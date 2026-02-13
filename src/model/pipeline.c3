module llm;

import vk;
import std::io;
import std::core::mem;

// Diffusion Pipeline: txt2img and img2img

struct DiffusionPipeline {
    DiffusionConfig config;
    DiffusionKernels kernels;
    ClipEncoder clip;
    UnetModel unet;
    VaeDecoder vae_decoder;
    VaeEncoder vae_encoder;
    VaeActivations vae_acts;
    SchedulerState scheduler;
    DeviceContext* ctx;
}

struct DiffusionParams {
    uint num_steps;
    float cfg_scale;
    uint seed;
    float strength;  // For img2img: 0.0 = no change, 1.0 = full denoise
    String output_path;
}

// --- Load complete diffusion pipeline ---

fn DiffusionPipeline? load_diffusion_pipeline(DeviceContext* ctx, GGUFFile* gf) {
    io::printfn("\n=== Loading Diffusion Pipeline ===");

    // Load config
    DiffusionConfig config = load_diffusion_config((String)&SD_CONFIG_JSON)!!;

    // Create kernels
    DiffusionKernels kernels = create_diffusion_kernels(ctx)!!;

    // Load components
    ClipEncoder clip = load_clip_encoder(ctx, gf, &config.clip, &kernels)!!;
    UnetModel unet = load_unet_model(ctx, gf, &config.unet, &kernels)!!;
    VaeDecoder vae_decoder = load_vae_decoder(ctx, gf, &config.vae, &kernels)!!;
    VaeEncoder vae_encoder = load_vae_encoder(ctx, gf, &config.vae, &kernels)!!;

    // Allocate VAE activation buffers (512x512 max, 64 channels)
    VaeActivations vae_acts = allocate_vae_activations(ctx, config.image.size, 64)!!;

    // Initialize scheduler
    SchedulerState scheduler = init_scheduler(
        &config.scheduler,
        DDIM,
        1,     // SDXS is 1-step by default
        1.0f   // No CFG for single-step
    )!!;

    io::printfn("\n=== Diffusion Pipeline Ready ===\n");

    return {
        .config = config,
        .kernels = kernels,
        .clip = clip,
        .unet = unet,
        .vae_decoder = vae_decoder,
        .vae_encoder = vae_encoder,
        .vae_acts = vae_acts,
        .scheduler = scheduler,
        .ctx = ctx,
    };
}

// Fix up internal pointers after struct copy (return-by-value creates dangling pointers)
fn void DiffusionPipeline.fixup_pointers(&self) {
    self.clip.kernels = &self.kernels;
    self.unet.kernels = &self.kernels;
    self.vae_decoder.kernels = &self.kernels;
    self.vae_encoder.kernels = &self.kernels;
}

// --- Text-to-Image ---

fn bool DiffusionPipeline.txt2img(&self, Tokenizer* tokenizer, String prompt, DiffusionParams* params) {
    io::printfn("=== Text-to-Image Generation ===");
    io::printfn("  Prompt: \"%s\"", prompt);
    io::printfn("  Steps: %d, Seed: %d", params.num_steps, params.seed);

    // 1. Tokenize prompt
    io::printfn("\n[1/4] Tokenizing...");
    uint[77] tokens;
    uint n_tokens = 0;
    if (tokenizer != null) {
        uint[] encoded = tokenizer.encode(prompt)!!;
        for (usz i = 0; i < encoded.len && i < 77; i++) {
            tokens[i] = encoded[i];
            n_tokens++;
        }
        mem::free(encoded);
    } else {
        // No tokenizer: use BOS + padding (will produce unconditional output)
        tokens[0] = 49406;  // CLIP BOS <|startoftext|>
        n_tokens = 1;
    }
    // Pad remaining with 0 (padding token)
    for (uint i = n_tokens; i < 77; i++) {
        tokens[i] = 0;
    }
    io::printfn("  Tokens: %d", n_tokens);

    // 2. CLIP encode
    io::printfn("[2/4] CLIP encoding...");
    uint[] token_slice = tokens[0..76];
    self.clip.forward(token_slice, n_tokens)!!;
    // CLIP output is in clip.acts.hidden [77, 768]

    // 3. Generate random latent and run UNet
    io::printfn("[3/4] UNet denoising (%d steps)...", params.num_steps);

    // Generate random latent on CPU
    uint latent_elems = 4 * 64 * 64;  // [4, 64, 64]
    float[] latent_data = mem::new_array(float, latent_elems);
    generate_random_latent(latent_data, params.seed);

    // Upload to GPU (into UNet's buf_a)
    vk::Memory staging = vk::new_buffer(
        allocator: &self.ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: latent_data.ptr,
        data_size: (usz)latent_elems * 4,
    )!!;

    self.ctx.device.@single_time_command(self.ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, staging.buffer, self.unet.acts.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;
    staging.free();
    mem::free(latent_data);

    // Denoising loop
    SchedulerState* sched = &self.scheduler;
    sched.num_inference_steps = params.num_steps;
    sched.cfg_scale = params.cfg_scale;

    // Re-initialize timesteps for this run
    mem::free(sched.timesteps);
    sched.timesteps = mem::new_array(float, params.num_steps);
    for (uint i = 0; i < params.num_steps; i++) {
        float t = (float)(sched.num_train_timesteps - 1) *
            (1.0f - (float)i / (float)(params.num_steps > 1 ? params.num_steps - 1 : 1));
        sched.timesteps[i] = t;
    }

    for (uint step = 0; step < params.num_steps; step++) {
        float t = sched.timesteps[step];
        io::printfn("  Step %d/%d (t=%.1f)", step + 1, params.num_steps, t);

        // Run UNet: predicts noise from noisy latent + text conditioning
        self.unet.forward(&self.clip.acts.hidden, t)!!;

        // Apply scheduler step
        float alpha_t = sched.get_alpha_cumprod(t);
        float alpha_prev = sched.get_prev_alpha_cumprod(step);

        CommandBuffer cmd = self.ctx.command_buffer;
        begin_compute(cmd)!!;
        if (sched.stype == DDIM) {
            // DDIM step: buf_a has predicted noise, need to update latent
            // Actually our UNet writes result to buf_a as predicted noise
            // We need a separate buffer for the noisy latent...
            // For single-step: just use the UNet output directly as denoised latent
            dispatch_ddim_step(cmd, &self.kernels,
                &self.unet.acts.buf_a,  // noisy/output
                &self.unet.acts.buf_b,  // predicted noise (temp)
                latent_elems, alpha_t, alpha_prev);
        } else {
            float sigma_t = 1.0f - alpha_t;
            float sigma_prev = 1.0f - alpha_prev;
            float dt = sigma_prev - sigma_t;
            dispatch_euler_step(cmd, &self.kernels,
                &self.unet.acts.buf_a,
                &self.unet.acts.buf_b,
                latent_elems, dt);
        }
        submit_and_wait(self.ctx)!!;
    }

    // 4. VAE decode
    io::printfn("[4/4] VAE decoding...");

    // Copy denoised latent from UNet buf_a to VAE buf_a
    self.ctx.device.@single_time_command(self.ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, self.unet.acts.buf_a.gpu_buffer.buffer,
            self.vae_acts.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;

    self.vae_decoder.forward(&self.vae_acts)!!;
    // Image [3, 512, 512] is now in vae_acts.buf_a

    // Download image from GPU
    Image img = download_image_from_gpu(self.ctx, &self.vae_acts.buf_a, 512, 512, 3)!!;

    // Save
    bool ok = save_png(&img, params.output_path);
    img.free();

    if (ok) {
        io::printfn("\nGeneration complete! Output: %s", params.output_path);
    }
    return ok;
}

// --- Image-to-Image ---

fn bool DiffusionPipeline.img2img(&self, Tokenizer* tokenizer, String prompt, String input_path, DiffusionParams* params) {
    io::printfn("=== Image-to-Image Generation ===");
    io::printfn("  Prompt: \"%s\"", prompt);
    io::printfn("  Input: %s", input_path);
    io::printfn("  Strength: %.2f, Steps: %d", params.strength, params.num_steps);

    // 1. Load input image
    Image input_img = load_png(input_path)!!;
    if (input_img.width != 512 || input_img.height != 512) {
        io::printfn("Error: Input image must be 512x512 (got %dx%d)", input_img.width, input_img.height);
        input_img.free();
        return false;
    }

    // 2. Normalize to [-1, 1] and upload
    for (usz i = 0; i < input_img.data.len; i++) {
        input_img.data[i] = input_img.data[i] * 2.0f - 1.0f;
    }

    // Upload to VAE acts buf_a
    usz img_bytes = (usz)3 * 512 * 512 * 4;
    vk::Memory img_staging = vk::new_buffer(
        allocator: &self.ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: input_img.data.ptr,
        data_size: img_bytes,
    )!!;

    self.ctx.device.@single_time_command(self.ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, img_staging.buffer, self.vae_acts.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = img_bytes }});
    }!!;
    img_staging.free();
    input_img.free();

    // 3. VAE encode
    io::printfn("[1/5] VAE encoding...");
    self.vae_encoder.forward(&self.vae_acts, 512, 512)!!;
    // Latent [4, 64, 64] in vae_acts.buf_a

    // 4. Tokenize + CLIP encode
    io::printfn("[2/5] Tokenizing...");
    uint[77] tokens;
    uint n_tokens = 0;
    if (tokenizer != null) {
        uint[] encoded = tokenizer.encode(prompt)!!;
        for (usz i = 0; i < encoded.len && i < 77; i++) {
            tokens[i] = encoded[i];
            n_tokens++;
        }
        mem::free(encoded);
    } else {
        tokens[0] = 49406;  // CLIP BOS
        n_tokens = 1;
    }
    for (uint i = n_tokens; i < 77; i++) tokens[i] = 0;

    io::printfn("[3/5] CLIP encoding...");
    uint[] token_slice = tokens[0..76];
    self.clip.forward(token_slice, n_tokens)!!;

    // 5. Add noise to latent based on strength
    io::printfn("[4/5] Adding noise and denoising...");
    uint latent_elems = 4 * 64 * 64;

    // Copy encoded latent to UNet buf_a
    self.ctx.device.@single_time_command(self.ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, self.vae_acts.buf_a.gpu_buffer.buffer,
            self.unet.acts.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;

    // Add noise according to strength
    // Skip early timesteps based on strength
    uint start_step = (uint)((1.0f - params.strength) * (float)params.num_steps);

    // Generate noise on CPU and add to latent
    float[] noise = mem::new_array(float, latent_elems);
    generate_random_latent(noise, params.seed);

    // TODO: Add noise using scheduler. For now, run full denoising
    mem::free(noise);

    // Denoising loop (from start_step)
    SchedulerState* sched = &self.scheduler;
    for (uint step = start_step; step < params.num_steps; step++) {
        float t = sched.timesteps[step];
        io::printfn("  Step %d/%d (t=%.1f)", step + 1, params.num_steps, t);

        self.unet.forward(&self.clip.acts.hidden, t)!!;

        float alpha_t = sched.get_alpha_cumprod(t);
        float alpha_prev = sched.get_prev_alpha_cumprod(step);

        CommandBuffer cmd = self.ctx.command_buffer;
        begin_compute(cmd)!!;
        dispatch_ddim_step(cmd, &self.kernels,
            &self.unet.acts.buf_a, &self.unet.acts.buf_b,
            latent_elems, alpha_t, alpha_prev);
        submit_and_wait(self.ctx)!!;
    }

    // 6. VAE decode
    io::printfn("[5/5] VAE decoding...");
    self.ctx.device.@single_time_command(self.ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, self.unet.acts.buf_a.gpu_buffer.buffer,
            self.vae_acts.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;

    self.vae_decoder.forward(&self.vae_acts)!!;

    Image img = download_image_from_gpu(self.ctx, &self.vae_acts.buf_a, 512, 512, 3)!!;
    bool ok = save_png(&img, params.output_path);
    img.free();

    if (ok) {
        io::printfn("\nGeneration complete! Output: %s", params.output_path);
    }
    return ok;
}

// --- Free ---

fn void DiffusionPipeline.free(&self) {
    self.clip.free();
    self.unet.free();
    self.vae_decoder.free();
    self.vae_encoder.free();
    self.vae_acts.free();
    self.scheduler.free();
    self.kernels.free(self.ctx.device);
}
