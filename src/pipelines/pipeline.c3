module llm::pipelines;

import vk;
import std::io;
import std::core::mem;
import image;
import llm;
import llm::diffusers;

// Diffusion Pipeline: PyTorch-style API
// Usage:
//   DiffusionPipeline pipeline = load_diffusion_pipeline(&ctx, &gf)!!;
//   pipeline.fixup_pointers();
//   DiffusionInputs inputs = DiffusionInputs.defaults();
//   inputs.prompt = "a photo of a cat";
//   image::Image output = pipeline.generate(&inputs)!!;
//   output.save("output.png");
//   output.free();
//   pipeline.free();

struct DiffusionPipeline {
    llm::DiffusionConfig config;
    llm::DiffusionKernels kernels;
    llm::diffusers::ClipEncoder clip;
    llm::diffusers::UnetModel unet;
    llm::diffusers::VaeDecoder vae_decoder;
    llm::diffusers::VaeEncoder vae_encoder;
    llm::diffusers::VaeActivations vae_acts;
    SchedulerState scheduler;
    DeviceContext* ctx;
}

// --- Load complete diffusion pipeline ---

fn DiffusionPipeline? load_diffusion_pipeline(DeviceContext* ctx, GGUFFile* gf) {
    io::printfn("\n=== Loading Diffusion Pipeline ===");

    llm::DiffusionConfig config = llm::load_diffusion_config((String)&llm::SD_CONFIG_JSON)!!;
    llm::DiffusionKernels kernels = llm::create_diffusion_kernels(ctx)!!;

    llm::diffusers::ClipEncoder clip = llm::diffusers::load_clip_encoder(ctx, gf, &config.clip, &kernels)!!;
    llm::diffusers::UnetModel unet = llm::diffusers::load_unet_model(ctx, gf, &config.unet, &kernels)!!;
    llm::diffusers::VaeDecoder vae_decoder = llm::diffusers::load_vae_decoder(ctx, gf, &config.vae, &kernels)!!;
    llm::diffusers::VaeEncoder vae_encoder = llm::diffusers::load_vae_encoder(ctx, gf, &config.vae, &kernels)!!;

    llm::diffusers::VaeActivations vae_acts = llm::diffusers::allocate_vae_activations(ctx, config.image.size, 64)!!;

    SchedulerState scheduler = init_scheduler(
        &config.scheduler,
        DDIM,
        1,
        1.0f
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

// Fix up internal pointers after struct copy
fn void DiffusionPipeline.fixup_pointers(&self) {
    self.clip.kernels = &self.kernels;
    self.unet.kernels = &self.kernels;
    self.vae_decoder.kernels = &self.kernels;
    self.vae_encoder.kernels = &self.kernels;
}

// --- Unified Generate Method ---
// Supports both txt2img and img2img based on inputs

fn image::Image? DiffusionPipeline.generate(&self, 
    Tokenizer* tokenizer, 
    DiffusionInputs* inputs
) {
    // Validate inputs
    if (!inputs.validate()) {
        return PIPELINE_INVALID_INPUT~;
    }
    
    bool is_img2img = inputs.input_image != null;
    
    if (is_img2img) {
        io::printfn("=== Image-to-Image Generation ===");
        io::printfn("  Prompt: \"%s\"", inputs.prompt);
        io::printfn("  Strength: %.2f, Steps: %d", inputs.strength, inputs.num_steps);
        
        if (inputs.input_image.width != 512 || inputs.input_image.height != 512) {
            io::printfn("Error: Input image must be 512x512 (got %dx%d)", 
                inputs.input_image.width, inputs.input_image.height);
            return PIPELINE_INVALID_INPUT~;
        }
        
        // Convert input image to latent
        float[] input_data = llm::image_to_float_nchw(inputs.input_image)!!;
        
        for (usz i = 0; i < input_data.len; i++) {
            input_data[i] = input_data[i] * 2.0f - 1.0f;
        }

        usz img_bytes = (usz)3 * 512 * 512 * 4;
        vk::Memory img_staging = vk::new_buffer(
            allocator: &self.ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: input_data.ptr,
            data_size: img_bytes,
        )!!;

        self.ctx.device.@single_time_command(self.ctx.compute_queue; vk::CommandBuffer cmd) {
            vk::cmdCopyBuffer(cmd, img_staging.buffer, self.vae_acts.buf_a.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = img_bytes }});
        }!!;
        img_staging.free();
        mem::free(input_data);

        // VAE encode
        io::printfn("[1/4] VAE encoding...");
        self.vae_encoder.forward(&self.vae_acts, 512, 512)!!;
        
        // Copy encoded latent to UNet buf_a
        uint latent_elems = 4 * 64 * 64;
        self.ctx.device.@single_time_command(self.ctx.compute_queue; vk::CommandBuffer cmd) {
            vk::cmdCopyBuffer(cmd, self.vae_acts.buf_a.gpu_buffer.buffer,
                self.unet.acts.buf_a.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;
        
        // Tokenize + CLIP
        io::printfn("[2/4] Tokenizing...");
        encode_prompt(self, tokenizer, inputs.prompt);
        
        // Denoise
        io::printfn("[3/4] Denoising...");
        uint start_step = (uint)((1.0f - inputs.strength) * (float)inputs.num_steps);
        denoise(self, inputs, start_step);
        
    } else {
        io::printfn("=== Text-to-Image Generation ===");
        io::printfn("  Prompt: \"%s\"", inputs.prompt);
        io::printfn("  Steps: %d, Seed: %d", inputs.num_steps, inputs.seed);

        // Tokenize
        io::printfn("\n[1/3] Tokenizing...");
        encode_prompt(self, tokenizer, inputs.prompt);

        // Generate latent and denoise
        io::printfn("[2/3] UNet denoising (%d steps)...", inputs.num_steps);
        generate_and_denoise(self, inputs);
    }

    // VAE decode
    io::printfn("[%s] VAE decoding...", is_img2img ? "4/4" : "3/3");
    
    uint latent_elems = 4 * 64 * 64;
    self.ctx.device.@single_time_command(self.ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, self.unet.acts.buf_a.gpu_buffer.buffer,
            self.vae_acts.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;

    self.vae_decoder.forward(&self.vae_acts)!!;

    // Download and convert to image
    usz total_px = (usz)3 * 512 * 512;
    float[] img_data = mem::new_array(float, total_px);
    
    vk::Memory download_staging = vk::new_buffer(
        allocator: &self.ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: total_px * 4,
    )!!;

    self.ctx.device.@single_time_command(self.ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, self.vae_acts.buf_a.gpu_buffer.buffer, download_staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = total_px * 4 }});
    }!!;

    float* mapped = (float*)download_staging.data();
    mem::copy(img_data.ptr, mapped, total_px * 4);
    download_staging.free();

    image::Image result = float_tensor_to_image_sd(img_data, 512, 512, 3)!!;
    mem::free(img_data);

    io::printfn("\nGeneration complete!");
    return result;
}

// Helper: encode prompt with CLIP
fn void encode_prompt(DiffusionPipeline* self, Tokenizer* tokenizer, String prompt) {
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
    
    for (uint i = n_tokens; i < 77; i++) {
        tokens[i] = 0;
    }
    
    io::printfn("  Tokens: %d", n_tokens);

    io::printfn("[2/3] CLIP encoding...");
    uint[] token_slice = tokens[0..76];
    self.clip.forward(token_slice, n_tokens)!!;
}

// Helper: generate random latent and denoise
fn void generate_and_denoise(DiffusionPipeline* self, DiffusionInputs* inputs) {
    uint latent_elems = 4 * 64 * 64;
    float[] latent_data = mem::new_array(float, latent_elems);
    generate_random_latent(latent_data, inputs.seed);

    vk::Memory staging = vk::new_buffer(
        allocator: &self.ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: latent_data.ptr,
        data_size: (usz)latent_elems * 4,
    )!!;

    self.ctx.device.@single_time_command(self.ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, staging.buffer, self.unet.acts.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;
    staging.free();
    mem::free(latent_data);

    // Setup scheduler
    SchedulerState* sched = &self.scheduler;
    sched.num_inference_steps = inputs.num_steps;
    sched.cfg_scale = inputs.cfg_scale;

    mem::free(sched.timesteps);
    sched.timesteps = mem::new_array(float, inputs.num_steps);
    for (uint i = 0; i < inputs.num_steps; i++) {
        float t = (float)(sched.num_train_timesteps - 1) *
            (1.0f - (float)i / (float)(inputs.num_steps > 1 ? inputs.num_steps - 1 : 1));
        sched.timesteps[i] = t;
    }

    // Denoise
    for (uint step = 0; step < inputs.num_steps; step++) {
        float t = sched.timesteps[step];
        io::printfn("  Step %d/%d (t=%.1f)", step + 1, inputs.num_steps, t);

        self.unet.forward(&self.clip.acts.hidden, t)!!;

        float alpha_t = sched.get_alpha_cumprod(t);
        float alpha_prev = sched.get_prev_alpha_cumprod(step);

        vk::CommandBuffer cmd = self.ctx.command_buffer;
        llm::begin_compute(cmd)!!;
        if (sched.stype == DDIM) {
            dispatch_ddim_step(cmd, &self.kernels,
                &self.unet.acts.buf_a,
                &self.unet.acts.buf_b,
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
        llm::submit_and_wait(self.ctx)!!;
    }
}

// Helper: denoise from a specific step (for img2img)
fn void denoise(DiffusionPipeline* self, DiffusionInputs* inputs, uint start_step) {
    uint latent_elems = 4 * 64 * 64;
    SchedulerState* sched = &self.scheduler;

    for (uint step = start_step; step < inputs.num_steps; step++) {
        float t = sched.timesteps[step];
        io::printfn("  Step %d/%d (t=%.1f)", step + 1, inputs.num_steps, t);

        self.unet.forward(&self.clip.acts.hidden, t)!!;

        float alpha_t = sched.get_alpha_cumprod(t);
        float alpha_prev = sched.get_prev_alpha_cumprod(step);

        vk::CommandBuffer cmd = self.ctx.command_buffer;
        llm::begin_compute(cmd)!!;
        dispatch_ddim_step(cmd, &self.kernels,
            &self.unet.acts.buf_a, &self.unet.acts.buf_b,
            latent_elems, alpha_t, alpha_prev);
        llm::submit_and_wait(self.ctx)!!;
    }
}

// Helper: convert float tensor to image::Image
fn image::Image? float_tensor_to_image_sd(float[] data, uint width, uint height, uint channels) {
    PixelFormat format;
    switch (channels) {
        case 1: format = PixelFormat.GRAYSCALE;
        case 3: format = PixelFormat.RGB;
        case 4: format = PixelFormat.RGBA;
        default: return PIPELINE_INVALID_INPUT~;
    }
    
    char[] pixels = mem::new_array(char, (usz)width * height * channels);
    
    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            uint pixel_idx = y * width + x;
            for (uint c = 0; c < channels; c++) {
                float val = data[c * width * height + pixel_idx];
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                pixels[(usz)pixel_idx * channels + c] = (char)(val * 255.0f + 0.5f);
            }
        }
    }
    
    return {
        .width = width,
        .height = height,
        .bit_depth = 8,
        .format = format,
        .pixels = pixels,
    };
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
