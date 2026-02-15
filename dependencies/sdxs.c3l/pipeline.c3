module sdxs;

import vk;
import llm;
import llm::pipelines;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::time::clock;
import image;
import image::png;

struct SDXSState {
    bool loaded;
    llm::DiffusionConfig config;
    llm::DiffusionKernels kernels;
    ClipEncoder clip;
    UnetModel unet;
    VaeDecoder vae_decoder;
    VaeEncoder vae_encoder;
    VaeActivations vae_acts;
    llm::pipelines::SchedulerState scheduler;
    llm::DeviceContext* ctx;
}

fn void? sdxs_load(void* data, llm::DeviceContext* ctx, String model_path, llm::pipelines::PipelineOptions* opts) {
    SDXSState* state = (SDXSState*)data;
    state.ctx = ctx;

    io::printfn("\n=== Loading SDXS Pipeline ===");

    mmap::FileMmap mm = file::mmap_open(model_path, "rb")!!;
    char[] file_data = mm.bytes();
    llm::GGUFFile gf = llm::gguf_parse(file_data)!!;

    state.config = llm::load_diffusion_config((String)&llm::SD_CONFIG_JSON)!!;
    state.kernels = llm::create_diffusion_kernels(ctx)!!;

    state.clip = load_clip_encoder(ctx, &gf, &state.config.clip, &state.kernels)!!;
    state.unet = load_unet_model(ctx, &gf, &state.config.unet, &state.kernels)!!;
    state.vae_decoder = load_vae_decoder(ctx, &gf, &state.config.vae, &state.kernels)!!;
    state.vae_encoder = load_vae_encoder(ctx, &gf, &state.config.vae, &state.kernels)!!;

    state.vae_acts = allocate_vae_activations(ctx, state.config.image.size, 64)!!;

    state.scheduler = llm::pipelines::init_scheduler(
        &state.config.scheduler,
        llm::pipelines::SchedulerType.DDIM,
        1,
        1.0f
    )!!;

    // Fix up internal pointers
    state.clip.kernels = &state.kernels;
    state.unet.kernels = &state.kernels;
    state.vae_decoder.kernels = &state.kernels;
    state.vae_encoder.kernels = &state.kernels;

    gf.free();
    mm.destroy();

    state.loaded = true;
    io::printfn("\n=== SDXS Pipeline Ready ===\n");
}

fn image::Image? sdxs_generate(void* data, llm::DeviceContext* ctx, llm::pipelines::GenerationInputs* inputs) {
    SDXSState* state = (SDXSState*)data;

    bool is_img2img = inputs.input_image != null;

    if (is_img2img) {
        io::printfn("=== Image-to-Image Generation ===");
        io::printfn("  Prompt: \"%s\"", inputs.prompt);
        io::printfn("  Strength: %.2f, Steps: %d", inputs.strength, inputs.num_steps);

        if (inputs.input_image.width != 512 || inputs.input_image.height != 512) {
            io::printfn("Error: Input image must be 512x512 (got %dx%d)",
                inputs.input_image.width, inputs.input_image.height);
            return llm::pipelines::PIPELINE_INVALID_INPUT~;
        }

        // Convert input image to latent
        float[] input_data = llm::image_to_float_nchw(inputs.input_image)!!;

        for (usz i = 0; i < input_data.len; i++) {
            input_data[i] = input_data[i] * 2.0f - 1.0f;
        }

        usz img_bytes = (usz)3 * 512 * 512 * 4;
        vk::Memory img_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: input_data.ptr,
            data_size: img_bytes,
        )!!;

        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
            vk::cmdCopyBuffer(cmd, img_staging.buffer, state.vae_acts.buf_a.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = img_bytes }});
        }!!;
        img_staging.free();
        mem::free(input_data);

        // VAE encode
        io::printfn("[1/4] VAE encoding...");
        state.vae_encoder.forward(&state.vae_acts, 512, 512)!!;

        // Copy encoded latent to UNet buf_a
        uint latent_elems = 4 * 64 * 64;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
            vk::cmdCopyBuffer(cmd, state.vae_acts.buf_a.gpu_buffer.buffer,
                state.unet.acts.buf_a.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;

        // Tokenize + CLIP
        io::printfn("[2/4] Tokenizing + CLIP encoding...");
        encode_prompt(state, inputs.prompt);

        // Denoise
        io::printfn("[3/4] Denoising...");
        uint start_step = (uint)((1.0f - inputs.strength) * (float)inputs.num_steps);
        denoise(state, inputs, start_step);

    } else {
        io::printfn("=== Text-to-Image Generation ===");
        io::printfn("  Prompt: \"%s\"", inputs.prompt);
        io::printfn("  Steps: %d, Seed: %d", inputs.num_steps, inputs.seed);

        // Tokenize
        io::printfn("\n[1/3] Tokenizing + CLIP encoding...");
        encode_prompt(state, inputs.prompt);

        // Generate latent and denoise
        io::printfn("[2/3] UNet denoising (%d steps)...", inputs.num_steps);
        generate_and_denoise(state, inputs);
    }

    // VAE decode
    io::printfn("[%s] VAE decoding...", is_img2img ? "4/4" : "3/3");

    uint latent_elems = 4 * 64 * 64;
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, state.unet.acts.buf_a.gpu_buffer.buffer,
            state.vae_acts.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;

    state.vae_decoder.forward(&state.vae_acts)!!;

    // Download and convert to image
    usz total_px = (usz)3 * 512 * 512;
    float[] img_data = mem::new_array(float, total_px);

    vk::Memory download_staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: total_px * 4,
    )!!;

    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, state.vae_acts.buf_a.gpu_buffer.buffer, download_staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = total_px * 4 }});
    }!!;

    float* mapped = (float*)download_staging.data();
    mem::copy(img_data.ptr, mapped, total_px * 4);
    download_staging.free();

    image::Image result = llm::pipelines::float_tensor_to_image_sd(img_data, 512, 512, 3)!!;
    mem::free(img_data);

    io::printfn("\nGeneration complete!");
    return result;
}

// Helper: encode prompt with CLIP
fn void encode_prompt(SDXSState* state, String prompt) {
    uint[77] tokens;
    uint n_tokens = 0;

    // TODO: load tokenizer from model metadata when available
    // For now, use dummy BOS token
    tokens[0] = 49406;  // CLIP BOS
    n_tokens = 1;

    for (uint i = n_tokens; i < 77; i++) {
        tokens[i] = 0;
    }

    io::printfn("  Tokens: %d", n_tokens);

    uint[] token_slice = tokens[0..76];
    state.clip.forward(token_slice, n_tokens)!!;
}

// Helper: generate random latent and denoise
fn void generate_and_denoise(SDXSState* state, llm::pipelines::GenerationInputs* inputs) {
    uint latent_elems = 4 * 64 * 64;
    float[] latent_data = mem::new_array(float, latent_elems);
    llm::pipelines::generate_random_latent(latent_data, inputs.seed);

    vk::Memory staging = vk::new_buffer(
        allocator: &state.ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: latent_data.ptr,
        data_size: (usz)latent_elems * 4,
    )!!;

    state.ctx.device.@single_time_command(state.ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, staging.buffer, state.unet.acts.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;
    staging.free();
    mem::free(latent_data);

    // Setup scheduler
    llm::pipelines::SchedulerState* sched = &state.scheduler;
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

        state.unet.forward(&state.clip.acts.hidden, t)!!;

        float alpha_t = sched.get_alpha_cumprod(t);
        float alpha_prev = sched.get_prev_alpha_cumprod(step);

        vk::CommandBuffer cmd = state.ctx.command_buffer;
        llm::begin_compute(cmd)!!;
        if (sched.stype == llm::pipelines::SchedulerType.DDIM) {
            llm::pipelines::dispatch_ddim_step(cmd, &state.kernels,
                &state.unet.acts.buf_a,
                &state.unet.acts.buf_b,
                latent_elems, alpha_t, alpha_prev);
        } else {
            float sigma_t = 1.0f - alpha_t;
            float sigma_prev = 1.0f - alpha_prev;
            float dt = sigma_prev - sigma_t;
            llm::pipelines::dispatch_euler_step(cmd, &state.kernels,
                &state.unet.acts.buf_a,
                &state.unet.acts.buf_b,
                latent_elems, dt);
        }
        llm::submit_and_wait(state.ctx)!!;
    }
}

// Helper: denoise from a specific step (for img2img)
fn void denoise(SDXSState* state, llm::pipelines::GenerationInputs* inputs, uint start_step) {
    uint latent_elems = 4 * 64 * 64;
    llm::pipelines::SchedulerState* sched = &state.scheduler;

    for (uint step = start_step; step < inputs.num_steps; step++) {
        float t = sched.timesteps[step];
        io::printfn("  Step %d/%d (t=%.1f)", step + 1, inputs.num_steps, t);

        state.unet.forward(&state.clip.acts.hidden, t)!!;

        float alpha_t = sched.get_alpha_cumprod(t);
        float alpha_prev = sched.get_prev_alpha_cumprod(step);

        vk::CommandBuffer cmd = state.ctx.command_buffer;
        llm::begin_compute(cmd)!!;
        llm::pipelines::dispatch_ddim_step(cmd, &state.kernels,
            &state.unet.acts.buf_a, &state.unet.acts.buf_b,
            latent_elems, alpha_t, alpha_prev);
        llm::submit_and_wait(state.ctx)!!;
    }
}

fn void sdxs_free(void* data) {
    SDXSState* state = (SDXSState*)data;
    if (state.loaded) {
        state.clip.free();
        state.unet.free();
        state.vae_decoder.free();
        state.vae_encoder.free();
        state.vae_acts.free();
        state.scheduler.free();
        state.kernels.free(state.ctx.device);
        state.loaded = false;
    }
}

fn llm::pipelines::PipelineImpl create_pipeline() {
    SDXSState* state = mem::alloc(SDXSState);
    *state = {};
    return {
        .data = state,
        .load = &sdxs_load,
        .generate = &sdxs_generate,
        .free_fn = &sdxs_free,
    };
}
