module llm;

import vk;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::random;
import std::time::clock;

// Z-Image Turbo Pipeline
// Pipeline: tokenize → Qwen3 encode → DiT denoise (×N) → Flux VAE decode → save PPM

struct ZImageParams {
    uint num_steps;
    uint seed;
    String output_path;
    String vae_path;         // Path to Flux VAE safetensors file
    String text_model_path;  // Path to text encoder GGUF (Qwen3)
    uint image_size;         // 512 default
}

fn bool is_dit_model(GGUFFile* gf) {
    // Check architecture metadata first
    if (try arch = gf.get_string("general.architecture")) {
        if (arch == "lumina2") return true;
    }
    // Fallback: check for DiT tensor name prefixes (Lumina2 / Z-Image)
    foreach (&t : gf.tensors) {
        if (t.name.starts_with("noise_refiner.") ||
            t.name.starts_with("context_refiner.") ||
            t.name.starts_with("x_pad_token")) {
            return true;
        }
    }
    return false;
}

fn bool run_zimage_pipeline(
    DeviceContext* ctx,
    GGUFFile* dit_gf,
    String prompt,
    ZImageParams* params
) {
    io::printfn("\n=== Z-Image Turbo Pipeline ===");
    io::printfn("  Prompt: \"%s\"", prompt);
    io::printfn("  Steps: %d, Seed: %d, Size: %d", params.num_steps, params.seed, params.image_size);

    Clock total_start = clock::now();

    uint img_size = params.image_size;
    uint lat_h = img_size / 8;  // latent height (64 for 512)
    uint lat_w = img_size / 8;  // latent width
    uint n_patches = (lat_h / DIT_PATCH_SIZE) * (lat_w / DIT_PATCH_SIZE);

    // ============================================================
    // Phase 1: Text Encoding via Qwen3
    // ============================================================
    io::printfn("\n[1/5] Text encoding (Qwen3)...");

    if (params.text_model_path.len == 0) {
        io::printfn("Error: --text-model required for DiT pipeline (path to Qwen3 GGUF)");
        return false;
    }

    // Load text encoder from separate GGUF file
    mmap::FileMmap text_mm = file::mmap_open(params.text_model_path, "rb")!!;
    char[] text_data = text_mm.bytes();
    io::printfn("  Text model: %s (%d bytes)", params.text_model_path, text_data.len);

    GGUFFile text_gf = gguf_parse(text_data)!!;

    String arch_name = detect_architecture(&text_gf);
    io::printfn("  Text encoder architecture: %s", arch_name);

    String config_json;
    if (arch_name == "qwen3") {
        config_json = (String)&QWEN3_CONFIG_JSON;
    } else if (arch_name == "qwen2") {
        config_json = (String)&QWEN2_CONFIG_JSON;
    } else {
        config_json = (String)&QWEN3_CONFIG_JSON;
    }

    ModelConfig text_config = load_arch_config(config_json, &text_gf)!!;
    WeightNames text_names = load_weight_names(config_json)!!;

    // Tokenize prompt first to know text_len for KV cache sizing
    Tokenizer tok = load_tokenizer(&text_gf)!!;
    uint[] prompt_tokens = tok.encode(prompt)!!;
    io::printfn("  Prompt: %d tokens", prompt_tokens.len);

    // Prepend BOS token
    uint text_len = (uint)prompt_tokens.len + 1;
    uint[] full_tokens = mem::new_array(uint, text_len);
    full_tokens[0] = tok.bos_id;
    for (usz i = 0; i < prompt_tokens.len; i++) {
        full_tokens[i + 1] = prompt_tokens[i];
    }

    // Load LLM model for text encoding (encode_only: skip F32 embedding + output weight, small KV cache)
    uint enc_max_seq = text_len + 16;
    LlmModel text_model = load_llm_model(ctx, &text_gf, &text_config, &text_names,
        encode_only: true, max_seq_len: enc_max_seq)!!;

    Clock encode_start = clock::now();
    // Encode text: run LLM forward, get hidden states [text_len, dim]
    Tensor text_embeddings = text_model.encode_text(full_tokens)!!;
    double encode_sec = encode_start.to_now().to_sec();
    io::printfn("  Text encoded: [%d, %d] in %.2fs", text_len, text_config.dim, encode_sec);

    // Free CPU-side data (keep text embeddings on GPU)
    mem::free(prompt_tokens);
    mem::free(full_tokens);
    tok.free();
    text_model.free();
    text_gf.free();
    text_mm.destroy();
    io::printfn("  Text encoding complete.");

    // ============================================================
    // Phase 2: Load DiT + VAE
    // ============================================================
    io::printfn("\n[2/5] Loading DiT + VAE...");

    // Load DiT weights
    DiTWeights dit_weights = load_dit_weights(ctx, dit_gf)!!;
    DiTKernels dit_kernels = create_dit_kernels(ctx)!!;

    // Allocate DiT activations
    uint max_seq = n_patches + text_len + 32;  // some headroom
    DiTActivations dit_acts = allocate_dit_activations(ctx, max_seq)!!;

    DiTModel* dit = mem::alloc(DiTModel);
    dit.weights = dit_weights;
    dit.acts = dit_acts;
    dit.kernels = dit_kernels;
    dit.ctx = ctx;
    dit.n_patches = n_patches;
    dit.text_len = text_len;
    dit.latent_h = lat_h;
    dit.latent_w = lat_w;

    io::printfn("  DiT loaded: %d patches, %d text tokens", n_patches, text_len);

    // Project text embeddings: cap_embedder = Sequential(RMSNorm(text_dim), Linear(text_dim, dim))
    // text_embeddings is [text_len, text_dim], need [text_len, DIT_DIM]
    Tensor projected_text = create_f32_tensor(ctx, { (ulong)text_len * DIT_DIM, 0, 0, 0 }, 1)!!;
    Tensor norm_scratch = create_f32_tensor(ctx, { (ulong)text_config.dim, 0, 0, 0 }, 1)!!;

    CommandBuffer cmd = ctx.command_buffer;
    begin_compute(cmd)!!;

    // RMSNorm each text position, then linear project
    for (uint pos = 0; pos < text_len; pos++) {
        usz in_off = (usz)pos * text_config.dim * 4;
        // Copy position to scratch
        vk::cmdCopyBuffer(cmd, text_embeddings.gpu_buffer.buffer, norm_scratch.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)text_config.dim * 4 }});
        compute_barrier(cmd);
        // RMSNorm in-place
        RMSNormPC rms_pc = { .dim = text_config.dim, .eps = 1e-6f };
        dispatch_kernel(cmd, &dit.kernels.shared.rmsnorm,
            { norm_scratch.gpu_buffer.buffer, dit.weights.cap_emb_norm.gpu_buffer.buffer,
              norm_scratch.gpu_buffer.buffer },
            { norm_scratch.size_bytes, dit.weights.cap_emb_norm.size_bytes,
              norm_scratch.size_bytes },
            &rms_pc, 1);
        compute_barrier(cmd);
        // Write back
        vk::cmdCopyBuffer(cmd, norm_scratch.gpu_buffer.buffer, text_embeddings.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = in_off, .size = (ulong)text_config.dim * 4 }});
        compute_barrier(cmd);
    }

    // Linear projection: text_embeddings [text_len, text_dim] -> projected_text [text_len, DIT_DIM]
    LinearProjPC cap_pc = { .out_dim = DIT_DIM, .in_dim = text_config.dim, .seq_len = text_len };
    dispatch_kernel(cmd, &dit.kernels.linear_proj,
        { dit.weights.cap_emb_weight.gpu_buffer.buffer, dit.weights.cap_emb_bias.gpu_buffer.buffer,
          text_embeddings.gpu_buffer.buffer, projected_text.gpu_buffer.buffer },
        { dit.weights.cap_emb_weight.size_bytes, dit.weights.cap_emb_bias.size_bytes,
          text_embeddings.size_bytes, projected_text.size_bytes },
        &cap_pc, text_len * DIT_DIM);
    compute_barrier(cmd);
    submit_and_wait(ctx)!!;
    norm_scratch.free();

    // Free original text embeddings
    text_embeddings.free();

    // Run context refiner on projected text (2 layers, no adaLN)
    begin_compute(cmd)!!;
    for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
        dit_transformer_layer(dit, cmd, &dit.weights.context_refiner[l],
            &projected_text, text_len, 0, false)!!;
    }
    submit_and_wait(ctx)!!;

    io::printfn("  Context refiner complete.");

    // Load VAE
    DiffusionKernels diff_kernels = create_diffusion_kernels(ctx)!!;
    FluxVaeDecoder vae_decoder;
    FluxVaeActivations vae_acts;
    bool has_vae = false;

    if (params.vae_path.len > 0) {
        SafetensorsFile sf = safetensors_open(params.vae_path)!!;
        vae_decoder = load_flux_vae_decoder(ctx, &sf, &diff_kernels)!!;
        sf.close();  // Release mmap after weights are on GPU
        vae_acts = allocate_flux_vae_activations(ctx, img_size)!!;
        has_vae = true;
        io::printfn("  Flux VAE loaded from %s", params.vae_path);
    } else {
        io::printfn("  Warning: No VAE path specified (--vae). Will output raw latent.");
    }

    io::printfn("  GPU memory: %d MB allocated, %d MB fragmented (reusable)",
        ctx.allocator.total_allocated() / (1024 * 1024),
        ctx.allocator.total_fragmented() / (1024 * 1024));

    // ============================================================
    // Phase 3: Generate random latent
    // ============================================================
    io::printfn("\n[3/5] Generating random latent...");

    uint latent_elems = DIT_LATENT_CHANNELS * lat_h * lat_w;
    float[] latent_data = mem::new_array(float, latent_elems);
    generate_random_latent(latent_data, params.seed);

    // Upload to DiT latent buffer
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: latent_data.ptr,
        data_size: (usz)latent_elems * 4,
    )!!;

    ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer xfer_cmd) {
        vk::cmdCopyBuffer(xfer_cmd, staging.buffer, dit.acts.latent.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;
    staging.free();
    mem::free(latent_data);

    // ============================================================
    // Phase 4: Denoising loop
    // ============================================================
    io::printfn("\n[4/5] Denoising (%d steps)...", params.num_steps);

    FlowMatchingState flow = init_flow_matching(params.num_steps);

    Clock denoise_start = clock::now();
    for (uint step = 0; step < params.num_steps; step++) {
        float t = flow.get_timestep(step);
        float dt = flow.get_dt(step);
        io::printfn("  Step %d/%d (t=%.3f, dt=%.3f)", step + 1, params.num_steps, t, dt);

        Clock step_start = clock::now();
        if (step == 0) { dit_profile_layer = true; }

        // DiT forward: predicts velocity
        dit.forward(&projected_text, t)!!;

        // Euler step: latent += dt * velocity
        begin_compute(cmd)!!;
        FlowEulerPC euler_pc = { .n = latent_elems, .dt = dt };
        dispatch_kernel(cmd, &dit.kernels.flow_euler_step,
            { dit.acts.latent.gpu_buffer.buffer, dit.acts.velocity.gpu_buffer.buffer },
            { dit.acts.latent.size_bytes, dit.acts.velocity.size_bytes },
            &euler_pc, ceil_div(latent_elems, 256));
        submit_and_wait(ctx)!!;

        double step_sec = step_start.to_now().to_sec();
        io::printfn("  Step %d done in %.2fs", step + 1, step_sec);
    }
    double denoise_sec = denoise_start.to_now().to_sec();
    io::printfn("  Denoising complete in %.2fs", denoise_sec);

    flow.free();

    // ============================================================
    // Phase 5: VAE decode
    // ============================================================
    io::printfn("\n[5/5] VAE decoding...");

    if (has_vae) {
        // Scale latent by VAE scaling factor (1/0.3611 for Flux VAE)
        begin_compute(cmd)!!;
        ScalePC scale_pc = { .n = latent_elems, .scale = 1.0f / 0.3611f };
        dispatch_kernel(cmd, &dit.kernels.scale_buffer,
            { dit.acts.latent.gpu_buffer.buffer },
            { dit.acts.latent.size_bytes },
            &scale_pc, ceil_div(latent_elems, 256));
        submit_and_wait(ctx)!!;

        // Copy latent to VAE input buffer
        ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer copy_cmd) {
            vk::cmdCopyBuffer(copy_cmd, dit.acts.latent.gpu_buffer.buffer,
                vae_acts.buf_a.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;

        // Run VAE decoder
        vae_decoder.forward(&vae_acts, lat_h, lat_w)!!;

        // Download image
        Image img = download_image_from_gpu(ctx, &vae_acts.buf_a, img_size, img_size, 3)!!;
        bool ok = save_ppm(&img, params.output_path);
        img.free();

        // Cleanup
        vae_decoder.free();
        vae_acts.free();
        diff_kernels.free(ctx.device);
        projected_text.free();
        dit.free();
        mem::free(dit);

        double total_sec = total_start.to_now().to_sec();
        io::printfn("\n=== Generation Complete ===");
        io::printfn("  Output: %s (%dx%d)", params.output_path, img_size, img_size);
        io::printfn("  Total time: %.2fs", total_sec);

        return ok;
    } else {
        // No VAE: save raw latent as debug output
        io::printfn("  No VAE loaded. Skipping decode.");

        // Cleanup
        projected_text.free();
        diff_kernels.free(ctx.device);
        dit.free();
        mem::free(dit);

        double total_sec = total_start.to_now().to_sec();
        io::printfn("\n=== Pipeline Complete (no VAE) ===");
        io::printfn("  Total time: %.2fs", total_sec);
        return true;
    }
}
