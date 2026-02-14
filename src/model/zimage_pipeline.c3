module llm;

import vk;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::random;
import std::time::clock;

// Z-Image Turbo Pipeline
// Pipeline: tokenize → Qwen3 encode → DiT denoise (×N) → Flux VAE decode → save PNG

struct ZImageParams {
    uint num_steps;
    uint seed;
    String output_path;
    String vae_path;         // Path to Flux VAE safetensors file
    String taesd_path;       // Path to TAESD safetensors file (alternative to full VAE)
    String text_model_path;  // Path to text encoder GGUF (Qwen3)
    uint image_size;         // 512 default
    float cfg_scale;         // Classifier-free guidance scale (default 7.0 for Z-Image)
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
    float cfg_scale = params.cfg_scale;
    bool use_cfg = cfg_scale > 1.0f;

    io::printfn("\n=== Z-Image Turbo Pipeline ===");
    io::printfn("  Prompt: \"%s\"", prompt);
    io::printfn("  Steps: %d, Seed: %d, Size: %d, CFG: %.1f", params.num_steps, params.seed, params.image_size, cfg_scale);

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

    // Tokenize with chat template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    Tokenizer tok = load_tokenizer(&text_gf)!!;

    // Build conditioned prompt with chat template
    char[4096] cond_buf;
    usz cond_len = 0;
    String cond_parts = "<|im_start|>user\n";
    for (usz i = 0; i < cond_parts.len; i++) { cond_buf[cond_len] = cond_parts[i]; cond_len++; }
    for (usz i = 0; i < prompt.len; i++) { cond_buf[cond_len] = prompt[i]; cond_len++; }
    String cond_suffix = "<|im_end|>\n<|im_start|>assistant\n";
    for (usz i = 0; i < cond_suffix.len; i++) { cond_buf[cond_len] = cond_suffix[i]; cond_len++; }
    String cond_prompt = (String)cond_buf[0..cond_len - 1];

    uint[] cond_tokens_raw = tok.encode_with_specials(cond_prompt)!!;
    io::printfn("  Cond prompt: %d tokens (with chat template)", cond_tokens_raw.len);

    // Prepend BOS
    uint cond_text_len = (uint)cond_tokens_raw.len + 1;
    uint[] cond_tokens = mem::new_array(uint, cond_text_len);
    cond_tokens[0] = tok.bos_id;
    for (usz i = 0; i < cond_tokens_raw.len; i++) {
        cond_tokens[i + 1] = cond_tokens_raw[i];
    }
    mem::free(cond_tokens_raw);

    // Build unconditioned prompt for CFG (empty user text)
    uint[] uncond_tokens;
    uint uncond_text_len = 0;
    if (use_cfg) {
        String uncond_prompt = "<|im_start|>user\n\n<|im_end|>\n<|im_start|>assistant\n";
        uint[] uncond_tokens_raw = tok.encode_with_specials(uncond_prompt)!!;
        io::printfn("  Uncond prompt: %d tokens (empty chat template)", uncond_tokens_raw.len);

        uncond_text_len = (uint)uncond_tokens_raw.len + 1;
        uncond_tokens = mem::new_array(uint, uncond_text_len);
        uncond_tokens[0] = tok.bos_id;
        for (usz i = 0; i < uncond_tokens_raw.len; i++) {
            uncond_tokens[i + 1] = uncond_tokens_raw[i];
        }
        mem::free(uncond_tokens_raw);
    }

    // Load LLM model for text encoding
    uint max_text_len = cond_text_len;
    if (use_cfg && uncond_text_len > max_text_len) max_text_len = uncond_text_len;
    uint enc_max_seq = max_text_len + 16;
    LlmModel text_model = load_llm_model(ctx, &text_gf, &text_config, &text_names,
        encode_only: true, max_seq_len: enc_max_seq)!!;

    Clock encode_start = clock::now();

    // Encode conditioned text
    Tensor cond_embeddings = text_model.encode_text(cond_tokens)!!;
    io::printfn("  Cond encoded: [%d, %d]", cond_text_len, text_config.dim);

    // Encode unconditioned text for CFG
    Tensor uncond_embeddings;
    if (use_cfg) {
        uncond_embeddings = text_model.encode_text(uncond_tokens)!!;
        io::printfn("  Uncond encoded: [%d, %d]", uncond_text_len, text_config.dim);
    }

    double encode_sec = encode_start.to_now().to_sec();
    io::printfn("  Text encoding complete in %.2fs", encode_sec);

    // Free tokenizer and text model (keep embeddings on GPU)
    mem::free(cond_tokens);
    if (use_cfg) mem::free(uncond_tokens);
    tok.free();
    text_model.free();
    text_gf.free();
    text_mm.destroy();

    // ============================================================
    // Phase 2: Load DiT + VAE
    // ============================================================
    io::printfn("\n[2/5] Loading DiT + VAE...");

    // Load DiT weights
    DiTWeights dit_weights = load_dit_weights(ctx, dit_gf)!!;
    DiTKernels dit_kernels = create_dit_kernels(ctx)!!;

    // Compute padded sequence lengths (multiples of 32)
    uint cond_padded_text = pad_to_multiple(cond_text_len, SEQ_MULTI_OF);
    uint padded_patches = pad_to_multiple(n_patches, SEQ_MULTI_OF);
    uint cond_n_text_pad = cond_padded_text - cond_text_len;
    uint n_img_pad = padded_patches - n_patches;

    uint uncond_padded_text = use_cfg ? pad_to_multiple(uncond_text_len, SEQ_MULTI_OF) : 0;
    uint uncond_n_text_pad = use_cfg ? uncond_padded_text - uncond_text_len : 0;

    // Use max padded_text for activation allocation (both must fit)
    uint max_padded_text = cond_padded_text;
    if (use_cfg && uncond_padded_text > max_padded_text) max_padded_text = uncond_padded_text;
    uint max_seq = padded_patches + max_padded_text;
    DiTActivations dit_acts = allocate_dit_activations(ctx, max_seq)!!;

    DiTModel* dit = mem::alloc(DiTModel);
    dit.weights = dit_weights;
    dit.acts = dit_acts;
    dit.kernels = dit_kernels;
    dit.ctx = ctx;
    dit.n_patches = n_patches;
    dit.text_len = cond_text_len;
    dit.padded_patches = padded_patches;
    dit.padded_text = cond_padded_text;
    dit.latent_h = lat_h;
    dit.latent_w = lat_w;

    // Precompute mRoPE cos/sin tables for conditioned text
    uint patches_w = lat_w / DIT_PATCH_SIZE;
    uint patches_h = lat_h / DIT_PATCH_SIZE;
    Tensor cond_rope_cos_main;
    Tensor cond_rope_sin_main;
    precompute_mrope_tables(ctx, &cond_rope_cos_main, &cond_rope_sin_main,
        n_patches, patches_w, patches_h, cond_text_len, MROPE_MODE_FULL)!!;
    dit.rope_cos_main = cond_rope_cos_main;
    dit.rope_sin_main = cond_rope_sin_main;
    precompute_mrope_tables(ctx, &dit.rope_cos_refiner, &dit.rope_sin_refiner,
        n_patches, patches_w, patches_h, cond_text_len, MROPE_MODE_IMAGE)!!;

    // Context refiner rope tables for conditioned text
    Tensor cond_rope_cos_ctx;
    Tensor cond_rope_sin_ctx;
    precompute_mrope_tables(ctx, &cond_rope_cos_ctx, &cond_rope_sin_ctx,
        n_patches, patches_w, patches_h, cond_text_len, MROPE_MODE_TEXT)!!;

    // Unconditioned mRoPE tables (for CFG)
    Tensor uncond_rope_cos_main;
    Tensor uncond_rope_sin_main;
    Tensor uncond_rope_cos_ctx;
    Tensor uncond_rope_sin_ctx;
    if (use_cfg) {
        precompute_mrope_tables(ctx, &uncond_rope_cos_main, &uncond_rope_sin_main,
            n_patches, patches_w, patches_h, uncond_text_len, MROPE_MODE_FULL)!!;
        precompute_mrope_tables(ctx, &uncond_rope_cos_ctx, &uncond_rope_sin_ctx,
            n_patches, patches_w, patches_h, uncond_text_len, MROPE_MODE_TEXT)!!;
    }

    io::printfn("  DiT loaded: %d patches (pad %d), cond=%d tokens (pad %d)",
        n_patches, n_img_pad, cond_text_len, cond_n_text_pad);
    if (use_cfg) {
        io::printfn("  CFG uncond: %d tokens (pad %d)", uncond_text_len, uncond_n_text_pad);
    }

    // --- Helper: Project text through cap_embedder + pad + context_refiner ---
    // Process conditioned text
    Tensor cond_projected = create_f32_tensor(ctx, { (ulong)cond_padded_text * DIT_DIM, 0, 0, 0 }, 1)!!;
    Tensor norm_scratch = create_f32_tensor(ctx, { (ulong)text_config.dim, 0, 0, 0 }, 1)!!;
    CommandBuffer cmd = ctx.command_buffer;

    // Project cond: RMSNorm + Linear + Pad
    begin_compute(cmd)!!;
    for (uint pos = 0; pos < cond_text_len; pos++) {
        usz in_off = (usz)pos * text_config.dim * 4;
        vk::cmdCopyBuffer(cmd, cond_embeddings.gpu_buffer.buffer, norm_scratch.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)text_config.dim * 4 }});
        compute_barrier(cmd);
        RMSNormPC rms_pc = { .dim = text_config.dim, .eps = 1e-6f };
        dispatch_kernel(cmd, &dit.kernels.shared.rmsnorm,
            { norm_scratch.gpu_buffer.buffer, dit.weights.cap_emb_norm.gpu_buffer.buffer,
              norm_scratch.gpu_buffer.buffer },
            { norm_scratch.size_bytes, dit.weights.cap_emb_norm.size_bytes,
              norm_scratch.size_bytes },
            &rms_pc, 1);
        compute_barrier(cmd);
        vk::cmdCopyBuffer(cmd, norm_scratch.gpu_buffer.buffer, cond_embeddings.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = in_off, .size = (ulong)text_config.dim * 4 }});
        compute_barrier(cmd);
    }
    LinearProjPC cap_pc = { .out_dim = DIT_DIM, .in_dim = text_config.dim, .seq_len = cond_text_len };
    dispatch_kernel(cmd, &dit.kernels.linear_proj,
        { dit.weights.cap_emb_weight.gpu_buffer.buffer, dit.weights.cap_emb_bias.gpu_buffer.buffer,
          cond_embeddings.gpu_buffer.buffer, cond_projected.gpu_buffer.buffer },
        { dit.weights.cap_emb_weight.size_bytes, dit.weights.cap_emb_bias.size_bytes,
          cond_embeddings.size_bytes, cond_projected.size_bytes },
        &cap_pc, cond_text_len * DIT_DIM);
    compute_barrier(cmd);
    if (cond_n_text_pad > 0 && dit.weights.cap_pad_token.size_bytes > 0) {
        for (uint p = 0; p < cond_n_text_pad; p++) {
            vk::cmdCopyBuffer(cmd, dit.weights.cap_pad_token.gpu_buffer.buffer,
                cond_projected.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0,
                    .dstOffset = (ulong)(cond_text_len + p) * DIT_DIM * 4,
                    .size = (ulong)DIT_DIM * 4 }});
        }
        compute_barrier(cmd);
    }
    submit_and_wait(ctx)!!;
    cond_embeddings.free();

    // Context refiner on cond text
    dit.rope_cos_context = cond_rope_cos_ctx;
    dit.rope_sin_context = cond_rope_sin_ctx;
    begin_compute(cmd)!!;
    for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
        dit_transformer_layer(dit, cmd, &dit.weights.context_refiner[l],
            &cond_projected, cond_padded_text, false,
            &dit.rope_cos_context, &dit.rope_sin_context)!!;
    }
    submit_and_wait(ctx)!!;
    io::printfn("  Cond context refiner complete (padded=%d).", cond_padded_text);

    // Process unconditioned text for CFG
    Tensor uncond_projected;
    if (use_cfg) {
        uncond_projected = create_f32_tensor(ctx, { (ulong)uncond_padded_text * DIT_DIM, 0, 0, 0 }, 1)!!;

        // Project uncond: RMSNorm + Linear + Pad
        begin_compute(cmd)!!;
        for (uint pos = 0; pos < uncond_text_len; pos++) {
            usz in_off = (usz)pos * text_config.dim * 4;
            vk::cmdCopyBuffer(cmd, uncond_embeddings.gpu_buffer.buffer, norm_scratch.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)text_config.dim * 4 }});
            compute_barrier(cmd);
            RMSNormPC rms_pc = { .dim = text_config.dim, .eps = 1e-6f };
            dispatch_kernel(cmd, &dit.kernels.shared.rmsnorm,
                { norm_scratch.gpu_buffer.buffer, dit.weights.cap_emb_norm.gpu_buffer.buffer,
                  norm_scratch.gpu_buffer.buffer },
                { norm_scratch.size_bytes, dit.weights.cap_emb_norm.size_bytes,
                  norm_scratch.size_bytes },
                &rms_pc, 1);
            compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, norm_scratch.gpu_buffer.buffer, uncond_embeddings.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = in_off, .size = (ulong)text_config.dim * 4 }});
            compute_barrier(cmd);
        }
        LinearProjPC uncond_cap_pc = { .out_dim = DIT_DIM, .in_dim = text_config.dim, .seq_len = uncond_text_len };
        dispatch_kernel(cmd, &dit.kernels.linear_proj,
            { dit.weights.cap_emb_weight.gpu_buffer.buffer, dit.weights.cap_emb_bias.gpu_buffer.buffer,
              uncond_embeddings.gpu_buffer.buffer, uncond_projected.gpu_buffer.buffer },
            { dit.weights.cap_emb_weight.size_bytes, dit.weights.cap_emb_bias.size_bytes,
              uncond_embeddings.size_bytes, uncond_projected.size_bytes },
            &uncond_cap_pc, uncond_text_len * DIT_DIM);
        compute_barrier(cmd);
        if (uncond_n_text_pad > 0 && dit.weights.cap_pad_token.size_bytes > 0) {
            for (uint p = 0; p < uncond_n_text_pad; p++) {
                vk::cmdCopyBuffer(cmd, dit.weights.cap_pad_token.gpu_buffer.buffer,
                    uncond_projected.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0,
                        .dstOffset = (ulong)(uncond_text_len + p) * DIT_DIM * 4,
                        .size = (ulong)DIT_DIM * 4 }});
            }
            compute_barrier(cmd);
        }
        submit_and_wait(ctx)!!;
        uncond_embeddings.free();

        // Context refiner on uncond text
        dit.rope_cos_context = uncond_rope_cos_ctx;
        dit.rope_sin_context = uncond_rope_sin_ctx;
        begin_compute(cmd)!!;
        for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
            dit_transformer_layer(dit, cmd, &dit.weights.context_refiner[l],
                &uncond_projected, uncond_padded_text, false,
                &dit.rope_cos_context, &dit.rope_sin_context)!!;
        }
        submit_and_wait(ctx)!!;
        io::printfn("  Uncond context refiner complete (padded=%d).", uncond_padded_text);
    }
    norm_scratch.free();

    // Load VAE (TAESD or full Flux VAE)
    DiffusionKernels diff_kernels = create_diffusion_kernels(ctx)!!;
    FluxVaeDecoder vae_decoder;
    FluxVaeActivations vae_acts;
    TAESDDecoder taesd_decoder;
    TAESDActivations taesd_acts;
    bool has_vae = false;
    bool use_taesd = false;

    if (params.taesd_path.len > 0) {
        SafetensorsFile sf = safetensors_open(params.taesd_path)!!;
        taesd_decoder = load_taesd_decoder(ctx, &sf, &diff_kernels)!!;
        sf.close();
        taesd_acts = allocate_taesd_activations(ctx, img_size)!!;
        has_vae = true;
        use_taesd = true;
        io::printfn("  TAESD loaded from %s", params.taesd_path);
    } else if (params.vae_path.len > 0) {
        SafetensorsFile sf = safetensors_open(params.vae_path)!!;
        vae_decoder = load_flux_vae_decoder(ctx, &sf, &diff_kernels)!!;
        sf.close();
        vae_acts = allocate_flux_vae_activations(ctx, img_size)!!;
        has_vae = true;
        io::printfn("  Flux VAE loaded from %s", params.vae_path);
    } else {
        io::printfn("  Warning: No VAE path specified (--vae or --taesd). Will output raw latent.");
    }

    io::printfn("  GPU memory: %d MB allocated, %d MB fragmented (reusable)",
        ctx.allocator.total_allocated() / (1024 * 1024),
        ctx.allocator.total_fragmented() / (1024 * 1024));

    // ============================================================
    // Phase 3: Generate or load random latent
    // ============================================================
    uint latent_elems = DIT_LATENT_CHANNELS * lat_h * lat_w;
    float[] latent_data = mem::new_array(float, latent_elems);
    
    // Try to load pre-generated latent from file (for debugging/comparison)
    String latent_file = "debug_output/initial_latent.bin";
    bool loaded_from_file = false;
    if (try f = io::file::open(latent_file, "rb")) {
        // Check file size
        if (try file_size = f.seek(0, io::Seek.END)) {
          f.seek(0, io::Seek.SET)!!;
        
          // Expected size: header (16 bytes) + 16*64*64*4 = 262160 bytes
          long expected_size = 16 + (long)latent_elems * 4;
        
          if ((long)file_size == expected_size) {
            io::printfn("\n[3/5] Loading pre-generated latent from %s...", latent_file);
            // Skip header
            char[16] header;
            f.read(header[:16])!!;
            // Read float data
            char[] raw_data = mem::new_array(char, (usz)latent_elems * 4);
            f.read(raw_data)!!;
            f.close()!!;
            
            // Copy to latent_data
            mem::copy(latent_data.ptr, raw_data.ptr, (usz)latent_elems * 4);
            mem::free(raw_data);
            loaded_from_file = true;
            
            // Verify stats
            float dmin = latent_data[0];
            float dmax = latent_data[0];
            float dsum = 0;
            for (uint i = 0; i < latent_elems; i++) {
                float v = latent_data[i];
                if (v < dmin) dmin = v;
                if (v > dmax) dmax = v;
                dsum += v;
            }
            io::printfn("  Loaded latent: min=%.4f max=%.4f mean=%.4f", dmin, dmax, dsum / latent_elems);
          } else {
            io::printfn("  Warning: latent file size mismatch (%d vs %d expected), generating fresh latent", file_size, expected_size);
            f.close()!!;
          }
        } else {
          io::printfn("  Warning: could not determine file size, generating fresh latent");
          f.close()!!;
        }
    }
    
    if (!loaded_from_file) {
        io::printfn("\n[3/5] Generating random latent...");
        generate_random_latent(latent_data, params.seed);
    }

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

    // Debug: Save initial latent for comparison
    debug_save_npy(ctx, dit.acts.latent.gpu_buffer.buffer, latent_elems, "debug_output/c3_latent_initial.bin")!!;

    // ============================================================
    // Phase 4: Denoising loop (with CFG)
    // ============================================================
    io::printfn("\n[4/5] Denoising (%d steps, CFG=%.1f)...", params.num_steps, cfg_scale);

    FlowMatchingState flow = init_flow_matching(params.num_steps);

    // Allocate CPU buffer for CFG velocity combining
    float[] cfg_cond_vel;
    float[] cfg_uncond_vel;
    if (use_cfg) {
        cfg_cond_vel = mem::new_array(float, latent_elems);
        cfg_uncond_vel = mem::new_array(float, latent_elems);
    }

    // Staging buffer for velocity readback/upload during CFG
    vk::Memory cfg_staging;
    if (use_cfg) {
        cfg_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: (usz)latent_elems * 4,
        )!!;
    }

    // Save original latent for re-use between cond/uncond forward passes
    vk::Memory latent_save;
    if (use_cfg) {
        latent_save = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: null, data_size: (usz)latent_elems * 4,
        )!!;
    }

    Clock denoise_start = clock::now();
    for (uint step = 0; step < params.num_steps; step++) {
        float t = flow.get_timestep(step);
        float dt = flow.get_dt(step);
        io::printfn("  Step %d/%d (t=%.3f, dt=%.3f)", step + 1, params.num_steps, t, dt);

        Clock step_start = clock::now();
        if (step == 0) { dit_profile_layer = true; }

        if (use_cfg) {
            // Save current latent (both cond and uncond forward start from same latent)
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer save_cmd) {
                vk::cmdCopyBuffer(save_cmd, dit.acts.latent.gpu_buffer.buffer, latent_save.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            // --- Conditioned forward ---
            dit.text_len = cond_text_len;
            dit.padded_text = cond_padded_text;
            dit.rope_cos_main = cond_rope_cos_main;
            dit.rope_sin_main = cond_rope_sin_main;
            dit.forward(&cond_projected, t)!!;

            // Read cond velocity to CPU
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer read_cmd) {
                vk::cmdCopyBuffer(read_cmd, dit.acts.velocity.gpu_buffer.buffer, cfg_staging.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;
            float* stg_ptr = (float*)cfg_staging.data();
            for (uint i = 0; i < latent_elems; i++) cfg_cond_vel[i] = stg_ptr[i];

            // Restore latent for uncond forward
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer restore_cmd) {
                vk::cmdCopyBuffer(restore_cmd, latent_save.buffer, dit.acts.latent.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            // --- Unconditioned forward ---
            dit.text_len = uncond_text_len;
            dit.padded_text = uncond_padded_text;
            dit.rope_cos_main = uncond_rope_cos_main;
            dit.rope_sin_main = uncond_rope_sin_main;
            dit_profile_layer = false;  // Don't profile uncond pass
            dit.forward(&uncond_projected, t)!!;

            // Read uncond velocity to CPU
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer read_cmd2) {
                vk::cmdCopyBuffer(read_cmd2, dit.acts.velocity.gpu_buffer.buffer, cfg_staging.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;
            stg_ptr = (float*)cfg_staging.data();
            for (uint i = 0; i < latent_elems; i++) cfg_uncond_vel[i] = stg_ptr[i];

            // CFG combine: velocity = uncond + cfg_scale * (cond - uncond)
            for (uint i = 0; i < latent_elems; i++) {
                stg_ptr[i] = cfg_uncond_vel[i] + cfg_scale * (cfg_cond_vel[i] - cfg_uncond_vel[i]);
            }

            // Upload combined velocity back to GPU
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer write_cmd) {
                vk::cmdCopyBuffer(write_cmd, cfg_staging.buffer, dit.acts.velocity.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            // Restore latent for Euler step
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer restore_cmd2) {
                vk::cmdCopyBuffer(restore_cmd2, latent_save.buffer, dit.acts.latent.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;
        } else {
            // No CFG: single forward pass
            dit.forward(&cond_projected, t)!!;
        }

        // Debug: dump velocity and latent stats before Euler step
        {
            debug_readback_stats(ctx, dit.acts.latent.gpu_buffer.buffer, latent_elems, "  latent (pre-euler)")!!;
            debug_readback_stats(ctx, dit.acts.velocity.gpu_buffer.buffer, latent_elems, "  velocity (cfg'd)")!!;
            // Dump first 10 velocity values for comparison with KoboldCpp
            vk::Memory vel_stg = vk::new_buffer(
                allocator: &ctx.allocator,
                usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data: null, data_size: 40,
            )!!;
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer vel_cmd) {
                vk::cmdCopyBuffer(vel_cmd, dit.acts.velocity.gpu_buffer.buffer, vel_stg.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = 40 }});
            }!!;
            float* vp = (float*)vel_stg.data();
            io::printf("  velocity[0..9]:");
            for (uint vi = 0; vi < 10; vi++) io::printf(" %.4f", vp[vi]);
            io::printn("");
            vel_stg.free();
        }

        // Euler step: latent += dt * velocity
        begin_compute(cmd)!!;
        FlowEulerPC euler_pc = { .n = latent_elems, .dt = dt };
        dispatch_kernel(cmd, &dit.kernels.flow_euler_step,
            { dit.acts.latent.gpu_buffer.buffer, dit.acts.velocity.gpu_buffer.buffer },
            { dit.acts.latent.size_bytes, dit.acts.velocity.size_bytes },
            &euler_pc, ceil_div(latent_elems, 256));
        submit_and_wait(ctx)!!;

        // Debug: latent after Euler step
        debug_readback_stats(ctx, dit.acts.latent.gpu_buffer.buffer, latent_elems, "  latent (post-euler)")!!;

        double step_sec = step_start.to_now().to_sec();
        io::printfn("  Step %d done in %.2fs", step + 1, step_sec);
    }
    double denoise_sec = denoise_start.to_now().to_sec();
    io::printfn("  Denoising complete in %.2fs", denoise_sec);

    // Debug: readback latent stats + save raw latent as PNG for visualization
    debug_readback_stats(ctx, dit.acts.latent.gpu_buffer.buffer, latent_elems, "final latent")!!;
    {
        // Save channels 0,1,2 of denoised latent as RGB PNG (64x64)
        vk::Memory lat_dbg = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: (usz)latent_elems * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer dbg_cmd) {
            vk::cmdCopyBuffer(dbg_cmd, dit.acts.latent.gpu_buffer.buffer, lat_dbg.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;
        float* lat_ptr = (float*)lat_dbg.data();
        uint spatial = lat_h * lat_w;
        // Find min/max across channels 0-2 for normalization
        float vmin = lat_ptr[0]; float vmax = lat_ptr[0];
        for (uint ch = 0; ch < 3; ch++) {
            for (uint i = 0; i < spatial; i++) {
                float v = lat_ptr[ch * spatial + i];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
        io::printfn("  [DBG] raw latent ch0-2: min=%.4f max=%.4f", vmin, vmax);
        float range = vmax - vmin;
        if (range < 1e-6f) range = 1.0f;
        // Write PNG using same pattern as save_png
        Image lat_img = {
            .width = lat_w, .height = lat_h, .channels = 3,
            .data = mem::new_array(float, (usz)3 * spatial),
        };
        for (uint ch = 0; ch < 3; ch++) {
            for (uint i = 0; i < spatial; i++) {
                float v = (lat_ptr[ch * spatial + i] - vmin) / range;
                if (v < 0) v = 0; if (v > 1) v = 1;
                lat_img.data[ch * spatial + i] = v;
            }
        }
        save_png(&lat_img, "latent_raw.png");
        mem::free(lat_img.data);
        io::printfn("  Raw latent saved: latent_raw.png (%dx%d)", lat_w, lat_h);
        lat_dbg.free();
    }

    // Cleanup CFG buffers
    if (use_cfg) {
        cfg_staging.free();
        latent_save.free();
        mem::free(cfg_cond_vel);
        mem::free(cfg_uncond_vel);
    }

    flow.free();

    // ============================================================
    // Phase 5: VAE decode
    // ============================================================
    io::printfn("\n[5/5] VAE decoding...");

    if (has_vae) {
        Tensor* decode_output;

        if (use_taesd) {
            // TAESD: no latent scaling needed (tanh clamp handles it)
            // Copy latent to TAESD input buffer
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer copy_cmd) {
                vk::cmdCopyBuffer(copy_cmd, dit.acts.latent.gpu_buffer.buffer,
                    taesd_acts.buf_a.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            taesd_decoder.forward(&taesd_acts, lat_h, lat_w)!!;
            decode_output = &taesd_acts.buf_a;
        } else {
            // Flux VAE: per-channel denormalization before decode
            // Formula: value = value * stds[c] / scale_factor + means[c]
            // From KoboldCpp's process_latent_out() for Z-Image (16 channels)
            const float SCALE_FACTOR = 0.3611f;
            const float[16] LATENT_MEANS = {
                -0.7571f, -0.7089f, -0.9113f,  0.1075f, -0.1745f,  0.9653f, -0.1517f,  1.5508f,
                 0.4134f, -0.0715f,  0.5517f, -0.3632f, -0.1922f, -0.9497f,  0.2503f, -0.2921f,
            };
            const float[16] LATENT_STDS = {
                2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f,
            };

            // Read back latent from GPU
            uint spatial = lat_h * lat_w;
            vk::Memory lat_readback = vk::new_buffer(
                allocator: &ctx.allocator,
                usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data: null, data_size: (usz)latent_elems * 4,
            )!!;
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer rb_cmd) {
                vk::cmdCopyBuffer(rb_cmd, dit.acts.latent.gpu_buffer.buffer, lat_readback.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            // Apply per-channel denormalization on CPU
            float* lat_cpu = (float*)lat_readback.data();
            for (uint ch = 0; ch < DIT_LATENT_CHANNELS; ch++) {
                float scale = LATENT_STDS[ch] / SCALE_FACTOR;
                float mean = LATENT_MEANS[ch];
                uint offset = ch * spatial;
                for (uint i = 0; i < spatial; i++) {
                    lat_cpu[offset + i] = lat_cpu[offset + i] * scale + mean;
                }
            }

            io::printfn("  [DEBUG] post-denorm latent ch0[0..3]: %.4f %.4f %.4f %.4f",
                lat_cpu[0], lat_cpu[1], lat_cpu[2], lat_cpu[3]);

            // Upload denormalized latent to VAE input buffer
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer copy_cmd) {
                vk::cmdCopyBuffer(copy_cmd, lat_readback.buffer,
                    vae_acts.buf_a.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;
            lat_readback.free();

            vae_decoder.forward(&vae_acts, lat_h, lat_w)!!;
            decode_output = &vae_acts.buf_a;
        }

        // Download image
        Image img = download_image_from_gpu(ctx, decode_output, img_size, img_size, 3)!!;

        // Debug: check pixel stats
        {
            float p_min = img.data[0]; float p_max = img.data[0]; float p_sum = 0;
            uint p_nan = 0;
            usz total_px = (usz)img_size * img_size * 3;
            for (usz i = 0; i < total_px; i++) {
                float p = img.data[i];
                if (p != p) { p_nan++; } else { if (p < p_min) p_min = p; if (p > p_max) p_max = p; p_sum += p; }
            }
            io::printfn("  [DEBUG] pixels: min=%.4f max=%.4f mean=%.4f nan=%d", p_min, p_max, p_sum / (float)total_px, p_nan);
        }

        bool ok = save_png(&img, params.output_path);
        img.free();

        // Cleanup
        if (use_taesd) {
            taesd_decoder.free();
            taesd_acts.free();
        } else {
            vae_decoder.free();
            vae_acts.free();
        }
        diff_kernels.free(ctx.device);
        cond_projected.free();
        if (use_cfg) uncond_projected.free();
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
        cond_projected.free();
        if (use_cfg) uncond_projected.free();
        diff_kernels.free(ctx.device);
        dit.free();
        mem::free(dit);

        double total_sec = total_start.to_now().to_sec();
        io::printfn("\n=== Pipeline Complete (no VAE) ===");
        io::printfn("  Total time: %.2fs", total_sec);
        return true;
    }
}
