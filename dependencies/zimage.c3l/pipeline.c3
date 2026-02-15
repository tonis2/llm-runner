module zimage;

import vk;
import llm;
import llm::pipelines;
import llm_text;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::random;
import std::time::clock;
import image;
import image::png;

struct ZImageState (llm::pipelines::Pipeline) {
    bool loaded;
    String model_path;
    String text_model_path;
    String vae_path;
    String taesd_path;
}

fn void? ZImageState.load(ZImageState* self, llm::DeviceContext* ctx, String model_path, llm::pipelines::PipelineOptions* opts) @dynamic {
    self.loaded = true;
    self.model_path = model_path;
    self.text_model_path = opts.text_model_path;
    self.vae_path = opts.vae_path;
    self.taesd_path = opts.taesd_path;
}

fn image::Image? ZImageState.generate(ZImageState* self, llm::DeviceContext* ctx, llm::pipelines::GenerationInputs* inputs) @dynamic {
    float cfg_scale = inputs.cfg_scale;
    bool use_cfg = cfg_scale > 1.0f;
    bool is_img2img = inputs.input_image != null;

    io::printfn("\n=== Z-Image Turbo Pipeline ===");
    io::printfn("  Prompt: \"%s\"", inputs.prompt);
    io::printfn("  Mode: %s", is_img2img ? "img2img" : "txt2img");
    io::printfn("  Steps: %d, Seed: %d, Size: %d, CFG: %.1f",
        inputs.num_steps, inputs.seed, inputs.image_size, cfg_scale);
    if (is_img2img) {
        io::printfn("  Strength: %.2f", inputs.strength);
    }

    Clock total_start = clock::now();

    uint img_size = inputs.image_size;
    uint lat_h = img_size / 8;
    uint lat_w = img_size / 8;
    uint n_patches = (lat_h / DIT_PATCH_SIZE) * (lat_w / DIT_PATCH_SIZE);
    uint latent_elems = DIT_LATENT_CHANNELS * lat_h * lat_w;

    // ---- Phase 1: Text Encoding via LLM ----
    io::printfn("\n[1/5] Text encoding (LLM)...");

    if (self.text_model_path.len == 0) {
        io::printfn("Error: text_model_path required for DiT pipeline");
        return llm::pipelines::PIPELINE_INVALID_INPUT~;
    }

    mmap::FileMmap text_mm = file::mmap_open(self.text_model_path, "rb")!!;
    char[] text_data = text_mm.bytes();
    io::printfn("  Text model: %s (%d bytes)", self.text_model_path, text_data.len);

    llm::GGUFFile text_gf = llm::gguf_parse(text_data)!!;

    String arch_name = llm::detect_architecture(&text_gf);
    io::printfn("  Text encoder architecture: %s", arch_name);

    String config_json;
    if (arch_name == "qwen3") {
        config_json = (String)&llm_text::QWEN3_CONFIG_JSON;
    } else if (arch_name == "qwen2") {
        config_json = (String)&llm_text::QWEN2_CONFIG_JSON;
    } else {
        config_json = (String)&llm_text::QWEN3_CONFIG_JSON;
    }

    llm::ModelConfig text_config = llm::load_arch_config(config_json, &text_gf)!!;
    llm::WeightNames text_names = llm::load_weight_names(config_json)!!;

    llm::Tokenizer tok = llm::load_tokenizer(&text_gf)!!;

    // Build conditioned prompt with chat template
    char[4096] cond_buf;
    usz cond_len = 0;
    String cond_parts = "<|im_start|>user\n";
    for (usz i = 0; i < cond_parts.len; i++) { cond_buf[cond_len] = cond_parts[i]; cond_len++; }
    for (usz i = 0; i < inputs.prompt.len; i++) { cond_buf[cond_len] = inputs.prompt[i]; cond_len++; }
    String cond_suffix = "<|im_end|>\n<|im_start|>assistant\n";
    for (usz i = 0; i < cond_suffix.len; i++) { cond_buf[cond_len] = cond_suffix[i]; cond_len++; }
    String cond_prompt = (String)cond_buf[0..cond_len - 1];

    uint[] cond_tokens_raw = tok.encode_with_specials(cond_prompt)!!;
    io::printfn("  Cond prompt: %d tokens (with chat template)", cond_tokens_raw.len);

    uint cond_text_len = (uint)cond_tokens_raw.len + 1;
    uint[] cond_tokens = mem::new_array(uint, cond_text_len);
    cond_tokens[0] = tok.bos_id;
    for (usz i = 0; i < cond_tokens_raw.len; i++) {
        cond_tokens[i + 1] = cond_tokens_raw[i];
    }
    mem::free(cond_tokens_raw);

    // Build unconditioned prompt for CFG
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
    llm_text::LlmModel text_model = llm_text::load_llm_model(ctx, &text_gf, &text_config, &text_names,
        encode_only: true, max_seq_len: enc_max_seq)!!;

    Clock encode_start = clock::now();

    llm::Tensor cond_embeddings = text_model.encode_text(cond_tokens)!!;
    io::printfn("  Cond encoded: [%d, %d]", cond_text_len, text_config.dim);

    llm::Tensor uncond_embeddings;
    if (use_cfg) {
        uncond_embeddings = text_model.encode_text(uncond_tokens)!!;
        io::printfn("  Uncond encoded: [%d, %d]", uncond_text_len, text_config.dim);
    }

    double encode_sec = encode_start.to_now().to_sec();
    io::printfn("  Text encoding complete in %.2fs", encode_sec);

    mem::free(cond_tokens);
    if (use_cfg) mem::free(uncond_tokens);
    tok.free();
    text_model.free();
    text_gf.free();
    text_mm.destroy();

    // ---- Phase 2: Load DiT + VAE ----
    io::printfn("\n[2/5] Loading DiT + VAE...");

    mmap::FileMmap dit_mm = file::mmap_open(self.model_path, "rb")!!;
    char[] dit_data = dit_mm.bytes();
    llm::GGUFFile dit_gf = llm::gguf_parse(dit_data)!!;

    DiTWeights dit_weights = load_dit_weights(ctx, &dit_gf)!!;
    DiTKernels dit_kernels = create_dit_kernels(ctx)!!;

    uint cond_padded_text = pad_to_multiple(cond_text_len, SEQ_MULTI_OF);
    uint padded_patches = pad_to_multiple(n_patches, SEQ_MULTI_OF);
    uint cond_n_text_pad = cond_padded_text - cond_text_len;
    uint n_img_pad = padded_patches - n_patches;

    uint uncond_padded_text = use_cfg ? pad_to_multiple(uncond_text_len, SEQ_MULTI_OF) : 0;
    uint uncond_n_text_pad = use_cfg ? uncond_padded_text - uncond_text_len : 0;

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

    // Precompute mRoPE tables
    uint patches_w = lat_w / DIT_PATCH_SIZE;
    uint patches_h = lat_h / DIT_PATCH_SIZE;
    llm::Tensor cond_rope_cos_main;
    llm::Tensor cond_rope_sin_main;
    precompute_mrope_tables(ctx, &cond_rope_cos_main, &cond_rope_sin_main,
        n_patches, patches_w, patches_h, cond_text_len, MROPE_MODE_FULL)!!;
    dit.rope_cos_main = cond_rope_cos_main;
    dit.rope_sin_main = cond_rope_sin_main;
    precompute_mrope_tables(ctx, &dit.rope_cos_refiner, &dit.rope_sin_refiner,
        n_patches, patches_w, patches_h, cond_text_len, MROPE_MODE_IMAGE)!!;

    llm::Tensor cond_rope_cos_ctx;
    llm::Tensor cond_rope_sin_ctx;
    precompute_mrope_tables(ctx, &cond_rope_cos_ctx, &cond_rope_sin_ctx,
        n_patches, patches_w, patches_h, cond_text_len, MROPE_MODE_TEXT)!!;

    llm::Tensor uncond_rope_cos_main;
    llm::Tensor uncond_rope_sin_main;
    llm::Tensor uncond_rope_cos_ctx;
    llm::Tensor uncond_rope_sin_ctx;
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

    // Project text through cap_embedder + context_refiner
    llm::Tensor cond_projected = llm::create_f32_tensor(ctx, { (ulong)cond_padded_text * DIT_DIM, 0, 0, 0 }, 1)!!;
    llm::Tensor norm_scratch = llm::create_f32_tensor(ctx, { (ulong)text_config.dim, 0, 0, 0 }, 1)!!;
    vk::CommandBuffer cmd = ctx.command_buffer;

    llm::begin_compute(cmd)!!;
    for (uint pos = 0; pos < cond_text_len; pos++) {
        usz in_off = (usz)pos * text_config.dim * 4;
        vk::cmdCopyBuffer(cmd, cond_embeddings.gpu_buffer.buffer, norm_scratch.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)text_config.dim * 4 }});
        llm::compute_barrier(cmd);
        RMSNormPC rms_pc = { .dim = text_config.dim, .eps = 1e-6f };
        llm::dispatch_kernel(cmd, &dit.kernels.shared.rmsnorm,
            { norm_scratch.gpu_buffer.buffer, dit.weights.cap_emb_norm.gpu_buffer.buffer,
              norm_scratch.gpu_buffer.buffer },
            { norm_scratch.size_bytes, dit.weights.cap_emb_norm.size_bytes,
              norm_scratch.size_bytes },
            &rms_pc, 1);
        llm::compute_barrier(cmd);
        vk::cmdCopyBuffer(cmd, norm_scratch.gpu_buffer.buffer, cond_embeddings.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = in_off, .size = (ulong)text_config.dim * 4 }});
        llm::compute_barrier(cmd);
    }
    LinearProjPC cap_pc = { .out_dim = DIT_DIM, .in_dim = text_config.dim, .seq_len = cond_text_len };
    llm::dispatch_kernel(cmd, &dit.kernels.linear_proj,
        { dit.weights.cap_emb_weight.gpu_buffer.buffer, dit.weights.cap_emb_bias.gpu_buffer.buffer,
          cond_embeddings.gpu_buffer.buffer, cond_projected.gpu_buffer.buffer },
        { dit.weights.cap_emb_weight.size_bytes, dit.weights.cap_emb_bias.size_bytes,
          cond_embeddings.size_bytes, cond_projected.size_bytes },
        &cap_pc, cond_text_len * DIT_DIM);
    llm::compute_barrier(cmd);
    if (cond_n_text_pad > 0 && dit.weights.cap_pad_token.size_bytes > 0) {
        for (uint p = 0; p < cond_n_text_pad; p++) {
            vk::cmdCopyBuffer(cmd, dit.weights.cap_pad_token.gpu_buffer.buffer,
                cond_projected.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0,
                    .dstOffset = (ulong)(cond_text_len + p) * DIT_DIM * 4,
                    .size = (ulong)DIT_DIM * 4 }});
        }
        llm::compute_barrier(cmd);
    }
    llm::submit_and_wait(ctx)!!;
    cond_embeddings.free();

    // Context refiner on cond text
    dit.rope_cos_context = cond_rope_cos_ctx;
    dit.rope_sin_context = cond_rope_sin_ctx;
    llm::begin_compute(cmd)!!;
    for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
        dit_transformer_layer(dit, cmd, &dit.weights.context_refiner[l],
            &cond_projected, cond_padded_text, false,
            &dit.rope_cos_context, &dit.rope_sin_context)!!;
    }
    llm::submit_and_wait(ctx)!!;
    io::printfn("  Cond context refiner complete (padded=%d).", cond_padded_text);

    // Process unconditioned text for CFG
    llm::Tensor uncond_projected;
    if (use_cfg) {
        uncond_projected = llm::create_f32_tensor(ctx, { (ulong)uncond_padded_text * DIT_DIM, 0, 0, 0 }, 1)!!;

        llm::begin_compute(cmd)!!;
        for (uint pos = 0; pos < uncond_text_len; pos++) {
            usz in_off = (usz)pos * text_config.dim * 4;
            vk::cmdCopyBuffer(cmd, uncond_embeddings.gpu_buffer.buffer, norm_scratch.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)text_config.dim * 4 }});
            llm::compute_barrier(cmd);
            RMSNormPC rms_pc = { .dim = text_config.dim, .eps = 1e-6f };
            llm::dispatch_kernel(cmd, &dit.kernels.shared.rmsnorm,
                { norm_scratch.gpu_buffer.buffer, dit.weights.cap_emb_norm.gpu_buffer.buffer,
                  norm_scratch.gpu_buffer.buffer },
                { norm_scratch.size_bytes, dit.weights.cap_emb_norm.size_bytes,
                  norm_scratch.size_bytes },
                &rms_pc, 1);
            llm::compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, norm_scratch.gpu_buffer.buffer, uncond_embeddings.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = in_off, .size = (ulong)text_config.dim * 4 }});
            llm::compute_barrier(cmd);
        }
        LinearProjPC uncond_cap_pc = { .out_dim = DIT_DIM, .in_dim = text_config.dim, .seq_len = uncond_text_len };
        llm::dispatch_kernel(cmd, &dit.kernels.linear_proj,
            { dit.weights.cap_emb_weight.gpu_buffer.buffer, dit.weights.cap_emb_bias.gpu_buffer.buffer,
              uncond_embeddings.gpu_buffer.buffer, uncond_projected.gpu_buffer.buffer },
            { dit.weights.cap_emb_weight.size_bytes, dit.weights.cap_emb_bias.size_bytes,
              uncond_embeddings.size_bytes, uncond_projected.size_bytes },
            &uncond_cap_pc, uncond_text_len * DIT_DIM);
        llm::compute_barrier(cmd);
        if (uncond_n_text_pad > 0 && dit.weights.cap_pad_token.size_bytes > 0) {
            for (uint p = 0; p < uncond_n_text_pad; p++) {
                vk::cmdCopyBuffer(cmd, dit.weights.cap_pad_token.gpu_buffer.buffer,
                    uncond_projected.gpu_buffer.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0,
                        .dstOffset = (ulong)(uncond_text_len + p) * DIT_DIM * 4,
                        .size = (ulong)DIT_DIM * 4 }});
            }
            llm::compute_barrier(cmd);
        }
        llm::submit_and_wait(ctx)!!;
        uncond_embeddings.free();

        // Context refiner on uncond text
        dit.rope_cos_context = uncond_rope_cos_ctx;
        dit.rope_sin_context = uncond_rope_sin_ctx;
        llm::begin_compute(cmd)!!;
        for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
            dit_transformer_layer(dit, cmd, &dit.weights.context_refiner[l],
                &uncond_projected, uncond_padded_text, false,
                &dit.rope_cos_context, &dit.rope_sin_context)!!;
        }
        llm::submit_and_wait(ctx)!!;
        io::printfn("  Uncond context refiner complete (padded=%d).", uncond_padded_text);
    }
    norm_scratch.free();

    // Load VAE/TAESD
    pipelines::DiffusionKernels diff_kernels = pipelines::create_diffusion_kernels(ctx)!!;
    FluxVaeDecoder vae_decoder;
    FluxVaeActivations vae_acts;
    TAESDDecoder taesd_decoder;
    TAESDActivations taesd_acts;
    bool has_vae = false;
    bool use_taesd = false;

    if (self.taesd_path.len > 0) {
        llm::SafetensorsFile sf = llm::safetensors_open(self.taesd_path)!!;
        taesd_decoder = load_taesd_decoder(ctx, &sf, &diff_kernels)!!;
        sf.close();
        taesd_acts = allocate_taesd_activations(ctx, img_size)!!;
        has_vae = true;
        use_taesd = true;
        io::printfn("  TAESD loaded from %s", self.taesd_path);
    } else if (self.vae_path.len > 0) {
        llm::SafetensorsFile sf = llm::safetensors_open(self.vae_path)!!;
        vae_decoder = load_flux_vae_decoder(ctx, &sf, &diff_kernels)!!;
        sf.close();
        vae_acts = allocate_flux_vae_activations(ctx, img_size)!!;
        has_vae = true;
        io::printfn("  Flux VAE loaded from %s", self.vae_path);
    } else {
        io::printfn("  Warning: No VAE path specified. Will output raw latent.");
    }

    io::printfn("  GPU memory: %d MB allocated, %d MB fragmented",
        ctx.allocator.total_allocated() / (1024 * 1024),
        ctx.allocator.total_fragmented() / (1024 * 1024));

    // ---- Phase 3: Generate or encode initial latent ----
    float[] latent_data = mem::new_array(float, latent_elems);

    if (is_img2img) {
        io::printfn("\n[3/5] Encoding input image...");
        io::printfn("  Warning: img2img encoding not fully implemented, using random latent");
        llm::pipelines::generate_random_latent(latent_data, inputs.seed);
    } else {
        io::printfn("\n[3/5] Generating random latent...");
        llm::pipelines::generate_random_latent(latent_data, inputs.seed);
    }

    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: latent_data.ptr,
        data_size: (usz)latent_elems * 4,
    )!!;

    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer xfer_cmd) {
        vk::cmdCopyBuffer(xfer_cmd, staging.buffer, dit.acts.latent.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;
    staging.free();
    mem::free(latent_data);

    // ---- Phase 4: Denoising loop ----
    io::printfn("\n[4/5] Denoising (%d steps, CFG=%.1f)...", inputs.num_steps, cfg_scale);

    llm::pipelines::FlowMatchingState flow = llm::pipelines::init_flow_matching(inputs.num_steps);

    float[] cfg_cond_vel;
    float[] cfg_uncond_vel;
    if (use_cfg) {
        cfg_cond_vel = mem::new_array(float, latent_elems);
        cfg_uncond_vel = mem::new_array(float, latent_elems);
    }

    vk::Memory cfg_staging;
    if (use_cfg) {
        cfg_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: (usz)latent_elems * 4,
        )!!;
    }

    vk::Memory latent_save;
    if (use_cfg) {
        latent_save = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: null, data_size: (usz)latent_elems * 4,
        )!!;
    }

    for (uint step = 0; step < inputs.num_steps; step++) {
        float t = flow.get_timestep(step);
        float dt = flow.get_dt(step);
        io::printfn("  Step %d/%d (t=%.3f, dt=%.3f)", step + 1, inputs.num_steps, t, dt);

        if (use_cfg) {
            // Save current latent
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer save_cmd) {
                vk::cmdCopyBuffer(save_cmd, dit.acts.latent.gpu_buffer.buffer, latent_save.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            // Conditioned forward
            dit.text_len = cond_text_len;
            dit.padded_text = cond_padded_text;
            dit.rope_cos_main = cond_rope_cos_main;
            dit.rope_sin_main = cond_rope_sin_main;
            dit.forward(&cond_projected, t)!!;

            // Read cond velocity to CPU
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer read_cmd) {
                vk::cmdCopyBuffer(read_cmd, dit.acts.velocity.gpu_buffer.buffer, cfg_staging.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;
            float* stg_ptr = (float*)cfg_staging.data();
            for (uint i = 0; i < latent_elems; i++) cfg_cond_vel[i] = stg_ptr[i];

            // Restore latent for uncond forward
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer restore_cmd) {
                vk::cmdCopyBuffer(restore_cmd, latent_save.buffer, dit.acts.latent.gpu_buffer.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            // Unconditioned forward
            dit.text_len = uncond_text_len;
            dit.padded_text = uncond_padded_text;
            dit.rope_cos_main = uncond_rope_cos_main;
            dit.rope_sin_main = uncond_rope_sin_main;
            dit.forward(&uncond_projected, t)!!;

            // Read uncond velocity to CPU
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer read_cmd2) {
                vk::cmdCopyBuffer(read_cmd2, dit.acts.velocity.gpu_buffer.buffer, cfg_staging.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;
            stg_ptr = (float*)cfg_staging.data();
            for (uint i = 0; i < latent_elems; i++) cfg_uncond_vel[i] = stg_ptr[i];

            // CFG combine
            for (uint i = 0; i < latent_elems; i++) {
                stg_ptr[i] = cfg_uncond_vel[i] + cfg_scale * (cfg_cond_vel[i] - cfg_uncond_vel[i]);
            }

            // DEBUG: Save velocity to file
            io::printfn("  [DEBUG] Saving velocity for step %d", step);
            char[256] filename_vel;
            usz fn_vel_len = 0;
            String vel_prefix = "/tmp/zimage_debug/velocity_step_";
            for (usz i = 0; i < vel_prefix.len; i++) filename_vel[fn_vel_len++] = vel_prefix[i];
            filename_vel[fn_vel_len++] = (char)('0' + (step + 1) / 10);
            filename_vel[fn_vel_len++] = (char)('0' + (step + 1) % 10);
            String vel_suffix = "_c3.bin";
            for (usz i = 0; i < vel_suffix.len; i++) filename_vel[fn_vel_len++] = vel_suffix[i];
            filename_vel[fn_vel_len] = 0;
            
            File f_vel = file::open((String)filename_vel[0..fn_vel_len-1], "wb")!!;
            uint n_vel = (uint)latent_elems;
            char[4] vel_header = {
                (char)(n_vel & 0xFF), 
                (char)((n_vel >> 8) & 0xFF),
                (char)((n_vel >> 16) & 0xFF), 
                (char)((n_vel >> 24) & 0xFF)
            };
            f_vel.write(&vel_header)!;
            f_vel.write(((char*)stg_ptr)[0..latent_elems*4-1])!;
            f_vel.close()!;

            // Upload combined velocity
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer write_cmd) {
                vk::cmdCopyBuffer(write_cmd, cfg_staging.buffer, dit.acts.velocity.gpu_buffer.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            // Restore latent for Euler step
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer restore_cmd2) {
                vk::cmdCopyBuffer(restore_cmd2, latent_save.buffer, dit.acts.latent.gpu_buffer.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;
        } else {
            // No CFG: single forward pass
            dit.forward(&cond_projected, t)!!;
        }

        // Euler step
        llm::begin_compute(cmd)!!;
        FlowEulerPC euler_pc = { .n = latent_elems, .dt = dt };
        llm::dispatch_kernel(cmd, &dit.kernels.flow_euler_step,
            { dit.acts.latent.gpu_buffer.buffer, dit.acts.velocity.gpu_buffer.buffer },
            { dit.acts.latent.size_bytes, dit.acts.velocity.size_bytes },
            &euler_pc, llm::ceil_div(latent_elems, 256));
        llm::submit_and_wait(ctx)!!;

        // DEBUG: Save latent after Euler step
        io::printfn("  [DEBUG] Saving latent for step %d", step);
        char[256] filename_lat;
        usz fn_lat_len = 0;
        String lat_prefix = "/tmp/zimage_debug/latent_step_";
        for (usz i = 0; i < lat_prefix.len; i++) filename_lat[fn_lat_len++] = lat_prefix[i];
        filename_lat[fn_lat_len++] = (char)('0' + (step + 1) / 10);
        filename_lat[fn_lat_len++] = (char)('0' + (step + 1) % 10);
        String lat_suffix = "_c3.bin";
        for (usz i = 0; i < lat_suffix.len; i++) filename_lat[fn_lat_len++] = lat_suffix[i];
        filename_lat[fn_lat_len] = 0;
        
        // Read latent from GPU
        vk::Memory lat_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null,
            data_size: (usz)latent_elems * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer lat_cmd) {
            vk::cmdCopyBuffer(lat_cmd, dit.acts.latent.gpu_buffer.buffer, lat_staging.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;
        float* lat_ptr = (float*)lat_staging.data();
        
        File f_lat = file::open((String)filename_lat[0..fn_lat_len-1], "wb")!!;
        uint n_lat = (uint)latent_elems;
        char[4] lat_header = {
            (char)(n_lat & 0xFF), 
            (char)((n_lat >> 8) & 0xFF),
            (char)((n_lat >> 16) & 0xFF), 
            (char)((n_lat >> 24) & 0xFF)
        };
        f_lat.write(&lat_header)!;
        f_lat.write(((char*)lat_ptr)[0..latent_elems*4-1])!;
        f_lat.close()!;
        lat_staging.free();
    }

    io::printfn("  Denoising complete");

    // Cleanup CFG buffers
    if (use_cfg) {
        cfg_staging.free();
        latent_save.free();
        mem::free(cfg_cond_vel);
        mem::free(cfg_uncond_vel);
    }

    flow.free();

    // ---- Phase 5: VAE decode and return image ----
    io::printfn("\n[5/5] VAE decoding...");

    image::Image result;

    if (has_vae) {
        llm::Tensor* decode_output;

        io::printfn("  Using VAE: %s", use_taesd ? "TAESD" : "Flux VAE");

        if (use_taesd) {
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer copy_cmd) {
                vk::cmdCopyBuffer(copy_cmd, dit.acts.latent.gpu_buffer.buffer,
                    taesd_acts.buf_a.gpu_buffer.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            taesd_decoder.forward(&taesd_acts, lat_h, lat_w)!!;
            decode_output = &taesd_acts.buf_a;
        } else {
            // Flux VAE denormalization
            const float SCALE_FACTOR = 0.3611f;
            const float[16] LATENT_MEANS = {
                -0.7571f, -0.7089f, -0.9113f,  0.1075f, -0.1745f,  0.9653f, -0.1517f,  1.5508f,
                 0.4134f, -0.0715f,  0.5517f, -0.3632f, -0.1922f, -0.9497f,  0.2503f, -0.2921f,
            };
            const float[16] LATENT_STDS = {
                2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f,
            };

            uint spatial = lat_h * lat_w;
            vk::Memory lat_readback = vk::new_buffer(
                allocator: &ctx.allocator,
                usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data: null, data_size: (usz)latent_elems * 4,
            )!!;
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer rb_cmd) {
                vk::cmdCopyBuffer(rb_cmd, dit.acts.latent.gpu_buffer.buffer, lat_readback.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;

            float* lat_cpu = (float*)lat_readback.data();

            for (uint ch = 0; ch < DIT_LATENT_CHANNELS; ch++) {
                float scale = LATENT_STDS[ch] / SCALE_FACTOR;
                float mean = LATENT_MEANS[ch];
                uint offset = ch * spatial;
                for (uint i = 0; i < spatial; i++) {
                    lat_cpu[offset + i] = lat_cpu[offset + i] * scale + mean;
                }
            }

            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer copy_cmd) {
                vk::cmdCopyBuffer(copy_cmd, lat_readback.buffer,
                    vae_acts.buf_a.gpu_buffer.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
            }!!;
            lat_readback.free();

            vae_decoder.forward(&vae_acts, lat_h, lat_w)!!;
            decode_output = &vae_acts.buf_a;
        }

        // Download image from GPU
        usz total_px = (usz)img_size * img_size * 3;
        float[] img_data = mem::new_array(float, total_px);

        vk::Memory download_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null,
            data_size: total_px * 4,
        )!!;

        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer download_cmd) {
            vk::cmdCopyBuffer(download_cmd, decode_output.gpu_buffer.buffer, download_staging.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = total_px * 4 }});
        }!!;

        float* mapped = (float*)download_staging.data();
        mem::copy(img_data.ptr, mapped, total_px * 4);
        download_staging.free();

        result = llm::pipelines::float_tensor_to_image(img_data, img_size, img_size, 3)!!;
        mem::free(img_data);

        if (use_taesd) {
            taesd_decoder.free();
            taesd_acts.free();
        } else {
            vae_decoder.free();
            vae_acts.free();
        }
    } else {
        // No VAE: return raw latent as grayscale image
        io::printfn("  Warning: No VAE loaded, returning raw latent (first channel)");

        float[] latent_img = mem::new_array(float, (usz)lat_h * lat_w);
        vk::Memory lat_readback = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: (usz)latent_elems * 4,
        )!!;

        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer rb_cmd) {
            vk::cmdCopyBuffer(rb_cmd, dit.acts.latent.gpu_buffer.buffer, lat_readback.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;

        float* lat_ptr = (float*)lat_readback.data();
        float lmin = lat_ptr[0];
        float lmax = lat_ptr[0];
        for (uint i = 0; i < lat_h * lat_w; i++) {
            float v = lat_ptr[i];
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        }
        float lrange = lmax - lmin;
        if (lrange < 1e-6f) lrange = 1.0f;

        for (uint i = 0; i < lat_h * lat_w; i++) {
            latent_img[i] = (lat_ptr[i] - lmin) / lrange;
        }

        lat_readback.free();

        result = llm::pipelines::float_tensor_to_image(latent_img, lat_w, lat_h, 1)!!;
        mem::free(latent_img);
    }

    // Cleanup
    cond_projected.free();
    if (use_cfg) uncond_projected.free();
    diff_kernels.free(ctx.device);
    dit_gf.free();
    dit_mm.destroy();
    dit.free();
    mem::free(dit);

    double total_sec = total_start.to_now().to_sec();
    io::printfn("\n=== Generation Complete ===");
    io::printfn("  Output: %dx%d image", img_size, img_size);
    io::printfn("  Total time: %.2fs", total_sec);

    return result;
}

fn void ZImageState.free(ZImageState* self) @dynamic {
    self.loaded = false;
}

fn llm::pipelines::Pipeline create_pipeline() {
    ZImageState* state = mem::new(ZImageState);
    *state = {};
    return (llm::pipelines::Pipeline)state;
}
