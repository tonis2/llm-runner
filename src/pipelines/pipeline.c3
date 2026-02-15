module llm::pipelines;

import vk;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::math::random;
import std::time::clock;
import image;
import image::png;
import llm;
import llm::diffusers;

// Unified Pipeline: PyTorch-style API
// Usage:
//   ImagePipeline pipeline = pipeline_from_gguf(&ctx, "model.gguf")!!;
//   pipeline.set_text_encoder("qwen3.gguf");
//   pipeline.set_taesd("taesd_f.embd");
//
//   GenerationInputs inputs = generation_inputs_defaults();
//   inputs.prompt = "a cat";
//   image::Image output = pipeline.generate(&inputs)!!;
//
//   output.save("out.png");
//   output.free();
//   pipeline.free();

struct ImagePipeline {
    llm::DeviceContext* ctx;

    TextEncoderKind text_encoder_kind;
    DenoiserKind denoiser_kind;
    DecoderKind decoder_kind;
    SchedulerType scheduler_kind;

    // Model paths
    String model_path;
    String text_model_path;
    String vae_path;
    String taesd_path;
}

// --- Model Detection ---

fn bool is_dit_model(llm::GGUFFile* gf) {
    if (try arch = gf.get_string("general.architecture")) {
        if (arch == "lumina2") return true;
    }
    foreach (&t : gf.tensors) {
        if (t.name.starts_with("noise_refiner.") ||
            t.name.starts_with("context_refiner.") ||
            t.name.starts_with("x_pad_token")) {
            return true;
        }
    }
    return false;
}

fn bool is_sdxs_model(llm::GGUFFile* gf) {
    foreach (&t : gf.tensors) {
        if (t.name.starts_with("cond_stage_model.") ||
            t.name.starts_with("model.diffusion_model.")) {
            return true;
        }
    }
    return false;
}

// --- Factory Functions ---

fn ImagePipeline? pipeline_from_gguf(llm::DeviceContext* ctx, String model_path) {
    mmap::FileMmap mm = file::mmap_open(model_path, "rb")!!;
    char[] data = mm.bytes();
    llm::GGUFFile gf = llm::gguf_parse(data)!!;
    defer gf.free();

    TextEncoderKind text_enc = NONE;
    DenoiserKind denoiser = NONE;
    DecoderKind decoder = NONE;
    SchedulerType scheduler = FLOW_MATCHING;

    if (is_dit_model(&gf)) {
        io::printfn("Detected: DiT model (Lumina2 / Z-Image)");
        text_enc = LLM;
        denoiser = DIT;
        decoder = TAESD;  // Default for DiT, can be overridden
        scheduler = FLOW_MATCHING;
    } else if (is_sdxs_model(&gf)) {
        io::printfn("Detected: SDXS model (CLIP + UNet + VAE)");
        text_enc = CLIP;
        denoiser = UNET;
        decoder = SD_VAE;
        scheduler = DDIM;
    } else {
        io::printfn("Error: Unrecognized model format");
        mm.destroy();
        return PIPELINE_LOAD_FAILED~;
    }

    mm.destroy();

    return {
        .ctx = ctx,
        .text_encoder_kind = text_enc,
        .denoiser_kind = denoiser,
        .decoder_kind = decoder,
        .scheduler_kind = scheduler,
        .model_path = model_path,
        .text_model_path = "",
        .vae_path = "",
        .taesd_path = "",
    };
}

fn ImagePipeline? pipeline_custom(
    llm::DeviceContext* ctx,
    TextEncoderKind text_enc,
    DenoiserKind denoiser,
    DecoderKind decoder,
    SchedulerType scheduler,
    String model_path,
) {
    return {
        .ctx = ctx,
        .text_encoder_kind = text_enc,
        .denoiser_kind = denoiser,
        .decoder_kind = decoder,
        .scheduler_kind = scheduler,
        .model_path = model_path,
        .text_model_path = "",
        .vae_path = "",
        .taesd_path = "",
    };
}

// --- Builder Methods ---

fn void ImagePipeline.set_text_encoder(ImagePipeline* self, String path) {
    self.text_model_path = path;
}

fn void ImagePipeline.set_vae(ImagePipeline* self, String path) {
    self.vae_path = path;
    // If user explicitly sets a VAE, update decoder kind
    if (path.len > 0 && self.decoder_kind != SD_VAE) {
        self.decoder_kind = FLUX_VAE;
    }
}

fn void ImagePipeline.set_taesd(ImagePipeline* self, String path) {
    self.taesd_path = path;
    if (path.len > 0) {
        self.decoder_kind = TAESD;
    }
}

// --- Main Generate Entry Point ---

fn image::Image? ImagePipeline.generate(ImagePipeline* self, GenerationInputs* inputs) {
    if (!inputs.validate()) {
        return PIPELINE_INVALID_INPUT~;
    }

    switch (self.denoiser_kind) {
        case DIT:
            return self.generate_dit(inputs);
        case UNET:
            return self.generate_unet(inputs);
        default:
            io::printfn("Error: No denoiser configured");
            return PIPELINE_NOT_INITIALIZED~;
    }
}

// --- Free ---

fn void ImagePipeline.free(ImagePipeline* self) {
    // Nothing to free — models are loaded per-generation
}

// ============================================================
// DiT Pipeline (Z-Image / Lumina2)
// ============================================================

fn image::Image? ImagePipeline.generate_dit(ImagePipeline* self, GenerationInputs* inputs) {
    llm::DeviceContext* ctx = self.ctx;
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
    uint n_patches = (lat_h / llm::diffusers::DIT_PATCH_SIZE) * (lat_w / llm::diffusers::DIT_PATCH_SIZE);
    uint latent_elems = llm::diffusers::DIT_LATENT_CHANNELS * lat_h * lat_w;

    // ---- Phase 1: Text Encoding via LLM ----
    io::printfn("\n[1/5] Text encoding (LLM)...");

    if (self.text_model_path.len == 0) {
        io::printfn("Error: text_model_path required for DiT pipeline");
        return PIPELINE_INVALID_INPUT~;
    }

    mmap::FileMmap text_mm = file::mmap_open(self.text_model_path, "rb")!!;
    char[] text_data = text_mm.bytes();
    io::printfn("  Text model: %s (%d bytes)", self.text_model_path, text_data.len);

    llm::GGUFFile text_gf = llm::gguf_parse(text_data)!!;

    String arch_name = llm::detect_architecture(&text_gf);
    io::printfn("  Text encoder architecture: %s", arch_name);

    String config_json;
    if (arch_name == "qwen3") {
        config_json = (String)&llm::diffusers::QWEN3_CONFIG_JSON;
    } else if (arch_name == "qwen2") {
        config_json = (String)&llm::diffusers::QWEN2_CONFIG_JSON;
    } else {
        config_json = (String)&llm::diffusers::QWEN3_CONFIG_JSON;
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
    llm::diffusers::LlmModel text_model = llm::diffusers::load_llm_model(ctx, &text_gf, &text_config, &text_names,
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

    llm::diffusers::DiTWeights dit_weights = llm::diffusers::load_dit_weights(ctx, &dit_gf)!!;
    llm::diffusers::DiTKernels dit_kernels = llm::diffusers::create_dit_kernels(ctx)!!;

    uint cond_padded_text = llm::diffusers::pad_to_multiple(cond_text_len, llm::diffusers::SEQ_MULTI_OF);
    uint padded_patches = llm::diffusers::pad_to_multiple(n_patches, llm::diffusers::SEQ_MULTI_OF);
    uint cond_n_text_pad = cond_padded_text - cond_text_len;
    uint n_img_pad = padded_patches - n_patches;

    uint uncond_padded_text = use_cfg ? llm::diffusers::pad_to_multiple(uncond_text_len, llm::diffusers::SEQ_MULTI_OF) : 0;
    uint uncond_n_text_pad = use_cfg ? uncond_padded_text - uncond_text_len : 0;

    uint max_padded_text = cond_padded_text;
    if (use_cfg && uncond_padded_text > max_padded_text) max_padded_text = uncond_padded_text;
    uint max_seq = padded_patches + max_padded_text;
    llm::diffusers::DiTActivations dit_acts = llm::diffusers::allocate_dit_activations(ctx, max_seq)!!;

    llm::diffusers::DiTModel* dit = mem::alloc(llm::diffusers::DiTModel);
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
    uint patches_w = lat_w / llm::diffusers::DIT_PATCH_SIZE;
    uint patches_h = lat_h / llm::diffusers::DIT_PATCH_SIZE;
    llm::Tensor cond_rope_cos_main;
    llm::Tensor cond_rope_sin_main;
    llm::diffusers::precompute_mrope_tables(ctx, &cond_rope_cos_main, &cond_rope_sin_main,
        n_patches, patches_w, patches_h, cond_text_len, llm::diffusers::MROPE_MODE_FULL)!!;
    dit.rope_cos_main = cond_rope_cos_main;
    dit.rope_sin_main = cond_rope_sin_main;
    llm::diffusers::precompute_mrope_tables(ctx, &dit.rope_cos_refiner, &dit.rope_sin_refiner,
        n_patches, patches_w, patches_h, cond_text_len, llm::diffusers::MROPE_MODE_IMAGE)!!;

    llm::Tensor cond_rope_cos_ctx;
    llm::Tensor cond_rope_sin_ctx;
    llm::diffusers::precompute_mrope_tables(ctx, &cond_rope_cos_ctx, &cond_rope_sin_ctx,
        n_patches, patches_w, patches_h, cond_text_len, llm::diffusers::MROPE_MODE_TEXT)!!;

    llm::Tensor uncond_rope_cos_main;
    llm::Tensor uncond_rope_sin_main;
    llm::Tensor uncond_rope_cos_ctx;
    llm::Tensor uncond_rope_sin_ctx;
    if (use_cfg) {
        llm::diffusers::precompute_mrope_tables(ctx, &uncond_rope_cos_main, &uncond_rope_sin_main,
            n_patches, patches_w, patches_h, uncond_text_len, llm::diffusers::MROPE_MODE_FULL)!!;
        llm::diffusers::precompute_mrope_tables(ctx, &uncond_rope_cos_ctx, &uncond_rope_sin_ctx,
            n_patches, patches_w, patches_h, uncond_text_len, llm::diffusers::MROPE_MODE_TEXT)!!;
    }

    io::printfn("  DiT loaded: %d patches (pad %d), cond=%d tokens (pad %d)",
        n_patches, n_img_pad, cond_text_len, cond_n_text_pad);
    if (use_cfg) {
        io::printfn("  CFG uncond: %d tokens (pad %d)", uncond_text_len, uncond_n_text_pad);
    }

    // Project text through cap_embedder + context_refiner
    llm::Tensor cond_projected = llm::create_f32_tensor(ctx, { (ulong)cond_padded_text * llm::diffusers::DIT_DIM, 0, 0, 0 }, 1)!!;
    llm::Tensor norm_scratch = llm::create_f32_tensor(ctx, { (ulong)text_config.dim, 0, 0, 0 }, 1)!!;
    vk::CommandBuffer cmd = ctx.command_buffer;

    llm::begin_compute(cmd)!!;
    for (uint pos = 0; pos < cond_text_len; pos++) {
        usz in_off = (usz)pos * text_config.dim * 4;
        vk::cmdCopyBuffer(cmd, cond_embeddings.gpu_buffer.buffer, norm_scratch.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)text_config.dim * 4 }});
        llm::compute_barrier(cmd);
        llm::diffusers::RMSNormPC rms_pc = { .dim = text_config.dim, .eps = 1e-6f };
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
    llm::diffusers::LinearProjPC cap_pc = { .out_dim = llm::diffusers::DIT_DIM, .in_dim = text_config.dim, .seq_len = cond_text_len };
    llm::dispatch_kernel(cmd, &dit.kernels.linear_proj,
        { dit.weights.cap_emb_weight.gpu_buffer.buffer, dit.weights.cap_emb_bias.gpu_buffer.buffer,
          cond_embeddings.gpu_buffer.buffer, cond_projected.gpu_buffer.buffer },
        { dit.weights.cap_emb_weight.size_bytes, dit.weights.cap_emb_bias.size_bytes,
          cond_embeddings.size_bytes, cond_projected.size_bytes },
        &cap_pc, cond_text_len * llm::diffusers::DIT_DIM);
    llm::compute_barrier(cmd);
    if (cond_n_text_pad > 0 && dit.weights.cap_pad_token.size_bytes > 0) {
        for (uint p = 0; p < cond_n_text_pad; p++) {
            vk::cmdCopyBuffer(cmd, dit.weights.cap_pad_token.gpu_buffer.buffer,
                cond_projected.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0,
                    .dstOffset = (ulong)(cond_text_len + p) * llm::diffusers::DIT_DIM * 4,
                    .size = (ulong)llm::diffusers::DIT_DIM * 4 }});
        }
        llm::compute_barrier(cmd);
    }
    llm::submit_and_wait(ctx)!!;
    cond_embeddings.free();

    // Context refiner on cond text
    dit.rope_cos_context = cond_rope_cos_ctx;
    dit.rope_sin_context = cond_rope_sin_ctx;
    llm::begin_compute(cmd)!!;
    for (uint l = 0; l < llm::diffusers::DIT_NUM_REFINER_LAYERS; l++) {
        llm::diffusers::dit_transformer_layer(dit, cmd, &dit.weights.context_refiner[l],
            &cond_projected, cond_padded_text, false,
            &dit.rope_cos_context, &dit.rope_sin_context)!!;
    }
    llm::submit_and_wait(ctx)!!;
    io::printfn("  Cond context refiner complete (padded=%d).", cond_padded_text);

    // Process unconditioned text for CFG
    llm::Tensor uncond_projected;
    if (use_cfg) {
        uncond_projected = llm::create_f32_tensor(ctx, { (ulong)uncond_padded_text * llm::diffusers::DIT_DIM, 0, 0, 0 }, 1)!!;

        llm::begin_compute(cmd)!!;
        for (uint pos = 0; pos < uncond_text_len; pos++) {
            usz in_off = (usz)pos * text_config.dim * 4;
            vk::cmdCopyBuffer(cmd, uncond_embeddings.gpu_buffer.buffer, norm_scratch.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)text_config.dim * 4 }});
            llm::compute_barrier(cmd);
            llm::diffusers::RMSNormPC rms_pc = { .dim = text_config.dim, .eps = 1e-6f };
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
        llm::diffusers::LinearProjPC uncond_cap_pc = { .out_dim = llm::diffusers::DIT_DIM, .in_dim = text_config.dim, .seq_len = uncond_text_len };
        llm::dispatch_kernel(cmd, &dit.kernels.linear_proj,
            { dit.weights.cap_emb_weight.gpu_buffer.buffer, dit.weights.cap_emb_bias.gpu_buffer.buffer,
              uncond_embeddings.gpu_buffer.buffer, uncond_projected.gpu_buffer.buffer },
            { dit.weights.cap_emb_weight.size_bytes, dit.weights.cap_emb_bias.size_bytes,
              uncond_embeddings.size_bytes, uncond_projected.size_bytes },
            &uncond_cap_pc, uncond_text_len * llm::diffusers::DIT_DIM);
        llm::compute_barrier(cmd);
        if (uncond_n_text_pad > 0 && dit.weights.cap_pad_token.size_bytes > 0) {
            for (uint p = 0; p < uncond_n_text_pad; p++) {
                vk::cmdCopyBuffer(cmd, dit.weights.cap_pad_token.gpu_buffer.buffer,
                    uncond_projected.gpu_buffer.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0,
                        .dstOffset = (ulong)(uncond_text_len + p) * llm::diffusers::DIT_DIM * 4,
                        .size = (ulong)llm::diffusers::DIT_DIM * 4 }});
            }
            llm::compute_barrier(cmd);
        }
        llm::submit_and_wait(ctx)!!;
        uncond_embeddings.free();

        // Context refiner on uncond text
        dit.rope_cos_context = uncond_rope_cos_ctx;
        dit.rope_sin_context = uncond_rope_sin_ctx;
        llm::begin_compute(cmd)!!;
        for (uint l = 0; l < llm::diffusers::DIT_NUM_REFINER_LAYERS; l++) {
            llm::diffusers::dit_transformer_layer(dit, cmd, &dit.weights.context_refiner[l],
                &uncond_projected, uncond_padded_text, false,
                &dit.rope_cos_context, &dit.rope_sin_context)!!;
        }
        llm::submit_and_wait(ctx)!!;
        io::printfn("  Uncond context refiner complete (padded=%d).", uncond_padded_text);
    }
    norm_scratch.free();

    // Load VAE/TAESD
    llm::DiffusionKernels diff_kernels = llm::create_diffusion_kernels(ctx)!!;
    llm::diffusers::FluxVaeDecoder vae_decoder;
    llm::diffusers::FluxVaeActivations vae_acts;
    llm::diffusers::TAESDDecoder taesd_decoder;
    llm::diffusers::TAESDActivations taesd_acts;
    bool has_vae = false;
    bool use_taesd = false;

    if (self.taesd_path.len > 0) {
        llm::SafetensorsFile sf = llm::safetensors_open(self.taesd_path)!!;
        taesd_decoder = llm::diffusers::load_taesd_decoder(ctx, &sf, &diff_kernels)!!;
        sf.close();
        taesd_acts = llm::diffusers::allocate_taesd_activations(ctx, img_size)!!;
        has_vae = true;
        use_taesd = true;
        io::printfn("  TAESD loaded from %s", self.taesd_path);
    } else if (self.vae_path.len > 0) {
        llm::SafetensorsFile sf = llm::safetensors_open(self.vae_path)!!;
        vae_decoder = llm::diffusers::load_flux_vae_decoder(ctx, &sf, &diff_kernels)!!;
        sf.close();
        vae_acts = llm::diffusers::allocate_flux_vae_activations(ctx, img_size)!!;
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
        generate_random_latent(latent_data, inputs.seed);
    } else {
        io::printfn("\n[3/5] Generating random latent...");
        generate_random_latent(latent_data, inputs.seed);
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

    FlowMatchingState flow = init_flow_matching(inputs.num_steps);

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
        llm::diffusers::FlowEulerPC euler_pc = { .n = latent_elems, .dt = dt };
        llm::dispatch_kernel(cmd, &dit.kernels.flow_euler_step,
            { dit.acts.latent.gpu_buffer.buffer, dit.acts.velocity.gpu_buffer.buffer },
            { dit.acts.latent.size_bytes, dit.acts.velocity.size_bytes },
            &euler_pc, llm::ceil_div(latent_elems, 256));
        llm::submit_and_wait(ctx)!!;
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

            for (uint ch = 0; ch < llm::diffusers::DIT_LATENT_CHANNELS; ch++) {
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

        result = float_tensor_to_image(img_data, img_size, img_size, 3)!!;
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

        result = float_tensor_to_image(latent_img, lat_w, lat_h, 1)!!;
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

// ============================================================
// UNet Pipeline (SDXS / Stable Diffusion)
// ============================================================

fn image::Image? ImagePipeline.generate_unet(ImagePipeline* self, GenerationInputs* inputs) {
    llm::DeviceContext* ctx = self.ctx;
    bool is_img2img = inputs.input_image != null;

    if (is_img2img) {
        io::printfn("=== Image-to-Image Generation ===");
        io::printfn("  Prompt: \"%s\"", inputs.prompt);
        io::printfn("  Strength: %.2f, Steps: %d", inputs.strength, inputs.num_steps);
    } else {
        io::printfn("=== Text-to-Image Generation ===");
        io::printfn("  Prompt: \"%s\"", inputs.prompt);
        io::printfn("  Steps: %d, Seed: %d", inputs.num_steps, inputs.seed);
    }

    // Load GGUF
    mmap::FileMmap mm = file::mmap_open(self.model_path, "rb")!!;
    char[] data = mm.bytes();
    llm::GGUFFile gf = llm::gguf_parse(data)!!;

    llm::DiffusionConfig config = llm::load_diffusion_config((String)&llm::SD_CONFIG_JSON)!!;
    llm::DiffusionKernels kernels = llm::create_diffusion_kernels(ctx)!!;

    llm::diffusers::ClipEncoder clip = llm::diffusers::load_clip_encoder(ctx, &gf, &config.clip, &kernels)!!;
    llm::diffusers::UnetModel unet = llm::diffusers::load_unet_model(ctx, &gf, &config.unet, &kernels)!!;
    llm::diffusers::VaeDecoder vae_decoder = llm::diffusers::load_vae_decoder(ctx, &gf, &config.vae, &kernels)!!;
    llm::diffusers::VaeEncoder vae_encoder = llm::diffusers::load_vae_encoder(ctx, &gf, &config.vae, &kernels)!!;
    llm::diffusers::VaeActivations vae_acts = llm::diffusers::allocate_vae_activations(ctx, config.image.size, 64)!!;

    // Fix up kernel pointers
    clip.kernels = &kernels;
    unet.kernels = &kernels;
    vae_decoder.kernels = &kernels;
    vae_encoder.kernels = &kernels;

    SchedulerState scheduler = init_scheduler(
        &config.scheduler,
        DDIM,
        inputs.num_steps,
        inputs.cfg_scale
    )!!;

    // Load tokenizer
    llm::Tokenizer tok = llm::load_tokenizer(&gf)!!;

    if (is_img2img) {
        if (inputs.input_image.width != 512 || inputs.input_image.height != 512) {
            io::printfn("Error: Input image must be 512x512 (got %dx%d)",
                inputs.input_image.width, inputs.input_image.height);
            tok.free();
            gf.free();
            mm.destroy();
            return PIPELINE_INVALID_INPUT~;
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
            vk::cmdCopyBuffer(cmd, img_staging.buffer, vae_acts.buf_a.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = img_bytes }});
        }!!;
        img_staging.free();
        mem::free(input_data);

        // VAE encode
        io::printfn("[1/4] VAE encoding...");
        vae_encoder.forward(&vae_acts, 512, 512)!!;

        // Copy encoded latent to UNet buf_a
        uint latent_elems = 4 * 64 * 64;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
            vk::cmdCopyBuffer(cmd, vae_acts.buf_a.gpu_buffer.buffer,
                unet.acts.buf_a.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;

        // Tokenize + CLIP
        io::printfn("[2/4] Tokenizing...");
        unet_encode_prompt(&clip, &tok, inputs.prompt);

        // Denoise
        io::printfn("[3/4] Denoising...");
        uint start_step = (uint)((1.0f - inputs.strength) * (float)inputs.num_steps);
        unet_denoise(ctx, &unet, &clip, &kernels, &scheduler, inputs, start_step);
    } else {
        // Tokenize
        io::printfn("\n[1/3] Tokenizing...");
        unet_encode_prompt(&clip, &tok, inputs.prompt);

        // Generate latent and denoise
        io::printfn("[2/3] UNet denoising (%d steps)...", inputs.num_steps);
        unet_generate_and_denoise(ctx, &unet, &clip, &kernels, &scheduler, inputs);
    }

    // VAE decode
    io::printfn("[%s] VAE decoding...", is_img2img ? "4/4" : "3/3");

    uint latent_elems = 4 * 64 * 64;
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, unet.acts.buf_a.gpu_buffer.buffer,
            vae_acts.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;

    vae_decoder.forward(&vae_acts)!!;

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
        vk::cmdCopyBuffer(cmd, vae_acts.buf_a.gpu_buffer.buffer, download_staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = total_px * 4 }});
    }!!;

    float* mapped = (float*)download_staging.data();
    mem::copy(img_data.ptr, mapped, total_px * 4);
    download_staging.free();

    image::Image result = float_tensor_to_image_sd(img_data, 512, 512, 3)!!;
    mem::free(img_data);

    io::printfn("\nGeneration complete!");

    // Cleanup
    tok.free();
    clip.free();
    unet.free();
    vae_decoder.free();
    vae_encoder.free();
    vae_acts.free();
    scheduler.free();
    kernels.free(ctx.device);
    gf.free();
    mm.destroy();

    return result;
}

// --- UNet helpers ---

fn void unet_encode_prompt(llm::diffusers::ClipEncoder* clip, llm::Tokenizer* tokenizer, String prompt) {
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
    clip.forward(token_slice, n_tokens)!!;
}

fn void unet_generate_and_denoise(
    llm::DeviceContext* ctx,
    llm::diffusers::UnetModel* unet,
    llm::diffusers::ClipEncoder* clip,
    llm::DiffusionKernels* kernels,
    SchedulerState* sched,
    GenerationInputs* inputs,
) {
    uint latent_elems = 4 * 64 * 64;
    float[] latent_data = mem::new_array(float, latent_elems);
    generate_random_latent(latent_data, inputs.seed);

    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: latent_data.ptr,
        data_size: (usz)latent_elems * 4,
    )!!;

    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, staging.buffer, unet.acts.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
    }!!;
    staging.free();
    mem::free(latent_data);

    // Setup scheduler
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

        unet.forward(&clip.acts.hidden, t)!!;

        float alpha_t = sched.get_alpha_cumprod(t);
        float alpha_prev = sched.get_prev_alpha_cumprod(step);

        vk::CommandBuffer cmd = ctx.command_buffer;
        llm::begin_compute(cmd)!!;
        if (sched.stype == DDIM) {
            dispatch_ddim_step(cmd, kernels,
                &unet.acts.buf_a,
                &unet.acts.buf_b,
                latent_elems, alpha_t, alpha_prev);
        } else {
            float sigma_t = 1.0f - alpha_t;
            float sigma_prev = 1.0f - alpha_prev;
            float dt = sigma_prev - sigma_t;
            dispatch_euler_step(cmd, kernels,
                &unet.acts.buf_a,
                &unet.acts.buf_b,
                latent_elems, dt);
        }
        llm::submit_and_wait(ctx)!!;
    }
}

fn void unet_denoise(
    llm::DeviceContext* ctx,
    llm::diffusers::UnetModel* unet,
    llm::diffusers::ClipEncoder* clip,
    llm::DiffusionKernels* kernels,
    SchedulerState* sched,
    GenerationInputs* inputs,
    uint start_step,
) {
    uint latent_elems = 4 * 64 * 64;

    for (uint step = start_step; step < inputs.num_steps; step++) {
        float t = sched.timesteps[step];
        io::printfn("  Step %d/%d (t=%.1f)", step + 1, inputs.num_steps, t);

        unet.forward(&clip.acts.hidden, t)!!;

        float alpha_t = sched.get_alpha_cumprod(t);
        float alpha_prev = sched.get_prev_alpha_cumprod(step);

        vk::CommandBuffer cmd = ctx.command_buffer;
        llm::begin_compute(cmd)!!;
        dispatch_ddim_step(cmd, kernels,
            &unet.acts.buf_a, &unet.acts.buf_b,
            latent_elems, alpha_t, alpha_prev);
        llm::submit_and_wait(ctx)!!;
    }
}

// --- Shared Helpers ---

// Convert float[0,1] tensor to image::Image (NCHW layout)
fn image::Image? float_tensor_to_image(float[] data, uint width, uint height, uint channels) {
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

// Convert float tensor to image with SD-style normalization
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
