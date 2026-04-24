module flux;

import vk;
import llm;
import llm::pipelines;
import llm_text;
import zimage;
import image;
import image::png;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::math;
import std::core::mem;

// Flux 2 Klein Pipeline: txt2img with Flux.2 Klein 9B
//
// Flow:
//   1. Encode text with Qwen3-8B → extract 3 hidden states → [seq_len, 12288]
//   2. Load Flux DiT from GGUF
//   3. Denoise with flow matching Euler steps
//   4. Pixel shuffle + VAE decode → output image
//   5. Save

struct FluxConfig {
    String model_path;       // Flux DiT GGUF
    String text_model_path;  // Qwen3-8B GGUF (text encoder)
    String vae_path;         // ae.safetensors (Flux VAE)
    String prompt;
    String input_path;       // Input image for img2img
    String output_path;
    uint image_size;
    uint num_steps;
    float cfg_scale;
    float strength;          // img2img strength (0-1)
    uint seed;
    bool offload;
}

fn FluxConfig flux_config_defaults() {
    return {
        .output_path = "output.png",
        .image_size = 1024,
        .num_steps = 4,       // Klein is distilled to 4 steps
        .cfg_scale = 1.0f,    // No CFG for distilled Klein
        .strength = 0.75f,
        .seed = 42,
    };
}

// Qwen3-8B layers to extract hidden states from (0-indexed, evenly spaced)
// For 32-layer model: layers at 25%, 50%, 75% depth
const uint[3] QWEN3_OUTPUT_LAYERS = { 8, 16, 24 };

// --- Pipeline Implementation ---

fn void? run_flux_pipeline(FluxConfig* config) {
    io::printfn("\n=== Flux 2 Klein Pipeline ===");
    io::printfn("  DiT model: %s", config.model_path);
    io::printfn("  Text model: %s", config.text_model_path);
    io::printfn("  VAE: %s", config.vae_path);
    io::printfn("  Prompt: %s", config.prompt);
    io::printfn("  Size: %d, Steps: %d, CFG: %.1f, Seed: %d",
        config.image_size, config.num_steps, config.cfg_scale, config.seed);

    // Setup Vulkan
    llm::DeviceContext ctx = llm::createContext()!!;
    defer ctx.free();

    // --- Phase 1: Text encoding with Qwen3 ---
    io::printfn("\n--- Phase 1: Qwen3 Text Encoding ---");

    mmap::FileMmap text_mm = file::mmap_open(config.text_model_path, "rb")!!;
    char[] text_data = text_mm.bytes();
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

    // Build prompt with chat template (official pipeline uses apply_chat_template with enable_thinking=False)
    char[4096] cond_buf;
    usz cond_len = 0;
    String cond_parts = "<|im_start|>user\n";
    for (usz i = 0; i < cond_parts.len; i++) { cond_buf[cond_len] = cond_parts[i]; cond_len++; }
    for (usz i = 0; i < config.prompt.len; i++) { cond_buf[cond_len] = config.prompt[i]; cond_len++; }
    String cond_suffix = "<|im_end|>\n<|im_start|>assistant\n";
    for (usz i = 0; i < cond_suffix.len; i++) { cond_buf[cond_len] = cond_suffix[i]; cond_len++; }
    String cond_prompt = (String)cond_buf[0..cond_len - 1];

    uint[] cond_tokens = tok.encode_with_specials(cond_prompt)!!;
    uint text_len = (uint)cond_tokens.len;
    io::printfn("  Tokenized: %d tokens (with chat template)", text_len);

    // Load LLM for text encoding
    llm_text::LlmModel text_model = llm_text::load_llm_model(&ctx, &text_gf, &text_config, &text_names,
        encode_only: true, max_seq_len: text_len + 16)!!;

    // Download output norm weight for normalizing extracted hidden states
    // The Qwen3 final RMSNorm maps raw hidden states (~100 magnitude) to unit scale
    float[] output_norm_cpu = mem::new_array(float, text_config.dim);
    {
        vk::Memory norm_dl = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: (usz)text_config.dim * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer nc) {
            vk::cmdCopyBuffer(nc, text_model.weights.output_norm.gpu_buffer.buffer, norm_dl.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)text_config.dim * 4 }});
        }!!;
        mem::copy(output_norm_cpu.ptr, (float*)norm_dl.data(), (usz)text_config.dim * 4);
        norm_dl.free();
    }
    float rms_eps = text_config.rms_eps;

    // Set global norm data for encode_text_multilayer
    g_output_norm_cpu = output_norm_cpu;
    g_rms_eps = rms_eps;

    // Encode text with multi-layer extraction
    // Extract hidden states at 3 layers, concatenate → [text_len, 3 * dim]
    llm::Tensor text_embeddings = encode_text_multilayer(&ctx, &text_model, cond_tokens, text_len, text_config.dim)!!;
    uint text_out_dim = text_config.dim * 3;  // 3 concatenated hidden states
    io::printfn("  Text embeddings: [%d, %d] (3 layers × %d)", text_len, text_out_dim, text_config.dim);

    // Free text encoder
    mem::free(cond_tokens);
    mem::free(output_norm_cpu);
    g_output_norm_cpu = {};
    tok.free();
    text_model.free();
    text_gf.free();
    text_mm.destroy();

    // --- Phase 2: Load Flux DiT ---
    io::printfn("\n--- Phase 2: Loading Flux DiT ---");

    FluxDiTKernels dit_kernels = create_flux_dit_kernels(&ctx)!!;
    defer dit_kernels.free(ctx.device);

    mmap::FileMmap dit_mm = file::mmap_open(config.model_path, "rb")!!;
    char[] dit_data = dit_mm.bytes();
    llm::GGUFFile dit_gf = llm::gguf_parse(dit_data)!!;
    FluxDiTModel dit = load_flux_dit(&ctx, &dit_gf, &dit_kernels)!!;
    defer dit.free();

    // Image dimensions: Flux.2 uses 16x spatial reduction (8x VAE + 2x pixel unshuffle)
    uint img_h = (config.image_size + 15) / 16 * 16; // Round to multiple of 16
    uint img_w = img_h;
    uint lat_h = img_h / 16;  // 16x reduction (8x VAE * 2x unshuffle)
    uint lat_w = img_w / 16;
    uint n_img = lat_h * lat_w;  // PATCH_SIZE=1, so n_patches = spatial elements
    uint n_txt = text_len;
    uint n_joint = n_img + n_txt;
    uint latent_elems = FLUX_LATENT_CH * lat_h * lat_w;  // 128 * lat_h * lat_w

    if (img_h != config.image_size) {
        io::printfn("  Note: size %d rounded to %d (must be multiple of 16)", config.image_size, img_h);
    }
    io::printfn("  Image: %dx%d → latent: [128, %d, %d] → %d patches",
        img_h, img_w, lat_h, lat_w, n_img);
    io::printfn("  Sequence: %d img + %d txt = %d joint", n_img, n_txt, n_joint);

    // Set model dimensions
    dit.n_img = n_img;
    dit.n_txt = n_txt;
    dit.n_joint = n_joint;
    dit.latent_h = lat_h;
    dit.latent_w = lat_w;
    dit.patches_h = lat_h;  // PATCH_SIZE=1
    dit.patches_w = lat_w;

    // Allocate activations
    dit.acts = allocate_flux_activations(&ctx, n_img, n_txt, latent_elems)!!;

    // Precompute mRoPE tables
    precompute_flux_mrope(&ctx, &dit.rope_cos, &dit.rope_sin, n_txt, n_img, lat_w, lat_h)!!;

    // --- Phase 3: Denoising ---
    io::printfn("\n--- Phase 3: Denoising (%d steps) ---", config.num_steps);

    // DEBUG: Zero out text embeddings to test unconditional generation
    // This tests whether the DiT can produce spatial structure without text
    // Generate random latent [128, lat_h, lat_w]
    float[] latent_cpu = mem::new_array(float, latent_elems);
    pipelines::generate_random_latent(latent_cpu, config.seed);

    // Upload latent to GPU
    {
        vk::Memory lat_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: latent_cpu.ptr, data_size: (usz)latent_elems * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer copy_cmd) {
            vk::cmdCopyBuffer(copy_cmd, lat_staging.buffer,
                dit.acts.latent.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;
        lat_staging.free();
    }

    // Scheduler: linear sigma schedule with dynamic shift
    float[] sigmas = mem::new_array(float, (usz)config.num_steps + 1);
    {
        // Compute dynamic shift based on image sequence length (same as Flux.1)
        float base_shift = 0.5f;
        float max_shift = 1.15f;
        float base_seq = 256.0f;
        float max_seq = 4096.0f;
        float m = (max_shift - base_shift) / (max_seq - base_seq);
        float b = base_shift - m * base_seq;
        float mu = (float)n_img * m + b;
        float flux_shift = math::exp(mu);  // time_snr_shift expects exp(mu), not mu

        for (uint i = 0; i < config.num_steps; i++) {
            float t = 1.0f - (float)i * (1.0f - 1.0f / (float)config.num_steps)
                / (float)(config.num_steps > 1 ? config.num_steps - 1 : 1);
            sigmas[i] = pipelines::time_snr_shift(flux_shift, t);
        }
        sigmas[config.num_steps] = 0.0f;
        io::printfn("  Scheduler: mu=%.3f, shift=%.3f", mu, flux_shift);
    }

    io::printf("  sigmas:");
    for (uint i = 0; i <= config.num_steps; i++) io::printf(" %.3f", sigmas[i]);
    io::printn("");

    pipelines::FlowMatchingState scheduler = {
        .num_steps = config.num_steps,
        .sigmas = sigmas,
    };
    defer scheduler.free();

    // Denoising loop
    for (uint step = 0; step < config.num_steps; step++) {
        float sigma = scheduler.get_timestep(step);
        float dt = scheduler.get_dt(step);
        io::printfn("  Step %d/%d: sigma=%.4f, dt=%.4f", step + 1, config.num_steps, sigma, dt);

        // DiT forward pass
        dit.forward(&text_embeddings, n_txt, sigma)!!;

        // Debug: print velocity and latent values
        if (step < 3 || step == config.num_steps - 1) {
            usz dbg_sz = 32;
            vk::Memory vel_dbg = vk::new_buffer(
                allocator: &ctx.allocator,
                usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data: null, data_size: dbg_sz,
            )!!;
            vk::Memory lat_dbg = vk::new_buffer(
                allocator: &ctx.allocator,
                usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data: null, data_size: dbg_sz,
            )!!;
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dbg_cmd) {
                vk::cmdCopyBuffer(dbg_cmd, dit.acts.velocity.gpu_buffer.buffer, vel_dbg.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = dbg_sz }});
                vk::cmdCopyBuffer(dbg_cmd, dit.acts.latent.gpu_buffer.buffer, lat_dbg.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = dbg_sz }});
            }!!;
            float* vptr = (float*)vel_dbg.data();
            float* lptr = (float*)lat_dbg.data();
            io::printf("    vel[0:4]= ");
            for (int i = 0; i < 4; i++) io::printf("%.4f ", vptr[i]);
            io::printf("  lat[0:4]= ");
            for (int i = 0; i < 4; i++) io::printf("%.4f ", lptr[i]);
            io::printn("");
            vel_dbg.free();
            lat_dbg.free();

            // Spatial variation diagnostic: velocity at different positions
            // velocity is [128, lat_h, lat_w] = [128, 32, 32]
            if (step == 0) {
                usz spatial_sz = (usz)latent_elems * 4;
                vk::Memory vel_full = vk::new_buffer(
                    allocator: &ctx.allocator,
                    usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                    properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    data: null, data_size: spatial_sz,
                )!!;
                ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer sc) {
                    vk::cmdCopyBuffer(sc, dit.acts.velocity.gpu_buffer.buffer, vel_full.buffer, 1,
                        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = spatial_sz }});
                }!!;
                float* vf = (float*)vel_full.data();
                // Check channel 0 at various positions
                uint hw = lat_h * lat_w;
                io::printfn("    vel ch0 (0,0)=%.4f (0,16)=%.4f (16,0)=%.4f (16,16)=%.4f (31,31)=%.4f",
                    vf[0*hw + 0*lat_w + 0], vf[0*hw + 0*lat_w + 16], vf[0*hw + 16*lat_w + 0],
                    vf[0*hw + 16*lat_w + 16], vf[0*hw + 31*lat_w + 31]);
                io::printfn("    vel ch64 (0,0)=%.4f (0,16)=%.4f (16,0)=%.4f (16,16)=%.4f (31,31)=%.4f",
                    vf[64*hw + 0*lat_w + 0], vf[64*hw + 0*lat_w + 16], vf[64*hw + 16*lat_w + 0],
                    vf[64*hw + 16*lat_w + 16], vf[64*hw + 31*lat_w + 31]);
                // Compute velocity spatial variance for channel 0
                float v_mean = 0; float v_var = 0;
                for (uint i = 0; i < hw; i++) v_mean += vf[i];
                v_mean /= (float)hw;
                for (uint i = 0; i < hw; i++) v_var += (vf[i] - v_mean) * (vf[i] - v_mean);
                v_var /= (float)hw;
                io::printfn("    vel ch0: mean=%.4f var=%.6f std=%.4f", v_mean, v_var, math::sqrt(v_var));
                vel_full.free();
            }
        }

        // Euler step: latent += dt * velocity
        llm::RenderGraph euler_g;
        euler_g.init(&ctx)!!;
        zimage::FlowEulerPC euler_pc = { .n = latent_elems, .dt = dt };
        euler_g.dispatch(&dit_kernels.dit.flow_euler_step,
            { &dit.acts.latent, &dit.acts.velocity },
            &euler_pc, llm::ceil_div(latent_elems, 256));
        euler_g.flush()!!;
    }

    io::printfn("  Denoising complete");

    // --- Phase 4: VAE decode ---
    io::printfn("\n--- Phase 4: VAE Decoding ---");

    // Download denoised latent [128, lat_h, lat_w] from GPU
    {
        vk::Memory dl_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: (usz)latent_elems * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dl_cmd) {
            vk::cmdCopyBuffer(dl_cmd, dit.acts.latent.gpu_buffer.buffer, dl_staging.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 }});
        }!!;
        float* mapped = (float*)dl_staging.data();
        mem::copy(latent_cpu.ptr, mapped, (usz)latent_elems * 4);
        dl_staging.free();
    }

    // Debug: save denoised latent channel 0 as grayscale to check spatial structure
    {
        uint hw = lat_h * lat_w;
        // Find min/max of channel 0
        float ch0_min = latent_cpu[0]; float ch0_max = latent_cpu[0];
        for (uint i = 0; i < hw; i++) {
            if (latent_cpu[i] < ch0_min) ch0_min = latent_cpu[i];
            if (latent_cpu[i] > ch0_max) ch0_max = latent_cpu[i];
        }
        io::printfn("  Latent ch0: min=%.4f max=%.4f", ch0_min, ch0_max);
        // Create grayscale image from channel 0
        float range = ch0_max - ch0_min;
        if (range < 0.001f) range = 1.0f;
        usz lat_px = (usz)lat_h * lat_w * 3;
        float[] lat_img = mem::new_array(float, lat_px);
        for (uint y = 0; y < lat_h; y++) {
            for (uint x = 0; x < lat_w; x++) {
                float v = (latent_cpu[y * lat_w + x] - ch0_min) / range;
                lat_img[(y * lat_w + x) * 3 + 0] = v;
                lat_img[(y * lat_w + x) * 3 + 1] = v;
                lat_img[(y * lat_w + x) * 3 + 2] = v;
            }
        }
        image::Image lat_result = pipelines::float_tensor_to_image(lat_img, lat_w, lat_h, 3)!!;
        png::save_file(&lat_result, "debug_latent_ch0.png")!!;
        io::printfn("  Saved latent ch0 debug image to debug_latent_ch0.png");
        mem::free(lat_img);
    }

    // Flux.2 VAE decode:
    // 1. Apply BatchNorm (inv_normalize) in 128-ch space using bn.running_mean/var
    // 2. Pixel shuffle: [128, lat_h, lat_w] → [32, 2*lat_h, 2*lat_w]
    // 3. post_quant_conv: 1x1 conv [32] → [32]
    // 4. Standard VAE decoder

    // Load BN and post_quant_conv weights from safetensors
    llm::SafetensorsFile vae_sf = llm::safetensors_open(config.vae_path)!!;
    defer vae_sf.close();

    // Diagnostic: list ALL tensors in VAE (to find BN and post_quant)
    io::printfn("  VAE has %d tensors:", (uint)vae_sf.n_tensors);
    for (usz i = 0; i < vae_sf.n_tensors; i++) {
        String name = vae_sf.tensors[i].name;
        // Only print non-decoder tensors and first few decoder tensors
        if (!name.contains("decoder.") || i < 10) {
            io::printf("    %s [", name);
            for (uint d = 0; d < vae_sf.tensors[i].n_dims; d++) {
                if (d > 0) io::printf(", ");
                io::printf("%d", (uint)vae_sf.tensors[i].shape[d]);
            }
            io::printfn("]");
        }
    }
    // Count decoder tensors
    uint dec_count = 0;
    for (usz i = 0; i < vae_sf.n_tensors; i++) {
        if (vae_sf.tensors[i].name.contains("decoder.")) dec_count++;
    }
    io::printfn("    ... (%d decoder tensors)", dec_count);

    // Apply BatchNorm inverse normalization: x = x * sqrt(var + eps) + mean
    bool skip_bn = false;
    {
        float[128] bn_mean;
        float[128] bn_var;
        bool has_bn = false;
        if (try info = vae_sf.find_tensor("bn.running_mean")) {
            float* data = (float*)(vae_sf.data_base + (usz)info.data_start);
            mem::copy(&bn_mean, data, 128 * 4);
            has_bn = true;
        }
        if (try info = vae_sf.find_tensor("bn.running_var")) {
            float* data = (float*)(vae_sf.data_base + (usz)info.data_start);
            mem::copy(&bn_var, data, 128 * 4);
        }
        if (has_bn) {
            float eps = 1e-5f;
            // Print some BN stats for debugging
            io::printfn("  BN mean[0:4] = %.4f %.4f %.4f %.4f", bn_mean[0], bn_mean[1], bn_mean[2], bn_mean[3]);
            io::printfn("  BN var[0:4] = %.4f %.4f %.4f %.4f", bn_var[0], bn_var[1], bn_var[2], bn_var[3]);
            // Print latent stats before BN
            float lat_min = latent_cpu[0]; float lat_max = latent_cpu[0]; float lat_sum = 0;
            for (uint i = 0; i < latent_elems; i++) {
                if (latent_cpu[i] < lat_min) lat_min = latent_cpu[i];
                if (latent_cpu[i] > lat_max) lat_max = latent_cpu[i];
                lat_sum += latent_cpu[i];
            }
            io::printfn("  Latent before BN: min=%.4f max=%.4f mean=%.4f", lat_min, lat_max, lat_sum / (float)latent_elems);

            if (!skip_bn) {
                for (uint c = 0; c < 128; c++) {
                    float std = math::sqrt(bn_var[c] + eps);
                    for (uint j = 0; j < lat_h * lat_w; j++) {
                        uint idx = c * lat_h * lat_w + j;
                        latent_cpu[idx] = latent_cpu[idx] * std + bn_mean[c];
                    }
                }
            } else {
                io::printfn("  *** SKIPPING BN inverse (debug) ***");
            }

            // Print latent stats after BN
            lat_min = latent_cpu[0]; lat_max = latent_cpu[0]; lat_sum = 0;
            for (uint i = 0; i < latent_elems; i++) {
                if (latent_cpu[i] < lat_min) lat_min = latent_cpu[i];
                if (latent_cpu[i] > lat_max) lat_max = latent_cpu[i];
                lat_sum += latent_cpu[i];
            }
            io::printfn("  Latent after BN: min=%.4f max=%.4f mean=%.4f", lat_min, lat_max, lat_sum / (float)latent_elems);
        }
        io::printfn("  Applied BatchNorm inv_normalize (128 channels)");
    }

    // Pixel shuffle: [128, lat_h, lat_w] → [32, 2*lat_h, 2*lat_w]
    uint vae_h = lat_h * 2;  // = img_h / 8
    uint vae_w = lat_w * 2;  // = img_w / 8
    uint vae_elems = FLUX_VAE_CH * vae_h * vae_w;  // 32 * vae_h * vae_w
    float[] vae_latent = mem::new_array(float, vae_elems);
    pixel_shuffle_cpu(latent_cpu, vae_latent, FLUX_LATENT_CH, lat_h, lat_w, 2);
    mem::free(latent_cpu);

    // Apply post_quant_conv (1x1 conv, [32,32,1,1] — per-pixel linear transform)
    {
        float[32 * 32] pqc_weight;
        float[32] pqc_bias;
        if (try info = vae_sf.find_tensor("post_quant_conv.weight")) {
            float* data = (float*)(vae_sf.data_base + (usz)info.data_start);
            mem::copy(&pqc_weight, data, 32 * 32 * 4);
        }
        if (try info = vae_sf.find_tensor("post_quant_conv.bias")) {
            float* data = (float*)(vae_sf.data_base + (usz)info.data_start);
            mem::copy(&pqc_bias, data, 32 * 4);
        }
        // Apply 1x1 conv per spatial position: out[c] = bias[c] + sum_k(weight[c,k] * in[k])
        uint spatial = vae_h * vae_w;
        float[32] tmp;
        for (uint s = 0; s < spatial; s++) {
            // Read input for this spatial position
            for (uint c = 0; c < 32; c++) {
                tmp[c] = vae_latent[c * spatial + s];
            }
            // Compute output
            for (uint oc = 0; oc < 32; oc++) {
                float val = pqc_bias[oc];
                for (uint ic = 0; ic < 32; ic++) {
                    val += pqc_weight[oc * 32 + ic] * tmp[ic];
                }
                vae_latent[oc * spatial + s] = val;
            }
        }
        io::printfn("  Applied post_quant_conv (32→32 1x1)");
    }

    // Load VAE decoder (reuse vae_sf already opened above)
    zimage::VaeKernels vae_kernels = zimage::create_vae_kernels(&ctx)!!;
    defer vae_kernels.free(ctx.device);

    zimage::FluxVaeDecoder vae_dec = zimage::load_flux_vae_decoder(&ctx, &vae_sf, &vae_kernels)!!;
    defer vae_dec.free();

    zimage::FluxVaeActivations vae_acts = zimage::allocate_flux_vae_activations(&ctx, img_h)!!;
    defer vae_acts.free();

    // Upload pixel-shuffled latent to VAE input
    // The VAE conv_in will read however many channels it expects from the buffer
    {
        vk::Memory lat_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: vae_latent.ptr, data_size: (usz)vae_elems * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer vae_cmd) {
            vk::cmdCopyBuffer(vae_cmd, lat_staging.buffer,
                vae_acts.buf_a.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)vae_elems * 4 }});
        }!!;
        lat_staging.free();
    }
    mem::free(vae_latent);

    io::printfn("  Running VAE decoder on [%d, %d, %d]...", FLUX_VAE_CH, vae_h, vae_w);
    vae_dec.forward(&vae_acts, vae_h, vae_w)!!;

    // --- Phase 5: Download and save ---
    io::printfn("\n--- Phase 5: Saving output ---");

    usz total_px = (usz)img_h * img_w * 3;
    float[] img_data = mem::new_array(float, total_px);

    {
        vk::Memory dl_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: total_px * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer img_cmd) {
            vk::cmdCopyBuffer(img_cmd, vae_acts.buf_a.gpu_buffer.buffer, dl_staging.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = total_px * 4 }});
        }!!;
        float* mapped = (float*)dl_staging.data();
        mem::copy(img_data.ptr, mapped, total_px * 4);
        dl_staging.free();
    }

    image::Image result = pipelines::float_tensor_to_image(img_data, img_w, img_h, 3)!!;
    mem::free(img_data);

    png::save_file(&result, config.output_path)!!;
    io::printfn("  Saved to %s (%dx%d)", config.output_path, img_w, img_h);

    // Cleanup
    text_embeddings.free();
    io::printfn("\n=== Pipeline complete ===");
}

// --- Pixel shuffle (depth-to-space) on CPU ---
// Input:  [C, H, W] where C = out_ch * r * r
// Output: [out_ch, H*r, W*r]
fn void pixel_shuffle_cpu(float[] input, float[] output, uint in_channels, uint in_h, uint in_w, uint r) {
    uint out_ch = in_channels / (r * r);
    uint out_h = in_h * r;
    uint out_w = in_w * r;

    for (uint c = 0; c < out_ch; c++) {
        for (uint oh = 0; oh < out_h; oh++) {
            for (uint ow = 0; ow < out_w; ow++) {
                // Map output (c, oh, ow) back to input channel
                uint sub_h = oh % r;
                uint sub_w = ow % r;
                uint in_c = c * r * r + sub_h * r + sub_w;
                uint ih = oh / r;
                uint iw = ow / r;

                uint in_idx = in_c * in_h * in_w + ih * in_w + iw;
                uint out_idx = c * out_h * out_w + oh * out_w + ow;
                output[out_idx] = input[in_idx];
            }
        }
    }
}

// --- Multi-layer text encoding ---
// Encode text through Qwen3, extract hidden states at 3 specific layers,
// concatenate them per-token: [seq_len, 3 * dim]
// Global state for text normalization (set before calling encode_text_multilayer)
float[] g_output_norm_cpu;
float g_rms_eps;

fn llm::Tensor? encode_text_multilayer(
    llm::DeviceContext* ctx,
    llm_text::LlmModel* model,
    uint[] tokens,
    uint seq_len,
    uint dim
) {
    llm::ModelConfig* c = &model.config;
    uint n_extract = 3;  // Number of layers to extract

    // Determine which layers to extract
    // Diffusers uses text_encoder_out_layers=(9, 18, 27) which are 1-indexed hidden_states indices
    // hidden_states[k] = output after layer k-1. So we capture after layers 8, 17, 26 (0-indexed)
    // For N-layer model: spacing = N/4, extract = {spacing-1, spacing*2-1, spacing*3-1}
    uint[3] extract_layers;
    if (c.n_layers >= 12) {
        uint spacing = c.n_layers / 4;
        extract_layers = { spacing - 1, spacing * 2 - 1, spacing * 3 - 1 };
    } else {
        extract_layers = { 0, c.n_layers / 2, c.n_layers - 1 };
    }
    io::printfn("  Extracting hidden states from layers: %d, %d, %d (of %d)",
        extract_layers[0], extract_layers[1], extract_layers[2], c.n_layers);

    // Working hidden states buffer
    llm::Tensor hidden_states = llm::create_f32_tensor(ctx, { (ulong)seq_len * dim, 0, 0, 0 }, 1)!!;

    // Buffers for 3 extracted layer outputs: each [seq_len, dim]
    llm::Tensor[3] layer_outputs;
    for (uint i = 0; i < 3; i++) {
        layer_outputs[i] = llm::create_f32_tensor(ctx, { (ulong)seq_len * dim, 0, 0, 0 }, 1)!!;
    }

    // 1. Embed all tokens
    for (uint pos = 0; pos < seq_len; pos++) {
        float* staging_ptr = (float*)model.embedding_staging.data();
        llm::dequant_embedding_row(model.weights.embedding_data, &model.weights.embedding_info, tokens[pos], staging_ptr, dim);
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer emb_cmd) {
            vk::cmdCopyBuffer(emb_cmd, model.embedding_staging.buffer, hidden_states.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)pos * dim * 4, .size = (ulong)dim * 4 }});
        }!!;
    }

    // 2. Process layers, snapshot at extract points
    for (uint l = 0; l < c.n_layers; l++) {
        llm_text::LayerWeights lw = llm_text::load_single_layer(ctx, model.gf, model.names, c, l)!!;

        llm::RenderGraph g;
        g.init(ctx)!!;
        for (uint pos = 0; pos < seq_len; pos++) {
            if (g.needs_barrier) llm::compute_barrier(g.cmd);
            vk::cmdCopyBuffer(g.cmd, hidden_states.gpu_buffer.buffer, model.acts.hidden.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = (ulong)pos * dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
            g.needs_barrier = true;

            model.forward_layer(&g, &lw, l, pos)!!;

            if (g.needs_barrier) llm::compute_barrier(g.cmd);
            vk::cmdCopyBuffer(g.cmd, model.acts.hidden.gpu_buffer.buffer, hidden_states.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)pos * dim * 4, .size = (ulong)dim * 4 }});
            g.needs_barrier = true;
        }
        g.flush()!!;
        llm_text::free_layer_weights(&lw, c);

        // Snapshot if this is an extraction layer
        for (uint i = 0; i < 3; i++) {
            if (l == extract_layers[i]) {
                ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer snap_cmd) {
                    vk::cmdCopyBuffer(snap_cmd, hidden_states.gpu_buffer.buffer,
                        layer_outputs[i].gpu_buffer.buffer, 1,
                        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * dim * 4 }});
                }!!;
                io::printfn("  Captured layer %d hidden states", l);
            }
        }
    }
    hidden_states.free();

    // 3. Concatenate: interleave 3 layer outputs → [seq_len, 3*dim]
    uint out_dim = dim * 3;
    usz out_size = (usz)seq_len * out_dim;
    float[] concat_cpu = mem::new_array(float, out_size);

    // Download all 3 layer outputs
    float*[3] layer_data;
    for (uint i = 0; i < 3; i++) {
        layer_data[i] = (float*)mem::calloc((usz)seq_len * dim * 4);
        vk::Memory dl = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: (usz)seq_len * dim * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dl_cmd) {
            vk::cmdCopyBuffer(dl_cmd, layer_outputs[i].gpu_buffer.buffer, dl.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * dim * 4 }});
        }!!;
        mem::copy(layer_data[i], (float*)dl.data(), (usz)seq_len * dim * 4);
        dl.free();
        layer_outputs[i].free();
    }

    // Diagnostic: check per-layer stats and per-position variation
    for (uint i = 0; i < 3; i++) {
        float rms_first = 0; float rms_last = 0;
        for (uint d = 0; d < dim; d++) {
            rms_first += layer_data[i][d] * layer_data[i][d];
            rms_last += layer_data[i][(seq_len-1)*dim + d] * layer_data[i][(seq_len-1)*dim + d];
        }
        rms_first = math::sqrt(rms_first / (float)dim);
        rms_last = math::sqrt(rms_last / (float)dim);
        io::printfn("  Layer %d raw: pos0[0:2]=%.2f,%.2f rms_pos0=%.2f rms_last=%.2f",
            extract_layers[i], layer_data[i][0], layer_data[i][1], rms_first, rms_last);
    }

    // Note: official Flux2KleinPipeline does NOT normalize hidden states
    // Raw Qwen3 hidden states are passed directly to txt_in projection

    // Concatenate per-token: [layer0_token0, layer1_token0, layer2_token0, layer0_token1, ...]
    for (uint t = 0; t < seq_len; t++) {
        for (uint i = 0; i < 3; i++) {
            mem::copy(
                &concat_cpu[t * out_dim + i * dim],
                &layer_data[i][t * dim],
                (usz)dim * 4
            );
        }
    }

    for (uint i = 0; i < 3; i++) mem::free(layer_data[i]);

    // Upload concatenated embeddings to GPU
    llm::Tensor result = llm::create_f32_tensor(ctx, { (ulong)out_size, 0, 0, 0 }, 1)!!;
    {
        vk::Memory ul = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: concat_cpu.ptr, data_size: out_size * 4,
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer ul_cmd) {
            vk::cmdCopyBuffer(ul_cmd, ul.buffer, result.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)out_size * 4 }});
        }!!;
        ul.free();
    }
    mem::free(concat_cpu);

    return result;
}
