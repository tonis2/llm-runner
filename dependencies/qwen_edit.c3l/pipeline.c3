module qwen_edit;

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
import zimage;

struct QwenEditState (llm::pipelines::Pipeline) {
    bool loaded;
    String model_path;
    String text_model_path;
    String vae_path;
    bool offload;
    
    // Model state (loaded during generate)
    MMDiTModel mmdit;
    bool mmdit_loaded;
    
    // Image dimensions
    uint img_h;
    uint img_w;
}

fn void? QwenEditState.load(QwenEditState* self, llm::DeviceContext* ctx, String model_path, llm::pipelines::PipelineOptions* opts) @dynamic {
    self.loaded = true;
    self.model_path = model_path;
    self.text_model_path = opts.text_model_path;
    self.vae_path = opts.vae_path;
    self.offload = opts.offload;
    self.mmdit_loaded = false;
}

fn image::Image? QwenEditState.generate(QwenEditState* self, llm::DeviceContext* ctx, llm::pipelines::GenerationInputs* inputs) @dynamic {
    if (!self.loaded) {
        io::printfn("Error: pipeline not loaded");
        return llm::pipelines::PIPELINE_INVALID_INPUT~;
    }
    
    io::printfn("\n=== Qwen Edit Pipeline (via Pipeline interface) ===");
    io::printfn("  Prompt: \"%s\"", inputs.prompt);
    io::printfn("  Steps: %d, Seed: %d, CFG: %.1f",
        inputs.num_steps, inputs.seed, inputs.cfg_scale);
    
    // Determine image size
    uint img_size = inputs.image_size;
    if (img_size % 16 != 0) {
        img_size = (img_size + 8) / 16 * 16;
        io::printfn("  Note: size rounded to %d (must be multiple of 16)", img_size);
    }
    self.img_h = img_size;
    self.img_w = img_size;
    
    uint lat_h = img_size / 8;
    uint lat_w = img_size / 8;
    uint patch_size = 2;
    uint n_patches = (lat_h / patch_size) * (lat_w / patch_size);
    uint patch_dim = 64;  // 16 channels * 2 * 2
    
    Clock total_start = clock::now();
    
    // === Phase 1: Text Encoding ===
    io::printfn("\n[1/4] Text encoding...");
    if (self.text_model_path.len == 0) {
        io::printfn("Error: text_model_path required");
        return llm::pipelines::PIPELINE_INVALID_INPUT~;
    }
    
    llm::Tensor text_emb = encode_text_prompt(ctx, inputs.prompt, self.text_model_path)!!;
    uint n_txt = (uint)text_emb.shape[0];
    uint txt_dim = (uint)text_emb.shape[1];
    io::printfn("  Cond text embeddings: [%d, %d]", n_txt, txt_dim);
    
    // Debug: print text embedding stats
    print_tensor_stats(ctx, &text_emb, "text_emb")!!;
    
    // Unconditional encoding
    llm::Tensor uncond_emb = encode_text_prompt(ctx, " ", self.text_model_path)!!;
    uint n_uncond = (uint)uncond_emb.shape[0];
    io::printfn("  Uncond text embeddings: [%d, %d]", n_uncond, txt_dim);
    print_tensor_stats(ctx, &uncond_emb, "uncond_emb")!!;
    
    uint n_txt_max = n_txt > n_uncond ? n_txt : n_uncond;
    
    // === Phase 2: Load MMDiT ===
    io::printfn("\n[2/4] Loading MMDiT...");
    self.mmdit = load_mmdit_model(ctx, self.model_path)!!;
    self.mmdit_loaded = true;
    self.mmdit.acts = allocate_mmdit_activations(ctx, n_patches, n_txt_max)!!;
    
    // Copy cond text embeddings to MMDiT activation buffer
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, text_emb.gpu_buffer.buffer, self.mmdit.acts.txt_emb_input.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_txt * txt_dim * 4 }});
    }!!;
    text_emb.free();
    
    // === Phase 3: Initialize latent ===
    io::printfn("\n[3/4] Initializing latent...");
    uint img_elems = n_patches * patch_dim;
    usz img_size_bytes = (usz)img_elems * 4;
    
    // Use seed for reproducibility
    uint seed = inputs.seed;
    
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: img_size_bytes,
    )!!;
    float* sdata = (float*)staging.data();
    
    // Gaussian noise initialization
    for (uint i = 0; i < img_elems; i++) {
        seed = seed * 1103515245 + 12345;
        float u1 = (float)((seed >> 16) & 0x7FFF) / 32767.0f;
        seed = seed * 1103515245 + 12345;
        float u2 = (float)((seed >> 16) & 0x7FFF) / 32767.0f;
        u1 = u1 < 1e-7f ? 1e-7f : u1;
        float z = std::math::sqrt(-2.0f * std::math::ln(u1)) * std::math::cos(2.0f * 3.14159265f * u2);
        sdata[i] = z;
    }
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, staging.buffer, self.mmdit.acts.patched_latent.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)img_size_bytes }});
    }!!;
    staging.free();
    
    print_tensor_stats(ctx, &self.mmdit.acts.patched_latent, "init_latent")!!;
    
    // === Phase 4: MMDiT Denoising ===
    io::printfn("\n[4/4] Denoising...");
    float cfg_scale = inputs.cfg_scale;
    self.mmdit.denoise(n_patches, n_txt, lat_h, lat_w, inputs.num_steps, cfg_scale, &uncond_emb, n_uncond)!!;
    
    print_tensor_stats(ctx, &self.mmdit.acts.patched_latent, "denoised_latent")!!;
    
    // === Phase 5: Unpatchify and VAE Decode ===
    io::printfn("\n[5/5] VAE Decode...");
    
    uint latent_elems = 16 * lat_h * lat_w;
    usz latent_bytes = (usz)latent_elems * 4;
    
    // Unpatchify on GPU
    llm::Tensor output_latent = llm::create_f32_tensor(ctx, { (ulong)latent_elems, 0, 0, 0 }, 1)!!;
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        MMDiTPatchifyPC unpatch_pc = { .channels = 16, .height = lat_h, .width = lat_w, .patch_size = patch_size };
        g.dispatch(&self.mmdit.kernels.unpatchify,
            { &self.mmdit.acts.patched_latent, &output_latent, &self.mmdit.acts.patched_latent, &output_latent },
            &unpatch_pc, llm::ceil_div(latent_elems, 256));
        g.flush()!!;
    }
    
    print_tensor_stats(ctx, &output_latent, "unpatchified_latent")!!;
    
    // Download latent to CPU for denormalization
    vk::Memory lat_staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: latent_bytes,
    )!!;
    
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, output_latent.gpu_buffer.buffer, lat_staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_bytes }});
    }!!;
    
    float* latent_cpu = (float*)lat_staging.data();
    
    // Denormalize latent: latents = latents * std + mean (per-channel, Qwen-Image VAE)
    float[16] lat_mean = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                           0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
    float[16] lat_std  = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                           3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.916f};
    uint spatial = lat_h * lat_w;
    for (uint c = 0; c < 16; c++) {
        for (uint s = 0; s < spatial; s++) {
            latent_cpu[c * spatial + s] = latent_cpu[c * spatial + s] * lat_std[c] + lat_mean[c];
        }
    }
    
    io::printf("  Latent first values (denorm): [");
    for (uint i = 0; i < 5; i++) {
        if (i > 0) io::printf(", ");
        io::printf("%.3f", latent_cpu[i]);
    }
    io::printfn("]");
    
    // Load Qwen-Image VAE
    io::printfn("  Loading Qwen-Image VAE...");
    zimage::VaeKernels vae_kernels = zimage::create_vae_kernels(ctx)!!;
    llm::SafetensorsFile sf = llm::safetensors_open(self.vae_path)!!;
    QwenVaeDecoder vae_decoder = load_qwen_vae_decoder(ctx, &sf, &vae_kernels)!!;
    sf.close();
    
    // Allocate VAE activation buffers
    zimage::FluxVaeActivations vae_acts = zimage::allocate_flux_vae_activations(ctx, self.img_h > self.img_w ? self.img_h : self.img_w)!!;
    
    // Upload denormalized latent to VAE input buffer
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, lat_staging.buffer, vae_acts.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_bytes }});
    }!!;
    lat_staging.free();
    output_latent.free();
    
    // Run VAE decode
    io::printfn("  Decoding latent [16, %d, %d] -> image [3, %d, %d]...", lat_h, lat_w, self.img_h, self.img_w);
    vae_decoder.forward(ctx, &vae_acts, lat_h, lat_w)!!;
    
    // Download decoded image
    usz total_px = (usz)self.img_h * self.img_w * 3;
    float[] img_data = mem::new_array(float, total_px);
    
    vk::Memory download_staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: total_px * 4,
    )!!;
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, vae_acts.buf_a.gpu_buffer.buffer, download_staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = total_px * 4 }});
    }!!;
    
    float* mapped = (float*)download_staging.data();
    mem::copy(img_data.ptr, mapped, total_px * 4);
    download_staging.free();
    
    // Convert to image and save
    image::Image result = pipelines::float_tensor_to_image(img_data, self.img_w, self.img_h, 3)!!;
    mem::free(img_data);
    
    String output_path = "output_qwen.png";
    png::save_file(&result, output_path)!!;
    io::printfn("\n=== Image saved to %s ===", output_path);
    
    // Cleanup
    vae_acts.free();
    vae_kernels.free(ctx.device);
    // TODO: free mmdit weights
    
    double total_ms = total_start.to_now().to_sec() * 1000.0;
    io::printfn("Total time: %.1f ms", total_ms);
    
    return result;
}

fn void QwenEditState.free(QwenEditState* self) @dynamic {
    if (self.mmdit_loaded) {
        // TODO: implement proper MMDiT cleanup
        self.mmdit_loaded = false;
    }
    self.loaded = false;
}

fn llm::pipelines::Pipeline create_qwen_edit_pipeline() {
    return (llm::pipelines::Pipeline)mem::new(QwenEditState);
}
