module vit_test;

import vk;
import llm;
import llm::pipelines;
import qwen_edit;
import zimage;
import std::io;
import std::math;
import std::core::mem;
import image;
import image::png;

fn int main(String[] args) {
    if (args.len < 4) {
        io::printfn("Usage: vit-test <mmdit.gguf> <text_model.gguf> <vae.safetensors> [prompt] [height] [width] [steps]");
        io::printfn("  Runs full text→MMDiT→VAE→image pipeline.");
        io::printfn("  height/width: output image size (default 256, must be divisible by 16)");
        return 1;
    }

    String mmdit_path = args[1];
    String text_path = args[2];
    String vae_path = args[3];
    String prompt = args.len > 4 ? args[4] : "a red circle on a white background";
    uint img_h = args.len > 5 ? (args[5].to_uint() ?? 256) : 256;
    uint img_w = args.len > 6 ? (args[6].to_uint() ?? 256) : 256;
    uint n_steps = args.len > 7 ? (args[7].to_uint() ?? 20) : 20;

    // Latent dimensions (8x downsampled from image)
    uint lat_h = img_h / 8;
    uint lat_w = img_w / 8;
    uint patch_size = 2;
    uint n_patches = (lat_h / patch_size) * (lat_w / patch_size);
    uint patch_dim = 64;  // 16 channels * 2 * 2

    io::printfn("=== Qwen Image Edit Pipeline ===");
    io::printfn("  Prompt: \"%s\"", prompt);
    io::printfn("  Output: %dx%d (latent: %dx%d, %d patches)", img_w, img_h, lat_w, lat_h, n_patches);
    io::printfn("  Steps: %d", n_steps);

    // Init Vulkan
    llm::DeviceContext ctx = llm::createContext()!!;
    defer ctx.free();

    // === Phase 1: Text Encoding (conditional + unconditional) ===
    llm::Tensor text_emb = qwen_edit::encode_text_prompt(&ctx, prompt, text_path)!!;
    uint n_txt = (uint)text_emb.shape[0];
    uint txt_dim = (uint)text_emb.shape[1];
    io::printfn("\n  Cond text embeddings: [%d, %d]", n_txt, txt_dim);

    // Encode unconditional (negative prompt = single space)
    llm::Tensor uncond_emb = qwen_edit::encode_text_prompt(&ctx, " ", text_path)!!;
    uint n_uncond = (uint)uncond_emb.shape[0];
    io::printfn("  Uncond text embeddings: [%d, %d]", n_uncond, txt_dim);

    // Use max of n_txt and n_uncond for activation allocation
    uint n_txt_max = n_txt > n_uncond ? n_txt : n_uncond;

    // === Phase 2: Load MMDiT ===
    io::printfn("\n  Loading MMDiT...");
    qwen_edit::MMDiTModel mmdit = qwen_edit::load_mmdit_model(&ctx, mmdit_path)!!;
    mmdit.acts = qwen_edit::allocate_mmdit_activations(&ctx, n_patches, n_txt_max)!!;

    // Copy cond text embeddings to MMDiT activation buffer
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, text_emb.gpu_buffer.buffer, mmdit.acts.txt_emb_input.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_txt * txt_dim * 4 }});
    }!!;
    text_emb.free();

    // Initialize patched_latent with random noise (Gaussian-like from deterministic pattern)
    {
        uint img_elems = n_patches * patch_dim;
        usz img_size = (usz)img_elems * 4;

        vk::Memory staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data_size: img_size,
        )!!;
        float* sdata = (float*)staging.data();

        // Simple pseudo-random noise (Box-Muller-like from LCG)
        uint seed = 42;
        for (uint i = 0; i < img_elems; i++) {
            seed = seed * 1103515245 + 12345;
            float u1 = (float)((seed >> 16) & 0x7FFF) / 32767.0f;
            seed = seed * 1103515245 + 12345;
            float u2 = (float)((seed >> 16) & 0x7FFF) / 32767.0f;
            // Box-Muller transform
            u1 = u1 < 1e-7f ? 1e-7f : u1;  // Avoid log(0)
            float z = std::math::sqrt(-2.0f * std::math::ln(u1)) * std::math::cos(2.0f * 3.14159265f * u2);
            sdata[i] = z;
        }
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
            vk::cmdCopyBuffer(cmd, staging.buffer, mmdit.acts.patched_latent.gpu_buffer.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)img_size }});
        }!!;
        staging.free();
        io::printfn("  Initialized latent with Gaussian noise.");
    }

    // === Phase 3: MMDiT Denoising ===
    float cfg_scale = 4.0f;
    mmdit.denoise(n_patches, n_txt, lat_h, lat_w, n_steps, cfg_scale, &uncond_emb, n_uncond)!!;

    // === Phase 4: VAE Decode ===
    io::printfn("\n=== VAE Decode ===");

    // Get denoised latent from patched_latent (unpatchify happens inside denoise)
    // Actually denoise() already unpatchifies. But let me get the output from it.
    // The denoise function currently creates and frees output_latent internally.
    // Let me read the patched_latent and unpatchify it here instead.

    // Unpatchify patched_latent → latent [16, lat_h, lat_w]
    uint latent_elems = 16 * lat_h * lat_w;

    // Read patched_latent back to CPU for denormalization
    usz latent_bytes = (usz)latent_elems * 4;
    vk::Memory lat_staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: latent_bytes,
    )!!;

    // Unpatchify on GPU first
    llm::Tensor output_latent = llm::create_f32_tensor(&ctx, { (ulong)latent_elems, 0, 0, 0 }, 1)!!;
    {
        llm::RenderGraph g;
        g.init(&ctx)!!;
        qwen_edit::MMDiTPatchifyPC unpatch_pc = { .channels = 16, .height = lat_h, .width = lat_w, .patch_size = patch_size };
        g.dispatch(&mmdit.kernels.unpatchify,
            { &mmdit.acts.patched_latent, &output_latent, &mmdit.acts.patched_latent, &output_latent },
            &unpatch_pc, llm::ceil_div(latent_elems, 256));
        g.flush()!!;
    }

    // Download latent to CPU for denormalization
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
    zimage::VaeKernels vae_kernels = zimage::create_vae_kernels(&ctx)!!;
    llm::SafetensorsFile sf = llm::safetensors_open(vae_path)!!;
    qwen_edit::QwenVaeDecoder vae_decoder = qwen_edit::load_qwen_vae_decoder(&ctx, &sf, &vae_kernels)!!;
    sf.close();

    // Allocate VAE activation buffers
    zimage::FluxVaeActivations vae_acts = zimage::allocate_flux_vae_activations(&ctx, img_h > img_w ? img_h : img_w)!!;

    // Upload denormalized latent to VAE input buffer
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, lat_staging.buffer, vae_acts.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_bytes }});
    }!!;
    lat_staging.free();
    output_latent.free();

    // Run VAE decode
    io::printfn("  Decoding latent [16, %d, %d] → image [3, %d, %d]...", lat_h, lat_w, img_h, img_w);
    vae_decoder.forward(&ctx, &vae_acts, lat_h, lat_w)!!;

    // Download decoded image
    usz total_px = (usz)img_h * img_w * 3;
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
    image::Image result = pipelines::float_tensor_to_image(img_data, img_w, img_h, 3)!!;
    mem::free(img_data);

    String output_path = "output.png";
    png::save_file(&result, output_path)!!;
    io::printfn("\n=== Image saved to %s ===", output_path);

    // Cleanup
    vae_acts.free();
    vae_kernels.free(ctx.device);

    return 0;
}
