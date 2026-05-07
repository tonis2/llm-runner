module llm;

import std::io;
import std::io::file;
import std::core::mem;
import image;
import image::png;
import vk;



// --- Load PNG as float tensor ---
// Returns image::Image (caller must call .free())
// Pixel data is in interleaved format, convert to NCHW float as needed
fn image::Image? load_png(String filename) {
    image::Image img = png::load_file(filename)!!;

    if (img.format != PixelFormat.RGB && img.format != PixelFormat.RGBA && img.format != PixelFormat.GRAYSCALE) {
        io::printfn("Error: Only RGB, RGBA, and Grayscale PNG images supported, got %s", img.format);
        img.free();
        return COMPUTE_ERROR~;
    }

    io::printfn("Image loaded: %s (%dx%d)", filename, img.width, img.height);
    return img;
}

// --- Bilinear resize + center crop to target_w x target_h ---
// Resizes the short side to fit `target` (preserving aspect ratio), then
// center-crops to exactly target_w x target_h. Output is a new image::Image
// (caller frees). Matches PIL.Image.resize(BILINEAR).crop(center) used by the
// diffusers VaeImageProcessor preprocess path with default settings.
fn image::Image? resize_center_crop(image::Image* src, uint target_w, uint target_h) {
    uint channels;
    switch (src.format) {
        case PixelFormat.GRAYSCALE: channels = 1;
        case PixelFormat.RGB: channels = 3;
        case PixelFormat.RGBA: channels = 4;
        default: channels = 3;
    }

    // Scale so that the short side maps to the matching target dimension —
    // this guarantees the long side covers the other target dimension fully,
    // so the subsequent center crop has source pixels for every output pixel.
    float scale_w = (float)target_w / (float)src.width;
    float scale_h = (float)target_h / (float)src.height;
    float scale = scale_w > scale_h ? scale_w : scale_h;
    uint scaled_w = (uint)((float)src.width * scale + 0.5f);
    uint scaled_h = (uint)((float)src.height * scale + 0.5f);
    if (scaled_w < target_w) scaled_w = target_w;
    if (scaled_h < target_h) scaled_h = target_h;

    // Center-crop offsets in the (virtual) post-resize image
    uint crop_x = (scaled_w - target_w) / 2;
    uint crop_y = (scaled_h - target_h) / 2;

    char[] out_pixels = mem::new_array(char, (usz)target_w * target_h * channels);
    usz src_bpp = (usz)src.bytes_per_pixel();

    // For each output pixel, compute the corresponding source coordinate via
    // the inverse of the scale, then bilinear-sample from the source. This
    // fuses resize + crop into a single sampling pass.
    for (uint oy = 0; oy < target_h; oy++) {
        // Map output row -> post-resize row -> source row (centered sampling)
        float src_y = ((float)(oy + crop_y) + 0.5f) / scale - 0.5f;
        if (src_y < 0.0f) src_y = 0.0f;
        if (src_y > (float)(src.height - 1)) src_y = (float)(src.height - 1);
        uint y0 = (uint)src_y;
        uint y1 = y0 + 1 < src.height ? y0 + 1 : y0;
        float fy = src_y - (float)y0;

        for (uint ox = 0; ox < target_w; ox++) {
            float src_x = ((float)(ox + crop_x) + 0.5f) / scale - 0.5f;
            if (src_x < 0.0f) src_x = 0.0f;
            if (src_x > (float)(src.width - 1)) src_x = (float)(src.width - 1);
            uint x0 = (uint)src_x;
            uint x1 = x0 + 1 < src.width ? x0 + 1 : x0;
            float fx = src_x - (float)x0;

            usz i00 = ((usz)y0 * src.width + x0) * src_bpp;
            usz i01 = ((usz)y0 * src.width + x1) * src_bpp;
            usz i10 = ((usz)y1 * src.width + x0) * src_bpp;
            usz i11 = ((usz)y1 * src.width + x1) * src_bpp;
            usz out_idx = ((usz)oy * target_w + ox) * channels;

            for (uint c = 0; c < channels; c++) {
                float v00 = (float)(uint)src.pixels[i00 + c];
                float v01 = (float)(uint)src.pixels[i01 + c];
                float v10 = (float)(uint)src.pixels[i10 + c];
                float v11 = (float)(uint)src.pixels[i11 + c];
                float v0 = v00 * (1.0f - fx) + v01 * fx;
                float v1 = v10 * (1.0f - fx) + v11 * fx;
                float v = v0 * (1.0f - fy) + v1 * fy;
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                out_pixels[out_idx + c] = (char)(v + 0.5f);
            }
        }
    }

    return {
        .width = target_w,
        .height = target_h,
        .bit_depth = 8,
        .format = src.format,
        .pixels = out_pixels,
    };
}

// --- Convert image::Image to NCHW float tensor in [-1, 1] ---
// Same as image_to_float_nchw but maps [0, 255] to [-1, 1] for VAE encoder input.
// Caller must free the returned data.
fn float[]? image_to_float_nchw_signed(image::Image* img) {
    uint channels;
    switch (img.format) {
        case PixelFormat.GRAYSCALE: channels = 1;
        case PixelFormat.RGB: channels = 3;
        case PixelFormat.RGBA: channels = 4;
        default: channels = 3;
    }

    uint total_pixels = img.width * img.height;
    float[] data = mem::new_array(float, (usz)total_pixels * channels);
    usz bpp = (usz)img.bytes_per_pixel();

    for (uint y = 0; y < img.height; y++) {
        for (uint x = 0; x < img.width; x++) {
            uint pixel_idx = y * img.width + x;
            for (uint c = 0; c < channels; c++) {
                char val = img.pixels[(usz)pixel_idx * bpp + c];
                data[c * total_pixels + pixel_idx] = (float)(uint)val / 127.5f - 1.0f;
            }
        }
    }

    return data;
}

// --- Convert image::Image to NCHW float tensor ---
// Caller must free the returned data
fn float[]? image_to_float_nchw(image::Image* img) {
    uint channels;
    switch (img.format) {
        case PixelFormat.GRAYSCALE: channels = 1;
        case PixelFormat.RGB: channels = 3;
        case PixelFormat.RGBA: channels = 4;
        default: channels = 3;
    }
    
    uint total_pixels = img.width * img.height;
    float[] data = mem::new_array(float, (usz)total_pixels * channels);
    usz bpp = (usz)img.bytes_per_pixel();
    
    for (uint y = 0; y < img.height; y++) {
        for (uint x = 0; x < img.width; x++) {
            uint pixel_idx = y * img.width + x;
            for (uint c = 0; c < channels; c++) {
                char val = img.pixels[(usz)pixel_idx * bpp + c];
                data[c * total_pixels + pixel_idx] = (float)(uint)val / 255.0f;
            }
        }
    }

    return data;
}

// --- Upload image::Image to GPU ---
fn Tensor? upload_image_to_gpu(DeviceContext* ctx, image::Image* img) {
    uint channels;
    switch (img.format) {
        case PixelFormat.GRAYSCALE: channels = 1;
        case PixelFormat.RGB: channels = 3;
        case PixelFormat.RGBA: channels = 4;
        default: channels = 3;
    }
    
    usz total = (usz)channels * img.width * img.height;
    ulong[4] shape = { (ulong)total, 0, 0, 0 };

    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: img.pixels.ptr,
        data_size: total,  // 1 byte per pixel component
    )!!;

    return {
        .gpu_buffer = gpu_buffer,
        .dtype = GGML_F32,
        .n_dims = 1,
        .shape = shape,
        .size_bytes = total,
    };
}

// --- Download float tensor from GPU and create image::Image ---
// Creates a grayscale image::Image from float data
fn image::Image? download_tensor_as_image(DeviceContext* ctx, Tensor* tensor, uint width, uint height, uint channels) {
    usz total = (usz)channels * width * height;
    float[] data = mem::new_array(float, total);

    // Create staging buffer
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: total * 4,
    )!!;

    // Copy from device to staging
    ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, tensor.gpu_buffer.buffer, staging.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = total * 4 }});
    }!!;

    // Read from staging buffer
    float* mapped = (float*)staging.data();
    mem::copy(data.ptr, mapped, total * 4);
    staging.free();

    // Convert float [0,1] to bytes
    char[] pixels = mem::new_array(char, total);
    for (usz i = 0; i < total; i++) {
        float val = data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        pixels[i] = (char)(val * 255.0f + 0.5f);
    }
    mem::free(data);

    PixelFormat format;
    switch (channels) {
        case 1: format = PixelFormat.GRAYSCALE;
        case 3: format = PixelFormat.RGB;
        case 4: format = PixelFormat.RGBA;
        default: format = PixelFormat.RGB;
    }

    return {
        .width = width,
        .height = height,
        .bit_depth = 8,
        .format = format,
        .pixels = pixels,
    };
}
