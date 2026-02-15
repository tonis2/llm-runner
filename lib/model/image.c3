module llm;

import std::io;
import std::io::file;
import std::core::mem;
import image;
import image::png;
import vk;

// Image I/O using image.c3l (PNG format) directly
// No local Image struct - use image::Image from the library

// --- Save float tensor as PNG ---
// Converts NCHW float [0,1] to interleaved RGB and saves as PNG
fn bool save_tensor_as_png(float[] data, uint width, uint height, uint channels, String filename) {
    uint total_pixels = width * height;
    
    // Convert NCHW float [0,1] to interleaved RGB bytes
    char[] pixels = mem::new_array(char, (usz)total_pixels * channels);

    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            uint pixel_idx = y * width + x;
            for (uint c = 0; c < channels; c++) {
                float val = data[c * total_pixels + pixel_idx];
                // Clamp to [0, 1]
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                pixels[pixel_idx * channels + c] = (char)(val * 255.0f + 0.5f);
            }
        }
    }

    // Determine pixel format
    PixelFormat format;
    switch (channels) {
        case 1: format = PixelFormat.GRAYSCALE;
        case 3: format = PixelFormat.RGB;
        case 4: format = PixelFormat.RGBA;
        default: format = PixelFormat.RGB;
    }

    // Create image struct and save
    image::Image img = {
        .width = width,
        .height = height,
        .bit_depth = 8,
        .format = format,
        .pixels = pixels,
    };

    bool ok = false;
    if (catch err = png::save_file(&img, filename)) {
        io::printfn("  ERROR: Failed to save PNG: %s", filename);
    } else {
        ok = true;
        io::printfn("Image saved: %s (%dx%d)", filename, width, height);
    }
    
    mem::free(pixels);
    return ok;
}

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
