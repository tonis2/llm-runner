module llm;

import std::io;
import std::io::file;
import std::core::mem;
import image;
import image::png;
import vk;

// Image I/O using image.c3l (PNG format)

struct Image {
    uint width;
    uint height;
    uint channels;  // 3 for RGB
    float[] data;   // [channels, height, width] NCHW layout
}

// --- Save image to PNG ---

fn bool save_png(Image* img, String filename) {
    // Convert NCHW float [0,1] to interleaved RGB bytes for PNG
    uint total_pixels = img.width * img.height;
    char[] pixels = mem::new_array(char, (usz)total_pixels * 3);

    for (uint y = 0; y < img.height; y++) {
        for (uint x = 0; x < img.width; x++) {
            uint pixel_idx = y * img.width + x;
            for (uint c = 0; c < 3; c++) {
                float val = img.data[c * total_pixels + pixel_idx];
                // Clamp to [0, 1]
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                pixels[pixel_idx * 3 + c] = (char)(val * 255.0f + 0.5f);
            }
        }
    }

    // Create image struct for image.c3l
    image::Image png_img = {
        .width = img.width,
        .height = img.height,
        .bit_depth = 8,
        .format = PixelFormat.RGB,
        .pixels = pixels,
    };

    // Save as PNG
    png::save_file(&png_img, filename)!!;
    mem::free(pixels);

    io::printfn("Image saved: %s (%dx%d)", filename, img.width, img.height);
    return true;
}

// --- Load image from PNG ---

fn Image? load_png(String filename) {
    image::Image png_img = png::load_file(filename)!!;
    defer png_img.free();

    if (png_img.format != PixelFormat.RGB && png_img.format != PixelFormat.RGBA) {
        io::printfn("Error: Only RGB and RGBA PNG images supported");
        return COMPUTE_ERROR~;
    }

    uint channels = (uint)(png_img.format == PixelFormat.RGBA ? 4 : 3);
    uint total_pixels = png_img.width * png_img.height;
    float[] img_data = mem::new_array(float, (usz)total_pixels * channels);

    usz bpp = (usz)png_img.bytes_per_pixel();
    
    for (uint y = 0; y < png_img.height; y++) {
        for (uint x = 0; x < png_img.width; x++) {
            uint pixel_idx = y * png_img.width + x;
            for (uint c = 0; c < channels; c++) {
                char val = png_img.pixels[(usz)pixel_idx * bpp + c];
                img_data[c * total_pixels + pixel_idx] = (float)(uint)val / 255.0f;
            }
        }
    }

    io::printfn("Image loaded: %s (%dx%d)", filename, png_img.width, png_img.height);

    return {
        .width = png_img.width,
        .height = png_img.height,
        .channels = channels,
        .data = img_data,
    };
}

// --- Upload image to GPU ---

fn Tensor? upload_image_to_gpu(DeviceContext* ctx, Image* img) {
    usz total = (usz)img.channels * img.width * img.height;
    ulong[4] shape = { (ulong)total, 0, 0, 0 };

    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: img.data.ptr,
        data_size: total * 4,
    )!!;

    return {
        .gpu_buffer = gpu_buffer,
        .dtype = GGML_F32,
        .n_dims = 1,
        .shape = shape,
        .size_bytes = total * 4,
    };
}

// --- Download image from GPU ---

fn Image? download_image_from_gpu(DeviceContext* ctx, Tensor* tensor, uint width, uint height, uint channels) {
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

    return {
        .width = width,
        .height = height,
        .channels = channels,
        .data = data,
    };
}

// --- Free ---

fn void Image.free(&self) {
    mem::free(self.data);
}
