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

// --- Save image to PPM (fallback format, no compression) ---

// Helper: convert uint to string, returns number of chars written
fn uint uint_to_str(uint n, char* buf) {
    if (n == 0) { buf[0] = '0'; return 1; }
    char[16] temp;
    uint i = 0;
    while (n > 0) { temp[i++] = (char)('0' + (n % 10)); n /= 10; }
    uint j = 0;
    while (i > 0) { buf[j++] = temp[--i]; }
    return j;
}

fn bool save_ppm(Image* img, String filename) {
    // Convert NCHW float [0,1] to interleaved RGB bytes
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

    // Build PPM header: "P6\n[width] [height]\n255\n"
    char[] header = mem::new_array(char, 256);
    uint header_len = 0;
    char* h = header.ptr;
    
    // Build header
    h[header_len++] = 'P';
    h[header_len++] = '6';
    h[header_len++] = '\n';
    header_len += uint_to_str(img.width, &h[header_len]);
    h[header_len++] = ' ';
    header_len += uint_to_str(img.height, &h[header_len]);
    h[header_len++] = '\n';
    h[header_len++] = '2';
    h[header_len++] = '5';
    h[header_len++] = '5';
    h[header_len++] = '\n';

    // Write file
    if (try f = io::file::open(filename, "wb")) {
        f.write(header[:header_len])!!;
        f.write(pixels)!!;
        f.close()!!;
        io::printfn("Image saved: %s (%dx%d) [PPM format]", filename, img.width, img.height);
        mem::free(pixels);
        mem::free(header);
        return true;
    }
    
    io::printfn("Error: failed to open %s for writing", filename);
    mem::free(pixels);
    mem::free(header);
    return false;
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

    // Save as PNG (with fallback to PPM on failure)
    bool ok = false;
    if (catch err = png::save_file(&png_img, filename)) {
        io::printfn("  Warning: PNG save failed, trying PPM format...");
        // Create PPM filename by appending .ppm extension
        char[] ppm_name = mem::new_array(char, filename.len + 5);
        mem::copy(ppm_name.ptr, filename.ptr, filename.len);
        ppm_name[filename.len] = '.';
        ppm_name[filename.len + 1] = 'p';
        ppm_name[filename.len + 2] = 'p';
        ppm_name[filename.len + 3] = 'm';
        ppm_name[filename.len + 4] = '\0';
        ok = save_ppm(img, (String)ppm_name);
        mem::free(ppm_name);
    } else {
        ok = true;
        io::printfn("Image saved: %s (%dx%d)", filename, img.width, img.height);
    }
    
    mem::free(pixels);
    return ok;
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
