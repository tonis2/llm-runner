module llm;

import std::io;
import std::io::file;
import std::core::mem;
import vk;

// PPM Image I/O (P6 binary format - no external dependencies)

struct Image {
    uint width;
    uint height;
    uint channels;  // 3 for RGB
    float[] data;   // [channels, height, width] NCHW layout
}

// --- Save image to PPM ---

fn bool save_ppm(Image* img, String filename) {
    // Write P6 binary PPM
    File f = file::open(filename, "wb")!!;
    defer (void)f.close();

    // PPM header
    char[64] header_buf;
    usz hlen = 0;
    // "P6\n"
    header_buf[hlen] = 'P'; hlen++;
    header_buf[hlen] = '6'; hlen++;
    header_buf[hlen] = '\n'; hlen++;

    // Width
    uint w = img.width;
    char[16] num_buf;
    usz nlen = uint_to_str(w, &num_buf);
    for (usz i = 0; i < nlen; i++) { header_buf[hlen] = num_buf[i]; hlen++; }
    header_buf[hlen] = ' '; hlen++;

    // Height
    nlen = uint_to_str(img.height, &num_buf);
    for (usz i = 0; i < nlen; i++) { header_buf[hlen] = num_buf[i]; hlen++; }
    header_buf[hlen] = '\n'; hlen++;

    // Max val
    header_buf[hlen] = '2'; hlen++;
    header_buf[hlen] = '5'; hlen++;
    header_buf[hlen] = '5'; hlen++;
    header_buf[hlen] = '\n'; hlen++;

    f.write(header_buf[0..hlen - 1])!!;

    // Pixel data: convert NCHW float [0,1] to interleaved RGB bytes
    uint total_pixels = img.width * img.height;
    char[] rgb_data = mem::new_array(char, (usz)total_pixels * 3);

    for (uint y = 0; y < img.height; y++) {
        for (uint x = 0; x < img.width; x++) {
            uint pixel_idx = y * img.width + x;
            for (uint c = 0; c < 3; c++) {
                float val = img.data[c * total_pixels + pixel_idx];
                // Clamp to [0, 1]
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                rgb_data[pixel_idx * 3 + c] = (char)(val * 255.0f + 0.5f);
            }
        }
    }

    f.write(rgb_data)!!;
    mem::free(rgb_data);

    io::printfn("Image saved: %s (%dx%d)", filename, img.width, img.height);
    return true;
}

// --- Load image from PPM ---

fn Image? load_ppm(String filename) {
    char[] data = file::load(mem, filename)!!;

    // Parse P6 header
    usz pos = 0;

    // Skip "P6\n" or "P6 "
    if (data.len < 3 || data[0] != 'P' || data[1] != '6') {
        io::printfn("Error: Not a P6 PPM file");
        return COMPUTE_ERROR~;
    }
    pos = 2;
    while (pos < data.len && (data[pos] == '\n' || data[pos] == '\r' || data[pos] == ' ')) pos++;

    // Skip comments
    while (pos < data.len && data[pos] == '#') {
        while (pos < data.len && data[pos] != '\n') pos++;
        pos++;
    }

    // Read width
    uint width = 0;
    while (pos < data.len && data[pos] >= '0' && data[pos] <= '9') {
        width = width * 10 + (uint)(data[pos] - '0');
        pos++;
    }
    while (pos < data.len && (data[pos] == ' ' || data[pos] == '\n' || data[pos] == '\r')) pos++;

    // Read height
    uint height = 0;
    while (pos < data.len && data[pos] >= '0' && data[pos] <= '9') {
        height = height * 10 + (uint)(data[pos] - '0');
        pos++;
    }
    while (pos < data.len && (data[pos] == ' ' || data[pos] == '\n' || data[pos] == '\r')) pos++;

    // Read maxval
    uint maxval = 0;
    while (pos < data.len && data[pos] >= '0' && data[pos] <= '9') {
        maxval = maxval * 10 + (uint)(data[pos] - '0');
        pos++;
    }
    pos++;  // skip single whitespace after maxval

    if (width == 0 || height == 0 || maxval == 0) {
        io::printfn("Error: Invalid PPM header");
        return COMPUTE_ERROR~;
    }

    // Read pixel data
    uint total_pixels = width * height;
    float[] img_data = mem::new_array(float, (usz)total_pixels * 3);

    float scale = 1.0f / (float)maxval;
    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            uint pixel_idx = y * width + x;
            for (uint c = 0; c < 3; c++) {
                if (pos < data.len) {
                    img_data[c * total_pixels + pixel_idx] = (float)(uint)data[pos] * scale;
                    pos++;
                }
            }
        }
    }

    mem::free(data);
    io::printfn("Image loaded: %s (%dx%d)", filename, width, height);

    return {
        .width = width,
        .height = height,
        .channels = 3,
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

// --- Helper: uint to string ---

fn usz uint_to_str(uint val, char[16]* buf) {
    if (val == 0) {
        (*buf)[0] = '0';
        return 1;
    }
    char[16] tmp;
    usz len = 0;
    uint v = val;
    while (v > 0) {
        tmp[len] = (char)('0' + v % 10);
        v /= 10;
        len++;
    }
    // Reverse
    for (usz i = 0; i < len; i++) {
        (*buf)[i] = tmp[len - 1 - i];
    }
    return len;
}

// --- Free ---

fn void Image.free(&self) {
    mem::free(self.data);
}
