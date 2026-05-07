module qwen_edit;

import vk;
import llm;
import std::io;
import std::io::file;
import std::math;
import std::core::mem;

// --- Tensor Debug Helpers ---
// These help diagnose issues by printing statistics and dumping tensors to files

fn void? print_tensor_stats(llm::DeviceContext* ctx, llm::Tensor* tensor, String label) {
    if (tensor.n_dims == 0) {
        io::printfn("  [%s] <empty tensor>", label);
        return;
    }
    
    // Calculate total elements
    usz total = 1;
    for (uint i = 0; i < tensor.n_dims; i++) {
        total *= (usz)tensor.shape[i];
    }
    if (total == 0) {
        io::printfn("  [%s] <zero-size tensor>", label);
        return;
    }
    
    // Download to CPU staging buffer
    usz bytes = total * 4; // Assume F32 for now
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: bytes,
    )!!;
    
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, tensor.gpu_buffer.buffer, staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = bytes }});
    }!!;
    
    float* data = (float*)staging.data();
    
    // Compute statistics
    float min_val = data[0];
    float max_val = data[0];
    double sum = 0.0;
    double sum_sq = 0.0;
    uint nan_count = 0;
    uint inf_count = 0;
    uint zero_count = 0;
    
    for (usz i = 0; i < total; i++) {
        float v = data[i];
        if (v != v) {
            nan_count++;
            continue;
        }
        if (v == 1.0f/0.0f || v == -1.0f/0.0f) {
            inf_count++;
            continue;
        }
        if (v == 0.0f) zero_count++;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += (double)v;
        sum_sq += (double)v * (double)v;
    }
    
    float mean = (float)(sum / (double)(total - nan_count - inf_count));
    float variance = (float)(sum_sq / (double)(total - nan_count - inf_count)) - mean * mean;
    float std_dev = variance > 0.0f ? std::math::sqrt(variance) : 0.0f;
    
    // Print shape
    io::printf("  [%s] shape=[", label);
    for (uint i = 0; i < tensor.n_dims; i++) {
        if (i > 0) io::printf(", ");
        io::printf("%d", (uint)tensor.shape[i]);
    }
    io::printf("] ");
    
    io::printfn("mean=%.4f std=%.4f min=%.4f max=%.4f zeros=%d/%d nan=%d inf=%d",
        mean, std_dev, min_val, max_val, zero_count, (uint)total, nan_count, inf_count);
    
    staging.free();
}

// Dump first N values of a tensor for detailed inspection
fn void? print_tensor_first_n(llm::DeviceContext* ctx, llm::Tensor* tensor, String label, uint n) {
    usz total = 1;
    for (uint i = 0; i < tensor.n_dims; i++) {
        total *= (usz)tensor.shape[i];
    }
    uint count = n < total ? n : (uint)total;
    if (count == 0) return;
    
    usz bytes = total * 4;
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: bytes,
    )!!;
    
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, tensor.gpu_buffer.buffer, staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = bytes }});
    }!!;
    
    float* data = (float*)staging.data();
    
    io::printf("  [%s] first %d: [", label, count);
    for (uint i = 0; i < count; i++) {
        if (i > 0) io::printf(", ");
        io::printf("%.4f", data[i]);
    }
    io::printfn("]");
    
    staging.free();
}

// Save tensor to a raw binary file for external analysis
fn void? dump_tensor_to_file(llm::DeviceContext* ctx, llm::Tensor* tensor, String filename) {
    usz total = 1;
    for (uint i = 0; i < tensor.n_dims; i++) {
        total *= (usz)tensor.shape[i];
    }
    if (total == 0) return;
    
    usz bytes = total * 4;
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: bytes,
    )!!;
    
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, tensor.gpu_buffer.buffer, staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = bytes }});
    }!!;
    
    float* data = (float*)staging.data();
    
    // Write as simple binary: 4 bytes for ndims, then 4 bytes per shape dim, then f32 data
    File f = io::file::open(filename, "wb")!!;
    uint ndims = tensor.n_dims;
    f.write(((char*)&ndims)[:4])!!;
    for (uint i = 0; i < tensor.n_dims; i++) {
        uint dim = (uint)tensor.shape[i];
        f.write(((char*)&dim)[:4])!!;
    }
    f.write(((char*)data)[:bytes])!!;
    f.close()!!;
    
    staging.free();
    io::printfn("  Dumped tensor to %s", filename);
}

// Validate that a tensor doesn't contain NaN or Inf (common sign of bugs)
fn bool? validate_tensor_finite(llm::DeviceContext* ctx, llm::Tensor* tensor, String label) {
    usz total = 1;
    for (uint i = 0; i < tensor.n_dims; i++) {
        total *= (usz)tensor.shape[i];
    }
    if (total == 0) return true;
    
    usz bytes = total * 4;
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data_size: bytes,
    )!!;
    
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, tensor.gpu_buffer.buffer, staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = bytes }});
    }!!;
    
    float* data = (float*)staging.data();
    uint nan_count = 0;
    uint inf_count = 0;
    
    for (usz i = 0; i < total; i++) {
        float v = data[i];
        if (v != v) {
            nan_count++;
        } else if (v == 1.0f/0.0f || v == -1.0f/0.0f) {
            inf_count++;
        }
    }
    
    staging.free();
    
    if (nan_count > 0 || inf_count > 0) {
        io::printfn("  WARNING [%s] contains %d NaN and %d Inf values!", label, nan_count, inf_count);
        return false;
    }
    return true;
}
