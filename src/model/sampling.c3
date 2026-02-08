module llm;

import vk;
import std::io;
import std::core::mem;

fn void? debug_readback(DeviceContext* ctx, Tensor* tensor, String label, uint count) {
    usz size = (usz)count * 4;
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: size,
    )!!;

    ctx.device.@single_time_command(queue: ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, tensor.gpu_buffer.buffer, staging.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = size }});
    }!!;

    float* data = (float*)staging.data();
    io::printf("  %s:", label);
    for (uint i = 0; i < count; i++) {
        io::printf(" %.6f", data[i]);
    }
    io::printfn("");
    staging.free();
}

fn uint? argmax_sample(DeviceContext* ctx, Tensor* logits, uint vocab_size) {
    usz size = (usz)vocab_size * 4;

    // Create host-visible staging buffer
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: size,
    )!!;

    // Copy logits from device to staging
    ctx.device.@single_time_command(queue: ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, logits.gpu_buffer.buffer, staging.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = size }});
    }!!;

    // Map and find argmax
    float* data = (float*)staging.data();
    float max_val = data[0];
    uint max_idx = 0;

    for (uint i = 1; i < vocab_size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }

    staging.free();
    return max_idx;
}
