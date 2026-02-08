module llm;

import vk;
import std::io;
import std::core::mem;

struct Tensor {
    vk::Memory gpu_buffer;
    GGMLType dtype;
    uint n_dims;
    ulong[4] shape;
    usz size_bytes;
}

fn usz compute_tensor_bytes(GGMLType dtype, ulong[4] shape, uint n_dims) {
    usz n_elements = 1;
    for (uint i = 0; i < n_dims; i++) {
        n_elements *= (usz)shape[i];
    }
    usz block_size = dtype.block_size();
    usz type_size = dtype.type_size();
    usz n_blocks = (n_elements + block_size - 1) / block_size;
    return n_blocks * type_size;
}

fn Tensor? create_tensor(DeviceContext* ctx, GGMLType dtype, ulong[4] shape, uint n_dims) {
    usz size_bytes = compute_tensor_bytes(dtype, shape, n_dims);
    if (size_bytes == 0) return COMPUTE_ERROR~;

    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: null,
        data_size: size_bytes
    )!!;

    return {
        .gpu_buffer = gpu_buffer,
        .dtype = dtype,
        .n_dims = n_dims,
        .shape = shape,
        .size_bytes = size_bytes,
    };
}

fn Tensor? upload_weight(DeviceContext* ctx, GGUFTensorInfo* info, char* tensor_data_base) {
    char* data_ptr = tensor_data_base + (usz)info.offset;

    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: data_ptr,
        data_size: info.data_size,
    )!!;

    return {
        .gpu_buffer = gpu_buffer,
        .dtype = info.type,
        .n_dims = info.n_dims,
        .shape = info.shape,
        .size_bytes = info.data_size,
    };
}

fn void Tensor.free(&self) {
    self.gpu_buffer.free();
}

fn Tensor? create_f32_tensor(DeviceContext* ctx, ulong[4] shape, uint n_dims) {
    return create_tensor(ctx, GGML_F32, shape, n_dims);
}
