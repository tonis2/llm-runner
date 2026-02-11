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

// CPU-side bf16 to f32 conversion
fn float cpu_bf16_to_f32(char* ptr) {
    ushort bits = bitcast(*(char[2]*)ptr, ushort);
    uint f32_bits = (uint)bits << 16;
    return bitcast(f32_bits, float);
}

// CPU-side f16 to f32 conversion
fn float cpu_f16_to_f32(char* ptr) {
    ushort bits = bitcast(*(char[2]*)ptr, ushort);
    uint sign = ((uint)bits >> 15) & 1;
    uint exp_bits = ((uint)bits >> 10) & 0x1F;
    uint mant = (uint)bits & 0x3FF;

    if (exp_bits == 0) {
        if (mant == 0) return sign != 0 ? -0.0f : 0.0f;
        // Subnormal
        float val = (float)mant / (1 << 24);
        return sign != 0 ? -val : val;
    }
    if (exp_bits == 31) {
        return sign != 0 ? -1.0f/0.0f : 1.0f/0.0f;
    }
    float val = (float)(mant | 0x400) / (1 << 25) * (float)(1 << exp_bits);
    return sign != 0 ? -val : val;
}

// Dequantize Q5_K block (256 elements from 176 bytes) to F32
fn void dequant_q5k_block(char* block, float* output) {
    float d = cpu_f16_to_f32(block);
    float dmin = cpu_f16_to_f32(block + 2);
    char* scales = block + 4;
    char* qh = block + 16;
    char* qs = block + 48;

    for (uint j = 0; j < 4; j++) {
        // Get scale and min for sub-blocks 2*j and 2*j+1
        uint sc1, m1, sc2, m2;
        uint is0 = 2 * j;
        uint is1 = 2 * j + 1;

        if (is0 < 4) {
            sc1 = (uint)scales[is0] & 63;
            m1 = (uint)scales[is0 + 4] & 63;
        } else {
            sc1 = ((uint)scales[is0 + 4] & 0xF) | (((uint)scales[is0 - 4] >> 6) << 4);
            m1 = ((uint)scales[is0 + 4] >> 4) | (((uint)scales[is0] >> 6) << 4);
        }
        if (is1 < 4) {
            sc2 = (uint)scales[is1] & 63;
            m2 = (uint)scales[is1 + 4] & 63;
        } else {
            sc2 = ((uint)scales[is1 + 4] & 0xF) | (((uint)scales[is1 - 4] >> 6) << 4);
            m2 = ((uint)scales[is1 + 4] >> 4) | (((uint)scales[is1] >> 6) << 4);
        }

        float d1 = d * (float)sc1;
        float dm1 = dmin * (float)m1;
        float d2 = d * (float)sc2;
        float dm2 = dmin * (float)m2;

        for (uint l = 0; l < 32; l++) {
            uint ql_byte = (uint)qs[j * 32 + l];
            uint qh_byte = (uint)qh[l];
            uint q_low = ql_byte & 0xF;
            uint q_high = (qh_byte >> (2 * j)) & 1;
            output[j * 64 + l] = d1 * (float)(q_low + q_high * 16) - dm1;
        }
        for (uint l = 0; l < 32; l++) {
            uint ql_byte = (uint)qs[j * 32 + l];
            uint qh_byte = (uint)qh[l];
            uint q_low = (ql_byte >> 4) & 0xF;
            uint q_high = (qh_byte >> (2 * j + 1)) & 1;
            output[j * 64 + 32 + l] = d2 * (float)(q_low + q_high * 16) - dm2;
        }
    }
}

// Dequantize Q6_K block (256 elements from 210 bytes) to F32
// Layout: ql[128] + qh[64] + scales[16] + f16 d(2)
fn void dequant_q6k_block(char* block, float* output) {
    char* ql = block;
    char* qh = block + 128;
    ichar* sc = (ichar*)(block + 192);
    float d = cpu_f16_to_f32(block + 208);

    for (uint half = 0; half < 2; half++) {
        uint ql_off = half * 64;
        uint qh_off = half * 32;
        uint sc_off = half * 8;
        uint elem_off = half * 128;

        for (uint l = 0; l < 32; l++) {
            uint is_idx = l / 16;

            uint ql0 = (uint)ql[ql_off + l];
            uint ql32 = (uint)ql[ql_off + 32 + l];
            uint qh_val = (uint)qh[qh_off + l];

            int q1 = (int)((ql0 & 0xF) | (((qh_val >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql32 & 0xF) | (((qh_val >> 2) & 3) << 4)) - 32;
            int q3 = (int)(((ql0 >> 4) & 0xF) | (((qh_val >> 4) & 3) << 4)) - 32;
            int q4 = (int)(((ql32 >> 4) & 0xF) | (((qh_val >> 6) & 3) << 4)) - 32;

            output[elem_off + l + 0]  = d * (float)sc[sc_off + is_idx + 0] * (float)q1;
            output[elem_off + l + 32] = d * (float)sc[sc_off + is_idx + 2] * (float)q2;
            output[elem_off + l + 64] = d * (float)sc[sc_off + is_idx + 4] * (float)q3;
            output[elem_off + l + 96] = d * (float)sc[sc_off + is_idx + 6] * (float)q4;
        }
    }
}

// Dequantize Q4_K block (256 elements from 144 bytes) to F32
fn void dequant_q4k_block(char* block, float* output) {
    float d = cpu_f16_to_f32(block);
    float dmin = cpu_f16_to_f32(block + 2);
    char* scales = block + 4;
    char* qs = block + 16;

    uint is = 0;
    for (uint j = 0; j < 4; j++) {
        uint sc1, m1, sc2, m2;
        uint is0 = is;
        uint is1 = is + 1;

        if (is0 < 4) {
            sc1 = (uint)scales[is0] & 63;
            m1 = (uint)scales[is0 + 4] & 63;
        } else {
            sc1 = ((uint)scales[is0 + 4] & 0xF) | (((uint)scales[is0 - 4] >> 6) << 4);
            m1 = ((uint)scales[is0 + 4] >> 4) | (((uint)scales[is0] >> 6) << 4);
        }
        if (is1 < 4) {
            sc2 = (uint)scales[is1] & 63;
            m2 = (uint)scales[is1 + 4] & 63;
        } else {
            sc2 = ((uint)scales[is1 + 4] & 0xF) | (((uint)scales[is1 - 4] >> 6) << 4);
            m2 = ((uint)scales[is1 + 4] >> 4) | (((uint)scales[is1] >> 6) << 4);
        }

        float d1 = d * (float)sc1;
        float dm1 = dmin * (float)m1;
        float d2 = d * (float)sc2;
        float dm2 = dmin * (float)m2;

        for (uint l = 0; l < 32; l++) {
            uint ql_byte = (uint)qs[j * 32 + l];
            output[j * 64 + l] = d1 * (float)(ql_byte & 0xF) - dm1;
        }
        for (uint l = 0; l < 32; l++) {
            uint ql_byte = (uint)qs[j * 32 + l];
            output[j * 64 + 32 + l] = d2 * (float)((ql_byte >> 4) & 0xF) - dm2;
        }

        is += 2;
    }
}

// Dequantize Q4_0 block (32 elements from 18 bytes) to F32
// Layout: f16 scale(2) + qs[16] (packed 4-bit nibbles)
// Low nibble → elements 0-15, high nibble → elements 16-31
fn void dequant_q4_0_block(char* block, float* output) {
    float d = cpu_f16_to_f32(block);
    char* qs = block + 2;

    for (uint i = 0; i < 16; i++) {
        uint byte_val = (uint)qs[i];
        int lo = (int)(byte_val & 0xF) - 8;
        int hi = (int)((byte_val >> 4) & 0xF) - 8;
        output[i] = d * (float)lo;
        output[i + 16] = d * (float)hi;
    }
}

// Dequantize Q8_0 block (32 elements from 34 bytes) to F32
fn void dequant_q8_0_block(char* block, float* output) {
    float d = cpu_f16_to_f32(block);
    ichar* qs = (ichar*)(block + 2);

    for (uint i = 0; i < 32; i++) {
        output[i] = d * (float)qs[i];
    }
}

// Upload a quantized tensor as F32 (dequantizes on CPU, uploads F32)
fn Tensor? upload_weight_as_f32(DeviceContext* ctx, GGUFTensorInfo* info, char* tensor_data_base) {
    char* data_ptr = tensor_data_base + (usz)info.offset;

    usz n_elements = 1;
    for (uint i = 0; i < info.n_dims; i++) {
        n_elements *= (usz)info.shape[i];
    }
    usz f32_size = n_elements * 4;
    float* f32_data = (float*)mem::calloc(f32_size);

    if (info.type == GGML_Q5_K) {
        usz n_blocks = n_elements / 256;
        for (usz b = 0; b < n_blocks; b++) {
            dequant_q5k_block(data_ptr + b * 176, f32_data + b * 256);
        }
    } else if (info.type == GGML_Q4_K) {
        usz n_blocks = n_elements / 256;
        for (usz b = 0; b < n_blocks; b++) {
            dequant_q4k_block(data_ptr + b * 144, f32_data + b * 256);
        }
    } else if (info.type == GGML_Q8_0) {
        usz n_blocks = n_elements / 32;
        for (usz b = 0; b < n_blocks; b++) {
            dequant_q8_0_block(data_ptr + b * 34, f32_data + b * 32);
        }
    } else if (info.type == GGML_Q4_0) {
        usz n_blocks = n_elements / 32;
        for (usz b = 0; b < n_blocks; b++) {
            dequant_q4_0_block(data_ptr + b * 18, f32_data + b * 32);
        }
    } else if (info.type == GGML_Q6_K) {
        usz n_blocks = n_elements / 256;
        for (usz b = 0; b < n_blocks; b++) {
            dequant_q6k_block(data_ptr + b * 210, f32_data + b * 256);
        }
    } else if (info.type == GGML_F16) {
        for (usz j = 0; j < n_elements; j++) {
            f32_data[j] = cpu_f16_to_f32(data_ptr + j * 2);
        }
    } else if (info.type == GGML_BF16) {
        for (usz j = 0; j < n_elements; j++) {
            f32_data[j] = cpu_bf16_to_f32(data_ptr + j * 2);
        }
    } else {
        // Copy raw for F32
        mem::copy(f32_data, data_ptr, f32_size);
    }

    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: f32_data,
        data_size: f32_size,
    )!!;

    mem::free(f32_data);

    return {
        .gpu_buffer = gpu_buffer,
        .dtype = GGML_F32,
        .n_dims = info.n_dims,
        .shape = info.shape,
        .size_bytes = f32_size,
    };
}
