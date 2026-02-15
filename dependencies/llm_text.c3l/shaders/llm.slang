// LLM inference compute shaders
// Single file, multiple entry points compiled to one SPIR-V module

// ============================================================
// Embedding lookup
// ============================================================

struct EmbeddingParams {
    uint token_id;
    uint dim;
};

[vk_push_constant] const EmbeddingParams embedding_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> embedding_table;
[vk_binding(1, 0)] RWStructuredBuffer<float> embedding_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void embedding(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= embedding_pc.dim) return;
    embedding_output[tid.x] = embedding_table[embedding_pc.token_id * embedding_pc.dim + tid.x];
}

// ============================================================
// RMS Normalization
// ============================================================

struct RMSNormParams {
    uint dim;
    float eps;
};

[vk_push_constant] const RMSNormParams rmsnorm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> rmsnorm_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> rmsnorm_weight;
[vk_binding(2, 0)] RWStructuredBuffer<float> rmsnorm_output;

groupshared float rmsnorm_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void rmsnorm(uint3 tid: SV_DispatchThreadID, uint gi: SV_GroupIndex) {
    uint num_threads = 256;

    float sum_sq = 0.0;
    for (uint i = gi; i < rmsnorm_pc.dim; i += num_threads) {
        float val = rmsnorm_input[i];
        sum_sq += val * val;
    }

    rmsnorm_shared[gi] = sum_sq;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            rmsnorm_shared[gi] += rmsnorm_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float rms = 1.0 / sqrt(rmsnorm_shared[0] / float(rmsnorm_pc.dim) + rmsnorm_pc.eps);

    for (uint i = gi; i < rmsnorm_pc.dim; i += num_threads) {
        rmsnorm_output[i] = rmsnorm_input[i] * rms * rmsnorm_weight[i];
    }
}

// ============================================================
// Layer Normalization
// ============================================================

struct LayerNormParams {
    uint dim;
    float eps;
};

[vk_push_constant] const LayerNormParams layernorm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> layernorm_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> layernorm_weight;
[vk_binding(2, 0)] RWStructuredBuffer<float> layernorm_bias;
[vk_binding(3, 0)] RWStructuredBuffer<float> layernorm_output;

groupshared float layernorm_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void layernorm(uint3 tid: SV_DispatchThreadID, uint gi: SV_GroupIndex) {
    uint num_threads = 256;

    // Compute mean
    float sum = 0.0;
    for (uint i = gi; i < layernorm_pc.dim; i += num_threads) {
        sum += layernorm_input[i];
    }
    layernorm_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            layernorm_shared[gi] += layernorm_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = layernorm_shared[0] / float(layernorm_pc.dim);
    GroupMemoryBarrierWithGroupSync();

    // Compute variance
    float var_sum = 0.0;
    for (uint i = gi; i < layernorm_pc.dim; i += num_threads) {
        float diff = layernorm_input[i] - mean;
        var_sum += diff * diff;
    }
    layernorm_shared[gi] = var_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            layernorm_shared[gi] += layernorm_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_std = 1.0 / sqrt(layernorm_shared[0] / float(layernorm_pc.dim) + layernorm_pc.eps);

    // Normalize: (x - mean) * inv_std * weight + bias
    for (uint i = gi; i < layernorm_pc.dim; i += num_threads) {
        layernorm_output[i] = (layernorm_input[i] - mean) * inv_std * layernorm_weight[i] + layernorm_bias[i];
    }
}

// ============================================================
// SiLU activation (in-place)
// ============================================================

struct SiluParams {
    uint n;
};

[vk_push_constant] const SiluParams silu_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> silu_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void silu(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= silu_pc.n) return;
    float x = silu_data[tid.x];
    silu_data[tid.x] = x / (1.0 + exp(-x));
}

// ============================================================
// GELU activation (in-place)
// ============================================================

struct GeluParams {
    uint n;
};

[vk_push_constant] const GeluParams gelu_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> gelu_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void gelu(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= gelu_pc.n) return;
    float x = gelu_data[tid.x];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c = 0.7978845608; // sqrt(2/pi)
    gelu_data[tid.x] = 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

// ============================================================
// Matrix-vector multiply (F32 weights)
// ============================================================

struct MatMulParams {
    uint out_dim;
    uint in_dim;
};

[vk_push_constant] const MatMulParams matmul_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> matmul_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> matmul_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> matmul_output;

groupshared float matmul_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    uint num_threads = 256;

    if (row >= matmul_pc.out_dim) return;

    float sum = 0.0;
    for (uint i = gi; i < matmul_pc.in_dim; i += num_threads) {
        sum += matmul_weight[row * matmul_pc.in_dim + i] * matmul_input[i];
    }

    matmul_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            matmul_shared[gi] += matmul_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0) {
        matmul_output[row] = matmul_shared[0];
    }
}

// ============================================================
// Matrix-vector multiply (Q8_0 quantized weights)
// ============================================================

// Q8_0 block: 2 bytes f16 scale + 32 bytes int8 quantized = 34 bytes
// Buffer is uint[] for 4-byte aligned access

[vk_push_constant] const MatMulParams matmul_q8_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> q8_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> q8_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> q8_output;

groupshared float q8_shared[256];

float unpack_f16(uint bits) {
    uint sign = (bits >> 15) & 0x1u;
    uint exp_bits = (bits >> 10) & 0x1Fu;
    uint mant = bits & 0x3FFu;

    if (exp_bits == 0) {
        if (mant == 0) return sign != 0 ? -0.0 : 0.0;
        float val = ldexp(float(mant), -24);
        return sign != 0 ? -val : val;
    }
    if (exp_bits == 31) {
        return sign != 0 ? asfloat(0xFF800000u) : asfloat(0x7F800000u);
    }

    float val = ldexp(float(mant | 0x400u), int(exp_bits) - 25);
    return sign != 0 ? -val : val;
}

int sign_extend_i8(uint v) {
    return (v & 0x80u) != 0 ? int(v) - 256 : int(v);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_q8(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    uint num_threads = 256;

    if (row >= matmul_q8_pc.out_dim) return;

    uint blocks_per_row = matmul_q8_pc.in_dim / 32;
    uint bytes_per_row = blocks_per_row * 34;
    uint row_byte_offset = row * bytes_per_row;

    float sum = 0.0;

    for (uint block_idx = gi; block_idx < blocks_per_row; block_idx += num_threads) {
        uint block_byte = row_byte_offset + block_idx * 34;

        // Read f16 scale
        uint scale_uint_idx = block_byte / 4;
        uint scale_byte_offset = block_byte % 4;
        uint scale_word = q8_weight[scale_uint_idx];
        uint scale_bits;
        if (scale_byte_offset == 0) {
            scale_bits = scale_word & 0xFFFFu;
        } else if (scale_byte_offset == 2) {
            scale_bits = (scale_word >> 16) & 0xFFFFu;
        } else {
            uint next_word = q8_weight[scale_uint_idx + 1];
            scale_bits = ((scale_word >> (scale_byte_offset * 8)) | (next_word << (32 - scale_byte_offset * 8))) & 0xFFFFu;
        }
        float scale = unpack_f16(scale_bits);

        // Read 32 int8 quantized values
        uint qs_byte = block_byte + 2;
        uint input_offset = block_idx * 32;

        for (uint j = 0; j < 32; j += 4) {
            uint byte_pos = qs_byte + j;
            uint uint_idx = byte_pos / 4;
            uint byte_off = byte_pos % 4;

            uint word = q8_weight[uint_idx];
            if (byte_off != 0) {
                uint next_word = q8_weight[uint_idx + 1];
                word = (word >> (byte_off * 8)) | (next_word << (32 - byte_off * 8));
            }

            int q0 = sign_extend_i8(word & 0xFFu);
            int q1 = sign_extend_i8((word >> 8) & 0xFFu);
            int q2 = sign_extend_i8((word >> 16) & 0xFFu);
            int q3 = sign_extend_i8((word >> 24) & 0xFFu);

            sum += scale * float(q0) * q8_input[input_offset + j];
            sum += scale * float(q1) * q8_input[input_offset + j + 1];
            sum += scale * float(q2) * q8_input[input_offset + j + 2];
            sum += scale * float(q3) * q8_input[input_offset + j + 3];
        }
    }

    q8_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            q8_shared[gi] += q8_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0) {
        q8_output[row] = q8_shared[0];
    }
}

// ============================================================
// Matrix-vector multiply (Q4_0 quantized weights)
// ============================================================
// Q4_0 block: 18 bytes per 32 elements
// Layout: f16 scale(2) + qs[16] (packed 4-bit nibbles)
// Low nibble → elements 0-15, high nibble → elements 16-31

[vk_push_constant] const MatMulParams matmul_q4_0_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> q4_0_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> q4_0_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> q4_0_output;

groupshared float q4_0_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_q4_0(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    uint num_threads = 256;

    if (row >= matmul_q4_0_pc.out_dim) return;

    uint blocks_per_row = matmul_q4_0_pc.in_dim / 32;
    uint bytes_per_row = blocks_per_row * 18;
    uint row_byte_offset = row * bytes_per_row;

    float sum = 0.0;

    for (uint block_idx = gi; block_idx < blocks_per_row; block_idx += num_threads) {
        uint block_byte = row_byte_offset + block_idx * 18;

        // Read f16 scale
        uint scale_uint_idx = block_byte / 4;
        uint scale_byte_offset = block_byte % 4;
        uint scale_word = q4_0_weight[scale_uint_idx];
        uint scale_bits;
        if (scale_byte_offset == 0) {
            scale_bits = scale_word & 0xFFFFu;
        } else if (scale_byte_offset == 2) {
            scale_bits = (scale_word >> 16) & 0xFFFFu;
        } else {
            uint next_word = q4_0_weight[scale_uint_idx + 1];
            scale_bits = ((scale_word >> (scale_byte_offset * 8)) | (next_word << (32 - scale_byte_offset * 8))) & 0xFFFFu;
        }
        float scale = unpack_f16(scale_bits);

        // Read 16 bytes of packed nibbles (32 elements)
        uint qs_byte = block_byte + 2;
        uint input_offset = block_idx * 32;

        for (uint j = 0; j < 16; j += 4) {
            uint byte_pos = qs_byte + j;
            uint uint_idx = byte_pos / 4;
            uint byte_off = byte_pos % 4;

            uint word = q4_0_weight[uint_idx];
            if (byte_off != 0) {
                uint next_word = q4_0_weight[uint_idx + 1];
                word = (word >> (byte_off * 8)) | (next_word << (32 - byte_off * 8));
            }

            // Low nibbles → elements 0-15
            int q0_lo = int(word & 0xFu) - 8;
            int q1_lo = int((word >> 8) & 0xFu) - 8;
            int q2_lo = int((word >> 16) & 0xFu) - 8;
            int q3_lo = int((word >> 24) & 0xFu) - 8;

            sum += scale * float(q0_lo) * q4_0_input[input_offset + j];
            sum += scale * float(q1_lo) * q4_0_input[input_offset + j + 1];
            sum += scale * float(q2_lo) * q4_0_input[input_offset + j + 2];
            sum += scale * float(q3_lo) * q4_0_input[input_offset + j + 3];

            // High nibbles → elements 16-31
            int q0_hi = int((word >> 4) & 0xFu) - 8;
            int q1_hi = int((word >> 12) & 0xFu) - 8;
            int q2_hi = int((word >> 20) & 0xFu) - 8;
            int q3_hi = int((word >> 28) & 0xFu) - 8;

            sum += scale * float(q0_hi) * q4_0_input[input_offset + 16 + j];
            sum += scale * float(q1_hi) * q4_0_input[input_offset + 16 + j + 1];
            sum += scale * float(q2_hi) * q4_0_input[input_offset + 16 + j + 2];
            sum += scale * float(q3_hi) * q4_0_input[input_offset + 16 + j + 3];
        }
    }

    q4_0_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            q4_0_shared[gi] += q4_0_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0) {
        q4_0_output[row] = q4_0_shared[0];
    }
}

// ============================================================
// Matrix-vector multiply (Q5_K quantized weights)
// ============================================================
// Q5_K block: 176 bytes per 256 elements
// Layout: f16 d(2) + f16 dmin(2) + scales[12] + qh[32] + qs[128]

[vk_push_constant] const MatMulParams matmul_q5k_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> q5k_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> q5k_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> q5k_output;

groupshared float q5k_shared[256];

// Extract 6-bit scale and min for Q5_K sub-block
void get_scale_min_k4(uint j, uint block_byte, RWStructuredBuffer<uint> w, out uint sc_val, out uint m_val) {
    // scales are at block_byte + 4, 12 bytes
    uint sc_base = block_byte + 4;
    if (j < 4) {
        uint byte_j = read_byte(w, sc_base + j);
        uint byte_j4 = read_byte(w, sc_base + j + 4);
        sc_val = byte_j & 63u;
        m_val = byte_j4 & 63u;
    } else {
        uint byte_j4 = read_byte(w, sc_base + j + 4);
        uint byte_jm4 = read_byte(w, sc_base + j - 4);
        uint byte_j0 = read_byte(w, sc_base + j);
        sc_val = (byte_j4 & 0xFu) | ((byte_jm4 >> 6) << 4);
        m_val = (byte_j4 >> 4) | ((byte_j0 >> 6) << 4);
    }
}

uint read_byte(RWStructuredBuffer<uint> buf, uint byte_idx) {
    return (buf[byte_idx / 4] >> ((byte_idx % 4) * 8)) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_q5k(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    uint num_threads = 256;

    if (row >= matmul_q5k_pc.out_dim) return;

    uint blocks_per_row = matmul_q5k_pc.in_dim / 256;
    uint row_byte_offset = row * blocks_per_row * 176;

    float sum = 0.0;

    for (uint block_idx = gi; block_idx < blocks_per_row; block_idx += num_threads) {
        uint bb = row_byte_offset + block_idx * 176;
        uint input_base = block_idx * 256;

        // Read d and dmin (f16 at bytes 0 and 2)
        // bb is always 4-aligned (176 is multiple of 4)
        uint dm_word = q5k_weight[bb / 4];
        float d = unpack_f16(dm_word & 0xFFFFu);
        float dmin = unpack_f16((dm_word >> 16) & 0xFFFFu);

        // Process 4 chunks of 64 elements each (8 sub-blocks of 32)
        // bb is always 4-aligned (176 is multiple of 4), so qh (offset 16)
        // and qs (offset 48) are word-aligned — use direct uint reads
        for (uint j = 0; j < 4; j++) {
            uint sc1, m1, sc2, m2;
            get_scale_min_k4(2 * j, bb, q5k_weight, sc1, m1);
            get_scale_min_k4(2 * j + 1, bb, q5k_weight, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint qs_base = bb + 48 + j * 32;
            uint qh_base = bb + 16;
            uint qh_shift_lo = 2 * j;
            uint qh_shift_hi = 2 * j + 1;

            // Read 4 bytes at a time, process both low+high nibbles together
            for (uint l = 0; l < 32; l += 4) {
                uint qs_word = q5k_weight[(qs_base + l) / 4];
                uint qh_word = q5k_weight[(qh_base + l) / 4];

                uint ql0 = qs_word & 0xFFu;
                uint ql1 = (qs_word >> 8) & 0xFFu;
                uint ql2 = (qs_word >> 16) & 0xFFu;
                uint ql3 = (qs_word >> 24) & 0xFFu;

                uint qh0 = qh_word & 0xFFu;
                uint qh1 = (qh_word >> 8) & 0xFFu;
                uint qh2 = (qh_word >> 16) & 0xFFu;
                uint qh3 = (qh_word >> 24) & 0xFFu;

                // Low nibble + high bit -> first 32 elements
                sum += (d1 * float((ql0 & 0xFu) + (((qh0 >> qh_shift_lo) & 1u) << 4)) - dm1) * q5k_input[input_base + j * 64 + l];
                sum += (d1 * float((ql1 & 0xFu) + (((qh1 >> qh_shift_lo) & 1u) << 4)) - dm1) * q5k_input[input_base + j * 64 + l + 1];
                sum += (d1 * float((ql2 & 0xFu) + (((qh2 >> qh_shift_lo) & 1u) << 4)) - dm1) * q5k_input[input_base + j * 64 + l + 2];
                sum += (d1 * float((ql3 & 0xFu) + (((qh3 >> qh_shift_lo) & 1u) << 4)) - dm1) * q5k_input[input_base + j * 64 + l + 3];

                // High nibble + high bit -> next 32 elements
                sum += (d2 * float(((ql0 >> 4) & 0xFu) + (((qh0 >> qh_shift_hi) & 1u) << 4)) - dm2) * q5k_input[input_base + j * 64 + 32 + l];
                sum += (d2 * float(((ql1 >> 4) & 0xFu) + (((qh1 >> qh_shift_hi) & 1u) << 4)) - dm2) * q5k_input[input_base + j * 64 + 32 + l + 1];
                sum += (d2 * float(((ql2 >> 4) & 0xFu) + (((qh2 >> qh_shift_hi) & 1u) << 4)) - dm2) * q5k_input[input_base + j * 64 + 32 + l + 2];
                sum += (d2 * float(((ql3 >> 4) & 0xFu) + (((qh3 >> qh_shift_hi) & 1u) << 4)) - dm2) * q5k_input[input_base + j * 64 + 32 + l + 3];
            }
        }
    }

    q5k_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            q5k_shared[gi] += q5k_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0) {
        q5k_output[row] = q5k_shared[0];
    }
}

// ============================================================
// Matrix-vector multiply (Q4_K quantized weights)
// ============================================================
// Q4_K block: 144 bytes per 256 elements
// Layout: f16 d(2) + f16 dmin(2) + scales[12] + qs[128]

[vk_push_constant] const MatMulParams matmul_q4k_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> q4k_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> q4k_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> q4k_output;

groupshared float q4k_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_q4k(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    uint num_threads = 256;

    if (row >= matmul_q4k_pc.out_dim) return;

    uint blocks_per_row = matmul_q4k_pc.in_dim / 256;
    uint row_byte_offset = row * blocks_per_row * 144;

    float sum = 0.0;

    for (uint block_idx = gi; block_idx < blocks_per_row; block_idx += num_threads) {
        uint bb = row_byte_offset + block_idx * 144;
        uint input_base = block_idx * 256;

        // Read d and dmin (f16 at bytes 0 and 2)
        // bb is always 4-aligned (144 is multiple of 4)
        uint dm_word = q4k_weight[bb / 4];
        float d = unpack_f16(dm_word & 0xFFFFu);
        float dmin = unpack_f16((dm_word >> 16) & 0xFFFFu);

        // Process 4 chunks of 64 elements (8 sub-blocks of 32)
        // scales at offset 4, qs at offset 16 (both 4-aligned)
        uint qs_off = bb + 16;

        for (uint j = 0; j < 4; j++) {
            uint sc1, m1, sc2, m2;
            get_scale_min_k4(2 * j, bb, q4k_weight, sc1, m1);
            get_scale_min_k4(2 * j + 1, bb, q4k_weight, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint qs_base = qs_off + j * 32;

            // Read 4 bytes at a time, process both low+high nibbles together
            for (uint l = 0; l < 32; l += 4) {
                uint qs_word = q4k_weight[(qs_base + l) / 4];

                uint b0 = qs_word & 0xFFu;
                uint b1 = (qs_word >> 8) & 0xFFu;
                uint b2 = (qs_word >> 16) & 0xFFu;
                uint b3 = (qs_word >> 24) & 0xFFu;

                // Low nibble -> first 32 elements
                sum += (d1 * float(b0 & 0xFu) - dm1) * q4k_input[input_base + j * 64 + l];
                sum += (d1 * float(b1 & 0xFu) - dm1) * q4k_input[input_base + j * 64 + l + 1];
                sum += (d1 * float(b2 & 0xFu) - dm1) * q4k_input[input_base + j * 64 + l + 2];
                sum += (d1 * float(b3 & 0xFu) - dm1) * q4k_input[input_base + j * 64 + l + 3];

                // High nibble -> next 32 elements
                sum += (d2 * float((b0 >> 4) & 0xFu) - dm2) * q4k_input[input_base + j * 64 + 32 + l];
                sum += (d2 * float((b1 >> 4) & 0xFu) - dm2) * q4k_input[input_base + j * 64 + 32 + l + 1];
                sum += (d2 * float((b2 >> 4) & 0xFu) - dm2) * q4k_input[input_base + j * 64 + 32 + l + 2];
                sum += (d2 * float((b3 >> 4) & 0xFu) - dm2) * q4k_input[input_base + j * 64 + 32 + l + 3];
            }
        }
    }

    q4k_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            q4k_shared[gi] += q4k_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0) {
        q4k_output[row] = q4k_shared[0];
    }
}

// ============================================================
// Matrix-vector multiply (Q6_K quantized weights)
// ============================================================
// Q6_K block: 210 bytes per 256 elements
// Layout: ql[128] + qh[64] + scales[16] + f16 d(2)

[vk_push_constant] const MatMulParams matmul_q6k_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> q6k_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> q6k_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> q6k_output;

groupshared float q6k_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_q6k(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    uint num_threads = 256;

    if (row >= matmul_q6k_pc.out_dim) return;

    uint blocks_per_row = matmul_q6k_pc.in_dim / 256;
    uint row_byte_offset = row * blocks_per_row * 210;

    float sum = 0.0;

    for (uint block_idx = gi; block_idx < blocks_per_row; block_idx += num_threads) {
        uint bb = row_byte_offset + block_idx * 210;
        uint input_base = block_idx * 256;

        // Read d (f16 at offset 208)
        // 210 mod 4 = 2, so bb is always even. d_byte = bb+208 is always even.
        uint d_byte = bb + 208;
        uint d_word = q6k_weight[d_byte / 4];
        uint d_byte_off = d_byte % 4;
        uint d_bits;
        if (d_byte_off == 0) {
            d_bits = d_word & 0xFFFFu;
        } else {
            d_bits = (d_word >> 16) & 0xFFFFu;
        }
        float d = unpack_f16(d_bits);

        uint ql_base = bb;         // ql[128] at offset 0
        uint qh_base = bb + 128;   // qh[64] at offset 128
        uint sc_base = bb + 192;   // scales[16] at offset 192

        // Process 2 halves of 128 elements each
        for (uint half_idx = 0; half_idx < 2; half_idx++) {
            uint ql_off = ql_base + half_idx * 64;
            uint qh_off = qh_base + half_idx * 32;
            uint sc_off = half_idx * 8;
            uint elem_off = half_idx * 128;

            // Pre-read 8 scale values for this half (only 8 read_byte calls vs 128)
            int sc_0_0 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + 0));
            int sc_0_2 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + 2));
            int sc_0_4 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + 4));
            int sc_0_6 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + 6));
            int sc_1_0 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + 1));
            int sc_1_2 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + 3));
            int sc_1_4 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + 5));
            int sc_1_6 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + 7));

            // Read 4 bytes at a time using unaligned word reads
            for (uint l = 0; l < 32; l += 4) {
                // Select pre-read scales based on l/16
                int s1, s2, s3, s4;
                if (l < 16) {
                    s1 = sc_0_0; s2 = sc_0_2; s3 = sc_0_4; s4 = sc_0_6;
                } else {
                    s1 = sc_1_0; s2 = sc_1_2; s3 = sc_1_4; s4 = sc_1_6;
                }

                float ds1 = d * float(s1);
                float ds2 = d * float(s2);
                float ds3 = d * float(s3);
                float ds4 = d * float(s4);

                // Read 4 bytes of ql[0..63], ql[32..95], qh
                uint ql0_pos = ql_off + l;
                uint ql0_idx = ql0_pos / 4;
                uint ql0_off = ql0_pos % 4;
                uint ql0_word = q6k_weight[ql0_idx];
                if (ql0_off != 0) {
                    ql0_word = (ql0_word >> (ql0_off * 8)) | (q6k_weight[ql0_idx + 1] << (32 - ql0_off * 8));
                }

                uint ql32_pos = ql_off + 32 + l;
                uint ql32_idx = ql32_pos / 4;
                uint ql32_off = ql32_pos % 4;
                uint ql32_word = q6k_weight[ql32_idx];
                if (ql32_off != 0) {
                    ql32_word = (ql32_word >> (ql32_off * 8)) | (q6k_weight[ql32_idx + 1] << (32 - ql32_off * 8));
                }

                uint qh_pos = qh_off + l;
                uint qh_idx = qh_pos / 4;
                uint qh_off2 = qh_pos % 4;
                uint qh_word = q6k_weight[qh_idx];
                if (qh_off2 != 0) {
                    qh_word = (qh_word >> (qh_off2 * 8)) | (q6k_weight[qh_idx + 1] << (32 - qh_off2 * 8));
                }

                // Process 4 elements
                for (uint k = 0; k < 4; k++) {
                    uint ql0_b = (ql0_word >> (k * 8)) & 0xFFu;
                    uint ql32_b = (ql32_word >> (k * 8)) & 0xFFu;
                    uint qh_b = (qh_word >> (k * 8)) & 0xFFu;

                    int q1 = int((ql0_b & 0xFu) | (((qh_b >> 0) & 3u) << 4)) - 32;
                    int q2 = int((ql32_b & 0xFu) | (((qh_b >> 2) & 3u) << 4)) - 32;
                    int q3 = int(((ql0_b >> 4) & 0xFu) | (((qh_b >> 4) & 3u) << 4)) - 32;
                    int q4 = int(((ql32_b >> 4) & 0xFu) | (((qh_b >> 6) & 3u) << 4)) - 32;

                    sum += ds1 * float(q1) * q6k_input[input_base + elem_off + l + k];
                    sum += ds2 * float(q2) * q6k_input[input_base + elem_off + l + k + 32];
                    sum += ds3 * float(q3) * q6k_input[input_base + elem_off + l + k + 64];
                    sum += ds4 * float(q4) * q6k_input[input_base + elem_off + l + k + 96];
                }
            }
        }
    }

    q6k_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            q6k_shared[gi] += q6k_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0) {
        q6k_output[row] = q6k_shared[0];
    }
}

// ============================================================
// RoPE (Rotary Position Embeddings)
// ============================================================

struct RoPEParams {
    uint dim;       // head_dim
    uint n_heads;
    uint position;
    float theta;    // base frequency (10000.0)
};

[vk_push_constant] const RoPEParams rope_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> rope_data;

[shader("compute")]
[numthreads(128, 1, 1)]
void rope(uint3 tid: SV_DispatchThreadID) {
    uint total_pairs = rope_pc.n_heads * (rope_pc.dim / 2);
    if (tid.x >= total_pairs) return;

    uint head = tid.x / (rope_pc.dim / 2);
    uint pair = tid.x % (rope_pc.dim / 2);

    float freq = 1.0 / pow(rope_pc.theta, float(2 * pair) / float(rope_pc.dim));
    float angle = float(rope_pc.position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    uint base = head * rope_pc.dim + pair * 2;
    float x0 = rope_data[base];
    float x1 = rope_data[base + 1];

    rope_data[base]     = x0 * cos_val - x1 * sin_val;
    rope_data[base + 1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================
// Softmax
// ============================================================

struct SoftmaxParams {
    uint n;
    uint offset;
};

[vk_push_constant] const SoftmaxParams softmax_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> softmax_data;

groupshared float softmax_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void softmax(uint gi: SV_GroupIndex) {
    uint num_threads = 256;

    // Find max
    float max_val = -1.0e30;
    for (uint i = gi; i < softmax_pc.n; i += num_threads) {
        max_val = max(max_val, softmax_data[softmax_pc.offset + i]);
    }

    softmax_shared[gi] = max_val;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            softmax_shared[gi] = max(softmax_shared[gi], softmax_shared[gi + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float global_max = softmax_shared[0];
    GroupMemoryBarrierWithGroupSync();

    // Exp and sum
    float sum = 0.0;
    for (uint i = gi; i < softmax_pc.n; i += num_threads) {
        float val = exp(softmax_data[softmax_pc.offset + i] - global_max);
        softmax_data[softmax_pc.offset + i] = val;
        sum += val;
    }

    softmax_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            softmax_shared[gi] += softmax_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float global_sum = softmax_shared[0];
    GroupMemoryBarrierWithGroupSync();

    // Normalize
    for (uint i = gi; i < softmax_pc.n; i += num_threads) {
        softmax_data[softmax_pc.offset + i] /= global_sum;
    }
}

// ============================================================
// Attention (GQA: grouped query attention)
// ============================================================

struct AttentionParams {
    uint head_dim;
    uint n_kv_heads;
    uint n_q_heads;
    uint seq_len;
    float scale;
    uint max_seq_len;
};

[vk_push_constant] const AttentionParams attn_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> attn_q;
[vk_binding(1, 0)] RWStructuredBuffer<float> attn_k_cache;
[vk_binding(2, 0)] RWStructuredBuffer<float> attn_v_cache;
[vk_binding(3, 0)] RWStructuredBuffer<float> attn_scores;
[vk_binding(4, 0)] RWStructuredBuffer<float> attn_output;

groupshared float attn_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void attention(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint q_head = gid.x;
    uint num_threads = 256;

    if (q_head >= attn_pc.n_q_heads) return;

    // GQA: map q_head to kv_head
    uint kv_head = q_head / (attn_pc.n_q_heads / attn_pc.n_kv_heads);

    // Phase 1: Q * K^T scores
    for (uint pos = gi; pos < attn_pc.seq_len; pos += num_threads) {
        float dot_val = 0.0;
        for (uint d = 0; d < attn_pc.head_dim; d++) {
            dot_val += attn_q[q_head * attn_pc.head_dim + d] *
                       attn_k_cache[kv_head * attn_pc.max_seq_len * attn_pc.head_dim + pos * attn_pc.head_dim + d];
        }
        attn_scores[q_head * attn_pc.seq_len + pos] = dot_val * attn_pc.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Softmax over scores
    float max_val = -1.0e30;
    for (uint i = gi; i < attn_pc.seq_len; i += num_threads) {
        max_val = max(max_val, attn_scores[q_head * attn_pc.seq_len + i]);
    }
    attn_shared[gi] = max_val;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            attn_shared[gi] = max(attn_shared[gi], attn_shared[gi + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float global_max = attn_shared[0];
    GroupMemoryBarrierWithGroupSync();

    float sum = 0.0;
    for (uint i = gi; i < attn_pc.seq_len; i += num_threads) {
        float val = exp(attn_scores[q_head * attn_pc.seq_len + i] - global_max);
        attn_scores[q_head * attn_pc.seq_len + i] = val;
        sum += val;
    }
    attn_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) {
            attn_shared[gi] += attn_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float global_sum = attn_shared[0];
    GroupMemoryBarrierWithGroupSync();

    for (uint i = gi; i < attn_pc.seq_len; i += num_threads) {
        attn_scores[q_head * attn_pc.seq_len + i] /= global_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: Weighted sum of V
    for (uint d = gi; d < attn_pc.head_dim; d += num_threads) {
        float acc = 0.0;
        for (uint pos = 0; pos < attn_pc.seq_len; pos++) {
            acc += attn_scores[q_head * attn_pc.seq_len + pos] *
                   attn_v_cache[kv_head * attn_pc.max_seq_len * attn_pc.head_dim + pos * attn_pc.head_dim + d];
        }
        attn_output[q_head * attn_pc.head_dim + d] = acc;
    }
}

// ============================================================
// Residual add (in-place: acc += residual)
// ============================================================

struct ResidualParams {
    uint n;
};

[vk_push_constant] const ResidualParams residual_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> residual_acc;
[vk_binding(1, 0)] RWStructuredBuffer<float> residual_src;

[shader("compute")]
[numthreads(256, 1, 1)]
void residual_add(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= residual_pc.n) return;
    residual_acc[tid.x] += residual_src[tid.x];
}

// ============================================================
// Element-wise multiply (in-place: a *= b, for SwiGLU gate*up)
// ============================================================

struct ElemwiseParams {
    uint n;
};

[vk_push_constant] const ElemwiseParams elemwise_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> elemwise_a;
[vk_binding(1, 0)] RWStructuredBuffer<float> elemwise_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void elemwise_mul(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= elemwise_pc.n) return;
    elemwise_a[tid.x] *= elemwise_b[tid.x];
}

// ============================================================
// Batch matrix multiply - all positions in one dispatch
// ============================================================
// weight[out_dim, in_dim] × input[seq_len, in_dim] → output[seq_len, out_dim]
// row_offset: skip rows in weight matrix (for fused QKV split)
// Dispatch: seq_len * out_dim workgroups

struct BatchMatMulParams {
    uint out_dim;
    uint in_dim;
    uint seq_len;
    uint row_offset;
};

// --- F32 batch matmul ---

[vk_push_constant] const BatchMatMulParams bm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bm_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> bm_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> bm_output;

// 2D Tiled GEMM: BM=64, BN=64, BK=32, TM=4, TN=4
// 256 threads (16x16), each computes 4x4 output elements
// Shared memory: [BK][BM+1] + [BK][BN+1] with +1 padding to avoid bank conflicts
// Dispatch: ceil(seq_len/64) * ceil(out_dim/64) workgroups

groupshared float bm_A[32 * 65];  // A tile [BK=32][BM=64+1] = 2080 floats
groupshared float bm_B[32 * 65];  // B tile [BK=32][BN=64+1] = 2080 floats

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_matmul(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint tiles_n = (bm_pc.out_dim + 63) / 64;
    uint tile_m = gid.x / tiles_n;
    uint tile_n = gid.x % tiles_n;
    uint pos_base = tile_m * 64;
    uint row_base = bm_pc.row_offset + tile_n * 64;

    uint tm = gi / 16;   // 0..15
    uint tn = gi % 16;   // 0..15

    float acc[16];
    for (uint i = 0; i < 16; i++) acc[i] = 0.0;

    uint k_tiles = (bm_pc.in_dim + 31) / 32;

    for (uint t = 0; t < k_tiles; t++) {
        uint k_base = t * 32;

        // Load A tile [BK=32][BM=64]: 2048 elements, 8 per thread
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint m_local = flat >> 5;     // flat / 32
            uint k_local = flat & 31u;    // flat % 32
            uint gm = pos_base + m_local;
            uint gk = k_base + k_local;
            bm_A[k_local * 65 + m_local] = (gm < bm_pc.seq_len && gk < bm_pc.in_dim) ?
                bm_input[gm * bm_pc.in_dim + gk] : 0.0;
        }

        // Load B tile [BK=32][BN=64]: 2048 elements, 8 per thread
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint n_local = flat >> 5;
            uint k_local = flat & 31u;
            uint gn = row_base + n_local;
            uint gk = k_base + k_local;
            bm_B[k_local * 65 + n_local] = (gn < bm_pc.row_offset + bm_pc.out_dim && gk < bm_pc.in_dim) ?
                bm_weight[gn * bm_pc.in_dim + gk] : 0.0;
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute: 4x4 outer products over BK=32
        for (uint k = 0; k < 32; k++) {
            float a0 = bm_A[k * 65 + tm * 4];
            float a1 = bm_A[k * 65 + tm * 4 + 1];
            float a2 = bm_A[k * 65 + tm * 4 + 2];
            float a3 = bm_A[k * 65 + tm * 4 + 3];
            float b0 = bm_B[k * 65 + tn * 4];
            float b1 = bm_B[k * 65 + tn * 4 + 1];
            float b2 = bm_B[k * 65 + tn * 4 + 2];
            float b3 = bm_B[k * 65 + tn * 4 + 3];
            acc[0]  += a0 * b0; acc[1]  += a0 * b1; acc[2]  += a0 * b2; acc[3]  += a0 * b3;
            acc[4]  += a1 * b0; acc[5]  += a1 * b1; acc[6]  += a1 * b2; acc[7]  += a1 * b3;
            acc[8]  += a2 * b0; acc[9]  += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
            acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write 4x4 outputs
    for (uint r = 0; r < 4; r++) {
        uint gm = pos_base + tm * 4 + r;
        if (gm < bm_pc.seq_len) {
            for (uint c = 0; c < 4; c++) {
                uint gn = tile_n * 64 + tn * 4 + c;
                if (gn < bm_pc.out_dim) {
                    bm_output[gm * bm_pc.out_dim + gn] = acc[r * 4 + c];
                }
            }
        }
    }
}

// --- Simple F32 batch matmul (1D dispatch, no tiling) ---
// This is a simplified version that uses 1D dispatch like the Q8 shader
// to avoid the tile indexing bugs in the tiled version
// Use this for debugging or when tiled version has permutation issues

[vk_push_constant] const BatchMatMulParams bm_simple_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bm_simple_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> bm_simple_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> bm_simple_output;

groupshared float bm_simple_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_matmul_simple(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    // Support both 1D and 2D dispatch
    // 2D dispatch: gid.x = pos, gid.y = out_idx
    // 1D dispatch: gid.x = pos * out_dim + out_idx, gid.y = 0
    uint pos, out_idx;
    if (gid.y == 0 && gid.x >= bm_simple_pc.seq_len) {
        // Assume 1D dispatch
        pos = gid.x / bm_simple_pc.out_dim;
        out_idx = gid.x % bm_simple_pc.out_dim;
    } else {
        // 2D dispatch
        pos = gid.x;
        out_idx = gid.y;
    }
    
    if (pos >= bm_simple_pc.seq_len) return;
    
    uint actual_out = bm_simple_pc.row_offset + out_idx;
    uint in_base = pos * bm_simple_pc.in_dim;
    
    // Each thread computes a partial sum
    float partial = 0.0;
    for (uint k = gi; k < bm_simple_pc.in_dim; k += 256) {
        // Weight layout: [in_dim, out_dim] - weight[k, actual_out]
        partial += bm_simple_weight[k * (bm_simple_pc.row_offset + bm_simple_pc.out_dim) + actual_out] 
                   * bm_simple_input[in_base + k];
    }
    
    bm_simple_shared[gi] = partial;
    GroupMemoryBarrierWithGroupSync();
    
    // Tree reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) {
            bm_simple_shared[gi] += bm_simple_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Thread 0 writes final result
    if (gi == 0) {
        bm_simple_output[pos * bm_simple_pc.out_dim + out_idx] = bm_simple_shared[0];
        
        // Debug: Also store gid.x in a separate location to trace workgroup mapping
        // Use last row (pos 1023) columns 0-4 to store debug info
        if (pos == 0 && out_idx < 5 && bm_simple_pc.seq_len > 1023) {
            bm_simple_output[1023 * bm_simple_pc.out_dim + out_idx] = float(gid.x);
        }
    }
}

// --- Q8_0 batch matmul ---

[vk_push_constant] const BatchMatMulParams bmq8_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> bmq8_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> bmq8_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> bmq8_output;

groupshared float bmq8_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_matmul_q8(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint pos = gid.x / bmq8_pc.out_dim;
    uint row = gid.x % bmq8_pc.out_dim;
    if (pos >= bmq8_pc.seq_len) return;
    uint actual_row = bmq8_pc.row_offset + row;

    uint blocks_per_row = bmq8_pc.in_dim / 32;
    uint bytes_per_row = blocks_per_row * 34;
    uint row_byte_offset = actual_row * bytes_per_row;
    uint in_base = pos * bmq8_pc.in_dim;
    float sum = 0.0;

    for (uint block_idx = gi; block_idx < blocks_per_row; block_idx += 256) {
        uint block_byte = row_byte_offset + block_idx * 34;
        uint scale_uint_idx = block_byte / 4;
        uint scale_byte_offset = block_byte % 4;
        uint scale_word = bmq8_weight[scale_uint_idx];
        uint scale_bits;
        if (scale_byte_offset == 0) {
            scale_bits = scale_word & 0xFFFFu;
        } else if (scale_byte_offset == 2) {
            scale_bits = (scale_word >> 16) & 0xFFFFu;
        } else {
            uint next_word = bmq8_weight[scale_uint_idx + 1];
            scale_bits = ((scale_word >> (scale_byte_offset * 8)) | (next_word << (32 - scale_byte_offset * 8))) & 0xFFFFu;
        }
        float scale = unpack_f16(scale_bits);
        uint qs_byte = block_byte + 2;
        uint input_offset = in_base + block_idx * 32;

        for (uint j = 0; j < 32; j += 4) {
            uint byte_pos = qs_byte + j;
            uint uint_idx = byte_pos / 4;
            uint byte_off = byte_pos % 4;
            uint word = bmq8_weight[uint_idx];
            if (byte_off != 0) {
                uint next_word = bmq8_weight[uint_idx + 1];
                word = (word >> (byte_off * 8)) | (next_word << (32 - byte_off * 8));
            }
            sum += scale * float(sign_extend_i8(word & 0xFFu)) * bmq8_input[input_offset + j];
            sum += scale * float(sign_extend_i8((word >> 8) & 0xFFu)) * bmq8_input[input_offset + j + 1];
            sum += scale * float(sign_extend_i8((word >> 16) & 0xFFu)) * bmq8_input[input_offset + j + 2];
            sum += scale * float(sign_extend_i8((word >> 24) & 0xFFu)) * bmq8_input[input_offset + j + 3];
        }
    }

    bmq8_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) bmq8_shared[gi] += bmq8_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) {
        bmq8_output[pos * bmq8_pc.out_dim + row] = bmq8_shared[0];
    }
}

// --- Q4_0 batch matmul ---

[vk_push_constant] const BatchMatMulParams bmq40_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> bmq40_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> bmq40_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> bmq40_output;

groupshared float bmq40_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_matmul_q4_0(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint pos = gid.x / bmq40_pc.out_dim;
    uint row = gid.x % bmq40_pc.out_dim;
    if (pos >= bmq40_pc.seq_len) return;
    uint actual_row = bmq40_pc.row_offset + row;

    uint blocks_per_row = bmq40_pc.in_dim / 32;
    uint bytes_per_row = blocks_per_row * 18;
    uint row_byte_offset = actual_row * bytes_per_row;
    uint in_base = pos * bmq40_pc.in_dim;
    float sum = 0.0;

    for (uint block_idx = gi; block_idx < blocks_per_row; block_idx += 256) {
        uint block_byte = row_byte_offset + block_idx * 18;
        uint scale_uint_idx = block_byte / 4;
        uint scale_byte_offset = block_byte % 4;
        uint scale_word = bmq40_weight[scale_uint_idx];
        uint scale_bits;
        if (scale_byte_offset == 0) {
            scale_bits = scale_word & 0xFFFFu;
        } else if (scale_byte_offset == 2) {
            scale_bits = (scale_word >> 16) & 0xFFFFu;
        } else {
            uint next_word = bmq40_weight[scale_uint_idx + 1];
            scale_bits = ((scale_word >> (scale_byte_offset * 8)) | (next_word << (32 - scale_byte_offset * 8))) & 0xFFFFu;
        }
        float scale = unpack_f16(scale_bits);
        uint qs_byte = block_byte + 2;
        uint input_offset = in_base + block_idx * 32;

        for (uint j = 0; j < 16; j += 4) {
            uint byte_pos = qs_byte + j;
            uint uint_idx = byte_pos / 4;
            uint byte_off = byte_pos % 4;
            uint word = bmq40_weight[uint_idx];
            if (byte_off != 0) {
                uint next_word = bmq40_weight[uint_idx + 1];
                word = (word >> (byte_off * 8)) | (next_word << (32 - byte_off * 8));
            }
            sum += scale * float(int(word & 0xFu) - 8) * bmq40_input[input_offset + j];
            sum += scale * float(int((word >> 8) & 0xFu) - 8) * bmq40_input[input_offset + j + 1];
            sum += scale * float(int((word >> 16) & 0xFu) - 8) * bmq40_input[input_offset + j + 2];
            sum += scale * float(int((word >> 24) & 0xFu) - 8) * bmq40_input[input_offset + j + 3];
            sum += scale * float(int((word >> 4) & 0xFu) - 8) * bmq40_input[input_offset + 16 + j];
            sum += scale * float(int((word >> 12) & 0xFu) - 8) * bmq40_input[input_offset + 16 + j + 1];
            sum += scale * float(int((word >> 20) & 0xFu) - 8) * bmq40_input[input_offset + 16 + j + 2];
            sum += scale * float(int((word >> 28) & 0xFu) - 8) * bmq40_input[input_offset + 16 + j + 3];
        }
    }

    bmq40_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) bmq40_shared[gi] += bmq40_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) {
        bmq40_output[pos * bmq40_pc.out_dim + row] = bmq40_shared[0];
    }
}

// --- Q4_K batch matmul ---

[vk_push_constant] const BatchMatMulParams bmq4k_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> bmq4k_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> bmq4k_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> bmq4k_output;

groupshared float q4k_A[32 * 65];  // [BK=32][BM=64+1]
groupshared float q4k_B[32 * 65];  // [BK=32][BN=64+1]

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_matmul_q4k(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint tiles_n = (bmq4k_pc.out_dim + 63) / 64;
    uint tile_m = gid.x / tiles_n;
    uint tile_n = gid.x % tiles_n;
    uint pos_base = tile_m * 64;
    uint row_base = bmq4k_pc.row_offset + tile_n * 64;

    uint tm = gi / 16;
    uint tn = gi % 16;

    float acc[16];
    for (uint i = 0; i < 16; i++) acc[i] = 0.0;

    uint blocks_per_row = bmq4k_pc.in_dim / 256;
    uint k_tiles = bmq4k_pc.in_dim / 32;  // in_dim is always multiple of 256

    for (uint t = 0; t < k_tiles; t++) {
        uint k_base = t * 32;

        // Load A tile (F32 input)
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint m_local = flat >> 5;
            uint k_local = flat & 31u;
            uint gm = pos_base + m_local;
            q4k_A[k_local * 65 + m_local] = (gm < bmq4k_pc.seq_len) ?
                bmq4k_input[gm * bmq4k_pc.in_dim + k_base + k_local] : 0.0;
        }

        // Load B tile (Q4_K dequantize)
        // BK=32 chunk within a 256-element block: all share same sub/half/scale
        uint block_idx = k_base / 256;
        uint chunk = (k_base % 256) / 32;  // 0..7
        uint sub = chunk / 2;
        uint half = chunk % 2;
        uint scale_idx = 2 * sub + half;

        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint n_local = flat >> 5;
            uint k_local = flat & 31u;
            uint gn = row_base + n_local;
            if (gn < bmq4k_pc.row_offset + bmq4k_pc.out_dim) {
                uint bb = gn * blocks_per_row * 144 + block_idx * 144;
                uint dm_word = bmq4k_weight[bb / 4];
                float d = unpack_f16(dm_word & 0xFFFFu);
                float dmin = unpack_f16((dm_word >> 16) & 0xFFFFu);
                uint sc, mn;
                get_scale_min_k4(scale_idx, bb, bmq4k_weight, sc, mn);
                uint qs_byte = read_byte(bmq4k_weight, bb + 16 + sub * 32 + k_local);
                uint q = (half == 0) ? (qs_byte & 0xFu) : ((qs_byte >> 4) & 0xFu);
                q4k_B[k_local * 65 + n_local] = d * float(sc) * float(q) - dmin * float(mn);
            } else {
                q4k_B[k_local * 65 + n_local] = 0.0;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < 32; k++) {
            float a0 = q4k_A[k * 65 + tm * 4];
            float a1 = q4k_A[k * 65 + tm * 4 + 1];
            float a2 = q4k_A[k * 65 + tm * 4 + 2];
            float a3 = q4k_A[k * 65 + tm * 4 + 3];
            float b0 = q4k_B[k * 65 + tn * 4];
            float b1 = q4k_B[k * 65 + tn * 4 + 1];
            float b2 = q4k_B[k * 65 + tn * 4 + 2];
            float b3 = q4k_B[k * 65 + tn * 4 + 3];
            acc[0]  += a0 * b0; acc[1]  += a0 * b1; acc[2]  += a0 * b2; acc[3]  += a0 * b3;
            acc[4]  += a1 * b0; acc[5]  += a1 * b1; acc[6]  += a1 * b2; acc[7]  += a1 * b3;
            acc[8]  += a2 * b0; acc[9]  += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
            acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    for (uint r = 0; r < 4; r++) {
        uint gm = pos_base + tm * 4 + r;
        if (gm < bmq4k_pc.seq_len) {
            for (uint c = 0; c < 4; c++) {
                uint gn = tile_n * 64 + tn * 4 + c;
                if (gn < bmq4k_pc.out_dim) {
                    bmq4k_output[gm * bmq4k_pc.out_dim + gn] = acc[r * 4 + c];
                }
            }
        }
    }
}

// --- Q5_K batch matmul ---

[vk_push_constant] const BatchMatMulParams bmq5k_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> bmq5k_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> bmq5k_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> bmq5k_output;

groupshared float q5k_A[32 * 65];
groupshared float q5k_B[32 * 65];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_matmul_q5k(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint tiles_n = (bmq5k_pc.out_dim + 63) / 64;
    uint tile_m = gid.x / tiles_n;
    uint tile_n = gid.x % tiles_n;
    uint pos_base = tile_m * 64;
    uint row_base = bmq5k_pc.row_offset + tile_n * 64;

    uint tm = gi / 16;
    uint tn = gi % 16;

    float acc[16];
    for (uint i = 0; i < 16; i++) acc[i] = 0.0;

    uint blocks_per_row = bmq5k_pc.in_dim / 256;
    uint k_tiles = bmq5k_pc.in_dim / 32;

    for (uint t = 0; t < k_tiles; t++) {
        uint k_base = t * 32;

        // Load A tile (F32 input)
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint m_local = flat >> 5;
            uint k_local = flat & 31u;
            uint gm = pos_base + m_local;
            q5k_A[k_local * 65 + m_local] = (gm < bmq5k_pc.seq_len) ?
                bmq5k_input[gm * bmq5k_pc.in_dim + k_base + k_local] : 0.0;
        }

        // Load B tile (Q5_K dequantize)
        uint block_idx = k_base / 256;
        uint chunk = (k_base % 256) / 32;
        uint sub = chunk / 2;
        uint half = chunk % 2;
        uint scale_idx = 2 * sub + half;

        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint n_local = flat >> 5;
            uint k_local = flat & 31u;
            uint gn = row_base + n_local;
            if (gn < bmq5k_pc.row_offset + bmq5k_pc.out_dim) {
                uint bb = gn * blocks_per_row * 176 + block_idx * 176;
                uint dm_word = bmq5k_weight[bb / 4];
                float d = unpack_f16(dm_word & 0xFFFFu);
                float dmin = unpack_f16((dm_word >> 16) & 0xFFFFu);
                uint sc, mn;
                get_scale_min_k4(scale_idx, bb, bmq5k_weight, sc, mn);
                uint qs_byte = read_byte(bmq5k_weight, bb + 48 + sub * 32 + k_local);
                uint ql = (half == 0) ? (qs_byte & 0xFu) : ((qs_byte >> 4) & 0xFu);
                uint qh_byte = read_byte(bmq5k_weight, bb + 16 + k_local);
                uint qh_bit = (qh_byte >> (2 * sub + half)) & 1u;
                uint q = ql + (qh_bit << 4);
                q5k_B[k_local * 65 + n_local] = d * float(sc) * float(q) - dmin * float(mn);
            } else {
                q5k_B[k_local * 65 + n_local] = 0.0;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < 32; k++) {
            float a0 = q5k_A[k * 65 + tm * 4];
            float a1 = q5k_A[k * 65 + tm * 4 + 1];
            float a2 = q5k_A[k * 65 + tm * 4 + 2];
            float a3 = q5k_A[k * 65 + tm * 4 + 3];
            float b0 = q5k_B[k * 65 + tn * 4];
            float b1 = q5k_B[k * 65 + tn * 4 + 1];
            float b2 = q5k_B[k * 65 + tn * 4 + 2];
            float b3 = q5k_B[k * 65 + tn * 4 + 3];
            acc[0]  += a0 * b0; acc[1]  += a0 * b1; acc[2]  += a0 * b2; acc[3]  += a0 * b3;
            acc[4]  += a1 * b0; acc[5]  += a1 * b1; acc[6]  += a1 * b2; acc[7]  += a1 * b3;
            acc[8]  += a2 * b0; acc[9]  += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
            acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    for (uint r = 0; r < 4; r++) {
        uint gm = pos_base + tm * 4 + r;
        if (gm < bmq5k_pc.seq_len) {
            for (uint c = 0; c < 4; c++) {
                uint gn = tile_n * 64 + tn * 4 + c;
                if (gn < bmq5k_pc.out_dim) {
                    bmq5k_output[gm * bmq5k_pc.out_dim + gn] = acc[r * 4 + c];
                }
            }
        }
    }
}

// --- Q6_K batch matmul ---

[vk_push_constant] const BatchMatMulParams bmq6k_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> bmq6k_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> bmq6k_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> bmq6k_output;

groupshared float q6k_A[32 * 65];
groupshared float q6k_B[32 * 65];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_matmul_q6k(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint tiles_n = (bmq6k_pc.out_dim + 63) / 64;
    uint tile_m = gid.x / tiles_n;
    uint tile_n = gid.x % tiles_n;
    uint pos_base = tile_m * 64;
    uint row_base = bmq6k_pc.row_offset + tile_n * 64;

    uint tm = gi / 16;
    uint tn = gi % 16;

    float acc[16];
    for (uint i = 0; i < 16; i++) acc[i] = 0.0;

    uint blocks_per_row = bmq6k_pc.in_dim / 256;
    uint k_tiles = bmq6k_pc.in_dim / 32;

    for (uint t = 0; t < k_tiles; t++) {
        uint k_base = t * 32;

        // Load A tile (F32 input)
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint m_local = flat >> 5;
            uint k_local = flat & 31u;
            uint gm = pos_base + m_local;
            q6k_A[k_local * 65 + m_local] = (gm < bmq6k_pc.seq_len) ?
                bmq6k_input[gm * bmq6k_pc.in_dim + k_base + k_local] : 0.0;
        }

        // Load B tile (Q6_K dequantize)
        // Map BK=32 chunk to Q6_K element positions
        uint block_idx = k_base / 256;
        uint local_k_start = k_base % 256;  // 0, 32, 64, ..., 224
        // Q6_K: half_idx = local_k / 128, quadrant = (local_k%128) / 32
        uint half_idx = local_k_start / 128;
        uint quadrant = (local_k_start % 128) / 32;

        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint n_local = flat >> 5;
            uint k_local = flat & 31u;
            uint gn = row_base + n_local;
            if (gn < bmq6k_pc.row_offset + bmq6k_pc.out_dim) {
                uint bb = gn * blocks_per_row * 210 + block_idx * 210;

                uint d_byte = bb + 208;
                uint d_word = bmq6k_weight[d_byte / 4];
                uint d_byte_off = d_byte % 4;
                uint d_bits;
                if (d_byte_off == 0) { d_bits = d_word & 0xFFFFu; }
                else { d_bits = (d_word >> 16) & 0xFFFFu; }
                float d = unpack_f16(d_bits);

                uint ql_byte = read_byte(bmq6k_weight, bb + half_idx * 64 + (quadrant % 2) * 32 + k_local);
                uint ql_nibble = (quadrant < 2) ? (ql_byte & 0xFu) : ((ql_byte >> 4) & 0xFu);
                uint qh_byte = read_byte(bmq6k_weight, bb + 128 + half_idx * 32 + k_local);
                uint qh_bits = (qh_byte >> (quadrant * 2)) & 3u;
                int q = int(ql_nibble | (qh_bits << 4)) - 32;
                int sc = sign_extend_i8(read_byte(bmq6k_weight, bb + 192 + half_idx * 8 + quadrant * 2 + (k_local >= 16 ? 1u : 0u)));

                q6k_B[k_local * 65 + n_local] = d * float(sc) * float(q);
            } else {
                q6k_B[k_local * 65 + n_local] = 0.0;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < 32; k++) {
            float a0 = q6k_A[k * 65 + tm * 4];
            float a1 = q6k_A[k * 65 + tm * 4 + 1];
            float a2 = q6k_A[k * 65 + tm * 4 + 2];
            float a3 = q6k_A[k * 65 + tm * 4 + 3];
            float b0 = q6k_B[k * 65 + tn * 4];
            float b1 = q6k_B[k * 65 + tn * 4 + 1];
            float b2 = q6k_B[k * 65 + tn * 4 + 2];
            float b3 = q6k_B[k * 65 + tn * 4 + 3];
            acc[0]  += a0 * b0; acc[1]  += a0 * b1; acc[2]  += a0 * b2; acc[3]  += a0 * b3;
            acc[4]  += a1 * b0; acc[5]  += a1 * b1; acc[6]  += a1 * b2; acc[7]  += a1 * b3;
            acc[8]  += a2 * b0; acc[9]  += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
            acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    for (uint r = 0; r < 4; r++) {
        uint gm = pos_base + tm * 4 + r;
        if (gm < bmq6k_pc.seq_len) {
            for (uint c = 0; c < 4; c++) {
                uint gn = tile_n * 64 + tn * 4 + c;
                if (gn < bmq6k_pc.out_dim) {
                    bmq6k_output[gm * bmq6k_pc.out_dim + gn] = acc[r * 4 + c];
                }
            }
        }
    }
}
