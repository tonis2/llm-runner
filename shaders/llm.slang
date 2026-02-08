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
        uint dm_word = q5k_weight[bb / 4];
        uint byte_off = bb % 4;
        uint d_bits, dmin_bits;
        if (byte_off == 0) {
            d_bits = dm_word & 0xFFFFu;
            dmin_bits = (dm_word >> 16) & 0xFFFFu;
        } else if (byte_off == 2) {
            d_bits = (dm_word >> 16) & 0xFFFFu;
            uint next = q5k_weight[bb / 4 + 1];
            dmin_bits = next & 0xFFFFu;
        } else {
            // Unaligned - read bytes
            d_bits = (read_byte(q5k_weight, bb) | (read_byte(q5k_weight, bb + 1) << 8));
            dmin_bits = (read_byte(q5k_weight, bb + 2) | (read_byte(q5k_weight, bb + 3) << 8));
        }
        float d = unpack_f16(d_bits);
        float dmin = unpack_f16(dmin_bits);

        // Process 4 chunks of 64 elements each (8 sub-blocks of 32)
        for (uint j = 0; j < 4; j++) {
            // Get scales for sub-blocks 2*j and 2*j+1
            uint sc1, m1, sc2, m2;
            get_scale_min_k4(2 * j, bb, q5k_weight, sc1, m1);
            get_scale_min_k4(2 * j + 1, bb, q5k_weight, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint qs_base = bb + 48 + j * 32; // qs starts at offset 48
            uint qh_base = bb + 16;           // qh starts at offset 16

            // First 32 elements: low nibble + high bit
            for (uint l = 0; l < 32; l++) {
                uint ql_byte = read_byte(q5k_weight, qs_base + l);
                uint qh_byte = read_byte(q5k_weight, qh_base + l);
                uint q_low = ql_byte & 0xFu;
                uint q_high = (qh_byte >> (2 * j)) & 1u;
                float val = d1 * float(q_low + q_high * 16u) - dm1;
                sum += val * q5k_input[input_base + j * 64 + l];
            }

            // Next 32 elements: high nibble + high bit
            for (uint l = 0; l < 32; l++) {
                uint ql_byte = read_byte(q5k_weight, qs_base + l);
                uint qh_byte = read_byte(q5k_weight, qh_base + l);
                uint q_low = (ql_byte >> 4) & 0xFu;
                uint q_high = (qh_byte >> (2 * j + 1)) & 1u;
                float val = d2 * float(q_low + q_high * 16u) - dm2;
                sum += val * q5k_input[input_base + j * 64 + 32 + l];
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
        uint d_byte = bb + 208;
        uint d_bits = read_byte(q6k_weight, d_byte) | (read_byte(q6k_weight, d_byte + 1) << 8);
        float d = unpack_f16(d_bits);

        uint ql_base = bb;         // ql[128] at offset 0
        uint qh_base = bb + 128;   // qh[64] at offset 128
        uint sc_base = bb + 192;   // scales[16] at offset 192

        // Process 2 halves of 128 elements each
        // Matches ggml reference: ql advances by 64 per half, qh by 32, scales by 8
        for (uint half = 0; half < 2; half++) {
            uint ql_off = ql_base + half * 64;
            uint qh_off = qh_base + half * 32;
            uint sc_off = half * 8;
            uint elem_off = half * 128;

            for (uint l = 0; l < 32; l++) {
                uint is_idx = l / 16;

                uint ql0 = read_byte(q6k_weight, ql_off + l);
                uint ql32 = read_byte(q6k_weight, ql_off + 32 + l);
                uint qh_val = read_byte(q6k_weight, qh_off + l);

                int q1 = int((ql0 & 0xFu) | (((qh_val >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql32 & 0xFu) | (((qh_val >> 2) & 3u) << 4)) - 32;
                int q3 = int(((ql0 >> 4) & 0xFu) | (((qh_val >> 4) & 3u) << 4)) - 32;
                int q4 = int(((ql32 >> 4) & 0xFu) | (((qh_val >> 6) & 3u) << 4)) - 32;

                int sc1 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + is_idx + 0));
                int sc2 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + is_idx + 2));
                int sc3 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + is_idx + 4));
                int sc4 = sign_extend_i8(read_byte(q6k_weight, sc_base + sc_off + is_idx + 6));

                sum += d * float(sc1) * float(q1) * q6k_input[input_base + elem_off + l + 0];
                sum += d * float(sc2) * float(q2) * q6k_input[input_base + elem_off + l + 32];
                sum += d * float(sc3) * float(q3) * q6k_input[input_base + elem_off + l + 64];
                sum += d * float(sc4) * float(q4) * q6k_input[input_base + elem_off + l + 96];
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
