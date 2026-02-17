// Diffusion model compute shaders
// Spatial operations for Conv2D, GroupNorm, attention, upsampling, scheduling

// Shared helpers from llm.slang (duplicated to keep shaders independent)
float unpack_f16_diff(uint bits) {
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

uint read_byte_diff(RWStructuredBuffer<uint> buf, uint byte_idx) {
    return (buf[byte_idx / 4] >> ((byte_idx % 4) * 8)) & 0xFFu;
}

int sign_extend_i8_diff(uint v) {
    return (v & 0x80u) != 0 ? int(v) - 256 : int(v);
}

// ============================================================
// Conv2D - Direct convolution
// ============================================================
// NCHW layout: idx = c * H * W + h * W + w
// Weight shape in GGUF: [kH, kW, C_in, C_out] stored as F32
// One workgroup per output pixel per output channel

struct Conv2dParams {
    uint in_c;
    uint out_c;
    uint in_h;
    uint in_w;
    uint kH;
    uint kW;
    uint stride;
    uint pad;
    uint groups;
    uint out_h;
    uint out_w;
    uint has_bias;
};

[vk_push_constant] const Conv2dParams conv2d_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> conv2d_weight;  // [kH, kW, C_in, C_out] or F32
[vk_binding(1, 0)] RWStructuredBuffer<float> conv2d_bias;
[vk_binding(2, 0)] RWStructuredBuffer<float> conv2d_input;   // [C_in, H, W]
[vk_binding(3, 0)] RWStructuredBuffer<float> conv2d_output;  // [C_out, out_H, out_W]

[shader("compute")]
[numthreads(256, 1, 1)]
void conv2d(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    // gid.x = output spatial position, gid.y = output channel
    uint oc = gid.y;
    uint spatial_idx = gid.x * 256 + gi;
    uint oh = spatial_idx / conv2d_pc.out_w;
    uint ow = spatial_idx % conv2d_pc.out_w;

    if (oc >= conv2d_pc.out_c || oh >= conv2d_pc.out_h || ow >= conv2d_pc.out_w) return;

    uint group_size_in = conv2d_pc.in_c / conv2d_pc.groups;
    uint group_size_out = conv2d_pc.out_c / conv2d_pc.groups;
    uint group = oc / group_size_out;
    uint ic_start = group * group_size_in;

    float sum = 0.0;

    for (uint ic_local = 0; ic_local < group_size_in; ic_local++) {
        uint ic = ic_start + ic_local;
        for (uint kh = 0; kh < conv2d_pc.kH; kh++) {
            for (uint kw = 0; kw < conv2d_pc.kW; kw++) {
                int ih = int(oh * conv2d_pc.stride + kh) - int(conv2d_pc.pad);
                int iw = int(ow * conv2d_pc.stride + kw) - int(conv2d_pc.pad);

                if (ih >= 0 && ih < int(conv2d_pc.in_h) && iw >= 0 && iw < int(conv2d_pc.in_w)) {
                    float input_val = conv2d_input[ic * conv2d_pc.in_h * conv2d_pc.in_w + uint(ih) * conv2d_pc.in_w + uint(iw)];
                    // Weight layout: [kH, kW, C_in_per_group, C_out]
                    uint w_idx = kh * conv2d_pc.kW * group_size_in * conv2d_pc.out_c
                               + kw * group_size_in * conv2d_pc.out_c
                               + ic_local * conv2d_pc.out_c
                               + oc;
                    sum += input_val * conv2d_weight[w_idx];
                }
            }
        }
    }

    if (conv2d_pc.has_bias != 0) {
        sum += conv2d_bias[oc];
    }

    conv2d_output[oc * conv2d_pc.out_h * conv2d_pc.out_w + oh * conv2d_pc.out_w + ow] = sum;
}

// ============================================================
// Q8_0 Conv2D - Dequantize weights on the fly
// ============================================================

struct Conv2dQ8Params {
    uint in_c;
    uint out_c;
    uint in_h;
    uint in_w;
    uint kH;
    uint kW;
    uint stride;
    uint pad;
    uint groups;
    uint out_h;
    uint out_w;
    uint has_bias;
};

[vk_push_constant] const Conv2dQ8Params conv2d_q8_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> conv2d_q8_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> conv2d_q8_bias;
[vk_binding(2, 0)] RWStructuredBuffer<float> conv2d_q8_input;
[vk_binding(3, 0)] RWStructuredBuffer<float> conv2d_q8_output;

float dequant_q8_element(RWStructuredBuffer<uint> w, uint flat_idx) {
    uint block_idx = flat_idx / 32;
    uint elem_in_block = flat_idx % 32;
    uint block_byte = block_idx * 34;
    // Read f16 scale
    uint scale_uint_idx = block_byte / 4;
    uint scale_byte_offset = block_byte % 4;
    uint scale_word = w[scale_uint_idx];
    uint scale_bits;
    if (scale_byte_offset == 0) {
        scale_bits = scale_word & 0xFFFFu;
    } else if (scale_byte_offset == 2) {
        scale_bits = (scale_word >> 16) & 0xFFFFu;
    } else {
        uint next_word = w[scale_uint_idx + 1];
        scale_bits = ((scale_word >> (scale_byte_offset * 8)) | (next_word << (32 - scale_byte_offset * 8))) & 0xFFFFu;
    }
    float scale = unpack_f16_diff(scale_bits);
    // Read int8 value
    uint qs_byte = block_byte + 2 + elem_in_block;
    int q = sign_extend_i8_diff(read_byte_diff(w, qs_byte));
    return scale * float(q);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void conv2d_q8(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint oc = gid.y;
    uint spatial_idx = gid.x * 256 + gi;
    uint oh = spatial_idx / conv2d_q8_pc.out_w;
    uint ow = spatial_idx % conv2d_q8_pc.out_w;

    if (oc >= conv2d_q8_pc.out_c || oh >= conv2d_q8_pc.out_h || ow >= conv2d_q8_pc.out_w) return;

    uint group_size_in = conv2d_q8_pc.in_c / conv2d_q8_pc.groups;
    uint group_size_out = conv2d_q8_pc.out_c / conv2d_q8_pc.groups;
    uint group = oc / group_size_out;
    uint ic_start = group * group_size_in;

    float sum = 0.0;

    for (uint ic_local = 0; ic_local < group_size_in; ic_local++) {
        uint ic = ic_start + ic_local;
        for (uint kh = 0; kh < conv2d_q8_pc.kH; kh++) {
            for (uint kw = 0; kw < conv2d_q8_pc.kW; kw++) {
                int ih = int(oh * conv2d_q8_pc.stride + kh) - int(conv2d_q8_pc.pad);
                int iw = int(ow * conv2d_q8_pc.stride + kw) - int(conv2d_q8_pc.pad);

                if (ih >= 0 && ih < int(conv2d_q8_pc.in_h) && iw >= 0 && iw < int(conv2d_q8_pc.in_w)) {
                    float input_val = conv2d_q8_input[ic * conv2d_q8_pc.in_h * conv2d_q8_pc.in_w + uint(ih) * conv2d_q8_pc.in_w + uint(iw)];
                    uint w_idx = kh * conv2d_q8_pc.kW * group_size_in * conv2d_q8_pc.out_c
                               + kw * group_size_in * conv2d_q8_pc.out_c
                               + ic_local * conv2d_q8_pc.out_c
                               + oc;
                    sum += input_val * dequant_q8_element(conv2d_q8_weight, w_idx);
                }
            }
        }
    }

    if (conv2d_q8_pc.has_bias != 0) {
        sum += conv2d_q8_bias[oc];
    }

    conv2d_q8_output[oc * conv2d_q8_pc.out_h * conv2d_q8_pc.out_w + oh * conv2d_q8_pc.out_w + ow] = sum;
}

// ============================================================
// Group Normalization
// ============================================================
// GroupNorm: divide channels into groups, normalize within each group
// Input: [C, H, W], Output: [C, H, W]
// One workgroup per group

struct GroupNormParams {
    uint channels;
    uint spatial;   // H * W
    uint num_groups;
    float eps;
};

[vk_push_constant] const GroupNormParams group_norm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> gn_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> gn_weight;  // per-channel scale
[vk_binding(2, 0)] RWStructuredBuffer<float> gn_bias;    // per-channel bias
[vk_binding(3, 0)] RWStructuredBuffer<float> gn_output;

groupshared float gn_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void group_norm(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint group = gid.x;
    uint num_threads = 256;

    if (group >= group_norm_pc.num_groups) return;

    uint channels_per_group = group_norm_pc.channels / group_norm_pc.num_groups;
    uint c_start = group * channels_per_group;
    uint n_elements = channels_per_group * group_norm_pc.spatial;

    // Compute mean
    float sum = 0.0;
    for (uint i = gi; i < n_elements; i += num_threads) {
        uint c = c_start + i / group_norm_pc.spatial;
        uint s = i % group_norm_pc.spatial;
        sum += gn_input[c * group_norm_pc.spatial + s];
    }
    gn_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) gn_shared[gi] += gn_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = gn_shared[0] / float(n_elements);
    GroupMemoryBarrierWithGroupSync();

    // Compute variance
    float var_sum = 0.0;
    for (uint i = gi; i < n_elements; i += num_threads) {
        uint c = c_start + i / group_norm_pc.spatial;
        uint sp = i % group_norm_pc.spatial;
        float diff = gn_input[c * group_norm_pc.spatial + sp] - mean;
        var_sum += diff * diff;
    }
    gn_shared[gi] = var_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) gn_shared[gi] += gn_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_std = 1.0 / sqrt(gn_shared[0] / float(n_elements) + group_norm_pc.eps);

    // Normalize with per-channel weight and bias
    for (uint i = gi; i < n_elements; i += num_threads) {
        uint c = c_start + i / group_norm_pc.spatial;
        uint sp = i % group_norm_pc.spatial;
        uint idx = c * group_norm_pc.spatial + sp;
        gn_output[idx] = (gn_input[idx] - mean) * inv_std * gn_weight[c] + gn_bias[c];
    }
}

// ============================================================
// Batched Matrix Multiply (F32)
// ============================================================
// C[M,N] = A[M,K] * B[K,N]
// Row-major, one workgroup per 16x16 output tile

struct BatchedMatMulParams {
    uint M;
    uint N;
    uint K;
};

[vk_push_constant] const BatchedMatMulParams bmm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bmm_A;
[vk_binding(1, 0)] RWStructuredBuffer<float> bmm_B;
[vk_binding(2, 0)] RWStructuredBuffer<float> bmm_output;

groupshared float bmm_As[16][16];
groupshared float bmm_Bs[16][16];

[shader("compute")]
[numthreads(16, 16, 1)]
void batched_matmul(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID) {
    uint row = gid.x * 16 + lid.x;
    uint col = gid.y * 16 + lid.y;

    float sum = 0.0;

    uint num_tiles = (bmm_pc.K + 15) / 16;
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * 16 + lid.y;
        uint b_row = t * 16 + lid.x;

        bmm_As[lid.x][lid.y] = (row < bmm_pc.M && a_col < bmm_pc.K) ? bmm_A[row * bmm_pc.K + a_col] : 0.0;
        bmm_Bs[lid.x][lid.y] = (b_row < bmm_pc.K && col < bmm_pc.N) ? bmm_B[b_row * bmm_pc.N + col] : 0.0;

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < 16; k++) {
            sum += bmm_As[lid.x][k] * bmm_Bs[k][lid.y];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (row < bmm_pc.M && col < bmm_pc.N) {
        bmm_output[row * bmm_pc.N + col] = sum;
    }
}

// ============================================================
// Q8_0 Batched Matrix Multiply
// ============================================================
// C[M,N] = A[M,K] * B_q8[K,N]
// A is F32, B is Q8_0 quantized

struct BatchedMatMulQ8Params {
    uint M;
    uint N;
    uint K;
};

[vk_push_constant] const BatchedMatMulQ8Params bmm_q8_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bmm_q8_A;
[vk_binding(1, 0)] RWStructuredBuffer<uint> bmm_q8_B;   // Q8_0 quantized
[vk_binding(2, 0)] RWStructuredBuffer<float> bmm_q8_output;

groupshared float bmm_q8_As[16][16];

[shader("compute")]
[numthreads(16, 16, 1)]
void batched_matmul_q8(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID) {
    uint row = gid.x * 16 + lid.x;
    uint col = gid.y * 16 + lid.y;

    if (row >= bmm_q8_pc.M || col >= bmm_q8_pc.N) return;

    // B is stored row-major as [K, N] in Q8_0
    // Each row of B has K elements; blocks_per_row = K/32
    // But we need B[k, col], so we access the weight column by column
    // Instead: accumulate directly without tiling B
    float sum = 0.0;

    // Accumulate A[row, k] * B_dequant[k, col]
    // B is row-major [K, N], so B[k, col] is at flat index k*N + col
    // In Q8_0: row k of length N has blocks_per_row = N/32, bytes_per_row = blocks_per_row * 34
    // Element at position col in row k: block = col/32, elem_in_block = col%32
    // But B is stored as [K, N] in Q8_0, meaning the K dimension is the "row"
    // Flat element index = k * N + col

    for (uint k = 0; k < bmm_q8_pc.K; k++) {
        float a_val = bmm_q8_A[row * bmm_q8_pc.K + k];
        float b_val = dequant_q8_element(bmm_q8_B, k * bmm_q8_pc.N + col);
        sum += a_val * b_val;
    }

    bmm_q8_output[row * bmm_q8_pc.N + col] = sum;
}

// ============================================================
// Non-causal Self Attention (for CLIP and spatial transformers)
// ============================================================
// Q, K, V: [n_heads, seq_len, head_dim]
// No causal mask, no KV cache

struct SpatialAttentionParams {
    uint head_dim;
    uint n_heads;
    uint seq_len;
    float scale;
};

[vk_push_constant] const SpatialAttentionParams spatial_attn_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> sa_Q;
[vk_binding(1, 0)] RWStructuredBuffer<float> sa_K;
[vk_binding(2, 0)] RWStructuredBuffer<float> sa_V;
[vk_binding(3, 0)] RWStructuredBuffer<float> sa_scores;
[vk_binding(4, 0)] RWStructuredBuffer<float> sa_output;

groupshared float sa_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void spatial_attention(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    // One workgroup per (head, query_position) pair
    uint head = gid.x / spatial_attn_pc.seq_len;
    uint q_pos = gid.x % spatial_attn_pc.seq_len;
    uint num_threads = 256;

    if (head >= spatial_attn_pc.n_heads) return;

    uint head_offset = head * spatial_attn_pc.seq_len * spatial_attn_pc.head_dim;
    uint scores_base = head * spatial_attn_pc.seq_len * spatial_attn_pc.seq_len + q_pos * spatial_attn_pc.seq_len;

    // Phase 1: Compute Q[q_pos] * K[kv_pos]^T for all kv_pos
    for (uint kv_pos = gi; kv_pos < spatial_attn_pc.seq_len; kv_pos += num_threads) {
        float dot_val = 0.0;
        for (uint d = 0; d < spatial_attn_pc.head_dim; d++) {
            dot_val += sa_Q[head_offset + q_pos * spatial_attn_pc.head_dim + d] *
                       sa_K[head_offset + kv_pos * spatial_attn_pc.head_dim + d];
        }
        sa_scores[scores_base + kv_pos] = dot_val * spatial_attn_pc.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Softmax over scores (no causal mask)
    float max_val = -1.0e30;
    for (uint i = gi; i < spatial_attn_pc.seq_len; i += num_threads) {
        max_val = max(max_val, sa_scores[scores_base + i]);
    }
    sa_shared[gi] = max_val;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) sa_shared[gi] = max(sa_shared[gi], sa_shared[gi + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float global_max = sa_shared[0];
    GroupMemoryBarrierWithGroupSync();

    float sum = 0.0;
    for (uint i = gi; i < spatial_attn_pc.seq_len; i += num_threads) {
        float val = exp(sa_scores[scores_base + i] - global_max);
        sa_scores[scores_base + i] = val;
        sum += val;
    }
    sa_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) sa_shared[gi] += sa_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float global_sum = sa_shared[0];
    GroupMemoryBarrierWithGroupSync();

    for (uint i = gi; i < spatial_attn_pc.seq_len; i += num_threads) {
        sa_scores[scores_base + i] /= global_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: Weighted sum of V
    uint out_base = head_offset + q_pos * spatial_attn_pc.head_dim;
    for (uint d = gi; d < spatial_attn_pc.head_dim; d += num_threads) {
        float acc = 0.0;
        for (uint pos = 0; pos < spatial_attn_pc.seq_len; pos++) {
            acc += sa_scores[scores_base + pos] *
                   sa_V[head_offset + pos * spatial_attn_pc.head_dim + d];
        }
        sa_output[out_base + d] = acc;
    }
}

// ============================================================
// Cross Attention (text -> image)
// ============================================================
// Q from image [n_heads, q_len, head_dim]
// K, V from text [n_heads, kv_len, head_dim]

struct CrossAttentionParams {
    uint head_dim;
    uint n_heads;
    uint q_len;
    uint kv_len;
    float scale;
};

[vk_push_constant] const CrossAttentionParams cross_attn_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> ca_Q;
[vk_binding(1, 0)] RWStructuredBuffer<float> ca_K;
[vk_binding(2, 0)] RWStructuredBuffer<float> ca_V;
[vk_binding(3, 0)] RWStructuredBuffer<float> ca_scores;
[vk_binding(4, 0)] RWStructuredBuffer<float> ca_output;

groupshared float ca_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cross_attention(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    // One workgroup per (head, query_position) pair
    uint head = gid.x / cross_attn_pc.q_len;
    uint q_pos = gid.x % cross_attn_pc.q_len;
    uint num_threads = 256;

    if (head >= cross_attn_pc.n_heads) return;

    uint q_head_offset = head * cross_attn_pc.q_len * cross_attn_pc.head_dim;
    uint kv_head_offset = head * cross_attn_pc.kv_len * cross_attn_pc.head_dim;
    uint scores_base = head * cross_attn_pc.q_len * cross_attn_pc.kv_len + q_pos * cross_attn_pc.kv_len;

    // Phase 1: Q[q_pos] * K^T
    for (uint kv_pos = gi; kv_pos < cross_attn_pc.kv_len; kv_pos += num_threads) {
        float dot_val = 0.0;
        for (uint d = 0; d < cross_attn_pc.head_dim; d++) {
            dot_val += ca_Q[q_head_offset + q_pos * cross_attn_pc.head_dim + d] *
                       ca_K[kv_head_offset + kv_pos * cross_attn_pc.head_dim + d];
        }
        ca_scores[scores_base + kv_pos] = dot_val * cross_attn_pc.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Softmax
    float max_val = -1.0e30;
    for (uint i = gi; i < cross_attn_pc.kv_len; i += num_threads) {
        max_val = max(max_val, ca_scores[scores_base + i]);
    }
    ca_shared[gi] = max_val;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) ca_shared[gi] = max(ca_shared[gi], ca_shared[gi + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float global_max = ca_shared[0];
    GroupMemoryBarrierWithGroupSync();

    float sum = 0.0;
    for (uint i = gi; i < cross_attn_pc.kv_len; i += num_threads) {
        float val = exp(ca_scores[scores_base + i] - global_max);
        ca_scores[scores_base + i] = val;
        sum += val;
    }
    ca_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) ca_shared[gi] += ca_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float global_sum = ca_shared[0];
    GroupMemoryBarrierWithGroupSync();

    for (uint i = gi; i < cross_attn_pc.kv_len; i += num_threads) {
        ca_scores[scores_base + i] /= global_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: Weighted sum of V
    uint out_base = q_head_offset + q_pos * cross_attn_pc.head_dim;
    for (uint d = gi; d < cross_attn_pc.head_dim; d += num_threads) {
        float acc = 0.0;
        for (uint pos = 0; pos < cross_attn_pc.kv_len; pos++) {
            acc += ca_scores[scores_base + pos] *
                   ca_V[kv_head_offset + pos * cross_attn_pc.head_dim + d];
        }
        ca_output[out_base + d] = acc;
    }
}

// ============================================================
// Upsample Nearest 2x
// ============================================================
// Input: [C, H, W] -> Output: [C, 2H, 2W]

struct UpsampleParams {
    uint channels;
    uint in_h;
    uint in_w;
};

[vk_push_constant] const UpsampleParams upsample_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> upsample_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> upsample_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void upsample_nearest(uint3 tid: SV_DispatchThreadID) {
    uint out_h = upsample_pc.in_h * 2;
    uint out_w = upsample_pc.in_w * 2;
    uint total = upsample_pc.channels * out_h * out_w;

    if (tid.x >= total) return;

    uint c = tid.x / (out_h * out_w);
    uint rem = tid.x % (out_h * out_w);
    uint oh = rem / out_w;
    uint ow = rem % out_w;

    uint ih = oh / 2;
    uint iw = ow / 2;

    upsample_output[tid.x] = upsample_input[c * upsample_pc.in_h * upsample_pc.in_w + ih * upsample_pc.in_w + iw];
}

// ============================================================
// Timestep Embedding (sinusoidal)
// ============================================================
// Produces [dim] embedding from a single timestep scalar

struct TimestepEmbedParams {
    uint dim;
    float timestep;
    float max_period;
};

[vk_push_constant] const TimestepEmbedParams timestep_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> timestep_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void timestep_embed(uint3 tid: SV_DispatchThreadID) {
    uint half_dim = timestep_pc.dim / 2;
    if (tid.x >= timestep_pc.dim) return;

    if (tid.x < half_dim) {
        float freq = exp(-log(timestep_pc.max_period) * float(tid.x) / float(half_dim));
        timestep_output[tid.x] = cos(timestep_pc.timestep * freq);
    } else {
        uint idx = tid.x - half_dim;
        float freq = exp(-log(timestep_pc.max_period) * float(idx) / float(half_dim));
        timestep_output[tid.x] = sin(timestep_pc.timestep * freq);
    }
}

// ============================================================
// Broadcast Add (add embedding across spatial dimensions)
// ============================================================
// hidden[C, H*W] += embedding[C]
// In-place on hidden

struct BroadcastAddParams {
    uint channels;
    uint spatial;   // H * W
};

[vk_push_constant] const BroadcastAddParams broadcast_add_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> ba_hidden;
[vk_binding(1, 0)] RWStructuredBuffer<float> ba_embedding;

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_add(uint3 tid: SV_DispatchThreadID) {
    uint total = broadcast_add_pc.channels * broadcast_add_pc.spatial;
    if (tid.x >= total) return;

    uint c = tid.x / broadcast_add_pc.spatial;
    ba_hidden[tid.x] += ba_embedding[c];
}

// ============================================================
// Channel Concatenation (for UNet skip connections)
// ============================================================
// A[C_a, spatial] + B[C_b, spatial] -> output[C_a+C_b, spatial]

struct ChannelConcatParams {
    uint channels_a;
    uint channels_b;
    uint spatial;
};

[vk_push_constant] const ChannelConcatParams concat_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> concat_A;
[vk_binding(1, 0)] RWStructuredBuffer<float> concat_B;
[vk_binding(2, 0)] RWStructuredBuffer<float> concat_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void channel_concat(uint3 tid: SV_DispatchThreadID) {
    uint total_c = concat_pc.channels_a + concat_pc.channels_b;
    uint total = total_c * concat_pc.spatial;
    if (tid.x >= total) return;

    uint c = tid.x / concat_pc.spatial;
    uint s = tid.x % concat_pc.spatial;

    if (c < concat_pc.channels_a) {
        concat_output[tid.x] = concat_A[c * concat_pc.spatial + s];
    } else {
        uint cb = c - concat_pc.channels_a;
        concat_output[tid.x] = concat_B[cb * concat_pc.spatial + s];
    }
}

// ============================================================
// DDIM Denoising Step
// ============================================================
// x_prev = sqrt(alpha_prev) * predicted_x0 + sqrt(1 - alpha_prev) * predicted_dir
// predicted_x0 = (noisy - sqrt(1-alpha_t) * predicted_noise) / sqrt(alpha_t)
// predicted_dir = sqrt(1-alpha_prev-sigma^2) * predicted_noise

struct DdimStepParams {
    uint n;
    float sqrt_alpha_t;
    float sqrt_one_minus_alpha_t;
    float sqrt_alpha_prev;
    float sqrt_one_minus_alpha_prev;
};

[vk_push_constant] const DdimStepParams ddim_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> ddim_noisy;
[vk_binding(1, 0)] RWStructuredBuffer<float> ddim_predicted_noise;

[shader("compute")]
[numthreads(256, 1, 1)]
void ddim_step(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= ddim_pc.n) return;

    float x = ddim_noisy[tid.x];
    float eps = ddim_predicted_noise[tid.x];

    // Predict x0
    float x0 = (x - ddim_pc.sqrt_one_minus_alpha_t * eps) / ddim_pc.sqrt_alpha_t;

    // Predict direction
    float dir = ddim_pc.sqrt_one_minus_alpha_prev * eps;

    // Update
    ddim_noisy[tid.x] = ddim_pc.sqrt_alpha_prev * x0 + dir;
}

// ============================================================
// Euler Denoising Step
// ============================================================
// x_next = x + (sigma_next - sigma) * predicted_noise

struct EulerStepParams {
    uint n;
    float dt;  // sigma_next - sigma
};

[vk_push_constant] const EulerStepParams euler_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> euler_noisy;
[vk_binding(1, 0)] RWStructuredBuffer<float> euler_predicted_noise;

[shader("compute")]
[numthreads(256, 1, 1)]
void euler_step(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= euler_pc.n) return;
    euler_noisy[tid.x] += euler_pc.dt * euler_predicted_noise[tid.x];
}

// ============================================================
// Scale-Shift-Clamp (VAE output post-processing)
// ============================================================
// output = clamp(input * scale + shift, 0, 1)

struct ScaleShiftParams {
    uint n;
    float scale;
    float shift;
};

[vk_push_constant] const ScaleShiftParams scale_shift_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> ss_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> ss_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void scale_shift_clamp(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= scale_shift_pc.n) return;
    float val = ss_input[tid.x] * scale_shift_pc.scale + scale_shift_pc.shift;
    ss_output[tid.x] = clamp(val, 0.0, 1.0);
}

// ============================================================
// ReLU (in-place element-wise)
// ============================================================

struct ReluParams {
    uint n;
};

[vk_push_constant] const ReluParams relu_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> relu_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void relu(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= relu_pc.n) return;
    relu_data[tid.x] = max(relu_data[tid.x], 0.0);
}

// ============================================================
// Tanh Clamp (in-place): x = tanh(x / 3.0) * 3.0
// ============================================================

struct TanhClampParams {
    uint n;
};

[vk_push_constant] const TanhClampParams tanh_clamp_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> tanh_clamp_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void tanh_clamp(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= tanh_clamp_pc.n) return;
    tanh_clamp_data[tid.x] = tanh(tanh_clamp_data[tid.x] / 3.0) * 3.0;
}

// ============================================================
// F16 to F32 conversion (for loading F16 bias/weights)
// ============================================================

struct F16ToF32Params {
    uint n;
};

[vk_push_constant] const F16ToF32Params f16_to_f32_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> f16_input;   // packed F16 as uint16 in uint32
[vk_binding(1, 0)] RWStructuredBuffer<float> f32_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void f16_to_f32(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= f16_to_f32_pc.n) return;
    // Each uint32 contains 2 f16 values
    uint word_idx = tid.x / 2;
    uint half_idx = tid.x % 2;
    uint word = f16_input[word_idx];
    uint bits = (half_idx == 0) ? (word & 0xFFFFu) : ((word >> 16) & 0xFFFFu);
    f32_output[tid.x] = unpack_f16_diff(bits);
}

// ============================================================
// Shared kernels (duplicated from llm.slang for independence)
// These are used by both LLM and diffusion pipelines
// ============================================================

// Helper for Q4_K and Q5_K scale/min extraction
void get_scale_min_k4_diff(uint j, uint block_byte, RWStructuredBuffer<uint> w, out uint sc_val, out uint m_val) {
    uint sc_base = block_byte + 4;
    if (j < 4) {
        uint byte_j = read_byte_diff(w, sc_base + j);
        uint byte_j4 = read_byte_diff(w, sc_base + j + 4);
        sc_val = byte_j & 63u;
        m_val = byte_j4 & 63u;
    } else {
        uint byte_j4 = read_byte_diff(w, sc_base + j + 4);
        uint byte_jm4 = read_byte_diff(w, sc_base + j - 4);
        uint byte_j0 = read_byte_diff(w, sc_base + j);
        sc_val = (byte_j4 & 0xFu) | ((byte_jm4 >> 6) << 4);
        m_val = (byte_j4 >> 4) | ((byte_j0 >> 6) << 4);
    }
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
    float c = 0.7978845608; // sqrt(2/pi)
    gelu_data[tid.x] = 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

// ============================================================
// Matrix Multiply (F32)
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
// Matrix Multiply Q8_0
// ============================================================

[vk_push_constant] const MatMulParams matmul_q8_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> q8_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> q8_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> q8_output;

groupshared float q8_shared[256];

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
        float scale = unpack_f16_diff(scale_bits);

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

            int q0 = sign_extend_i8_diff(word & 0xFFu);
            int q1 = sign_extend_i8_diff((word >> 8) & 0xFFu);
            int q2 = sign_extend_i8_diff((word >> 16) & 0xFFu);
            int q3 = sign_extend_i8_diff((word >> 24) & 0xFFu);

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
// Matrix Multiply Q4_0
// ============================================================

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
        float scale = unpack_f16_diff(scale_bits);

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

            int q0_lo = int(word & 0xFu) - 8;
            int q1_lo = int((word >> 8) & 0xFu) - 8;
            int q2_lo = int((word >> 16) & 0xFu) - 8;
            int q3_lo = int((word >> 24) & 0xFu) - 8;

            sum += scale * float(q0_lo) * q4_0_input[input_offset + j];
            sum += scale * float(q1_lo) * q4_0_input[input_offset + j + 1];
            sum += scale * float(q2_lo) * q4_0_input[input_offset + j + 2];
            sum += scale * float(q3_lo) * q4_0_input[input_offset + j + 3];

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
// Matrix Multiply Q4_K
// ============================================================

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

        uint dm_word = q4k_weight[bb / 4];
        float d = unpack_f16_diff(dm_word & 0xFFFFu);
        float dmin = unpack_f16_diff((dm_word >> 16) & 0xFFFFu);

        uint qs_off = bb + 16;

        for (uint j = 0; j < 4; j++) {
            uint sc1, m1, sc2, m2;
            get_scale_min_k4_diff(2 * j, bb, q4k_weight, sc1, m1);
            get_scale_min_k4_diff(2 * j + 1, bb, q4k_weight, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint qs_base = qs_off + j * 32;

            for (uint l = 0; l < 32; l += 4) {
                uint qs_word = q4k_weight[(qs_base + l) / 4];

                uint b0 = qs_word & 0xFFu;
                uint b1 = (qs_word >> 8) & 0xFFu;
                uint b2 = (qs_word >> 16) & 0xFFu;
                uint b3 = (qs_word >> 24) & 0xFFu;

                sum += (d1 * float(b0 & 0xFu) - dm1) * q4k_input[input_base + j * 64 + l];
                sum += (d1 * float(b1 & 0xFu) - dm1) * q4k_input[input_base + j * 64 + l + 1];
                sum += (d1 * float(b2 & 0xFu) - dm1) * q4k_input[input_base + j * 64 + l + 2];
                sum += (d1 * float(b3 & 0xFu) - dm1) * q4k_input[input_base + j * 64 + l + 3];

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
// Matrix Multiply Q5_K
// ============================================================

[vk_push_constant] const MatMulParams matmul_q5k_pc;

[vk_binding(0, 0)] RWStructuredBuffer<uint> q5k_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> q5k_input;
[vk_binding(2, 0)] RWStructuredBuffer<float> q5k_output;

groupshared float q5k_shared[256];

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

        uint dm_word = q5k_weight[bb / 4];
        float d = unpack_f16_diff(dm_word & 0xFFFFu);
        float dmin = unpack_f16_diff((dm_word >> 16) & 0xFFFFu);

        for (uint j = 0; j < 4; j++) {
            uint sc1, m1, sc2, m2;
            get_scale_min_k4_diff(2 * j, bb, q5k_weight, sc1, m1);
            get_scale_min_k4_diff(2 * j + 1, bb, q5k_weight, sc2, m2);

            float d1 = d * float(sc1);
            float dm1 = dmin * float(m1);
            float d2 = d * float(sc2);
            float dm2 = dmin * float(m2);

            uint qs_base = bb + 48 + j * 32;
            uint qh_base = bb + 16;
            uint qh_shift_lo = 2 * j;
            uint qh_shift_hi = 2 * j + 1;

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

                sum += (d1 * float((ql0 & 0xFu) + (((qh0 >> qh_shift_lo) & 1u) << 4)) - dm1) * q5k_input[input_base + j * 64 + l];
                sum += (d1 * float((ql1 & 0xFu) + (((qh1 >> qh_shift_lo) & 1u) << 4)) - dm1) * q5k_input[input_base + j * 64 + l + 1];
                sum += (d1 * float((ql2 & 0xFu) + (((qh2 >> qh_shift_lo) & 1u) << 4)) - dm1) * q5k_input[input_base + j * 64 + l + 2];
                sum += (d1 * float((ql3 & 0xFu) + (((qh3 >> qh_shift_lo) & 1u) << 4)) - dm1) * q5k_input[input_base + j * 64 + l + 3];

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
// Matrix Multiply Q6_K
// ============================================================

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

        uint d_byte = bb + 208;
        uint d_word = q6k_weight[d_byte / 4];
        uint d_byte_off = d_byte % 4;
        uint d_bits;
        if (d_byte_off == 0) {
            d_bits = d_word & 0xFFFFu;
        } else {
            d_bits = (d_word >> 16) & 0xFFFFu;
        }
        float d = unpack_f16_diff(d_bits);

        uint ql_base = bb;
        uint qh_base = bb + 128;
        uint sc_base = bb + 192;

        for (uint half_idx = 0; half_idx < 2; half_idx++) {
            uint ql_off = ql_base + half_idx * 64;
            uint qh_off = qh_base + half_idx * 32;
            uint sc_off = half_idx * 8;
            uint elem_off = half_idx * 128;

            int sc_0_0 = sign_extend_i8_diff(read_byte_diff(q6k_weight, sc_base + sc_off + 0));
            int sc_0_2 = sign_extend_i8_diff(read_byte_diff(q6k_weight, sc_base + sc_off + 2));
            int sc_0_4 = sign_extend_i8_diff(read_byte_diff(q6k_weight, sc_base + sc_off + 4));
            int sc_0_6 = sign_extend_i8_diff(read_byte_diff(q6k_weight, sc_base + sc_off + 6));
            int sc_1_0 = sign_extend_i8_diff(read_byte_diff(q6k_weight, sc_base + sc_off + 1));
            int sc_1_2 = sign_extend_i8_diff(read_byte_diff(q6k_weight, sc_base + sc_off + 3));
            int sc_1_4 = sign_extend_i8_diff(read_byte_diff(q6k_weight, sc_base + sc_off + 5));
            int sc_1_6 = sign_extend_i8_diff(read_byte_diff(q6k_weight, sc_base + sc_off + 7));

            for (uint l = 0; l < 32; l += 4) {
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

    for (uint i = gi; i < softmax_pc.n; i += num_threads) {
        softmax_data[softmax_pc.offset + i] /= global_sum;
    }
}

// ============================================================
// Residual Add (in-place: acc += src)
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
// Element-wise Multiply (in-place: a *= b)
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
