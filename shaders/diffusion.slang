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
