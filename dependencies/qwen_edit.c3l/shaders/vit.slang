// ============================================================
// Vision Transformer (ViT) Compute Shaders
// For Qwen2.5-VL mmproj vision encoder
// ============================================================

// ============================================================
// Batch LayerNorm with Affine (weight only, no bias)
// ============================================================
// One workgroup per row. Input [n_rows, dim] → Output [n_rows, dim]
// output[row, i] = weight[i] * (input[row, i] - mean) / sqrt(var + eps)

struct BatchLayerNormAffineParams {
    uint dim;
    float eps;
    uint n_rows;
    uint _pad;
};

[vk_push_constant] const BatchLayerNormAffineParams blna_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> blna_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> blna_weight;
[vk_binding(2, 0)] RWStructuredBuffer<float> blna_output;

groupshared float blna_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_layernorm_affine(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    if (row >= blna_pc.n_rows) return;

    uint base = row * blna_pc.dim;

    // Pass 1: Compute mean
    float sum = 0.0;
    for (uint i = gi; i < blna_pc.dim; i += 256) {
        sum += blna_input[base + i];
    }
    blna_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) blna_shared[gi] += blna_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = blna_shared[0] / float(blna_pc.dim);

    // Pass 2: Compute variance
    float var_sum = 0.0;
    for (uint i = gi; i < blna_pc.dim; i += 256) {
        float diff = blna_input[base + i] - mean;
        var_sum += diff * diff;
    }
    blna_shared[gi] = var_sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) blna_shared[gi] += blna_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_std = 1.0 / sqrt(blna_shared[0] / float(blna_pc.dim) + blna_pc.eps);

    // Normalize with affine weight
    for (uint i = gi; i < blna_pc.dim; i += 256) {
        blna_output[base + i] = (blna_input[base + i] - mean) * inv_std * blna_weight[i];
    }
}

// ============================================================
// Batch RMSNorm with Affine
// ============================================================
// One workgroup per row. Input [n_rows, dim] → Output [n_rows, dim]
// output[row, i] = weight[i] * input[row, i] / sqrt(mean(input^2) + eps)

struct BatchRMSNormAffineParams {
    uint dim;
    float eps;
    uint n_rows;
    uint _pad;
};

[vk_push_constant] const BatchRMSNormAffineParams brna_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> brna_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> brna_weight;
[vk_binding(2, 0)] RWStructuredBuffer<float> brna_output;

groupshared float brna_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_rmsnorm_affine(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    if (row >= brna_pc.n_rows) return;

    uint base = row * brna_pc.dim;

    float sum_sq = 0.0;
    for (uint i = gi; i < brna_pc.dim; i += 256) {
        float val = brna_input[base + i];
        sum_sq += val * val;
    }
    brna_shared[gi] = sum_sq;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) brna_shared[gi] += brna_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float rms = 1.0 / sqrt(brna_shared[0] / float(brna_pc.dim) + brna_pc.eps);

    for (uint i = gi; i < brna_pc.dim; i += 256) {
        brna_output[base + i] = brna_input[base + i] * rms * brna_weight[i];
    }
}

// ============================================================
// Broadcast Bias Add
// ============================================================
// output[row, col] += bias[col] for all rows
// 2D dispatch: (n_rows, ceil(dim/256))

struct BroadcastBiasParams {
    uint dim;
    uint n_rows;
};

[vk_push_constant] const BroadcastBiasParams bb_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bb_data;
[vk_binding(1, 0)] RWStructuredBuffer<float> bb_bias;

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_bias_add(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    uint col = gid.y * 256 + gi;
    if (row >= bb_pc.n_rows || col >= bb_pc.dim) return;
    bb_data[row * bb_pc.dim + col] += bb_bias[col];
}

// ============================================================
// Image Normalize (channel-wise mean/std)
// ============================================================
// Applies: pixel = (pixel - mean[c]) / std[c]  per channel NCHW
// Simple element-wise, 1D dispatch over total elements

struct ImageNormParams {
    uint height;
    uint width;
    float mean_r; float mean_g; float mean_b;
    float std_r;  float std_g;  float std_b;
};

[vk_push_constant] const ImageNormParams inorm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> inorm_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void image_normalize(uint3 tid: SV_DispatchThreadID) {
    uint spatial = inorm_pc.height * inorm_pc.width;
    uint total = 3 * spatial;
    if (tid.x >= total) return;

    uint c = tid.x / spatial;
    float mean_val = (c == 0) ? inorm_pc.mean_r : (c == 1) ? inorm_pc.mean_g : inorm_pc.mean_b;
    float std_val  = (c == 0) ? inorm_pc.std_r  : (c == 1) ? inorm_pc.std_g  : inorm_pc.std_b;

    inorm_data[tid.x] = (inorm_data[tid.x] - mean_val) / std_val;
}

// ============================================================
// Patch Embedding (Conv2D with stride = patch_size)
// ============================================================
// Input: [3, H, W] normalized float image
// Weight: [out_channels, 3, patch_h, patch_w] stored as F32
// Output: [n_patches, out_channels]
// Dispatch: one workgroup per (patch_index, output_channel_block)
// Each workgroup computes one output element: dot(weight[oc], input_patch)

struct PatchEmbedParams {
    uint height;
    uint width;
    uint patch_size;
    uint out_channels;
    uint in_channels;  // always 3
    uint patches_h;    // height / patch_size
    uint patches_w;    // width / patch_size
    uint _pad;
};

[vk_push_constant] const PatchEmbedParams pe_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> pe_image;   // [3, H, W]
[vk_binding(1, 0)] RWStructuredBuffer<float> pe_weight;  // [out_ch, 3, patch_h, patch_w]
[vk_binding(2, 0)] RWStructuredBuffer<float> pe_output;  // [n_patches, out_ch]

groupshared float pe_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void patch_embed_conv2d(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    // gid.x = patch_index, gid.y = output_channel
    uint patch_idx = gid.x;
    uint oc = gid.y;
    uint n_patches = pe_pc.patches_h * pe_pc.patches_w;

    if (patch_idx >= n_patches || oc >= pe_pc.out_channels) return;

    uint patch_row = patch_idx / pe_pc.patches_w;
    uint patch_col = patch_idx % pe_pc.patches_w;
    uint y0 = patch_row * pe_pc.patch_size;
    uint x0 = patch_col * pe_pc.patch_size;

    uint kernel_size = pe_pc.in_channels * pe_pc.patch_size * pe_pc.patch_size; // 3*14*14 = 588

    // Each thread accumulates partial dot product
    float partial = 0.0;
    for (uint k = gi; k < kernel_size; k += 256) {
        uint c = k / (pe_pc.patch_size * pe_pc.patch_size);
        uint rem = k % (pe_pc.patch_size * pe_pc.patch_size);
        uint py = rem / pe_pc.patch_size;
        uint px = rem % pe_pc.patch_size;

        float img_val = pe_image[c * pe_pc.height * pe_pc.width + (y0 + py) * pe_pc.width + (x0 + px)];
        float w_val = pe_weight[oc * kernel_size + k];
        partial += img_val * w_val;
    }

    // Reduction
    pe_shared[gi] = partial;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) pe_shared[gi] += pe_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0) {
        pe_output[patch_idx * pe_pc.out_channels + oc] = pe_shared[0];
    }
}

// ============================================================
// 2D Rotary Position Embeddings (RoPE)
// ============================================================
// Applied to Q and K tensors in [n_heads, seq_len, head_dim] layout
// head_dim is split in half: first half uses height positions, second half uses width positions
// Each pair (2i, 2i+1) gets rotated by angle = pos * freq[i]

struct Rope2DParams {
    uint seq_len;
    uint n_heads;
    uint head_dim;
    uint half_dim;      // head_dim / 2
    uint patches_w;     // width in patches (for computing col position)
    float theta;        // rope base frequency (10000.0)
    uint _pad0;
    uint _pad1;
};

[vk_push_constant] const Rope2DParams rope2d_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> rope2d_qk;  // [n_heads, seq_len, head_dim]

[shader("compute")]
[numthreads(256, 1, 1)]
void rope_2d(uint3 tid: SV_DispatchThreadID) {
    // Each thread handles one (head, position, pair) triple
    uint pair_idx = tid.x;
    uint total_pairs = rope2d_pc.n_heads * rope2d_pc.seq_len * rope2d_pc.half_dim;
    if (pair_idx >= total_pairs) return;

    uint pairs_per_head = rope2d_pc.seq_len * rope2d_pc.half_dim;
    uint head = pair_idx / pairs_per_head;
    uint rem = pair_idx % pairs_per_head;
    uint pos = rem / rope2d_pc.half_dim;
    uint pair = rem % rope2d_pc.half_dim;

    // Determine spatial position
    uint row_pos = pos / rope2d_pc.patches_w;
    uint col_pos = pos % rope2d_pc.patches_w;

    // First quarter of half_dim uses height, second quarter uses width
    uint quarter = rope2d_pc.half_dim / 2;
    float position;
    uint dim_idx;
    if (pair < quarter) {
        position = float(row_pos);
        dim_idx = pair;
    } else {
        position = float(col_pos);
        dim_idx = pair - quarter;
    }

    float freq = 1.0 / pow(rope2d_pc.theta, float(2 * dim_idx) / float(quarter * 2));
    float angle = position * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    uint base = head * rope2d_pc.seq_len * rope2d_pc.head_dim + pos * rope2d_pc.head_dim;
    uint idx0 = base + pair * 2;
    uint idx1 = base + pair * 2 + 1;

    float x0 = rope2d_qk[idx0];
    float x1 = rope2d_qk[idx1];
    rope2d_qk[idx0] = x0 * cos_val - x1 * sin_val;
    rope2d_qk[idx1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================
// Transpose Heads: [seq_len, dim] ↔ [n_heads, seq_len, head_dim]
// ============================================================
// direction=0: [seq, dim] → [heads, seq, head_dim]  (forward)
// direction=1: [heads, seq, head_dim] → [seq, dim]  (inverse)

struct TransposeHeadsParams {
    uint seq_len;
    uint n_heads;
    uint head_dim;
    uint direction;
};

[vk_push_constant] const TransposeHeadsParams th_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> th_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> th_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void transpose_heads(uint3 tid: SV_DispatchThreadID) {
    uint total = th_pc.seq_len * th_pc.n_heads * th_pc.head_dim;
    if (tid.x >= total) return;

    uint dim = th_pc.n_heads * th_pc.head_dim;

    if (th_pc.direction == 0) {
        // [seq, dim] → [heads, seq, head_dim]
        uint seq_pos = tid.x / dim;
        uint d = tid.x % dim;
        uint head = d / th_pc.head_dim;
        uint hd = d % th_pc.head_dim;

        uint dst = head * th_pc.seq_len * th_pc.head_dim + seq_pos * th_pc.head_dim + hd;
        th_output[dst] = th_input[tid.x];
    } else {
        // [heads, seq, head_dim] → [seq, dim]
        uint head = tid.x / (th_pc.seq_len * th_pc.head_dim);
        uint rem = tid.x % (th_pc.seq_len * th_pc.head_dim);
        uint seq_pos = rem / th_pc.head_dim;
        uint hd = rem % th_pc.head_dim;

        uint dst = seq_pos * dim + head * th_pc.head_dim + hd;
        th_output[dst] = th_input[tid.x];
    }
}

// ============================================================
// Window Partition
// ============================================================
// Rearranges [H_p, W_p, dim] into [n_windows, window_h * window_w, dim]
// where each window is a contiguous block of window_h x window_w patches

struct WindowPartitionParams {
    uint patches_h;
    uint patches_w;
    uint dim;
    uint window_size;     // e.g., 8
    uint direction;       // 0 = partition, 1 = unpartition
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[vk_push_constant] const WindowPartitionParams wp_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> wp_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> wp_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void window_partition(uint3 tid: SV_DispatchThreadID) {
    uint total = wp_pc.patches_h * wp_pc.patches_w * wp_pc.dim;
    if (tid.x >= total) return;

    uint elem = tid.x;
    uint d = elem % wp_pc.dim;
    uint spatial = elem / wp_pc.dim;
    uint row = spatial / wp_pc.patches_w;
    uint col = spatial % wp_pc.patches_w;

    uint win_h = wp_pc.window_size;
    uint win_w = wp_pc.window_size;
    uint windows_per_row = wp_pc.patches_w / win_w;

    uint win_row = row / win_h;
    uint win_col = col / win_w;
    uint win_idx = win_row * windows_per_row + win_col;

    uint local_row = row % win_h;
    uint local_col = col % win_w;
    uint local_pos = local_row * win_w + local_col;

    uint win_size = win_h * win_w;

    if (wp_pc.direction == 0) {
        // partition: [H_p, W_p, dim] → [n_windows, win_size, dim]
        uint dst = win_idx * win_size * wp_pc.dim + local_pos * wp_pc.dim + d;
        wp_output[dst] = wp_input[elem];
    } else {
        // unpartition: [n_windows, win_size, dim] → [H_p, W_p, dim]
        uint src = win_idx * win_size * wp_pc.dim + local_pos * wp_pc.dim + d;
        wp_output[elem] = wp_input[src];
    }
}

// ============================================================
// Spatial Merge 2x2
// ============================================================
// Groups 2x2 adjacent patches and concatenates their features
// Input: [H_p, W_p, dim] → Output: [H_p/2 * W_p/2, 4*dim]
// Layout: [top-left, top-right, bottom-left, bottom-right] concatenated

struct SpatialMerge2x2Params {
    uint patches_h;
    uint patches_w;
    uint dim;
    uint _pad;
};

[vk_push_constant] const SpatialMerge2x2Params sm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> sm_input;   // [patches_h * patches_w, dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> sm_output;  // [patches_h/2 * patches_w/2, 4*dim]

[shader("compute")]
[numthreads(256, 1, 1)]
void spatial_merge_2x2(uint3 tid: SV_DispatchThreadID) {
    uint merged_h = sm_pc.patches_h / 2;
    uint merged_w = sm_pc.patches_w / 2;
    uint out_dim = sm_pc.dim * 4;
    uint total = merged_h * merged_w * out_dim;
    if (tid.x >= total) return;

    uint out_pos = tid.x / out_dim;
    uint out_d = tid.x % out_dim;

    uint mr = out_pos / merged_w;
    uint mc = out_pos % merged_w;

    // Which of the 4 sub-patches does this element come from?
    uint sub_patch = out_d / sm_pc.dim;  // 0=TL, 1=TR, 2=BL, 3=BR
    uint feat_idx = out_d % sm_pc.dim;

    uint sr = mr * 2 + (sub_patch / 2);  // row: 0,1 for TL/TR, 2,3 for BL/BR
    uint sc = mc * 2 + (sub_patch % 2);  // col: 0,2 for TL/BL, 1,3 for TR/BR

    uint src_idx = (sr * sm_pc.patches_w + sc) * sm_pc.dim + feat_idx;
    sm_output[tid.x] = sm_input[src_idx];
}

// ============================================================
// MMDiT Shaders
// ============================================================

// ============================================================
// GEGLU Activation (in-place on first half)
// ============================================================
// Input buffer has [n_rows, 2*half_dim]. Split into gate[half_dim] and value[half_dim].
// Result: output[i] = GELU(gate[i]) * value[i], stored in first half_dim of each row.

struct GeGLUParams {
    uint n_rows;
    uint half_dim;   // 6144 (full dim is 12288)
};

[vk_push_constant] const GeGLUParams geglu_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> geglu_data;   // [n_rows, 2*half_dim] → in-place first half
[vk_binding(1, 0)] RWStructuredBuffer<float> geglu_output;  // [n_rows, half_dim]

[shader("compute")]
[numthreads(256, 1, 1)]
void geglu_activation(uint3 tid: SV_DispatchThreadID) {
    uint total = geglu_pc.n_rows * geglu_pc.half_dim;
    if (tid.x >= total) return;

    uint row = tid.x / geglu_pc.half_dim;
    uint col = tid.x % geglu_pc.half_dim;
    uint full_dim = geglu_pc.half_dim * 2;

    float gate_val = geglu_data[row * full_dim + col];
    float value_val = geglu_data[row * full_dim + geglu_pc.half_dim + col];

    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x = gate_val;
    float gelu = x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));

    geglu_output[tid.x] = gelu * value_val;
}

// ============================================================
// AdaLN Modulate: apply shift + scale to LayerNorm output
// One workgroup per row. Computes:
//   normed = LayerNorm(input[row])
//   output[row, d] = normed[d] * (1 + scale[d]) + shift[d]
// shift and scale are extracted from the modulation buffer at given offsets

struct AdaLNModParams {
    uint n_rows;
    uint dim;
    uint shift_offset;  // offset in modulation buffer (in floats) for shift[dim]
    uint scale_offset;  // offset in modulation buffer (in floats) for scale[dim]
};

[vk_push_constant] const AdaLNModParams adaln_mod_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> adaln_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> adaln_mod_buf;   // modulation params [6*dim]
[vk_binding(2, 0)] RWStructuredBuffer<float> adaln_output;

groupshared float adaln_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void adaln_modulate_shift_scale(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    if (row >= adaln_mod_pc.n_rows) return;

    uint base = row * adaln_mod_pc.dim;

    // Pass 1: Compute mean
    float sum = 0.0;
    for (uint i = gi; i < adaln_mod_pc.dim; i += 256) {
        sum += adaln_input[base + i];
    }
    adaln_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) adaln_shared[gi] += adaln_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = adaln_shared[0] / float(adaln_mod_pc.dim);

    // Pass 2: Compute variance
    float var_sum = 0.0;
    for (uint i = gi; i < adaln_mod_pc.dim; i += 256) {
        float diff = adaln_input[base + i] - mean;
        var_sum += diff * diff;
    }
    adaln_shared[gi] = var_sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) adaln_shared[gi] += adaln_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_std = 1.0 / sqrt(adaln_shared[0] / float(adaln_mod_pc.dim) + 1e-6);

    // Normalize + apply AdaLN modulation
    for (uint i = gi; i < adaln_mod_pc.dim; i += 256) {
        float normed = (adaln_input[base + i] - mean) * inv_std;
        float shift = adaln_mod_buf[adaln_mod_pc.shift_offset + i];
        float scale = adaln_mod_buf[adaln_mod_pc.scale_offset + i];
        adaln_output[base + i] = normed * (1.0 + scale) + shift;
    }
}

// ============================================================
// Per-head RMSNorm
// ============================================================
// Input is [seq_len, n_heads * head_dim] (contiguous per-row).
// Weight is [head_dim] (shared across all heads and positions).
// One workgroup per (seq_pos, head) pair.
// Dispatch: seq_len * n_heads workgroups.

struct PerHeadRMSNormParams {
    uint seq_len;
    uint n_heads;
    uint head_dim;
    float eps;
};

[vk_push_constant] const PerHeadRMSNormParams phrms_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> phrms_data;    // in-place [seq_len, n_heads * head_dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> phrms_weight;  // [head_dim]

groupshared float phrms_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void perhead_rmsnorm(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint idx = gid.x;
    if (idx >= phrms_pc.seq_len * phrms_pc.n_heads) return;

    uint pos = idx / phrms_pc.n_heads;
    uint head = idx % phrms_pc.n_heads;
    uint base = pos * (phrms_pc.n_heads * phrms_pc.head_dim) + head * phrms_pc.head_dim;

    // Compute sum of squares
    float sum_sq = 0.0;
    for (uint i = gi; i < phrms_pc.head_dim; i += 256) {
        float val = phrms_data[base + i];
        sum_sq += val * val;
    }
    phrms_shared[gi] = sum_sq;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) phrms_shared[gi] += phrms_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float rms = 1.0 / sqrt(phrms_shared[0] / float(phrms_pc.head_dim) + phrms_pc.eps);

    // Normalize in-place with weight
    for (uint i = gi; i < phrms_pc.head_dim; i += 256) {
        phrms_data[base + i] = phrms_data[base + i] * rms * phrms_weight[i];
    }
}

// ============================================================
// Gated Residual Add
// ============================================================
// hidden[row, d] += gate[d] * value[row, d]
// gate is broadcast from modulation buffer at a given offset

struct GatedResidualParams {
    uint n_rows;
    uint dim;
    uint gate_offset;   // offset in modulation buffer for gate[dim]
};

[vk_push_constant] const GatedResidualParams gated_res_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> gated_hidden;
[vk_binding(1, 0)] RWStructuredBuffer<float> gated_value;
[vk_binding(2, 0)] RWStructuredBuffer<float> gated_mod_buf;

[shader("compute")]
[numthreads(256, 1, 1)]
void gated_residual_add(uint3 tid: SV_DispatchThreadID) {
    uint total = gated_res_pc.n_rows * gated_res_pc.dim;
    if (tid.x >= total) return;

    uint col = tid.x % gated_res_pc.dim;
    float gate = gated_mod_buf[gated_res_pc.gate_offset + col];

    gated_hidden[tid.x] += gate * gated_value[tid.x];
}

// ============================================================
// Sinusoidal Timestep Embedding
// ============================================================
// Generates [dim] sinusoidal embedding from a scalar timestep.
// Even indices get cos, odd indices get sin.

struct TimestepEmbedParams {
    uint dim;
    float timestep;
};

[vk_push_constant] const TimestepEmbedParams te_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> te_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void sinusoidal_timestep_embed(uint3 tid: SV_DispatchThreadID) {
    uint half_dim = te_pc.dim / 2;
    if (tid.x >= te_pc.dim) return;

    uint pair = tid.x / 2;
    float freq = exp(-log(10000.0) * float(pair) / float(half_dim));
    float angle = te_pc.timestep * freq;

    if (tid.x % 2 == 0) {
        te_output[tid.x] = cos(angle);
    } else {
        te_output[tid.x] = sin(angle);
    }
}

// ============================================================
// Patchify: [C, H, W] → [n_patches, C * patch_h * patch_w]
// ============================================================

struct MMDiTPatchifyParams {
    uint channels;
    uint height;
    uint width;
    uint patch_size;
};

[vk_push_constant] const MMDiTPatchifyParams mpatch_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> mpatch_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> mpatch_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void mmdit_patchify(uint3 tid: SV_DispatchThreadID) {
    uint patch_dim = mpatch_pc.channels * mpatch_pc.patch_size * mpatch_pc.patch_size;
    uint patches_w = mpatch_pc.width / mpatch_pc.patch_size;
    uint patches_h = mpatch_pc.height / mpatch_pc.patch_size;
    uint n_patches = patches_h * patches_w;
    uint total = n_patches * patch_dim;
    if (tid.x >= total) return;

    uint patch_idx = tid.x / patch_dim;
    uint inner = tid.x % patch_dim;

    uint c = inner / (mpatch_pc.patch_size * mpatch_pc.patch_size);
    uint rem = inner % (mpatch_pc.patch_size * mpatch_pc.patch_size);
    uint py = rem / mpatch_pc.patch_size;
    uint px = rem % mpatch_pc.patch_size;

    uint pr = patch_idx / patches_w;
    uint pc_col = patch_idx % patches_w;
    uint y = pr * mpatch_pc.patch_size + py;
    uint x = pc_col * mpatch_pc.patch_size + px;

    mpatch_output[tid.x] = mpatch_input[c * mpatch_pc.height * mpatch_pc.width + y * mpatch_pc.width + x];
}

// ============================================================
// Unpatchify: [n_patches, C * patch_h * patch_w] → [C, H, W]
// ============================================================

[vk_binding(2, 0)] RWStructuredBuffer<float> unpatch_input;
[vk_binding(3, 0)] RWStructuredBuffer<float> unpatch_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void mmdit_unpatchify(uint3 tid: SV_DispatchThreadID) {
    uint patch_dim = mpatch_pc.channels * mpatch_pc.patch_size * mpatch_pc.patch_size;
    uint patches_w = mpatch_pc.width / mpatch_pc.patch_size;
    uint patches_h = mpatch_pc.height / mpatch_pc.patch_size;
    uint total = mpatch_pc.channels * mpatch_pc.height * mpatch_pc.width;
    if (tid.x >= total) return;

    uint c = tid.x / (mpatch_pc.height * mpatch_pc.width);
    uint rem = tid.x % (mpatch_pc.height * mpatch_pc.width);
    uint y = rem / mpatch_pc.width;
    uint x = rem % mpatch_pc.width;

    uint pr = y / mpatch_pc.patch_size;
    uint py = y % mpatch_pc.patch_size;
    uint pc_col = x / mpatch_pc.patch_size;
    uint px = x % mpatch_pc.patch_size;

    uint patch_idx = pr * patches_w + pc_col;
    uint inner = c * mpatch_pc.patch_size * mpatch_pc.patch_size + py * mpatch_pc.patch_size + px;

    unpatch_output[tid.x] = unpatch_input[patch_idx * patch_dim + inner];
}

// ============================================================
// Flow Euler Step: latent += dt * velocity
// ============================================================

struct FlowEulerParams {
    uint n;
    float dt;
};

[vk_push_constant] const FlowEulerParams feuler_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> feuler_latent;
[vk_binding(1, 0)] RWStructuredBuffer<float> feuler_velocity;

[shader("compute")]
[numthreads(256, 1, 1)]
void flow_euler_step(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= feuler_pc.n) return;
    feuler_latent[tid.x] += feuler_pc.dt * feuler_velocity[tid.x];
}

// ============================================================
// VAE RMS Norm (channel-wise per spatial position)
// ============================================================
// Data in [C, spatial] layout. For each spatial position s:
//   rms = sqrt(mean(x[c,s]^2 for c in 0..C) + eps)
//   output[c*S+s] = input[c*S+s] / rms * gamma[c]
// One workgroup per spatial position, reducing over channels.

struct VaeRMSNormParams {
    uint channels;    // C
    uint spatial;     // H*W
    float eps;
    uint _pad;
};

[vk_push_constant] const VaeRMSNormParams vrn_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> vrn_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> vrn_gamma;   // [channels]
[vk_binding(2, 0)] RWStructuredBuffer<float> vrn_output;

groupshared float vrn_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void vae_rmsnorm(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint s = gid.x;
    if (s >= vrn_pc.spatial) return;

    // Sum of squares across channels (strided: input[c * spatial + s])
    float sum_sq = 0.0;
    for (uint c = gi; c < vrn_pc.channels; c += 256) {
        float val = vrn_input[c * vrn_pc.spatial + s];
        sum_sq += val * val;
    }
    vrn_shared[gi] = sum_sq;
    GroupMemoryBarrierWithGroupSync();
    for (uint r = 128; r > 0; r >>= 1) {
        if (gi < r) vrn_shared[gi] += vrn_shared[gi + r];
        GroupMemoryBarrierWithGroupSync();
    }
    float rms = 1.0 / sqrt(vrn_shared[0] / float(vrn_pc.channels) + vrn_pc.eps);

    // Normalize with gamma weight
    for (uint c = gi; c < vrn_pc.channels; c += 256) {
        vrn_output[c * vrn_pc.spatial + s] = vrn_input[c * vrn_pc.spatial + s] * rms * vrn_gamma[c];
    }
}

// ============================================================
// Norm-preserving CFG
// ============================================================
// cond_vel[row] = uncond[row] + scale * (cond[row] - uncond[row])
// Then rescale so ||result_row|| = ||cond_row||
// One workgroup per row.

struct CFGParams {
    uint n_rows;
    uint dim;
    float scale;
    uint _pad;
};

[vk_push_constant] const CFGParams cfg_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> cfg_cond;     // [n_rows, dim] in/out
[vk_binding(1, 0)] RWStructuredBuffer<float> cfg_uncond;    // [n_rows, dim]

groupshared float cfg_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void norm_preserving_cfg(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    if (row >= cfg_pc.n_rows) return;

    uint base = row * cfg_pc.dim;

    // Compute ||cond_row|| and combined = uncond + scale * (cond - uncond)
    // First pass: compute cond norm squared and store combined
    float cond_norm_sq = 0.0;
    for (uint i = gi; i < cfg_pc.dim; i += 256) {
        float c = cfg_cond[base + i];
        float u = cfg_uncond[base + i];
        cond_norm_sq += c * c;
        cfg_cond[base + i] = u + cfg_pc.scale * (c - u);  // store combined
    }
    cfg_shared[gi] = cond_norm_sq;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) cfg_shared[gi] += cfg_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float cond_norm = sqrt(cfg_shared[0]);

    // Second pass: compute combined norm
    float comb_norm_sq = 0.0;
    for (uint i = gi; i < cfg_pc.dim; i += 256) {
        float v = cfg_cond[base + i];
        comb_norm_sq += v * v;
    }
    cfg_shared[gi] = comb_norm_sq;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) cfg_shared[gi] += cfg_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float comb_norm = sqrt(cfg_shared[0]);

    // Rescale: result = combined * (cond_norm / comb_norm)
    float ratio = (comb_norm > 1e-8) ? (cond_norm / comb_norm) : 1.0;
    for (uint i = gi; i < cfg_pc.dim; i += 256) {
        cfg_cond[base + i] *= ratio;
    }
}
