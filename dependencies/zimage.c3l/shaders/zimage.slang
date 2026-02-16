// Z-Image Turbo compute shaders
// Patchify/unpatchify, AdaLN modulation, Flow Euler step, Linear projection with bias

// ============================================================
// Patchify - Convert [C, H, W] latent to [n_patches, patch_dim]
// ============================================================
// patch_dim = channels * patch_size * patch_size
// n_patches = (H / patch_size) * (W / patch_size)
// Output layout: patches[patch_idx * patch_dim + c * patch_size * patch_size + py * patch_size + px]

struct PatchifyParams {
    uint channels;
    uint height;
    uint width;
    uint patch_size;
};

[vk_push_constant] const PatchifyParams patchify_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> patchify_input;   // [C, H, W]
[vk_binding(1, 0)] RWStructuredBuffer<float> patchify_output;  // [n_patches, patch_dim]

[shader("compute")]
[numthreads(256, 1, 1)]
void patchify(uint3 tid: SV_DispatchThreadID) {
    uint ps = patchify_pc.patch_size;
    uint patch_dim = patchify_pc.channels * ps * ps;
    uint patches_w = patchify_pc.width / ps;
    uint patches_h = patchify_pc.height / ps;
    uint n_patches = patches_h * patches_w;
    uint total = n_patches * patch_dim;

    if (tid.x >= total) return;

    uint patch_idx = tid.x / patch_dim;
    uint elem = tid.x % patch_dim;

    uint c = elem / (ps * ps);
    uint rem = elem % (ps * ps);
    uint py = rem / ps;
    uint px = rem % ps;

    uint patch_y = patch_idx / patches_w;
    uint patch_x = patch_idx % patches_w;

    uint ih = patch_y * ps + py;
    uint iw = patch_x * ps + px;

    patchify_output[tid.x] = patchify_input[c * patchify_pc.height * patchify_pc.width + ih * patchify_pc.width + iw];
}

// ============================================================
// Unpatchify - Convert [n_patches, patch_dim] back to [C, H, W]
// ============================================================

struct UnpatchifyParams {
    uint channels;
    uint height;
    uint width;
    uint patch_size;
};

[vk_push_constant] const UnpatchifyParams unpatchify_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> unpatchify_input;   // [n_patches, patch_dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> unpatchify_output;  // [C, H, W]

[shader("compute")]
[numthreads(256, 1, 1)]
void unpatchify(uint3 tid: SV_DispatchThreadID) {
    uint total = unpatchify_pc.channels * unpatchify_pc.height * unpatchify_pc.width;
    if (tid.x >= total) return;

    uint ps = unpatchify_pc.patch_size;
    uint c = tid.x / (unpatchify_pc.height * unpatchify_pc.width);
    uint rem = tid.x % (unpatchify_pc.height * unpatchify_pc.width);
    uint h = rem / unpatchify_pc.width;
    uint w = rem % unpatchify_pc.width;

    uint patches_w = unpatchify_pc.width / ps;
    uint patch_y = h / ps;
    uint patch_x = w / ps;
    uint py = h % ps;
    uint px = w % ps;

    uint patch_idx = patch_y * patches_w + patch_x;
    uint patch_dim = unpatchify_pc.channels * ps * ps;
    uint elem = c * ps * ps + py * ps + px;

    unpatchify_output[tid.x] = unpatchify_input[patch_idx * patch_dim + elem];
}

// ============================================================
// AdaLN Modulate - output = input * (1 + scale) + shift
// ============================================================
// Applied per-element, scale and shift are broadcast across positions

struct AdalnModParams {
    uint n_elements;    // total elements to process
    uint dim;           // dimension (for broadcasting)
};

[vk_push_constant] const AdalnModParams adaln_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> adaln_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> adaln_scale;   // [dim]
[vk_binding(2, 0)] RWStructuredBuffer<float> adaln_shift;   // [dim]
[vk_binding(3, 0)] RWStructuredBuffer<float> adaln_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void adaln_modulate(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= adaln_pc.n_elements) return;

    uint d = tid.x % adaln_pc.dim;
    adaln_output[tid.x] = adaln_input[tid.x] * (1.0 + adaln_scale[d]) + adaln_shift[d];
}

// ============================================================
// Flow Euler Step - x = x + dt * velocity
// ============================================================

struct FlowEulerParams {
    uint n;
    float dt;
};

[vk_push_constant] const FlowEulerParams flow_euler_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> flow_x;         // current state (updated in-place)
[vk_binding(1, 0)] RWStructuredBuffer<float> flow_velocity;   // predicted velocity

[shader("compute")]
[numthreads(256, 1, 1)]
void flow_euler_step(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= flow_euler_pc.n) return;
    flow_x[tid.x] += flow_euler_pc.dt * flow_velocity[tid.x];
}

// ============================================================
// Linear Projection with Bias - output = input @ weight^T + bias
// ============================================================
// Handles F32 weights with bias (existing matmul has no bias)
// Processes one output row per workgroup
// Input: [seq_len, in_dim], Weight: [out_dim, in_dim], Bias: [out_dim]
// Output: [seq_len, out_dim]

struct LinearProjParams {
    uint out_dim;
    uint in_dim;
    uint seq_len;
};

[vk_push_constant] const LinearProjParams linear_proj_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> lp_weight;   // [out_dim, in_dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> lp_bias;     // [out_dim]
[vk_binding(2, 0)] RWStructuredBuffer<float> lp_input;    // [seq_len, in_dim]
[vk_binding(3, 0)] RWStructuredBuffer<float> lp_output;   // [seq_len, out_dim]

groupshared float lp_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void linear_proj(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    // gid.x = output element index (seq_pos * out_dim + out_idx)
    uint flat_idx = gid.x;
    uint seq_pos = flat_idx / linear_proj_pc.out_dim;
    uint out_idx = flat_idx % linear_proj_pc.out_dim;

    if (seq_pos >= linear_proj_pc.seq_len || out_idx >= linear_proj_pc.out_dim) return;

    // Parallel dot product: each thread handles a stride of elements
    // GGML stores weights as [out_dim, in_dim] row-major (same as batch_matmul)
    float partial = 0.0;
    for (uint k = gi; k < linear_proj_pc.in_dim; k += 256) {
        partial += lp_input[seq_pos * linear_proj_pc.in_dim + k] *
                   lp_weight[out_idx * linear_proj_pc.in_dim + k];
    }

    // Tree reduction in shared memory
    lp_shared[gi] = partial;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) {
            lp_shared[gi] += lp_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0) {
        lp_output[seq_pos * linear_proj_pc.out_dim + out_idx] = lp_shared[0] + lp_bias[out_idx];
    }
}

// ============================================================
// Linear Projection SIMPLIFIED + DEBUG VERSION
// ============================================================
// One workgroup per output row (seq_len workgroups total)
// Each thread computes a subset of output columns using strided loop
// Includes debug output buffer for validation

[vk_binding(4, 0)] RWStructuredBuffer<uint> lp_debug;     // Debug buffer [seq_len, out_dim, 6] entries
[vk_binding(5, 0)] RWStructuredBuffer<uint> lp_debug_count; // Atomic counter for debug entries

struct LinearProjDebugEntry {
    uint seq_pos;
    uint out_idx;
    uint thread_idx;
    uint workgroup_id;
    float value;
    uint flags;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void linear_proj_debug(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    // gid.x = row index (sequence position)
    uint seq_pos = gid.x;
    
    // Bounds check
    if (seq_pos >= linear_proj_pc.seq_len) return;
    
    // Strided loop: thread gi computes columns gi, gi+256, gi+512, ...
    for (uint out_idx = gi; out_idx < linear_proj_pc.out_dim; out_idx += 256) {
        // Compute dot product for this output element
        float sum = 0.0;
        for (uint k = 0; k < linear_proj_pc.in_dim; k++) {
            float input_val = lp_input[seq_pos * linear_proj_pc.in_dim + k];
            // Weight layout: [out_dim, in_dim] row-major from GGML
            float weight_val = lp_weight[out_idx * linear_proj_pc.in_dim + k];
            sum += input_val * weight_val;
        }
        
        float result = sum + lp_bias[out_idx];
        uint output_idx = seq_pos * linear_proj_pc.out_dim + out_idx;
        lp_output[output_idx] = result;
        
        // Debug logging for first workgroup and first few outputs
        if (seq_pos < 2 && out_idx < 10 && gi < 2) {
            uint debug_idx = 0;
            if (lp_debug_count[0] < 100) {
                InterlockedAdd(lp_debug_count[0], 10, debug_idx);
                if (debug_idx + 9 < 1000) {  // Max 100 entries * 10 uints
                    lp_debug[debug_idx + 0] = seq_pos;
                    lp_debug[debug_idx + 1] = out_idx;
                    lp_debug[debug_idx + 2] = gi;
                    lp_debug[debug_idx + 3] = gid.x;
                    // Store float as bits
                    lp_debug[debug_idx + 4] = asuint(result);
                    // First weight value accessed (k=0)
                    float first_weight = lp_weight[0 * linear_proj_pc.out_dim + out_idx];
                    lp_debug[debug_idx + 5] = asuint(first_weight);
                    // First input value accessed (k=0)
                    float first_input = lp_input[seq_pos * linear_proj_pc.in_dim + 0];
                    lp_debug[debug_idx + 6] = asuint(first_input);
                    // Input access index for k=0
                    lp_debug[debug_idx + 7] = seq_pos * linear_proj_pc.in_dim + 0;
                    // Validation flags
                    uint flags = 0;
                    if (seq_pos >= linear_proj_pc.seq_len) flags |= 1;
                    if (out_idx >= linear_proj_pc.out_dim) flags |= 2;
                    if (output_idx >= linear_proj_pc.seq_len * linear_proj_pc.out_dim) flags |= 4;
                    lp_debug[debug_idx + 8] = flags;
                    // Checksum of first 5 input values
                    float input_sum = 0.0;
                    for (uint kk = 0; kk < 5 && kk < linear_proj_pc.in_dim; kk++) {
                        input_sum += lp_input[seq_pos * linear_proj_pc.in_dim + kk];
                    }
                    lp_debug[debug_idx + 9] = asuint(input_sum);
                }
            }
        }
    }
}

// ============================================================
// Sinusoidal Timestep Embedding (for DiT)
// ============================================================
// Produces [dim] embedding from scalar timestep
// Uses log-spaced frequencies

struct DiTTimestepParams {
    uint dim;
    float timestep;
};

[vk_push_constant] const DiTTimestepParams dit_timestep_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> dit_timestep_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void dit_timestep_embed(uint3 tid: SV_DispatchThreadID) {
    uint half_dim = dit_timestep_pc.dim / 2;
    if (tid.x >= dit_timestep_pc.dim) return;

    if (tid.x < half_dim) {
        float freq = exp(-log(10000.0) * float(tid.x) / float(half_dim));
        dit_timestep_output[tid.x] = cos(dit_timestep_pc.timestep * freq);
    } else {
        uint idx = tid.x - half_dim;
        float freq = exp(-log(10000.0) * float(idx) / float(half_dim));
        dit_timestep_output[tid.x] = sin(dit_timestep_pc.timestep * freq);
    }
}

// ============================================================
// Element-wise add with broadcast - a[n] += b[dim] (broadcast b over positions)
// ============================================================

struct BroadcastAddDimParams {
    uint n_total;
    uint dim;
};

[vk_push_constant] const BroadcastAddDimParams bcast_add_dim_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bcast_a;  // [seq_len, dim] - modified in place
[vk_binding(1, 0)] RWStructuredBuffer<float> bcast_b;  // [dim]

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_add_dim(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= bcast_add_dim_pc.n_total) return;
    uint d = tid.x % bcast_add_dim_pc.dim;
    bcast_a[tid.x] += bcast_b[d];
}

// ============================================================
// Scale latent by VAE scaling factor
// ============================================================

struct ScaleParams {
    uint n;
    float scale;
};

[vk_push_constant] const ScaleParams scale_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> scale_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void scale_buffer(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= scale_pc.n) return;
    scale_data[tid.x] *= scale_pc.scale;
}

// ============================================================
// Scale Modulate - output = input * (1 + scale)
// ============================================================
// Like adaln_modulate but without shift term

struct ScaleModParams {
    uint n_elements;
    uint dim;
};

[vk_push_constant] const ScaleModParams scale_mod_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> smod_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> smod_scale;   // [dim]
[vk_binding(2, 0)] RWStructuredBuffer<float> smod_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void scale_modulate(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= scale_mod_pc.n_elements) return;
    uint d = tid.x % scale_mod_pc.dim;
    smod_output[tid.x] = smod_input[tid.x] * (1.0 + smod_scale[d]);
}

// ============================================================
// Gated Residual Add - residual += scale * tanh(gate) * clip(input, -clip_thresh, clip_thresh)
// ============================================================
// gate is [dim], broadcast across sequence positions
// scale allows residual scaling (e.g., 0.5 for DeepNet-style stabilization)
// clip_thresh clamps extreme outlier values to prevent divergence

struct GatedResidualParams {
    uint n_elements;
    uint dim;
    float scale;        // Residual scale factor (default: 1.0)
    float clip_thresh;  // Clip threshold (0.0 = no clipping, typical: 10.0)
};

[vk_push_constant] const GatedResidualParams gated_res_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> gres_residual;  // [seq_len, dim] modified in-place
[vk_binding(1, 0)] RWStructuredBuffer<float> gres_gate;      // [dim]
[vk_binding(2, 0)] RWStructuredBuffer<float> gres_input;     // [seq_len, dim]

[shader("compute")]
[numthreads(256, 1, 1)]
void gated_residual(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= gated_res_pc.n_elements) return;
    uint d = tid.x % gated_res_pc.dim;
    float residual_scale = (gated_res_pc.scale > 0.0) ? gated_res_pc.scale : 1.0;
    float clip_thresh = (gated_res_pc.clip_thresh > 0.0) ? gated_res_pc.clip_thresh : 1e10;
    float gated_val = residual_scale * tanh(gres_gate[d]) * gres_input[tid.x];
    // Clip to prevent extreme outliers from accumulating
    gated_val = clamp(gated_val, -clip_thresh, clip_thresh);
    gres_residual[tid.x] += gated_val;
}

// ============================================================
// Value Clipping - Clip extreme outliers in-place
// ============================================================

struct ClipParams {
    uint n_elements;
    float threshold;  // Values outside [-threshold, threshold] are clamped
};

[vk_push_constant] const ClipParams clip_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> clip_buffer;

[shader("compute")]
[numthreads(256, 1, 1)]
void clip_values(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= clip_pc.n_elements) return;
    float val = clip_buffer[tid.x];
    clip_buffer[tid.x] = clamp(val, -clip_pc.threshold, clip_pc.threshold);
}

// ============================================================
// Batch RMSNorm - One workgroup per row
// ============================================================
// Processes [n_rows, dim] with weight [dim], in separate input/output buffers
// row_offset: offset into input/output for first row (for dual norm second pass)

struct BatchRMSNormParams {
    uint dim;
    float eps;
    uint n_rows;
    uint row_offset;
};

[vk_push_constant] const BatchRMSNormParams batch_rmsnorm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> brn_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> brn_weight;
[vk_binding(2, 0)] RWStructuredBuffer<float> brn_output;

groupshared float brn_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_rmsnorm(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    if (row >= batch_rmsnorm_pc.n_rows) return;

    uint base = (row + batch_rmsnorm_pc.row_offset) * batch_rmsnorm_pc.dim;

    // Compute sum of squares
    float sum_sq = 0.0;
    for (uint i = gi; i < batch_rmsnorm_pc.dim; i += 256) {
        float val = brn_input[base + i];
        sum_sq += val * val;
    }

    brn_shared[gi] = sum_sq;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) {
            brn_shared[gi] += brn_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float rms = 1.0 / sqrt(brn_shared[0] / float(batch_rmsnorm_pc.dim) + batch_rmsnorm_pc.eps);

    for (uint i = gi; i < batch_rmsnorm_pc.dim; i += 256) {
        brn_output[base + i] = brn_input[base + i] * rms * brn_weight[i];
    }
}

// ============================================================
// Batch LayerNorm (no affine) - normalizes each row to zero mean, unit variance
// ============================================================
// Processes [n_rows, dim] in-place: y = (x - mean) / sqrt(var + eps)
// No learnable weight or bias parameters

struct BatchLayerNormParams {
    uint dim;
    float eps;
    uint n_rows;
    uint _pad;
};

[vk_push_constant] const BatchLayerNormParams batch_layernorm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bln_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> bln_output;

groupshared float bln_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_layernorm(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    if (row >= batch_layernorm_pc.n_rows) return;

    uint base = row * batch_layernorm_pc.dim;

    // Pass 1: Compute mean
    float sum = 0.0;
    for (uint i = gi; i < batch_layernorm_pc.dim; i += 256) {
        sum += bln_input[base + i];
    }
    bln_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) bln_shared[gi] += bln_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = bln_shared[0] / float(batch_layernorm_pc.dim);

    // Pass 2: Compute variance
    float var_sum = 0.0;
    for (uint i = gi; i < batch_layernorm_pc.dim; i += 256) {
        float diff = bln_input[base + i] - mean;
        var_sum += diff * diff;
    }
    bln_shared[gi] = var_sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) bln_shared[gi] += bln_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_std = 1.0 / sqrt(bln_shared[0] / float(batch_layernorm_pc.dim) + batch_layernorm_pc.eps);

    // Normalize
    for (uint i = gi; i < batch_layernorm_pc.dim; i += 256) {
        bln_output[base + i] = (bln_input[base + i] - mean) * inv_std;
    }
}

// ============================================================
// Transpose Heads - Remap between [seq, heads, head_dim] <-> [heads, seq, head_dim]
// ============================================================
// direction 0: [seq, heads, head_dim] -> [heads, seq, head_dim]
// direction 1: [heads, seq, head_dim] -> [seq, heads, head_dim]

struct TransposeHeadsParams {
    uint seq_len;
    uint n_heads;
    uint head_dim;
    uint direction;   // 0 = forward (seq-major -> head-major), 1 = reverse
};

[vk_push_constant] const TransposeHeadsParams transpose_heads_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> th_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> th_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void transpose_heads(uint3 tid: SV_DispatchThreadID) {
    uint S = transpose_heads_pc.seq_len;
    uint H = transpose_heads_pc.n_heads;
    uint D = transpose_heads_pc.head_dim;
    uint total = S * H * D;

    if (tid.x >= total) return;

    uint d = tid.x % D;
    uint rem = tid.x / D;

    if (transpose_heads_pc.direction == 0) {
        // src layout: [seq, heads, head_dim]
        uint seq = rem / H;
        uint h = rem % H;
        // dst layout: [heads, seq, head_dim]
        th_output[h * S * D + seq * D + d] = th_input[tid.x];
    } else {
        // src layout: [heads, seq, head_dim]
        uint h = rem / S;
        uint seq = rem % S;
        // dst layout: [seq, heads, head_dim]
        th_output[seq * H * D + h * D + d] = th_input[tid.x];
    }
}

// ============================================================
// Batch Head Norm - Per-head RMSNorm on [seq_len, n_heads*head_dim]
// ============================================================
// One workgroup per (position, head) pair
// Each workgroup normalizes head_dim elements in-place

struct BatchHeadNormParams {
    uint n_heads;
    uint head_dim;
    uint seq_len;
    float eps;
};

[vk_push_constant] const BatchHeadNormParams batch_head_norm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bhn_data;    // [seq_len, n_heads * head_dim] in-place
[vk_binding(1, 0)] RWStructuredBuffer<float> bhn_weight;   // [head_dim]

groupshared float bhn_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batch_head_norm(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint H = batch_head_norm_pc.n_heads;
    uint D = batch_head_norm_pc.head_dim;

    uint flat_id = gid.x;  // workgroup index = pos * n_heads + head
    uint pos = flat_id / H;
    uint h = flat_id % H;

    if (pos >= batch_head_norm_pc.seq_len) return;

    uint base = (pos * H + h) * D;

    // Compute sum of squares over head_dim elements
    float sum_sq = 0.0;
    for (uint i = gi; i < D; i += 256) {
        float val = bhn_data[base + i];
        sum_sq += val * val;
    }

    bhn_shared[gi] = sum_sq;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) {
            bhn_shared[gi] += bhn_shared[gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float rms = 1.0 / sqrt(bhn_shared[0] / float(D) + batch_head_norm_pc.eps);

    for (uint i = gi; i < D; i += 256) {
        bhn_data[base + i] = bhn_data[base + i] * rms * bhn_weight[i];
    }
}

// ============================================================
// Flash Attention - Shared-memory scores + fused reverse transpose
// ============================================================
// Q, K, V in [n_heads, seq_len, head_dim] layout (after forward transpose)
// Output in [seq_len, dim] layout (reverse transpose fused into write)
// Scores stay in groupshared memory — eliminates 140MB global buffer
// One workgroup per (head, query_position) pair

struct FlashAttentionParams {
    uint head_dim;    // 128
    uint n_heads;     // 30
    uint seq_len;     // ~1034
    float scale;      // 1/sqrt(128)
};

[vk_push_constant] const FlashAttentionParams flash_attn_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> fa_Q;       // [n_heads, seq_len, head_dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> fa_K;       // [n_heads, seq_len, head_dim]
[vk_binding(2, 0)] RWStructuredBuffer<float> fa_V;       // [n_heads, seq_len, head_dim]
[vk_binding(3, 0)] RWStructuredBuffer<float> fa_output;   // [seq_len, n_heads * head_dim]

groupshared float fa_scores[2048];   // max seq_len scores in shared memory
groupshared float fa_reduce[256];    // for tree reductions
groupshared float fa_q[128];         // cached Q vector for this query position

[shader("compute")]
[numthreads(256, 1, 1)]
void flash_attention(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint head = gid.x / flash_attn_pc.seq_len;
    uint q_pos = gid.x % flash_attn_pc.seq_len;
    uint D = flash_attn_pc.head_dim;
    uint S = flash_attn_pc.seq_len;

    if (head >= flash_attn_pc.n_heads) return;

    uint head_offset = head * S * D;

    // Pre-load Q vector into shared memory (128 floats)
    if (gi < D) {
        fa_q[gi] = fa_Q[head_offset + q_pos * D + gi];
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 1: Q * K^T -> scores in shared memory
    for (uint kv_pos = gi; kv_pos < S; kv_pos += 256) {
        float dot_val = 0.0;
        for (uint d = 0; d < D; d++) {
            dot_val += fa_q[d] * fa_K[head_offset + kv_pos * D + d];
        }
        fa_scores[kv_pos] = dot_val * flash_attn_pc.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Softmax entirely in shared memory
    float max_val = -1.0e30;
    for (uint i = gi; i < S; i += 256) {
        max_val = max(max_val, fa_scores[i]);
    }
    fa_reduce[gi] = max_val;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) fa_reduce[gi] = max(fa_reduce[gi], fa_reduce[gi + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float global_max = fa_reduce[0];
    GroupMemoryBarrierWithGroupSync();

    float sum = 0.0;
    for (uint i = gi; i < S; i += 256) {
        float val = exp(fa_scores[i] - global_max);
        fa_scores[i] = val;
        sum += val;
    }
    fa_reduce[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) fa_reduce[gi] += fa_reduce[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float global_sum = fa_reduce[0];
    GroupMemoryBarrierWithGroupSync();

    for (uint i = gi; i < S; i += 256) {
        fa_scores[i] /= global_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: Weighted V sum -> output in [seq_len, dim] layout (fused reverse transpose)
    uint dim = flash_attn_pc.n_heads * D;
    for (uint d = gi; d < D; d += 256) {
        float acc = 0.0;
        for (uint pos = 0; pos < S; pos++) {
            acc += fa_scores[pos] * fa_V[head_offset + pos * D + d];
        }
        // Write directly in [seq_len, dim] layout instead of [n_heads, seq_len, head_dim]
        fa_output[q_pos * dim + head * D + d] = acc;
    }
}

// mRoPE shader is in separate file (mrope.slang -> mrope.spv)
