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
        dit_timestep_output[tid.x] = sin(dit_timestep_pc.timestep * freq);
    } else {
        uint idx = tid.x - half_dim;
        float freq = exp(-log(10000.0) * float(idx) / float(half_dim));
        dit_timestep_output[tid.x] = cos(dit_timestep_pc.timestep * freq);
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
