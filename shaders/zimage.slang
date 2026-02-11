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

    if (gi == 0) {
        // Compute dot product
        float sum = 0.0;
        for (uint k = 0; k < linear_proj_pc.in_dim; k++) {
            sum += lp_input[seq_pos * linear_proj_pc.in_dim + k] *
                   lp_weight[out_idx * linear_proj_pc.in_dim + k];
        }
        sum += lp_bias[out_idx];
        lp_output[seq_pos * linear_proj_pc.out_dim + out_idx] = sum;
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
