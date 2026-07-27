// Kernels for the SkinTokens mesh encoder — a Michelangelo/3DShape2VecSet
// perceiver at width 512.
//
// Everything here is row-batched: the encoder runs one cross-attention block
// over a 54000-point cloud, so the single-row layernorm and matmul in
// diffusion.slang would need 54000 dispatches each.
//
// The attention kernel lives in mesh_attention.slang. SPIR-V accumulates
// groupshared allocation across every entry point in a module, and its ~14 KB
// of tiles would otherwise be charged to these cheap elementwise kernels too.

// ============================================================
// Batched LayerNorm with affine parameters
// ============================================================
// One workgroup per row. weight is required; bias is optional so that a
// LayerNorm(bias=False) can bind any buffer and set has_bias = 0.

struct MeshLayerNormParams {
    uint dim;
    uint n_rows;
    float eps;
    uint has_bias;
};

[vk_push_constant] const MeshLayerNormParams me_ln_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> me_ln_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> me_ln_weight;
[vk_binding(2, 0)] RWStructuredBuffer<float> me_ln_bias;
[vk_binding(3, 0)] RWStructuredBuffer<float> me_ln_output;

groupshared float me_ln_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void mesh_layernorm(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    if (row >= me_ln_pc.n_rows) return;

    uint base = row * me_ln_pc.dim;

    float sum = 0.0;
    for (uint i = gi; i < me_ln_pc.dim; i += 256) {
        sum += me_ln_input[base + i];
    }
    me_ln_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) me_ln_shared[gi] += me_ln_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = me_ln_shared[0] / float(me_ln_pc.dim);
    GroupMemoryBarrierWithGroupSync();

    float var_sum = 0.0;
    for (uint i = gi; i < me_ln_pc.dim; i += 256) {
        float diff = me_ln_input[base + i] - mean;
        var_sum += diff * diff;
    }
    me_ln_shared[gi] = var_sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) me_ln_shared[gi] += me_ln_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    // Biased variance, matching torch.nn.LayerNorm.
    float inv_std = 1.0 / sqrt(me_ln_shared[0] / float(me_ln_pc.dim) + me_ln_pc.eps);

    for (uint i = gi; i < me_ln_pc.dim; i += 256) {
        float normed = (me_ln_input[base + i] - mean) * inv_std * me_ln_weight[i];
        me_ln_output[base + i] = me_ln_pc.has_bias != 0 ? normed + me_ln_bias[i] : normed;
    }
}

// ============================================================
// Batched RMSNorm with a weight
// ============================================================
// The bridge into the LLM (output_proj.1) carries a weight and no bias, which
// is how Qwen3 spells its norms and not how anything else in this checkpoint
// spells a LayerNorm.

struct MeshRmsNormParams {
    uint dim;
    uint n_rows;
    float eps;
    uint _pad;
};

[vk_push_constant] const MeshRmsNormParams me_rms_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> me_rms_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> me_rms_weight;
[vk_binding(2, 0)] RWStructuredBuffer<float> me_rms_output;

groupshared float me_rms_shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void mesh_rmsnorm(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    if (row >= me_rms_pc.n_rows) return;

    uint base = row * me_rms_pc.dim;

    float sum = 0.0;
    for (uint i = gi; i < me_rms_pc.dim; i += 256) {
        float v = me_rms_input[base + i];
        sum += v * v;
    }
    me_rms_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) me_rms_shared[gi] += me_rms_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_rms = 1.0 / sqrt(me_rms_shared[0] / float(me_rms_pc.dim) + me_rms_pc.eps);

    for (uint i = gi; i < me_rms_pc.dim; i += 256) {
        me_rms_output[base + i] = me_rms_input[base + i] * inv_rms * me_rms_weight[i];
    }
}

// ============================================================
// Linear: out[M, N] = in[M, K] @ weight[N, K]^T + bias[N]
// ============================================================
// Weight stays in the [out, in] layout safetensors ships it in. Both tile
// loads run along K so that consecutive threads touch consecutive addresses;
// the pad to 17 keeps the transposed read in the inner product off a single
// groupshared bank.

struct MeshLinearParams {
    uint M;
    uint N;
    uint K;
    uint has_bias;
};

[vk_push_constant] const MeshLinearParams me_lin_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> me_lin_input;
[vk_binding(1, 0)] RWStructuredBuffer<float> me_lin_weight;
[vk_binding(2, 0)] RWStructuredBuffer<float> me_lin_bias;
[vk_binding(3, 0)] RWStructuredBuffer<float> me_lin_output;

groupshared float me_lin_a[16][17];
groupshared float me_lin_w[16][17];

[shader("compute")]
[numthreads(16, 16, 1)]
void mesh_linear(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID) {
    uint col = gid.x * 16 + lid.x;
    uint row = gid.y * 16 + lid.y;
    uint w_row = gid.x * 16 + lid.y;

    float sum = 0.0;
    uint tiles = (me_lin_pc.K + 15) / 16;

    for (uint t = 0; t < tiles; t++) {
        uint k = t * 16 + lid.x;

        me_lin_a[lid.y][lid.x] = (row < me_lin_pc.M && k < me_lin_pc.K)
            ? me_lin_input[row * me_lin_pc.K + k] : 0.0;
        me_lin_w[lid.y][lid.x] = (w_row < me_lin_pc.N && k < me_lin_pc.K)
            ? me_lin_weight[w_row * me_lin_pc.K + k] : 0.0;

        GroupMemoryBarrierWithGroupSync();

        for (uint kk = 0; kk < 16; kk++) {
            sum += me_lin_a[lid.y][kk] * me_lin_w[lid.x][kk];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (row < me_lin_pc.M && col < me_lin_pc.N) {
        me_lin_output[row * me_lin_pc.N + col] =
            me_lin_pc.has_bias != 0 ? sum + me_lin_bias[col] : sum;
    }
}

// ============================================================
// GELU (in-place)
// ============================================================
// The exact erf form, because that is what torch's default nn.GELU() is and
// what Michelangelo's MLP was trained with. The tanh approximation everything
// else in this engine uses is off by up to ~1e-3 around |x| = 2, which is the
// part of the curve most activations land in.
//
// Slang has no erf, so this is Abramowitz & Stegun 7.1.26 -- max absolute
// error 1.5e-7, three orders below the tanh form and well past what F32
// accumulation downstream can carry.

struct MeshGeluParams {
    uint n;
};

[vk_push_constant] const MeshGeluParams me_gelu_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> me_gelu_data;

float mesh_erf(float x) {
    float sign = x < 0.0 ? -1.0 : 1.0;
    float a = abs(x);

    float t = 1.0 / (1.0 + 0.3275911 * a);
    float poly = t * (0.254829592
              + t * (-0.284496736
              + t * (1.421413741
              + t * (-1.453152027
              + t * 1.061405429))));

    return sign * (1.0 - poly * exp(-a * a));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mesh_gelu(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= me_gelu_pc.n) return;
    float x = me_gelu_data[tid.x];
    me_gelu_data[tid.x] = 0.5 * x * (1.0 + mesh_erf(x * 0.70710678118654752));
}

// ============================================================
// Residual add (in-place on the accumulator)
// ============================================================

struct MeshAddParams {
    uint n;
};

[vk_push_constant] const MeshAddParams me_add_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> me_add_acc;
[vk_binding(1, 0)] RWStructuredBuffer<float> me_add_delta;

[shader("compute")]
[numthreads(256, 1, 1)]
void mesh_residual_add(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= me_add_pc.n) return;
    me_add_acc[tid.x] += me_add_delta[tid.x];
}
