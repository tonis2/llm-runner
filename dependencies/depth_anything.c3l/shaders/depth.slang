// Depth Anything V2 — kernels not covered by the shared/vit/diffusion shaders.
//
// Phase 2 (encoder):
//   broadcast_scale_mul  — LayerScale: x[row, col] *= gamma[col]
//   split_qkv            — fused QKV [N, 3*dim] -> Q,K,V [N, dim]
//
// (Phase 3 DPT kernels — bilinear resize, transposed conv — will be added here.)

// ============================================================
// LayerScale: per-channel broadcast multiply (in place)
// ============================================================
struct BroadcastScaleParams {
    uint dim;
    uint n_rows;
};

[vk_push_constant] const BroadcastScaleParams bsm_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bsm_data;   // [n_rows, dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> bsm_scale;  // [dim]

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_scale_mul(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint row = gid.x;
    uint col = gid.y * 256 + gi;
    if (row >= bsm_pc.n_rows || col >= bsm_pc.dim) return;
    bsm_data[row * bsm_pc.dim + col] *= bsm_scale[col];
}

// ============================================================
// Split fused QKV: [N, 3*dim] (q|k|v concatenated per row) -> 3x [N, dim]
// ============================================================
struct SplitQkvParams {
    uint dim;
    uint n_rows;
};

[vk_push_constant] const SplitQkvParams sq_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> sq_qkv;  // [n_rows, 3*dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> sq_q;    // [n_rows, dim]
[vk_binding(2, 0)] RWStructuredBuffer<float> sq_k;    // [n_rows, dim]
[vk_binding(3, 0)] RWStructuredBuffer<float> sq_v;    // [n_rows, dim]

[shader("compute")]
[numthreads(256, 1, 1)]
void split_qkv(uint3 tid: SV_DispatchThreadID) {
    uint total = sq_pc.n_rows * sq_pc.dim;
    if (tid.x >= total) return;
    uint row = tid.x / sq_pc.dim;
    uint d = tid.x % sq_pc.dim;
    uint base = row * 3 * sq_pc.dim;
    sq_q[tid.x] = sq_qkv[base + d];
    sq_k[tid.x] = sq_qkv[base + sq_pc.dim + d];
    sq_v[tid.x] = sq_qkv[base + 2 * sq_pc.dim + d];
}

// ============================================================
// Tokens [N, C] (skip CLS row 0) -> CHW [C, gh, gw]  (for the DPT head)
// ============================================================
struct TokensToChwParams {
    uint dim;
    uint n_patches;
};

[vk_push_constant] const TokensToChwParams t2c_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> t2c_tokens;  // [1 + n_patches, dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> t2c_out;     // [dim, n_patches]

[shader("compute")]
[numthreads(256, 1, 1)]
void tokens_to_chw(uint3 tid: SV_DispatchThreadID) {
    uint total = t2c_pc.dim * t2c_pc.n_patches;
    if (tid.x >= total) return;
    uint c = tid.x / t2c_pc.n_patches;
    uint p = tid.x % t2c_pc.n_patches;
    t2c_out[c * t2c_pc.n_patches + p] = t2c_tokens[(1 + p) * t2c_pc.dim + c];
}

// ============================================================
// Bilinear resize [C, in_h, in_w] -> [C, out_h, out_w], align_corners=True
// ============================================================
struct BilinearParams {
    uint channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
};

[vk_push_constant] const BilinearParams bil_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> bil_in;
[vk_binding(1, 0)] RWStructuredBuffer<float> bil_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void bilinear_resize(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint c = gid.y;
    uint sidx = gid.x * 256 + gi;
    if (c >= bil_pc.channels) return;
    uint oh = sidx / bil_pc.out_w;
    uint ow = sidx % bil_pc.out_w;
    if (oh >= bil_pc.out_h) return;

    // align_corners=True coordinate mapping
    float sy = (bil_pc.out_h > 1) ? float(oh) * float(bil_pc.in_h - 1) / float(bil_pc.out_h - 1) : 0.0;
    float sx = (bil_pc.out_w > 1) ? float(ow) * float(bil_pc.in_w - 1) / float(bil_pc.out_w - 1) : 0.0;

    uint y0 = uint(floor(sy));
    uint x0 = uint(floor(sx));
    uint y1 = min(y0 + 1, bil_pc.in_h - 1);
    uint x1 = min(x0 + 1, bil_pc.in_w - 1);
    float dy = sy - float(y0);
    float dx = sx - float(x0);

    uint plane = c * bil_pc.in_h * bil_pc.in_w;
    float v00 = bil_in[plane + y0 * bil_pc.in_w + x0];
    float v01 = bil_in[plane + y0 * bil_pc.in_w + x1];
    float v10 = bil_in[plane + y1 * bil_pc.in_w + x0];
    float v11 = bil_in[plane + y1 * bil_pc.in_w + x1];

    float v = v00 * (1.0 - dy) * (1.0 - dx)
            + v01 * (1.0 - dy) * dx
            + v10 * dy * (1.0 - dx)
            + v11 * dy * dx;
    bil_out[c * bil_pc.out_h * bil_pc.out_w + oh * bil_pc.out_w + ow] = v;
}

// ============================================================
// Non-overlapping transposed conv (stride == kernel, pad 0)
//   input  [Cin, in_h, in_w]
//   weight [Cin, Cout, k, k]  (PyTorch ConvTranspose2d layout, raw)
//   output [Cout, in_h*k, in_w*k]
// ============================================================
struct ConvTransposeParams {
    uint in_c;
    uint out_c;
    uint in_h;
    uint in_w;
    uint k;
    uint has_bias;
};

[vk_push_constant] const ConvTransposeParams ct_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> ct_weight;
[vk_binding(1, 0)] RWStructuredBuffer<float> ct_bias;
[vk_binding(2, 0)] RWStructuredBuffer<float> ct_input;
[vk_binding(3, 0)] RWStructuredBuffer<float> ct_output;

[shader("compute")]
[numthreads(256, 1, 1)]
void conv_transpose(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint oc = gid.y;
    uint out_h = ct_pc.in_h * ct_pc.k;
    uint out_w = ct_pc.in_w * ct_pc.k;
    uint sidx = gid.x * 256 + gi;
    uint oh = sidx / out_w;
    uint ow = sidx % out_w;
    if (oc >= ct_pc.out_c || oh >= out_h) return;

    uint ih = oh / ct_pc.k;
    uint iw = ow / ct_pc.k;
    uint ky = oh % ct_pc.k;
    uint kx = ow % ct_pc.k;
    uint ksz = ct_pc.k * ct_pc.k;

    float sum = 0.0;
    for (uint ic = 0; ic < ct_pc.in_c; ic++) {
        float inv = ct_input[ic * ct_pc.in_h * ct_pc.in_w + ih * ct_pc.in_w + iw];
        float w = ct_weight[ic * ct_pc.out_c * ksz + oc * ksz + ky * ct_pc.k + kx];
        sum += inv * w;
    }
    if (ct_pc.has_bias != 0) sum += ct_bias[oc];
    ct_output[oc * out_h * out_w + oh * out_w + ow] = sum;
}

