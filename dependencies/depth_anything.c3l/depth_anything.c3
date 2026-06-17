module depth_anything;

import vk;
import llm;
import llm::pipelines;
import llm_text;
import image;
import image::png;
import std::io;
import std::io::file;
import std::math;
import std::core::mem;

// Embedded depth SPIR-V (broadcast_scale_mul, split_qkv, + Phase 3 DPT kernels)
const char[*] DEPTH_SPV = $embed("./shaders/depth.spv");

// =====================================================================
// Depth Anything V2 (ViT-S/14 DINOv2 encoder + DPT decoder head)
//
// Checkpoint layout (safetensors, MackinationsAi conversion):
//   pretrained.*    -> DINOv2 ViT-S/14 encoder
//   depth_head.*    -> DPT decoder head
//
// This file (Phase 1) defines the config + weight structs and the
// safetensors loader. Forward pass (encoder) and DPT head are added in
// later phases.
// =====================================================================

// --- Configuration (ViT-S/14) ---

struct DepthConfig {
    uint dim;            // 384  encoder hidden dim
    uint n_heads;        // 6
    uint head_dim;       // 64   (dim / n_heads)
    uint ffn_dim;        // 1536 (dim * 4)
    uint n_layers;       // 12
    uint patch_size;     // 14
    uint pretrain_grid;  // 37   (pos_embed trained at 518/14 = 37x37)
    uint infer_grid;     // inference grid (>= pretrain_grid for higher-res depth); set by loader
    float ln_eps;        // 1e-6
    uint[4] tap_layers;  // {2, 5, 8, 11}  intermediate features fed to DPT
    uint[4] proj_dims;   // {48, 96, 192, 384}  DPT reassemble channels
    uint feature_dim;    // 64   DPT "scratch" common channel count
    float[3] image_mean; // ImageNet normalization
    float[3] image_std;
}

fn DepthConfig vits_config() {
    return {
        .dim = 384,
        .n_heads = 6,
        .head_dim = 64,
        .ffn_dim = 1536,
        .n_layers = 12,
        .patch_size = 14,
        .pretrain_grid = 37,
        .infer_grid = 37,
        .ln_eps = 1e-6f,
        .tap_layers = { 2, 5, 8, 11 },
        .proj_dims = { 48, 96, 192, 384 },
        .feature_dim = 64,
        .image_mean = { 0.485f, 0.456f, 0.406f },
        .image_std = { 0.229f, 0.224f, 0.225f },
    };
}

// --- Encoder weight structs ---

struct DinoBlock {
    llm::Tensor norm1_w;   // [dim]
    llm::Tensor norm1_b;   // [dim]
    llm::Tensor qkv_w;     // [3*dim, dim]  (Linear: out=3*dim, in=dim)
    llm::Tensor qkv_b;     // [3*dim]
    llm::Tensor proj_w;    // [dim, dim]
    llm::Tensor proj_b;    // [dim]
    llm::Tensor ls1;       // [dim]  LayerScale gamma (attn residual)
    llm::Tensor norm2_w;   // [dim]
    llm::Tensor norm2_b;   // [dim]
    llm::Tensor fc1_w;     // [ffn_dim, dim]
    llm::Tensor fc1_b;     // [ffn_dim]
    llm::Tensor fc2_w;     // [dim, ffn_dim]
    llm::Tensor fc2_b;     // [dim]
    llm::Tensor ls2;       // [dim]  LayerScale gamma (mlp residual)
}

struct DinoEncoder {
    llm::Tensor patch_w;   // conv [dim, 3, 14, 14] -> stored [kH,kW,Cin,Cout]
    llm::Tensor patch_b;   // [dim]
    llm::Tensor cls_token; // [dim]
    llm::Tensor pos_embed; // [1 + grid*grid, dim]  (1370 for 37x37)
    llm::Tensor norm_w;    // [dim]  final layernorm
    llm::Tensor norm_b;    // [dim]
    float[] pos_cpu;       // CPU copy of pos_embed [1+37*37, dim] for per-image interpolation
    DinoBlock[] blocks;
}

// --- DPT head weight structs ---

// Generic conv holder. `transposed` flags resize_layers 0/1 which are
// ConvTranspose2d (PyTorch weight layout [Cin, Cout, kH, kW]); those are
// loaded raw (no layout swap) and interpreted by the transposed-conv kernel.
struct DptConv {
    llm::Tensor weight;
    llm::Tensor bias;
    bool has_bias;
    bool transposed;
    uint in_c;
    uint out_c;
    uint k;
    uint stride;
    uint pad;
}

struct RefineNet {
    DptConv rcu1_conv1;  // 3x3 64->64
    DptConv rcu1_conv2;  // 3x3 64->64
    DptConv rcu2_conv1;  // 3x3 64->64
    DptConv rcu2_conv2;  // 3x3 64->64
    DptConv out_conv;    // 1x1 64->64
}

struct DptHead {
    DptConv[4] projects;     // 1x1: 384 -> {48,96,192,384}
    DptConv resize0;         // convT 4x4 s4: 48 -> 48   (4x up)
    DptConv resize1;         // convT 2x2 s2: 96 -> 96   (2x up)
    // resize index 2 is identity (no weights)
    DptConv resize3;         // conv 3x3 s2 p1: 384 -> 384 (1/2 down)
    DptConv[4] layer_rn;     // 3x3 p1, no bias: {48,96,192,384} -> 64
    RefineNet[4] refinenet;  // refinenet1..4 (top-down fusion)
    DptConv output_conv1;    // 3x3 p1: 64 -> 32
    DptConv output_conv2_0;  // 3x3 p1: 32 -> 32
    DptConv output_conv2_2;  // 1x1: 32 -> 1
}

// --- Model ---

struct DepthModel {
    DepthConfig config;
    DinoEncoder encoder;
    DptHead head;
    DepthKernels kernels;
    DepthActivations acts;
    DptActivations dpt;
    llm::DeviceContext* ctx;
}

// =====================================================================
// Name building helpers (mirrors flux_vae_name idiom — no heap alloc)
// =====================================================================

// prefix + suffix -> String view into buf
fn String da_name(char[256]* buf, String prefix, String suffix) {
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) { (*buf)[pos] = prefix[i]; pos++; }
    for (usz i = 0; i < suffix.len; i++) { (*buf)[pos] = suffix[i]; pos++; }
    return (String)(*buf)[0..pos - 1];
}

// "pretrained.blocks.{layer}.{suffix}"
fn String da_blk_name(char[256]* buf, uint layer, String suffix) {
    usz pos = 0;
    String pfx = "pretrained.blocks.";
    for (usz i = 0; i < pfx.len; i++) { (*buf)[pos] = pfx[i]; pos++; }
    if (layer >= 10) { (*buf)[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    (*buf)[pos] = (char)('0' + layer % 10); pos++;
    (*buf)[pos] = '.'; pos++;
    for (usz i = 0; i < suffix.len; i++) { (*buf)[pos] = suffix[i]; pos++; }
    return (String)(*buf)[0..pos - 1];
}

// --- Per-tensor loaders ---

// Linear weight / bias / norm / token: raw F32 upload (no layout change).
fn llm::Tensor da_lin(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String name) {
    return llm::upload_safetensor_f32(ctx, sf, name)!!;
}

fn llm::Tensor da_blk_lin(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, uint layer, String suffix) {
    char[256] buf;
    return llm::upload_safetensor_f32(ctx, sf, da_blk_name(&buf, layer, suffix))!!;
}

// Conv weight (+optional bias). transposed=true keeps PyTorch layout raw.
fn DptConv load_dpt_conv(llm::DeviceContext* ctx, llm::SafetensorsFile* sf,
        String prefix, uint in_c, uint out_c, uint k, uint stride, uint pad,
        bool has_bias, bool transposed) {
    char[256] wbuf;
    String wname = da_name(&wbuf, prefix, "weight");

    llm::Tensor w = transposed
        ? llm::upload_safetensor_f32(ctx, sf, wname)!!
        : llm::upload_safetensor_conv_f32(ctx, sf, wname)!!;

    DptConv c = {
        .weight = w,
        .has_bias = has_bias,
        .transposed = transposed,
        .in_c = in_c, .out_c = out_c, .k = k, .stride = stride, .pad = pad,
    };
    if (has_bias) {
        char[256] bbuf;
        c.bias = llm::upload_safetensor_f32(ctx, sf, da_name(&bbuf, prefix, "bias"))!!;
    }
    return c;
}

fn RefineNet load_refinenet(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String pfx) {
    char[256] b1; char[256] b2; char[256] b3; char[256] b4; char[256] b5;
    return {
        .rcu1_conv1 = load_dpt_conv(ctx, sf, da_name(&b1, pfx, "resConfUnit1.conv1."), 64, 64, 3, 1, 1, true, false),
        .rcu1_conv2 = load_dpt_conv(ctx, sf, da_name(&b2, pfx, "resConfUnit1.conv2."), 64, 64, 3, 1, 1, true, false),
        .rcu2_conv1 = load_dpt_conv(ctx, sf, da_name(&b3, pfx, "resConfUnit2.conv1."), 64, 64, 3, 1, 1, true, false),
        .rcu2_conv2 = load_dpt_conv(ctx, sf, da_name(&b4, pfx, "resConfUnit2.conv2."), 64, 64, 3, 1, 1, true, false),
        .out_conv   = load_dpt_conv(ctx, sf, da_name(&b5, pfx, "out_conv."),           64, 64, 1, 1, 0, true, false),
    };
}

// =====================================================================
// Loader
// =====================================================================

// box_px = requested longest-side inference resolution (snapped to a patch multiple).
// Larger -> finer depth detail, but attention memory grows O(grid^4): grid 37 (518px)
// uses a ~45 MB score buffer, grid 74 (1036px) ~720 MB, grid 111 (1554px) ~3.6 GB.
fn DepthModel? load_depth_model(llm::DeviceContext* ctx, String path, uint box_px) {
    io::printfn("\n=== Loading Depth Anything V2 (ViT-S/14) ===");
    io::printfn("  Path: %s", path);

    llm::SafetensorsFile sf = llm::safetensors_open(path)!!;
    io::printfn("  Tensors in file: %d", sf.n_tensors);

    DepthConfig config = vits_config();
    // Inference grid from the requested resolution (snap to a multiple of patch_size).
    uint infer_grid = (box_px + config.patch_size / 2) / config.patch_size;
    if (infer_grid < 1) infer_grid = 1;
    config.infer_grid = infer_grid;

    // --- Encoder ---
    DinoEncoder enc;
    // patch_embed uses vit.slang patch_embed_conv2d which indexes the weight as
    // raw PyTorch [Cout, Cin, kH, kW] (pe_weight[oc*kernel_size + k]), so load it
    // raw (NO conv layout swap).
    enc.patch_w   = llm::upload_safetensor_f32(ctx, &sf, "pretrained.patch_embed.proj.weight")!!;
    enc.patch_b   = da_lin(ctx, &sf, "pretrained.patch_embed.proj.bias");
    enc.cls_token = da_lin(ctx, &sf, "pretrained.cls_token");
    enc.pos_embed = da_lin(ctx, &sf, "pretrained.pos_embed");
    enc.norm_w    = da_lin(ctx, &sf, "pretrained.norm.weight");
    enc.norm_b    = da_lin(ctx, &sf, "pretrained.norm.bias");
    enc.pos_cpu   = read_st_f32(&sf, "pretrained.pos_embed");  // keep for pos-embed interpolation

    enc.blocks = mem::new_array(DinoBlock, config.n_layers);
    for (uint l = 0; l < config.n_layers; l++) {
        enc.blocks[l] = {
            .norm1_w = da_blk_lin(ctx, &sf, l, "norm1.weight"),
            .norm1_b = da_blk_lin(ctx, &sf, l, "norm1.bias"),
            .qkv_w   = da_blk_lin(ctx, &sf, l, "attn.qkv.weight"),
            .qkv_b   = da_blk_lin(ctx, &sf, l, "attn.qkv.bias"),
            .proj_w  = da_blk_lin(ctx, &sf, l, "attn.proj.weight"),
            .proj_b  = da_blk_lin(ctx, &sf, l, "attn.proj.bias"),
            .ls1     = da_blk_lin(ctx, &sf, l, "ls1.gamma"),
            .norm2_w = da_blk_lin(ctx, &sf, l, "norm2.weight"),
            .norm2_b = da_blk_lin(ctx, &sf, l, "norm2.bias"),
            .fc1_w   = da_blk_lin(ctx, &sf, l, "mlp.fc1.weight"),
            .fc1_b   = da_blk_lin(ctx, &sf, l, "mlp.fc1.bias"),
            .fc2_w   = da_blk_lin(ctx, &sf, l, "mlp.fc2.weight"),
            .fc2_b   = da_blk_lin(ctx, &sf, l, "mlp.fc2.bias"),
            .ls2     = da_blk_lin(ctx, &sf, l, "ls2.gamma"),
        };
    }
    io::printfn("  Encoder: patch_embed + %d blocks + final norm loaded", config.n_layers);

    // --- DPT head ---
    DptHead head;

    // projects: 1x1 conv 384 -> {48,96,192,384}
    head.projects[0] = load_dpt_conv(ctx, &sf, "depth_head.projects.0.", 384, 48,  1, 1, 0, true, false);
    head.projects[1] = load_dpt_conv(ctx, &sf, "depth_head.projects.1.", 384, 96,  1, 1, 0, true, false);
    head.projects[2] = load_dpt_conv(ctx, &sf, "depth_head.projects.2.", 384, 192, 1, 1, 0, true, false);
    head.projects[3] = load_dpt_conv(ctx, &sf, "depth_head.projects.3.", 384, 384, 1, 1, 0, true, false);

    // resize_layers: 0/1 are ConvTranspose (upsample), 3 is strided conv (downsample)
    head.resize0 = load_dpt_conv(ctx, &sf, "depth_head.resize_layers.0.", 48,  48,  4, 4, 0, true, true);
    head.resize1 = load_dpt_conv(ctx, &sf, "depth_head.resize_layers.1.", 96,  96,  2, 2, 0, true, true);
    head.resize3 = load_dpt_conv(ctx, &sf, "depth_head.resize_layers.3.", 384, 384, 3, 2, 1, true, false);

    // scratch.layerN_rn: 3x3 conv -> 64, NO bias
    head.layer_rn[0] = load_dpt_conv(ctx, &sf, "depth_head.scratch.layer1_rn.", 48,  64, 3, 1, 1, false, false);
    head.layer_rn[1] = load_dpt_conv(ctx, &sf, "depth_head.scratch.layer2_rn.", 96,  64, 3, 1, 1, false, false);
    head.layer_rn[2] = load_dpt_conv(ctx, &sf, "depth_head.scratch.layer3_rn.", 192, 64, 3, 1, 1, false, false);
    head.layer_rn[3] = load_dpt_conv(ctx, &sf, "depth_head.scratch.layer4_rn.", 384, 64, 3, 1, 1, false, false);

    // refinenet1..4
    head.refinenet[0] = load_refinenet(ctx, &sf, "depth_head.scratch.refinenet1.");
    head.refinenet[1] = load_refinenet(ctx, &sf, "depth_head.scratch.refinenet2.");
    head.refinenet[2] = load_refinenet(ctx, &sf, "depth_head.scratch.refinenet3.");
    head.refinenet[3] = load_refinenet(ctx, &sf, "depth_head.scratch.refinenet4.");

    // output convs
    head.output_conv1   = load_dpt_conv(ctx, &sf, "depth_head.scratch.output_conv1.",   64, 32, 3, 1, 1, true, false);
    head.output_conv2_0 = load_dpt_conv(ctx, &sf, "depth_head.scratch.output_conv2.0.", 32, 32, 3, 1, 1, true, false);
    head.output_conv2_2 = load_dpt_conv(ctx, &sf, "depth_head.scratch.output_conv2.2.", 32, 1,  1, 1, 0, true, false);

    io::printfn("  DPT head: projects + resize + scratch + 4 refinenets + output convs loaded");

    // Kernels + activation buffers (sized for the chosen inference grid; upper bounds).
    DepthKernels kernels = create_depth_kernels(ctx)!!;
    uint grid = config.infer_grid;
    uint n_tokens = 1 + grid * grid;        // + CLS
    io::printfn("  Inference grid: %dx%d (input %dpx)", grid, grid, grid * config.patch_size);
    DepthActivations acts = allocate_activations(ctx, &config, n_tokens)!!;
    DptActivations dpt = allocate_dpt_activations(ctx, grid, grid * config.patch_size)!!;

    io::printfn("=== Depth Anything loaded OK (%d tokens) ===\n", n_tokens);

    return {
        .config = config,
        .encoder = enc,
        .head = head,
        .kernels = kernels,
        .acts = acts,
        .dpt = dpt,
        .ctx = ctx,
    };
}

// =====================================================================
// Phase 2 — DINOv2 encoder forward
// =====================================================================

// --- Push constant structs (match shader layouts) ---

struct BatchLayerNormAffinePC { uint dim; float eps; uint n_rows; uint _pad; }
struct BroadcastBiasPC        { uint dim; uint n_rows; }
struct PatchEmbedPC           { uint height; uint width; uint patch_size; uint out_channels; uint in_channels; uint patches_h; uint patches_w; uint _pad; }
struct TransposeHeadsPC       { uint seq_len; uint n_heads; uint head_dim; uint direction; }
struct BatchMatMulPC          { uint out_dim; uint in_dim; uint seq_len; uint row_offset; }
struct SpatialAttentionPC     { uint head_dim; uint n_heads; uint seq_len; float scale; }
struct BroadcastScalePC       { uint dim; uint n_rows; }
struct SplitQkvPC             { uint dim; uint n_rows; }
struct TokensToChwPC          { uint dim; uint n_patches; }
struct BilinearPC             { uint channels; uint in_h; uint in_w; uint out_h; uint out_w; }
struct ConvTransposePC        { uint in_c; uint out_c; uint in_h; uint in_w; uint k; uint has_bias; }

// --- Kernels ---

struct DepthKernels {
    llm::ComputeKernel patch_embed;       // vit.slang
    llm::ComputeKernel batch_ln_affine;   // vit.slang (gamma only)
    llm::ComputeKernel broadcast_bias;    // vit.slang (add beta / bias)
    llm::ComputeKernel transpose_heads;   // vit.slang
    llm::ComputeKernel batch_matmul;      // llm.slang batch_matmul_simple
    llm::ComputeKernel spatial_attention; // diffusion.slang
    llm::ComputeKernel broadcast_scale;   // depth.slang (LayerScale)
    llm::ComputeKernel split_qkv;         // depth.slang
    // DPT head
    llm::ComputeKernel conv2d;            // diffusion.slang
    llm::ComputeKernel relu;              // diffusion.slang
    llm::ComputeKernel tokens_to_chw;     // depth.slang
    llm::ComputeKernel bilinear;          // depth.slang
    llm::ComputeKernel conv_transpose;    // depth.slang
    llm::SharedKernels shared;            // gelu, residual_add, ...
}

fn DepthKernels? create_depth_kernels(llm::DeviceContext* ctx) {
    char[] vit_spv = &VIT_SPV;
    vk::ShaderModule vit_shader = vk::shaderModuleCreateInfo().setCodeSize(
        vit_spv.len).setCode((uint*)&vit_spv[0]).build(ctx.device)!!;

    char[] llm_spv = &llm_text::LLM_SPV;
    vk::ShaderModule llm_shader = vk::shaderModuleCreateInfo().setCodeSize(
        llm_spv.len).setCode((uint*)&llm_spv[0]).build(ctx.device)!!;

    char[] diff_spv = &llm::pipelines::DIFFUSION_SPV;
    vk::ShaderModule diff_shader = vk::shaderModuleCreateInfo().setCodeSize(
        diff_spv.len).setCode((uint*)&diff_spv[0]).build(ctx.device)!!;

    char[] depth_spv = &DEPTH_SPV;
    vk::ShaderModule depth_shader = vk::shaderModuleCreateInfo().setCodeSize(
        depth_spv.len).setCode((uint*)&depth_spv[0]).build(ctx.device)!!;

    DepthKernels k = {
        .patch_embed       = llm::create_kernel(ctx, vit_shader, 3, PatchEmbedPC::size, "patch_embed_conv2d")!!,
        .batch_ln_affine   = llm::create_kernel(ctx, vit_shader, 3, BatchLayerNormAffinePC::size, "batch_layernorm_affine")!!,
        .broadcast_bias    = llm::create_kernel(ctx, vit_shader, 2, BroadcastBiasPC::size, "broadcast_bias_add")!!,
        .transpose_heads   = llm::create_kernel(ctx, vit_shader, 2, TransposeHeadsPC::size, "transpose_heads")!!,
        .batch_matmul      = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC::size, "batch_matmul_simple")!!,
        .spatial_attention = llm::create_kernel(ctx, diff_shader, 5, SpatialAttentionPC::size, "spatial_attention")!!,
        .broadcast_scale   = llm::create_kernel(ctx, depth_shader, 2, BroadcastScalePC::size, "broadcast_scale_mul")!!,
        .split_qkv         = llm::create_kernel(ctx, depth_shader, 4, SplitQkvPC::size, "split_qkv")!!,
        .conv2d            = llm::create_kernel(ctx, diff_shader, 4, llm::pipelines::Conv2dPC::size, "conv2d")!!,
        .relu              = llm::create_kernel(ctx, diff_shader, 1, llm::pipelines::ReluPC::size, "relu")!!,
        .tokens_to_chw     = llm::create_kernel(ctx, depth_shader, 2, TokensToChwPC::size, "tokens_to_chw")!!,
        .bilinear          = llm::create_kernel(ctx, depth_shader, 2, BilinearPC::size, "bilinear_resize")!!,
        .conv_transpose    = llm::create_kernel(ctx, depth_shader, 4, ConvTransposePC::size, "conv_transpose")!!,
        .shared            = llm::create_shared_kernels(ctx, llm_shader)!!,
    };

    vit_shader.free(ctx.device);
    llm_shader.free(ctx.device);
    diff_shader.free(ctx.device);
    depth_shader.free(ctx.device);
    return k;
}

// vit.spv lives in the qwen_edit module; re-embed it here so we can reuse its kernels.
const char[*] VIT_SPV = $embed("../qwen_edit.c3l/shaders/vit.spv");

// --- Activations ---

struct DepthActivations {
    llm::Tensor hidden;    // [N, dim]
    llm::Tensor norm_out;  // [N, dim]   scratch (LN output / attn-token output)
    llm::Tensor qkv;       // [N, 3*dim]
    llm::Tensor q;         // [N, dim]
    llm::Tensor k;         // [N, dim]
    llm::Tensor v;         // [N, dim]
    llm::Tensor attn_out;  // [N, dim]   ([heads,N,head_dim] during attention)
    llm::Tensor scores;    // [heads, N, N]
    llm::Tensor branch;    // [N, dim]   attn-proj / mlp output before residual
    llm::Tensor ffn;       // [N, ffn_dim]
    llm::Tensor patch_tok; // [n_patches, dim]
    llm::Tensor[4] taps;   // [N, dim] each — features after blocks 2/5/8/11
}

fn DepthActivations? allocate_activations(llm::DeviceContext* ctx, DepthConfig* c, uint n) {
    uint dim = c.dim;
    ulong[4] s_nd  = { (ulong)n * dim, 0, 0, 0 };
    ulong[4] s_qkv = { (ulong)n * 3 * dim, 0, 0, 0 };
    ulong[4] s_ffn = { (ulong)n * c.ffn_dim, 0, 0, 0 };
    ulong[4] s_sc  = { (ulong)c.n_heads * n * n, 0, 0, 0 };

    DepthActivations a;
    a.hidden    = llm::create_f32_tensor(ctx, s_nd, 1)!!;
    a.norm_out  = llm::create_f32_tensor(ctx, s_nd, 1)!!;
    a.qkv       = llm::create_f32_tensor(ctx, s_qkv, 1)!!;
    a.q         = llm::create_f32_tensor(ctx, s_nd, 1)!!;
    a.k         = llm::create_f32_tensor(ctx, s_nd, 1)!!;
    a.v         = llm::create_f32_tensor(ctx, s_nd, 1)!!;
    a.attn_out  = llm::create_f32_tensor(ctx, s_nd, 1)!!;
    a.scores    = llm::create_f32_tensor(ctx, s_sc, 1)!!;
    a.branch    = llm::create_f32_tensor(ctx, s_nd, 1)!!;
    a.ffn       = llm::create_f32_tensor(ctx, s_ffn, 1)!!;
    a.patch_tok = llm::create_f32_tensor(ctx, s_nd, 1)!!;
    for (uint i = 0; i < 4; i++) { a.taps[i] = llm::create_f32_tensor(ctx, s_nd, 1)!!; }
    return a;
}

// --- Helpers ---

// Upload a CPU float[] as a 1-D F32 GPU tensor.
fn llm::Tensor upload_f32_array(llm::DeviceContext* ctx, float[] data) {
    ulong[4] shape = { (ulong)data.len, 0, 0, 0 };
    vk::Memory buf = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT | vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: data.ptr,
        data_size: (sz)(data.len * 4),
    )!!;
    return {
        .gpu_buffer = buf,
        .dtype = llm::GGML_F32,
        .n_dims = 1,
        .shape = shape,
        .size_bytes = data.len * 4,
    };
}

// Copy `size` bytes from src[src_off] to dst[dst_off] (byte offsets).
fn void copy_region(llm::DeviceContext* ctx, vk::Memory src, vk::Memory dst, ulong src_off, ulong dst_off, ulong size) {
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, src.buffer, dst.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = src_off, .dstOffset = dst_off, .size = size }});
    }!!;
}

// Read an F32 tensor from a safetensors file into a CPU float[] (caller owns).
fn float[] read_st_f32(llm::SafetensorsFile* sf, String name) {
    llm::SafetensorsTensorInfo* info = sf.find_tensor(name)!!;
    usz n = (usz)(info.data_end - info.data_start) / 4;
    float[] out = mem::new_array(float, (sz)n);
    mem::copy(out.ptr, sf.data_base + (usz)info.data_start, (sz)(n * 4));
    return out;
}

// Result of preprocessing: GPU image + interpolated pos-embed + grid + source dims.
struct PreprocInfo {
    llm::Tensor image;      // [3, gh*14, gw*14]
    llm::Tensor pos_embed;  // [1 + gh*gw, dim]
    uint gh;
    uint gw;
    uint orig_w;
    uint orig_h;
}

// Bilinear (align_corners=False) interpolation of the patch positional embeddings
// from the pretrained src_grid x src_grid layout to gh x gw; CLS row copied through.
fn float[] interp_pos_embed(float[] pos, uint src_grid, uint gh, uint gw, uint dim) {
    usz n_out = (usz)(1 + gh * gw);
    float[] out = mem::new_array(float, (sz)(n_out * dim));
    // CLS row (token 0)
    for (uint c = 0; c < dim; c++) out[c] = pos[c];

    for (uint oy = 0; oy < gh; oy++) {
        float sy = ((float)oy + 0.5f) * (float)src_grid / (float)gh - 0.5f;
        if (sy < 0.0f) sy = 0.0f;
        uint y0 = (uint)sy;
        uint y1 = y0 + 1 < src_grid ? y0 + 1 : src_grid - 1;
        float dy = sy - (float)y0;
        for (uint ox = 0; ox < gw; ox++) {
            float sx = ((float)ox + 0.5f) * (float)src_grid / (float)gw - 0.5f;
            if (sx < 0.0f) sx = 0.0f;
            uint x0 = (uint)sx;
            uint x1 = x0 + 1 < src_grid ? x0 + 1 : src_grid - 1;
            float dx = sx - (float)x0;

            usz t00 = (usz)(1 + y0 * src_grid + x0) * dim;
            usz t01 = (usz)(1 + y0 * src_grid + x1) * dim;
            usz t10 = (usz)(1 + y1 * src_grid + x0) * dim;
            usz t11 = (usz)(1 + y1 * src_grid + x1) * dim;
            usz dst = (usz)(1 + oy * gw + ox) * dim;
            for (uint c = 0; c < dim; c++) {
                float v = pos[t00 + c] * (1.0f - dy) * (1.0f - dx)
                        + pos[t01 + c] * (1.0f - dy) * dx
                        + pos[t10 + c] * dy * (1.0f - dx)
                        + pos[t11 + c] * dy * dx;
                out[dst + c] = v;
            }
        }
    }
    return out;
}

// Load image, aspect-preserving resize to fit the (pretrain_grid*patch) box with both
// sides multiples of patch_size, ImageNet-normalize, and interpolate pos-embed to grid.
fn PreprocInfo? preprocess_image(llm::DeviceContext* ctx, String path, DepthModel* model) {
    DepthConfig* c = &model.config;
    uint patch = c.patch_size;
    uint box = c.infer_grid * patch;             // longest-side inference resolution

    image::Image img = llm::load_image(path)!!;
    uint ow = img.width;
    uint oh = img.height;

    // Fit inside the box, preserve aspect, snap each side to a multiple of patch.
    uint longest = ow > oh ? ow : oh;
    float scale = (float)box / (float)longest;
    uint gw = (uint)((float)ow * scale / (float)patch + 0.5f);
    uint gh = (uint)((float)oh * scale / (float)patch + 0.5f);
    if (gw < 1) gw = 1;
    if (gh < 1) gh = 1;
    if (gw > c.infer_grid) gw = c.infer_grid;
    if (gh > c.infer_grid) gh = c.infer_grid;
    uint tw = gw * patch;
    uint th = gh * patch;

    image::Image rs = img.resize(tw, th);
    img.free();
    float[] data = llm::image_to_float_nchw(&rs)!!;   // [3, th, tw] in [0,1]
    rs.free();

    usz tp = (usz)tw * th;
    for (uint ch = 0; ch < 3; ch++) {
        float m = c.image_mean[ch];
        float sd = c.image_std[ch];
        for (usz i = 0; i < tp; i++) {
            data[(usz)ch * tp + i] = (data[(usz)ch * tp + i] - m) / sd;
        }
    }
    llm::Tensor t = upload_f32_array(ctx, data);
    mem::free(data);

    float[] pos = interp_pos_embed(model.encoder.pos_cpu, c.pretrain_grid, gh, gw, c.dim);
    llm::Tensor pos_t = upload_f32_array(ctx, pos);
    mem::free(pos);

    return { .image = t, .pos_embed = pos_t, .gh = gh, .gw = gw, .orig_w = ow, .orig_h = oh };
}

// --- One transformer block (in place on acts.hidden) ---

fn void? DepthModel.forward_block(&self, uint layer, uint n) {
    DepthConfig* c = &self.config;
    DepthKernels* k = &self.kernels;
    DepthActivations* a = &self.acts;
    DinoBlock* b = &self.encoder.blocks[layer];
    uint dim = c.dim;
    uint qkv_dim = 3 * dim;
    uint total = n * dim;

    llm::RenderGraph g;
    g.init(self.ctx)!!;

    BatchLayerNormAffinePC ln_pc = { .dim = dim, .eps = c.ln_eps, .n_rows = n };
    BroadcastBiasPC bb_dim  = { .dim = dim, .n_rows = n };
    BroadcastScalePC bs_dim = { .dim = dim, .n_rows = n };
    llm::ResidualPC res_pc  = { .n = total };

    // --- Attention ---
    // LN1 (gamma then beta)
    g.dispatch(&k.batch_ln_affine, { &a.hidden, &b.norm1_w, &a.norm_out }, &ln_pc, n);
    g.dispatch(&k.broadcast_bias, { &a.norm_out, &b.norm1_b }, &bb_dim, n, llm::ceil_div(dim, 256));

    // QKV projection + bias
    BatchMatMulPC qkv_mm = { .out_dim = qkv_dim, .in_dim = dim, .seq_len = n, .row_offset = 0 };
    g.dispatch(&k.batch_matmul, { &b.qkv_w, &a.norm_out, &a.qkv }, &qkv_mm, n, qkv_dim);
    BroadcastBiasPC bb_qkv = { .dim = qkv_dim, .n_rows = n };
    g.dispatch(&k.broadcast_bias, { &a.qkv, &b.qkv_b }, &bb_qkv, n, llm::ceil_div(qkv_dim, 256));

    // Split into Q, K, V
    SplitQkvPC sq_pc = { .dim = dim, .n_rows = n };
    g.dispatch(&k.split_qkv, { &a.qkv, &a.q, &a.k, &a.v }, &sq_pc, llm::ceil_div(total, 256));

    // Transpose Q,K,V to [heads, seq, head_dim] (via attn_out scratch + copy back)
    TransposeHeadsPC th_fwd = { .seq_len = n, .n_heads = c.n_heads, .head_dim = c.head_dim, .direction = 0 };
    g.dispatch(&k.transpose_heads, { &a.q, &a.attn_out }, &th_fwd, llm::ceil_div(total, 256));
    g.copy(&a.attn_out, &a.q, (ulong)total * 4);
    g.dispatch(&k.transpose_heads, { &a.k, &a.attn_out }, &th_fwd, llm::ceil_div(total, 256));
    g.copy(&a.attn_out, &a.k, (ulong)total * 4);
    g.dispatch(&k.transpose_heads, { &a.v, &a.attn_out }, &th_fwd, llm::ceil_div(total, 256));
    g.copy(&a.attn_out, &a.v, (ulong)total * 4);

    // Scaled dot-product attention -> attn_out [heads, seq, head_dim]
    float scale = 1.0f / std::math::sqrt((float)c.head_dim);
    SpatialAttentionPC sa_pc = { .head_dim = c.head_dim, .n_heads = c.n_heads, .seq_len = n, .scale = scale };
    g.dispatch(&k.spatial_attention, { &a.q, &a.k, &a.v, &a.scores, &a.attn_out }, &sa_pc, c.n_heads * n);

    // Transpose back -> norm_out [seq, dim]
    TransposeHeadsPC th_inv = { .seq_len = n, .n_heads = c.n_heads, .head_dim = c.head_dim, .direction = 1 };
    g.dispatch(&k.transpose_heads, { &a.attn_out, &a.norm_out }, &th_inv, llm::ceil_div(total, 256));

    // Output projection + bias -> branch
    BatchMatMulPC proj_mm = { .out_dim = dim, .in_dim = dim, .seq_len = n, .row_offset = 0 };
    g.dispatch(&k.batch_matmul, { &b.proj_w, &a.norm_out, &a.branch }, &proj_mm, n, dim);
    g.dispatch(&k.broadcast_bias, { &a.branch, &b.proj_b }, &bb_dim, n, llm::ceil_div(dim, 256));

    // LayerScale (ls1) then residual: hidden += ls1 * branch
    g.dispatch(&k.broadcast_scale, { &a.branch, &b.ls1 }, &bs_dim, n, llm::ceil_div(dim, 256));
    g.dispatch(&k.shared.residual_add, { &a.hidden, &a.branch }, &res_pc, llm::ceil_div(total, 256));

    // --- MLP ---
    g.dispatch(&k.batch_ln_affine, { &a.hidden, &b.norm2_w, &a.norm_out }, &ln_pc, n);
    g.dispatch(&k.broadcast_bias, { &a.norm_out, &b.norm2_b }, &bb_dim, n, llm::ceil_div(dim, 256));

    // fc1 + bias -> ffn, GELU
    BatchMatMulPC fc1_mm = { .out_dim = c.ffn_dim, .in_dim = dim, .seq_len = n, .row_offset = 0 };
    g.dispatch(&k.batch_matmul, { &b.fc1_w, &a.norm_out, &a.ffn }, &fc1_mm, n, c.ffn_dim);
    BroadcastBiasPC bb_ffn = { .dim = c.ffn_dim, .n_rows = n };
    g.dispatch(&k.broadcast_bias, { &a.ffn, &b.fc1_b }, &bb_ffn, n, llm::ceil_div(c.ffn_dim, 256));
    llm::GeluPC gelu_pc = { .n = n * c.ffn_dim };
    g.dispatch(&k.shared.gelu, { &a.ffn }, &gelu_pc, llm::ceil_div(n * c.ffn_dim, 256));

    // fc2 + bias -> branch
    BatchMatMulPC fc2_mm = { .out_dim = dim, .in_dim = c.ffn_dim, .seq_len = n, .row_offset = 0 };
    g.dispatch(&k.batch_matmul, { &b.fc2_w, &a.ffn, &a.branch }, &fc2_mm, n, dim);
    g.dispatch(&k.broadcast_bias, { &a.branch, &b.fc2_b }, &bb_dim, n, llm::ceil_div(dim, 256));

    // LayerScale (ls2) then residual: hidden += ls2 * branch
    g.dispatch(&k.broadcast_scale, { &a.branch, &b.ls2 }, &bs_dim, n, llm::ceil_div(dim, 256));
    g.dispatch(&k.shared.residual_add, { &a.hidden, &a.branch }, &res_pc, llm::ceil_div(total, 256));

    g.flush()!!;
}

// --- Full encoder: image -> 4 tapped, final-normed features in acts.taps ---
// gh/gw = patch grid height/width; pos_embed must be the interpolated [1+gh*gw, dim].
fn void? DepthModel.encode(&self, llm::Tensor* image_chw, llm::Tensor* pos_embed, uint gh, uint gw) {
    DepthConfig* c = &self.config;
    DepthKernels* k = &self.kernels;
    DepthActivations* a = &self.acts;
    uint dim = c.dim;
    uint n_patches = gh * gw;
    uint n = 1 + n_patches;

    // Patch embed -> patch_tok [n_patches, dim]
    {
        llm::RenderGraph g;
        g.init(self.ctx)!!;
        PatchEmbedPC pe = {
            .height = gh * c.patch_size, .width = gw * c.patch_size, .patch_size = c.patch_size,
            .out_channels = dim, .in_channels = 3, .patches_h = gh, .patches_w = gw,
        };
        g.dispatch(&k.patch_embed, { image_chw, &self.encoder.patch_w, &a.patch_tok }, &pe, n_patches, dim);
        // patch_embed_conv2d does NOT add the conv bias — add it here.
        BroadcastBiasPC pb = { .dim = dim, .n_rows = n_patches };
        g.dispatch(&k.broadcast_bias, { &a.patch_tok, &self.encoder.patch_b }, &pb, n_patches, llm::ceil_div(dim, 256));
        g.flush()!!;
    }

    // Assemble tokens: hidden[0] = CLS, hidden[1..] = patches; then += pos_embed.
    copy_region(self.ctx, self.encoder.cls_token.gpu_buffer, a.hidden.gpu_buffer, 0, 0, (ulong)dim * 4);
    copy_region(self.ctx, a.patch_tok.gpu_buffer, a.hidden.gpu_buffer, 0, (ulong)dim * 4, (ulong)n_patches * dim * 4);
    {
        llm::RenderGraph g;
        g.init(self.ctx)!!;
        llm::ResidualPC pos_pc = { .n = n * dim };
        g.dispatch(&k.shared.residual_add, { &a.hidden, pos_embed }, &pos_pc, llm::ceil_div(n * dim, 256));
        g.flush()!!;
    }

    // Transformer blocks; tap configured layers.
    uint tap_i = 0;
    for (uint l = 0; l < c.n_layers; l++) {
        self.forward_block(l, n)!!;
        for (uint t = 0; t < 4; t++) {
            if (c.tap_layers[t] == l) {
                copy_region(self.ctx, a.hidden.gpu_buffer, a.taps[tap_i].gpu_buffer, 0, 0, (ulong)n * dim * 4);
                tap_i++;
            }
        }
    }

    // Apply final LayerNorm (norm_w, norm_b) to each tapped feature, in place.
    BatchLayerNormAffinePC ln_pc = { .dim = dim, .eps = c.ln_eps, .n_rows = n };
    BroadcastBiasPC bb_dim = { .dim = dim, .n_rows = n };
    for (uint t = 0; t < 4; t++) {
        llm::RenderGraph g;
        g.init(self.ctx)!!;
        g.dispatch(&k.batch_ln_affine, { &a.taps[t], &self.encoder.norm_w, &a.taps[t] }, &ln_pc, n);
        g.dispatch(&k.broadcast_bias, { &a.taps[t], &self.encoder.norm_b }, &bb_dim, n, llm::ceil_div(dim, 256));
        g.flush()!!;
    }
}

// =====================================================================
// Phase 3 — DPT decoder head
// =====================================================================

struct DptActivations {
    llm::Tensor chw;        // [384, grid, grid]   tokens reshaped to CHW
    llm::Tensor proj;       // projects[i] output   (<= 384*grid*grid)
    llm::Tensor resize_buf; // resize_layers output (<= 48*4grid*4grid)
    llm::Tensor rn1;        // [64, 4g, 4g]
    llm::Tensor rn2;        // [64, 2g, 2g]
    llm::Tensor rn3;        // [64, g, g]
    llm::Tensor rn4;        // [64, g/2-ish, ...]
    llm::Tensor a;          // conv-chain scratch  [64, 2*4g, 2*4g]
    llm::Tensor b;
    llm::Tensor cc;
    llm::Tensor fused;
    llm::Tensor up;
    llm::Tensor pathP;
    llm::Tensor pathQ;
    llm::Tensor o1;         // [32, 2*4g, 2*4g]
    llm::Tensor interp;     // [32, side, side]
    llm::Tensor convb;      // [32, side, side]
    llm::Tensor depth;      // [1, side, side]
}

fn DptActivations? allocate_dpt_activations(llm::DeviceContext* ctx, uint grid, uint side) {
    uint g2 = grid * 2;
    uint g4 = grid * 4;                 // largest reassembled feature side (148)
    uint fmax = g4 * 2;                 // rn1 is 4g; refinenet1 upsamples 4g -> 8g (296)
    ulong c64  = (ulong)64 * fmax * fmax;       // [64, 296, 296]
    ulong c32s = (ulong)32 * side * side;       // [32, 518, 518]

    DptActivations d;
    d.chw        = llm::create_f32_tensor(ctx, { (ulong)384 * grid * grid, 0, 0, 0 }, 1)!!;
    d.proj       = llm::create_f32_tensor(ctx, { (ulong)384 * grid * grid, 0, 0, 0 }, 1)!!;
    d.resize_buf = llm::create_f32_tensor(ctx, { (ulong)48 * g4 * g4, 0, 0, 0 }, 1)!!;
    d.rn1        = llm::create_f32_tensor(ctx, { (ulong)64 * g4 * g4, 0, 0, 0 }, 1)!!;
    d.rn2        = llm::create_f32_tensor(ctx, { (ulong)64 * g2 * g2, 0, 0, 0 }, 1)!!;
    d.rn3        = llm::create_f32_tensor(ctx, { (ulong)64 * grid * grid, 0, 0, 0 }, 1)!!;
    d.rn4        = llm::create_f32_tensor(ctx, { (ulong)64 * grid * grid, 0, 0, 0 }, 1)!!;
    d.a          = llm::create_f32_tensor(ctx, { c64, 0, 0, 0 }, 1)!!;
    d.b          = llm::create_f32_tensor(ctx, { c64, 0, 0, 0 }, 1)!!;
    d.cc         = llm::create_f32_tensor(ctx, { c64, 0, 0, 0 }, 1)!!;
    d.fused      = llm::create_f32_tensor(ctx, { c64, 0, 0, 0 }, 1)!!;
    d.up         = llm::create_f32_tensor(ctx, { c64, 0, 0, 0 }, 1)!!;
    d.pathP      = llm::create_f32_tensor(ctx, { c64, 0, 0, 0 }, 1)!!;
    d.pathQ      = llm::create_f32_tensor(ctx, { c64, 0, 0, 0 }, 1)!!;
    d.o1         = llm::create_f32_tensor(ctx, { (ulong)32 * fmax * fmax, 0, 0, 0 }, 1)!!;
    d.interp     = llm::create_f32_tensor(ctx, { c32s, 0, 0, 0 }, 1)!!;
    d.convb      = llm::create_f32_tensor(ctx, { c32s, 0, 0, 0 }, 1)!!;
    d.depth      = llm::create_f32_tensor(ctx, { (ulong)side * side, 0, 0, 0 }, 1)!!;
    return d;
}

// --- Dispatch helpers ---

fn void dpt_conv(llm::RenderGraph* g, DepthKernels* k, DptConv* c, llm::Tensor* inp, llm::Tensor* out,
        uint in_h, uint in_w, uint out_h, uint out_w) {
    llm::Tensor* bias = c.has_bias ? &c.bias : &c.weight;  // placeholder when no bias
    llm::pipelines::Conv2dPC pc = {
        .in_c = c.in_c, .out_c = c.out_c, .in_h = in_h, .in_w = in_w,
        .kH = c.k, .kW = c.k, .stride = c.stride, .pad = c.pad, .groups = 1,
        .out_h = out_h, .out_w = out_w, .has_bias = c.has_bias ? 1 : 0,
    };
    g.dispatch(&k.conv2d, { &c.weight, bias, inp, out }, &pc, llm::ceil_div(out_h * out_w, 256), c.out_c);
}

fn void dpt_convT(llm::RenderGraph* g, DepthKernels* k, DptConv* c, llm::Tensor* inp, llm::Tensor* out,
        uint in_h, uint in_w) {
    llm::Tensor* bias = c.has_bias ? &c.bias : &c.weight;
    ConvTransposePC pc = { .in_c = c.in_c, .out_c = c.out_c, .in_h = in_h, .in_w = in_w, .k = c.k, .has_bias = c.has_bias ? 1 : 0 };
    uint oh = in_h * c.k; uint ow = in_w * c.k;
    g.dispatch(&k.conv_transpose, { &c.weight, bias, inp, out }, &pc, llm::ceil_div(oh * ow, 256), c.out_c);
}

fn void dpt_bilinear(llm::RenderGraph* g, DepthKernels* k, llm::Tensor* inp, llm::Tensor* out,
        uint ch, uint in_h, uint in_w, uint out_h, uint out_w) {
    BilinearPC pc = { .channels = ch, .in_h = in_h, .in_w = in_w, .out_h = out_h, .out_w = out_w };
    g.dispatch(&k.bilinear, { inp, out }, &pc, llm::ceil_div(out_h * out_w, 256), ch);
}

fn void dpt_relu(llm::RenderGraph* g, DepthKernels* k, llm::Tensor* buf, uint n) {
    llm::pipelines::ReluPC pc = { .n = n };
    g.dispatch(&k.relu, { buf }, &pc, llm::ceil_div(n, 256));
}

// Residual conv unit (64ch): dst = x + conv2(relu(conv1(relu(x)))). a,b are scratch.
fn void dpt_rcu(llm::RenderGraph* g, DepthKernels* k, DptConv* c1, DptConv* c2,
        llm::Tensor* x, llm::Tensor* dst, llm::Tensor* a, llm::Tensor* b, uint h, uint w) {
    uint n = 64 * h * w;
    g.copy(x, a, (ulong)n * 4); dpt_relu(g, k, a, n);
    dpt_conv(g, k, c1, a, b, h, w, h, w); dpt_relu(g, k, b, n);
    dpt_conv(g, k, c2, b, a, h, w, h, w);
    if (dst != x) g.copy(x, dst, (ulong)n * 4);
    llm::ResidualPC rp = { .n = n };
    g.dispatch(&k.shared.residual_add, { dst, a }, &rp, llm::ceil_div(n, 256));
}

// FeatureFusionBlock: out_buf = out_conv( upsample( resConfUnit2( primary [+ resConfUnit1(feat)] ) ) )
fn void dpt_fuse(llm::RenderGraph* g, DepthKernels* k, RefineNet* rn,
        llm::Tensor* primary, llm::Tensor* feat, bool has_feat,
        uint h, uint w, uint out_h, uint out_w, DptActivations* d, llm::Tensor* out_buf) {
    uint n = 64 * h * w;
    g.copy(primary, &d.fused, (ulong)n * 4);
    if (has_feat) {
        dpt_rcu(g, k, &rn.rcu1_conv1, &rn.rcu1_conv2, feat, &d.cc, &d.a, &d.b, h, w);
        llm::ResidualPC rp = { .n = n };
        g.dispatch(&k.shared.residual_add, { &d.fused, &d.cc }, &rp, llm::ceil_div(n, 256));
    }
    dpt_rcu(g, k, &rn.rcu2_conv1, &rn.rcu2_conv2, &d.fused, &d.fused, &d.a, &d.b, h, w);
    dpt_bilinear(g, k, &d.fused, &d.up, 64, h, w, out_h, out_w);
    dpt_conv(g, k, &rn.out_conv, &d.up, out_buf, out_h, out_w, out_h, out_w);
}

// --- Full DPT head: acts.taps -> dpt.depth [1, gh*patch, gw*patch] ---
fn void? DepthModel.decode(&self, uint gh, uint gw) {
    DepthConfig* c = &self.config;
    DepthKernels* k = &self.kernels;
    DptHead* hd = &self.head;
    DptActivations* d = &self.dpt;
    uint sh = gh * c.patch_size;
    uint sw = gw * c.patch_size;
    uint np = gh * gw;

    llm::Tensor*[4] rn = { &d.rn1, &d.rn2, &d.rn3, &d.rn4 };
    uint[4] rh; uint[4] rw;

    // Reassemble: tokens -> CHW -> projects -> resize -> layerN_rn (-> rn[i])
    for (uint i = 0; i < 4; i++) {
        llm::RenderGraph g;
        g.init(self.ctx)!!;

        TokensToChwPC tc = { .dim = c.dim, .n_patches = np };
        g.dispatch(&k.tokens_to_chw, { &self.acts.taps[i], &d.chw }, &tc, llm::ceil_div(c.dim * np, 256));

        dpt_conv(&g, k, &hd.projects[i], &d.chw, &d.proj, gh, gw, gh, gw);

        uint ph; uint pw;
        switch (i) {
            case 0: dpt_convT(&g, k, &hd.resize0, &d.proj, &d.resize_buf, gh, gw); ph = gh * 4; pw = gw * 4;
            case 1: dpt_convT(&g, k, &hd.resize1, &d.proj, &d.resize_buf, gh, gw); ph = gh * 2; pw = gw * 2;
            case 2: g.copy(&d.proj, &d.resize_buf, (ulong)192 * np * 4); ph = gh; pw = gw;
            default: {
                uint oh = (gh + 2 - 3) / 2 + 1;
                uint ovw = (gw + 2 - 3) / 2 + 1;
                dpt_conv(&g, k, &hd.resize3, &d.proj, &d.resize_buf, gh, gw, oh, ovw);
                ph = oh; pw = ovw;
            }
        }

        dpt_conv(&g, k, &hd.layer_rn[i], &d.resize_buf, rn[i], ph, pw, ph, pw);
        rh[i] = ph; rw[i] = pw;
        g.flush()!!;
    }

    // Top-down fusion
    llm::RenderGraph g;
    g.init(self.ctx)!!;
    dpt_fuse(&g, k, &hd.refinenet[3], &d.rn4, null, false, rh[3], rw[3], rh[2], rw[2], d, &d.pathP);
    dpt_fuse(&g, k, &hd.refinenet[2], &d.pathP, &d.rn3, true, rh[2], rw[2], rh[1], rw[1], d, &d.pathQ);
    dpt_fuse(&g, k, &hd.refinenet[1], &d.pathQ, &d.rn2, true, rh[1], rw[1], rh[0], rw[0], d, &d.pathP);
    dpt_fuse(&g, k, &hd.refinenet[0], &d.pathP, &d.rn1, true, rh[0], rw[0], rh[0] * 2, rw[0] * 2, d, &d.pathQ);

    uint fh = rh[0] * 2; uint fw = rw[0] * 2;

    // output_conv1 (3x3 pad1, 64->32)
    dpt_conv(&g, k, &hd.output_conv1, &d.pathQ, &d.o1, fh, fw, fh, fw);
    // upsample to model-input resolution (sh, sw)
    dpt_bilinear(&g, k, &d.o1, &d.interp, 32, fh, fw, sh, sw);
    // output_conv2: (3x3 pad1 32->32) + relu, then (1x1 32->1) + relu
    dpt_conv(&g, k, &hd.output_conv2_0, &d.interp, &d.convb, sh, sw, sh, sw);
    dpt_relu(&g, k, &d.convb, 32 * sh * sw);
    dpt_conv(&g, k, &hd.output_conv2_2, &d.convb, &d.depth, sh, sw, sh, sw);
    dpt_relu(&g, k, &d.depth, sh * sw);
    g.flush()!!;
}

// =====================================================================
// Phase 4 — depth map -> PNG
// =====================================================================

// Resize the depth map ([sh, sw]) back to the original resolution (ow x oh) with the
// bilinear (align_corners=True) kernel, min/max normalize to [0,255] (near = bright,
// matching Depth Anything's inverse-depth convention) and save as a grayscale PNG.
fn void? save_depth_png(llm::DeviceContext* ctx, DepthKernels* k, llm::Tensor* depth,
        uint sh, uint sw, uint ow, uint oh, String path) {
    usz count = (usz)ow * oh;

    // GPU bilinear resize [1, sh, sw] -> [1, oh, ow]
    llm::Tensor full = llm::create_f32_tensor(ctx, { (ulong)count, 0, 0, 0 }, 1)!!;
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        dpt_bilinear(&g, k, depth, &full, 1, sh, sw, oh, ow);
        g.flush()!!;
    }

    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: (sz)(count * 4),
    )!!;
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, full.gpu_buffer.buffer, staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = count * 4 }});
    }!!;
    float* m = (float*)staging.data();

    float mn = m[0]; float mx = m[0];
    for (usz i = 0; i < count; i++) {
        if (m[i] < mn) mn = m[i];
        if (m[i] > mx) mx = m[i];
    }
    float range = mx - mn;
    if (range < 1e-8f) range = 1.0f;

    char[] pixels = mem::new_array(char, (sz)count);
    for (usz i = 0; i < count; i++) {
        float v = (m[i] - mn) / range;       // [0,1], near (large inverse depth) -> bright
        pixels[i] = (char)(v * 255.0f + 0.5f);
    }
    staging.free();

    image::Image img = {
        .width = ow,
        .height = oh,
        .bit_depth = 8,
        .format = image::PixelFormat.GRAYSCALE,
        .pixels = pixels,
    };
    png::save_file(&img, path)!!;
    mem::free(pixels);
    io::printfn("  wrote %s (%dx%d grayscale, depth range %.3f..%.3f)", path, ow, oh, mn, mx);
}

// =====================================================================
// Texture height-map pipeline (detrend + albedo detail blend)
//
// Depth Anything predicts *scene* depth, which for a flat, head-on game
// texture shows up as a smooth global gradient (e.g. top brighter than
// bottom) plus low-frequency blobs — useless as a tiling height/displacement
// map. This path removes that low-frequency component (high-pass via a large
// box blur) so the result reads as surface relief, then optionally folds in
// the albedo's fine luminance detail that the model can't resolve.
// =====================================================================

// Clamped-window box blur via a double-precision summed-area table. O(W*H).
fn void box_blur(float[] src, float[] dst, uint w, uint h, int r) {
    usz sw = (usz)w + 1;
    usz sat_len = sw * ((usz)h + 1);
    double[] sat = mem::new_array(double, (sz)sat_len);
    for (uint y = 0; y < h; y++) {
        double rowsum = 0.0;
        usz cur = (usz)(y + 1) * sw;
        usz prev = (usz)y * sw;
        usz srow = (usz)y * w;
        for (uint x = 0; x < w; x++) {
            rowsum += (double)src[srow + x];
            sat[cur + (usz)x + 1] = sat[prev + (usz)x + 1] + rowsum;
        }
    }
    for (uint y = 0; y < h; y++) {
        int ya = (int)y - r; if (ya < 0) ya = 0;
        int yb = (int)y + r; if (yb > (int)h - 1) yb = (int)h - 1;
        usz rb = (usz)(yb + 1) * sw;
        usz ra = (usz)ya * sw;
        usz drow = (usz)y * w;
        for (uint x = 0; x < w; x++) {
            int xa = (int)x - r; if (xa < 0) xa = 0;
            int xb = (int)x + r; if (xb > (int)w - 1) xb = (int)w - 1;
            usz ca = (usz)xa;
            usz cb = (usz)(xb + 1);
            double s = sat[rb + cb] - sat[ra + cb] - sat[rb + ca] + sat[ra + ca];
            double area = (double)((yb - ya + 1) * (xb - xa + 1));
            dst[drow + (usz)x] = (float)(s / area);
        }
    }
    mem::free(sat);
}

// Normalize `buf` in place to [lo, hi] by its own min/max.
fn void normalize_to(float[] buf, usz count, float lo, float hi) {
    float mn = buf[0]; float mx = buf[0];
    for (usz i = 0; i < count; i++) {
        if (buf[i] < mn) mn = buf[i];
        if (buf[i] > mx) mx = buf[i];
    }
    float range = mx - mn;
    if (range < 1e-8f) range = 1.0f;
    float span = hi - lo;
    for (usz i = 0; i < count; i++) buf[i] = lo + (buf[i] - mn) / range * span;
}

// Build a tiling height/relief map. Two height sources are blended by `albedo_w`:
//   * model depth (good for real 3D shape, e.g. rounded stones): high-passed by a
//     box blur of radius detrend_frac*longest-side to strip the scene-depth gradient.
//   * albedo luminance (good for flat-but-detailed surfaces, e.g. wood grain, carving,
//     cracks, nails — which the depth model cannot resolve): partially flattened by
//     `albedo_flatten` (1 = fully local relief, 0 = keep all plank-to-plank tone).
// `detail_w` adds an extra fine high-pass of the luminance on top for crispness.
//   rocks:  albedo_w 0, detail ~0.3   |   flat carved wood: albedo_w ~1, flatten ~0.7
fn void? save_height_map(llm::DeviceContext* ctx, DepthKernels* k, llm::Tensor* depth,
        uint sh, uint sw, uint ow, uint oh, String albedo_path,
        float detrend_frac, float detail_w, float albedo_w, float albedo_flatten, String path) {
    usz count = (usz)ow * oh;
    uint longest = ow > oh ? ow : oh;
    int r = (int)(detrend_frac * (float)longest + 0.5f);
    if (r < 1) r = 1;

    // --- Model-depth base: GPU resize [sh,sw] -> [oh,ow], download, high-pass, normalize.
    float[] base = mem::new_array(float, (sz)count);
    if (albedo_w < 1.0f) {
        llm::Tensor full = llm::create_f32_tensor(ctx, { (ulong)count, 0, 0, 0 }, 1)!!;
        {
            llm::RenderGraph g;
            g.init(ctx)!!;
            dpt_bilinear(&g, k, depth, &full, 1, sh, sw, oh, ow);
            g.flush()!!;
        }
        vk::Memory staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null,
            data_size: (sz)(count * 4),
        )!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
            vk::cmdCopyBuffer(cmd, full.gpu_buffer.buffer, staging.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = count * 4 }});
        }!!;
        float* sp = (float*)staging.data();
        for (usz i = 0; i < count; i++) base[i] = sp[i];
        staging.free();

        float[] mean = mem::new_array(float, (sz)count);
        box_blur(base, mean, ow, oh, r);
        for (usz i = 0; i < count; i++) base[i] -= mean[i];   // strip scene-depth gradient
        mem::free(mean);
        normalize_to(base, count, 0.0f, 1.0f);
    }

    // --- Albedo luminance (needed if it contributes to the base or to the detail pass).
    bool need_albedo = albedo_w > 0.0f || detail_w > 0.0f;
    float[] lum;
    if (need_albedo) {
        image::Image img = llm::load_image(albedo_path)!!;
        image::Image rs = img.resize(ow, oh);
        img.free();
        float[] rgb = llm::image_to_float_nchw(&rs)!!;   // [3, oh, ow] in [0,1]
        rs.free();
        lum = mem::new_array(float, (sz)count);
        for (usz i = 0; i < count; i++) {
            lum[i] = 0.299f * rgb[i] + 0.587f * rgb[count + i] + 0.114f * rgb[2 * count + i];
        }
        mem::free(rgb);
    }

    // --- Albedo height base: partially flatten plank-to-plank tone, keep the grain.
    if (albedo_w > 0.0f) {
        float[] amean = mem::new_array(float, (sz)count);
        box_blur(lum, amean, ow, oh, r);
        float[] alb = mem::new_array(float, (sz)count);
        for (usz i = 0; i < count; i++) alb[i] = lum[i] - albedo_flatten * amean[i];
        mem::free(amean);
        normalize_to(alb, count, 0.0f, 1.0f);
        for (usz i = 0; i < count; i++) base[i] = base[i] * (1.0f - albedo_w) + alb[i] * albedo_w;
        mem::free(alb);
    }

    // --- Optional fine detail: small-radius luminance high-pass added on top.
    if (detail_w > 0.0f) {
        int r2 = r / 8; if (r2 < 1) r2 = 1;
        float[] lmean = mem::new_array(float, (sz)count);
        box_blur(lum, lmean, ow, oh, r2);
        float[] hp = mem::new_array(float, (sz)count);
        for (usz i = 0; i < count; i++) hp[i] = lum[i] - lmean[i];
        mem::free(lmean);
        normalize_to(hp, count, -0.5f, 0.5f);
        for (usz i = 0; i < count; i++) {
            float v = base[i] + detail_w * hp[i];
            base[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        }
        mem::free(hp);
    }
    if (need_albedo) mem::free(lum);

    // --- Quantize and save.
    char[] pixels = mem::new_array(char, (sz)count);
    for (usz i = 0; i < count; i++) pixels[i] = (char)(base[i] * 255.0f + 0.5f);
    mem::free(base);

    image::Image out = {
        .width = ow, .height = oh, .bit_depth = 8,
        .format = image::PixelFormat.GRAYSCALE, .pixels = pixels,
    };
    png::save_file(&out, path)!!;
    mem::free(pixels);
    io::printfn("  wrote %s (%dx%d height map, detrend r=%d, detail=%.2f, albedo=%.2f)",
        path, ow, oh, r, detail_w, albedo_w);
}

// --- Debug: download a buffer region and write raw F32 to a file ---
fn void? dump_raw(llm::DeviceContext* ctx, llm::Tensor* t, String path, usz count) {
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: (sz)(count * 4),
    )!!;
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, t.gpu_buffer.buffer, staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = count * 4 }});
    }!!;
    char* bytes = (char*)staging.data();
    File f = file::open(path, "wb")!!;
    f.write(bytes[0:count * 4])!!;
    f.close()!!;
    staging.free();
    io::printfn("  dumped %s (%d floats)", path, count);
}

// --- Debug: download a buffer region and print stats ---
fn void? print_stats(llm::DeviceContext* ctx, llm::Tensor* t, String label, usz count) {
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: (sz)(count * 4),
    )!!;
    ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, t.gpu_buffer.buffer, staging.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = count * 4 }});
    }!!;
    float* m = (float*)staging.data();
    float mn = m[0]; float mx = m[0]; float sum = 0.0f;
    for (usz i = 0; i < count; i++) {
        float v = m[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    io::printfn("  %s[%d]: min=%.4f max=%.4f mean=%.5f  first=[%.4f %.4f %.4f %.4f]",
        label, count, mn, mx, sum / (float)count, m[0], m[1], m[2], m[3]);
    staging.free();
}
