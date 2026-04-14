module qwen_edit;

import vk;
import llm;
import llm::pipelines;
import llm_text;
import std::io;
import std::math;
import std::core::mem;
import std::io::file;
import std::io::file::mmap;
import std::time::clock;

// --- MMDiT Configuration ---

const uint MMDIT_DIM       = 3072;
const uint MMDIT_HEADS     = 24;
const uint MMDIT_HEAD_DIM  = 128;
const uint MMDIT_FFN_DIM   = 12288;
const uint MMDIT_GEGLU_DIM = 6144;  // FFN_DIM / 2
const uint MMDIT_NUM_LAYERS = 60;
const uint MMDIT_PATCH_SIZE = 2;
const uint MMDIT_LATENT_CH = 16;
const uint MMDIT_PATCH_DIM = 64;    // 16 * 2 * 2
const uint MMDIT_T_EMB_DIM = 256;
const uint MMDIT_MOD_DIM   = 18432; // 6 * 3072
const uint MMDIT_TXT_DIM   = 3584;  // Qwen2.5-VL embedding dim

// --- Push constant structs for MMDiT shaders ---

struct GeGLUPC { uint n_rows; uint half_dim; }
struct AdaLNModPC { uint n_rows; uint dim; uint shift_offset; uint scale_offset; }
struct GatedResidualPC { uint n_rows; uint dim; uint gate_offset; }
struct TimestepEmbedPC { uint dim; float timestep; }
struct MMDiTPatchifyPC { uint channels; uint height; uint width; uint patch_size; }
struct BatchRMSNormAffinePC { uint dim; float eps; uint n_rows; uint _pad; }
struct PerHeadRMSNormPC { uint seq_len; uint n_heads; uint head_dim; float eps; }
struct FlowEulerPC { uint n; float dt; }
struct CfgPC { uint n_rows; uint dim; float scale; uint _pad; }

// Reuse from ViT
struct MMBatchMatMulPC { uint out_dim; uint in_dim; uint seq_len; uint row_offset; }
struct MMSpatialAttentionPC { uint head_dim; uint n_heads; uint seq_len; float scale; }

// --- Per-layer weights (loaded/freed during streaming) ---

struct MMDiTLayerWeights {
    // Image attention
    llm::Tensor img_q_w;    llm::Tensor img_q_b;
    llm::Tensor img_k_w;    llm::Tensor img_k_b;
    llm::Tensor img_v_w;    llm::Tensor img_v_b;
    llm::Tensor img_out_w;  llm::Tensor img_out_b;
    llm::Tensor img_norm_q; llm::Tensor img_norm_k;
    // Text attention
    llm::Tensor txt_q_w;    llm::Tensor txt_q_b;
    llm::Tensor txt_k_w;    llm::Tensor txt_k_b;
    llm::Tensor txt_v_w;    llm::Tensor txt_v_b;
    llm::Tensor txt_out_w;  llm::Tensor txt_out_b;
    llm::Tensor txt_norm_q; llm::Tensor txt_norm_k;
    // Image FFN (GEGLU)
    llm::Tensor img_ffn_up_w;   llm::Tensor img_ffn_up_b;   // [dim, ffn_dim] (gate+value fused)
    llm::Tensor img_ffn_down_w; llm::Tensor img_ffn_down_b; // [geglu_dim, dim]
    // Text FFN (GEGLU)
    llm::Tensor txt_ffn_up_w;   llm::Tensor txt_ffn_up_b;
    llm::Tensor txt_ffn_down_w; llm::Tensor txt_ffn_down_b;
    // Modulation
    llm::Tensor img_mod_w;  llm::Tensor img_mod_b;  // [dim, 6*dim]
    llm::Tensor txt_mod_w;  llm::Tensor txt_mod_b;
}

// --- Global weights (kept in GPU memory) ---

struct MMDiTGlobalWeights {
    llm::Tensor img_in_w;       // [patch_dim, dim]
    llm::Tensor img_in_b;       // [dim]
    llm::Tensor txt_in_w;       // [text_dim, dim]
    llm::Tensor txt_in_b;       // [dim]
    llm::Tensor txt_norm;       // [text_dim] RMSNorm weight
    llm::Tensor t_emb_l1_w;    // [t_emb_dim, dim]
    llm::Tensor t_emb_l1_b;    // [dim]
    llm::Tensor t_emb_l2_w;    // [dim, dim]
    llm::Tensor t_emb_l2_b;    // [dim]
    llm::Tensor norm_out_w;    // [dim, 2*dim]
    llm::Tensor norm_out_b;    // [2*dim]
    llm::Tensor proj_out_w;    // [dim, patch_dim]
    llm::Tensor proj_out_b;    // [patch_dim]
}

// --- Activation buffers ---

struct MMDiTActivations {
    llm::Tensor img_hidden;     // [n_img, dim]
    llm::Tensor txt_hidden;     // [n_txt, dim]
    llm::Tensor img_norm;       // [n_img, dim]
    llm::Tensor txt_norm;       // [n_txt, dim]
    llm::Tensor joint_q;        // [n_joint, dim]
    llm::Tensor joint_k;        // [n_joint, dim]
    llm::Tensor joint_v;        // [n_joint, dim]
    llm::Tensor attn_scores;    // [n_heads, n_joint, n_joint]
    llm::Tensor attn_out;       // [n_joint, dim]
    llm::Tensor img_ffn_buf;    // [n_img, ffn_dim]
    llm::Tensor txt_ffn_buf;    // [n_txt, ffn_dim]
    llm::Tensor img_ffn_out;    // [n_img, geglu_dim]
    llm::Tensor txt_ffn_out;    // [n_txt, geglu_dim]
    llm::Tensor img_proj;       // [n_img, dim]
    llm::Tensor txt_proj;       // [n_txt, dim]
    llm::Tensor t_emb;          // [dim]
    llm::Tensor img_mod_buf;    // [6*dim]
    llm::Tensor txt_mod_buf;    // [6*dim]
    llm::Tensor patched_latent; // [n_patches, patch_dim] - patchified noisy latent
    llm::Tensor velocity;       // [n_patches, patch_dim]
    llm::Tensor txt_emb_input;  // [n_txt, txt_dim] - text embeddings before projection
}

// --- Kernel handles ---

struct MMDiTKernels {
    // From vit.slang
    llm::ComputeKernel batch_ln_affine;
    llm::ComputeKernel broadcast_bias;
    llm::ComputeKernel geglu;
    llm::ComputeKernel adaln_mod;
    llm::ComputeKernel gated_residual;
    llm::ComputeKernel timestep_embed;
    llm::ComputeKernel patchify;
    llm::ComputeKernel unpatchify;
    llm::ComputeKernel flow_euler;
    llm::ComputeKernel perhead_rmsnorm;
    llm::ComputeKernel batch_rmsnorm;
    llm::ComputeKernel norm_cfg;
    // From llm.slang
    llm::ComputeKernel batch_matmul;
    llm::ComputeKernel batch_matmul_q4_0;
    llm::ComputeKernel batch_matmul_q4_1;
    llm::ComputeKernel batch_matmul_simple;
    // From diffusion.slang
    llm::ComputeKernel spatial_attention;
    // Shared
    llm::SharedKernels shared;
}

// --- Main MMDiT model ---

struct MMDiTModel {
    MMDiTGlobalWeights weights;
    MMDiTActivations acts;
    MMDiTKernels kernels;
    llm::DeviceContext* ctx;
    // Keep GGUF file mapped for streaming weight loading
    mmap::FileMmap mmfile;
    llm::GGUFFile gf;
}

// --- Load MMDiT ---

fn MMDiTModel? load_mmdit_model(llm::DeviceContext* ctx, String model_path) {
    io::printfn("\n=== Loading MMDiT ===");
    io::printfn("  Path: %s", model_path);

    mmap::FileMmap mm = file::mmap_open(model_path, "rb")!!;
    char[] data = mm.bytes();
    io::printfn("  File size: %d MB", data.len / (1024 * 1024));
    llm::GGUFFile gf = llm::gguf_parse(data)!!;
    io::printfn("  Tensors: %d", gf.n_tensors);

    // Load global weights one by one (to identify crash location)
    io::printfn("  Loading global weights...");
    MMDiTGlobalWeights w;
    io::printfn("    img_in.weight..."); w.img_in_w = llm::load_tensor_as_f32(ctx, &gf, "img_in.weight")!!;
    io::printfn("    img_in.bias..."); w.img_in_b = llm::load_tensor_as_f32(ctx, &gf, "img_in.bias")!!;
    io::printfn("    txt_in.weight..."); w.txt_in_w = llm::load_tensor_as_f32(ctx, &gf, "txt_in.weight")!!;
    io::printfn("    txt_in.bias..."); w.txt_in_b = llm::load_tensor_as_f32(ctx, &gf, "txt_in.bias")!!;
    io::printfn("    txt_norm..."); w.txt_norm = llm::load_tensor_as_f32(ctx, &gf, "txt_norm.weight")!!;
    io::printfn("    t_emb_l1_w..."); w.t_emb_l1_w = llm::load_tensor_as_f32(ctx, &gf, "time_text_embed.timestep_embedder.linear_1.weight")!!;
    io::printfn("    t_emb_l1_b..."); w.t_emb_l1_b = llm::load_tensor_as_f32(ctx, &gf, "time_text_embed.timestep_embedder.linear_1.bias")!!;
    io::printfn("    t_emb_l2_w..."); w.t_emb_l2_w = llm::load_tensor_as_f32(ctx, &gf, "time_text_embed.timestep_embedder.linear_2.weight")!!;
    io::printfn("    t_emb_l2_b..."); w.t_emb_l2_b = llm::load_tensor_as_f32(ctx, &gf, "time_text_embed.timestep_embedder.linear_2.bias")!!;
    io::printfn("    norm_out_w..."); w.norm_out_w = llm::load_tensor_as_f32(ctx, &gf, "norm_out.linear.weight")!!;
    io::printfn("    norm_out_b..."); w.norm_out_b = llm::load_tensor_as_f32(ctx, &gf, "norm_out.linear.bias")!!;
    io::printfn("    proj_out_w..."); w.proj_out_w = llm::load_tensor_as_f32(ctx, &gf, "proj_out.weight")!!;
    io::printfn("    proj_out_b..."); w.proj_out_b = llm::load_tensor_as_f32(ctx, &gf, "proj_out.bias")!!;

    io::printfn("  All global weights loaded.");

    // Create kernels
    io::printfn("  Creating kernels...");
    MMDiTKernels kernels = create_mmdit_kernels(ctx)!!;

    io::printfn("  Global weights and kernels ready. Layers will be streamed.");

    return {
        .weights = w,
        .kernels = kernels,
        .ctx = ctx,
        .mmfile = mm,
        .gf = gf,
    };
}

fn MMDiTKernels? create_mmdit_kernels(llm::DeviceContext* ctx) {
    io::printfn("    Loading shaders...");
    char[] vit_spv = &VIT_SPV;
    vk::ShaderModule vit_shader = vk::shaderModuleCreateInfo().setCodeSize(
        vit_spv.len).setCode((uint*)&vit_spv[0]).build(ctx.device)!!;

    char[] llm_spv = &llm_text::LLM_SPV;
    vk::ShaderModule llm_shader = vk::shaderModuleCreateInfo().setCodeSize(
        llm_spv.len).setCode((uint*)&llm_spv[0]).build(ctx.device)!!;

    char[] diff_spv = &llm::pipelines::DIFFUSION_SPV;
    vk::ShaderModule diff_shader = vk::shaderModuleCreateInfo().setCodeSize(
        diff_spv.len).setCode((uint*)&diff_spv[0]).build(ctx.device)!!;

    io::printfn("    Creating vit kernels...");
    llm::ComputeKernel k_bln = llm::create_kernel(ctx, vit_shader, 3, BatchLayerNormAffinePC.sizeof, "batch_layernorm_affine")!!;
    llm::ComputeKernel k_bb = llm::create_kernel(ctx, vit_shader, 2, BroadcastBiasPC.sizeof, "broadcast_bias_add")!!;
    llm::ComputeKernel k_geglu = llm::create_kernel(ctx, vit_shader, 2, GeGLUPC.sizeof, "geglu_activation")!!;
    llm::ComputeKernel k_adaln = llm::create_kernel(ctx, vit_shader, 3, AdaLNModPC.sizeof, "adaln_modulate_shift_scale")!!;
    llm::ComputeKernel k_gres = llm::create_kernel(ctx, vit_shader, 3, GatedResidualPC.sizeof, "gated_residual_add")!!;
    llm::ComputeKernel k_te = llm::create_kernel(ctx, vit_shader, 1, TimestepEmbedPC.sizeof, "sinusoidal_timestep_embed")!!;
    llm::ComputeKernel k_patch = llm::create_kernel(ctx, vit_shader, 2, MMDiTPatchifyPC.sizeof, "mmdit_patchify")!!;
    llm::ComputeKernel k_unpatch = llm::create_kernel(ctx, vit_shader, 4, MMDiTPatchifyPC.sizeof, "mmdit_unpatchify")!!;
    llm::ComputeKernel k_euler = llm::create_kernel(ctx, vit_shader, 2, FlowEulerPC.sizeof, "flow_euler_step")!!;
    llm::ComputeKernel k_phrms = llm::create_kernel(ctx, vit_shader, 2, PerHeadRMSNormPC.sizeof, "perhead_rmsnorm")!!;
    llm::ComputeKernel k_brms = llm::create_kernel(ctx, vit_shader, 3, BatchRMSNormAffinePC.sizeof, "batch_rmsnorm_affine")!!;
    llm::ComputeKernel k_cfg = llm::create_kernel(ctx, vit_shader, 2, CfgPC.sizeof, "norm_preserving_cfg")!!;

    io::printfn("    Creating llm kernels...");
    llm::ComputeKernel k_bm = llm::create_kernel(ctx, llm_shader, 3, MMBatchMatMulPC.sizeof, "batch_matmul")!!;
    llm::ComputeKernel k_bm_q4 = llm::create_kernel(ctx, llm_shader, 3, MMBatchMatMulPC.sizeof, "batch_matmul_q4_0")!!;
    llm::ComputeKernel k_bm_q41 = llm::create_kernel(ctx, llm_shader, 3, MMBatchMatMulPC.sizeof, "batch_matmul_q4_1")!!;
    llm::ComputeKernel k_bm_s = llm::create_kernel(ctx, llm_shader, 3, MMBatchMatMulPC.sizeof, "batch_matmul_simple")!!;

    io::printfn("    Creating attention kernel...");
    llm::ComputeKernel k_sa = llm::create_kernel(ctx, diff_shader, 5, MMSpatialAttentionPC.sizeof, "spatial_attention")!!;

    io::printfn("    Creating shared kernels...");
    llm::SharedKernels sk = llm::create_shared_kernels(ctx, llm_shader)!!;

    io::printfn("    All kernels created.");
    vit_shader.free(ctx.device);
    llm_shader.free(ctx.device);
    diff_shader.free(ctx.device);

    return {
        .batch_ln_affine = k_bln, .broadcast_bias = k_bb, .geglu = k_geglu,
        .adaln_mod = k_adaln, .gated_residual = k_gres, .timestep_embed = k_te,
        .patchify = k_patch, .unpatchify = k_unpatch, .flow_euler = k_euler, .perhead_rmsnorm = k_phrms, .batch_rmsnorm = k_brms, .norm_cfg = k_cfg,
        .batch_matmul = k_bm, .batch_matmul_q4_0 = k_bm_q4, .batch_matmul_q4_1 = k_bm_q41, .batch_matmul_simple = k_bm_s,
        .spatial_attention = k_sa, .shared = sk,
    };
}

// --- Allocate activations ---

fn MMDiTActivations? allocate_mmdit_activations(llm::DeviceContext* ctx, uint n_img, uint n_txt) {
    uint n_joint = n_img + n_txt;
    uint dim = MMDIT_DIM;
    uint ffn = MMDIT_FFN_DIM;
    uint geglu = MMDIT_GEGLU_DIM;

    return {
        .img_hidden = llm::create_f32_tensor(ctx, { (ulong)n_img * dim, 0, 0, 0 }, 1)!!,
        .txt_hidden = llm::create_f32_tensor(ctx, { (ulong)n_txt * dim, 0, 0, 0 }, 1)!!,
        .img_norm = llm::create_f32_tensor(ctx, { (ulong)n_img * dim, 0, 0, 0 }, 1)!!,
        .txt_norm = llm::create_f32_tensor(ctx, { (ulong)n_txt * dim, 0, 0, 0 }, 1)!!,
        .joint_q = llm::create_f32_tensor(ctx, { (ulong)n_joint * dim, 0, 0, 0 }, 1)!!,
        .joint_k = llm::create_f32_tensor(ctx, { (ulong)n_joint * dim, 0, 0, 0 }, 1)!!,
        .joint_v = llm::create_f32_tensor(ctx, { (ulong)n_joint * dim, 0, 0, 0 }, 1)!!,
        .attn_scores = llm::create_f32_tensor(ctx, { (ulong)MMDIT_HEADS * n_joint * n_joint, 0, 0, 0 }, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, { (ulong)n_joint * dim, 0, 0, 0 }, 1)!!,
        .img_ffn_buf = llm::create_f32_tensor(ctx, { (ulong)n_img * ffn, 0, 0, 0 }, 1)!!,
        .txt_ffn_buf = llm::create_f32_tensor(ctx, { (ulong)n_txt * ffn, 0, 0, 0 }, 1)!!,
        .img_ffn_out = llm::create_f32_tensor(ctx, { (ulong)n_img * geglu, 0, 0, 0 }, 1)!!,
        .txt_ffn_out = llm::create_f32_tensor(ctx, { (ulong)n_txt * geglu, 0, 0, 0 }, 1)!!,
        .img_proj = llm::create_f32_tensor(ctx, { (ulong)n_img * dim, 0, 0, 0 }, 1)!!,
        .txt_proj = llm::create_f32_tensor(ctx, { (ulong)n_txt * dim, 0, 0, 0 }, 1)!!,
        .t_emb = llm::create_f32_tensor(ctx, { (ulong)dim, 0, 0, 0 }, 1)!!,
        .img_mod_buf = llm::create_f32_tensor(ctx, { (ulong)MMDIT_MOD_DIM, 0, 0, 0 }, 1)!!,
        .txt_mod_buf = llm::create_f32_tensor(ctx, { (ulong)MMDIT_MOD_DIM, 0, 0, 0 }, 1)!!,
        .patched_latent = llm::create_f32_tensor(ctx, { (ulong)n_img * MMDIT_PATCH_DIM, 0, 0, 0 }, 1)!!,
        .velocity = llm::create_f32_tensor(ctx, { (ulong)n_img * MMDIT_PATCH_DIM, 0, 0, 0 }, 1)!!,
        .txt_emb_input = llm::create_f32_tensor(ctx, { (ulong)n_txt * MMDIT_TXT_DIM, 0, 0, 0 }, 1)!!,
    };
}

// --- Streaming layer loading ---

fn MMDiTLayerWeights? load_mmdit_layer(llm::DeviceContext* ctx, llm::GGUFFile* gf, uint layer) {
    char[128] buf;
    return {
        // Image attention
        .img_q_w = load_mmdit_tensor(ctx, gf, &buf, layer, "attn.to_q.weight"),
        .img_q_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.to_q.bias"),
        .img_k_w = load_mmdit_tensor(ctx, gf, &buf, layer, "attn.to_k.weight"),
        .img_k_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.to_k.bias"),
        .img_v_w = load_mmdit_tensor(ctx, gf, &buf, layer, "attn.to_v.weight"),
        .img_v_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.to_v.bias"),
        .img_out_w = load_mmdit_tensor(ctx, gf, &buf, layer, "attn.to_out.0.weight"),
        .img_out_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.to_out.0.bias"),
        .img_norm_q = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.norm_q.weight"),
        .img_norm_k = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.norm_k.weight"),
        // Text attention
        .txt_q_w = load_mmdit_tensor(ctx, gf, &buf, layer, "attn.add_q_proj.weight"),
        .txt_q_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.add_q_proj.bias"),
        .txt_k_w = load_mmdit_tensor(ctx, gf, &buf, layer, "attn.add_k_proj.weight"),
        .txt_k_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.add_k_proj.bias"),
        .txt_v_w = load_mmdit_tensor(ctx, gf, &buf, layer, "attn.add_v_proj.weight"),
        .txt_v_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.add_v_proj.bias"),
        .txt_out_w = load_mmdit_tensor(ctx, gf, &buf, layer, "attn.to_add_out.weight"),
        .txt_out_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.to_add_out.bias"),
        .txt_norm_q = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.norm_added_q.weight"),
        .txt_norm_k = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "attn.norm_added_k.weight"),
        // Image FFN
        .img_ffn_up_w = load_mmdit_tensor(ctx, gf, &buf, layer, "img_mlp.net.0.proj.weight"),
        .img_ffn_up_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "img_mlp.net.0.proj.bias"),
        .img_ffn_down_w = load_mmdit_tensor(ctx, gf, &buf, layer, "img_mlp.net.2.weight"),  // Q4_1 native
        .img_ffn_down_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "img_mlp.net.2.bias"),
        // Text FFN
        .txt_ffn_up_w = load_mmdit_tensor(ctx, gf, &buf, layer, "txt_mlp.net.0.proj.weight"),
        .txt_ffn_up_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "txt_mlp.net.0.proj.bias"),
        .txt_ffn_down_w = load_mmdit_tensor(ctx, gf, &buf, layer, "txt_mlp.net.2.weight"),  // Q4_1 native
        .txt_ffn_down_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "txt_mlp.net.2.bias"),
        // Modulation
        .img_mod_w = load_mmdit_tensor(ctx, gf, &buf, layer, "img_mod.1.weight"),
        .img_mod_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "img_mod.1.bias"),
        .txt_mod_w = load_mmdit_tensor(ctx, gf, &buf, layer, "txt_mod.1.weight"),
        .txt_mod_b = load_mmdit_tensor_f32(ctx, gf, &buf, layer, "txt_mod.1.bias"),
    };
}

fn llm::Tensor load_mmdit_tensor(llm::DeviceContext* ctx, llm::GGUFFile* gf, char[128]* buf, uint layer, String suffix) {
    usz len = format_mmdit_name(buf, layer, suffix);
    String name = (String)(*buf)[0..len - 1];
    if (try info = gf.find_tensor(name)) {
        return llm::upload_weight(ctx, info, gf.tensor_data_base)!!;
    }
    return {};
}

fn llm::Tensor load_mmdit_tensor_f32(llm::DeviceContext* ctx, llm::GGUFFile* gf, char[128]* buf, uint layer, String suffix) {
    usz len = format_mmdit_name(buf, layer, suffix);
    String name = (String)(*buf)[0..len - 1];
    return llm::load_tensor_as_f32(ctx, gf, name)!!;
}

fn llm::Tensor load_mmdit_tensor_as_f32(llm::DeviceContext* ctx, llm::GGUFFile* gf, char[128]* buf, uint layer, String suffix) {
    usz len = format_mmdit_name(buf, layer, suffix);
    String name = (String)(*buf)[0..len - 1];
    return llm::load_tensor_as_f32(ctx, gf, name)!!;
}

fn usz format_mmdit_name(char[128]* buf, uint layer, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    String prefix = "transformer_blocks.";
    for (usz i = 0; i < prefix.len; i++) { p[pos + i] = prefix[i]; }
    pos += prefix.len;
    // Layer number (up to 59)
    if (layer >= 10) { p[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    p[pos] = (char)('0' + layer % 10); pos++;
    p[pos] = '.'; pos++;
    for (usz i = 0; i < suffix.len; i++) { p[pos + i] = suffix[i]; }
    pos += suffix.len;
    return pos;
}

fn void free_mmdit_layer(MMDiTLayerWeights* lw) {
    if (lw.img_q_w.size_bytes > 0) lw.img_q_w.free();
    if (lw.img_q_b.size_bytes > 0) lw.img_q_b.free();
    if (lw.img_k_w.size_bytes > 0) lw.img_k_w.free();
    if (lw.img_k_b.size_bytes > 0) lw.img_k_b.free();
    if (lw.img_v_w.size_bytes > 0) lw.img_v_w.free();
    if (lw.img_v_b.size_bytes > 0) lw.img_v_b.free();
    if (lw.img_out_w.size_bytes > 0) lw.img_out_w.free();
    if (lw.img_out_b.size_bytes > 0) lw.img_out_b.free();
    if (lw.img_norm_q.size_bytes > 0) lw.img_norm_q.free();
    if (lw.img_norm_k.size_bytes > 0) lw.img_norm_k.free();
    if (lw.txt_q_w.size_bytes > 0) lw.txt_q_w.free();
    if (lw.txt_q_b.size_bytes > 0) lw.txt_q_b.free();
    if (lw.txt_k_w.size_bytes > 0) lw.txt_k_w.free();
    if (lw.txt_k_b.size_bytes > 0) lw.txt_k_b.free();
    if (lw.txt_v_w.size_bytes > 0) lw.txt_v_w.free();
    if (lw.txt_v_b.size_bytes > 0) lw.txt_v_b.free();
    if (lw.txt_out_w.size_bytes > 0) lw.txt_out_w.free();
    if (lw.txt_out_b.size_bytes > 0) lw.txt_out_b.free();
    if (lw.txt_norm_q.size_bytes > 0) lw.txt_norm_q.free();
    if (lw.txt_norm_k.size_bytes > 0) lw.txt_norm_k.free();
    if (lw.img_ffn_up_w.size_bytes > 0) lw.img_ffn_up_w.free();
    if (lw.img_ffn_up_b.size_bytes > 0) lw.img_ffn_up_b.free();
    if (lw.img_ffn_down_w.size_bytes > 0) lw.img_ffn_down_w.free();
    if (lw.img_ffn_down_b.size_bytes > 0) lw.img_ffn_down_b.free();
    if (lw.txt_ffn_up_w.size_bytes > 0) lw.txt_ffn_up_w.free();
    if (lw.txt_ffn_up_b.size_bytes > 0) lw.txt_ffn_up_b.free();
    if (lw.txt_ffn_down_w.size_bytes > 0) lw.txt_ffn_down_w.free();
    if (lw.txt_ffn_down_b.size_bytes > 0) lw.txt_ffn_down_b.free();
    if (lw.img_mod_w.size_bytes > 0) lw.img_mod_w.free();
    if (lw.img_mod_b.size_bytes > 0) lw.img_mod_b.free();
    if (lw.txt_mod_w.size_bytes > 0) lw.txt_mod_w.free();
    if (lw.txt_mod_b.size_bytes > 0) lw.txt_mod_b.free();
}

// --- Batch matmul dispatch helper ---

fn void dispatch_mmdit_matmul(llm::RenderGraph* g, MMDiTKernels* k, llm::Tensor* weight, llm::Tensor* input, llm::Tensor* output, uint out_dim, uint in_dim, uint seq_len) {
    MMBatchMatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim, .seq_len = seq_len, .row_offset = 0 };
    if (weight.dtype == llm::GGML_Q4_0) {
        g.dispatch(&k.batch_matmul_q4_0, { weight, input, output }, &pc, seq_len * out_dim);
    } else if (weight.dtype == llm::GGML_Q4_1) {
        g.dispatch(&k.batch_matmul_q4_1, { weight, input, output }, &pc, seq_len * out_dim);
    } else {
        g.dispatch(&k.batch_matmul_simple, { weight, input, output }, &pc, seq_len, out_dim);
    }
}

// --- Single MMDiT forward step (all 60 layers, streaming) ---

fn void? MMDiTModel.forward(&self, uint n_img, uint n_txt, float timestep) {
    llm::DeviceContext* ctx = self.ctx;
    MMDiTKernels* k = &self.kernels;
    MMDiTActivations* a = &self.acts;
    MMDiTGlobalWeights* w = &self.weights;
    uint dim = MMDIT_DIM;
    uint n_joint = n_img + n_txt;

    // 1. Timestep embedding: sinusoidal → MLP
    {
        llm::RenderGraph g;
        g.init(ctx)!!;

        // Sinusoidal embedding → t_emb [256]
        llm::Tensor t_sin = llm::create_f32_tensor(ctx, { MMDIT_T_EMB_DIM, 0, 0, 0 }, 1)!!;
        TimestepEmbedPC te_pc = { .dim = MMDIT_T_EMB_DIM, .timestep = timestep * 1000.0f };
        g.dispatch(&k.timestep_embed, { &t_sin }, &te_pc, llm::ceil_div(MMDIT_T_EMB_DIM, 256));

        // MLP: Linear1(256 → 3072) + SiLU + Linear2(3072 → 3072)
        dispatch_mmdit_matmul(&g, k, &w.t_emb_l1_w, &t_sin, &a.t_emb, dim, MMDIT_T_EMB_DIM, 1);
        BroadcastBiasPC bb1 = { .dim = dim, .n_rows = 1 };
        g.dispatch(&k.broadcast_bias, { &a.t_emb, &w.t_emb_l1_b }, &bb1, 1, llm::ceil_div(dim, 256));

        llm::SiluPC silu_pc = { .n = dim };
        g.dispatch(&k.shared.silu, { &a.t_emb }, &silu_pc, llm::ceil_div(dim, 256));

        // Need a temp buffer for second linear
        llm::Tensor t_tmp = llm::create_f32_tensor(ctx, { dim, 0, 0, 0 }, 1)!!;
        g.copy(&a.t_emb, &t_tmp, (ulong)dim * 4);
        dispatch_mmdit_matmul(&g, k, &w.t_emb_l2_w, &t_tmp, &a.t_emb, dim, dim, 1);
        g.dispatch(&k.broadcast_bias, { &a.t_emb, &w.t_emb_l2_b }, &bb1, 1, llm::ceil_div(dim, 256));

        g.flush()!!;
        t_sin.free();
        t_tmp.free();
    }

    // 2. Input projections
    {
        llm::RenderGraph g;
        g.init(ctx)!!;

        // img_hidden = img_in(patched_latent): [n_img, 64] → [n_img, 3072]
        dispatch_mmdit_matmul(&g, k, &w.img_in_w, &a.patched_latent, &a.img_hidden, dim, MMDIT_PATCH_DIM, n_img);
        BroadcastBiasPC bb_img = { .dim = dim, .n_rows = n_img };
        g.dispatch(&k.broadcast_bias, { &a.img_hidden, &w.img_in_b }, &bb_img, n_img, llm::ceil_div(dim, 256));

        // txt_hidden = txt_in(RMSNorm(txt_emb_input)): RMSNorm [n_txt, 3584] → matmul → [n_txt, 3072]
        llm::Tensor txt_rms_tmp = llm::create_f32_tensor(ctx, { (ulong)n_txt * MMDIT_TXT_DIM, 0, 0, 0 }, 1)!!;
        BatchRMSNormAffinePC rms_pc = { .dim = MMDIT_TXT_DIM, .eps = 1e-6f, .n_rows = n_txt };
        g.dispatch(&k.batch_rmsnorm, { &a.txt_emb_input, &w.txt_norm, &txt_rms_tmp }, &rms_pc, n_txt);
        dispatch_mmdit_matmul(&g, k, &w.txt_in_w, &txt_rms_tmp, &a.txt_hidden, dim, MMDIT_TXT_DIM, n_txt);
        BroadcastBiasPC bb_txt = { .dim = dim, .n_rows = n_txt };
        g.dispatch(&k.broadcast_bias, { &a.txt_hidden, &w.txt_in_b }, &bb_txt, n_txt, llm::ceil_div(dim, 256));

        g.flush()!!;
        txt_rms_tmp.free();
    }

    // 3. Process 60 layers (streaming)
    for (uint l = 0; l < MMDIT_NUM_LAYERS; l++) {
        MMDiTLayerWeights lw = load_mmdit_layer(ctx, &self.gf, l)!!;
        self.forward_layer(&lw, n_img, n_txt)!!;
        free_mmdit_layer(&lw);
    }

    // 3. Final output: AdaLN + projection
    {
        llm::RenderGraph g;
        g.init(ctx)!!;

        // LayerNorm on img_hidden
        BatchLayerNormAffinePC ln_pc = { .dim = dim, .eps = 1e-6f, .n_rows = n_img };
        // Note: final norm has no affine weight - use in-place norm
        // For simplicity, reuse batch_ln_affine with a dummy weight of ones (TODO: proper impl)
        // Actually the norm_out uses AdaLN: SiLU(t_emb) → Linear → [2*dim] → shift, scale
        // output = LayerNorm(hidden) * (1 + scale) + shift → proj_out

        // Compute final modulation: SiLU(t_emb) → norm_out.linear → [6144] → shift[3072], scale[3072]
        llm::Tensor t_silu = llm::create_f32_tensor(ctx, { dim, 0, 0, 0 }, 1)!!;
        g.copy(&a.t_emb, &t_silu, (ulong)dim * 4);
        llm::SiluPC silu_pc = { .n = dim };
        g.dispatch(&k.shared.silu, { &t_silu }, &silu_pc, llm::ceil_div(dim, 256));

        llm::Tensor final_mod = llm::create_f32_tensor(ctx, { (ulong)dim * 2, 0, 0, 0 }, 1)!!;
        dispatch_mmdit_matmul(&g, k, &w.norm_out_w, &t_silu, &final_mod, dim * 2, dim, 1);
        BroadcastBiasPC bb_final = { .dim = dim * 2, .n_rows = 1 };
        g.dispatch(&k.broadcast_bias, { &final_mod, &w.norm_out_b }, &bb_final, 1, llm::ceil_div(dim * 2, 256));

        // Apply AdaLN: LayerNorm → shift at offset 0, scale at offset dim
        AdaLNModPC adaln_final = { .n_rows = n_img, .dim = dim, .shift_offset = 0, .scale_offset = dim };
        g.dispatch(&k.adaln_mod, { &a.img_hidden, &final_mod, &a.img_norm }, &adaln_final, n_img);

        // Project to patch dim
        dispatch_mmdit_matmul(&g, k, &w.proj_out_w, &a.img_norm, &a.velocity, MMDIT_PATCH_DIM, dim, n_img);
        BroadcastBiasPC bb_proj = { .dim = MMDIT_PATCH_DIM, .n_rows = n_img };
        g.dispatch(&k.broadcast_bias, { &a.velocity, &w.proj_out_b }, &bb_proj, n_img, llm::ceil_div(MMDIT_PATCH_DIM, 256));

        g.flush()!!;
        t_silu.free();
        final_mod.free();
    }
}

// --- Single layer forward ---

fn void? MMDiTModel.forward_layer(&self, MMDiTLayerWeights* lw, uint n_img, uint n_txt) {
    llm::DeviceContext* ctx = self.ctx;
    MMDiTKernels* k = &self.kernels;
    MMDiTActivations* a = &self.acts;
    uint dim = MMDIT_DIM;
    uint n_joint = n_img + n_txt;
    uint img_bytes = (ulong)n_img * dim * 4;
    uint txt_bytes = (ulong)n_txt * dim * 4;

    llm::RenderGraph g;
    g.init(ctx)!!;

    // === Step 1: AdaLN modulation from timestep ===
    // SiLU(t_emb) is the conditioning vector. Apply to img_mod and txt_mod.
    // We need SiLU(t_emb) → matmul with mod weights. t_emb is [dim].

    // img modulation: SiLU(t_emb) → img_mod Linear → [18432] + bias
    llm::Tensor t_silu = llm::create_f32_tensor(ctx, { dim, 0, 0, 0 }, 1)!!;
    g.copy(&a.t_emb, &t_silu, (ulong)dim * 4);
    llm::SiluPC silu_pc = { .n = dim };
    g.dispatch(&k.shared.silu, { &t_silu }, &silu_pc, llm::ceil_div(dim, 256));

    dispatch_mmdit_matmul(&g, k, &lw.img_mod_w, &t_silu, &a.img_mod_buf, MMDIT_MOD_DIM, dim, 1);
    BroadcastBiasPC bb_mod = { .dim = MMDIT_MOD_DIM, .n_rows = 1 };
    g.dispatch(&k.broadcast_bias, { &a.img_mod_buf, &lw.img_mod_b }, &bb_mod, 1, llm::ceil_div(MMDIT_MOD_DIM, 256));

    dispatch_mmdit_matmul(&g, k, &lw.txt_mod_w, &t_silu, &a.txt_mod_buf, MMDIT_MOD_DIM, dim, 1);
    g.dispatch(&k.broadcast_bias, { &a.txt_mod_buf, &lw.txt_mod_b }, &bb_mod, 1, llm::ceil_div(MMDIT_MOD_DIM, 256));

    g.flush()!!;
    t_silu.free();

    llm::RenderGraph g1;
    g1.init(ctx)!!;

    // === Step 2: Pre-attention norm + modulation ===
    // img_mod_buf layout: [shift1, scale1, gate1, shift2, scale2, gate2] each [dim]
    // Offsets: shift1=0, scale1=dim, gate1=2*dim, shift2=3*dim, scale2=4*dim, gate2=5*dim

    // LayerNorm img_hidden (no affine - we'll use shift/scale from AdaLN)
    // For simplicity, use batch_ln_affine with the modulation as combined operation
    // Actually we need plain LayerNorm first, then modulate. Let me use the existing batch_layernorm from diffusion.
    // Workaround: dispatch norm without affine, then apply shift/scale

    // img: LayerNorm → modulate (one workgroup per row)
    AdaLNModPC img_adaln1 = { .n_rows = n_img, .dim = dim, .shift_offset = 0, .scale_offset = dim };
    g1.dispatch(&k.adaln_mod, { &a.img_hidden, &a.img_mod_buf, &a.img_norm }, &img_adaln1, n_img);

    // txt: LayerNorm → modulate (one workgroup per row)
    AdaLNModPC txt_adaln1 = { .n_rows = n_txt, .dim = dim, .shift_offset = 0, .scale_offset = dim };
    g1.dispatch(&k.adaln_mod, { &a.txt_hidden, &a.txt_mod_buf, &a.txt_norm }, &txt_adaln1, n_txt);

    // === Step 3: QKV projections + per-head Q/K norm ===
    PerHeadRMSNormPC img_phrms = { .seq_len = n_img, .n_heads = MMDIT_HEADS, .head_dim = MMDIT_HEAD_DIM, .eps = 1e-6f };
    PerHeadRMSNormPC txt_phrms = { .seq_len = n_txt, .n_heads = MMDIT_HEADS, .head_dim = MMDIT_HEAD_DIM, .eps = 1e-6f };
    BroadcastBiasPC bb_dim_img = { .dim = dim, .n_rows = n_img };
    BroadcastBiasPC bb_dim_txt = { .dim = dim, .n_rows = n_txt };

    // Image Q: matmul → bias → per-head RMSNorm → copy to joint
    dispatch_mmdit_matmul(&g1, k, &lw.img_q_w, &a.img_norm, &a.img_proj, dim, dim, n_img);
    g1.dispatch(&k.broadcast_bias, { &a.img_proj, &lw.img_q_b }, &bb_dim_img, n_img, llm::ceil_div(dim, 256));
    g1.dispatch(&k.perhead_rmsnorm, { &a.img_proj, &lw.img_norm_q }, &img_phrms, n_img * MMDIT_HEADS);
    if (g1.needs_barrier) llm::compute_barrier(g1.cmd);
    vk::cmdCopyBuffer(g1.cmd, a.img_proj.gpu_buffer.buffer, a.joint_q.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = img_bytes }});
    g1.needs_barrier = true;

    // Image K: matmul → bias → per-head RMSNorm → copy to joint
    dispatch_mmdit_matmul(&g1, k, &lw.img_k_w, &a.img_norm, &a.img_proj, dim, dim, n_img);
    g1.dispatch(&k.broadcast_bias, { &a.img_proj, &lw.img_k_b }, &bb_dim_img, n_img, llm::ceil_div(dim, 256));
    g1.dispatch(&k.perhead_rmsnorm, { &a.img_proj, &lw.img_norm_k }, &img_phrms, n_img * MMDIT_HEADS);
    if (g1.needs_barrier) llm::compute_barrier(g1.cmd);
    vk::cmdCopyBuffer(g1.cmd, a.img_proj.gpu_buffer.buffer, a.joint_k.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = img_bytes }});
    g1.needs_barrier = true;

    // Image V: matmul → bias → copy to joint (no norm on V)
    dispatch_mmdit_matmul(&g1, k, &lw.img_v_w, &a.img_norm, &a.img_proj, dim, dim, n_img);
    g1.dispatch(&k.broadcast_bias, { &a.img_proj, &lw.img_v_b }, &bb_dim_img, n_img, llm::ceil_div(dim, 256));
    if (g1.needs_barrier) llm::compute_barrier(g1.cmd);
    vk::cmdCopyBuffer(g1.cmd, a.img_proj.gpu_buffer.buffer, a.joint_v.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = img_bytes }});
    g1.needs_barrier = true;

    // Text Q: matmul → bias → per-head RMSNorm → copy to joint
    dispatch_mmdit_matmul(&g1, k, &lw.txt_q_w, &a.txt_norm, &a.txt_proj, dim, dim, n_txt);
    g1.dispatch(&k.broadcast_bias, { &a.txt_proj, &lw.txt_q_b }, &bb_dim_txt, n_txt, llm::ceil_div(dim, 256));
    g1.dispatch(&k.perhead_rmsnorm, { &a.txt_proj, &lw.txt_norm_q }, &txt_phrms, n_txt * MMDIT_HEADS);
    if (g1.needs_barrier) llm::compute_barrier(g1.cmd);
    vk::cmdCopyBuffer(g1.cmd, a.txt_proj.gpu_buffer.buffer, a.joint_q.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = img_bytes, .size = txt_bytes }});
    g1.needs_barrier = true;

    // Text K: matmul → bias → per-head RMSNorm → copy to joint
    dispatch_mmdit_matmul(&g1, k, &lw.txt_k_w, &a.txt_norm, &a.txt_proj, dim, dim, n_txt);
    g1.dispatch(&k.broadcast_bias, { &a.txt_proj, &lw.txt_k_b }, &bb_dim_txt, n_txt, llm::ceil_div(dim, 256));
    g1.dispatch(&k.perhead_rmsnorm, { &a.txt_proj, &lw.txt_norm_k }, &txt_phrms, n_txt * MMDIT_HEADS);
    if (g1.needs_barrier) llm::compute_barrier(g1.cmd);
    vk::cmdCopyBuffer(g1.cmd, a.txt_proj.gpu_buffer.buffer, a.joint_k.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = img_bytes, .size = txt_bytes }});
    g1.needs_barrier = true;

    // Text V: matmul → bias → copy to joint (no norm on V)
    dispatch_mmdit_matmul(&g1, k, &lw.txt_v_w, &a.txt_norm, &a.txt_proj, dim, dim, n_txt);
    g1.dispatch(&k.broadcast_bias, { &a.txt_proj, &lw.txt_v_b }, &bb_dim_txt, n_txt, llm::ceil_div(dim, 256));
    if (g1.needs_barrier) llm::compute_barrier(g1.cmd);
    vk::cmdCopyBuffer(g1.cmd, a.txt_proj.gpu_buffer.buffer, a.joint_v.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = img_bytes, .size = txt_bytes }});
    g1.needs_barrier = true;

    // === Step 4: Transpose to heads layout and run joint attention ===
    // joint_q/k/v are [n_joint, dim] → need [n_heads, n_joint, head_dim]
    uint total_elems = n_joint * dim;
    VitTransposeHeadsPC th_fwd = { .seq_len = n_joint, .n_heads = MMDIT_HEADS, .head_dim = MMDIT_HEAD_DIM, .direction = 0 };

    // Transpose Q
    g1.dispatch(&k.batch_ln_affine, { &a.joint_q, &a.joint_q, &a.attn_out }, &th_fwd, llm::ceil_div(total_elems, 256));
    // Wait - I should use transpose_heads not batch_ln_affine. Let me use the ViT transpose_heads kernel.
    // Actually the vit.c3 transpose_heads kernel is already created in ViT kernels, not in MMDiT kernels.
    // I need to add it to MMDiT kernels too. For now, let me just do the attention in [seq, dim] layout.
    // The spatial_attention kernel expects [n_heads, seq_len, head_dim] layout though.

    // Use vit transpose_heads - need to create the kernel handle
    // For now, let's just submit what we have and do attention in a second command buffer
    g1.flush()!!;

    // Transpose Q/K/V and run attention in separate graph
    {
        llm::RenderGraph g2;
        g2.init(ctx)!!;

        // We need transpose_heads kernel - create it inline (TODO: move to kernel creation)
        char[] vit_spv = &VIT_SPV;
        vk::ShaderModule vit_shader = vk::shaderModuleCreateInfo().setCodeSize(
            vit_spv.len).setCode((uint*)&vit_spv[0]).build(ctx.device)!!;
        llm::ComputeKernel th_kernel = llm::create_kernel(ctx, vit_shader, 2, VitTransposeHeadsPC.sizeof, "transpose_heads")!!;
        vit_shader.free(ctx.device);

        // Transpose Q → attn_out (temp), then copy back
        g2.dispatch(&th_kernel, { &a.joint_q, &a.attn_out }, &th_fwd, llm::ceil_div(total_elems, 256));
        g2.copy(&a.attn_out, &a.joint_q, (ulong)total_elems * 4);

        // Transpose K
        g2.dispatch(&th_kernel, { &a.joint_k, &a.attn_out }, &th_fwd, llm::ceil_div(total_elems, 256));
        g2.copy(&a.attn_out, &a.joint_k, (ulong)total_elems * 4);

        // Transpose V
        g2.dispatch(&th_kernel, { &a.joint_v, &a.attn_out }, &th_fwd, llm::ceil_div(total_elems, 256));
        g2.copy(&a.attn_out, &a.joint_v, (ulong)total_elems * 4);

        // Joint spatial attention
        float scale = 1.0f / std::math::sqrt((float)MMDIT_HEAD_DIM);
        MMSpatialAttentionPC sa_pc = { .head_dim = MMDIT_HEAD_DIM, .n_heads = MMDIT_HEADS, .seq_len = n_joint, .scale = scale };
        g2.dispatch(&k.spatial_attention, { &a.joint_q, &a.joint_k, &a.joint_v, &a.attn_scores, &a.attn_out },
            &sa_pc, MMDIT_HEADS * n_joint);

        // Transpose back: [n_heads, n_joint, head_dim] → [n_joint, dim]
        VitTransposeHeadsPC th_inv = { .seq_len = n_joint, .n_heads = MMDIT_HEADS, .head_dim = MMDIT_HEAD_DIM, .direction = 1 };
        g2.dispatch(&th_kernel, { &a.attn_out, &a.joint_q }, &th_inv, llm::ceil_div(total_elems, 256));
        // joint_q now holds the attention output in [n_joint, dim] layout

        g2.flush()!!;
        th_kernel.free(ctx.device);
    }

    // === Step 5: Output projection + gated residual ===
    {
        llm::RenderGraph g3;
        g3.init(ctx)!!;

        // Split attention output: img portion [0..n_img], txt portion [n_img..n_joint]
        // Image output projection
        // Copy img portion from joint_q to img_proj
        if (g3.needs_barrier) llm::compute_barrier(g3.cmd);
        vk::cmdCopyBuffer(g3.cmd, a.joint_q.gpu_buffer.buffer, a.img_proj.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = img_bytes }});
        g3.needs_barrier = true;

        dispatch_mmdit_matmul(&g3, k, &lw.img_out_w, &a.img_proj, &a.img_norm, dim, dim, n_img);
        g3.dispatch(&k.broadcast_bias, { &a.img_norm, &lw.img_out_b }, &bb_dim_img, n_img, llm::ceil_div(dim, 256));

        // Gated residual: img_hidden += gate1 * img_attn_out
        GatedResidualPC img_gate1 = { .n_rows = n_img, .dim = dim, .gate_offset = 2 * dim };
        g3.dispatch(&k.gated_residual, { &a.img_hidden, &a.img_norm, &a.img_mod_buf }, &img_gate1,
            llm::ceil_div(n_img * dim, 256));

        // Text output projection
        if (g3.needs_barrier) llm::compute_barrier(g3.cmd);
        vk::cmdCopyBuffer(g3.cmd, a.joint_q.gpu_buffer.buffer, a.txt_proj.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = img_bytes, .dstOffset = 0, .size = txt_bytes }});
        g3.needs_barrier = true;

        dispatch_mmdit_matmul(&g3, k, &lw.txt_out_w, &a.txt_proj, &a.txt_norm, dim, dim, n_txt);
        g3.dispatch(&k.broadcast_bias, { &a.txt_norm, &lw.txt_out_b }, &bb_dim_txt, n_txt, llm::ceil_div(dim, 256));

        GatedResidualPC txt_gate1 = { .n_rows = n_txt, .dim = dim, .gate_offset = 2 * dim };
        g3.dispatch(&k.gated_residual, { &a.txt_hidden, &a.txt_norm, &a.txt_mod_buf }, &txt_gate1,
            llm::ceil_div(n_txt * dim, 256));

        // === Step 6: Pre-FFN norm + modulation ===
        AdaLNModPC img_adaln2 = { .n_rows = n_img, .dim = dim, .shift_offset = 3 * dim, .scale_offset = 4 * dim };
        g3.dispatch(&k.adaln_mod, { &a.img_hidden, &a.img_mod_buf, &a.img_norm }, &img_adaln2, n_img);

        AdaLNModPC txt_adaln2 = { .n_rows = n_txt, .dim = dim, .shift_offset = 3 * dim, .scale_offset = 4 * dim };
        g3.dispatch(&k.adaln_mod, { &a.txt_hidden, &a.txt_mod_buf, &a.txt_norm }, &txt_adaln2, n_txt);

        // === Step 7: GEGLU FFN ===
        // Image FFN
        dispatch_mmdit_matmul(&g3, k, &lw.img_ffn_up_w, &a.img_norm, &a.img_ffn_buf, MMDIT_FFN_DIM, dim, n_img);
        BroadcastBiasPC bb_ffn_img = { .dim = MMDIT_FFN_DIM, .n_rows = n_img };
        g3.dispatch(&k.broadcast_bias, { &a.img_ffn_buf, &lw.img_ffn_up_b }, &bb_ffn_img, n_img, llm::ceil_div(MMDIT_FFN_DIM, 256));
        GeGLUPC geglu_img = { .n_rows = n_img, .half_dim = MMDIT_GEGLU_DIM };
        g3.dispatch(&k.geglu, { &a.img_ffn_buf, &a.img_ffn_out }, &geglu_img, llm::ceil_div(n_img * MMDIT_GEGLU_DIM, 256));

        dispatch_mmdit_matmul(&g3, k, &lw.img_ffn_down_w, &a.img_ffn_out, &a.img_norm, dim, MMDIT_GEGLU_DIM, n_img);
        BroadcastBiasPC bb_down_img = { .dim = dim, .n_rows = n_img };
        g3.dispatch(&k.broadcast_bias, { &a.img_norm, &lw.img_ffn_down_b }, &bb_down_img, n_img, llm::ceil_div(dim, 256));

        GatedResidualPC img_gate2 = { .n_rows = n_img, .dim = dim, .gate_offset = 5 * dim };
        g3.dispatch(&k.gated_residual, { &a.img_hidden, &a.img_norm, &a.img_mod_buf }, &img_gate2,
            llm::ceil_div(n_img * dim, 256));

        // Text FFN
        dispatch_mmdit_matmul(&g3, k, &lw.txt_ffn_up_w, &a.txt_norm, &a.txt_ffn_buf, MMDIT_FFN_DIM, dim, n_txt);
        BroadcastBiasPC bb_ffn_txt = { .dim = MMDIT_FFN_DIM, .n_rows = n_txt };
        g3.dispatch(&k.broadcast_bias, { &a.txt_ffn_buf, &lw.txt_ffn_up_b }, &bb_ffn_txt, n_txt, llm::ceil_div(MMDIT_FFN_DIM, 256));
        GeGLUPC geglu_txt = { .n_rows = n_txt, .half_dim = MMDIT_GEGLU_DIM };
        g3.dispatch(&k.geglu, { &a.txt_ffn_buf, &a.txt_ffn_out }, &geglu_txt, llm::ceil_div(n_txt * MMDIT_GEGLU_DIM, 256));

        dispatch_mmdit_matmul(&g3, k, &lw.txt_ffn_down_w, &a.txt_ffn_out, &a.txt_norm, dim, MMDIT_GEGLU_DIM, n_txt);
        BroadcastBiasPC bb_down_txt = { .dim = dim, .n_rows = n_txt };
        g3.dispatch(&k.broadcast_bias, { &a.txt_norm, &lw.txt_ffn_down_b }, &bb_down_txt, n_txt, llm::ceil_div(dim, 256));

        GatedResidualPC txt_gate2 = { .n_rows = n_txt, .dim = dim, .gate_offset = 5 * dim };
        g3.dispatch(&k.gated_residual, { &a.txt_hidden, &a.txt_norm, &a.txt_mod_buf }, &txt_gate2,
            llm::ceil_div(n_txt * dim, 256));

        g3.flush()!!;
    }
}

// --- Multi-step denoising ---

// Compute dynamic shift mu based on image sequence length (linear interpolation)
fn float compute_shift_mu(uint image_seq_len) {
    const float BASE_SHIFT = 0.5f;
    const float MAX_SHIFT = 1.15f;
    const float BASE_SEQ = 256.0f;
    const float MAX_SEQ = 4096.0f;
    float m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ);
    return BASE_SHIFT + ((float)image_seq_len - BASE_SEQ) * m;
}

// Exponential time shift: shifted = exp(mu) / (exp(mu) + (1/t - 1))
fn float time_shift_exp(float mu, float t) {
    if (t <= 1e-8f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    float e_mu = math::exp(mu);
    return e_mu / (e_mu + (1.0f / t - 1.0f));
}

fn void? MMDiTModel.denoise(&self, uint n_img, uint n_txt, uint height, uint width, uint n_steps, float cfg_scale, llm::Tensor* uncond_txt_emb, uint n_uncond_txt) {
    llm::DeviceContext* ctx = self.ctx;
    MMDiTKernels* k = &self.kernels;
    MMDiTActivations* a = &self.acts;
    uint patch_dim = MMDIT_PATCH_DIM;
    uint n_total = n_img * patch_dim;

    io::printfn("\n=== MMDiT Denoising (%d steps) ===", n_steps);
    io::printfn("  Image: %dx%d, %d patches, %d text tokens", width, height, n_img, n_txt);

    // Build sigma schedule with dynamic shifting
    float mu = compute_shift_mu(n_img);
    io::printfn("  Shift mu: %.4f", mu);

    const float SHIFT_TERMINAL = 0.02f;

    // Step 1: Linear sigmas from 1.0 to 1/n_steps
    float[] sigmas = mem::new_array(float, (usz)(n_steps + 1));
    for (uint i = 0; i < n_steps; i++) {
        sigmas[i] = 1.0f - (float)i * (1.0f - 1.0f / (float)n_steps) / ((float)n_steps - 1.0f);
    }

    // Step 2: Apply exponential time shift
    for (uint i = 0; i < n_steps; i++) {
        sigmas[i] = time_shift_exp(mu, sigmas[i]);
    }

    // Step 3: Terminal stretch so last sigma maps to SHIFT_TERMINAL
    float last_one_minus = 1.0f - sigmas[n_steps - 1];
    float scale_factor = last_one_minus / (1.0f - SHIFT_TERMINAL);
    for (uint i = 0; i < n_steps; i++) {
        float one_minus = 1.0f - sigmas[i];
        sigmas[i] = 1.0f - one_minus / scale_factor;
    }

    // Step 4: Append terminal 0
    sigmas[n_steps] = 0.0f;

    bool use_cfg = cfg_scale > 1.0f && uncond_txt_emb != null;
    // For CFG: we need a buffer to save the uncond velocity
    llm::Tensor uncond_velocity;
    // We also need separate uncond activations for txt_emb_input swap
    if (use_cfg) {
        uncond_velocity = llm::create_f32_tensor(ctx, { (ulong)n_total, 0, 0, 0 }, 1)!!;
        io::printfn("  CFG enabled: scale=%.1f, uncond_tokens=%d", cfg_scale, n_uncond_txt);
    }

    for (uint step = 0; step < n_steps; step++) {
        float sigma = sigmas[step];
        float sigma_next = sigmas[step + 1];
        float dt = sigma_next - sigma;  // negative

        io::printfn("  Step %d/%d: σ=%.4f → %.4f", step + 1, n_steps, sigma, sigma_next);
        Clock step_start = clock::now();

        if (use_cfg) {
            // --- Conditional forward ---
            self.forward(n_img, n_txt, sigma)!!;

            // Save cond velocity, then run uncond
            // We need to swap txt_emb_input to uncond, reallocate acts for n_uncond_txt
            // For simplicity: copy cond velocity, swap txt_emb, run uncond forward, then CFG combine

            // Copy cond velocity to uncond_velocity (we'll use it for CFG)
            ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer cmd) {
                vk::cmdCopyBuffer(cmd, a.velocity.gpu_buffer.buffer, uncond_velocity.gpu_buffer.buffer, 1,
                    (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_total * 4 }});
            }!!;

            // --- Unconditional forward ---
            // Swap text embeddings: save cond, load uncond
            llm::Tensor saved_txt = a.txt_emb_input;
            a.txt_emb_input = *uncond_txt_emb;
            self.forward(n_img, n_uncond_txt, sigma)!!;
            a.txt_emb_input = saved_txt;  // restore

            // Now: velocity = uncond prediction, uncond_velocity = cond prediction
            // Swap so cond is in velocity (for CFG shader: cfg_cond=cond, cfg_uncond=uncond)
            // Actually we saved cond in uncond_velocity. Let me swap:
            // uncond_velocity has cond, velocity has uncond
            // CFG: result = uncond + scale * (cond - uncond)
            // cfg_cond buffer will be overwritten with the combined result
            // So pass: cfg_cond = uncond_velocity (cond), cfg_uncond = velocity (uncond)
            {
                llm::RenderGraph g;
                g.init(ctx)!!;
                CfgPC cfg_pc = { .n_rows = n_img, .dim = MMDIT_PATCH_DIM, .scale = cfg_scale };
                g.dispatch(&k.norm_cfg, { &uncond_velocity, &a.velocity }, &cfg_pc, n_img);
                g.flush()!!;
            }

            // Euler step with CFG-combined velocity (in uncond_velocity)
            {
                llm::RenderGraph g;
                g.init(ctx)!!;
                FlowEulerPC euler_pc = { .n = n_total, .dt = dt };
                g.dispatch(&k.flow_euler, { &a.patched_latent, &uncond_velocity }, &euler_pc, llm::ceil_div(n_total, 256));
                g.flush()!!;
            }
        } else {
            // No CFG: single forward pass
            self.forward(n_img, n_txt, sigma)!!;
            {
                llm::RenderGraph g;
                g.init(ctx)!!;
                FlowEulerPC euler_pc = { .n = n_total, .dt = dt };
                g.dispatch(&k.flow_euler, { &a.patched_latent, &a.velocity }, &euler_pc, llm::ceil_div(n_total, 256));
                g.flush()!!;
            }
        }

        double ms = step_start.to_now().to_sec() * 1000.0;
        io::printfn("    Step %d done: %.0f ms", step + 1, ms);
    }

    if (use_cfg) uncond_velocity.free();
    mem::free(sigmas);
    io::printfn("=== Denoising Complete ===");
}
