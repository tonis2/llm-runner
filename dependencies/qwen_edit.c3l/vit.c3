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

// Embedded ViT SPIR-V shader
const char[*] VIT_SPV = $embed("./shaders/vit.spv");

// --- ViT Configuration (from mmproj GGUF metadata) ---

struct ViTConfig {
    uint dim;           // 1280
    uint n_heads;       // 16
    uint head_dim;      // 80  (dim / n_heads)
    uint ffn_dim;       // 3420
    uint n_layers;      // 32
    uint patch_size;    // 14
    uint image_size;    // 560
    uint merger_hidden; // 5120 (dim * 4 for 2x2 merge)
    uint proj_dim;      // 3584 (LLM hidden dim)
    uint window_size;   // 8 (n_wa_pattern)
    float ln_eps;       // 1e-6
    float[3] image_mean;
    float[3] image_std;
}

// --- Push constant structs (matching vit.slang) ---

struct BatchLayerNormAffinePC { uint dim; float eps; uint n_rows; uint _pad; }
struct BroadcastBiasPC { uint dim; uint n_rows; }
struct ImageNormPC { uint height; uint width; float mean_r; float mean_g; float mean_b; float std_r; float std_g; float std_b; }
struct PatchEmbedPC { uint height; uint width; uint patch_size; uint out_channels; uint in_channels; uint patches_h; uint patches_w; uint _pad; }
struct Rope2DPC { uint seq_len; uint n_heads; uint head_dim; uint half_dim; uint patches_w; float theta; uint _pad0; uint _pad1; }
struct VitTransposeHeadsPC { uint seq_len; uint n_heads; uint head_dim; uint direction; }
struct WindowPartitionPC { uint patches_h; uint patches_w; uint dim; uint window_size; uint direction; uint _pad0; uint _pad1; uint _pad2; }
struct SpatialMerge2x2PC { uint patches_h; uint patches_w; uint dim; uint _pad; }

// Reuse these from zimage (same layout as llm.slang batch_matmul)
struct BatchMatMulPC { uint out_dim; uint in_dim; uint seq_len; uint row_offset; }
struct SpatialAttentionPC { uint head_dim; uint n_heads; uint seq_len; float scale; }
struct FFNSwigluSplitPC { uint n_elements; uint _pad; }

// --- Layer Weights ---

struct ViTLayerWeights {
    llm::Tensor ln1;          // [dim]
    llm::Tensor ln2;          // [dim]
    llm::Tensor wq;           // [dim, dim]
    llm::Tensor wk;           // [dim, dim]
    llm::Tensor wv;           // [dim, dim]
    llm::Tensor wo;           // [dim, dim]
    llm::Tensor bq;           // [dim]
    llm::Tensor bk;           // [dim]
    llm::Tensor bv;           // [dim]
    llm::Tensor bo;           // [dim]
    llm::Tensor ffn_gate;     // [dim, ffn_dim]
    llm::Tensor ffn_up;       // [dim, ffn_dim]
    llm::Tensor ffn_down;     // [ffn_dim, dim]
    llm::Tensor ffn_gate_b;   // [ffn_dim]
    llm::Tensor ffn_up_b;     // [ffn_dim]
    llm::Tensor ffn_down_b;   // [dim]
}

struct ViTWeights {
    llm::Tensor patch_embd;   // [out_ch, 3, 14, 14] as F32
    llm::Tensor post_ln;      // [dim]
    llm::Tensor mm0_weight;   // [merger_hidden, merger_hidden]
    llm::Tensor mm0_bias;     // [merger_hidden]
    llm::Tensor mm2_weight;   // [merger_hidden, proj_dim]
    llm::Tensor mm2_bias;     // [proj_dim]
    ViTLayerWeights[] layers;
}

// --- Activation Buffers ---

struct ViTActivations {
    llm::Tensor hidden;       // [n_patches, dim]
    llm::Tensor norm_out;     // [n_patches, dim]
    llm::Tensor q;            // [n_patches, dim]
    llm::Tensor k;            // [n_patches, dim]
    llm::Tensor v;            // [n_patches, dim]
    llm::Tensor attn_out;     // [n_patches, dim]
    llm::Tensor attn_scores;  // [n_heads, n_patches, n_patches]
    llm::Tensor ffn_gate_out; // [n_patches, ffn_dim]
    llm::Tensor ffn_up_out;   // [n_patches, ffn_dim]
    llm::Tensor proj_out;     // [n_patches, dim]
    llm::Tensor window_buf;   // [n_patches, dim] for window rearrange
    llm::Tensor merged;       // [n_patches/4, merger_hidden]
    llm::Tensor merger_tmp;   // [n_patches/4, merger_hidden]
}

// --- Kernel Handles ---

struct ViTKernels {
    // From vit.slang
    llm::ComputeKernel batch_ln_affine;
    llm::ComputeKernel broadcast_bias;
    llm::ComputeKernel image_normalize;
    llm::ComputeKernel patch_embed;
    llm::ComputeKernel rope_2d;
    llm::ComputeKernel transpose_heads;
    llm::ComputeKernel window_partition;
    llm::ComputeKernel spatial_merge;
    // From llm.slang (batch matmul variants)
    llm::ComputeKernel batch_matmul;
    llm::ComputeKernel batch_matmul_q8;
    llm::ComputeKernel batch_matmul_simple;
    llm::ComputeKernel ffn_swiglu_split;
    // From diffusion.slang (non-causal attention)
    llm::ComputeKernel spatial_attention;
    // Shared
    llm::SharedKernels shared;
}

// --- Main ViT Model ---

struct ViTModel {
    ViTConfig config;
    ViTWeights weights;
    ViTActivations acts;
    ViTKernels kernels;
    llm::DeviceContext* ctx;
}

// Full attention layers (0-indexed)
fn bool is_full_attention_layer(uint layer, uint n_wa_pattern) {
    // Full attention every n_wa_pattern layers, at the last position
    // For n_wa_pattern=8: layers 7, 15, 23, 31
    return (layer + 1) % n_wa_pattern == 0;
}

// --- Load mmproj GGUF ---

fn ViTModel? load_vit_model(llm::DeviceContext* ctx, String mmproj_path) {
    io::printfn("\n=== Loading ViT (mmproj) ===");
    io::printfn("  Path: %s", mmproj_path);

    mmap::FileMmap mm = file::mmap_open(mmproj_path, "rb")!!;
    char[] data = mm.bytes();
    llm::GGUFFile gf = llm::gguf_parse(data)!!;

    // Read config from metadata
    ViTConfig config = {
        .dim = gf.get_u32("clip.vision.embedding_length") ?? 1280,
        .n_heads = gf.get_u32("clip.vision.attention.head_count") ?? 16,
        .ffn_dim = gf.get_u32("clip.vision.feed_forward_length") ?? 3420,
        .n_layers = gf.get_u32("clip.vision.block_count") ?? 32,
        .patch_size = gf.get_u32("clip.vision.patch_size") ?? 14,
        .image_size = gf.get_u32("clip.vision.image_size") ?? 560,
        .proj_dim = gf.get_u32("clip.vision.projection_dim") ?? 3584,
        .window_size = gf.get_u32("clip.vision.n_wa_pattern") ?? 8,
        .ln_eps = gf.get_f32("clip.vision.attention.layer_norm_epsilon") ?? 1e-6f,
    };
    config.head_dim = config.dim / config.n_heads;
    config.merger_hidden = config.dim * 4;  // 2x2 spatial merge: 1280*4 = 5120

    // Image mean/std arrays - read from metadata or use defaults
    config.image_mean = { 0.48145467f, 0.45782751f, 0.40821072f };
    config.image_std = { 0.26862954f, 0.26130259f, 0.27577711f };

    io::printfn("  dim=%d, heads=%d, head_dim=%d, layers=%d", config.dim, config.n_heads, config.head_dim, config.n_layers);
    io::printfn("  ffn_dim=%d, patch=%d, image=%d, window=%d", config.ffn_dim, config.patch_size, config.image_size, config.window_size);
    io::printfn("  merger_hidden=%d, proj_dim=%d", config.merger_hidden, config.proj_dim);

    // Load weights
    io::printfn("  Loading patch embedding...");
    llm::Tensor patch_embd = llm::load_tensor_as_f32(ctx, &gf, "v.patch_embd.weight")!!;

    io::printfn("  Loading post_ln...");
    llm::Tensor post_ln = llm::load_tensor_by_name(ctx, &gf, "v.post_ln.weight")!!;

    io::printfn("  Loading merger MLP...");
    llm::Tensor mm0_weight = llm::upload_weight(ctx, gf.find_tensor("mm.0.weight")!!, gf.tensor_data_base)!!;
    llm::Tensor mm0_bias = llm::load_tensor_as_f32(ctx, &gf, "mm.0.bias")!!;
    llm::Tensor mm2_weight = llm::upload_weight(ctx, gf.find_tensor("mm.2.weight")!!, gf.tensor_data_base)!!;
    llm::Tensor mm2_bias = llm::load_tensor_as_f32(ctx, &gf, "mm.2.bias")!!;

    // Load layer weights
    ViTLayerWeights[] layers = mem::new_array(ViTLayerWeights, config.n_layers);
    for (uint l = 0; l < config.n_layers; l++) {
        layers[l] = load_vit_layer(ctx, &gf, l)!!;
        if ((l + 1) % 8 == 0 || l == config.n_layers - 1) {
            io::printfn("  Loaded ViT layer %d / %d", l + 1, config.n_layers);
        }
    }

    ViTWeights weights = {
        .patch_embd = patch_embd,
        .post_ln = post_ln,
        .mm0_weight = mm0_weight,
        .mm0_bias = mm0_bias,
        .mm2_weight = mm2_weight,
        .mm2_bias = mm2_bias,
        .layers = layers,
    };

    // Create kernels
    ViTKernels kernels = create_vit_kernels(ctx)!!;

    // Allocate activations for max image size
    uint patches_h = config.image_size / config.patch_size;
    uint patches_w = config.image_size / config.patch_size;
    uint n_patches = patches_h * patches_w;
    ViTActivations acts = allocate_vit_activations(ctx, &config, n_patches)!!;

    io::printfn("  ViT loaded: %d patches (%dx%d)", n_patches, patches_h, patches_w);

    // Keep mmap alive (weights reference it for quantized formats)
    // Note: in a real impl we'd store the mmap handle; for now weights are uploaded to GPU

    return {
        .config = config,
        .weights = weights,
        .acts = acts,
        .kernels = kernels,
        .ctx = ctx,
    };
}

fn ViTLayerWeights? load_vit_layer(llm::DeviceContext* ctx, llm::GGUFFile* gf, uint layer) {
    char[80] buf;

    return {
        .ln1 = load_vit_tensor(ctx, gf, &buf, layer, "ln1.weight")!!,
        .ln2 = load_vit_tensor(ctx, gf, &buf, layer, "ln2.weight")!!,
        .wq = load_vit_tensor(ctx, gf, &buf, layer, "attn_q.weight")!!,
        .wk = load_vit_tensor(ctx, gf, &buf, layer, "attn_k.weight")!!,
        .wv = load_vit_tensor(ctx, gf, &buf, layer, "attn_v.weight")!!,
        .wo = load_vit_tensor(ctx, gf, &buf, layer, "attn_out.weight")!!,
        .bq = load_vit_tensor_f32(ctx, gf, &buf, layer, "attn_q.bias")!!,
        .bk = load_vit_tensor_f32(ctx, gf, &buf, layer, "attn_k.bias")!!,
        .bv = load_vit_tensor_f32(ctx, gf, &buf, layer, "attn_v.bias")!!,
        .bo = load_vit_tensor_f32(ctx, gf, &buf, layer, "attn_out.bias")!!,
        .ffn_gate = load_vit_tensor(ctx, gf, &buf, layer, "ffn_gate.weight")!!,
        .ffn_up = load_vit_tensor(ctx, gf, &buf, layer, "ffn_up.weight")!!,
        .ffn_down = load_vit_tensor(ctx, gf, &buf, layer, "ffn_down.weight")!!,
        .ffn_gate_b = load_vit_tensor_f32(ctx, gf, &buf, layer, "ffn_gate.bias")!!,
        .ffn_up_b = load_vit_tensor_f32(ctx, gf, &buf, layer, "ffn_up.bias")!!,
        .ffn_down_b = load_vit_tensor_f32(ctx, gf, &buf, layer, "ffn_down.bias")!!,
    };
}

// Format "v.blk.{layer}.{suffix}" and upload preserving quantization
fn llm::Tensor? load_vit_tensor(llm::DeviceContext* ctx, llm::GGUFFile* gf, char[80]* buf, uint layer, String suffix) {
    usz len = format_vit_name(buf, layer, suffix);
    String name = (String)(*buf)[0..len - 1];
    llm::GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return llm::upload_weight(ctx, info, gf.tensor_data_base);
}

// Format "v.blk.{layer}.{suffix}" and upload as F32 (for biases)
fn llm::Tensor? load_vit_tensor_f32(llm::DeviceContext* ctx, llm::GGUFFile* gf, char[80]* buf, uint layer, String suffix) {
    usz len = format_vit_name(buf, layer, suffix);
    String name = (String)(*buf)[0..len - 1];
    return llm::load_tensor_as_f32(ctx, gf, name);
}

fn usz format_vit_name(char[80]* buf, uint layer, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;

    // "v.blk."
    String prefix = "v.blk.";
    for (usz i = 0; i < prefix.len; i++) { p[pos + i] = prefix[i]; }
    pos += prefix.len;

    // Layer number
    if (layer >= 10) { p[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    p[pos] = (char)('0' + layer % 10); pos++;

    // "."
    p[pos] = '.'; pos++;

    // Suffix
    for (usz i = 0; i < suffix.len; i++) { p[pos + i] = suffix[i]; }
    pos += suffix.len;

    return pos;
}

// --- Kernel Creation ---

fn ViTKernels? create_vit_kernels(llm::DeviceContext* ctx) {
    // ViT shader
    char[] vit_spv = &VIT_SPV;
    vk::ShaderModule vit_shader = vk::shaderModuleCreateInfo().setCodeSize(
        vit_spv.len).setCode((uint*)&vit_spv[0]).build(ctx.device)!!;

    // LLM shader (for batch matmul variants)
    char[] llm_spv = &llm_text::LLM_SPV;
    vk::ShaderModule llm_shader = vk::shaderModuleCreateInfo().setCodeSize(
        llm_spv.len).setCode((uint*)&llm_spv[0]).build(ctx.device)!!;

    // Diffusion shader (for spatial_attention)
    char[] diff_spv = &llm::pipelines::DIFFUSION_SPV;
    vk::ShaderModule diff_shader = vk::shaderModuleCreateInfo().setCodeSize(
        diff_spv.len).setCode((uint*)&diff_spv[0]).build(ctx.device)!!;

    ViTKernels kernels = {
        .batch_ln_affine = llm::create_kernel(ctx, vit_shader, 3, BatchLayerNormAffinePC.sizeof, "batch_layernorm_affine")!!,
        .broadcast_bias = llm::create_kernel(ctx, vit_shader, 2, BroadcastBiasPC.sizeof, "broadcast_bias_add")!!,
        .image_normalize = llm::create_kernel(ctx, vit_shader, 1, ImageNormPC.sizeof, "image_normalize")!!,
        .patch_embed = llm::create_kernel(ctx, vit_shader, 3, PatchEmbedPC.sizeof, "patch_embed_conv2d")!!,
        .rope_2d = llm::create_kernel(ctx, vit_shader, 1, Rope2DPC.sizeof, "rope_2d")!!,
        .transpose_heads = llm::create_kernel(ctx, vit_shader, 2, VitTransposeHeadsPC.sizeof, "transpose_heads")!!,
        .window_partition = llm::create_kernel(ctx, vit_shader, 2, WindowPartitionPC.sizeof, "window_partition")!!,
        .spatial_merge = llm::create_kernel(ctx, vit_shader, 2, SpatialMerge2x2PC.sizeof, "spatial_merge_2x2")!!,
        .batch_matmul = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul")!!,
        .batch_matmul_q8 = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q8")!!,
        .batch_matmul_simple = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_simple")!!,
        .ffn_swiglu_split = llm::create_kernel(ctx, llm_shader, 2, FFNSwigluSplitPC.sizeof, "ffn_swiglu_split")!!,
        .spatial_attention = llm::create_kernel(ctx, diff_shader, 5, SpatialAttentionPC.sizeof, "spatial_attention")!!,
        .shared = llm::create_shared_kernels(ctx, llm_shader)!!,
    };

    vit_shader.free(ctx.device);
    llm_shader.free(ctx.device);
    diff_shader.free(ctx.device);

    return kernels;
}

// --- Activation Allocation ---

fn ViTActivations? allocate_vit_activations(llm::DeviceContext* ctx, ViTConfig* c, uint n_patches) {
    uint dim = c.dim;
    uint ffn = c.ffn_dim;
    uint merged_patches = n_patches / 4;

    return {
        .hidden = llm::create_f32_tensor(ctx, { (ulong)n_patches * dim, 0, 0, 0 }, 1)!!,
        .norm_out = llm::create_f32_tensor(ctx, { (ulong)n_patches * dim, 0, 0, 0 }, 1)!!,
        .q = llm::create_f32_tensor(ctx, { (ulong)n_patches * dim, 0, 0, 0 }, 1)!!,
        .k = llm::create_f32_tensor(ctx, { (ulong)n_patches * dim, 0, 0, 0 }, 1)!!,
        .v = llm::create_f32_tensor(ctx, { (ulong)n_patches * dim, 0, 0, 0 }, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, { (ulong)n_patches * dim, 0, 0, 0 }, 1)!!,
        .attn_scores = llm::create_f32_tensor(ctx, { (ulong)c.n_heads * n_patches * n_patches, 0, 0, 0 }, 1)!!,
        .ffn_gate_out = llm::create_f32_tensor(ctx, { (ulong)n_patches * ffn, 0, 0, 0 }, 1)!!,
        .ffn_up_out = llm::create_f32_tensor(ctx, { (ulong)n_patches * ffn, 0, 0, 0 }, 1)!!,
        .proj_out = llm::create_f32_tensor(ctx, { (ulong)n_patches * dim, 0, 0, 0 }, 1)!!,
        .window_buf = llm::create_f32_tensor(ctx, { (ulong)n_patches * dim, 0, 0, 0 }, 1)!!,
        .merged = llm::create_f32_tensor(ctx, { (ulong)merged_patches * c.merger_hidden, 0, 0, 0 }, 1)!!,
        .merger_tmp = llm::create_f32_tensor(ctx, { (ulong)merged_patches * c.merger_hidden, 0, 0, 0 }, 1)!!,
    };
}

// --- Batch MatMul Dispatch Helper ---

fn void dispatch_vit_matmul(llm::RenderGraph* g, ViTKernels* k, llm::Tensor* weight, llm::Tensor* input, llm::Tensor* output, uint out_dim, uint in_dim, uint seq_len) {
    BatchMatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim, .seq_len = seq_len, .row_offset = 0 };

    if (weight.dtype == llm::GGML_Q8_0) {
        uint groups = ((seq_len + 63) / 64) * ((out_dim + 63) / 64);
        g.dispatch(&k.batch_matmul_q8, { weight, input, output }, &pc, groups);
    } else {
        // F32: use simple dispatch
        g.dispatch(&k.batch_matmul_simple, { weight, input, output }, &pc, seq_len, out_dim);
    }
}

// --- Forward Pass ---

fn llm::Tensor? ViTModel.forward(&self, llm::Tensor* image_gpu, uint height, uint width) {
    llm::DeviceContext* ctx = self.ctx;
    ViTConfig* c = &self.config;
    ViTWeights* w = &self.weights;
    ViTActivations* a = &self.acts;
    ViTKernels* k = &self.kernels;

    uint patches_h = height / c.patch_size;
    uint patches_w = width / c.patch_size;
    uint n_patches = patches_h * patches_w;
    uint dim = c.dim;

    io::printfn("  ViT forward: %dx%d image → %dx%d patches (%d total)", width, height, patches_w, patches_h, n_patches);

    // Step 1: Normalize image
    {
        llm::RenderGraph g;
        g.init(ctx)!!;

        ImageNormPC norm_pc = {
            .height = height, .width = width,
            .mean_r = c.image_mean[0], .mean_g = c.image_mean[1], .mean_b = c.image_mean[2],
            .std_r = c.image_std[0], .std_g = c.image_std[1], .std_b = c.image_std[2],
        };
        g.dispatch(&k.image_normalize, { image_gpu }, &norm_pc, llm::ceil_div(3 * height * width, 256));

        // Step 2: Patch embedding
        PatchEmbedPC pe_pc = {
            .height = height, .width = width,
            .patch_size = c.patch_size, .out_channels = dim,
            .in_channels = 3, .patches_h = patches_h, .patches_w = patches_w,
        };
        g.dispatch(&k.patch_embed, { image_gpu, &w.patch_embd, &a.hidden }, &pe_pc, n_patches, dim);
        g.flush()!!;
    }
    io::printfn("  Patch embedding done: [%d, %d]", n_patches, dim);

    // Step 3: Transformer layers
    for (uint l = 0; l < c.n_layers; l++) {
        self.forward_layer(l, n_patches, patches_h, patches_w)!!;
    }
    io::printfn("  %d transformer layers done", c.n_layers);

    // Step 4: Post LayerNorm
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        BatchLayerNormAffinePC ln_pc = { .dim = dim, .eps = c.ln_eps, .n_rows = n_patches };
        g.dispatch(&k.batch_ln_affine, { &a.hidden, &w.post_ln, &a.norm_out }, &ln_pc, n_patches);

        // Step 5: Spatial merge 2x2
        SpatialMerge2x2PC sm_pc = { .patches_h = patches_h, .patches_w = patches_w, .dim = dim };
        uint merged_h = patches_h / 2;
        uint merged_w = patches_w / 2;
        uint merged_patches = merged_h * merged_w;
        g.dispatch(&k.spatial_merge, { &a.norm_out, &a.merged }, &sm_pc,
            llm::ceil_div(merged_patches * c.merger_hidden, 256));

        g.flush()!!;

        // Step 6: Merger MLP
        llm::RenderGraph g2;
        g2.init(ctx)!!;

        // Linear 0: [merged_patches, 5120] → [merged_patches, 5120]
        dispatch_vit_matmul(&g2, k, &w.mm0_weight, &a.merged, &a.merger_tmp, c.merger_hidden, c.merger_hidden, merged_patches);
        BroadcastBiasPC bb0 = { .dim = c.merger_hidden, .n_rows = merged_patches };
        g2.dispatch(&k.broadcast_bias, { &a.merger_tmp, &w.mm0_bias }, &bb0, merged_patches, llm::ceil_div(c.merger_hidden, 256));

        // SiLU activation
        llm::SiluPC silu_pc = { .n = merged_patches * c.merger_hidden };
        g2.dispatch(&k.shared.silu, { &a.merger_tmp }, &silu_pc, llm::ceil_div(merged_patches * c.merger_hidden, 256));

        // Linear 2: [merged_patches, 5120] → [merged_patches, 3584]
        dispatch_vit_matmul(&g2, k, &w.mm2_weight, &a.merger_tmp, &a.merged, c.proj_dim, c.merger_hidden, merged_patches);
        BroadcastBiasPC bb2 = { .dim = c.proj_dim, .n_rows = merged_patches };
        g2.dispatch(&k.broadcast_bias, { &a.merged, &w.mm2_bias }, &bb2, merged_patches, llm::ceil_div(c.proj_dim, 256));

        g2.flush()!!;

        io::printfn("  Merger MLP done: [%d, %d]", merged_patches, c.proj_dim);

        // Return merged tensor (caller should not free — it's part of acts)
        a.merged.n_dims = 2;
        a.merged.shape = { (ulong)merged_patches, (ulong)c.proj_dim, 0, 0 };
        return a.merged;
    }
}

fn void? ViTModel.forward_layer(&self, uint layer, uint n_patches, uint patches_h, uint patches_w) {
    ViTConfig* c = &self.config;
    ViTWeights* w = &self.weights;
    ViTActivations* a = &self.acts;
    ViTKernels* k = &self.kernels;
    ViTLayerWeights* lw = &w.layers[layer];
    uint dim = c.dim;

    llm::RenderGraph g;
    g.init(self.ctx)!!;

    // (a) Pre-attention LayerNorm
    BatchLayerNormAffinePC ln1_pc = { .dim = dim, .eps = c.ln_eps, .n_rows = n_patches };
    g.dispatch(&k.batch_ln_affine, { &a.hidden, &lw.ln1, &a.norm_out }, &ln1_pc, n_patches);

    // (b) Q/K/V projections
    dispatch_vit_matmul(&g, k, &lw.wq, &a.norm_out, &a.q, dim, dim, n_patches);
    dispatch_vit_matmul(&g, k, &lw.wk, &a.norm_out, &a.k, dim, dim, n_patches);
    dispatch_vit_matmul(&g, k, &lw.wv, &a.norm_out, &a.v, dim, dim, n_patches);

    // Add biases
    BroadcastBiasPC bb_q = { .dim = dim, .n_rows = n_patches };
    g.dispatch(&k.broadcast_bias, { &a.q, &lw.bq }, &bb_q, n_patches, llm::ceil_div(dim, 256));
    g.dispatch(&k.broadcast_bias, { &a.k, &lw.bk }, &bb_q, n_patches, llm::ceil_div(dim, 256));
    g.dispatch(&k.broadcast_bias, { &a.v, &lw.bv }, &bb_q, n_patches, llm::ceil_div(dim, 256));

    // (c) 2D RoPE on Q and K
    Rope2DPC rope_pc = {
        .seq_len = n_patches, .n_heads = c.n_heads, .head_dim = c.head_dim,
        .half_dim = c.head_dim / 2, .patches_w = patches_w, .theta = 10000.0f,
    };
    // RoPE expects [n_heads, seq_len, head_dim] layout, but Q is [seq_len, dim]
    // We need to transpose first, apply rope, then handle attention

    // (d) Transpose to heads layout: [seq, dim] → [heads, seq, head_dim]
    VitTransposeHeadsPC th_fwd = { .seq_len = n_patches, .n_heads = c.n_heads, .head_dim = c.head_dim, .direction = 0 };
    uint total_elems = n_patches * dim;
    g.dispatch(&k.transpose_heads, { &a.q, &a.attn_out }, &th_fwd, llm::ceil_div(total_elems, 256));
    g.copy(&a.attn_out, &a.q, (ulong)total_elems * 4);
    g.dispatch(&k.transpose_heads, { &a.k, &a.attn_out }, &th_fwd, llm::ceil_div(total_elems, 256));
    g.copy(&a.attn_out, &a.k, (ulong)total_elems * 4);
    g.dispatch(&k.transpose_heads, { &a.v, &a.attn_out }, &th_fwd, llm::ceil_div(total_elems, 256));
    g.copy(&a.attn_out, &a.v, (ulong)total_elems * 4);

    // Apply 2D RoPE (in-place on Q and K in [heads, seq, head_dim] layout)
    uint total_pairs = c.n_heads * n_patches * (c.head_dim / 2);
    g.dispatch(&k.rope_2d, { &a.q }, &rope_pc, llm::ceil_div(total_pairs, 256));
    g.dispatch(&k.rope_2d, { &a.k }, &rope_pc, llm::ceil_div(total_pairs, 256));

    // (e) Attention
    bool full_attn = is_full_attention_layer(layer, c.window_size);

    if (full_attn) {
        // Full non-causal attention over all patches
        float scale = 1.0f / std::math::sqrt((float)c.head_dim);
        SpatialAttentionPC sa_pc = { .head_dim = c.head_dim, .n_heads = c.n_heads, .seq_len = n_patches, .scale = scale };
        g.dispatch(&k.spatial_attention, { &a.q, &a.k, &a.v, &a.attn_scores, &a.attn_out },
            &sa_pc, c.n_heads * n_patches);
    } else {
        // Window attention: partition → attention per window → unpartition
        // For now, fall back to full attention (window attention optimization TODO)
        float scale = 1.0f / std::math::sqrt((float)c.head_dim);
        SpatialAttentionPC sa_pc = { .head_dim = c.head_dim, .n_heads = c.n_heads, .seq_len = n_patches, .scale = scale };
        g.dispatch(&k.spatial_attention, { &a.q, &a.k, &a.v, &a.attn_scores, &a.attn_out },
            &sa_pc, c.n_heads * n_patches);
    }

    // (f) Transpose back: [heads, seq, head_dim] → [seq, dim]
    VitTransposeHeadsPC th_inv = { .seq_len = n_patches, .n_heads = c.n_heads, .head_dim = c.head_dim, .direction = 1 };
    g.dispatch(&k.transpose_heads, { &a.attn_out, &a.proj_out }, &th_inv, llm::ceil_div(total_elems, 256));

    // (g) Output projection
    dispatch_vit_matmul(&g, k, &lw.wo, &a.proj_out, &a.norm_out, dim, dim, n_patches);
    g.dispatch(&k.broadcast_bias, { &a.norm_out, &lw.bo }, &bb_q, n_patches, llm::ceil_div(dim, 256));

    // (h) Residual: hidden += attn_output
    llm::ResidualPC res_pc = { .n = n_patches * dim };
    g.dispatch(&k.shared.residual_add, { &a.hidden, &a.norm_out }, &res_pc, llm::ceil_div(n_patches * dim, 256));

    // (i) Pre-FFN LayerNorm
    BatchLayerNormAffinePC ln2_pc = { .dim = dim, .eps = c.ln_eps, .n_rows = n_patches };
    g.dispatch(&k.batch_ln_affine, { &a.hidden, &lw.ln2, &a.norm_out }, &ln2_pc, n_patches);

    // (j) SwiGLU FFN
    dispatch_vit_matmul(&g, k, &lw.ffn_gate, &a.norm_out, &a.ffn_gate_out, c.ffn_dim, dim, n_patches);
    dispatch_vit_matmul(&g, k, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, dim, n_patches);

    // Add FFN biases
    BroadcastBiasPC bb_ffn = { .dim = c.ffn_dim, .n_rows = n_patches };
    g.dispatch(&k.broadcast_bias, { &a.ffn_gate_out, &lw.ffn_gate_b }, &bb_ffn, n_patches, llm::ceil_div(c.ffn_dim, 256));
    g.dispatch(&k.broadcast_bias, { &a.ffn_up_out, &lw.ffn_up_b }, &bb_ffn, n_patches, llm::ceil_div(c.ffn_dim, 256));

    // Fused SwiGLU: gate = silu(gate) * up
    FFNSwigluSplitPC swiglu_pc = { .n_elements = n_patches * c.ffn_dim };
    g.dispatch(&k.ffn_swiglu_split, { &a.ffn_gate_out, &a.ffn_up_out }, &swiglu_pc, llm::ceil_div(n_patches * c.ffn_dim, 256));

    // FFN down projection
    dispatch_vit_matmul(&g, k, &lw.ffn_down, &a.ffn_gate_out, &a.norm_out, dim, c.ffn_dim, n_patches);
    BroadcastBiasPC bb_down = { .dim = dim, .n_rows = n_patches };
    g.dispatch(&k.broadcast_bias, { &a.norm_out, &lw.ffn_down_b }, &bb_down, n_patches, llm::ceil_div(dim, 256));

    // (k) Residual: hidden += ffn_output
    g.dispatch(&k.shared.residual_add, { &a.hidden, &a.norm_out }, &res_pc, llm::ceil_div(n_patches * dim, 256));

    g.flush()!!;
}

// =============================================================================
// Multimodal Encoding: ViT + LLM integration
// =============================================================================

// Build a token sequence with visual placeholders and encode through the LLM.
// Returns [total_seq_len, dim] embeddings from the penultimate LLM layer.
fn llm::Tensor? encode_multimodal(
    llm::DeviceContext* ctx,
    ViTModel* vit,
    llm::Tensor* image_gpu,
    uint image_height,
    uint image_width,
    String prompt,
    String text_model_path
) {
    io::printfn("\n=== Multimodal Encoding ===");

    // 1. Run ViT to get visual tokens
    io::printfn("[1/4] Running ViT...");
    llm::Tensor visual_tokens = vit.forward(image_gpu, image_height, image_width)!!;
    uint n_visual = (uint)visual_tokens.shape[0];
    io::printfn("  Visual tokens: [%d, %d]", visual_tokens.shape[0], visual_tokens.shape[1]);

    // 2. Load text model
    io::printfn("[2/4] Loading text model...");
    mmap::FileMmap text_mm = file::mmap_open(text_model_path, "rb")!!;
    char[] text_data = text_mm.bytes();
    llm::GGUFFile text_gf = llm::gguf_parse(text_data)!!;

    String arch_name = llm::detect_architecture(&text_gf);
    io::printfn("  Architecture: %s", arch_name);

    String config_json;
    if (arch_name == "qwen3") {
        config_json = (String)&llm_text::QWEN3_CONFIG_JSON;
    } else if (arch_name == "qwen2") {
        config_json = (String)&llm_text::QWEN2_CONFIG_JSON;
    } else if (arch_name == "qwen2vl") {
        config_json = (String)&llm_text::QWEN2_VL_CONFIG_JSON;
    } else {
        config_json = (String)&llm_text::QWEN3_CONFIG_JSON;
    }

    llm::ModelConfig text_config = llm::load_arch_config(config_json, &text_gf)!!;
    llm::WeightNames text_names = llm::load_weight_names(config_json)!!;
    llm::Tokenizer tok = llm::load_tokenizer(&text_gf)!!;

    // 3. Build token sequence with visual placeholders
    io::printfn("[3/4] Building token sequence...");

    // Find special token IDs
    uint im_start_id = tok.find_special_token("<|im_start|>");
    uint im_end_id = tok.find_special_token("<|im_end|>");
    uint vision_start_id = tok.find_special_token("<|vision_start|>");
    uint vision_end_id = tok.find_special_token("<|vision_end|>");
    uint image_pad_id = tok.find_special_token("<|image_pad|>");

    io::printfn("  Special tokens: im_start=%d, im_end=%d, vision_start=%d, vision_end=%d, image_pad=%d",
        im_start_id, im_end_id, vision_start_id, vision_end_id, image_pad_id);

    // Tokenize text segments
    uint[] user_tokens = tok.encode("user\n")!!;
    uint[] prompt_tokens = tok.encode(prompt)!!;
    uint[] newline_tokens = tok.encode("\n")!!;
    uint[] assistant_tokens = tok.encode("assistant\n")!!;

    // Build full sequence:
    // [im_start] + "user\n" + [vision_start] + [image_pad × n_visual] + [vision_end] + prompt + [im_end] + "\n" + [im_start] + "assistant\n"
    uint total_len = 1 + (uint)user_tokens.len + 1 + n_visual + 1 + (uint)prompt_tokens.len + 1 + (uint)newline_tokens.len + 1 + (uint)assistant_tokens.len;
    uint[] tokens = mem::new_array(uint, total_len);
    uint idx = 0;

    // <|im_start|>user\n
    tokens[idx] = im_start_id; idx++;
    for (usz i = 0; i < user_tokens.len; i++) { tokens[idx] = user_tokens[i]; idx++; }

    // <|vision_start|> + <|image_pad|> × n_visual + <|vision_end|>
    tokens[idx] = vision_start_id; idx++;
    for (uint i = 0; i < n_visual; i++) { tokens[idx] = image_pad_id; idx++; }
    tokens[idx] = vision_end_id; idx++;

    // prompt text + <|im_end|>\n
    for (usz i = 0; i < prompt_tokens.len; i++) { tokens[idx] = prompt_tokens[i]; idx++; }
    tokens[idx] = im_end_id; idx++;
    for (usz i = 0; i < newline_tokens.len; i++) { tokens[idx] = newline_tokens[i]; idx++; }

    // <|im_start|>assistant\n
    tokens[idx] = im_start_id; idx++;
    for (usz i = 0; i < assistant_tokens.len; i++) { tokens[idx] = assistant_tokens[i]; idx++; }

    io::printfn("  Total sequence: %d tokens (%d visual + %d text/special)", idx, n_visual, idx - n_visual);

    mem::free(user_tokens);
    mem::free(prompt_tokens);
    mem::free(newline_tokens);
    mem::free(assistant_tokens);

    // 4. Load LLM and encode with visual injection
    io::printfn("[4/4] Running LLM with visual tokens...");
    uint max_seq = idx;
    llm_text::LlmModel text_model = llm_text::load_llm_model(
        ctx, &text_gf, &text_config, &text_names,
        encode_only: true,
        max_seq_len: max_seq
    )!!;

    llm::Tensor result = text_model.encode_text_with_visual(tokens[0..idx-1], &visual_tokens, image_pad_id)!!;

    io::printfn("  Output: [%d, %d]", result.shape[0], result.shape[1]);

    // Cleanup
    mem::free(tokens);
    tok.free();
    text_model.free();
    text_gf.free();
    text_mm.destroy();

    return result;
}

// --- Text-only encoding (no image) ---
// Encodes a text prompt through Qwen2.5-VL LLM, returning [seq_len, 3584] embeddings
// for use as MMDiT text conditioning.

fn llm::Tensor? encode_text_prompt(
    llm::DeviceContext* ctx,
    String prompt,
    String text_model_path
) {
    io::printfn("\n=== Text Encoding ===");

    // Load text model
    io::printfn("  Loading text model...");
    mmap::FileMmap text_mm = file::mmap_open(text_model_path, "rb")!!;
    char[] text_data = text_mm.bytes();
    llm::GGUFFile text_gf = llm::gguf_parse(text_data)!!;

    String arch_name = llm::detect_architecture(&text_gf);
    io::printfn("  Architecture: %s", arch_name);

    String config_json;
    if (arch_name == "qwen3") {
        config_json = (String)&llm_text::QWEN3_CONFIG_JSON;
    } else if (arch_name == "qwen2") {
        config_json = (String)&llm_text::QWEN2_CONFIG_JSON;
    } else if (arch_name == "qwen2vl") {
        config_json = (String)&llm_text::QWEN2_VL_CONFIG_JSON;
    } else {
        config_json = (String)&llm_text::QWEN3_CONFIG_JSON;
    }

    llm::ModelConfig text_config = llm::load_arch_config(config_json, &text_gf)!!;
    llm::WeightNames text_names = llm::load_weight_names(config_json)!!;
    llm::Tokenizer tok = llm::load_tokenizer(&text_gf)!!;

    // Build token sequence: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    uint im_start_id = tok.find_special_token("<|im_start|>");
    uint im_end_id = tok.find_special_token("<|im_end|>");

    uint[] user_tokens = tok.encode("user\n")!!;
    uint[] prompt_tokens = tok.encode(prompt)!!;
    uint[] newline_tokens = tok.encode("\n")!!;
    uint[] assistant_tokens = tok.encode("assistant\n")!!;

    uint total_len = 1 + (uint)user_tokens.len + (uint)prompt_tokens.len + 1 + (uint)newline_tokens.len + 1 + (uint)assistant_tokens.len;
    uint[] tokens = mem::new_array(uint, total_len);
    uint idx = 0;

    tokens[idx] = im_start_id; idx++;
    for (usz i = 0; i < user_tokens.len; i++) { tokens[idx] = user_tokens[i]; idx++; }
    for (usz i = 0; i < prompt_tokens.len; i++) { tokens[idx] = prompt_tokens[i]; idx++; }
    tokens[idx] = im_end_id; idx++;
    for (usz i = 0; i < newline_tokens.len; i++) { tokens[idx] = newline_tokens[i]; idx++; }
    tokens[idx] = im_start_id; idx++;
    for (usz i = 0; i < assistant_tokens.len; i++) { tokens[idx] = assistant_tokens[i]; idx++; }

    io::printfn("  Sequence: %d tokens", idx);

    mem::free(user_tokens);
    mem::free(prompt_tokens);
    mem::free(newline_tokens);
    mem::free(assistant_tokens);

    // Load and run LLM
    llm_text::LlmModel text_model = llm_text::load_llm_model(
        ctx, &text_gf, &text_config, &text_names,
        encode_only: true,
        max_seq_len: idx
    )!!;

    llm::Tensor result = text_model.encode_text_streaming(tokens[0..idx-1])!!;
    io::printfn("  Output: [%d, %d]", result.shape[0], result.shape[1]);

    mem::free(tokens);
    tok.free();
    text_model.free();
    text_gf.free();
    text_mm.destroy();

    return result;
}
