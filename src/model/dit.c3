module llm;

import vk;
import std::io;
import std::math;
import std::core::mem;

// Embedded Z-Image SPIR-V shader
const char[*] ZIMAGE_SPV = $embed("../../shaders/zimage.spv");

// DiT (Diffusion Transformer) for Z-Image Turbo / Lumina2 architecture
// 30-layer transformer with AdaLN modulation, dual RMSNorm, SwiGLU FFN

const uint DIT_NUM_LAYERS = 30;
const uint DIT_NUM_REFINER_LAYERS = 2;
const uint DIT_DIM = 3840;        // Hidden dimension
const uint DIT_HEADS = 30;        // Number of attention heads
const uint DIT_HEAD_DIM = 128;    // 3840 / 30
const uint DIT_FFN_DIM = 10240;   // FFN intermediate dimension
const uint DIT_PATCH_SIZE = 2;
const uint DIT_LATENT_CHANNELS = 16;
const uint DIT_PATCH_DIM = 64;    // 16 * 2 * 2
const uint DIT_T_EMB_DIM = 256;   // Timestep embedding dimension
const uint DIT_T_MLP_DIM = 1024;  // t_embedder MLP intermediate dim

// --- Push constant structs for Z-Image shaders ---

struct PatchifyPC {
    uint channels;
    uint height;
    uint width;
    uint patch_size;
}

struct UnpatchifyPC {
    uint channels;
    uint height;
    uint width;
    uint patch_size;
}

struct AdalnModPC {
    uint n_elements;
    uint dim;
}

struct FlowEulerPC {
    uint n;
    float dt;
}

struct LinearProjPC {
    uint out_dim;
    uint in_dim;
    uint seq_len;
}

struct DiTTimestepPC {
    uint dim;
    float timestep;
}

struct BroadcastAddDimPC {
    uint n_total;
    uint dim;
}

struct ScalePC {
    uint n;
    float scale;
}

// --- DiT Layer Weights ---

struct DiTLayerWeights {
    // Fused QKV: [3*dim, dim]
    Tensor wqkv;
    // Q/K norms: [head_dim]
    Tensor q_norm;
    Tensor k_norm;
    // Output projection: [dim, dim]
    Tensor wo;
    // FFN SwiGLU: gate [ffn_dim, dim], up [ffn_dim, dim], down [dim, ffn_dim]
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
    // Dual norms
    Tensor attn_norm1;   // image tokens
    Tensor attn_norm2;   // text tokens
    Tensor ffn_norm1;    // image tokens
    Tensor ffn_norm2;    // text tokens
    // AdaLN: projects t_emb to modulation params [4*dim]
    Tensor adaln_linear;
    Tensor adaln_bias;
}

struct DiTWeights {
    DiTLayerWeights[DIT_NUM_LAYERS] layers;

    // Noise refiner (2 layers, timestep-conditioned)
    DiTLayerWeights[DIT_NUM_REFINER_LAYERS] noise_refiner;
    // Context refiner (2 layers, plain transformer - no adaLN)
    DiTLayerWeights[DIT_NUM_REFINER_LAYERS] context_refiner;

    // cap_embedder: Sequential(RMSNorm(text_dim), Linear(text_dim, dim))
    Tensor cap_emb_norm;    // RMSNorm weight [text_dim]
    Tensor cap_emb_weight;  // Linear weight [text_dim, dim]
    Tensor cap_emb_bias;    // Linear bias [dim]

    // t_embedder MLP: sinusoidal [256] -> Linear(256,1024) -> SiLU -> Linear(1024,256) -> [256]
    Tensor t_emb_mlp0_weight;
    Tensor t_emb_mlp0_bias;
    Tensor t_emb_mlp2_weight;
    Tensor t_emb_mlp2_bias;

    // x_embedder: [dim, patch_dim] + bias
    Tensor x_emb_weight;
    Tensor x_emb_bias;

    // Final layer
    Tensor final_adaln_linear;
    Tensor final_adaln_bias;
    Tensor final_linear;    // [patch_dim, dim]
    Tensor final_bias;

    // Pad tokens: [dim] (learnable padding for text)
    Tensor x_pad_token;
    Tensor cap_pad_token;
}

// --- DiT Activations ---

struct DiTActivations {
    Tensor buf_a;       // General purpose [max_seq * dim]
    Tensor buf_b;       // General purpose [max_seq * dim]
    Tensor buf_c;       // General purpose [max_seq * dim]
    Tensor patches;     // [n_patches, patch_dim]
    Tensor hidden;      // [n_patches + text_len, dim]
    Tensor norm_out;    // [dim] per-position scratch
    Tensor q;           // [n_heads, seq_len, head_dim]
    Tensor k;           // [n_heads, seq_len, head_dim]
    Tensor v;           // [n_heads, seq_len, head_dim]
    Tensor attn_scores; // [n_heads, seq_len, seq_len]
    Tensor attn_out;    // [seq_len, dim]
    Tensor ffn_gate_out; // [seq_len, ffn_dim]
    Tensor ffn_up_out;   // [seq_len, ffn_dim]
    Tensor ffn_down_out; // [seq_len, dim]
    Tensor t_emb;       // [t_emb_dim]
    Tensor t_emb_mlp;   // [DIT_T_MLP_DIM] intermediate / [DIT_T_EMB_DIM] final
    Tensor adaln_params; // [4 * dim]
    Tensor adaln_scale;  // [dim] (split from adaln_params)
    Tensor adaln_shift;  // [dim]
    Tensor velocity;     // [n_patches, patch_dim]
    Tensor latent;       // [channels, h, w]
}

// --- DiT Kernels ---

struct DiTKernels {
    ComputeKernel patchify;
    ComputeKernel unpatchify;
    ComputeKernel adaln_modulate;
    ComputeKernel flow_euler_step;
    ComputeKernel linear_proj;
    ComputeKernel dit_timestep_embed;
    ComputeKernel broadcast_add_dim;
    ComputeKernel scale_buffer;
    ComputeKernel spatial_attention;
    SharedKernels shared;
}

// --- DiT Model ---

struct DiTModel {
    DiTWeights weights;
    DiTActivations acts;
    DiTKernels kernels;
    DeviceContext* ctx;
    uint n_patches;     // Number of image patches
    uint text_len;      // Text sequence length
    uint latent_h;      // Latent height
    uint latent_w;      // Latent width
}

// --- Kernel Creation ---

fn DiTKernels? create_dit_kernels(DeviceContext* ctx) {
    io::printfn("Creating DiT compute kernels...");

    char[] zi_spv = &ZIMAGE_SPV;
    ShaderModule zi_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(zi_spv.len)
        .setCode((uint*)&zi_spv[0])
        .build(ctx.device)!!;

    // Also need diffusion shader for spatial_attention
    char[] diff_spv = &DIFFUSION_SPV;
    ShaderModule diff_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(diff_spv.len)
        .setCode((uint*)&diff_spv[0])
        .build(ctx.device)!!;

    // Shared kernels from LLM shader (rmsnorm, matmul variants, silu, etc.)
    char[] llm_spv = &LLM_SPV;
    ShaderModule llm_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(llm_spv.len)
        .setCode((uint*)&llm_spv[0])
        .build(ctx.device)!!;

    DiTKernels kernels = {
        .patchify         = create_kernel(ctx, zi_shader, 2, PatchifyPC.sizeof, "patchify")!!,
        .unpatchify       = create_kernel(ctx, zi_shader, 2, UnpatchifyPC.sizeof, "unpatchify")!!,
        .adaln_modulate   = create_kernel(ctx, zi_shader, 4, AdalnModPC.sizeof, "adaln_modulate")!!,
        .flow_euler_step  = create_kernel(ctx, zi_shader, 2, FlowEulerPC.sizeof, "flow_euler_step")!!,
        .linear_proj      = create_kernel(ctx, zi_shader, 4, LinearProjPC.sizeof, "linear_proj")!!,
        .dit_timestep_embed = create_kernel(ctx, zi_shader, 1, DiTTimestepPC.sizeof, "dit_timestep_embed")!!,
        .broadcast_add_dim = create_kernel(ctx, zi_shader, 2, BroadcastAddDimPC.sizeof, "broadcast_add_dim")!!,
        .scale_buffer     = create_kernel(ctx, zi_shader, 1, ScalePC.sizeof, "scale_buffer")!!,
        .spatial_attention = create_kernel(ctx, diff_shader, 5, SpatialAttentionPC.sizeof, "spatial_attention")!!,
        .shared           = create_shared_kernels(ctx, llm_shader)!!,
    };

    zi_shader.free(ctx.device);
    diff_shader.free(ctx.device);
    llm_shader.free(ctx.device);
    return kernels;
}

// --- Activation Allocation ---

fn DiTActivations? allocate_dit_activations(DeviceContext* ctx, uint max_seq_len) {
    io::printfn("Allocating DiT activation buffers (max_seq=%d)...", max_seq_len);

    ulong max_dim_total = (ulong)max_seq_len * DIT_DIM;
    ulong max_ffn_total = (ulong)max_seq_len * DIT_FFN_DIM;
    ulong attn_total = (ulong)DIT_HEADS * max_seq_len * max_seq_len;
    ulong latent_total = (ulong)DIT_LATENT_CHANNELS * 64 * 64;  // max 512x512 image

    return {
        .buf_a = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .buf_b = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .buf_c = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .patches = create_f32_tensor(ctx, { (ulong)1024 * DIT_PATCH_DIM, 0, 0, 0 }, 1)!!,
        .hidden = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .norm_out = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .q = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .k = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .v = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .attn_scores = create_f32_tensor(ctx, { attn_total, 0, 0, 0 }, 1)!!,
        .attn_out = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .ffn_gate_out = create_f32_tensor(ctx, { max_ffn_total, 0, 0, 0 }, 1)!!,
        .ffn_up_out = create_f32_tensor(ctx, { max_ffn_total, 0, 0, 0 }, 1)!!,
        .ffn_down_out = create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .t_emb = create_f32_tensor(ctx, { DIT_T_EMB_DIM, 0, 0, 0 }, 1)!!,
        .t_emb_mlp = create_f32_tensor(ctx, { DIT_T_MLP_DIM, 0, 0, 0 }, 1)!!,
        .adaln_params = create_f32_tensor(ctx, { (ulong)4 * DIT_DIM, 0, 0, 0 }, 1)!!,
        .adaln_scale = create_f32_tensor(ctx, { DIT_DIM, 0, 0, 0 }, 1)!!,
        .adaln_shift = create_f32_tensor(ctx, { DIT_DIM, 0, 0, 0 }, 1)!!,
        .velocity = create_f32_tensor(ctx, { (ulong)1024 * DIT_PATCH_DIM, 0, 0, 0 }, 1)!!,
        .latent = create_f32_tensor(ctx, { latent_total, 0, 0, 0 }, 1)!!,
    };
}

// --- Weight Loading ---

fn String dit_layer_tensor_name(char[128]* buf, String prefix, uint layer, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) p[pos + i] = prefix[i];
    pos += prefix.len;
    if (layer >= 100) { p[pos] = (char)('0' + layer / 100); pos++; }
    if (layer >= 10) { p[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    p[pos] = (char)('0' + layer % 10); pos++;
    p[pos] = '.'; pos++;
    for (usz i = 0; i < suffix.len; i++) p[pos + i] = suffix[i];
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

fn Tensor load_dit_tensor(DeviceContext* ctx, GGUFFile* gf, String name) {
    if (try info = gf.find_tensor(name)) {
        // F16/BF16 weights get dequantized to F32
        if (info.type == GGML_F16 || info.type == GGML_BF16 || info.type == GGML_F32) {
            return upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
        }
        // Quantized weights stay quantized
        return upload_weight(ctx, info, gf.tensor_data_base)!!;
    }
    io::printfn("Warning: DiT tensor not found: %s", name);
    return (Tensor){};
}

fn Tensor load_dit_tensor_f32(DeviceContext* ctx, GGUFFile* gf, String name) {
    GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
}

fn DiTLayerWeights load_dit_layer_weights(DeviceContext* ctx, GGUFFile* gf, String prefix, uint layer, bool has_adaln) {
    char[128] buf;

    DiTLayerWeights lw;

    // Fused QKV: prefix.L.attention.qkv.weight
    String qkv_name = dit_layer_tensor_name(&buf, prefix, layer, "attention.qkv.weight");
    io::printfn("    [%s%d] loading qkv...", prefix, layer);
    if (try info = gf.find_tensor(qkv_name)) {
        if (info.type == GGML_F16 || info.type == GGML_BF16 || info.type == GGML_F32) {
            lw.wqkv = upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
        } else {
            lw.wqkv = upload_weight(ctx, info, gf.tensor_data_base)!!;
        }
    } else {
        io::printfn("    Warning: QKV not found: %s", qkv_name);
    }

    io::printfn("    [%s%d] loading norms + out...", prefix, layer);
    lw.q_norm = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "attention.q_norm.weight"));
    lw.k_norm = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "attention.k_norm.weight"));
    lw.wo = load_dit_tensor(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "attention.out.weight"));

    io::printfn("    [%s%d] loading ffn...", prefix, layer);
    lw.ffn_gate = load_dit_tensor(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w1.weight"));
    lw.ffn_up = load_dit_tensor(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w3.weight"));
    lw.ffn_down = load_dit_tensor(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w2.weight"));

    lw.attn_norm1 = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "attention_norm1.weight"));

    // Try loading second norm (for dual-norm layers)
    String norm2_name = dit_layer_tensor_name(&buf, prefix, layer, "attention_norm2.weight");
    if (try info = gf.find_tensor(norm2_name)) {
        lw.attn_norm2 = upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
    }

    lw.ffn_norm1 = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "ffn_norm1.weight"));

    String ffn_norm2_name = dit_layer_tensor_name(&buf, prefix, layer, "ffn_norm2.weight");
    if (try info = gf.find_tensor(ffn_norm2_name)) {
        lw.ffn_norm2 = upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
    }

    if (has_adaln) {
        io::printfn("    [%s%d] loading adaln...", prefix, layer);
        lw.adaln_linear = load_dit_tensor(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "adaLN_modulation.0.weight"));
        lw.adaln_bias = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "adaLN_modulation.0.bias"));
    }

    return lw;
}

fn DiTWeights? load_dit_weights(DeviceContext* ctx, GGUFFile* gf) {
    io::printfn("\nLoading DiT weights...");

    DiTWeights w;

    // Main layers (0-29)
    for (uint l = 0; l < DIT_NUM_LAYERS; l++) {
        io::printf("  DiT layer %d / %d...\r", l + 1, DIT_NUM_LAYERS);
        w.layers[l] = load_dit_layer_weights(ctx, gf, "layers.", l, true);
        if ((l + 1) % 5 == 0 || l == DIT_NUM_LAYERS - 1) {
            io::printfn("  DiT layer %d / %d loaded", l + 1, DIT_NUM_LAYERS);
        }
    }

    // Noise refiner layers
    io::printfn("  Loading noise refiner...");
    for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
        w.noise_refiner[l] = load_dit_layer_weights(ctx, gf, "noise_refiner.", l, true);
    }

    // Context refiner layers (no adaLN)
    io::printfn("  Loading context refiner...");
    for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
        w.context_refiner[l] = load_dit_layer_weights(ctx, gf, "context_refiner.", l, false);
    }

    // Embedders
    io::printfn("  Loading embedders...");
    // cap_embedder = Sequential(RMSNorm(text_dim), Linear(text_dim, dim))
    w.cap_emb_norm = load_dit_tensor_f32(ctx, gf, "cap_embedder.0.weight");
    w.cap_emb_weight = load_dit_tensor_f32(ctx, gf, "cap_embedder.1.weight");
    w.cap_emb_bias = load_dit_tensor_f32(ctx, gf, "cap_embedder.1.bias");

    // t_embedder = MLP: sinusoidal[256] -> Linear(256,1024) -> SiLU -> Linear(1024,256)
    w.t_emb_mlp0_weight = load_dit_tensor_f32(ctx, gf, "t_embedder.mlp.0.weight");
    w.t_emb_mlp0_bias = load_dit_tensor_f32(ctx, gf, "t_embedder.mlp.0.bias");
    w.t_emb_mlp2_weight = load_dit_tensor_f32(ctx, gf, "t_embedder.mlp.2.weight");
    w.t_emb_mlp2_bias = load_dit_tensor_f32(ctx, gf, "t_embedder.mlp.2.bias");

    w.x_emb_weight = load_dit_tensor_f32(ctx, gf, "x_embedder.weight");
    w.x_emb_bias = load_dit_tensor_f32(ctx, gf, "x_embedder.bias");

    // Final layer: adaln = Sequential(SiLU(), Linear(t_emb_dim, dim)) at index 1
    io::printfn("  Loading final layer...");
    w.final_adaln_linear = load_dit_tensor_f32(ctx, gf, "final_layer.adaLN_modulation.1.weight");
    w.final_adaln_bias = load_dit_tensor_f32(ctx, gf, "final_layer.adaLN_modulation.1.bias");
    w.final_linear = load_dit_tensor_f32(ctx, gf, "final_layer.linear.weight");
    w.final_bias = load_dit_tensor_f32(ctx, gf, "final_layer.linear.bias");

    // Pad tokens (optional)
    if (try info = gf.find_tensor("x_pad_token")) {
        w.x_pad_token = upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
    }
    if (try info = gf.find_tensor("cap_pad_token")) {
        w.cap_pad_token = upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
    }

    io::printfn("  DiT weights loaded.");
    return w;
}

// --- DiT Forward Pass Helpers ---

// Apply per-head RMSNorm to Q or K buffer
// qk: [seq_len, n_heads * head_dim]  (already projected)
// norm_weight: [head_dim]
fn void? dit_apply_head_norm(
    DiTModel* model,
    CommandBuffer cmd,
    Tensor* qk,
    Tensor* norm_weight,
    uint seq_len,
    uint n_heads,
    uint head_dim
) {
    SharedKernels* sk = &model.kernels.shared;
    DiTActivations* a = &model.acts;

    // For each head, extract head slice, normalize, write back
    for (uint h = 0; h < n_heads; h++) {
        for (uint pos = 0; pos < seq_len; pos++) {
            // Offset in the qk buffer: (pos * n_heads + h) * head_dim
            // Actually, layout is [seq_len, dim] where dim = n_heads * head_dim
            // head slice at pos: offset = (pos * n_heads * head_dim + h * head_dim) * 4
            usz src_offset = ((usz)pos * n_heads * head_dim + (usz)h * head_dim) * 4;

            // Copy to norm_out scratch
            vk::cmdCopyBuffer(cmd, qk.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = src_offset, .dstOffset = 0, .size = (ulong)head_dim * 4 }});
            compute_barrier(cmd);

            RMSNormPC rms_pc = { .dim = head_dim, .eps = 1e-6f };
            dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, norm_weight.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
                { (usz)head_dim * 4, norm_weight.size_bytes, (usz)head_dim * 4 },
                &rms_pc, 1);
            compute_barrier(cmd);

            vk::cmdCopyBuffer(cmd, a.ffn_down_out.gpu_buffer.buffer, qk.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = src_offset, .size = (ulong)head_dim * 4 }});
            compute_barrier(cmd);
        }
    }
}

// Run a single DiT transformer layer
fn void? dit_transformer_layer(
    DiTModel* model,
    CommandBuffer cmd,
    DiTLayerWeights* lw,
    Tensor* hidden,        // [seq_len, dim] - modified in place
    uint seq_len,
    uint img_len,          // number of image tokens (0 if no dual norm)
    bool has_adaln
) {
    SharedKernels* sk = &model.kernels.shared;
    DiTKernels* dk = &model.kernels;
    DiTActivations* a = &model.acts;
    uint dim = DIT_DIM;

    // 1. AdaLN modulation: project t_emb [256] to scale/shift params [4*dim]
    if (has_adaln && lw.adaln_linear.size_bytes > 0) {
        // adaln_params = t_emb @ adaln_weight^T + adaln_bias
        LinearProjPC adaln_proj_pc = { .out_dim = 4 * dim, .in_dim = DIT_T_EMB_DIM, .seq_len = 1 };
        dispatch_kernel(cmd, &dk.linear_proj,
            { lw.adaln_linear.gpu_buffer.buffer, lw.adaln_bias.gpu_buffer.buffer,
              a.t_emb.gpu_buffer.buffer, a.adaln_params.gpu_buffer.buffer },
            { lw.adaln_linear.size_bytes, lw.adaln_bias.size_bytes,
              a.t_emb.size_bytes, a.adaln_params.size_bytes },
            &adaln_proj_pc, 4 * dim);
        compute_barrier(cmd);

        // Split adaln_params [4*dim] into: attn_scale[dim], attn_shift[dim], ffn_scale[dim], ffn_shift[dim]
        // Copy attn_scale and attn_shift
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)dim * 4 }});
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);
    }

    // 2. Pre-attention RMSNorm (dual norm: norm1 for image, norm2 for text)
    if (img_len > 0 && lw.attn_norm2.size_bytes > 0) {
        // Image portion: positions [0..img_len-1]
        for (uint pos = 0; pos < img_len; pos++) {
            usz off = (usz)pos * dim * 4;
            vk::cmdCopyBuffer(cmd, hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = off, .dstOffset = 0, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
            RMSNormPC rms_pc = { .dim = dim, .eps = 1e-6f };
            dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, lw.attn_norm1.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer },
                { (usz)dim * 4, lw.attn_norm1.size_bytes, (usz)dim * 4 },
                &rms_pc, 1);
            compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = off, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
        }
        // Text portion: positions [img_len..seq_len-1]
        for (uint pos = img_len; pos < seq_len; pos++) {
            usz off = (usz)pos * dim * 4;
            vk::cmdCopyBuffer(cmd, hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = off, .dstOffset = 0, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
            RMSNormPC rms_pc = { .dim = dim, .eps = 1e-6f };
            dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, lw.attn_norm2.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer },
                { (usz)dim * 4, lw.attn_norm2.size_bytes, (usz)dim * 4 },
                &rms_pc, 1);
            compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = off, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
        }
    } else {
        // Single norm
        for (uint pos = 0; pos < seq_len; pos++) {
            usz off = (usz)pos * dim * 4;
            vk::cmdCopyBuffer(cmd, hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = off, .dstOffset = 0, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
            RMSNormPC rms_pc = { .dim = dim, .eps = 1e-6f };
            dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, lw.attn_norm1.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer },
                { (usz)dim * 4, lw.attn_norm1.size_bytes, (usz)dim * 4 },
                &rms_pc, 1);
            compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = off, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
        }
    }

    // 2b. Apply AdaLN modulation on image portion only
    if (has_adaln && img_len > 0) {
        AdalnModPC adaln_mod_pc = { .n_elements = img_len * dim, .dim = dim };
        dispatch_kernel(cmd, &dk.adaln_modulate,
            { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer,
              a.adaln_shift.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { a.buf_a.size_bytes, a.adaln_scale.size_bytes,
              a.adaln_shift.size_bytes, a.buf_a.size_bytes },
            &adaln_mod_pc, ceil_div(img_len * dim, 256));
        compute_barrier(cmd);
    }

    // 3. QKV projection (loop over positions, using existing matmul for quantized weights)
    // buf_a contains normed input [seq_len, dim]
    for (uint pos = 0; pos < seq_len; pos++) {
        usz in_off = (usz)pos * dim * 4;
        // Copy position to norm_out for matmul input
        vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);

        // Matmul: norm_out [dim] @ wqkv [3*dim, dim] -> buf_c [3*dim]
        // Output Q at buf_b[pos*dim], K at separate offset, V at separate offset
        MatMulPC mm_pc = { .out_dim = 3 * dim, .in_dim = dim };
        ComputeKernel* mm_kernel;
        if (lw.wqkv.dtype == GGML_Q4_K) {
            mm_kernel = &sk.matmul_q4k;
        } else if (lw.wqkv.dtype == GGML_Q5_K) {
            mm_kernel = &sk.matmul_q5k;
        } else if (lw.wqkv.dtype == GGML_Q6_K) {
            mm_kernel = &sk.matmul_q6k;
        } else if (lw.wqkv.dtype == GGML_Q8_0) {
            mm_kernel = &sk.matmul_q8;
        } else if (lw.wqkv.dtype == GGML_Q4_0) {
            mm_kernel = &sk.matmul_q4_0;
        } else {
            mm_kernel = &sk.matmul;
        }

        dispatch_kernel(cmd, mm_kernel,
            { lw.wqkv.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer },
            { lw.wqkv.size_bytes, (usz)dim * 4, (usz)3 * dim * 4 },
            &mm_pc, 3 * dim);
        compute_barrier(cmd);

        // Split QKV: Q=[0..dim-1], K=[dim..2*dim-1], V=[2*dim..3*dim-1]
        usz q_off = (usz)pos * dim * 4;
        vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.q.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = q_off, .size = (ulong)dim * 4 }});
        vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.k.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)dim * 4, .dstOffset = q_off, .size = (ulong)dim * 4 }});
        vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.v.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)dim * 2 * 4, .dstOffset = q_off, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);
    }

    // 4. Per-head Q/K norm
    dit_apply_head_norm(model, cmd, &a.q, &lw.q_norm, seq_len, DIT_HEADS, DIT_HEAD_DIM)!!;
    dit_apply_head_norm(model, cmd, &a.k, &lw.k_norm, seq_len, DIT_HEADS, DIT_HEAD_DIM)!!;

    // 5. Reshape Q,K,V from [seq_len, n_heads * head_dim] to [n_heads, seq_len, head_dim]
    // The spatial_attention shader expects [n_heads, seq_len, head_dim]
    // Currently layout is [seq_len, n_heads, head_dim] - need to transpose
    // Copy to buf_b as transposed
    for (uint h = 0; h < DIT_HEADS; h++) {
        for (uint pos = 0; pos < seq_len; pos++) {
            usz src = ((usz)pos * DIT_HEADS + h) * DIT_HEAD_DIM * 4;
            usz dst = ((usz)h * seq_len + pos) * DIT_HEAD_DIM * 4;
            vk::cmdCopyBuffer(cmd, a.q.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = src, .dstOffset = dst, .size = (ulong)DIT_HEAD_DIM * 4 }});
        }
    }
    compute_barrier(cmd);
    // Copy transposed Q back
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.q.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * dim * 4 }});
    compute_barrier(cmd);

    // Same for K
    for (uint h = 0; h < DIT_HEADS; h++) {
        for (uint pos = 0; pos < seq_len; pos++) {
            usz src = ((usz)pos * DIT_HEADS + h) * DIT_HEAD_DIM * 4;
            usz dst = ((usz)h * seq_len + pos) * DIT_HEAD_DIM * 4;
            vk::cmdCopyBuffer(cmd, a.k.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = src, .dstOffset = dst, .size = (ulong)DIT_HEAD_DIM * 4 }});
        }
    }
    compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.k.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * dim * 4 }});
    compute_barrier(cmd);

    // Same for V
    for (uint h = 0; h < DIT_HEADS; h++) {
        for (uint pos = 0; pos < seq_len; pos++) {
            usz src = ((usz)pos * DIT_HEADS + h) * DIT_HEAD_DIM * 4;
            usz dst = ((usz)h * seq_len + pos) * DIT_HEAD_DIM * 4;
            vk::cmdCopyBuffer(cmd, a.v.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = src, .dstOffset = dst, .size = (ulong)DIT_HEAD_DIM * 4 }});
        }
    }
    compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.v.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * dim * 4 }});
    compute_barrier(cmd);

    // 6. Spatial attention (non-causal)
    float scale = 1.0f / math::sqrt((float)DIT_HEAD_DIM);
    SpatialAttentionPC sa_pc = {
        .head_dim = DIT_HEAD_DIM,
        .n_heads = DIT_HEADS,
        .seq_len = seq_len,
        .scale = scale,
    };
    dispatch_kernel(cmd, &dk.spatial_attention,
        { a.q.gpu_buffer.buffer, a.k.gpu_buffer.buffer, a.v.gpu_buffer.buffer,
          a.attn_scores.gpu_buffer.buffer, a.attn_out.gpu_buffer.buffer },
        { a.q.size_bytes, a.k.size_bytes, a.v.size_bytes,
          a.attn_scores.size_bytes, a.attn_out.size_bytes },
        &sa_pc, DIT_HEADS * seq_len);
    compute_barrier(cmd);

    // 7. Transpose attention output back: [n_heads, seq_len, head_dim] -> [seq_len, dim]
    for (uint h = 0; h < DIT_HEADS; h++) {
        for (uint pos = 0; pos < seq_len; pos++) {
            usz src = ((usz)h * seq_len + pos) * DIT_HEAD_DIM * 4;
            usz dst = ((usz)pos * DIT_HEADS + h) * DIT_HEAD_DIM * 4;
            vk::cmdCopyBuffer(cmd, a.attn_out.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = src, .dstOffset = dst, .size = (ulong)DIT_HEAD_DIM * 4 }});
        }
    }
    compute_barrier(cmd);

    // 8. Output projection (loop over positions)
    for (uint pos = 0; pos < seq_len; pos++) {
        usz in_off = (usz)pos * dim * 4;
        vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);

        dispatch_matmul(cmd, sk, &lw.wo, &a.norm_out, &a.buf_c, dim, dim);
        compute_barrier(cmd);

        vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = in_off, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);
    }

    // 9. Residual add: hidden += attn_output
    ResidualPC res_pc = { .n = seq_len * dim };
    dispatch_kernel(cmd, &sk.residual_add,
        { hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { hidden.size_bytes, a.buf_a.size_bytes },
        &res_pc, ceil_div(seq_len * dim, 256));
    compute_barrier(cmd);

    // 10. FFN: Pre-FFN norm (dual or single)
    if (has_adaln) {
        // Get FFN scale/shift from adaln_params [2*dim..3*dim-1] and [3*dim..4*dim-1]
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)2 * dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)3 * dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);
    }

    if (img_len > 0 && lw.ffn_norm2.size_bytes > 0) {
        // Dual norm for FFN
        for (uint pos = 0; pos < img_len; pos++) {
            usz off = (usz)pos * dim * 4;
            vk::cmdCopyBuffer(cmd, hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = off, .dstOffset = 0, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
            RMSNormPC rms_pc = { .dim = dim, .eps = 1e-6f };
            dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, lw.ffn_norm1.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer },
                { (usz)dim * 4, lw.ffn_norm1.size_bytes, (usz)dim * 4 },
                &rms_pc, 1);
            compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = off, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
        }
        for (uint pos = img_len; pos < seq_len; pos++) {
            usz off = (usz)pos * dim * 4;
            vk::cmdCopyBuffer(cmd, hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = off, .dstOffset = 0, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
            RMSNormPC rms_pc = { .dim = dim, .eps = 1e-6f };
            dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, lw.ffn_norm2.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer },
                { (usz)dim * 4, lw.ffn_norm2.size_bytes, (usz)dim * 4 },
                &rms_pc, 1);
            compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = off, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
        }
    } else {
        for (uint pos = 0; pos < seq_len; pos++) {
            usz off = (usz)pos * dim * 4;
            vk::cmdCopyBuffer(cmd, hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = off, .dstOffset = 0, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
            RMSNormPC rms_pc = { .dim = dim, .eps = 1e-6f };
            dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, lw.ffn_norm1.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer },
                { (usz)dim * 4, lw.ffn_norm1.size_bytes, (usz)dim * 4 },
                &rms_pc, 1);
            compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = off, .size = (ulong)dim * 4 }});
            compute_barrier(cmd);
        }
    }

    // Apply AdaLN on image portion for FFN
    if (has_adaln && img_len > 0) {
        AdalnModPC adaln_mod_pc = { .n_elements = img_len * dim, .dim = dim };
        dispatch_kernel(cmd, &dk.adaln_modulate,
            { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer,
              a.adaln_shift.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { a.buf_a.size_bytes, a.adaln_scale.size_bytes,
              a.adaln_shift.size_bytes, a.buf_a.size_bytes },
            &adaln_mod_pc, ceil_div(img_len * dim, 256));
        compute_barrier(cmd);
    }

    // 11. SwiGLU FFN (loop over positions)
    for (uint pos = 0; pos < seq_len; pos++) {
        usz in_off = (usz)pos * dim * 4;
        vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);

        // Gate and Up projections
        dispatch_matmul(cmd, sk, &lw.ffn_gate, &a.norm_out, &a.ffn_gate_out, DIT_FFN_DIM, dim);
        dispatch_matmul(cmd, sk, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, DIT_FFN_DIM, dim);
        compute_barrier(cmd);

        // SiLU on gate
        SiluPC silu_pc = { .n = DIT_FFN_DIM };
        dispatch_kernel(cmd, &sk.silu,
            { a.ffn_gate_out.gpu_buffer.buffer },
            { a.ffn_gate_out.size_bytes },
            &silu_pc, ceil_div(DIT_FFN_DIM, 256));
        compute_barrier(cmd);

        // gate * up
        ElemwisePC emul_pc = { .n = DIT_FFN_DIM };
        dispatch_kernel(cmd, &sk.elemwise_mul,
            { a.ffn_gate_out.gpu_buffer.buffer, a.ffn_up_out.gpu_buffer.buffer },
            { a.ffn_gate_out.size_bytes, a.ffn_up_out.size_bytes },
            &emul_pc, ceil_div(DIT_FFN_DIM, 256));
        compute_barrier(cmd);

        // Down projection
        dispatch_matmul(cmd, sk, &lw.ffn_down, &a.ffn_gate_out, &a.ffn_down_out, dim, DIT_FFN_DIM);
        compute_barrier(cmd);

        // Write back
        vk::cmdCopyBuffer(cmd, a.ffn_down_out.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = in_off, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);
    }

    // 12. Residual add: hidden += ffn_output
    dispatch_kernel(cmd, &sk.residual_add,
        { hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { hidden.size_bytes, a.buf_a.size_bytes },
        &res_pc, ceil_div(seq_len * dim, 256));
    compute_barrier(cmd);
}

// --- Main DiT Forward Pass ---
// Takes: latent [16, h, w], text_embeddings [text_len, 3840], timestep scalar
// Returns: velocity [16, h, w] in acts.latent

fn void? DiTModel.forward(&self, Tensor* text_embeddings, float timestep) {
    DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    DiTWeights* w = &self.weights;
    DiTActivations* a = &self.acts;
    DiTKernels* dk = &self.kernels;
    SharedKernels* sk = &dk.shared;

    uint dim = DIT_DIM;
    uint n_patches = self.n_patches;
    uint text_len = self.text_len;

    begin_compute(cmd)!!;

    // 1. Patchify latent -> [n_patches, patch_dim]
    PatchifyPC patch_pc = {
        .channels = DIT_LATENT_CHANNELS,
        .height = self.latent_h,
        .width = self.latent_w,
        .patch_size = DIT_PATCH_SIZE,
    };
    dispatch_kernel(cmd, &dk.patchify,
        { a.latent.gpu_buffer.buffer, a.patches.gpu_buffer.buffer },
        { a.latent.size_bytes, a.patches.size_bytes },
        &patch_pc, ceil_div(n_patches * DIT_PATCH_DIM, 256));
    compute_barrier(cmd);

    // 2. x_embedder: [n_patches, 64] -> [n_patches, 3840]
    // Use linear_proj shader
    LinearProjPC x_emb_pc = { .out_dim = dim, .in_dim = DIT_PATCH_DIM, .seq_len = n_patches };
    dispatch_kernel(cmd, &dk.linear_proj,
        { w.x_emb_weight.gpu_buffer.buffer, w.x_emb_bias.gpu_buffer.buffer,
          a.patches.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
        { w.x_emb_weight.size_bytes, w.x_emb_bias.size_bytes,
          a.patches.size_bytes, a.hidden.size_bytes },
        &x_emb_pc, n_patches * dim);
    compute_barrier(cmd);

    // 3. Timestep embedding
    DiTTimestepPC t_pc = { .dim = DIT_T_EMB_DIM, .timestep = timestep };
    dispatch_kernel(cmd, &dk.dit_timestep_embed,
        { a.t_emb.gpu_buffer.buffer },
        { a.t_emb.size_bytes },
        &t_pc, ceil_div(DIT_T_EMB_DIM, 256));
    compute_barrier(cmd);

    // t_embedder MLP: t_emb [256] -> mlp_0 [1024] -> silu -> mlp_2 [256]
    LinearProjPC t_mlp0_pc = { .out_dim = DIT_T_MLP_DIM, .in_dim = DIT_T_EMB_DIM, .seq_len = 1 };
    dispatch_kernel(cmd, &dk.linear_proj,
        { w.t_emb_mlp0_weight.gpu_buffer.buffer, w.t_emb_mlp0_bias.gpu_buffer.buffer,
          a.t_emb.gpu_buffer.buffer, a.t_emb_mlp.gpu_buffer.buffer },
        { w.t_emb_mlp0_weight.size_bytes, w.t_emb_mlp0_bias.size_bytes,
          a.t_emb.size_bytes, a.t_emb_mlp.size_bytes },
        &t_mlp0_pc, DIT_T_MLP_DIM);
    compute_barrier(cmd);

    SiluPC silu_pc = { .n = DIT_T_MLP_DIM };
    dispatch_kernel(cmd, &sk.silu,
        { a.t_emb_mlp.gpu_buffer.buffer },
        { a.t_emb_mlp.size_bytes },
        &silu_pc, ceil_div(DIT_T_MLP_DIM, 256));
    compute_barrier(cmd);

    // mlp_2: [1024] -> [256]
    LinearProjPC t_mlp2_pc = { .out_dim = DIT_T_EMB_DIM, .in_dim = DIT_T_MLP_DIM, .seq_len = 1 };
    dispatch_kernel(cmd, &dk.linear_proj,
        { w.t_emb_mlp2_weight.gpu_buffer.buffer, w.t_emb_mlp2_bias.gpu_buffer.buffer,
          a.t_emb_mlp.gpu_buffer.buffer, a.t_emb.gpu_buffer.buffer },
        { w.t_emb_mlp2_weight.size_bytes, w.t_emb_mlp2_bias.size_bytes,
          a.t_emb_mlp.size_bytes, a.t_emb.size_bytes },
        &t_mlp2_pc, DIT_T_EMB_DIM);
    compute_barrier(cmd);
    // t_emb now contains the final [256] timestep embedding

    // 4. Noise refiner (2 layers on image tokens, with adaLN from t_emb_mlp)
    for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
        dit_transformer_layer(self, cmd, &w.noise_refiner[l], &a.hidden, n_patches, n_patches, true)!!;
    }

    // Submit and wait between major phases to avoid command buffer overflow
    submit_and_wait(ctx)!!;
    begin_compute(cmd)!!;

    // 5. Concat image + text tokens -> [n_patches + text_len, dim]
    // hidden already has image tokens [n_patches, dim] at offset 0
    // Copy text embeddings after image tokens
    // text_embeddings is [text_len, text_dim], need cap_embedder to project to [text_len, dim]
    // But text is already projected by pipeline before calling forward, so just copy
    vk::cmdCopyBuffer(cmd, text_embeddings.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)n_patches * dim * 4,
                         .size = (ulong)text_len * dim * 4 }});
    compute_barrier(cmd);

    uint total_seq = n_patches + text_len;

    // 6. Main DiT (30 layers with adaLN, dual norm)
    for (uint l = 0; l < DIT_NUM_LAYERS; l++) {
        dit_transformer_layer(self, cmd, &w.layers[l], &a.hidden, total_seq, n_patches, true)!!;

        // Submit periodically to prevent command buffer overflow
        if ((l + 1) % 5 == 0) {
            submit_and_wait(ctx)!!;
            begin_compute(cmd)!!;
        }
    }

    submit_and_wait(ctx)!!;
    begin_compute(cmd)!!;

    // 7. Final layer: extract image portion, adaLN modulate, linear projection
    // Final adaLN: SiLU(t_emb) @ weight + bias -> [dim] scale
    // First apply SiLU to t_emb copy
    vk::cmdCopyBuffer(cmd, a.t_emb.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)DIT_T_EMB_DIM * 4 }});
    compute_barrier(cmd);
    SiluPC final_silu_pc = { .n = DIT_T_EMB_DIM };
    dispatch_kernel(cmd, &sk.silu,
        { a.buf_c.gpu_buffer.buffer },
        { (usz)DIT_T_EMB_DIM * 4 },
        &final_silu_pc, ceil_div(DIT_T_EMB_DIM, 256));
    compute_barrier(cmd);

    // Linear: [256] -> [dim]
    LinearProjPC final_adaln_pc = { .out_dim = dim, .in_dim = DIT_T_EMB_DIM, .seq_len = 1 };
    dispatch_kernel(cmd, &dk.linear_proj,
        { w.final_adaln_linear.gpu_buffer.buffer, w.final_adaln_bias.gpu_buffer.buffer,
          a.buf_c.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer },
        { w.final_adaln_linear.size_bytes, w.final_adaln_bias.size_bytes,
          (usz)DIT_T_EMB_DIM * 4, a.adaln_scale.size_bytes },
        &final_adaln_pc, dim);
    compute_barrier(cmd);

    // Scale image portion: output = hidden * (1 + scale)
    // Copy image portion to buf_a
    vk::cmdCopyBuffer(cmd, a.hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_patches * dim * 4 }});
    compute_barrier(cmd);

    // Use adaln_modulate with shift=0: output = input * (1 + scale) + 0
    // Zero the shift buffer
    AdalnModPC final_mod_pc = { .n_elements = n_patches * dim, .dim = dim };
    dispatch_kernel(cmd, &dk.adaln_modulate,
        { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer,
          a.adaln_shift.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.adaln_scale.size_bytes,
          a.adaln_shift.size_bytes, a.buf_a.size_bytes },
        &final_mod_pc, ceil_div(n_patches * dim, 256));
    compute_barrier(cmd);

    // Final linear: [dim] -> [patch_dim] per position
    for (uint pos = 0; pos < n_patches; pos++) {
        usz in_off = (usz)pos * dim * 4;
        vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = in_off, .dstOffset = 0, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);

        // matmul: [patch_dim, dim] @ norm_out [dim] -> [patch_dim]
        dispatch_matmul(cmd, sk, &w.final_linear, &a.norm_out, &a.buf_c, DIT_PATCH_DIM, dim);
        compute_barrier(cmd);

        // Add bias
        BroadcastAddDimPC bias_pc = { .n_total = DIT_PATCH_DIM, .dim = DIT_PATCH_DIM };
        dispatch_kernel(cmd, &dk.broadcast_add_dim,
            { a.buf_c.gpu_buffer.buffer, w.final_bias.gpu_buffer.buffer },
            { (usz)DIT_PATCH_DIM * 4, w.final_bias.size_bytes },
            &bias_pc, ceil_div(DIT_PATCH_DIM, 256));
        compute_barrier(cmd);

        vk::cmdCopyBuffer(cmd, a.buf_c.gpu_buffer.buffer, a.velocity.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (usz)pos * DIT_PATCH_DIM * 4,
                             .size = (ulong)DIT_PATCH_DIM * 4 }});
        compute_barrier(cmd);
    }

    // 8. Unpatchify velocity -> [16, h, w]
    UnpatchifyPC unpatch_pc = {
        .channels = DIT_LATENT_CHANNELS,
        .height = self.latent_h,
        .width = self.latent_w,
        .patch_size = DIT_PATCH_SIZE,
    };
    uint latent_total = DIT_LATENT_CHANNELS * self.latent_h * self.latent_w;
    dispatch_kernel(cmd, &dk.unpatchify,
        { a.velocity.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.velocity.size_bytes, a.buf_a.size_bytes },
        &unpatch_pc, ceil_div(latent_total, 256));
    compute_barrier(cmd);

    // Copy unpatchified velocity back to velocity buffer (for Euler step)
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.velocity.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_total * 4 }});

    submit_and_wait(ctx)!!;
}

// --- Free ---

fn void DiTLayerWeights.free(&self) {
    if (self.wqkv.size_bytes > 0) self.wqkv.free();
    if (self.q_norm.size_bytes > 0) self.q_norm.free();
    if (self.k_norm.size_bytes > 0) self.k_norm.free();
    if (self.wo.size_bytes > 0) self.wo.free();
    if (self.ffn_gate.size_bytes > 0) self.ffn_gate.free();
    if (self.ffn_up.size_bytes > 0) self.ffn_up.free();
    if (self.ffn_down.size_bytes > 0) self.ffn_down.free();
    if (self.attn_norm1.size_bytes > 0) self.attn_norm1.free();
    if (self.attn_norm2.size_bytes > 0) self.attn_norm2.free();
    if (self.ffn_norm1.size_bytes > 0) self.ffn_norm1.free();
    if (self.ffn_norm2.size_bytes > 0) self.ffn_norm2.free();
    if (self.adaln_linear.size_bytes > 0) self.adaln_linear.free();
    if (self.adaln_bias.size_bytes > 0) self.adaln_bias.free();
}

fn void DiTModel.free(&self) {
    // Free layer weights
    for (uint l = 0; l < DIT_NUM_LAYERS; l++) {
        self.weights.layers[l].free();
    }
    for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
        self.weights.noise_refiner[l].free();
        self.weights.context_refiner[l].free();
    }

    // Free embedder weights
    if (self.weights.cap_emb_norm.size_bytes > 0) self.weights.cap_emb_norm.free();
    if (self.weights.cap_emb_weight.size_bytes > 0) self.weights.cap_emb_weight.free();
    if (self.weights.cap_emb_bias.size_bytes > 0) self.weights.cap_emb_bias.free();
    if (self.weights.t_emb_mlp0_weight.size_bytes > 0) self.weights.t_emb_mlp0_weight.free();
    if (self.weights.t_emb_mlp0_bias.size_bytes > 0) self.weights.t_emb_mlp0_bias.free();
    if (self.weights.t_emb_mlp2_weight.size_bytes > 0) self.weights.t_emb_mlp2_weight.free();
    if (self.weights.t_emb_mlp2_bias.size_bytes > 0) self.weights.t_emb_mlp2_bias.free();
    if (self.weights.x_emb_weight.size_bytes > 0) self.weights.x_emb_weight.free();
    if (self.weights.x_emb_bias.size_bytes > 0) self.weights.x_emb_bias.free();
    if (self.weights.final_adaln_linear.size_bytes > 0) self.weights.final_adaln_linear.free();
    if (self.weights.final_adaln_bias.size_bytes > 0) self.weights.final_adaln_bias.free();
    if (self.weights.final_linear.size_bytes > 0) self.weights.final_linear.free();
    if (self.weights.final_bias.size_bytes > 0) self.weights.final_bias.free();
    if (self.weights.x_pad_token.size_bytes > 0) self.weights.x_pad_token.free();
    if (self.weights.cap_pad_token.size_bytes > 0) self.weights.cap_pad_token.free();

    // Free activations
    self.acts.buf_a.free();
    self.acts.buf_b.free();
    self.acts.buf_c.free();
    self.acts.patches.free();
    self.acts.hidden.free();
    self.acts.norm_out.free();
    self.acts.q.free();
    self.acts.k.free();
    self.acts.v.free();
    self.acts.attn_scores.free();
    self.acts.attn_out.free();
    self.acts.ffn_gate_out.free();
    self.acts.ffn_up_out.free();
    self.acts.ffn_down_out.free();
    self.acts.t_emb.free();
    self.acts.t_emb_mlp.free();
    self.acts.adaln_params.free();
    self.acts.adaln_scale.free();
    self.acts.adaln_shift.free();
    self.acts.velocity.free();
    self.acts.latent.free();

    // Free kernels
    self.kernels.patchify.free(self.ctx.device);
    self.kernels.unpatchify.free(self.ctx.device);
    self.kernels.adaln_modulate.free(self.ctx.device);
    self.kernels.flow_euler_step.free(self.ctx.device);
    self.kernels.linear_proj.free(self.ctx.device);
    self.kernels.dit_timestep_embed.free(self.ctx.device);
    self.kernels.broadcast_add_dim.free(self.ctx.device);
    self.kernels.scale_buffer.free(self.ctx.device);
    self.kernels.spatial_attention.free(self.ctx.device);
    self.kernels.shared.free(self.ctx.device);
}
