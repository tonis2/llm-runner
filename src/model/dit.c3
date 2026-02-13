module llm;

import vk;
import std::io;
import std::math;
import std::core::mem;
import std::time::clock;

// Embedded Z-Image SPIR-V shaders
const char[*] ZIMAGE_SPV = $embed("../../shaders/zimage.spv");
const char[*] MROPE_SPV = $embed("../../shaders/mrope.spv");

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

// mRoPE config (from Z-Image-Turbo config.json)
const float DIT_ROPE_THETA = 256.0;
const uint[3] DIT_ROPE_AXES_DIM = { 32, 48, 48 };  // text, height, width
const float DIT_T_PERIOD = 1000.0;  // Maps sigma [0,1] to model timestep [1000,0]

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

struct BatchLayerNormPC {
    uint dim;
    float eps;
    uint n_rows;
    uint _pad;
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

struct BatchRMSNormPC {
    uint dim;
    float eps;
    uint n_rows;
    uint row_offset;
}

struct TransposeHeadsPC {
    uint seq_len;
    uint n_heads;
    uint head_dim;
    uint direction;
}

struct BatchHeadNormPC {
    uint n_heads;
    uint head_dim;
    uint seq_len;
    float eps;
}

struct FlashAttentionPC {
    uint head_dim;
    uint n_heads;
    uint seq_len;
    float scale;
}

struct ScaleModPC {
    uint n_elements;
    uint dim;
}

struct GatedResidualPC {
    uint n_elements;
    uint dim;
}

struct MRoPEPC {
    uint seq_len;
    uint n_heads;
    uint _pad0;
    uint _pad1;
}

struct BatchMatMulPC {
    uint out_dim;
    uint in_dim;
    uint seq_len;
    uint row_offset;
}

// Profile flag: when true, the next transformer layer call inserts GPU syncs per phase
bool dit_profile_layer = false;

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
    // Norms (Lumina2: norm1=pre-norm, norm2=post-norm)
    Tensor attn_norm1;   // pre-attention RMSNorm
    Tensor attn_norm2;   // post-attention RMSNorm
    Tensor ffn_norm1;    // pre-FFN RMSNorm
    Tensor ffn_norm2;    // post-FFN RMSNorm
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
    Tensor final_norm;          // RMSNorm weight [dim]
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
    ComputeKernel flash_attention;
    ComputeKernel batch_rmsnorm;
    ComputeKernel transpose_heads;
    ComputeKernel batch_head_norm;
    ComputeKernel scale_modulate;
    ComputeKernel gated_residual;
    ComputeKernel batch_layernorm;
    ComputeKernel mrope;
    ComputeKernel batch_matmul;
    ComputeKernel batch_matmul_q8;
    ComputeKernel batch_matmul_q4_0;
    ComputeKernel batch_matmul_q4k;
    ComputeKernel batch_matmul_q5k;
    ComputeKernel batch_matmul_q6k;
    SharedKernels shared;
}

// --- DiT Model ---

struct DiTModel {
    DiTWeights weights;
    DiTActivations acts;
    DiTKernels kernels;
    DeviceContext* ctx;
    uint n_patches;     // Number of real image patches
    uint text_len;      // Real text sequence length
    uint padded_patches; // n_patches padded to multiple of 32
    uint padded_text;    // text_len padded to multiple of 32
    uint latent_h;      // Latent height
    uint latent_w;      // Latent width
    // mRoPE cos/sin tables (precomputed once per config)
    Tensor rope_cos_main;    // [n_patches + text_len, head_dim/2] for main layers
    Tensor rope_sin_main;    // [n_patches + text_len, head_dim/2]
    Tensor rope_cos_refiner; // [n_patches, head_dim/2] for noise refiner
    Tensor rope_sin_refiner; // [n_patches, head_dim/2]
    Tensor rope_cos_context; // [text_len, head_dim/2] for context refiner
    Tensor rope_sin_context; // [text_len, head_dim/2]
}

// --- Kernel Creation ---

fn DiTKernels? create_dit_kernels(DeviceContext* ctx) {
    io::printfn("Creating DiT compute kernels...");

    char[] zi_spv = &ZIMAGE_SPV;
    ShaderModule zi_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(zi_spv.len)
        .setCode((uint*)&zi_spv[0])
        .build(ctx.device)!!;

    // Shared kernels from LLM shader (rmsnorm, matmul variants, silu, etc.)
    char[] llm_spv = &LLM_SPV;
    ShaderModule llm_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(llm_spv.len)
        .setCode((uint*)&llm_spv[0])
        .build(ctx.device)!!;

    io::printfn("  Creating zimage kernels...");
    ComputeKernel k_patchify = create_kernel(ctx, zi_shader, 2, PatchifyPC.sizeof, "patchify")!!;
    ComputeKernel k_unpatchify = create_kernel(ctx, zi_shader, 2, UnpatchifyPC.sizeof, "unpatchify")!!;
    ComputeKernel k_adaln = create_kernel(ctx, zi_shader, 4, AdalnModPC.sizeof, "adaln_modulate")!!;
    ComputeKernel k_euler = create_kernel(ctx, zi_shader, 2, FlowEulerPC.sizeof, "flow_euler_step")!!;
    ComputeKernel k_linproj = create_kernel(ctx, zi_shader, 4, LinearProjPC.sizeof, "linear_proj")!!;
    ComputeKernel k_timestep = create_kernel(ctx, zi_shader, 1, DiTTimestepPC.sizeof, "dit_timestep_embed")!!;
    ComputeKernel k_bcastadd = create_kernel(ctx, zi_shader, 2, BroadcastAddDimPC.sizeof, "broadcast_add_dim")!!;
    ComputeKernel k_scalebuf = create_kernel(ctx, zi_shader, 1, ScalePC.sizeof, "scale_buffer")!!;
    ComputeKernel k_flashattn = create_kernel(ctx, zi_shader, 4, FlashAttentionPC.sizeof, "flash_attention")!!;
    ComputeKernel k_scalemod = create_kernel(ctx, zi_shader, 3, ScaleModPC.sizeof, "scale_modulate")!!;
    ComputeKernel k_gatedres = create_kernel(ctx, zi_shader, 3, GatedResidualPC.sizeof, "gated_residual")!!;
    io::printfn("  Creating mrope kernel...");
    char[] mrope_spv = &MROPE_SPV;
    ShaderModule mrope_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(mrope_spv.len)
        .setCode((uint*)&mrope_spv[0])
        .build(ctx.device)!!;
    ComputeKernel k_mrope = create_kernel(ctx, mrope_shader, 3, MRoPEPC.sizeof, "mrope")!!;
    mrope_shader.free(ctx.device);
    io::printfn("  mrope kernel created.");
    DiTKernels kernels = {
        .patchify         = k_patchify,
        .unpatchify       = k_unpatchify,
        .adaln_modulate   = k_adaln,
        .flow_euler_step  = k_euler,
        .linear_proj      = k_linproj,
        .dit_timestep_embed = k_timestep,
        .broadcast_add_dim = k_bcastadd,
        .scale_buffer     = k_scalebuf,
        .flash_attention   = k_flashattn,
        .scale_modulate   = k_scalemod,
        .gated_residual   = k_gatedres,
        .batch_layernorm  = create_kernel(ctx, zi_shader, 2, BatchLayerNormPC.sizeof, "batch_layernorm")!!,
        .mrope            = k_mrope,
        .batch_rmsnorm    = create_kernel(ctx, zi_shader, 3, BatchRMSNormPC.sizeof, "batch_rmsnorm")!!,
        .transpose_heads  = create_kernel(ctx, zi_shader, 2, TransposeHeadsPC.sizeof, "transpose_heads")!!,
        .batch_head_norm  = create_kernel(ctx, zi_shader, 2, BatchHeadNormPC.sizeof, "batch_head_norm")!!,
        .batch_matmul     = create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul")!!,
        .batch_matmul_q8  = create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q8")!!,
        .batch_matmul_q4_0 = create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q4_0")!!,
        .batch_matmul_q4k = create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q4k")!!,
        .batch_matmul_q5k = create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q5k")!!,
        .batch_matmul_q6k = create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q6k")!!,
        .shared           = create_shared_kernels(ctx, llm_shader)!!,
    };

    zi_shader.free(ctx.device);
    llm_shader.free(ctx.device);
    return kernels;
}

// --- Activation Allocation ---

fn DiTActivations? allocate_dit_activations(DeviceContext* ctx, uint max_seq_len) {
    io::printfn("Allocating DiT activation buffers (max_seq=%d)...", max_seq_len);

    ulong max_dim_total = (ulong)max_seq_len * DIT_DIM;
    ulong max_ffn_total = (ulong)max_seq_len * DIT_FFN_DIM;
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

// --- mRoPE Precomputation ---
// Precompute cos/sin tables for multimodal RoPE
// Our layout: [image_real, image_pad, text_real, text_pad]
// Reference layout: [text_real, text_pad, image_real, image_pad]
// Position IDs:
//   image_real[i]: (padded_text_len + 1, row, col)
//   image_pad[j]:  (0, 0, 0) — identity rotation
//   text_real[k]:  (k + 1, 0, 0) — 1-indexed
//   text_pad[m]:   (text_len + m + 1, 0, 0) — continues text sequence

const uint MROPE_MODE_FULL = 0;     // padded image + padded text (main layers)
const uint MROPE_MODE_IMAGE = 1;    // padded image only (noise refiner)
const uint MROPE_MODE_TEXT = 2;     // padded text only (context refiner)
const uint SEQ_MULTI_OF = 32;      // Pad sequences to multiples of this

fn uint pad_to_multiple(uint n, uint multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

fn void? precompute_mrope_tables(
    DeviceContext* ctx,
    Tensor* cos_out, Tensor* sin_out,
    uint n_patches, uint patches_w, uint patches_h,
    uint text_len, uint mode
) {
    uint padded_text = pad_to_multiple(text_len, SEQ_MULTI_OF);
    uint padded_patches = pad_to_multiple(n_patches, SEQ_MULTI_OF);
    uint n_text_pad = padded_text - text_len;
    uint n_img_pad = padded_patches - n_patches;

    uint seq_len;
    switch (mode) {
        case MROPE_MODE_FULL: seq_len = padded_patches + padded_text;
        case MROPE_MODE_IMAGE: seq_len = padded_patches;
        case MROPE_MODE_TEXT: seq_len = padded_text;
        default: seq_len = padded_patches + padded_text;
    }
    uint half_dim = DIT_HEAD_DIM / 2;  // 64
    usz table_size = (usz)seq_len * half_dim;

    float* cos_data = (float*)mem::calloc(table_size * 4);
    float* sin_data = (float*)mem::calloc(table_size * 4);

    // Image text-axis position = padded_text_len + 1
    uint image_index = padded_text + 1;

    // For each token, compute rotation angles for all 3 axes
    for (uint t = 0; t < seq_len; t++) {
        uint pos_text;   // axis 0
        uint pos_row;    // axis 1
        uint pos_col;    // axis 2

        if (mode == MROPE_MODE_TEXT) {
            // All tokens are text (real + pad), 1-indexed
            pos_text = t + 1;
            pos_row = 0;
            pos_col = 0;
        } else if (mode == MROPE_MODE_IMAGE) {
            if (t < n_patches) {
                pos_text = image_index;
                pos_row = t / patches_w;
                pos_col = t % patches_w;
            } else {
                // Image padding: identity rotation (all zeros)
                pos_text = 0;
                pos_row = 0;
                pos_col = 0;
            }
        } else {
            // FULL mode: [image_real, image_pad, text_real, text_pad]
            if (t < n_patches) {
                pos_text = image_index;
                pos_row = t / patches_w;
                pos_col = t % patches_w;
            } else if (t < padded_patches) {
                // Image padding
                pos_text = 0;
                pos_row = 0;
                pos_col = 0;
            } else {
                // Text (real + pad), 1-indexed
                uint text_idx = t - padded_patches;
                pos_text = text_idx + 1;
                pos_row = 0;
                pos_col = 0;
            }
        }

        uint pair = 0;
        // Axis 0: text position (dims 0..15)
        uint half_ax0 = DIT_ROPE_AXES_DIM[0] / 2;  // 16
        for (uint k = 0; k < half_ax0; k++) {
            float freq = 1.0f / math::pow(DIT_ROPE_THETA, (float)(2 * k) / (float)DIT_ROPE_AXES_DIM[0]);
            float angle = (float)pos_text * freq;
            cos_data[t * half_dim + pair] = math::cos(angle);
            sin_data[t * half_dim + pair] = math::sin(angle);
            pair++;
        }
        // Axis 1: row position (dims 16..39)
        uint half_ax1 = DIT_ROPE_AXES_DIM[1] / 2;  // 24
        for (uint k = 0; k < half_ax1; k++) {
            float freq = 1.0f / math::pow(DIT_ROPE_THETA, (float)(2 * k) / (float)DIT_ROPE_AXES_DIM[1]);
            float angle = (float)pos_row * freq;
            cos_data[t * half_dim + pair] = math::cos(angle);
            sin_data[t * half_dim + pair] = math::sin(angle);
            pair++;
        }
        // Axis 2: col position (dims 40..63)
        uint half_ax2 = DIT_ROPE_AXES_DIM[2] / 2;  // 24
        for (uint k = 0; k < half_ax2; k++) {
            float freq = 1.0f / math::pow(DIT_ROPE_THETA, (float)(2 * k) / (float)DIT_ROPE_AXES_DIM[2]);
            float angle = (float)pos_col * freq;
            cos_data[t * half_dim + pair] = math::cos(angle);
            sin_data[t * half_dim + pair] = math::sin(angle);
            pair++;
        }
    }

    // Upload to GPU
    *cos_out = (Tensor){
        .gpu_buffer = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: cos_data, data_size: table_size * 4,
        )!!,
        .size_bytes = table_size * 4,
        .shape = { seq_len, half_dim, 0, 0 },
        .n_dims = 2,
        .dtype = GGML_F32,
    };
    *sin_out = (Tensor){
        .gpu_buffer = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: sin_data, data_size: table_size * 4,
        )!!,
        .size_bytes = table_size * 4,
        .shape = { seq_len, half_dim, 0, 0 },
        .n_dims = 2,
        .dtype = GGML_F32,
    };

    mem::free(cos_data);
    mem::free(sin_data);

    String mode_name;
    switch (mode) {
        case MROPE_MODE_FULL: mode_name = "main";
        case MROPE_MODE_IMAGE: mode_name = "noise_refiner";
        case MROPE_MODE_TEXT: mode_name = "context_refiner";
        default: mode_name = "unknown";
    }
    io::printfn("  mRoPE tables: seq_len=%d, half_dim=%d, pad=%d (%s)",
        seq_len, half_dim, seq_len - (mode == MROPE_MODE_TEXT ? text_len : mode == MROPE_MODE_IMAGE ? n_patches : n_patches + text_len), mode_name);
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
        // Must load as F32 since linear_proj shader expects F32 weights
        lw.adaln_linear = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "adaLN_modulation.0.weight"));
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

    // Verify tensor shapes for t_embedder and adaLN
    if (try info = gf.find_tensor("t_embedder.mlp.0.weight")) {
        io::printfn("  [shape] t_emb.mlp.0.weight: [%d, %d]", info.shape[0], info.shape[1]);
    }
    if (try info = gf.find_tensor("t_embedder.mlp.2.weight")) {
        io::printfn("  [shape] t_emb.mlp.2.weight: [%d, %d]", info.shape[0], info.shape[1]);
    }
    if (try info = gf.find_tensor("layers.0.adaLN_modulation.0.weight")) {
        io::printfn("  [shape] layers.0.adaLN_modulation.0.weight: [%d, %d]", info.shape[0], info.shape[1]);
    }
    if (try info = gf.find_tensor("noise_refiner.0.adaLN_modulation.0.weight")) {
        io::printfn("  [shape] noise_refiner.0.adaLN_modulation.0.weight: [%d, %d]", info.shape[0], info.shape[1]);
    }
    if (try info = gf.find_tensor("layers.0.attention_norm2.weight")) {
        io::printfn("  [shape] layers.0.attention_norm2.weight: [%d]", info.shape[0]);
    }
    if (try info = gf.find_tensor("final_layer.adaLN_modulation.1.weight")) {
        io::printfn("  [shape] final_layer.adaLN_modulation.1.weight: [%d, %d]", info.shape[0], info.shape[1]);
    }

    w.x_emb_weight = load_dit_tensor_f32(ctx, gf, "x_embedder.weight");
    w.x_emb_bias = load_dit_tensor_f32(ctx, gf, "x_embedder.bias");

    // Final layer: norm + adaLN (scale-only) + linear projection
    io::printfn("  Loading final layer...");
    // Try loading final norm weight (Lumina2: LuminaLayerNormContinuous has RMSNorm)
    w.final_norm = load_dit_tensor(ctx, gf, "final_layer.norm.weight");
    if (w.final_norm.size_bytes == 0) {
        w.final_norm = load_dit_tensor(ctx, gf, "norm_out.norm.weight");
    }
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

// Apply per-head RMSNorm to Q or K buffer using batch_head_norm shader
// qk: [seq_len, n_heads * head_dim]  (in-place)
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
    DiTKernels* dk = &model.kernels;

    BatchHeadNormPC pc = {
        .n_heads = n_heads,
        .head_dim = head_dim,
        .seq_len = seq_len,
        .eps = 1e-6f,
    };
    dispatch_kernel(cmd, &dk.batch_head_norm,
        { qk.gpu_buffer.buffer, norm_weight.gpu_buffer.buffer },
        { qk.size_bytes, norm_weight.size_bytes },
        &pc, seq_len * n_heads);
    compute_barrier(cmd);
}

// Dispatch a batch matmul: weight[out_dim, in_dim] × input[seq_len, in_dim] → output[seq_len, out_dim]
// row_offset: skip rows in weight (for fused QKV split)
fn void dispatch_batch_matmul(
    CommandBuffer cmd,
    DiTKernels* dk,
    Tensor* weight,
    Tensor* input,
    Tensor* output,
    uint out_dim,
    uint in_dim,
    uint seq_len,
    uint row_offset
) {
    BatchMatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim, .seq_len = seq_len, .row_offset = row_offset };
    ComputeKernel* kernel;
    bool tiled_2d = false;
    if (weight.dtype == GGML_Q4_K) {
        kernel = &dk.batch_matmul_q4k;
        tiled_2d = true;
    } else if (weight.dtype == GGML_Q5_K) {
        kernel = &dk.batch_matmul_q5k;
        tiled_2d = true;
    } else if (weight.dtype == GGML_Q6_K) {
        kernel = &dk.batch_matmul_q6k;
        tiled_2d = true;
    } else if (weight.dtype == GGML_Q8_0) {
        kernel = &dk.batch_matmul_q8;
    } else if (weight.dtype == GGML_Q4_0) {
        kernel = &dk.batch_matmul_q4_0;
    } else {
        kernel = &dk.batch_matmul;
        tiled_2d = true;
    }
    // 2D tiled GEMM: ceil(seq_len/64) * ceil(out_dim/64) workgroups
    // Legacy (Q8_0, Q4_0): seq_len * out_dim workgroups
    uint groups = tiled_2d ? ((seq_len + 63) / 64) * ((out_dim + 63) / 64) : seq_len * out_dim;
    dispatch_kernel(cmd, kernel,
        { weight.gpu_buffer.buffer, input.gpu_buffer.buffer, output.gpu_buffer.buffer },
        { weight.size_bytes, input.size_bytes, output.size_bytes },
        &pc, groups);
}

// Check if float bits represent NaN (fast-math safe)
fn bool is_nan_bits(float v) {
    uint bits = bitcast(v, uint);
    return (bits & 0x7FFFFFFF) > 0x7F800000;
}

// Check if float bits represent Inf (fast-math safe)
fn bool is_inf_bits(float v) {
    uint bits = bitcast(v, uint);
    return (bits & 0x7FFFFFFF) == 0x7F800000;
}

// Debug: readback hidden states and compare positions to detect spatial uniformity
// Reads back [n_rows, dim] buffer and computes cross-position variance
fn void? debug_position_variance(DeviceContext* ctx, vk::Buffer src_buffer, uint n_rows, uint dim, String label) {
    usz total = (usz)n_rows * dim;
    vk::Memory stg = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null, data_size: total * 4,
    )!!;
    ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer dbg_cmd) {
        vk::cmdCopyBuffer(dbg_cmd, src_buffer, stg.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)total * 4 }});
    }!!;
    float* ptr = (float*)stg.data();

    // Print first 5 elements of 4 sample positions
    uint[4] sample_pos = { 0, n_rows / 4, n_rows / 2, 3 * n_rows / 4 };
    io::printfn("  [POS] %s (%d rows x %d dim):", label, n_rows, dim);
    for (uint si = 0; si < 4; si++) {
        uint p = sample_pos[si];
        if (p >= n_rows) continue;
        float row_sum = 0;
        for (uint d = 0; d < dim; d++) row_sum += ptr[p * dim + d];
        float row_mean = row_sum / (float)dim;
        float row_sq = 0;
        for (uint d = 0; d < dim; d++) {
            float diff = ptr[p * dim + d] - row_mean;
            row_sq += diff * diff;
        }
        float row_std = math::sqrt(row_sq / (float)dim);
        io::printfn("    pos[%d]: mean=%.6f std=%.6f first5=[%.3f %.3f %.3f %.3f %.3f]",
            p, row_mean, row_std,
            ptr[p * dim + 0], ptr[p * dim + 1], ptr[p * dim + 2],
            ptr[p * dim + 3], ptr[p * dim + 4]);
    }

    // Compute cross-position variance: for each dim, compute variance across all positions
    double cross_var_sum = 0;
    for (uint d = 0; d < 10; d++) {  // Sample first 10 dims
        double col_sum = 0;
        for (uint r = 0; r < n_rows; r++) col_sum += ptr[r * dim + d];
        double col_mean = col_sum / (double)n_rows;
        double col_var = 0;
        for (uint r = 0; r < n_rows; r++) {
            double diff = ptr[r * dim + d] - col_mean;
            col_var += diff * diff;
        }
        cross_var_sum += col_var / (double)n_rows;
    }
    io::printfn("    cross-pos variance (avg first 10 dims): %.6f", cross_var_sum / 10.0);
    stg.free();
}

// Debug: readback a GPU buffer and print stats (min/max/mean/nan count)
// Must be called OUTSIDE an active command buffer recording (after submit_and_wait)
fn void? debug_readback_stats(DeviceContext* ctx, vk::Buffer src_buffer, uint n_floats, String label) {
    vk::Memory stg = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null, data_size: (usz)n_floats * 4,
    )!!;
    ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer dbg_cmd) {
        vk::cmdCopyBuffer(dbg_cmd, src_buffer, stg.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_floats * 4 }});
    }!!;
    float* ptr = (float*)stg.data();
    float mn = 1e30; float mx = -1e30; double sm = 0;
    uint nan_count = 0; uint inf_count = 0;
    for (uint i = 0; i < n_floats; i++) {
        float v = ptr[i];
        if (is_nan_bits(v)) { nan_count++; continue; }
        if (is_inf_bits(v)) { inf_count++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sm += v;
    }
    uint valid = n_floats - nan_count - inf_count;
    float mean = valid > 0 ? (float)(sm / (double)valid) : 0;
    io::printfn("  [DBG] %s: min=%.6f max=%.6f mean=%.6f nan=%d inf=%d (%d elems)",
        label, mn, mx, mean, nan_count, inf_count, n_floats);
    stg.free();
}

// Run a single DiT transformer layer (Lumina2 architecture)
// Lumina2 adaLN: 4 params = [scale_msa, gate_msa, scale_mlp, gate_mlp]
// Pre-norm + scale modulation (no shift), post-norm + tanh-gated residual
fn void? dit_transformer_layer(
    DiTModel* model,
    CommandBuffer cmd,
    DiTLayerWeights* lw,
    Tensor* hidden,        // [seq_len, dim] - modified in place
    uint seq_len,
    bool has_adaln,
    Tensor* rope_cos,      // [seq_len, head_dim/2] precomputed cos table
    Tensor* rope_sin       // [seq_len, head_dim/2] precomputed sin table
) {
    DeviceContext* ctx = model.ctx;
    SharedKernels* sk = &model.kernels.shared;
    DiTKernels* dk = &model.kernels;
    DiTActivations* a = &model.acts;
    uint dim = DIT_DIM;
    bool profiling = dit_profile_layer;
    Clock prof_t;
    if (profiling) { prof_t = clock::now(); }

    // === Phase A: AdaLN params + Pre-attention norm + scale ===
    if (has_adaln && lw.adaln_linear.size_bytes > 0) {
        // Project t_emb -> 4 modulation params: [scale_msa, gate_msa, scale_mlp, gate_mlp]
        LinearProjPC adaln_proj_pc = { .out_dim = 4 * dim, .in_dim = DIT_T_EMB_DIM, .seq_len = 1 };
        dispatch_kernel(cmd, &dk.linear_proj,
            { lw.adaln_linear.gpu_buffer.buffer, lw.adaln_bias.gpu_buffer.buffer,
              a.t_emb.gpu_buffer.buffer, a.adaln_params.gpu_buffer.buffer },
            { lw.adaln_linear.size_bytes, lw.adaln_bias.size_bytes,
              a.t_emb.size_bytes, a.adaln_params.size_bytes },
            &adaln_proj_pc, 4 * dim);
        compute_barrier(cmd);

        // Extract scale_msa (chunk 0) and gate_msa (chunk 1)
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)dim * 4 }});
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);

        // Debug: dump adaln modulation values for first call
        if (profiling) {
            submit_and_wait(ctx)!!;
            vk::Memory adaln_stg = vk::new_buffer(
                allocator: &ctx.allocator,
                usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data: null, data_size: (usz)4 * dim * 4,
            )!!;
            begin_compute(cmd)!!;
            vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, adaln_stg.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)4 * dim * 4 }});
            submit_and_wait(ctx)!!;
            float* ap = (float*)adaln_stg.data();
            String[4] names = { "scale_msa", "gate_msa", "scale_mlp", "gate_mlp" };
            for (uint c = 0; c < 4; c++) {
                float cmin = ap[c*dim]; float cmax = cmin; float csum = 0; float csq = 0;
                for (uint i = 0; i < dim; i++) {
                    float v = ap[c * dim + i];
                    if (v < cmin) { cmin = v; }
                    if (v > cmax) { cmax = v; }
                    csum += v; csq += v * v;
                }
                float cmean = csum / (float)dim;
                io::printfn("      adaln[%s]: min=%.4f max=%.4f mean=%.4f std=%.4f",
                    names[c], cmin, cmax, cmean, math::sqrt(csq / (float)dim - cmean * cmean));
            }
            adaln_stg.free();
            begin_compute(cmd)!!;
        }
    }

    // Pre-attention RMSNorm: all tokens get same norm1
    BatchRMSNormPC rms_pc = { .dim = dim, .eps = 1e-6f, .n_rows = seq_len, .row_offset = 0 };
    dispatch_kernel(cmd, &dk.batch_rmsnorm,
        { hidden.gpu_buffer.buffer, lw.attn_norm1.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { hidden.size_bytes, lw.attn_norm1.size_bytes, a.buf_a.size_bytes },
        &rms_pc, seq_len);
    compute_barrier(cmd);

    // Scale modulation: buf_a = buf_a * (1 + scale_msa) — applied to ALL tokens
    if (has_adaln) {
        ScaleModPC smod_pc = { .n_elements = seq_len * dim, .dim = dim };
        dispatch_kernel(cmd, &dk.scale_modulate,
            { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { a.buf_a.size_bytes, a.adaln_scale.size_bytes, a.buf_a.size_bytes },
            &smod_pc, ceil_div(seq_len * dim, 256));
        compute_barrier(cmd);
    }

    if (profiling) { submit_and_wait(ctx)!!; io::printfn("      norm+adaln: %.3fs", prof_t.to_now().to_sec()); prof_t = clock::now(); begin_compute(cmd)!!; }

    // === Phase B: QKV batch matmul (3 dispatches) ===
    // wqkv is [3*dim, dim] fused — split via row_offset: Q=0, K=dim, V=2*dim
    dispatch_batch_matmul(cmd, dk, &lw.wqkv, &a.buf_a, &a.q, dim, dim, seq_len, 0);
    dispatch_batch_matmul(cmd, dk, &lw.wqkv, &a.buf_a, &a.k, dim, dim, seq_len, dim);
    dispatch_batch_matmul(cmd, dk, &lw.wqkv, &a.buf_a, &a.v, dim, dim, seq_len, 2 * dim);
    compute_barrier(cmd);

    if (profiling) { submit_and_wait(ctx)!!; io::printfn("      qkv_matmul: %.3fs", prof_t.to_now().to_sec()); prof_t = clock::now(); begin_compute(cmd)!!; }

    // === Phase C: Head norm + Forward Transpose + Flash Attention ===
    dit_apply_head_norm(model, cmd, &a.q, &lw.q_norm, seq_len, DIT_HEADS, DIT_HEAD_DIM)!!;
    dit_apply_head_norm(model, cmd, &a.k, &lw.k_norm, seq_len, DIT_HEADS, DIT_HEAD_DIM)!!;

    // Forward transpose Q/K/V: [seq_len, dim] -> [n_heads, seq_len, head_dim]
    uint total_elems = seq_len * dim;
    uint transpose_groups = ceil_div(total_elems, 256);
    TransposeHeadsPC th_fwd_pc = {
        .seq_len = seq_len, .n_heads = DIT_HEADS, .head_dim = DIT_HEAD_DIM, .direction = 0,
    };

    dispatch_kernel(cmd, &dk.transpose_heads,
        { a.q.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.q.size_bytes, a.buf_b.size_bytes },
        &th_fwd_pc, transpose_groups);
    compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.q.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)total_elems * 4 }});
    compute_barrier(cmd);

    dispatch_kernel(cmd, &dk.transpose_heads,
        { a.k.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.k.size_bytes, a.buf_b.size_bytes },
        &th_fwd_pc, transpose_groups);
    compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.k.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)total_elems * 4 }});
    compute_barrier(cmd);

    dispatch_kernel(cmd, &dk.transpose_heads,
        { a.v.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.v.size_bytes, a.buf_b.size_bytes },
        &th_fwd_pc, transpose_groups);
    compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.v.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)total_elems * 4 }});
    compute_barrier(cmd);

    // === Apply mRoPE to Q and K (after transpose, before attention) ===
    if (rope_cos != null && rope_cos.size_bytes > 0) {
        MRoPEPC mrope_pc = { .seq_len = seq_len, .n_heads = DIT_HEADS };
        // 2D dispatch: x = pairs (64 threads per workgroup), y = head*seq_len
        uint groups_y = DIT_HEADS * seq_len;

        dispatch_kernel(cmd, &dk.mrope,
            { a.q.gpu_buffer.buffer, rope_cos.gpu_buffer.buffer, rope_sin.gpu_buffer.buffer },
            { a.q.size_bytes, rope_cos.size_bytes, rope_sin.size_bytes },
            &mrope_pc, 1, groups_y);

        dispatch_kernel(cmd, &dk.mrope,
            { a.k.gpu_buffer.buffer, rope_cos.gpu_buffer.buffer, rope_sin.gpu_buffer.buffer },
            { a.k.size_bytes, rope_cos.size_bytes, rope_sin.size_bytes },
            &mrope_pc, 1, groups_y);
        compute_barrier(cmd);
    }

    // Flash attention: Q/K/V in [n_heads, seq_len, head_dim], output to buf_b in [seq_len, dim]
    // Scale = 1/head_dim (not 1/sqrt), matching KoboldCpp z_image.hpp Rope::attention scale=1.f/128.f
    // With QK-norm (head_norm), Q·K are bounded, so 1/d_k gives proper softmax range
    float scale = 1.0f / (float)DIT_HEAD_DIM;
    FlashAttentionPC fa_pc = {
        .head_dim = DIT_HEAD_DIM,
        .n_heads = DIT_HEADS,
        .seq_len = seq_len,
        .scale = scale,
    };
    dispatch_kernel(cmd, &dk.flash_attention,
        { a.q.gpu_buffer.buffer, a.k.gpu_buffer.buffer, a.v.gpu_buffer.buffer,
          a.buf_b.gpu_buffer.buffer },
        { a.q.size_bytes, a.k.size_bytes, a.v.size_bytes,
          a.buf_b.size_bytes },
        &fa_pc, DIT_HEADS * seq_len);
    compute_barrier(cmd);

    if (profiling) { submit_and_wait(ctx)!!; io::printfn("      attention:  %.3fs", prof_t.to_now().to_sec()); prof_t = clock::now(); begin_compute(cmd)!!; }

    // === Phase D: Output projection + Post-attention norm + Gated residual ===
    // Output projection: buf_b -> buf_a
    dispatch_batch_matmul(cmd, dk, &lw.wo, &a.buf_b, &a.buf_a, dim, dim, seq_len, 0);
    compute_barrier(cmd);

    // Post-attention norm: norm2(attn_output)
    if (lw.attn_norm2.size_bytes > 0) {
        BatchRMSNormPC post_attn_rms = { .dim = dim, .eps = 1e-6f, .n_rows = seq_len, .row_offset = 0 };
        dispatch_kernel(cmd, &dk.batch_rmsnorm,
            { a.buf_a.gpu_buffer.buffer, lw.attn_norm2.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
            { a.buf_a.size_bytes, lw.attn_norm2.size_bytes, a.buf_b.size_bytes },
            &post_attn_rms, seq_len);
        compute_barrier(cmd);

        // Gated residual: hidden += tanh(gate_msa) * norm2(attn_output)
        if (has_adaln) {
            GatedResidualPC gres_pc = { .n_elements = seq_len * dim, .dim = dim };
            dispatch_kernel(cmd, &dk.gated_residual,
                { hidden.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
                { hidden.size_bytes, a.adaln_shift.size_bytes, a.buf_b.size_bytes },
                &gres_pc, ceil_div(seq_len * dim, 256));
        } else {
            // Context refiner: no gate, just residual += post_norm(output)
            ResidualPC res_pc = { .n = seq_len * dim };
            dispatch_kernel(cmd, &sk.residual_add,
                { hidden.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
                { hidden.size_bytes, a.buf_b.size_bytes },
                &res_pc, ceil_div(seq_len * dim, 256));
        }
        compute_barrier(cmd);
    } else {
        // No post-norm available: direct residual
        ResidualPC res_pc = { .n = seq_len * dim };
        dispatch_kernel(cmd, &sk.residual_add,
            { hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { hidden.size_bytes, a.buf_a.size_bytes },
            &res_pc, ceil_div(seq_len * dim, 256));
        compute_barrier(cmd);
    }

    if (profiling) { submit_and_wait(ctx)!!; io::printfn("      out_proj:   %.3fs", prof_t.to_now().to_sec()); prof_t = clock::now(); begin_compute(cmd)!!; }

    // === Phase E: FFN pre-norm + scale ===
    if (has_adaln) {
        // Extract scale_mlp (chunk 2) and gate_mlp (chunk 3) from adaln_params
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)2 * dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)3 * dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
        compute_barrier(cmd);
    }

    // Pre-FFN RMSNorm: all tokens get same ffn_norm1
    BatchRMSNormPC ffn_rms_pc = { .dim = dim, .eps = 1e-6f, .n_rows = seq_len, .row_offset = 0 };
    dispatch_kernel(cmd, &dk.batch_rmsnorm,
        { hidden.gpu_buffer.buffer, lw.ffn_norm1.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { hidden.size_bytes, lw.ffn_norm1.size_bytes, a.buf_a.size_bytes },
        &ffn_rms_pc, seq_len);
    compute_barrier(cmd);

    // Scale modulation: buf_a = buf_a * (1 + scale_mlp) — applied to ALL tokens
    if (has_adaln) {
        ScaleModPC smod_pc = { .n_elements = seq_len * dim, .dim = dim };
        dispatch_kernel(cmd, &dk.scale_modulate,
            { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { a.buf_a.size_bytes, a.adaln_scale.size_bytes, a.buf_a.size_bytes },
            &smod_pc, ceil_div(seq_len * dim, 256));
        compute_barrier(cmd);
    }

    if (profiling) { submit_and_wait(ctx)!!; io::printfn("      ffn_norm:   %.3fs", prof_t.to_now().to_sec()); prof_t = clock::now(); begin_compute(cmd)!!; }

    // === Phase F: SwiGLU FFN batch matmul ===
    // Gate and Up projections (both read buf_a, write to different outputs)
    dispatch_batch_matmul(cmd, dk, &lw.ffn_gate, &a.buf_a, &a.ffn_gate_out, DIT_FFN_DIM, dim, seq_len, 0);
    dispatch_batch_matmul(cmd, dk, &lw.ffn_up, &a.buf_a, &a.ffn_up_out, DIT_FFN_DIM, dim, seq_len, 0);
    compute_barrier(cmd);

    // SiLU on gate
    SiluPC silu_pc = { .n = seq_len * DIT_FFN_DIM };
    dispatch_kernel(cmd, &sk.silu,
        { a.ffn_gate_out.gpu_buffer.buffer },
        { a.ffn_gate_out.size_bytes },
        &silu_pc, ceil_div(seq_len * DIT_FFN_DIM, 256));
    compute_barrier(cmd);

    // gate * up
    ElemwisePC emul_pc = { .n = seq_len * DIT_FFN_DIM };
    dispatch_kernel(cmd, &sk.elemwise_mul,
        { a.ffn_gate_out.gpu_buffer.buffer, a.ffn_up_out.gpu_buffer.buffer },
        { a.ffn_gate_out.size_bytes, a.ffn_up_out.size_bytes },
        &emul_pc, ceil_div(seq_len * DIT_FFN_DIM, 256));
    compute_barrier(cmd);

    // Down projection: [seq_len, ffn_dim] -> [seq_len, dim]
    dispatch_batch_matmul(cmd, dk, &lw.ffn_down, &a.ffn_gate_out, &a.ffn_down_out, dim, DIT_FFN_DIM, seq_len, 0);
    compute_barrier(cmd);

    // === Phase G: Post-FFN norm + Gated residual ===
    if (lw.ffn_norm2.size_bytes > 0) {
        BatchRMSNormPC post_ffn_rms = { .dim = dim, .eps = 1e-6f, .n_rows = seq_len, .row_offset = 0 };
        dispatch_kernel(cmd, &dk.batch_rmsnorm,
            { a.ffn_down_out.gpu_buffer.buffer, lw.ffn_norm2.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { a.ffn_down_out.size_bytes, lw.ffn_norm2.size_bytes, a.buf_a.size_bytes },
            &post_ffn_rms, seq_len);
        compute_barrier(cmd);

        // Gated residual: hidden += tanh(gate_mlp) * ffn_norm2(ffn_output)
        if (has_adaln) {
            GatedResidualPC gres_pc = { .n_elements = seq_len * dim, .dim = dim };
            dispatch_kernel(cmd, &dk.gated_residual,
                { hidden.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
                { hidden.size_bytes, a.adaln_shift.size_bytes, a.buf_a.size_bytes },
                &gres_pc, ceil_div(seq_len * dim, 256));
        } else {
            // Context refiner: no gate
            ResidualPC res_pc = { .n = seq_len * dim };
            dispatch_kernel(cmd, &sk.residual_add,
                { hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
                { hidden.size_bytes, a.buf_a.size_bytes },
                &res_pc, ceil_div(seq_len * dim, 256));
        }
        compute_barrier(cmd);
    } else {
        // No post-norm: direct residual
        ResidualPC res_pc = { .n = seq_len * dim };
        dispatch_kernel(cmd, &sk.residual_add,
            { hidden.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
            { hidden.size_bytes, a.ffn_down_out.size_bytes },
            &res_pc, ceil_div(seq_len * dim, 256));
        compute_barrier(cmd);
    }

    if (profiling) { submit_and_wait(ctx)!!; io::printfn("      ffn:        %.3fs", prof_t.to_now().to_sec()); dit_profile_layer = false; begin_compute(cmd)!!; }
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
    uint padded_patches = self.padded_patches;
    uint padded_text = self.padded_text;
    uint n_img_pad = padded_patches - n_patches;

    Clock step_start = clock::now();
    Clock phase_start = step_start;

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
    LinearProjPC x_emb_pc = { .out_dim = dim, .in_dim = DIT_PATCH_DIM, .seq_len = n_patches };
    dispatch_kernel(cmd, &dk.linear_proj,
        { w.x_emb_weight.gpu_buffer.buffer, w.x_emb_bias.gpu_buffer.buffer,
          a.patches.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
        { w.x_emb_weight.size_bytes, w.x_emb_bias.size_bytes,
          a.patches.size_bytes, a.hidden.size_bytes },
        &x_emb_pc, n_patches * dim);
    compute_barrier(cmd);

    // 2b. Pad image tokens with x_pad_token (append n_img_pad copies)
    if (n_img_pad > 0 && w.x_pad_token.size_bytes > 0) {
        for (uint p = 0; p < n_img_pad; p++) {
            vk::cmdCopyBuffer(cmd, w.x_pad_token.gpu_buffer.buffer,
                a.hidden.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0,
                    .dstOffset = (ulong)(n_patches + p) * dim * 4,
                    .size = (ulong)dim * 4 }});
        }
        compute_barrier(cmd);
    }

    // Debug: check spatial diversity after x_embedder
    submit_and_wait(ctx)!!;
    debug_position_variance(ctx, a.hidden.gpu_buffer.buffer, n_patches, dim, "after x_embedder")!!;
    begin_compute(cmd)!!;

    // 3. Timestep embedding: sigma is already in [0,1], map to timestep [0,1000]
    // sigma=1.0 (fully noisy) -> timestep=1000, sigma=0.0 (fully denoised) -> timestep=0
    DiTTimestepPC t_pc = { .dim = DIT_T_EMB_DIM, .timestep = DIT_T_PERIOD * timestep };
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

    // NOTE: No SiLU here — JointTransformerBlock's adaLN uses t_emb directly (weight at .0)
    // Only FinalLayer applies SiLU before its adaLN Linear (weight at .1 in Sequential(SiLU, Linear))

    // Debug: readback t_emb after MLP + SiLU
    if (dit_profile_layer) {
        submit_and_wait(ctx)!!;
        vk::Memory t_stg = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: (usz)DIT_T_EMB_DIM * 4,
        )!!;
        begin_compute(cmd)!!;
        vk::cmdCopyBuffer(cmd, a.t_emb.gpu_buffer.buffer, t_stg.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)DIT_T_EMB_DIM * 4 }});
        submit_and_wait(ctx)!!;
        float* t_data = (float*)t_stg.data();
        io::printf("  [DBG] t_emb (after MLP+SiLU, 256d): first10=[");
        float t_min = t_data[0]; float t_max = t_data[0]; float t_sum = 0; float t_sq = 0;
        for (uint i = 0; i < DIT_T_EMB_DIM; i++) {
            if (i < 10) { io::printf("%.4f ", t_data[i]); }
            if (t_data[i] < t_min) { t_min = t_data[i]; }
            if (t_data[i] > t_max) { t_max = t_data[i]; }
            t_sum += t_data[i]; t_sq += t_data[i] * t_data[i];
        }
        io::printfn("] min=%.4f max=%.4f mean=%.4f std=%.4f",
            t_min, t_max, t_sum / DIT_T_EMB_DIM,
            math::sqrt(t_sq / DIT_T_EMB_DIM - (t_sum / DIT_T_EMB_DIM) * (t_sum / DIT_T_EMB_DIM)));
        t_stg.free();
        begin_compute(cmd)!!;
    }

    // Save debug flag before noise refiner consumes it
    bool want_layer_debug = dit_profile_layer;

    // 4. Noise refiner (2 layers on padded image tokens, with adaLN)
    for (uint l = 0; l < DIT_NUM_REFINER_LAYERS; l++) {
        dit_transformer_layer(self, cmd, &w.noise_refiner[l], &a.hidden, padded_patches, true,
            &self.rope_cos_refiner, &self.rope_sin_refiner)!!;
    }

    submit_and_wait(ctx)!!;
    // Debug: check spatial diversity after noise refiner
    debug_position_variance(ctx, a.hidden.gpu_buffer.buffer, n_patches, dim, "after noise_refiner")!!;
    double setup_sec = phase_start.to_now().to_sec();
    phase_start = clock::now();
    begin_compute(cmd)!!;

    // 5. Concat padded_image + padded_text -> [padded_patches + padded_text, dim]
    // hidden already has image tokens [padded_patches, dim] at offset 0
    // Copy padded text embeddings after padded image tokens
    // text_embeddings is [padded_text, dim] (already padded by pipeline)
    vk::cmdCopyBuffer(cmd, text_embeddings.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)padded_patches * dim * 4,
                         .size = (ulong)padded_text * dim * 4 }});
    compute_barrier(cmd);

    uint total_seq = padded_patches + padded_text;

    // 6. Main DiT (30 layers with adaLN, post-norm, tanh gating)
    double[16] batch_times;
    uint batch_idx = 0;
    Clock batch_start = clock::now();
    for (uint l = 0; l < DIT_NUM_LAYERS; l++) {
        if (l == 0 && dit_profile_layer) {
            io::printfn("    --- Layer profile (layer 1, seq=%d) ---", total_seq);
        }
        dit_transformer_layer(self, cmd, &w.layers[l], &a.hidden, total_seq, true,
            &self.rope_cos_main, &self.rope_sin_main)!!;

        // Debug: readback after layers 0, 1, 5, 10, 20, 29
        if (want_layer_debug && (l == 0 || l == 1 || l == 5 || l == 10 || l == 20 || l == 29)) {
            submit_and_wait(ctx)!!;
            if (l == 0) {
                debug_position_variance(ctx, a.hidden.gpu_buffer.buffer, n_patches, dim, "after main layer 0")!!;
            } else if (l == 1) {
                debug_position_variance(ctx, a.hidden.gpu_buffer.buffer, n_patches, dim, "after main layer 1")!!;
            } else if (l == 5) {
                debug_position_variance(ctx, a.hidden.gpu_buffer.buffer, n_patches, dim, "after main layer 5")!!;
            } else if (l == 10) {
                debug_position_variance(ctx, a.hidden.gpu_buffer.buffer, n_patches, dim, "after main layer 10")!!;
            } else if (l == 20) {
                debug_position_variance(ctx, a.hidden.gpu_buffer.buffer, n_patches, dim, "after main layer 20")!!;
            } else if (l == 29) {
                debug_position_variance(ctx, a.hidden.gpu_buffer.buffer, n_patches, dim, "after main layer 29")!!;
            }
            batch_start = clock::now();
            begin_compute(cmd)!!;
        }

        if ((l + 1) % 10 == 0 || l == DIT_NUM_LAYERS - 1) {
            submit_and_wait(ctx)!!;
            batch_times[batch_idx] = batch_start.to_now().to_sec();
            batch_idx++;
            batch_start = clock::now();
            begin_compute(cmd)!!;
        }
    }
    double dit_sec = phase_start.to_now().to_sec();
    phase_start = clock::now();
    io::printf("    layer batches:");
    for (uint i = 0; i < batch_idx; i++) {
        io::printf(" %.2f", batch_times[i]);
    }
    io::printn("");

    // 7. Final layer: extract REAL image tokens (first n_patches), skip padding
    // FinalLayer adaLN: Sequential(SiLU(), Linear) — weight at .1, so apply SiLU to t_emb first
    // Use t_emb_mlp as temp buffer for SiLU(t_emb) (safe: t_emb_mlp not used after MLP phase)
    vk::cmdCopyBuffer(cmd, a.t_emb.gpu_buffer.buffer, a.t_emb_mlp.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)DIT_T_EMB_DIM * 4 }});
    compute_barrier(cmd);
    SiluPC final_silu_pc = { .n = DIT_T_EMB_DIM };
    dispatch_kernel(cmd, &sk.silu,
        { a.t_emb_mlp.gpu_buffer.buffer },
        { a.t_emb_mlp.size_bytes },
        &final_silu_pc, ceil_div(DIT_T_EMB_DIM, 256));
    compute_barrier(cmd);

    // Final adaLN: SiLU(t_emb) @ weight + bias -> [dim] scale
    LinearProjPC final_adaln_pc = { .out_dim = dim, .in_dim = DIT_T_EMB_DIM, .seq_len = 1 };
    dispatch_kernel(cmd, &dk.linear_proj,
        { w.final_adaln_linear.gpu_buffer.buffer, w.final_adaln_bias.gpu_buffer.buffer,
          a.t_emb_mlp.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer },
        { w.final_adaln_linear.size_bytes, w.final_adaln_bias.size_bytes,
          a.t_emb_mlp.size_bytes, a.adaln_scale.size_bytes },
        &final_adaln_pc, dim);
    compute_barrier(cmd);

    // Copy only real image portion to buf_a (first n_patches tokens, skip img/text padding)
    vk::cmdCopyBuffer(cmd, a.hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_patches * dim * 4 }});
    compute_barrier(cmd);

    // Apply LayerNorm (no affine) to image portion — Lumina2 FinalLayer uses elementwise_affine=False
    if (w.final_norm.size_bytes > 0) {
        // Learnable RMSNorm (if weight tensor exists in GGUF)
        BatchRMSNormPC final_rms = { .dim = dim, .eps = 1e-6f, .n_rows = n_patches, .row_offset = 0 };
        dispatch_kernel(cmd, &dk.batch_rmsnorm,
            { a.buf_a.gpu_buffer.buffer, w.final_norm.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
            { a.buf_a.size_bytes, w.final_norm.size_bytes, a.buf_b.size_bytes },
            &final_rms, n_patches);
        compute_barrier(cmd);
        vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_patches * dim * 4 }});
        compute_barrier(cmd);
    } else {
        // No-affine LayerNorm (Lumina2 default: no learnable weight)
        BatchLayerNormPC final_ln = { .dim = dim, .eps = 1e-6f, .n_rows = n_patches };
        dispatch_kernel(cmd, &dk.batch_layernorm,
            { a.buf_a.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
            { a.buf_a.size_bytes, a.buf_b.size_bytes },
            &final_ln, n_patches);
        compute_barrier(cmd);
        vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_patches * dim * 4 }});
        compute_barrier(cmd);
    }

    // Scale modulate: buf_a = buf_a * (1 + scale)
    ScaleModPC final_smod_pc = { .n_elements = n_patches * dim, .dim = dim };
    dispatch_kernel(cmd, &dk.scale_modulate,
        { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.adaln_scale.size_bytes, a.buf_a.size_bytes },
        &final_smod_pc, ceil_div(n_patches * dim, 256));
    compute_barrier(cmd);

    // Final linear: batch matmul [n_patches, dim] -> [n_patches, patch_dim] + bias
    dispatch_batch_matmul(cmd, dk, &w.final_linear, &a.buf_a, &a.velocity, DIT_PATCH_DIM, dim, n_patches, 0);
    compute_barrier(cmd);

    // Add bias (broadcast [patch_dim] across all patches)
    BroadcastAddDimPC bias_pc = { .n_total = n_patches * DIT_PATCH_DIM, .dim = DIT_PATCH_DIM };
    dispatch_kernel(cmd, &dk.broadcast_add_dim,
        { a.velocity.gpu_buffer.buffer, w.final_bias.gpu_buffer.buffer },
        { a.velocity.size_bytes, w.final_bias.size_bytes },
        &bias_pc, ceil_div(n_patches * DIT_PATCH_DIM, 256));
    compute_barrier(cmd);

    // Debug: check velocity (patch space) after final layer
    if (want_layer_debug) {
        submit_and_wait(ctx)!!;
        debug_position_variance(ctx, a.velocity.gpu_buffer.buffer, n_patches, DIT_PATCH_DIM, "velocity (patch space)")!!;
        // Also check hidden state after final layernorm+scale
        debug_position_variance(ctx, a.buf_a.gpu_buffer.buffer, n_patches, dim, "after final layer (pre-linear)")!!;
        begin_compute(cmd)!!;
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
    // Note: No negation needed - flow matching predicts velocity from noise->data,
    // and Euler step x = x + dt * v with negative dt naturally moves noise->data
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.velocity.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_total * 4 }});

    submit_and_wait(ctx)!!;
    double final_sec = phase_start.to_now().to_sec();
    double total_sec = step_start.to_now().to_sec();
    io::printfn("    setup+refiner: %.2fs | dit_layers: %.2fs | final: %.2fs | total: %.2fs",
        setup_sec, dit_sec, final_sec, total_sec);
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
    if (self.weights.final_norm.size_bytes > 0) self.weights.final_norm.free();
    if (self.weights.final_adaln_linear.size_bytes > 0) self.weights.final_adaln_linear.free();
    if (self.weights.final_adaln_bias.size_bytes > 0) self.weights.final_adaln_bias.free();
    if (self.weights.final_linear.size_bytes > 0) self.weights.final_linear.free();
    if (self.weights.final_bias.size_bytes > 0) self.weights.final_bias.free();
    if (self.weights.x_pad_token.size_bytes > 0) self.weights.x_pad_token.free();
    if (self.weights.cap_pad_token.size_bytes > 0) self.weights.cap_pad_token.free();

    // Free RoPE tables
    if (self.rope_cos_main.size_bytes > 0) self.rope_cos_main.free();
    if (self.rope_sin_main.size_bytes > 0) self.rope_sin_main.free();
    if (self.rope_cos_refiner.size_bytes > 0) self.rope_cos_refiner.free();
    if (self.rope_sin_refiner.size_bytes > 0) self.rope_sin_refiner.free();
    if (self.rope_cos_context.size_bytes > 0) self.rope_cos_context.free();
    if (self.rope_sin_context.size_bytes > 0) self.rope_sin_context.free();

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
    self.kernels.flash_attention.free(self.ctx.device);
    self.kernels.scale_modulate.free(self.ctx.device);
    self.kernels.gated_residual.free(self.ctx.device);
    self.kernels.batch_layernorm.free(self.ctx.device);
    self.kernels.mrope.free(self.ctx.device);
    self.kernels.batch_rmsnorm.free(self.ctx.device);
    self.kernels.transpose_heads.free(self.ctx.device);
    self.kernels.batch_head_norm.free(self.ctx.device);
    self.kernels.batch_matmul.free(self.ctx.device);
    self.kernels.batch_matmul_q8.free(self.ctx.device);
    self.kernels.batch_matmul_q4_0.free(self.ctx.device);
    self.kernels.batch_matmul_q4k.free(self.ctx.device);
    self.kernels.batch_matmul_q5k.free(self.ctx.device);
    self.kernels.batch_matmul_q6k.free(self.ctx.device);
    self.kernels.shared.free(self.ctx.device);
}
