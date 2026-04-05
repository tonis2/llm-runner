module zimage;

import vk;
import llm;
import llm_text;
import std::io;
import std::math;
import std::core::mem;
import std::time::clock;

// Embedded Z-Image SPIR-V shaders
const char[*] ZIMAGE_SPV = $embed("./shaders/zimage.spv");
const char[*] MROPE_SPV = $embed("./shaders/mrope.spv");

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
    float scale;
    float clip_thresh;
}

struct ClipParams {
    uint n_elements;
    float threshold;
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


// --- DiT Layer Weights ---

struct DiTLayerWeights {
    // Fused QKV: [3*dim, dim]
    llm::Tensor wqkv;
    // Q/K norms: [head_dim]
    llm::Tensor q_norm;
    llm::Tensor k_norm;
    // Output projection: [dim, dim]
    llm::Tensor wo;
    // FFN SwiGLU: gate [ffn_dim, dim], up [ffn_dim, dim], down [dim, ffn_dim]
    llm::Tensor ffn_gate;
    llm::Tensor ffn_up;
    llm::Tensor ffn_down;
    // Norms (Lumina2: norm1=pre-norm, norm2=post-norm)
    llm::Tensor attn_norm1;   // pre-attention RMSNorm
    llm::Tensor attn_norm2;   // post-attention RMSNorm
    llm::Tensor ffn_norm1;    // pre-FFN RMSNorm
    llm::Tensor ffn_norm2;    // post-FFN RMSNorm
    // AdaLN: projects t_emb to modulation params [4*dim]
    llm::Tensor adaln_linear;
    llm::Tensor adaln_bias;
}

// CPU-cached layer weights for streaming/offloading mode.
// Mirrors DiTLayerWeights but holds CPU data + original GGML dtype.
struct DiTLayerCPUWeights {
    llm::CpuTensorData wqkv;
    llm::GGMLType wqkv_dtype;
    llm::CpuTensorData q_norm;
    llm::CpuTensorData k_norm;
    llm::CpuTensorData wo;
    llm::GGMLType wo_dtype;
    llm::CpuTensorData ffn_gate;
    llm::GGMLType ffn_gate_dtype;
    llm::CpuTensorData ffn_up;
    llm::GGMLType ffn_up_dtype;
    llm::CpuTensorData ffn_down;
    llm::GGMLType ffn_down_dtype;
    llm::CpuTensorData attn_norm1;
    llm::CpuTensorData attn_norm2;
    llm::CpuTensorData ffn_norm1;
    llm::CpuTensorData ffn_norm2;
    llm::CpuTensorData adaln_linear;
    llm::CpuTensorData adaln_bias;
}

struct DiTWeights {
    DiTLayerWeights[DIT_NUM_LAYERS] layers;

    // Noise refiner (2 layers, timestep-conditioned)
    DiTLayerWeights[DIT_NUM_REFINER_LAYERS] noise_refiner;
    // Context refiner (2 layers, plain transformer - no adaLN)
    DiTLayerWeights[DIT_NUM_REFINER_LAYERS] context_refiner;

    // Actual layer counts (may differ from constants if GGUF has fewer layers)
    uint actual_num_layers;
    uint actual_num_refiner_layers;

    // Streaming/offload mode: CPU-cached layer weights
    bool streaming;
    DiTLayerCPUWeights[DIT_NUM_LAYERS]* cpu_layers;
    DiTLayerCPUWeights[DIT_NUM_REFINER_LAYERS]* cpu_noise_refiner;
    DiTLayerCPUWeights[DIT_NUM_REFINER_LAYERS]* cpu_context_refiner;

    // cap_embedder: Sequential(RMSNorm(text_dim), Linear(text_dim, dim))
    llm::Tensor cap_emb_norm;    // RMSNorm weight [text_dim]
    llm::Tensor cap_emb_weight;  // Linear weight [text_dim, dim]
    llm::Tensor cap_emb_bias;    // Linear bias [dim]

    // t_embedder MLP: sinusoidal [256] -> Linear(256,1024) -> SiLU -> Linear(1024,256) -> [256]
    llm::Tensor t_emb_mlp0_weight;
    llm::Tensor t_emb_mlp0_bias;
    llm::Tensor t_emb_mlp2_weight;
    llm::Tensor t_emb_mlp2_bias;

    // x_embedder: [dim, patch_dim] + bias
    llm::Tensor x_emb_weight;
    llm::Tensor x_emb_bias;

    // Final layer
    llm::Tensor final_norm;          // RMSNorm weight [dim]
    llm::Tensor final_adaln_linear;
    llm::Tensor final_adaln_bias;
    llm::Tensor final_linear;    // [patch_dim, dim]
    llm::Tensor final_bias;

    // Pad tokens: [dim] (learnable padding for text)
    llm::Tensor x_pad_token;
    llm::Tensor cap_pad_token;
}

// --- DiT Activations ---

struct DiTActivations {
    llm::Tensor buf_a;       // General purpose [max_seq * dim]
    llm::Tensor buf_b;       // General purpose [max_seq * dim]
    llm::Tensor buf_c;       // General purpose [max_seq * dim]
    llm::Tensor patches;     // [n_patches, patch_dim]
    llm::Tensor hidden;      // [n_patches + text_len, dim]
    llm::Tensor norm_out;    // [dim] per-position scratch
    llm::Tensor q;           // [n_heads, seq_len, head_dim]
    llm::Tensor k;           // [n_heads, seq_len, head_dim]
    llm::Tensor v;           // [n_heads, seq_len, head_dim]
    llm::Tensor ffn_gate_out; // [seq_len, ffn_dim]
    llm::Tensor ffn_up_out;   // [seq_len, ffn_dim]
    llm::Tensor ffn_down_out; // [seq_len, dim]
    llm::Tensor t_emb;       // [t_emb_dim]
    llm::Tensor t_emb_mlp;   // [DIT_T_MLP_DIM] intermediate / [DIT_T_EMB_DIM] final
    llm::Tensor adaln_params; // [4 * dim]
    llm::Tensor adaln_scale;  // [dim] (split from adaln_params)
    llm::Tensor adaln_shift;  // [dim]
    llm::Tensor velocity;     // [n_patches, patch_dim]
    llm::Tensor latent;       // [channels, h, w]
}

// --- DiT Kernels ---

struct DiTKernels {
    llm::ComputeKernel patchify;
    llm::ComputeKernel unpatchify;
    llm::ComputeKernel adaln_modulate;
    llm::ComputeKernel flow_euler_step;
    llm::ComputeKernel linear_proj;
    llm::ComputeKernel linear_proj_debug;  // Simplified version with debug output
    llm::ComputeKernel dit_timestep_embed;
    llm::ComputeKernel broadcast_add_dim;
    llm::ComputeKernel scale_buffer;
    llm::ComputeKernel flash_attention;
    llm::ComputeKernel batch_rmsnorm;
    llm::ComputeKernel transpose_heads;
    llm::ComputeKernel batch_head_norm;
    llm::ComputeKernel scale_modulate;
    llm::ComputeKernel gated_residual;
    llm::ComputeKernel clip_values;
    llm::ComputeKernel batch_layernorm;
    llm::ComputeKernel mrope;
    llm::ComputeKernel batch_matmul;
    llm::ComputeKernel batch_matmul_simple;  // Simple 1D dispatch version (no tiling bugs)
    llm::ComputeKernel batch_matmul_q8;
    llm::ComputeKernel batch_matmul_q4_0;
    llm::ComputeKernel batch_matmul_q4k;
    llm::ComputeKernel batch_matmul_q5k;
    llm::ComputeKernel batch_matmul_q6k;
    llm::SharedKernels shared;
}

// --- DiT Model ---

struct DiTModel {
    DiTWeights weights;
    DiTActivations acts;
    DiTKernels kernels;
    llm::DeviceContext* ctx;
    uint n_patches;     // Number of real image patches
    uint text_len;      // Real text sequence length
    uint padded_patches; // n_patches padded to multiple of 32
    uint padded_text;    // text_len padded to multiple of 32
    uint latent_h;      // Latent height
    uint latent_w;      // Latent width
    // mRoPE cos/sin tables (precomputed once per config)
    llm::Tensor rope_cos_main;    // [n_patches + text_len, head_dim/2] for main layers
    llm::Tensor rope_sin_main;    // [n_patches + text_len, head_dim/2]
    llm::Tensor rope_cos_refiner; // [n_patches, head_dim/2] for noise refiner
    llm::Tensor rope_sin_refiner; // [n_patches, head_dim/2]
    llm::Tensor rope_cos_context; // [text_len, head_dim/2] for context refiner
    llm::Tensor rope_sin_context; // [text_len, head_dim/2]
}

// --- Kernel Creation ---

fn DiTKernels? create_dit_kernels(llm::DeviceContext* ctx) {
    io::printfn("Creating DiT compute kernels...");

    char[] zi_spv = &ZIMAGE_SPV;
    vk::ShaderModule zi_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(zi_spv.len)
        .setCode((uint*)&zi_spv[0])
        .build(ctx.device)!!;

    // Shared kernels from LLM shader (rmsnorm, matmul variants, silu, etc.)
    char[] llm_spv = &llm_text::LLM_SPV;
    vk::ShaderModule llm_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(llm_spv.len)
        .setCode((uint*)&llm_spv[0])
        .build(ctx.device)!!;

    io::printfn("  Creating zimage kernels...");
    llm::ComputeKernel k_patchify = llm::create_kernel(ctx, zi_shader, 2, PatchifyPC.sizeof, "patchify")!!;
    llm::ComputeKernel k_unpatchify = llm::create_kernel(ctx, zi_shader, 2, UnpatchifyPC.sizeof, "unpatchify")!!;
    llm::ComputeKernel k_adaln = llm::create_kernel(ctx, zi_shader, 4, AdalnModPC.sizeof, "adaln_modulate")!!;
    llm::ComputeKernel k_euler = llm::create_kernel(ctx, zi_shader, 2, FlowEulerPC.sizeof, "flow_euler_step")!!;
    llm::ComputeKernel k_linproj = llm::create_kernel(ctx, zi_shader, 4, LinearProjPC.sizeof, "linear_proj")!!;
    llm::ComputeKernel k_linproj_debug = llm::create_kernel(ctx, zi_shader, 6, LinearProjPC.sizeof, "linear_proj_debug")!!;
    llm::ComputeKernel k_timestep = llm::create_kernel(ctx, zi_shader, 1, DiTTimestepPC.sizeof, "dit_timestep_embed")!!;
    llm::ComputeKernel k_bcastadd = llm::create_kernel(ctx, zi_shader, 2, BroadcastAddDimPC.sizeof, "broadcast_add_dim")!!;
    llm::ComputeKernel k_scalebuf = llm::create_kernel(ctx, zi_shader, 1, ScalePC.sizeof, "scale_buffer")!!;
    llm::ComputeKernel k_flashattn = llm::create_kernel(ctx, zi_shader, 4, FlashAttentionPC.sizeof, "flash_attention")!!;
    llm::ComputeKernel k_scalemod = llm::create_kernel(ctx, zi_shader, 3, ScaleModPC.sizeof, "scale_modulate")!!;
    llm::ComputeKernel k_gatedres = llm::create_kernel(ctx, zi_shader, 3, GatedResidualPC.sizeof, "gated_residual")!!;
    llm::ComputeKernel k_clip = llm::create_kernel(ctx, zi_shader, 1, ClipParams.sizeof, "clip_values")!!;
    io::printfn("  Creating mrope kernel...");
    char[] mrope_spv = &MROPE_SPV;
    vk::ShaderModule mrope_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(mrope_spv.len)
        .setCode((uint*)&mrope_spv[0])
        .build(ctx.device)!!;
    llm::ComputeKernel k_mrope = llm::create_kernel(ctx, mrope_shader, 3, MRoPEPC.sizeof, "mrope")!!;
    mrope_shader.free(ctx.device);
    io::printfn("  mrope kernel created.");
    DiTKernels kernels = {
        .patchify         = k_patchify,
        .unpatchify       = k_unpatchify,
        .adaln_modulate   = k_adaln,
        .flow_euler_step  = k_euler,
        .linear_proj      = k_linproj,
        .linear_proj_debug = k_linproj_debug,
        .dit_timestep_embed = k_timestep,
        .broadcast_add_dim = k_bcastadd,
        .scale_buffer     = k_scalebuf,
        .flash_attention   = k_flashattn,
        .scale_modulate   = k_scalemod,
        .gated_residual   = k_gatedres,
        .clip_values      = k_clip,
        .batch_layernorm  = llm::create_kernel(ctx, zi_shader, 2, BatchLayerNormPC.sizeof, "batch_layernorm")!!,
        .mrope            = k_mrope,
        .batch_rmsnorm    = llm::create_kernel(ctx, zi_shader, 3, BatchRMSNormPC.sizeof, "batch_rmsnorm")!!,
        .transpose_heads  = llm::create_kernel(ctx, zi_shader, 2, TransposeHeadsPC.sizeof, "transpose_heads")!!,
        .batch_head_norm  = llm::create_kernel(ctx, zi_shader, 2, BatchHeadNormPC.sizeof, "batch_head_norm")!!,
        .batch_matmul     = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul")!!,
        .batch_matmul_simple = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_simple")!!,
        .batch_matmul_q8  = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q8")!!,
        .batch_matmul_q4_0 = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q4_0")!!,
        .batch_matmul_q4k = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q4k")!!,
        .batch_matmul_q5k = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q5k")!!,
        .batch_matmul_q6k = llm::create_kernel(ctx, llm_shader, 3, BatchMatMulPC.sizeof, "batch_matmul_q6k")!!,
        .shared           = llm::create_shared_kernels(ctx, llm_shader)!!,
    };

    zi_shader.free(ctx.device);
    llm_shader.free(ctx.device);
    return kernels;
}

// --- Activation Allocation ---

fn DiTActivations? allocate_dit_activations(llm::DeviceContext* ctx, uint max_seq_len) {
    io::printfn("Allocating DiT activation buffers (max_seq=%d)...", max_seq_len);

    ulong max_dim_total = (ulong)max_seq_len * DIT_DIM;
    ulong max_ffn_total = (ulong)max_seq_len * DIT_FFN_DIM;
    ulong latent_total = (ulong)DIT_LATENT_CHANNELS * 64 * 64;  // max 512x512 image

    return {
        .buf_a = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .buf_b = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .buf_c = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .patches = llm::create_f32_tensor(ctx, { (ulong)1024 * DIT_PATCH_DIM, 0, 0, 0 }, 1)!!,
        .hidden = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .norm_out = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .q = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .k = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .v = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .ffn_gate_out = llm::create_f32_tensor(ctx, { max_ffn_total, 0, 0, 0 }, 1)!!,
        .ffn_up_out = llm::create_f32_tensor(ctx, { max_ffn_total, 0, 0, 0 }, 1)!!,
        .ffn_down_out = llm::create_f32_tensor(ctx, { max_dim_total, 0, 0, 0 }, 1)!!,
        .t_emb = llm::create_f32_tensor(ctx, { DIT_T_EMB_DIM, 0, 0, 0 }, 1)!!,
        .t_emb_mlp = llm::create_f32_tensor(ctx, { DIT_T_MLP_DIM, 0, 0, 0 }, 1)!!,
        .adaln_params = llm::create_f32_tensor(ctx, { (ulong)4 * DIT_DIM, 0, 0, 0 }, 1)!!,
        .adaln_scale = llm::create_f32_tensor(ctx, { DIT_DIM, 0, 0, 0 }, 1)!!,
        .adaln_shift = llm::create_f32_tensor(ctx, { DIT_DIM, 0, 0, 0 }, 1)!!,
        .velocity = llm::create_f32_tensor(ctx, { (ulong)1024 * DIT_PATCH_DIM, 0, 0, 0 }, 1)!!,
        .latent = llm::create_f32_tensor(ctx, { latent_total, 0, 0, 0 }, 1)!!,
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
    llm::DeviceContext* ctx,
    llm::Tensor* cos_out, llm::Tensor* sin_out,
    uint n_patches, uint patches_w, uint patches_h,
    uint text_len, uint mode
) {
    uint padded_text = pad_to_multiple(text_len, SEQ_MULTI_OF);
    uint padded_patches = pad_to_multiple(n_patches, SEQ_MULTI_OF);
    uint n_text_pad = padded_text - text_len;
    uint n_img_pad = padded_patches - n_patches;

    uint seq_len;
    switch (mode) {
        case MROPE_MODE_FULL: seq_len = padded_patches + padded_text; break;
        case MROPE_MODE_IMAGE: seq_len = padded_patches; break;
        case MROPE_MODE_TEXT: seq_len = padded_text; break;
        default: seq_len = padded_patches + padded_text; break;
    }
    uint half_dim = DIT_HEAD_DIM / 2;  // 64
    usz table_size = (usz)seq_len * half_dim;

    float* cos_data = (float*)mem::calloc(table_size * 4);
    float* sin_data = (float*)mem::calloc(table_size * 4);

    // Image text-axis position = text_len (matches official Lumina2 position_ids)
    // Official: position_ids[:, cap_len:cap_len+img_len, 0] = cap_len
    uint image_index = text_len;

    // For each token, compute rotation angles for all 3 axes
    for (uint t = 0; t < seq_len; t++) {
        uint pos_text;   // axis 0
        uint pos_row;    // axis 1
        uint pos_col;    // axis 2

        if (mode == MROPE_MODE_TEXT) {
            // All tokens are text (real + pad), 1-indexed (Diffusers: pos_start=(1,0,0))
            pos_text = t + 1;
            pos_row = 0;
            pos_col = 0;
        } else if (mode == MROPE_MODE_IMAGE) {
            if (t < n_patches) {
                // Image text-axis: padded_text + 1 (Diffusers: pos_start=(cap_len+1, 0, 0))
                pos_text = padded_text + 1;
                pos_row = t / patches_w;
                pos_col = t % patches_w;
            } else {
                // Image padding: identity rotation (all zeros)
                pos_text = 0;
                pos_row = 0;
                pos_col = 0;
            }
        } else {
            // FULL mode: [text_real, text_pad, image_real, image_pad]
            // C3 ordering: caption tokens first, then image tokens
            // Matches Diffusers position IDs:
            //   Text: axis0 = 1..padded_text (1-indexed, padding gets sequential positions)
            //   Image: axis0 = padded_text + 1 (constant), axis1/2 = row/col
            if (t < text_len) {
                // Real text token, 1-indexed (Diffusers: pos_start=(1,0,0))
                pos_text = t + 1;
                pos_row = 0;
                pos_col = 0;
            } else if (t < padded_text) {
                // Text padding: sequential positions (Diffusers gives padding real positions)
                pos_text = t + 1;
                pos_row = 0;
                pos_col = 0;
            } else if (t < padded_text + n_patches) {
                // Real image token: text-axis = padded_text + 1
                uint img_idx = t - padded_text;
                pos_text = padded_text + 1;
                pos_row = img_idx / patches_w;
                pos_col = img_idx % patches_w;
            } else {
                // Image padding
                pos_text = 0;
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
    *cos_out = (llm::Tensor){
        .gpu_buffer = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: cos_data, data_size: table_size * 4,
        )!!,
        .size_bytes = table_size * 4,
        .shape = { seq_len, half_dim, 0, 0 },
        .n_dims = 2,
        .dtype = llm::GGML_F32,
    };
    *sin_out = (llm::Tensor){
        .gpu_buffer = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: sin_data, data_size: table_size * 4,
        )!!,
        .size_bytes = table_size * 4,
        .shape = { seq_len, half_dim, 0, 0 },
        .n_dims = 2,
        .dtype = llm::GGML_F32,
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

fn llm::Tensor load_dit_tensor(llm::DeviceContext* ctx, llm::GGUFFile* gf, String name) {
    if (try info = gf.find_tensor(name)) {
        // F16/BF16 weights get dequantized to F32
        if (info.type == llm::GGML_F16 || info.type == llm::GGML_BF16 || info.type == llm::GGML_F32) {
            return llm::upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
        }
        // Quantized weights stay quantized
        return llm::upload_weight(ctx, info, gf.tensor_data_base)!!;
    }
    io::printfn("Warning: DiT tensor not found: %s", name);
    return (llm::Tensor){};
}

fn llm::Tensor load_dit_tensor_f32(llm::DeviceContext* ctx, llm::GGUFFile* gf, String name) {
    llm::GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return llm::upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
}

fn DiTLayerWeights load_dit_layer_weights(llm::DeviceContext* ctx, llm::GGUFFile* gf, String prefix, uint layer, bool has_adaln) {
    char[128] buf;

    DiTLayerWeights lw;

    // Fused QKV: prefix.L.attention.qkv.weight
    String qkv_name = dit_layer_tensor_name(&buf, prefix, layer, "attention.qkv.weight");
    io::printfn("    [%s%d] loading qkv...", prefix, layer);
    if (try info = gf.find_tensor(qkv_name)) {
        io::printfn("    [DEBUG] QKV GGUF type: %d (llm::GGML_Q5_K=%d, llm::GGML_BF16=%d)", info.type, llm::GGML_Q5_K, llm::GGML_BF16);
        if (info.type == llm::GGML_F16 || info.type == llm::GGML_BF16 || info.type == llm::GGML_F32) {
            lw.wqkv = llm::upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
        } else {
            lw.wqkv = llm::upload_weight(ctx, info, gf.tensor_data_base)!!;
        }
        io::printfn("    [DEBUG] QKV loaded dtype: %d (llm::GGML_F32=%d, llm::GGML_BF16=%d)", lw.wqkv.dtype, llm::GGML_F32, llm::GGML_BF16);
    } else {
        io::printfn("    Warning: QKV not found: %s", qkv_name);
    }

    io::printfn("    [%s%d] loading norms + out...", prefix, layer);
    lw.q_norm = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "attention.q_norm.weight"));
    lw.k_norm = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "attention.k_norm.weight"));
    String wo_name = dit_layer_tensor_name(&buf, prefix, layer, "attention.out.weight");
    if (try wo_info = gf.find_tensor(wo_name)) {
        io::printfn("    [DEBUG] Wo GGUF type: %d (%s)", wo_info.type, wo_info.type == llm::GGML_Q5_K ? "Q5_K" : (wo_info.type == llm::GGML_Q4_K ? "Q4_K" : (wo_info.type == llm::GGML_F32 ? "F32" : "other")));
    }
    lw.wo = load_dit_tensor(ctx, gf, wo_name);
    io::printfn("    [DEBUG] Wo loaded dtype: %d", lw.wo.dtype);

    io::printfn("    [%s%d] loading ffn...", prefix, layer);
    lw.ffn_gate = load_dit_tensor(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w1.weight"));
    lw.ffn_up = load_dit_tensor(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w3.weight"));
    lw.ffn_down = load_dit_tensor(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w2.weight"));

    lw.attn_norm1 = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "attention_norm1.weight"));

    // Try loading second norm (for dual-norm layers)
    String norm2_name = dit_layer_tensor_name(&buf, prefix, layer, "attention_norm2.weight");
    if (try info = gf.find_tensor(norm2_name)) {
        lw.attn_norm2 = llm::upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
    }

    lw.ffn_norm1 = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "ffn_norm1.weight"));

    String ffn_norm2_name = dit_layer_tensor_name(&buf, prefix, layer, "ffn_norm2.weight");
    if (try info = gf.find_tensor(ffn_norm2_name)) {
        lw.ffn_norm2 = llm::upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
    }

    if (has_adaln) {
        io::printfn("    [%s%d] loading adaln...", prefix, layer);
        // Must load as F32 since linear_proj shader expects F32 weights
        lw.adaln_linear = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "adaLN_modulation.0.weight"));
        lw.adaln_bias = load_dit_tensor_f32(ctx, gf, dit_layer_tensor_name(&buf, prefix, layer, "adaLN_modulation.0.bias"));
    }

    return lw;
}

// --- Streaming helpers: load to CPU, upload to GPU, free GPU layer ---

// Helper: load a tensor to CPU. For F16/BF16/F32 → dequant to F32. For quantized → raw copy.
fn llm::CpuTensorData load_tensor_to_cpu(llm::GGUFFile* gf, String name, bool force_f32) {
    if (try info = gf.find_tensor(name)) {
        if (force_f32 || info.type == llm::GGML_F16 || info.type == llm::GGML_BF16 || info.type == llm::GGML_F32) {
            return llm::dequant_to_cpu(info, gf.tensor_data_base)!!;
        }
        return llm::copy_raw_to_cpu(info, gf.tensor_data_base)!!;
    }
    return (llm::CpuTensorData){};
}

// Helper: get GGML type for a tensor (returns F32 if not found or if it's F16/BF16)
fn llm::GGMLType get_tensor_dtype(llm::GGUFFile* gf, String name) {
    if (try info = gf.find_tensor(name)) {
        if (info.type == llm::GGML_F16 || info.type == llm::GGML_BF16 || info.type == llm::GGML_F32) {
            return llm::GGML_F32;
        }
        return info.type;
    }
    return llm::GGML_F32;
}

// Load a DiT layer's weights to CPU RAM (for streaming mode)
fn DiTLayerCPUWeights load_dit_layer_cpu(llm::GGUFFile* gf, String prefix, uint layer, bool has_adaln) {
    char[128] buf;
    DiTLayerCPUWeights cw;

    String qkv_name = dit_layer_tensor_name(&buf, prefix, layer, "attention.qkv.weight");
    cw.wqkv = load_tensor_to_cpu(gf, qkv_name, false);
    cw.wqkv_dtype = get_tensor_dtype(gf, qkv_name);

    cw.q_norm = load_tensor_to_cpu(gf, dit_layer_tensor_name(&buf, prefix, layer, "attention.q_norm.weight"), true);
    cw.k_norm = load_tensor_to_cpu(gf, dit_layer_tensor_name(&buf, prefix, layer, "attention.k_norm.weight"), true);

    String wo_name = dit_layer_tensor_name(&buf, prefix, layer, "attention.out.weight");
    cw.wo = load_tensor_to_cpu(gf, wo_name, false);
    cw.wo_dtype = get_tensor_dtype(gf, wo_name);

    String gate_name = dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w1.weight");
    cw.ffn_gate = load_tensor_to_cpu(gf, gate_name, false);
    cw.ffn_gate_dtype = get_tensor_dtype(gf, gate_name);

    String up_name = dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w3.weight");
    cw.ffn_up = load_tensor_to_cpu(gf, up_name, false);
    cw.ffn_up_dtype = get_tensor_dtype(gf, up_name);

    String down_name = dit_layer_tensor_name(&buf, prefix, layer, "feed_forward.w2.weight");
    cw.ffn_down = load_tensor_to_cpu(gf, down_name, false);
    cw.ffn_down_dtype = get_tensor_dtype(gf, down_name);

    cw.attn_norm1 = load_tensor_to_cpu(gf, dit_layer_tensor_name(&buf, prefix, layer, "attention_norm1.weight"), true);
    cw.attn_norm2 = load_tensor_to_cpu(gf, dit_layer_tensor_name(&buf, prefix, layer, "attention_norm2.weight"), true);
    cw.ffn_norm1 = load_tensor_to_cpu(gf, dit_layer_tensor_name(&buf, prefix, layer, "ffn_norm1.weight"), true);
    cw.ffn_norm2 = load_tensor_to_cpu(gf, dit_layer_tensor_name(&buf, prefix, layer, "ffn_norm2.weight"), true);

    if (has_adaln) {
        cw.adaln_linear = load_tensor_to_cpu(gf, dit_layer_tensor_name(&buf, prefix, layer, "adaLN_modulation.0.weight"), true);
        cw.adaln_bias = load_tensor_to_cpu(gf, dit_layer_tensor_name(&buf, prefix, layer, "adaLN_modulation.0.bias"), true);
    }

    return cw;
}

// Upload a CPU-cached layer to GPU, returning GPU tensor handles
fn DiTLayerWeights upload_dit_layer(llm::DeviceContext* ctx, DiTLayerCPUWeights* cw) {
    DiTLayerWeights lw;
    if (cw.wqkv.data != null) {
        if (cw.wqkv_dtype == llm::GGML_F32) {
            lw.wqkv = llm::upload_cpu_data(ctx, &cw.wqkv)!!;
        } else {
            lw.wqkv = llm::upload_cpu_data_raw(ctx, &cw.wqkv, cw.wqkv_dtype)!!;
        }
    }
    if (cw.q_norm.data != null) lw.q_norm = llm::upload_cpu_data(ctx, &cw.q_norm)!!;
    if (cw.k_norm.data != null) lw.k_norm = llm::upload_cpu_data(ctx, &cw.k_norm)!!;
    if (cw.wo.data != null) {
        if (cw.wo_dtype == llm::GGML_F32) {
            lw.wo = llm::upload_cpu_data(ctx, &cw.wo)!!;
        } else {
            lw.wo = llm::upload_cpu_data_raw(ctx, &cw.wo, cw.wo_dtype)!!;
        }
    }
    if (cw.ffn_gate.data != null) {
        if (cw.ffn_gate_dtype == llm::GGML_F32) {
            lw.ffn_gate = llm::upload_cpu_data(ctx, &cw.ffn_gate)!!;
        } else {
            lw.ffn_gate = llm::upload_cpu_data_raw(ctx, &cw.ffn_gate, cw.ffn_gate_dtype)!!;
        }
    }
    if (cw.ffn_up.data != null) {
        if (cw.ffn_up_dtype == llm::GGML_F32) {
            lw.ffn_up = llm::upload_cpu_data(ctx, &cw.ffn_up)!!;
        } else {
            lw.ffn_up = llm::upload_cpu_data_raw(ctx, &cw.ffn_up, cw.ffn_up_dtype)!!;
        }
    }
    if (cw.ffn_down.data != null) {
        if (cw.ffn_down_dtype == llm::GGML_F32) {
            lw.ffn_down = llm::upload_cpu_data(ctx, &cw.ffn_down)!!;
        } else {
            lw.ffn_down = llm::upload_cpu_data_raw(ctx, &cw.ffn_down, cw.ffn_down_dtype)!!;
        }
    }
    if (cw.attn_norm1.data != null) lw.attn_norm1 = llm::upload_cpu_data(ctx, &cw.attn_norm1)!!;
    if (cw.attn_norm2.data != null) lw.attn_norm2 = llm::upload_cpu_data(ctx, &cw.attn_norm2)!!;
    if (cw.ffn_norm1.data != null) lw.ffn_norm1 = llm::upload_cpu_data(ctx, &cw.ffn_norm1)!!;
    if (cw.ffn_norm2.data != null) lw.ffn_norm2 = llm::upload_cpu_data(ctx, &cw.ffn_norm2)!!;
    if (cw.adaln_linear.data != null) lw.adaln_linear = llm::upload_cpu_data(ctx, &cw.adaln_linear)!!;
    if (cw.adaln_bias.data != null) lw.adaln_bias = llm::upload_cpu_data(ctx, &cw.adaln_bias)!!;
    return lw;
}

fn void DiTLayerCPUWeights.free(&self) {
    self.wqkv.free();
    self.q_norm.free();
    self.k_norm.free();
    self.wo.free();
    self.ffn_gate.free();
    self.ffn_up.free();
    self.ffn_down.free();
    self.attn_norm1.free();
    self.attn_norm2.free();
    self.ffn_norm1.free();
    self.ffn_norm2.free();
    self.adaln_linear.free();
    self.adaln_bias.free();
}

fn DiTWeights? load_dit_weights(llm::DeviceContext* ctx, llm::GGUFFile* gf, bool streaming) {
    io::printfn("\nLoading DiT weights%s...", streaming ? " (streaming/offload mode)" : "");

    DiTWeights w;
    w.streaming = streaming;

    // Read actual layer counts from GGUF metadata (for mini/debug models)
    uint num_layers = DIT_NUM_LAYERS;
    uint num_refiner = DIT_NUM_REFINER_LAYERS;
    if (try n = gf.get_u32("zimage.num_layers")) {
        num_layers = n;
        io::printfn("  [MINI] Using %d main layers (from GGUF metadata)", num_layers);
    }
    if (try n = gf.get_u32("zimage.num_refiner_layers")) {
        num_refiner = n;
        io::printfn("  [MINI] Using %d refiner layers (from GGUF metadata)", num_refiner);
    }
    w.actual_num_layers = num_layers;
    w.actual_num_refiner_layers = num_refiner;

    if (streaming) {
        // Streaming mode: load layer weights to CPU RAM
        w.cpu_layers = mem::alloc(DiTLayerCPUWeights[DIT_NUM_LAYERS]);
        w.cpu_noise_refiner = mem::alloc(DiTLayerCPUWeights[DIT_NUM_REFINER_LAYERS]);
        w.cpu_context_refiner = mem::alloc(DiTLayerCPUWeights[DIT_NUM_REFINER_LAYERS]);

        for (uint l = 0; l < num_layers; l++) {
            io::printf("  DiT layer %d / %d (CPU)...\r", l + 1, num_layers);
            (*w.cpu_layers)[l] = load_dit_layer_cpu(gf, "layers.", l, true);
            if ((l + 1) % 5 == 0 || l == num_layers - 1) {
                io::printfn("  DiT layer %d / %d cached to CPU", l + 1, num_layers);
            }
        }

        io::printfn("  Loading noise refiner (CPU)...");
        for (uint l = 0; l < num_refiner; l++) {
            (*w.cpu_noise_refiner)[l] = load_dit_layer_cpu(gf, "noise_refiner.", l, true);
        }

        io::printfn("  Loading context refiner (CPU)...");
        for (uint l = 0; l < num_refiner; l++) {
            (*w.cpu_context_refiner)[l] = load_dit_layer_cpu(gf, "context_refiner.", l, false);
        }
    } else {
        // Normal mode: load directly to GPU
        for (uint l = 0; l < num_layers; l++) {
            io::printf("  DiT layer %d / %d...\r", l + 1, num_layers);
            w.layers[l] = load_dit_layer_weights(ctx, gf, "layers.", l, true);
            if ((l + 1) % 5 == 0 || l == num_layers - 1) {
                io::printfn("  DiT layer %d / %d loaded", l + 1, num_layers);
            }
        }

        io::printfn("  Loading noise refiner...");
        for (uint l = 0; l < num_refiner; l++) {
            w.noise_refiner[l] = load_dit_layer_weights(ctx, gf, "noise_refiner.", l, true);
        }

        io::printfn("  Loading context refiner...");
        for (uint l = 0; l < num_refiner; l++) {
            w.context_refiner[l] = load_dit_layer_weights(ctx, gf, "context_refiner.", l, false);
        }
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

    // Load x_embedder weights as F32 (using simple shader to avoid tiling bugs)
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
        w.x_pad_token = llm::upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
    }
    if (try info = gf.find_tensor("cap_pad_token")) {
        w.cap_pad_token = llm::upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
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
    llm::Tensor* qk,
    llm::Tensor* norm_weight,
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
llm::dispatch_kernel(cmd, &dk.batch_head_norm,
        { qk.gpu_buffer.buffer, norm_weight.gpu_buffer.buffer },
        { qk.size_bytes, norm_weight.size_bytes },
        &pc, seq_len * n_heads);
llm::compute_barrier(cmd);
}

// Dispatch a batch matmul: weight[out_dim, in_dim] × input[seq_len, in_dim] → output[seq_len, out_dim]
// row_offset: skip rows in weight (for fused QKV split)
fn void dispatch_batch_matmul(
    CommandBuffer cmd,
    DiTKernels* dk,
    llm::Tensor* weight,
    llm::Tensor* input,
    llm::Tensor* output,
    uint out_dim,
    uint in_dim,
    uint seq_len,
    uint row_offset
) {
    BatchMatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim, .seq_len = seq_len, .row_offset = row_offset };
    llm::ComputeKernel* kernel;
    bool tiled_2d = false;
    if (weight.dtype == llm::GGML_Q4_K) {
        kernel = &dk.batch_matmul_q4k;
        tiled_2d = true;
    } else if (weight.dtype == llm::GGML_Q5_K) {
        kernel = &dk.batch_matmul_q5k;
        tiled_2d = true;
    } else if (weight.dtype == llm::GGML_Q6_K) {
        kernel = &dk.batch_matmul_q6k;
        tiled_2d = true;
    } else if (weight.dtype == llm::GGML_Q8_0) {
        kernel = &dk.batch_matmul_q8;
    } else if (weight.dtype == llm::GGML_Q4_0) {
        kernel = &dk.batch_matmul_q4_0;
    } else {
        // Use batch_matmul_simple for F32 weights (avoids tiled indexing bugs)
        // 2D dispatch: (seq_len, out_dim) workgroups
        kernel = &dk.batch_matmul_simple;
llm::dispatch_kernel(cmd, kernel,
            { weight.gpu_buffer.buffer, input.gpu_buffer.buffer, output.gpu_buffer.buffer },
            { weight.size_bytes, input.size_bytes, output.size_bytes },
            &pc, seq_len, out_dim);
        return;
    }
    // 2D tiled GEMM: ceil(seq_len/64) * ceil(out_dim/64) workgroups
    // Legacy (Q8_0, Q4_0): seq_len * out_dim workgroups
    uint groups = tiled_2d ? ((seq_len + 63) / 64) * ((out_dim + 63) / 64) : seq_len * out_dim;
llm::dispatch_kernel(cmd, kernel,
        { weight.gpu_buffer.buffer, input.gpu_buffer.buffer, output.gpu_buffer.buffer },
        { weight.size_bytes, input.size_bytes, output.size_bytes },
        &pc, groups);
}

// Run a single DiT transformer layer (Lumina2 architecture)
// Lumina2 adaLN: 4 params = [scale_msa, gate_msa, scale_mlp, gate_mlp]
// Pre-norm + scale modulation (no shift), post-norm + tanh-gated residual
fn void? dit_transformer_layer(
    DiTModel* model,
    CommandBuffer cmd,
    DiTLayerWeights* lw,
    llm::Tensor* hidden,        // [seq_len, dim] - modified in place
    uint seq_len,
    bool has_adaln,
    llm::Tensor* rope_cos,      // [seq_len, head_dim/2] precomputed cos table
    llm::Tensor* rope_sin       // [seq_len, head_dim/2] precomputed sin table
) {
    llm::DeviceContext* ctx = model.ctx;
    llm::SharedKernels* sk = &model.kernels.shared;
    DiTKernels* dk = &model.kernels;
    DiTActivations* a = &model.acts;
    uint dim = DIT_DIM;

    // === Phase A: AdaLN params + Pre-attention norm + scale ===
    if (has_adaln && lw.adaln_linear.size_bytes > 0) {
        // Z-Image: per-layer adaLN does NOT apply SiLU (Sequential(Linear) only, weight at .0)
        // Only FinalLayer applies SiLU (Sequential(SiLU, Linear), weight at .1)
        // Project t_emb -> 4 modulation params: [scale_msa, gate_msa, scale_mlp, gate_mlp]
        LinearProjPC adaln_proj_pc = { .out_dim = 4 * dim, .in_dim = DIT_T_EMB_DIM, .seq_len = 1 };
llm::dispatch_kernel(cmd, &dk.linear_proj,
            { lw.adaln_linear.gpu_buffer.buffer, lw.adaln_bias.gpu_buffer.buffer,
              a.t_emb.gpu_buffer.buffer, a.adaln_params.gpu_buffer.buffer },
            { lw.adaln_linear.size_bytes, lw.adaln_bias.size_bytes,
              a.t_emb.size_bytes, a.adaln_params.size_bytes },
            &adaln_proj_pc, 4 * dim);
llm::compute_barrier(cmd);

        // Extract scale_msa (chunk 0) and gate_msa (chunk 1)
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)dim * 4 }});
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
llm::compute_barrier(cmd);
    }

    // Pre-attention RMSNorm: all tokens get same norm1
    BatchRMSNormPC rms_pc = { .dim = dim, .eps = 1e-6f, .n_rows = seq_len, .row_offset = 0 };
llm::dispatch_kernel(cmd, &dk.batch_rmsnorm,
        { hidden.gpu_buffer.buffer, lw.attn_norm1.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { hidden.size_bytes, lw.attn_norm1.size_bytes, a.buf_a.size_bytes },
        &rms_pc, seq_len);
llm::compute_barrier(cmd);

    // Scale modulation: buf_a = buf_a * (1 + scale_msa) — applied to ALL tokens
    if (has_adaln) {
        ScaleModPC smod_pc = { .n_elements = seq_len * dim, .dim = dim };
llm::dispatch_kernel(cmd, &dk.scale_modulate,
            { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { a.buf_a.size_bytes, a.adaln_scale.size_bytes, a.buf_a.size_bytes },
            &smod_pc, llm::ceil_div(seq_len * dim, 256));
llm::compute_barrier(cmd);
    }

    // === Phase B: QKV batch matmul (3 dispatches) ===
    // wqkv is [3*dim, dim] fused — split via row_offset: Q=0, K=dim, V=2*dim
    dispatch_batch_matmul(cmd, dk, &lw.wqkv, &a.buf_a, &a.q, dim, dim, seq_len, 0);
    dispatch_batch_matmul(cmd, dk, &lw.wqkv, &a.buf_a, &a.k, dim, dim, seq_len, dim);
    dispatch_batch_matmul(cmd, dk, &lw.wqkv, &a.buf_a, &a.v, dim, dim, seq_len, 2 * dim);
llm::compute_barrier(cmd);

    // === Phase C: Head norm + Forward Transpose + Flash Attention ===
    dit_apply_head_norm(model, cmd, &a.q, &lw.q_norm, seq_len, DIT_HEADS, DIT_HEAD_DIM)!!;
    dit_apply_head_norm(model, cmd, &a.k, &lw.k_norm, seq_len, DIT_HEADS, DIT_HEAD_DIM)!!;

    // Forward transpose Q/K/V: [seq_len, dim] -> [n_heads, seq_len, head_dim]
    uint total_elems = seq_len * dim;
    uint transpose_groups = llm::ceil_div(total_elems, 256);
    TransposeHeadsPC th_fwd_pc = {
        .seq_len = seq_len, .n_heads = DIT_HEADS, .head_dim = DIT_HEAD_DIM, .direction = 0,
    };

llm::dispatch_kernel(cmd, &dk.transpose_heads,
        { a.q.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.q.size_bytes, a.buf_b.size_bytes },
        &th_fwd_pc, transpose_groups);
llm::compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.q.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)total_elems * 4 }});
llm::compute_barrier(cmd);

llm::dispatch_kernel(cmd, &dk.transpose_heads,
        { a.k.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.k.size_bytes, a.buf_b.size_bytes },
        &th_fwd_pc, transpose_groups);
llm::compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.k.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)total_elems * 4 }});
llm::compute_barrier(cmd);

llm::dispatch_kernel(cmd, &dk.transpose_heads,
        { a.v.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.v.size_bytes, a.buf_b.size_bytes },
        &th_fwd_pc, transpose_groups);
llm::compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.v.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)total_elems * 4 }});
llm::compute_barrier(cmd);

    // === Apply mRoPE to Q and K (after transpose, before attention) ===
    if (rope_cos != null && rope_cos.size_bytes > 0) {
        MRoPEPC mrope_pc = { .seq_len = seq_len, .n_heads = DIT_HEADS };
        // 2D dispatch: x = pairs (64 threads per workgroup), y = head*seq_len
        uint groups_y = DIT_HEADS * seq_len;

llm::dispatch_kernel(cmd, &dk.mrope,
            { a.q.gpu_buffer.buffer, rope_cos.gpu_buffer.buffer, rope_sin.gpu_buffer.buffer },
            { a.q.size_bytes, rope_cos.size_bytes, rope_sin.size_bytes },
            &mrope_pc, 1, groups_y);

llm::dispatch_kernel(cmd, &dk.mrope,
            { a.k.gpu_buffer.buffer, rope_cos.gpu_buffer.buffer, rope_sin.gpu_buffer.buffer },
            { a.k.size_bytes, rope_cos.size_bytes, rope_sin.size_bytes },
            &mrope_pc, 1, groups_y);
llm::compute_barrier(cmd);
    }

    // Flash attention: Q/K/V in [n_heads, seq_len, head_dim], output to buf_b in [seq_len, dim]
    // Standard 1/sqrt(head_dim) scaling, matching official diffusers
    float scale = 1.0f / math::sqrt((float)DIT_HEAD_DIM);
    FlashAttentionPC fa_pc = {
        .head_dim = DIT_HEAD_DIM,
        .n_heads = DIT_HEADS,
        .seq_len = seq_len,
        .scale = scale,
    };
llm::dispatch_kernel(cmd, &dk.flash_attention,
        { a.q.gpu_buffer.buffer, a.k.gpu_buffer.buffer, a.v.gpu_buffer.buffer,
          a.buf_b.gpu_buffer.buffer },
        { a.q.size_bytes, a.k.size_bytes, a.v.size_bytes,
          a.buf_b.size_bytes },
        &fa_pc, DIT_HEADS * seq_len);
llm::compute_barrier(cmd);

    // === Phase D: Output projection + Post-attention norm + Gated residual ===
    // Output projection: buf_b -> buf_a
    dispatch_batch_matmul(cmd, dk, &lw.wo, &a.buf_b, &a.buf_a, dim, dim, seq_len, 0);
llm::compute_barrier(cmd);

    // Post-attention norm: norm2(attn_output)
    if (lw.attn_norm2.size_bytes > 0) {
        BatchRMSNormPC post_attn_rms = { .dim = dim, .eps = 1e-6f, .n_rows = seq_len, .row_offset = 0 };
llm::dispatch_kernel(cmd, &dk.batch_rmsnorm,
            { a.buf_a.gpu_buffer.buffer, lw.attn_norm2.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
            { a.buf_a.size_bytes, lw.attn_norm2.size_bytes, a.buf_b.size_bytes },
            &post_attn_rms, seq_len);
llm::compute_barrier(cmd);

        // Gated residual: hidden += tanh(gate_msa) * norm2(attn_output)
        if (has_adaln) {
            GatedResidualPC gres_pc = { .n_elements = seq_len * dim, .dim = dim, .scale = 1.0f, .clip_thresh = 0.0f };
llm::dispatch_kernel(cmd, &dk.gated_residual,
                { hidden.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
                { hidden.size_bytes, a.adaln_shift.size_bytes, a.buf_b.size_bytes },
                &gres_pc, llm::ceil_div(seq_len * dim, 256));
        } else {
            // Context refiner: no gate, just residual += post_norm(output)
            llm::ResidualPC res_pc = { .n = seq_len * dim };
llm::dispatch_kernel(cmd, &sk.residual_add,
                { hidden.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
                { hidden.size_bytes, a.buf_b.size_bytes },
                &res_pc, llm::ceil_div(seq_len * dim, 256));
        }
llm::compute_barrier(cmd);
    } else {
        // No post-norm available: direct residual
        llm::ResidualPC res_pc = { .n = seq_len * dim };
llm::dispatch_kernel(cmd, &sk.residual_add,
            { hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { hidden.size_bytes, a.buf_a.size_bytes },
            &res_pc, llm::ceil_div(seq_len * dim, 256));
llm::compute_barrier(cmd);
    }

    // === Phase E: FFN pre-norm + scale ===
    if (has_adaln) {
        // Extract scale_mlp (chunk 2) and gate_mlp (chunk 3) from adaln_params
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)2 * dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
        vk::cmdCopyBuffer(cmd, a.adaln_params.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)3 * dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 }});
llm::compute_barrier(cmd);
    }

    // Pre-FFN RMSNorm: all tokens get same ffn_norm1
    BatchRMSNormPC ffn_rms_pc = { .dim = dim, .eps = 1e-6f, .n_rows = seq_len, .row_offset = 0 };
llm::dispatch_kernel(cmd, &dk.batch_rmsnorm,
        { hidden.gpu_buffer.buffer, lw.ffn_norm1.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { hidden.size_bytes, lw.ffn_norm1.size_bytes, a.buf_a.size_bytes },
        &ffn_rms_pc, seq_len);
llm::compute_barrier(cmd);

    // Scale modulation: buf_a = buf_a * (1 + scale_mlp) — applied to ALL tokens
    if (has_adaln) {
        ScaleModPC smod_pc = { .n_elements = seq_len * dim, .dim = dim };
llm::dispatch_kernel(cmd, &dk.scale_modulate,
            { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { a.buf_a.size_bytes, a.adaln_scale.size_bytes, a.buf_a.size_bytes },
            &smod_pc, llm::ceil_div(seq_len * dim, 256));
llm::compute_barrier(cmd);
    }

    // === Phase F: SwiGLU FFN batch matmul ===
    // Gate and Up projections (both read buf_a, write to different outputs)
    dispatch_batch_matmul(cmd, dk, &lw.ffn_gate, &a.buf_a, &a.ffn_gate_out, DIT_FFN_DIM, dim, seq_len, 0);
    dispatch_batch_matmul(cmd, dk, &lw.ffn_up, &a.buf_a, &a.ffn_up_out, DIT_FFN_DIM, dim, seq_len, 0);
llm::compute_barrier(cmd);

    // SiLU on gate
    llm::SiluPC silu_pc = { .n = seq_len * DIT_FFN_DIM };
llm::dispatch_kernel(cmd, &sk.silu,
        { a.ffn_gate_out.gpu_buffer.buffer },
        { a.ffn_gate_out.size_bytes },
        &silu_pc, llm::ceil_div(seq_len * DIT_FFN_DIM, 256));
llm::compute_barrier(cmd);

    // gate * up
    llm::ElemwisePC emul_pc = { .n = seq_len * DIT_FFN_DIM };
llm::dispatch_kernel(cmd, &sk.elemwise_mul,
        { a.ffn_gate_out.gpu_buffer.buffer, a.ffn_up_out.gpu_buffer.buffer },
        { a.ffn_gate_out.size_bytes, a.ffn_up_out.size_bytes },
        &emul_pc, llm::ceil_div(seq_len * DIT_FFN_DIM, 256));
llm::compute_barrier(cmd);

    // Down projection: [seq_len, ffn_dim] -> [seq_len, dim]
    dispatch_batch_matmul(cmd, dk, &lw.ffn_down, &a.ffn_gate_out, &a.ffn_down_out, dim, DIT_FFN_DIM, seq_len, 0);
llm::compute_barrier(cmd);

    // === Phase G: Post-FFN norm + Gated residual ===
    if (lw.ffn_norm2.size_bytes > 0) {
        BatchRMSNormPC post_ffn_rms = { .dim = dim, .eps = 1e-6f, .n_rows = seq_len, .row_offset = 0 };
llm::dispatch_kernel(cmd, &dk.batch_rmsnorm,
            { a.ffn_down_out.gpu_buffer.buffer, lw.ffn_norm2.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
            { a.ffn_down_out.size_bytes, lw.ffn_norm2.size_bytes, a.buf_a.size_bytes },
            &post_ffn_rms, seq_len);
llm::compute_barrier(cmd);

        // Gated residual: hidden += tanh(gate_mlp) * ffn_norm2(ffn_output)
        if (has_adaln) {
            GatedResidualPC gres_pc = { .n_elements = seq_len * dim, .dim = dim, .scale = 1.0f, .clip_thresh = 0.0f };
llm::dispatch_kernel(cmd, &dk.gated_residual,
                { hidden.gpu_buffer.buffer, a.adaln_shift.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
                { hidden.size_bytes, a.adaln_shift.size_bytes, a.buf_a.size_bytes },
                &gres_pc, llm::ceil_div(seq_len * dim, 256));
        } else {
            // Context refiner: no gate
            llm::ResidualPC res_pc = { .n = seq_len * dim };
llm::dispatch_kernel(cmd, &sk.residual_add,
                { hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
                { hidden.size_bytes, a.buf_a.size_bytes },
                &res_pc, llm::ceil_div(seq_len * dim, 256));
        }
llm::compute_barrier(cmd);
    } else {
        // No post-norm: direct residual
        llm::ResidualPC res_pc = { .n = seq_len * dim };
llm::dispatch_kernel(cmd, &sk.residual_add,
            { hidden.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
            { hidden.size_bytes, a.ffn_down_out.size_bytes },
            &res_pc, llm::ceil_div(seq_len * dim, 256));
llm::compute_barrier(cmd);
    }

}

// --- Main DiT Forward Pass ---
// Takes: latent [16, h, w], text_embeddings [text_len, 3840], timestep scalar
// Returns: velocity [16, h, w] in acts.latent

fn void? DiTModel.forward(&self, llm::Tensor* text_embeddings, float timestep) {
    llm::DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    DiTWeights* w = &self.weights;
    DiTActivations* a = &self.acts;
    DiTKernels* dk = &self.kernels;
    llm::SharedKernels* sk = &dk.shared;

    uint dim = DIT_DIM;
    uint n_patches = self.n_patches;
    uint text_len = self.text_len;
    uint padded_patches = self.padded_patches;
    uint padded_text = self.padded_text;
    uint n_img_pad = padded_patches - n_patches;

    Clock step_start = clock::now();
    Clock phase_start = step_start;

llm::begin_compute(cmd)!!;

    // 1. Patchify latent -> [n_patches, patch_dim]
    PatchifyPC patch_pc = {
        .channels = DIT_LATENT_CHANNELS,
        .height = self.latent_h,
        .width = self.latent_w,
        .patch_size = DIT_PATCH_SIZE,
    };
llm::dispatch_kernel(cmd, &dk.patchify,
        { a.latent.gpu_buffer.buffer, a.patches.gpu_buffer.buffer },
        { a.latent.size_bytes, a.patches.size_bytes },
        &patch_pc, llm::ceil_div(n_patches * DIT_PATCH_DIM, 256));
llm::compute_barrier(cmd);

    // 2. x_embedder: [n_patches, 64] -> [n_patches, 3840]
    // Use batch_matmul_simple (1D dispatch, no tiling bugs) + separate bias add
    BatchMatMulPC x_emb_pc = { .out_dim = dim, .in_dim = DIT_PATCH_DIM, .seq_len = n_patches, .row_offset = 0 };
    uint groups_x = n_patches;
    uint groups_y = dim;
llm::dispatch_kernel(cmd, &dk.batch_matmul_simple,
        { w.x_emb_weight.gpu_buffer.buffer, a.patches.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
        { w.x_emb_weight.size_bytes, a.patches.size_bytes, a.hidden.size_bytes },
        &x_emb_pc, groups_x, groups_y);
llm::compute_barrier(cmd);

    // Then add bias using broadcast_add_dim
llm::dispatch_kernel(cmd, &dk.broadcast_add_dim,
        { a.hidden.gpu_buffer.buffer, w.x_emb_bias.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
        { a.hidden.size_bytes, w.x_emb_bias.size_bytes, a.hidden.size_bytes },
        null, n_patches * dim);
llm::compute_barrier(cmd);

    // 2b. Pad image tokens with x_pad_token (append n_img_pad copies)
    if (n_img_pad > 0 && w.x_pad_token.size_bytes > 0) {
        for (uint p = 0; p < n_img_pad; p++) {
            vk::cmdCopyBuffer(cmd, w.x_pad_token.gpu_buffer.buffer,
                a.hidden.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0,
                    .dstOffset = (ulong)(n_patches + p) * dim * 4,
                    .size = (ulong)dim * 4 }});
        }
llm::compute_barrier(cmd);
    }

    // 3. Timestep embedding: Z-Image pipeline passes (1000-t)/1000 = 1-sigma to model.
    // Model computes: t_embedder(timestep * t_scale) = sincos((1-sigma) * 1000)
    // At sigma=1.0 (noisy): model_t = 0
    // At sigma=0.0 (clean): model_t = 1000
    float actual_timestep = (1.0f - timestep) * DIT_T_PERIOD;
    DiTTimestepPC t_pc = { .dim = DIT_T_EMB_DIM, .timestep = actual_timestep };
llm::dispatch_kernel(cmd, &dk.dit_timestep_embed,
        { a.t_emb.gpu_buffer.buffer },
        { a.t_emb.size_bytes },
        &t_pc, llm::ceil_div(DIT_T_EMB_DIM, 256));
llm::compute_barrier(cmd);

    // t_embedder MLP: t_emb [256] -> mlp_0 [1024] -> silu -> mlp_2 [256]
    LinearProjPC t_mlp0_pc = { .out_dim = DIT_T_MLP_DIM, .in_dim = DIT_T_EMB_DIM, .seq_len = 1 };
llm::dispatch_kernel(cmd, &dk.linear_proj,
        { w.t_emb_mlp0_weight.gpu_buffer.buffer, w.t_emb_mlp0_bias.gpu_buffer.buffer,
          a.t_emb.gpu_buffer.buffer, a.t_emb_mlp.gpu_buffer.buffer },
        { w.t_emb_mlp0_weight.size_bytes, w.t_emb_mlp0_bias.size_bytes,
          a.t_emb.size_bytes, a.t_emb_mlp.size_bytes },
        &t_mlp0_pc, DIT_T_MLP_DIM);
llm::compute_barrier(cmd);

    llm::SiluPC silu_pc = { .n = DIT_T_MLP_DIM };
llm::dispatch_kernel(cmd, &sk.silu,
        { a.t_emb_mlp.gpu_buffer.buffer },
        { a.t_emb_mlp.size_bytes },
        &silu_pc, llm::ceil_div(DIT_T_MLP_DIM, 256));
llm::compute_barrier(cmd);

    // mlp_2: [1024] -> [256]
    LinearProjPC t_mlp2_pc = { .out_dim = DIT_T_EMB_DIM, .in_dim = DIT_T_MLP_DIM, .seq_len = 1 };
llm::dispatch_kernel(cmd, &dk.linear_proj,
        { w.t_emb_mlp2_weight.gpu_buffer.buffer, w.t_emb_mlp2_bias.gpu_buffer.buffer,
          a.t_emb_mlp.gpu_buffer.buffer, a.t_emb.gpu_buffer.buffer },
        { w.t_emb_mlp2_weight.size_bytes, w.t_emb_mlp2_bias.size_bytes,
          a.t_emb_mlp.size_bytes, a.t_emb.size_bytes },
        &t_mlp2_pc, DIT_T_EMB_DIM);
llm::compute_barrier(cmd);

    // NOTE: No SiLU here — JointTransformerBlock's adaLN uses t_emb directly (weight at .0)
    // Only FinalLayer applies SiLU before its adaLN Linear (weight at .1 in Sequential(SiLU, Linear))

    // 4. Noise refiner (2 layers on padded image tokens, with adaLN)
    if (w.streaming) {
        for (uint l = 0; l < self.weights.actual_num_refiner_layers; l++) {
            DiTLayerWeights lw = upload_dit_layer(ctx, &(*w.cpu_noise_refiner)[l]);
            dit_transformer_layer(self, cmd, &lw, &a.hidden, padded_patches, true,
                &self.rope_cos_refiner, &self.rope_sin_refiner)!!;
            llm::submit_and_wait(ctx)!!;
            lw.free();
            llm::begin_compute(cmd)!!;
        }
    } else {
        for (uint l = 0; l < self.weights.actual_num_refiner_layers; l++) {
            dit_transformer_layer(self, cmd, &w.noise_refiner[l], &a.hidden, padded_patches, true,
                &self.rope_cos_refiner, &self.rope_sin_refiner)!!;
        }
    }

llm::submit_and_wait(ctx)!!;
    double setup_sec = phase_start.to_now().to_sec();
    phase_start = clock::now();
llm::begin_compute(cmd)!!;

    // 5. Concat [padded_text, padded_image] -> [padded_text + padded_patches, dim]
    // Official Lumina2 ordering: text tokens first, then image tokens
    // hidden currently has image tokens [padded_patches, dim] at offset 0 (from noise_refiner)
    // Need to rearrange to [text, image] order

    // Step 1: Save image tokens from hidden to buf_a (temporary)
    vk::cmdCopyBuffer(cmd, a.hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0,
                         .size = (ulong)padded_patches * dim * 4 }});
llm::compute_barrier(cmd);

    // Step 2: Copy text embeddings to the BEGINNING of hidden
    vk::cmdCopyBuffer(cmd, text_embeddings.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0,
                         .size = (ulong)padded_text * dim * 4 }});
llm::compute_barrier(cmd);

    // Step 3: Copy image tokens AFTER text in hidden
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)padded_text * dim * 4,
                         .size = (ulong)padded_patches * dim * 4 }});
llm::compute_barrier(cmd);

    uint total_seq = padded_patches + padded_text;

    // 6. Main DiT (30 layers with adaLN, post-norm, tanh gating)
    double[16] batch_times;
    uint batch_idx = 0;
    Clock batch_start = clock::now();

    uint actual_layers = self.weights.actual_num_layers;
    if (w.streaming) {
        // Streaming mode: upload each layer, run, free
        for (uint l = 0; l < actual_layers; l++) {
            DiTLayerWeights lw = upload_dit_layer(ctx, &(*w.cpu_layers)[l]);
            dit_transformer_layer(self, cmd, &lw, &a.hidden, total_seq, true,
                &self.rope_cos_main, &self.rope_sin_main)!!;
            llm::submit_and_wait(ctx)!!;
            lw.free();

            if ((l + 1) % 10 == 0 || l == actual_layers - 1) {
                batch_times[batch_idx] = batch_start.to_now().to_sec();
                batch_idx++;
                batch_start = clock::now();
            }
            llm::begin_compute(cmd)!!;
        }
    } else {
        for (uint l = 0; l < actual_layers; l++) {
            dit_transformer_layer(self, cmd, &w.layers[l], &a.hidden, total_seq, true,
                &self.rope_cos_main, &self.rope_sin_main)!!;

            // Flush GPU every 5 layers to prevent command buffer overflow / GPU timeout
            if ((l + 1) % 5 == 0 || l == actual_layers - 1) {
llm::submit_and_wait(ctx)!!;
                if ((l + 1) % 10 == 0 || l == actual_layers - 1) {
                    batch_times[batch_idx] = batch_start.to_now().to_sec();
                    batch_idx++;
                    batch_start = clock::now();
                }
llm::begin_compute(cmd)!!;
            }
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
llm::compute_barrier(cmd);
    llm::SiluPC final_silu_pc = { .n = DIT_T_EMB_DIM };
llm::dispatch_kernel(cmd, &sk.silu,
        { a.t_emb_mlp.gpu_buffer.buffer },
        { a.t_emb_mlp.size_bytes },
        &final_silu_pc, llm::ceil_div(DIT_T_EMB_DIM, 256));
llm::compute_barrier(cmd);

    // Final adaLN: SiLU(t_emb) @ weight + bias -> [dim] scale
    LinearProjPC final_adaln_pc = { .out_dim = dim, .in_dim = DIT_T_EMB_DIM, .seq_len = 1 };
llm::dispatch_kernel(cmd, &dk.linear_proj,
        { w.final_adaln_linear.gpu_buffer.buffer, w.final_adaln_bias.gpu_buffer.buffer,
          a.t_emb_mlp.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer },
        { w.final_adaln_linear.size_bytes, w.final_adaln_bias.size_bytes,
          a.t_emb_mlp.size_bytes, a.adaln_scale.size_bytes },
        &final_adaln_pc, dim);
llm::compute_barrier(cmd);

    // Copy only real image portion to buf_a (image tokens start after padded_text)
    // Official Lumina2: begin = cap_size, end = begin + n_patches
    vk::cmdCopyBuffer(cmd, a.hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = (ulong)padded_text * dim * 4, .dstOffset = 0, .size = (ulong)n_patches * dim * 4 }});
llm::compute_barrier(cmd);

    // Apply LayerNorm (no affine) to image portion — Lumina2 FinalLayer uses elementwise_affine=False
    if (w.final_norm.size_bytes > 0) {
        // Learnable RMSNorm (if weight tensor exists in GGUF)
        BatchRMSNormPC final_rms = { .dim = dim, .eps = 1e-6f, .n_rows = n_patches, .row_offset = 0 };
llm::dispatch_kernel(cmd, &dk.batch_rmsnorm,
            { a.buf_a.gpu_buffer.buffer, w.final_norm.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
            { a.buf_a.size_bytes, w.final_norm.size_bytes, a.buf_b.size_bytes },
            &final_rms, n_patches);
llm::compute_barrier(cmd);
        vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_patches * dim * 4 }});
llm::compute_barrier(cmd);
    } else {
        // No-affine LayerNorm (Lumina2 default: no learnable weight)
        BatchLayerNormPC final_ln = { .dim = dim, .eps = 1e-6f, .n_rows = n_patches };
llm::dispatch_kernel(cmd, &dk.batch_layernorm,
            { a.buf_a.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
            { a.buf_a.size_bytes, a.buf_b.size_bytes },
            &final_ln, n_patches);
llm::compute_barrier(cmd);
        vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n_patches * dim * 4 }});
llm::compute_barrier(cmd);
    }

    // Scale modulate: buf_a = buf_a * (1 + scale)
    ScaleModPC final_smod_pc = { .n_elements = n_patches * dim, .dim = dim };
llm::dispatch_kernel(cmd, &dk.scale_modulate,
        { a.buf_a.gpu_buffer.buffer, a.adaln_scale.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.adaln_scale.size_bytes, a.buf_a.size_bytes },
        &final_smod_pc, llm::ceil_div(n_patches * dim, 256));
llm::compute_barrier(cmd);

    // Final linear: batch matmul [n_patches, dim] -> [n_patches, patch_dim] + bias
    dispatch_batch_matmul(cmd, dk, &w.final_linear, &a.buf_a, &a.velocity, DIT_PATCH_DIM, dim, n_patches, 0);
llm::compute_barrier(cmd);

    // Add bias (broadcast [patch_dim] across all patches)
    BroadcastAddDimPC bias_pc = { .n_total = n_patches * DIT_PATCH_DIM, .dim = DIT_PATCH_DIM };
llm::dispatch_kernel(cmd, &dk.broadcast_add_dim,
        { a.velocity.gpu_buffer.buffer, w.final_bias.gpu_buffer.buffer },
        { a.velocity.size_bytes, w.final_bias.size_bytes },
        &bias_pc, llm::ceil_div(n_patches * DIT_PATCH_DIM, 256));
llm::compute_barrier(cmd);

    // 8. Unpatchify velocity -> [16, h, w]
    UnpatchifyPC unpatch_pc = {
        .channels = DIT_LATENT_CHANNELS,
        .height = self.latent_h,
        .width = self.latent_w,
        .patch_size = DIT_PATCH_SIZE,
    };
    uint latent_total = DIT_LATENT_CHANNELS * self.latent_h * self.latent_w;
llm::dispatch_kernel(cmd, &dk.unpatchify,
        { a.velocity.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.velocity.size_bytes, a.buf_a.size_bytes },
        &unpatch_pc, llm::ceil_div(latent_total, 256));
llm::compute_barrier(cmd);

    // Copy unpatchified velocity back to velocity buffer (for Euler step)
    // Python pipeline.py applies: noise_pred = -noise_pred.squeeze(2)
    // The model predicts velocity from data->noise, but we need noise->data direction
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.velocity.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_total * 4 }});
llm::compute_barrier(cmd);

    // Negate velocity to get correct direction (noise -> data)
    ScalePC neg_pc = { .n = latent_total, .scale = -1.0f };
llm::dispatch_kernel(cmd, &dk.scale_buffer,
        { a.velocity.gpu_buffer.buffer },
        { a.velocity.size_bytes },
        &neg_pc, llm::ceil_div(latent_total, 256));

llm::submit_and_wait(ctx)!!;

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
    // Free layer weights (CPU or GPU depending on streaming mode)
    uint num_layers = self.weights.actual_num_layers;
    uint num_refiner = self.weights.actual_num_refiner_layers;
    if (self.weights.streaming) {
        if (self.weights.cpu_layers != null) {
            for (uint l = 0; l < num_layers; l++) {
                (*self.weights.cpu_layers)[l].free();
            }
            mem::free(self.weights.cpu_layers);
        }
        if (self.weights.cpu_noise_refiner != null) {
            for (uint l = 0; l < num_refiner; l++) {
                (*self.weights.cpu_noise_refiner)[l].free();
            }
            mem::free(self.weights.cpu_noise_refiner);
        }
        if (self.weights.cpu_context_refiner != null) {
            for (uint l = 0; l < num_refiner; l++) {
                (*self.weights.cpu_context_refiner)[l].free();
            }
            mem::free(self.weights.cpu_context_refiner);
        }
    } else {
        for (uint l = 0; l < num_layers; l++) {
            self.weights.layers[l].free();
        }
        for (uint l = 0; l < num_refiner; l++) {
            self.weights.noise_refiner[l].free();
            self.weights.context_refiner[l].free();
        }
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
    self.kernels.linear_proj_debug.free(self.ctx.device);
    self.kernels.dit_timestep_embed.free(self.ctx.device);
    self.kernels.broadcast_add_dim.free(self.ctx.device);
    self.kernels.scale_buffer.free(self.ctx.device);
    self.kernels.flash_attention.free(self.ctx.device);
    self.kernels.scale_modulate.free(self.ctx.device);
    self.kernels.gated_residual.free(self.ctx.device);
    self.kernels.clip_values.free(self.ctx.device);
    self.kernels.batch_layernorm.free(self.ctx.device);
    self.kernels.mrope.free(self.ctx.device);
    self.kernels.batch_rmsnorm.free(self.ctx.device);
    self.kernels.transpose_heads.free(self.ctx.device);
    self.kernels.batch_head_norm.free(self.ctx.device);
    self.kernels.batch_matmul.free(self.ctx.device);
    self.kernels.batch_matmul_simple.free(self.ctx.device);
    self.kernels.batch_matmul_q8.free(self.ctx.device);
    self.kernels.batch_matmul_q4_0.free(self.ctx.device);
    self.kernels.batch_matmul_q4k.free(self.ctx.device);
    self.kernels.batch_matmul_q5k.free(self.ctx.device);
    self.kernels.batch_matmul_q6k.free(self.ctx.device);
    self.kernels.shared.free(self.ctx.device);
}
