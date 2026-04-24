module flux;

import vk;
import llm;
import llm::pipelines;
import zimage;
import std::io;
import std::math;
import std::core::mem;

// Flux 2 Klein Hybrid DiT Transformer
// dim=4096, heads=32, head_dim=128, mlp_dim=12288
// 8 dual-stream blocks + 24 single-stream blocks (Klein distilled)
// Shared modulations (same for all dual blocks, same for all single blocks)
// No biases, no CLIP pooled conditioning
//
// Flux.2 VAE latent: 32 channels at 8x spatial reduction
// Patchify = pixel unshuffle(2): [32, H/8, W/8] → [128, H/16, W/16]
// So patch_dim=128, effectively patch_size=1 over 128-ch latent

const uint FLUX_DIM        = 4096;
const uint FLUX_HEADS      = 32;   // dim / head_dim
const uint FLUX_HEAD_DIM   = 128;
const uint FLUX_MLP_DIM    = 12288;  // dim * 3
const uint FLUX_PATCH_SIZE = 1;      // patches are 1x1 over 128-channel latent
const uint FLUX_LATENT_CH  = 128;    // after pixel unshuffle: 32 VAE ch * 4
const uint FLUX_PATCH_DIM  = 128;    // LATENT_CH * PATCH_SIZE^2 = 128
const uint FLUX_VAE_CH     = 32;     // actual VAE latent channels before unshuffle
const float FLUX_ROPE_THETA = 2000.0f;  // Flux.2 uses theta=2000 (NOT 10000 like Flux.1)

// Text encoder: Qwen3-8B, 3 hidden states concatenated → 12288 dim
const uint FLUX_TEXT_DIM   = 12288;  // 3 * 4096 (Qwen3-8B hidden states)

// mRoPE axes for Flux.2 Klein: 4 axes [T, H, W, L] — each 32 dims, sum to head_dim=128
// T = always 0 (unused temporal axis)
// H = spatial height (image rows, 0 for text)
// W = spatial width (image cols, 0 for text)
// L = sequential position (text token index, 0 for image)
const uint FLUX_ROPE_AXES_T   = 32;  // temporal (always 0)
const uint FLUX_ROPE_AXES_H   = 32;  // height
const uint FLUX_ROPE_AXES_W   = 32;  // width
const uint FLUX_ROPE_AXES_L   = 32;  // sequential (text)

// Timestep embedding dims
const uint FLUX_T_EMB_DIM  = 256;

// --- Weight Structures ---

struct FluxDualBlockWeights {
    // Image attention (fused QKV)
    llm::Tensor img_qkv_w;        // [3*dim, dim]
    llm::Tensor img_proj_w;       // [dim, dim]
    llm::Tensor img_q_norm;       // [head_dim]
    llm::Tensor img_k_norm;       // [head_dim]

    // Text attention (fused QKV)
    llm::Tensor txt_qkv_w;        // [3*dim, dim]
    llm::Tensor txt_proj_w;       // [dim, dim]
    llm::Tensor txt_q_norm;       // [head_dim]
    llm::Tensor txt_k_norm;       // [head_dim]

    // Image FFN (GEGLU: up maps dim->2*mlp_dim, down maps mlp_dim->dim)
    llm::Tensor img_mlp_up_w;     // [2*mlp_dim, dim]
    llm::Tensor img_mlp_down_w;   // [dim, mlp_dim]

    // Text FFN
    llm::Tensor txt_mlp_up_w;
    llm::Tensor txt_mlp_down_w;
}

struct FluxSingleBlockWeights {
    // Fused QKV + MLP_in: [3*dim + 2*mlp_dim, dim]
    llm::Tensor linear1_w;

    // Fused output_proj + MLP_down: [dim, dim + mlp_dim]
    llm::Tensor linear2_w;

    // Per-head Q/K norms
    llm::Tensor q_norm;           // [head_dim]
    llm::Tensor k_norm;           // [head_dim]
}

struct FluxDiTGlobalWeights {
    // Input projections
    llm::Tensor img_in_w;         // [dim, patch_dim]
    llm::Tensor txt_in_w;         // [dim, txt_dim=4096]

    // Timestep embedding MLP (no bias)
    llm::Tensor time_in_w0;       // [dim, 256]
    llm::Tensor time_in_w1;       // [dim, dim]

    // Shared modulations (Klein: same for all blocks)
    llm::Tensor double_mod_img_w; // [6*dim, dim]
    llm::Tensor double_mod_txt_w; // [6*dim, dim]
    llm::Tensor single_mod_w;     // [3*dim, dim]

    // Final layer
    llm::Tensor final_linear_w;   // [patch_dim, dim]
    llm::Tensor final_adaln_w;    // [2*dim, dim]
}

// --- Activations ---

struct FluxDiTActivations {
    llm::Tensor img_hidden;     // [n_img, dim]
    llm::Tensor txt_hidden;     // [n_txt, dim]
    llm::Tensor hidden;         // [max_seq, dim] — combined for single-stream
    llm::Tensor buf_a;          // [max_seq, dim] scratch
    llm::Tensor buf_b;          // [max_seq, dim] scratch
    llm::Tensor buf_c;          // [max_seq, dim] scratch
    llm::Tensor q;              // [max_seq, dim] → [heads, seq, head_dim]
    llm::Tensor k;              // [max_seq, dim]
    llm::Tensor v;              // [max_seq, dim]
    llm::Tensor attn_out;       // [max_seq, dim] flash attention output
    llm::Tensor ffn_gate_out;   // [max_seq, mlp_dim]
    llm::Tensor ffn_up_out;     // [max_seq, mlp_dim]
    llm::Tensor concat_buf;     // [max_seq, dim+mlp_dim] for single linear2
    llm::Tensor patches;        // [n_patches, patch_dim]
    llm::Tensor velocity;       // [n_patches, patch_dim]
    llm::Tensor latent;         // [channels, h, w]
    llm::Tensor t_emb;          // [256] sinusoidal
    llm::Tensor t_emb_mlp;      // [dim] MLP output
    llm::Tensor silu_t;         // [dim] SiLU(t_emb_mlp)
    llm::Tensor dual_mod_img;   // [6*dim]
    llm::Tensor dual_mod_txt;   // [6*dim]
    llm::Tensor single_mod_buf; // [3*dim]
    llm::Tensor mod_shift;      // [dim] temp
    llm::Tensor mod_scale;      // [dim] temp
    llm::Tensor mod_gate;       // [dim] temp
}

// --- Model ---

struct FluxDiTModel {
    FluxDiTGlobalWeights global;
    FluxDualBlockWeights[] dual_blocks;
    FluxSingleBlockWeights[] single_blocks;
    uint n_dual_blocks;
    uint n_single_blocks;

    // mRoPE tables [seq_len, head_dim/2]
    llm::Tensor rope_cos;
    llm::Tensor rope_sin;

    FluxDiTKernels* dk;
    FluxDiTActivations acts;
    llm::DeviceContext* ctx;

    uint n_img;       // actual image patch count
    uint n_txt;       // actual text token count
    uint n_joint;     // n_img + n_txt
    uint latent_h;
    uint latent_w;
    uint patches_h;
    uint patches_w;
}

// --- Tensor Name Helpers ---

fn String flux_dit_name(char[256]* buf, String prefix, String suffix) {
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) { (*buf)[pos] = prefix[i]; pos++; }
    for (usz i = 0; i < suffix.len; i++) { (*buf)[pos] = suffix[i]; pos++; }
    return (String)(*buf)[0..pos - 1];
}

fn String flux_block_name(char[256]* buf, String prefix, uint layer, String suffix) {
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) { (*buf)[pos] = prefix[i]; pos++; }
    if (layer >= 10) { (*buf)[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    (*buf)[pos] = (char)('0' + layer % 10); pos++;
    (*buf)[pos] = '.'; pos++;
    for (usz i = 0; i < suffix.len; i++) { (*buf)[pos] = suffix[i]; pos++; }
    return (String)(*buf)[0..pos - 1];
}

fn llm::Tensor load_dit_tensor(llm::DeviceContext* ctx, llm::GGUFFile* gf, String name) {
    llm::GGUFTensorInfo* info = gf.find_tensor(name)!!;
    // BF16/F16 weights need conversion to F32 — no GPU matmul kernel for these types
    if (info.type == llm::GGML_BF16 || info.type == llm::GGML_F16) {
        return llm::upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
    }
    return llm::upload_weight(ctx, info, gf.tensor_data_base)!!;
}

// --- Detect Block Counts ---

fn void detect_flux_block_counts(llm::GGUFFile* gf, uint* n_dual, uint* n_single) {
    uint dual = 0;
    uint single = 0;

    for (usz i = 0; i < gf.n_tensors; i++) {
        String name = gf.tensors[i].name;
        if (name.len > 14) {
            if (name.contains("double_blocks.")) {
                if (try idx = name.index_of("double_blocks.")) {
                    usz start = idx + 14;
                    uint n = 0;
                    while (start < name.len && name[start] >= '0' && name[start] <= '9') {
                        n = n * 10 + (uint)(name[start] - '0');
                        start++;
                    }
                    if (n + 1 > dual) dual = n + 1;
                }
            }
            if (name.contains("single_blocks.")) {
                if (try idx = name.index_of("single_blocks.")) {
                    usz start = idx + 14;
                    uint n = 0;
                    while (start < name.len && name[start] >= '0' && name[start] <= '9') {
                        n = n * 10 + (uint)(name[start] - '0');
                        start++;
                    }
                    if (n + 1 > single) single = n + 1;
                }
            }
        }
    }

    *n_dual = dual;
    *n_single = single;
    io::printfn("Detected Flux blocks: %d dual-stream, %d single-stream", dual, single);
}

// --- Activation Allocation ---

fn FluxDiTActivations? allocate_flux_activations(
    llm::DeviceContext* ctx, uint n_img, uint n_txt, uint latent_elems
) {
    uint n_joint = n_img + n_txt;
    uint max_seq = n_joint;  // single-stream uses full sequence
    uint dim = FLUX_DIM;
    uint mlp = FLUX_MLP_DIM;
    ulong sd = (ulong)max_seq * dim;
    ulong sm = (ulong)max_seq * mlp;
    ulong sc = (ulong)max_seq * (ulong)(dim + mlp);

    return {
        .img_hidden = llm::create_f32_tensor(ctx, { (ulong)n_img * dim, 0, 0, 0 }, 1)!!,
        .txt_hidden = llm::create_f32_tensor(ctx, { (ulong)n_txt * dim, 0, 0, 0 }, 1)!!,
        .hidden = llm::create_f32_tensor(ctx, { sd, 0, 0, 0 }, 1)!!,
        .buf_a = llm::create_f32_tensor(ctx, { sd, 0, 0, 0 }, 1)!!,
        .buf_b = llm::create_f32_tensor(ctx, { sd, 0, 0, 0 }, 1)!!,
        .buf_c = llm::create_f32_tensor(ctx, { sd, 0, 0, 0 }, 1)!!,
        .q = llm::create_f32_tensor(ctx, { sd, 0, 0, 0 }, 1)!!,
        .k = llm::create_f32_tensor(ctx, { sd, 0, 0, 0 }, 1)!!,
        .v = llm::create_f32_tensor(ctx, { sd, 0, 0, 0 }, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, { sd, 0, 0, 0 }, 1)!!,
        .ffn_gate_out = llm::create_f32_tensor(ctx, { sm, 0, 0, 0 }, 1)!!,
        .ffn_up_out = llm::create_f32_tensor(ctx, { sm, 0, 0, 0 }, 1)!!,
        .concat_buf = llm::create_f32_tensor(ctx, { sc, 0, 0, 0 }, 1)!!,
        .patches = llm::create_f32_tensor(ctx, { (ulong)n_img * FLUX_PATCH_DIM, 0, 0, 0 }, 1)!!,
        .velocity = llm::create_f32_tensor(ctx, { (ulong)n_img * FLUX_PATCH_DIM, 0, 0, 0 }, 1)!!,
        .latent = llm::create_f32_tensor(ctx, { (ulong)latent_elems, 0, 0, 0 }, 1)!!,
        .t_emb = llm::create_f32_tensor(ctx, { FLUX_T_EMB_DIM, 0, 0, 0 }, 1)!!,
        .t_emb_mlp = llm::create_f32_tensor(ctx, { dim, 0, 0, 0 }, 1)!!,
        .silu_t = llm::create_f32_tensor(ctx, { dim, 0, 0, 0 }, 1)!!,
        .dual_mod_img = llm::create_f32_tensor(ctx, { (ulong)6 * dim, 0, 0, 0 }, 1)!!,
        .dual_mod_txt = llm::create_f32_tensor(ctx, { (ulong)6 * dim, 0, 0, 0 }, 1)!!,
        .single_mod_buf = llm::create_f32_tensor(ctx, { (ulong)3 * dim, 0, 0, 0 }, 1)!!,
        .mod_shift = llm::create_f32_tensor(ctx, { dim, 0, 0, 0 }, 1)!!,
        .mod_scale = llm::create_f32_tensor(ctx, { dim, 0, 0, 0 }, 1)!!,
        .mod_gate = llm::create_f32_tensor(ctx, { dim, 0, 0, 0 }, 1)!!,
    };
}

// --- mRoPE Precomputation ---
// Flux.2 Klein uses 4-axis mRoPE: (T, H, W, L) each 32 dims
// Layout: [text tokens, image tokens] — text first, then image
// Text token at position p: ids = (T=0, H=0, W=0, L=p)
// Image token at (row, col): ids = (T=0, H=row, W=col, L=0)

fn void? precompute_flux_mrope(
    llm::DeviceContext* ctx,
    llm::Tensor* cos_out,
    llm::Tensor* sin_out,
    uint n_txt,
    uint n_img,
    uint patches_w,
    uint patches_h
) {
    uint seq_len = n_txt + n_img;
    uint half_dim = FLUX_HEAD_DIM / 2; // 64
    usz table_size = (usz)seq_len * half_dim;

    float* cos_data = (float*)mem::calloc(table_size * 4);
    float* sin_data = (float*)mem::calloc(table_size * 4);

    for (uint t = 0; t < seq_len; t++) {
        // 4-axis position: (pos_t, pos_h, pos_w, pos_l)
        uint pos_t, pos_h, pos_w, pos_l;

        if (t < n_txt) {
            // Text token: sequential position on L axis
            pos_t = 0;
            pos_h = 0;
            pos_w = 0;
            pos_l = t;
        } else {
            // Image token: spatial position on H/W axes
            uint img_idx = t - n_txt;
            pos_t = 0;
            pos_h = img_idx / patches_w;
            pos_w = img_idx % patches_w;
            pos_l = 0;
        }

        uint pair = 0;
        // Axis 0: T (32 dims → 16 pairs) — temporal, always 0
        uint half_ax_t = FLUX_ROPE_AXES_T / 2;
        for (uint k = 0; k < half_ax_t; k++) {
            float freq = 1.0f / math::pow(FLUX_ROPE_THETA, (float)(2 * k) / (float)FLUX_ROPE_AXES_T);
            float angle = (float)pos_t * freq;
            cos_data[t * half_dim + pair] = math::cos(angle);
            sin_data[t * half_dim + pair] = math::sin(angle);
            pair++;
        }
        // Axis 1: H (32 dims → 16 pairs) — height
        uint half_ax_h = FLUX_ROPE_AXES_H / 2;
        for (uint k = 0; k < half_ax_h; k++) {
            float freq = 1.0f / math::pow(FLUX_ROPE_THETA, (float)(2 * k) / (float)FLUX_ROPE_AXES_H);
            float angle = (float)pos_h * freq;
            cos_data[t * half_dim + pair] = math::cos(angle);
            sin_data[t * half_dim + pair] = math::sin(angle);
            pair++;
        }
        // Axis 2: W (32 dims → 16 pairs) — width
        uint half_ax_w = FLUX_ROPE_AXES_W / 2;
        for (uint k = 0; k < half_ax_w; k++) {
            float freq = 1.0f / math::pow(FLUX_ROPE_THETA, (float)(2 * k) / (float)FLUX_ROPE_AXES_W);
            float angle = (float)pos_w * freq;
            cos_data[t * half_dim + pair] = math::cos(angle);
            sin_data[t * half_dim + pair] = math::sin(angle);
            pair++;
        }
        // Axis 3: L (32 dims → 16 pairs) — sequential (text position)
        uint half_ax_l = FLUX_ROPE_AXES_L / 2;
        for (uint k = 0; k < half_ax_l; k++) {
            float freq = 1.0f / math::pow(FLUX_ROPE_THETA, (float)(2 * k) / (float)FLUX_ROPE_AXES_L);
            float angle = (float)pos_l * freq;
            cos_data[t * half_dim + pair] = math::cos(angle);
            sin_data[t * half_dim + pair] = math::sin(angle);
            pair++;
        }
    }

    *cos_out = (llm::Tensor) {
        .gpu_buffer = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: cos_data,
            data_size: table_size * 4,
        )!!,
        .size_bytes = table_size * 4,
        .shape = { seq_len, half_dim, 0, 0 },
        .n_dims = 2,
        .dtype = llm::GGML_F32,
    };
    *sin_out = (llm::Tensor) {
        .gpu_buffer = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            data: sin_data,
            data_size: table_size * 4,
        )!!,
        .size_bytes = table_size * 4,
        .shape = { seq_len, half_dim, 0, 0 },
        .n_dims = 2,
        .dtype = llm::GGML_F32,
    };

    mem::free(cos_data);
    mem::free(sin_data);
    io::printfn("  mRoPE tables: [%d, %d] (theta=%.0f)", seq_len, half_dim, FLUX_ROPE_THETA);
}

// --- Batch Matmul Dispatch Helper ---

fn void flux_matmul(
    llm::RenderGraph* g,
    zimage::DiTKernels* dk,
    llm::Tensor* weight,
    llm::Tensor* input,
    llm::Tensor* output,
    uint out_dim,
    uint in_dim,
    uint seq_len,
    uint row_offset
) {
    zimage::dispatch_batch_matmul(g, dk, weight, input, output, out_dim, in_dim, seq_len, row_offset);
}

// --- Copy a chunk from a larger buffer to a separate buffer ---

fn void copy_chunk(llm::RenderGraph* g, llm::Tensor* src, llm::Tensor* dst, uint offset_floats, uint n_floats) {
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    vk::cmdCopyBuffer(g.cmd, src.gpu_buffer.buffer, dst.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = (ulong)offset_floats * 4, .dstOffset = 0, .size = (ulong)n_floats * 4 }});
    g.needs_barrier = true;
}

// --- Copy from src to dst at a specific destination offset ---

fn void copy_to_offset(llm::RenderGraph* g, llm::Tensor* src, llm::Tensor* dst, uint dst_offset_floats, uint n_floats) {
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    vk::cmdCopyBuffer(g.cmd, src.gpu_buffer.buffer, dst.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)dst_offset_floats * 4, .size = (ulong)n_floats * 4 }});
    g.needs_barrier = true;
}

// --- Concat rows for single-stream linear2 ---
// Interleave attn_out[seq, dim] and mlp_out[seq, mlp] into concat[seq, dim+mlp]

fn void concat_rows_copy(
    vk::CommandBuffer cmd,
    llm::Tensor* attn_out,
    llm::Tensor* mlp_out,
    llm::Tensor* concat_out,
    uint seq_len,
    uint dim,
    uint mlp_dim
) {
    uint total = dim + mlp_dim;

    // Build copy regions for attn and mlp
    vk::BufferCopy[] attn_regions = mem::new_array(vk::BufferCopy, seq_len);
    vk::BufferCopy[] mlp_regions = mem::new_array(vk::BufferCopy, seq_len);

    for (uint i = 0; i < seq_len; i++) {
        attn_regions[i] = {
            .srcOffset = (ulong)i * dim * 4,
            .dstOffset = (ulong)i * total * 4,
            .size = (ulong)dim * 4,
        };
        mlp_regions[i] = {
            .srcOffset = (ulong)i * mlp_dim * 4,
            .dstOffset = (ulong)(i * total + dim) * 4,
            .size = (ulong)mlp_dim * 4,
        };
    }

    vk::cmdCopyBuffer(cmd, attn_out.gpu_buffer.buffer, concat_out.gpu_buffer.buffer,
        seq_len, &attn_regions[0]);
    vk::cmdCopyBuffer(cmd, mlp_out.gpu_buffer.buffer, concat_out.gpu_buffer.buffer,
        seq_len, &mlp_regions[0]);

    mem::free(attn_regions);
    mem::free(mlp_regions);
}

// --- Load Model ---

fn FluxDiTModel? load_flux_dit(llm::DeviceContext* ctx, llm::GGUFFile* gf, FluxDiTKernels* dk) {
    io::printfn("\nLoading Flux DiT model...");

    uint n_dual, n_single;
    detect_flux_block_counts(gf, &n_dual, &n_single);

    // Diagnostic: check key tensor shapes and existence
    {
        uint bias_count = 0;
        bool has_guidance = false;
        bool has_vector = false;
        for (usz i = 0; i < gf.n_tensors; i++) {
            String name = gf.tensors[i].name;
            if (name.contains("bias")) bias_count++;
            if (name.contains("guidance")) has_guidance = true;
            if (name.contains("vector")) has_vector = true;
        }
        io::printfn("  Tensor stats: %d total, %d with bias, guidance=%s, vector=%s",
            (uint)gf.n_tensors, bias_count, has_guidance ? "yes" : "no", has_vector ? "yes" : "no");

        // Check key tensor shapes
        if (try info = gf.find_tensor("txt_in.weight")) {
            io::printfn("  txt_in.weight: [%d, %d] type=%d", (uint)info.shape[0], (uint)info.shape[1], (uint)info.type);
        }
        if (try info = gf.find_tensor("img_in.weight")) {
            io::printfn("  img_in.weight: [%d, %d] type=%d", (uint)info.shape[0], (uint)info.shape[1], (uint)info.type);
        }
        if (try info = gf.find_tensor("time_in.in_layer.weight")) {
            io::printfn("  time_in.in_layer.weight: [%d, %d] type=%d", (uint)info.shape[0], (uint)info.shape[1], (uint)info.type);
        }
        // Check for biases in time_in
        if (try info = gf.find_tensor("time_in.in_layer.bias")) {
            io::printfn("  time_in.in_layer.bias: [%d] EXISTS", (uint)info.shape[0]);
        }
        if (try info = gf.find_tensor("time_in.out_layer.bias")) {
            io::printfn("  time_in.out_layer.bias: [%d] EXISTS", (uint)info.shape[0]);
        }
        // Check guidance_in
        if (try info = gf.find_tensor("guidance_in.in_layer.weight")) {
            io::printfn("  guidance_in.in_layer.weight: [%d, %d] EXISTS", (uint)info.shape[0], (uint)info.shape[1]);
        }
        // Check vector_in
        if (try info = gf.find_tensor("vector_in.in_layer.weight")) {
            io::printfn("  vector_in.in_layer.weight: [%d, %d] EXISTS", (uint)info.shape[0], (uint)info.shape[1]);
        }
        // Check modulation biases
        if (try info = gf.find_tensor("double_stream_modulation_img.lin.bias")) {
            io::printfn("  double_stream_modulation_img.lin.bias: [%d] EXISTS", (uint)info.shape[0]);
        }
        // Check final layer biases
        if (try info = gf.find_tensor("final_layer.linear.bias")) {
            io::printfn("  final_layer.linear.bias: [%d] EXISTS", (uint)info.shape[0]);
        }
        if (try info = gf.find_tensor("final_layer.adaLN_modulation.1.bias")) {
            io::printfn("  final_layer.adaLN_modulation.1.bias: [%d] EXISTS", (uint)info.shape[0]);
        }
        // Print a few bias tensor names for reference
        uint printed = 0;
        for (usz i = 0; i < gf.n_tensors && printed < 5; i++) {
            String name = gf.tensors[i].name;
            if (name.contains("bias")) {
                io::printfn("    bias tensor: %s [%d]", name, (uint)gf.tensors[i].shape[0]);
                printed++;
            }
        }
    }

    char[256] buf;

    // Load global weights
    FluxDiTGlobalWeights global;
    global.img_in_w = load_dit_tensor(ctx, gf, "img_in.weight");
    global.txt_in_w = load_dit_tensor(ctx, gf, "txt_in.weight");
    global.time_in_w0 = load_dit_tensor(ctx, gf, "time_in.in_layer.weight");
    global.time_in_w1 = load_dit_tensor(ctx, gf, "time_in.out_layer.weight");

    global.double_mod_img_w = load_dit_tensor(ctx, gf, "double_stream_modulation_img.lin.weight");
    global.double_mod_txt_w = load_dit_tensor(ctx, gf, "double_stream_modulation_txt.lin.weight");
    global.single_mod_w = load_dit_tensor(ctx, gf, "single_stream_modulation.lin.weight");

    global.final_linear_w = load_dit_tensor(ctx, gf, "final_layer.linear.weight");
    global.final_adaln_w = load_dit_tensor(ctx, gf, "final_layer.adaLN_modulation.1.weight");

    io::printfn("  Global weights loaded");

    // Load dual-stream blocks
    FluxDualBlockWeights[] dual_blocks = mem::new_array(FluxDualBlockWeights, n_dual);
    for (uint l = 0; l < n_dual; l++) {
        String blk_prefix = "double_blocks.";

        dual_blocks[l] = {
            .img_qkv_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "img_attn.qkv.weight")),
            .img_proj_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "img_attn.proj.weight")),
            .img_q_norm = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "img_attn.norm.query_norm.scale")),
            .img_k_norm = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "img_attn.norm.key_norm.scale")),

            .txt_qkv_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "txt_attn.qkv.weight")),
            .txt_proj_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "txt_attn.proj.weight")),
            .txt_q_norm = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "txt_attn.norm.query_norm.scale")),
            .txt_k_norm = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "txt_attn.norm.key_norm.scale")),

            .img_mlp_up_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "img_mlp.0.weight")),
            .img_mlp_down_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "img_mlp.2.weight")),

            .txt_mlp_up_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "txt_mlp.0.weight")),
            .txt_mlp_down_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "txt_mlp.2.weight")),
        };

        if ((l + 1) % 5 == 0 || l == n_dual - 1) {
            io::printfn("  Dual block %d / %d loaded", l + 1, n_dual);
        }
    }

    // Load single-stream blocks
    FluxSingleBlockWeights[] single_blocks = mem::new_array(FluxSingleBlockWeights, n_single);
    for (uint l = 0; l < n_single; l++) {
        String blk_prefix = "single_blocks.";

        single_blocks[l] = {
            .linear1_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "linear1.weight")),
            .linear2_w = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "linear2.weight")),
            .q_norm = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "norm.query_norm.scale")),
            .k_norm = load_dit_tensor(ctx, gf, flux_block_name(&buf, blk_prefix, l, "norm.key_norm.scale")),
        };

        if ((l + 1) % 10 == 0 || l == n_single - 1) {
            io::printfn("  Single block %d / %d loaded", l + 1, n_single);
        }
    }

    io::printfn("  Flux DiT loaded: %d dual + %d single blocks, dim=%d", n_dual, n_single, FLUX_DIM);

    return {
        .global = global,
        .dual_blocks = dual_blocks,
        .single_blocks = single_blocks,
        .n_dual_blocks = n_dual,
        .n_single_blocks = n_single,
        .dk = dk,
        .ctx = ctx,
    };
}

// --- Dual-Stream Block ---

fn void? flux_dual_stream_block(
    FluxDiTModel* model,
    llm::RenderGraph* g,
    FluxDualBlockWeights* lw
) {
    zimage::DiTKernels* dk = &model.dk.dit;
    FluxDiTActivations* a = &model.acts;
    uint dim = FLUX_DIM;
    uint mlp = FLUX_MLP_DIM;
    uint n_img = model.n_img;
    uint n_txt = model.n_txt;
    uint n_joint = model.n_joint;

    // === Pre-attention: LayerNorm + AdaLN modulation ===

    // IMG: LayerNorm(img_hidden) → buf_a
    copy_chunk(g, &a.dual_mod_img, &a.mod_shift, 0, dim);       // shift1
    copy_chunk(g, &a.dual_mod_img, &a.mod_scale, dim, dim);     // scale1
    zimage::BatchLayerNormPC img_ln = { .dim = dim, .eps = 1e-6f, .n_rows = n_img };
    g.dispatch(&dk.batch_layernorm, { &a.img_hidden, &a.buf_a }, &img_ln, n_img);
    zimage::AdalnModPC img_mod = { .n_elements = n_img * dim, .dim = dim };
    g.dispatch(&dk.adaln_modulate, { &a.buf_a, &a.mod_scale, &a.mod_shift, &a.buf_a },
        &img_mod, llm::ceil_div(n_img * dim, 256));

    // TXT: LayerNorm(txt_hidden) → buf_b
    copy_chunk(g, &a.dual_mod_txt, &a.mod_shift, 0, dim);
    copy_chunk(g, &a.dual_mod_txt, &a.mod_scale, dim, dim);
    zimage::BatchLayerNormPC txt_ln = { .dim = dim, .eps = 1e-6f, .n_rows = n_txt };
    g.dispatch(&dk.batch_layernorm, { &a.txt_hidden, &a.buf_b }, &txt_ln, n_txt);
    zimage::AdalnModPC txt_mod = { .n_elements = n_txt * dim, .dim = dim };
    g.dispatch(&dk.adaln_modulate, { &a.buf_b, &a.mod_scale, &a.mod_shift, &a.buf_b },
        &txt_mod, llm::ceil_div(n_txt * dim, 256));

    // === QKV projections ===
    // IMG Q/K/V → first n_img rows of q/k/v (txt first in joint layout, so img at offset n_txt)
    flux_matmul(g, dk, &lw.img_qkv_w, &a.buf_a, &a.buf_c, dim, dim, n_img, 0);       // Q
    copy_to_offset(g, &a.buf_c, &a.q, n_txt * dim, n_img * dim);
    flux_matmul(g, dk, &lw.img_qkv_w, &a.buf_a, &a.buf_c, dim, dim, n_img, dim);     // K
    copy_to_offset(g, &a.buf_c, &a.k, n_txt * dim, n_img * dim);
    flux_matmul(g, dk, &lw.img_qkv_w, &a.buf_a, &a.buf_c, dim, dim, n_img, 2 * dim); // V
    copy_to_offset(g, &a.buf_c, &a.v, n_txt * dim, n_img * dim);

    // TXT Q/K/V → first n_txt rows of q/k/v (txt at offset 0)
    flux_matmul(g, dk, &lw.txt_qkv_w, &a.buf_b, &a.q, dim, dim, n_txt, 0);       // Q
    flux_matmul(g, dk, &lw.txt_qkv_w, &a.buf_b, &a.k, dim, dim, n_txt, dim);     // K
    flux_matmul(g, dk, &lw.txt_qkv_w, &a.buf_b, &a.v, dim, dim, n_txt, 2 * dim); // V

    // === Per-head Q/K RMSNorm (applied to full joint Q/K) ===
    // IMG norms applied to img portion, TXT norms to txt portion
    // For simplicity: apply img norms to full buffer, then overwrite txt portion
    // Actually, need separate norm calls for each portion since they use different weights

    // IMG portion: q/k at offset [n_txt*dim .. n_joint*dim]
    // batch_head_norm operates on the full buffer, so we need to use it per-portion
    // We'll use the img norm weights on the img portion and txt norm weights on txt portion

    // Apply head norm to txt portion (offset 0)
    zimage::BatchHeadNormPC txt_hn = { .n_heads = FLUX_HEADS, .head_dim = FLUX_HEAD_DIM, .seq_len = n_txt, .eps = 1e-6f };
    g.dispatch(&dk.batch_head_norm, { &a.q, &lw.txt_q_norm }, &txt_hn, n_txt * FLUX_HEADS);
    g.dispatch(&dk.batch_head_norm, { &a.k, &lw.txt_k_norm }, &txt_hn, n_txt * FLUX_HEADS);

    // Apply head norm to img portion (need to offset the buffer)
    // batch_head_norm reads input[gid * head_dim ... (gid+1) * head_dim - 1]
    // gid ranges from 0 to seq_len * n_heads - 1
    // For the img portion starting at offset n_txt*dim, we need gid offset = n_txt * FLUX_HEADS
    // Unfortunately batch_head_norm doesn't support offset. We need to use buf_c as temp.

    // Copy img Q/K portions to buf_c, apply norm, copy back
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    vk::cmdCopyBuffer(g.cmd, a.q.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = (ulong)n_txt * dim * 4, .dstOffset = 0, .size = (ulong)n_img * dim * 4 }});
    g.needs_barrier = true;
    zimage::BatchHeadNormPC img_hn = { .n_heads = FLUX_HEADS, .head_dim = FLUX_HEAD_DIM, .seq_len = n_img, .eps = 1e-6f };
    g.dispatch(&dk.batch_head_norm, { &a.buf_c, &lw.img_q_norm }, &img_hn, n_img * FLUX_HEADS);
    // Copy back
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    vk::cmdCopyBuffer(g.cmd, a.buf_c.gpu_buffer.buffer, a.q.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)n_txt * dim * 4, .size = (ulong)n_img * dim * 4 }});
    g.needs_barrier = true;

    // Same for K
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    vk::cmdCopyBuffer(g.cmd, a.k.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = (ulong)n_txt * dim * 4, .dstOffset = 0, .size = (ulong)n_img * dim * 4 }});
    g.needs_barrier = true;
    g.dispatch(&dk.batch_head_norm, { &a.buf_c, &lw.img_k_norm }, &img_hn, n_img * FLUX_HEADS);
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    vk::cmdCopyBuffer(g.cmd, a.buf_c.gpu_buffer.buffer, a.k.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)n_txt * dim * 4, .size = (ulong)n_img * dim * 4 }});
    g.needs_barrier = true;

    // === Transpose Q/K/V: [n_joint, dim] → [heads, n_joint, head_dim] ===
    uint total_elems = n_joint * dim;
    uint transpose_groups = llm::ceil_div(total_elems, 256);
    zimage::TransposeHeadsPC th_fwd = {
        .seq_len = n_joint, .n_heads = FLUX_HEADS, .head_dim = FLUX_HEAD_DIM, .direction = 0,
    };
    // Q → buf_a (transposed Q), K → buf_b (transposed K), V → buf_c (transposed V)
    g.dispatch(&dk.transpose_heads, { &a.q, &a.buf_a }, &th_fwd, transpose_groups);
    g.dispatch(&dk.transpose_heads, { &a.k, &a.buf_b }, &th_fwd, transpose_groups);
    g.dispatch(&dk.transpose_heads, { &a.v, &a.buf_c }, &th_fwd, transpose_groups);

    // === Apply mRoPE to Q (buf_a) and K (buf_b) ===
    zimage::MRoPEPC mrope_pc = { .seq_len = n_joint, .n_heads = FLUX_HEADS };
    uint mrope_groups_y = FLUX_HEADS * n_joint;
    g.dispatch(&dk.mrope, { &a.buf_a, &model.rope_cos, &model.rope_sin }, &mrope_pc, 1, mrope_groups_y);
    g.dispatch(&dk.mrope, { &a.buf_b, &model.rope_cos, &model.rope_sin }, &mrope_pc, 1, mrope_groups_y);

    // === Flash Attention ===
    // Input: Q=buf_a, K=buf_b, V=buf_c in [heads, n_joint, head_dim]
    // Output: attn_out in [n_joint, dim] (already transposed back by FA kernel)
    float scale = 1.0f / math::sqrt((float)FLUX_HEAD_DIM);
    zimage::FlashAttentionPC fa_pc = { .head_dim = FLUX_HEAD_DIM, .n_heads = FLUX_HEADS, .seq_len = n_joint, .scale = scale };
    uint fa_q_blocks = llm::ceil_div(n_joint, 8);
    g.dispatch(&dk.flash_attention, { &a.buf_a, &a.buf_b, &a.buf_c, &a.attn_out },
        &fa_pc, fa_q_blocks, FLUX_HEADS);

    // === Split attention output and project ===
    // TXT portion: attn_out[0:n_txt*dim] → txt_proj → buf_b
    flux_matmul(g, dk, &lw.txt_proj_w, &a.attn_out, &a.buf_b, dim, dim, n_txt, 0);

    // IMG portion: attn_out[n_txt*dim:] → buf_c, then img_proj → buf_a
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    vk::cmdCopyBuffer(g.cmd, a.attn_out.gpu_buffer.buffer, a.buf_c.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = (ulong)n_txt * dim * 4, .dstOffset = 0, .size = (ulong)n_img * dim * 4 }});
    g.needs_barrier = true;
    flux_matmul(g, dk, &lw.img_proj_w, &a.buf_c, &a.buf_a, dim, dim, n_img, 0);

    // === Gated residual (post-attention) ===
    // img_hidden += gate1_img * buf_a
    copy_chunk(g, &a.dual_mod_img, &a.mod_gate, 2 * dim, dim); // gate1
    zimage::GatedResidualPC img_gres = { .n_elements = n_img * dim, .dim = dim, .scale = 0, .clip_thresh = 0 };
    g.dispatch(&model.dk.gated_residual_linear, { &a.img_hidden, &a.mod_gate, &a.buf_a },
        &img_gres, llm::ceil_div(n_img * dim, 256));

    // txt_hidden += gate1_txt * buf_b
    copy_chunk(g, &a.dual_mod_txt, &a.mod_gate, 2 * dim, dim);
    zimage::GatedResidualPC txt_gres = { .n_elements = n_txt * dim, .dim = dim, .scale = 0, .clip_thresh = 0 };
    g.dispatch(&model.dk.gated_residual_linear, { &a.txt_hidden, &a.mod_gate, &a.buf_b },
        &txt_gres, llm::ceil_div(n_txt * dim, 256));

    g.checkpoint()!!;

    // === Pre-FFN: LayerNorm + AdaLN modulation ===

    // IMG
    copy_chunk(g, &a.dual_mod_img, &a.mod_shift, 3 * dim, dim);  // shift2
    copy_chunk(g, &a.dual_mod_img, &a.mod_scale, 4 * dim, dim);  // scale2
    g.dispatch(&dk.batch_layernorm, { &a.img_hidden, &a.buf_a }, &img_ln, n_img);
    g.dispatch(&dk.adaln_modulate, { &a.buf_a, &a.mod_scale, &a.mod_shift, &a.buf_a },
        &img_mod, llm::ceil_div(n_img * dim, 256));

    // IMG GEGLU FFN: up→[2*mlp], split x/gate, x*GELU(gate), down→[dim]
    // Reference: x, gate = chunk(2); return out_proj(x * GELU(gate))
    // x = first half (row_offset=0), gate = second half (row_offset=mlp)
    flux_matmul(g, dk, &lw.img_mlp_up_w, &a.buf_a, &a.ffn_gate_out, mlp, dim, n_img, 0);     // x (features)
    flux_matmul(g, dk, &lw.img_mlp_up_w, &a.buf_a, &a.ffn_up_out, mlp, dim, n_img, mlp);      // gate
    llm::GeluPC gelu_pc = { .n = n_img * mlp };
    g.dispatch(&dk.shared.gelu, { &a.ffn_up_out }, &gelu_pc, llm::ceil_div(n_img * mlp, 256));
    llm::ElemwisePC emul_pc = { .n = n_img * mlp };
    g.dispatch(&dk.shared.elemwise_mul, { &a.ffn_gate_out, &a.ffn_up_out }, &emul_pc, llm::ceil_div(n_img * mlp, 256));
    flux_matmul(g, dk, &lw.img_mlp_down_w, &a.ffn_gate_out, &a.buf_a, dim, mlp, n_img, 0);

    // IMG gated residual: img_hidden += gate2_img * buf_a
    copy_chunk(g, &a.dual_mod_img, &a.mod_gate, 5 * dim, dim);  // gate2
    g.dispatch(&model.dk.gated_residual_linear, { &a.img_hidden, &a.mod_gate, &a.buf_a },
        &img_gres, llm::ceil_div(n_img * dim, 256));

    // TXT
    copy_chunk(g, &a.dual_mod_txt, &a.mod_shift, 3 * dim, dim);
    copy_chunk(g, &a.dual_mod_txt, &a.mod_scale, 4 * dim, dim);
    g.dispatch(&dk.batch_layernorm, { &a.txt_hidden, &a.buf_b }, &txt_ln, n_txt);
    g.dispatch(&dk.adaln_modulate, { &a.buf_b, &a.mod_scale, &a.mod_shift, &a.buf_b },
        &txt_mod, llm::ceil_div(n_txt * dim, 256));

    // TXT GEGLU FFN: x = first half, gate = second half, GELU on gate
    flux_matmul(g, dk, &lw.txt_mlp_up_w, &a.buf_b, &a.ffn_gate_out, mlp, dim, n_txt, 0);
    flux_matmul(g, dk, &lw.txt_mlp_up_w, &a.buf_b, &a.ffn_up_out, mlp, dim, n_txt, mlp);
    llm::GeluPC gelu_txt = { .n = n_txt * mlp };
    g.dispatch(&dk.shared.gelu, { &a.ffn_up_out }, &gelu_txt, llm::ceil_div(n_txt * mlp, 256));
    llm::ElemwisePC emul_txt = { .n = n_txt * mlp };
    g.dispatch(&dk.shared.elemwise_mul, { &a.ffn_gate_out, &a.ffn_up_out }, &emul_txt, llm::ceil_div(n_txt * mlp, 256));
    flux_matmul(g, dk, &lw.txt_mlp_down_w, &a.ffn_gate_out, &a.buf_b, dim, mlp, n_txt, 0);

    copy_chunk(g, &a.dual_mod_txt, &a.mod_gate, 5 * dim, dim);
    g.dispatch(&model.dk.gated_residual_linear, { &a.txt_hidden, &a.mod_gate, &a.buf_b },
        &txt_gres, llm::ceil_div(n_txt * dim, 256));

    g.checkpoint()!!;
}

// --- Single-Stream Block ---

fn void? flux_single_stream_block(
    FluxDiTModel* model,
    llm::RenderGraph* g,
    FluxSingleBlockWeights* lw
) {
    zimage::DiTKernels* dk = &model.dk.dit;
    FluxDiTActivations* a = &model.acts;
    uint dim = FLUX_DIM;
    uint mlp = FLUX_MLP_DIM;
    uint seq = model.n_joint;

    // === Pre-norm: LayerNorm + AdaLN modulation ===
    copy_chunk(g, &a.single_mod_buf, &a.mod_shift, 0, dim);       // shift
    copy_chunk(g, &a.single_mod_buf, &a.mod_scale, dim, dim);     // scale
    zimage::BatchLayerNormPC ln_pc = { .dim = dim, .eps = 1e-6f, .n_rows = seq };
    g.dispatch(&dk.batch_layernorm, { &a.hidden, &a.buf_a }, &ln_pc, seq);
    zimage::AdalnModPC mod_pc = { .n_elements = seq * dim, .dim = dim };
    g.dispatch(&dk.adaln_modulate, { &a.buf_a, &a.mod_scale, &a.mod_shift, &a.buf_a },
        &mod_pc, llm::ceil_div(seq * dim, 256));

    // === Fused linear1: [seq, dim] → split into Q/K/V + gate/up ===
    // linear1_w: [3*dim + 2*mlp, dim]
    flux_matmul(g, dk, &lw.linear1_w, &a.buf_a, &a.q, dim, dim, seq, 0);                          // Q
    flux_matmul(g, dk, &lw.linear1_w, &a.buf_a, &a.k, dim, dim, seq, dim);                        // K
    flux_matmul(g, dk, &lw.linear1_w, &a.buf_a, &a.v, dim, dim, seq, 2 * dim);                    // V
    flux_matmul(g, dk, &lw.linear1_w, &a.buf_a, &a.ffn_gate_out, mlp, dim, seq, 3 * dim);         // gate
    flux_matmul(g, dk, &lw.linear1_w, &a.buf_a, &a.ffn_up_out, mlp, dim, seq, 3 * dim + mlp);     // up

    // === Per-head Q/K norm ===
    zimage::BatchHeadNormPC hn_pc = { .n_heads = FLUX_HEADS, .head_dim = FLUX_HEAD_DIM, .seq_len = seq, .eps = 1e-6f };
    g.dispatch(&dk.batch_head_norm, { &a.q, &lw.q_norm }, &hn_pc, seq * FLUX_HEADS);
    g.dispatch(&dk.batch_head_norm, { &a.k, &lw.k_norm }, &hn_pc, seq * FLUX_HEADS);

    // === Transpose Q/K/V: [seq, dim] → [heads, seq, head_dim] ===
    uint total_elems = seq * dim;
    uint transpose_groups = llm::ceil_div(total_elems, 256);
    zimage::TransposeHeadsPC th_fwd = {
        .seq_len = seq, .n_heads = FLUX_HEADS, .head_dim = FLUX_HEAD_DIM, .direction = 0,
    };
    g.dispatch(&dk.transpose_heads, { &a.q, &a.buf_a }, &th_fwd, transpose_groups);   // Q transposed → buf_a
    g.dispatch(&dk.transpose_heads, { &a.k, &a.buf_b }, &th_fwd, transpose_groups);   // K transposed → buf_b
    g.dispatch(&dk.transpose_heads, { &a.v, &a.buf_c }, &th_fwd, transpose_groups);   // V transposed → buf_c

    // === mRoPE on Q and K ===
    zimage::MRoPEPC mrope_pc = { .seq_len = seq, .n_heads = FLUX_HEADS };
    uint mrope_groups_y = FLUX_HEADS * seq;
    g.dispatch(&dk.mrope, { &a.buf_a, &model.rope_cos, &model.rope_sin }, &mrope_pc, 1, mrope_groups_y);
    g.dispatch(&dk.mrope, { &a.buf_b, &model.rope_cos, &model.rope_sin }, &mrope_pc, 1, mrope_groups_y);

    // === Flash Attention ===
    float scale = 1.0f / math::sqrt((float)FLUX_HEAD_DIM);
    zimage::FlashAttentionPC fa_pc = { .head_dim = FLUX_HEAD_DIM, .n_heads = FLUX_HEADS, .seq_len = seq, .scale = scale };
    uint fa_q_blocks = llm::ceil_div(seq, 8);
    g.dispatch(&dk.flash_attention, { &a.buf_a, &a.buf_b, &a.buf_c, &a.attn_out },
        &fa_pc, fa_q_blocks, FLUX_HEADS);

    // === GEGLU on MLP portion: x * GELU(gate), gate is the second half ===
    llm::GeluPC gelu_pc = { .n = seq * mlp };
    g.dispatch(&dk.shared.gelu, { &a.ffn_up_out }, &gelu_pc, llm::ceil_div(seq * mlp, 256));
    llm::ElemwisePC emul_pc = { .n = seq * mlp };
    g.dispatch(&dk.shared.elemwise_mul, { &a.ffn_gate_out, &a.ffn_up_out }, &emul_pc, llm::ceil_div(seq * mlp, 256));

    // === Concat attn_out + mlp_out → linear2 input ===
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    concat_rows_copy(g.cmd, &a.attn_out, &a.ffn_gate_out, &a.concat_buf, seq, dim, mlp);
    g.needs_barrier = true;

    // === linear2: [seq, dim+mlp] → [seq, dim] ===
    flux_matmul(g, dk, &lw.linear2_w, &a.concat_buf, &a.buf_a, dim, dim + mlp, seq, 0);

    // === Gated residual: hidden += gate * output ===
    copy_chunk(g, &a.single_mod_buf, &a.mod_gate, 2 * dim, dim);  // gate
    zimage::GatedResidualPC gres = { .n_elements = seq * dim, .dim = dim, .scale = 0, .clip_thresh = 0 };
    g.dispatch(&model.dk.gated_residual_linear, { &a.hidden, &a.mod_gate, &a.buf_a },
        &gres, llm::ceil_div(seq * dim, 256));

    g.checkpoint()!!;
}

// --- Main Forward Pass ---

fn void? FluxDiTModel.forward(
    &self,
    llm::Tensor* t5_embeddings, // [n_txt, 4096] T5 output
    uint t5_seq_len,
    float sigma                 // timestep (0=clean, 1=noise)
) {
    llm::DeviceContext* ctx = self.ctx;
    zimage::DiTKernels* dk = &self.dk.dit;
    FluxDiTActivations* a = &self.acts;
    FluxDiTGlobalWeights* w = &self.global;
    uint dim = FLUX_DIM;
    uint n_img = self.n_img;
    uint n_txt = self.n_txt;
    uint n_joint = self.n_joint;

    llm::RenderGraph g;
    g.init(ctx)!!;

    // 1. Patchify latent → [n_patches, 128]
    // Flux.2 latent is [128, H/16, W/16] after pixel unshuffle.
    // With PATCH_SIZE=1, patchify just reshapes [128, H, W] → [H*W, 128]
    zimage::PatchifyPC patch_pc = {
        .channels = FLUX_LATENT_CH, .height = self.latent_h, .width = self.latent_w, .patch_size = FLUX_PATCH_SIZE,
    };
    g.dispatch(&dk.patchify, { &a.latent, &a.patches }, &patch_pc,
        llm::ceil_div(n_img * FLUX_PATCH_DIM, 256));

    // 2. img_in: [n_patches, 128] → [n_patches, dim]
    flux_matmul(&g, dk, &w.img_in_w, &a.patches, &a.img_hidden, dim, FLUX_PATCH_DIM, n_img, 0);

    // 3. txt_in: [n_txt, 12288] → [n_txt, dim]
    flux_matmul(&g, dk, &w.txt_in_w, t5_embeddings, &a.txt_hidden, dim, FLUX_TEXT_DIM, n_txt, 0);

    // 4. Timestep embedding: sigma * 1000 → sincos[256] → MLP → [dim]
    float timestep_val = sigma * 1000.0f;
    zimage::DiTTimestepPC ts_pc = { .dim = FLUX_T_EMB_DIM, .timestep = timestep_val };
    g.dispatch(&dk.dit_timestep_embed, { &a.t_emb }, &ts_pc, llm::ceil_div(FLUX_T_EMB_DIM, 256));

    // MLP: t_emb[256] → w0 → [dim] → SiLU → w1 → [dim]
    flux_matmul(&g, dk, &w.time_in_w0, &a.t_emb, &a.t_emb_mlp, dim, FLUX_T_EMB_DIM, 1, 0);
    llm::SiluPC silu1_pc = { .n = dim };
    g.dispatch(&dk.shared.silu, { &a.t_emb_mlp }, &silu1_pc, llm::ceil_div(dim, 256));
    flux_matmul(&g, dk, &w.time_in_w1, &a.t_emb_mlp, &a.silu_t, dim, dim, 1, 0);

    // 5. Shared modulations: SiLU(t_emb_final) → matmul → mod params
    g.copy(&a.silu_t, &a.t_emb_mlp, (ulong)dim * 4);
    llm::SiluPC silu2_pc = { .n = dim };
    g.dispatch(&dk.shared.silu, { &a.t_emb_mlp }, &silu2_pc, llm::ceil_div(dim, 256));

    // dual_mod_img = SiLU(t_emb) @ double_mod_img_w^T → [6*dim]
    flux_matmul(&g, dk, &w.double_mod_img_w, &a.t_emb_mlp, &a.dual_mod_img, 6 * dim, dim, 1, 0);
    flux_matmul(&g, dk, &w.double_mod_txt_w, &a.t_emb_mlp, &a.dual_mod_txt, 6 * dim, dim, 1, 0);
    flux_matmul(&g, dk, &w.single_mod_w, &a.t_emb_mlp, &a.single_mod_buf, 3 * dim, dim, 1, 0);

    g.checkpoint()!!;

    // Debug: print img_hidden and txt_hidden after embedding
    g.checkpoint()!!;
    {
        float[4] dbg;
        vk::Memory dbgm = vk::new_buffer(
            allocator: &ctx.allocator, usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null, data_size: 16)!!;
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dc) {
            vk::cmdCopyBuffer(dc, a.img_hidden.gpu_buffer.buffer, dbgm.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = 16 }});
        }!!;
        float* dp = (float*)dbgm.data();
        io::printfn("    img_hidden[0:4] = %.4f %.4f %.4f %.4f", dp[0], dp[1], dp[2], dp[3]);
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dc2) {
            vk::cmdCopyBuffer(dc2, a.txt_hidden.gpu_buffer.buffer, dbgm.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = 16 }});
        }!!;
        dp = (float*)dbgm.data();
        io::printfn("    txt_hidden[0:4] = %.4f %.4f %.4f %.4f", dp[0], dp[1], dp[2], dp[3]);
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dc3) {
            vk::cmdCopyBuffer(dc3, a.dual_mod_img.gpu_buffer.buffer, dbgm.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = 16 }});
        }!!;
        dp = (float*)dbgm.data();
        io::printfn("    dual_mod_img[0:4] = %.4f %.4f %.4f %.4f", dp[0], dp[1], dp[2], dp[3]);
        // Check gate values (at offset 2*dim=8192)
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dc4) {
            vk::cmdCopyBuffer(dc4, a.dual_mod_img.gpu_buffer.buffer, dbgm.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = (ulong)2 * FLUX_DIM * 4, .dstOffset = 0, .size = 16 }});
        }!!;
        dp = (float*)dbgm.data();
        io::printfn("    dual_gate1_img[0:4] = %.4f %.4f %.4f %.4f", dp[0], dp[1], dp[2], dp[3]);
        // Check t_emb_mlp (SiLU(vec))
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dc5) {
            vk::cmdCopyBuffer(dc5, a.t_emb_mlp.gpu_buffer.buffer, dbgm.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = 16 }});
        }!!;
        dp = (float*)dbgm.data();
        io::printfn("    silu_vec[0:4] = %.4f %.4f %.4f %.4f", dp[0], dp[1], dp[2], dp[3]);
        // Check raw t_emb (sinusoidal)
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dc6) {
            vk::cmdCopyBuffer(dc6, a.t_emb.gpu_buffer.buffer, dbgm.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = 16 }});
        }!!;
        dp = (float*)dbgm.data();
        io::printfn("    t_emb_sincos[0:4] = %.4f %.4f %.4f %.4f", dp[0], dp[1], dp[2], dp[3]);
        // Check silu_t (vec before final SiLU) — should be the time_in MLP output
        ctx.device.@single_time_command(ctx.compute_queue; vk::CommandBuffer dc7) {
            vk::cmdCopyBuffer(dc7, a.silu_t.gpu_buffer.buffer, dbgm.buffer, 1,
                (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = 16 }});
        }!!;
        dp = (float*)dbgm.data();
        io::printfn("    vec (pre-SiLU)[0:4] = %.4f %.4f %.4f %.4f", dp[0], dp[1], dp[2], dp[3]);
        // Print weight dtypes  (0=F32, 1=F16, 8=Q8_0, 30=BF16)
        io::printfn("    weight dtypes: time_w0=%d time_w1=%d mod_img=%d img_in=%d",
            w.time_in_w0.dtype, w.time_in_w1.dtype, w.double_mod_img_w.dtype, w.img_in_w.dtype);
        io::printfn("    block dtypes: dual_qkv=%d dual_proj=%d single_l1=%d single_l2=%d",
            self.dual_blocks[0].img_qkv_w.dtype, self.dual_blocks[0].img_proj_w.dtype,
            self.single_blocks[0].linear1_w.dtype, self.single_blocks[0].linear2_w.dtype);
        dbgm.free();
    }
    g.init(ctx)!!;

    // 6. Dual-stream blocks
    io::printfn("    Running %d dual-stream blocks...", self.n_dual_blocks);
    for (uint l = 0; l < self.n_dual_blocks; l++) {
        flux_dual_stream_block(self, &g, &self.dual_blocks[l])!!;
    }

    // 7. Concatenate streams: hidden = [txt_hidden; img_hidden]
    g.copy(&a.txt_hidden, &a.hidden, (ulong)n_txt * dim * 4);
    copy_to_offset(&g, &a.img_hidden, &a.hidden, n_txt * dim, n_img * dim);

    // 8. Single-stream blocks
    io::printfn("    Running %d single-stream blocks...", self.n_single_blocks);
    for (uint l = 0; l < self.n_single_blocks; l++) {
        flux_single_stream_block(self, &g, &self.single_blocks[l])!!;
    }

    // 9. Final layer: extract img portion, AdaLN, project to patch_dim
    {
        llm::RenderGraph gf;
        gf.init(ctx)!!;

        // Extract img portion from hidden[n_txt*dim:]
        if (gf.needs_barrier) llm::compute_barrier(gf.cmd);
        vk::cmdCopyBuffer(gf.cmd, a.hidden.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = (ulong)n_txt * dim * 4, .dstOffset = 0, .size = (ulong)n_img * dim * 4 }});
        gf.needs_barrier = true;

        // Final AdaLN: SiLU(t_emb) → matmul → [2*dim] → (shift, scale)
        // Use buf_b as temp for the [2*dim] output (mod_shift is only [dim] sized)
        flux_matmul(&gf, dk, &w.final_adaln_w, &a.t_emb_mlp, &a.buf_b, 2 * dim, dim, 1, 0);
        // buf_b[0:dim] = shift, buf_b[dim:2*dim] = scale
        copy_chunk(&gf, &a.buf_b, &a.mod_shift, 0, dim);
        copy_chunk(&gf, &a.buf_b, &a.mod_scale, dim, dim);

        // LayerNorm on img portion
        zimage::BatchLayerNormPC final_ln = { .dim = dim, .eps = 1e-6f, .n_rows = n_img };
        gf.dispatch(&dk.batch_layernorm, { &a.buf_a, &a.buf_b }, &final_ln, n_img);

        // AdaLN: buf_b * (1 + scale) + shift
        zimage::AdalnModPC final_mod = { .n_elements = n_img * dim, .dim = dim };
        gf.dispatch(&dk.adaln_modulate, { &a.buf_b, &a.mod_scale, &a.mod_shift, &a.buf_a },
            &final_mod, llm::ceil_div(n_img * dim, 256));

        // Final projection: [n_img, dim] → [n_img, patch_dim]
        flux_matmul(&gf, dk, &w.final_linear_w, &a.buf_a, &a.velocity, FLUX_PATCH_DIM, dim, n_img, 0);

        // Unpatchify: [n_patches, 128] → [128, h, w]
        zimage::UnpatchifyPC unpatch_pc = {
            .channels = FLUX_LATENT_CH, .height = self.latent_h, .width = self.latent_w, .patch_size = FLUX_PATCH_SIZE,
        };
        uint latent_total = FLUX_LATENT_CH * self.latent_h * self.latent_w;
        gf.dispatch(&dk.unpatchify, { &a.velocity, &a.buf_a }, &unpatch_pc,
            llm::ceil_div(latent_total, 256));

        // Copy unpatchified velocity to velocity buffer
        gf.copy(&a.buf_a, &a.velocity, (ulong)latent_total * 4);

        gf.flush()!!;
    }
}

// --- Free ---

fn void FluxDualBlockWeights.free(&self) {
    self.img_qkv_w.free();
    self.img_proj_w.free();
    self.img_q_norm.free(); self.img_k_norm.free();
    self.txt_qkv_w.free();
    self.txt_proj_w.free();
    self.txt_q_norm.free(); self.txt_k_norm.free();
    self.img_mlp_up_w.free();
    self.img_mlp_down_w.free();
    self.txt_mlp_up_w.free();
    self.txt_mlp_down_w.free();
}

fn void FluxSingleBlockWeights.free(&self) {
    self.linear1_w.free();
    self.linear2_w.free();
    self.q_norm.free(); self.k_norm.free();
}

fn void FluxDiTActivations.free(&self) {
    self.img_hidden.free();
    self.txt_hidden.free();
    self.hidden.free();
    self.buf_a.free();
    self.buf_b.free();
    self.buf_c.free();
    self.q.free();
    self.k.free();
    self.v.free();
    self.attn_out.free();
    self.ffn_gate_out.free();
    self.ffn_up_out.free();
    self.concat_buf.free();
    self.patches.free();
    self.velocity.free();
    self.latent.free();
    self.t_emb.free();
    self.t_emb_mlp.free();
    self.silu_t.free();
    self.dual_mod_img.free();
    self.dual_mod_txt.free();
    self.single_mod_buf.free();
    self.mod_shift.free();
    self.mod_scale.free();
    self.mod_gate.free();
}

fn void FluxDiTModel.free(&self) {
    self.global.img_in_w.free();
    self.global.txt_in_w.free();
    self.global.time_in_w0.free();
    self.global.time_in_w1.free();
    self.global.double_mod_img_w.free();
    self.global.double_mod_txt_w.free();
    self.global.single_mod_w.free();
    self.global.final_linear_w.free();
    self.global.final_adaln_w.free();

    for (uint i = 0; i < self.n_dual_blocks; i++) self.dual_blocks[i].free();
    for (uint i = 0; i < self.n_single_blocks; i++) self.single_blocks[i].free();
    mem::free(self.dual_blocks);
    mem::free(self.single_blocks);

    if (self.rope_cos.size_bytes > 0) self.rope_cos.free();
    if (self.rope_sin.size_bytes > 0) self.rope_sin.free();

    self.acts.free();
}
