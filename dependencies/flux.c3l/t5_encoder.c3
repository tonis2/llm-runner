module flux;

import vk;
import llm;
import llm::pipelines;
import std::io;
import std::math;
import std::core::mem;

// T5-XXL Encoder for Flux text conditioning
// 24-layer encoder-only transformer with relative position bias
//
// Architecture: dim=4096, heads=64, head_dim=64, FFN=10240 (GEGLU)
// Uses RMSNorm, no absolute position embeddings
// Relative position bias shared across all layers

const uint T5_DIM       = 4096;
const uint T5_HEADS     = 64;
const uint T5_HEAD_DIM  = 64;
const uint T5_FFN_DIM   = 10240;
const uint T5_LAYERS    = 24;
const uint T5_MAX_SEQ   = 512;
const float T5_EPS      = 1e-6f;

// Relative position bias constants
const uint T5_REL_BUCKETS = 32;
const uint T5_REL_MAX_DIST = 128;

struct T5AttentionPC {
    uint head_dim;
    uint n_heads;
    uint seq_len;
    float scale;
}

struct T5LayerWeights {
    llm::Tensor attn_norm;       // [dim]
    llm::Tensor wq;              // [n_heads*head_dim, dim]
    llm::Tensor wk;              // [n_heads*head_dim, dim]
    llm::Tensor wv;              // [n_heads*head_dim, dim]
    llm::Tensor wo;              // [dim, n_heads*head_dim]
    llm::Tensor ffn_norm;        // [dim]
    llm::Tensor ffn_gate;        // [ffn_dim, dim]
    llm::Tensor ffn_up;          // [ffn_dim, dim]
    llm::Tensor ffn_down;        // [dim, ffn_dim]
}

struct T5Weights {
    llm::Tensor token_embedding;    // [vocab_size, dim]
    llm::Tensor output_norm;        // [dim]
    llm::Tensor rel_pos_bias;       // [n_heads, seq_len, seq_len] precomputed
    T5LayerWeights[] layers;
    // CPU-side embedding for per-token dequant
    char* embedding_data;
    llm::GGUFTensorInfo embedding_info;
}

struct T5Activations {
    llm::Tensor hidden;         // [seq, dim]
    llm::Tensor norm_out;       // [seq, dim]
    llm::Tensor q;              // [n_heads, seq, head_dim]
    llm::Tensor k;              // [n_heads, seq, head_dim]
    llm::Tensor v;              // [n_heads, seq, head_dim]
    llm::Tensor attn_out;       // [n_heads, seq, head_dim]
    llm::Tensor attn_scores;    // [n_heads, seq, seq]
    llm::Tensor ffn_gate_up;    // [seq, 2*ffn_dim]
    llm::Tensor ffn_out;        // [seq, ffn_dim]
    llm::Tensor ffn_down_out;   // [seq, dim]
}

struct T5Kernels {
    llm::ComputeKernel spatial_attention;  // Non-causal attention (bias added separately later)
    llm::ComputeKernel broadcast_add;      // For adding bias to scores
    llm::SharedKernels shared;
}

struct FluxT5Encoder {
    T5Weights weights;
    T5Activations acts;
    T5Kernels kernels;
    FluxKernels* flux_kernels;  // For batched matmul
    llm::DeviceContext* ctx;
    llm::GGUFFile* gf;         // Keep reference for streaming
    uint seq_len;               // Actual sequence length for current input
}

// --- T5 Relative Position Bias ---

// Compute the bucket index for a relative position, following T5's scheme:
//   - Bidirectional: first half positive, second half negative
//   - Each half: first n/2 linear, rest logarithmic up to max_distance
fn uint t5_relative_position_bucket(int relative_position) {
    uint n_buckets = T5_REL_BUCKETS;
    uint max_distance = T5_REL_MAX_DIST;

    int ret = 0;
    int n = -relative_position;

    // Bidirectional: use half the buckets for each direction
    uint half_buckets = n_buckets / 2;
    ret += (n < 0) ? (int)half_buckets : 0;
    n = n < 0 ? -n : n;  // abs(n)

    // First half: linear (1 bucket per distance)
    uint max_exact = half_buckets / 2;
    bool is_small = (uint)n < max_exact;

    // Logarithmic bucketing for larger distances
    uint val_if_large;
    if ((uint)n >= max_distance) {
        val_if_large = half_buckets - 1;
    } else {
        float log_ratio = math::ln((float)n / (float)max_exact) / math::ln((float)max_distance / (float)max_exact);
        val_if_large = (uint)((float)max_exact + log_ratio * (float)(half_buckets - max_exact));
    }

    ret += is_small ? n : (int)val_if_large;
    return (uint)ret;
}

// Precompute the full [n_heads, seq_len, seq_len] bias table on CPU
fn llm::Tensor? precompute_t5_position_bias(
    llm::DeviceContext* ctx,
    llm::GGUFFile* gf,
    uint seq_len
) {
    io::printfn("Precomputing T5 relative position bias [%d, %d, %d]...", T5_HEADS, seq_len, seq_len);

    // Load the learned bias table from GGUF (stored at layer 0, shared across layers)
    llm::GGUFTensorInfo* bias_info = gf.find_tensor("enc.blk.0.attn_rel_b.weight")!!;
    char* bias_data = gf.tensor_data_base + (usz)bias_info.offset;

    // GGUF shape [64, 32]: shape[0]=n_heads (inner dim), shape[1]=n_buckets (outer dim)
    // Memory layout: [n_buckets, n_heads] = data[bucket * n_heads + head]
    uint n_heads = (uint)bias_info.shape[0];
    uint n_buckets = (uint)bias_info.shape[1];

    io::printfn("  Bias table: [%d buckets, %d heads]", n_buckets, n_heads);

    // Dequantize bias table to float
    usz table_elements = (usz)n_buckets * n_heads;
    llm::CpuTensorData cpu_data = llm::dequant_to_cpu(bias_info, gf.tensor_data_base)!!;
    float* table_f32 = cpu_data.data;

    // Build full bias: [n_heads, seq_len, seq_len]
    usz bias_size = (usz)n_heads * seq_len * seq_len;
    float* full_bias = (float*)mem::calloc(bias_size * 4);

    for (uint h = 0; h < n_heads; h++) {
        for (uint i = 0; i < seq_len; i++) {
            for (uint j = 0; j < seq_len; j++) {
                int rel_pos = (int)j - (int)i;
                uint bucket = t5_relative_position_bucket(rel_pos);
                if (bucket >= n_buckets) bucket = n_buckets - 1;

                // Table layout: [n_buckets, n_heads] -> index = bucket * n_heads + h
                float bias_val = table_f32[(usz)bucket * n_heads + h];
                full_bias[(usz)h * seq_len * seq_len + (usz)i * seq_len + j] = bias_val;
            }
        }
    }

    mem::free(cpu_data.data);

    // Upload to GPU
    vk::Memory gpu_buffer = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: full_bias,
        data_size: bias_size * 4,
    )!!;

    mem::free(full_bias);

    ulong[4] shape = { (ulong)(n_heads * seq_len * seq_len), 0, 0, 0 };
    return {
        .gpu_buffer = gpu_buffer,
        .dtype = llm::GGML_F32,
        .n_dims = 1,
        .shape = shape,
        .size_bytes = bias_size * 4,
    };
}

// --- Load T5 Model ---

fn usz t5_layer_name(char[64]* buf, uint layer, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    String prefix = "enc.blk.";
    for (usz i = 0; i < prefix.len; i++) p[pos + i] = prefix[i];
    pos += prefix.len;
    if (layer >= 10) { p[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    p[pos] = (char)('0' + layer % 10); pos++;
    p[pos] = '.'; pos++;
    for (usz i = 0; i < suffix.len; i++) p[pos + i] = suffix[i];
    pos += suffix.len;
    return pos;
}

fn llm::Tensor load_t5_layer_tensor(llm::DeviceContext* ctx, llm::GGUFFile* gf, uint layer, String suffix) {
    char[64] buf;
    usz len = t5_layer_name(&buf, layer, suffix);
    String name = (String)buf[0..len - 1];
    llm::GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return llm::upload_weight(ctx, info, gf.tensor_data_base)!!;
}

fn T5Kernels? create_t5_kernels(llm::DeviceContext* ctx) {
    // Use the shared diffusion shader for T5 operations
    char[] spv = &pipelines::DIFFUSION_SPV;
    vk::ShaderModule shader = vk::shaderModuleCreateInfo()
        .setCodeSize(spv.len)
        .setCode((uint*)&spv[0])
        .build(ctx.device)!!;

    T5Kernels kernels = {
        .spatial_attention = llm::create_kernel(ctx, shader, 5, SpatialAttentionPC.sizeof, "spatial_attention")!!,
        .broadcast_add = llm::create_kernel(ctx, shader, 2, BroadcastAddPC.sizeof, "broadcast_add")!!,
        .shared = llm::create_shared_kernels(ctx, shader)!!,
    };

    shader.free(ctx.device);
    return kernels;
}

fn T5Activations? allocate_t5_activations(llm::DeviceContext* ctx, uint seq_len) {
    ulong seq = seq_len;
    ulong dim = T5_DIM;
    ulong ffn = T5_FFN_DIM;
    ulong heads = T5_HEADS;

    ulong[4] hidden_shape = { seq * dim, 0, 0, 0 };
    ulong[4] qkv_shape = { heads * seq * T5_HEAD_DIM, 0, 0, 0 };
    ulong[4] scores_shape = { heads * seq * seq, 0, 0, 0 };
    ulong[4] ffn_gate_up_shape = { seq * ffn * 2, 0, 0, 0 };
    ulong[4] ffn_shape = { seq * ffn, 0, 0, 0 };

    return {
        .hidden = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .norm_out = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .q = llm::create_f32_tensor(ctx, qkv_shape, 1)!!,
        .k = llm::create_f32_tensor(ctx, qkv_shape, 1)!!,
        .v = llm::create_f32_tensor(ctx, qkv_shape, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, qkv_shape, 1)!!,
        .attn_scores = llm::create_f32_tensor(ctx, scores_shape, 1)!!,
        .ffn_gate_up = llm::create_f32_tensor(ctx, ffn_gate_up_shape, 1)!!,
        .ffn_out = llm::create_f32_tensor(ctx, ffn_shape, 1)!!,
        .ffn_down_out = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
    };
}

fn FluxT5Encoder? load_t5_encoder(
    llm::DeviceContext* ctx,
    llm::GGUFFile* gf,
    FluxKernels* flux_kernels,
    uint seq_len
) {
    io::printfn("\nLoading T5-XXL encoder (%d layers, dim=%d)...", T5_LAYERS, T5_DIM);

    if (seq_len > T5_MAX_SEQ) seq_len = T5_MAX_SEQ;

    // Create T5-specific kernels
    io::printfn("  Creating T5 kernels...");
    T5Kernels t5k = create_t5_kernels(ctx)!!;

    // Precompute relative position bias
    io::printfn("  Precomputing relative position bias...");
    llm::Tensor rel_bias = precompute_t5_position_bias(ctx, gf, seq_len)!!;

    // Load embedding (keep CPU reference for per-token dequant)
    io::printfn("  Loading embedding info...");
    llm::GGUFTensorInfo* emb_info = gf.find_tensor("token_embd.weight")!!;
    char* emb_data = gf.tensor_data_base + (usz)emb_info.offset;

    // Load output norm
    io::printfn("  Loading output norm...");
    llm::GGUFTensorInfo* norm_info = gf.find_tensor("enc.output_norm.weight")!!;
    llm::Tensor output_norm = llm::upload_weight(ctx, norm_info, gf.tensor_data_base)!!;

    // Load all layers
    T5LayerWeights[] layers = mem::new_array(T5LayerWeights, T5_LAYERS);
    for (uint l = 0; l < T5_LAYERS; l++) {
        layers[l] = {
            .attn_norm = load_t5_layer_tensor(ctx, gf, l, "attn_norm.weight"),
            .wq = load_t5_layer_tensor(ctx, gf, l, "attn_q.weight"),
            .wk = load_t5_layer_tensor(ctx, gf, l, "attn_k.weight"),
            .wv = load_t5_layer_tensor(ctx, gf, l, "attn_v.weight"),
            .wo = load_t5_layer_tensor(ctx, gf, l, "attn_o.weight"),
            .ffn_norm = load_t5_layer_tensor(ctx, gf, l, "ffn_norm.weight"),
            .ffn_gate = load_t5_layer_tensor(ctx, gf, l, "ffn_gate.weight"),
            .ffn_up = load_t5_layer_tensor(ctx, gf, l, "ffn_up.weight"),
            .ffn_down = load_t5_layer_tensor(ctx, gf, l, "ffn_down.weight"),
        };
        if ((l + 1) % 6 == 0 || l == T5_LAYERS - 1) {
            io::printfn("  T5 layer %d / %d loaded", l + 1, T5_LAYERS);
        }
    }

    T5Weights weights = {
        .token_embedding = {},
        .output_norm = output_norm,
        .rel_pos_bias = rel_bias,
        .layers = layers,
        .embedding_data = emb_data,
        .embedding_info = *emb_info,
    };

    T5Activations acts = allocate_t5_activations(ctx, seq_len)!!;

    return {
        .weights = weights,
        .acts = acts,
        .kernels = t5k,
        .flux_kernels = flux_kernels,
        .ctx = ctx,
        .gf = gf,
        .seq_len = seq_len,
    };
}

// T5 forward pass: tokens[n_tokens] -> hidden[seq_len, 4096]
// Result is in self.acts.hidden after this call
fn void? FluxT5Encoder.forward(&self, uint[] tokens, uint n_tokens) {
    llm::DeviceContext* ctx = self.ctx;
    vk::CommandBuffer cmd = ctx.command_buffer;
    T5Weights* w = &self.weights;
    T5Activations* a = &self.acts;
    T5Kernels* tk = &self.kernels;
    FluxKernels* fk = self.flux_kernels;
    llm::SharedKernels* sk = &tk.shared;

    uint seq = self.seq_len;
    uint dim = T5_DIM;
    uint qkv_dim = T5_HEADS * T5_HEAD_DIM;

    io::printfn("T5 forward: %d tokens, seq_len=%d", n_tokens, seq);

    // 1. Token embedding: dequantize all rows on CPU and bulk upload
    // T5 has no position embeddings — positions come from relative bias
    {
        usz total_bytes = (usz)seq * dim * 4;
        float* emb_buf = (float*)mem::calloc(total_bytes);

        for (uint pos = 0; pos < seq; pos++) {
            uint token_id = pos < n_tokens ? tokens[pos] : 0;
            llm::dequant_embedding_row(w.embedding_data, &w.embedding_info, token_id,
                emb_buf + (usz)pos * dim, dim);
        }

        // Upload entire embedding sequence to GPU
        vk::Memory staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: emb_buf,
            data_size: total_bytes,
        )!!;

        llm::begin_compute(cmd)!!;
        vk::cmdCopyBuffer(cmd, staging.buffer, a.hidden.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = total_bytes }});
        llm::submit_and_wait(ctx)!!;
        staging.free();
        mem::free(emb_buf);
    }

    // 2. Transformer layers
    for (uint l = 0; l < T5_LAYERS; l++) {
        T5LayerWeights* lw = &w.layers[l];

        llm::begin_compute(cmd)!!;

        // --- Self-Attention ---
        // RMSNorm(hidden) -> norm_out  (per position)
        for (uint pos = 0; pos < seq; pos++) {
            llm::RMSNormPC rms_pc = { .dim = dim, .eps = T5_EPS };
            llm::dispatch_kernel(cmd, &sk.rmsnorm,
                { a.hidden.gpu_buffer.buffer, lw.attn_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.attn_norm.size_bytes, a.norm_out.size_bytes },
                &rms_pc, 1);
        }
        llm::compute_barrier(cmd);

        // Q/K/V projections (per position): weight [qkv_dim, dim] x input [dim] -> [qkv_dim]
        t5_dispatch_matmul_seq(cmd, sk, &lw.wq, &a.norm_out, &a.q, qkv_dim, dim, seq);
        t5_dispatch_matmul_seq(cmd, sk, &lw.wk, &a.norm_out, &a.k, qkv_dim, dim, seq);
        t5_dispatch_matmul_seq(cmd, sk, &lw.wv, &a.norm_out, &a.v, qkv_dim, dim, seq);
        llm::compute_barrier(cmd);

        // T5 attention (using spatial_attention without bias for now)
        // TODO: Add relative position bias before softmax for correct T5 behavior
        float scale = 1.0f;  // T5 does not scale by 1/sqrt(d_k)
        SpatialAttentionPC attn_pc = {
            .head_dim = T5_HEAD_DIM,
            .n_heads = T5_HEADS,
            .seq_len = seq,
            .scale = scale,
        };
        llm::dispatch_kernel(cmd, &tk.spatial_attention,
            { a.q.gpu_buffer.buffer, a.k.gpu_buffer.buffer, a.v.gpu_buffer.buffer,
              a.attn_scores.gpu_buffer.buffer, a.attn_out.gpu_buffer.buffer },
            { a.q.size_bytes, a.k.size_bytes, a.v.size_bytes,
              a.attn_scores.size_bytes, a.attn_out.size_bytes },
            &attn_pc, T5_HEADS * seq);
        llm::compute_barrier(cmd);

        // Output projection (per position): [qkv_dim] -> [dim]
        t5_dispatch_matmul_seq(cmd, sk, &lw.wo, &a.attn_out, &a.norm_out, dim, qkv_dim, seq);
        llm::compute_barrier(cmd);

        // Residual: hidden += attn_output
        llm::ResidualPC res_pc = { .n = seq * dim };
        llm::dispatch_kernel(cmd, &sk.residual_add,
            { a.hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.norm_out.size_bytes },
            &res_pc, llm::ceil_div(seq * dim, 256));
        llm::compute_barrier(cmd);

        // --- FFN (GEGLU) ---
        // RMSNorm(hidden) -> norm_out
        for (uint pos = 0; pos < seq; pos++) {
            llm::RMSNormPC rms_pc2 = { .dim = dim, .eps = T5_EPS };
            llm::dispatch_kernel(cmd, &sk.rmsnorm,
                { a.hidden.gpu_buffer.buffer, lw.ffn_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.ffn_norm.size_bytes, a.norm_out.size_bytes },
                &rms_pc2, 1);
        }
        llm::compute_barrier(cmd);

        // Gate: [seq, dim] -> [seq, ffn_dim] -> first half of ffn_gate_up
        t5_dispatch_matmul_seq(cmd, sk, &lw.ffn_gate, &a.norm_out, &a.ffn_out, T5_FFN_DIM, dim, seq);
        llm::compute_barrier(cmd);

        usz gate_bytes = (usz)seq * T5_FFN_DIM * 4;
        vk::cmdCopyBuffer(cmd, a.ffn_out.gpu_buffer.buffer, a.ffn_gate_up.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = gate_bytes }});
        llm::compute_barrier(cmd);

        // Up: [seq, dim] -> [seq, ffn_dim] -> second half of ffn_gate_up
        t5_dispatch_matmul_seq(cmd, sk, &lw.ffn_up, &a.norm_out, &a.ffn_out, T5_FFN_DIM, dim, seq);
        llm::compute_barrier(cmd);

        vk::cmdCopyBuffer(cmd, a.ffn_out.gpu_buffer.buffer, a.ffn_gate_up.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = gate_bytes, .size = gate_bytes }});
        llm::compute_barrier(cmd);

        // GEGLU: output = GELU(gate) * up
        // gate is in ffn_gate_up[0..gate_bytes], up is in ffn_out
        // Apply GELU to gate in-place (first half of ffn_gate_up)
        uint geglu_n = seq * T5_FFN_DIM;
        llm::GeluPC gelu_pc = { .n = geglu_n };
        llm::dispatch_kernel(cmd, &sk.gelu,
            { a.ffn_gate_up.gpu_buffer.buffer },
            { a.ffn_gate_up.size_bytes },
            &gelu_pc, llm::ceil_div(geglu_n, 256));
        llm::compute_barrier(cmd);

        // Multiply: ffn_out = GELU(gate) * up (element-wise)
        llm::ElemwisePC emul_pc = { .n = geglu_n };
        llm::dispatch_kernel(cmd, &sk.elemwise_mul,
            { a.ffn_out.gpu_buffer.buffer, a.ffn_gate_up.gpu_buffer.buffer },
            { a.ffn_out.size_bytes, a.ffn_gate_up.size_bytes },
            &emul_pc, llm::ceil_div(geglu_n, 256));
        llm::compute_barrier(cmd);

        // Down: [seq, ffn_dim] -> [seq, dim]
        t5_dispatch_matmul_seq(cmd, sk, &lw.ffn_down, &a.ffn_out, &a.ffn_down_out, dim, T5_FFN_DIM, seq);
        llm::compute_barrier(cmd);

        // Residual: hidden += ffn_output
        llm::dispatch_kernel(cmd, &sk.residual_add,
            { a.hidden.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.ffn_down_out.size_bytes },
            &res_pc, llm::ceil_div(seq * dim, 256));

        // Submit each layer to avoid GPU timeout
        llm::submit_and_wait(ctx)!!;
    }

    // 3. Final RMSNorm
    llm::begin_compute(cmd)!!;
    for (uint pos = 0; pos < seq; pos++) {
        llm::RMSNormPC final_rms = { .dim = dim, .eps = T5_EPS };
        llm::dispatch_kernel(cmd, &sk.rmsnorm,
            { a.hidden.gpu_buffer.buffer, w.output_norm.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
            { a.hidden.size_bytes, w.output_norm.size_bytes, a.hidden.size_bytes },
            &final_rms, 1);
    }

    llm::submit_and_wait(ctx)!!;
    io::printfn("  T5 encoding complete");
}

// --- Dispatch Helper ---
// Per-position matmul for quantized weights over sequence
// Each position: weight [out_dim, in_dim] x input[pos, in_dim] -> output[pos, out_dim]
fn void t5_dispatch_matmul_seq(
    vk::CommandBuffer cmd,
    llm::SharedKernels* sk,
    llm::Tensor* weight,
    llm::Tensor* input,
    llm::Tensor* output,
    uint out_dim,
    uint in_dim,
    uint seq_len
) {
    llm::MatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim };
    llm::ComputeKernel* kernel;
    if (weight.dtype == llm::GGML_Q8_0) {
        kernel = &sk.matmul_q8;
    } else if (weight.dtype == llm::GGML_Q4_0) {
        kernel = &sk.matmul_q4_0;
    } else if (weight.dtype == llm::GGML_Q4_K) {
        kernel = &sk.matmul_q4k;
    } else if (weight.dtype == llm::GGML_Q5_K) {
        kernel = &sk.matmul_q5k;
    } else if (weight.dtype == llm::GGML_Q6_K) {
        kernel = &sk.matmul_q6k;
    } else {
        kernel = &sk.matmul;
    }

    // The shared matmul kernel operates on [out_dim, in_dim] x [in_dim] -> [out_dim]
    // with one workgroup per output element. We dispatch for each position,
    // using buffer offsets implicit in the shader (it reads from binding start).
    // Since we can't easily offset bindings, we dispatch all positions.
    // The kernel reads input[0..in_dim-1] and writes output[0..out_dim-1].
    // For sequence processing, the LLM encode_text handles this with a position loop.
    // Here we just dispatch seq_len times with the kernel.
    for (uint pos = 0; pos < seq_len; pos++) {
        llm::dispatch_kernel(cmd, kernel,
            { weight.gpu_buffer.buffer, input.gpu_buffer.buffer, output.gpu_buffer.buffer },
            { weight.size_bytes, input.size_bytes, output.size_bytes },
            &pc, out_dim);
    }
}

fn void FluxT5Encoder.free(&self) {
    // Free weights
    if (self.weights.output_norm.size_bytes > 0) self.weights.output_norm.free();
    if (self.weights.rel_pos_bias.size_bytes > 0) self.weights.rel_pos_bias.free();
    for (uint l = 0; l < T5_LAYERS; l++) {
        T5LayerWeights* lw = &self.weights.layers[l];
        lw.attn_norm.free();
        lw.wq.free(); lw.wk.free(); lw.wv.free(); lw.wo.free();
        lw.ffn_norm.free();
        lw.ffn_gate.free(); lw.ffn_up.free(); lw.ffn_down.free();
    }
    mem::free(self.weights.layers);

    // Free activations
    self.acts.hidden.free();
    self.acts.norm_out.free();
    self.acts.q.free(); self.acts.k.free(); self.acts.v.free();
    self.acts.attn_out.free();
    self.acts.attn_scores.free();
    self.acts.ffn_gate_up.free();
    self.acts.ffn_out.free();
    self.acts.ffn_down_out.free();

    // Free T5 kernels
    self.kernels.spatial_attention.free(self.ctx.device);
    self.kernels.broadcast_add.free(self.ctx.device);
    self.kernels.shared.free(self.ctx.device);
}
