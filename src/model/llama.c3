module llm;

import vk;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::math;
import std::core::mem;

const MAX_SEQ_LEN = 2048;

// Embedded SPIR-V shaders
const char[*] EMBEDDING_SPV   = $embed("../../shaders/embedding.spv");
const char[*] RMSNORM_SPV     = $embed("../../shaders/rmsnorm.spv");
const char[*] SILU_SPV        = $embed("../../shaders/silu.spv");
const char[*] MATMUL_SPV      = $embed("../../shaders/matmul.spv");
const char[*] MATMUL_Q8_SPV   = $embed("../../shaders/matmul_q8.spv");
const char[*] MATMUL_Q4K_SPV  = $embed("../../shaders/matmul_q4k.spv");
const char[*] MATMUL_Q5K_SPV  = $embed("../../shaders/matmul_q5k.spv");
const char[*] MATMUL_Q6K_SPV  = $embed("../../shaders/matmul_q6k.spv");
const char[*] ROPE_SPV        = $embed("../../shaders/rope.spv");
const char[*] SOFTMAX_SPV     = $embed("../../shaders/softmax.spv");
const char[*] ATTENTION_SPV   = $embed("../../shaders/attention.spv");
const char[*] RESIDUAL_SPV    = $embed("../../shaders/residual_add.spv");
const char[*] ELEMWISE_SPV    = $embed("../../shaders/elemwise_mul.spv");

struct LlamaConfig {
    uint n_layers;
    uint n_heads;
    uint n_kv_heads;
    uint dim;
    uint ffn_dim;
    uint vocab_size;
    uint head_dim;
    float rope_theta;
    float rms_eps;
}

struct LlamaLayerWeights {
    Tensor attn_norm;
    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor wo;
    Tensor ffn_norm;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
}

struct LlamaWeights {
    Tensor token_embedding;
    Tensor output_norm;
    Tensor output;
    LlamaLayerWeights[] layers;
}

struct LlamaActivations {
    Tensor hidden;
    Tensor residual;
    Tensor norm_out;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor attn_out;
    Tensor ffn_gate_out;
    Tensor ffn_up_out;
    Tensor ffn_down_out;
    Tensor logits;
    Tensor attn_scores;
}

struct LlamaKVCache {
    Tensor[] k_cache;
    Tensor[] v_cache;
}

struct LlamaKernels {
    ComputeKernel embedding;
    ComputeKernel rmsnorm;
    ComputeKernel silu;
    ComputeKernel matmul;
    ComputeKernel matmul_q8;
    ComputeKernel matmul_q4k;
    ComputeKernel matmul_q5k;
    ComputeKernel matmul_q6k;
    ComputeKernel rope;
    ComputeKernel softmax;
    ComputeKernel attention;
    ComputeKernel residual_add;
    ComputeKernel elemwise_mul;
}

struct LlamaModel {
    LlamaConfig config;
    LlamaWeights weights;
    LlamaActivations acts;
    LlamaKVCache kv;
    LlamaKernels kernels;
    DeviceContext* ctx;
}

struct EmbeddingPC { uint token_id; uint dim; }
struct RMSNormPC { uint dim; float eps; }
struct SiluPC { uint n; }
struct MatMulPC { uint out_dim; uint in_dim; }
struct RoPEPC { uint dim; uint n_heads; uint position; float theta; }
struct SoftmaxPC { uint n; uint offset; }
struct AttentionPC { uint head_dim; uint n_kv_heads; uint n_q_heads; uint seq_len; float scale; uint max_seq_len; }
struct ResidualPC { uint n; }
struct ElemwisePC { uint n; }

fn LlamaConfig extract_config(GGUFFile* gf) {
    uint dim = gf.get_u32("llama.embedding_length") ?? 2048;
    uint n_heads = gf.get_u32("llama.attention.head_count") ?? 32;
    uint n_kv_heads = gf.get_u32("llama.attention.head_count_kv") ?? 4;
    uint n_layers = gf.get_u32("llama.block_count") ?? 22;
    uint ffn_dim = gf.get_u32("llama.feed_forward_length") ?? 5632;
    uint vocab_size = 0;

    // Vocab size comes from tokenizer.ggml.tokens array or the embedding tensor
    GGUFTensorInfo*? emb = gf.find_tensor("token_embd.weight");
    if (try emb) {
        vocab_size = (uint)emb.shape[1];
    }

    float rope_theta = gf.get_f32("llama.rope.freq_base") ?? 10000.0;
    float rms_eps = gf.get_f32("llama.attention.layer_norm_rms_epsilon") ?? 1e-5;

    LlamaConfig config = {
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .dim = dim,
        .ffn_dim = ffn_dim,
        .vocab_size = vocab_size,
        .head_dim = dim / n_heads,
        .rope_theta = rope_theta,
        .rms_eps = rms_eps,
    };

    io::printfn("\n=== LLaMA Config ===");
    io::printfn("  dim: %d", config.dim);
    io::printfn("  n_layers: %d", config.n_layers);
    io::printfn("  n_heads: %d", config.n_heads);
    io::printfn("  n_kv_heads: %d", config.n_kv_heads);
    io::printfn("  ffn_dim: %d", config.ffn_dim);
    io::printfn("  vocab_size: %d", config.vocab_size);
    io::printfn("  head_dim: %d", config.head_dim);
    io::printfn("  rope_theta: %f", config.rope_theta);
    io::printfn("  rms_eps: %f", config.rms_eps);

    return config;
}

fn Tensor load_tensor_by_name(DeviceContext* ctx, GGUFFile* gf, String name) {
    GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return upload_weight(ctx, info, gf.tensor_data_base)!!;
}

fn Tensor load_tensor_as_f32(DeviceContext* ctx, GGUFFile* gf, String name) {
    GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
}

fn LlamaWeights? load_weights(DeviceContext* ctx, GGUFFile* gf, LlamaConfig* config) {
    io::printfn("\nLoading weights...");

    // Embedding must be F32 for the embedding lookup shader
    Tensor token_embedding = load_tensor_as_f32(ctx, gf, "token_embd.weight");
    Tensor output_norm = load_tensor_by_name(ctx, gf, "output_norm.weight");

    // Output weight - may be same as embedding (tied weights) or separate
    Tensor output_weight;
    GGUFTensorInfo*? output_info = gf.find_tensor("output.weight");
    if (try output_info) {
        output_weight = upload_weight(ctx, output_info, gf.tensor_data_base)!!;
    } else {
        // Tied weights - use embedding
        output_weight = token_embedding;
    }

    LlamaLayerWeights[] layers = mem::new_array(LlamaLayerWeights, config.n_layers);

    for (uint l = 0; l < config.n_layers; l++) {
        // Format: blk.{l}.attn_norm.weight, etc.
        // We'll build the name string manually
        layers[l] = {
            .attn_norm = load_layer_tensor(ctx, gf, l, "attn_norm.weight"),
            .wq = load_layer_tensor(ctx, gf, l, "attn_q.weight"),
            .wk = load_layer_tensor(ctx, gf, l, "attn_k.weight"),
            .wv = load_layer_tensor(ctx, gf, l, "attn_v.weight"),
            .wo = load_layer_tensor(ctx, gf, l, "attn_output.weight"),
            .ffn_norm = load_layer_tensor(ctx, gf, l, "ffn_norm.weight"),
            .ffn_gate = load_layer_tensor(ctx, gf, l, "ffn_gate.weight"),
            .ffn_up = load_layer_tensor(ctx, gf, l, "ffn_up.weight"),
            .ffn_down = load_layer_tensor(ctx, gf, l, "ffn_down.weight"),
        };

        if ((l + 1) % 4 == 0 || l == config.n_layers - 1) {
            io::printfn("  Loaded layer %d / %d", l + 1, config.n_layers);
        }
    }

    return {
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output = output_weight,
        .layers = layers,
    };
}

fn Tensor load_layer_tensor(DeviceContext* ctx, GGUFFile* gf, uint layer, String suffix) {
    // Build "blk.{layer}.{suffix}" name
    // Layer numbers are small (< 100), so a small buffer suffices
    char[64] buf;
    usz len = format_layer_name(&buf, layer, suffix);
    String name = (String)buf[0..len - 1];

    GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return upload_weight(ctx, info, gf.tensor_data_base)!!;
}

fn usz format_layer_name(char[64]* buf, uint layer, String suffix) {
    // "blk." = 4 chars
    char* p = &(*buf)[0];
    p[0] = 'b'; p[1] = 'l'; p[2] = 'k'; p[3] = '.';
    usz pos = 4;

    // Write layer number
    if (layer >= 10) {
        p[pos] = (char)('0' + layer / 10);
        pos++;
    }
    p[pos] = (char)('0' + layer % 10);
    pos++;

    // Write "."
    p[pos] = '.';
    pos++;

    // Write suffix
    for (usz i = 0; i < suffix.len; i++) {
        p[pos + i] = suffix[i];
    }
    pos += suffix.len;

    return pos;
}

fn LlamaActivations? allocate_activations(DeviceContext* ctx, LlamaConfig* config) {
    io::printfn("Allocating activation buffers...");

    ulong dim = config.dim;
    ulong ffn_dim = config.ffn_dim;
    ulong vocab = config.vocab_size;
    ulong n_heads = config.n_heads;

    ulong[4] dim_shape = { dim, 0, 0, 0 };
    ulong[4] ffn_shape = { ffn_dim, 0, 0, 0 };
    ulong[4] vocab_shape = { vocab, 0, 0, 0 };
    ulong[4] qkv_shape = { dim, 0, 0, 0 };  // n_heads * head_dim = dim
    ulong[4] scores_shape = { n_heads * MAX_SEQ_LEN, 0, 0, 0 };

    return {
        .hidden = create_f32_tensor(ctx, dim_shape, 1)!!,
        .residual = create_f32_tensor(ctx, dim_shape, 1)!!,
        .norm_out = create_f32_tensor(ctx, dim_shape, 1)!!,
        .q = create_f32_tensor(ctx, qkv_shape, 1)!!,
        .k = create_f32_tensor(ctx, { (ulong)config.n_kv_heads * config.head_dim, 0, 0, 0 }, 1)!!,
        .v = create_f32_tensor(ctx, { (ulong)config.n_kv_heads * config.head_dim, 0, 0, 0 }, 1)!!,
        .attn_out = create_f32_tensor(ctx, dim_shape, 1)!!,
        .ffn_gate_out = create_f32_tensor(ctx, ffn_shape, 1)!!,
        .ffn_up_out = create_f32_tensor(ctx, ffn_shape, 1)!!,
        .ffn_down_out = create_f32_tensor(ctx, dim_shape, 1)!!,
        .logits = create_f32_tensor(ctx, vocab_shape, 1)!!,
        .attn_scores = create_f32_tensor(ctx, scores_shape, 1)!!,
    };
}

fn LlamaKVCache? allocate_kv_cache(DeviceContext* ctx, LlamaConfig* config) {
    io::printfn("Allocating KV cache...");

    Tensor[] k_cache = mem::new_array(Tensor, config.n_layers);
    Tensor[] v_cache = mem::new_array(Tensor, config.n_layers);

    ulong cache_size = (ulong)config.n_kv_heads * MAX_SEQ_LEN * config.head_dim;
    ulong[4] cache_shape = { cache_size, 0, 0, 0 };

    for (uint l = 0; l < config.n_layers; l++) {
        k_cache[l] = create_f32_tensor(ctx, cache_shape, 1)!!;
        v_cache[l] = create_f32_tensor(ctx, cache_shape, 1)!!;
    }

    return {
        .k_cache = k_cache,
        .v_cache = v_cache,
    };
}

fn LlamaKernels? create_kernels(DeviceContext* ctx) {
    io::printfn("Creating compute kernels...");
    return {
        .embedding   = create_kernel(ctx, &EMBEDDING_SPV, 2, EmbeddingPC.sizeof)!!,
        .rmsnorm     = create_kernel(ctx, &RMSNORM_SPV, 3, RMSNormPC.sizeof)!!,
        .silu        = create_kernel(ctx, &SILU_SPV, 1, SiluPC.sizeof)!!,
        .matmul      = create_kernel(ctx, &MATMUL_SPV, 3, MatMulPC.sizeof)!!,
        .matmul_q8   = create_kernel(ctx, &MATMUL_Q8_SPV, 3, MatMulPC.sizeof)!!,
        .matmul_q4k  = create_kernel(ctx, &MATMUL_Q4K_SPV, 3, MatMulPC.sizeof)!!,
        .matmul_q5k  = create_kernel(ctx, &MATMUL_Q5K_SPV, 3, MatMulPC.sizeof)!!,
        .matmul_q6k  = create_kernel(ctx, &MATMUL_Q6K_SPV, 3, MatMulPC.sizeof)!!,
        .rope        = create_kernel(ctx, &ROPE_SPV, 1, RoPEPC.sizeof)!!,
        .softmax     = create_kernel(ctx, &SOFTMAX_SPV, 1, SoftmaxPC.sizeof)!!,
        .attention   = create_kernel(ctx, &ATTENTION_SPV, 5, AttentionPC.sizeof)!!,
        .residual_add = create_kernel(ctx, &RESIDUAL_SPV, 2, ResidualPC.sizeof)!!,
        .elemwise_mul = create_kernel(ctx, &ELEMWISE_SPV, 2, ElemwisePC.sizeof)!!,
    };
}

fn uint ceil_div(uint a, uint b) {
    return (a + b - 1) / b;
}

fn void dispatch_matmul(
    CommandBuffer cmd,
    LlamaKernels* k,
    Tensor* weight,
    Tensor* input,
    Tensor* output,
    uint out_dim,
    uint in_dim
) {
    MatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim };
    ComputeKernel* kernel;
    if (weight.dtype == GGML_Q8_0) {
        kernel = &k.matmul_q8;
    } else if (weight.dtype == GGML_Q4_K) {
        kernel = &k.matmul_q4k;
    } else if (weight.dtype == GGML_Q5_K) {
        kernel = &k.matmul_q5k;
    } else if (weight.dtype == GGML_Q6_K) {
        kernel = &k.matmul_q6k;
    } else {
        kernel = &k.matmul;
    }
    dispatch_kernel(cmd, kernel,
        { weight.gpu_buffer.buffer, input.gpu_buffer.buffer, output.gpu_buffer.buffer },
        { weight.size_bytes, input.size_bytes, output.size_bytes },
        &pc, out_dim);
}

fn LlamaModel? load_model(DeviceContext* ctx, String model_path) {
    io::printfn("Loading model from: %s", model_path);

    // Memory map the GGUF file
    mmap::FileMmap mm = file::mmap_open(model_path, "rb")!!;
    char[] data = mm.bytes();
    io::printfn("File mapped: %d bytes", data.len);

    GGUFFile gf = gguf_parse(data)!!;
    gguf_print_info(&gf);

    LlamaConfig config = extract_config(&gf);
    LlamaWeights weights = load_weights(ctx, &gf, &config)!!;
    LlamaActivations acts = allocate_activations(ctx, &config)!!;
    LlamaKVCache kv = allocate_kv_cache(ctx, &config)!!;
    LlamaKernels kernels = create_kernels(ctx)!!;

    io::printfn("\nModel loaded successfully!");
    io::printfn("GPU memory used: %d MB", ctx.allocator.total_used() / (1024 * 1024));

    return {
        .config = config,
        .weights = weights,
        .acts = acts,
        .kv = kv,
        .kernels = kernels,
        .ctx = ctx,
    };
}

fn void? LlamaModel.forward(&self, uint token, uint position) {
    DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    LlamaConfig* c = &self.config;
    LlamaWeights* w = &self.weights;
    LlamaActivations* a = &self.acts;
    LlamaKVCache* kv = &self.kv;
    LlamaKernels* k = &self.kernels;

    begin_compute(cmd)!!;

    // 1. Embedding lookup -> hidden
    EmbeddingPC emb_pc = { .token_id = token, .dim = c.dim };
    dispatch_kernel(cmd, &k.embedding,
        { w.token_embedding.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
        { w.token_embedding.size_bytes, a.hidden.size_bytes },
        &emb_pc, ceil_div(c.dim, 256));

    compute_barrier(cmd);

    // 2. For each transformer layer
    for (uint l = 0; l < c.n_layers; l++) {
        LlamaLayerWeights* lw = &w.layers[l];

        // 2a. Copy hidden -> residual (via buffer copy)
        vk::cmdCopyBuffer(cmd, a.hidden.gpu_buffer.buffer, a.residual.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)c.dim * 4}});
        compute_barrier(cmd);

        // 2b. RMSNorm(hidden, attn_norm) -> norm_out
        RMSNormPC rms_pc = { .dim = c.dim, .eps = c.rms_eps };
        dispatch_kernel(cmd, &k.rmsnorm,
            { a.hidden.gpu_buffer.buffer, lw.attn_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, lw.attn_norm.size_bytes, a.norm_out.size_bytes },
            &rms_pc, 1);
        compute_barrier(cmd);

        // 2c. Q/K/V projections
        uint kv_dim = c.n_kv_heads * c.head_dim;
        dispatch_matmul(cmd, k, &lw.wq, &a.norm_out, &a.q, c.dim, c.dim);
        dispatch_matmul(cmd, k, &lw.wk, &a.norm_out, &a.k, kv_dim, c.dim);
        dispatch_matmul(cmd, k, &lw.wv, &a.norm_out, &a.v, kv_dim, c.dim);
        compute_barrier(cmd);

        // 2d. RoPE on Q and K
        RoPEPC rope_q_pc = { .dim = c.head_dim, .n_heads = c.n_heads, .position = position, .theta = c.rope_theta };
        dispatch_kernel(cmd, &k.rope,
            { a.q.gpu_buffer.buffer },
            { a.q.size_bytes },
            &rope_q_pc, ceil_div(c.n_heads * (c.head_dim / 2), 128));

        RoPEPC rope_k_pc = { .dim = c.head_dim, .n_heads = c.n_kv_heads, .position = position, .theta = c.rope_theta };
        dispatch_kernel(cmd, &k.rope,
            { a.k.gpu_buffer.buffer },
            { a.k.size_bytes },
            &rope_k_pc, ceil_div(c.n_kv_heads * (c.head_dim / 2), 128));
        compute_barrier(cmd);

        // 2e. Copy K, V into KV cache at position
        usz kv_head_bytes = (usz)c.head_dim * 4;
        for (uint h = 0; h < c.n_kv_heads; h++) {
            usz src_offset = h * kv_head_bytes;
            usz dst_offset = (usz)h * MAX_SEQ_LEN * kv_head_bytes + (usz)position * kv_head_bytes;

            vk::cmdCopyBuffer(cmd, a.k.gpu_buffer.buffer, kv.k_cache[l].gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = kv_head_bytes }});
            vk::cmdCopyBuffer(cmd, a.v.gpu_buffer.buffer, kv.v_cache[l].gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = kv_head_bytes }});
        }
        compute_barrier(cmd);

        // 2f. Attention
        uint seq_len = position + 1;
        AttentionPC attn_pc = {
            .head_dim = c.head_dim,
            .n_kv_heads = c.n_kv_heads,
            .n_q_heads = c.n_heads,
            .seq_len = seq_len,
            .scale = 1.0f / std::math::sqrt((float)c.head_dim),
            .max_seq_len = MAX_SEQ_LEN,
        };
        dispatch_kernel(cmd, &k.attention,
            { a.q.gpu_buffer.buffer, kv.k_cache[l].gpu_buffer.buffer, kv.v_cache[l].gpu_buffer.buffer,
              a.attn_scores.gpu_buffer.buffer, a.attn_out.gpu_buffer.buffer },
            { a.q.size_bytes, kv.k_cache[l].size_bytes, kv.v_cache[l].size_bytes,
              a.attn_scores.size_bytes, a.attn_out.size_bytes },
            &attn_pc, c.n_heads);
        compute_barrier(cmd);

        // 2g. Output projection
        dispatch_matmul(cmd, k, &lw.wo, &a.attn_out, &a.norm_out, c.dim, c.dim);
        compute_barrier(cmd);

        // 2h. Residual add: hidden = residual + attn_output
        // First copy residual back to hidden
        vk::cmdCopyBuffer(cmd, a.residual.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)c.dim * 4}});
        compute_barrier(cmd);

        ResidualPC res_pc = { .n = c.dim };
        dispatch_kernel(cmd, &k.residual_add,
            { a.hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.norm_out.size_bytes },
            &res_pc, ceil_div(c.dim, 256));
        compute_barrier(cmd);

        // 2i. Copy hidden -> residual for FFN
        vk::cmdCopyBuffer(cmd, a.hidden.gpu_buffer.buffer, a.residual.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)c.dim * 4}});
        compute_barrier(cmd);

        // 2j. RMSNorm for FFN
        dispatch_kernel(cmd, &k.rmsnorm,
            { a.hidden.gpu_buffer.buffer, lw.ffn_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, lw.ffn_norm.size_bytes, a.norm_out.size_bytes },
            &rms_pc, 1);
        compute_barrier(cmd);

        // 2k. FFN: gate and up projections
        dispatch_matmul(cmd, k, &lw.ffn_gate, &a.norm_out, &a.ffn_gate_out, c.ffn_dim, c.dim);
        dispatch_matmul(cmd, k, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, c.dim);
        compute_barrier(cmd);

        // 2l. SiLU(gate) then gate *= up (SwiGLU)
        SiluPC silu_pc = { .n = c.ffn_dim };
        dispatch_kernel(cmd, &k.silu,
            { a.ffn_gate_out.gpu_buffer.buffer },
            { a.ffn_gate_out.size_bytes },
            &silu_pc, ceil_div(c.ffn_dim, 256));
        compute_barrier(cmd);

        ElemwisePC emul_pc = { .n = c.ffn_dim };
        dispatch_kernel(cmd, &k.elemwise_mul,
            { a.ffn_gate_out.gpu_buffer.buffer, a.ffn_up_out.gpu_buffer.buffer },
            { a.ffn_gate_out.size_bytes, a.ffn_up_out.size_bytes },
            &emul_pc, ceil_div(c.ffn_dim, 256));
        compute_barrier(cmd);

        // 2m. Down projection
        dispatch_matmul(cmd, k, &lw.ffn_down, &a.ffn_gate_out, &a.ffn_down_out, c.dim, c.ffn_dim);
        compute_barrier(cmd);

        // 2n. Residual add
        vk::cmdCopyBuffer(cmd, a.residual.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)c.dim * 4}});
        compute_barrier(cmd);

        dispatch_kernel(cmd, &k.residual_add,
            { a.hidden.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.ffn_down_out.size_bytes },
            &res_pc, ceil_div(c.dim, 256));
        compute_barrier(cmd);
    }

    // 3. Final RMSNorm
    RMSNormPC final_rms = { .dim = c.dim, .eps = c.rms_eps };
    dispatch_kernel(cmd, &k.rmsnorm,
        { a.hidden.gpu_buffer.buffer, w.output_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
        { a.hidden.size_bytes, w.output_norm.size_bytes, a.norm_out.size_bytes },
        &final_rms, 1);
    compute_barrier(cmd);

    // 4. Output projection -> logits
    dispatch_matmul(cmd, k, &w.output, &a.norm_out, &a.logits, c.vocab_size, c.dim);

    // 5. Submit and wait
    submit_and_wait(ctx)!!;
}

fn void LlamaModel.free(&self) {
    // Free kernels
    self.kernels.embedding.free(self.ctx.device);
    self.kernels.rmsnorm.free(self.ctx.device);
    self.kernels.silu.free(self.ctx.device);
    self.kernels.matmul.free(self.ctx.device);
    self.kernels.matmul_q8.free(self.ctx.device);
    self.kernels.matmul_q4k.free(self.ctx.device);
    self.kernels.matmul_q5k.free(self.ctx.device);
    self.kernels.matmul_q6k.free(self.ctx.device);
    self.kernels.rope.free(self.ctx.device);
    self.kernels.softmax.free(self.ctx.device);
    self.kernels.attention.free(self.ctx.device);
    self.kernels.residual_add.free(self.ctx.device);
    self.kernels.elemwise_mul.free(self.ctx.device);

    // Free activations
    self.acts.hidden.free();
    self.acts.residual.free();
    self.acts.norm_out.free();
    self.acts.q.free();
    self.acts.k.free();
    self.acts.v.free();
    self.acts.attn_out.free();
    self.acts.ffn_gate_out.free();
    self.acts.ffn_up_out.free();
    self.acts.ffn_down_out.free();
    self.acts.logits.free();
    self.acts.attn_scores.free();

    // Free KV cache
    for (uint l = 0; l < self.config.n_layers; l++) {
        self.kv.k_cache[l].free();
        self.kv.v_cache[l].free();
    }
    mem::free(self.kv.k_cache);
    mem::free(self.kv.v_cache);

    // Free weights
    self.weights.token_embedding.free();
    self.weights.output_norm.free();
    // Only free output if it's not the same as embedding (tied weights)
    if (self.weights.output.gpu_buffer.buffer != self.weights.token_embedding.gpu_buffer.buffer) {
        self.weights.output.free();
    }
    for (uint l = 0; l < self.config.n_layers; l++) {
        LlamaLayerWeights* lw = &self.weights.layers[l];
        lw.attn_norm.free();
        lw.wq.free();
        lw.wk.free();
        lw.wv.free();
        lw.wo.free();
        lw.ffn_norm.free();
        lw.ffn_gate.free();
        lw.ffn_up.free();
        lw.ffn_down.free();
    }
    mem::free(self.weights.layers);
}
