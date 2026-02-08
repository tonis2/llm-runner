module llm;

import vk;
import std::io;
import std::math;
import std::core::mem;

const MAX_SEQ_LEN = 2048;

// Embedded SPIR-V shader (single module with all entry points)
const char[*] LLM_SPV = $embed("../../shaders/llm.spv");

// Embedded architecture configs
const char[*] LLAMA_CONFIG_JSON = $embed("../../configs/llama.json");
const char[*] QWEN2_CONFIG_JSON = $embed("../../configs/qwen2.json");
const char[*] PHI3_CONFIG_JSON = $embed("../../configs/phi3.json");

struct LayerWeights {
    Tensor attn_norm;
    Tensor attn_norm_bias;
    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor wo;
    Tensor ffn_norm;
    Tensor ffn_norm_bias;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
}

struct ModelWeights {
    Tensor token_embedding;
    Tensor output_norm;
    Tensor output_norm_bias;
    Tensor output;
    LayerWeights[] layers;
}

struct ModelActivations {
    Tensor hidden;
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

struct KVCache {
    Tensor[] k_cache;
    Tensor[] v_cache;
}

struct ModelKernels {
    ComputeKernel embedding;
    ComputeKernel rmsnorm;
    ComputeKernel layernorm;
    ComputeKernel silu;
    ComputeKernel gelu;
    ComputeKernel matmul;
    ComputeKernel matmul_q8;
    ComputeKernel matmul_q4_0;
    ComputeKernel matmul_q4k;
    ComputeKernel matmul_q5k;
    ComputeKernel matmul_q6k;
    ComputeKernel rope;
    ComputeKernel softmax;
    ComputeKernel attention;
    ComputeKernel residual_add;
    ComputeKernel elemwise_mul;
}

struct Model {
    ModelConfig config;
    ModelWeights weights;
    ModelActivations acts;
    KVCache kv;
    ModelKernels kernels;
    DeviceContext* ctx;
}

struct EmbeddingPC { uint token_id; uint dim; }
struct RMSNormPC { uint dim; float eps; }
struct LayerNormPC { uint dim; float eps; }
struct SiluPC { uint n; }
struct GeluPC { uint n; }
struct MatMulPC { uint out_dim; uint in_dim; }
struct RoPEPC { uint dim; uint n_heads; uint position; float theta; }
struct SoftmaxPC { uint n; uint offset; }
struct AttentionPC { uint head_dim; uint n_kv_heads; uint n_q_heads; uint seq_len; float scale; uint max_seq_len; }
struct ResidualPC { uint n; }
struct ElemwisePC { uint n; }

fn Tensor load_tensor_by_name(DeviceContext* ctx, GGUFFile* gf, String name) {
    GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return upload_weight(ctx, info, gf.tensor_data_base)!!;
}

fn Tensor load_tensor_as_f32(DeviceContext* ctx, GGUFFile* gf, String name) {
    GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return upload_weight_as_f32(ctx, info, gf.tensor_data_base)!!;
}

fn ModelWeights? load_weights(DeviceContext* ctx, GGUFFile* gf, ModelConfig* config, WeightNames* names) {
    io::printfn("\nLoading weights...");

    // Embedding must be F32 for the embedding lookup shader
    Tensor token_embedding = load_tensor_as_f32(ctx, gf, names.embedding);
    Tensor output_norm = load_tensor_by_name(ctx, gf, names.output_norm);

    // Load output norm bias for LayerNorm architectures
    Tensor output_norm_bias;
    if (config.norm_type == LAYERNORM && names.output_norm_bias.len > 0) {
        output_norm_bias = load_tensor_by_name(ctx, gf, names.output_norm_bias);
    }

    // Output weight - may be same as embedding (tied weights) or separate
    Tensor output_weight;
    if (try output_info = gf.find_tensor(names.output)) {
        output_weight = upload_weight(ctx, output_info, gf.tensor_data_base)!!;
    } else {
        // Tied weights - use embedding
        output_weight = token_embedding;
    }

    LayerWeights[] layers = mem::new_array(LayerWeights, config.n_layers);

    for (uint l = 0; l < config.n_layers; l++) {
        layers[l] = {
            .attn_norm = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.attn_norm),
            .wq = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wq),
            .wk = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wk),
            .wv = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wv),
            .wo = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wo),
            .ffn_norm = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.ffn_norm),
            .ffn_up = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.ffn_up),
            .ffn_down = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.ffn_down),
        };

        // Load gate weight only for SwiGLU FFN
        if (config.ffn_type == SWIGLU) {
            layers[l].ffn_gate = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.ffn_gate);
        }

        // Load norm biases for LayerNorm architectures
        if (config.norm_type == LAYERNORM) {
            if (names.attn_norm_bias.len > 0) {
                layers[l].attn_norm_bias = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.attn_norm_bias);
            }
            if (names.ffn_norm_bias.len > 0) {
                layers[l].ffn_norm_bias = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.ffn_norm_bias);
            }
        }

        if ((l + 1) % 4 == 0 || l == config.n_layers - 1) {
            io::printfn("  Loaded layer %d / %d", l + 1, config.n_layers);
        }
    }

    return {
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output_norm_bias = output_norm_bias,
        .output = output_weight,
        .layers = layers,
    };
}

fn Tensor load_layer_tensor(DeviceContext* ctx, GGUFFile* gf, String prefix, uint layer, String suffix) {
    char[64] buf;
    usz len = format_layer_name(&buf, prefix, layer, suffix);
    String name = (String)buf[0..len - 1];

    GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return upload_weight(ctx, info, gf.tensor_data_base)!!;
}

fn usz format_layer_name(char[64]* buf, String prefix, uint layer, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;

    // Write prefix (e.g. "blk")
    for (usz i = 0; i < prefix.len; i++) {
        p[pos + i] = prefix[i];
    }
    pos += prefix.len;

    // Write "."
    p[pos] = '.';
    pos++;

    // Write layer number
    if (layer >= 100) {
        p[pos] = (char)('0' + layer / 100);
        pos++;
    }
    if (layer >= 10) {
        p[pos] = (char)('0' + (layer / 10) % 10);
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

fn ModelActivations? allocate_activations(DeviceContext* ctx, ModelConfig* config) {
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

fn KVCache? allocate_kv_cache(DeviceContext* ctx, ModelConfig* config) {
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

fn ModelKernels? create_kernels(DeviceContext* ctx) {
    io::printfn("Creating compute kernels...");
    char[] spv = &LLM_SPV;
    ShaderModule shader = vk::shaderModuleCreateInfo()
        .setCodeSize(spv.len)
        .setCode((uint*)&spv[0])
        .build(ctx.device)!!;

    ModelKernels kernels = {
        .embedding    = create_kernel(ctx, shader, 2, EmbeddingPC.sizeof, "embedding")!!,
        .rmsnorm      = create_kernel(ctx, shader, 3, RMSNormPC.sizeof, "rmsnorm")!!,
        .layernorm    = create_kernel(ctx, shader, 4, LayerNormPC.sizeof, "layernorm")!!,
        .silu         = create_kernel(ctx, shader, 1, SiluPC.sizeof, "silu")!!,
        .gelu         = create_kernel(ctx, shader, 1, GeluPC.sizeof, "gelu")!!,
        .matmul       = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul")!!,
        .matmul_q8    = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q8")!!,
        .matmul_q4_0  = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q4_0")!!,
        .matmul_q4k   = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q4k")!!,
        .matmul_q5k   = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q5k")!!,
        .matmul_q6k   = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q6k")!!,
        .rope         = create_kernel(ctx, shader, 1, RoPEPC.sizeof, "rope")!!,
        .softmax      = create_kernel(ctx, shader, 1, SoftmaxPC.sizeof, "softmax")!!,
        .attention    = create_kernel(ctx, shader, 5, AttentionPC.sizeof, "attention")!!,
        .residual_add = create_kernel(ctx, shader, 2, ResidualPC.sizeof, "residual_add")!!,
        .elemwise_mul = create_kernel(ctx, shader, 2, ElemwisePC.sizeof, "elemwise_mul")!!,
    };

    shader.free(ctx.device);
    return kernels;
}

fn uint ceil_div(uint a, uint b) {
    return (a + b - 1) / b;
}

fn void dispatch_matmul(
    CommandBuffer cmd,
    ModelKernels* k,
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
    } else if (weight.dtype == GGML_Q4_0) {
        kernel = &k.matmul_q4_0;
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

fn Model? load_model(DeviceContext* ctx, GGUFFile* gf, ModelConfig* config, WeightNames* names) {
    ModelWeights weights = load_weights(ctx, gf, config, names)!!;
    ModelActivations acts = allocate_activations(ctx, config)!!;
    KVCache kv = allocate_kv_cache(ctx, config)!!;
    ModelKernels kernels = create_kernels(ctx)!!;

    io::printfn("\nModel loaded successfully!");
    io::printfn("GPU memory used: %d MB", ctx.allocator.total_used() / (1024 * 1024));

    return {
        .config = *config,
        .weights = weights,
        .acts = acts,
        .kv = kv,
        .kernels = kernels,
        .ctx = ctx,
    };
}

fn void? Model.forward(&self, uint token, uint position) {
    DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    ModelConfig* c = &self.config;
    ModelWeights* w = &self.weights;
    ModelActivations* a = &self.acts;
    KVCache* kv = &self.kv;
    ModelKernels* k = &self.kernels;

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
        LayerWeights* lw = &w.layers[l];

        // 2b. Norm(hidden, attn_norm) -> norm_out
        if (c.norm_type == RMSNORM) {
            RMSNormPC rms_pc = { .dim = c.dim, .eps = c.rms_eps };
            dispatch_kernel(cmd, &k.rmsnorm,
                { a.hidden.gpu_buffer.buffer, lw.attn_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.attn_norm.size_bytes, a.norm_out.size_bytes },
                &rms_pc, 1);
        } else {
            LayerNormPC ln_pc = { .dim = c.dim, .eps = c.rms_eps };
            dispatch_kernel(cmd, &k.layernorm,
                { a.hidden.gpu_buffer.buffer, lw.attn_norm.gpu_buffer.buffer, lw.attn_norm_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.attn_norm.size_bytes, lw.attn_norm_bias.size_bytes, a.norm_out.size_bytes },
                &ln_pc, 1);
        }
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

        // 2h. Residual add: hidden += attn_output (hidden still has original value)
        ResidualPC res_pc = { .n = c.dim };
        dispatch_kernel(cmd, &k.residual_add,
            { a.hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.norm_out.size_bytes },
            &res_pc, ceil_div(c.dim, 256));
        compute_barrier(cmd);

        // 2j. Norm for FFN
        if (c.norm_type == RMSNORM) {
            RMSNormPC rms_pc2 = { .dim = c.dim, .eps = c.rms_eps };
            dispatch_kernel(cmd, &k.rmsnorm,
                { a.hidden.gpu_buffer.buffer, lw.ffn_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.ffn_norm.size_bytes, a.norm_out.size_bytes },
                &rms_pc2, 1);
        } else {
            LayerNormPC ln_pc2 = { .dim = c.dim, .eps = c.rms_eps };
            dispatch_kernel(cmd, &k.layernorm,
                { a.hidden.gpu_buffer.buffer, lw.ffn_norm.gpu_buffer.buffer, lw.ffn_norm_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.ffn_norm.size_bytes, lw.ffn_norm_bias.size_bytes, a.norm_out.size_bytes },
                &ln_pc2, 1);
        }
        compute_barrier(cmd);

        // 2k. FFN
        if (c.ffn_type == SWIGLU) {
            // SwiGLU: gate+up → SiLU(gate) → gate*up → down
            dispatch_matmul(cmd, k, &lw.ffn_gate, &a.norm_out, &a.ffn_gate_out, c.ffn_dim, c.dim);
            dispatch_matmul(cmd, k, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, c.dim);
            compute_barrier(cmd);

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

            dispatch_matmul(cmd, k, &lw.ffn_down, &a.ffn_gate_out, &a.ffn_down_out, c.dim, c.ffn_dim);
        } else {
            // GELU MLP: up → GELU → down
            dispatch_matmul(cmd, k, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, c.dim);
            compute_barrier(cmd);

            GeluPC gelu_pc = { .n = c.ffn_dim };
            dispatch_kernel(cmd, &k.gelu,
                { a.ffn_up_out.gpu_buffer.buffer },
                { a.ffn_up_out.size_bytes },
                &gelu_pc, ceil_div(c.ffn_dim, 256));
            compute_barrier(cmd);

            dispatch_matmul(cmd, k, &lw.ffn_down, &a.ffn_up_out, &a.ffn_down_out, c.dim, c.ffn_dim);
        }
        compute_barrier(cmd);

        // 2n. Residual add: hidden += ffn_down_out
        dispatch_kernel(cmd, &k.residual_add,
            { a.hidden.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.ffn_down_out.size_bytes },
            &res_pc, ceil_div(c.dim, 256));
        compute_barrier(cmd);
    }

    // 3. Final norm
    if (c.norm_type == RMSNORM) {
        RMSNormPC final_rms = { .dim = c.dim, .eps = c.rms_eps };
        dispatch_kernel(cmd, &k.rmsnorm,
            { a.hidden.gpu_buffer.buffer, w.output_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, w.output_norm.size_bytes, a.norm_out.size_bytes },
            &final_rms, 1);
    } else {
        LayerNormPC final_ln = { .dim = c.dim, .eps = c.rms_eps };
        dispatch_kernel(cmd, &k.layernorm,
            { a.hidden.gpu_buffer.buffer, w.output_norm.gpu_buffer.buffer, w.output_norm_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, w.output_norm.size_bytes, w.output_norm_bias.size_bytes, a.norm_out.size_bytes },
            &final_ln, 1);
    }
    compute_barrier(cmd);

    // 4. Output projection -> logits
    dispatch_matmul(cmd, k, &w.output, &a.norm_out, &a.logits, c.vocab_size, c.dim);

    // 5. Submit and wait
    submit_and_wait(ctx)!!;
}

fn void Model.free(&self) {
    // Free kernels
    self.kernels.embedding.free(self.ctx.device);
    self.kernels.rmsnorm.free(self.ctx.device);
    self.kernels.layernorm.free(self.ctx.device);
    self.kernels.silu.free(self.ctx.device);
    self.kernels.gelu.free(self.ctx.device);
    self.kernels.matmul.free(self.ctx.device);
    self.kernels.matmul_q8.free(self.ctx.device);
    self.kernels.matmul_q4_0.free(self.ctx.device);
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
    if (self.weights.output_norm_bias.size_bytes > 0) self.weights.output_norm_bias.free();
    // Only free output if it's not the same as embedding (tied weights)
    if (self.weights.output.gpu_buffer.buffer != self.weights.token_embedding.gpu_buffer.buffer) {
        self.weights.output.free();
    }
    for (uint l = 0; l < self.config.n_layers; l++) {
        LayerWeights* lw = &self.weights.layers[l];
        lw.attn_norm.free();
        if (lw.attn_norm_bias.size_bytes > 0) lw.attn_norm_bias.free();
        lw.wq.free();
        lw.wk.free();
        lw.wv.free();
        lw.wo.free();
        lw.ffn_norm.free();
        if (lw.ffn_norm_bias.size_bytes > 0) lw.ffn_norm_bias.free();
        if (lw.ffn_gate.size_bytes > 0) lw.ffn_gate.free();
        lw.ffn_up.free();
        lw.ffn_down.free();
    }
    mem::free(self.weights.layers);
}
