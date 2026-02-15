module llm_text;

import vk;
import llm;
import std::io;
import std::math;
import std::core::mem;

const MAX_SEQ_LEN = 2048;

// Embedded SPIR-V shader (single module with all entry points)
const char[*] LLM_SPV = $embed("./shaders/llm.spv");

// Embedded architecture configs
const char[*] LLAMA_CONFIG_JSON = $embed("./configs/llama.json");
const char[*] QWEN2_CONFIG_JSON = $embed("./configs/qwen2.json");
const char[*] QWEN3_CONFIG_JSON = $embed("./configs/qwen3.json");

struct LayerWeights {
    llm::Tensor attn_norm;
    llm::Tensor attn_norm_bias;
    llm::Tensor wq;
    llm::Tensor wk;
    llm::Tensor wv;
    llm::Tensor wo;
    llm::Tensor wq_norm;
    llm::Tensor wk_norm;
    llm::Tensor ffn_norm;
    llm::Tensor ffn_norm_bias;
    llm::Tensor ffn_gate;
    llm::Tensor ffn_up;
    llm::Tensor ffn_down;
}

struct LlmWeights {
    llm::Tensor token_embedding;
    llm::Tensor output_norm;
    llm::Tensor output_norm_bias;
    llm::Tensor output;
    LayerWeights[] layers;
    // CPU-side embedding for encode_only mode (avoids F32 expansion on GPU)
    char* embedding_data;
    llm::GGUFTensorInfo embedding_info;
}

struct LlmActivations {
    llm::Tensor hidden;
    llm::Tensor norm_out;
    llm::Tensor q;
    llm::Tensor k;
    llm::Tensor v;
    llm::Tensor attn_out;
    llm::Tensor ffn_gate_out;
    llm::Tensor ffn_up_out;
    llm::Tensor ffn_down_out;
    llm::Tensor logits;
    llm::Tensor attn_scores;
}

struct KVCache {
    llm::Tensor[] k_cache;
    llm::Tensor[] v_cache;
}

// LLM-specific kernels (rope, embedding, causal attention)
struct LlmKernels {
    llm::ComputeKernel embedding;
    llm::ComputeKernel rope;
    llm::ComputeKernel attention;
    llm::SharedKernels shared;
}

struct LlmModel {
    llm::ModelConfig config;
    LlmWeights weights;
    LlmActivations acts;
    KVCache kv;
    LlmKernels kernels;
    llm::DeviceContext* ctx;
    bool encode_only;
    uint max_seq_len;
    vk::Memory embedding_staging;  // HOST_VISIBLE staging for CPU embedding upload
    // References for streaming layer loading (encode_only mode)
    llm::GGUFFile* gf;
    llm::WeightNames* names;
}

// LLM-specific push constant structs (shared PC structs are in llm::kernels.c3)
struct EmbeddingPC { uint token_id; uint dim; }
struct RoPEPC { uint dim; uint n_heads; uint position; float theta; }
struct AttentionPC { uint head_dim; uint n_kv_heads; uint n_q_heads; uint seq_len; float scale; uint max_seq_len; }

fn LlmWeights? load_llm_weights(llm::DeviceContext* ctx, llm::GGUFFile* gf, llm::ModelConfig* config, llm::WeightNames* names, bool encode_only = false) {
    io::printfn("\nLoading weights...");

    llm::Tensor token_embedding;
    char* embedding_data = null;
    llm::GGUFTensorInfo embedding_info;

    if (encode_only) {
        // Store CPU-side reference for per-token dequant (saves ~1.5GB for large vocabs)
        llm::GGUFTensorInfo* info = gf.find_tensor(names.embedding)!!;
        embedding_info = *info;
        embedding_data = gf.tensor_data_base + (usz)info.offset;
        io::printfn("  Embedding: CPU-side dequant (encode_only), type=%d", info.type);
    } else {
        // Embedding must be F32 for the embedding lookup shader
        token_embedding = llm::load_tensor_as_f32(ctx, gf, names.embedding)!!;
    }

    llm::Tensor output_norm = llm::load_tensor_by_name(ctx, gf, names.output_norm)!!;

    // Load output norm bias for LayerNorm architectures
    llm::Tensor output_norm_bias;
    if (config.norm_type == LAYERNORM && names.output_norm_bias.len > 0) {
        output_norm_bias = llm::load_tensor_by_name(ctx, gf, names.output_norm_bias)!!;
    }

    // Output weight - skip entirely in encode_only mode (saves ~200-700MB)
    llm::Tensor output_weight;
    if (encode_only) {
        io::printfn("  Skipping output weight (encode_only)");
    } else if (try output_info = gf.find_tensor(names.output)) {
        output_weight = llm::upload_weight(ctx, output_info, gf.tensor_data_base)!!;
    } else {
        // Tied weights - use embedding
        output_weight = token_embedding;
    }

    LayerWeights[] layers;

    if (encode_only) {
        // Streaming mode: layers loaded on-demand during encode_text
        io::printfn("  Layers: streaming mode (loaded per-layer during encoding)");
    } else {
        layers = mem::new_array(LayerWeights, config.n_layers);

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

            // Load Q/K norm weights if configured
            if (names.wq_norm.len > 0) {
                layers[l].wq_norm = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wq_norm);
                layers[l].wk_norm = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wk_norm);
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
    }

    return {
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output_norm_bias = output_norm_bias,
        .output = output_weight,
        .layers = layers,
        .embedding_data = embedding_data,
        .embedding_info = embedding_info,
    };
}

fn llm::Tensor load_layer_tensor(llm::DeviceContext* ctx, llm::GGUFFile* gf, String prefix, uint layer, String suffix) {
    char[64] buf;
    usz len = format_layer_name(&buf, prefix, layer, suffix);
    String name = (String)buf[0..len - 1];

    llm::GGUFTensorInfo* info = gf.find_tensor(name)!!;
    return llm::upload_weight(ctx, info, gf.tensor_data_base)!!;
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

// Upload one layer's weights to GPU (for streaming mode)
fn LayerWeights? load_single_layer(llm::DeviceContext* ctx, llm::GGUFFile* gf, llm::WeightNames* names, llm::ModelConfig* config, uint layer) {
    LayerWeights lw = {
        .attn_norm = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.attn_norm),
        .wq = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wq),
        .wk = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wk),
        .wv = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wv),
        .wo = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wo),
        .ffn_norm = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.ffn_norm),
        .ffn_up = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.ffn_up),
        .ffn_down = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.ffn_down),
    };
    if (config.ffn_type == SWIGLU) {
        lw.ffn_gate = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.ffn_gate);
    }
    if (names.wq_norm.len > 0) {
        lw.wq_norm = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wq_norm);
        lw.wk_norm = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wk_norm);
    }
    if (config.norm_type == LAYERNORM) {
        if (names.attn_norm_bias.len > 0) {
            lw.attn_norm_bias = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.attn_norm_bias);
        }
        if (names.ffn_norm_bias.len > 0) {
            lw.ffn_norm_bias = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.ffn_norm_bias);
        }
    }
    return lw;
}

// Free one layer's GPU weights
fn void free_layer_weights(LayerWeights* lw, llm::ModelConfig* config) {
    lw.attn_norm.free();
    if (lw.attn_norm_bias.size_bytes > 0) lw.attn_norm_bias.free();
    lw.wq.free();
    lw.wk.free();
    lw.wv.free();
    lw.wo.free();
    lw.ffn_norm.free();
    if (lw.ffn_norm_bias.size_bytes > 0) lw.ffn_norm_bias.free();
    if (lw.wq_norm.size_bytes > 0) lw.wq_norm.free();
    if (lw.wk_norm.size_bytes > 0) lw.wk_norm.free();
    if (lw.ffn_gate.size_bytes > 0) lw.ffn_gate.free();
    lw.ffn_up.free();
    lw.ffn_down.free();
}

fn LlmActivations? allocate_llm_activations(llm::DeviceContext* ctx, llm::ModelConfig* config, uint max_seq_len = MAX_SEQ_LEN, bool encode_only = false) {
    io::printfn("Allocating activation buffers (max_seq=%d)...", max_seq_len);

    ulong dim = config.dim;
    ulong ffn_dim = config.ffn_dim;
    ulong n_heads = config.n_heads;

    ulong[4] dim_shape = { dim, 0, 0, 0 };
    ulong[4] ffn_shape = { ffn_dim, 0, 0, 0 };
    ulong[4] qkv_shape = { dim, 0, 0, 0 };  // n_heads * head_dim = dim
    ulong[4] scores_shape = { n_heads * max_seq_len, 0, 0, 0 };

    llm::Tensor logits;
    if (!encode_only) {
        ulong[4] vocab_shape = { (ulong)config.vocab_size, 0, 0, 0 };
        logits = llm::create_f32_tensor(ctx, vocab_shape, 1)!!;
    }

    return {
        .hidden = llm::create_f32_tensor(ctx, dim_shape, 1)!!,
        .norm_out = llm::create_f32_tensor(ctx, dim_shape, 1)!!,
        .q = llm::create_f32_tensor(ctx, qkv_shape, 1)!!,
        .k = llm::create_f32_tensor(ctx, { (ulong)config.n_kv_heads * config.head_dim, 0, 0, 0 }, 1)!!,
        .v = llm::create_f32_tensor(ctx, { (ulong)config.n_kv_heads * config.head_dim, 0, 0, 0 }, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, dim_shape, 1)!!,
        .ffn_gate_out = llm::create_f32_tensor(ctx, ffn_shape, 1)!!,
        .ffn_up_out = llm::create_f32_tensor(ctx, ffn_shape, 1)!!,
        .ffn_down_out = llm::create_f32_tensor(ctx, dim_shape, 1)!!,
        .logits = logits,
        .attn_scores = llm::create_f32_tensor(ctx, scores_shape, 1)!!,
    };
}

fn KVCache? allocate_kv_cache(llm::DeviceContext* ctx, llm::ModelConfig* config, uint max_seq_len = MAX_SEQ_LEN) {
    io::printfn("Allocating KV cache (max_seq=%d, %d layers)...", max_seq_len, config.n_layers);

    llm::Tensor[] k_cache = mem::new_array(llm::Tensor, config.n_layers);
    llm::Tensor[] v_cache = mem::new_array(llm::Tensor, config.n_layers);

    ulong cache_size = (ulong)config.n_kv_heads * max_seq_len * config.head_dim;
    ulong[4] cache_shape = { cache_size, 0, 0, 0 };

    for (uint l = 0; l < config.n_layers; l++) {
        k_cache[l] = llm::create_f32_tensor(ctx, cache_shape, 1)!!;
        v_cache[l] = llm::create_f32_tensor(ctx, cache_shape, 1)!!;
    }

    return {
        .k_cache = k_cache,
        .v_cache = v_cache,
    };
}

fn LlmKernels? create_llm_kernels(llm::DeviceContext* ctx) {
    io::printfn("Creating compute kernels...");
    char[] spv = &LLM_SPV;
    ShaderModule shader = vk::shaderModuleCreateInfo()
        .setCodeSize(spv.len)
        .setCode((uint*)&spv[0])
        .build(ctx.device)!!;

    LlmKernels kernels = {
        .embedding    = llm::create_kernel(ctx, shader, 2, EmbeddingPC.sizeof, "embedding")!!,
        .rope         = llm::create_kernel(ctx, shader, 1, RoPEPC.sizeof, "rope")!!,
        .attention    = llm::create_kernel(ctx, shader, 5, AttentionPC.sizeof, "attention")!!,
        .shared       = llm::create_shared_kernels(ctx, shader)!!,
    };

    shader.free(ctx.device);
    return kernels;
}

fn LlmModel? load_llm_model(llm::DeviceContext* ctx, llm::GGUFFile* gf, llm::ModelConfig* config, llm::WeightNames* names, bool encode_only = false, uint max_seq_len = MAX_SEQ_LEN) {
    LlmWeights weights = load_llm_weights(ctx, gf, config, names, encode_only)!!;
    LlmActivations acts = allocate_llm_activations(ctx, config, max_seq_len, encode_only)!!;
    KVCache kv = allocate_kv_cache(ctx, config, max_seq_len)!!;
    LlmKernels kernels = create_llm_kernels(ctx)!!;

    // Create host-visible staging buffer for CPU embedding upload in encode_only mode
    vk::Memory embedding_staging;
    if (encode_only) {
        embedding_staging = vk::new_buffer(
            allocator: &ctx.allocator,
            usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null,
            data_size: (usz)config.dim * 4,
        )!!;
    }

    io::printfn("\nModel loaded successfully!");
    io::printfn("GPU memory used: %d MB", ctx.allocator.total_used() / (1024 * 1024));

    return {
        .config = *config,
        .weights = weights,
        .acts = acts,
        .kv = kv,
        .kernels = kernels,
        .ctx = ctx,
        .encode_only = encode_only,
        .max_seq_len = max_seq_len,
        .embedding_staging = embedding_staging,
        .gf = encode_only ? gf : null,
        .names = encode_only ? names : null,
    };
}

// Process a single transformer layer. Reads/writes acts.hidden.
// Must be called within a begin_compute/submit_and_wait block.
fn void? LlmModel.forward_layer(&self, CommandBuffer cmd, LayerWeights* lw, uint layer, uint position) {
    llm::ModelConfig* c = &self.config;
    LlmActivations* a = &self.acts;
    KVCache* kv = &self.kv;
    LlmKernels* k = &self.kernels;
    llm::SharedKernels* sk = &k.shared;

    // Norm(hidden, attn_norm) -> norm_out
    if (c.norm_type == RMSNORM) {
        llm::RMSNormPC rms_pc = { .dim = c.dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &sk.rmsnorm,
            { a.hidden.gpu_buffer.buffer, lw.attn_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, lw.attn_norm.size_bytes, a.norm_out.size_bytes },
            &rms_pc, 1);
    } else {
        llm::LayerNormPC ln_pc = { .dim = c.dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &sk.layernorm,
            { a.hidden.gpu_buffer.buffer, lw.attn_norm.gpu_buffer.buffer, lw.attn_norm_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, lw.attn_norm.size_bytes, lw.attn_norm_bias.size_bytes, a.norm_out.size_bytes },
            &ln_pc, 1);
    }
llm::compute_barrier(cmd);

    // Q/K/V projections
    uint kv_dim = c.n_kv_heads * c.head_dim;
llm::dispatch_matmul(cmd, sk, &lw.wq, &a.norm_out, &a.q, c.dim, c.dim);
llm::dispatch_matmul(cmd, sk, &lw.wk, &a.norm_out, &a.k, kv_dim, c.dim);
llm::dispatch_matmul(cmd, sk, &lw.wv, &a.norm_out, &a.v, kv_dim, c.dim);
llm::compute_barrier(cmd);

    // Q/K norm (per-head RMSNorm, e.g. Qwen3)
    if (lw.wq_norm.size_bytes > 0) {
        for (uint h = 0; h < c.n_heads; h++) {
            usz h_off = (usz)h * c.head_dim * 4;
            vk::cmdCopyBuffer(cmd, a.q.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = h_off, .dstOffset = 0, .size = (ulong)c.head_dim * 4 }});
llm::compute_barrier(cmd);
            llm::RMSNormPC qn_pc = { .dim = c.head_dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, lw.wq_norm.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
                { (usz)c.head_dim * 4, lw.wq_norm.size_bytes, (usz)c.head_dim * 4 },
                &qn_pc, 1);
llm::compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, a.ffn_down_out.gpu_buffer.buffer, a.q.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = h_off, .size = (ulong)c.head_dim * 4 }});
llm::compute_barrier(cmd);
        }
        for (uint h = 0; h < c.n_kv_heads; h++) {
            usz h_off = (usz)h * c.head_dim * 4;
            vk::cmdCopyBuffer(cmd, a.k.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = h_off, .dstOffset = 0, .size = (ulong)c.head_dim * 4 }});
llm::compute_barrier(cmd);
            llm::RMSNormPC kn_pc = { .dim = c.head_dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &sk.rmsnorm,
                { a.norm_out.gpu_buffer.buffer, lw.wk_norm.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
                { (usz)c.head_dim * 4, lw.wk_norm.size_bytes, (usz)c.head_dim * 4 },
                &kn_pc, 1);
llm::compute_barrier(cmd);
            vk::cmdCopyBuffer(cmd, a.ffn_down_out.gpu_buffer.buffer, a.k.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = h_off, .size = (ulong)c.head_dim * 4 }});
llm::compute_barrier(cmd);
        }
    }

    // RoPE on Q and K
    RoPEPC rope_q_pc = { .dim = c.head_dim, .n_heads = c.n_heads, .position = position, .theta = c.rope_theta };
llm::dispatch_kernel(cmd, &k.rope,
        { a.q.gpu_buffer.buffer },
        { a.q.size_bytes },
        &rope_q_pc, llm::ceil_div(c.n_heads * (c.head_dim / 2), 128));

    RoPEPC rope_k_pc = { .dim = c.head_dim, .n_heads = c.n_kv_heads, .position = position, .theta = c.rope_theta };
llm::dispatch_kernel(cmd, &k.rope,
        { a.k.gpu_buffer.buffer },
        { a.k.size_bytes },
        &rope_k_pc, llm::ceil_div(c.n_kv_heads * (c.head_dim / 2), 128));
llm::compute_barrier(cmd);

    // Copy K, V into KV cache at position
    usz kv_head_bytes = (usz)c.head_dim * 4;
    for (uint h = 0; h < c.n_kv_heads; h++) {
        usz src_offset = h * kv_head_bytes;
        usz dst_offset = (usz)h * self.max_seq_len * kv_head_bytes + (usz)position * kv_head_bytes;

        vk::cmdCopyBuffer(cmd, a.k.gpu_buffer.buffer, kv.k_cache[layer].gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = kv_head_bytes }});
        vk::cmdCopyBuffer(cmd, a.v.gpu_buffer.buffer, kv.v_cache[layer].gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = kv_head_bytes }});
    }
llm::compute_barrier(cmd);

    // Attention
    uint seq_len = position + 1;
    AttentionPC attn_pc = {
        .head_dim = c.head_dim,
        .n_kv_heads = c.n_kv_heads,
        .n_q_heads = c.n_heads,
        .seq_len = seq_len,
        .scale = 1.0f / std::math::sqrt((float)c.head_dim),
        .max_seq_len = self.max_seq_len,
    };
llm::dispatch_kernel(cmd, &k.attention,
        { a.q.gpu_buffer.buffer, kv.k_cache[layer].gpu_buffer.buffer, kv.v_cache[layer].gpu_buffer.buffer,
          a.attn_scores.gpu_buffer.buffer, a.attn_out.gpu_buffer.buffer },
        { a.q.size_bytes, kv.k_cache[layer].size_bytes, kv.v_cache[layer].size_bytes,
          a.attn_scores.size_bytes, a.attn_out.size_bytes },
        &attn_pc, c.n_heads);
llm::compute_barrier(cmd);

    // Output projection
llm::dispatch_matmul(cmd, sk, &lw.wo, &a.attn_out, &a.norm_out, c.dim, c.dim);
llm::compute_barrier(cmd);

    // Residual add: hidden += attn_output
    llm::ResidualPC res_pc = { .n = c.dim };
llm::dispatch_kernel(cmd, &sk.residual_add,
        { a.hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
        { a.hidden.size_bytes, a.norm_out.size_bytes },
        &res_pc, llm::ceil_div(c.dim, 256));
llm::compute_barrier(cmd);

    // Norm for FFN
    if (c.norm_type == RMSNORM) {
        llm::RMSNormPC rms_pc2 = { .dim = c.dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &sk.rmsnorm,
            { a.hidden.gpu_buffer.buffer, lw.ffn_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, lw.ffn_norm.size_bytes, a.norm_out.size_bytes },
            &rms_pc2, 1);
    } else {
        llm::LayerNormPC ln_pc2 = { .dim = c.dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &sk.layernorm,
            { a.hidden.gpu_buffer.buffer, lw.ffn_norm.gpu_buffer.buffer, lw.ffn_norm_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, lw.ffn_norm.size_bytes, lw.ffn_norm_bias.size_bytes, a.norm_out.size_bytes },
            &ln_pc2, 1);
    }
llm::compute_barrier(cmd);

    // FFN
    if (c.ffn_type == SWIGLU) {
llm::dispatch_matmul(cmd, sk, &lw.ffn_gate, &a.norm_out, &a.ffn_gate_out, c.ffn_dim, c.dim);
llm::dispatch_matmul(cmd, sk, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, c.dim);
llm::compute_barrier(cmd);

        llm::SiluPC silu_pc = { .n = c.ffn_dim };
llm::dispatch_kernel(cmd, &sk.silu,
            { a.ffn_gate_out.gpu_buffer.buffer },
            { a.ffn_gate_out.size_bytes },
            &silu_pc, llm::ceil_div(c.ffn_dim, 256));
llm::compute_barrier(cmd);

        llm::ElemwisePC emul_pc = { .n = c.ffn_dim };
llm::dispatch_kernel(cmd, &sk.elemwise_mul,
            { a.ffn_gate_out.gpu_buffer.buffer, a.ffn_up_out.gpu_buffer.buffer },
            { a.ffn_gate_out.size_bytes, a.ffn_up_out.size_bytes },
            &emul_pc, llm::ceil_div(c.ffn_dim, 256));
llm::compute_barrier(cmd);

llm::dispatch_matmul(cmd, sk, &lw.ffn_down, &a.ffn_gate_out, &a.ffn_down_out, c.dim, c.ffn_dim);
    } else {
llm::dispatch_matmul(cmd, sk, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, c.dim);
llm::compute_barrier(cmd);

        llm::GeluPC gelu_pc = { .n = c.ffn_dim };
llm::dispatch_kernel(cmd, &sk.gelu,
            { a.ffn_up_out.gpu_buffer.buffer },
            { a.ffn_up_out.size_bytes },
            &gelu_pc, llm::ceil_div(c.ffn_dim, 256));
llm::compute_barrier(cmd);

llm::dispatch_matmul(cmd, sk, &lw.ffn_down, &a.ffn_up_out, &a.ffn_down_out, c.dim, c.ffn_dim);
    }
llm::compute_barrier(cmd);

    // Residual add: hidden += ffn_down_out
llm::dispatch_kernel(cmd, &sk.residual_add,
        { a.hidden.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
        { a.hidden.size_bytes, a.ffn_down_out.size_bytes },
        &res_pc, llm::ceil_div(c.dim, 256));
llm::compute_barrier(cmd);
}

fn void? LlmModel.forward(&self, uint token, uint position) {
    llm::DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    llm::ModelConfig* c = &self.config;
    LlmWeights* w = &self.weights;
    LlmActivations* a = &self.acts;
    KVCache* kv = &self.kv;
    LlmKernels* k = &self.kernels;
    llm::SharedKernels* sk = &k.shared;

    // 1. Embedding lookup -> hidden
    if (self.encode_only) {
        // CPU-side dequant: dequant one row, upload via staging buffer
        float* staging_ptr = (float*)self.embedding_staging.data();
        llm::dequant_embedding_row(w.embedding_data, &w.embedding_info, token, staging_ptr, c.dim);

        // Copy staging -> hidden on GPU
        ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer emb_cmd) {
            vk::cmdCopyBuffer(emb_cmd, self.embedding_staging.buffer, a.hidden.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)c.dim * 4 }});
        }!!;
    }

llm::begin_compute(cmd)!!;

    if (!self.encode_only) {
        EmbeddingPC emb_pc = { .token_id = token, .dim = c.dim };
llm::dispatch_kernel(cmd, &k.embedding,
            { w.token_embedding.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
            { w.token_embedding.size_bytes, a.hidden.size_bytes },
            &emb_pc, llm::ceil_div(c.dim, 256));
llm::compute_barrier(cmd);
    }

    // 2. For each transformer layer
    for (uint l = 0; l < c.n_layers; l++) {
        self.forward_layer(cmd, &w.layers[l], l, position)!!;
    }

    // 3. Final norm
    if (c.norm_type == RMSNORM) {
        llm::RMSNormPC final_rms = { .dim = c.dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &sk.rmsnorm,
            { a.hidden.gpu_buffer.buffer, w.output_norm.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, w.output_norm.size_bytes, a.norm_out.size_bytes },
            &final_rms, 1);
    } else {
        llm::LayerNormPC final_ln = { .dim = c.dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &sk.layernorm,
            { a.hidden.gpu_buffer.buffer, w.output_norm.gpu_buffer.buffer, w.output_norm_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, w.output_norm.size_bytes, w.output_norm_bias.size_bytes, a.norm_out.size_bytes },
            &final_ln, 1);
    }
llm::compute_barrier(cmd);

    // 4. Output projection -> logits (skip in encode_only mode)
    if (!self.encode_only) {
llm::dispatch_matmul(cmd, sk, &w.output, &a.norm_out, &a.logits, c.vocab_size, c.dim);
    }

    // 5. Submit and wait
llm::submit_and_wait(ctx)!!;
}

// Encode text: run forward pass for each token, extract post-final-norm hidden states.
// Returns a GPU buffer with [seq_len, dim] embeddings.
// Caller is responsible for freeing the returned tensor.
fn llm::Tensor? LlmModel.encode_text(&self, uint[] tokens) {
    if (self.encode_only) {
        return self.encode_text_streaming(tokens);
    }

    llm::ModelConfig* c = &self.config;
    llm::DeviceContext* ctx = self.ctx;

    // Allocate output buffer: [seq_len, dim]
    ulong[4] out_shape = { (ulong)tokens.len, (ulong)c.dim, 0, 0 };
    llm::Tensor text_embeddings = llm::create_f32_tensor(ctx, { (ulong)tokens.len * c.dim, 0, 0, 0 }, 1)!!;

    // Process BOS + each token
    self.forward(tokens[0], 0)!!;

    ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, self.acts.norm_out.gpu_buffer.buffer,
            text_embeddings.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)c.dim * 4 }});
    }!!;

    for (usz i = 1; i < tokens.len; i++) {
        self.forward(tokens[i], (uint)i)!!;
        usz dst_offset = i * c.dim * 4;
        ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer cmd) {
            vk::cmdCopyBuffer(cmd, self.acts.norm_out.gpu_buffer.buffer,
                text_embeddings.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)dst_offset, .size = (ulong)c.dim * 4 }});
        }!!;
    }

    text_embeddings.n_dims = 2;
    text_embeddings.shape = out_shape;

    return text_embeddings;
}

// Streaming encode: load one layer's weights at a time, process all positions, free weights.
// GPU memory = 1 layer of weights + KV cache + activations (instead of all 36 layers).
fn llm::Tensor? LlmModel.encode_text_streaming(&self, uint[] tokens) {
    llm::ModelConfig* c = &self.config;
    llm::DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    uint seq_len = (uint)tokens.len;

    // Allocate intermediate hidden states: [seq_len, dim]
    llm::Tensor hidden_states = llm::create_f32_tensor(ctx, { (ulong)seq_len * c.dim, 0, 0, 0 }, 1)!!;

    // 1. Embed all tokens into hidden_states via CPU dequant
    for (uint pos = 0; pos < seq_len; pos++) {
        float* staging_ptr = (float*)self.embedding_staging.data();
        llm::dequant_embedding_row(self.weights.embedding_data, &self.weights.embedding_info, tokens[pos], staging_ptr, c.dim);

        ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer emb_cmd) {
            vk::cmdCopyBuffer(emb_cmd, self.embedding_staging.buffer, hidden_states.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)pos * c.dim * 4, .size = (ulong)c.dim * 4 }});
        }!!;
    }
    io::printfn("  Embedded %d tokens", seq_len);

    // 2. Process layers one at a time (streaming weights)
    for (uint l = 0; l < c.n_layers; l++) {
        // Upload this layer's weights
        LayerWeights lw = load_single_layer(ctx, self.gf, self.names, c, l)!!;

        // Process all positions through this layer
llm::begin_compute(cmd)!!;
        for (uint pos = 0; pos < seq_len; pos++) {
            // Copy hidden_states[pos] → acts.hidden
            vk::cmdCopyBuffer(cmd, hidden_states.gpu_buffer.buffer, self.acts.hidden.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = (ulong)pos * c.dim * 4, .dstOffset = 0, .size = (ulong)c.dim * 4 }});
llm::compute_barrier(cmd);

            // Process layer
            self.forward_layer(cmd, &lw, l, pos)!!;

            // Copy acts.hidden → hidden_states[pos]
            vk::cmdCopyBuffer(cmd, self.acts.hidden.gpu_buffer.buffer, hidden_states.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)pos * c.dim * 4, .size = (ulong)c.dim * 4 }});
llm::compute_barrier(cmd);
        }
llm::submit_and_wait(ctx)!!;

        // Free this layer's weights (space reused for next layer)
        free_layer_weights(&lw, c);

        if ((l + 1) % 4 == 0 || l == c.n_layers - 1) {
            io::printfn("  Processed layer %d / %d", l + 1, c.n_layers);
        }
    }

    // 3. Apply final norm to each position, build output
    llm::Tensor text_embeddings = llm::create_f32_tensor(ctx, { (ulong)seq_len * c.dim, 0, 0, 0 }, 1)!!;

llm::begin_compute(cmd)!!;
    for (uint pos = 0; pos < seq_len; pos++) {
        // Copy hidden_states[pos] → acts.hidden
        vk::cmdCopyBuffer(cmd, hidden_states.gpu_buffer.buffer, self.acts.hidden.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = (ulong)pos * c.dim * 4, .dstOffset = 0, .size = (ulong)c.dim * 4 }});
llm::compute_barrier(cmd);

        // Final norm
        if (c.norm_type == RMSNORM) {
            llm::RMSNormPC final_rms = { .dim = c.dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &self.kernels.shared.rmsnorm,
                { self.acts.hidden.gpu_buffer.buffer, self.weights.output_norm.gpu_buffer.buffer, self.acts.norm_out.gpu_buffer.buffer },
                { self.acts.hidden.size_bytes, self.weights.output_norm.size_bytes, self.acts.norm_out.size_bytes },
                &final_rms, 1);
        } else {
            llm::LayerNormPC final_ln = { .dim = c.dim, .eps = c.rms_eps };
llm::dispatch_kernel(cmd, &self.kernels.shared.layernorm,
                { self.acts.hidden.gpu_buffer.buffer, self.weights.output_norm.gpu_buffer.buffer, self.weights.output_norm_bias.gpu_buffer.buffer, self.acts.norm_out.gpu_buffer.buffer },
                { self.acts.hidden.size_bytes, self.weights.output_norm.size_bytes, self.weights.output_norm_bias.size_bytes, self.acts.norm_out.size_bytes },
                &final_ln, 1);
        }
llm::compute_barrier(cmd);

        // Copy norm_out → text_embeddings[pos]
        vk::cmdCopyBuffer(cmd, self.acts.norm_out.gpu_buffer.buffer, text_embeddings.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)pos * c.dim * 4, .size = (ulong)c.dim * 4 }});
llm::compute_barrier(cmd);
    }
llm::submit_and_wait(ctx)!!;

    hidden_states.free();

    text_embeddings.n_dims = 2;
    text_embeddings.shape = { (ulong)seq_len, (ulong)c.dim, 0, 0 };

    return text_embeddings;
}

fn void LlmModel.free(&self) {
    // Free LLM-specific kernels
    self.kernels.embedding.free(self.ctx.device);
    self.kernels.rope.free(self.ctx.device);
    self.kernels.attention.free(self.ctx.device);
    // Free shared kernels
    self.kernels.shared.free(self.ctx.device);

    // Free encode_only staging buffer
    if (self.encode_only) {
        self.embedding_staging.free();
    }

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
    if (!self.encode_only) self.acts.logits.free();
    self.acts.attn_scores.free();

    // Free KV cache
    for (uint l = 0; l < self.config.n_layers; l++) {
        self.kv.k_cache[l].free();
        self.kv.v_cache[l].free();
    }
    mem::free(self.kv.k_cache);
    mem::free(self.kv.v_cache);

    // Free weights
    if (!self.encode_only) {
        self.weights.token_embedding.free();
        // Only free output if it's not the same as embedding (tied weights)
        if (self.weights.output.gpu_buffer.buffer != self.weights.token_embedding.gpu_buffer.buffer) {
            self.weights.output.free();
        }
    }
    self.weights.output_norm.free();
    if (self.weights.output_norm_bias.size_bytes > 0) self.weights.output_norm_bias.free();
    if (!self.encode_only) {
        for (uint l = 0; l < self.config.n_layers; l++) {
            free_layer_weights(&self.weights.layers[l], &self.config);
        }
        mem::free(self.weights.layers);
    }
}
