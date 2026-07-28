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
const char[*] QWEN2_VL_CONFIG_JSON = $embed("./configs/qwen2_vl.json");

struct LayerWeights {
    llm::Tensor attn_norm;
    llm::Tensor attn_norm_bias;
    llm::Tensor wq;
    llm::Tensor wk;
    llm::Tensor wv;
    llm::Tensor wo;
    llm::Tensor wq_bias;
    llm::Tensor wk_bias;
    llm::Tensor wv_bias;
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
    // Flash-decode partials: per (head, chunk) an unnormalised weighted-V sum
    // and its chunk's softmax max + expsum.
    llm::Tensor attn_partial;
    llm::Tensor attn_meta;
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
    // Fused decode path (forward_layer_fused)
    llm::ComputeKernel silu_mul;
    llm::ComputeKernel residual_rmsnorm;
    llm::ComputeKernel head_norm_rope;
    llm::ComputeKernel attention_split;
    llm::ComputeKernel attention_combine;
    llm::ComputeKernel matmul_q8_gemv;
    llm::ComputeKernel embedding_batch;
    llm::ComputeKernel kv_cache_store_multi;
    llm::ComputeKernel matmul_q8_gemv_multi;
    // Batched prefill path (encode_text_multilayer)
    llm::ComputeKernel rmsnorm_batch;
    llm::ComputeKernel head_rmsnorm_batch;
    llm::ComputeKernel rope_batch;
    llm::ComputeKernel kv_cache_store;
    llm::ComputeKernel attention_prefill;
    llm::ComputeKernel batch_matmul_simple;
    llm::ComputeKernel batch_matmul_q8;
    llm::ComputeKernel batch_matmul_q4_0;
    llm::ComputeKernel batch_matmul_q4k;
    llm::ComputeKernel batch_matmul_q5k;
    llm::ComputeKernel batch_matmul_q6k;
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
// Batched prefill push constants (must match llm.slang prefill kernels)
struct RMSNormBatchPC { uint dim; float eps; uint n_rows; }
struct HeadRMSNormBatchPC { uint head_dim; uint n_heads; uint n_rows; float eps; }
struct SiluMulPC { uint n; }
struct ResidualRMSNormPC { uint dim; float eps; uint n_rows; }
struct HeadNormRopePC { uint head_dim; uint n_heads; uint n_rows; float eps; uint position; float theta; uint pos_stride; }
// The shader-side `uint4 tokens[4]` sits on a 16-byte boundary (std430), so
// the two pads are load-bearing.
struct EmbeddingBatchPC { uint dim; uint n_rows; uint pad0; uint pad1; uint[16] tokens; }
struct KVStoreMultiPC { uint n_rows; uint kv_dim; uint head_dim; uint max_seq_len; uint position; }
struct MatMulMultiPC { uint out_dim; uint in_dim; uint n_rows; }
struct AttentionSplitPC { uint head_dim; uint n_kv_heads; uint n_q_heads; uint seq_len; float scale; uint max_seq_len; uint n_chunks; }
struct AttentionCombinePC { uint head_dim; uint n_q_heads; uint n_chunks; }

// Positions per attention_split workgroup; mirrors ATTN_CHUNK in llm.slang.
const uint ATTN_SPLIT_CHUNK = 256;
struct RoPEBatchPC { uint dim; uint n_heads; uint n_tokens; float theta; }
struct KVStorePC { uint n_tokens; uint kv_dim; uint head_dim; uint max_seq_len; uint pos_offset; }
struct AttentionPrefillPC { uint head_dim; uint n_kv_heads; uint n_q_heads; uint n_tokens; float scale; uint max_seq_len; }
struct BatchMatMulPC { uint out_dim; uint in_dim; uint seq_len; uint row_offset; }

fn LlmWeights? load_llm_weights(llm::DeviceContext* ctx, llm::GGUFFile* gf, llm::ModelConfig* config, llm::WeightNames* names, bool encode_only = false) {

    llm::Tensor token_embedding;
    char* embedding_data = null;
    llm::GGUFTensorInfo embedding_info;

    if (encode_only) {
        // Store CPU-side reference for per-token dequant (saves ~1.5GB for large vocabs)
        llm::GGUFTensorInfo* info = gf.find_tensor(names.embedding)!!;
        embedding_info = *info;
        embedding_data = gf.tensor_data_base + (usz)info.offset;
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
    } else if (try output_info = gf.find_tensor(names.output)) {
        output_weight = llm::upload_weight(ctx, output_info, gf.tensor_data_base)!!;
    } else {
        // Tied weights - use embedding
        output_weight = token_embedding;
    }

    LayerWeights[] layers;

    if (encode_only) {
        // Streaming mode: layers loaded on-demand during encode_text
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

            // Load Q/K/V bias weights if configured (e.g. Qwen2.5-VL)
            if (names.wq_bias.len > 0) {
                layers[l].wq_bias = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wq_bias);
                layers[l].wk_bias = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wk_bias);
                layers[l].wv_bias = load_layer_tensor(ctx, gf, names.layer_prefix, l, names.wv_bias);
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
    if (names.wq_bias.len > 0) {
        lw.wq_bias = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wq_bias);
        lw.wk_bias = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wk_bias);
        lw.wv_bias = load_layer_tensor(ctx, gf, names.layer_prefix, layer, names.wv_bias);
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
    if (lw.wq_bias.size_bytes > 0) lw.wq_bias.free();
    if (lw.wk_bias.size_bytes > 0) lw.wk_bias.free();
    if (lw.wv_bias.size_bytes > 0) lw.wv_bias.free();
    lw.ffn_norm.free();
    if (lw.ffn_norm_bias.size_bytes > 0) lw.ffn_norm_bias.free();
    if (lw.wq_norm.size_bytes > 0) lw.wq_norm.free();
    if (lw.wk_norm.size_bytes > 0) lw.wk_norm.free();
    if (lw.ffn_gate.size_bytes > 0) lw.ffn_gate.free();
    lw.ffn_up.free();
    lw.ffn_down.free();
}

fn LlmActivations? allocate_llm_activations(llm::DeviceContext* ctx, llm::ModelConfig* config, uint max_seq_len = MAX_SEQ_LEN, bool encode_only = false) {

    ulong dim = config.dim;
    ulong ffn_dim = config.ffn_dim;
    ulong n_heads = config.n_heads;
    ulong q_dim = (ulong)config.n_heads * config.head_dim;  // may differ from dim (e.g. Qwen3)

    ulong[4] dim_shape = { dim, 0, 0, 0 };
    ulong[4] ffn_shape = { ffn_dim, 0, 0, 0 };
    ulong[4] q_shape = { q_dim, 0, 0, 0 };
    ulong[4] scores_shape = { n_heads * max_seq_len, 0, 0, 0 };

    llm::Tensor logits;
    if (!encode_only) {
        ulong[4] vocab_shape = { (ulong)config.vocab_size, 0, 0, 0 };
        logits = llm::create_f32_tensor(ctx, vocab_shape, 1)!!;
    }

    return {
        .hidden = llm::create_f32_tensor(ctx, dim_shape, 1)!!,
        .norm_out = llm::create_f32_tensor(ctx, dim_shape, 1)!!,
        .q = llm::create_f32_tensor(ctx, q_shape, 1)!!,
        .k = llm::create_f32_tensor(ctx, { (ulong)config.n_kv_heads * config.head_dim, 0, 0, 0 }, 1)!!,
        .v = llm::create_f32_tensor(ctx, { (ulong)config.n_kv_heads * config.head_dim, 0, 0, 0 }, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, q_shape, 1)!!,
        .ffn_gate_out = llm::create_f32_tensor(ctx, ffn_shape, 1)!!,
        .ffn_up_out = llm::create_f32_tensor(ctx, ffn_shape, 1)!!,
        .ffn_down_out = llm::create_f32_tensor(ctx, dim_shape, 1)!!,
        .logits = logits,
        .attn_scores = llm::create_f32_tensor(ctx, scores_shape, 1)!!,
        .attn_partial = llm::create_f32_tensor(ctx,
            { n_heads * llm::ceil_div(max_seq_len, ATTN_SPLIT_CHUNK) * config.head_dim, 0, 0, 0 }, 1)!!,
        .attn_meta = llm::create_f32_tensor(ctx,
            { n_heads * llm::ceil_div(max_seq_len, ATTN_SPLIT_CHUNK) * 2, 0, 0, 0 }, 1)!!,
    };
}

fn KVCache? allocate_kv_cache(llm::DeviceContext* ctx, llm::ModelConfig* config, uint max_seq_len = MAX_SEQ_LEN) {

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
    char[] spv = &LLM_SPV;
    ShaderModule shader = vk::shaderModuleCreateInfo()
        .setCodeSize(spv.len)
        .setCode((uint*)&spv[0])
        .build(ctx.device)!!;

    LlmKernels kernels = {
        .embedding    = llm::create_kernel(ctx, shader, 2, EmbeddingPC::size, "embedding")!!,
        .rope         = llm::create_kernel(ctx, shader, 1, RoPEPC::size, "rope")!!,
        .attention    = llm::create_kernel(ctx, shader, 5, AttentionPC::size, "attention")!!,
        .silu_mul         = llm::create_kernel(ctx, shader, 2, SiluMulPC::size, "silu_mul")!!,
        .residual_rmsnorm = llm::create_kernel(ctx, shader, 4, ResidualRMSNormPC::size, "residual_rmsnorm")!!,
        .head_norm_rope   = llm::create_kernel(ctx, shader, 2, HeadNormRopePC::size, "head_norm_rope")!!,
        .attention_split  = llm::create_kernel(ctx, shader, 5, AttentionSplitPC::size, "attention_split")!!,
        .attention_combine = llm::create_kernel(ctx, shader, 3, AttentionCombinePC::size, "attention_combine")!!,
        .matmul_q8_gemv   = llm::create_kernel(ctx, shader, 3, llm::MatMulPC::size, "matmul_q8_gemv")!!,
        .embedding_batch  = llm::create_kernel(ctx, shader, 2, EmbeddingBatchPC::size, "embedding_batch")!!,
        .kv_cache_store_multi = llm::create_kernel(ctx, shader, 4, KVStoreMultiPC::size, "kv_cache_store_multi")!!,
        .matmul_q8_gemv_multi = llm::create_kernel(ctx, shader, 3, MatMulMultiPC::size, "matmul_q8_gemv_multi")!!,
        .rmsnorm_batch      = llm::create_kernel(ctx, shader, 3, RMSNormBatchPC::size, "rmsnorm_batch")!!,
        .head_rmsnorm_batch = llm::create_kernel(ctx, shader, 2, HeadRMSNormBatchPC::size, "head_rmsnorm_batch")!!,
        .rope_batch         = llm::create_kernel(ctx, shader, 1, RoPEBatchPC::size, "rope_batch")!!,
        .kv_cache_store     = llm::create_kernel(ctx, shader, 4, KVStorePC::size, "kv_cache_store")!!,
        .attention_prefill  = llm::create_kernel(ctx, shader, 4, AttentionPrefillPC::size, "attention_prefill")!!,
        .batch_matmul_simple = llm::create_kernel(ctx, shader, 3, BatchMatMulPC::size, "batch_matmul_simple")!!,
        .batch_matmul_q8    = llm::create_kernel(ctx, shader, 3, BatchMatMulPC::size, "batch_matmul_q8")!!,
        .batch_matmul_q4_0  = llm::create_kernel(ctx, shader, 3, BatchMatMulPC::size, "batch_matmul_q4_0")!!,
        .batch_matmul_q4k   = llm::create_kernel(ctx, shader, 3, BatchMatMulPC::size, "batch_matmul_q4k")!!,
        .batch_matmul_q5k   = llm::create_kernel(ctx, shader, 3, BatchMatMulPC::size, "batch_matmul_q5k")!!,
        .batch_matmul_q6k   = llm::create_kernel(ctx, shader, 3, BatchMatMulPC::size, "batch_matmul_q6k")!!,
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
            usage: BUFFER_USAGE_TRANSFER_SRC_BIT,
            properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data: null,
            data_size: (usz)config.dim * 4,
        )!!;
    }


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
// Appends ops to the provided RenderGraph; the caller owns the submission frame.
fn void? LlmModel.forward_layer(&self, llm::RenderGraph* g, LayerWeights* lw, uint layer, uint position) {
    llm::ModelConfig* c = &self.config;
    LlmActivations* a = &self.acts;
    KVCache* kv = &self.kv;
    LlmKernels* k = &self.kernels;
    llm::SharedKernels* sk = &k.shared;

    // Norm(hidden, attn_norm) -> norm_out
    if (c.norm_type == RMSNORM) {
        llm::RMSNormPC rms_pc = { .dim = c.dim, .eps = c.rms_eps };
        g.dispatch(&sk.rmsnorm, { &a.hidden, &lw.attn_norm, &a.norm_out }, &rms_pc, 1);
    } else {
        llm::LayerNormPC ln_pc = { .dim = c.dim, .eps = c.rms_eps };
        g.dispatch(&sk.layernorm,
            { &a.hidden, &lw.attn_norm, &lw.attn_norm_bias, &a.norm_out },
            &ln_pc, 1);
    }

    // Q/K/V projections
    // Q output = n_heads * head_dim (may differ from dim, e.g. Qwen3: dim=2560 but Q=32*128=4096)
    uint q_dim = c.n_heads * c.head_dim;
    uint kv_dim = c.n_kv_heads * c.head_dim;
    // All three read the same norm and write elsewhere, so they need one
    // barrier between them and the norm, not three between each other.
    llm::dispatch_matmul(g, sk, &lw.wq, &a.norm_out, &a.q, q_dim, c.dim);
    llm::dispatch_matmul(g, sk, &lw.wk, &a.norm_out, &a.k, kv_dim, c.dim, independent: true);
    llm::dispatch_matmul(g, sk, &lw.wv, &a.norm_out, &a.v, kv_dim, c.dim, independent: true);

    // Q/K/V bias addition (e.g. Qwen2.5-VL)
    if (lw.wq_bias.size_bytes > 0) {
        llm::ResidualPC q_bias_pc = { .n = q_dim };
        g.dispatch(&sk.residual_add, { &a.q, &lw.wq_bias }, &q_bias_pc, llm::ceil_div(q_dim, 256));
        llm::ResidualPC k_bias_pc = { .n = kv_dim };
        g.dispatch(&sk.residual_add, { &a.k, &lw.wk_bias }, &k_bias_pc, llm::ceil_div(kv_dim, 256));
        llm::ResidualPC v_bias_pc = { .n = kv_dim };
        g.dispatch(&sk.residual_add, { &a.v, &lw.wv_bias }, &v_bias_pc, llm::ceil_div(kv_dim, 256));
    }

    // Q/K norm (per-head RMSNorm, e.g. Qwen3). The same in-place kernel the
    // prefill path uses, with one row: a head is a row whether the batch is
    // 512 positions or the single token being decoded. The staged version
    // this replaces cost 3 commands and 3 full pipeline barriers per head —
    // (n_heads + n_kv_heads) * 3 per layer, on 128-element norms, which on a
    // 0.6B model is most of the decode's command stream.
    if (lw.wq_norm.size_bytes > 0) {
        HeadRMSNormBatchPC qn_pc = { .head_dim = c.head_dim, .n_heads = c.n_heads, .n_rows = 1, .eps = c.rms_eps };
        g.dispatch(&k.head_rmsnorm_batch, { &a.q, &lw.wq_norm }, &qn_pc, llm::ceil_div(c.n_heads, 256));
        HeadRMSNormBatchPC kn_pc = { .head_dim = c.head_dim, .n_heads = c.n_kv_heads, .n_rows = 1, .eps = c.rms_eps };
        g.dispatch(&k.head_rmsnorm_batch, { &a.k, &lw.wk_norm }, &kn_pc, llm::ceil_div(c.n_kv_heads, 256), independent: true);
    }

    // RoPE on Q and K — one touches Q, the other K, so the barrier belongs in
    // front of the pair rather than between them.
    RoPEPC rope_q_pc = { .dim = c.head_dim, .n_heads = c.n_heads, .position = position, .theta = c.rope_theta };
    g.dispatch(&k.rope, { &a.q }, &rope_q_pc,
        llm::ceil_div(c.n_heads * (c.head_dim / 2), 128));

    RoPEPC rope_k_pc = { .dim = c.head_dim, .n_heads = c.n_kv_heads, .position = position, .theta = c.rope_theta };
    g.dispatch(&k.rope, { &a.k }, &rope_k_pc,
        llm::ceil_div(c.n_kv_heads * (c.head_dim / 2), 128), independent: true);

    // Scatter K, V into the cache at `position` — the prefill's kernel with a
    // one-row batch, so the 2 * n_kv_heads little region copies this used to
    // record become a single dispatch.
    KVStorePC kvs_pc = {
        .n_tokens = 1,
        .kv_dim = kv_dim,
        .head_dim = c.head_dim,
        .max_seq_len = self.max_seq_len,
        .pos_offset = position,
    };
    g.dispatch(&k.kv_cache_store,
        { &a.k, &a.v, &kv.k_cache[layer], &kv.v_cache[layer] },
        &kvs_pc, llm::ceil_div(kv_dim, 256));

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
    g.dispatch(&k.attention,
        { &a.q, &kv.k_cache[layer], &kv.v_cache[layer], &a.attn_scores, &a.attn_out },
        &attn_pc, c.n_heads);

    // Output projection: attn_out [q_dim] -> norm_out [dim]
    llm::dispatch_matmul(g, sk, &lw.wo, &a.attn_out, &a.norm_out, c.dim, q_dim);

    // Residual add: hidden += attn_output
    llm::ResidualPC res_pc = { .n = c.dim };
    g.dispatch(&sk.residual_add, { &a.hidden, &a.norm_out }, &res_pc, llm::ceil_div(c.dim, 256));

    // Norm for FFN
    if (c.norm_type == RMSNORM) {
        llm::RMSNormPC rms_pc2 = { .dim = c.dim, .eps = c.rms_eps };
        g.dispatch(&sk.rmsnorm, { &a.hidden, &lw.ffn_norm, &a.norm_out }, &rms_pc2, 1);
    } else {
        llm::LayerNormPC ln_pc2 = { .dim = c.dim, .eps = c.rms_eps };
        g.dispatch(&sk.layernorm,
            { &a.hidden, &lw.ffn_norm, &lw.ffn_norm_bias, &a.norm_out },
            &ln_pc2, 1);
    }

    // FFN
    if (c.ffn_type == SWIGLU) {
        llm::dispatch_matmul(g, sk, &lw.ffn_gate, &a.norm_out, &a.ffn_gate_out, c.ffn_dim, c.dim);
        llm::dispatch_matmul(g, sk, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, c.dim, independent: true);

        llm::SiluPC silu_pc = { .n = c.ffn_dim };
        g.dispatch(&sk.silu, { &a.ffn_gate_out }, &silu_pc, llm::ceil_div(c.ffn_dim, 256));

        llm::ElemwisePC emul_pc = { .n = c.ffn_dim };
        g.dispatch(&sk.elemwise_mul, { &a.ffn_gate_out, &a.ffn_up_out },
            &emul_pc, llm::ceil_div(c.ffn_dim, 256));

        llm::dispatch_matmul(g, sk, &lw.ffn_down, &a.ffn_gate_out, &a.ffn_down_out, c.dim, c.ffn_dim);
    } else {
        llm::dispatch_matmul(g, sk, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, c.dim);

        llm::GeluPC gelu_pc = { .n = c.ffn_dim };
        g.dispatch(&sk.gelu, { &a.ffn_up_out }, &gelu_pc, llm::ceil_div(c.ffn_dim, 256));

        llm::dispatch_matmul(g, sk, &lw.ffn_down, &a.ffn_up_out, &a.ffn_down_out, c.dim, c.ffn_dim);
    }

    // Residual add: hidden += ffn_down_out
    g.dispatch(&sk.residual_add, { &a.hidden, &a.ffn_down_out }, &res_pc, llm::ceil_div(c.dim, 256));
}

// Decode GEMV router: Q8_0 goes to the eight-rows-per-workgroup kernel, other
// dtypes to the shared one-row-per-workgroup matmuls. Decode path only — the
// prefill's tiled batch kernels are a different regime.
fn void dispatch_gemv(
    llm::RenderGraph* g,
    LlmKernels* k,
    llm::SharedKernels* sk,
    llm::Tensor* weight,
    llm::Tensor* input,
    llm::Tensor* output,
    uint out_dim,
    uint in_dim,
    bool independent = false
) @local {
    if (weight.dtype == llm::GGML_Q8_0) {
        llm::MatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim };
        g.dispatch(&k.matmul_q8_gemv, { weight, input, output }, &pc,
            llm::ceil_div(out_dim, 8), independent: independent);
        return;
    }
    llm::dispatch_matmul(g, sk, weight, input, output, out_dim, in_dim, independent: independent);
}

<*
 One decode layer with the elementwise chains fused.

 The decode loop is bound on dispatch and barrier overhead — the model's whole
 weight traffic would take about a tenth of what a token actually costs — so
 the win here is structural, not arithmetic: 19 dispatches and 13 barriers per
 layer become 14 and 9. The layer expects `norm_out` to already hold the
 attn-normed hidden (the previous layer's fused tail leaves it there; forward()
 seeds it for layer 0), and its own tail norms `hidden` with `next_norm` —
 the next layer's attn norm, or the model's output norm after the last layer.

 Requires RMSNorm + SwiGLU without QKV bias, same gate as the prefill path;
 forward() checks and falls back to forward_layer otherwise.
*>
fn void? LlmModel.forward_layer_fused(&self, llm::RenderGraph* g, LayerWeights* lw, uint layer, uint position, llm::Tensor* next_norm) {
    llm::ModelConfig* c = &self.config;
    LlmActivations* a = &self.acts;
    KVCache* kv = &self.kv;
    LlmKernels* k = &self.kernels;
    llm::SharedKernels* sk = &k.shared;
    uint q_dim = c.n_heads * c.head_dim;
    uint kv_dim = c.n_kv_heads * c.head_dim;

    // Q/K/V off the pre-normed input; siblings, one barrier for the group.
    dispatch_gemv(g, k, sk, &lw.wq, &a.norm_out, &a.q, q_dim, c.dim);
    dispatch_gemv(g, k, sk, &lw.wk, &a.norm_out, &a.k, kv_dim, c.dim, independent: true);
    dispatch_gemv(g, k, sk, &lw.wv, &a.norm_out, &a.v, kv_dim, c.dim, independent: true);

    if (lw.wq_norm.size_bytes > 0) {
        // Per-head norm and rotation in one pass per tensor.
        HeadNormRopePC qh = { .head_dim = c.head_dim, .n_heads = c.n_heads, .n_rows = 1,
            .eps = c.rms_eps, .position = position, .theta = c.rope_theta, .pos_stride = 0 };
        g.dispatch(&k.head_norm_rope, { &a.q, &lw.wq_norm }, &qh, llm::ceil_div(c.n_heads, 256));
        HeadNormRopePC kh = { .head_dim = c.head_dim, .n_heads = c.n_kv_heads, .n_rows = 1,
            .eps = c.rms_eps, .position = position, .theta = c.rope_theta, .pos_stride = 0 };
        g.dispatch(&k.head_norm_rope, { &a.k, &lw.wk_norm }, &kh, llm::ceil_div(c.n_kv_heads, 256), independent: true);
    } else {
        RoPEPC rope_q_pc = { .dim = c.head_dim, .n_heads = c.n_heads, .position = position, .theta = c.rope_theta };
        g.dispatch(&k.rope, { &a.q }, &rope_q_pc, llm::ceil_div(c.n_heads * (c.head_dim / 2), 128));
        RoPEPC rope_k_pc = { .dim = c.head_dim, .n_heads = c.n_kv_heads, .position = position, .theta = c.rope_theta };
        g.dispatch(&k.rope, { &a.k }, &rope_k_pc, llm::ceil_div(c.n_kv_heads * (c.head_dim / 2), 128), independent: true);
    }

    KVStorePC kvs_pc = {
        .n_tokens = 1,
        .kv_dim = kv_dim,
        .head_dim = c.head_dim,
        .max_seq_len = self.max_seq_len,
        .pos_offset = position,
    };
    g.dispatch(&k.kv_cache_store,
        { &a.k, &a.v, &kv.k_cache[layer], &kv.v_cache[layer] },
        &kvs_pc, llm::ceil_div(kv_dim, 256));

    // Flash-decode: (head, chunk) workgroups over the sequence, then a fold.
    // Not the sequential kernel forward_layer uses — with one query there is
    // nothing else to parallelise over, and the sequence is the long axis.
    uint n_chunks = llm::ceil_div(position + 1, ATTN_SPLIT_CHUNK);
    AttentionSplitPC as_pc = {
        .head_dim = c.head_dim,
        .n_kv_heads = c.n_kv_heads,
        .n_q_heads = c.n_heads,
        .seq_len = position + 1,
        .scale = 1.0f / std::math::sqrt((float)c.head_dim),
        .max_seq_len = self.max_seq_len,
        .n_chunks = n_chunks,
    };
    g.dispatch(&k.attention_split,
        { &a.q, &kv.k_cache[layer], &kv.v_cache[layer], &a.attn_partial, &a.attn_meta },
        &as_pc, c.n_heads, n_chunks);
    AttentionCombinePC ac_pc = { .head_dim = c.head_dim, .n_q_heads = c.n_heads, .n_chunks = n_chunks };
    g.dispatch(&k.attention_combine,
        { &a.attn_partial, &a.attn_meta, &a.attn_out },
        &ac_pc, c.n_heads);

    dispatch_gemv(g, k, sk, &lw.wo, &a.attn_out, &a.norm_out, c.dim, q_dim);

    // hidden += attn projection, then norm for the FFN — one dispatch. The
    // delta and the output are the same buffer; the kernel is written for it.
    ResidualRMSNormPC rr_pc = { .dim = c.dim, .eps = c.rms_eps, .n_rows = 1 };
    g.dispatch(&k.residual_rmsnorm, { &a.hidden, &a.norm_out, &lw.ffn_norm, &a.norm_out }, &rr_pc, 1);

    dispatch_gemv(g, k, sk, &lw.ffn_gate, &a.norm_out, &a.ffn_gate_out, c.ffn_dim, c.dim);
    dispatch_gemv(g, k, sk, &lw.ffn_up, &a.norm_out, &a.ffn_up_out, c.ffn_dim, c.dim, independent: true);

    SiluMulPC sm_pc = { .n = c.ffn_dim };
    g.dispatch(&k.silu_mul, { &a.ffn_gate_out, &a.ffn_up_out }, &sm_pc, llm::ceil_div(c.ffn_dim, 256));

    dispatch_gemv(g, k, sk, &lw.ffn_down, &a.ffn_gate_out, &a.ffn_down_out, c.dim, c.ffn_dim);

    // hidden += FFN, then the next layer's attn norm (or the output norm).
    g.dispatch(&k.residual_rmsnorm, { &a.hidden, &a.ffn_down_out, next_norm, &a.norm_out }, &rr_pc, 1);
}

// --- Batched lockstep decode ---
//
// A decode step is a chain of small dispatches whose wall clock is close to
// the dispatch count times ~10 us, not the work inside them: an empty submit
// is ~0.1 ms, the full step ~5 ms across ~450 dispatches. Independent
// sequences at the same position can therefore share the chain — but only if
// they share the dispatches too: recording the stages once per stream was
// measured at ~7 us per extra dispatch, five times slower per step than one
// stream. So the streams live batch-major in one set of [n, ...] tensors and
// every stage is a single dispatch. This is what a seed search is: one
// prefix decoded under different rngs, in lockstep.

struct BatchDecode {
    uint n;
    llm::Tensor hidden;         // [n, dim]
    llm::Tensor norm_out;
    llm::Tensor q;              // [n, q_dim]
    llm::Tensor k;              // [n, kv_dim]
    llm::Tensor v;
    llm::Tensor attn_out;       // [n, q_dim]
    llm::Tensor ffn_gate_out;   // [n, ffn_dim]
    llm::Tensor ffn_up_out;
    llm::Tensor ffn_down_out;   // [n, dim]
    llm::Tensor logits;         // [n, vocab]
    llm::Tensor attn_partial;   // [n, heads, chunks, head_dim]
    llm::Tensor attn_meta;      // [n, heads, chunks, 2]
    llm::Tensor[] k_cache;      // per layer: [n, kv_heads, max_seq, head_dim]
    llm::Tensor[] v_cache;
}

const uint MAX_BATCH_DECODE = 16;   // embedding_batch carries the ids in push constants

fn BatchDecode? LlmModel.create_batch_decode(&self, uint n) {
    llm::ModelConfig* c = &self.config;
    llm::DeviceContext* ctx = self.ctx;
    if (n == 0 || n > MAX_BATCH_DECODE) return llm::COMPUTE_ERROR~;
    if (self.encode_only || c.norm_type != RMSNORM || c.ffn_type != SWIGLU
        || c.n_layers == 0 || self.weights.layers[0].wq_bias.size_bytes > 0) return llm::COMPUTE_ERROR~;

    ulong nn = n;
    ulong q_dim = (ulong)c.n_heads * c.head_dim;
    ulong kv_dim = (ulong)c.n_kv_heads * c.head_dim;
    ulong chunks = llm::ceil_div(self.max_seq_len, ATTN_SPLIT_CHUNK);

    BatchDecode bd = { .n = n };
    bd.hidden = llm::create_f32_tensor(ctx, { nn * c.dim, 0, 0, 0 }, 1)!;
    bd.norm_out = llm::create_f32_tensor(ctx, { nn * c.dim, 0, 0, 0 }, 1)!;
    bd.q = llm::create_f32_tensor(ctx, { nn * q_dim, 0, 0, 0 }, 1)!;
    bd.k = llm::create_f32_tensor(ctx, { nn * kv_dim, 0, 0, 0 }, 1)!;
    bd.v = llm::create_f32_tensor(ctx, { nn * kv_dim, 0, 0, 0 }, 1)!;
    bd.attn_out = llm::create_f32_tensor(ctx, { nn * q_dim, 0, 0, 0 }, 1)!;
    bd.ffn_gate_out = llm::create_f32_tensor(ctx, { nn * c.ffn_dim, 0, 0, 0 }, 1)!;
    bd.ffn_up_out = llm::create_f32_tensor(ctx, { nn * c.ffn_dim, 0, 0, 0 }, 1)!;
    bd.ffn_down_out = llm::create_f32_tensor(ctx, { nn * c.dim, 0, 0, 0 }, 1)!;
    bd.logits = llm::create_f32_tensor(ctx, { nn * c.vocab_size, 0, 0, 0 }, 1)!;
    bd.attn_partial = llm::create_f32_tensor(ctx, { nn * c.n_heads * chunks * c.head_dim, 0, 0, 0 }, 1)!;
    bd.attn_meta = llm::create_f32_tensor(ctx, { nn * c.n_heads * chunks * 2, 0, 0, 0 }, 1)!;

    bd.k_cache = mem::new_array(llm::Tensor, c.n_layers);
    bd.v_cache = mem::new_array(llm::Tensor, c.n_layers);
    ulong slab = nn * kv_dim * self.max_seq_len;
    for (uint l = 0; l < c.n_layers; l++) {
        bd.k_cache[l] = llm::create_f32_tensor(ctx, { slab, 0, 0, 0 }, 1)!;
        bd.v_cache[l] = llm::create_f32_tensor(ctx, { slab, 0, 0, 0 }, 1)!;
    }
    return bd;
}

fn void BatchDecode.free(&self) {
    self.hidden.free();
    self.norm_out.free();
    self.q.free();
    self.k.free();
    self.v.free();
    self.attn_out.free();
    self.ffn_gate_out.free();
    self.ffn_up_out.free();
    self.ffn_down_out.free();
    self.logits.free();
    self.attn_partial.free();
    self.attn_meta.free();
    foreach (&tensor : self.k_cache) tensor.free();
    foreach (&tensor : self.v_cache) tensor.free();
    if (self.k_cache.len > 0) mem::free(self.k_cache);
    if (self.v_cache.len > 0) mem::free(self.v_cache);
    *self = {};
}

// Batched decode GEMV: the fp32 eight-rows-per-workgroup kernel with gid.y
// as the stream, so each row's arithmetic — and therefore each stream's
// tokens — is exactly the single-stream kernel's. The tiled batch_matmul
// path is wrong twice over here: its fp16 LDS tiles drift against the
// sequential decode, and its 64-row tiles do eight times the work for an
// eight-row batch.
fn void? dispatch_gemv_batch(
    llm::RenderGraph* g,
    LlmKernels* k,
    llm::Tensor* weight,
    llm::Tensor* input,
    llm::Tensor* output,
    uint out_dim,
    uint in_dim,
    uint n_rows,
    bool independent = false
) @local {
    if (weight.dtype != llm::GGML_Q8_0) return llm::COMPUTE_ERROR~;
    MatMulMultiPC pc = { .out_dim = out_dim, .in_dim = in_dim, .n_rows = n_rows };
    g.dispatch(&k.matmul_q8_gemv_multi, { weight, input, output }, &pc,
        llm::ceil_div(out_dim, 8), independent: independent);
}

<*
 Seed every stream's cache slab with positions [0, n_prefix) of the model's
 own KV cache — the state a prefill just left there.
*>
fn void? LlmModel.batch_load_prefix(&self, BatchDecode* bd, uint n_prefix) {
    llm::ModelConfig* c = &self.config;
    uint heads = c.n_kv_heads;
    if (heads > 32) return llm::COMPUTE_ERROR~;

    ulong head_stride = (ulong)self.max_seq_len * c.head_dim * 4;
    ulong length = (ulong)n_prefix * c.head_dim * 4;

    self.ctx.device.@single_time_command(self.ctx.compute_queue; CommandBuffer cmd) {
        for (uint l = 0; l < c.n_layers; l++) {
            for (uint row = 0; row < bd.n; row++) {
                BufferCopy[32] regions;
                for (uint h = 0; h < heads; h++) {
                    regions[h] = {
                        .srcOffset = (ulong)h * head_stride,
                        .dstOffset = ((ulong)row * heads + h) * head_stride,
                        .size = length,
                    };
                }
                vk::cmdCopyBuffer(cmd, self.kv.k_cache[l].gpu_buffer.buffer,
                    bd.k_cache[l].gpu_buffer.buffer, heads, regions[:heads]);
                vk::cmdCopyBuffer(cmd, self.kv.v_cache[l].gpu_buffer.buffer,
                    bd.v_cache[l].gpu_buffer.buffer, heads, regions[:heads]);
            }
        }
    }!!;
}

<*
 One lockstep decode step for the whole batch: every stage is a single
 dispatch over [n, ...] tensors. The step's logits land in `logits_staging`
 — stream s at byte offset s * vocab_size * 4 — inside the same submission,
 ready to sample from when this returns.
*>
fn void? LlmModel.forward_batch(&self, BatchDecode* bd, uint[] tokens, uint position, vk::Memory* logits_staging) {
    llm::ModelConfig* c = &self.config;
    LlmWeights* w = &self.weights;
    LlmKernels* k = &self.kernels;
    llm::SharedKernels* sk = &k.shared;
    uint n = bd.n;
    uint q_dim = c.n_heads * c.head_dim;
    uint kv_dim = c.n_kv_heads * c.head_dim;

    llm::RenderGraph g;
    g.init(self.ctx)!!;

    EmbeddingBatchPC emb_pc = { .dim = c.dim, .n_rows = n };
    foreach (usz s, token : tokens) emb_pc.tokens[s] = token;
    g.dispatch(&k.embedding_batch, { &w.token_embedding, &bd.hidden }, &emb_pc,
        llm::ceil_div(n * c.dim, 256));

    RMSNormBatchPC first_rms = { .dim = c.dim, .eps = c.rms_eps, .n_rows = n };
    g.dispatch(&k.rmsnorm_batch, { &bd.hidden, &w.layers[0].attn_norm, &bd.norm_out }, &first_rms, n);

    ResidualRMSNormPC rr_pc = { .dim = c.dim, .eps = c.rms_eps, .n_rows = n };
    uint n_chunks = llm::ceil_div(position + 1, ATTN_SPLIT_CHUNK);

    for (uint l = 0; l < c.n_layers; l++) {
        LayerWeights* lw = &w.layers[l];
        llm::Tensor* next_norm = l + 1 < c.n_layers ? &w.layers[l + 1].attn_norm : &w.output_norm;

        dispatch_gemv_batch(&g, k, &lw.wq, &bd.norm_out, &bd.q, q_dim, c.dim, n)!;
        dispatch_gemv_batch(&g, k, &lw.wk, &bd.norm_out, &bd.k, kv_dim, c.dim, n, independent: true)!;
        dispatch_gemv_batch(&g, k, &lw.wv, &bd.norm_out, &bd.v, kv_dim, c.dim, n, independent: true)!;

        if (lw.wq_norm.size_bytes > 0) {
            HeadNormRopePC qh = { .head_dim = c.head_dim, .n_heads = c.n_heads, .n_rows = n,
                .eps = c.rms_eps, .position = position, .theta = c.rope_theta, .pos_stride = 0 };
            g.dispatch(&k.head_norm_rope, { &bd.q, &lw.wq_norm }, &qh,
                llm::ceil_div(n * c.n_heads, 256));
            HeadNormRopePC kh = { .head_dim = c.head_dim, .n_heads = c.n_kv_heads, .n_rows = n,
                .eps = c.rms_eps, .position = position, .theta = c.rope_theta, .pos_stride = 0 };
            g.dispatch(&k.head_norm_rope, { &bd.k, &lw.wk_norm }, &kh,
                llm::ceil_div(n * c.n_kv_heads, 256), independent: true);
        } else {
            return llm::COMPUTE_ERROR~;   // no arch without Q/K norm reaches this path today
        }

        KVStoreMultiPC kvm_pc = { .n_rows = n, .kv_dim = kv_dim, .head_dim = c.head_dim,
            .max_seq_len = self.max_seq_len, .position = position };
        g.dispatch(&k.kv_cache_store_multi,
            { &bd.k, &bd.v, &bd.k_cache[l], &bd.v_cache[l] },
            &kvm_pc, llm::ceil_div(n * kv_dim, 256));

        AttentionSplitPC as_pc = {
            .head_dim = c.head_dim,
            .n_kv_heads = c.n_kv_heads,
            .n_q_heads = c.n_heads,
            .seq_len = position + 1,
            .scale = 1.0f / std::math::sqrt((float)c.head_dim),
            .max_seq_len = self.max_seq_len,
            .n_chunks = n_chunks,
        };
        g.dispatch(&k.attention_split,
            { &bd.q, &bd.k_cache[l], &bd.v_cache[l], &bd.attn_partial, &bd.attn_meta },
            &as_pc, c.n_heads, n_chunks, n);
        AttentionCombinePC ac_pc = { .head_dim = c.head_dim, .n_q_heads = c.n_heads, .n_chunks = n_chunks };
        g.dispatch(&k.attention_combine,
            { &bd.attn_partial, &bd.attn_meta, &bd.attn_out },
            &ac_pc, c.n_heads, n);

        dispatch_gemv_batch(&g, k, &lw.wo, &bd.attn_out, &bd.norm_out, c.dim, q_dim, n)!;

        g.dispatch(&k.residual_rmsnorm,
            { &bd.hidden, &bd.norm_out, &lw.ffn_norm, &bd.norm_out }, &rr_pc, n);

        dispatch_gemv_batch(&g, k, &lw.ffn_gate, &bd.norm_out, &bd.ffn_gate_out, c.ffn_dim, c.dim, n)!;
        dispatch_gemv_batch(&g, k, &lw.ffn_up, &bd.norm_out, &bd.ffn_up_out, c.ffn_dim, c.dim, n, independent: true)!;
        SiluMulPC sm_pc = { .n = n * c.ffn_dim };
        g.dispatch(&k.silu_mul, { &bd.ffn_gate_out, &bd.ffn_up_out }, &sm_pc,
            llm::ceil_div(n * c.ffn_dim, 256));
        dispatch_gemv_batch(&g, k, &lw.ffn_down, &bd.ffn_gate_out, &bd.ffn_down_out, c.dim, c.ffn_dim, n)!;

        g.dispatch(&k.residual_rmsnorm,
            { &bd.hidden, &bd.ffn_down_out, next_norm, &bd.norm_out }, &rr_pc, n);
    }

    dispatch_gemv_batch(&g, k, &w.output, &bd.norm_out, &bd.logits, c.vocab_size, c.dim, n)!;

    // The logits ride home in the same submission.
    ulong logits_bytes = (ulong)c.vocab_size * 4;
    if (g.needs_barrier) llm::compute_barrier(g.cmd);
    vk::cmdCopyBuffer(g.cmd, bd.logits.gpu_buffer.buffer, logits_staging.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)n * logits_bytes }});
    g.needs_barrier = true;

    g.flush()!!;
}

// --- Batched prefill path ---
// Processes all prompt positions of a layer with batched GEMMs / norms /
// RoPE / attention instead of one forward_layer call per token. Used by
// encode_text_multilayer; ~15 dispatches per layer instead of ~150 commands
// per token position.

struct PrefillActivations {
    llm::Tensor norm_out;
    llm::Tensor q;
    llm::Tensor k;
    llm::Tensor v;
    llm::Tensor attn_out;
    llm::Tensor ffn_gate_out;
    llm::Tensor ffn_up_out;
    llm::Tensor ffn_down_out;
}

fn PrefillActivations? allocate_prefill_activations(llm::DeviceContext* ctx, llm::ModelConfig* config, uint n_tokens) {
    ulong n = n_tokens;
    ulong dim = config.dim;
    ulong q_dim = (ulong)config.n_heads * config.head_dim;
    ulong kv_dim = (ulong)config.n_kv_heads * config.head_dim;
    ulong ffn_dim = config.ffn_dim;
    return {
        .norm_out     = llm::create_f32_tensor(ctx, { n * dim, 0, 0, 0 }, 1)!!,
        .q            = llm::create_f32_tensor(ctx, { n * q_dim, 0, 0, 0 }, 1)!!,
        .k            = llm::create_f32_tensor(ctx, { n * kv_dim, 0, 0, 0 }, 1)!!,
        .v            = llm::create_f32_tensor(ctx, { n * kv_dim, 0, 0, 0 }, 1)!!,
        .attn_out     = llm::create_f32_tensor(ctx, { n * q_dim, 0, 0, 0 }, 1)!!,
        .ffn_gate_out = llm::create_f32_tensor(ctx, { n * ffn_dim, 0, 0, 0 }, 1)!!,
        .ffn_up_out   = llm::create_f32_tensor(ctx, { n * ffn_dim, 0, 0, 0 }, 1)!!,
        .ffn_down_out = llm::create_f32_tensor(ctx, { n * dim, 0, 0, 0 }, 1)!!,
    };
}

fn void PrefillActivations.free(&self) {
    self.norm_out.free();
    self.q.free();
    self.k.free();
    self.v.free();
    self.attn_out.free();
    self.ffn_gate_out.free();
    self.ffn_up_out.free();
    self.ffn_down_out.free();
}

// Arch gate for the prefill path (per-layer bias is checked at dispatch time).
fn bool LlmModel.prefill_supported(&self, uint n_tokens) {
    return self.config.norm_type == RMSNORM
        && self.config.ffn_type == SWIGLU
        && n_tokens <= 1024              // attention_prefill LDS score buffer
        && n_tokens <= self.max_seq_len;
}

fn void dispatch_matmul_batch(
    llm::RenderGraph* g,
    LlmKernels* k,
    llm::Tensor* weight,
    llm::Tensor* input,
    llm::Tensor* output,
    uint out_dim,
    uint in_dim,
    uint n_tokens,
    bool independent = false
) {
    BatchMatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim, .seq_len = n_tokens, .row_offset = 0 };
    llm::ComputeKernel* kernel;
    bool tiled_2d = false;
    if (weight.dtype == llm::GGML_Q8_0) {
        kernel = &k.batch_matmul_q8;
        tiled_2d = true;
    } else if (weight.dtype == llm::GGML_Q4_K) {
        kernel = &k.batch_matmul_q4k;
        tiled_2d = true;
    } else if (weight.dtype == llm::GGML_Q5_K) {
        kernel = &k.batch_matmul_q5k;
        tiled_2d = true;
    } else if (weight.dtype == llm::GGML_Q6_K) {
        kernel = &k.batch_matmul_q6k;
        tiled_2d = true;
    } else if (weight.dtype == llm::GGML_Q4_0) {
        kernel = &k.batch_matmul_q4_0;
    } else {
        // F32/F16-as-F32 weights: 2D dispatch (seq_len, out_dim)
        g.dispatch(&k.batch_matmul_simple, { weight, input, output }, &pc, n_tokens, out_dim, independent: independent);
        return;
    }
    uint groups = tiled_2d
        ? llm::ceil_div(n_tokens, 64) * llm::ceil_div(out_dim, 64)
        : n_tokens * out_dim;
    g.dispatch(kernel, { weight, input, output }, &pc, groups, independent: independent);
}

// Batched layer forward over `hidden` [n_tokens, dim]. Requires RMSNorm +
// SwiGLU without QKV bias (e.g. Qwen3) — callers check prefill_supported()
// and lw.wq_bias before dispatching.
fn void? LlmModel.forward_layer_prefill(&self, llm::RenderGraph* g, LayerWeights* lw, uint layer, llm::Tensor* hidden, PrefillActivations* pa, uint n_tokens) {
    llm::ModelConfig* c = &self.config;
    LlmKernels* k = &self.kernels;
    llm::SharedKernels* sk = &k.shared;
    uint q_dim = c.n_heads * c.head_dim;
    uint kv_dim = c.n_kv_heads * c.head_dim;

    // Attention norm
    RMSNormBatchPC rms_pc = { .dim = c.dim, .eps = c.rms_eps, .n_rows = n_tokens };
    g.dispatch(&k.rmsnorm_batch, { hidden, &lw.attn_norm, &pa.norm_out }, &rms_pc, n_tokens);

    // Q/K/V projections (batched GEMM)
    dispatch_matmul_batch(g, k, &lw.wq, &pa.norm_out, &pa.q, q_dim, c.dim, n_tokens);
    dispatch_matmul_batch(g, k, &lw.wk, &pa.norm_out, &pa.k, kv_dim, c.dim, n_tokens, independent: true);
    dispatch_matmul_batch(g, k, &lw.wv, &pa.norm_out, &pa.v, kv_dim, c.dim, n_tokens, independent: true);

    // Q/K per-head RMSNorm (e.g. Qwen3) — two dispatches instead of
    // 3 commands x heads per position
    if (lw.wq_norm.size_bytes > 0) {
        HeadRMSNormBatchPC qn_pc = { .head_dim = c.head_dim, .n_heads = c.n_heads, .n_rows = n_tokens, .eps = c.rms_eps };
        g.dispatch(&k.head_rmsnorm_batch, { &pa.q, &lw.wq_norm }, &qn_pc, llm::ceil_div(n_tokens * c.n_heads, 256));
        HeadRMSNormBatchPC kn_pc = { .head_dim = c.head_dim, .n_heads = c.n_kv_heads, .n_rows = n_tokens, .eps = c.rms_eps };
        g.dispatch(&k.head_rmsnorm_batch, { &pa.k, &lw.wk_norm }, &kn_pc, llm::ceil_div(n_tokens * c.n_kv_heads, 256), independent: true);
    }

    // RoPE (position = row index)
    RoPEBatchPC rope_q_pc = { .dim = c.head_dim, .n_heads = c.n_heads, .n_tokens = n_tokens, .theta = c.rope_theta };
    g.dispatch(&k.rope_batch, { &pa.q }, &rope_q_pc, llm::ceil_div(n_tokens * c.n_heads * (c.head_dim / 2), 128));
    RoPEBatchPC rope_k_pc = { .dim = c.head_dim, .n_heads = c.n_kv_heads, .n_tokens = n_tokens, .theta = c.rope_theta };
    g.dispatch(&k.rope_batch, { &pa.k }, &rope_k_pc, llm::ceil_div(n_tokens * c.n_kv_heads * (c.head_dim / 2), 128), independent: true);

    // KV cache fill (one dispatch instead of 2 copies x kv_heads x positions)
    KVStorePC kvs_pc = { .n_tokens = n_tokens, .kv_dim = kv_dim, .head_dim = c.head_dim, .max_seq_len = self.max_seq_len };
    g.dispatch(&k.kv_cache_store,
        { &pa.k, &pa.v, &self.kv.k_cache[layer], &self.kv.v_cache[layer] },
        &kvs_pc, llm::ceil_div(n_tokens * kv_dim, 256));

    // Causal attention over all positions: one workgroup per (head, position)
    AttentionPrefillPC apf_pc = {
        .head_dim = c.head_dim,
        .n_kv_heads = c.n_kv_heads,
        .n_q_heads = c.n_heads,
        .n_tokens = n_tokens,
        .scale = 1.0f / std::math::sqrt((float)c.head_dim),
        .max_seq_len = self.max_seq_len,
    };
    g.dispatch(&k.attention_prefill,
        { &pa.q, &self.kv.k_cache[layer], &self.kv.v_cache[layer], &pa.attn_out },
        &apf_pc, c.n_heads, n_tokens);

    // Output projection + residual
    dispatch_matmul_batch(g, k, &lw.wo, &pa.attn_out, &pa.norm_out, c.dim, q_dim, n_tokens);
    llm::ResidualPC res_pc = { .n = n_tokens * c.dim };
    g.dispatch(&sk.residual_add, { hidden, &pa.norm_out }, &res_pc, llm::ceil_div(n_tokens * c.dim, 256));

    // FFN norm
    g.dispatch(&k.rmsnorm_batch, { hidden, &lw.ffn_norm, &pa.norm_out }, &rms_pc, n_tokens);

    // FFN (SwiGLU)
    dispatch_matmul_batch(g, k, &lw.ffn_gate, &pa.norm_out, &pa.ffn_gate_out, c.ffn_dim, c.dim, n_tokens);
    dispatch_matmul_batch(g, k, &lw.ffn_up, &pa.norm_out, &pa.ffn_up_out, c.ffn_dim, c.dim, n_tokens, independent: true);
    llm::SiluPC silu_pc = { .n = n_tokens * c.ffn_dim };
    g.dispatch(&sk.silu, { &pa.ffn_gate_out }, &silu_pc, llm::ceil_div(n_tokens * c.ffn_dim, 256));
    llm::ElemwisePC emul_pc = { .n = n_tokens * c.ffn_dim };
    g.dispatch(&sk.elemwise_mul, { &pa.ffn_gate_out, &pa.ffn_up_out }, &emul_pc, llm::ceil_div(n_tokens * c.ffn_dim, 256));
    dispatch_matmul_batch(g, k, &lw.ffn_down, &pa.ffn_gate_out, &pa.ffn_down_out, c.dim, c.ffn_dim, n_tokens);
    g.dispatch(&sk.residual_add, { hidden, &pa.ffn_down_out }, &res_pc, llm::ceil_div(n_tokens * c.dim, 256));
}

<*
 Copy positions [from, to) of one KV cache into another.

 What beam search needs: a beam that descends from another has to inherit its
 attention state, and re-running the prefix per beam is not an option. The
 cache is laid out head-major — `h * max_seq_len * head_dim + pos * head_dim` —
 so a position range is contiguous within a head and the whole move is one
 region per head.
*>
fn void? LlmModel.copy_kv_range(&self, KVCache* src, KVCache* dst, uint from_pos, uint to_pos) {
    if (to_pos <= from_pos) return;

    llm::ModelConfig* c = &self.config;
    uint heads = c.n_kv_heads;
    if (heads > 32) return llm::COMPUTE_ERROR~;

    ulong head_stride = (ulong)self.max_seq_len * c.head_dim * 4;
    ulong offset = (ulong)from_pos * c.head_dim * 4;
    ulong length = (ulong)(to_pos - from_pos) * c.head_dim * 4;

    BufferCopy[32] regions;
    for (uint h = 0; h < heads; h++) {
        regions[h] = {
            .srcOffset = (ulong)h * head_stride + offset,
            .dstOffset = (ulong)h * head_stride + offset,
            .size = length,
        };
    }

    self.ctx.device.@single_time_command(self.ctx.compute_queue; CommandBuffer cmd) {
        for (uint l = 0; l < c.n_layers; l++) {
            vk::cmdCopyBuffer(cmd, src.k_cache[l].gpu_buffer.buffer,
                dst.k_cache[l].gpu_buffer.buffer, heads, regions[:heads]);
            vk::cmdCopyBuffer(cmd, src.v_cache[l].gpu_buffer.buffer,
                dst.v_cache[l].gpu_buffer.buffer, heads, regions[:heads]);
        }
    }!!;
}

<*
 Swap in a different KV cache, returning the one that was in use.

 Beams take turns on the same weights and activations; only their attention
 state differs, so switching beams is switching caches.
*>
fn void KVCache.free(&self) {
    foreach (&tensor : self.k_cache) tensor.free();
    foreach (&tensor : self.v_cache) tensor.free();
    if (self.k_cache.len > 0) mem::free(self.k_cache);
    if (self.v_cache.len > 0) mem::free(self.v_cache);
    *self = {};
}

fn KVCache LlmModel.use_kv_cache(&self, KVCache cache) {
    KVCache previous = self.kv;
    self.kv = cache;
    return previous;
}

<*
 Run a batch of pre-computed embeddings through the stack instead of a token
 lookup, filling the KV cache for positions [0, n_tokens) and leaving the last
 row's logits in `acts.logits`.

 This is what conditioning on something that is not text needs — SkinTokens
 prefixes a mesh encoder's 512 latents before any token exists. Decoding then
 continues through the ordinary `forward(token, position)` from position
 `n_tokens`, because both paths share the same KV cache.

 `embeddings` is [n_tokens, dim] and doubles as the residual stream, so it is
 overwritten. Requires prefill_supported(n_tokens).
*>
fn void? LlmModel.prefill_embeddings(&self, llm::Tensor* embeddings, uint n_tokens) {
    if (n_tokens == 0 || !self.prefill_supported(n_tokens)) return llm::COMPUTE_ERROR~;
    if (self.encode_only) return llm::COMPUTE_ERROR~;

    llm::ModelConfig* c = &self.config;
    LlmWeights* w = &self.weights;
    LlmActivations* a = &self.acts;
    LlmKernels* k = &self.kernels;
    llm::SharedKernels* sk = &k.shared;

    PrefillActivations pa = allocate_prefill_activations(self.ctx, c, n_tokens)!!;
    defer pa.free();

    llm::RenderGraph g;
    g.init(self.ctx)!!;

    for (uint l = 0; l < c.n_layers; l++) {
        self.forward_layer_prefill(&g, &w.layers[l], l, embeddings, &pa, n_tokens)!!;
        // A whole stack of batched layers is a long command buffer; break it up
        // so a slow device is not asked to chew through all of it in one go.
        if ((l + 1) % 8 == 0 && l + 1 < c.n_layers) g.checkpoint()!!;
    }

    // Only the final position predicts the next token, so the rest of the
    // batch is done and just its row goes on to the output head.
    ulong row_bytes = (ulong)c.dim * 4;
    g.copy_region(embeddings, (usz)((ulong)(n_tokens - 1) * row_bytes), &a.hidden, 0, row_bytes);

    // prefill_supported() already established RMSNorm, so there is no
    // layernorm branch to mirror from forward().
    llm::RMSNormPC final_rms = { .dim = c.dim, .eps = c.rms_eps };
    g.dispatch(&sk.rmsnorm, { &a.hidden, &w.output_norm, &a.norm_out }, &final_rms, 1);
    llm::dispatch_matmul(&g, sk, &w.output, &a.norm_out, &a.logits, c.vocab_size, c.dim);

    g.flush()!!;
}

fn void? LlmModel.forward(&self, uint token, uint position) {
    llm::DeviceContext* ctx = self.ctx;
    llm::ModelConfig* c = &self.config;
    LlmWeights* w = &self.weights;
    LlmActivations* a = &self.acts;
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

    llm::RenderGraph g;
    g.init(ctx)!!;

    if (!self.encode_only) {
        EmbeddingPC emb_pc = { .token_id = token, .dim = c.dim };
        g.dispatch(&k.embedding, { &w.token_embedding, &a.hidden }, &emb_pc,
            llm::ceil_div(c.dim, 256));
    }

    // 2. The transformer stack. The fused path carries the arch gate of the
    // batched prefill: RMSNorm + SwiGLU, no QKV bias. Its layers chain through
    // norm_out — each one's tail norms the hidden for the next — so the first
    // norm is seeded here and the last layer's tail is already the final norm.
    bool fused = !self.encode_only
        && c.norm_type == RMSNORM && c.ffn_type == SWIGLU
        && c.n_layers > 0 && w.layers[0].wq_bias.size_bytes == 0;

    if (fused) {
        llm::RMSNormPC first_rms = { .dim = c.dim, .eps = c.rms_eps };
        g.dispatch(&sk.rmsnorm, { &a.hidden, &w.layers[0].attn_norm, &a.norm_out }, &first_rms, 1);
        for (uint l = 0; l < c.n_layers; l++) {
            llm::Tensor* next_norm = l + 1 < c.n_layers ? &w.layers[l + 1].attn_norm : &w.output_norm;
            self.forward_layer_fused(&g, &w.layers[l], l, position, next_norm)!!;
        }
    } else {
        for (uint l = 0; l < c.n_layers; l++) {
            self.forward_layer(&g, &w.layers[l], l, position)!!;
        }

        // 3. Final norm
        if (c.norm_type == RMSNORM) {
            llm::RMSNormPC final_rms = { .dim = c.dim, .eps = c.rms_eps };
            g.dispatch(&sk.rmsnorm, { &a.hidden, &w.output_norm, &a.norm_out }, &final_rms, 1);
        } else {
            llm::LayerNormPC final_ln = { .dim = c.dim, .eps = c.rms_eps };
            g.dispatch(&sk.layernorm,
                { &a.hidden, &w.output_norm, &w.output_norm_bias, &a.norm_out },
                &final_ln, 1);
        }
    }

    // 4. Output projection -> logits (skip in encode_only mode)
    if (!self.encode_only) {
        dispatch_gemv(&g, k, sk, &w.output, &a.norm_out, &a.logits, c.vocab_size, c.dim);
    }

    // 5. Submit and wait
    g.flush()!!;
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

    // 2. Process layers one at a time (streaming weights)
    // Diffusers uses hidden_states[-2] (second-to-last layer output, no final norm).
    // We save hidden_states after layer n_layers-2 and return those directly.
    uint penultimate_layer = c.n_layers >= 2 ? c.n_layers - 2 : 0;
    llm::Tensor text_embeddings;

    for (uint l = 0; l < c.n_layers; l++) {
        // Upload this layer's weights
        LayerWeights lw = load_single_layer(ctx, self.gf, self.names, c, l)!!;

        // Process all positions through this layer
        llm::RenderGraph g;
        g.init(ctx)!!;
        for (uint pos = 0; pos < seq_len; pos++) {
            // Copy hidden_states[pos] → acts.hidden (offset read, raw cmd)
            if (g.needs_barrier) llm::compute_barrier(g.cmd);
            vk::cmdCopyBuffer(g.cmd, hidden_states.gpu_buffer.buffer, self.acts.hidden.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = (ulong)pos * c.dim * 4, .dstOffset = 0, .size = (ulong)c.dim * 4 }});
            g.needs_barrier = true;

            // Process layer
            self.forward_layer(&g, &lw, l, pos)!!;

            // Copy acts.hidden → hidden_states[pos] (offset write, raw cmd)
            if (g.needs_barrier) llm::compute_barrier(g.cmd);
            vk::cmdCopyBuffer(g.cmd, self.acts.hidden.gpu_buffer.buffer, hidden_states.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)pos * c.dim * 4, .size = (ulong)c.dim * 4 }});
            g.needs_barrier = true;
        }
        g.flush()!!;

        // Free this layer's weights (space reused for next layer)
        free_layer_weights(&lw, c);

        // Snapshot hidden_states after second-to-last layer (hidden_states[-2])
        if (l == penultimate_layer) {
            text_embeddings = llm::create_f32_tensor(ctx, { (ulong)seq_len * c.dim, 0, 0, 0 }, 1)!!;
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer snap_cmd) {
                vk::cmdCopyBuffer(snap_cmd, hidden_states.gpu_buffer.buffer,
                    text_embeddings.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * c.dim * 4 }});
            }!!;
        }
    }

    // Skip final norm — diffusers uses raw hidden_states[-2] without normalization
    hidden_states.free();

    text_embeddings.n_dims = 2;
    text_embeddings.shape = { (ulong)seq_len, (ulong)c.dim, 0, 0 };

    return text_embeddings;
}

// Streaming encode with visual token injection.
// Same as encode_text_streaming, but positions where tokens[pos] == image_pad_id
// get their embeddings from visual_embeddings (GPU tensor [n_visual, dim]) instead
// of from the token embedding table.
fn llm::Tensor? LlmModel.encode_text_with_visual(&self, uint[] tokens, llm::Tensor* visual_embeddings, uint image_pad_id) {
    llm::ModelConfig* c = &self.config;
    llm::DeviceContext* ctx = self.ctx;
    uint seq_len = (uint)tokens.len;

    llm::Tensor hidden_states = llm::create_f32_tensor(ctx, { (ulong)seq_len * c.dim, 0, 0, 0 }, 1)!!;

    // 1. Embed all tokens — visual positions get ViT embeddings, text positions get vocab embeddings
    uint visual_idx = 0;
    for (uint pos = 0; pos < seq_len; pos++) {
        if (tokens[pos] == image_pad_id && visual_embeddings != null) {
            // Copy from ViT output (GPU → GPU)
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer vis_cmd) {
                vk::cmdCopyBuffer(vis_cmd, visual_embeddings.gpu_buffer.buffer, hidden_states.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = (ulong)visual_idx * c.dim * 4, .dstOffset = (ulong)pos * c.dim * 4, .size = (ulong)c.dim * 4 }});
            }!!;
            visual_idx++;
        } else {
            // Normal token embedding via CPU dequant
            float* staging_ptr = (float*)self.embedding_staging.data();
            llm::dequant_embedding_row(self.weights.embedding_data, &self.weights.embedding_info, tokens[pos], staging_ptr, c.dim);

            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer emb_cmd) {
                vk::cmdCopyBuffer(emb_cmd, self.embedding_staging.buffer, hidden_states.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)pos * c.dim * 4, .size = (ulong)c.dim * 4 }});
            }!!;
        }
    }

    io::printfn("  Embedded %d tokens (%d visual, %d text)", seq_len, visual_idx, seq_len - visual_idx);

    // 2. Process layers one at a time (identical to encode_text_streaming)
    uint penultimate_layer = c.n_layers >= 2 ? c.n_layers - 2 : 0;
    llm::Tensor text_embeddings;

    for (uint l = 0; l < c.n_layers; l++) {
        LayerWeights lw = load_single_layer(ctx, self.gf, self.names, c, l)!!;

        llm::RenderGraph g;
        g.init(ctx)!!;
        for (uint pos = 0; pos < seq_len; pos++) {
            if (g.needs_barrier) llm::compute_barrier(g.cmd);
            vk::cmdCopyBuffer(g.cmd, hidden_states.gpu_buffer.buffer, self.acts.hidden.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = (ulong)pos * c.dim * 4, .dstOffset = 0, .size = (ulong)c.dim * 4 }});
            g.needs_barrier = true;

            self.forward_layer(&g, &lw, l, pos)!!;

            if (g.needs_barrier) llm::compute_barrier(g.cmd);
            vk::cmdCopyBuffer(g.cmd, self.acts.hidden.gpu_buffer.buffer, hidden_states.gpu_buffer.buffer, 1,
                (BufferCopy[]){{ .srcOffset = 0, .dstOffset = (ulong)pos * c.dim * 4, .size = (ulong)c.dim * 4 }});
            g.needs_barrier = true;
        }
        g.flush()!!;

        free_layer_weights(&lw, c);

        if (l == penultimate_layer) {
            text_embeddings = llm::create_f32_tensor(ctx, { (ulong)seq_len * c.dim, 0, 0, 0 }, 1)!!;
            ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer snap_cmd) {
                vk::cmdCopyBuffer(snap_cmd, hidden_states.gpu_buffer.buffer,
                    text_embeddings.gpu_buffer.buffer, 1,
                    (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * c.dim * 4 }});
            }!!;
        }
    }

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
    self.kernels.silu_mul.free(self.ctx.device);
    self.kernels.residual_rmsnorm.free(self.ctx.device);
    self.kernels.head_norm_rope.free(self.ctx.device);
    self.kernels.attention_split.free(self.ctx.device);
    self.kernels.attention_combine.free(self.ctx.device);
    self.kernels.matmul_q8_gemv.free(self.ctx.device);
    self.kernels.embedding_batch.free(self.ctx.device);
    self.kernels.kv_cache_store_multi.free(self.ctx.device);
    self.kernels.matmul_q8_gemv_multi.free(self.ctx.device);
    self.kernels.rmsnorm_batch.free(self.ctx.device);
    self.kernels.head_rmsnorm_batch.free(self.ctx.device);
    self.kernels.rope_batch.free(self.ctx.device);
    self.kernels.kv_cache_store.free(self.ctx.device);
    self.kernels.attention_prefill.free(self.ctx.device);
    self.kernels.batch_matmul_simple.free(self.ctx.device);
    self.kernels.batch_matmul_q8.free(self.ctx.device);
    self.kernels.batch_matmul_q4_0.free(self.ctx.device);
    self.kernels.batch_matmul_q4k.free(self.ctx.device);
    self.kernels.batch_matmul_q5k.free(self.ctx.device);
    self.kernels.batch_matmul_q6k.free(self.ctx.device);
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
    self.acts.attn_partial.free();
    self.acts.attn_meta.free();

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
