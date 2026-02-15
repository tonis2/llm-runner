module sdxs;

import vk;
import llm;
import std::io;
import std::math;
import std::core::mem;

struct EmbeddingPC { uint token_id; uint dim; }

// CLIP Text Encoder for Stable Diffusion
// 12-layer transformer with non-causal self-attention
// Processes all 77 token positions at once using batched matmul

struct ClipLayerWeights {
    llm::Tensor ln1_weight;
    llm::Tensor ln1_bias;
    llm::Tensor q_proj;
    llm::Tensor q_proj_bias;
    llm::Tensor k_proj;
    llm::Tensor k_proj_bias;
    llm::Tensor v_proj;
    llm::Tensor v_proj_bias;
    llm::Tensor out_proj;
    llm::Tensor out_proj_bias;
    llm::Tensor ln2_weight;
    llm::Tensor ln2_bias;
    llm::Tensor fc1;
    llm::Tensor fc1_bias;
    llm::Tensor fc2;
    llm::Tensor fc2_bias;
}

struct ClipWeights {
    llm::Tensor token_embedding;     // [vocab_size, dim]
    llm::Tensor position_embedding;  // [max_tokens, dim]
    llm::Tensor final_ln_weight;
    llm::Tensor final_ln_bias;
    ClipLayerWeights[] layers;
}

struct ClipActivations {
    llm::Tensor hidden;      // [max_tokens, dim] = [77, 768]
    llm::Tensor norm_out;    // [max_tokens, dim]
    llm::Tensor q;           // [max_tokens, dim]
    llm::Tensor k;           // [max_tokens, dim]
    llm::Tensor v;           // [max_tokens, dim]
    llm::Tensor attn_out;    // [max_tokens, dim]
    llm::Tensor ffn_out;     // [max_tokens, ffn_dim] (ffn_dim = 4 * dim = 3072)
    llm::Tensor ffn_down_out; // [max_tokens, dim]
    llm::Tensor scores;      // [n_heads, max_tokens, max_tokens]
}

struct ClipEncoder {
    llm::ClipConfig config;
    ClipWeights weights;
    ClipActivations acts;
    llm::DiffusionKernels* kernels;
    llm::DeviceContext* ctx;
}

fn String clip_tensor_name(char[128]* buf, String prefix, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) {
        p[pos + i] = prefix[i];
    }
    pos += prefix.len;
    for (usz i = 0; i < suffix.len; i++) {
        p[pos + i] = suffix[i];
    }
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

fn String clip_layer_tensor_name(char[128]* buf, String prefix, uint layer, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    // prefix
    for (usz i = 0; i < prefix.len; i++) p[pos + i] = prefix[i];
    pos += prefix.len;
    // "encoder.layers."
    String mid = "encoder.layers.";
    for (usz i = 0; i < mid.len; i++) p[pos + i] = mid[i];
    pos += mid.len;
    // layer number
    if (layer >= 10) { p[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    p[pos] = (char)('0' + layer % 10); pos++;
    // "."
    p[pos] = '.'; pos++;
    // suffix
    for (usz i = 0; i < suffix.len; i++) p[pos + i] = suffix[i];
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

fn ClipWeights? load_clip_weights(llm::DeviceContext* ctx, llm::GGUFFile* gf, llm::ClipConfig* config) {
    io::printfn("\nLoading CLIP weights...");
    String pfx = config.prefix;
    char[128] buf;

    llm::Tensor token_emb = llm::load_tensor_as_f32(ctx, gf,
        clip_tensor_name(&buf, pfx, "embeddings.token_embedding.weight"))!!;
    llm::Tensor pos_emb = llm::load_tensor_as_f32(ctx, gf,
        clip_tensor_name(&buf, pfx, "embeddings.position_embedding.weight"))!!;
    llm::Tensor final_ln_w = llm::load_tensor_by_name(ctx, gf,
        clip_tensor_name(&buf, pfx, "final_layer_norm.weight"))!!;
    llm::Tensor final_ln_b = llm::load_tensor_by_name(ctx, gf,
        clip_tensor_name(&buf, pfx, "final_layer_norm.bias"))!!;

    ClipLayerWeights[] layers = mem::new_array(ClipLayerWeights, config.n_layers);

    for (uint l = 0; l < config.n_layers; l++) {
        layers[l] = {
            .ln1_weight = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "layer_norm1.weight"))!!,
            .ln1_bias = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "layer_norm1.bias"))!!,
            .q_proj = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "self_attn.q_proj.weight"))!!,
            .q_proj_bias = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "self_attn.q_proj.bias"))!!,
            .k_proj = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "self_attn.k_proj.weight"))!!,
            .k_proj_bias = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "self_attn.k_proj.bias"))!!,
            .v_proj = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "self_attn.v_proj.weight"))!!,
            .v_proj_bias = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "self_attn.v_proj.bias"))!!,
            .out_proj = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "self_attn.out_proj.weight"))!!,
            .out_proj_bias = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "self_attn.out_proj.bias"))!!,
            .ln2_weight = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "layer_norm2.weight"))!!,
            .ln2_bias = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "layer_norm2.bias"))!!,
            .fc1 = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "mlp.fc1.weight"))!!,
            .fc1_bias = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "mlp.fc1.bias"))!!,
            .fc2 = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "mlp.fc2.weight"))!!,
            .fc2_bias = llm::load_tensor_by_name(ctx, gf,
                clip_layer_tensor_name(&buf, pfx, l, "mlp.fc2.bias"))!!,
        };

        if ((l + 1) % 4 == 0 || l == config.n_layers - 1) {
            io::printfn("  CLIP layer %d / %d loaded", l + 1, config.n_layers);
        }
    }

    return {
        .token_embedding = token_emb,
        .position_embedding = pos_emb,
        .final_ln_weight = final_ln_w,
        .final_ln_bias = final_ln_b,
        .layers = layers,
    };
}

fn ClipActivations? allocate_clip_activations(llm::DeviceContext* ctx, llm::ClipConfig* config) {
    io::printfn("Allocating CLIP activation buffers...");

    ulong seq = config.max_tokens;  // 77
    ulong dim = config.dim;         // 768
    ulong ffn_dim = dim * 4;        // 3072
    ulong n_heads = config.n_heads; // 12

    ulong[4] hidden_shape = { seq * dim, 0, 0, 0 };
    ulong[4] ffn_shape = { seq * ffn_dim, 0, 0, 0 };
    ulong[4] scores_shape = { n_heads * seq * seq, 0, 0, 0 };

    return {
        .hidden = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .norm_out = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .q = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .k = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .v = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .ffn_out = llm::create_f32_tensor(ctx, ffn_shape, 1)!!,
        .ffn_down_out = llm::create_f32_tensor(ctx, hidden_shape, 1)!!,
        .scores = llm::create_f32_tensor(ctx, scores_shape, 1)!!,
    };
}

fn ClipEncoder? load_clip_encoder(llm::DeviceContext* ctx, llm::GGUFFile* gf, llm::ClipConfig* config, llm::DiffusionKernels* kernels) {
    ClipWeights weights = load_clip_weights(ctx, gf, config)!!;
    ClipActivations acts = allocate_clip_activations(ctx, config)!!;

    return {
        .config = *config,
        .weights = weights,
        .acts = acts,
        .kernels = kernels,
        .ctx = ctx,
    };
}

// CLIP forward pass: tokens[max_tokens] -> hidden[max_tokens, dim]
// Result is in self.acts.hidden after this call
fn void? ClipEncoder.forward(&self, uint[] tokens, uint n_tokens) {
    llm::DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    llm::ClipConfig* c = &self.config;
    ClipWeights* w = &self.weights;
    ClipActivations* a = &self.acts;
    llm::DiffusionKernels* k = self.kernels;
    llm::SharedKernels* sk = &k.shared;

    uint seq = c.max_tokens;  // 77
    uint dim = c.dim;         // 768
    uint ffn_dim = dim * 4;   // 3072

llm::begin_compute(cmd)!!;

    // 1. Token embedding + position embedding -> hidden
    // For each position: hidden[pos] = token_emb[token_id] + pos_emb[pos]
    // We use embedding lookup for each position, then add position embeddings
    for (uint pos = 0; pos < seq; pos++) {
        uint token_id = pos < n_tokens ? tokens[pos] : 0;  // Pad with 0
        EmbeddingPC emb_pc = { .token_id = token_id, .dim = dim };
        // Write token embedding to hidden[pos*dim .. (pos+1)*dim-1]
        // We reuse the LLM embedding shader but need to offset output
        // For simplicity, write to a temp and copy, or do it on CPU
        // Actually: upload token IDs and use a batched approach
        // Simple approach: CPU-side embedding lookup
    }

    // Actually, for CLIP we need batched embedding. Let's do it by uploading
    // token embeddings and position embeddings and adding them on GPU.
    // Step 1: For each token position, copy the embedding row into hidden
    // Step 2: Add position embedding

    // Upload token IDs to staging and copy embeddings via shader
    for (uint pos = 0; pos < seq; pos++) {
        uint token_id = pos < n_tokens ? tokens[pos] : 0;
        // Copy token embedding row: token_emb[token_id * dim .. (token_id+1)*dim - 1]
        // to hidden[pos * dim .. (pos+1)*dim - 1]
        usz row_bytes = (usz)dim * 4;
        usz src_offset = (usz)token_id * row_bytes;
        usz dst_offset = (usz)pos * row_bytes;
        vk::cmdCopyBuffer(cmd, w.token_embedding.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = row_bytes }});
    }
llm::compute_barrier(cmd);

    // Add position embedding: hidden += position_embedding (element-wise, full seq*dim)
    llm::ResidualPC res_pc = { .n = seq * dim };
llm::dispatch_kernel(cmd, &sk.residual_add,
        { a.hidden.gpu_buffer.buffer, w.position_embedding.gpu_buffer.buffer },
        { a.hidden.size_bytes, w.position_embedding.size_bytes },
        &res_pc, llm::ceil_div(seq * dim, 256));
llm::compute_barrier(cmd);

    // 2. For each transformer layer
    for (uint l = 0; l < c.n_layers; l++) {
        ClipLayerWeights* lw = &w.layers[l];

        // 2a. LayerNorm1(hidden) -> norm_out
        // Apply layernorm per-position (77 times, each of dim=768)
        for (uint pos = 0; pos < seq; pos++) {
            llm::LayerNormPC ln_pc = { .dim = dim, .eps = c.eps };
            usz offset = (usz)pos * dim * 4;
            // We need per-position layernorm. Use the existing layernorm shader
            // but with an offset. Since our layernorm operates on binding 0 from start,
            // we'd need to offset the buffer. For now, dispatch per position.
            // TODO: optimize with a batched layernorm
llm::dispatch_kernel(cmd, &sk.layernorm,
                { a.hidden.gpu_buffer.buffer, lw.ln1_weight.gpu_buffer.buffer,
                  lw.ln1_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.ln1_weight.size_bytes,
                  lw.ln1_bias.size_bytes, a.norm_out.size_bytes },
                &ln_pc, 1);
        }
llm::compute_barrier(cmd);

        // 2b. Q/K/V projections: batched matmul [77, 768] x [768, 768]^T -> [77, 768]
        // Weight is stored as [out_dim, in_dim], matmul is input * weight^T
        // output[seq, dim] = norm_out[seq, dim] x weight^T[dim, dim]
        llm::BatchedMatMulPC qkv_pc = { .m_dim = seq, .n_dim = dim, .k_dim = dim };
        llm::dispatch_batched_matmul_auto(cmd, k, &lw.q_proj, &a.norm_out, &a.q, &qkv_pc);
        llm::dispatch_batched_matmul_auto(cmd, k, &lw.k_proj, &a.norm_out, &a.k, &qkv_pc);
        llm::dispatch_batched_matmul_auto(cmd, k, &lw.v_proj, &a.norm_out, &a.v, &qkv_pc);
llm::compute_barrier(cmd);

        // Add biases to Q, K, V
        llm::ResidualPC bias_pc = { .n = seq * dim };
        // TODO: broadcast add bias[dim] across seq positions
        // For now we need a broadcast add for [seq, dim] += [dim]
        llm::BroadcastAddPC ba_pc = { .channels = dim, .spatial = seq };
llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.q.gpu_buffer.buffer, lw.q_proj_bias.gpu_buffer.buffer },
            { a.q.size_bytes, lw.q_proj_bias.size_bytes },
            &ba_pc, llm::ceil_div(seq * dim, 256));
llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.k.gpu_buffer.buffer, lw.k_proj_bias.gpu_buffer.buffer },
            { a.k.size_bytes, lw.k_proj_bias.size_bytes },
            &ba_pc, llm::ceil_div(seq * dim, 256));
llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.v.gpu_buffer.buffer, lw.v_proj_bias.gpu_buffer.buffer },
            { a.v.size_bytes, lw.v_proj_bias.size_bytes },
            &ba_pc, llm::ceil_div(seq * dim, 256));
llm::compute_barrier(cmd);

        // 2c. Self-attention (non-causal)
        // Reshape Q/K/V from [seq, dim] to [n_heads, seq, head_dim]
        // The spatial_attention shader expects [n_heads, seq, head_dim] layout
        float scale = 1.0f / math::sqrt((float)c.head_dim);
        llm::SpatialAttentionPC sa_pc = {
            .head_dim = c.head_dim,
            .n_heads = c.n_heads,
            .seq_len = seq,
            .scale = scale,
        };
llm::dispatch_kernel(cmd, &k.spatial_attention,
            { a.q.gpu_buffer.buffer, a.k.gpu_buffer.buffer, a.v.gpu_buffer.buffer,
              a.scores.gpu_buffer.buffer, a.attn_out.gpu_buffer.buffer },
            { a.q.size_bytes, a.k.size_bytes, a.v.size_bytes,
              a.scores.size_bytes, a.attn_out.size_bytes },
            &sa_pc, c.n_heads * seq);
llm::compute_barrier(cmd);

        // 2d. Output projection
        llm::BatchedMatMulPC out_pc = { .m_dim = seq, .n_dim = dim, .k_dim = dim };
        llm::dispatch_batched_matmul_auto(cmd, k, &lw.out_proj, &a.attn_out, &a.norm_out, &out_pc);
llm::compute_barrier(cmd);

        // Add output projection bias
llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.norm_out.gpu_buffer.buffer, lw.out_proj_bias.gpu_buffer.buffer },
            { a.norm_out.size_bytes, lw.out_proj_bias.size_bytes },
            &ba_pc, llm::ceil_div(seq * dim, 256));
llm::compute_barrier(cmd);

        // 2e. Residual: hidden += attn_output
llm::dispatch_kernel(cmd, &sk.residual_add,
            { a.hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.norm_out.size_bytes },
            &res_pc, llm::ceil_div(seq * dim, 256));
llm::compute_barrier(cmd);

        // 2f. LayerNorm2(hidden) -> norm_out
        for (uint pos = 0; pos < seq; pos++) {
            llm::LayerNormPC ln_pc2 = { .dim = dim, .eps = c.eps };
llm::dispatch_kernel(cmd, &sk.layernorm,
                { a.hidden.gpu_buffer.buffer, lw.ln2_weight.gpu_buffer.buffer,
                  lw.ln2_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.ln2_weight.size_bytes,
                  lw.ln2_bias.size_bytes, a.norm_out.size_bytes },
                &ln_pc2, 1);
        }
llm::compute_barrier(cmd);

        // 2g. FFN: fc1 -> GELU -> fc2
        llm::BatchedMatMulPC fc1_pc = { .m_dim = seq, .n_dim = ffn_dim, .k_dim = dim };
        llm::dispatch_batched_matmul_auto(cmd, k, &lw.fc1, &a.norm_out, &a.ffn_out, &fc1_pc);
llm::compute_barrier(cmd);

        // Add fc1 bias
        llm::BroadcastAddPC fc1_ba = { .channels = ffn_dim, .spatial = seq };
llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.ffn_out.gpu_buffer.buffer, lw.fc1_bias.gpu_buffer.buffer },
            { a.ffn_out.size_bytes, lw.fc1_bias.size_bytes },
            &fc1_ba, llm::ceil_div(seq * ffn_dim, 256));
llm::compute_barrier(cmd);

        // GELU activation
        llm::GeluPC gelu_pc = { .n = seq * ffn_dim };
llm::dispatch_kernel(cmd, &sk.gelu,
            { a.ffn_out.gpu_buffer.buffer },
            { a.ffn_out.size_bytes },
            &gelu_pc, llm::ceil_div(seq * ffn_dim, 256));
llm::compute_barrier(cmd);

        // fc2
        llm::BatchedMatMulPC fc2_pc = { .m_dim = seq, .n_dim = dim, .k_dim = ffn_dim };
        llm::dispatch_batched_matmul_auto(cmd, k, &lw.fc2, &a.ffn_out, &a.ffn_down_out, &fc2_pc);
llm::compute_barrier(cmd);

        // Add fc2 bias
llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.ffn_down_out.gpu_buffer.buffer, lw.fc2_bias.gpu_buffer.buffer },
            { a.ffn_down_out.size_bytes, lw.fc2_bias.size_bytes },
            &ba_pc, llm::ceil_div(seq * dim, 256));
llm::compute_barrier(cmd);

        // 2h. Residual: hidden += ffn_output
llm::dispatch_kernel(cmd, &sk.residual_add,
            { a.hidden.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.ffn_down_out.size_bytes },
            &res_pc, llm::ceil_div(seq * dim, 256));
llm::compute_barrier(cmd);
    }

    // 3. Final LayerNorm
    for (uint pos = 0; pos < seq; pos++) {
        llm::LayerNormPC final_ln = { .dim = dim, .eps = c.eps };
llm::dispatch_kernel(cmd, &sk.layernorm,
            { a.hidden.gpu_buffer.buffer, w.final_ln_weight.gpu_buffer.buffer,
              w.final_ln_bias.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
            { a.hidden.size_bytes, w.final_ln_weight.size_bytes,
              w.final_ln_bias.size_bytes, a.hidden.size_bytes },
            &final_ln, 1);
    }

llm::submit_and_wait(ctx)!!;
}

fn void ClipEncoder.free(&self) {
    // Free weights
    self.weights.token_embedding.free();
    self.weights.position_embedding.free();
    self.weights.final_ln_weight.free();
    self.weights.final_ln_bias.free();
    for (uint l = 0; l < self.config.n_layers; l++) {
        ClipLayerWeights* lw = &self.weights.layers[l];
        lw.ln1_weight.free(); lw.ln1_bias.free();
        lw.q_proj.free(); lw.q_proj_bias.free();
        lw.k_proj.free(); lw.k_proj_bias.free();
        lw.v_proj.free(); lw.v_proj_bias.free();
        lw.out_proj.free(); lw.out_proj_bias.free();
        lw.ln2_weight.free(); lw.ln2_bias.free();
        lw.fc1.free(); lw.fc1_bias.free();
        lw.fc2.free(); lw.fc2_bias.free();
    }
    mem::free(self.weights.layers);

    // Free activations
    self.acts.hidden.free();
    self.acts.norm_out.free();
    self.acts.q.free();
    self.acts.k.free();
    self.acts.v.free();
    self.acts.attn_out.free();
    self.acts.ffn_out.free();
    self.acts.ffn_down_out.free();
    self.acts.scores.free();
}
