module flux;

import vk;
import llm;
import std::io;
import std::math;
import std::core::mem;

// CLIP-L Text Encoder for Flux
// 12-layer transformer with non-causal self-attention
// Produces pooled output [768] for vector conditioning
//
// Architecture: dim=768, heads=12, head_dim=64, FFN=3072, max_tokens=77

const uint CLIP_DIM       = 768;
const uint CLIP_HEADS     = 12;
const uint CLIP_HEAD_DIM  = 64;
const uint CLIP_FFN_DIM   = 3072;
const uint CLIP_LAYERS    = 12;
const uint CLIP_MAX_TOKENS = 77;
const uint CLIP_VOCAB_SIZE = 49408;
const float CLIP_EPS      = 1e-5f;

// Special token IDs (standard OpenAI CLIP)
const uint CLIP_BOS_ID    = 49406;  // <|startoftext|>
const uint CLIP_EOS_ID    = 49407;  // <|endoftext|>

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
    llm::Tensor text_projection;     // [dim, dim] optional — for pooled output
    bool has_text_projection;
    ClipLayerWeights[] layers;
}

struct ClipActivations {
    llm::Tensor hidden;       // [max_tokens, dim]
    llm::Tensor norm_out;     // [max_tokens, dim]
    llm::Tensor q;            // [max_tokens, dim]
    llm::Tensor k;            // [max_tokens, dim]
    llm::Tensor v;            // [max_tokens, dim]
    llm::Tensor attn_out;     // [max_tokens, dim]
    llm::Tensor ffn_out;      // [max_tokens, ffn_dim]
    llm::Tensor ffn_down_out; // [max_tokens, dim]
    llm::Tensor scores;       // [n_heads, max_tokens, max_tokens]
    llm::Tensor pooled;       // [dim] — pooled output for Flux vector_in
}

struct FluxClipEncoder {
    ClipWeights weights;
    ClipActivations acts;
    FluxKernels* kernels;
    llm::DeviceContext* ctx;
}

// --- Tensor name helpers ---

fn String clip_name(char[128]* buf, String prefix, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) p[pos + i] = prefix[i];
    pos += prefix.len;
    for (usz i = 0; i < suffix.len; i++) p[pos + i] = suffix[i];
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

fn String clip_layer_name(char[128]* buf, String prefix, uint layer, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) p[pos + i] = prefix[i];
    pos += prefix.len;
    String mid = "encoder.layers.";
    for (usz i = 0; i < mid.len; i++) p[pos + i] = mid[i];
    pos += mid.len;
    if (layer >= 10) { p[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    p[pos] = (char)('0' + layer % 10); pos++;
    p[pos] = '.'; pos++;
    for (usz i = 0; i < suffix.len; i++) p[pos + i] = suffix[i];
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

// --- Load from safetensors ---

fn ClipWeights? load_clip_weights_safetensors(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix) {
    io::printfn("\nLoading CLIP-L weights from safetensors...");
    char[128] buf;

    llm::Tensor token_emb = llm::upload_safetensor_f32(ctx, sf,
        clip_name(&buf, prefix, "embeddings.token_embedding.weight"))!!;
    llm::Tensor pos_emb = llm::upload_safetensor_f32(ctx, sf,
        clip_name(&buf, prefix, "embeddings.position_embedding.weight"))!!;
    llm::Tensor final_ln_w = llm::upload_safetensor_f32(ctx, sf,
        clip_name(&buf, prefix, "final_layer_norm.weight"))!!;
    llm::Tensor final_ln_b = llm::upload_safetensor_f32(ctx, sf,
        clip_name(&buf, prefix, "final_layer_norm.bias"))!!;

    // text_projection is optional (may be at top level without prefix)
    llm::Tensor text_proj;
    bool has_text_proj = false;
    if (try info = sf.find_tensor("text_projection.weight")) {
        text_proj = llm::upload_safetensor_f32(ctx, sf, "text_projection.weight")!!;
        has_text_proj = true;
        io::printfn("  Found text_projection.weight");
    }

    ClipLayerWeights[] layers = mem::new_array(ClipLayerWeights, CLIP_LAYERS);

    for (uint l = 0; l < CLIP_LAYERS; l++) {
        layers[l] = {
            .ln1_weight = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "layer_norm1.weight"))!!,
            .ln1_bias = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "layer_norm1.bias"))!!,
            .q_proj = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "self_attn.q_proj.weight"))!!,
            .q_proj_bias = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "self_attn.q_proj.bias"))!!,
            .k_proj = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "self_attn.k_proj.weight"))!!,
            .k_proj_bias = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "self_attn.k_proj.bias"))!!,
            .v_proj = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "self_attn.v_proj.weight"))!!,
            .v_proj_bias = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "self_attn.v_proj.bias"))!!,
            .out_proj = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "self_attn.out_proj.weight"))!!,
            .out_proj_bias = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "self_attn.out_proj.bias"))!!,
            .ln2_weight = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "layer_norm2.weight"))!!,
            .ln2_bias = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "layer_norm2.bias"))!!,
            .fc1 = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "mlp.fc1.weight"))!!,
            .fc1_bias = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "mlp.fc1.bias"))!!,
            .fc2 = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "mlp.fc2.weight"))!!,
            .fc2_bias = llm::upload_safetensor_f32(ctx, sf,
                clip_layer_name(&buf, prefix, l, "mlp.fc2.bias"))!!,
        };

        if ((l + 1) % 4 == 0 || l == CLIP_LAYERS - 1) {
            io::printfn("  CLIP layer %d / %d loaded", l + 1, CLIP_LAYERS);
        }
    }

    return {
        .token_embedding = token_emb,
        .position_embedding = pos_emb,
        .final_ln_weight = final_ln_w,
        .final_ln_bias = final_ln_b,
        .text_projection = text_proj,
        .has_text_projection = has_text_proj,
        .layers = layers,
    };
}

fn ClipActivations? allocate_clip_activations(llm::DeviceContext* ctx) {
    io::printfn("Allocating CLIP activation buffers...");

    ulong seq = CLIP_MAX_TOKENS;
    ulong dim = CLIP_DIM;
    ulong ffn_dim = CLIP_FFN_DIM;
    ulong n_heads = CLIP_HEADS;

    ulong[4] hidden_shape = { seq * dim, 0, 0, 0 };
    ulong[4] ffn_shape = { seq * ffn_dim, 0, 0, 0 };
    ulong[4] scores_shape = { n_heads * seq * seq, 0, 0, 0 };
    ulong[4] pooled_shape = { dim, 0, 0, 0 };

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
        .pooled = llm::create_f32_tensor(ctx, pooled_shape, 1)!!,
    };
}

fn FluxClipEncoder? load_clip_encoder(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, FluxKernels* kernels) {
    ClipWeights weights = load_clip_weights_safetensors(ctx, sf, "text_model.")!!;
    ClipActivations acts = allocate_clip_activations(ctx)!!;

    return {
        .weights = weights,
        .acts = acts,
        .kernels = kernels,
        .ctx = ctx,
    };
}

// CLIP forward pass: tokens[n_tokens] -> hidden[max_tokens, dim] + pooled[dim]
// After this call:
//   - self.acts.hidden contains the full sequence hidden states [max_tokens, dim]
//   - self.acts.pooled contains the pooled output [dim] (EOS token embedding)
fn void? FluxClipEncoder.forward(&self, uint[] tokens, uint n_tokens) {
    llm::DeviceContext* ctx = self.ctx;
    vk::CommandBuffer cmd = ctx.command_buffer;
    ClipWeights* w = &self.weights;
    ClipActivations* a = &self.acts;
    FluxKernels* k = self.kernels;
    llm::SharedKernels* sk = &k.shared;

    uint seq = CLIP_MAX_TOKENS;
    uint dim = CLIP_DIM;
    uint ffn_dim = CLIP_FFN_DIM;

    llm::begin_compute(cmd)!!;

    // 1. Token embedding: copy embedding rows for each token position
    for (uint pos = 0; pos < seq; pos++) {
        uint token_id = pos < n_tokens ? tokens[pos] : 0;
        usz row_bytes = (usz)dim * 4;
        usz src_offset = (usz)token_id * row_bytes;
        usz dst_offset = (usz)pos * row_bytes;
        vk::cmdCopyBuffer(cmd, w.token_embedding.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = row_bytes }});
    }
    llm::compute_barrier(cmd);

    // 2. Add position embedding: hidden += position_embedding
    llm::ResidualPC res_pc = { .n = seq * dim };
    llm::dispatch_kernel(cmd, &sk.residual_add,
        { a.hidden.gpu_buffer.buffer, w.position_embedding.gpu_buffer.buffer },
        { a.hidden.size_bytes, w.position_embedding.size_bytes },
        &res_pc, llm::ceil_div(seq * dim, 256));
    llm::compute_barrier(cmd);

    // 3. Transformer layers
    for (uint l = 0; l < CLIP_LAYERS; l++) {
        ClipLayerWeights* lw = &w.layers[l];

        // LayerNorm1(hidden) -> norm_out
        for (uint pos = 0; pos < seq; pos++) {
            llm::LayerNormPC ln_pc = { .dim = dim, .eps = CLIP_EPS };
            llm::dispatch_kernel(cmd, &sk.layernorm,
                { a.hidden.gpu_buffer.buffer, lw.ln1_weight.gpu_buffer.buffer,
                  lw.ln1_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.ln1_weight.size_bytes,
                  lw.ln1_bias.size_bytes, a.norm_out.size_bytes },
                &ln_pc, 1);
        }
        llm::compute_barrier(cmd);

        // Q/K/V projections: [77, 768] x [768, 768]^T -> [77, 768]
        BatchedMatMulPC qkv_pc = { .m_dim = seq, .n_dim = dim, .k_dim = dim };
        dispatch_batched_matmul(cmd, k, &lw.q_proj, &a.norm_out, &a.q, &qkv_pc);
        dispatch_batched_matmul(cmd, k, &lw.k_proj, &a.norm_out, &a.k, &qkv_pc);
        dispatch_batched_matmul(cmd, k, &lw.v_proj, &a.norm_out, &a.v, &qkv_pc);
        llm::compute_barrier(cmd);

        // Add biases
        BroadcastAddPC ba_pc = { .channels = dim, .spatial = seq };
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

        // Self-attention (non-causal)
        float scale = 1.0f / math::sqrt((float)CLIP_HEAD_DIM);
        SpatialAttentionPC sa_pc = {
            .head_dim = CLIP_HEAD_DIM,
            .n_heads = CLIP_HEADS,
            .seq_len = seq,
            .scale = scale,
        };
        llm::dispatch_kernel(cmd, &k.spatial_attention,
            { a.q.gpu_buffer.buffer, a.k.gpu_buffer.buffer, a.v.gpu_buffer.buffer,
              a.scores.gpu_buffer.buffer, a.attn_out.gpu_buffer.buffer },
            { a.q.size_bytes, a.k.size_bytes, a.v.size_bytes,
              a.scores.size_bytes, a.attn_out.size_bytes },
            &sa_pc, CLIP_HEADS * seq);
        llm::compute_barrier(cmd);

        // Output projection
        BatchedMatMulPC out_pc = { .m_dim = seq, .n_dim = dim, .k_dim = dim };
        dispatch_batched_matmul(cmd, k, &lw.out_proj, &a.attn_out, &a.norm_out, &out_pc);
        llm::compute_barrier(cmd);

        llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.norm_out.gpu_buffer.buffer, lw.out_proj_bias.gpu_buffer.buffer },
            { a.norm_out.size_bytes, lw.out_proj_bias.size_bytes },
            &ba_pc, llm::ceil_div(seq * dim, 256));
        llm::compute_barrier(cmd);

        // Residual: hidden += attn_output
        llm::dispatch_kernel(cmd, &sk.residual_add,
            { a.hidden.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.norm_out.size_bytes },
            &res_pc, llm::ceil_div(seq * dim, 256));
        llm::compute_barrier(cmd);

        // LayerNorm2(hidden) -> norm_out
        for (uint pos = 0; pos < seq; pos++) {
            llm::LayerNormPC ln_pc2 = { .dim = dim, .eps = CLIP_EPS };
            llm::dispatch_kernel(cmd, &sk.layernorm,
                { a.hidden.gpu_buffer.buffer, lw.ln2_weight.gpu_buffer.buffer,
                  lw.ln2_bias.gpu_buffer.buffer, a.norm_out.gpu_buffer.buffer },
                { a.hidden.size_bytes, lw.ln2_weight.size_bytes,
                  lw.ln2_bias.size_bytes, a.norm_out.size_bytes },
                &ln_pc2, 1);
        }
        llm::compute_barrier(cmd);

        // FFN: fc1 -> GELU -> fc2
        BatchedMatMulPC fc1_pc = { .m_dim = seq, .n_dim = ffn_dim, .k_dim = dim };
        dispatch_batched_matmul(cmd, k, &lw.fc1, &a.norm_out, &a.ffn_out, &fc1_pc);
        llm::compute_barrier(cmd);

        BroadcastAddPC fc1_ba = { .channels = ffn_dim, .spatial = seq };
        llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.ffn_out.gpu_buffer.buffer, lw.fc1_bias.gpu_buffer.buffer },
            { a.ffn_out.size_bytes, lw.fc1_bias.size_bytes },
            &fc1_ba, llm::ceil_div(seq * ffn_dim, 256));
        llm::compute_barrier(cmd);

        llm::GeluPC gelu_pc = { .n = seq * ffn_dim };
        llm::dispatch_kernel(cmd, &sk.gelu,
            { a.ffn_out.gpu_buffer.buffer },
            { a.ffn_out.size_bytes },
            &gelu_pc, llm::ceil_div(seq * ffn_dim, 256));
        llm::compute_barrier(cmd);

        BatchedMatMulPC fc2_pc = { .m_dim = seq, .n_dim = dim, .k_dim = ffn_dim };
        dispatch_batched_matmul(cmd, k, &lw.fc2, &a.ffn_out, &a.ffn_down_out, &fc2_pc);
        llm::compute_barrier(cmd);

        llm::dispatch_kernel(cmd, &k.broadcast_add,
            { a.ffn_down_out.gpu_buffer.buffer, lw.fc2_bias.gpu_buffer.buffer },
            { a.ffn_down_out.size_bytes, lw.fc2_bias.size_bytes },
            &ba_pc, llm::ceil_div(seq * dim, 256));
        llm::compute_barrier(cmd);

        // Residual: hidden += ffn_output
        llm::dispatch_kernel(cmd, &sk.residual_add,
            { a.hidden.gpu_buffer.buffer, a.ffn_down_out.gpu_buffer.buffer },
            { a.hidden.size_bytes, a.ffn_down_out.size_bytes },
            &res_pc, llm::ceil_div(seq * dim, 256));
        llm::compute_barrier(cmd);
    }

    // 4. Final LayerNorm
    for (uint pos = 0; pos < seq; pos++) {
        llm::LayerNormPC final_ln = { .dim = dim, .eps = CLIP_EPS };
        llm::dispatch_kernel(cmd, &sk.layernorm,
            { a.hidden.gpu_buffer.buffer, w.final_ln_weight.gpu_buffer.buffer,
              w.final_ln_bias.gpu_buffer.buffer, a.hidden.gpu_buffer.buffer },
            { a.hidden.size_bytes, w.final_ln_weight.size_bytes,
              w.final_ln_bias.size_bytes, a.hidden.size_bytes },
            &final_ln, 1);
    }

    // 5. Extract pooled output: hidden state at EOS token position
    // The EOS token is the last real token (position n_tokens - 1)
    // Copy hidden[eos_pos * dim .. (eos_pos+1)*dim - 1] -> pooled[0..dim-1]
    {
        uint eos_pos = n_tokens > 0 ? n_tokens - 1 : 0;
        usz row_bytes = (usz)dim * 4;
        usz src_offset = (usz)eos_pos * row_bytes;
        vk::cmdCopyBuffer(cmd, a.hidden.gpu_buffer.buffer, a.pooled.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = src_offset, .dstOffset = 0, .size = row_bytes }});
    }

    // 6. If text_projection exists, apply it: pooled = pooled @ text_projection^T
    if (w.has_text_projection) {
        llm::compute_barrier(cmd);
        // text_projection is [dim, dim], pooled is [1, dim]
        // Output goes to norm_out[0..dim-1] temporarily, then copy back
        BatchedMatMulPC proj_pc = { .m_dim = 1, .n_dim = dim, .k_dim = dim };
        dispatch_batched_matmul(cmd, k, &w.text_projection, &a.pooled, &a.norm_out, &proj_pc);
        llm::compute_barrier(cmd);
        usz row_bytes = (usz)dim * 4;
        vk::cmdCopyBuffer(cmd, a.norm_out.gpu_buffer.buffer, a.pooled.gpu_buffer.buffer, 1,
            (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = row_bytes }});
    }

    llm::submit_and_wait(ctx)!!;
}

// Tokenize text for CLIP using a pre-loaded tokenizer (GPT-2 BPE)
// Returns token array with BOS and EOS, padded/truncated to max_tokens
fn uint[]? clip_tokenize(llm::Tokenizer* tokenizer, String text) {
    uint[] raw_tokens = tokenizer.encode(text)!!;

    // Build: [BOS, ...tokens..., EOS, padding...]
    uint[] tokens = mem::new_array(uint, CLIP_MAX_TOKENS);
    tokens[0] = CLIP_BOS_ID;

    uint max_text_tokens = CLIP_MAX_TOKENS - 2;  // Reserve BOS + EOS
    uint n_text = (uint)raw_tokens.len;
    if (n_text > max_text_tokens) n_text = max_text_tokens;

    for (uint i = 0; i < n_text; i++) {
        tokens[i + 1] = raw_tokens[i];
    }
    tokens[n_text + 1] = CLIP_EOS_ID;

    // Pad rest with EOS
    for (uint i = n_text + 2; i < CLIP_MAX_TOKENS; i++) {
        tokens[i] = CLIP_EOS_ID;
    }

    uint n_tokens = n_text + 2;  // BOS + text + EOS
    mem::free(raw_tokens);

    io::printfn("CLIP tokenized: %d tokens (text: %d)", n_tokens, n_text);
    return tokens;
}

fn void FluxClipEncoder.free(&self) {
    self.weights.token_embedding.free();
    self.weights.position_embedding.free();
    self.weights.final_ln_weight.free();
    self.weights.final_ln_bias.free();
    if (self.weights.has_text_projection) self.weights.text_projection.free();
    for (uint l = 0; l < CLIP_LAYERS; l++) {
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

    self.acts.hidden.free();
    self.acts.norm_out.free();
    self.acts.q.free();
    self.acts.k.free();
    self.acts.v.free();
    self.acts.attn_out.free();
    self.acts.ffn_out.free();
    self.acts.ffn_down_out.free();
    self.acts.scores.free();
    self.acts.pooled.free();
}
