// T5 Encoder Shader - attention with additive relative position bias
//
// The T5 attention adds a precomputed bias [n_heads, seq_len, seq_len]
// to the QK^T scores before softmax. This is the only architectural
// difference from standard non-causal self-attention.

struct T5AttentionPC {
    uint head_dim;
    uint n_heads;
    uint seq_len;
    float scale;
};

// Bindings: [0] Q, [1] K, [2] V, [3] bias, [4] scores, [5] output
[[vk::binding(0)]] RWStructuredBuffer<float> t5_Q;
[[vk::binding(1)]] RWStructuredBuffer<float> t5_K;
[[vk::binding(2)]] RWStructuredBuffer<float> t5_V;
[[vk::binding(3)]] RWStructuredBuffer<float> t5_bias;    // [n_heads, seq, seq] precomputed
[[vk::binding(4)]] RWStructuredBuffer<float> t5_scores;  // [n_heads, seq, seq] workspace
[[vk::binding(5)]] RWStructuredBuffer<float> t5_out;

[[vk::push_constant]] T5AttentionPC t5_attn_pc;

groupshared float t5_shared[256];

// T5 self-attention with relative position bias
// One workgroup per (head, query_position) pair
// Layout: Q/K/V in [n_heads, seq_len, head_dim], bias in [n_heads, seq_len, seq_len]
[shader("compute")]
[numthreads(256, 1, 1)]
void t5_attention(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint head = gid.x / t5_attn_pc.seq_len;
    uint q_pos = gid.x % t5_attn_pc.seq_len;
    uint num_threads = 256;

    if (head >= t5_attn_pc.n_heads) return;

    uint head_offset = head * t5_attn_pc.seq_len * t5_attn_pc.head_dim;
    uint scores_base = head * t5_attn_pc.seq_len * t5_attn_pc.seq_len + q_pos * t5_attn_pc.seq_len;
    uint bias_base = scores_base;  // Same layout as scores

    // Phase 1: Compute Q[q_pos] * K[kv_pos]^T + bias
    for (uint kv_pos = gi; kv_pos < t5_attn_pc.seq_len; kv_pos += num_threads) {
        float dot_val = 0.0;
        for (uint d = 0; d < t5_attn_pc.head_dim; d++) {
            dot_val += t5_Q[head_offset + q_pos * t5_attn_pc.head_dim + d] *
                       t5_K[head_offset + kv_pos * t5_attn_pc.head_dim + d];
        }
        // Scale and add relative position bias
        t5_scores[scores_base + kv_pos] = dot_val * t5_attn_pc.scale + t5_bias[bias_base + kv_pos];
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Softmax over scores (no causal mask)
    float max_val = -1.0e30;
    for (uint i = gi; i < t5_attn_pc.seq_len; i += num_threads) {
        max_val = max(max_val, t5_scores[scores_base + i]);
    }
    t5_shared[gi] = max_val;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) t5_shared[gi] = max(t5_shared[gi], t5_shared[gi + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float global_max = t5_shared[0];
    GroupMemoryBarrierWithGroupSync();

    float sum = 0.0;
    for (uint i = gi; i < t5_attn_pc.seq_len; i += num_threads) {
        float val = exp(t5_scores[scores_base + i] - global_max);
        t5_scores[scores_base + i] = val;
        sum += val;
    }
    t5_shared[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (gi < s) t5_shared[gi] += t5_shared[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float global_sum = t5_shared[0];
    GroupMemoryBarrierWithGroupSync();

    for (uint i = gi; i < t5_attn_pc.seq_len; i += num_threads) {
        t5_scores[scores_base + i] /= global_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: Weighted sum of V
    uint out_base = head_offset + q_pos * t5_attn_pc.head_dim;
    for (uint d = gi; d < t5_attn_pc.head_dim; d += num_threads) {
        float acc = 0.0;
        for (uint pos = 0; pos < t5_attn_pc.seq_len; pos++) {
            acc += t5_scores[scores_base + pos] * t5_V[head_offset + pos * t5_attn_pc.head_dim + d];
        }
        t5_out[out_base + d] = acc;
    }
}

// GEGLU activation: output[i] = GELU(gate[i]) * up[i]
// gate and up are interleaved in input: [seq, 2*ffn_dim]
// gate = input[..., :ffn_dim], up = input[..., ffn_dim:]
struct GeGLUPC {
    uint n;          // total elements = seq * ffn_dim
    uint ffn_dim;    // FFN hidden dim
    uint seq_len;    // sequence length
};

[[vk::binding(0)]] RWStructuredBuffer<float> geglu_input;   // [seq, 2*ffn_dim]
[[vk::binding(1)]] RWStructuredBuffer<float> geglu_output;  // [seq, ffn_dim]

[[vk::push_constant]] GeGLUPC geglu_pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void geglu_activation(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint idx = gid.x * 256 + gi;
    if (idx >= geglu_pc.n) return;

    // Determine position in [seq, ffn_dim]
    uint seq_idx = idx / geglu_pc.ffn_dim;
    uint dim_idx = idx % geglu_pc.ffn_dim;

    // Input layout: [seq, 2*ffn_dim] where first half is gate, second is up
    uint base = seq_idx * geglu_pc.ffn_dim * 2;
    float gate_val = geglu_input[base + dim_idx];
    float up_val = geglu_input[base + geglu_pc.ffn_dim + dim_idx];

    // GELU approximation (same as in diffusion.slang)
    float gelu_val = gate_val * 0.5 * (1.0 + tanh(0.7978845608 * (gate_val + 0.044715 * gate_val * gate_val * gate_val)));

    geglu_output[idx] = gelu_val * up_val;
}
