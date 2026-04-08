// Flash Attention v2 - Tiled FA2-style with online softmax
// Separate file to avoid shared memory accumulation across entry points in SPIR-V

struct FlashAttentionParams {
    uint head_dim;    // 128
    uint n_heads;     // 30
    uint seq_len;     // ~1056
    float scale;      // 1/sqrt(128)
};

[vk_push_constant] const FlashAttentionParams fa2_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> fa2_Q;      // [n_heads, seq_len, head_dim]
[vk_binding(1, 0)] RWStructuredBuffer<float> fa2_K;      // [n_heads, seq_len, head_dim]
[vk_binding(2, 0)] RWStructuredBuffer<float> fa2_V;      // [n_heads, seq_len, head_dim]
[vk_binding(3, 0)] RWStructuredBuffer<float> fa2_output;  // [seq_len, n_heads * head_dim]

static const uint FA2_BR = 8;       // query rows per workgroup
static const uint FA2_BC = 32;      // KV positions per tile
static const uint FA2_D  = 128;     // head dimension
static const uint FA2_D_PAD = 129;  // +1 padding to avoid LDS bank conflicts

// LDS: ~26 KB total
groupshared float fa2_KV[FA2_BC * FA2_D_PAD];   // 4128 floats — reused for K then V
groupshared float fa2_Qtile[FA2_BR * FA2_D];     // 1024 floats
groupshared float fa2_S[FA2_BR * FA2_BC];         // 256 floats — score tile
groupshared float fa2_O[FA2_BR * FA2_D];          // 1024 floats — output accumulator
groupshared float fa2_m[FA2_BR];                   // running max per query row
groupshared float fa2_l[FA2_BR];                   // running sum per query row
groupshared float fa2_rescale[FA2_BR];             // exp(m_old - m_new) per row
groupshared float fa2_reduce[256];                  // tree reduction scratch

[shader("compute")]
[numthreads(256, 1, 1)]
void flash_attention_v2(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint head = gid.y;
    uint q_start = gid.x * FA2_BR;
    uint S = fa2_pc.seq_len;
    uint D = fa2_pc.head_dim;
    float scale = fa2_pc.scale;

    if (head >= fa2_pc.n_heads) return;

    uint head_offset = head * S * D;

    // Step 1: Load Q tile [BR x D] into LDS (coalesced)
    for (uint i = 0; i < 4; i++) {
        uint flat = gi + i * 256;
        if (flat < FA2_BR * FA2_D) {
            uint qr = flat / FA2_D;
            uint qd = flat % FA2_D;
            uint actual_q = q_start + qr;
            fa2_Qtile[flat] = (actual_q < S) ? fa2_Q[head_offset + actual_q * D + qd] : 0.0;
        }
    }

    // Step 2: Initialize accumulators
    for (uint i = 0; i < 4; i++) {
        uint flat = gi + i * 256;
        if (flat < FA2_BR * FA2_D)
            fa2_O[flat] = 0.0;
    }
    if (gi < FA2_BR) {
        fa2_m[gi] = -1.0e30;
        fa2_l[gi] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Step 3: Outer loop over KV blocks
    uint n_kv_blocks = (S + FA2_BC - 1) / FA2_BC;

    for (uint kv_block = 0; kv_block < n_kv_blocks; kv_block++) {
        uint kv_start = kv_block * FA2_BC;

        // 3a: Load K tile [BC x D] into fa2_KV (coalesced, padded stride)
        for (uint i = 0; i < 16; i++) {
            uint flat = gi + i * 256;
            if (flat < FA2_BC * FA2_D) {
                uint kv_r = flat / FA2_D;
                uint kv_d = flat % FA2_D;
                uint actual_kv = kv_start + kv_r;
                fa2_KV[kv_r * FA2_D_PAD + kv_d] = (actual_kv < S)
                    ? fa2_K[head_offset + actual_kv * D + kv_d] : 0.0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3b: Compute scores S_ij = Q @ K^T  [BR x BC] = [8 x 32]
        // Thread gi -> score[gi/32][gi%32], 256 threads = 256 scores (1:1)
        uint qr = gi / FA2_BC;    // 0..7
        uint kc = gi % FA2_BC;    // 0..31
        float dot_val = 0.0;
        if (qr < FA2_BR) {
            for (uint d = 0; d < FA2_D; d++) {
                dot_val += fa2_Qtile[qr * FA2_D + d] * fa2_KV[kc * FA2_D_PAD + d];
            }
            dot_val *= scale;
        }
        // Mask out-of-bounds
        uint actual_kv = kv_start + kc;
        uint actual_q = q_start + qr;
        bool valid = (actual_kv < S) && (actual_q < S) && (qr < FA2_BR);
        fa2_S[gi] = valid ? dot_val : -1.0e30;
        GroupMemoryBarrierWithGroupSync();

        // 3c: Row-max reduction (32 threads per row via shared memory)
        fa2_reduce[gi] = fa2_S[gi];
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 16; s > 0; s >>= 1) {
            if (kc < s)
                fa2_reduce[gi] = max(fa2_reduce[gi], fa2_reduce[gi + s]);
            GroupMemoryBarrierWithGroupSync();
        }
        float tile_max = fa2_reduce[qr * FA2_BC];

        // 3d: Online softmax update
        float m_old = fa2_m[qr];
        float m_new = max(m_old, tile_max);
        float rescale_val = exp(m_old - m_new);

        // Compute exp(score - m_new)
        float exp_val = valid ? exp(fa2_S[gi] - m_new) : 0.0;
        fa2_S[gi] = exp_val;

        // Row-sum reduction
        fa2_reduce[gi] = exp_val;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 16; s > 0; s >>= 1) {
            if (kc < s)
                fa2_reduce[gi] += fa2_reduce[gi + s];
            GroupMemoryBarrierWithGroupSync();
        }
        float tile_sum = fa2_reduce[qr * FA2_BC];

        // Update running state (one thread per row)
        if (kc == 0 && qr < FA2_BR) {
            fa2_m[qr] = m_new;
            fa2_l[qr] = rescale_val * fa2_l[qr] + tile_sum;
            fa2_rescale[qr] = rescale_val;
        }
        GroupMemoryBarrierWithGroupSync();

        // 3e: Load V tile into fa2_KV (overwrites K, coalesced)
        for (uint i = 0; i < 16; i++) {
            uint flat = gi + i * 256;
            if (flat < FA2_BC * FA2_D) {
                uint kv_r = flat / FA2_D;
                uint kv_d = flat % FA2_D;
                uint actual_kv2 = kv_start + kv_r;
                fa2_KV[kv_r * FA2_D_PAD + kv_d] = (actual_kv2 < S)
                    ? fa2_V[head_offset + actual_kv2 * D + kv_d] : 0.0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3f: Update output: O = rescale * O + exp_scores @ V_tile
        for (uint i = 0; i < 4; i++) {
            uint flat = gi + i * 256;
            if (flat < FA2_BR * FA2_D) {
                uint o_qr = flat / FA2_D;
                uint o_d  = flat % FA2_D;

                // Rescale old output
                float val = fa2_O[flat] * fa2_rescale[o_qr];

                // Accumulate: scores[o_qr, :] @ V_tile[:, o_d]
                float new_acc = 0.0;
                for (uint kv = 0; kv < FA2_BC; kv++) {
                    new_acc += fa2_S[o_qr * FA2_BC + kv] * fa2_KV[kv * FA2_D_PAD + o_d];
                }
                fa2_O[flat] = val + new_acc;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Step 4: Final normalization and output write (fused reverse transpose)
    uint dim = fa2_pc.n_heads * D;
    for (uint i = 0; i < 4; i++) {
        uint flat = gi + i * 256;
        if (flat < FA2_BR * FA2_D) {
            uint o_qr = flat / FA2_D;
            uint o_d  = flat % FA2_D;
            uint actual_q = q_start + o_qr;
            if (actual_q < S) {
                fa2_output[actual_q * dim + head * D + o_d] = fa2_O[flat] / fa2_l[o_qr];
            }
        }
    }
}
