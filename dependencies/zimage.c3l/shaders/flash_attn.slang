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

static const uint FA2_BR = 16;      // query rows per workgroup (dispatch grid uses this)
static const uint FA2_BC = 32;      // KV positions per tile
static const uint FA2_D  = 128;     // head dimension
static const uint FA2_DP = 64;      // head dimension in half2 pairs (D/2)
static const uint FA2_DP_PAD = 65;  // +1 padding to avoid LDS bank conflicts

// LDS: ~16.3 KB total. Q/K/V tiles are packed fp16 pairs (same recipe as
// batch_matmul_q8): one ds_read_b32 returns two D-adjacent halves, and the
// dot loops are shaped `acc += a.x*b.x + a.y*b.y` (V_DOT2 form). Scores and
// all online-softmax state stay fp32; the output accumulator lives in
// registers (8 floats/thread), which is what funds BR=16 without leaving
// the 16 KB LDS occupancy bucket.
groupshared vector<float16_t,2> fa2_KV[FA2_BC * FA2_DP_PAD]; // reused for K then V
groupshared vector<float16_t,2> fa2_Qtile[FA2_BR * FA2_DP];
groupshared float fa2_S[FA2_BR * FA2_BC];          // score tile
groupshared float fa2_m[FA2_BR];                   // running max per query row
groupshared float fa2_l[FA2_BR];                   // running sum per query row
groupshared float fa2_rescale[FA2_BR];             // exp(m_old - m_new) per row
groupshared float fa2_reduce[FA2_BR * FA2_BC];     // tree reduction scratch

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

    // Step 1: Load Q tile [BR x DP half2] into LDS (coalesced pair loads)
    for (uint i = 0; i < 4; i++) {
        uint flat = gi + i * 256;
        if (flat < FA2_BR * FA2_DP) {
            uint qr = flat / FA2_DP;
            uint qd = flat % FA2_DP;
            uint actual_q = q_start + qr;
            float v0 = 0.0, v1 = 0.0;
            if (actual_q < S) {
                uint base = head_offset + actual_q * D + qd * 2;
                v0 = fa2_Q[base];
                v1 = fa2_Q[base + 1];
            }
            fa2_Qtile[flat] = vector<float16_t,2>(float16_t(v0), float16_t(v1));
        }
    }

    // Step 2: Initialize accumulators. Output lives in registers: thread gi
    // owns pair slots (gi + i*256) -> (row flat/DP, pair flat%DP), i = 0..3.
    float acc_x[4];
    float acc_y[4];
    for (uint i = 0; i < 4; i++) {
        acc_x[i] = 0.0;
        acc_y[i] = 0.0;
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

        // 3a: Load K tile [BC x DP half2] into fa2_KV (coalesced pair loads)
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            if (flat < FA2_BC * FA2_DP) {
                uint kv_r = flat / FA2_DP;
                uint kv_d = flat % FA2_DP;
                uint actual_kv = kv_start + kv_r;
                float v0 = 0.0, v1 = 0.0;
                if (actual_kv < S) {
                    uint base = head_offset + actual_kv * D + kv_d * 2;
                    v0 = fa2_K[base];
                    v1 = fa2_K[base + 1];
                }
                fa2_KV[kv_r * FA2_DP_PAD + kv_d] = vector<float16_t,2>(float16_t(v0), float16_t(v1));
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3b: Compute scores S_ij = Q @ K^T  [BR x BC] = [16 x 32]
        // 512 scores over 256 threads: thread gi handles flats gi and gi+256.
        for (uint i = 0; i < 2; i++) {
            uint flat = gi + i * 256;
            uint qr = flat / FA2_BC;
            uint kc = flat % FA2_BC;
            float dot_val = 0.0;
            for (uint dp = 0; dp < FA2_DP; dp++) {
                vector<float16_t,2> q2 = fa2_Qtile[qr * FA2_DP + dp];
                vector<float16_t,2> k2 = fa2_KV[kc * FA2_DP_PAD + dp];
                dot_val += float(q2.x) * float(k2.x) + float(q2.y) * float(k2.y);
            }
            dot_val *= scale;
            // Mask out-of-bounds
            bool valid = (kv_start + kc < S) && (q_start + qr < S);
            fa2_S[flat] = valid ? dot_val : -1.0e30;
        }
        GroupMemoryBarrierWithGroupSync();

        // 3c: Row-max reduction (32 threads per row via shared memory)
        for (uint i = 0; i < 2; i++) {
            uint flat = gi + i * 256;
            fa2_reduce[flat] = fa2_S[flat];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 16; s > 0; s >>= 1) {
            for (uint i = 0; i < 2; i++) {
                uint flat = gi + i * 256;
                if ((flat % FA2_BC) < s)
                    fa2_reduce[flat] = max(fa2_reduce[flat], fa2_reduce[flat + s]);
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // 3d: Online softmax update. Read tile max + m_old into registers for
        // both flats BEFORE fa2_reduce is overwritten with exp values.
        float m_new_r[2];
        float exp_r[2];
        for (uint i = 0; i < 2; i++) {
            uint flat = gi + i * 256;
            uint qr = flat / FA2_BC;
            uint kc = flat % FA2_BC;
            float tile_max = fa2_reduce[qr * FA2_BC];
            float m_old = fa2_m[qr];
            m_new_r[i] = max(m_old, tile_max);
            bool valid = (kv_start + kc < S) && (q_start + qr < S);
            exp_r[i] = valid ? exp(fa2_S[flat] - m_new_r[i]) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        // Row-sum reduction over the exp values
        for (uint i = 0; i < 2; i++) {
            uint flat = gi + i * 256;
            fa2_S[flat] = exp_r[i];
            fa2_reduce[flat] = exp_r[i];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 16; s > 0; s >>= 1) {
            for (uint i = 0; i < 2; i++) {
                uint flat = gi + i * 256;
                if ((flat % FA2_BC) < s)
                    fa2_reduce[flat] += fa2_reduce[flat + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Update running state (row-leader thread per row)
        for (uint i = 0; i < 2; i++) {
            uint flat = gi + i * 256;
            uint qr = flat / FA2_BC;
            if ((flat % FA2_BC) == 0) {
                float tile_sum = fa2_reduce[flat];
                float m_old = fa2_m[qr];
                float rescale_val = exp(m_old - m_new_r[i]);
                fa2_m[qr] = m_new_r[i];
                fa2_l[qr] = rescale_val * fa2_l[qr] + tile_sum;
                fa2_rescale[qr] = rescale_val;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3e: Load V tile into fa2_KV (overwrites K, coalesced pair loads)
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            if (flat < FA2_BC * FA2_DP) {
                uint kv_r = flat / FA2_DP;
                uint kv_d = flat % FA2_DP;
                uint actual_kv2 = kv_start + kv_r;
                float v0 = 0.0, v1 = 0.0;
                if (actual_kv2 < S) {
                    uint base = head_offset + actual_kv2 * D + kv_d * 2;
                    v0 = fa2_V[base];
                    v1 = fa2_V[base + 1];
                }
                fa2_KV[kv_r * FA2_DP_PAD + kv_d] = vector<float16_t,2>(float16_t(v0), float16_t(v1));
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3f: Update output: O = rescale * O + exp_scores @ V_tile
        // Each thread owns 4 (row, half2-pair) register slots.
        for (uint i = 0; i < 4; i++) {
            uint flat = gi + i * 256;
            uint o_qr = flat / FA2_DP;
            uint o_dp = flat % FA2_DP;

            float new_x = 0.0;
            float new_y = 0.0;
            for (uint kv = 0; kv < FA2_BC; kv++) {
                float s_val = fa2_S[o_qr * FA2_BC + kv];
                vector<float16_t,2> v2 = fa2_KV[kv * FA2_DP_PAD + o_dp];
                new_x += s_val * float(v2.x);
                new_y += s_val * float(v2.y);
            }
            float r = fa2_rescale[o_qr];
            acc_x[i] = acc_x[i] * r + new_x;
            acc_y[i] = acc_y[i] * r + new_y;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Step 4: Final normalization and output write (fused reverse transpose)
    uint dim = fa2_pc.n_heads * D;
    for (uint i = 0; i < 4; i++) {
        uint flat = gi + i * 256;
        uint o_qr = flat / FA2_DP;
        uint o_dp = flat % FA2_DP;
        uint actual_q = q_start + o_qr;
        if (actual_q < S) {
            float inv_l = 1.0 / fa2_l[o_qr];
            uint out_base = actual_q * dim + head * D + o_dp * 2;
            fa2_output[out_base]     = acc_x[i] * inv_l;
            fa2_output[out_base + 1] = acc_y[i] * inv_l;
        }
    }
}
