// Flash-attention for the SkinTokens mesh encoder.
//
// Its own module, like flash_attn.slang: SPIR-V charges a module's whole
// groupshared footprint to every entry point in it, and the tiles here are
// ~14 KB.
//
// Two things it does that the zimage kernel does not:
//
//  - Query and key/value lengths are independent. The encoder's one cross
//    attention block has 256 queries against 54000 points, which is the whole
//    reason this is tiled rather than materialised: the score matrix alone
//    would be 442 MB.
//
//  - Q, K and V are addressed by stride rather than assumed to live in three
//    separate [head, seq, dim] buffers. Michelangelo fuses them per head --
//    c_qkv emits [q|k|v] for head 0, then head 1, and c_kv emits [k|v] the
//    same way -- so a fused projection can be attended straight out of, with
//    no split or transpose pass in between. Self attention binds the same
//    buffer as both Q and KV and just moves the offsets.
//
// head_dim is fixed at 64. That is what the mesh encoder (512/8) and the skin
// VAE (768/12) both use, and pinning it lets the tiles be statically sized.

static const uint ME_BR = 16;    // query rows per workgroup
static const uint ME_BC = 32;    // kv positions per tile
static const uint ME_D = 64;     // head dimension
static const uint ME_DPAD = 65;  // +1 so the score loop's strided read is conflict-free

struct MeshAttentionParams {
    uint n_heads;
    uint q_len;
    uint kv_len;
    uint q_row_stride;
    uint q_head_stride;
    uint q_offset;
    uint kv_row_stride;
    uint kv_head_stride;
    uint k_offset;
    uint v_offset;
    uint out_row_stride;
    float scale;
};

[vk_push_constant] const MeshAttentionParams me_attn_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> me_attn_q;
[vk_binding(1, 0)] RWStructuredBuffer<float> me_attn_kv;
[vk_binding(2, 0)] RWStructuredBuffer<float> me_attn_out;

groupshared float me_attn_qtile[ME_BR * ME_D];
groupshared float me_attn_kvtile[ME_BC * ME_DPAD];   // K, then overwritten by V
groupshared float me_attn_scores[ME_BR * ME_BC];
groupshared float me_attn_m[ME_BR];
groupshared float me_attn_l[ME_BR];
groupshared float me_attn_rescale[ME_BR];

[shader("compute")]
[numthreads(256, 1, 1)]
void mesh_attention(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    uint head = gid.y;
    if (head >= me_attn_pc.n_heads) return;

    uint q_start = gid.x * ME_BR;
    uint q_len = me_attn_pc.q_len;
    uint kv_len = me_attn_pc.kv_len;

    uint q_head_base = head * me_attn_pc.q_head_stride + me_attn_pc.q_offset;
    uint k_head_base = head * me_attn_pc.kv_head_stride + me_attn_pc.k_offset;
    uint v_head_base = head * me_attn_pc.kv_head_stride + me_attn_pc.v_offset;

    // Q tile: ME_BR * ME_D = 1024 slots over 256 threads.
    for (uint i = 0; i < 4; i++) {
        uint flat = gi + i * 256;
        uint qr = flat / ME_D;
        uint qd = flat % ME_D;
        uint aq = q_start + qr;
        me_attn_qtile[flat] = aq < q_len
            ? me_attn_q[aq * me_attn_pc.q_row_stride + q_head_base + qd] : 0.0;
    }

    // Output accumulator stays in registers, one slot per Q tile slot.
    float acc[4];
    for (uint i = 0; i < 4; i++) acc[i] = 0.0;

    if (gi < ME_BR) {
        me_attn_m[gi] = -1.0e30;
        me_attn_l[gi] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint blocks = (kv_len + ME_BC - 1) / ME_BC;
    for (uint b = 0; b < blocks; b++) {
        uint kv_start = b * ME_BC;

        // K tile: ME_BC * ME_D = 2048 slots over 256 threads.
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint kr = flat / ME_D;
            uint kd = flat % ME_D;
            uint akv = kv_start + kr;
            me_attn_kvtile[kr * ME_DPAD + kd] = akv < kv_len
                ? me_attn_kv[akv * me_attn_pc.kv_row_stride + k_head_base + kd] : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        // Scores: ME_BR * ME_BC = 512 slots over 256 threads.
        for (uint i = 0; i < 2; i++) {
            uint flat = gi + i * 256;
            uint qr = flat / ME_BC;
            uint kc = flat % ME_BC;

            float dot_val = 0.0;
            for (uint d = 0; d < ME_D; d++) {
                dot_val += me_attn_qtile[qr * ME_D + d] * me_attn_kvtile[kc * ME_DPAD + d];
            }
            bool valid = (kv_start + kc < kv_len) && (q_start + qr < q_len);
            me_attn_scores[flat] = valid ? dot_val * me_attn_pc.scale : -1.0e30;
        }
        GroupMemoryBarrierWithGroupSync();

        // Online softmax, one thread per query row. ME_BC is small enough that
        // scanning it serially costs less than the barriers a tree reduction
        // over 256 threads would need.
        if (gi < ME_BR) {
            uint base = gi * ME_BC;

            float tile_max = -1.0e30;
            for (uint c = 0; c < ME_BC; c++) tile_max = max(tile_max, me_attn_scores[base + c]);

            float m_old = me_attn_m[gi];
            float m_new = max(m_old, tile_max);
            float rescale = exp(m_old - m_new);

            float tile_sum = 0.0;
            for (uint c = 0; c < ME_BC; c++) {
                float p = exp(me_attn_scores[base + c] - m_new);
                me_attn_scores[base + c] = p;
                tile_sum += p;
            }

            me_attn_m[gi] = m_new;
            me_attn_l[gi] = me_attn_l[gi] * rescale + tile_sum;
            me_attn_rescale[gi] = rescale;
        }

        // V tile overwrites K. Safe to start before the softmax above finishes:
        // the barrier after the score loop already retired every read of K, and
        // the softmax touches only the score and running-state arrays.
        for (uint i = 0; i < 8; i++) {
            uint flat = gi + i * 256;
            uint kr = flat / ME_D;
            uint kd = flat % ME_D;
            uint akv = kv_start + kr;
            me_attn_kvtile[kr * ME_DPAD + kd] = akv < kv_len
                ? me_attn_kv[akv * me_attn_pc.kv_row_stride + v_head_base + kd] : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint i = 0; i < 4; i++) {
            uint flat = gi + i * 256;
            uint qr = flat / ME_D;
            uint qd = flat % ME_D;

            float tile_out = 0.0;
            for (uint c = 0; c < ME_BC; c++) {
                tile_out += me_attn_scores[qr * ME_BC + c] * me_attn_kvtile[c * ME_DPAD + qd];
            }
            acc[i] = acc[i] * me_attn_rescale[qr] + tile_out;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint i = 0; i < 4; i++) {
        uint flat = gi + i * 256;
        uint qr = flat / ME_D;
        uint qd = flat % ME_D;
        uint aq = q_start + qr;
        if (aq < q_len) {
            me_attn_out[aq * me_attn_pc.out_row_stride + head * ME_D + qd] = acc[i] / me_attn_l[qr];
        }
    }
}
