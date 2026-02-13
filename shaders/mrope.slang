// Multimodal Rotary Position Embedding (3-axis mRoPE) for DiT
// Separate file to avoid AMD driver SPIR-V compilation issues
// Applies RoPE to Q or K in [n_heads, seq_len, head_dim] layout
// Uses precomputed cos/sin tables of shape [seq_len, head_dim/2]
// INTERLEAVED format: pair j rotates (x[2j], x[2j+1])
// Rotation convention matches KoboldCpp rope.hpp apply_rope (standard RoPE e^{+iθ}):
//   out[2j]   =  x[2j]*cos - x[2j+1]*sin
//   out[2j+1] =  x[2j]*sin + x[2j+1]*cos
// Dispatched as 2D: gid.x = pair index (0..63), gid.y = head*seq_len + pos

struct MRoPEParams {
    uint seq_len;
    uint n_heads;
    uint _pad0;
    uint _pad1;
};

[vk_push_constant] const MRoPEParams mrope_pc;

[vk_binding(0, 0)] RWStructuredBuffer<float> mrope_data;
[vk_binding(1, 0)] RWStructuredBuffer<float> mrope_cos;
[vk_binding(2, 0)] RWStructuredBuffer<float> mrope_sin;

static const uint ROPE_HALF_DIM = 64;
static const uint ROPE_HEAD_DIM = 128;

[shader("compute")]
[numthreads(64, 1, 1)]
void mrope(uint3 tid: SV_DispatchThreadID) {
    uint pair = tid.x;
    uint head_pos = tid.y;
    if (pair >= ROPE_HALF_DIM) return;
    if (head_pos >= mrope_pc.n_heads * mrope_pc.seq_len) return;

    uint head = head_pos / mrope_pc.seq_len;
    uint pos = head_pos % mrope_pc.seq_len;

    uint cs_idx = pos * ROPE_HALF_DIM + pair;
    float cv = mrope_cos[cs_idx];
    float sv = mrope_sin[cs_idx];

    // Interleaved: pair j uses elements (2j) and (2j+1)
    uint base = head * mrope_pc.seq_len * ROPE_HEAD_DIM + pos * ROPE_HEAD_DIM;
    float x0 = mrope_data[base + 2 * pair];
    float x1 = mrope_data[base + 2 * pair + 1];

    mrope_data[base + 2 * pair]     =  x0 * cv + x1 * sv;
    mrope_data[base + 2 * pair + 1] = -x0 * sv + x1 * cv;
}
