# Writes llm/core/kernels/matmul_q8_8x8.shady: the shader language has no
# loops the compiler is sure to unroll, so the 64 accumulators are written out.
#
#   python3 tools/kernels/gen_matmul_q8_8x8.py
import os
out = os.path.join(os.path.dirname(__file__), '../../llm/core/kernels/matmul_q8_8x8.shady')
L = []; w = L.append
w("""// y[seq, out_dim] = x[seq, in_dim] @ W[row_offset .. row_offset + out_dim, :]^T, W in Q8_0,
// without matrix cores.
//
// A 128x128 output tile per workgroup and an 8x8 patch of it per thread - rows
// tm*4.. and 64 + tm*4.., columns likewise - so a thread's k step is four
// 128-bit shared-memory reads for 64 multiply-adds, twice the 4x4 kernel's
// (matmul_q8), whose reads are what bound it on GPUs without matrix cores. K in
// steps of 16, half a Q8_0 block; the weight is dequantised on its way into
// shared memory. Tiles sit k-major as float4s, rows 33 float4s apart.
// Written out unrolled (tools/kernels/gen_matmul_q8_8x8.py) so the
// accumulators stay in registers.
// Dispatch: ceil(seq/128) * ceil(out_dim/128).
uint8 w[];
float x[];
float y[];
struct Push @pushconstant { uint out_dim; uint in_dim; uint seq_len; uint row_offset; }
Push pc;
shared float4 As[528];      // [16 k][32 m4 + 1]
shared float4 Bs[528];      // [16 k][32 n4 + 1]
struct ComputeIn { uint3 gid @builtin(workgroup_id); uint3 lid @builtin(local_invocation_id); }

fn float a_at(uint gm, uint col)
{
    return gm < pc.seq_len ? x[gm * pc.in_dim + col] : 0.0;
}

fn float b_at(uint gn, uint byte, float s)
{
    return gn < pc.row_offset + pc.out_dim ? s * float(sign_extend8(uint(w[byte]))) : 0.0;
}

fn float scale_at(uint gn, uint bb)
{
    return gn < pc.row_offset + pc.out_dim ? unpack_f16(uint(w[bb]) | (uint(w[bb + 1u]) << 8u)) : 0.0;
}

fn void main(ComputeIn input) @compute @threads(256, 1, 1)
{
    uint gi = input.lid.x;
    uint tiles_n = (pc.out_dim + 127u) / 128u;
    uint tile_m = input.gid.x / tiles_n;
    uint tile_n = input.gid.x % tiles_n;
    uint pos_base = tile_m * 128u;
    uint row_base = pc.row_offset + tile_n * 128u;
    uint tm = gi / 16u;
    uint tn = gi % 16u;
    uint row_bytes = (pc.in_dim / 32u) * 34u;
""")
for i in range(8):
    for j in range(2):
        w(f"    float4 c{i}{j} = float4(0.0);")
w("""
    uint steps = pc.in_dim / 16u;
    for (uint t = 0u; t < steps; t++) {
        uint k_base = t * 16u;
        uint block = t / 2u;
        uint half_k = (t % 2u) * 16u;
        // A: two float4s a thread, four rows at one k; lanes run along k.
        for (uint i = 0u; i < 2u; i++) {
            uint flat = gi + i * 256u;
            uint k = flat & 15u;
            uint m4 = flat >> 4u;
            uint gm = pos_base + m4 * 4u;
            uint col = k_base + k;
            As[k * 33u + m4] = float4(a_at(gm, col), a_at(gm + 1u, col), a_at(gm + 2u, col), a_at(gm + 3u, col));
        }
        // B: four weight rows at two k a thread, each row's scale read once.
        {
            uint k = (gi & 7u) * 2u;
            uint n4 = gi >> 3u;
            uint gn = row_base + n4 * 4u;
            uint bb = gn * row_bytes + block * 34u;
            uint byte = bb + 2u + half_k + k;
            float s0 = scale_at(gn, bb);
            float s1 = scale_at(gn + 1u, bb + row_bytes);
            float s2 = scale_at(gn + 2u, bb + 2u * row_bytes);
            float s3 = scale_at(gn + 3u, bb + 3u * row_bytes);
            Bs[k * 33u + n4] = float4(b_at(gn, byte, s0), b_at(gn + 1u, byte + row_bytes, s1),
                b_at(gn + 2u, byte + 2u * row_bytes, s2), b_at(gn + 3u, byte + 3u * row_bytes, s3));
            Bs[k * 33u + 33u + n4] = float4(b_at(gn, byte + 1u, s0), b_at(gn + 1u, byte + 1u + row_bytes, s1),
                b_at(gn + 2u, byte + 1u + 2u * row_bytes, s2), b_at(gn + 3u, byte + 1u + 3u * row_bytes, s3));
        }
        barrier;
        for (uint k = 0u; k < 16u; k++) {
            float4 a0 = As[k * 33u + tm];
            float4 a1 = As[k * 33u + 16u + tm];
            float4 b0 = Bs[k * 33u + tn];
            float4 b1 = Bs[k * 33u + 16u + tn];""")
comps = 'xyzw'
for i in range(8):
    a = f"a{i // 4}.{comps[i % 4]}"
    for j in range(2):
        w(f"            c{i}{j} += {a} * b{j};")
w("""        }
        barrier;
    }
""")
for i in range(8):
    row = f"pos_base + {64 if i >= 4 else 0}u + tm * 4u + {i % 4}u"
    for j in range(2):
        w(f"    store4({row}, tile_n * 128u + {64 * j}u + tn * 4u, c{i}{j});")
w("""}

fn void store4(uint gm, uint col0, float4 v)
{
    if (gm >= pc.seq_len) return;
    uint base = gm * pc.out_dim + col0;
    if (col0 < pc.out_dim) y[base] = v.x;
    if (col0 + 1u < pc.out_dim) y[base + 1u] = v.y;
    if (col0 + 2u < pc.out_dim) y[base + 2u] = v.z;
    if (col0 + 3u < pc.out_dim) y[base + 3u] = v.w;
}""")
open(out, 'w').write("\n".join(L) + "\n")
