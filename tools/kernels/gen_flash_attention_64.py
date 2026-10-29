# Writes llm/core/kernels/flash_attention_64.shady, unrolled so a thread's
# query row and output stay in registers.
#
#   python3 tools/kernels/gen_flash_attention_64.py
#
# CH keys a step and WG threads a workgroup were picked on an Apple M5 (8 and
# 128: 27 ms for 20 heads of 1029 tokens, against 32 ms at 8 and 64, 36 ms at
# 4 and 64, 49 ms at 8 and 32). ops.js dispatches ceil(seq / WG) groups.
import os
CH, WG = 8, 128
out = os.path.join(os.path.dirname(__file__), '../../llm/core/kernels/flash_attention_64.shady')
D4 = 16
L = []; w = L.append
w(f"""// Flash attention for head_dim 64 without matrix cores: one thread a query
// row, its Q (pre-scaled) and output in registers, so a row's softmax needs no
// reductions and a 32-key tile takes two barriers. The K and V tiles sit in
// shared memory, which every thread of the workgroup reads at the same address
// (a broadcast). Keys go {CH} at a time: their scores, one rescale of the
// output, then their V rows. Non-causal; written out unrolled (by
// tools/kernels/gen_flash_attention_64.py) so Q and the output stay in registers.
// Q, K, V: [heads, seq, 64]. Output: [seq, heads * 64].
// Dispatch: (ceil(seq/{WG}), heads).
float4 q[];
float4 k[];
float4 v[];
float output[];
struct Push @pushconstant {{ uint n_heads; uint seq_len; float scale; }}
Push pc;
shared float4 ks[512];      // [32 keys][16]
shared float4 vs[512];
struct ComputeIn {{ uint3 gid @builtin(workgroup_id); uint3 lid @builtin(local_invocation_id); }}
fn void main(ComputeIn input) @compute @threads({WG}, 1, 1)
{{
    uint t = input.lid.x;
    uint head = input.gid.y;
    uint row = input.gid.x * {WG}u + t;
    uint S = pc.seq_len;
    uint base = head * S * 16u;
    uint qi = base + min(row, S - 1u) * 16u;""")
for d in range(D4): w(f"    float4 q{d} = q[qi + {d}u] * pc.scale;")
for d in range(D4): w(f"    float4 o{d} = float4(0.0);")
w(f"""    float m = -1.0e30;
    float l = 0.0;
    uint tiles = (S + 31u) / 32u;
    for (uint tile = 0u; tile < tiles; tile++) {{
        uint k0 = tile * 32u;
        barrier;
        for (uint i = 0u; i < {512 // WG}u; i++) {{
            uint f = t + i * {WG}u;
            uint key = k0 + f / 16u;
            uint at = base + key * 16u + f % 16u;
            ks[f] = key < S ? k[at] : float4(0.0);
            vs[f] = key < S ? v[at] : float4(0.0);
        }}
        barrier;
        uint valid = min(32u, S - k0);
        for (uint c = 0u; c < 32u; c += {CH}u) {{""")
for j in range(CH):
    terms = " + ".join(f"q{d} * ks[r{j} + {d}u]" for d in range(D4))
    w(f"            uint r{j} = (c + {j}u) * 16u;")
    w(f"            float4 a{j} = {terms};")
    w(f"            float s{j} = c + {j}u < valid ? a{j}.x + a{j}.y + a{j}.z + a{j}.w : -1.0e30;")
def tree(xs, op):
    while len(xs) > 1:
        xs = [op(xs[i], xs[i+1]) if i+1 < len(xs) else xs[i] for i in range(0, len(xs), 2)]
    return xs[0]
w(f"            float mx = max(m, {tree([f's{j}' for j in range(CH)], lambda a,b: f'max({a}, {b})')});")
w("            float r = exp(m - mx);")
for j in range(CH): w(f"            float p{j} = exp(s{j} - mx);")
w(f"            l = l * r + {tree([f'p{j}' for j in range(CH)], lambda a,b: f'({a} + {b})')};")
w("            m = mx;")
for d in range(D4):
    terms = " + ".join(f"p{j} * vs[r{j} + {d}u]" for j in range(CH))
    w(f"            o{d} = o{d} * r + {terms};")
w("""        }
    }
    if (row >= S) return;
    float inv = 1.0 / l;
    uint ob = row * pc.n_heads * 64u + head * 64u;""")
for d in range(D4):
    for c, comp in enumerate('xyzw'):
        w(f"    output[ob + {4*d+c}u] = o{d}.{comp} * inv;")
w("}")
open(out,'w').write("\n".join(L)+"\n")
