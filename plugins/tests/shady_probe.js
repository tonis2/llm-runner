import { llm } from '../lib/llm.js';
const C = three.compute;
const results = {};
function tryKernel(name, src) {
	try { const k = C.kernel(src, { name }); results[name] = 'ok'; return k; }
	catch (e) { results[name] = String(e.message).slice(0, 400); return null; }
}
tryKernel('loops_shared', `
float a[];
float o[];
shared float tile[256];
struct Push @pushconstant { uint n; float s; }
Push push;
struct ComputeIn { uint3 gid @builtin(workgroup_id); uint3 lid @builtin(local_invocation_id); }
fn float helper(uint i) { return a[i] * 2.0; }
fn void main(ComputeIn input) @compute @threads(256, 1, 1)
{
    uint gi = input.lid.x;
    float acc[4];
    for (uint k = 0u; k < 4u; k++) acc[k] = 0.0;
    float partial = 0.0;
    for (uint i = gi; i < push.n; i += 256u) { partial += helper(i); }
    tile[gi] = partial;
    barrier;
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (gi < s) tile[gi] += tile[gi + s];
        barrier;
    }
    acc[gi & 3u] = exp(0.0) + sqrt(4.0) + max(1.0, 2.0) + pow(2.0, 3.0) + cos(0.0) + log(1.0);
    if (gi == 0u) o[input.gid.x] = tile[0] * push.s + acc[0];
}
`);
tryKernel('bytes', `
uint8 w[];
float o[];
struct ComputeIn { uint3 id @builtin(global_invocation_id); }
fn float unpack_f16(uint bits)
{
    uint sign = (bits >> 15u) & 1u;
    uint e = (bits >> 10u) & 31u;
    uint m = bits & 1023u;
    float v;
    if (e == 0u) { v = float(m) * 5.9604645e-8; }
    else if (e == 31u) { v = asfloat(0x7F800000u); }
    else { v = asfloat(((e + 112u) << 23u) | (m << 13u)); }
    return sign != 0u ? -v : v;
}
fn void main(ComputeIn input) @compute @threads(64, 1, 1)
{
    uint i = input.id.x;
    uint lo = uint(w[i * 2u]);
    uint hi = uint(w[i * 2u + 1u]);
    int q = int(uint(w[i]));
    if (q > 127) q = q - 256;
    o[i] = unpack_f16(lo | (hi << 8u)) + float(q) * 0.0;
}
`);
tryKernel('int_mix', `
float o[];
struct ComputeIn { uint3 id @builtin(global_invocation_id); }
fn void main(ComputeIn input) @compute @threads(64, 1, 1)
{
    uint i = input.id.x;
    int ih = int(i) - 1;
    bool valid = ih >= 0 && ih < 10;
    float v = valid ? 1.0 : 0.0;
    o[i] = v + float(ih) * 0.5 + (float)(i % 3u);
}
`);
tryKernel('tanh_min', `
float o[];
struct ComputeIn { uint3 id @builtin(global_invocation_id); }
fn void main(ComputeIn input) @compute @threads(64, 1, 1)
{
    uint i = input.id.x;
    o[i] = clamp(float(i), 0.0, 1.0) + min(1u, i) * 0u + float(min(3u, i));
}
`);
tryKernel('shared2d', `
float o[];
shared float t[2112];
shared float b[33];
struct ComputeIn { uint3 id @builtin(global_invocation_id); uint3 lid @builtin(local_invocation_id); }
fn void main(ComputeIn input) @compute @threads(256, 1, 1)
{
    uint i = input.lid.x;
    t[i * 8u] = 1.0;
    b[i % 33u] = 2.0;
    barrier;
    o[input.id.x] = t[i] + b[0];
}
`);
// run the bytes kernel on a known f16 value 1.5 = 0x3E00
const k = C.kernel(`
uint8 w[];
float o[];
struct ComputeIn { uint3 id @builtin(global_invocation_id); }
fn float unpack_f16(uint bits)
{
    uint sign = (bits >> 15u) & 1u;
    uint e = (bits >> 10u) & 31u;
    uint m = bits & 1023u;
    float v;
    if (e == 0u) { v = float(m) * 5.9604645e-8; }
    else if (e == 31u) { v = asfloat(0x7F800000u); }
    else { v = asfloat(((e + 112u) << 23u) | (m << 13u)); }
    return sign != 0u ? -v : v;
}
fn void main(ComputeIn input) @compute @threads(64, 1, 1)
{
    uint i = input.id.x;
    if (i >= 3u) return;
    uint lo = uint(w[i * 2u]);
    uint hi = uint(w[i * 2u + 1u]);
    o[i] = unpack_f16(lo | (hi << 8u));
}
`, { name: 'f16run' });
const w = C.bytes(8, new Uint8Array([0x00, 0x3E, 0x00, 0xC0, 0x01, 0x00, 0, 0]));
const o = C.f32(64);
k.run([w, o], { threads: 3 });
o.read();
results.f16 = [o.f32(0), o.f32(1), o.f32(2)];
results.limits = C.limits;
llm.print(JSON.stringify(results, null, 1));
