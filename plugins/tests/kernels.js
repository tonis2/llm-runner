// Parity: every shady kernel against the Slang kernel it replaces.
//
// Each case builds the same random inputs twice, runs the shady kernel on one
// set and the precompiled Slang entry point (the `.spv` the C3 pipelines ship)
// on the other, and compares the outputs. Slang's matmul and attention keep
// their tiles in fp16 where shady keeps f32, so those cases allow a relative
// error of about fp16's precision; everything else should agree to rounding.
//
//   ./build/llm-runner tests/kernels.js [only=name]

import { llm } from '../lib/llm.js';
import { kernel, pc, flashDefines } from '../lib/gpu.js';

const C = three.compute;
const only = globalThis.__llm_config?.only;

const SPV = {
	llm: 'dependencies/llm_text.c3l/shaders/llm.spv',
	zimage: 'dependencies/zimage.c3l/shaders/zimage.spv',
	flash: 'dependencies/zimage.c3l/shaders/flash_attn.spv',
	mrope: 'dependencies/zimage.c3l/shaders/mrope.spv',
	diffusion: 'lib/shaders/diffusion.spv',
	depth: 'dependencies/depth_anything.c3l/shaders/depth.spv',
	vit: 'dependencies/qwen_edit.c3l/shaders/vit.spv',
};
// The Slang reference of a pipeline whose C3 version has been deleted is gone
// with it; its cases are skipped rather than failed.
const modules = {};
class Skip extends Error {}
function slang(file, entry) {
	if (!modules[file]) {
		if (!llm.exists(SPV[file])) throw new Skip(`${SPV[file]} is gone with its C3 pipeline`);
		modules[file] = C.spirv(SPV[file]);
	}
	return C.kernel(modules[file], { name: entry, entry });
}

// A deterministic generator, so a failing case fails the same way twice.
let seed = 12345;
function rand() {
	seed = (seed * 1664525 + 1013904223) >>> 0;
	return seed / 4294967296;
}
function randFloats(n, scale = 1) {
	const a = new Float32Array(n);
	for (let i = 0; i < n; i++) a[i] = (rand() * 2 - 1) * scale;
	return a;
}
function f16bits(v) {
	const f = new Float32Array([v]);
	const u = new Uint32Array(f.buffer)[0];
	const sign = (u >>> 16) & 0x8000;
	const e = ((u >>> 23) & 0xff) - 127 + 15;
	if (e <= 0) return sign;
	if (e >= 31) return sign | 0x7c00;
	return sign | (e << 10) | ((u >>> 13) & 0x3ff);
}
// Random Q8_0 rows: [rows, cols/32 blocks of (f16 scale, 32 x int8)].
function randQ8(rows, cols) {
	const blocks = rows * cols / 32;
	const out = new Uint8Array(blocks * 34);
	for (let b = 0; b < blocks; b++) {
		const s = f16bits(0.002 + rand() * 0.02);
		out[b * 34] = s & 0xff;
		out[b * 34 + 1] = s >> 8;
		for (let i = 0; i < 32; i++) out[b * 34 + 2 + i] = Math.floor(rand() * 255) - 127 & 0xff;
	}
	return out;
}

// A binding spec: { f32: n, data?: Float32Array } or { bytes: Uint8Array }.
function makeBuffer(spec) {
	if (spec.bytes) return C.bytes(spec.bytes.length, spec.bytes);
	const n = spec.f32;
	return spec.data ? C.f32(n, spec.data) : C.f32(n);
}

// Pad with throwaway buffers when a Slang module declares more bindings than
// this entry uses — its reflection counts the whole module's.
function dispatchPadded(k, buffers, opts) {
	try {
		k.dispatch(buffers, opts);
	} catch (e) {
		const m = /binds (\d+) buffers/.exec(String(e.message));
		if (!m) throw e;
		const want = Number(m[1]);
		const padded = buffers.slice();
		while (padded.length < want) padded.push(C.f32(16));
		k.dispatch(padded, opts);
	}
}

const report = [];
let failures = 0;

function compare(name, def) {
	if (only && only !== name) return;
	try {
		const specs = def.bindings();
		const mine = specs.map(makeBuffer);
		const theirs = specs.map(makeBuffer);
		const push = def.push;
		kernel(def.shady ?? name).dispatch(mine, { workgroups: def.groups, push });
		dispatchPadded(slang(def.spv, def.entry ?? name), theirs, { workgroups: def.slangGroups ?? def.groups, push: def.slangPush ?? push });
		C.submit();
		let worst = 0;
		let worstAt = -1;
		let scale = 0;
		for (const o of def.outputs) {
			const a = new Float32Array(mine[o].readBytes().buffer);
			const b = new Float32Array(theirs[o].readBytes().buffer);
			const n = def.compareCount ?? a.length;
			for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(b[i]));
			for (let i = 0; i < n; i++) {
				const d = Math.abs(a[i] - b[i]);
				if (!(d <= worst)) { worst = d; worstAt = i; }
			}
		}
		const rel = scale > 0 ? worst / scale : worst;
		const tol = def.tol ?? 1e-5;
		const ok = rel <= tol;
		if (!ok) failures++;
		report.push(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(28)} max|diff| ${worst.toExponential(2)} (${rel.toExponential(2)} of max ${scale.toExponential(2)}) at ${worstAt}`);
		for (const b of mine.concat(theirs)) b.dispose();
	} catch (e) {
		if (e instanceof Skip) {
			report.push(`skip ${name.padEnd(28)} ${e.message}`);
			return;
		}
		failures++;
		report.push(`FAIL ${name.padEnd(28)} ${String(e.message).split('\n').slice(0, 4).join(' | ')}`);
	}
}

const N = 5000;
const g256 = (n) => [Math.ceil(n / 256), 1, 1];

compare('silu', { spv: 'llm', bindings: () => [{ f32: N, data: randFloats(N, 4) }], outputs: [0], push: pc('u', N), groups: g256(N) });
compare('elemwise_mul', { spv: 'llm', bindings: () => [{ f32: N, data: randFloats(N) }, { f32: N, data: randFloats(N) }], outputs: [0], push: pc('u', N), groups: g256(N) });
compare('residual_add', { spv: 'llm', bindings: () => [{ f32: N, data: randFloats(N) }, { f32: N, data: randFloats(N) }], outputs: [0], push: pc('u', N), groups: g256(N) });
compare('silu_mul', { spv: 'llm', bindings: () => [{ f32: N, data: randFloats(N, 3) }, { f32: N, data: randFloats(N) }], outputs: [0], push: pc('u', N), groups: g256(N) });
{
	const dim = 96, rows = 20, n = dim * rows;
	compare('adaln_modulate', { spv: 'zimage', bindings: () => [{ f32: n, data: randFloats(n) }, { f32: dim, data: randFloats(dim) }, { f32: dim, data: randFloats(dim) }, { f32: n }], outputs: [3], push: pc('uuuu', n, dim, 0, 0), groups: g256(n) });
	compare('gated_residual_linear', { spv: 'zimage', bindings: () => [{ f32: n, data: randFloats(n) }, { f32: dim, data: randFloats(dim) }, { f32: n, data: randFloats(n) }], outputs: [0], push: pc('uuuu', n, dim, 0, 0), groups: g256(n) });
}
compare('flow_euler_step', { spv: 'zimage', bindings: () => [{ f32: N, data: randFloats(N) }, { f32: N, data: randFloats(N) }], outputs: [0], push: pc('uf', N, -0.3), groups: g256(N) });
compare('scale_shift_clamp', { spv: 'diffusion', bindings: () => [{ f32: N, data: randFloats(N, 2) }, { f32: N }], outputs: [1], push: pc('uff', N, 0.5, 0.5), groups: g256(N) });
compare('upsample_nearest', { spv: 'diffusion', bindings: () => [{ f32: 3 * 7 * 9, data: randFloats(3 * 7 * 9) }, { f32: 3 * 14 * 18 }], outputs: [1], push: pc('uuu', 3, 7, 9), groups: g256(3 * 14 * 18) });
compare('patchify', { spv: 'zimage', bindings: () => [{ f32: 16 * 8 * 6, data: randFloats(16 * 8 * 6) }, { f32: 16 * 8 * 6 }], outputs: [1], push: pc('uuuu', 16, 8, 6, 2), groups: g256(16 * 8 * 6) });
compare('unpatchify', { spv: 'zimage', bindings: () => [{ f32: 16 * 8 * 6, data: randFloats(16 * 8 * 6) }, { f32: 16 * 8 * 6 }], outputs: [1], push: pc('uuuu', 16, 8, 6, 2), groups: g256(16 * 8 * 6) });
for (const dir of [0, 1]) {
	const n = 37 * 4 * 32;
	compare(`transpose_heads/${dir}`, { shady: 'transpose_heads', entry: 'transpose_heads', spv: 'zimage', bindings: () => [{ f32: n, data: randFloats(n) }, { f32: n }], outputs: [1], push: pc('uuuu', 37, 4, 32, dir), groups: g256(n) });
	compare(`transpose_channel_spatial/${dir}`, { shady: 'transpose_channel_spatial', entry: 'transpose_channel_spatial', spv: 'zimage', bindings: () => [{ f32: 12 * 50, data: randFloats(12 * 50) }, { f32: 12 * 50 }], outputs: [1], push: pc('uuu', 12, 50, dir), groups: g256(600) });
}
compare('timestep_embed', { entry: 'dit_timestep_embed', spv: 'zimage', bindings: () => [{ f32: 256 }], outputs: [0], push: pc('uf', 256, 734.5), groups: [1, 1, 1], tol: 1e-4 });
{
	const seq = 21, heads = 3, n = heads * seq * 128;
	compare('mrope', { spv: 'mrope', bindings: () => [{ f32: n, data: randFloats(n) }, { f32: seq * 64, data: randFloats(seq * 64) }, { f32: seq * 64, data: randFloats(seq * 64) }], outputs: [0], push: pc('uuuu', seq, heads, 0, 0), groups: [1, seq, heads] });
}
{
	const tokens = 9, heads = 4, dim = 64, n = tokens * heads * dim;
	compare('rope_batch', { spv: 'llm', bindings: () => [{ f32: n, data: randFloats(n) }], outputs: [0], push: pc('uuuf', dim, heads, tokens, 10000), groups: [Math.ceil(tokens * heads * dim / 2 / 128), 1, 1], tol: 1e-4 });
	compare('head_rmsnorm_batch', { spv: 'llm', bindings: () => [{ f32: n, data: randFloats(n) }, { f32: dim, data: randFloats(dim) }], outputs: [0], push: pc('uuuf', dim, heads, tokens, 1e-6), groups: g256(tokens * heads) });
}
{
	const dim = 700, rows = 5, n = dim * rows;
	compare('batch_layernorm', { spv: 'zimage', bindings: () => [{ f32: n, data: randFloats(n, 3) }, { f32: n }], outputs: [1], push: pc('ufuu', dim, 1e-6, rows, 0), groups: [rows, 1, 1] });
	compare('rmsnorm_batch', { spv: 'llm', bindings: () => [{ f32: n, data: randFloats(n, 3) }, { f32: dim, data: randFloats(dim) }, { f32: n }], outputs: [2], push: pc('ufu', dim, 1e-6, rows), groups: [rows, 1, 1] });
	compare('batch_head_norm', { spv: 'zimage', bindings: () => [{ f32: 6 * 4 * 128, data: randFloats(6 * 4 * 128) }, { f32: 128, data: randFloats(128) }], outputs: [0], push: pc('uuuf', 4, 128, 6, 1e-6), groups: [24, 1, 1] });
	compare('group_norm', { spv: 'diffusion', bindings: () => [{ f32: 64 * 30, data: randFloats(64 * 30, 2) }, { f32: 64, data: randFloats(64) }, { f32: 64, data: randFloats(64) }, { f32: 64 * 30 }], outputs: [3], push: pc('uuuf', 64, 30, 32, 1e-6), groups: [32, 1, 1] });
}
{
	// Q8_0 matmul: a fused-QKV-style slot (row_offset) with ragged edges.
	const out = 130, inDim = 256, seq = 70, off = 64, rows = 256;
	const w = randQ8(rows, inDim);
	const groups = [Math.ceil(seq / 64) * Math.ceil(out / 64), 1, 1];
	compare('matmul_q8', { entry: 'batch_matmul_q8', spv: 'llm', bindings: () => [{ bytes: w }, { f32: seq * inDim, data: randFloats(seq * inDim) }, { f32: seq * out }], outputs: [2], push: pc('uuuu', out, inDim, seq, off), groups, tol: 2e-3 });
	const wf = randFloats(rows * inDim, 0.05);
	compare('matmul_f32', { entry: 'batch_matmul', spv: 'llm', bindings: () => [{ f32: rows * inDim, data: wf }, { f32: seq * inDim, data: randFloats(seq * inDim) }, { f32: seq * out }], outputs: [2], push: pc('uuuu', out, inDim, seq, off), groups, tol: 1e-5 });
	compare('matmul_f32_rows', { entry: 'batch_matmul_simple', spv: 'llm', bindings: () => [{ f32: rows * inDim, data: wf }, { f32: 3 * inDim, data: randFloats(3 * inDim) }, { f32: 3 * out }], outputs: [2], push: pc('uuuu', out, inDim, 3, off), groups: [3, out, 1], tol: 1e-5 });
	compare('linear_bias', { entry: 'linear_proj', spv: 'diffusion', bindings: () => [{ f32: out * inDim, data: randFloats(out * inDim, 0.05) }, { f32: out, data: randFloats(out) }, { f32: 5 * inDim, data: randFloats(5 * inDim) }, { f32: 5 * out }], outputs: [3], push: pc('uuu', out, inDim, 5), groups: [5 * out, 1, 1], tol: 1e-5 });
}
{
	const heads = 3, seq = 77, n = heads * seq * 128;
	compare('flash_attention', { entry: 'flash_attention_v2', spv: 'flash', bindings: () => [{ f32: n, data: randFloats(n) }, { f32: n, data: randFloats(n) }, { f32: n, data: randFloats(n) }, { f32: n }], outputs: [3], push: pc('uuuf', 128, heads, seq, 1 / Math.sqrt(128)), groups: [Math.ceil(seq / 16), heads, 1], tol: 3e-3 });
}
{
	// Causal attention: the Slang kernel reads a [kv_heads, max_seq, hd] cache,
	// so its K/V are laid out for it from the same numbers.
	const hd = 64, nq = 8, nkv = 2, n = 13, maxSeq = 16;
	const q = randFloats(n * nq * hd), k = randFloats(n * nkv * hd), v = randFloats(n * nkv * hd);
	const kc = new Float32Array(nkv * maxSeq * hd), vc = new Float32Array(nkv * maxSeq * hd);
	for (let p = 0; p < n; p++) for (let h = 0; h < nkv; h++) for (let d = 0; d < hd; d++) {
		kc[h * maxSeq * hd + p * hd + d] = k[p * nkv * hd + h * hd + d];
		vc[h * maxSeq * hd + p * hd + d] = v[p * nkv * hd + h * hd + d];
	}
	if (!only || only === 'attention_causal') {
		const mine = [C.f32(q.length, q), C.f32(k.length, k), C.f32(v.length, v), C.f32(n * nq * hd)];
		const theirs = [C.f32(q.length, q), C.f32(kc.length, kc), C.f32(vc.length, vc), C.f32(n * nq * hd)];
		kernel('attention_causal').dispatch(mine, { workgroups: [nq, n, 1], push: pc('uuuuf', hd, nkv, nq, n, 1 / Math.sqrt(hd)) });
		dispatchPadded(slang('llm', 'attention_prefill'), theirs, { workgroups: [nq, n, 1], push: pc('uuuufu', hd, nkv, nq, n, 1 / Math.sqrt(hd), maxSeq) });
		C.submit();
		const a = new Float32Array(mine[3].readBytes().buffer), b = new Float32Array(theirs[3].readBytes().buffer);
		let worst = 0;
		for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
		const ok = worst < 1e-5;
		if (!ok) failures++;
		report.push(`${ok ? 'ok  ' : 'FAIL'} ${'attention_causal'.padEnd(28)} max|diff| ${worst.toExponential(2)}`);
	}
}
{
	const S = 300, Cn = 64;
	compare('vae_attention', { spv: 'diffusion', bindings: () => [{ f32: S * Cn, data: randFloats(S * Cn) }, { f32: S * Cn, data: randFloats(S * Cn) }, { f32: S * Cn, data: randFloats(S * Cn) }, { f32: S * Cn }], outputs: [3], push: pc('uuufu', Cn, 1, S, 1 / 8, Cn), groups: [S, 1, 1], tol: 1e-5 });
}
{
	const ic = 6, oc = 10, h = 19, w = 23;
	const push3 = pc('uuuuuuuuuuuu', ic, oc, h, w, 3, 3, 1, 1, 1, h, w, 1);
	const weights = randFloats(9 * ic * oc, 0.2), bias = randFloats(oc), input = randFloats(ic * h * w);
	compare('conv2d_3x3', { spv: 'diffusion', bindings: () => [{ f32: weights.length, data: weights }, { f32: oc, data: bias }, { f32: input.length, data: input }, { f32: oc * h * w }], outputs: [3], push: push3, groups: [Math.ceil(w / 16), Math.ceil(h / 16), Math.ceil(oc / 4)], tol: 1e-5 });
	const oh = Math.floor((h + 2 - 3) / 2) + 1, ow = Math.floor((w + 2 - 3) / 2) + 1;
	const pushS = pc('uuuuuuuuuuuu', ic, oc, h, w, 3, 3, 2, 1, 1, oh, ow, 1);
	compare('conv2d', { spv: 'diffusion', bindings: () => [{ f32: weights.length, data: weights }, { f32: oc, data: bias }, { f32: input.length, data: input }, { f32: oc * oh * ow }], outputs: [3], push: pushS, groups: [Math.ceil(oh * ow / 256), oc, 1], tol: 1e-5 });
}

compare('gelu', { spv: 'llm', bindings: () => [{ f32: N, data: randFloats(N, 5) }], outputs: [0], push: pc('u', N), groups: g256(N), tol: 1e-6 });
compare('relu', { spv: 'diffusion', bindings: () => [{ f32: N, data: randFloats(N) }], outputs: [0], push: pc('u', N), groups: g256(N) });
compare('bilinear_resize', { spv: 'depth', bindings: () => [{ f32: 3 * 9 * 13, data: randFloats(3 * 9 * 13) }, { f32: 3 * 20 * 17 }], outputs: [1], push: pc('uuuuu', 3, 9, 13, 20, 17), groups: [Math.ceil(20 * 17 / 256), 3, 1] });
{
	const ic = 5, oc = 7, h = 6, w = 9, k = 4;
	compare('conv_transpose', { spv: 'depth', bindings: () => [{ f32: ic * oc * k * k, data: randFloats(ic * oc * k * k) }, { f32: oc, data: randFloats(oc) }, { f32: ic * h * w, data: randFloats(ic * h * w) }, { f32: oc * h * k * w * k }], outputs: [3], push: pc('uuuuuu', ic, oc, h, w, k, 1), groups: [Math.ceil(h * k * w * k / 256), oc, 1] });
}
if ((!only || only === 'layernorm_affine') && !llm.exists(SPV.vit)) {
	report.push(`skip ${'layernorm_affine'.padEnd(28)} ${SPV.vit} is gone with its C3 pipeline`);
} else if (!only || only === 'layernorm_affine') {
	// Slang does it in two passes: the scaled norm, then the bias.
	const dim = 384, rows = 7, n = dim * rows;
	const x = randFloats(n, 3), w = randFloats(dim), b = randFloats(dim);
	const mine = [C.f32(n, x), C.f32(dim, w), C.f32(dim, b), C.f32(n)];
	const t0 = [C.f32(n, x), C.f32(dim, w), C.f32(n)], tb = C.f32(dim, b);
	kernel('layernorm_affine').dispatch(mine, { workgroups: [rows, 1, 1], push: pc('ufuu', dim, 1e-6, rows, 0) });
	dispatchPadded(slang('vit', 'batch_layernorm_affine'), t0, { workgroups: [rows, 1, 1], push: pc('ufuu', dim, 1e-6, rows, 0) });
	dispatchPadded(slang('vit', 'broadcast_bias_add'), [t0[2], tb], { workgroups: [rows, 2, 1], push: pc('uu', dim, rows) });
	C.submit();
	const a = new Float32Array(mine[3].readBytes().buffer), bb = new Float32Array(t0[2].readBytes().buffer);
	let worst = 0;
	for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(a[i] - bb[i]));
	const ok = worst < 1e-5;
	if (!ok) failures++;
	report.push(`${ok ? 'ok  ' : 'FAIL'} ${'layernorm_affine'.padEnd(28)} max|diff| ${worst.toExponential(2)}`);
}
if (!only || only === 'flash_attention/64') {
	// Head width 64 against Slang's spatial_attention, whose output stays [heads, seq, hd].
	const heads = 6, seq = 97, hd = 64, n = heads * seq * hd;
	const q = randFloats(n), k = randFloats(n), v = randFloats(n);
	const mine = [C.f32(n, q), C.f32(n, k), C.f32(n, v), C.f32(n)];
	const theirs = [C.f32(n, q), C.f32(n, k), C.f32(n, v), C.f32(heads * seq * seq), C.f32(n)];
	kernel('flash_attention', flashDefines(hd)).dispatch(mine, { workgroups: [Math.ceil(seq / 16), heads, 1], push: pc('uuuf', hd, heads, seq, 1 / 8) });
	dispatchPadded(slang('diffusion', 'spatial_attention'), theirs, { workgroups: [heads * seq, 1, 1], push: pc('uuuf', hd, heads, seq, 1 / 8) });
	C.submit();
	const a = new Float32Array(mine[3].readBytes().buffer), b = new Float32Array(theirs[4].readBytes().buffer);
	let worst = 0;
	for (let h = 0; h < heads; h++) for (let p = 0; p < seq; p++) for (let d = 0; d < hd; d++) {
		worst = Math.max(worst, Math.abs(a[p * heads * hd + h * hd + d] - b[h * seq * hd + p * hd + d]));
	}
	const ok = worst < 1e-5;
	if (!ok) failures++;
	report.push(`${ok ? 'ok  ' : 'FAIL'} ${'flash_attention/64'.padEnd(28)} max|diff| ${worst.toExponential(2)}`);
}

for (const line of report) llm.print(line);
const skipped = report.filter((l) => l.startsWith('skip')).length;
llm.print(failures === 0 ? `all ${report.length - skipped} kernels agree${skipped ? `, ${skipped} skipped` : ''}` : `${failures} of ${report.length} disagree`);
