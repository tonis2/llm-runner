// The 3D and sparse kernels LATO.2 is written in, each against a plain
// JavaScript version on small random inputs.
//
//   llm-runner tests/lato2_kernels.js
import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit } from '/core/gpu.js';

let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
const C = three.compute;
const up = (a) => { const t = f32(a.length); t.buffer.write(a); return t; };
const u32 = (a) => C.u32(a.length, Uint32Array.from(a));
const read = (t, n) => new Float32Array(t.buffer.readBytes(0, n * 4).buffer);
const rand = (n, seed, s = 1) => llm.randomNormal(n, seed).map((v) => v * s);
function worst(a, b) {
	let w = 0, m = 0;
	for (let i = 0; i < b.length; i++) { w = Math.max(w, Math.abs(a[i] - b[i])); m = Math.max(m, Math.abs(b[i])); }
	return w / Math.max(m, 1e-6);
}
const gelu = (x) => {
	const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
	let e = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z);
	if (x < 0) e = -e;
	return 0.5 * x * (1 + e);
};

{ // rope_pairs
	const T = 37, H = 3, D = 16, x = rand(T * H * D, 1), cos = new Float32Array(T * D / 2), sin = new Float32Array(T * D / 2);
	for (let i = 0; i < cos.length; i++) { cos[i] = Math.cos(i * 0.37); sin[i] = Math.sin(i * 0.37); }
	const want = x.slice();
	for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) for (let p = 0; p < D / 2; p++) {
		const at = (t * H + h) * D + 2 * p, c = cos[t * D / 2 + p], s = sin[t * D / 2 + p], a = x[at], b = x[at + 1];
		want[at] = a * c - b * s; want[at + 1] = a * s + b * c;
	}
	const g = up(x), gc = up(cos), gs = up(sin);
	op.ropePairs(g, gc, gs, D, H, T);
	check(worst(read(g, x.length), want) < 1e-5, 'rope_pairs');
}
{ // head_norm_gamma
	const R = 21, H = 4, D = 64, x = rand(R * H * D, 2), w = rand(H * D, 3), want = new Float32Array(x.length);
	for (let r = 0; r < R; r++) for (let h = 0; h < H; h++) {
		let s = 0;
		for (let d = 0; d < D; d++) s += x[(r * H + h) * D + d] ** 2;
		const inv = 1 / Math.sqrt(s / D);
		for (let d = 0; d < D; d++) want[(r * H + h) * D + d] = x[(r * H + h) * D + d] * inv * w[h * D + d];
	}
	const g = up(x);
	op.headNormGamma(g, up(w), H, D, R);
	check(worst(read(g, x.length), want) < 1e-5, 'head_norm_gamma');
}
{ // group_norm_rows (+ silu)
	const N = 1000, Cn = 32, G = 8, x = rand(N * Cn, 4, 3), w = rand(Cn, 5), b = rand(Cn, 6), want = new Float32Array(x.length);
	const cpg = Cn / G;
	for (let g = 0; g < G; g++) {
		let s = 0, s2 = 0;
		for (let r = 0; r < N; r++) for (let c = g * cpg; c < (g + 1) * cpg; c++) s += x[r * Cn + c];
		const mean = s / (N * cpg);
		for (let r = 0; r < N; r++) for (let c = g * cpg; c < (g + 1) * cpg; c++) s2 += (x[r * Cn + c] - mean) ** 2;
		const inv = 1 / Math.sqrt(s2 / (N * cpg) + 1e-5);
		for (let r = 0; r < N; r++) for (let c = g * cpg; c < (g + 1) * cpg; c++) {
			const y = (x[r * Cn + c] - mean) * inv * w[c] + b[c];
			want[r * Cn + c] = y / (1 + Math.exp(-y));
		}
	}
	const out = f32(x.length);
	op.groupNormRows(up(x), up(w), up(b), out, Cn, N, G, 1e-5, true);
	check(worst(read(out, x.length), want) < 1e-5, 'group_norm_rows');
}
{ // im2col3d both layouts, stride 1 and 2, against a direct conv through a matmul
	for (const [stride, cMajor] of [[1, false], [2, true]]) {
		const X = 5, Y = 6, Z = 7, Cn = 3, K = 3, O = 4;
		const ox = Math.floor((X + 2 - K) / stride) + 1, oy = Math.floor((Y + 2 - K) / stride) + 1, oz = Math.floor((Z + 2 - K) / stride) + 1;
		const x = rand(X * Y * Z * Cn, 7), w = rand(O * K * K * K * Cn, 8); // torch [O, C, K, K, K] when cMajor, else [O, K, K, K, C]
		const want = new Float32Array(ox * oy * oz * O);
		for (let a = 0; a < ox; a++) for (let b = 0; b < oy; b++) for (let c = 0; c < oz; c++) for (let o = 0; o < O; o++) {
			let s = 0;
			for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) for (let l = 0; l < K; l++) {
				const ix = a * stride + i - 1, iy = b * stride + j - 1, iz = c * stride + l - 1;
				if (ix < 0 || iy < 0 || iz < 0 || ix >= X || iy >= Y || iz >= Z) continue;
				for (let ch = 0; ch < Cn; ch++) {
					const k = (i * K + j) * K + l;
					const wi = cMajor ? ((o * Cn + ch) * 27 + k) : ((o * 27 + k) * Cn + ch);
					s += x[((ix * Y + iy) * Z + iz) * Cn + ch] * w[wi];
				}
			}
			want[((a * oy + b) * oz + c) * O + o] = s;
		}
		const rows = ox * oy * oz, cols = f32(rows * 27 * Cn), y = f32(rows * O);
		op.im2col3d(up(x), cols, Cn, [X, Y, Z], K, stride, 1, [ox, oy, oz], 0, rows, cMajor);
		op.matmul(up(w), cols, y, O, 27 * Cn, rows);
		check(worst(read(y, want.length), want) < 1e-5, `im2col3d stride ${stride} ${cMajor ? 'c-major' : 'k-major'}`);
	}
}
{ // gather_k
	const N = 50, Cn = 5, K = 3, R = 40, x = rand(N * Cn, 9), idx = [];
	for (let i = 0; i < R * K; i++) idx.push(i % 7 === 0 ? 0xFFFFFFFF : (i * 13) % N);
	const want = new Float32Array(R * K * Cn);
	for (let r = 0; r < R; r++) for (let k = 0; k < K; k++) for (let c = 0; c < Cn; c++) {
		const s = idx[r * K + k];
		want[(r * K + k) * Cn + c] = s === 0xFFFFFFFF ? 0 : x[s * Cn + c];
	}
	const out = f32(want.length);
	op.gatherK(up(x), u32(idx), out, Cn, K, R);
	check(worst(read(out, want.length), want) === 0, 'gather_k');
}
{ // pixel_shuffle3d
	const X = 2, Y = 3, Z = 2, Cn = 3, x = rand(X * Y * Z * Cn * 8, 10), want = new Float32Array(x.length);
	for (let a = 0; a < 2 * X; a++) for (let b = 0; b < 2 * Y; b++) for (let c = 0; c < 2 * Z; c++) for (let ch = 0; ch < Cn; ch++) {
		const src = ((a >> 1) * Y + (b >> 1)) * Z + (c >> 1), sub = (a & 1) * 4 + (b & 1) * 2 + (c & 1);
		want[((a * 2 * Y + b) * 2 * Z + c) * Cn + ch] = x[src * Cn * 8 + ch * 8 + sub];
	}
	const out = f32(x.length);
	op.pixelShuffle3d(up(x), out, Cn, X, Y, Z);
	check(worst(read(out, x.length), want) === 0, 'pixel_shuffle3d');
}
{ // attention_window: three groups of uneven size, rows interleaved
	const n = 90, H = 2, D = 64, q = rand(n * H * D, 11), k = rand(n * H * D, 12), v = rand(n * H * D, 13);
	const group = (t) => (t * 7) % 3;
	const order = [], span = new Array(2 * n);
	for (let g = 0; g < 3; g++) {
		const start = order.length;
		for (let t = 0; t < n; t++) if (group(t) === g) order.push(t);
		for (let t = 0; t < n; t++) if (group(t) === g) { span[2 * t] = start; span[2 * t + 1] = order.length; }
	}
	const want = new Float32Array(n * H * D);
	for (let t = 0; t < n; t++) for (let h = 0; h < H; h++) {
		const keys = order.slice(span[2 * t], span[2 * t + 1]);
		const s = keys.map((p) => { let d = 0; for (let i = 0; i < D; i++) d += q[(t * H + h) * D + i] * k[(p * H + h) * D + i]; return d / 8; });
		const m = Math.max(...s), e = s.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
		for (let i = 0; i < D; i++) want[(t * H + h) * D + i] = keys.reduce((a, p, j) => a + e[j] * v[(p * H + h) * D + i], 0) / z;
	}
	const out = f32(want.length);
	op.attentionWindow(up(q), up(k), up(v), u32(order), u32(span), out, H, D, n);
	check(worst(read(out, want.length), want) < 1e-5, 'attention_window');
}
{ // attentionPadded: cross-attention at head width 96 (float16 operands)
	const H = 2, D = 96, L = 70, S = 130, q = rand(L * H * D, 14), k = rand(S * H * D, 15), v = rand(S * H * D, 16);
	const want = new Float32Array(L * H * D);
	for (let t = 0; t < L; t++) for (let h = 0; h < H; h++) {
		const s = [];
		for (let p = 0; p < S; p++) { let d = 0; for (let i = 0; i < D; i++) d += q[(t * H + h) * D + i] * k[(p * H + h) * D + i]; s.push(d / Math.sqrt(D)); }
		const m = Math.max(...s), e = s.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
		for (let i = 0; i < D; i++) { let a = 0; for (let p = 0; p < S; p++) a += e[p] * v[(p * H + h) * D + i]; want[(t * H + h) * D + i] = a / z; }
	}
	const pad = { q: f32(H * S * 128), k: f32(H * S * 128), v: f32(H * S * 128), o: f32(H * S * 128) };
	const out = f32(want.length);
	op.attentionPadded(up(q), up(k), up(v), out, pad, H, D, L, S);
	const err = worst(read(out, want.length), want);
	check(err < 5e-3, `attentionPadded cross, head 96 (${err.toExponential(1)})`);
}
{ // pair_edges
	const n = 45, Hd = 32, a = rand(n * Hd, 17), b = rand(n * Hd, 18), w2 = rand(Hd, 19), b2 = 0.1;
	const want = [];
	for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
		let s = 0;
		for (let h = 0; h < Hd; h++) s += w2[h] * (gelu(a[i * Hd + h] + b[j * Hd + h]) + gelu(a[j * Hd + h] + b[i * Hd + h]));
		if (0.5 * s + b2 > 0) want.push(i * n + j);
	}
	const count = C.u32(1), cap = n * n, edges = C.u32(2 * cap);
	op.pairEdges(up(a), up(b), up(w2), count, edges, n, Hd, b2, 0, cap);
	submit();
	const got = count.read().bytes[0];
	const e = new Uint32Array(edges.readBytes(0, got * 8).buffer);
	const pairs = [];
	for (let i = 0; i < got; i++) pairs.push(e[2 * i] * n + e[2 * i + 1]);
	pairs.sort((x, y) => x - y);
	check(got === want.length && pairs.every((p, i) => p === want[i]), `pair_edges (${got} of ${n * (n - 1) / 2} pairs, want ${want.length})`);
}

llm.print(failures ? `${failures} failed` : 'all passed');
