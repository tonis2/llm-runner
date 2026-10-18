// The head-64 register attention (flash_attention_64) against the general
// kernel and a float64 reference on the host, at DINOv3's 1029 tokens and an
// odd small length, and its time against the general kernel's.
//
//   llm-runner tests/attention_64.js
import { llm, f32 } from '../lib/llm.js';
import { dispatch, pc, groups, flashDefines, submit } from '../lib/gpu.js';

let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
let seed = 3;
const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}
const read = (t, n) => new Float32Array(t.buffer.readBytes(0, n * 4).buffer);

const HD = 64;
function run(name, q, k, v, out, heads, seq) {
	if (name === 'flash_attention_64') dispatch(name, [q, k, v, out], [groups(seq, 128), heads], pc('uuf', heads, seq, 1 / Math.sqrt(HD)));
	else dispatch(name, [q, k, v, out], [groups(seq, 16), heads], pc('uuuf', HD, heads, seq, 1 / Math.sqrt(HD)), false, flashDefines(HD));
}

// Host attention for a few query rows of one head.
function reference(Q, K, V, heads, seq, head, row) {
	const at = (X, r, d) => X[(head * seq + r) * HD + d];
	const s = new Float64Array(seq);
	let m = -Infinity;
	for (let j = 0; j < seq; j++) {
		let dot = 0;
		for (let d = 0; d < HD; d++) dot += at(Q, row, d) * at(K, j, d);
		s[j] = dot / Math.sqrt(HD);
		m = Math.max(m, s[j]);
	}
	let l = 0;
	for (let j = 0; j < seq; j++) { s[j] = Math.exp(s[j] - m); l += s[j]; }
	return Array.from({ length: HD }, (_, d) => { let o = 0; for (let j = 0; j < seq; j++) o += s[j] * at(V, j, d); return o / l; });
}

for (const [heads, seq] of [[20, 1029], [3, 37]]) {
	const n = heads * seq * HD;
	// Scores with some spread, as trained Q and K give.
	const Q = Float32Array.from({ length: n }, () => random() * 4);
	const K = Float32Array.from({ length: n }, () => random() * 4);
	const V = Float32Array.from({ length: n }, () => random() * 2);
	const q = upload(Q), k = upload(K), v = upload(V), a = f32(n), b = f32(n);
	run('flash_attention', q, k, v, a, heads, seq);
	run('flash_attention_64', q, k, v, b, heads, seq);
	submit();
	const A = read(a, n), B = read(b, n);
	let worst = 0, worstRef = 0;
	for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(A[i] - B[i]));
	for (const [head, row] of [[0, 0], [heads - 1, seq - 1], [heads >> 1, seq >> 1]]) {
		const want = reference(Q, K, V, heads, seq, head, row);
		want.forEach((w, d) => { worstRef = Math.max(worstRef, Math.abs(w - B[row * heads * HD + head * HD + d])); });
	}
	check(worst < 1e-4, `${heads} heads x ${seq}: matches the general kernel (worst ${worst.toExponential(2)})`);
	check(worstRef < 1e-4, `${heads} heads x ${seq}: matches float64 (worst ${worstRef.toExponential(2)})`);
	if (seq > 1000) {
		for (const name of ['flash_attention', 'flash_attention_64']) {
			run(name, q, k, v, b, heads, seq);
			submit();
			const t0 = llm.now();
			for (let i = 0; i < 10; i++) run(name, q, k, v, b, heads, seq);
			submit();
			const ms = (llm.now() - t0) / 10;
			const flops = 4 * heads * seq * seq * HD;
			llm.print(`     ${name}: ${ms.toFixed(2)} ms, ${(flops / ms / 1e9).toFixed(2)} TFLOPS`);
		}
	}
	for (const t of [q, k, v, a, b]) t.dispose();
}
llm.print(failures ? `${failures} FAILED` : 'all passed');
