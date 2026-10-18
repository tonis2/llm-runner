// The 8x8 Q8_0 matmul (matmul_q8_8x8) against the 4x4 one (matmul_q8) and the
// host, on DINOv3's shapes and on edge shapes, and the two kernels' times.
//
//   llm-runner tests/matmul_q8_8x8.js
import { llm, f32 } from '../lib/llm.js';
import { dispatch, pc, groups, submit } from '../lib/gpu.js';

let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
let seed = 11;
const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };

// A float as IEEE half bits (round to nearest; normal range only, as scales are).
function halfBits(v) {
	const f = new Float32Array([v]), u = new Uint32Array(f.buffer)[0];
	const sign = (u >>> 16) & 0x8000, exp = ((u >>> 23) & 0xff) - 127 + 15;
	if (v === 0) return 0;
	let mant = (u & 0x7fffff) + 0x1000;
	let e = exp;
	if (mant & 0x800000) { mant = 0; e++; }
	return sign | (e << 10) | (mant >>> 13);
}
function halfValue(bits) {
	const e = (bits >>> 10) & 0x1f, m = bits & 0x3ff, s = bits & 0x8000 ? -1 : 1;
	return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

// W [out, in] as Q8_0 bytes, and the values those bytes stand for.
function quantize(out, inDim) {
	const blocks = inDim / 32, bytes = new Uint8Array(out * blocks * 34), values = new Float32Array(out * inDim);
	for (let r = 0; r < out; r++) {
		for (let b = 0; b < blocks; b++) {
			const vals = Array.from({ length: 32 }, () => random() * 0.2);
			const amax = Math.max(...vals.map(Math.abs)) || 1;
			const hb = halfBits(amax / 127), d = halfValue(hb);
			const at = (r * blocks + b) * 34;
			bytes[at] = hb & 0xff;
			bytes[at + 1] = hb >> 8;
			for (let i = 0; i < 32; i++) {
				const qv = Math.max(-127, Math.min(127, Math.round(vals[i] / d)));
				bytes[at + 2 + i] = qv & 0xff;
				values[r * inDim + b * 32 + i] = qv * d;
			}
		}
	}
	return { bytes, values };
}

function upload(data) {
	const t = f32(Math.ceil(data.byteLength / 4));
	const padded = new Uint8Array(t.elements * 4);
	padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
	t.buffer.write(new Float32Array(padded.buffer));
	return t;
}
const read = (t, n) => new Float32Array(t.buffer.readBytes(0, n * 4).buffer);

function run(name, W, x, y, out, inDim, seq, rowOffset = 0) {
	const per = name === 'matmul_q8_8x8' ? 128 : 64;
	dispatch(name, [W, x, y], [groups(seq, per) * groups(out, per)], pc('uuuu', out, inDim, seq, rowOffset));
}

for (const [seq, inDim, out, rowOffset, timed] of [
	[37, 64, 70, 0, false], [145, 1024, 1404, 0, false], [130, 96, 200, 50, false],
	[1029, 1280, 5120, 0, true], [1029, 5120, 1280, 0, true], [1029, 1280, 1280, 1280, true],
]) {
	const rows = out + rowOffset;
	const { bytes, values } = quantize(rows, inDim);
	const X = Float32Array.from({ length: seq * inDim }, () => random() * 4);
	const W = upload(bytes), x = upload(X), a = f32(seq * out), b = f32(seq * out);
	run('matmul_q8', W, x, a, out, inDim, seq, rowOffset);
	run('matmul_q8_8x8', W, x, b, out, inDim, seq, rowOffset);
	submit();
	const A = read(a, seq * out), B = read(b, seq * out);
	let worst = 0, scale = 0;
	for (let i = 0; i < A.length; i++) { worst = Math.max(worst, Math.abs(A[i] - B[i])); scale = Math.max(scale, Math.abs(A[i])); }
	let worstRef = 0;
	for (const r of [0, seq - 1, seq >> 1]) {
		for (const c of [0, out - 1, out >> 1, (out * 7) >> 3]) {
			let s = 0;
			for (let k = 0; k < inDim; k++) s += X[r * inDim + k] * values[(rowOffset + c) * inDim + k];
			worstRef = Math.max(worstRef, Math.abs(s - B[r * out + c]));
		}
	}
	const label = `${seq}x${inDim}->${out}${rowOffset ? ` (rows from ${rowOffset})` : ''}`;
	check(worst <= 1e-5 * scale, `${label}: matches matmul_q8 (worst ${worst.toExponential(2)} of ${scale.toFixed(1)})`);
	check(worstRef <= 1e-5 * scale, `${label}: matches the host (worst ${worstRef.toExponential(2)})`);
	if (timed) {
		for (const name of ['matmul_q8', 'matmul_q8_8x8']) {
			run(name, W, x, b, out, inDim, seq, rowOffset);
			submit();
			const t0 = llm.now();
			for (let i = 0; i < 10; i++) run(name, W, x, b, out, inDim, seq, rowOffset);
			submit();
			const ms = (llm.now() - t0) / 10;
			llm.print(`     ${name}: ${ms.toFixed(2)} ms, ${((2 * seq * inDim * out) / ms / 1e9).toFixed(2)} TFLOPS`);
		}
	}
	for (const t of [W, x, a, b]) t.dispose();
}
llm.print(failures ? `${failures} FAILED` : 'all passed');
