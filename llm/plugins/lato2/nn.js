// The layers LATO.2's models (and the TRELLIS stage it borrows) are built
// from, over rows: a token, a voxel or a sparse point is a row [n, channels].
//
// Weights come from an open llm.js Model (a .pt through openTorch, or a
// safetensors); `as` is how a matmul weight goes to the GPU ('f32' or 'q8').

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';

export function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

export function uploadU32(values) {
	return three.compute.u32(values.length, values instanceof Uint32Array ? values : Uint32Array.from(values));
}

export function disposeAll(v) {
	if (!v) return;
	if (typeof v.dispose === 'function') { v.dispose(); return; }
	if (Array.isArray(v)) { for (const x of v) disposeAll(x); return; }
	if (typeof v === 'object') for (const x of Object.values(v)) disposeAll(x);
}

export function readFloats(t, count, first = 0) {
	const buffer = t.buffer ?? t;
	return new Float32Array(buffer.readBytes(first * 4, count * 4).buffer);
}

// Rows of a [n, width] tensor as a view: rows first .. first + count.
export function rows(t, width, first, count) {
	return t.view(first * width * 4, count * width * 4);
}

// A torch nn.Linear: y[rows, out] = x[rows, in] W^T + b.
export class Linear {
	constructor(m, name, as = 'f32', { bias = true } = {}) {
		const shape = m.shape(`${name}.weight`);
		this.in = shape[0];
		this.out = shape[1];
		this.w = m.upload(`${name}.weight`, as);
		this.b = bias && m.has(`${name}.bias`) ? m.upload(`${name}.bias`, 'f32') : null;
	}
	run(x, y, n, independent = false) {
		op.matmul(this.w, x, y, this.out, this.in, n, 0, independent && !this.b);
		if (this.b) op.biasAdd(y, this.b, n * this.out, this.out);
	}
	// Output columns first .. first + count only (a third of a fused QKV).
	part(x, y, n, first, count, bias = null) {
		op.matmul(this.w, x, y, count, this.in, n, first);
		if (this.b) op.biasAdd(y, this.b.view(first * 4, count * 4), n * count, count);
	}
	dispose() { this.w.dispose(); this.b?.dispose(); }
}

// TRELLIS's AbsolutePositionEmbedder: per coordinate, sin then cos over
// channels / dims / 2 frequencies, the axes side by side, zero-padded to
// `channels`. `coords` is [n, dims] integers (an Int32Array or a list).
export function sincosPositions(coords, n, channels, dims = 3) {
	const fd = Math.floor(channels / dims / 2);
	const freqs = new Float32Array(fd);
	for (let i = 0; i < fd; i++) freqs[i] = 1 / Math.pow(10000, Math.fround(i / fd));
	const out = new Float32Array(n * channels);
	for (let t = 0; t < n; t++) {
		for (let d = 0; d < dims; d++) {
			const v = coords[t * dims + d], at = t * channels + d * 2 * fd;
			for (let i = 0; i < fd; i++) {
				const a = Math.fround(v * freqs[i]);
				out[at + i] = Math.sin(a);
				out[at + fd + i] = Math.cos(a);
			}
		}
	}
	return out;
}

// cos/sin tables [n, headDim / 2] for 3D RoPE on adjacent pairs (op.ropePairs):
// pair p turns by coords[axis] * freq[f] with axis = p / fd, f = p % fd, fd =
// headDim / 2 / 3, freq = 1 / 10000^(f / fd); pairs past 3 * fd are left as
// they are. TRELLIS's sparse RotaryPositionEmbedder and LATO.2's
// RotaryPositionPhasesEmbedder (rope_freq (1, 10000)) are both this.
export function ropeTables(coords, n, headDim) {
	const half = headDim / 2, fd = Math.floor(half / 3);
	const freqs = new Float32Array(fd);
	for (let i = 0; i < fd; i++) freqs[i] = 1 / Math.pow(10000, Math.fround(i / fd));
	const cos = new Float32Array(n * half), sin = new Float32Array(n * half);
	for (let t = 0; t < n; t++) {
		for (let p = 0; p < half; p++) {
			if (p >= 3 * fd) { cos[t * half + p] = 1; continue; }
			const a = Math.fround(coords[t * 3 + Math.floor(p / fd)] * freqs[p % fd]);
			cos[t * half + p] = Math.cos(a);
			sin[t * half + p] = Math.sin(a);
		}
	}
	return { cos: upload(cos), sin: upload(sin), dispose() { this.cos.dispose(); this.sin.dispose(); } };
}

// TimestepEmbedder: the 256-wide [cos, sin] embedding of t, then
// Linear, SiLU, Linear. `out` is a [1, dim] tensor.
export class TimestepEmbedder {
	constructor(m, name, as = 'f32') {
		this.l0 = new Linear(m, `${name}.mlp.0`, as);
		this.l2 = new Linear(m, `${name}.mlp.2`, as);
		this.freq = f32(256);
		this.mid = f32(this.l0.out);
	}
	run(t, out) {
		op.timestepEmbed(this.freq, 256, t);
		this.l0.run(this.freq, this.mid, 1);
		op.silu(this.mid, this.l0.out);
		this.l2.run(this.mid, out, 1);
	}
	dispose() { disposeAll([this.l0, this.l2, this.freq, this.mid]); }
}

// A dense 3D convolution over a channel-last grid [X, Y, Z, inC] to [OX, OY,
// OZ, outC], as patch rows times the weight (torch's [out, in, k, k, k], read
// flattened). The patches are made a slab of rows at a time in `scratch`.
export class Conv3d {
	// `m` opened with openTorch / openSafetensors, whose `dims` is torch's
	// [out, in, k, k, k] (the table has it as [out, in * k^3]).
	constructor(m, name, as = 'f32') {
		const [outC, inC, k] = m.dims(`${name}.weight`);
		this.k = k;
		this.inC = inC;
		this.outC = outC;
		this.w = m.upload(`${name}.weight`, as);
		this.b = m.has(`${name}.bias`) ? m.upload(`${name}.bias`, 'f32') : null;
	}
	// `grid` [X, Y, Z]; returns the output grid.
	run(x, y, grid, scratch, stride = 1) {
		const k = this.k, pad = k === 3 ? 1 : 0, kk = k ** 3;
		const og = grid.map((s) => Math.floor((s + 2 * pad - k) / stride) + 1);
		const total = og[0] * og[1] * og[2], width = kk * this.inC;
		const slab = Math.max(1, Math.min(total, Math.floor(scratch.elements / width)));
		for (let first = 0; first < total; first += slab) {
			const count = Math.min(slab, total - first);
			const out = rows(y, this.outC, first, count);
			if (k === 1 && stride === 1) {
				op.matmul(this.w, rows(x, this.inC, first, count), out, this.outC, this.inC, count);
			} else {
				op.im2col3d(x, scratch, this.inC, grid, k, stride, pad, og, first, count, true);
				op.matmul(this.w, scratch, out, this.outC, width, count);
			}
			if (this.b) op.biasAdd(out, this.b, count * this.outC, this.outC);
		}
		return og;
	}
	dispose() { this.w.dispose(); this.b?.dispose(); }
}

// Self- or cross-attention over rows, any head width up to 128, on the matrix
// cores (op.attentionPadded); the padding buffers grow to the largest call.
export class Attention {
	constructor() { this.pad = null; this.size = 0; }
	run(q, k, v, out, heads, hd, qLen, kLen) {
		const need = heads * Math.max(qLen, kLen) * 128;
		if (need > this.size) {
			if (this.pad) { three.compute.submit(); disposeAll(this.pad); }
			this.pad = { q: f32(need), k: f32(need), v: f32(need), o: f32(need) };
			this.size = need;
		}
		op.attentionPadded(q, k, v, out, this.pad, heads, hd, qLen, kLen);
	}
	dispose() { disposeAll(this.pad); this.pad = null; this.size = 0; }
}

// Buffers for activations, reused across calls and grown as needed: `get(name,
// count)` is a tensor of at least `count` floats.
export class Scratch {
	constructor() { this.t = new Map(); }
	get(name, count) {
		let t = this.t.get(name);
		if (!t || t.elements < count) {
			if (t) { three.compute.submit(); t.dispose(); }
			t = f32(Math.max(count, 1));
			this.t.set(name, t);
		}
		return t;
	}
	dispose() { for (const t of this.t.values()) t.dispose(); this.t.clear(); }
}

export function seconds(t0) { return llm.since(t0); }
