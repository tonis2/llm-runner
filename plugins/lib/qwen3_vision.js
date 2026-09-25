// Qwen3-VL's vision encoder, from an mmproj GGUF: an image in, the rows the
// language model reads in its image-pad tokens' places out, and the three
// "deepstack" sets it adds to its first three layers' output there.
//
// 16-pixel patches (the still image counts for both of the patch embedding's
// two frames), a learned position table bilinearly resized to the patch grid,
// then 27 pre-norm blocks with 2D rotary attention over all the patches. Every
// 2x2 of patches becomes one token: through the main merger at the end, and
// through a deepstack merger after blocks 8, 16 and 24.
//
// Ported from transformers' `modeling_qwen3_vl.py` (Qwen3VLVisionModel).

import { llm, f32 } from './llm.js';
import * as op from './ops.js';
import { submit, breathe, uploadEach } from './gpu.js';

const PATCH = 16;
const HEAD_PAD = 80;   // flash attention's head width: 72 padded to a multiple of 16

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

function disposeAll(obj) {
	for (const v of Object.values(obj)) {
		if (!v) continue;
		if (typeof v.dispose === 'function') v.dispose();
		else if (typeof v === 'object') disposeAll(v);
	}
}

export class Qwen3Vision {
	constructor(path) {
		const m = llm.open(path);
		this.model = m;
		if (!m.has('v.patch_embd.weight')) {
			m.close();
			throw new Error(`${path} is not a vision encoder (mmproj) file`);
		}
		this.dim = m.meta('clip.vision.embedding_length', 1152);
		this.heads = m.meta('clip.vision.attention.head_count', 16);
		this.headDim = this.dim / this.heads;
		this.ffn = m.meta('clip.vision.feed_forward_length', 4304);
		this.eps = m.meta('clip.vision.attention.layer_norm_epsilon', 1e-6);
		this.outDim = m.meta('clip.vision.projection_dim', 4096);
		let n = 0;
		while (m.has(`v.blk.${n}.attn_qkv.weight`)) n++;
		this.nLayers = n;
		this.deepstackAt = [];
		for (let l = 0; l < n; l++) if (m.has(`v.deepstack.${l}.fc1.weight`)) this.deepstackAt.push(l);
		this.grid = Math.round(Math.sqrt(m.shape('v.position_embd.weight')[1]));
	}

	async load() {
		const m = this.model, D = this.dim;
		const t0 = llm.now();
		const f = (name) => m.upload(name, 'f32');
		// The two frames of the Conv3d see the same image: their kernels add.
		// Reordered from (c, y, x) to patchify's (y, x, c).
		const w0 = m.floats('v.patch_embd.weight'), w1 = m.floats('v.patch_embd.weight.1');
		const pd = 3 * PATCH * PATCH;
		const patchW = new Float32Array(D * pd);
		for (let o = 0; o < D; o++) {
			for (let c = 0; c < 3; c++) {
				for (let p = 0; p < PATCH * PATCH; p++) {
					const src = (o * 3 + c) * PATCH * PATCH + p;
					patchW[o * pd + p * 3 + c] = w0[src] + w1[src];
				}
			}
		}
		this.g = {
			patchW: upload(patchW),
			patchB: f('v.patch_embd.bias'),
			pos: f('v.position_embd.weight'),
			postW: f('v.post_ln.weight'), postB: f('v.post_ln.bias'),
			fc1: f('mm.0.weight'), fc1b: f('mm.0.bias'),
			fc2: f('mm.2.weight'), fc2b: f('mm.2.bias'),
		};
		this.blocks = [];
		for (let l = 0; l < this.nLayers; l++) {
			const b = `v.blk.${l}.`;
			const bias = m.floats(`${b}attn_qkv.bias`);
			this.blocks.push(await uploadEach({
				ln1w: () => f(`${b}ln1.weight`), ln1b: () => f(`${b}ln1.bias`),
				qkv: () => f(`${b}attn_qkv.weight`),
				qb: () => upload(bias.subarray(0, D)),
				kb: () => upload(bias.subarray(D, 2 * D)),
				vb: () => upload(bias.subarray(2 * D, 3 * D)),
				out: () => f(`${b}attn_out.weight`), outb: () => f(`${b}attn_out.bias`),
				ln2w: () => f(`${b}ln2.weight`), ln2b: () => f(`${b}ln2.bias`),
				up: () => f(`${b}ffn_up.weight`), upb: () => f(`${b}ffn_up.bias`),
				down: () => f(`${b}ffn_down.weight`), downb: () => f(`${b}ffn_down.bias`),
			}));
		}
		this.deepstack = [];
		for (const l of this.deepstackAt) {
			const p = `v.deepstack.${l}.`;
			this.deepstack.push(await uploadEach({
				normW: () => f(`${p}norm.weight`), normB: () => f(`${p}norm.bias`),
				fc1: () => f(`${p}fc1.weight`), fc1b: () => f(`${p}fc1.bias`),
				fc2: () => f(`${p}fc2.weight`), fc2b: () => f(`${p}fc2.bias`),
			}));
		}
		llm.print(`  vision: ${this.nLayers} blocks, dim ${this.dim}, deepstack after ${this.deepstackAt.join('/')}, loaded in ${llm.since(t0)}`);
	}

	close() {
		if (this.g) disposeAll(this.g);
		for (const b of this.blocks ?? []) disposeAll(b);
		for (const d of this.deepstack ?? []) disposeAll(d);
		this.model.close();
	}

	// cos/sin [n, headDim / 2] for the 2D rotary: the first half of the pairs turn
	// with the patch's row, the second with its column.
	ropeTables(gh, gw) {
		const half = this.headDim / 2, quarter = half / 2;
		const inv = new Float32Array(quarter);
		for (let k = 0; k < quarter; k++) inv[k] = Math.fround(1 / Math.fround(Math.pow(10000, Math.fround((2 * k) / half))));
		const n = gh * gw;
		const cos = new Float32Array(n * half), sin = new Float32Array(n * half);
		for (let t = 0; t < n; t++) {
			const row = Math.floor(t / gw), col = t % gw;
			for (let j = 0; j < half; j++) {
				const angle = Math.fround((j < quarter ? row : col) * inv[j % quarter]);
				cos[t * half + j] = Math.cos(angle);
				sin[t * half + j] = Math.sin(angle);
			}
		}
		return { cos: upload(cos), sin: upload(sin) };
	}

	// `pixels` is [3, H, W] in [-1, 1], the sides multiples of 32. Returns
	// { main, deepstack: [...], tokens, gh, gw }: [tokens, outDim] tensors, a
	// token per 2x2 of patches in row order over the (H/32, W/32) grid.
	async encode(pixels, H, W) {
		const D = this.dim, F = this.ffn, g = this.g, eps = this.eps;
		const gh = H / PATCH, gw = W / PATCH, N = gh * gw, M = N / 4;
		const t0 = llm.now();
		const img = upload(pixels);
		const patches = f32(N * 3 * PATCH * PATCH);
		const a = {
			h: f32(N * D), x: f32(N * D), q: f32(N * D), k: f32(N * D), v: f32(N * D),
			qp: f32(N * this.heads * HEAD_PAD), kp: f32(N * this.heads * HEAD_PAD), vp: f32(N * this.heads * HEAD_PAD),
			o: f32(N * this.heads * HEAD_PAD), ffn: f32(N * F),
			merged: f32(M * 4 * D), m1: f32(M * 4 * D),
			posT: f32(this.grid * this.grid * D),
		};
		const rope = this.ropeTables(gh, gw);
		try {
			op.patchify(img, patches, 3, H, W, PATCH);
			op.matmul(g.patchW, patches, a.h, D, 3 * PATCH * PATCH, N);
			op.biasAdd(a.h, g.patchB, N * D, D);
			// The position table, [grid^2, D], resized over the patch grid with
			// aligned corners and added.
			op.transposeChannelSpatial(g.pos, a.posT, D, this.grid * this.grid, 1);
			op.bilinear(a.posT, a.q, D, this.grid, this.grid, gh, gw);
			op.transposeChannelSpatial(a.q, a.x, D, N, 0);
			op.add(a.h, a.x, N * D);
			submit();

			const deepstack = [];
			for (let l = 0; l < this.nLayers; l++) {
				this.block(this.blocks[l], a, N, rope);
				const at = this.deepstackAt.indexOf(l);
				if (at >= 0) {
					const d = this.deepstack[at];
					const out = f32(M * this.outDim);
					op.merge2x2(a.h, a.merged, gh, gw, D);
					op.layerNormAffine(a.merged, d.normW, d.normB, a.m1, 4 * D, M, 1e-6);
					this.mlp(d.fc1, d.fc1b, d.fc2, d.fc2b, a.m1, a.merged, out, M);
					deepstack.push(out);
				}
				await breathe(true);
			}
			const main = f32(M * this.outDim);
			op.layerNormAffine(a.h, g.postW, g.postB, a.x, D, N, 1e-6);
			op.merge2x2(a.x, a.merged, gh, gw, D);
			this.mlp(g.fc1, g.fc1b, g.fc2, g.fc2b, a.merged, a.m1, main, M);
			submit();
			llm.print(`  vision: ${W}x${H} to ${M} tokens in ${llm.since(t0)}`);
			return { main, deepstack, tokens: M, gh: gh / 2, gw: gw / 2 };
		} finally {
			submit();
			disposeAll(a);
			disposeAll(rope);
			img.dispose();
			patches.dispose();
		}
	}

	// The mergers' MLP: fc1, exact GELU, fc2, over `rows` rows of 4 * dim.
	mlp(fc1, fc1b, fc2, fc2b, x, tmp, out, rows) {
		const W = 4 * this.dim;
		op.matmul(fc1, x, tmp, W, W, rows);
		op.biasAdd(tmp, fc1b, rows * W, W);
		op.geluErf(tmp, rows * W);
		op.matmul(fc2, tmp, out, this.outDim, W, rows);
		op.biasAdd(out, fc2b, rows * this.outDim, this.outDim);
	}

	// One pre-norm block over a.h [N, dim], in place.
	block(b, a, N, rope) {
		const D = this.dim, H = this.heads, hd = this.headDim, F = this.ffn, n = N * D;
		op.layerNormAffine(a.h, b.ln1w, b.ln1b, a.x, D, N, this.eps);
		op.matmul(b.qkv, a.x, a.q, D, D, N, 0, true);
		op.matmul(b.qkv, a.x, a.k, D, D, N, D, true);
		op.matmul(b.qkv, a.x, a.v, D, D, N, 2 * D);
		op.biasAdd(a.q, b.qb, n, D);
		op.biasAdd(a.k, b.kb, n, D);
		op.biasAdd(a.v, b.vb, n, D);
		op.ropeNeox(a.q, rope.cos, rope.sin, hd, H, N, true);
		op.ropeNeox(a.k, rope.cos, rope.sin, hd, H, N);
		op.padHeads(a.q, a.qp, N, H, hd, HEAD_PAD);
		op.padHeads(a.k, a.kp, N, H, hd, HEAD_PAD);
		op.padHeads(a.v, a.vp, N, H, hd, HEAD_PAD);
		op.flashAttentionScalar(a.qp, a.kp, a.vp, a.o, H, N, HEAD_PAD, 1 / Math.sqrt(hd));
		// [N, heads * 80] back to [N, heads * 72]: each (token, head) a row.
		op.copyRows(a.o, a.x, N * H, hd, hd, 0, HEAD_PAD);
		op.matmul(b.out, a.x, a.q, D, D, N);
		op.biasAdd(a.q, b.outb, n, D);
		op.add(a.h, a.q, n);

		op.layerNormAffine(a.h, b.ln2w, b.ln2b, a.x, D, N, this.eps);
		op.matmul(b.up, a.x, a.ffn, F, D, N);
		op.biasAdd(a.ffn, b.upb, N * F, F);
		op.gelu(a.ffn, N * F);
		op.matmul(b.down, a.ffn, a.q, D, F, N);
		op.biasAdd(a.q, b.downb, n, D);
		op.add(a.h, a.q, n);
	}
}
