// DINOv3's ViT (facebookresearch/dinov3), as SAM 3D Body uses it: an image in,
// the last block's normed patch tokens out as a [dim, gh, gw] feature map (or
// as the tokens [gh * gw, dim], for matching patches against each other).
//
// 16-pixel patches, then a CLS token and four storage ("register") tokens in
// front of them, and no learned position table: every block rotates the patch
// tokens' queries and keys by a 2D axial RoPE (the prefix tokens are left as
// they are). Pre-norm blocks with LayerScale on both branches and a SwiGLU MLP.
// SAM 3D Body takes `get_intermediate_layers(n=1, norm=True, reshape=True)`:
// the final norm over the last block, the prefix dropped.
//
// Ported from dinov3's `models/vision_transformer.py`, `layers/attention.py`
// (SelfAttention, LinearKMaskedBias), `layers/ffn_layers.py` (SwiGLUFFN) and
// `layers/rope_position_encoding.py`; the depth plugin's DINOv2 is the model
// this one is written after.

import { llm, f32 } from './llm.js';
import * as op from './ops.js';
import { submit, copy, breathe, breathing, uploadEach } from './gpu.js';

export const PATCH = 16;
const HEAD = 64;
const EPS = 1e-5;

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

function disposeAll(v) {
	if (!v) return;
	if (typeof v.dispose === 'function') { v.dispose(); return; }
	if (typeof v === 'object') for (const x of Object.values(v)) disposeAll(x);
}

export class DinoV3 {
	// `m` is an open llm.js Model (safetensors or `openTorch`); `prefix` is where
	// the ViT's tensors are in it. `linear` is how the blocks' matmul weights go
	// to the GPU: 'q8' (Q8_0, a quarter of float32, the matrix cores' format) or
	// 'f32'. `exact` keeps the SwiGLU down-projection in float32 on the matrix
	// cores, whose float16 operands an MLP's hidden values can overflow.
	constructor(m, { prefix = 'backbone.encoder.', linear = 'q8', exact = true } = {}) {
		this.m = m;
		this.p = prefix;
		this.linear = linear;
		this.exact = exact;
		const shape = (name) => m.shape(`${prefix}${name}`); // GGUF order, fastest first
		this.dim = shape('cls_token')[0];
		this.heads = this.dim / HEAD;
		this.storage = m.has(`${prefix}storage_tokens`) ? shape('storage_tokens')[1] : 0;
		this.prefixTokens = 1 + this.storage;
		this.ffn = shape('blocks.0.mlp.w1.weight')[1];
		let n = 0;
		while (m.has(`${prefix}blocks.${n}.norm1.weight`)) n++;
		this.depth = n;
		if (shape('patch_embed.proj.weight')[0] !== PATCH) throw new Error(`DINOv3: patch size ${shape('patch_embed.proj.weight')[0]}, expected ${PATCH}`);
		// Stored in the checkpoint's own type (bf16 in SAM 3D Body's), and used as
		// stored: the rounded periods are the ones the model was trained with.
		this.periods = m.floats(`${prefix}rope_embed.periods`);
		if (this.periods.length !== HEAD / 4) throw new Error(`DINOv3: ${this.periods.length} RoPE periods for head width ${HEAD}`);
	}

	async load() {
		const m = this.m, D = this.dim, P = this.p;
		const t0 = llm.now();
		const f = (name) => m.upload(`${P}${name}`, 'f32');
		const w = (name) => m.upload(`${P}${name}`, this.linear);
		// The conv's [out, 3, 16, 16] reordered to patchify's (y, x, c) per patch.
		const conv = m.floats(`${P}patch_embed.proj.weight`);
		const pd = 3 * PATCH * PATCH;
		const patchW = new Float32Array(D * pd);
		for (let o = 0; o < D; o++) {
			for (let c = 0; c < 3; c++) {
				for (let p = 0; p < PATCH * PATCH; p++) patchW[o * pd + p * 3 + c] = conv[(o * 3 + c) * PATCH * PATCH + p];
			}
		}
		const prefixRows = new Float32Array(this.prefixTokens * D);
		prefixRows.set(m.floats(`${P}cls_token`));
		if (this.storage) prefixRows.set(m.floats(`${P}storage_tokens`), D);
		this.g = {
			patchW: upload(patchW), patchB: f('patch_embed.proj.bias'),
			prefix: upload(prefixRows),
			normW: f('norm.weight'), normB: f('norm.bias'),
		};
		this.blocks = [];
		for (let l = 0; l < this.depth; l++) {
			const b = `blocks.${l}.`;
			// LinearKMaskedBias: the bias times its mask (zero over K). The mask
			// is a buffer, so it is in the checkpoint; without one the bias is
			// used whole.
			let bias = m.has(`${P}${b}attn.qkv.bias`) ? m.floats(`${P}${b}attn.qkv.bias`) : null;
			if (bias && m.has(`${P}${b}attn.qkv.bias_mask`)) {
				const mask = m.floats(`${P}${b}attn.qkv.bias_mask`);
				bias = bias.map((v, i) => v * mask[i]);
			}
			const third = (i) => {
				const part = bias?.subarray(i * D, (i + 1) * D);
				return part && part.some((v) => v !== 0) ? upload(part) : null;
			};
			this.blocks.push(await uploadEach({
				n1w: () => f(`${b}norm1.weight`), n1b: () => f(`${b}norm1.bias`),
				qkv: () => w(`${b}attn.qkv.weight`),
				qb: () => third(0), kb: () => third(1), vb: () => third(2),
				proj: () => w(`${b}attn.proj.weight`), projB: () => f(`${b}attn.proj.bias`),
				ls1: () => f(`${b}ls1.gamma`),
				n2w: () => f(`${b}norm2.weight`), n2b: () => f(`${b}norm2.bias`),
				w1: () => w(`${b}mlp.w1.weight`), w1b: () => f(`${b}mlp.w1.bias`),
				w2: () => w(`${b}mlp.w2.weight`), w2b: () => f(`${b}mlp.w2.bias`),
				w3: () => w(`${b}mlp.w3.weight`), w3b: () => f(`${b}mlp.w3.bias`),
				ls2: () => f(`${b}ls2.gamma`),
			}));
		}
		llm.print(`  DINOv3: dim ${D}, ${this.depth} blocks, ${this.heads} heads, ${this.storage} storage tokens, ${this.linear} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll(this.g);
		for (const b of this.blocks ?? []) disposeAll(b);
		this.g = null;
		this.blocks = null;
	}

	// cos/sin [gh * gw, HEAD / 2] for rope_neox. A patch's coordinates are its
	// centre over the grid, each side on its own (`normalize_coords="separate"`),
	// mapped to [-1, 1]; angle j turns with the row for j < HEAD / 4 and with the
	// column after, 2 pi * coord / period. Pair j is (j, j + HEAD / 2), DINOv3's
	// rotate-half. In float32, as `pos_embed_rope_dtype="fp32"` has it.
	ropeTables(gh, gw) {
		const half = HEAD / 2, quarter = HEAD / 4;
		const n = gh * gw;
		const cos = new Float32Array(n * half), sin = new Float32Array(n * half);
		const twoPi = Math.fround(2 * Math.PI);
		for (let t = 0; t < n; t++) {
			const y = Math.floor(t / gw), x = t % gw;
			const cy = Math.fround(2 * Math.fround((y + 0.5) / gh) - 1);
			const cx = Math.fround(2 * Math.fround((x + 0.5) / gw) - 1);
			for (let j = 0; j < half; j++) {
				const c = j < quarter ? cy : cx;
				const angle = Math.fround(Math.fround(twoPi * c) / this.periods[j % quarter]);
				cos[t * half + j] = Math.cos(angle);
				sin[t * half + j] = Math.sin(angle);
			}
		}
		return { cos: upload(cos), sin: upload(sin) };
	}

	// `pixels` is [3, H, W], normalised (ImageNet mean and std), the sides
	// multiples of 16: a Float32Array, or a Tensor already on the GPU (which
	// stays the caller's). Returns the feature map [dim, H / 16, W / 16] as a
	// Tensor, which the caller disposes. `tokens` returns the same patches as
	// rows [gh * gw, dim] instead; `quiet` leaves out the timing line.
	// `stopAfter` (a block index) returns the tokens [prefix + patches, dim]
	// after that block instead, for comparing a port stage by stage. `layers`
	// (block indices, with `tokens`) lays those blocks' patches beside the last
	// one's, each through the final norm: rows [gh * gw, dim * (layers + 1)],
	// the earlier blocks first. A middle block knows better where things are,
	// the last what they are.
	async encode(pixels, H, W, { stopAfter = -1, tokens = false, quiet = false, layers = null } = {}) {
		const D = this.dim, F = this.ffn, nh = this.heads, g = this.g;
		const gh = H / PATCH, gw = W / PATCH, N = gh * gw, pre = this.prefixTokens, n = pre + N;
		if (!Number.isInteger(gh) || !Number.isInteger(gw)) throw new Error(`DINOv3: ${W}x${H} is not a multiple of ${PATCH}`);
		const t0 = llm.now();
		const img = pixels instanceof Float32Array ? upload(pixels) : pixels;
		const a = {
			patches: f32(N * 3 * PATCH * PATCH),
			h: f32(n * D), x: f32(n * D), q: f32(n * D), k: f32(n * D), v: f32(n * D),
			qt: f32(n * D), kt: f32(n * D), vt: f32(n * D), o: f32(n * D),
			gate: f32(n * F), up: f32(n * F),
		};
		const rope = this.ropeTables(gh, gw);
		const rows = (t, first, count) => t.view(first * D * 4, count * D * 4);
		const extra = tokens && layers?.length ? layers : null, wide = D * ((extra?.length ?? 0) + 1);
		const out = tokens ? f32(wide * N, [wide, N]) : f32(D * N, [gw, gh, D]);
		try {
			// Prefix tokens, then the patches' embeddings after them.
			copy(g.prefix, a.h, pre * D * 4);
			op.patchify(img, a.patches, 3, H, W, PATCH);
			op.matmul(g.patchW, a.patches, rows(a.h, pre, N), D, 3 * PATCH * PATCH, N);
			op.biasAdd(rows(a.h, pre, N), g.patchB, N * D, D);
			submit();

			for (const [l, b] of this.blocks.entries()) {
				this.block(b, a, n, N, rope);
				if (l === stopAfter) {
					out.dispose();
					const all = f32(n * D, [D, n]);
					copy(a.h, all, n * D * 4);
					return all;
				}
				const at = extra ? extra.indexOf(l) : -1;
				if (at >= 0) {
					op.layerNormAffine(a.h, g.normW, g.normB, a.x, D, n, EPS);
					op.copyRows(rows(a.x, pre, N), out, N, D, wide, at * D);
				}
				if (breathing()) await breathe(true);
				else submit();
			}
			op.layerNormAffine(a.h, g.normW, g.normB, a.x, D, n, EPS);
			if (extra) op.copyRows(rows(a.x, pre, N), out, N, D, wide, extra.length * D);
			else if (tokens) copy(a.x, out, N * D * 4, pre * D * 4);
			else op.transposeChannelSpatial(rows(a.x, pre, N), out, D, N, 1);
			submit();
			if (!quiet) llm.print(`  DINOv3: ${W}x${H}, ${n} tokens, in ${llm.since(t0)}`);
			return out;
		} catch (e) {
			out.dispose();
			throw e;
		} finally {
			submit();
			disposeAll(a);
			disposeAll(rope);
			if (img !== pixels) img.dispose();
		}
	}

	// One block over a.h [n, dim] in place; the last N rows are patches.
	block(b, a, n, N, rope) {
		const D = this.dim, F = this.ffn, nh = this.heads, pre = this.prefixTokens;
		const rows = (t, first, count) => t.view(first * D * 4, count * D * 4);
		op.layerNormAffine(a.h, b.n1w, b.n1b, a.x, D, n, EPS);
		op.matmul(b.qkv, a.x, a.q, D, D, n, 0, true);
		op.matmul(b.qkv, a.x, a.k, D, D, n, D, true);
		op.matmul(b.qkv, a.x, a.v, D, D, n, 2 * D);
		if (b.qb) op.biasAdd(a.q, b.qb, n * D, D);
		if (b.kb) op.biasAdd(a.k, b.kb, n * D, D);
		if (b.vb) op.biasAdd(a.v, b.vb, n * D, D);
		op.ropeNeox(rows(a.q, pre, N), rope.cos, rope.sin, HEAD, nh, N, true);
		op.ropeNeox(rows(a.k, pre, N), rope.cos, rope.sin, HEAD, nh, N);
		op.transposeHeads(a.q, a.qt, n, nh, HEAD, 0, true);
		op.transposeHeads(a.k, a.kt, n, nh, HEAD, 0, true);
		op.transposeHeads(a.v, a.vt, n, nh, HEAD, 0);
		op.flashAttention(a.qt, a.kt, a.vt, a.o, nh, n, HEAD);
		op.linearBiasRows(b.proj, b.projB, a.o, a.q, D, D, n);
		op.gatedAdd(a.h, b.ls1, a.q, n * D, D, 0);

		op.layerNormAffine(a.h, b.n2w, b.n2b, a.x, D, n, EPS);
		op.matmul(b.w1, a.x, a.gate, F, D, n, 0, true);
		op.matmul(b.w2, a.x, a.up, F, D, n);
		op.biasAdd(a.gate, b.w1b, n * F, F);
		op.biasAdd(a.up, b.w2b, n * F, F);
		op.siluMul(a.gate, a.up, n * F);
		op.matmul(b.w3, a.gate, a.q, D, F, n, 0, false, this.exact);
		op.biasAdd(a.q, b.w3b, n * D, D);
		op.gatedAdd(a.h, b.ls2, a.q, n * D, D, 0);
	}
}
