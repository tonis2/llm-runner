// DINOv2's ViT (facebookresearch/dinov2), with or without register tokens:
// an image in, its tokens out - by default the last block's, before the final
// norm (`x_prenorm`), as TRELLIS and LATO.2 condition on them.
//
// 14-pixel patches; a CLS token and a learned position table added to the
// CLS and patch tokens; then the registers (no position) between them. Pre-norm
// blocks with LayerScale on both branches and a GELU (erf) MLP.
//
// Ported from dinov2's `models/vision_transformer.py` (DinoVisionTransformer,
// prepare_tokens_with_masks, forward_features) and `layers/block.py`.
//
// The Hugging Face checkpoints (transformers' Dinov2Model, as Hi3D loads
// facebook/dinov2-large) are read too: q, k and v each their own matrix, and
// the final norm as `layernorm`. Other grids than the pretrained one get the
// position table resized as transformers 4.46 does it (bicubic to the new
// size, align_corners off), which is what `encode(..., { norm: 'model' })` at
// Hi3D's 756 pixels needs.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, copy, breathe, breathing, uploadEach } from '/core/gpu.js';

export const PATCH = 14;
const HEAD = 64;
const EPS = 1e-6;

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

// Where each tensor is in the two checkpoint layouts (block names after the
// block prefix).
const FB_NAMES = {
	cls: 'cls_token', registers: 'register_tokens', pos: 'pos_embed',
	patch: 'patch_embed.proj.weight', patchBias: 'patch_embed.proj.bias', block: 'blocks.', fc1: 'mlp.fc1.weight',
	norm: 'norm.',
};
const HF_NAMES = {
	cls: 'embeddings.cls_token', registers: 'embeddings.register_tokens', pos: 'embeddings.position_embeddings',
	patch: 'embeddings.patch_embeddings.projection.weight', patchBias: 'embeddings.patch_embeddings.projection.bias',
	block: 'encoder.layer.', fc1: 'mlp.fc1.weight', norm: 'layernorm.',
};

// torch's bicubic weights (A = -0.75) for the 4 taps around t in [0, 1).
function cubicWeights(t) {
	const A = -0.75;
	const near = (x) => ((A + 2) * x - (A + 3)) * x * x + 1;
	const far = (x) => ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
	return [far(t + 1), near(t), near(1 - t), far(2 - t)];
}

// Per output index: the 4 input indices (clamped to the edge) and weights of
// F.interpolate(mode='bicubic', align_corners=False) from inSize to outSize.
function cubicTaps(inSize, outSize) {
	const scale = inSize / outSize, taps = [];
	for (let o = 0; o < outSize; o++) {
		const src = Math.fround(Math.fround(scale * (o + 0.5)) - 0.5);
		const i = Math.floor(src);
		const w = cubicWeights(Math.fround(src - i));
		const at = [i - 1, i, i + 1, i + 2].map((v) => Math.min(Math.max(v, 0), inSize - 1));
		taps.push({ at, w });
	}
	return taps;
}

// A [g * g, dim] position table (rows in raster order) resized to [gh * gw, dim].
export function resizePositions(table, g, gh, gw, dim) {
	const tx = cubicTaps(g, gw), ty = cubicTaps(g, gh);
	// Along x first, then y, as separable passes (torch does both at once; the
	// sums agree to float rounding).
	const mid = new Float32Array(g * gw * dim);
	for (let y = 0; y < g; y++) {
		for (let x = 0; x < gw; x++) {
			const { at, w } = tx[x], o = (y * gw + x) * dim;
			const r0 = (y * g + at[0]) * dim, r1 = (y * g + at[1]) * dim, r2 = (y * g + at[2]) * dim, r3 = (y * g + at[3]) * dim;
			for (let c = 0; c < dim; c++) mid[o + c] = table[r0 + c] * w[0] + table[r1 + c] * w[1] + table[r2 + c] * w[2] + table[r3 + c] * w[3];
		}
	}
	const out = new Float32Array(gh * gw * dim);
	for (let y = 0; y < gh; y++) {
		const { at, w } = ty[y];
		for (let x = 0; x < gw; x++) {
			const o = (y * gw + x) * dim;
			const r0 = (at[0] * gw + x) * dim, r1 = (at[1] * gw + x) * dim, r2 = (at[2] * gw + x) * dim, r3 = (at[3] * gw + x) * dim;
			for (let c = 0; c < dim; c++) out[o + c] = mid[r0 + c] * w[0] + mid[r1 + c] * w[1] + mid[r2 + c] * w[2] + mid[r3 + c] * w[3];
		}
	}
	return out;
}

export class DinoV2 {
	// `m` is an open llm.js Model (safetensors or `openTorch`); `prefix` is where
	// the ViT's tensors are in it. `linear` is how the blocks' matmul weights go
	// to the GPU: 'f32' or 'q8' (Q8_0, run on the matrix cores in float16).
	constructor(m, { prefix = '', linear = 'f32' } = {}) {
		this.m = m;
		this.p = prefix;
		this.linear = linear;
		const hf = this.hf = m.has(`${prefix}embeddings.cls_token`);
		const N = this.names = hf ? HF_NAMES : FB_NAMES;
		const shape = (name) => m.shape(`${prefix}${N[name] ?? name}`); // GGUF order, fastest first
		this.dim = shape('cls')[0];
		this.heads = this.dim / HEAD;
		this.registers = m.has(`${prefix}${N.registers}`) ? shape('registers')[1] : 0;
		this.ffn = m.shape(`${prefix}${N.block}0.${N.fc1}`)[1];
		this.grid = Math.round(Math.sqrt(shape('pos')[1] - 1));
		let n = 0;
		while (m.has(`${prefix}${N.block}${n}.norm1.weight`)) n++;
		this.depth = n;
		if (shape('patch')[0] !== PATCH) throw new Error(`DINOv2: patch size ${shape('patch')[0]}, expected ${PATCH}`);
	}

	async load() {
		const m = this.m, D = this.dim, P = this.p, N = this.names;
		const t0 = llm.now();
		const f = (name) => m.upload(`${P}${name}`, 'f32');
		const w = (name) => m.upload(`${P}${name}`, this.linear);
		// The conv's [out, 3, 14, 14] reordered to patchify's (y, x, c) per patch.
		const conv = m.floats(`${P}${N.patch}`);
		const pd = 3 * PATCH * PATCH;
		const patchW = new Float32Array(D * pd);
		for (let o = 0; o < D; o++) {
			for (let c = 0; c < 3; c++) {
				for (let p = 0; p < PATCH * PATCH; p++) patchW[o * pd + p * 3 + c] = conv[(o * 3 + c) * PATCH * PATCH + p];
			}
		}
		this.pos = m.floats(`${P}${N.pos}`);
		this.cls = m.floats(`${P}${N.cls}`);
		this.reg = this.registers ? m.floats(`${P}${N.registers}`) : null;
		this.g = {
			patchW: upload(patchW), patchB: f(N.patchBias),
			normW: m.has(`${P}${N.norm}weight`) ? f(`${N.norm}weight`) : null,
			normB: m.has(`${P}${N.norm}bias`) ? f(`${N.norm}bias`) : null,
		};
		this.blocks = [];
		for (let l = 0; l < this.depth; l++) {
			const b = `${N.block}${l}.`;
			const parts = this.hf ? {
				wq: () => w(`${b}attention.attention.query.weight`), qb: () => f(`${b}attention.attention.query.bias`),
				wk: () => w(`${b}attention.attention.key.weight`), kb: () => f(`${b}attention.attention.key.bias`),
				wv: () => w(`${b}attention.attention.value.weight`), vb: () => f(`${b}attention.attention.value.bias`),
				proj: () => w(`${b}attention.output.dense.weight`), projB: () => f(`${b}attention.output.dense.bias`),
				ls1: () => f(`${b}layer_scale1.lambda1`), ls2: () => f(`${b}layer_scale2.lambda1`),
			} : {
				qkv: () => w(`${b}attn.qkv.weight`), qkvB: () => f(`${b}attn.qkv.bias`),
				proj: () => w(`${b}attn.proj.weight`), projB: () => f(`${b}attn.proj.bias`),
				ls1: () => f(`${b}ls1.gamma`), ls2: () => f(`${b}ls2.gamma`),
			};
			this.blocks.push(await uploadEach({
				n1w: () => f(`${b}norm1.weight`), n1b: () => f(`${b}norm1.bias`),
				n2w: () => f(`${b}norm2.weight`), n2b: () => f(`${b}norm2.bias`),
				fc1: () => w(`${b}mlp.fc1.weight`), fc1B: () => f(`${b}mlp.fc1.bias`),
				fc2: () => w(`${b}mlp.fc2.weight`), fc2B: () => f(`${b}mlp.fc2.bias`),
				...parts,
			}));
		}
		llm.print(`  DINOv2: dim ${D}, ${this.depth} blocks, ${this.heads} heads, ${this.registers} registers, ${this.linear} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll(this.g);
		for (const b of this.blocks ?? []) disposeAll(b);
		this.g = null;
		this.blocks = null;
	}

	// The tokens' starting rows [1 + registers + gh * gw, dim] less the patch
	// embeddings: CLS plus its position, the registers, then each patch's
	// position (resized when the grid is not the pretrained one).
	prefixRows(gh, gw) {
		const D = this.dim, R = this.registers, N = gh * gw;
		const rows = new Float32Array((1 + R + N) * D);
		for (let c = 0; c < D; c++) rows[c] = this.cls[c] + this.pos[c];
		if (R) rows.set(this.reg, D);
		const table = this.pos.subarray(D);
		rows.set(gh === this.grid && gw === this.grid ? table : resizePositions(table, this.grid, gh, gw, D), (1 + R) * D);
		return rows;
	}

	// `pixels` [3, H, W] normalised (ImageNet mean and std), H and W multiples
	// of 14. Returns the tokens [1 + registers + patches, dim] as a Tensor the
	// caller disposes: the last block's before the final norm, or after it with
	// `norm: 'model'` (Hugging Face's `last_hidden_state`).
	async encode(pixels, H, W, { quiet = false, norm = 'none' } = {}) {
		const D = this.dim, F = this.ffn, nh = this.heads;
		const gh = H / PATCH, gw = W / PATCH, N = gh * gw, pre = 1 + this.registers, n = pre + N;
		if (!Number.isInteger(gh) || !Number.isInteger(gw)) throw new Error(`DINOv2: ${W}x${H} is not a multiple of ${PATCH}`);
		if (norm === 'model' && !this.g.normW) throw new Error('DINOv2: the checkpoint has no final norm');
		const t0 = llm.now();
		const img = pixels instanceof Float32Array ? upload(pixels) : pixels;
		const a = {
			patches: f32(N * 3 * PATCH * PATCH), start: upload(this.prefixRows(gh, gw)),
			x: f32(n * D), q: f32(n * D), k: f32(n * D), v: f32(n * D), o: f32(n * D), up: f32(n * F),
			pad: { q: f32(n * nh * 128), k: f32(n * nh * 128), v: f32(n * nh * 128), o: f32(n * nh * 128) },
		};
		const out = f32(n * D, [D, n]);
		const rows = (t, first, count) => t.view(first * D * 4, count * D * 4);
		try {
			op.patchify(img, a.patches, 3, H, W, PATCH);
			op.matmul(this.g.patchW, a.patches, rows(out, pre, N), D, 3 * PATCH * PATCH, N);
			op.biasAdd(rows(out, pre, N), this.g.patchB, N * D, D);
			op.add(out, a.start, n * D);
			submit();
			for (const b of this.blocks) {
				this.block(b, out, a, n);
				if (breathing()) await breathe(true);
			}
			if (norm === 'model') {
				op.layerNormAffine(out, this.g.normW, this.g.normB, a.x, D, n, EPS);
				copy(a.x, out, n * D * 4);
			}
			submit();
			if (!quiet) llm.print(`  DINOv2: ${W}x${H}, ${n} tokens, in ${llm.since(t0)}`);
			return out;
		} catch (e) {
			out.dispose();
			throw e;
		} finally {
			submit();
			disposeAll(a);
			if (img !== pixels) img.dispose();
		}
	}

	// One block over h [n, dim] in place.
	block(b, h, a, n) {
		const D = this.dim, F = this.ffn, nh = this.heads;
		op.layerNormAffine(h, b.n1w, b.n1b, a.x, D, n, EPS);
		if (b.qkv) {
			op.matmul(b.qkv, a.x, a.q, D, D, n, 0, true);
			op.matmul(b.qkv, a.x, a.k, D, D, n, D, true);
			op.matmul(b.qkv, a.x, a.v, D, D, n, 2 * D);
			op.biasAdd(a.q, b.qkvB.view(0, D * 4), n * D, D);
			op.biasAdd(a.k, b.qkvB.view(D * 4, D * 4), n * D, D);
			op.biasAdd(a.v, b.qkvB.view(2 * D * 4, D * 4), n * D, D);
		} else {
			op.linearBiasRows(b.wq, b.qb, a.x, a.q, D, D, n);
			op.linearBiasRows(b.wk, b.kb, a.x, a.k, D, D, n);
			op.linearBiasRows(b.wv, b.vb, a.x, a.v, D, D, n);
		}
		op.attention64Padded(a.q, a.k, a.v, a.o, a.pad, nh, n);
		op.linearBiasRows(b.proj, b.projB, a.o, a.q, D, D, n);
		op.gatedAdd(h, b.ls1, a.q, n * D, D, 0);

		op.layerNormAffine(h, b.n2w, b.n2b, a.x, D, n, EPS);
		op.linearBiasRows(b.fc1, b.fc1B, a.x, a.up, F, D, n);
		op.geluErf(a.up, n * F);
		op.linearBiasRows(b.fc2, b.fc2B, a.up, a.q, D, F, n);
		op.gatedAdd(h, b.ls2, a.q, n * D, D, 0);
	}
}
