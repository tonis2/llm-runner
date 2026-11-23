// DINOv2's ViT (facebookresearch/dinov2), with or without register tokens:
// an image in, its tokens out - the last block's, before the final norm
// (`x_prenorm`), as TRELLIS and LATO.2 condition on them.
//
// 14-pixel patches; a CLS token and a learned position table added to the
// CLS and patch tokens; then the registers (no position) between them. Pre-norm
// blocks with LayerScale on both branches and a GELU (erf) MLP.
//
// Ported from dinov2's `models/vision_transformer.py` (DinoVisionTransformer,
// prepare_tokens_with_masks, forward_features) and `layers/block.py`.

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

export class DinoV2 {
	// `m` is an open llm.js Model (safetensors or `openTorch`); `prefix` is where
	// the ViT's tensors are in it. `linear` is how the blocks' matmul weights go
	// to the GPU: 'f32' or 'q8' (Q8_0, run on the matrix cores in float16).
	constructor(m, { prefix = '', linear = 'f32' } = {}) {
		this.m = m;
		this.p = prefix;
		this.linear = linear;
		const shape = (name) => m.shape(`${prefix}${name}`); // GGUF order, fastest first
		this.dim = shape('cls_token')[0];
		this.heads = this.dim / HEAD;
		this.registers = m.has(`${prefix}register_tokens`) ? shape('register_tokens')[1] : 0;
		this.ffn = shape('blocks.0.mlp.fc1.weight')[1];
		const pos = shape('pos_embed');
		this.grid = Math.round(Math.sqrt(pos[1] - 1));
		let n = 0;
		while (m.has(`${prefix}blocks.${n}.norm1.weight`)) n++;
		this.depth = n;
		if (shape('patch_embed.proj.weight')[0] !== PATCH) throw new Error(`DINOv2: patch size ${shape('patch_embed.proj.weight')[0]}, expected ${PATCH}`);
	}

	async load() {
		const m = this.m, D = this.dim, P = this.p;
		const t0 = llm.now();
		const f = (name) => m.upload(`${P}${name}`, 'f32');
		const w = (name) => m.upload(`${P}${name}`, this.linear);
		// The conv's [out, 3, 14, 14] reordered to patchify's (y, x, c) per patch.
		const conv = m.floats(`${P}patch_embed.proj.weight`);
		const pd = 3 * PATCH * PATCH;
		const patchW = new Float32Array(D * pd);
		for (let o = 0; o < D; o++) {
			for (let c = 0; c < 3; c++) {
				for (let p = 0; p < PATCH * PATCH; p++) patchW[o * pd + p * 3 + c] = conv[(o * 3 + c) * PATCH * PATCH + p];
			}
		}
		this.pos = m.floats(`${P}pos_embed`);
		this.cls = m.floats(`${P}cls_token`);
		this.reg = this.registers ? m.floats(`${P}register_tokens`) : null;
		this.g = { patchW: upload(patchW), patchB: f('patch_embed.proj.bias') };
		this.blocks = [];
		for (let l = 0; l < this.depth; l++) {
			const b = `blocks.${l}.`;
			this.blocks.push(await uploadEach({
				n1w: () => f(`${b}norm1.weight`), n1b: () => f(`${b}norm1.bias`),
				qkv: () => w(`${b}attn.qkv.weight`), qkvB: () => f(`${b}attn.qkv.bias`),
				proj: () => w(`${b}attn.proj.weight`), projB: () => f(`${b}attn.proj.bias`),
				ls1: () => f(`${b}ls1.gamma`),
				n2w: () => f(`${b}norm2.weight`), n2b: () => f(`${b}norm2.bias`),
				fc1: () => w(`${b}mlp.fc1.weight`), fc1B: () => f(`${b}mlp.fc1.bias`),
				fc2: () => w(`${b}mlp.fc2.weight`), fc2B: () => f(`${b}mlp.fc2.bias`),
				ls2: () => f(`${b}ls2.gamma`),
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
	// position. Only the pretrained grid is supported (518 for the ViT-L/14).
	prefixRows(gh, gw) {
		const D = this.dim, R = this.registers, N = gh * gw;
		if (gh !== this.grid || gw !== this.grid) throw new Error(`DINOv2: ${gw}x${gh} patches; only the pretrained ${this.grid}x${this.grid} is supported`);
		const rows = new Float32Array((1 + R + N) * D);
		for (let c = 0; c < D; c++) rows[c] = this.cls[c] + this.pos[c];
		if (R) rows.set(this.reg, D);
		rows.set(this.pos.subarray(D), (1 + R) * D);
		return rows;
	}

	// `pixels` [3, H, W] normalised (ImageNet mean and std), H and W the
	// pretrained grid's 14 pixels a patch. Returns the last block's tokens
	// [1 + registers + patches, dim] before the final norm, a Tensor the caller
	// disposes.
	async encode(pixels, H, W, { quiet = false } = {}) {
		const D = this.dim, F = this.ffn, nh = this.heads;
		const gh = H / PATCH, gw = W / PATCH, N = gh * gw, pre = 1 + this.registers, n = pre + N;
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
		op.matmul(b.qkv, a.x, a.q, D, D, n, 0, true);
		op.matmul(b.qkv, a.x, a.k, D, D, n, D, true);
		op.matmul(b.qkv, a.x, a.v, D, D, n, 2 * D);
		op.biasAdd(a.q, b.qkvB.view(0, D * 4), n * D, D);
		op.biasAdd(a.k, b.qkvB.view(D * 4, D * 4), n * D, D);
		op.biasAdd(a.v, b.qkvB.view(2 * D * 4, D * 4), n * D, D);
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
