// CLIP's vision tower (transformers' CLIPVisionModel, as in
// openai/clip-vit-large-patch14): an image in, the pooled embedding out - the
// CLS token after the last block and `post_layernorm`, not projected. Hi3D
// conditions on it.
//
// 14-pixel patches (a conv without bias), a class embedding in front, a learned
// position table, `pre_layrnorm`, then pre-norm blocks: attention with biased
// q/k/v/out projections and a quick_gelu MLP, both residuals plain.
//
// Ported from transformers' `models/clip/modeling_clip.py` (CLIPVisionEmbeddings,
// CLIPEncoderLayer, CLIPVisionTransformer).

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, uploadEach } from '/core/gpu.js';

const HEAD = 64;

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

export class ClipVision {
	// `m` is an open llm.js Model holding the tower under `prefix`; `linear` is
	// how the blocks' matmul weights go to the GPU ('f32' or 'q8'). `eps` is the
	// config's layer_norm_eps.
	constructor(m, { prefix = 'vision_model.', linear = 'f32', eps = 1e-5 } = {}) {
		this.m = m;
		this.p = prefix;
		this.linear = linear;
		this.eps = eps;
		const shape = (name) => m.shape(`${prefix}${name}`); // GGUF order, fastest first
		const conv = shape('embeddings.patch_embedding.weight');
		this.patch = conv[0];
		this.dim = conv[3];
		this.heads = this.dim / HEAD;
		this.positions = shape('embeddings.position_embedding.weight')[1];
		this.grid = Math.round(Math.sqrt(this.positions - 1));
		this.size = this.grid * this.patch;
		this.ffn = shape('encoder.layers.0.mlp.fc1.weight')[1];
		let n = 0;
		while (m.has(`${prefix}encoder.layers.${n}.layer_norm1.weight`)) n++;
		this.depth = n;
	}

	async load() {
		const m = this.m, D = this.dim, P = this.p, K = this.patch;
		const t0 = llm.now();
		const f = (name) => m.upload(`${P}${name}`, 'f32');
		const w = (name) => m.upload(`${P}${name}`, this.linear);
		// The conv's [out, 3, k, k] reordered to patchify's (y, x, c) per patch.
		const conv = m.floats(`${P}embeddings.patch_embedding.weight`);
		const pd = 3 * K * K;
		const patchW = new Float32Array(D * pd);
		for (let o = 0; o < D; o++) {
			for (let c = 0; c < 3; c++) {
				for (let p = 0; p < K * K; p++) patchW[o * pd + p * 3 + c] = conv[(o * 3 + c) * K * K + p];
			}
		}
		// The class embedding goes in row 0, then each row gets its position.
		const start = Float32Array.from(m.floats(`${P}embeddings.position_embedding.weight`));
		const cls = m.floats(`${P}embeddings.class_embedding`);
		for (let c = 0; c < D; c++) start[c] += cls[c];
		this.g = {
			patchW: upload(patchW), start: upload(start), zero: upload(new Float32Array(D)),
			preW: f('pre_layrnorm.weight'), preB: f('pre_layrnorm.bias'),
			postW: f('post_layernorm.weight'), postB: f('post_layernorm.bias'),
		};
		this.blocks = [];
		for (let l = 0; l < this.depth; l++) {
			const b = `encoder.layers.${l}.`;
			this.blocks.push(await uploadEach({
				n1w: () => f(`${b}layer_norm1.weight`), n1b: () => f(`${b}layer_norm1.bias`),
				wq: () => w(`${b}self_attn.q_proj.weight`), qb: () => f(`${b}self_attn.q_proj.bias`),
				wk: () => w(`${b}self_attn.k_proj.weight`), kb: () => f(`${b}self_attn.k_proj.bias`),
				wv: () => w(`${b}self_attn.v_proj.weight`), vb: () => f(`${b}self_attn.v_proj.bias`),
				wo: () => w(`${b}self_attn.out_proj.weight`), ob: () => f(`${b}self_attn.out_proj.bias`),
				n2w: () => f(`${b}layer_norm2.weight`), n2b: () => f(`${b}layer_norm2.bias`),
				fc1: () => w(`${b}mlp.fc1.weight`), fc1B: () => f(`${b}mlp.fc1.bias`),
				fc2: () => w(`${b}mlp.fc2.weight`), fc2B: () => f(`${b}mlp.fc2.bias`),
			}));
		}
		llm.print(`  CLIP vision: dim ${D}, ${this.depth} blocks, ${this.heads} heads, ${this.size} pixels, ${this.linear} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll(this.g);
		for (const b of this.blocks ?? []) disposeAll(b);
		this.g = null;
		this.blocks = null;
	}

	// `pixels` [3, size, size], normalised with CLIP's mean and std. Returns the
	// pooled embedding (post_layernorm of the CLS token) as a Float32Array.
	async pooled(pixels, { quiet = false } = {}) {
		const D = this.dim, F = this.ffn, nh = this.heads, K = this.patch, S = this.size;
		const N = this.grid * this.grid, n = N + 1, eps = this.eps;
		const t0 = llm.now();
		const img = pixels instanceof Float32Array ? upload(pixels) : pixels;
		const a = {
			patches: f32(N * 3 * K * K), h: f32(n * D),
			x: f32(n * D), q: f32(n * D), k: f32(n * D), v: f32(n * D), o: f32(n * D), up: f32(n * F),
			pad: { q: f32(n * nh * 128), k: f32(n * nh * 128), v: f32(n * nh * 128), o: f32(n * nh * 128) },
			pool: f32(D),
		};
		try {
			op.patchify(img, a.patches, 3, S, S, K);
			op.fillRows(this.g.zero, a.x, 0, 1, D);
			op.matmul(this.g.patchW, a.patches, a.x.view(D * 4, N * D * 4), D, 3 * K * K, N);
			op.add(a.x, this.g.start, n * D);
			op.layerNormAffine(a.x, this.g.preW, this.g.preB, a.h, D, n, eps);
			for (const b of this.blocks) this.block(b, a, n);
			op.layerNormAffine(a.h, this.g.postW, this.g.postB, a.pool, D, 1, eps);
			submit();
			const out = new Float32Array(a.pool.buffer.readBytes(0, D * 4).buffer);
			if (!quiet) llm.print(`  CLIP vision: ${S}x${S}, ${n} tokens, in ${llm.since(t0)}`);
			return out;
		} finally {
			submit();
			disposeAll(a);
			if (img !== pixels) img.dispose();
		}
	}

	// One block over a.h [n, dim] in place.
	block(b, a, n) {
		const D = this.dim, F = this.ffn, nh = this.heads, eps = this.eps;
		op.layerNormAffine(a.h, b.n1w, b.n1b, a.x, D, n, eps);
		op.linearBiasRows(b.wq, b.qb, a.x, a.q, D, D, n);
		op.linearBiasRows(b.wk, b.kb, a.x, a.k, D, D, n);
		op.linearBiasRows(b.wv, b.vb, a.x, a.v, D, D, n);
		op.attention64Padded(a.q, a.k, a.v, a.o, a.pad, nh, n);
		op.linearBiasRows(b.wo, b.ob, a.o, a.q, D, D, n);
		op.add(a.h, a.q, n * D);

		op.layerNormAffine(a.h, b.n2w, b.n2b, a.x, D, n, eps);
		op.linearBiasRows(b.fc1, b.fc1B, a.x, a.up, F, D, n);
		op.quickGelu(a.up, n * F);
		op.linearBiasRows(b.fc2, b.fc2B, a.up, a.q, D, F, n);
		op.add(a.h, a.q, n * D);
	}
}
