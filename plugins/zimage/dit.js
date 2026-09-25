// The Z-Image Turbo DiT (Lumina2): a context refiner over the text, a noise
// refiner over the image patches, then 30 joint layers over [image | text],
// each with scale-only AdaLN, post-norms and tanh-gated residuals.
//
// The graph is `dependencies/zimage.c3l/dit.c3`'s.

import { llm, f32, GGML } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit, copy } from '../lib/gpu.js';

export const PATCH = 2;
const PAD_TO = 32;
const THETA = 256;
const AXES = [32, 48, 48];

export function padTo(n, m = PAD_TO) { return Math.ceil(n / m) * m; }

// cos/sin [seq, 64] for the 3-axis rope. `mode` 'full' is [image | image pad |
// text | text pad], 'image' the padded image alone, 'text' the padded text
// alone. Text runs 1-indexed along axis 0; image tokens sit at axis 0 =
// paddedText + 1 with their row and column on axes 1 and 2; image padding
// does not rotate.
export function ropeTables(nPatches, patchesW, textLen, mode) {
	const paddedText = padTo(textLen);
	const paddedPatches = padTo(nPatches);
	const seq = mode === 'full' ? paddedPatches + paddedText : mode === 'image' ? paddedPatches : paddedText;
	const half = 64;
	const cos = new Float32Array(seq * half);
	const sin = new Float32Array(seq * half);
	const freqs = AXES.map((ax) => {
		const f = new Float32Array(ax / 2);
		for (let k = 0; k < ax / 2; k++) f[k] = Math.fround(1 / Math.pow(THETA, (2 * k) / ax));
		return f;
	});
	for (let t = 0; t < seq; t++) {
		let pos;
		if (mode === 'text') {
			pos = [t + 1, 0, 0];
		} else if (t < nPatches) {
			pos = [paddedText + 1, Math.floor(t / patchesW), t % patchesW];
		} else if (t < paddedPatches) {
			pos = [0, 0, 0];
		} else {
			pos = [t - paddedPatches + 1, 0, 0];
		}
		let pair = 0;
		for (let a = 0; a < 3; a++) {
			for (let k = 0; k < freqs[a].length; k++) {
				const angle = Math.fround(pos[a] * freqs[a][k]);
				cos[t * half + pair] = Math.cos(angle);
				sin[t * half + pair] = Math.sin(angle);
				pair++;
			}
		}
	}
	return { cos, sin };
}

function upload(t) {
	const buf = f32(t.length);
	buf.buffer.write(t);
	return buf;
}

export class ZImageDiT {
	constructor(model) {
		this.model = model;
		const m = model;
		this.dim = m.shape('x_embedder.weight')[1];
		this.patchDim = m.shape('x_embedder.weight')[0];
		this.channels = this.patchDim / (PATCH * PATCH);
		this.headDim = m.shape('layers.0.attention.q_norm.weight')[0];
		this.heads = this.dim / this.headDim;
		this.ffn = m.shape('layers.0.feed_forward.w1.weight')[1];
		this.tDim = m.shape('t_embedder.mlp.2.weight')[1];
		this.tMlp = m.shape('t_embedder.mlp.0.weight')[1];
		this.textDim = m.shape('cap_embedder.1.weight')[0];
		const count = (prefix) => {
			let n = 0;
			while (m.has(`${prefix}.${n}.attention.qkv.weight`)) n++;
			return n;
		};
		this.nLayers = count('layers');
		this.nRefiner = count('noise_refiner');
		this.nContext = count('context_refiner');
	}

	load() {
		const m = this.model;
		const t0 = llm.now();
		const up = (name) => m.upload(name, 'auto');
		const f = (name) => m.upload(name, 'f32');
		const layer = (prefix, l, adaln) => ({
			qkv: up(`${prefix}.${l}.attention.qkv.weight`),
			qNorm: f(`${prefix}.${l}.attention.q_norm.weight`),
			kNorm: f(`${prefix}.${l}.attention.k_norm.weight`),
			out: up(`${prefix}.${l}.attention.out.weight`),
			w1: up(`${prefix}.${l}.feed_forward.w1.weight`),
			w2: up(`${prefix}.${l}.feed_forward.w2.weight`),
			w3: up(`${prefix}.${l}.feed_forward.w3.weight`),
			attnNorm1: f(`${prefix}.${l}.attention_norm1.weight`),
			attnNorm2: m.has(`${prefix}.${l}.attention_norm2.weight`) ? f(`${prefix}.${l}.attention_norm2.weight`) : null,
			ffnNorm1: f(`${prefix}.${l}.ffn_norm1.weight`),
			ffnNorm2: m.has(`${prefix}.${l}.ffn_norm2.weight`) ? f(`${prefix}.${l}.ffn_norm2.weight`) : null,
			adaW: adaln ? f(`${prefix}.${l}.adaLN_modulation.0.weight`) : null,
			adaB: adaln ? f(`${prefix}.${l}.adaLN_modulation.0.bias`) : null,
		});
		this.layers = [];
		for (let l = 0; l < this.nLayers; l++) this.layers.push(layer('layers', l, true));
		this.noiseRefiner = [];
		for (let l = 0; l < this.nRefiner; l++) this.noiseRefiner.push(layer('noise_refiner', l, true));
		this.contextRefiner = [];
		for (let l = 0; l < this.nContext; l++) this.contextRefiner.push(layer('context_refiner', l, false));
		const finalNorm = ['final_layer.norm.weight', 'norm_out.norm.weight'].find((n) => m.has(n));
		this.g = {
			capNorm: f('cap_embedder.0.weight'),
			capW: f('cap_embedder.1.weight'),
			capB: f('cap_embedder.1.bias'),
			t0w: f('t_embedder.mlp.0.weight'), t0b: f('t_embedder.mlp.0.bias'),
			t2w: f('t_embedder.mlp.2.weight'), t2b: f('t_embedder.mlp.2.bias'),
			xw: f('x_embedder.weight'), xb: f('x_embedder.bias'),
			finalNorm: finalNorm ? f(finalNorm) : null,
			finalAdaW: f('final_layer.adaLN_modulation.1.weight'),
			finalAdaB: f('final_layer.adaLN_modulation.1.bias'),
			finalW: f('final_layer.linear.weight'),
			finalB: f('final_layer.linear.bias'),
			xPad: m.has('x_pad_token') ? f('x_pad_token') : null,
			capPad: m.has('cap_pad_token') ? f('cap_pad_token') : null,
		};
		llm.print(`  DiT: ${this.nLayers} layers + ${this.nRefiner}/${this.nContext} refiners, dim ${this.dim}, loaded in ${llm.since(t0)}`);
	}

	unload() {
		for (const lw of [...this.layers, ...this.noiseRefiner, ...this.contextRefiner]) {
			for (const t of Object.values(lw)) if (t) t.dispose();
		}
		for (const t of Object.values(this.g)) if (t) t.dispose();
	}

	// Activations for one image size and the longest text it will see.
	prepare(latentH, latentW, maxText) {
		const dim = this.dim;
		this.latentH = latentH;
		this.latentW = latentW;
		this.patchesW = latentW / PATCH;
		this.nPatches = (latentH / PATCH) * this.patchesW;
		this.paddedPatches = padTo(this.nPatches);
		const maxSeq = this.paddedPatches + padTo(maxText);
		this.a = {
			hidden: f32(maxSeq * dim), bufA: f32(maxSeq * dim), bufB: f32(maxSeq * dim), bufC: f32(maxSeq * dim),
			q: f32(maxSeq * dim), k: f32(maxSeq * dim), v: f32(maxSeq * dim), attn: f32(maxSeq * dim),
			gate: f32(maxSeq * this.ffn), up: f32(maxSeq * this.ffn), down: f32(maxSeq * dim),
			patches: f32(this.nPatches * this.patchDim), velocity: f32(this.channels * latentH * latentW),
			latent: f32(this.channels * latentH * latentW),
			tEmb: f32(this.tDim), tMlp: f32(this.tMlp), tSilu: f32(this.tDim),
			ada: f32(4 * dim), finalScale: f32(dim),
		};
	}

	release() {
		if (this.a) for (const t of Object.values(this.a)) t.dispose();
		this.a = null;
	}

	// Text embeddings [n, textDim] (a Tensor) -> the conditioning the main layers
	// read: RMSNorm, cap_embedder, padded with cap_pad_token to a multiple of 32,
	// through the context refiner. Returns { tensor, textLen, paddedText, cos, sin }.
	encodeText(embeddings, textLen) {
		const dim = this.dim;
		const padded = padTo(textLen);
		const normed = f32(textLen * this.textDim);
		const out = f32(padded * dim);
		op.rmsNorm(embeddings, this.g.capNorm, normed, this.textDim, textLen, 1e-6);
		op.linearBias(this.g.capW, this.g.capB, normed, out, dim, this.textDim, textLen);
		if (this.g.capPad) op.fillRows(this.g.capPad, out, textLen, padded - textLen, dim);
		const ctx = ropeTables(this.nPatches, this.patchesW, textLen, 'text');
		const cos = upload(ctx.cos), sin = upload(ctx.sin);
		for (const lw of this.contextRefiner) this.layer(lw, out, padded, false, cos, sin);
		submit();
		cos.dispose(); sin.dispose(); normed.dispose();
		// The image tokens' axis-0 position is the padded text length + 1, so the
		// noise refiner's table belongs to the text too.
		const main = ropeTables(this.nPatches, this.patchesW, textLen, 'full');
		const img = ropeTables(this.nPatches, this.patchesW, textLen, 'image');
		return {
			tensor: out, textLen, paddedText: padded,
			cos: upload(main.cos), sin: upload(main.sin),
			imgCos: upload(img.cos), imgSin: upload(img.sin),
			dispose() { for (const t of [this.tensor, this.cos, this.sin, this.imgCos, this.imgSin]) t.dispose(); },
		};
	}

	// velocity for the latent in this.a.latent at `sigma`, into this.a.velocity.
	forward(text, sigma) {
		const a = this.a, g = this.g, dim = this.dim;
		const nP = this.nPatches, pP = this.paddedPatches;
		op.patchify(a.latent, a.patches, this.channels, this.latentH, this.latentW, PATCH);
		op.matmul(g.xw, a.patches, a.hidden, dim, this.patchDim, nP);
		op.biasAdd(a.hidden, g.xb, nP * dim, dim);
		if (g.xPad) op.fillRows(g.xPad, a.hidden, nP, pP - nP, dim);

		// The pipeline hands the model 1 - sigma, and the model scales by 1000.
		op.timestepEmbed(a.tEmb, this.tDim, (1 - sigma) * 1000);
		op.linearBias(g.t0w, g.t0b, a.tEmb, a.tMlp, this.tMlp, this.tDim, 1);
		op.silu(a.tMlp, this.tMlp);
		op.linearBias(g.t2w, g.t2b, a.tMlp, a.tEmb, this.tDim, this.tMlp, 1);

		for (const lw of this.noiseRefiner) this.layer(lw, a.hidden, pP, true, text.imgCos, text.imgSin);
		copy(text.tensor, a.hidden, text.paddedText * dim * 4, 0, pP * dim * 4);
		const seq = pP + text.paddedText;
		for (const lw of this.layers) {
			this.layer(lw, a.hidden, seq, true, text.cos, text.sin);
			submit();
		}

		// Final layer: SiLU(t) -> scale; LayerNorm (or the RMSNorm a checkpoint
		// may carry) of the real image rows; * (1 + scale); linear + bias.
		copy(a.tEmb, a.tSilu, this.tDim * 4);
		op.silu(a.tSilu, this.tDim);
		op.linearBias(g.finalAdaW, g.finalAdaB, a.tSilu, a.finalScale, dim, this.tDim, 1);
		if (g.finalNorm) op.rmsNorm(a.hidden, g.finalNorm, a.bufA, dim, nP, 1e-5);
		else op.layerNorm(a.hidden, a.bufA, dim, nP, 1e-5);
		op.scaleModulate(a.bufA, a.finalScale, a.bufA, nP * dim, dim, 0);
		op.matmul(g.finalW, a.bufA, a.patches, this.patchDim, dim, nP);
		op.biasAdd(a.patches, g.finalB, nP * this.patchDim, this.patchDim);
		op.unpatchify(a.patches, a.velocity, this.channels, this.latentH, this.latentW, PATCH);
		// The model predicts data -> noise; the sampler steps noise -> data.
		op.scale(a.velocity, this.channels * this.latentH * this.latentW, -1);
		submit();
	}

	// One Lumina2 block over `hidden` [seq, dim], in place.
	layer(lw, hidden, seq, adaln, cos, sin) {
		const a = this.a, dim = this.dim, n = seq * dim;
		if (adaln) op.linearBias(lw.adaW, lw.adaB, a.tEmb, a.ada, 4 * dim, this.tDim, 1);
		op.rmsNorm(hidden, lw.attnNorm1, a.bufA, dim, seq, 1e-5);
		if (adaln) op.scaleModulate(a.bufA, a.ada, a.bufA, n, dim, 0);
		op.matmul(lw.qkv, a.bufA, a.q, dim, dim, seq, 0, true);
		op.matmul(lw.qkv, a.bufA, a.k, dim, dim, seq, dim, true);
		op.matmul(lw.qkv, a.bufA, a.v, dim, dim, seq, 2 * dim);
		op.headNorm(a.q, lw.qNorm, this.heads, this.headDim, seq, 1e-5);
		op.headNorm(a.k, lw.kNorm, this.heads, this.headDim, seq, 1e-5);
		op.transposeHeads(a.q, a.bufC, seq, this.heads, this.headDim, 0, true);
		op.transposeHeads(a.k, a.bufB, seq, this.heads, this.headDim, 0, true);
		op.transposeHeads(a.v, a.bufA, seq, this.heads, this.headDim, 0);
		op.mrope(a.bufC, cos, sin, seq, this.heads, true);
		op.mrope(a.bufB, cos, sin, seq, this.heads);
		op.flashAttention(a.bufC, a.bufB, a.bufA, a.attn, this.heads, seq);
		op.matmul(lw.out, a.attn, a.bufA, dim, dim, seq);
		this.residual(lw.attnNorm2, a.bufA, hidden, seq, adaln, dim);

		op.rmsNorm(hidden, lw.ffnNorm1, a.bufA, dim, seq, 1e-5);
		if (adaln) op.scaleModulate(a.bufA, a.ada, a.bufA, n, dim, 2 * dim);
		op.matmul(lw.w1, a.bufA, a.gate, this.ffn, dim, seq, 0, true);
		op.matmul(lw.w3, a.bufA, a.up, this.ffn, dim, seq);
		op.siluMul(a.gate, a.up, seq * this.ffn);
		op.matmul(lw.w2, a.gate, a.down, dim, this.ffn, seq);
		this.residual(lw.ffnNorm2, a.down, hidden, seq, adaln, 3 * dim);
	}

	// hidden += [tanh(gate)] * norm(x), the gate at `gateAt` of the AdaLN
	// params; no norm weight is a plain residual.
	residual(normW, x, hidden, seq, adaln, gateAt) {
		const a = this.a, dim = this.dim, n = seq * dim;
		if (!normW) { op.add(hidden, x, n); return; }
		op.rmsNorm(x, normW, a.bufB, dim, seq, 1e-5);
		if (adaln) op.gatedAddTanh(hidden, a.ada, a.bufB, n, dim, gateAt);
		else op.add(hidden, a.bufB, n);
	}
}
