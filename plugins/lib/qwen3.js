// Qwen-family decoder as a text encoder: a prompt in, hidden states out.
//
// The weights are streamed a layer at a time - uploaded, run over the whole
// prompt in one batched pass, dropped - so an 8B encoder costs one layer of VRAM
// rather than nine gigabytes, which is what lets it share a card with a 9B DiT.
// The hidden states of chosen layers are written into slots of one
// [seq, layers * dim] tensor as they go past, which is the conditioning a
// diffusion model reads.

import { llm, f32 } from './llm.js';
import * as op from './ops.js';
import { breathe, copy } from './gpu.js';

// The architecture's numbers, from the GGUF metadata the way
// `lib/model/config.c3` reads them.
export function textConfig(model) {
	const arch = model.meta('general.architecture', 'qwen3');
	const m = (key, fallback) => model.meta(`${arch}.${key}`, fallback);
	const dim = m('embedding_length', 2560);
	const nHeads = m('attention.head_count', 32);
	return {
		arch,
		dim,
		nHeads,
		nKvHeads: m('attention.head_count_kv', nHeads),
		nLayers: m('block_count', 36),
		ffnDim: m('feed_forward_length', 6912),
		headDim: m('attention.key_length', dim / nHeads),
		ropeTheta: m('rope.freq_base', 1000000),
		eps: m('attention.layer_norm_rms_epsilon', 1e-6),
		// Qwen3-VL's MRoPE: how many frequency pairs follow the height and width
		// axes (interleaved t, h, w, t, h, w, ...); the rest follow time.
		ropeSections: model.metadata[`${arch}.rope.dimension_sections`] !== undefined
			? Array.from(model.array(`${arch}.rope.dimension_sections`))
			: null,
	};
}

// cos/sin [n, headDim / 2] for rotate-half RoPE. `positions` is [3 * n]
// (time, height, width a token) or null for 0, 1, 2, ... on every axis. With
// `sections`, pair j < 3 * sections[1] with j % 3 == 1 reads the height
// position and j < 3 * sections[2] with j % 3 == 2 the width; the rest time.
// Float32 steps as transformers takes them.
export function ropeTables(n, headDim, theta, positions = null, sections = null) {
	const half = headDim / 2;
	const inv = new Float32Array(half);
	for (let j = 0; j < half; j++) inv[j] = Math.fround(1 / Math.fround(Math.pow(theta, Math.fround((2 * j) / headDim))));
	const axis = new Uint8Array(half);
	if (sections) {
		for (let j = 0; j < half; j++) {
			if (j % 3 === 1 && j < 3 * sections[1]) axis[j] = 1;
			else if (j % 3 === 2 && j < 3 * sections[2]) axis[j] = 2;
		}
	}
	const cos = new Float32Array(n * half);
	const sin = new Float32Array(n * half);
	for (let t = 0; t < n; t++) {
		for (let j = 0; j < half; j++) {
			const pos = positions ? positions[3 * t + axis[j]] : t;
			const angle = Math.fround(pos * inv[j]);
			cos[t * half + j] = Math.cos(angle);
			sin[t * half + j] = Math.sin(angle);
		}
	}
	return { cos, sin };
}

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

export class TextEncoder {
	constructor(path) {
		this.model = llm.open(path);
		this.config = textConfig(this.model);
		this.tokenizer = this.model.tokenizer();
		if (!this.model.has('blk.0.attn_q_norm.weight')) {
			// Qwen2 has biases and no per-head norm; this encoder is Qwen3's graph.
			llm.print(`  note: ${this.config.arch} has no per-head Q/K norm; encoding without it`);
		}
	}

	// Run `tokens` through the first `upTo` layers and return a tensor
	// [n, layers.length * dim] holding, per token, the hidden state after each
	// layer in `layers` (0-indexed), side by side.
	//
	// For a prompt with images (Qwen3-VL), `options` carries:
	//   positions  [3 * n] time/height/width positions (see ropeTables)
	//   inject     [{ at, rows, tensor }]: rows written over the token
	//              embeddings from row `at` - the vision encoder's output in
	//              the image-pad tokens' places
	//   deepstack  [{ at, rows, tensors }]: tensors[l] added to those rows after
	//              layer l
	async encodeLayers(tokens, layers, options = {}) {
		const c = this.config;
		const n = tokens.length;
		const dim = c.dim;
		const qDim = c.nHeads * c.headDim;
		const kvDim = c.nKvHeads * c.headDim;
		const upTo = Math.max(...layers) + 1;
		const width = layers.length * dim;

		const hidden = f32(n * dim);
		const norm = f32(n * dim);
		const q = f32(n * qDim);
		const k = f32(n * kvDim);
		const v = f32(n * kvDim);
		const attn = f32(n * qDim);
		const gate = f32(n * c.ffnDim);
		const up = f32(n * c.ffnDim);
		const down = f32(n * dim);
		const out = f32(n * width);

		this.model.embedRows('token_embd.weight', tokens, hidden);
		for (const { at, rows, tensor } of options.inject ?? []) copy(tensor, hidden, rows * dim * 4, 0, at * dim * 4);
		const rope = ropeTables(n, c.headDim, c.ropeTheta, options.positions ?? null, c.ropeSections);
		const cos = upload(rope.cos), sin = upload(rope.sin);
		const deep = f32(Math.max(0, ...(options.deepstack ?? []).map((d) => d.rows)) * dim || 1);

		let loadMs = 0;
		let runMs = 0;
		for (let l = 0; l < upTo; l++) {
			const t0 = llm.now();
			const w = this.layer(l);
			const t1 = llm.now();
			loadMs += t1 - t0;

			op.rmsNorm(hidden, w.attnNorm, norm, dim, n, c.eps);
			op.matmul(w.q, norm, q, qDim, dim, n, 0, true);
			op.matmul(w.k, norm, k, kvDim, dim, n, 0, true);
			op.matmul(w.v, norm, v, kvDim, dim, n);
			if (w.qNorm) {
				op.headNormSmall(q, w.qNorm, c.nHeads, c.headDim, n, c.eps, true);
				op.headNormSmall(k, w.kNorm, c.nKvHeads, c.headDim, n, c.eps);
			}
			op.ropeNeox(q, cos, sin, c.headDim, c.nHeads, n, true);
			op.ropeNeox(k, cos, sin, c.headDim, c.nKvHeads, n);
			op.attentionCausal(q, k, v, attn, c.headDim, c.nKvHeads, c.nHeads, n);
			op.matmul(w.o, attn, norm, dim, qDim, n);
			op.add(hidden, norm, n * dim);

			op.rmsNorm(hidden, w.ffnNorm, norm, dim, n, c.eps);
			op.matmul(w.gate, norm, gate, c.ffnDim, dim, n, 0, true);
			op.matmul(w.up, norm, up, c.ffnDim, dim, n);
			op.siluMul(gate, up, n * c.ffnDim);
			op.matmul(w.down, gate, down, dim, c.ffnDim, n, 0, false, true); // SwiGLU output: can pass 65504
			op.add(hidden, down, n * dim);
			for (const { at, rows, tensors } of options.deepstack ?? []) {
				if (l >= tensors.length) continue;
				copy(hidden, deep, rows * dim * 4, at * dim * 4);
				op.add(deep, tensors[l], rows * dim);
				copy(deep, hidden, rows * dim * 4, 0, at * dim * 4);
			}

			const slot = layers.indexOf(l);
			if (slot >= 0) op.copyRows(hidden, out, n, dim, width, slot * dim);
			await breathe(true);
			for (const t of Object.values(w)) if (t) t.dispose();
			runMs += llm.now() - t1;
		}
		llm.print(`  [text] ${upTo} layers x ${n} tokens: weight upload ${(loadMs / 1000).toFixed(2)}s, forward ${(runMs / 1000).toFixed(2)}s`);

		for (const t of [hidden, norm, q, k, v, attn, gate, up, down, cos, sin, deep]) t.dispose();
		return out;
	}

	layer(l) {
		const m = this.model;
		const name = (s) => `blk.${l}.${s}`;
		const has = (s) => m.has(name(s));
		return {
			attnNorm: m.upload(name('attn_norm.weight'), 'f32'),
			q: m.upload(name('attn_q.weight'), 'auto'),
			k: m.upload(name('attn_k.weight'), 'auto'),
			v: m.upload(name('attn_v.weight'), 'auto'),
			o: m.upload(name('attn_output.weight'), 'auto'),
			qNorm: has('attn_q_norm.weight') ? m.upload(name('attn_q_norm.weight'), 'f32') : null,
			kNorm: has('attn_k_norm.weight') ? m.upload(name('attn_k_norm.weight'), 'f32') : null,
			ffnNorm: m.upload(name('ffn_norm.weight'), 'f32'),
			gate: m.upload(name('ffn_gate.weight'), 'auto'),
			up: m.upload(name('ffn_up.weight'), 'auto'),
			down: m.upload(name('ffn_down.weight'), 'auto'),
		};
	}

	close() {
		this.model.close();
	}
}
