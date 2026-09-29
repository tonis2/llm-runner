// Kimodo's text encoder: LLM2Vec over Llama-3-8B-Instruct, a prompt in and one
// 4096-wide embedding out, the conditioning the motion denoiser reads.
//
// The Llama-3 graph with three differences from a chat model: attention is
// bidirectional, every projection carries the supervised adapter as a rank-16
// F32 LoRA branch beside its (MNTP-merged) Q8_0 base - y = W x + 2 B (A x), the
// 2 folded into B at upload - and the final-normed hidden states are averaged
// over every token but BOS. The weights are Kimodo's GGUF (`kimodo-llm2vec`,
// as kimodo.cpp packs it), streamed a layer at a time like `lib/qwen3.js`.

import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { breathe, submit } from '../lib/gpu.js';
import { ropeTables } from '../lib/qwen3.js';
import { KimodoTokenizer } from './tokenizer.js';

const EPS = 1e-5;
const MAX_TOKENS = 512;
const PROJECTIONS = ['attn_q_proj', 'attn_k_proj', 'attn_v_proj', 'attn_o_proj', 'ffn_gate_proj', 'ffn_up_proj', 'ffn_down_proj'];

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

// `tokenizerPath` defaults to `tokenizer.gguf` beside the model, where
// kimodo.cpp looks for it.
export class KimodoTextEncoder {
	constructor(path, tokenizerPath = null) {
		const m = (this.model = llm.open(path));
		if (m.meta('general.architecture') !== 'kimodo-llm2vec') {
			m.close();
			throw new Error(`${path} is not a Kimodo LLM2Vec text encoder GGUF (general.architecture = ${m.meta('general.architecture')})`);
		}
		const dim = m.meta('kimodo.hidden_size', 4096);
		const nHeads = m.meta('kimodo.heads', 32);
		this.config = {
			dim,
			nHeads,
			nKvHeads: m.meta('kimodo.key_value_heads', 8),
			headDim: dim / nHeads,
			nLayers: m.meta('kimodo.layer_count', 32),
			ropeTheta: m.meta('kimodo.rope_theta', 500000),
			ffnDim: m.shape('layer.00.ffn_gate_proj_base.weight')[1],
			rank: m.shape('layer.00.attn_q_proj_lora_a.weight')[1],
		};
		this.tokenizer = new KimodoTokenizer(tokenizerPath ?? path.replace(/[^/]*$/, 'tokenizer.gguf'));
	}

	// The prompt's 4096 floats.
	async encode(prompt) {
		const tokens = this.tokenizer.encode(prompt);
		if (tokens.length < 2 || tokens.length > MAX_TOKENS) throw new Error(`a Kimodo prompt is 1 to ${MAX_TOKENS - 1} tokens (this one is ${tokens.length - 1})`);
		const c = this.config;
		const n = tokens.length;
		const dim = c.dim;
		const kvDim = c.nKvHeads * c.headDim;

		const hidden = f32(n * dim);
		const norm = f32(n * dim);
		const q = f32(n * dim);
		const k = f32(n * kvDim);
		const v = f32(n * kvDim);
		const attn = f32(n * dim);
		const gate = f32(n * c.ffnDim);
		const up = f32(n * c.ffnDim);
		const lowRank = [f32(n * c.rank), f32(n * c.rank), f32(n * c.rank)];
		const branch = f32(n * c.ffnDim);

		// y = W x + B' (A x) for projection `p` of layer weights `w`; `slot`
		// picks a low-rank scratch so Q, K and V can be recorded back to back.
		const project = (w, p, x, y, out, inDim, slot = 0, exact = false) => {
			op.matmul(w[p].base, x, y, out, inDim, n, 0, false, exact);
			op.matmul(w[p].a, x, lowRank[slot], c.rank, inDim, n);
			op.matmul(w[p].b, lowRank[slot], branch, out, c.rank, n);
			op.add(y, branch, n * out);
		};

		this.model.embedRows('token_embedding.weight', tokens, hidden);
		const rope = ropeTables(n, c.headDim, c.ropeTheta);
		const cos = upload(rope.cos), sin = upload(rope.sin);

		const t0 = llm.now();
		let loadMs = 0;
		for (let l = 0; l < c.nLayers; l++) {
			const t1 = llm.now();
			const w = this.layer(l);
			loadMs += llm.now() - t1;

			op.rmsNorm(hidden, w.attnNorm, norm, dim, n, EPS);
			project(w, 'attn_q_proj', norm, q, dim, dim, 0);
			project(w, 'attn_k_proj', norm, k, kvDim, dim, 1);
			project(w, 'attn_v_proj', norm, v, kvDim, dim, 2);
			op.ropeNeox(q, cos, sin, c.headDim, c.nHeads, n, true);
			op.ropeNeox(k, cos, sin, c.headDim, c.nKvHeads, n);
			op.attentionGqa(q, k, v, attn, c.headDim, c.nKvHeads, c.nHeads, n, false);
			project(w, 'attn_o_proj', attn, norm, dim, dim);
			op.add(hidden, norm, n * dim);

			op.rmsNorm(hidden, w.ffnNorm, norm, dim, n, EPS);
			project(w, 'ffn_gate_proj', norm, gate, c.ffnDim, dim, 0);
			project(w, 'ffn_up_proj', norm, up, c.ffnDim, dim, 1);
			op.siluMul(gate, up, n * c.ffnDim);
			project(w, 'ffn_down_proj', gate, norm, dim, c.ffnDim, 0, true); // SwiGLU output: can pass 65504
			op.add(hidden, norm, n * dim);

			await breathe(true);
			disposeLayer(w);
		}

		const finalNorm = this.model.upload('final_norm.weight', 'f32');
		op.rmsNorm(hidden, finalNorm, norm, dim, n, EPS);
		submit();
		const states = new Float32Array(norm.buffer.readBytes(0, n * dim * 4).buffer);
		llm.print(`  [kimodo text] ${n} tokens x ${c.nLayers} layers: weight upload ${(loadMs / 1000).toFixed(2)}s, total ${llm.since(t0)}`);

		for (const t of [hidden, norm, q, k, v, attn, gate, up, branch, cos, sin, finalNorm, ...lowRank]) t.dispose();

		const pooled = new Float32Array(dim);
		for (let t = 1; t < n; t++) for (let d = 0; d < dim; d++) pooled[d] += states[t * dim + d];
		for (let d = 0; d < dim; d++) pooled[d] /= n - 1;
		return pooled;
	}

	layer(l) {
		const m = this.model;
		const name = (s) => `layer.${String(l).padStart(2, '0')}.${s}.weight`;
		const w = {
			attnNorm: m.upload(name('attn_norm'), 'f32'),
			ffnNorm: m.upload(name('ffn_norm'), 'f32'),
		};
		for (const p of PROJECTIONS) {
			const b = m.floats(name(`${p}_lora_b`));
			for (let i = 0; i < b.length; i++) b[i] *= 2;
			w[p] = { base: m.upload(name(`${p}_base`), 'auto'), a: m.upload(name(`${p}_lora_a`), 'f32'), b: upload(b) };
		}
		return w;
	}

	close() {
		this.model.close();
	}
}

function disposeLayer(w) {
	for (const t of Object.values(w)) {
		if (t.dispose) t.dispose();
		else for (const u of Object.values(t)) u.dispose();
	}
}
