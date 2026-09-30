// T5's encoder (T5 1.1 / FLAN-T5: gated-GELU feed-forward, untied head) and its
// SentencePiece unigram tokenizer, from the files Hugging Face ships: a folder
// with model.safetensors and tokenizer.json (google/flan-t5-base, say).
//
//   const t5 = new T5Encoder(folder);
//   const states = t5.encode(['a cat', 'a dog']);   // [tokens * d_model] each
//   const vectors = t5.pooled(['a cat']);           // mean over the tokens
//
// Strings are run as one padded batch: every sequence attends only its own
// tokens (the padding keys carry -1e30 in the bias, next to T5's bucketed
// relative-position bias), so a batch gives what each string alone would.

import { llm, f32 } from './llm.js';
import * as op from './ops.js';
import { submit } from './gpu.js';

const F = 4;

// The mean of token states [tokens * width].
export function meanPool(states, width) {
	const n = states.length / width, v = new Float32Array(width);
	for (let t = 0; t < n; t++) for (let d = 0; d < width; d++) v[d] += states[t * width + d];
	for (let d = 0; d < width; d++) v[d] /= n;
	return v;
}

// ------------------------------------------------------------- tokenizer

export class T5Tokenizer {
	constructor(path) {
		const json = JSON.parse(llm.readText(path));
		const model = json.model;
		if (model?.type !== 'Unigram') throw new Error(`${path}: a ${model?.type} tokenizer, not T5's unigram one`);
		this.pieces = new Map();
		this.maxLength = 0;
		let minScore = 0;
		model.vocab.forEach(([piece, score], id) => {
			this.pieces.set(piece, { id, score });
			this.maxLength = Math.max(this.maxLength, piece.length);
			minScore = Math.min(minScore, score);
		});
		this.unk = model.unk_id ?? 2;
		this.unkScore = minScore - 10;
		this.eos = this.pieces.get('</s>')?.id ?? 1;
		this.pad = this.pieces.get('<pad>')?.id ?? 0;
	}

	// SentencePiece's nmt_nfkc for plain text: NFKC, runs of white space as one
	// space, none at the ends; then '▁' for each space and one in front.
	normalize(text) {
		const t = text.normalize('NFKC').replace(/\s+/g, ' ').trim();
		return t.length ? '▁' + t.replace(/ /g, '▁') : '';
	}

	// The ids of `text`, best-scoring segmentation (Viterbi), then </s>.
	encode(text) {
		const s = [...this.normalize(text)];
		const n = s.length;
		const best = new Float64Array(n + 1).fill(-Infinity);
		const from = new Int32Array(n + 1).fill(-1);
		const ids = new Int32Array(n + 1).fill(-1);
		best[0] = 0;
		for (let i = 0; i < n; i++) {
			if (best[i] === -Infinity) continue;
			let piece = '';
			let matched = false;
			for (let len = 1; len <= this.maxLength && i + len <= n; len++) {
				piece += s[i + len - 1];
				const p = this.pieces.get(piece);
				if (!p || p.id === this.unk) continue;
				if (len === 1) matched = true;
				const score = best[i] + p.score;
				if (score > best[i + len]) {
					best[i + len] = score;
					from[i + len] = i;
					ids[i + len] = p.id;
				}
			}
			if (!matched) {
				const score = best[i] + this.unkScore;
				if (score > best[i + 1]) {
					best[i + 1] = score;
					from[i + 1] = i;
					ids[i + 1] = this.unk;
				}
			}
		}
		const out = [];
		for (let at = n; at > 0; at = from[at]) out.push(ids[at]);
		out.reverse();
		// SentencePiece keeps a run of unknown characters as one <unk>.
		const merged = out.filter((id, k) => !(id === this.unk && out[k - 1] === this.unk));
		merged.push(this.eos);
		return merged;
	}
}

// --------------------------------------------------------------- encoder

// T5's relative position bucket for key - query, bidirectional.
function bucket(rel, numBuckets, maxDistance) {
	const half = numBuckets / 2;
	let ret = rel > 0 ? half : 0;
	const n = Math.abs(rel);
	const exact = half / 2;
	if (n < exact) return ret + n;
	const large = exact + Math.floor(Math.log(n / exact) / Math.log(maxDistance / exact) * (half - exact));
	return ret + Math.min(large, half - 1);
}

export class T5Encoder {
	// `dir` holds model.safetensors and tokenizer.json; or give both paths.
	constructor(dir, { model = `${dir}/model.safetensors`, tokenizer = `${dir}/tokenizer.json`, maxRows = 4096 } = {}) {
		const t0 = llm.now();
		const m = llm.open(model);
		const pre = 'encoder.block.';
		if (!m.has(`${pre}0.layer.1.DenseReluDense.wi_0.weight`)) {
			m.close();
			throw new Error(`${model} is not a gated-GELU T5 encoder (T5 1.1 / FLAN-T5)`);
		}
		this.tokenizer = new T5Tokenizer(tokenizer);
		const [dModel, vocab] = m.shape('shared.weight');
		this.dModel = dModel;
		this.dFF = m.shape(`${pre}0.layer.1.DenseReluDense.wi_0.weight`)[1];
		const [heads, buckets] = m.shape(`${pre}0.layer.0.SelfAttention.relative_attention_bias.weight`); // torch [buckets, heads]
		this.heads = heads;
		this.buckets = buckets;
		this.maxDistance = 128;
		this.inner = m.shape(`${pre}0.layer.0.SelfAttention.q.weight`)[1];
		this.headDim = this.inner / heads;
		this.embedding = m.floats('shared.weight');
		this.vocab = vocab;
		this.relBias = m.floats(`${pre}0.layer.0.SelfAttention.relative_attention_bias.weight`); // [bucket][head]
		const up = (name) => m.upload(name, 'f32');
		this.layers = [];
		for (let l = 0; m.has(`${pre}${l}.layer.0.SelfAttention.q.weight`); l++) {
			const a = `${pre}${l}.layer.0.`, f = `${pre}${l}.layer.1.`;
			this.layers.push({
				ln0: up(a + 'layer_norm.weight'),
				q: up(a + 'SelfAttention.q.weight'), k: up(a + 'SelfAttention.k.weight'),
				v: up(a + 'SelfAttention.v.weight'), o: up(a + 'SelfAttention.o.weight'),
				ln1: up(f + 'layer_norm.weight'),
				wi0: up(f + 'DenseReluDense.wi_0.weight'), wi1: up(f + 'DenseReluDense.wi_1.weight'),
				wo: up(f + 'DenseReluDense.wo.weight'),
			});
		}
		this.finalNorm = up('encoder.final_layer_norm.weight');
		m.close();
		this.maxRows = maxRows;
		llm.print(`  [t5] ${this.layers.length} layers, d_model ${dModel}, resident in ${llm.since(t0)}`);
	}

	// The last hidden state of each string: a Float32Array of tokens * d_model.
	encode(texts) {
		const ids = texts.map((t) => this.tokenizer.encode(t));
		const out = new Array(texts.length);
		// Batches of strings of one padded length, at most maxRows rows each.
		const order = ids.map((_, i) => i).sort((a, b) => ids[a].length - ids[b].length);
		for (let s = 0; s < order.length;) {
			const L0 = ids[order[s]].length;
			let e = s, L = L0;
			while (e < order.length) {
				const L2 = Math.max(L, ids[order[e]].length);
				if ((e - s + 1) * L2 > this.maxRows && e > s) break;
				L = L2;
				e++;
			}
			const batch = order.slice(s, e);
			const states = this.run(batch.map((i) => ids[i]), L);
			batch.forEach((i, b) => { out[i] = states[b]; });
			s = e;
		}
		return out;
	}

	// Mean of each string's token states (its </s> included): what UniMate
	// conditions on.
	pooled(texts) {
		return this.encode(texts).map((st) => meanPool(st, this.dModel));
	}

	run(seqs, L) {
		const B = seqs.length, D = this.dModel, H = this.heads, I = this.inner, FF = this.dFF, rows = B * L;
		const x = new Float32Array(rows * D);
		seqs.forEach((s, b) => s.forEach((id, t) => x.set(this.embedding.subarray(id * D, (id + 1) * D), (b * L + t) * D)));
		// Position bias, shared by every layer, plus the padding mask.
		const bias = new Float32Array(B * H * L * L);
		const rel = new Float32Array(2 * L - 1 + 1);
		for (let h = 0; h < H; h++) {
			for (let r = -(L - 1); r < L; r++) rel[r + L - 1] = this.relBias[bucket(r, this.buckets, this.maxDistance) * H + h];
			for (let b = 0; b < B; b++) {
				const len = seqs[b].length;
				for (let i = 0; i < L; i++) {
					const at = ((b * H + h) * L + i) * L;
					for (let j = 0; j < L; j++) bias[at + j] = j < len ? rel[j - i + L - 1] : -1e30;
				}
			}
		}
		const g = {
			x: f32(rows * D), h: f32(rows * D), q: f32(rows * I), k: f32(rows * I), v: f32(rows * I),
			att: f32(rows * I), o: f32(rows * D), a: f32(rows * FF), b: f32(rows * FF), bias: f32(bias.length),
		};
		g.x.buffer.write(x);
		g.bias.buffer.write(bias);
		const shape = { lq: L, lk: L, sequences: B, qBs: L, qIs: 1, kvBs: L, kvIs: 1, scale: 1 };
		for (const l of this.layers) {
			op.rmsNorm(g.x, l.ln0, g.h, D, rows, 1e-6);
			op.matmul(l.q, g.h, g.q, I, D, rows, 0, true);
			op.matmul(l.k, g.h, g.k, I, D, rows, 0, true);
			op.matmul(l.v, g.h, g.v, I, D, rows);
			op.attentionRows(g.q, g.k, g.v, g.att, H, this.headDim, shape, g.bias, H * L * L);
			op.matmul(l.o, g.att, g.o, D, I, rows);
			op.add(g.x, g.o, rows * D);
			op.rmsNorm(g.x, l.ln1, g.h, D, rows, 1e-6);
			op.matmul(l.wi0, g.h, g.a, FF, D, rows, 0, true);
			op.matmul(l.wi1, g.h, g.b, FF, D, rows);
			op.gelu(g.a, rows * FF);
			op.mul(g.a, g.b, rows * FF);
			op.matmul(l.wo, g.a, g.o, D, FF, rows);
			op.add(g.x, g.o, rows * D);
		}
		op.rmsNorm(g.x, this.finalNorm, g.h, D, rows, 1e-6);
		submit();
		const all = new Float32Array(g.h.buffer.readBytes(0, rows * D * F).buffer);
		for (const t of Object.values(g)) t.dispose();
		return seqs.map((s, b) => all.slice(b * L * D, (b * L + s.length) * D));
	}

	close() {
		for (const l of this.layers) for (const t of Object.values(l)) t.dispose();
		this.finalNorm.dispose();
		this.layers = [];
		this.embedding = null;
	}
}
