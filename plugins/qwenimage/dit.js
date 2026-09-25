// The Qwen-Image 2.1 DiT: 32 single-stream blocks over one sequence of text
// and image tokens, with one modulation shared by every block.
//
// A block is scale-only AdaLN round attention and a SwiGLU MLP, each with a
// tanh-gated residual; attention is block-causal - text sees what came before
// it, an image block also sees itself whole, and the target image (last) sees
// everything. Text and reference-image tokens are modulated from t = 0, so
// nothing before the target depends on the step: their keys and values are
// worked out once per prompt (`encodePrefix`) and each step runs only the
// target's tokens against them (`forward`).
//
// Ported from diffusers' `transformer_qwenimage21.py`.

import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit, copy, breathe, uploadEach } from '../lib/gpu.js';

const THETA = 10000;
const AXES = [16, 56, 56];
const EPS = 1e-6;

// cos/sin [rows, 64] for the 3-axis rope of a sequence of segments, each
// { kind: 'text', n } or { kind: 'image', h, w }. Text advances one position
// on all three axes a token; an image block holds the first axis at the
// position the text reached and centres its rows and columns on zero, then
// moves the position on by its longer side.
export function ropeTable(segments) {
	const freqs = AXES.map((ax) => {
		const f = new Float32Array(ax / 2);
		for (let k = 0; k < ax / 2; k++) f[k] = Math.fround(1 / Math.fround(Math.pow(THETA, Math.fround((2 * k) / ax))));
		return f;
	});
	let rows = 0;
	for (const s of segments) rows += s.kind === 'text' ? s.n : s.h * s.w;
	const cos = new Float32Array(rows * 64);
	const sin = new Float32Array(rows * 64);
	let at = 0;
	const put = (p) => {
		let pair = 0;
		for (let a = 0; a < 3; a++) {
			for (let k = 0; k < freqs[a].length; k++) {
				const angle = Math.fround(p[a] * freqs[a][k]);
				cos[at * 64 + pair] = Math.cos(angle);
				sin[at * 64 + pair] = Math.sin(angle);
				pair++;
			}
		}
		at++;
	};
	let pos = 0;
	for (const s of segments) {
		if (s.kind === 'text') {
			for (let i = 0; i < s.n; i++, pos++) put([pos, pos, pos]);
		} else {
			const top = s.h - (s.h >> 1), left = s.w - (s.w >> 1);
			for (let r = 0; r < s.h; r++) for (let c = 0; c < s.w; c++) put([pos, r - top, c - left]);
			pos += Math.max(s.h, s.w);
		}
	}
	return { cos, sin, rows };
}

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

function disposeAll(obj) {
	for (const t of Object.values(obj)) if (t) t.dispose();
}

export class QwenImageDiT {
	constructor(model) {
		const m = model;
		this.model = m;
		this.p = m.has('model.diffusion_model.img_in.weight') ? 'model.diffusion_model.' : '';
		const shape = (name) => m.shape(this.p + name);
		this.channels = shape('img_in.weight')[0];
		this.dim = shape('img_in.weight')[1];
		this.headDim = shape('transformer_blocks.0.attn.norm_q.weight')[0];
		this.heads = this.dim / this.headDim;
		this.ffn = shape('transformer_blocks.0.img_mlp.gate_up.weight')[1] / 2;
		this.textDim = shape('txt_in.in_layer.weight')[0];
		this.tDim = shape('time_text_embed.timestep_embedder.linear_1.weight')[0];
		let n = 0;
		while (m.has(`${this.p}transformer_blocks.${n}.attn.to_q.weight`)) n++;
		this.nLayers = n;
	}

	async load() {
		const m = this.model, p = this.p;
		const t0 = llm.now();
		const up = (name) => m.upload(p + name, 'auto');
		const f = (name) => m.upload(p + name, 'f32');
		this.blocks = [];
		for (let l = 0; l < this.nLayers; l++) {
			const b = `transformer_blocks.${l}.`;
			this.blocks.push(await uploadEach({
				q: () => up(`${b}attn.to_q.weight`),
				k: () => up(`${b}attn.to_k.weight`),
				v: () => up(`${b}attn.to_v.weight`),
				out: () => up(`${b}attn.to_out.0.weight`),
				qNorm: () => f(`${b}attn.norm_q.weight`),
				kNorm: () => f(`${b}attn.norm_k.weight`),
				gateUp: () => up(`${b}img_mlp.gate_up.weight`),
				down: () => up(`${b}img_mlp.out.weight`),
			}));
		}
		// The text norm's weight is stored as scale - 1.
		const textNorm = m.floats(p + 'txt_in.text_norm.weight');
		for (let i = 0; i < textNorm.length; i++) textNorm[i] += 1;
		this.g = {
			imgIn: up('img_in.weight'),
			textNorm: upload(textNorm),
			textIn: up('txt_in.in_layer.weight'),
			textOut: up('txt_in.out_layer.weight'),
			t1: up('time_text_embed.timestep_embedder.linear_1.weight'),
			t2: up('time_text_embed.timestep_embedder.linear_2.weight'),
			mod: up('modulation.1.weight'),
			finalMod: up('norm_out.linear.weight'),
			projOut: up('proj_out.weight'),
		};
		llm.print(`  DiT: ${this.nLayers} blocks, dim ${this.dim}, ${this.channels} latent channels, loaded in ${llm.since(t0)}`);
	}

	unload() {
		for (const b of this.blocks ?? []) disposeAll(b);
		if (this.g) disposeAll(this.g);
		this.blocks = null;
		this.g = null;
	}

	// Activations for `seq` rows of one block.
	scratch(seq) {
		const d = seq * this.dim;
		return {
			bufA: f32(d), bufB: f32(d), q: f32(d), k: f32(d), v: f32(d), attn: f32(d),
			gate: f32(seq * this.ffn), up: f32(seq * this.ffn),
		};
	}

	// The target's activations, for one latent size.
	prepare(latentH, latentW) {
		const T = latentH * latentW, d = T * this.dim;
		this.latentH = latentH;
		this.latentW = latentW;
		this.tokens = T;
		this.s = this.scratch(T);
		this.a = {
			hidden: f32(d), kT: f32(d), vT: f32(d),
			patches: f32(T * this.channels),
			latent: f32(T * this.channels), velocity: f32(T * this.channels),
			tSin: f32(this.tDim), tH: f32(this.dim), temb: f32(this.dim),
			mod: f32(4 * this.dim), finalScale: f32(this.dim),
		};
	}

	release() {
		if (this.s) disposeAll(this.s);
		if (this.a) disposeAll(this.a);
		this.s = null;
		this.a = null;
	}

	// The shared modulation at `sigma` into `mod` [4 * dim]: scale and gate for
	// attention, then for the MLP. With `finalScale`, the output norm's scale too.
	modulation(sigma, mod, finalScale = null) {
		const a = this.a, g = this.g, D = this.dim;
		op.timestepEmbed(a.tSin, this.tDim, sigma * 1000);
		op.matmul(g.t1, a.tSin, a.tH, D, this.tDim, 1);
		op.silu(a.tH, D);
		op.matmul(g.t2, a.tH, a.temb, D, D, 1);
		// Everything downstream reads SiLU(temb).
		op.silu(a.temb, D);
		op.matmul(g.mod, a.temb, mod, 4 * D, D, 1);
		if (finalScale) op.matmul(g.finalMod, a.temb, finalScale, D, D, 1);
	}

	// Keys and values of everything before the target, once per prompt.
	//
	// `cond` is { tensor [n, textDim], n, slots }: the text encoder's rows, with
	// `slots` ([{ at, h, w }], in order) the text positions each reference
	// image's tokens go in at. `refs` are those images' latents [C, h, w]. Returns
	// the per-block caches and the rope for the target, which must follow
	// `prepare`.
	async encodePrefix(cond, refs = []) {
		const D = this.dim, g = this.g;
		const slots = cond.slots ?? [];
		if (slots.length !== refs.length) throw new Error(`the prompt has ${slots.length} image slots; ${refs.length} reference latents came`);

		// The sequence: text runs with the reference blocks between them, then
		// the target.
		const segments = [];
		let text = 0;
		slots.forEach((s, i) => {
			if (s.at > text) segments.push({ kind: 'text', n: s.at - text, from: text });
			segments.push({ kind: 'image', h: refs[i].h, w: refs[i].w, ref: i });
			text = s.at;
		});
		if (cond.n > text) segments.push({ kind: 'text', n: cond.n - text, from: text });
		const P = segments.reduce((sum, s) => sum + (s.kind === 'text' ? s.n : s.h * s.w), 0);
		const rope = ropeTable([...segments, { kind: 'image', h: this.latentH, w: this.latentW }]);

		// Each row's key limit: a text row sees up to itself, an image row its block.
		const limits = new Uint32Array(P);
		{
			let at = 0;
			for (const s of segments) {
				const len = s.kind === 'text' ? s.n : s.h * s.w;
				for (let i = 0; i < len; i++) limits[at + i] = s.kind === 'text' ? at + i + 1 : at + len;
				at += len;
			}
		}

		const t0 = llm.now();
		const hidden = f32(P * D);
		const sc = this.scratch(P);
		const limitBuf = f32(P);
		limitBuf.buffer.write(limits);
		const cos = upload(rope.cos.subarray(0, P * 64)), sin = upload(rope.sin.subarray(0, P * 64));

		// txt_in over all the text rows, then each run to its place.
		const tn = f32(cond.n * this.textDim);
		const tt = f32(cond.n * D);
		op.rmsNorm(cond.tensor, g.textNorm, tn, this.textDim, cond.n, EPS);
		op.matmul(g.textIn, tn, tt, D, this.textDim, cond.n);
		op.gelu(tt, cond.n * D);
		op.matmul(g.textOut, tt, sc.bufA, D, D, cond.n);
		let at = 0;
		for (const s of segments) {
			if (s.kind === 'text') {
				copy(sc.bufA, hidden, s.n * D * 4, s.from * D * 4, at * D * 4);
				at += s.n;
			} else {
				const n = s.h * s.w;
				const lat = upload(refs[s.ref].data);
				const patches = f32(n * this.channels);
				op.patchify(lat, patches, this.channels, s.h, s.w, 1);
				op.matmul(g.imgIn, patches, sc.bufB, D, this.channels, n);
				copy(sc.bufB, hidden, n * D * 4, 0, at * D * 4);
				submit();
				lat.dispose();
				patches.dispose();
				at += n;
			}
		}
		submit();
		tn.dispose();
		tt.dispose();

		const mod0 = f32(4 * D);
		this.modulation(0, mod0);
		const caches = [];
		for (let l = 0; l < this.nLayers; l++) {
			const kv = { k: f32(P * D), v: f32(P * D) };
			caches.push(kv);
			// The last block's own output is never read: only its keys and values.
			const attend = l < this.nLayers - 1
				? (q, out) => op.flashAttentionSplit(q, null, null, kv.k, kv.v, out, this.heads, P, 0, P, limitBuf)
				: null;
			this.block(this.blocks[l], sc, hidden, P, mod0, cos, sin, kv, attend);
			await breathe(true);
		}
		submit();
		for (const t of [hidden, limitBuf, cos, sin, mod0]) t.dispose();
		disposeAll(sc);
		llm.print(`  prefix: ${P} tokens (${segments.length} segments) through ${this.nLayers} blocks in ${llm.since(t0)}`);
		return {
			length: P,
			caches,
			cos: upload(rope.cos.subarray(P * 64)),
			sin: upload(rope.sin.subarray(P * 64)),
			dispose() {
				for (const kv of this.caches) { kv.k.dispose(); kv.v.dispose(); }
				this.cos.dispose();
				this.sin.dispose();
			},
		};
	}

	// The velocity for the latent in this.a.latent at `sigma`, into this.a.velocity.
	async forward(prefix, sigma) {
		const a = this.a, g = this.g, D = this.dim, T = this.tokens, C = this.channels;
		op.patchify(a.latent, a.patches, C, this.latentH, this.latentW, 1);
		op.matmul(g.imgIn, a.patches, a.hidden, D, C, T);
		this.modulation(sigma, a.mod, a.finalScale);
		for (let l = 0; l < this.nLayers; l++) {
			const cache = prefix.caches[l];
			const attend = (q, out) => op.flashAttentionSplit(q, cache.k, cache.v, a.kT, a.vT, out, this.heads, T, prefix.length, T);
			this.block(this.blocks[l], this.s, a.hidden, T, a.mod, prefix.cos, prefix.sin, { k: a.kT, v: a.vT }, attend);
			await breathe(true);
		}
		op.layerNorm(a.hidden, this.s.bufA, D, T, EPS);
		op.scaleModulate(this.s.bufA, a.finalScale, this.s.bufA, T * D, D, 0);
		op.matmul(g.projOut, this.s.bufA, a.patches, C, D, T);
		op.unpatchify(a.patches, a.velocity, C, this.latentH, this.latentW, 1);
		submit();
	}

	// One block over `hidden` [seq, dim], in place. Its keys and values (after
	// rope) are left in kv.k and kv.v as [heads, seq, headDim]; `attend(q, out)`
	// runs the attention, and without it the block stops there.
	block(lw, s, hidden, seq, mod, cos, sin, kv, attend) {
		const D = this.dim, H = this.heads, hd = this.headDim, F = this.ffn, n = seq * D;
		op.layerNorm(hidden, s.bufA, D, seq, EPS);
		op.scaleModulate(s.bufA, mod, s.bufA, n, D, 0);
		op.matmul(lw.q, s.bufA, s.q, D, D, seq, 0, true);
		op.matmul(lw.k, s.bufA, s.k, D, D, seq, 0, true);
		op.matmul(lw.v, s.bufA, s.v, D, D, seq);
		op.headNorm(s.q, lw.qNorm, H, hd, seq, EPS);
		op.headNorm(s.k, lw.kNorm, H, hd, seq, EPS);
		op.transposeHeads(s.q, s.bufB, seq, H, hd, 0, true);
		op.transposeHeads(s.k, kv.k, seq, H, hd, 0, true);
		op.transposeHeads(s.v, kv.v, seq, H, hd, 0);
		op.mrope(s.bufB, cos, sin, seq, H, true);
		op.mrope(kv.k, cos, sin, seq, H);
		if (!attend) return;
		attend(s.bufB, s.attn);
		op.matmul(lw.out, s.attn, s.bufA, D, D, seq);
		op.gatedAddTanh(hidden, mod, s.bufA, n, D, D);

		op.layerNorm(hidden, s.bufA, D, seq, EPS);
		op.scaleModulate(s.bufA, mod, s.bufA, n, D, 2 * D);
		// gate_up is [gate; up], each ffn rows.
		op.matmul(lw.gateUp, s.bufA, s.gate, F, D, seq, 0, true);
		op.matmul(lw.gateUp, s.bufA, s.up, F, D, seq, F);
		op.siluMul(s.gate, s.up, seq * F);
		op.matmul(lw.down, s.gate, s.bufA, D, F, seq);
		op.gatedAddTanh(hidden, mod, s.bufA, n, D, 3 * D);
	}
}
