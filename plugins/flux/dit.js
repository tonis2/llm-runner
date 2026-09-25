// The Flux 2 DiT: dual-stream blocks over image and text tokens, then
// single-stream blocks over both, in one [txt | img] sequence.
//
// The graph is `dependencies/flux.c3l/flux_dit.c3`'s. Where that copied a slice
// of a buffer out to feed a kernel - a modulation vector's third, the image half
// of the joint Q - this binds a view of it or passes an offset, so a block is
// its dispatches and nothing else.

import { llm, f32, GGML } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit, breathe, uploadEach } from '../lib/gpu.js';

// Architecture from the tensor shapes, the way `configure_flux_from_gguf` reads
// it: Klein-9B and FLUX.2-dev share a graph and differ in these numbers.
export function fluxConfig(model) {
	const img = model.shape('img_in.weight');          // [patch_dim, dim]
	const txt = model.shape('txt_in.weight');          // [txt_dim, dim]
	const tin = model.shape('time_in.in_layer.weight'); // [t_dim, dim]
	const headDim = model.shape('double_blocks.0.img_attn.norm.query_norm.scale')[0];
	const mlp = model.shape('double_blocks.0.img_mlp.0.weight')[1] / 2;
	let nDual = 0, nSingle = 0;
	for (const name of model.tensors.keys()) {
		let m = /^double_blocks\.(\d+)\./.exec(name);
		if (m) nDual = Math.max(nDual, Number(m[1]) + 1);
		m = /^single_blocks\.(\d+)\./.exec(name);
		if (m) nSingle = Math.max(nSingle, Number(m[1]) + 1);
	}
	const dim = img[1];
	const quarter = headDim / 4;
	return {
		dim,
		heads: dim / headDim,
		headDim,
		mlp,
		patchDim: img[0],
		textDim: txt[0],
		tDim: tin[0],
		nDual,
		nSingle,
		theta: model.meta('flux.rope.freq_base', 2000),
		axes: [quarter, quarter, quarter, headDim - 3 * quarter],
	};
}

const REF_T_OFFSET_STEP = 10;

// cos/sin tables [seq, headDim/2] for 4-axis rope (T, H, W, L). Sequence layout
// [text | noise image | ref image 0 | ref image 1 ...]; text runs along L, the
// images along H and W, and each reference image is told apart by T.
export function ropeTables(c, nTxt, latentW, latentH, refs = []) {
	const nNoise = latentW * latentH;
	const nRef = refs.reduce((a, r) => a + r.w * r.h, 0);
	const seq = nTxt + nNoise + nRef;
	const half = c.headDim / 2;
	const cos = new Float32Array(seq * half);
	const sin = new Float32Array(seq * half);
	const freqs = c.axes.map((ax) => {
		const f = new Float32Array(ax / 2);
		for (let k = 0; k < ax / 2; k++) f[k] = Math.fround(1 / Math.pow(c.theta, (2 * k) / ax));
		return f;
	});
	for (let t = 0; t < seq; t++) {
		let pos;
		if (t < nTxt) {
			pos = [0, 0, 0, t];
		} else if (t < nTxt + nNoise) {
			const i = t - nTxt;
			pos = [0, Math.floor(i / latentW), i % latentW, 0];
		} else {
			let local = t - nTxt - nNoise;
			let r = 0;
			while (local >= refs[r].w * refs[r].h) { local -= refs[r].w * refs[r].h; r++; }
			pos = [REF_T_OFFSET_STEP * (r + 1), Math.floor(local / refs[r].w), local % refs[r].w, 0];
		}
		let pair = 0;
		for (let a = 0; a < 4; a++) {
			for (let k = 0; k < freqs[a].length; k++) {
				const angle = Math.fround(pos[a] * freqs[a][k]);
				cos[t * half + pair] = Math.cos(angle);
				sin[t * half + pair] = Math.sin(angle);
				pair++;
			}
		}
	}
	return { cos, sin, seq };
}

export class FluxDiT {
	constructor(model) {
		this.model = model;
		this.config = fluxConfig(model);
		this.loaded = false;
	}

	// Upload every weight. Q8_0 stays Q8_0; BF16/F16 matrices and the norm
	// scales are widened to f32, which is what `load_dit_tensor` did.
	async load() {
		const m = this.model;
		const c = this.config;
		const up = (name) => m.upload(name, 'auto');
		const t0 = llm.now();
		this.g = {
			imgIn: up('img_in.weight'),
			txtIn: up('txt_in.weight'),
			time0: up('time_in.in_layer.weight'),
			time1: up('time_in.out_layer.weight'),
			modImg: up('double_stream_modulation_img.lin.weight'),
			modTxt: up('double_stream_modulation_txt.lin.weight'),
			modSingle: up('single_stream_modulation.lin.weight'),
			finalLinear: up('final_layer.linear.weight'),
			finalAdaLN: up('final_layer.adaLN_modulation.1.weight'),
		};
		this.dual = [];
		for (let l = 0; l < c.nDual; l++) {
			const p = (s) => `double_blocks.${l}.${s}`;
			this.dual.push(await uploadEach({
				imgQkv: () => up(p('img_attn.qkv.weight')),
				imgProj: () => up(p('img_attn.proj.weight')),
				imgQNorm: () => up(p('img_attn.norm.query_norm.scale')),
				imgKNorm: () => up(p('img_attn.norm.key_norm.scale')),
				txtQkv: () => up(p('txt_attn.qkv.weight')),
				txtProj: () => up(p('txt_attn.proj.weight')),
				txtQNorm: () => up(p('txt_attn.norm.query_norm.scale')),
				txtKNorm: () => up(p('txt_attn.norm.key_norm.scale')),
				imgUp: () => up(p('img_mlp.0.weight')),
				imgDown: () => up(p('img_mlp.2.weight')),
				txtUp: () => up(p('txt_mlp.0.weight')),
				txtDown: () => up(p('txt_mlp.2.weight')),
			}));
		}
		this.single = [];
		for (let l = 0; l < c.nSingle; l++) {
			const p = (s) => `single_blocks.${l}.${s}`;
			this.single.push(await uploadEach({
				linear1: () => up(p('linear1.weight')),
				linear2: () => up(p('linear2.weight')),
				qNorm: () => up(p('norm.query_norm.scale')),
				kNorm: () => up(p('norm.key_norm.scale')),
			}));
		}
		this.loaded = true;
		llm.print(`  DiT: ${c.nDual} dual + ${c.nSingle} single blocks, dim ${c.dim}, loaded in ${llm.since(t0)}`);
	}

	unload() {
		if (!this.loaded) return;
		for (const t of Object.values(this.g)) t.dispose();
		for (const b of this.dual.concat(this.single)) for (const t of Object.values(b)) t.dispose();
		this.loaded = false;
	}

	// Activations for one request's shapes.
	prepare({ nTxt, latentH, latentW, refs = [], refPatches = null }) {
		const c = this.config;
		this.release();
		const nNoise = latentH * latentW;
		const nRef = refs.reduce((a, r) => a + r.w * r.h, 0);
		const nImg = nNoise + nRef;
		const nJoint = nImg + nTxt;
		const dim = c.dim;
		this.shape = { nTxt, nNoise, nRef, nImg, nJoint, latentH, latentW };
		const tables = ropeTables(c, nTxt, latentW, latentH, refs);
		const C = three.compute;
		this.a = {
			hidden: f32(nJoint * dim),       // [txt | img], the single-stream state
			bufA: f32(nJoint * dim),
			bufB: f32(nJoint * dim),
			bufC: f32(nJoint * dim),
			q: f32(nJoint * dim),
			k: f32(nJoint * dim),
			v: f32(nJoint * dim),
			attn: f32(nJoint * dim),
			gate: f32(nJoint * c.mlp),
			up: f32(nJoint * c.mlp),
			concat: f32(nJoint * (dim + c.mlp)),
			patches: f32(nImg * c.patchDim),
			velocity: f32(nImg * c.patchDim),
			latent: f32(c.patchDim * nNoise),
			tEmb: f32(c.tDim),
			tMlp: f32(dim),
			siluT: f32(dim),
			modImg: f32(6 * dim),
			modTxt: f32(6 * dim),
			modSingle: f32(3 * dim),
			modFinal: f32(2 * dim),
			cos: f32(tables.cos.length),
			sin: f32(tables.sin.length),
		};
		this.a.cos.buffer.write(tables.cos);
		this.a.sin.buffer.write(tables.sin);
		if (nRef > 0) {
			this.refPatches = f32(nRef * c.patchDim);
			this.refPatches.buffer.write(refPatches);
		}
	}

	release() {
		if (this.a) for (const t of Object.values(this.a)) t.dispose();
		if (this.refPatches) this.refPatches.dispose();
		this.a = null;
		this.refPatches = null;
	}

	// One forward pass at `sigma`: this.a.latent in, this.a.velocity out (in the
	// latent's [C, H, W] layout). `text` is [nTxt, textDim].
	async forward(text, sigma) {
		const c = this.config;
		const a = this.a;
		const s = this.shape;
		const dim = c.dim;
		const F = 4; // bytes per float
		const nTxt = s.nTxt, nImg = s.nImg, nJoint = s.nJoint;
		// The two streams live in the joint buffer from the start: text rows first.
		const txtH = a.hidden.view(0, nTxt * dim * F);
		const imgH = a.hidden.view(nTxt * dim * F, nImg * dim * F);

		op.patchify(a.latent, a.patches, c.patchDim, s.latentH, s.latentW, 1);
		if (s.nRef > 0) {
			three.compute.copy(this.refPatches.buffer, a.patches.buffer, { dstOffset: s.nNoise * c.patchDim * F, size: s.nRef * c.patchDim * F });
		}
		op.matmul(this.g.imgIn, a.patches, imgH, dim, c.patchDim, nImg, 0, true);
		op.matmul(this.g.txtIn, text, txtH, dim, c.textDim, nTxt);

		op.timestepEmbed(a.tEmb, c.tDim, sigma * 1000);
		op.matmul(this.g.time0, a.tEmb, a.tMlp, dim, c.tDim, 1);
		op.silu(a.tMlp, dim);
		op.matmul(this.g.time1, a.tMlp, a.siluT, dim, dim, 1);
		op.silu(a.siluT, dim);
		op.matmul(this.g.modImg, a.siluT, a.modImg, 6 * dim, dim, 1, 0, true);
		op.matmul(this.g.modTxt, a.siluT, a.modTxt, 6 * dim, dim, 1, 0, true);
		op.matmul(this.g.modSingle, a.siluT, a.modSingle, 3 * dim, dim, 1, 0, true);
		op.matmul(this.g.finalAdaLN, a.siluT, a.modFinal, 2 * dim, dim, 1);

		for (const b of this.dual) {
			this.dualBlock(b, txtH, imgH);
			await breathe();
		}
		submit();
		for (const b of this.single) {
			this.singleBlock(b);
			await breathe();
		}

		// Final layer over the image rows: LayerNorm, AdaLN (shift first, then
		// scale, in this layer), project to patches, back to [C, H, W].
		op.layerNorm(imgH, a.bufA, dim, nImg);
		op.modulate(a.bufA, a.modFinal, a.bufA, nImg * dim, dim, dim, 0);
		op.matmul(this.g.finalLinear, a.bufA, a.patches, c.patchDim, dim, nImg);
		op.unpatchify(a.patches, a.velocity, c.patchDim, s.latentH, s.latentW, 1);
		submit();
	}

	dualBlock(w, txtH, imgH) {
		const c = this.config;
		const a = this.a;
		const s = this.shape;
		const dim = c.dim, mlp = c.mlp, F = 4;
		const nTxt = s.nTxt, nImg = s.nImg, nJoint = s.nJoint;
		const imgRows = (t) => t.view(nTxt * dim * F, nImg * dim * F);
		const txtRows = (t) => t.view(0, nTxt * dim * F);

		// Pre-attention norm + modulation (shift1, scale1 = mod[0], mod[1]).
		op.layerNorm(imgH, a.bufA, dim, nImg);
		op.layerNorm(txtH, a.bufB, dim, nTxt);
		op.modulate(a.bufA, a.modImg, a.bufA, nImg * dim, dim, dim, 0);
		op.modulate(a.bufB, a.modTxt, a.bufB, nTxt * dim, dim, dim, 0);

		// Q/K/V straight into their rows of the joint buffers.
		op.matmul(w.imgQkv, a.bufA, imgRows(a.q), dim, dim, nImg, 0, true);
		op.matmul(w.imgQkv, a.bufA, imgRows(a.k), dim, dim, nImg, dim, true);
		op.matmul(w.imgQkv, a.bufA, imgRows(a.v), dim, dim, nImg, 2 * dim, true);
		op.matmul(w.txtQkv, a.bufB, txtRows(a.q), dim, dim, nTxt, 0, true);
		op.matmul(w.txtQkv, a.bufB, txtRows(a.k), dim, dim, nTxt, dim, true);
		op.matmul(w.txtQkv, a.bufB, txtRows(a.v), dim, dim, nTxt, 2 * dim);

		op.headNorm(txtRows(a.q), w.txtQNorm, c.heads, c.headDim, nTxt);
		op.headNorm(txtRows(a.k), w.txtKNorm, c.heads, c.headDim, nTxt);
		op.headNorm(imgRows(a.q), w.imgQNorm, c.heads, c.headDim, nImg);
		op.headNorm(imgRows(a.k), w.imgKNorm, c.heads, c.headDim, nImg);

		this.attention();

		op.matmul(w.txtProj, txtRows(a.attn), a.bufB, dim, dim, nTxt, 0, true);
		op.matmul(w.imgProj, imgRows(a.attn), a.bufA, dim, dim, nImg);
		op.gatedAdd(imgH, a.modImg, a.bufA, nImg * dim, dim, 2 * dim);
		op.gatedAdd(txtH, a.modTxt, a.bufB, nTxt * dim, dim, 2 * dim);

		// FFN (shift2, scale2, gate2 = mod[3], mod[4], mod[5]).
		op.layerNorm(imgH, a.bufA, dim, nImg);
		op.modulate(a.bufA, a.modImg, a.bufA, nImg * dim, dim, 4 * dim, 3 * dim);
		op.matmul(w.imgUp, a.bufA, a.gate, mlp, dim, nImg, 0, true);
		op.matmul(w.imgUp, a.bufA, a.up, mlp, dim, nImg, mlp);
		op.siluMul(a.gate, a.up, nImg * mlp);
		op.matmul(w.imgDown, a.gate, a.bufA, dim, mlp, nImg);
		op.gatedAdd(imgH, a.modImg, a.bufA, nImg * dim, dim, 5 * dim);

		op.layerNorm(txtH, a.bufB, dim, nTxt);
		op.modulate(a.bufB, a.modTxt, a.bufB, nTxt * dim, dim, 4 * dim, 3 * dim);
		op.matmul(w.txtUp, a.bufB, a.gate, mlp, dim, nTxt, 0, true);
		op.matmul(w.txtUp, a.bufB, a.up, mlp, dim, nTxt, mlp);
		op.siluMul(a.gate, a.up, nTxt * mlp);
		op.matmul(w.txtDown, a.gate, a.bufB, dim, mlp, nTxt);
		op.gatedAdd(txtH, a.modTxt, a.bufB, nTxt * dim, dim, 5 * dim);
	}

	singleBlock(w) {
		const c = this.config;
		const a = this.a;
		const dim = c.dim, mlp = c.mlp;
		const seq = this.shape.nJoint;

		op.layerNorm(a.hidden, a.bufA, dim, seq);
		op.modulate(a.bufA, a.modSingle, a.bufA, seq * dim, dim, dim, 0);

		op.matmul(w.linear1, a.bufA, a.q, dim, dim, seq, 0, true);
		op.matmul(w.linear1, a.bufA, a.k, dim, dim, seq, dim, true);
		op.matmul(w.linear1, a.bufA, a.v, dim, dim, seq, 2 * dim, true);
		op.matmul(w.linear1, a.bufA, a.gate, mlp, dim, seq, 3 * dim, true);
		op.matmul(w.linear1, a.bufA, a.up, mlp, dim, seq, 3 * dim + mlp);

		op.headNorm(a.q, w.qNorm, c.heads, c.headDim, seq);
		op.headNorm(a.k, w.kNorm, c.heads, c.headDim, seq);
		this.attention();

		op.siluMul(a.gate, a.up, seq * mlp);
		op.concatRows(a.attn, a.gate, a.concat, seq, dim, mlp);
		op.matmul(w.linear2, a.concat, a.bufA, dim, dim + mlp, seq);
		op.gatedAdd(a.hidden, a.modSingle, a.bufA, seq * dim, dim, 2 * dim);
	}

	// q, k, v [joint, dim] -> attn [joint, dim]: transpose to heads, rotate,
	// flash attention (which transposes back as it writes).
	attention() {
		const c = this.config;
		const a = this.a;
		const seq = this.shape.nJoint;
		op.transposeHeads(a.q, a.bufA, seq, c.heads, c.headDim, 0, true);
		op.transposeHeads(a.k, a.bufB, seq, c.heads, c.headDim, 0, true);
		op.transposeHeads(a.v, a.bufC, seq, c.heads, c.headDim, 0);
		op.mrope(a.bufA, a.cos, a.sin, seq, c.heads, true);
		op.mrope(a.bufB, a.cos, a.sin, seq, c.heads);
		op.flashAttention(a.bufA, a.bufB, a.bufC, a.attn, c.heads, seq);
	}
}
