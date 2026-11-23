// TRELLIS's stage one (microsoft/TRELLIS-image-large), as LATO.2 uses it: the
// sparse-structure flow, a DiT over the 16^3 latent grid conditioned on DINOv2
// tokens, and the decoder that turns the latent into 64^3 occupancy.
//
// Ported from LATO.2's `models/trellis_ss/` (TRELLIS's sparse_structure_flow.py
// and sparse_structure_vae.py). Grids are channel-last rows here: voxel
// (x, y, z) of an R^3 grid is row (x * R + y) * R + z.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, breathe, breathing, copy } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { Linear, TimestepEmbedder, Conv3d, Attention, Scratch, sincosPositions, upload, disposeAll, readFloats, rows } from './nn.js';

const HEAD = 64;

export class SparseStructureFlow {
	// `dir` holds ckpts' ss_flow_img_dit_L_16l8_fp16.{json,safetensors}.
	constructor(dir, { linear = 'f32' } = {}) {
		const name = `${dir}/ss_flow_img_dit_L_16l8_fp16`;
		this.cfg = JSON.parse(llm.readText(`${name}.json`)).args;
		const c = this.cfg;
		if (c.patch_size !== 1 || c.pe_mode !== 'ape') throw new Error('sparse-structure flow: only patch_size 1 with absolute positions');
		this.m = openSafetensors(`${name}.safetensors`);
		this.linear = linear;
		this.R = c.resolution;
		this.C = c.model_channels;
		this.heads = c.num_heads;
		this.inC = c.in_channels;
		this.outC = c.out_channels;
		this.ffn = c.model_channels * c.mlp_ratio;
	}

	async load() {
		const m = this.m, as = this.linear, t0 = llm.now();
		this.t = new TimestepEmbedder(m, 't_embedder', 'f32');
		this.input = new Linear(m, 'input_layer', 'f32');
		this.output = new Linear(m, 'out_layer', 'f32');
		const R = this.R, coords = new Int32Array(R * R * R * 3);
		for (let i = 0; i < R * R * R; i++) { coords[3 * i] = Math.floor(i / (R * R)); coords[3 * i + 1] = Math.floor(i / R) % R; coords[3 * i + 2] = i % R; }
		this.pos = upload(sincosPositions(coords, R * R * R, this.C));
		this.blocks = [];
		for (let l = 0; l < this.cfg.num_blocks; l++) {
			const p = `blocks.${l}`;
			this.blocks.push({
				ada: new Linear(m, `${p}.adaLN_modulation.1`, as),
				qkv: new Linear(m, `${p}.self_attn.to_qkv`, as),
				qn: m.has(`${p}.self_attn.q_rms_norm.gamma`) ? m.upload(`${p}.self_attn.q_rms_norm.gamma`, 'f32') : null,
				kn: m.has(`${p}.self_attn.k_rms_norm.gamma`) ? m.upload(`${p}.self_attn.k_rms_norm.gamma`, 'f32') : null,
				out: new Linear(m, `${p}.self_attn.to_out`, as),
				n2w: m.upload(`${p}.norm2.weight`, 'f32'), n2b: m.upload(`${p}.norm2.bias`, 'f32'),
				q: new Linear(m, `${p}.cross_attn.to_q`, as),
				kv: new Linear(m, `${p}.cross_attn.to_kv`, as),
				cout: new Linear(m, `${p}.cross_attn.to_out`, as),
				fc1: new Linear(m, `${p}.mlp.mlp.0`, as),
				fc2: new Linear(m, `${p}.mlp.mlp.2`, as),
			});
			if (breathing()) await breathe();
		}
		this.attn = new Attention();
		this.s = new Scratch();
		llm.print(`  sparse-structure flow: ${this.cfg.num_blocks} blocks x ${this.C}, ${as} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		for (const b of this.blocks ?? []) disposeAll(b);
		disposeAll([this.t, this.input, this.output, this.pos, this.attn, this.s]);
		this.blocks = null;
		this.m.close();
	}

	// Each block's cross-attention keys and values for conditioning tokens
	// `cond` [L, condDim] (a GPU tensor): computed once for every step.
	context(cond, L) {
		const C = this.C;
		return this.blocks.map((b) => {
			const k = f32(L * C), v = f32(L * C);
			b.kv.part(cond, k, L, 0, C);
			b.kv.part(cond, v, L, C, C);
			return { k, v, L, dispose() { k.dispose(); v.dispose(); } };
		});
	}

	// The velocity [R^3, out] at noisy latent x [R^3, in] and time t (0..1, as
	// TRELLIS's 1000 t is given to its timestep embedder), for a context().
	forward(x, t, ctx, v) {
		const C = this.C, n = this.R ** 3, F = this.ffn, nh = this.heads, s = this.s;
		const h = s.get('h', n * C), a = s.get('a', n * C), q = s.get('q', n * C), k = s.get('k', n * C), vv = s.get('v', n * C);
		const o = s.get('o', n * C), up = s.get('up', n * F), temb = s.get('temb', C), silu = s.get('silu', C), mod = s.get('mod', 6 * C);
		this.input.run(x, h, n);
		op.add(h, this.pos, n * C);
		this.t.run(1000 * t, temb);
		copy(temb, silu, C * 4);
		op.silu(silu, C);
		for (const [l, b] of this.blocks.entries()) {
			b.ada.run(silu, mod, 1);
			// self-attention, modulated
			op.layerNorm(h, a, C, n, 1e-6);
			op.modulate(a, mod, a, n * C, C, C, 0);
			b.qkv.part(a, q, n, 0, C);
			b.qkv.part(a, k, n, C, C);
			b.qkv.part(a, vv, n, 2 * C, C);
			if (b.qn) op.headNormGamma(q, b.qn, nh, HEAD, n);
			if (b.kn) op.headNormGamma(k, b.kn, nh, HEAD, n);
			this.attn.run(q, k, vv, o, nh, HEAD, n, n);
			b.out.run(o, a, n);
			op.gatedAdd(h, mod, a, n * C, C, 2 * C);
			// cross-attention
			op.layerNormAffine(h, b.n2w, b.n2b, a, C, n, 1e-6);
			b.q.run(a, q, n);
			this.attn.run(q, ctx[l].k, ctx[l].v, o, nh, HEAD, n, ctx[l].L);
			b.cout.run(o, a, n);
			op.add(h, a, n * C);
			// MLP, modulated
			op.layerNorm(h, a, C, n, 1e-6);
			op.modulate(a, mod, a, n * C, C, 4 * C, 3 * C);
			b.fc1.run(a, up, n);
			op.gelu(up, n * F);
			b.fc2.run(up, a, n);
			op.gatedAdd(h, mod, a, n * C, C, 5 * C);
		}
		op.layerNorm(h, a, C, n, 1e-5);
		this.output.run(a, v, n);
	}

	// TRELLIS's FlowEulerGuidanceIntervalSampler from noise x [R^3, in] (a
	// tensor, overwritten with the result): guidance only while t is inside
	// `interval`, the schedule warped by `rescale`.
	async sample(x, cond, uncond, L, { steps = 25, cfg = 5.0, interval = [0.5, 1.0], rescale = 3.0 } = {}) {
		const n = this.R ** 3 * this.outC, t0 = llm.now();
		const ctx = this.context(cond, L), nctx = this.context(uncond, L);
		const v = f32(n), vu = f32(n);
		try {
			for (let i = 0; i < steps; i++) {
				const lin = (j) => 1 - j / steps;
				const warp = (u) => rescale * u / (1 + (rescale - 1) * u);
				const t = warp(lin(i)), tPrev = warp(lin(i + 1));
				this.forward(x, t, ctx, v);
				if (t >= interval[0] && t <= interval[1]) {
					this.forward(x, t, nctx, vu);
					// v = (1 + cfg) v - cfg vu: cfgCombine's vu + s (v - vu) at s = 1 + cfg
					op.cfgCombine(v, vu, 1, n, 1 + cfg, false);
					op.eulerStep(x, vu, n, -(t - tPrev));
				} else {
					op.eulerStep(x, v, n, -(t - tPrev));
				}
				submit();
				if (breathing()) await breathe();
			}
		} finally {
			submit();
			disposeAll([ctx, nctx, v, vu]);
		}
		llm.print(`  sparse structure: ${steps} steps in ${llm.since(t0)}`);
	}
}

// The sparse-structure decoder: latent [16^3, 8] to occupancy logits [64^3].
export class SparseStructureDecoder {
	constructor(dir) {
		const name = `${dir}/ss_dec_conv3d_16l8_fp16`;
		this.cfg = JSON.parse(llm.readText(`${name}.json`)).args;
		this.m = openSafetensors(`${name}.safetensors`);
	}

	load() {
		const m = this.m, c = this.cfg, f = (n) => m.upload(n, 'f32');
		const res = (p) => ({ n1w: f(`${p}.norm1.weight`), n1b: f(`${p}.norm1.bias`), c1: new Conv3d(m, `${p}.conv1`),
			n2w: f(`${p}.norm2.weight`), n2b: f(`${p}.norm2.bias`), c2: new Conv3d(m, `${p}.conv2`),
			skip: m.has(`${p}.skip_connection.weight`) ? new Conv3d(m, `${p}.skip_connection`) : null });
		this.input = new Conv3d(m, 'input_layer');
		this.middle = [];
		for (let i = 0; i < c.num_res_blocks_middle; i++) this.middle.push(res(`middle_block.${i}`));
		this.blocks = [];
		let at = 0;
		for (const [i, ch] of c.channels.entries()) {
			for (let j = 0; j < c.num_res_blocks; j++) this.blocks.push({ ch, ...res(`blocks.${at++}`) });
			if (i < c.channels.length - 1) this.blocks.push({ up: new Conv3d(m, `blocks.${at++}.conv`), ch: c.channels[i + 1] });
		}
		this.outNw = f('out_layer.0.weight');
		this.outNb = f('out_layer.0.bias');
		this.out = new Conv3d(m, 'out_layer.2');
		this.s = new Scratch();
	}

	dispose() {
		disposeAll([this.input, this.middle, this.blocks, this.outNw, this.outNb, this.out, this.s]);
		this.m.close();
	}

	// One ResBlock3d over x [R^3, ch] in place.
	res(b, x, R, ch) {
		const n = R ** 3, s = this.s, cols = s.get('cols', 1 << 25);
		const a = s.get('ra', n * ch), c = s.get('rc', n * ch);
		op.layerNormAffine(x, b.n1w, b.n1b, a, ch, n, 1e-5);
		op.silu(a, n * ch);
		b.c1.run(a, c, [R, R, R], cols);
		op.layerNormAffine(c, b.n2w, b.n2b, a, ch, n, 1e-5);
		op.silu(a, n * ch);
		b.c2.run(a, c, [R, R, R], cols);
		op.add(x, c, n * ch);
	}

	// latent [R^3, 8] -> logits [(4R)^3] (a new tensor).
	run(latent, R) {
		const s = this.s, cols = s.get('cols', 1 << 25), ch0 = this.cfg.channels[0];
		let x = s.get('x0', R ** 3 * ch0);
		this.input.run(latent, x, [R, R, R], cols);
		for (const b of this.middle) this.res(b, x, R, ch0);
		let ch = ch0, flip = 0;
		for (const b of this.blocks) {
			if (b.up) {
				const wide = s.get('wide', R ** 3 * b.up.outC);
				b.up.run(x, wide, [R, R, R], cols);
				const y = s.get(`x${++flip % 2}`, (2 * R) ** 3 * b.ch);
				op.pixelShuffle3d(wide, y, b.ch, R, R, R);
				x = y;
				R *= 2;
				ch = b.ch;
			} else {
				this.res(b, x, R, ch);
			}
			submit();
		}
		const n = R ** 3, a = s.get('ra', n * ch), logits = f32(n);
		op.layerNormAffine(x, this.outNw, this.outNb, a, ch, n, 1e-5);
		op.silu(a, n * ch);
		this.out.run(a, logits, [R, R, R], cols);
		submit();
		return { logits, R };
	}
}
