// LATO.2's topology stage: the voxel encoder that turns the 64^3 sparse
// structure into conditioning tokens, T-Flow (TopologySiTFlow) that samples a
// latent per vertex, and the T-VAE decoder that reads edges out of it.
//
// Ported from LATO.2's `models/voxel_encoder.py`, `models/topo_flow.py` and
// `models/topo_autoencoder.py` (decode) with `modules/transformer/hybrid.py`.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, breathe, breathing, copy } from '/core/gpu.js';
import { openTorch } from '/core/weights/torch.js';
import { Linear, TimestepEmbedder, Conv3d, Attention, Scratch, ropeTables, upload, disposeAll, readFloats } from './nn.js';

// ---- voxel encoder

// GroupNorm's group count as LATO.2 picks it: up to 8, dividing the channels.
function groupsOf(channels, most = 8) {
	let g = Math.min(most, channels);
	while (g > 1 && channels % g !== 0) g--;
	return g;
}

// 1D sin/cos rows [n, dim]: sin(pos w) then cos(pos w), w = 1 / 10000^(i / (dim/2)).
function sincos1d(n, dim) {
	const half = dim / 2, out = new Float32Array(n * dim);
	for (let p = 0; p < n; p++) for (let i = 0; i < half; i++) {
		const a = Math.fround(p * Math.fround(1 / Math.pow(10000, Math.fround(i / half))));
		out[p * dim + i] = Math.sin(a);
		out[p * dim + half + i] = Math.cos(a);
	}
	return out;
}

export class VoxelEncoder {
	constructor(path) {
		this.m = openTorch(path);
		const c = this.m.pickle.get('config');
		const a = c.get('args');
		this.resolution = c.get('resolution');
		this.down = a.get('num_downsamples');
		this.perLevel = a.get('blocks_per_level');
		this.dim = a.get('cond_in_dim');
		if (a.get('pos_embed') !== 'sincos') throw new Error('voxel encoder: only sincos positions are ported');
	}

	load() {
		const m = this.m, f = (n) => m.upload(n, 'f32');
		const gn = (p) => ({ w: f(`${p}.weight`), b: f(`${p}.bias`) });
		const res = (p) => ({ n1: gn(`${p}.block.0`), c1: new Conv3d(m, `${p}.block.2`), n2: gn(`${p}.block.3`), c2: new Conv3d(m, `${p}.block.5`) });
		this.stem = { conv: new Conv3d(m, 'stem.0'), n: gn('stem.1') };
		this.levels = [];
		for (let i = 0; i < this.down; i++) {
			const p = `down_blocks.${i}.net`, blocks = [];
			for (let j = 0; j < this.perLevel; j++) blocks.push(res(`${p}.${j}`));
			this.levels.push({ blocks, conv: new Conv3d(m, `${p}.${this.perLevel}`), n: gn(`${p}.${this.perLevel + 1}`) });
		}
		this.tail = [];
		for (let j = 0; j < this.perLevel; j++) this.tail.push(res(`tail_blocks.${j}`));
		this.proj = new Conv3d(m, 'proj');
		this.s = new Scratch();
	}

	dispose() {
		disposeAll([this.stem, this.levels, this.tail, this.proj, this.s]);
		this.m.close();
	}

	res(b, x, R, ch) {
		const n = R ** 3, s = this.s, cols = s.get('cols', 1 << 26), a = s.get('ra', n * ch), c = s.get('rc', n * ch), g = groupsOf(ch);
		op.groupNormRows(x, b.n1.w, b.n1.b, a, ch, n, g, 1e-5, true);
		b.c1.run(a, c, [R, R, R], cols);
		op.groupNormRows(c, b.n2.w, b.n2.b, a, ch, n, g, 1e-5, true);
		b.c2.run(a, c, [R, R, R], cols);
		op.add(x, c, n * ch);
	}

	// The occupied voxels `coords` (Int32Array [n * 3] on the 64^3 grid) ->
	// tokens [(R / 2^down)^3, dim] (a new tensor) with their positions added.
	run(coords, n) {
		const s = this.s, R0 = this.resolution, cols = s.get('cols', 1 << 26);
		const field = new Float32Array(R0 ** 3);
		for (let i = 0; i < n; i++) field[(coords[3 * i] * R0 + coords[3 * i + 1]) * R0 + coords[3 * i + 2]] = 1;
		const input = upload(field);
		let R = R0, ch = this.stem.conv.outC;
		let x = s.get('x0', R ** 3 * ch);
		this.stem.conv.run(input, x, [R, R, R], cols);
		op.groupNormRows(x, this.stem.n.w, this.stem.n.b, x, ch, R ** 3, groupsOf(ch), 1e-5, true);
		let flip = 0;
		for (const L of this.levels) {
			for (const b of L.blocks) this.res(b, x, R, ch);
			const out = L.conv.outC, y = s.get(`x${++flip % 2}`, (R / 2) ** 3 * out);
			L.conv.run(x, y, [R, R, R], cols, 2);
			R /= 2;
			ch = out;
			op.groupNormRows(y, L.n.w, L.n.b, y, ch, R ** 3, groupsOf(ch), 1e-5, true);
			x = y;
			submit();
		}
		for (const b of this.tail) this.res(b, x, R, ch);
		const T = R ** 3, tokens = f32(T * this.dim);
		this.proj.run(x, tokens, [R, R, R], cols);
		// 3D sin/cos: each axis a third (rounded down to even), zero-padded
		const axis = Math.floor(Math.floor(this.dim / 3) / 2) * 2, e = sincos1d(R, axis), pe = new Float32Array(T * this.dim);
		for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) for (let k = 0; k < R; k++) {
			const t = (i * R + j) * R + k;
			pe.set(e.subarray(i * axis, (i + 1) * axis), t * this.dim);
			pe.set(e.subarray(j * axis, (j + 1) * axis), t * this.dim + axis);
			pe.set(e.subarray(k * axis, (k + 1) * axis), t * this.dim + 2 * axis);
		}
		const peT = upload(pe);
		op.add(tokens, peT, T * this.dim);
		submit();
		disposeAll([input, peT]);
		return { tokens, count: T };
	}
}

// ---- T-Flow

const T_HEAD = 64;

export class TopologyFlow {
	constructor(path, { linear = 'f32' } = {}) {
		this.m = openTorch(path);
		const c = this.m.pickle.get('config'), a = c.get('args');
		this.zDim = a.get('z_dim');
		this.C = a.get('hidden_size');
		this.depth = a.get('depth');
		this.heads = a.get('num_heads') ?? 12;
		this.ffn = this.C * a.get('mlp_ratio');
		this.maxVertices = a.get('max_vertices');
		this.discrete = a.get('num_discrete');
		this.condDim = a.get('cond_in_dim');
		this.eps = a.get('qk_norm_eps');
		this.latentScale = c.get('latent_scale');
		this.linear = linear;
		if (this.C / this.heads !== T_HEAD) throw new Error(`T-Flow: head width ${this.C / this.heads}`);
	}

	async load() {
		const m = this.m, as = this.linear, f = (n) => m.upload(n, 'f32'), t0 = llm.now();
		this.input = new Linear(m, 'input_proj');
		this.coord0 = new Linear(m, 'coord_embed.0');
		this.coord2 = new Linear(m, 'coord_embed.2');
		this.t = new TimestepEmbedder(m, 't_embedder');
		this.cond0 = new Linear(m, 'cond_proj.0');
		this.cond2 = new Linear(m, 'cond_proj.2');
		this.blocks = [];
		for (let l = 0; l < this.depth; l++) {
			const p = `blocks.${l}`;
			this.blocks.push({
				ada: new Linear(m, `${p}.adaLN_modulation.1`, as),
				qkv: new Linear(m, `${p}.qkv`, as), out: new Linear(m, `${p}.proj_out`, as),
				nq: f(`${p}.norm_q.weight`), nk: f(`${p}.norm_k.weight`),
				q: new Linear(m, `${p}.q_ca`, as), kv: new Linear(m, `${p}.kv_ca`, as), cout: new Linear(m, `${p}.proj_ca_out`, as),
				nqc: f(`${p}.norm_q_ca.weight`), nkc: f(`${p}.norm_k_ca.weight`),
				fc1: new Linear(m, `${p}.mlp.0`, as), fc2: new Linear(m, `${p}.mlp.2`, as),
			});
			if (breathing()) await breathe();
		}
		this.finalAda = new Linear(m, 'final_layer.adaLN_modulation.1');
		this.final = new Linear(m, 'final_layer.linear');
		this.attn = new Attention();
		this.s = new Scratch();
		llm.print(`  T-Flow: ${this.depth} blocks x ${this.C}, ${as} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		for (const b of this.blocks ?? []) disposeAll(b);
		disposeAll([this.input, this.coord0, this.coord2, this.t, this.cond0, this.cond2, this.finalAda, this.final, this.attn, this.s]);
		this.m.close();
	}

	// Per-run inputs: the vertices' fixed rows (positions, coordinate
	// embedding), RoPE tables and each block's cross-attention keys/values.
	prepare(verts, n, cond, L) {
		const C = this.C, s = this.s, D = this.discrete;
		const base = f32(n * C), mid = s.get('pmid', Math.max(n, L) * C);
		const coords = new Float32Array(n * 3);
		for (let i = 0; i < n * 3; i++) coords[i] = Math.fround(Math.fround((verts[i] + 0.5) / D) * 2 - 1);
		const cu = upload(coords);
		this.coord0.run(cu, mid, n);
		op.silu(mid, n * C);
		this.coord2.run(mid, base, n);
		const pe = upload(sincos1d(n, C));
		op.add(base, pe, n * C);
		const ce = f32(L * C);
		this.cond0.run(cond, mid, L);
		op.silu(mid, L * C);
		this.cond2.run(mid, ce, L);
		const nh = this.heads;
		const ctx = this.blocks.map((b) => {
			const k = f32(L * C), v = f32(L * C);
			b.kv.part(ce, k, L, 0, C);
			b.kv.part(ce, v, L, C, C);
			op.headNorm(k, b.nkc, nh, T_HEAD, L, this.eps);
			return { k, v, dispose() { k.dispose(); v.dispose(); } };
		});
		submit();
		disposeAll([cu, pe, ce]);
		return { base, ctx, L, rope: ropeTables(verts, n, T_HEAD), dispose() { disposeAll([this.base, this.ctx, this.rope]); } };
	}

	forward(x, n, t, p, v) {
		const C = this.C, F = this.ffn, nh = this.heads, s = this.s, eps = this.eps;
		const h = s.get('h', n * C), a = s.get('a', n * C), q = s.get('q', n * C), k = s.get('k', n * C), vv = s.get('v', n * C);
		const o = s.get('o', n * C), up = s.get('up', n * F), temb = s.get('temb', C), mod = s.get('mod', 7 * C);
		this.input.run(x, h, n);
		op.add(h, p.base, n * C);
		this.t.run(t, temb);
		op.silu(temb, C);
		for (const [l, b] of this.blocks.entries()) {
			b.ada.run(temb, mod, 1);
			// shift_msa, scale_msa, gate_msa, gate_mca, shift_mlp, scale_mlp, gate_mlp
			op.layerNorm(h, a, C, n, 1e-6);
			op.modulate(a, mod, a, n * C, C, C, 0);
			b.qkv.part(a, q, n, 0, C);
			b.qkv.part(a, k, n, C, C);
			b.qkv.part(a, vv, n, 2 * C, C);
			op.headNorm(q, b.nq, nh, T_HEAD, n, eps);
			op.headNorm(k, b.nk, nh, T_HEAD, n, eps);
			op.ropePairs(q, p.rope.cos, p.rope.sin, T_HEAD, nh, n, true);
			op.ropePairs(k, p.rope.cos, p.rope.sin, T_HEAD, nh, n);
			this.attn.run(q, k, vv, o, nh, T_HEAD, n, n);
			b.out.run(o, a, n);
			op.gatedAdd(h, mod, a, n * C, C, 2 * C);
			op.layerNorm(h, a, C, n, 1e-6);
			b.q.run(a, q, n);
			op.headNorm(q, b.nqc, nh, T_HEAD, n, eps);
			this.attn.run(q, p.ctx[l].k, p.ctx[l].v, o, nh, T_HEAD, n, p.L);
			b.cout.run(o, a, n);
			op.gatedAdd(h, mod, a, n * C, C, 3 * C);
			op.layerNorm(h, a, C, n, 1e-6);
			op.modulate(a, mod, a, n * C, C, 5 * C, 4 * C);
			b.fc1.run(a, up, n);
			op.gelu(up, n * F);
			b.fc2.run(up, a, n);
			op.gatedAdd(h, mod, a, n * C, C, 6 * C);
			if (l % 4 === 3) submit();
		}
		this.finalAda.run(temb, mod, 1);
		op.layerNorm(h, a, C, n, 1e-6);
		op.modulate(a, mod, a, n * C, C, C, 0);
		this.final.run(a, v, n);
	}

	// Euler from t = 0 (noise x [n, z], overwritten) to 1 on vertices `verts`
	// (Int32Array [n * 3] on the 1024^3 grid), then divided by the latent scale.
	async sample(x, verts, n, cond, L, { steps = 50 } = {}) {
		if (n > this.maxVertices) throw new Error(`T-Flow: ${n} vertices, at most ${this.maxVertices}`);
		const t0 = llm.now(), size = n * this.zDim, v = f32(size);
		const p = this.prepare(verts, n, cond, L);
		try {
			for (let i = 0; i < steps; i++) {
				this.forward(x, n, i / steps, p, v);
				op.eulerStep(x, v, size, 1 / steps);
				submit();
				if (breathing()) await breathe();
			}
			op.scale(x, size, 1 / this.latentScale);
			submit();
		} finally {
			submit();
			disposeAll([p, v]);
		}
		llm.print(`  T-Flow: ${n} vertices, ${steps} steps in ${llm.since(t0)}`);
	}
}

// ---- T-VAE decoder

export class TopologyDecoder {
	constructor(path) {
		this.m = openTorch(path);
		const a = this.m.pickle.get('config').get('args');
		this.heads = a.get('num_heads');
		this.C = a.get('hidden_dim');
		this.hd = this.C / this.heads;
		this.stages = a.get('num_hybrid_stages');
		this.flashPer = a.get('num_flash_per_stage');
	}

	load() {
		const m = this.m, f = (n) => m.upload(n, 'f32');
		const block = (p) => ({
			qkv: new Linear(m, `${p}.qkv`), out: new Linear(m, `${p}.proj_out`),
			fc1: new Linear(m, `${p}.ffn.0`), fc2: new Linear(m, `${p}.ffn.2`),
			sa: f(`${p}.scale_msa`), sm: f(`${p}.scale_mlp`),
		});
		this.input = new Linear(m, 'decoder.input_proj');
		this.backbone = [];
		for (let i = 0; i < this.stages; i++) {
			this.backbone.push(block(`decoder.backbone.stages.${i}.graph_block`));
			for (let j = 0; j < this.flashPer; j++) this.backbone.push(block(`decoder.backbone.stages.${i}.flash_blocks.${j}`));
		}
		this.connection = [];
		for (let i = 0; m.has(`connection_transformer_blocks.${i}.qkv.weight`); i++) this.connection.push(block(`connection_transformer_blocks.${i}`));
		// The pair MLP's first layer split into the halves that see u and v.
		const w1 = m.floats('connection_predictor.mlp.0.weight'), H = m.shape('connection_predictor.mlp.0.weight')[1], C = this.C;
		const wa = new Float32Array(H * C), wb = new Float32Array(H * C);
		for (let r = 0; r < H; r++) { wa.set(w1.subarray(r * 2 * C, r * 2 * C + C), r * C); wb.set(w1.subarray(r * 2 * C + C, (r + 1) * 2 * C), r * C); }
		this.hidden = H;
		this.wa = upload(wa);
		this.wb = upload(wb);
		this.b1 = f('connection_predictor.mlp.0.bias');
		this.w2 = f('connection_predictor.mlp.2.weight');
		this.b2 = m.floats('connection_predictor.mlp.2.bias')[0];
		this.attn = new Attention();
		this.s = new Scratch();
	}

	dispose() {
		disposeAll([this.input, this.backbone, this.connection, this.wa, this.wb, this.b1, this.w2, this.attn, this.s]);
		this.m.close();
	}

	block(b, h, n, rope) {
		const C = this.C, nh = this.heads, hd = this.hd, s = this.s;
		const a = s.get('a', n * C), q = s.get('q', n * C), k = s.get('k', n * C), v = s.get('v', n * C), o = s.get('o', n * C), up = s.get('up', n * 4 * C);
		op.layerNorm(h, a, C, n, 1e-6);
		b.qkv.part(a, q, n, 0, C);
		b.qkv.part(a, k, n, C, C);
		b.qkv.part(a, v, n, 2 * C, C);
		if (rope) {
			op.ropePairs(q, rope.cos, rope.sin, hd, nh, n, true);
			op.ropePairs(k, rope.cos, rope.sin, hd, nh, n);
		}
		this.attn.run(q, k, v, o, nh, hd, n, n);
		b.out.run(o, a, n);
		op.gatedAdd(h, b.sa, a, n * C, C, 0);
		op.layerNorm(h, a, C, n, 1e-6);
		b.fc1.run(a, up, n);
		op.gelu(up, n * 4 * C);
		b.fc2.run(up, a, n);
		op.gatedAdd(h, b.sm, a, n * C, C, 0);
	}

	// Per-vertex features [n, C] (a new tensor) from latent z [n, zDim] on
	// vertices `verts` (Int32Array [n * 3], 1024^3 grid).
	features(z, verts, n) {
		const h = f32(n * this.C);
		this.input.run(z, h, n);
		for (const b of this.backbone) this.block(b, h, n, null);
		const rope = ropeTables(verts, n, this.hd);
		for (const b of this.connection) this.block(b, h, n, rope);
		submit();
		rope.dispose();
		return h;
	}

	// The edges [[i, j], ...] (i < j, sorted) whose pair logit passes `threshold`.
	edges(feats, n, threshold = 0) {
		const H = this.hidden, s = this.s, A = s.get('pa', n * H), B = s.get('pb', n * H);
		op.matmul(this.wa, feats, A, H, this.C, n);
		op.biasAdd(A, this.b1, n * H, H);
		op.matmul(this.wb, feats, B, H, this.C, n);
		let cap = Math.max(1024, 16 * n);
		for (;;) {
			const count = three.compute.u32(1), out = three.compute.u32(2 * cap);
			op.pairEdges(A, B, this.w2, count, out, n, H, this.b2, threshold, cap);
			submit();
			const got = count.read().bytes[0];
			if (got > cap) { count.dispose(); out.dispose(); cap = got; continue; }
			const raw = got ? new Uint32Array(out.readBytes(0, got * 8).buffer) : new Uint32Array(0);
			count.dispose();
			out.dispose();
			// fastjs's typed arrays have no sort
			const keys = new Array(got);
			for (let e = 0; e < got; e++) keys[e] = raw[2 * e] * n + raw[2 * e + 1];
			keys.sort((a, b) => a - b);
			const edges = new Int32Array(got * 2);
			for (let e = 0; e < got; e++) { edges[2 * e] = Math.floor(keys[e] / n); edges[2 * e + 1] = keys[e] % n; }
			return edges;
		}
	}
}
