// LATO.2's V-Flow (VertexSLatFlowModel): a flow over the vertex latent, one
// row per active voxel of the 64^3 sparse structure, conditioned on DINOv2
// tokens by cross-attention and on the wanted vertex count (`density`) added
// to the timestep embedding.
//
// Ported from LATO.2's `models/vertex_structured_flow.py` and
// `modules/sparse/transformer/modulated.py` (ModulatedSparseTransformerCrossBlock):
// full self-attention over the points with 3D RoPE (on the voxel coordinates)
// and per-head RMSNorm on Q and K, in that order.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, breathe, breathing, copy } from '/core/gpu.js';
import { openTorch } from '/core/weights/torch.js';
import { Linear, TimestepEmbedder, Attention, Scratch, ropeTables, disposeAll } from './nn.js';

const HEAD = 64;

export class VertexFlow {
	constructor(path, { linear = 'f32' } = {}) {
		this.m = openTorch(path);
		const config = this.m.pickle.get('config');
		this.config = config;
		const a = config.get('args');
		this.args = a;
		if (a.get('pe_mode') !== 'rope') throw new Error(`V-Flow: pe_mode ${a.get('pe_mode')}, only rope is ported`);
		if (a.get('share_mod')) throw new Error('V-Flow: share_mod is not ported');
		this.C = a.get('model_channels');
		this.heads = this.C / HEAD;
		this.inC = a.get('in_channels');
		this.outC = a.get('out_channels');
		this.depth = a.get('num_blocks');
		this.ffn = this.C * a.get('mlp_ratio');
		this.maxVertices = config.get('max_vertex_num');
		this.latentDim = config.get('latent_dim');
		this.dino = config.get('dino_version');
		this.imageResolution = config.get('image_resolution');
		this.linear = linear;
	}

	async load() {
		const m = this.m, as = this.linear, t0 = llm.now();
		this.t = new TimestepEmbedder(m, 't_embedder');
		this.density = this.args.get('use_density') ? new TimestepEmbedder(m, 'density_embedder') : null;
		this.input = new Linear(m, 'input_layer');
		this.output = new Linear(m, 'out_layer');
		this.blocks = [];
		for (let l = 0; l < this.depth; l++) {
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
		llm.print(`  V-Flow: ${this.depth} blocks x ${this.C}, ${as} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		for (const b of this.blocks ?? []) disposeAll(b);
		disposeAll([this.t, this.density, this.input, this.output, this.attn, this.s]);
		this.blocks = null;
		this.m.close();
	}

	// Each block's cross-attention keys and values for tokens `cond` [L, dim].
	context(cond, L) {
		const C = this.C;
		return this.blocks.map((b) => {
			const k = f32(L * C), v = f32(L * C);
			b.kv.part(cond, k, L, 0, C);
			b.kv.part(cond, v, L, C, C);
			return { k, v, L, dispose() { k.dispose(); v.dispose(); } };
		});
	}

	// The velocity [n, out] at x [n, in] on points whose RoPE tables are `rope`
	// (ropeTables of their 64^3 coordinates), time t in 0..1, for a context().
	forward(x, n, rope, t, density, ctx, v) {
		const C = this.C, F = this.ffn, nh = this.heads, s = this.s;
		const h = s.get('h', n * C), a = s.get('a', n * C), q = s.get('q', n * C), k = s.get('k', n * C), vv = s.get('v', n * C);
		const o = s.get('o', n * C), up = s.get('up', n * F), temb = s.get('temb', C), demb = s.get('demb', C), mod = s.get('mod', 6 * C);
		this.input.run(x, h, n);
		this.t.run(1000 * t, temb);
		if (this.density) {
			this.density.run(density, demb);
			op.add(temb, demb, C);
		}
		op.silu(temb, C);
		for (const [l, b] of this.blocks.entries()) {
			b.ada.run(temb, mod, 1);
			op.layerNorm(h, a, C, n, 1e-6);
			op.modulate(a, mod, a, n * C, C, C, 0);
			b.qkv.part(a, q, n, 0, C);
			b.qkv.part(a, k, n, C, C);
			b.qkv.part(a, vv, n, 2 * C, C);
			op.ropePairs(q, rope.cos, rope.sin, HEAD, nh, n, true);
			op.ropePairs(k, rope.cos, rope.sin, HEAD, nh, n);
			if (b.qn) op.headNormGamma(q, b.qn, nh, HEAD, n, 1e-12, true);
			if (b.kn) op.headNormGamma(k, b.kn, nh, HEAD, n);
			this.attn.run(q, k, vv, o, nh, HEAD, n, n);
			b.out.run(o, a, n);
			op.gatedAdd(h, mod, a, n * C, C, 2 * C);
			op.layerNormAffine(h, b.n2w, b.n2b, a, C, n, 1e-6);
			b.q.run(a, q, n);
			this.attn.run(q, ctx[l].k, ctx[l].v, o, nh, HEAD, n, ctx[l].L);
			b.cout.run(o, a, n);
			op.add(h, a, n * C);
			op.layerNorm(h, a, C, n, 1e-6);
			op.modulate(a, mod, a, n * C, C, 4 * C, 3 * C);
			b.fc1.run(a, up, n);
			op.gelu(up, n * F);
			b.fc2.run(up, a, n);
			op.gatedAdd(h, mod, a, n * C, C, 5 * C);
			if (l % 4 === 3) submit();
		}
		op.layerNorm(h, a, C, n, 1e-5);
		this.output.run(a, v, n);
	}

	// The density LATO.2 conditions on for a wanted vertex count.
	densityOf(vertices) { return vertices / this.maxVertices * 1000; }

	// Euler with classifier-free guidance at every step from noise x [n, latent]
	// (a tensor, overwritten with the result) on points `coords` [n * 3].
	async sample(x, coords, n, cond, uncond, L, { steps = 24, cfg = 3.0, rescale = 1.0, vertices = 2000 } = {}) {
		const t0 = llm.now(), size = n * this.outC;
		const rope = ropeTables(coords, n, HEAD);
		const ctx = this.context(cond, L), nctx = this.context(uncond, L);
		const v = f32(size), vu = f32(size), density = this.densityOf(vertices);
		try {
			for (let i = 0; i < steps; i++) {
				const warp = (u) => rescale * u / (1 + (rescale - 1) * u);
				const t = warp(1 - i / steps), tPrev = warp(1 - (i + 1) / steps);
				this.forward(x, n, rope, t, density, ctx, v);
				this.forward(x, n, rope, t, density, nctx, vu);
				op.cfgCombine(v, vu, 1, size, 1 + cfg, false);
				op.eulerStep(x, vu, size, -(t - tPrev));
				submit();
				if (breathing()) await breathe();
			}
		} finally {
			submit();
			disposeAll([rope, ctx, nctx, v, vu]);
		}
		llm.print(`  V-Flow: ${n} points, ${steps} steps in ${llm.since(t0)}`);
	}
}
