// Pixal3D's flow transformers (TRELLIS.2's DiT with Pixal3D's projection
// conditioning), one class for all four: the sparse-structure flow over the
// dense 16^3 grid and the shape and texture flows over sparse voxels are the
// same blocks over token rows; full attention does not care whether the
// tokens are a whole grid or some of its cells, only their coordinates (RoPE)
// differ.
//
// A block: adaLN-modulated self-attention (per-head RMSNorm on q and k, 3D
// RoPE), then cross-attention over the 5 global DINO tokens plus the block's
// own linear map of the voxel's proj features, then the modulated GELU MLP.
// The modulation is shared: one adaLN linear of the timestep, plus each
// block's learned offset.
//
// Ported from Pixal3D's models/sparse_structure_flow.py,
// models/structured_latent_flow.py, modules/transformer/modulated.py,
// modules/sparse/transformer/modulated.py, modules/attention/modules.py,
// modules/attention/proj_attention.py and pipelines/samplers/flow_euler.py
// (FlowEulerGuidanceIntervalSampler, with the CFG rescale of
// classifier_free_guidance_mixin.py).

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, breathe, breathing, copy } from '/core/gpu.js';
import { Linear, TimestepEmbedder, Attention, Scratch, ropeTables, upload, disposeAll, readFloats } from '/plugins/lato2/nn.js';

const SIGMA_MIN = 1e-5;

// The flows in the GGUF, by prefix.
export const FLOWS = {
	ss: { prefix: 'ss', meta: 'ss_flow' },
	shape512: { prefix: 'sh512', meta: 'shape_flow_512' },
	shape1024: { prefix: 'sh1024', meta: 'shape_flow_1024' },
	tex1024: { prefix: 'tx1024', meta: 'tex_flow_1024' },
};

export class Flow {
	// `m` the open GGUF (llm.open), `which` a key of FLOWS.
	constructor(m, which, { linear = 'f32' } = {}) {
		const { prefix } = FLOWS[which];
		this.m = m;
		this.p = `${prefix}.`;
		this.name = which;
		this.linear = linear;
		const shape = (n) => m.shape(`${this.p}${n}`);
		this.C = shape('input_layer.weight')[1];
		this.inC = shape('input_layer.weight')[0];
		this.outC = shape('out_layer.weight')[1];
		this.ffn = shape('blocks.0.mlp.mlp.0.weight')[1];
		this.projC = shape('blocks.0.cross_attn.proj_linear.weight')[0];
		this.ctxC = shape('blocks.0.cross_attn.cross_attn_block.to_kv.weight')[0];
		this.hd = shape('blocks.0.self_attn.q_rms_norm.gamma')[0];
		this.heads = this.C / this.hd;
		let n = 0;
		while (m.has(`${this.p}blocks.${n}.modulation`)) n++;
		this.depth = n;
	}

	async load() {
		// 'f16' is the GGUF's own F16, uploaded as it is (matmul_f16_coop).
		const m = this.m, P = this.p, as = this.linear === 'f16' ? 'raw' : this.linear, t0 = llm.now();
		const f = (n) => m.upload(`${P}${n}`, 'f32');
		this.t = new TimestepEmbedder(m, `${P}t_embedder`, 'f32');
		this.ada = new Linear(m, `${P}adaLN_modulation.1`, as);
		this.input = new Linear(m, `${P}input_layer`, 'f32');
		this.output = new Linear(m, `${P}out_layer`, 'f32');
		this.blocks = [];
		for (let l = 0; l < this.depth; l++) {
			const b = `blocks.${l}.`, c = `${b}cross_attn.cross_attn_block.`;
			this.blocks.push({
				mod: f(`${b}modulation`),
				qkv: new Linear(m, `${P}${b}self_attn.to_qkv`, as),
				qn: f(`${b}self_attn.q_rms_norm.gamma`), kn: f(`${b}self_attn.k_rms_norm.gamma`),
				out: new Linear(m, `${P}${b}self_attn.to_out`, as),
				n2w: f(`${b}norm2.weight`), n2b: f(`${b}norm2.bias`),
				q: new Linear(m, `${P}${c}to_q`, as),
				kv: new Linear(m, `${P}${c}to_kv`, as),
				cqn: f(`${c}q_rms_norm.gamma`), ckn: f(`${c}k_rms_norm.gamma`),
				cout: new Linear(m, `${P}${c}to_out`, as),
				proj: new Linear(m, `${P}${b}cross_attn.proj_linear`, as),
				fc1: new Linear(m, `${P}${b}mlp.mlp.0`, as),
				fc2: new Linear(m, `${P}${b}mlp.mlp.2`, as),
			});
			if (breathing()) await breathe();
		}
		this.attn = new Attention();
		this.s = new Scratch();
		llm.print(`  ${this.name} flow: ${this.depth} blocks x ${this.C}, ${this.heads} heads, ${this.linear} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		for (const b of this.blocks ?? []) disposeAll(b);
		disposeAll([this.t, this.ada, this.input, this.output, this.attn, this.s]);
		this.blocks = null;
	}

	// The conditioning a sampling run reuses: per block the cross-attention's
	// keys (head-normed) and values from `global` [5, ctx] (null: the negative,
	// zeros), and `proj` [n, projC] (null: zeros, so proj_linear is its bias).
	context(global, proj, n) {
		const C = this.C, L = 5;
		const zeros = global ? null : upload(new Float32Array(L * this.ctxC));
		const g = global ?? zeros;
		const blocks = this.blocks.map((b) => {
			const k = f32(L * C), v = f32(L * C);
			b.kv.part(g, k, L, 0, C);
			b.kv.part(g, v, L, C, C);
			op.headNormGamma(k, b.ckn, this.heads, this.hd, L);
			return { k, v };
		});
		submit();
		zeros?.dispose();
		return { blocks, proj, L, n, dispose() { disposeAll(blocks); } };
	}

	// cos/sin tables of the tokens' (i, j, k) cells [n, 3].
	rope(coords, n) { return ropeTables(coords, n, this.hd); }

	// The velocity v [n, out] at x [n, in] and time t (0..1; the model sees
	// 1000 t), with a context() and rope().
	forward(x, t, ctx, rope, v) {
		const C = this.C, n = ctx.n, F = this.ffn, nh = this.heads, hd = this.hd, s = this.s;
		const h = s.get('h', n * C), a = s.get('a', n * C), q = s.get('q', n * C), k = s.get('k', n * C), vv = s.get('v', n * C);
		const o = s.get('o', n * C), up = s.get('up', n * F), temb = s.get('temb', C), silu = s.get('silu', C);
		const gmod = s.get('gmod', 6 * C), mod = s.get('mod', 6 * C);
		this.input.run(x, h, n);
		this.t.run(1000 * t, temb);
		copy(temb, silu, C * 4);
		op.silu(silu, C);
		this.ada.run(silu, gmod, 1);
		for (const [l, b] of this.blocks.entries()) {
			copy(gmod, mod, 6 * C * 4);
			op.add(mod, b.mod, 6 * C);
			// self-attention: shift 0, scale C, gate 2C
			op.layerNorm(h, a, C, n, 1e-6);
			op.modulate(a, mod, a, n * C, C, C, 0);
			b.qkv.part(a, q, n, 0, C);
			b.qkv.part(a, k, n, C, C);
			b.qkv.part(a, vv, n, 2 * C, C);
			op.headNormGamma(q, b.qn, nh, hd, n);
			op.headNormGamma(k, b.kn, nh, hd, n);
			op.ropePairs(q, rope.cos, rope.sin, hd, nh, n);
			op.ropePairs(k, rope.cos, rope.sin, hd, nh, n);
			this.attn.run(q, k, vv, o, nh, hd, n, n);
			b.out.run(o, a, n);
			op.gatedAdd(h, mod, a, n * C, C, 2 * C);
			// cross-attention over the global tokens, plus the proj features' map
			op.layerNormAffine(h, b.n2w, b.n2b, a, C, n, 1e-6);
			b.q.run(a, q, n);
			op.headNormGamma(q, b.cqn, nh, hd, n);
			const c = ctx.blocks[l];
			this.attn.run(q, c.k, c.v, o, nh, hd, n, ctx.L);
			b.cout.run(o, a, n);
			op.add(h, a, n * C);
			if (ctx.proj) {
				b.proj.run(ctx.proj, a, n);
				op.add(h, a, n * C);
			} else {
				op.biasAdd(h, b.proj.b, n * C, C);
			}
			// MLP: shift 3C, scale 4C, gate 5C
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

	// FlowEulerGuidanceIntervalSampler from noise x [n, out] (a tensor,
	// overwritten with the sample): guidance (strength g, the negative being
	// zero conditioning) only while t is inside `interval`, rescaled toward the
	// positive prediction's std by `rescale`; the schedule warped by
	// `rescaleT`. `concat` [n, in - out] (the texture flow's shape latent) is
	// put beside x as the model's input. `onCall(i, t, v)` sees each model
	// call's output, for comparing with a reference.
	async sample(x, coords, cond, n, { steps = 12, guidance = 7.5, rescale = 0.7, interval = [0.6, 1], rescaleT = 5, concat = null, onCall = null, onStep = null } = {}) {
		const t0 = llm.now(), outN = n * this.outC;
		const pos = this.context(cond.global, cond.proj, n);
		const neg = guidance !== 1 ? this.context(null, null, n) : null;
		const rope = this.rope(coords, n);
		const v = f32(outN), vn = f32(outN);
		const input = concat ? f32(n * this.inC) : null;
		let calls = 0, guided = 0;
		const run = (t, ctx, out) => {
			if (concat) {
				op.concatRows(x, concat, input, n, this.outC, this.inC - this.outC);
				this.forward(input, t, ctx, rope, out);
			} else {
				this.forward(x, t, ctx, rope, out);
			}
			if (onCall) { submit(); onCall(calls, t, out); }
			calls++;
		};
		try {
			const warp = (u) => (rescaleT * u) / (1 + (rescaleT - 1) * u);
			for (let i = 0; i < steps; i++) {
				const t = warp(1 - i / steps), tPrev = warp(1 - (i + 1) / steps);
				run(t, pos, v);
				if (neg && t >= interval[0] && t <= interval[1]) {
					run(t, neg, vn);
					guided++;
					this.guide(x, v, vn, t, outN, guidance, rescale);
				}
				op.eulerStep(x, v, outN, -(t - tPrev));
				submit();
				if (breathing()) await breathe();
				if (onStep) await onStep(i + 1, steps);
			}
		} finally {
			submit();
			disposeAll([pos, neg, rope, v, vn, input]);
		}
		llm.print(`  ${this.name}: ${n} tokens, ${steps} steps (${guided} guided) in ${llm.since(t0)}`);
	}

	// v = g v + (1 - g) vn, then with `rescale` the implied x0 is pulled back
	// toward the positive one's std (over the whole latent). On the host: the
	// std is one number over every element.
	guide(x, v, vn, t, count, g, rescale) {
		if (!rescale) {
			// cfgCombine: vn + s (v - vn) into vn, s = g
			op.cfgCombine(v, vn, 1, count, g, false);
			copy(vn, v, count * 4);
			return;
		}
		submit();
		const xs = readFloats(x, count), p = readFloats(v, count), u = readFloats(vn, count);
		const a = 1 - SIGMA_MIN, b = SIGMA_MIN + (1 - SIGMA_MIN) * t;
		const x0p = new Float64Array(count), x0c = new Float64Array(count);
		for (let i = 0; i < count; i++) {
			const pred = g * p[i] + (1 - g) * u[i];
			x0p[i] = a * xs[i] - b * p[i];
			x0c[i] = a * xs[i] - b * pred;
		}
		const ratio = std(x0p) / std(x0c);
		const out = new Float32Array(count);
		for (let i = 0; i < count; i++) {
			const x0 = rescale * x0c[i] * ratio + (1 - rescale) * x0c[i];
			out[i] = (a * xs[i] - x0) / b;
		}
		v.buffer.write(out);
	}
}

// torch's std (unbiased).
function std(a) {
	let mean = 0;
	for (let i = 0; i < a.length; i++) mean += a[i];
	mean /= a.length;
	let s = 0;
	for (let i = 0; i < a.length; i++) s += (a[i] - mean) ** 2;
	return Math.sqrt(s / (a.length - 1));
}
