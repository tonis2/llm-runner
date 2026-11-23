// Hi3D's flow transformer (sparc3d's Sparc3DTransformer: TRELLIS's sparse
// structured-latent flow, at Hi3D's widths) and its Heun sampler.
//
// Over the voxels of a 128^3 grid: a linear 32 -> 256, a ResBlock there, a
// ResBlock that halves the grid (each 64^3 token its voxels' sum over their
// count plus one, as torch's scatter_reduce mean has it) and widens to 2112; 24 blocks over the tokens; then back: a ResBlock that copies
// each token to its voxels (its input the blocks' output beside the
// downsampling block's, 4224 wide), one over [that, the first ResBlock's
// output] (512), a plain layer norm and a linear 256 -> 32.
//
// A ResBlock: LayerNorm (affine), SiLU, 3^3 submanifold conv to half the output
// width, LayerNorm (no affine) times 1 + scale plus shift (both from the
// timestep embedding, scale first), SiLU, conv, SiLU, linear to the output
// width, plus the input (through a linear when the widths differ). Resampling
// comes first, on the input.
//
// A transformer block: adaLN-modulated self-attention (22 heads of 96; 3D RoPE
// on adjacent pairs, then per-head RMSNorm with a gamma, on q and k), plain
// cross-attention over the 2917 conditioning tokens [2917, 2048], then the
// modulated MLP (GELU, tanh). The modulation is shared: one adaLN linear of the
// timestep, plus each block's learned offset.
//
// The sampler (sparc3d's FlowMatchHeunScheduler and FlowMatchSchedule): Heun
// over sigma = 0.002 + 0.998 t, t warped twice (rational rescales by 2, then by
// a factor that grows with the voxel count), each step evaluated a little
// above its sigma (gamma 0.04 near t = 1, else 0.002) without adding noise.
// Each slope is guided at its own time from 0.7 up: scale 4 at t >= 1, else
// 10 t^2, against an all-zero condition.
//
// Ported from sparc3d/models/transformers/sparc3d_transformer.py,
// nn/sparse/transformer/modulated.py, nn/sparse/attention/modules.py,
// nn/rope.py, nn/sparse/spatial.py, schedulers/schedule.py,
// schedulers/flow_match_heun.py and pipelines/sparc3d_pipeline.py.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, breathe, breathing, copy } from '/core/gpu.js';
import { Linear, TimestepEmbedder, Attention, Scratch, ropeTables, upload, uploadU32, disposeAll, readFloats, rows } from '/plugins/lato2/nn.js';
import { neighbourTable } from '/plugins/pixal3d/decoder.js';

const NONE = 0xFFFFFFFF;
const f = Math.fround;

// A submanifold 3^3 conv, weight [out, 3, 3, 3, in] as stored, its patches
// gathered a slab of rows at a time through `table` (27 rows per voxel, NONE
// for an empty neighbour) out of x.
class SparseConv {
	constructor(m, name, as) {
		const d = m.dims(`${name}.weight`);
		this.outC = d[0];
		this.inC = d[4];
		this.w = m.upload(`${name}.weight`, as);
		this.b = m.upload(`${name}.bias`, 'f32');
	}
	run(x, y, n, table, scratch) {
		const width = 27 * this.inC;
		const slab = Math.max(1, Math.min(n, Math.floor(scratch.elements / width)));
		for (let first = 0; first < n; first += slab) {
			const count = Math.min(slab, n - first);
			op.gatherK(x, table.view(first * 27 * 4, count * 27 * 4), scratch, this.inC, 27, count);
			op.matmul(this.w, scratch, rows(y, this.outC, first, count), this.outC, width, count);
		}
		op.biasAdd(y, this.b, n * this.outC, this.outC);
	}
	dispose() { this.w.dispose(); this.b.dispose(); }
}

// SparseResBlock3d. `kind` 'plain', 'down' (input on the voxels, the block on
// the tokens) or 'up' (input on the tokens, the block on the voxels).
class ResBlock {
	constructor(m, name, as, kind) {
		this.kind = kind;
		this.n1w = m.upload(`${name}.norm1.weight`, 'f32');
		this.n1b = m.upload(`${name}.norm1.bias`, 'f32');
		this.inC = m.shape(`${name}.norm1.weight`)[0];
		this.conv1 = new SparseConv(m, `${name}.conv1.conv`, as);
		this.conv2 = new SparseConv(m, `${name}.conv2.conv`, as);
		this.hid = this.conv1.outC;
		this.emb = new Linear(m, `${name}.emb_layers.1`, as);
		this.linear = new Linear(m, `${name}.linear`, as);
		this.outC = this.linear.out;
		this.skip = m.has(`${name}.skip_connection.weight`) ? new Linear(m, `${name}.skip_connection`, as) : null;
	}

	// The block's scale and shift [2 * hid] from silu(t_emb) [C].
	modulation(siluT, s, key) {
		const mod = s.get(key, 2 * this.hid);
		this.emb.run(siluT, mod, 1);
		return mod;
	}

	// x [rows in, inC] to y [rows out, outC]; `st` the structure().
	run(x, y, st, mod, s) {
		const C = this.inC, H = this.hid, fine = st.n, coarse = st.m;
		let src = x, n, table;
		if (this.kind === 'down') {
			// torch's scatter_reduce mean counts the zero it starts from: the sum
			// over one more than the voxels
			src = s.get('rb.down', coarse * C);
			op.childMean(x, st.kids, src, C, coarse, 1);
			n = coarse;
			table = st.coarse;
		} else if (this.kind === 'up') {
			// Upsampling copies each token to its voxels, and norm1 and SiLU
			// are per row: both are done on the tokens, and conv1 gathers
			// through each voxel's neighbours' tokens.
			n = fine;
			table = st.fine;
		} else {
			n = fine;
			table = st.fine;
		}
		const rowsIn = this.kind === 'up' ? coarse : n;
		const a = s.get('rb.a', rowsIn * C), h1 = s.get('rb.h1', n * H), h2 = s.get('rb.h2', n * H);
		op.layerNormAffine(src, this.n1w, this.n1b, a, C, rowsIn, 1e-6);
		op.silu(a, rowsIn * C);
		if (this.kind === 'up') {
			// Every voxel takes its token's features, so conv1 is the tokens times
			// each tap's weights ([hid, 27, C] read as [hid * 27, C]) summed through
			// the voxels' neighbours' tokens: a quarter of gathering the voxels'
			// patches, and one well-shaped matmul.
			const z = s.get('rb.z', coarse * 27 * H);
			op.matmul(this.conv1.w, a, z, 27 * H, C, coarse);
			op.tapSum(z, st.upTable, h1, H, n);
			op.biasAdd(h1, this.conv1.b, n * H, H);
		} else {
			this.conv1.run(a, h1, n, table, s.get('conv', st.scratch));
		}
		op.layerNorm(h1, h2, H, n, 1e-6);
		op.modulate(h2, mod, h2, n * H, H, 0, H);
		op.silu(h2, n * H);
		this.conv2.run(h2, h1, n, table, s.get('conv', st.scratch));
		op.silu(h1, n * H);
		this.linear.run(h1, y, n);
		if (this.kind === 'up') {
			const sk = s.get('rb.skip', coarse * this.outC), up = s.get('rb.skipUp', n * this.outC);
			this.skip.run(src, sk, coarse);
			op.gatherK(sk, st.parentOf, up, this.outC, 1, n);
			op.add(y, up, n * this.outC);
		} else if (this.skip) {
			const sk = s.get('rb.skip', n * this.outC);
			this.skip.run(src, sk, n);
			op.add(y, sk, n * this.outC);
		} else {
			op.add(y, src, n * this.outC);
		}
	}

	dispose() { disposeAll([this.n1w, this.n1b, this.conv1, this.conv2, this.emb, this.linear, this.skip]); }
}

// The voxels' tokens on the half grid, as SparseDownsample orders them (x
// slowest, z fastest), with each voxel's token and each token's voxels.
export function tokensOf(coords, n) {
	const parent = new Int32Array(n), keys = new Map();
	const list = [];
	for (let i = 0; i < n; i++) {
		const x = coords[3 * i] >> 1, y = coords[3 * i + 1] >> 1, z = coords[3 * i + 2] >> 1;
		const key = (x * 1024 + y) * 1024 + z;
		if (!keys.has(key)) { keys.set(key, list.length); list.push(key); }
	}
	const sorted = list.slice().sort((p, q) => p - q);
	const rank = new Map();
	sorted.forEach((key, i) => rank.set(key, i));
	const m = sorted.length, tokens = new Int32Array(3 * m), kids = new Uint32Array(8 * m).fill(NONE), filled = new Uint8Array(m);
	sorted.forEach((key, i) => {
		tokens[3 * i] = Math.floor(key / (1024 * 1024));
		tokens[3 * i + 1] = Math.floor(key / 1024) % 1024;
		tokens[3 * i + 2] = key % 1024;
	});
	for (let i = 0; i < n; i++) {
		const x = coords[3 * i] >> 1, y = coords[3 * i + 1] >> 1, z = coords[3 * i + 2] >> 1;
		const p = rank.get((x * 1024 + y) * 1024 + z);
		parent[i] = p;
		kids[8 * p + filled[p]++] = i;
	}
	return { tokens, m, parent, kids };
}

// The sampler's times for `steps` updates over n voxels: sigmas [steps + 1]
// and the inflated times the model is evaluated at [steps], in float32 as
// torch has them.
export function schedule(steps, n, { sigmaMin = 0.002, rescale = 2 } = {}) {
	const warpBy = Math.max(1, Math.min(5, (19 * n - 20000) / (3 * n + 60000)));
	const rational = (t, w) => f(f(f(w) * t) / f(1 + f(f(w - 1) * t)));
	const sigmas = new Float32Array(steps + 1), warped = new Float32Array(steps + 1), hats = new Float32Array(steps);
	// torch.linspace(1, 0, steps + 1) in float32: from the start for the first
	// half, back from the end for the rest
	const points = steps + 1, step = f(-1 / steps), half = Math.floor(points / 2);
	for (let i = 0; i <= steps; i++) {
		const base = i < half ? f(1 + f(step * i)) : f(0 - f(step * (points - i - 1)));
		warped[i] = rational(rational(base, rescale), warpBy);
		sigmas[i] = f(f(sigmaMin) + f(f(1 - sigmaMin) * warped[i]));
	}
	for (let i = 0; i < steps; i++) hats[i] = f(sigmas[i] * f(1 + (warped[i] > f(0.985) ? f(0.04) : f(0.002))));
	return { sigmas, hats, warped, warp: warpBy };
}

// 4 at t >= 1, 10 t^2 from 0.7, else none (1).
export function guidanceScale(t) { return t >= 1 ? 4 : t >= f(0.7) ? f(10 * f(t * t)) : 1; }

export class Hi3DFlow {
	// `m` the open transformer_1024/model.safetensors, `config` its config.json.
	// `linear` 'f16' (the stored halves on the matrix cores) or 'f32'.
	constructor(m, config, { linear = 'f16' } = {}) {
		this.m = m;
		this.cfg = config;
		this.linear = linear;
		this.C = config.model_channels;
		this.heads = config.num_heads;
		this.hd = this.C / this.heads;
		this.depth = config.num_blocks;
		this.ctxC = config.cond_channels;
		this.inC = config.in_channels;
		this.outC = config.out_channels;
		this.ffn = m.shape('blocks.0.mlp.mlp.0.weight')[1];
	}

	async load() {
		const m = this.m, as = this.linear === 'f16' ? 'raw' : this.linear, t0 = llm.now();
		const fl = (n) => m.upload(n, 'f32');
		this.t = new TimestepEmbedder(m, 't_embedder', 'f32');
		this.ada = new Linear(m, 'adaLN_modulation.1', 'f32');
		this.input = new Linear(m, 'input_layer', 'f32');
		this.output = new Linear(m, 'out_layer', 'f32');
		this.io = {
			in0: new ResBlock(m, 'input_blocks.0', as, 'plain'),
			in1: new ResBlock(m, 'input_blocks.1', as, 'down'),
			out0: new ResBlock(m, 'out_blocks.0', as, 'up'),
			out1: new ResBlock(m, 'out_blocks.1', as, 'plain'),
		};
		this.blocks = [];
		for (let l = 0; l < this.depth; l++) {
			const b = `blocks.${l}.`;
			this.blocks.push({
				mod: fl(`${b}modulation`),
				qkv: new Linear(m, `${b}self_attn.to_qkv`, as),
				qn: fl(`${b}self_attn.q_rms_norm.gamma`), kn: fl(`${b}self_attn.k_rms_norm.gamma`),
				out: new Linear(m, `${b}self_attn.to_out`, as),
				n2w: fl(`${b}norm2.weight`), n2b: fl(`${b}norm2.bias`),
				q: new Linear(m, `${b}cross_attn.to_q`, as),
				kv: new Linear(m, `${b}cross_attn.to_kv`, as),
				cout: new Linear(m, `${b}cross_attn.to_out`, as),
				fc1: new Linear(m, `${b}mlp.mlp.0`, as),
				fc2: new Linear(m, `${b}mlp.mlp.2`, as),
			});
			if (breathing()) await breathe();
		}
		this.attn = new Attention();
		this.s = new Scratch();
		llm.print(`  Hi3D flow: ${this.depth} blocks x ${this.C}, ${this.heads} heads, ${this.linear} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		for (const b of this.blocks ?? []) disposeAll(b);
		disposeAll([this.t, this.ada, this.input, this.output, this.io, this.attn, this.s]);
		this.blocks = null;
	}

	// What every call over the same voxels (an Int32Array [n, 3] on the 128^3
	// grid) reuses: the voxels' and tokens' neighbour tables, the voxel ->
	// token map and back, conv1's table for the upsampling block (each voxel's
	// neighbours' tokens) and the tokens' RoPE.
	structure(coords, n) {
		const t0 = llm.now();
		const { tokens, m, parent, kids } = tokensOf(coords, n);
		let max = 0;
		for (const v of tokens) max = Math.max(max, v);
		if (max >= 64) throw new Error(`Hi3D flow: a token at ${max}; RoPE's NTK rescale (tokens past 63) is not supported`);
		const fine = neighbourTable(coords, n, 128);
		const up = new Uint32Array(fine.length);
		for (let i = 0; i < fine.length; i++) up[i] = fine[i] === NONE ? NONE : parent[fine[i]];
		const st = {
			n, m, tokens,
			fine: uploadU32(fine), coarse: uploadU32(neighbourTable(tokens, m, 64)), upTable: uploadU32(up),
			parentOf: uploadU32(Uint32Array.from(parent)), kids: uploadU32(kids),
			rope: ropeTables(tokens, m, this.hd),
			// gather scratch for the convs: 2^26 floats (256 MB)
			scratch: 1 << 26,
			dispose() { disposeAll([this.fine, this.coarse, this.upTable, this.parentOf, this.kids, this.rope]); },
		};
		llm.print(`  Hi3D flow: ${n} voxels, ${m} tokens, tables in ${llm.since(t0)}`);
		return st;
	}

	// The cross-attention's keys and values per block from cond [2917, 2048]
	// (a tensor); for the zero condition (null), whose keys are all the bias
	// and so whose attention is uniform, each block's output row is
	// to_out(bias of v), added as it is.
	context(cond, L = 2917) {
		const C = this.C;
		if (!cond) {
			const bv = f32(C);
			const rowsOut = this.blocks.map((b) => {
				copy(b.kv.b, bv, C * 4, C * 4);
				const r = f32(C);
				b.cout.run(bv, r, 1);
				return r;
			});
			submit();
			bv.dispose();
			return { negative: rowsOut, dispose() { disposeAll(rowsOut); } };
		}
		const blocks = this.blocks.map((b) => {
			const k = f32(L * C), v = f32(L * C);
			b.kv.part(cond, k, L, 0, C);
			b.kv.part(cond, v, L, C, C);
			return { k, v };
		});
		submit();
		return { blocks, L, dispose() { disposeAll(blocks); } };
	}

	// The velocity v [n, 32] at x [n, 32] and time t (the model sees 1000 t),
	// with a structure() and a context(). `grab(name, tensor, rows, width)`
	// sees intermediate tensors (for comparing with a reference). In a window
	// (the studio) a frame is drawn between blocks.
	async forward(x, t, st, ctx, v, grab = null) {
		const C = this.C, n = st.n, m = st.m, F = this.ffn, nh = this.heads, hd = this.hd, s = this.s, io = this.io;
		const temb = s.get('temb', C), silu = s.get('silu', C), gmod = s.get('gmod', 6 * C), mod = s.get('mod', 6 * C);
		this.t.run(f(1000 * f(t)), temb);
		copy(temb, silu, C * 4);
		op.silu(silu, C);
		this.ada.run(silu, gmod, 1);
		const mods = {};
		for (const [k, b] of Object.entries(io)) mods[k] = b.modulation(silu, s, `mod.${k}`);

		const h0 = s.get('h0', n * 256), skip0 = s.get('skip0', n * 256), skip1 = s.get('skip1', m * C);
		this.input.run(x, h0, n);
		io.in0.run(h0, skip0, st, mods.in0, s);
		io.in1.run(skip0, skip1, st, mods.in1, s);
		submit();
		if (grab) { submit(); grab('temb', temb, 1, C); grab('h0', h0, n, 256); grab('skip0', skip0, n, 256); }
		if (grab) { submit(); grab('h_in', skip1, m, C); }

		const h = s.get('h', m * C), a = s.get('a', m * C), q = s.get('q', m * C), k = s.get('k', m * C), vv = s.get('v', m * C);
		const o = s.get('o', m * C), up = s.get('up', m * F);
		copy(skip1, h, m * C * 4);
		for (const [l, b] of this.blocks.entries()) {
			copy(gmod, mod, 6 * C * 4);
			op.add(mod, b.mod, 6 * C);
			// self-attention: shift 0, scale C, gate 2C
			op.layerNorm(h, a, C, m, 1e-6);
			op.modulate(a, mod, a, m * C, C, C, 0);
			b.qkv.part(a, q, m, 0, C);
			b.qkv.part(a, k, m, C, C);
			b.qkv.part(a, vv, m, 2 * C, C);
			op.ropePairs(q, st.rope.cos, st.rope.sin, hd, nh, m);
			op.ropePairs(k, st.rope.cos, st.rope.sin, hd, nh, m);
			op.headNormGamma(q, b.qn, nh, hd, m);
			op.headNormGamma(k, b.kn, nh, hd, m);
			this.attn.run(q, k, vv, o, nh, hd, m, m);
			b.out.run(o, a, m);
			op.gatedAdd(h, mod, a, m * C, C, 2 * C);
			// cross-attention
			if (ctx.negative) {
				op.biasAdd(h, ctx.negative[l], m * C, C);
			} else {
				op.layerNormAffine(h, b.n2w, b.n2b, a, C, m, 1e-6);
				b.q.run(a, q, m);
				const c = ctx.blocks[l];
				this.attn.run(q, c.k, c.v, o, nh, hd, m, ctx.L);
				b.cout.run(o, a, m);
				op.add(h, a, m * C);
			}
			// MLP: shift 3C, scale 4C, gate 5C
			op.layerNorm(h, a, C, m, 1e-6);
			op.modulate(a, mod, a, m * C, C, 4 * C, 3 * C);
			b.fc1.run(a, up, m);
			op.gelu(up, m * F);
			b.fc2.run(up, a, m);
			op.gatedAdd(h, mod, a, m * C, C, 5 * C);
			if (grab && l === 0) { submit(); grab('h_block0', h, m, C); }
			// A block a submission: the whole pass as one runs past the
			// driver's hang timeout and the GPU is reset.
			submit();
			if (breathing()) await breathe();
		}
		if (grab) { submit(); grab('h_blocks', h, m, C); }

		const cat1 = s.get('cat1', m * 2 * C), y0 = s.get('y0', n * 256), cat0 = s.get('cat0', n * 512), y1 = s.get('y1', n * 256);
		op.concatRows(h, skip1, cat1, m, C, C);
		io.out0.run(cat1, y0, st, mods.out0, s);
		submit();
		op.concatRows(y0, skip0, cat0, n, 256, 256);
		io.out1.run(cat0, y1, st, mods.out1, s);
		op.layerNorm(y1, y0, 256, n, 1e-5);
		this.output.run(y0, v, n);
	}

	// The guided velocity at x and t into v (vn scratch): the negative only from
	// t = 0.7.
	async guided(x, t, st, pos, neg, v, vn, onCall) {
		const N = st.n * this.outC;
		await this.forward(x, t, st, pos, v);
		onCall?.(t, true, v);
		if (t < 0.7) return 1;
		await this.forward(x, t, st, neg, vn);
		onCall?.(t, false, vn);
		// vn + s (v - vn), into vn, then back into v
		op.cfgCombine(v, vn, 1, N, guidanceScale(t), false);
		copy(vn, v, N * 4);
		return 2;
	}

	// Heun from noise x [n, 32] (a tensor, overwritten with the denormalised
	// latent) over a structure(), conditioned on cond [2917, 2048] (a tensor).
	// `onCall(t, conditioned, v)` sees each model call's output (submitted);
	// `onStep(done, steps)` is awaited after each step.
	async sample(x, st, cond, { steps = 25, onCall = null, onStep = null } = {}) {
		const t0 = llm.now(), N = st.n * this.outC;
		const sch = schedule(steps, st.n);
		const pos = this.context(cond), neg = this.context(null);
		const v1 = f32(N), v2 = f32(N), vn = f32(N), pred = f32(N);
		const seen = onCall ? (t, c, v) => { submit(); onCall(t, c, v); } : null;
		let calls = 0;
		try {
			for (let i = 0; i < steps; i++) {
				const t = sch.hats[i], next = sch.sigmas[i + 1], dt = f(t - next);
				calls += await this.guided(x, t, st, pos, neg, v1, vn, seen);
				copy(x, pred, N * 4);
				op.eulerStep(pred, v1, N, -dt);
				calls += await this.guided(pred, next, st, pos, neg, v2, vn, seen);
				op.eulerStep(x, v1, N, f(-0.5 * dt));
				op.eulerStep(x, v2, N, f(-0.5 * dt));
				submit();
				if (breathing()) await breathe();
				if (onStep) await onStep(i + 1, steps);
			}
		} finally {
			submit();
			disposeAll([pos, neg, v1, v2, vn, pred]);
		}
		// the latent's per-channel statistics put back
		const lat = readFloats(x, N), mean = this.cfg.latent_mean, std = this.cfg.latent_std, C = this.outC;
		for (let i = 0; i < N; i++) lat[i] = f(f(lat[i] * f(std[i % C])) + f(mean[i % C]));
		x.buffer.write(lat);
		llm.print(`  Hi3D flow: ${st.n} voxels, ${steps} Heun steps (${calls} calls, warp ${sch.warp.toFixed(3)}) in ${llm.since(t0)}`);
		return lat;
	}
}
