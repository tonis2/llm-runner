// LATO.2's V-VAE decoder (VertexVAE.decode at inference) and offset head:
// the vertex latent on the 64^3 active voxels in, vertex positions out.
//
// The latent is expanded by a windowed (swin) sparse transformer, the voxels
// that hold vertices are kept, and four levels then each split every point
// into its eight children (64 -> 128 -> ... -> 1024), run sparse 3x3x3
// convolutions and attention over them and prune the children that hold no
// vertex. The survivors at 1024^3 are the vertices; the offset head places
// each inside its voxel.
//
// Ported from LATO.2's `models/vertex_autoencoder.py` with
// `modules/sparse/transformer/` (SparseTransformerBase, SparseTransformerCrossBase)
// and the spconv submanifold convolution (weights [out, 3, 3, 3, in],
// cross-correlation over the voxel coordinates).

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, breathe, breathing } from '/core/gpu.js';
import { openTorch } from '/core/weights/torch.js';
import { Linear, Attention, Scratch, ropeTables, sincosPositions, upload, uploadU32, disposeAll, readFloats, rows } from './nn.js';

const NONE = 0xFFFFFFFF;

// ---- point sets on the host

// Each point's 27 neighbours (offsets -1..1 per axis, x slowest) as row
// indices, NONE where there is no point: a submanifold convolution's table.
export function neighbours(coords, n, R) {
	const at = new Map();
	for (let i = 0; i < n; i++) at.set((coords[3 * i] * R + coords[3 * i + 1]) * R + coords[3 * i + 2], i);
	const out = new Uint32Array(n * 27);
	for (let i = 0; i < n; i++) {
		const x = coords[3 * i], y = coords[3 * i + 1], z = coords[3 * i + 2];
		let k = 0;
		for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++, k++) {
			const px = x + a, py = y + b, pz = z + c;
			const j = px < 0 || py < 0 || pz < 0 || px >= R || py >= R || pz >= R ? undefined : at.get((px * R + py) * R + pz);
			out[i * 27 + k] = j === undefined ? NONE : j;
		}
	}
	return out;
}

// The swin partition: points grouped by the `size`^3 window their coordinates
// plus `shift` fall in. `order` lists the points window by window, `span`
// gives each point its window's [start, end) in it.
export function windows(coords, n, size, shift) {
	const groups = new Map();
	for (let i = 0; i < n; i++) {
		const key = ((Math.floor((coords[3 * i] + shift) / size) * 4096 + Math.floor((coords[3 * i + 1] + shift) / size)) * 4096)
			+ Math.floor((coords[3 * i + 2] + shift) / size);
		let g = groups.get(key);
		if (!g) groups.set(key, (g = []));
		g.push(i);
	}
	const order = new Uint32Array(n), span = new Uint32Array(2 * n);
	let at = 0, largest = 0;
	for (const g of groups.values()) {
		const start = at;
		for (const i of g) order[at++] = i;
		for (const i of g) { span[2 * i] = start; span[2 * i + 1] = at; }
		largest = Math.max(largest, g.length);
	}
	if (largest > 1024) throw new Error(`V-VAE: a window of ${largest} points (the kernel takes 1024)`);
	return { order, span, count: groups.size };
}

// ---- layers over point rows

// A submanifold sparse convolution: weight [out, k^3 * in] (spconv's
// [out, k, k, k, in] flattened), patches gathered by a neighbours() table.
class SparseConv {
	constructor(m, name) {
		const d = m.dims(`${name}.weight`);
		this.outC = d[0];
		this.k = d[1];
		this.inC = d[4];
		this.w = m.upload(`${name}.weight`, 'f32');
		this.b = m.upload(`${name}.bias`, 'f32');
	}
	run(x, y, n, table, scratch) {
		const kk = this.k ** 3, width = kk * this.inC;
		if (kk === 1) {
			op.matmul(this.w, x, y, this.outC, this.inC, n);
		} else {
			const slab = Math.max(1, Math.min(n, Math.floor(scratch.elements / width)));
			for (let first = 0; first < n; first += slab) {
				const count = Math.min(slab, n - first);
				op.gatherK(x, table.view(first * 27 * 4, count * 27 * 4), scratch, this.inC, 27, count);
				op.matmul(this.w, scratch, rows(y, this.outC, first, count), this.outC, width, count);
			}
		}
		op.biasAdd(y, this.b, n * this.outC, this.outC);
	}
	dispose() { this.w.dispose(); this.b.dispose(); }
}

// Linear, GELU (tanh), Linear: TRELLIS's feed-forward and LATO.2's heads.
class Mlp {
	constructor(m, name) {
		this.l0 = new Linear(m, `${name}.0`);
		this.l2 = new Linear(m, `${name}.2`);
	}
	run(x, y, n, s) {
		const mid = s.get('mlp', n * this.l0.out);
		this.l0.run(x, mid, n);
		op.gelu(mid, n * this.l0.out);
		this.l2.run(mid, y, n);
	}
	dispose() { this.l0.dispose(); this.l2.dispose(); }
}

// A sparse transformer's blocks over point rows [n, C]: pre-norm (no affine)
// self-attention, optional cross-attention to a context, and the MLP. Self-
// attention is full, or windowed (swin: `windows` alternate by block), with
// 3D RoPE when `rope` tables are given.
class SparseTransformer {
	constructor(m, prefix, heads, { cross = false } = {}) {
		this.input = new Linear(m, `${prefix}.input_layer`);
		this.C = this.input.out;
		this.heads = heads;
		this.hd = this.C / heads;
		this.cross = cross;
		const attn = cross ? 'self_attn' : 'attn';
		this.blocks = [];
		for (let l = 0; m.has(`${prefix}.blocks.${l}.${attn}.to_qkv.weight`); l++) {
			const p = `${prefix}.blocks.${l}`;
			this.blocks.push({
				qkv: new Linear(m, `${p}.${attn}.to_qkv`), out: new Linear(m, `${p}.${attn}.to_out`),
				q: cross ? new Linear(m, `${p}.cross_attn.to_q`) : null,
				kv: cross ? new Linear(m, `${p}.cross_attn.to_kv`) : null,
				cout: cross ? new Linear(m, `${p}.cross_attn.to_out`) : null,
				mlp: new Mlp(m, `${p}.mlp.mlp`),
			});
		}
	}

	// Context keys and values per block for rows ctx [L, ctxC].
	context(ctx, L) {
		const C = this.C;
		return this.blocks.map((b) => {
			const k = f32(L * C), v = f32(L * C);
			b.kv.part(ctx, k, L, 0, C);
			b.kv.part(ctx, v, L, C, C);
			return { k, v, L, dispose() { k.dispose(); v.dispose(); } };
		});
	}

	// x [n, in] -> h [n, C]. `pos` (rows [n, C]) is added after the input
	// layer; `rope` are ropeTables; `win` [{ order, span }] per block parity.
	run(x, h, n, { s, attn, pos = null, rope = null, win = null, ctx = null }) {
		const C = this.C, nh = this.heads, hd = this.hd;
		const a = s.get('ta', n * C), q = s.get('tq', n * C), k = s.get('tk', n * C), v = s.get('tv', n * C), o = s.get('to', n * C);
		this.input.run(x, h, n);
		if (pos) op.add(h, pos, n * C);
		for (const [l, b] of this.blocks.entries()) {
			op.layerNorm(h, a, C, n, 1e-6);
			b.qkv.part(a, q, n, 0, C);
			b.qkv.part(a, k, n, C, C);
			b.qkv.part(a, v, n, 2 * C, C);
			if (rope) {
				op.ropePairs(q, rope.cos, rope.sin, hd, nh, n, true);
				op.ropePairs(k, rope.cos, rope.sin, hd, nh, n);
			}
			if (win) {
				const w = win[l % 2];
				op.attentionWindow(q, k, v, w.order, w.span, o, nh, hd, n);
			} else {
				attn.run(q, k, v, o, nh, hd, n, n);
			}
			b.out.run(o, a, n);
			op.add(h, a, n * C);
			if (this.cross) {
				op.layerNorm(h, a, C, n, 1e-6);
				b.q.run(a, q, n);
				attn.run(q, ctx[l].k, ctx[l].v, o, nh, hd, n, ctx[l].L);
				b.cout.run(o, a, n);
				op.add(h, a, n * C);
			}
			op.layerNorm(h, a, C, n, 1e-6);
			b.mlp.run(a, o, n, s);
			op.add(h, o, n * C);
		}
	}
	dispose() { disposeAll([this.input, this.blocks]); }
}

// ---- the decoder

export class VertexDecoder {
	constructor(vvaePath, offsetPath) {
		this.m = openTorch(vvaePath);
		this.mo = openTorch(offsetPath);
		const a = this.m.pickle.get('config').get('args');
		this.levels = a.get('decoder_cfg').map((c) => Object.fromEntries(c));
		this.expander = Object.fromEntries(a.get('expander_cfg'));
		this.window = a.get('window_size');
		if (a.get('attn_mode') !== 'swin' || a.get('pe_mode') !== 'rope') throw new Error('V-VAE: only the swin / rope decoder is ported');
		this.latentDim = a.get('latent_dim');
		this.resolution = a.get('resolution');
	}

	load() {
		const m = this.m, t0 = llm.now();
		this.expand = new SparseTransformer(m, 'latent_expander', this.expander.num_heads);
		this.vtxHead = new Mlp(m, 'vtx_pruning_head.mlp');
		this.vtxProj = new Linear(m, 'vtx_proj');
		this.level = this.levels.map((c, i) => {
			const p = `decoder_vtx.${i}.upsample`;
			return {
				cfg: c,
				gnW: m.upload(`${p}.act_layers.0.weight`, 'f32'), gnB: m.upload(`${p}.act_layers.0.bias`, 'f32'),
				conv1: new SparseConv(m, `${p}.out_layers.0.conv`),
				gn2W: m.upload(`${p}.out_layers.1.weight`, 'f32'), gn2B: m.upload(`${p}.out_layers.1.bias`, 'f32'),
				conv2: new SparseConv(m, `${p}.out_layers.3.conv`),
				skip: m.has(`${p}.skip_connection.conv.weight`) ? new SparseConv(m, `${p}.skip_connection.conv`) : null,
				ca: new SparseTransformer(m, `${p}.ca`, c.num_heads, { cross: true }),
				projCtx: new Linear(m, `${p}.proj_ctx`),
				projOut: new Linear(m, `${p}.proj_out`),
				prune: new Mlp(m, `${p}.pruning_head.mlp`),
				latentProj: new Linear(m, `latent_proj.${i}`),
				vtxCa: new SparseTransformer(m, `decoder_vtx_ca.${i}`, c.num_heads, { cross: true }),
			};
		});
		this.offset = new Mlp(this.mo, 'mlp');
		this.attn = new Attention();
		this.s = new Scratch();
		llm.print(`  V-VAE decoder: ${this.levels.length} levels to ${this.resolution}^3, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll([this.expand, this.vtxHead, this.vtxProj, this.level, this.offset, this.attn, this.s]);
		this.m.close();
		this.mo.close();
	}

	// Which children survive: logits [8 * parents], a parent's eight in a row.
	// Children at `threshold` (a logit) or over stay; a parent none of whose
	// children would is given its likeliest one.
	static survivors(logits, parents, threshold) {
		const keep = [];
		for (let p = 0; p < parents; p++) {
			let any = false, best = 0;
			for (let c = 0; c < 8; c++) {
				const l = logits[p * 8 + c];
				if (l >= threshold) { keep.push(p * 8 + c); any = true; }
				if (l > logits[p * 8 + best]) best = c;
			}
			if (!any) keep.push(p * 8 + best);
		}
		return keep;
	}

	// latent [n, 32] (a GPU tensor) on the 64^3 points `coords` (Int32Array
	// [n * 3]) -> { coords Int32Array [V * 3] at 1024^3, offsets Float32Array
	// [V * 3] in (-1, 1), stats }. `golden` (optional) records intermediate
	// values for a test.
	async decode(latent, coords, n, { threshold = 0.5, record = null } = {}) {
		const s = this.s, attn = this.attn, t0 = llm.now();
		const logit = Math.log(threshold / (1 - threshold));
		const keep = [];
		const stats = { points: [n] };
		try {
			// the latent expanded over windows, the vertex voxels kept
			const E = this.expand.C;
			const ropeE = ropeTables(coords, n, this.expand.hd);
			keep.push(ropeE);
			const win = [0, 1].map((par) => {
				const w = windows(coords, n, this.window, par * (this.window / 2));
				const g = { order: uploadU32(w.order), span: uploadU32(w.span), dispose() { this.order.dispose(); this.span.dispose(); } };
				keep.push(g);
				return g;
			});
			const ex = s.get('expanded', n * E);
			this.expand.run(latent, ex, n, { s, attn, rope: ropeE, win });
			const vl = s.get('vtxLogit', n);
			this.vtxHead.run(ex, vl, n, s);
			submit();
			if (record) record.expanded = readFloats(ex, n * E);
			const scores = readFloats(vl, n);
			if (record) record.vtxLogits = scores;
			let kept = [];
			for (let i = 0; i < n; i++) if (scores[i] >= logit) kept.push(i);
			if (kept.length === 0) kept = [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, Math.min(2, n));
			let pts = kept.length;
			let cur = new Int32Array(pts * 3);
			kept.forEach((i, j) => cur.set(coords.subarray(3 * i, 3 * i + 3), 3 * j));
			const idx = uploadU32(kept);
			keep.push(idx);
			const picked = s.get('picked', pts * E);
			op.gatherK(ex, idx, picked, E, 1, pts);
			let C = this.vtxProj.out;
			let x = s.get('x', pts * C);
			this.vtxProj.run(picked, x, pts);
			stats.points.push(pts);

			let R = 64;
			for (const [i, L] of this.level.entries()) {
				const out = L.conv1.outC, n8 = pts * 8, R2 = R * 2;
				// norm, then GroupNorm + SiLU, each split into its eight children
				const xn = s.get('xn', pts * C), h = s.get('h', pts * C);
				op.layerNorm(x, xn, C, pts, 1e-5);
				op.groupNormRows(xn, L.gnW, L.gnB, h, C, pts, 32, 1e-5, true);
				const parent = new Uint32Array(n8), sub = new Int32Array(n8 * 3);
				for (let p = 0; p < pts; p++) for (let c = 0; c < 8; c++) {
					parent[p * 8 + c] = p;
					sub[(p * 8 + c) * 3] = 2 * cur[3 * p] + ((c >> 2) & 1);
					sub[(p * 8 + c) * 3 + 1] = 2 * cur[3 * p + 1] + ((c >> 1) & 1);
					sub[(p * 8 + c) * 3 + 2] = 2 * cur[3 * p + 2] + (c & 1);
				}
				const par = uploadU32(parent), table = uploadU32(neighbours(sub, n8, R2));
				const hs = s.get('hs', n8 * C), xs = s.get('xs', n8 * C);
				op.gatherK(h, par, hs, C, 1, n8);
				op.gatherK(xn, par, xs, C, 1, n8);
				// the convolutions, and the shortcut
				const cols = s.get('cols', 1 << 26), c1 = s.get('c1', n8 * out), c2 = s.get('c2', n8 * out);
				L.conv1.run(hs, c1, n8, table, cols);
				op.groupNormRows(c1, L.gn2W, L.gn2B, c1, out, n8, 32, 1e-5, true);
				L.conv2.run(c1, c2, n8, table, cols);
				if (L.skip) {
					L.skip.run(xs, c1, n8, null, cols);
					op.add(c2, c1, n8 * out);
				} else {
					op.add(c2, xs, n8 * out);
				}
				// attention over the children, positions absolute; then the pruning
				const M = L.ca.C, pos = upload(sincosPositions(sub, n8, M));
				const ctxRows = s.get('ctxRows', n8 * M);
				L.projCtx.run(c2, ctxRows, n8);
				op.add(ctxRows, pos, n8 * M);
				const ctx = L.ca.context(ctxRows, n8);
				const ca = s.get('ca', n8 * M);
				L.ca.run(c2, ca, n8, { s, attn, pos, ctx });
				L.projOut.run(ca, c1, n8);
				const occ = s.get('occ', n8);
				L.prune.run(c1, occ, n8, s);
				submit();
				disposeAll([pos, ctx, par, table]);
				const logits = readFloats(occ, n8);
				const kids = VertexDecoder.survivors(logits, pts, logit);
				const next = kids.length;
				const kidIdx = uploadU32(kids);
				const y = s.get('y', next * out);
				op.gatherK(c1, kidIdx, y, out, 1, next);
				cur = new Int32Array(next * 3);
				kids.forEach((k, j) => cur.set(sub.subarray(3 * k, 3 * k + 3), 3 * j));
				// attention over the survivors, RoPE, and to the latent
				const lat = s.get('lat', n * L.latentProj.out);
				L.latentProj.run(latent, lat, n);
				const lctx = L.vtxCa.context(lat, n);
				const rope = ropeTables(cur, next, L.vtxCa.hd);
				const z = s.get('x', next * L.vtxCa.C);
				L.vtxCa.run(y, z, next, { s, attn, rope, ctx: lctx });
				submit();
				disposeAll([kidIdx, lctx, rope]);
				if (record) record[`l${i}`] = { coords: cur, feats: readFloats(z, next * L.vtxCa.C), logits, kept: kids };
				x = z;
				C = L.vtxCa.C;
				pts = next;
				R = R2;
				stats.points.push(pts);
				if (breathing()) await breathe();
			}
			// offsets in each final voxel
			const off = s.get('off', pts * 3);
			this.offset.run(x, off, pts, s);
			submit();
			const offsets = readFloats(off, pts * 3).map(Math.tanh);
			llm.print(`  V-VAE: points ${stats.points.join(' -> ')} at ${R}^3, in ${llm.since(t0)}`);
			return { coords: cur, offsets, resolution: R, stats };
		} finally {
			submit();
			disposeAll(keep);
		}
	}
}
