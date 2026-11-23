// Hi3D's decoder (sparc3d's SparconvVAE: a PTv3MeshDecoder), from the flow's
// latent on the 128^3 voxels to per-corner signs, distances and deformations
// on the cells of a 1024^3 grid.
//
// A 3^3 conv 32 -> 512; at that level a ResBlock, two serialized attention
// blocks (windows of 48 points along a z-order curve, then along the curve
// with x and y swapped) and a ResBlock; then four stages (512, 512 -> 256,
// 256 -> 128, 128 channels) of two ResBlocks each, the first three ending in
// an upsampling; then GroupNorm, SiLU and a 3^3 conv to 8 sign logits and 32
// channels (8 distances, tanh; 24 deformations, tanh).
//
// - ResBlock: GroupNorm (32 groups, statistics over every point), SiLU, conv,
//   GroupNorm, SiLU, conv, plus the input (through a linear when the widths
//   differ). Above 12M points a ResBlock is skipped, as sparc3d skips it.
// - Attention block: x += LayerNorm(linear(conv(x))) (a conditional
//   position encoding); x += attention(LayerNorm(x)) (32 heads of 16, within
//   the windows; a window past the last full one is the curve's last 48
//   points); x += MLP(LayerNorm(x)) (GELU, erf).
// - Upsampling: each cell's 8 children (child k = 4x + 2y + z) are kept where
//   a head (GroupNorm, SiLU, conv to 8 logits) says > 0; a kept child's
//   features are its chunk of a linear C -> 8C (no bias).
//
// Ported from sparc3d/models/autoencoders/ptv3.py, blocks.py,
// sparconv_vae.py, structures/point.py and utils/serialization/.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, breathe, breathing } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { Linear, Scratch, uploadU32, disposeAll, readFloats, rows } from '/plugins/lato2/nn.js';
import { neighbourTable } from '/plugins/pixal3d/decoder.js';

const NONE = 0xFFFFFFFF;
const P = 'decoder.';
const SKIP_POINTS = 12_000_000;
const PATCH = 48;

// A submanifold 3^3 conv, weight [out, 3, 3, 3, in], patches gathered a slab
// of rows at a time. With `norm` (a GroupNorm and its stats) each gathered
// value is normalised and SiLU'd on the way, so the input needs no normalised
// copy; with `accumulate` the output is added into y instead of written.
class SparseConv {
	constructor(m, name, as) {
		const d = m.dims(`${name}.weight`);
		this.outC = d[0];
		this.inC = d[4];
		this.w = m.upload(`${name}.weight`, as);
		this.b = m.has(`${name}.bias`) ? m.upload(`${name}.bias`, 'f32') : null;
	}
	run(x, y, n, table, s, { norm = null, accumulate = false } = {}) {
		const width = 27 * this.inC, scratch = s.get('cols', 1 << 26);
		const slab = Math.max(1, Math.min(n, Math.floor(scratch.elements / width), accumulate ? Math.floor((1 << 24) / this.outC) : n));
		const tmp = accumulate ? s.get('conv.acc', slab * this.outC) : null;
		for (let first = 0; first < n; first += slab) {
			const count = Math.min(slab, n - first);
			const t = table.view(first * 27 * 4, count * 27 * 4);
			if (norm) op.gatherKGroupNorm(x, t, norm.stats, norm.gn.w, norm.gn.b, scratch, this.inC, 27, count, norm.gn.groups);
			else op.gatherK(x, t, scratch, this.inC, 27, count);
			const out = rows(y, this.outC, first, count);
			if (accumulate) {
				op.matmul(this.w, scratch, tmp, this.outC, width, count);
				if (this.b) op.biasAdd(tmp, this.b, count * this.outC, this.outC);
				op.add(out, tmp, count * this.outC);
			} else {
				op.matmul(this.w, scratch, out, this.outC, width, count);
			}
			submit();
		}
		if (this.b && !accumulate) op.biasAdd(y, this.b, n * this.outC, this.outC);
	}
	dispose() { this.w.dispose(); this.b?.dispose(); }
}

// sparc3d's PointGroupNorm: 32 groups (fewer when a group would have under 4
// channels), statistics over all the points; weight and bias [1, C]. `stats`
// makes them for a conv's gather to apply (SiLU after, as every use has it).
class GroupNorm {
	constructor(m, name, C) {
		this.C = C;
		this.groups = 4 * 32 > C ? Math.floor(C / 4) : 32;
		this.w = m.upload(`${name}.weights`, 'f32');
		this.b = m.upload(`${name}.bias`, 'f32');
	}
	stats(x, n, s) {
		const st = s.get('gn.stats', 2 * this.groups);
		op.groupStats(x, st, this.C, n, this.groups, 1e-6);
		return { gn: this, stats: st };
	}
	dispose() { this.w.dispose(); this.b.dispose(); }
}

// GroupNorm, SiLU, conv, GroupNorm, SiLU, conv, plus the input (through a
// linear when the widths differ). The norms are applied in the convs'
// gathers and the second conv is added into the shortcut, so a block at the
// 1024^3 level holds its input, the first conv's output and nothing else that
// size.
class ResBlock {
	constructor(m, name, inC, outC, as) {
		this.inC = inC;
		this.outC = outC;
		this.n1 = new GroupNorm(m, `${name}.block.0`, inC);
		this.c1 = new SparseConv(m, `${name}.block.2`, as);
		this.n2 = new GroupNorm(m, `${name}.block.3`, outC);
		this.c2 = new SparseConv(m, `${name}.block.5`, as);
		this.res = m.has(`${name}.res_conv.weight`) ? new Linear(m, `${name}.res_conv`, as) : null;
	}
	// x [n, inC] to the returned tensor [n, outC]: x itself (updated in place)
	// when the widths match or the block is skipped, else `out`.
	run(x, n, table, s, out) {
		if (n > SKIP_POINTS) return x;
		const h = s.get('rb.h', n * this.outC);
		this.c1.run(x, h, n, table, s, { norm: this.n1.stats(x, n, s) });
		const norm = this.n2.stats(h, n, s);
		let y = x;
		if (this.res) {
			y = out;
			this.res.run(x, y, n);
		}
		this.c2.run(h, y, n, table, s, { norm, accumulate: true });
		return y;
	}
	dispose() { disposeAll([this.n1, this.c1, this.n2, this.c2, this.res]); }
}

class AttentionBlock {
	constructor(m, name, C, as) {
		const f = (n) => m.upload(`${name}.${n}`, 'f32');
		this.C = C;
		this.heads = 32;
		this.cpeConv = new SparseConv(m, `${name}.cpe.0`, as);
		this.cpeLinear = new Linear(m, `${name}.cpe.1`, as);
		this.cpeW = f('cpe.2.weight');
		this.cpeB = f('cpe.2.bias');
		this.n1w = f('norm1.0.weight');
		this.n1b = f('norm1.0.bias');
		this.qkv = new Linear(m, `${name}.attn.qkv`, as);
		this.proj = new Linear(m, `${name}.attn.proj`, as);
		this.n2w = f('norm2.0.weight');
		this.n2b = f('norm2.0.bias');
		this.fc1 = new Linear(m, `${name}.mlp.0.fc1`, as);
		this.fc2 = new Linear(m, `${name}.mlp.0.fc2`, as);
	}
	// x [n, C] in place; `win` the curve's { order, span } (windows()).
	run(x, n, table, win, s) {
		const C = this.C, nh = this.heads, hd = C / nh;
		const a = s.get('at.a', n * C), b = s.get('at.b', n * C);
		this.cpeConv.run(x, a, n, table, s);
		this.cpeLinear.run(a, b, n);
		op.layerNormAffine(b, this.cpeW, this.cpeB, a, C, n, 1e-5);
		op.add(x, a, n * C);
		op.layerNormAffine(x, this.n1w, this.n1b, a, C, n, 1e-5);
		const q = s.get('at.q', n * C), k = s.get('at.k', n * C), v = s.get('at.v', n * C);
		this.qkv.part(a, q, n, 0, C);
		this.qkv.part(a, k, n, C, C);
		this.qkv.part(a, v, n, 2 * C, C);
		op.attentionWindow(q, k, v, win.order, win.span, a, nh, hd, n);
		this.proj.run(a, b, n);
		op.add(x, b, n * C);
		op.layerNormAffine(x, this.n2w, this.n2b, a, C, n, 1e-5);
		const up = s.get('at.up', n * 4 * C);
		this.fc1.run(a, up, n);
		op.geluErf(up, n * 4 * C);
		this.fc2.run(up, b, n);
		op.add(x, b, n * C);
	}
	dispose() { disposeAll(Object.values(this)); }
}

// z-order (Morton) keys of cells: bit i of x at 3i + 2, y at 3i + 1, z at 3i;
// `swap` exchanges x and y ("z-trans").
function mortonKeys(coords, n, depth, swap) {
	const keys = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		let x = coords[3 * i], y = coords[3 * i + 1];
		const z = coords[3 * i + 2];
		if (swap) [x, y] = [y, x];
		let key = 0;
		for (let b = 0; b < depth; b++) {
			const bit = 2 ** (3 * b);
			key += (((x >> b) & 1) * 4 + ((y >> b) & 1) * 2 + ((z >> b) & 1)) * bit;
		}
		keys[i] = key;
	}
	return keys;
}

// The attention windows along a curve: `order` the points sorted by key, and
// per point the [start, end) of its window in `order`. Windows of 48 along the
// curve; a point past the last full window is in the last 48 points' window
// (sparc3d pads the last window with the points before it).
export function windows(coords, n, swap) {
	let max = 0;
	for (let i = 0; i < 3 * n; i++) max = Math.max(max, coords[i]);
	const depth = Math.max(1, Math.ceil(Math.log2(max + 1)));
	const keys = mortonKeys(coords, n, depth, swap);
	const order = Uint32Array.from(Array.from({ length: n }, (_, i) => i).sort((a, b) => keys[a] - keys[b]));
	const span = new Uint32Array(2 * n);
	const full = n > PATCH ? Math.ceil(n / PATCH) - 1 : 0;
	for (let i = 0; i < n; i++) {
		const w = Math.floor(i / PATCH);
		let lo, hi;
		if (n <= PATCH) { lo = 0; hi = n; }
		else if (w < full) { lo = w * PATCH; hi = lo + PATCH; }
		else { lo = n - PATCH; hi = n; }
		span[2 * order[i]] = lo;
		span[2 * order[i] + 1] = hi;
	}
	return { order: uploadU32(order), span: uploadU32(span), dispose() { this.order.dispose(); this.span.dispose(); } };
}

class Upsample {
	constructor(m, name, C, as) {
		this.C = C;
		this.headNorm = new GroupNorm(m, `${name}.upsample.cls_head.fc.0`, C);
		this.head = new SparseConv(m, `${name}.upsample.cls_head.fc.2`, 'f32');
		this.proj = new Linear(m, `${name}.upsample.proj`, as);
	}
	dispose() { disposeAll([this.headNorm, this.head, this.proj]); }
}

export class Hi3DDecoder {
	// `path` the vae/model.safetensors; `linear` 'f32' or 'q8'.
	constructor(path, { linear = 'f32' } = {}) {
		this.path = path;
		this.as = linear;
	}

	load() {
		const m = this.m = openSafetensors(this.path), as = this.as, t0 = llm.now();
		this.zToF = new SparseConv(m, `${P}z_to_f.conv.0`, as);
		this.mid = {
			in: new ResBlock(m, `${P}mid.block_in`, 512, 512, as),
			attn: [new AttentionBlock(m, `${P}mid.block0`, 512, as), new AttentionBlock(m, `${P}mid.block1`, 512, as)],
			out: new ResBlock(m, `${P}mid.block_out`, 512, 512, as),
		};
		const widths = [[512, 512, 512], [512, 256, 256], [256, 128, 128], [128, 128, 128]];
		this.stages = [3, 2, 1, 0].map((s, i) => {
			const [inC, outC] = widths[i];
			return {
				blocks: [new ResBlock(m, `${P}dec.dec${s}.block0`, inC, outC, as), new ResBlock(m, `${P}dec.dec${s}.block1`, outC, outC, as)],
				up: s > 0 ? new Upsample(m, `${P}dec.dec${s}.up`, outC, as) : null,
				C: outC,
			};
		});
		this.outNorm = new GroupNorm(m, `${P}conv_out.conv.0`, 128);
		this.outConv = new SparseConv(m, `${P}conv_out.conv.2`, as);
		this.s = new Scratch();
		llm.print(`  Hi3D decoder: loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll([this.zToF, this.mid, this.stages, this.outNorm, this.outConv, this.s]);
		this.stages = null;
		this.m?.close();
	}

	// The latent [n, 32] (a tensor) on `coords` [n, 3] of the 128^3 grid. Returns
	// { coords [m, 3] at 1024^3, r, raw [m, 40] } on the host: per cell 8 sign
	// logits, then 8 distances and 24 deformations before their tanh (as
	// meshFlexiCubes takes them). `grab(name, tensor, rows, width, coords)` sees
	// each level's output.
	async run(latent, coords, { grab = null } = {}) {
		const s = this.s, t0 = llm.now();
		let n = coords.length / 3, cur = coords, r = 128, name = 'x0';
		let table = uploadU32(neighbourTable(cur, n, r));
		const tables = [table];
		try {
			let x = s.get(name, n * 512);
			this.zToF.run(latent, x, n, table, s);
			// two buffers in turn: a block's output goes to the one x is not
			const other = () => (name === 'x0' ? 'x1' : 'x0');
			const step = (y) => { if (y !== x) name = other(); x = y; };
			const spare = (count) => s.get(other(), count);
			step(this.mid.in.run(x, n, table, s, spare(n * 512)));
			for (const [i, blk] of this.mid.attn.entries()) {
				const win = windows(cur, n, i === 1);
				blk.run(x, n, table, win, s);
				submit();
				win.dispose();
			}
			step(this.mid.out.run(x, n, table, s, spare(n * 512)));
			submit();
			if (grab) grab('mid', x, n, 512, cur);
			for (const [i, st] of this.stages.entries()) {
				for (const b of st.blocks) {
					step(b.run(x, n, table, s, spare(n * st.C)));
					submit();
					if (breathing()) await breathe();
				}
				if (!st.up) {
					if (grab) grab(`dec${3 - i}`, x, n, st.C, cur);
					break;
				}
				// children where the head's logit is > 0
				const U = st.up, C = st.C;
				const logit = s.get('up.logit', n * 8);
				U.head.run(x, logit, n, table, s, { norm: U.headNorm.stats(x, n, s) });
				submit();
				const lg = readFloats(logit, n * 8);
				let m = 0;
				for (let k = 0; k < n * 8; k++) if (lg[k] > 0) m++;
				// The proj (C -> 8C) a slab of parents at a time, its kept children
				// gathered out: `rel` is a child's chunk (k = 4x + 2y + z) in its
				// slab's output. Neighbour codes use child_neighbours' sub-cells (x
				// in bit 0).
				const per = Math.max(1, Math.floor((1 << 26) / (8 * C)));
				const rel = new Uint32Array(m), codes = new Uint32Array(m), kids = new Uint32Array(n * 8).fill(NONE), next = new Int32Array(m * 3);
				const start = new Uint32Array(n + 1);
				for (let p = 0, j = 0; p < n; p++) {
					start[p] = j;
					const p0 = p - (p % per);
					for (let k = 0; k < 8; k++) {
						if (!(lg[p * 8 + k] > 0)) continue;
						const dx = (k >> 2) & 1, dy = (k >> 1) & 1, dz = k & 1, sc = dx | (dy << 1) | (dz << 2);
						rel[j] = (p - p0) * 8 + k;
						codes[j] = p * 8 + sc;
						kids[p * 8 + sc] = j;
						next[3 * j] = 2 * cur[3 * p] + dx;
						next[3 * j + 1] = 2 * cur[3 * p + 1] + dy;
						next[3 * j + 2] = 2 * cur[3 * p + 2] + dz;
						j++;
					}
					if (p === n - 1) start[n] = j;
				}
				const g = uploadU32(rel), code = uploadU32(codes), kid = uploadU32(kids), table2 = three.compute.u32(m * 27);
				op.childNeighbours(table, kid, code, table2, m);
				const y = spare(m * C), wide = s.get('up.wide', Math.min(per, n) * 8 * C);
				for (let p0 = 0; p0 < n; p0 += per) {
					const count = Math.min(per, n - p0), j0 = start[p0], j1 = start[p0 + count];
					if (j1 === j0) continue;
					U.proj.run(rows(x, C, p0, count), wide, count);
					op.c2sGather(wide, g.view(j0 * 4, (j1 - j0) * 4), rows(y, C, j0, j1 - j0), j1 - j0, C, 8 * C, C, 1);
					submit();
				}
				disposeAll([g, code, kid]);
				table.dispose();
				tables.length = 0;
				table = table2;
				tables.push(table);
				step(y);
				n = m;
				cur = next;
				r *= 2;
				if (grab) grab(`dec${3 - i}`, x, n, C, cur);
			}
			// GroupNorm, SiLU and the conv to 8 sign logits and 32 fields
			const out = s.get('out.y', n * 40);
			this.outConv.run(x, out, n, table, s, { norm: this.outNorm.stats(x, n, s) });
			submit();
			const raw = readFloats(out, n * 40);
			llm.print(`  Hi3D decoder: ${coords.length / 3} voxels to ${n} cells at ${r}^3 in ${llm.since(t0)}`);
			return { coords: cur, r, raw };
		} finally {
			submit();
			disposeAll(tables);
		}
	}
}
