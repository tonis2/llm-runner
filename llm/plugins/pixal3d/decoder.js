// TRELLIS.2's shape decoder as Pixal3D ships it (FlexiDualGridVaeDecoder,
// "next_dc_f16c32"): a sparse UNet from the shape latent [n, 32] on its 32^3
// (or 64^3) voxels up to four times finer. Each level is ConvNeXt blocks (a
// submanifold 3x3x3 conv, LayerNorm, an MLP, the residual), then a
// channel-to-spatial block that predicts which of each voxel's 8 children
// exist and hands the kept ones its channels. The last level's 7 channels are
// FlexiDualGrid's: a vertex offset in the voxel, three edge-crossing flags
// and a quad-split weight, which mesh.js turns into triangles.
//
// Ported from Pixal3D's models/sc_vaes/sparse_unet_vae.py
// (SparseConvNeXtBlock3d, SparseResBlockC2S3d, SparseUnetVaeDecoder),
// models/sc_vaes/fdg_vae.py and modules/sparse/spatial/spatial2channel.py.
// Voxels are rows; a child's sub-cell s has its x offset in bit 0, y in bit 1
// and z in bit 2, and children follow their parent's order.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, breathe, breathing } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { Linear, Scratch, uploadU32, disposeAll, readFloats, rows } from '/plugins/lato2/nn.js';

const NONE = 0xFFFFFFFF;

// Voxel cells (an Int32Array [n, 3] on an r^3 grid) to row indices, in typed
// arrays: open addressing on (x r + y) r + z.
export class CellIndex {
	constructor(coords, n, r) {
		let size = 1;
		while (size < 2 * n) size <<= 1;
		this.mask = size - 1;
		this.r = r;
		this.keys = new Int32Array(size).fill(-1);
		this.vals = new Int32Array(size);
		for (let i = 0; i < n; i++) this.insert((coords[3 * i] * r + coords[3 * i + 1]) * r + coords[3 * i + 2], i);
	}
	slot(key) {
		let h = Math.imul(key, 0x9E3779B1) & this.mask;
		while (this.keys[h] !== -1 && this.keys[h] !== key) h = (h + 1) & this.mask;
		return h;
	}
	insert(key, v) {
		const h = this.slot(key);
		this.keys[h] = key;
		this.vals[h] = v;
	}
	// The row of cell (x, y, z), or -1.
	get(x, y, z) {
		const r = this.r;
		if (x < 0 || y < 0 || z < 0 || x >= r || y >= r || z >= r) return -1;
		const h = this.slot((x * r + y) * r + z);
		return this.keys[h] === -1 ? -1 : this.vals[h];
	}
}

// A submanifold conv's table: each voxel's 27 neighbours (offsets -1..1, x
// slowest), NONE where there is no voxel.
export function neighbourTable(coords, n, r) {
	const at = new CellIndex(coords, n, r), out = new Uint32Array(n * 27);
	for (let i = 0; i < n; i++) {
		const x = coords[3 * i], y = coords[3 * i + 1], z = coords[3 * i + 2];
		let k = i * 27;
		for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
			const j = at.get(x + a, y + b, z + c);
			out[k++] = j < 0 ? NONE : j;
		}
	}
	return out;
}

// A submanifold 3x3x3 conv, weight [out, 3, 3, 3, in] as stored (F16 on the
// matrix cores, else f32), patches gathered a slab of rows at a time.
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

export class ShapeDecoder {
	// `dir` is Pixal3D's ckpts/ (shape_dec_next_dc_f16c32_fp16.{json,safetensors}).
	// `linear` 'f16' runs the weights as stored, on the matrix cores.
	constructor(dir, { linear = 'f16', name = 'shape_dec_next_dc_f16c32_fp16' } = {}) {
		this.cfg = JSON.parse(llm.readText(`${dir}/${name}.json`)).args;
		this.path = `${dir}/${name}.safetensors`;
		this.as = linear === 'f16' ? 'raw' : linear;
	}

	load() {
		const m = this.m = openSafetensors(this.path), as = this.as, c = this.cfg, t0 = llm.now();
		const f = (n) => m.upload(n, 'f32');
		this.fromLatent = new Linear(m, 'from_latent', 'f32');
		this.levels = [];
		for (const [i, C] of c.model_channels.entries()) {
			const L = { C, blocks: [], up: null };
			for (let j = 0; j < c.num_blocks[i]; j++) {
				const p = `blocks.${i}.${j}`;
				L.blocks.push({
					conv: new SparseConv(m, `${p}.conv`, as),
					nw: f(`${p}.norm.weight`), nb: f(`${p}.norm.bias`),
					fc1: new Linear(m, `${p}.mlp.0`, as), fc2: new Linear(m, `${p}.mlp.2`, as),
				});
			}
			if (i < c.model_channels.length - 1) {
				const p = `blocks.${i}.${c.num_blocks[i]}`;
				L.up = {
					out: c.model_channels[i + 1],
					subdiv: new Linear(m, `${p}.to_subdiv`, 'f32'),
					n1w: f(`${p}.norm1.weight`), n1b: f(`${p}.norm1.bias`),
					conv1: new SparseConv(m, `${p}.conv1`, as),
					conv2: new SparseConv(m, `${p}.conv2`, as),
				};
			}
			this.levels.push(L);
		}
		this.output = new Linear(m, 'output_layer', 'f32');
		this.s = new Scratch();
		llm.print(`  shape decoder: ${c.model_channels.join('/')} channels, ${c.num_blocks.join('+')} blocks, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll([this.fromLatent, this.levels, this.output, this.s]);
		this.levels = null;
		this.m?.close();
	}

	// One ConvNeXt block over x [n, C] in place.
	convnext(b, x, n, C, table) {
		const s = this.s, cols = s.get('cols', 1 << 26);
		const h = s.get('h', n * C), a = s.get('a', n * C), up = s.get('up', n * 4 * C);
		b.conv.run(x, h, n, table, cols);
		op.layerNormAffine(h, b.nw, b.nb, a, C, n, 1e-6);
		b.fc1.run(a, up, n);
		op.silu(up, n * 4 * C);
		b.fc2.run(up, h, n);
		op.add(x, h, n * C);
	}

	// The latent [n, 32] (a tensor) on `coords` [n, 3] of an r^3 grid, through
	// the UNet. `upsampleOnly` (1 .. 4) stops after that many subdivisions and
	// returns { coords, r }: the voxels there (the cascade's next coordinates). Otherwise returns { coords, r,
	// fields, table }: fields [m, 7] and the last level's neighbour table
	// [m, 27] on the host, and with `keepSubs` each level's subdivision logits
	// (for comparing with the reference).
	//
	// Only the first level's neighbour table is made on the host; each next
	// one comes from its parents' on the GPU (child_neighbours.shady).
	async run(latent, coords, r, { upsampleOnly = 0, keepSubs = false } = {}) {
		const s = this.s, t0 = llm.now();
		let n = coords.length / 3, C = this.levels[0].C, cur = coords;
		let x = s.get('x0', n * C), flip = 0;
		this.fromLatent.run(latent, x, n);
		let table = uploadU32(neighbourTable(cur, n, r));
		const subs = [];
		try {
			for (const [i, L] of this.levels.entries()) {
				const tb = llm.now();
				for (const b of L.blocks) {
					this.convnext(b, x, n, C, table);
					if (breathing()) await breathe();
				}
				if (!L.up) break;
				// channel-to-spatial: children where the subdivision logit is > 0
				const U = L.up, O = U.out, cols = s.get('cols', 1 << 26);
				const logit = s.get('sub', n * 8);
				U.subdiv.run(x, logit, n);
				// the last subdivision an upsampleOnly run needs: its children's cells alone
				const last = upsampleOnly && i === upsampleOnly - 1;
				let wide = null;
				if (!last) {
					const a = s.get('a', n * C);
					wide = s.get('wide', n * 8 * O);
					op.layerNormAffine(x, U.n1w, U.n1b, a, C, n, 1e-6);
					op.silu(a, n * C);
					U.conv1.run(a, wide, n, table, cols);
				}
				submit();
				const lg = readFloats(logit, n * 8);
				if (keepSubs) subs.push({ coords: cur, logits: lg });
				let m = 0;
				for (let k = 0; k < n * 8; k++) if (lg[k] > 0) m++;
				const codes = new Uint32Array(m), kids = new Uint32Array(n * 8).fill(NONE), next = new Int32Array(m * 3);
				for (let p = 0, j = 0; p < n; p++) {
					for (let sc = 0; sc < 8; sc++) {
						if (!(lg[p * 8 + sc] > 0)) continue;
						codes[j] = p * 8 + sc;
						kids[p * 8 + sc] = j;
						next[3 * j] = 2 * cur[3 * p] + (sc & 1);
						next[3 * j + 1] = 2 * cur[3 * p + 1] + ((sc >> 1) & 1);
						next[3 * j + 2] = 2 * cur[3 * p + 2] + ((sc >> 2) & 1);
						j++;
					}
				}
				if (last) {
					llm.print(`  shape decoder: ${coords.length / 3} voxels to ${m} at ${2 * r}^3 (subdivisions only) in ${llm.since(t0)}`);
					return { coords: next, r: 2 * r };
				}
				const code = uploadU32(codes), kid = uploadU32(kids), table2 = three.compute.u32(m * 27);
				op.childNeighbours(table, kid, code, table2, m);
				const h = s.get('ch', m * O), y = s.get(`x${++flip % 2}`, m * O), skip = s.get('skip', m * O);
				op.c2sGather(wide, code, h, m, O, 8 * O, O, 1);
				op.c2sGather(x, code, skip, m, O, C, C / 8, (8 * O) / C);
				op.layerNorm(h, h, O, m, 1e-6);
				op.silu(h, m * O);
				U.conv2.run(h, y, m, table2, cols);
				op.add(y, skip, m * O);
				submit();
				disposeAll([code, kid, table]);
				table = table2;
				if (this.verbose) llm.print(`    level ${i}: ${n} voxels to ${m} in ${llm.since(tb)}`);
				x = y;
				n = m;
				C = O;
				cur = next;
				r *= 2;
			}
			const a = s.get('a', n * C), out = s.get('out', n * 7);
			op.layerNorm(x, a, C, n, 1e-5);
			this.output.run(a, out, n);
			submit();
			const fields = readFloats(out, n * 7);
			const nb = new Uint32Array(table.readBytes(0, n * 27 * 4).buffer);
			llm.print(`  shape decoder: ${coords.length / 3} voxels to ${n} at ${r}^3 in ${llm.since(t0)}`);
			return { coords: cur, r, fields, table: nb, subs };
		} finally {
			submit();
			table.dispose();
		}
	}
}
