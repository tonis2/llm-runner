// Meshes from the shape decoder's FlexiDualGrid fields (per voxel: a vertex
// offset, three edge-crossing flags and a quad-split weight).
//
// dualGridMesh is o_voxel's flexible_dual_grid_to_mesh (the inference
// branch), for comparing with the reference: a vertex in every voxel, and for
// every flagged edge a quad through the four voxels around it. Its quads are
// unoriented (every one faces -axis), and the mesh cannot be oriented at all:
// thin plates make tens of thousands of non-manifold edges.
//
// What the pipeline uses instead is the voxel shell's outer surface: the
// outside flooded on the GPU (floodOutside), then surface nets over the
// shell's faces toward it (shellSurface): closed, facing out, with vertices
// from the predicted ones.

import { f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit } from '/core/gpu.js';
import { CellIndex } from './decoder.js';

// Per axis, the 4 voxels around the edge from a voxel's corner (o_voxel's
// edge_neighbor_voxel_offset), in quad order.
const AROUND = [
	[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
	[[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
	[[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
];

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

// coords [n, 3] on an r^3 grid, fields [n, 7] (the decoder's raw output) to
// { positions [n, 3] in [-0.5, 0.5]^3, faces [f, 3] }, as o_voxel winds them.
// `table` (the voxels' neighbour table [n, 27], offsets -1..1 with x slowest,
// as the decoder gives it) saves looking the quads' voxels up.
export function dualGridMesh(coords, fields, r, { margin = 0.5, table = null } = {}) {
	const n = coords.length / 3;
	const positions = dualVertices(coords, fields, r, { margin }), weight = new Float32Array(n);
	for (let i = 0; i < n; i++) weight[i] = softplus(fields[7 * i + 6]);
	const at = table ? null : new CellIndex(coords, n, r);
	const slot = AROUND.map((corners) => corners.map((o) => (o[0] + 1) * 9 + (o[1] + 1) * 3 + o[2] + 1));
	let faces = new Uint32Array(1 << 20), f = 0;
	const q = [0, 0, 0, 0];
	for (let i = 0; i < n; i++) {
		const x = coords[3 * i], y = coords[3 * i + 1], z = coords[3 * i + 2];
		for (let a = 0; a < 3; a++) {
			if (!(fields[7 * i + 3 + a] > 0)) continue;
			let ok = true;
			for (let k = 0; k < 4 && ok; k++) {
				if (table) {
					const v = table[27 * i + slot[a][k]];
					q[k] = v === 0xFFFFFFFF ? -1 : v;
				} else {
					const o = AROUND[a][k];
					q[k] = at.get(x + o[0], y + o[1], z + o[2]);
				}
				ok = q[k] >= 0;
			}
			if (!ok) continue;
			if (f + 6 > faces.length) {
				const bigger = new Uint32Array(faces.length * 2);
				bigger.set(faces);
				faces = bigger;
			}
			if (weight[q[0]] * weight[q[2]] > weight[q[1]] * weight[q[3]]) {
				faces[f++] = q[0]; faces[f++] = q[1]; faces[f++] = q[2];
				faces[f++] = q[0]; faces[f++] = q[2]; faces[f++] = q[3];
			} else {
				faces[f++] = q[0]; faces[f++] = q[1]; faces[f++] = q[3];
				faces[f++] = q[3]; faces[f++] = q[1]; faces[f++] = q[2];
			}
		}
	}
	return { positions, faces: faces.slice(0, f) };
}

// The cells outside a voxel shell (coords [n, 3] on an r^3 grid): flooded on
// the GPU from the border of the (r + 2)^3 grid around it, the shell one cell
// in. Cells are bits, 32 to a word along z: cell (x, y, z) of the padded grid
// is bit z % 32 of word (x g + y) words + z / 32. Returns { bits, g, words,
// rounds }.
export function floodOutside(coords, r) {
	const g = r + 2, words = Math.ceil(g / 32), count = g * g * words, n = coords.length / 3;
	const packed = new Uint32Array(count);
	for (let i = 0; i < n; i++) {
		const z = coords[3 * i + 2] + 1;
		packed[((coords[3 * i] + 1) * g + coords[3 * i + 1] + 1) * words + (z >>> 5)] |= 1 << (z & 31);
	}
	const shell = three.compute.u32(count, packed), outside = three.compute.u32(count), changed = three.compute.u32(1);
	try {
		op.floodInit(outside, g, words);
		let rounds = 0;
		for (;;) {
			changed.write(new Uint32Array(1));
			for (let a = 0; a < 3; a++) for (let d = 0; d < 2; d++) op.floodSweep(shell, outside, changed, g, words, a, d);
			submit();
			rounds++;
			if (new Uint32Array(changed.readBytes(0, 4).buffer)[0] === 0 || rounds >= 256) break;
		}
		return { bits: new Uint32Array(outside.readBytes(0, count * 4).buffer), g, words, rounds };
	} finally {
		submit();
		shell.dispose();
		outside.dispose();
		changed.dispose();
	}
}

// The predicted vertex of every voxel, in [-0.5, 0.5]^3: [n, 3].
export function dualVertices(coords, fields, r, { margin = 0.5 } = {}) {
	const n = coords.length / 3, out = new Float32Array(n * 3);
	for (let i = 0; i < n; i++) {
		for (let a = 0; a < 3; a++) out[3 * i + a] = (coords[3 * i + a] + (1 + 2 * margin) * sigmoid(fields[7 * i + a]) - margin) / r - 0.5;
	}
	return out;
}

// The shell's outer surface, closed and facing out (what TRELLIS.2's
// narrow-band remesh is for): surface nets over the cells, a voxel being
// inside unless the flood reached it. Every face between a shell voxel and an
// outside cell gives a quad through the four dual cubes (2^3 cells each)
// around it, wound to face the outside; a cube's vertex is the mean of the
// predicted vertices of the shell voxels at its corners. A thin plate, with
// outside on both sides, gets a face on each. `table` is the voxels'
// neighbour table [n, 27]. Returns { positions, faces }.
export function shellSurface(coords, vertices, r, table, outside) {
	const n = coords.length / 3, { bits, g, words } = outside;
	const isOut = (x, y, z) => (bits[((x + 1) * g + y + 1) * words + ((z + 1) >>> 5)] >>> ((z + 1) & 31)) & 1;
	// cubes by their min-corner cell (padded, so -1 is fine): key -> vertex
	let size = 1;
	while (size < 4 * n) size *= 2;
	// keys (up to g^3) can pass 2^31: unsigned, 0xFFFFFFFF empty
	const EMPTY = 0xFFFFFFFF, mask = size - 1, keys = new Uint32Array(size).fill(EMPTY), ids = new Int32Array(size);
	let positions = new Float32Array(n * 3 * 2), nv = 0;
	let faces = new Uint32Array(n * 12), nf = 0;
	const sum = [0, 0, 0];
	// the vertex of the cube at min corner v + (ox, oy, oz), each -1 or 0
	const cube = (v, ox, oy, oz) => {
		const x = coords[3 * v] + ox, y = coords[3 * v + 1] + oy, z = coords[3 * v + 2] + oz;
		const key = ((x + 1) * g + (y + 1)) * g + z + 1;
		let h = Math.imul(key, 0x9E3779B1) & mask;
		while (keys[h] !== EMPTY && keys[h] !== key) h = (h + 1) & mask;
		if (keys[h] === key) return ids[h];
		keys[h] = key;
		sum[0] = sum[1] = sum[2] = 0;
		let count = 0;
		for (let a = 0; a <= 1; a++) for (let b = 0; b <= 1; b++) for (let c = 0; c <= 1; c++) {
			const w = table[27 * v + (ox + a + 1) * 9 + (oy + b + 1) * 3 + oz + c + 1];
			if (w === 0xFFFFFFFF) continue;
			sum[0] += vertices[3 * w]; sum[1] += vertices[3 * w + 1]; sum[2] += vertices[3 * w + 2];
			count++;
		}
		if (3 * nv + 3 > positions.length) {
			const bigger = new Float32Array(positions.length * 2);
			bigger.set(positions);
			positions = bigger;
		}
		positions[3 * nv] = sum[0] / count;
		positions[3 * nv + 1] = sum[1] / count;
		positions[3 * nv + 2] = sum[2] / count;
		ids[h] = nv;
		return nv++;
	};
	const q = [0, 0, 0, 0], off = [0, 0, 0];
	for (let v = 0; v < n; v++) {
		const x = coords[3 * v], y = coords[3 * v + 1], z = coords[3 * v + 2];
		for (let a = 0; a < 3; a++) {
			const b = (a + 1) % 3, c = (a + 2) % 3;
			for (const s of [-1, 1]) {
				const nx = x + (a === 0 ? s : 0), ny = y + (a === 1 ? s : 0), nz = z + (a === 2 ? s : 0);
				if (!isOut(nx, ny, nz)) continue;
				// cubes around the edge v -> v + s e_a, counter-clockwise about +e_a
				const lo = s > 0 ? 0 : -1;
				const ring = [[-1, -1], [0, -1], [0, 0], [-1, 0]];
				for (let k = 0; k < 4; k++) {
					off[a] = lo;
					off[b] = ring[k][0];
					off[c] = ring[k][1];
					q[k] = cube(v, off[0], off[1], off[2]);
				}
				if (nf + 6 > faces.length) {
					const bigger = new Uint32Array(faces.length * 2);
					bigger.set(faces);
					faces = bigger;
				}
				// facing +e_a as it stands; reversed when the outside is at -e_a
				if (s > 0) {
					faces[nf++] = q[0]; faces[nf++] = q[1]; faces[nf++] = q[2];
					faces[nf++] = q[0]; faces[nf++] = q[2]; faces[nf++] = q[3];
				} else {
					faces[nf++] = q[0]; faces[nf++] = q[2]; faces[nf++] = q[1];
					faces[nf++] = q[0]; faces[nf++] = q[3]; faces[nf++] = q[2];
				}
			}
		}
	}
	return { positions: positions.slice(0, 3 * nv), faces: faces.slice(0, nf) };
}
