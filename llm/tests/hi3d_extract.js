// Hi3D's mesh extraction (lib/host/mesh_flexicubes.c3) against the torch
// reference's (small.safetensors from ref_hi3d.py small): FlexiCubes and the
// hole filling over the reference's own decoded cells, the vertices matched by
// position and the faces compared as sets (each turned to start at its
// smallest vertex, so the winding counts).
//
//   llm-runner tests/hi3d_extract.js golden=build/hi3d/golden/example4
import { llm } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';

const cfg = globalThis.__llm_config;
const gold = openSafetensors(`${cfg.golden}/small.safetensors`);
const cells = gold.ints('decoded_coord');
const t0 = llm.now();
// the decoder's rows [n, 40]: sign logits, then the fields (already tanh'd here)
const sign = gold.floats('decoded_sign'), feat = gold.floats('decoded_feat'), nc = cells.length / 3;
const packed = new Float32Array(nc * 40);
for (let i = 0; i < nc; i++) { packed.set(sign.subarray(8 * i, 8 * i + 8), 40 * i); packed.set(feat.subarray(32 * i, 32 * i + 32), 40 * i + 8); }
const mesh = llm.meshFlexiCubes(Int32Array.from(cells), packed, 1024, 10000, 1);
llm.print(`ours: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} faces (${mesh.cubes} surface cubes, ${mesh.quads} quads, ${mesh.holes} holes filled) in ${llm.since(t0)}`);

// The reference's frame: grid units over the latent voxels' mean and largest
// radius (Point's matrix, made at 128^3 and applied as it is).
const lat = gold.ints('coords'), n = lat.length / 3;
const c = [0, 0, 0];
for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) c[a] += lat[3 * i + a];
for (let a = 0; a < 3; a++) c[a] = Math.fround(c[a] / n);
let s = 0;
for (let i = 0; i < n; i++) s = Math.max(s, Math.hypot(lat[3 * i] - c[0], lat[3 * i + 1] - c[1], lat[3 * i + 2] - c[2]));
s += 1e-8;
const ours = mesh.positions.map((v, i) => (v - c[i % 3]) / s);
const wantV = gold.floats('mesh_vertices'), wantF = gold.ints('mesh_faces');
llm.print(`reference: ${wantV.length / 3} vertices, ${wantF.length / 3} faces`);

// each of our vertices matched to a reference vertex within 1e-5 (a hash on a
// 1e-3 grid, its 27 cells searched); faces compared through the match
const cell = (p, i, a) => Math.floor(p[3 * i + a] * 1e3);
const grid = new Map();
for (let j = 0; j < wantV.length / 3; j++) {
	const k = `${cell(wantV, j, 0)},${cell(wantV, j, 1)},${cell(wantV, j, 2)}`;
	if (!grid.has(k)) grid.set(k, []);
	grid.get(k).push(j);
}
const map = new Int32Array(ours.length / 3).fill(-1);
let matched = 0, worst = 0;
for (let i = 0; i < ours.length / 3; i++) {
	const cx = cell(ours, i, 0), cy = cell(ours, i, 1), cz = cell(ours, i, 2);
	let best = -1, bd = 1e-5;
	for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
		for (const j of grid.get(`${cx + dx},${cy + dy},${cz + dz}`) ?? []) {
			const d = Math.max(Math.abs(ours[3 * i] - wantV[3 * j]), Math.abs(ours[3 * i + 1] - wantV[3 * j + 1]), Math.abs(ours[3 * i + 2] - wantV[3 * j + 2]));
			if (d < bd) { bd = d; best = j; }
		}
	}
	if (best < 0) continue;
	map[i] = best;
	matched++;
	worst = Math.max(worst, bd);
}
llm.print(`vertices matched within 1e-5: ${matched} of ${ours.length / 3} (worst ${worst.toExponential(2)})`);
const face = (a, b, d) => {
	const m = Math.min(a, b, d);
	return m === a ? `${a},${b},${d}` : m === b ? `${b},${d},${a}` : `${d},${a},${b}`;
};
const want = new Set();
for (let f = 0; f < wantF.length / 3; f++) want.add(face(wantF[3 * f], wantF[3 * f + 1], wantF[3 * f + 2]));
let same = 0, flipped = 0;
const idx = mesh.indices;
for (let f = 0; f < idx.length / 3; f++) {
	const a = map[idx[3 * f]], b = map[idx[3 * f + 1]], d = map[idx[3 * f + 2]];
	if (a < 0 || b < 0 || d < 0) continue;
	if (want.has(face(a, b, d))) same++;
	else if (want.has(face(a, d, b))) flipped++;
}
llm.print(`faces: ${same} of ${idx.length / 3} in the reference's, ${flipped} there turned the other way`);
