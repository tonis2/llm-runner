// Pixal3D's shape decoder and mesh extraction against the torch reference's,
// from the reference's own 512 shape latent: the subdivision logits of the
// first level, the voxels at each level, and the mesh. Writes the mesh as .glb.
//
//   llm-runner tests/pixal3d_decode.js golden=build/pixal3d/golden/example4 weights="/path/pixal3d" [out=build/pixal3d/example4_512.glb]
import { llm } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { writeMeshGlb, vertexNormals } from '/core/mesh/glb.js';
import { ShapeDecoder } from '/plugins/pixal3d/decoder.js';
import { dualGridMesh, floodOutside, dualVertices, shellSurface } from '/plugins/pixal3d/mesh.js';
import { upload } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config;
const sh = openSafetensors(`${cfg.golden}/shape512.safetensors`);
const gold = openSafetensors(`${cfg.golden}/decode512.safetensors`);
const c4 = sh.ints('coords'), n = c4.length / 4, coords = new Int32Array(n * 3);
for (let t = 0; t < n; t++) for (let a = 0; a < 3; a++) coords[3 * t + a] = c4[4 * t + 1 + a];
const dec = new ShapeDecoder(`${cfg.weights}/ckpts`, { linear: cfg.linear ?? 'f16' });
dec.load();
dec.verbose = true;
const latent = upload(sh.floats('slat'));
const t0 = llm.now();
const out = await dec.run(latent, coords, 32, { keepSubs: true });
llm.print(`decoded in ${llm.since(t0)}`);
for (const [i, sub] of out.subs.entries()) {
	const want = gold.floats(`sub${i}_logits`), wc = gold.ints(`sub${i}_coords`);
	let line = `level ${i}: ${sub.logits.length / 8} voxels (reference ${wc.length / 4})`;
	if (sub.logits.length === want.length) {
		let worst = 0, agree = 0;
		for (let k = 0; k < want.length; k++) { worst = Math.max(worst, Math.abs(sub.logits[k] - want[k])); if ((sub.logits[k] > 0) === (want[k] > 0)) agree++; }
		line += `, subdivision logits worst ${worst.toFixed(4)}, ${want.length - agree} of ${want.length} decisions differ`;
	}
	llm.print(line);
}
const t1 = llm.now();
const mesh = dualGridMesh(out.coords, out.fields, out.r, { table: out.table });
const wv = gold.floats('vertices'), wf = gold.ints('faces');
llm.print(`mesh: ${mesh.positions.length / 3} vertices, ${mesh.faces.length / 3} faces (reference ${wv.length / 3}, ${wf.length / 3}) in ${llm.since(t1)}`);
if (mesh.positions.length === wv.length) {
	let worst = 0;
	for (let i = 0; i < wv.length; i++) worst = Math.max(worst, Math.abs(mesh.positions[i] - wv[i]));
	let same = 0;
	const F = Math.min(wf.length, mesh.faces.length);
	for (let i = 0; i < F; i++) if (wf[i] === mesh.faces[i]) same++;
	llm.print(`vertices worst ${worst.toExponential(2)} (a voxel is ${(1 / 512).toExponential(2)}), face indices equal ${same} of ${wf.length}`);
}
if (cfg.out) {
	const t2 = llm.now();
	const outside = floodOutside(out.coords, out.r);
	const flooded = llm.since(t2), t3 = llm.now();
	const surf = shellSurface(out.coords, dualVertices(out.coords, out.fields, out.r), out.r, out.table, outside);
	llm.print(`outside flooded in ${outside.rounds} rounds, ${flooded}; surface ${surf.positions.length / 3} vertices, ${surf.faces.length / 3} faces in ${llm.since(t3)}`);
	writeMeshGlb(cfg.out, { name: 'pixal3d', positions: surf.positions, indices: surf.faces, normals: vertexNormals(surf.positions, surf.faces) });
	llm.print(`wrote ${cfg.out}`);
}
dec.dispose();
