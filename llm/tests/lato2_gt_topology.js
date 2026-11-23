// LATO.2's topology stage on a real mesh's own vertices: T-Flow and the T-VAE
// should find that mesh's edges. Precision and recall of the edges and faces
// against the mesh, and both meshes written as .glb to compare by eye.
//
//   llm-runner tests/lato2_gt_topology.js gt=build/lato2/gt/dragon.safetensors weights="/path/lato2" [out=build/lato2/gt/dragon]
//
// The .safetensors holds `verts` [n, 3] on the 1024 grid, `offsets`, `faces`,
// `edges` (i < j) and `voxels` [m, 3] of 64^3, as LATO.2's topo dataset makes them
// (tools/lato2_mesh_prep.py writes it from a .glb).
import { llm } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { writeMeshGlb, vertexNormals } from '/core/mesh/glb.js';
import { VoxelEncoder, TopologyFlow, TopologyDecoder } from '/plugins/lato2/topology.js';
import { edgesToFaces, orientFaces } from '/plugins/lato2/faces.js';
import { upload } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config, W = cfg.weights;
const g = openSafetensors(cfg.gt);
const verts = Int32Array.from(g.ints('verts')), n = verts.length / 3;
const voxels = Int32Array.from(g.ints('voxels')), nv = voxels.length / 3;
const offsets = g.floats('offsets');

const venc = new VoxelEncoder(`${W}/voxel_encoder.pt`);
venc.load();
const { tokens, count } = venc.run(voxels, nv);
venc.dispose();
const tflow = new TopologyFlow(`${W}/tflow.pt`);
await tflow.load();
const z = upload(llm.randomNormal(n * tflow.zDim, cfg.seed ?? 44));
await tflow.sample(z, verts, n, tokens, count, { steps: cfg.steps ?? 50 });
tflow.dispose();
const tvae = new TopologyDecoder(`${W}/tvae.pt`);
tvae.load();
const edges = tvae.edges(tvae.features(z, verts, n), n, 0);
tvae.dispose();

function score(name, got, want, width) {
	const key = (a, i) => {
		const r = Array.from(a.slice(i, i + width)).sort((x, y) => x - y);
		return r.join(',');
	};
	const set = new Set();
	for (let i = 0; i < want.length; i += width) set.add(key(want, i));
	let hit = 0;
	for (let i = 0; i < got.length; i += width) if (set.has(key(got, i))) hit++;
	const p = hit / (got.length / width), r = hit / (want.length / width);
	llm.print(`${name}: ${got.length / width} predicted, ${want.length / width} in the mesh, precision ${(100 * p).toFixed(1)}%, recall ${(100 * r).toFixed(1)}%`);
}
llm.print(`${n} vertices, ${nv} voxels`);
score('edges', edges, g.ints('edges'), 2);
const faces = edgesToFaces(edges, n, verts, { fill: true, maxRing: 12 });
score('faces', faces, g.ints('faces'), 3);

if (cfg.out) {
	const positions = new Float32Array(n * 3);
	for (let i = 0; i < n * 3; i++) positions[i] = verts[i] / 1024 - 0.5 + offsets[i] / 2048;
	for (const [suffix, f] of [['pred', orientFaces(faces, positions)], ['mesh', Int32Array.from(g.ints('faces'))]]) {
		const indices = Uint32Array.from(f);
		writeMeshGlb(`${cfg.out}_${suffix}.glb`, { name: suffix, positions, indices, normals: vertexNormals(positions, indices) });
	}
	llm.print(`wrote ${cfg.out}_pred.glb and ${cfg.out}_mesh.glb`);
}
