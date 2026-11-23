// How much LATO.2's mesh depends on its voxel scaffold: the same picture of a
// real mesh run twice, once on the mesh's own 64^3 voxels (as LATO.2's
// mesh-to-mesh does) and once on the voxels TRELLIS's sparse-structure flow
// guesses from the picture (the image-to-mesh path).
//
//   llm-runner tests/lato2_scaffold.js gt=build/lato2/gt/dragon.safetensors image=build/lato2/gt/dragon_render.png weights="/path/lato2" out=build/lato2/gt/dragon
//
// `gt` from tools/lato2_mesh_prep.py; `image` an RGBA picture of the same mesh
// (LATO.2 trained on grey renders from azimuth 45, elevation 30).
import { llm, image } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { writeMeshGlb } from '/core/mesh/glb.js';
import { Lato2 } from '/plugins/lato2/pipeline.js';

const cfg = globalThis.__llm_config;
const voxels = Int32Array.from(openSafetensors(cfg.gt).ints('voxels'));
const img = image.load(cfg.image);
const lato = new Lato2(cfg.weights);
for (const [name, options] of [['real_voxels', { voxels }], ['trellis_voxels', {}]]) {
	const r = await lato.generate(img, { vertices: cfg.vertices ?? 2000, ...options });
	const path = `${cfg.out}_${name}.glb`;
	writeMeshGlb(path, { name, positions: r.positions, indices: r.indices, normals: r.normals });
	llm.print(`${name}: ${r.stats.voxels} voxels, ${r.stats.vertices} vertices, ${r.stats.edges} edges, ${r.stats.faces} faces -> ${path}`);
}
lato.dispose();
