// core/mesh/simplify.js on a .glb: the pieces dropped, the vertices
// separated and the faces left, timed. Writes the result when out= is given.
//
//   llm-runner tests/mesh_simplify.js mesh=build/pixal3d/out/example4_1024.glb [faces=1000000] [gap=0.02] [separate=0.000244] [smooth=3] [crease=60] [out=x.glb]
import { llm } from '/core/llm.js';
import { readMeshGlb, writeMeshGlb, vertexNormals } from '/core/mesh/glb.js';
import { simplifyMesh } from '/core/mesh/simplify.js';

const cfg = globalThis.__llm_config;
let t = llm.now();
const mesh = readMeshGlb(cfg.mesh);
llm.print(`read ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} faces in ${llm.since(t)}`);
t = llm.now();
const r = simplifyMesh(mesh.positions, mesh.indices, {
	faces: Number(cfg.faces ?? 1000000), partGap: Number(cfg.gap ?? 0.02), separate: Number(cfg.separate ?? 0.25 / 1024), smooth: Number(cfg.smooth ?? 3), crease: Number(cfg.crease ?? 60),
});
llm.print(`simplified in ${llm.since(t)}: ${r.parts} pieces, ${r.dropped} dropped, ${r.separated} vertices separated, ${r.passes} passes; ${r.positions.length / 3} vertices, ${r.indices.length / 3} faces (steps ${Array.from(r.seconds, (x) => x.toFixed(2)).join(' / ')} s)`);
if (cfg.out) {
	writeMeshGlb(cfg.out, { name: 'simplified', positions: r.positions, indices: r.indices, normals: r.normals ?? vertexNormals(r.positions, r.indices) });
	llm.print(`wrote ${cfg.out}`);
}
