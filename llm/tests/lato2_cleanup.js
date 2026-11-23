// LATO.2's mesh cleanup (plugins/lato2/cleanup.js) on a .glb this plugin
// wrote: what it removed and filled, and the cleaned mesh written out.
//
//   llm-runner tests/lato2_cleanup.js in=build/qwen_alpha/lantern.glb out=build/qwen_alpha/lantern_clean.glb [smooth=0]
import { llm } from '/core/llm.js';
import { writeMeshGlb, vertexNormals } from '/core/mesh/glb.js';
import { cleanMesh } from '/plugins/lato2/cleanup.js';

// The positions and indices of a .glb with one primitive (what meshGlb writes).
function readMeshGlb(path) {
	const bytes = llm.readBytes(path);
	const u32 = (at) => bytes[at] | (bytes[at + 1] << 8) | (bytes[at + 2] << 16) | (bytes[at + 3] << 24);
	const jsonLength = u32(12);
	let text = '';
	for (let i = 0; i < jsonLength; i++) text += String.fromCharCode(bytes[20 + i]);
	const g = JSON.parse(text), bin = 20 + jsonLength + 8;
	const read = (a, Type) => {
		const acc = g.accessors[a], view = g.bufferViews[acc.bufferView];
		const comps = acc.type === 'VEC3' ? 3 : 1;
		const start = bin + (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);
		return new Type(bytes.slice(start, start + acc.count * comps * 4).buffer);
	};
	const prim = g.meshes[0].primitives[0];
	return { positions: read(prim.attributes.POSITION, Float32Array), indices: read(prim.indices, Uint32Array) };
}

const cfg = globalThis.__llm_config;
const { positions, indices } = readMeshGlb(cfg.in);
const t0 = llm.now();
const opts = {};
for (const k of ['smooth', 'minPiece', 'minSize', 'flapPasses', 'maxHole', 'foldCos']) if (cfg[k] !== undefined) opts[k] = Number(cfg[k]);
const r = cleanMesh(positions, indices, opts);
llm.print(`${cfg.in}: ${positions.length / 3} vertices, ${indices.length / 3} faces -> ${JSON.stringify(r.stats)} in ${llm.since(t0)}`);
const idx = Uint32Array.from(r.faces);
writeMeshGlb(cfg.out, { name: 'cleaned', positions: r.positions, indices: idx, normals: vertexNormals(r.positions, idx) });
