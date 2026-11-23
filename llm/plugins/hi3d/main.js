// Hi3D: a picture and a coarse mesh of it to a finer mesh
// (github.com/hi3d-official/Hi3Dv1.0, a Sparc3D refiner).
//
//   llm-runner hi3d weights=/path/hi3d input=picture.png mesh=coarse.glb output=out/refined.glb
//   llm-runner hi3d weights=/path/hi3d --server --port 7863
//       POST /generate {"input": "picture.png", "mesh": "coarse.glb", "output": "out/refined.glb", ...}
//
// The mesh (a .glb, e.g. Pixal3D's) only places the voxels Hi3D regenerates
// the surface in; the output stays in its frame. Settings, with their
// defaults: steps 25, budget 90000 (voxels on the 128^3 grid), faces 1000000
// (0 keeps them all), part_keep 0.005, part_gap 0.02, seed 42, linear f16 |
// f32, decoder f32 | q8, cond (a path: save the squared-up picture there too).

import { llm, image } from '/core/llm.js';
import { readMeshGlb, writeMeshGlb } from '/core/mesh/glb.js';
import { Hi3D, DEFAULTS } from './pipeline.js';

let pipeline = null;

function settings(config) {
	const o = {};
	for (const k of Object.keys(DEFAULTS)) if (config[k] !== undefined) o[k] = typeof DEFAULTS[k] === 'number' ? Number(config[k]) : config[k];
	return o;
}

async function run(config) {
	if (!config.input) throw new Error('input= is needed: the picture (.png; an RGBA cutout or RGB)');
	if (!config.mesh) throw new Error('mesh= is needed: the coarse mesh (.glb)');
	if (!config.weights) throw new Error('weights= is needed: the Hi3D folder (transformer_1024/, vae/, dinov2-large/, clip-vit-large-patch14/)');
	const output = config.output ?? 'output/hi3d.glb';
	if (!pipeline || pipeline.w !== config.weights) {
		pipeline?.dispose();
		pipeline = new Hi3D(config.weights);
	}
	const r = await pipeline.refine(image.load(config.input), readMeshGlb(config.mesh), settings(config));
	writeMeshGlb(output, {
		name: 'hi3d', positions: r.positions, indices: r.indices, normals: r.normals,
		generator: 'llm-runner hi3d', extras: { source: config.input, mesh: config.mesh, ...r.stats },
	});
	if (config.cond) image.savePng(config.cond, r.pre);
	llm.print(`  wrote ${output}: ${r.stats.vertices} vertices, ${r.stats.faces} faces, in ${r.stats.seconds.toFixed(1)}s`);
	return { output, ...r.stats };
}

llm.plugin({
	name: 'hi3d',
	load() {},
	async generate(config) {
		return run(config);
	},
	async handle(request) {
		const base = globalThis.__llm_config ?? {};
		if (request.method === 'GET' && request.path === '/info') {
			llm.respond(200, 'application/json', JSON.stringify({ name: 'hi3d', defaults: DEFAULTS }));
			return;
		}
		if (request.method === 'POST' && request.path === '/generate') {
			try {
				const body = request.body ? JSON.parse(request.body) : {};
				const result = await run({ ...base, ...body });
				llm.respond(200, 'application/json', JSON.stringify(result));
			} catch (e) {
				llm.respond(400, 'application/json', JSON.stringify({ error: String(e.message ?? e) }));
			}
			return;
		}
		llm.respond(404, 'application/json', JSON.stringify({ error: `no route ${request.method} ${request.path}` }));
	},
});
