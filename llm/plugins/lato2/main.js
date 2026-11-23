// LATO.2: an image to a 3D mesh (github.com/LoHhhha/LATO.2).
//
//   llm-runner lato2 weights=/path/lato2 input=cutout.png output=out/model.glb
//   llm-runner lato2 weights=/path/lato2 keep=true --server --port 7861
//       POST /generate {"input": "cutout.png", "output": "out/model.glb", ...}
//
// The input must be a cutout: an RGBA image whose background is transparent
// (no background removal is done). Settings, with their defaults:
//   seed 42, vertices 2000 (the vertex count asked for),
//   ss_steps 25, ss_cfg 5, vflow_steps 24, cfg 3, tflow_steps 50,
//   threshold 0.5 (vertex pruning), edge_threshold 0, fill_holes true,
//   max_ring 12, orient true (consistent winding), linear f32 | q8,
//   cond (a path: save the conditioning picture there too).

import { llm, image } from '/core/llm.js';
import { writeMeshGlb } from '/core/mesh/glb.js';
import { Lato2, DEFAULTS } from './pipeline.js';

let pipeline = null;

function settings(config) {
	const o = {};
	for (const k of Object.keys(DEFAULTS)) if (config[k] !== undefined) o[k] = config[k];
	return o;
}

async function run(config) {
	if (!config.input) throw new Error('input= is needed: an RGBA cutout (.png)');
	if (!config.weights) throw new Error('weights= is needed: the folder with LATO.2, TRELLIS and DINOv2 weights');
	const output = config.output ?? 'output/lato2.glb';
	if (!pipeline || pipeline.w !== config.weights) {
		pipeline?.dispose();
		pipeline = new Lato2(config.weights, { keep: !!config.keep, linear: config.linear ?? 'f32' });
	}
	const img = image.load(config.input);
	const r = await pipeline.generate(img, settings(config));
	writeMeshGlb(output, {
		name: 'lato2', positions: r.positions, indices: r.indices, normals: r.normals,
		generator: 'llm-runner lato2', extras: { source: config.input, ...r.stats },
	});
	if (config.cond) image.savePng(config.cond, r.cond);
	llm.print(`  wrote ${output}: ${r.stats.vertices} vertices, ${r.stats.faces} faces, in ${r.stats.seconds.toFixed(1)}s`);
	return { output, ...r.stats };
}

llm.plugin({
	name: 'lato2',
	load() {},
	async generate(config) {
		return run(config);
	},
	async handle(request) {
		const base = globalThis.__llm_config ?? {};
		if (request.method === 'GET' && request.path === '/info') {
			llm.respond(200, 'application/json', JSON.stringify({ name: 'lato2', defaults: DEFAULTS }));
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
