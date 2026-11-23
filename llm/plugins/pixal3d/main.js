// Pixal3D: an image to a dense 3D mesh (github.com/TencentARC/Pixal3D).
//
//   llm-runner pixal3d weights=/path/pixal3d input=cutout.png output=out/model.glb
//   llm-runner pixal3d weights=/path/pixal3d --server --port 7862
//       POST /generate {"input": "cutout.png", "output": "out/model.glb", ...}
//
// The input must be a cutout: an RGBA image whose background is transparent.
// `gguf` is the flows' file when it is not in weights/ or beside it.
// Settings, with their defaults: resolution 1024 (512: one stage, about a
// third of the time, coarser), faces 1000000 (decimated to that; 0 keeps the
// surface's every face), seed 42, fov 0.2 (radians), ss_steps 12,
// ss_guidance 7.5, ss_rescale 0.7, ss_rescale_t 5, shape_steps 12,
// shape_guidance 7.5, shape_rescale 0.5, shape_rescale_t 3, linear f16 | q8 | f32,
// cond (a path: save the squared-up picture there too).

import { llm, image } from '/core/llm.js';
import { writeMeshGlb } from '/core/mesh/glb.js';
import { Pixal3D, DEFAULTS } from './pipeline.js';

let pipeline = null;

function settings(config) {
	const o = {};
	for (const k of Object.keys(DEFAULTS)) if (config[k] !== undefined) o[k] = config[k];
	return o;
}

async function run(config) {
	if (!config.input) throw new Error('input= is needed: an RGBA cutout (.png)');
	if (!config.weights) throw new Error('weights= is needed: the Pixal3D folder (ckpts/, dinov3-vitl16/, naf/, pipeline.json)');
	const output = config.output ?? 'output/pixal3d.glb';
	if (!pipeline || pipeline.w !== config.weights) {
		pipeline?.dispose();
		pipeline = new Pixal3D(config.weights, { gguf: config.gguf ?? null, linear: config.linear ?? DEFAULTS.linear });
	}
	const r = await pipeline.generate(image.load(config.input), settings(config));
	writeMeshGlb(output, {
		name: 'pixal3d', positions: r.positions, indices: r.indices, normals: r.normals,
		generator: 'llm-runner pixal3d', extras: { source: config.input, ...r.stats },
	});
	if (config.cond) image.savePng(config.cond, r.pre);
	llm.print(`  wrote ${output}: ${r.stats.vertices} vertices, ${r.stats.faces} faces, in ${r.stats.seconds.toFixed(1)}s`);
	return { output, ...r.stats };
}

llm.plugin({
	name: 'pixal3d',
	load() {},
	async generate(config) {
		return run(config);
	},
	async handle(request) {
		const base = globalThis.__llm_config ?? {};
		if (request.method === 'GET' && request.path === '/info') {
			llm.respond(200, 'application/json', JSON.stringify({ name: 'pixal3d', defaults: DEFAULTS }));
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
