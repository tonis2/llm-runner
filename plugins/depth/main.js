// Depth Anything V2 (ViT-S/14) as a plugin: an image in, a depth map out.
//
//   llm-runner depth model=depth_anything_v2_vits_fp32.safetensors input=photo.jpg output=depth.png
//   llm-runner depth ... height=true albedo=1 flatten=0.7     (a tiling height map)
//
// The model is the `depth.estimate` node in `nodes.js`; this runs it between a
// load and a save. Settings are the C3 CLI's options.

import { llm } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { Executor } from '../lib/graph/executor.js';
import '../core/nodes.js';
import './nodes.js';

export function depthGraph(config) {
	const params = { image: { from: 'input.image' }, path: config.model };
	if (config.res) params.res = config.res;
	params.height_map = config.height === true || config.height === 'true';
	for (const k of ['detrend', 'detail', 'albedo', 'flatten']) if (config[k] !== undefined) params[k] = config[k];
	return {
		nodes: [
			{ id: 'input', type: 'core.load_image', params: { path: config.input } },
			{ id: 'depth', type: 'depth.estimate', params },
			{ id: 'save', type: 'core.save_image', params: { image: { from: 'depth.image' }, path: config.output ?? 'depth.png' } },
		],
	};
}

llm.plugin({
	name: 'depth',
	async generate(config) {
		const start = llm.now();
		op.useMatrixCores(config.matrix_cores ?? true);
		llm.print(`=== Depth: ${config.input} ===`);
		const { results } = await new Executor().run(depthGraph(config));
		llm.print(`=== done in ${llm.since(start)} ===`);
		return results.save;
	},
});
