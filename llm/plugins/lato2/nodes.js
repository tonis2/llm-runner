// LATO.2's node: a cutout image in, a .glb mesh written out.

import { llm } from '/core/llm.js';
import { defineNodes } from '/core/graph/registry.js';
import { writeMeshGlb } from '/core/mesh/glb.js';
import { Lato2 } from './pipeline.js';

function numbered(path) {
	if (!path.includes('#')) return path;
	for (let n = 1; ; n++) {
		const p = path.replace('#', String(n));
		if (!llm.exists(p)) return p;
	}
}

function makeDirFor(path) {
	const slash = path.lastIndexOf('/');
	if (slash > 0) globalThis.__llm.makeDir(path.slice(0, slash));
}

let pipeline = null;

defineNodes('lato2', {
	'lato2.image_to_mesh': {
		title: 'LATO.2 image to mesh',
		category: '3d',
		description: 'An RGBA cutout (transparent background) to a triangle mesh, saved as a .glb. '
			+ '`weights` is the folder with LATO.2\'s .pt files, trellis/ and dinov2_vitl14_reg4_pretrain.pth. '
			+ '`cleanup` removes crowded edges, folds and loose specks and fills holes; `smooth` adds Taubin steps (good on rounded shapes, shrinks thin parts). '
				+ 'A # in the name becomes the next free number.',
		output: true,
		inputs: {
			image: 'IMAGE', weights: 'PATH(dir)', path: 'STRING=output/lato2-#.glb',
			seed: 'INT=42', vertices: 'INT=2000',
			ss_steps: 'INT=25', ss_cfg: 'FLOAT=5', vflow_steps: 'INT=24', cfg: 'FLOAT=3', tflow_steps: 'INT=50',
			fill_holes: 'BOOL=true', cleanup: 'BOOL=true', smooth: 'INT=0',
		},
		outputs: {},
		async run({ image: img, weights, path, ...settings }) {
			if (!pipeline || pipeline.w !== weights) {
				pipeline?.dispose();
				pipeline = new Lato2(weights);
			}
			const r = await pipeline.generate(img, settings);
			const out = numbered(path);
			makeDirFor(out);
			writeMeshGlb(out, { name: 'lato2', positions: r.positions, indices: r.indices, normals: r.normals, generator: 'llm-runner lato2', extras: r.stats });
			llm.print(`  saved ${out} (${r.stats.vertices} vertices, ${r.stats.faces} faces)`);
			return { output: out, ...r.stats };
		},
	},
});
