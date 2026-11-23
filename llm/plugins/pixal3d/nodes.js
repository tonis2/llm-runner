// Pixal3D's node: an RGBA cutout to a dense mesh (MESH), for Save mesh or
// Hi3D's refiner.

import { defineNodes } from '/core/graph/registry.js';
import { Pixal3D } from './pipeline.js';

let pipeline = null;

defineNodes('pixal3d', {
	'pixal3d.image_to_mesh': {
		title: 'Pixal3D image to mesh',
		category: '3d',
		description: 'An RGBA cutout (transparent background) to a dense triangle mesh. '
			+ '`weights` is the Pixal3D folder (ckpts/, dinov3-vitl16/, naf/, pipeline.json); `gguf` is the flows\' file when it is not in that folder or beside it. '
			+ 'Resolution 512 is one stage, about a third of the time, and coarser. `faces` decimates to that many (0 keeps them all).',
		inputs: {
			image: 'IMAGE', weights: 'PATH(dir)', gguf: 'PATH(gguf)?',
			seed: 'INT=42', resolution: 'ENUM(1024|512)=1024', faces: 'INT=1000000', fov: 'FLOAT=0.2',
			ss_steps: 'INT=12', ss_guidance: 'FLOAT=7.5', shape_steps: 'INT=12', shape_guidance: 'FLOAT=7.5',
			linear: 'ENUM(f16|q8|f32)=f16',
		},
		outputs: { mesh: 'MESH' },
		async run({ image: img, weights, gguf, linear, resolution, ...settings }, ctx) {
			if (!pipeline || pipeline.w !== weights || pipeline.linear !== linear || (gguf && pipeline.gguf !== gguf)) {
				pipeline?.dispose();
				pipeline = new Pixal3D(weights, { gguf: gguf || null, linear });
			}
			const r = await pipeline.generate(img, { ...settings, resolution: Number(resolution), progress: ctx.progress });
			return { mesh: { name: 'pixal3d', positions: r.positions, indices: r.indices, normals: r.normals, stats: r.stats } };
		},
	},
});
