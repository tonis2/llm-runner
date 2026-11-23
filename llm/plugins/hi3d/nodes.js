// Hi3D's node: a picture and a coarse mesh of it (Pixal3D's, LATO.2's, a
// loaded .glb) to a finer mesh in the same frame.

import { llm } from '/core/llm.js';
import { defineNodes } from '/core/graph/registry.js';
import { Hi3D } from './pipeline.js';

let pipeline = null;

defineNodes('hi3d', {
	'hi3d.refine_mesh': {
		title: 'Hi3D refine mesh',
		category: '3d',
		description: 'Regenerates the mesh\'s surface from the picture, inside the mesh\'s voxels on a 128^3 grid, at 1024^3. '
			+ 'The mesh only places the voxels; the output stays in its frame and has no texture. '
			+ '`weights` is the Hi3D folder (transformer_1024/, vae/, dinov2-large/, clip-vit-large-patch14/). '
			+ '`budget` caps the voxels (the grid shrinks to fit); `faces` decimates to that many (0 keeps them all); '
			+ 'pieces under `part_keep` of the faces with nothing within `part_gap` are dropped. About 8 minutes for 75K voxels.',
		inputs: {
			image: 'IMAGE', mesh: 'MESH', weights: 'PATH(dir)',
			seed: 'INT=42', steps: 'INT=25', budget: 'INT=90000', faces: 'INT=1000000',
			part_keep: 'FLOAT=0.005', part_gap: 'FLOAT=0.02',
			linear: 'ENUM(f16|f32)=f16', decoder: 'ENUM(f32|q8)=f32',
		},
		outputs: { mesh: 'MESH' },
		async run({ image: img, mesh, weights, ...settings }, ctx) {
			if (!weights || !llm.exists(`${weights}/transformer_1024/config.json`)) {
				throw new Error(`weights: ${weights ? `${weights} is not the Hi3D folder (no transformer_1024/ in it)` : 'not set'}; choose the folder with transformer_1024/, vae/, dinov2-large/ and clip-vit-large-patch14/`);
			}
			if (!pipeline || pipeline.w !== weights) pipeline = new Hi3D(weights);
			const r = await pipeline.refine(img, mesh, { ...settings, progress: ctx.progress });
			return { mesh: { name: 'hi3d', positions: r.positions, indices: r.indices, normals: r.normals, stats: r.stats } };
		},
	},
});
