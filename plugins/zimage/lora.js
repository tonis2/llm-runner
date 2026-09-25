// Z-Image's LoRA and LoKr sites: the attention and SwiGLU FFN of the 30 main
// layers, under the naming schemes the old C3 loader read, plus the checkpoint's
// own fused names. Unfused Q/K/V land in row ranges of the fused QKV weight.
// The merge is `lib/lora.js`.

import * as lora from '../lib/lora.js';

const PREFIXES = [
	'layers.',
	'transformer_blocks.',
	'transformer.transformer_blocks.',
	'base_model.model.transformer.transformer_blocks.',
	'diffusion_model.layers.',
	'transformer.layers.',
];
const SCHEMES = PREFIXES.map((prefix) => ({ prefix }));

// [file names under the block, weight key, row offset in units of dim].
const SITES = [
	[['attention.qkv'], 'qkv', 0],
	[['attention.to_q'], 'qkv', 0],
	[['attention.to_k'], 'qkv', 1],
	[['attention.to_v'], 'qkv', 2],
	[['attention.out', 'attention.to_out', 'attention.to_out.0'], 'out', 0],
	[['feed_forward.w1'], 'w1', 0],
	[['feed_forward.w3'], 'w3', 0],
	[['feed_forward.w2'], 'w2', 0],
];

function* sites(dit, scheme, count) {
	for (let l = 0; l < count; l++) {
		for (const [modules, weight, slot] of SITES) {
			yield {
				module: modules.map((m) => `${scheme.prefix}${l}.${m}`),
				weight: dit.layers[l][weight],
				rowOffset: slot * dit.dim,
			};
		}
	}
}

// Fold `loras` ([{ path, strength }]) into `dit`'s resident weights, in order.
export function mergeLoras(dit, loras) {
	lora.mergeLoras({
		schemes: SCHEMES,
		sites: (scheme) => sites(dit, scheme, dit.layers.length),
		firstSites: (scheme) => sites(dit, scheme, 1),
	}, loras);
}
