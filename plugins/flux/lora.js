// Flux 2 Klein's LoRA and LoKr sites: the naming schemes, and where each
// site lands in the dual- and single-stream blocks. The merge is `lib/lora.js`.

import * as lora from '../lib/lora.js';

// Where each site lives in a file, per naming scheme. An absent name means the
// scheme has no such site.
const SCHEMES = [
	{
		dual: 'double_blocks.', single: 'single_blocks.', sep: '.',
		img_qkv: 'img_attn.qkv', img_proj: 'img_attn.proj', img_mlp_up: 'img_mlp.0', img_mlp_down: 'img_mlp.2',
		txt_qkv: 'txt_attn.qkv', txt_proj: 'txt_attn.proj', txt_mlp_up: 'txt_mlp.0', txt_mlp_down: 'txt_mlp.2',
		linear1: 'linear1', linear2: 'linear2',
	},
	{
		dual: 'diffusion_model.double_blocks.', single: 'diffusion_model.single_blocks.', sep: '.',
		img_qkv: 'img_attn.qkv', img_proj: 'img_attn.proj', img_mlp_up: 'img_mlp.0', img_mlp_down: 'img_mlp.2',
		txt_qkv: 'txt_attn.qkv', txt_proj: 'txt_attn.proj', txt_mlp_up: 'txt_mlp.0', txt_mlp_down: 'txt_mlp.2',
		linear1: 'linear1', linear2: 'linear2',
	},
	{
		// Diffusers: Q/K/V unfused, landing in row ranges of the fused weights.
		// No MLP sites - Flux 1's GELU FFN has no place in Klein's SwiGLU.
		dual: 'transformer.transformer_blocks.', single: 'transformer.single_transformer_blocks.', sep: '.',
		img_q: 'attn.to_q', img_k: 'attn.to_k', img_v: 'attn.to_v', img_proj: 'attn.to_out.0',
		txt_q: 'attn.add_q_proj', txt_k: 'attn.add_k_proj', txt_v: 'attn.add_v_proj', txt_proj: 'attn.to_add_out',
		single_q: 'attn.to_q', single_k: 'attn.to_k', single_v: 'attn.to_v', linear2: 'proj_out',
	},
	{
		dual: 'lora_unet_double_blocks_', single: 'lora_unet_single_blocks_', sep: '_',
		img_qkv: 'img_attn_qkv', img_proj: 'img_attn_proj', img_mlp_up: 'img_mlp_0', img_mlp_down: 'img_mlp_2',
		txt_qkv: 'txt_attn_qkv', txt_proj: 'txt_attn_proj', txt_mlp_up: 'txt_mlp_0', txt_mlp_down: 'txt_mlp_2',
		linear1: 'linear1', linear2: 'linear2',
	},
];

// Every site: [scheme key, block list, weight key, row offset in units of dim].
const DUAL_SITES = [
	['img_qkv', 'imgQkv', 0], ['img_q', 'imgQkv', 0], ['img_k', 'imgQkv', 1], ['img_v', 'imgQkv', 2],
	['img_proj', 'imgProj', 0], ['img_mlp_up', 'imgUp', 0], ['img_mlp_down', 'imgDown', 0],
	['txt_qkv', 'txtQkv', 0], ['txt_q', 'txtQkv', 0], ['txt_k', 'txtQkv', 1], ['txt_v', 'txtQkv', 2],
	['txt_proj', 'txtProj', 0], ['txt_mlp_up', 'txtUp', 0], ['txt_mlp_down', 'txtDown', 0],
];
const SINGLE_SITES = [
	['linear1', 'linear1', 0], ['single_q', 'linear1', 0], ['single_k', 'linear1', 1], ['single_v', 'linear1', 2],
	['linear2', 'linear2', 0],
];

function modulePath(prefix, layer, sep, module) { return `${prefix}${layer}${sep}${module}`; }

// Every (site, weight) pair of `dit` under `scheme`, as { module, weight, rowOffset },
// through block `last` of each stream.
function* sites(dit, scheme, last = Infinity) {
	const dim = dit.config.dim;
	for (let l = 0; l < Math.min(dit.dual.length, last + 1); l++) {
		for (const [key, weight, slot] of DUAL_SITES) {
			if (scheme[key]) yield { module: modulePath(scheme.dual, l, scheme.sep, scheme[key]), weight: dit.dual[l]?.[weight], rowOffset: slot * dim };
		}
	}
	for (let l = 0; l < Math.min(dit.single.length, last + 1); l++) {
		for (const [key, weight, slot] of SINGLE_SITES) {
			if (scheme[key]) yield { module: modulePath(scheme.single, l, scheme.sep, scheme[key]), weight: dit.single[l]?.[weight], rowOffset: slot * dim };
		}
	}
}

// Fold `loras` ([{ path, strength }]) into `dit`'s resident weights, in order.
export function mergeLoras(dit, loras) {
	lora.mergeLoras({
		schemes: SCHEMES,
		sites: (scheme) => sites(dit, scheme),
		firstSites: (scheme) => sites(dit, scheme, 0),
	}, loras);
}
