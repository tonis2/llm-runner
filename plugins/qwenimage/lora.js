// Qwen-Image 2.1's LoRA sites: the naming schemes, and where each site lands
// in the single-stream blocks. The merge is `lib/lora.js`.
//
// The DiT keeps the SwiGLU MLP fused as `img_mlp.gate_up` ([gate; up] — what
// diffusers' module calls `gate_layer` and `proj`), so a file naming the two
// halves lands each in its own rows of the fused weight.

import * as lora from '../lib/lora.js';

// Where each site lives in a file, per naming scheme.
const SCHEMES = [
	{
		// PEFT on the diffusers model, saved from the full pipeline.
		prefix: 'transformer.transformer_blocks.', sep: '.',
		q: 'attn.to_q', k: 'attn.to_k', v: 'attn.to_v', out: 'attn.to_out.0',
		gate_up: 'img_mlp.gate_up', gate_layer: 'img_mlp.gate_layer', proj: 'img_mlp.proj', down: 'img_mlp.out',
	},
	{
		// ai-toolkit's transformer_blocks format.
		prefix: 'diffusion_model.transformer_blocks.', sep: '.',
		q: 'attn.to_q', k: 'attn.to_k', v: 'attn.to_v', out: 'attn.to_out.0',
		gate_up: 'img_mlp.gate_up', gate_layer: 'img_mlp.gate_layer', proj: 'img_mlp.proj', down: 'img_mlp.out',
	},
];

// Every (site, weight) pair of `dit` under `scheme`, through block `last`, as
// { module, weight, rowOffset } with the offset in output rows of the weight.
function* sites(dit, scheme, last = Infinity) {
	for (let l = 0; l < Math.min(dit.nLayers, last + 1); l++) {
		const b = dit.blocks[l];
		for (const [key, weight, offset] of [
			['q', b.q, 0], ['k', b.k, 0], ['v', b.v, 0], ['out', b.out, 0],
			['gate_up', b.gateUp, 0],
			['gate_layer', b.gateUp, 0],
			['proj', b.gateUp, dit.ffn],
			['down', b.down, 0],
		]) {
			if (scheme[key]) yield { module: `${scheme.prefix}${l}${scheme.sep}${scheme[key]}`, weight, rowOffset: offset };
		}
	}
}

// Put `loras` ([{ path, strength }]) on `dit`'s resident weights, in order.
// They stay unmerged: Qwen-Image LoRA deltas are about one Q8_0 step, and
// requantising the merged weights rounds much of them away.
export function mergeLoras(dit, loras) {
	lora.mergeLoras({
		schemes: SCHEMES,
		sites: (scheme) => sites(dit, scheme),
		firstSites: (scheme) => sites(dit, scheme, 0),
		// Q, K and V read one input; so do the gate and up halves.
		inputs: () => dit.blocks.flatMap((b) => [[b.q, b.k, b.v], [b.gateUp]]),
	}, loras, { unmerged: true });
}
