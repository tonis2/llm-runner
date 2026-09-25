// Z-Image Turbo as a plugin.
//
//   llm-runner zimage --config zimage.json
//   llm-runner zimage --config zimage.json input=photo.png strength=0.6
//
// Qwen3 hidden states (the second-to-last layer) as the text conditioning, a
// flow-matching Euler loop through the Lumina2 DiT with optional CFG, and the
// Flux 1 VAE, or TAESD (`taesd`) for a fast, rough decode. `input` starts from
// an image instead of noise (img2img, `strength` of the way back to noise).
// `lora` / `lora_path` fold adapters into the DiT. `text_embeddings` may name a
// file of precomputed embeddings in the C3 pipeline's format (u32 tokens, u32
// dim, then floats) instead of a text model. Settings are the C3 CLI's JSON
// keys; `width` and `height` may stand in for `size`.
//
// The work is the nodes in `nodes.js`; `zimageGraph(config)` wires them.

import { llm } from '../lib/llm.js';
import { useMatrixCores } from '../lib/ops.js';
import { loraList } from '../lib/lora.js';
import { Executor, link } from '../lib/graph/executor.js';
import '../core/nodes.js';
import './nodes.js';

export function zimageGraph(config, { sink = 'save' } = {}) {
	const width = Math.round((config.width ?? config.size ?? 1024) / 16) * 16;
	const height = Math.round((config.height ?? config.size ?? 1024) / 16) * 16;
	const cfg = config.cfg_scale ?? 0;
	const input = config.input_image ?? config.input ?? null;
	const nodes = [];
	const add = (id, type, params) => nodes.push({ id, type, params });

	add('vae', 'core.vae', { path: config.vae });
	const sample = { width, height, steps: config.steps ?? 4, seed: config.seed ?? 42, cfg };
	if (input) {
		add('input', 'core.load_image', { path: input });
		add('encode', 'core.vae_encode', { vae: link('vae', 'vae'), image: link('input', 'image'), width, height });
		sample.latent = link('encode', 'latent');
		sample.strength = config.strength ?? 0.6;
	}
	if (config.text_model) add('text_encoder', 'core.text_encoder', { path: config.text_model });
	if (config.text_embeddings) {
		add('prompt', 'zimage.embeddings', { path: config.text_embeddings });
	} else {
		add('prompt', 'zimage.text_encode', { encoder: link('text_encoder', 'encoder'), prompt: config.prompt ?? '' });
	}
	sample.positive = link('prompt', 'cond');
	if (cfg > 1) {
		add('negative', 'zimage.text_encode', { encoder: link('text_encoder', 'encoder'), prompt: config.negative_prompt ?? '' });
		sample.negative = link('negative', 'cond');
	}
	let lora = null;
	loraList(config).forEach((l, i) => {
		add(`lora${i}`, 'core.lora', { path: l.path, strength: l.strength, ...(lora ? { lora } : {}) });
		lora = link(`lora${i}`, 'lora');
	});
	add('dit', 'zimage.load', { path: config.model, ...(lora ? { lora } : {}) });
	sample.model = link('dit', 'model');
	add('sample', 'zimage.sample', sample);
	// taef1 reads the same latents as the Flux 1 VAE: swapping it in is
	// swapping the decode's VAE.
	if (config.taesd) add('taesd', 'core.vae', { path: config.taesd });
	add('decode', 'core.vae_decode', { vae: link(config.taesd ? 'taesd' : 'vae', 'vae'), latent: link('sample', 'latent') });
	if (sink === 'save') add('save', 'core.save_image', { image: link('decode', 'image'), path: config.output ?? 'output.png' });
	else add('preview', 'core.preview', { image: link('decode', 'image') });
	return { nodes };
}

llm.plugin({
	name: 'zimage',
	async generate(config) {
		const start = llm.now();
		useMatrixCores(config.matrix_cores ?? true);
		const width = Math.round((config.width ?? config.size ?? 1024) / 16) * 16;
		const height = Math.round((config.height ?? config.size ?? 1024) / 16) * 16;
		const cfg = config.cfg_scale ?? 0;
		const input = config.input_image ?? config.input ?? null;
		llm.print(`\n=== Z-Image: ${width}x${height}, ${config.steps ?? 4} steps, seed ${config.seed ?? 42}${cfg > 1 ? ', cfg ' + cfg : ''}${input ? ', img2img strength ' + (config.strength ?? 0.6) : ''} ===`);
		const { results } = await new Executor().run(zimageGraph(config));
		llm.print(`=== done in ${llm.since(start)} ===`);
		return results.save;
	},
});
