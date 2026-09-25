// Qwen-Image 2.1 as a plugin.
//
//   llm-runner qwenimage --config qwenimage.json
//
// Qwen3-VL-8B's last hidden layer as the text conditioning, a flow-matching
// Euler loop through the single-stream DiT (optionally with true CFG and a
// negative prompt), and the model's own 64-channel VAE. `input` starts from an
// image instead of noise (img2img, `strength` of the way back to noise).
//
// The work is the nodes in `nodes.js`; `qwenImageGraph(config)` wires them.

import { llm } from '../lib/llm.js';
import { useMatrixCores } from '../lib/ops.js';
import { Executor, link } from '../lib/graph/executor.js';
import '../core/nodes.js';
import './nodes.js';

export function qwenImageGraph(config, { sink = 'save' } = {}) {
	const width = Math.round((config.width ?? config.size ?? 1024) / 32) * 32;
	const height = Math.round((config.height ?? config.size ?? 1024) / 32) * 32;
	const cfg = config.cfg_scale ?? 1;
	const input = config.input_image ?? config.input ?? null;
	const nodes = [];
	const add = (id, type, params) => nodes.push({ id, type, params });

	add('vae', 'core.vae', { path: config.vae });
	const sample = { width, height, steps: config.steps ?? 40, seed: config.seed ?? 42, cfg };
	if (input) {
		add('input', 'core.load_image', { path: input });
		add('encode', 'core.vae_encode', { vae: link('vae', 'vae'), image: link('input', 'image'), width, height });
		sample.latent = link('encode', 'latent');
		sample.strength = config.strength ?? 0.6;
	}
	add('text_encoder', 'core.text_encoder', { path: config.text_model });
	add('prompt', 'qwenimage.text_encode', { encoder: link('text_encoder', 'encoder'), prompt: config.prompt ?? '' });
	sample.positive = link('prompt', 'cond');
	if (cfg > 1) {
		add('negative', 'qwenimage.text_encode', { encoder: link('text_encoder', 'encoder'), prompt: config.negative_prompt ?? '' });
		sample.negative = link('negative', 'cond');
	}
	add('dit', 'qwenimage.load', { path: config.model });
	sample.model = link('dit', 'model');
	add('sample', 'qwenimage.sample', sample);
	add('decode', 'core.vae_decode', { vae: link('vae', 'vae'), latent: link('sample', 'latent') });
	if (sink === 'save') add('save', 'core.save_image', { image: link('decode', 'image'), path: config.output ?? 'output.png' });
	else add('preview', 'core.preview', { image: link('decode', 'image') });
	return { nodes };
}

llm.plugin({
	name: 'qwenimage',
	async generate(config) {
		const start = llm.now();
		useMatrixCores(config.matrix_cores ?? true);
		llm.print(`\n=== Qwen-Image 2.1: ${config.width ?? config.size ?? 1024}x${config.height ?? config.size ?? 1024}, ${config.steps ?? 40} steps, seed ${config.seed ?? 42} ===`);
		const { results } = await new Executor().run(qwenImageGraph(config));
		llm.print(`=== done in ${llm.since(start)} ===`);
		return results.save;
	},
});
