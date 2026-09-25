// Flux 2 (Klein) as a plugin: text to image, img2img and Kontext-style editing.
//
//   llm-runner flux --config flux-t2i.json
//   llm-runner flux --config flux.json --server --port 7860
//
// Qwen3 hidden states from three layers as the text conditioning, flow-matching
// Euler steps through the DiT, and the Flux 2 VAE. The work is the nodes in
// `nodes.js`; this wires them into a fixed graph from the C3 CLI's JSON keys, so
// `fluxGraph(config)` is also what the same run looks like as a graph file.

import { llm, image } from '../lib/llm.js';
import { precompile } from '../lib/gpu.js';
import * as op from '../lib/ops.js';
import { loraList } from '../lib/lora.js';
import { Executor, link } from '../lib/graph/executor.js';
import '../core/nodes.js';
import { KERNELS } from './nodes.js';

// The run a config describes, as a graph. `images` are the reference or
// starting images (paths or decoded images); `sink` is 'save' (to
// config.output) or 'preview' (handed back).
export function fluxGraph(config, { images = [], sink = 'save' } = {}) {
	const width = Math.ceil((config.width ?? config.size ?? 1024) / 16) * 16;
	const height = Math.ceil((config.height ?? config.size ?? 1024) / 16) * 16;
	const mode = config.edit_mode ?? '';
	const nodes = [];
	const add = (id, type, params) => nodes.push({ id, type, params });

	add('text_encoder', 'core.text_encoder', { path: config.text_model });
	add('prompt', 'flux.text_encode', { encoder: link('text_encoder', 'encoder'), prompt: config.prompt ?? '', text_pad: config.text_pad ?? 0 });
	add('vae', 'core.vae', { path: config.vae });
	let lora = null;
	loraList(config).forEach((l, i) => {
		add(`lora${i}`, 'core.lora', { path: l.path, strength: l.strength, ...(lora ? { lora } : {}) });
		lora = link(`lora${i}`, 'lora');
	});
	add('dit', 'flux.load', { path: config.model, ...(lora ? { lora } : {}) });

	const sample = {
		cond: link('prompt', 'cond'), model: link('dit', 'model'),
		width, height, steps: config.steps ?? 4, seed: config.seed ?? 42,
	};
	if ((mode === 'img2img' || mode === 'kontext') && images.length === 0) {
		throw new Error(`edit_mode ${mode} needs an input image`);
	}
	if (mode === 'img2img') {
		add('input', 'core.load_image', { path: images[0] });
		add('encode', 'core.vae_encode', { vae: link('vae', 'vae'), image: link('input', 'image'), width, height });
		sample.latent = link('encode', 'latent');
		sample.strength = config.strength ?? 0.6;
	} else if (mode === 'kontext') {
		if (images.length > 4) throw new Error(`kontext takes at most 4 reference images, not ${images.length}`);
		// References are only ever scaled down: to the output's long side, or to
		// max_ref_long_side when that is set lower.
		const longSide = config.max_ref_long_side ? Math.floor(config.max_ref_long_side / 16) * 16 : Math.max(width, height);
		images.forEach((img, i) => {
			add(`input${i}`, 'core.load_image', { path: img });
			add(`ref${i}`, 'flux.reference', {
				vae: link('vae', 'vae'), image: link(`input${i}`, 'image'), long_side: longSide,
				...(i > 0 ? { refs: link(`ref${i - 1}`, 'refs') } : {}),
			});
		});
		sample.refs = link(`ref${images.length - 1}`, 'refs');
	}
	add('sample', 'flux.sample', sample);
	add('decode', 'core.vae_decode', { vae: link('vae', 'vae'), latent: link('sample', 'latent') });
	if (sink === 'save') add('save', 'core.save_image', { image: link('decode', 'image'), path: config.output ?? 'output.png' });
	else add('preview', 'core.preview', { image: link('decode', 'image') });
	return { nodes };
}

function inputImages(config) {
	if (config.input_images) return config.input_images;
	if (config.input) return Array.isArray(config.input) ? config.input : [config.input];
	return [];
}

const state = {
	config: null,
	executor: null, // with keep_dit, results (the DiT above all) stay between requests
};

async function generate(request, sink) {
	const config = { ...state.config, ...request };
	const start = llm.now();
	const width = Math.ceil((config.width ?? config.size ?? 1024) / 16) * 16;
	const height = Math.ceil((config.height ?? config.size ?? 1024) / 16) * 16;
	const mode = config.edit_mode ?? '';
	llm.print(`\n=== Flux: ${width}x${height}, ${config.steps ?? 4} steps, seed ${config.seed ?? 42}${mode ? ', ' + mode : ''} ===`);
	const graph = fluxGraph(config, { images: inputImages(config), sink });
	const { results } = await state.executor.run(graph);
	llm.print(`=== done in ${llm.since(start)} ===`);
	return results[sink];
}

llm.plugin({
	name: 'flux',
	load(config) {
		state.config = config;
		state.executor = new Executor({ keep: !!config.keep_dit });
		const t0 = llm.now();
		op.useMatrixCores(config.matrix_cores ?? true);
		const kernels = op.matrixCoresOn() ? [...KERNELS, 'matmul_q8_coop', 'flash_attention_coop', 'conv2d_coop'] : KERNELS;
		precompile(kernels);
		llm.print(`flux: ${kernels.length} kernels ready in ${llm.since(t0)}${op.matrixCoresOn() ? ', matmuls on the matrix cores' : ''}`);
		// Load the DiT now rather than on the first request.
		if (config.keep_dit) return state.executor.run(fluxGraph({ ...config, edit_mode: '' }), { targets: ['dit'] });
	},
	async generate(config) {
		return generate(config, 'save');
	},
	// A1111-style: POST /sdapi/v1/txt2img or /img2img with { prompt, init_images,
	// steps, seed, width, height, loras } -> { data: base64 png, images: [same] }.
	async handle(req) {
		if (req.method !== 'POST' || !req.path.startsWith('/sdapi/v1/')) {
			llm.respond(404, 'application/json', JSON.stringify({ error: 'POST /sdapi/v1/txt2img or /sdapi/v1/img2img' }));
			return;
		}
		const body = JSON.parse(req.body || '{}');
		const request = { prompt: body.prompt ?? '' };
		for (const k of ['steps', 'seed', 'width', 'height', 'strength']) if (body[k] !== undefined) request[k] = body[k];
		if (body.loras ?? body.lora) request.lora = body.loras ?? body.lora;
		if (Array.isArray(body.init_images) && body.init_images.length > 0) {
			request.input_images = body.init_images.map((b64) => image.decode(llm.base64Decode(b64)));
			request.edit_mode = body.edit_mode ?? 'kontext';
		} else {
			request.edit_mode = '';
			request.input_images = [];
		}
		const { image: img } = await generate(request, 'preview');
		const png = image.encodePng(img);
		// `data` is what the C3 server answered with; `images` is A1111's shape.
		const b64 = llm.base64Encode(png);
		llm.respond(200, 'application/json', JSON.stringify({ data: b64, images: [b64] }));
	},
});
