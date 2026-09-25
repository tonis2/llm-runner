// Flux 2 (Klein) as a plugin: text to image, img2img and Kontext-style editing.
//
//   llm-runner flux --config flux-t2i.json
//   llm-runner flux --config flux.json --server --port 7860
//
// The pipeline is `dependencies/flux.c3l`'s: Qwen3 hidden states from three
// layers as the text conditioning, flow-matching Euler steps through the DiT,
// and the Flux 2 VAE. Settings are the C3 CLI's JSON keys.

import { llm, image } from '../lib/llm.js';
import { precompile, submit } from '../lib/gpu.js';
import { eulerStep } from '../lib/ops.js';
import * as op from '../lib/ops.js';
import { TextEncoder } from '../lib/qwen3.js';
import { FluxVAEDecoder } from '../lib/flux_vae.js';
import { FluxDiT } from './dit.js';
import { sigmas, noise, decodeLatent, encodeReference } from './latent.js';
import { mergeLoras } from './lora.js';
import { loraList } from '../lib/lora.js';

const KERNELS = [
	'matmul_q8', 'matmul_f32', 'matmul_f32_rows', 'rmsnorm_batch', 'head_rmsnorm_batch', 'rope_batch',
	'attention_causal', 'residual_add', 'silu_mul', 'copy_rows', 'patchify', 'unpatchify', 'timestep_embed',
	'silu', 'batch_layernorm', 'adaln_modulate', 'batch_head_norm', 'transpose_heads', 'mrope',
	'flash_attention', 'gated_residual_linear', 'concat_rows', 'flow_euler_step',
	'conv2d', 'conv2d_3x3', 'group_norm', 'upsample_nearest', 'transpose_channel_spatial', 'linear_bias',
	'vae_attention', 'scale_shift_clamp', 'lora_merge_q8', 'lokr_merge_q8',
];

const TEXT_MAX = 512;

const state = {
	config: null,
	dit: null,           // FluxDiT, weights resident between requests with keep_dit
	ditModel: null,
};

// Qwen3 chat-template wrap, padded with <|endoftext|> to a multiple of 64.
function encodePrompt(config, prompt) {
	const t0 = llm.now();
	const enc = new TextEncoder(config.text_model);
	const wrapped = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
	const raw = enc.tokenizer.encode(wrapped, true);
	const real = Math.min(raw.length, TEXT_MAX);
	// Padded to the matmul tile, not a fixed 512: there is no attention mask, so
	// pad tokens take part in the joint attention. `text_pad` forces a length
	// (the C3 pipeline's FLUX_TEXT_PAD), for comparing against a fixed-pad run.
	const want = config.text_pad ? Math.ceil(config.text_pad / 64) * 64 : Math.ceil(real / 64) * 64;
	const padded = Math.min(Math.max(want, 64), TEXT_MAX);
	const pad = enc.tokenizer.find('<|endoftext|>') || enc.tokenizer.pad;
	const tokens = new Uint32Array(padded).fill(pad);
	tokens.set(raw.subarray(0, real));
	// Klein reads hidden states after layers n/4, n/2 and 3n/4 (0-indexed
	// 8, 17, 26 for Qwen3-8B's 36).
	const n = enc.config.nLayers;
	const spacing = Math.floor(n / 4);
	const layers = n >= 12 ? [spacing - 1, 2 * spacing - 1, 3 * spacing - 1] : [0, Math.floor(n / 2), n - 1];
	llm.print(`  text: ${real} tokens, padded to ${padded}; layers ${layers.join(', ')} of ${n}`);
	const text = enc.encodeLayers(tokens, layers);
	enc.close();
	llm.print(`  [phase] text_encode: ${llm.since(t0)}`);
	return { text, nTxt: padded, textDim: 3 * enc.config.dim };
}

async function generate(request) {
	const config = { ...state.config, ...request };
	const start = llm.now();
	const width = Math.ceil((config.width ?? config.size ?? 1024) / 16) * 16;
	const height = Math.ceil((config.height ?? config.size ?? 1024) / 16) * 16;
	const steps = config.steps ?? 4;
	const seed = config.seed ?? 42;
	const mode = config.edit_mode ?? '';
	const latentH = height / 16, latentW = width / 16;
	llm.print(`\n=== Flux: ${width}x${height}, ${steps} steps, seed ${seed}${mode ? ', ' + mode : ''} ===`);

	const { text, nTxt } = encodePrompt(config, config.prompt ?? '');

	// Reference images: img2img starts from one, kontext attends to several.
	let refs = [];
	let refPatches = null;
	let initLatent = null;
	const inputs = config.input_images ?? (config.input ? (Array.isArray(config.input) ? config.input : [config.input]) : []);
	if (mode === 'img2img' || mode === 'kontext') {
		if (inputs.length === 0) throw new Error(`edit_mode ${mode} needs an input image`);
		const t0 = llm.now();
		const encoded = encodeReference(config, inputs, mode, width, height);
		refs = encoded.refs;
		refPatches = encoded.refPatches;
		initLatent = encoded.initLatent;
		llm.print(`  [phase] ref_vae_encode: ${llm.since(t0)}`);
	}

	// DiT: resident across requests with keep_dit, else loaded for this one.
	const t1 = llm.now();
	if (!state.dit) {
		state.ditModel = llm.open(config.model);
		state.dit = new FluxDiT(state.ditModel);
	}
	const dit = state.dit;
	const loras = loraList(config);
	if (dit.loaded && dit.loraSig !== JSON.stringify(loras)) dit.unload();
	if (!dit.loaded) {
		dit.load();
		if (loras.length > 0) mergeLoras(dit, loras);
		dit.loraSig = JSON.stringify(loras);
	}
	dit.prepare({ nTxt, latentH, latentW, refs, refPatches });
	llm.print(`  [phase] dit_setup: ${llm.since(t1)}`);

	// Denoise.
	const t2 = llm.now();
	const nImg = latentH * latentW;
	const schedule = sigmas(steps, nImg, mode === 'img2img' ? (config.strength ?? 0.6) : 1);
	const latent = noise(dit.config.patchDim * nImg, seed);
	if (initLatent) {
		const s0 = schedule[0];
		for (let i = 0; i < latent.length; i++) latent[i] = s0 * latent[i] + (1 - s0) * initLatent[i];
	}
	dit.a.latent.buffer.write(latent);
	for (let s = 0; s < steps; s++) {
		const ts = llm.now();
		dit.forward(text, schedule[s]);
		eulerStep(dit.a.latent, dit.a.velocity, latent.length, schedule[s + 1] - schedule[s]);
		submit();
		llm.print(`  step ${s + 1}/${steps}: sigma ${schedule[s].toFixed(4)}  ${(llm.now() - ts).toFixed(0)}ms`);
	}
	const finalLatent = new Float32Array(dit.a.latent.buffer.readBytes().buffer);
	dit.release();
	text.dispose();
	if (!config.keep_dit) {
		dit.unload();
		state.ditModel.close();
		state.dit = null;
		state.ditModel = null;
	}
	llm.print(`  [phase] denoise: ${llm.since(t2)}`);

	// Decode.
	const t3 = llm.now();
	const pixels = decodeLatent(config, finalLatent, latentH, latentW);
	const img = image.fromTensor(pixels, width, height, 3);
	llm.print(`  [phase] vae_decode: ${llm.since(t3)}`);
	llm.print(`=== done in ${llm.since(start)} ===`);
	return img;
}

llm.plugin({
	name: 'flux',
	load(config) {
		state.config = config;
		const t0 = llm.now();
		op.useMatrixCores(config.matrix_cores ?? true);
		const kernels = op.matrixCoresOn() ? [...KERNELS, 'matmul_q8_coop', 'flash_attention_coop', 'conv2d_coop'] : KERNELS;
		precompile(kernels);
		llm.print(`flux: ${kernels.length} kernels ready in ${llm.since(t0)}${op.matrixCoresOn() ? ', matmuls on the matrix cores' : ''}`);
		if (config.keep_dit) {
			state.ditModel = llm.open(config.model);
			state.dit = new FluxDiT(state.ditModel);
			state.dit.load();
			state.dit.loraSig = JSON.stringify([]);
		}
	},
	async generate(config) {
		const img = await generate(config);
		const out = config.output ?? 'output.png';
		image.savePng(out, img);
		llm.print(`  saved ${out} (${img.width}x${img.height})`);
		return { output: out, width: img.width, height: img.height };
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
		}
		const img = await generate(request);
		const png = image.encodePng(img);
		// `data` is what the C3 server answered with; `images` is A1111's shape.
		const b64 = llm.base64Encode(png);
		llm.respond(200, 'application/json', JSON.stringify({ data: b64, images: [b64] }));
	},
});
