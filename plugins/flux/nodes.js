// Flux 2 (Klein) as graph nodes: the prompt encoding, the DiT loader, the
// kontext references and the sampler. The VAE is the core nodes' (format
// 'flux2'); `main.js` wires these into the fixed text-to-image, img2img and
// kontext graphs the CLI and the A1111 server run.

import { llm } from '../lib/llm.js';
import { submit } from '../lib/gpu.js';
import { eulerStep } from '../lib/ops.js';
import { TextEncoder } from '../lib/qwen3.js';
import { makeLatent, encodeImage } from '../lib/latents.js';
import { image } from '../lib/llm.js';
import { defineNodes } from '../lib/graph/registry.js';
import { FluxDiT } from './dit.js';
import { sigmas, noise } from './latent.js';
import { mergeLoras } from './lora.js';

export const KERNELS = [
	'matmul_q8', 'matmul_f32', 'matmul_f32_rows', 'rmsnorm_batch', 'head_rmsnorm_batch', 'rope_batch',
	'attention_causal', 'residual_add', 'silu_mul', 'copy_rows', 'patchify', 'unpatchify', 'timestep_embed',
	'silu', 'batch_layernorm', 'adaln_modulate', 'batch_head_norm', 'transpose_heads', 'mrope',
	'flash_attention', 'gated_residual_linear', 'concat_rows', 'flow_euler_step',
	'conv2d', 'conv2d_3x3', 'group_norm', 'upsample_nearest', 'transpose_channel_spatial', 'linear_bias',
	'vae_attention', 'scale_shift_clamp', 'lora_merge_q8', 'lokr_merge_q8',
];

const TEXT_MAX = 512;
const MAX_REFS = 4;

// Qwen3 chat-template wrap, padded with <|endoftext|> to a multiple of 64.
export async function encodePrompt(textModel, prompt, textPad = 0) {
	const t0 = llm.now();
	const enc = new TextEncoder(textModel);
	const wrapped = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
	const raw = enc.tokenizer.encode(wrapped, true);
	const real = Math.min(raw.length, TEXT_MAX);
	// Padded to the matmul tile, not a fixed 512: there is no attention mask, so
	// pad tokens take part in the joint attention. `text_pad` forces a length
	// (the C3 pipeline's FLUX_TEXT_PAD), for comparing against a fixed-pad run.
	const want = textPad ? Math.ceil(textPad / 64) * 64 : Math.ceil(real / 64) * 64;
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
	const text = await enc.encodeLayers(tokens, layers);
	enc.close();
	llm.print(`  [phase] text_encode: ${llm.since(t0)}`);
	return { text, nTxt: padded, textDim: 3 * enc.config.dim };
}

// References packed token-major, one after another: [tokens, channels].
function packReferences(refs) {
	const channels = refs[0].data.length / (refs[0].w * refs[0].h);
	const total = refs.reduce((a, r) => a + r.w * r.h, 0);
	const packed = new Float32Array(total * channels);
	let at = 0;
	for (const r of refs) {
		const n = r.w * r.h;
		for (let p = 0; p < n; p++) for (let c = 0; c < channels; c++) packed[(at + p) * channels + c] = r.data[c * n + p];
		at += n;
	}
	return { packed, total };
}

defineNodes('flux', {
	'flux.load': {
		title: 'Flux 2 DiT',
		category: 'loaders',
		description: 'A Flux 2 (Klein) DiT GGUF, with any LoRAs folded in. Stays in VRAM while it is wired to something.',
		inputs: { path: 'PATH(dit)', lora: 'LORA?' },
		outputs: { model: 'MODEL' },
		async run({ path, lora }) {
			const t0 = llm.now();
			const file = llm.open(path);
			const dit = new FluxDiT(file);
			await dit.load();
			if (lora && lora.length > 0) mergeLoras(dit, lora);
			llm.print(`  [phase] dit_load: ${llm.since(t0)}`);
			return {
				model: {
					family: 'flux2',
					dit,
					dispose() { dit.release(); dit.unload(); file.close(); },
				},
			};
		},
	},

	'flux.text_encode': {
		title: 'Flux 2 prompt',
		category: 'conditioning',
		description: 'Qwen3 hidden states from three layers, side by side.',
		inputs: { encoder: 'TEXT_ENCODER', prompt: 'STRING*=', text_pad: 'INT=0' },
		outputs: { cond: 'CONDITIONING' },
		async run({ encoder, prompt, text_pad }) {
			const { text, nTxt, textDim } = await encodePrompt(encoder.path, prompt, text_pad);
			return { cond: { family: 'flux2', tensor: text, n: nTxt, dim: textDim, dispose() { text.dispose(); } } };
		},
	},

	'flux.reference': {
		title: 'Flux 2 reference image',
		category: 'conditioning',
		description: 'An image for kontext editing, encoded at its own aspect ratio with the long side at most `long_side`. Chain up to four.',
		inputs: { vae: 'VAE', image: 'IMAGE', refs: 'REFERENCES?', long_side: 'INT=1024' },
		outputs: { refs: 'REFERENCES' },
		async run({ vae, image: src, refs, long_side }) {
			if (vae.format !== 'flux2') throw new Error(`flux.reference needs a Flux 2 VAE; ${vae.path} is ${vae.kind}`);
			const list = refs ?? [];
			if (list.length >= MAX_REFS) throw new Error(`kontext takes at most ${MAX_REFS} reference images`);
			// References are only ever scaled down.
			const ceiling = Math.floor(long_side / 16) * 16;
			const native = Math.max(src.width, src.height);
			const img = image.fit16(src, Math.min(ceiling, native));
			const t0 = llm.now();
			const latent = await encodeImage(vae.path, img);
			llm.print(`  [phase] ref_vae_encode: ${llm.since(t0)}`);
			return { refs: [...list, { w: img.width / 16, h: img.height / 16, data: latent.data }] };
		},
	},

	'flux.sample': {
		title: 'Flux 2 sampler',
		category: 'sampling',
		description: 'Flow-matching Euler steps. With a latent it is img2img from `strength` of the way to noise; references make it a kontext edit.',
		inputs: {
			cond: 'CONDITIONING',
			refs: 'REFERENCES?',
			latent: 'LATENT(flux2)?',
			model: 'MODEL',
			width: 'INT=1024',
			height: 'INT=1024',
			steps: 'INT=4',
			seed: 'INT=42',
			strength: 'FLOAT=0.6',
		},
		outputs: { latent: 'LATENT(flux2)' },
		async run({ cond, refs, latent: init, model, width, height, steps, seed, strength }, ctx) {
			if (cond.family !== 'flux2') throw new Error(`flux.sample needs a Flux 2 prompt, not ${cond.family}`);
			if (model.family !== 'flux2') throw new Error(`flux.sample needs a Flux 2 DiT, not ${model.family}`);
			if (init && init.format !== 'flux2') throw new Error(`flux.sample starts from flux2 latents; this one is ${init.format}`);
			const latentH = init ? init.h : Math.ceil(height / 16);
			const latentW = init ? init.w : Math.ceil(width / 16);
			const dit = model.dit;

			const t1 = llm.now();
			let refDims = [];
			let refPatches = null;
			if (refs && refs.length > 0) {
				const p = packReferences(refs);
				refDims = refs.map((r) => ({ w: r.w, h: r.h }));
				refPatches = p.packed;
				llm.print(`  kontext: ${refs.length} reference image(s), ${p.total} tokens`);
			}
			dit.prepare({ nTxt: cond.n, latentH, latentW, refs: refDims, refPatches });
			llm.print(`  [phase] dit_setup: ${llm.since(t1)}`);

			const t2 = llm.now();
			const nImg = latentH * latentW;
			const schedule = sigmas(steps, nImg, init ? strength : 1);
			const x = noise(dit.config.patchDim * nImg, seed);
			if (init) {
				const s0 = schedule[0];
				for (let i = 0; i < x.length; i++) x[i] = s0 * x[i] + (1 - s0) * init.data[i];
			}
			dit.a.latent.buffer.write(x);
			try {
				for (let s = 0; s < steps; s++) {
					const ts = llm.now();
					await dit.forward(cond.tensor, schedule[s]);
					eulerStep(dit.a.latent, dit.a.velocity, x.length, schedule[s + 1] - schedule[s]);
					submit();
					llm.print(`  step ${s + 1}/${steps}: sigma ${schedule[s].toFixed(4)}  ${(llm.now() - ts).toFixed(0)}ms`);
					// `latent` is read back only if whoever is watching asks for it.
					await ctx.progress(s + 1, steps, {
						latent: () => makeLatent('flux2', new Float32Array(dit.a.latent.buffer.readBytes().buffer), latentH, latentW),
					});
				}
				const out = new Float32Array(dit.a.latent.buffer.readBytes().buffer);
				llm.print(`  [phase] denoise: ${llm.since(t2)}`);
				return { latent: makeLatent('flux2', out, latentH, latentW) };
			} finally {
				dit.release();
			}
		},
	},
});
