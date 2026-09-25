// The Qwen-Image 2.1 VAE: Wan 2.2's residual VAE cut down to one frame. RGBA
// in and out, 64 latent channels at /16.
//
// Its convolutions are already 2D (a causal 3D conv over a single frame sees
// only its last temporal tap), and the `time_conv`s never run on a first frame,
// so they are not loaded. What is left of time is in the parameter-free
// shortcuts round each down and up block (`op.avgDownAdd`, `op.dupUpAdd`).
// The norms are RMS over the channels of each pixel (`op.channelRmsNorm`).
//
//   encoder: conv_in -> five down blocks of two resnets (the first four ending
//            in a stride-2 conv), each plus its shortcut -> mid (resnet,
//            attention, resnet) -> norm, SiLU, conv_out -> quant_conv
//   decoder: post_quant_conv -> conv_in -> mid -> five up blocks of three
//            resnets (the first four ending in a 2x nearest upsample and a
//            conv), each plus its shortcut -> norm, SiLU, conv_out
//
// Ported from diffusers' `autoencoder_kl_qwenimage21.py`.

import { llm, f32 } from './llm.js';
import * as op from './ops.js';
import { submit, copy, breathe } from './gpu.js';
import { conv, disposeAll } from './flux_vae.js';

// Per-channel statistics of the latent space, from the model's `vae/config.json`:
// the DiT works in (z - mean) / std.
export const LATENT_MEAN = new Float32Array([
	0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
	2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
	-0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
	-2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
]);
export const LATENT_STD = new Float32Array([
	3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756,
	3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467,
	3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
	3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
]);

// Whether an open file is this VAE: its blocks have `downsampler.resample`s,
// where the Flux VAEs have `downsamplers.0.conv`.
export function isQwenVAE(model) {
	return model.has('decoder.up_blocks.0.upsampler.resample.1.weight') || model.has('encoder.down_blocks.0.downsampler.resample.1.weight');
}

// A resnet's weights, then a breath: the decoder is 1.35 GB of them, widened
// from bf16 on the host.
async function resblock(m, prefix) {
	const r = {
		norm1: m.upload(`${prefix}norm1.gamma`, 'f32'),
		conv1: conv(m, `${prefix}conv1.`),
		norm2: m.upload(`${prefix}norm2.gamma`, 'f32'),
		conv2: conv(m, `${prefix}conv2.`),
		shortcut: m.has(`${prefix}conv_shortcut.weight`) ? conv(m, `${prefix}conv_shortcut.`, 1) : null,
	};
	r.inC = r.conv1.inC;
	r.outC = r.conv1.outC;
	await breathe();
	return r;
}

function attention(m, prefix) {
	const qkv = conv(m, `${prefix}to_qkv.`, 1);
	return { norm: m.upload(`${prefix}norm.gamma`, 'f32'), qkv, proj: conv(m, `${prefix}proj.`, 1), channels: qkv.inC };
}

// x = conv2(silu(norm2(conv1(silu(norm1(x)))))) + shortcut(x). `a` is
// { x, t, s }; the result is left in a.x (x and s trade places).
async function runResnet(r, a, h, w) {
	const n = h * w;
	op.channelRmsNorm(a.x, r.norm1, a.t, r.inC, n, true);
	op.conv2d(r.conv1, a.t, a.s, h, w);
	await breathe();
	op.channelRmsNorm(a.s, r.norm2, a.t, r.outC, n, true);
	op.conv2d(r.conv2, a.t, a.s, h, w);
	if (r.shortcut) {
		op.conv2d(r.shortcut, a.x, a.t, h, w);
		op.add(a.s, a.t, r.outC * n);
	} else {
		op.add(a.s, a.x, r.outC * n);
	}
	[a.x, a.s] = [a.s, a.x];
}

// Queries per attention dispatch, each its own submission: the mid block's
// 1152 channels over 64x64 pixels are seconds of work in all.
const ATTENTION_CHUNK = 256;

function attentionScratch(channels, spatial) {
	const n = channels * spatial;
	return { q: f32(n), k: f32(n), v: f32(n), o: f32(n) };
}

// Single-head self-attention over the pixels, with a residual; q, k and v are
// the channel thirds of one 1x1 conv.
async function runAttention(at, a, sc, h, w) {
	const C = at.channels, S = h * w, bytes = C * S * 4;
	op.channelRmsNorm(a.x, at.norm, a.t, C, S, false);
	op.conv2d(at.qkv, a.t, a.s, h, w);
	op.transposeChannelSpatial(a.s, sc.q, C, S, 0);
	copy(a.s, a.t, bytes, bytes);
	op.transposeChannelSpatial(a.t, sc.k, C, S, 0);
	copy(a.s, a.t, bytes, 2 * bytes);
	op.transposeChannelSpatial(a.t, sc.v, C, S, 0);
	for (let start = 0; start < S; start += ATTENTION_CHUNK) {
		op.vaeAttention(sc.q, sc.k, sc.v, sc.o, C, S, start, Math.min(ATTENTION_CHUNK, S - start));
		await breathe(true);
	}
	op.transposeChannelSpatial(sc.o, a.t, C, S, 1);
	op.conv2d(at.proj, a.t, a.s, h, w);
	op.add(a.s, a.x, C * S);
	[a.x, a.s] = [a.s, a.x];
}

async function runMid(mid, a, h, w) {
	const sc = attentionScratch(mid.attn.channels, h * w);
	await runResnet(mid.res0, a, h, w);
	await runAttention(mid.attn, a, sc, h, w);
	await runResnet(mid.res1, a, h, w);
	submit();
	disposeAll(sc);
}

async function loadMid(m, prefix) {
	return {
		res0: await resblock(m, `${prefix}resnets.0.`),
		attn: attention(m, `${prefix}attentions.0.`),
		res1: await resblock(m, `${prefix}resnets.1.`),
	};
}

export class QwenVAEDecoder {
	constructor(model) { this.model = model; }

	// The weights to the GPU, a block at a time.
	async load() {
		const m = this.model;
		const t0 = llm.now();
		this.postQuant = m.has('post_quant_conv.weight') ? conv(m, 'post_quant_conv.', 1) : null;
		this.convIn = conv(m, 'decoder.conv_in.');
		this.mid = await loadMid(m, 'decoder.mid_block.');
		this.ups = [];
		for (let i = 0; m.has(`decoder.up_blocks.${i}.resnets.0.conv1.weight`); i++) {
			const p = `decoder.up_blocks.${i}.`;
			const resnets = [];
			for (let j = 0; m.has(`${p}resnets.${j}.conv1.weight`); j++) resnets.push(await resblock(m, `${p}resnets.${j}.`));
			const up = m.has(`${p}upsampler.resample.1.weight`) ? conv(m, `${p}upsampler.resample.1.`) : null;
			// A block that upsamples in time (it has a time_conv) has a temporal
			// factor of 2 in its shortcut too.
			const ftN = m.has(`${p}upsampler.time_conv.weight`) ? 2 : 1;
			this.ups.push({ resnets, up, ftN, inC: resnets[0].inC, outC: resnets[resnets.length - 1].outC });
		}
		this.normOut = m.upload('decoder.norm_out.gamma', 'f32');
		this.convOut = conv(m, 'decoder.conv_out.');
		this.latentChannels = this.convIn.inC;
		llm.print(`  VAE decoder: ${this.latentChannels} latent channels, ${this.ups.length} up blocks, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll([this.postQuant, this.convIn, this.mid, this.ups, this.normOut, this.convOut]);
	}

	// The largest activation, and the largest block input (kept for the shortcut).
	sizes(h, w) {
		let size = this.convIn.outC * h * w, save = 0;
		for (const b of this.ups) {
			save = Math.max(save, b.inC * h * w);
			size = Math.max(size, b.inC * h * w, b.outC * h * w);
			if (b.up) { h *= 2; w *= 2; size = Math.max(size, b.outC * h * w); }
		}
		return { size, save };
	}

	// `latent` is [C, h, w] in the VAE's own scale (the DiT's latent times std
	// plus mean); returns [3, 16h, 16w] floats in [0, 1], the alpha dropped.
	async decode(latent, h, w) {
		const { size, save } = this.sizes(h, w);
		const a = { x: f32(size), t: f32(size), s: f32(size) };
		const kept = f32(save);
		try {
			a.x.buffer.write(latent);
			if (this.postQuant) {
				op.conv2d(this.postQuant, a.x, a.t, h, w);
				[a.x, a.t] = [a.t, a.x];
			}
			op.conv2d(this.convIn, a.x, a.t, h, w);
			[a.x, a.t] = [a.t, a.x];
			await runMid(this.mid, a, h, w);
			await breathe(true);

			for (const b of this.ups) {
				if (b.up) copy(a.x, kept, b.inC * h * w * 4);
				for (const r of b.resnets) {
					await runResnet(r, a, h, w);
					await breathe(true);
				}
				if (b.up) {
					op.upsample2x(a.x, a.t, b.outC, h, w);
					op.conv2d(b.up, a.t, a.x, h * 2, w * 2);
					op.dupUpAdd(kept, a.x, b.inC, b.outC, h, w, b.ftN);
					h *= 2; w *= 2;
					await breathe(true);
				}
			}

			const ch = this.convOut.inC;
			op.channelRmsNorm(a.x, this.normOut, a.t, ch, h * w, true);
			op.conv2d(this.convOut, a.t, a.x, h, w);
			// [-1, 1] to [0, 1] on the first three channels (RGB; A is dropped).
			op.scaleShiftClamp(a.x, a.t, 3 * h * w, 0.5, 0.5);
			submit();
			return new Float32Array(a.t.buffer.readBytes(0, 3 * h * w * 4).buffer);
		} finally {
			disposeAll([a, kept]);
		}
	}
}

export class QwenVAEEncoder {
	constructor(model) { this.model = model; }

	async load() {
		const m = this.model;
		const t0 = llm.now();
		if (!m.has('encoder.conv_in.weight')) throw new Error(`${m.path} has no VAE encoder`);
		this.convIn = conv(m, 'encoder.conv_in.');
		this.downs = [];
		for (let i = 0; m.has(`encoder.down_blocks.${i}.resnets.0.conv1.weight`); i++) {
			const p = `encoder.down_blocks.${i}.`;
			const resnets = [];
			for (let j = 0; m.has(`${p}resnets.${j}.conv1.weight`); j++) resnets.push(await resblock(m, `${p}resnets.${j}.`));
			const down = m.has(`${p}downsampler.resample.1.weight`) ? conv(m, `${p}downsampler.resample.1.`) : null;
			const ftN = m.has(`${p}downsampler.time_conv.weight`) ? 2 : 1;
			this.downs.push({ resnets, down, ftN, inC: resnets[0].inC, outC: resnets[resnets.length - 1].outC });
		}
		this.mid = await loadMid(m, 'encoder.mid_block.');
		this.normOut = m.upload('encoder.norm_out.gamma', 'f32');
		this.convOut = conv(m, 'encoder.conv_out.');
		this.quant = m.has('quant_conv.weight') ? conv(m, 'quant_conv.', 1) : null;
		this.inChannels = this.convIn.inC;
		llm.print(`  VAE encoder: ${this.inChannels} image channels, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll([this.convIn, this.downs, this.mid, this.normOut, this.convOut, this.quant]);
	}

	// `pixels` is [inChannels, H, W] in [-1, 1]; returns the latent mean
	// [C, H/16, W/16] in the VAE's own scale.
	async encode(pixels, H, W) {
		let size = Math.max(this.inChannels, this.convIn.outC) * H * W, save = 0;
		{
			let h = H, w = W;
			for (const b of this.downs) {
				save = Math.max(save, b.inC * h * w);
				size = Math.max(size, b.inC * h * w, b.outC * h * w);
				if (b.down) { h /= 2; w /= 2; }
			}
		}
		const a = { x: f32(size), t: f32(size), s: f32(size) };
		const kept = f32(save);
		try {
			a.x.buffer.write(pixels);
			let h = H, w = W;
			op.conv2d(this.convIn, a.x, a.t, h, w);
			[a.x, a.t] = [a.t, a.x];
			await breathe(true);
			for (const b of this.downs) {
				copy(a.x, kept, b.inC * h * w * 4);
				for (const r of b.resnets) {
					await runResnet(r, a, h, w);
					await breathe(true);
				}
				if (b.down) {
					// Zero pad right and bottom, then a 3x3 stride-2 conv.
					op.conv2d(b.down, a.x, a.t, h, w, 2, 0, { h: h / 2, w: w / 2 });
					[a.x, a.t] = [a.t, a.x];
					h /= 2; w /= 2;
				}
				op.avgDownAdd(kept, a.x, b.inC, b.outC, h, w, b.ftN, b.down ? 2 : 1);
				await breathe(true);
			}
			await runMid(this.mid, a, h, w);
			const ch = this.convOut.inC;
			op.channelRmsNorm(a.x, this.normOut, a.t, ch, h * w, true);
			op.conv2d(this.convOut, a.t, a.x, h, w);
			if (this.quant) {
				op.conv2d(this.quant, a.x, a.t, h, w);
				[a.x, a.t] = [a.t, a.x];
			}
			submit();
			// The first half of the channels is the mean, the second the log-variance.
			const half = this.convOut.outC / 2;
			return { data: new Float32Array(a.x.buffer.readBytes(0, half * h * w * 4).buffer), channels: half, h, w };
		} finally {
			disposeAll([a, kept]);
		}
	}
}
