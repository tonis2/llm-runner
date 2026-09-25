// The Flux KL-VAE decoder: latent [C, h, w] to image [3, 8h, 8w].
//
// conv_in -> mid (resnet, attention, resnet) -> four up stages of three
// resnets each, the first three ending in a 2x nearest upsample and a conv ->
// GroupNorm, SiLU, conv_out. Both tensor namings are read: diffusers'
// (`decoder.up_blocks.N.resnets.M`) and the original checkpoint's
// (`decoder.up.N.block.M`, numbered from the output end).
//
// Ported from the C3 Z-Image pipeline (`flux_vae.c3`, since deleted).

import { llm, f32 } from './llm.js';
import * as op from './ops.js';
import { submit, copy, breathe } from './gpu.js';

function conv(model, prefix, k = 3) {
	const shape = model.shape(`${prefix}weight`); // [kw, kh, in, out]
	return {
		weight: model.upload(`${prefix}weight`, 'conv'),
		bias: model.upload(`${prefix}bias`, 'f32'),
		inC: shape[2],
		outC: shape[3],
		k: shape[0] ?? k,
	};
}

function resnet(model, prefix) {
	const r = {
		norm1w: model.upload(`${prefix}norm1.weight`, 'f32'),
		norm1b: model.upload(`${prefix}norm1.bias`, 'f32'),
		conv1: conv(model, `${prefix}conv1.`),
		norm2w: model.upload(`${prefix}norm2.weight`, 'f32'),
		norm2b: model.upload(`${prefix}norm2.bias`, 'f32'),
		conv2: conv(model, `${prefix}conv2.`),
		shortcut: null,
	};
	for (const s of ['conv_shortcut.', 'nin_shortcut.']) {
		if (model.has(`${prefix}${s}weight`)) r.shortcut = conv(model, `${prefix}${s}`, 1);
	}
	r.inC = r.conv1.inC;
	r.outC = r.conv1.outC;
	return r;
}

function attention(model, prefix) {
	const diffusers = model.has(`${prefix}group_norm.weight`);
	const n = diffusers
		? { norm: 'group_norm', q: 'to_q', k: 'to_k', v: 'to_v', out: 'to_out.0' }
		: { norm: 'norm', q: 'q', k: 'k', v: 'v', out: 'proj_out' };
	const t = (name) => model.upload(`${prefix}${name}`, 'f32');
	return {
		normw: t(`${n.norm}.weight`), normb: t(`${n.norm}.bias`),
		qw: t(`${n.q}.weight`), qb: t(`${n.q}.bias`),
		kw: t(`${n.k}.weight`), kb: t(`${n.k}.bias`),
		vw: t(`${n.v}.weight`), vb: t(`${n.v}.bias`),
		ow: t(`${n.out}.weight`), ob: t(`${n.out}.bias`),
		// [in, out] for a linear, [1, 1, in, out] for the old 1x1-conv spelling.
		channels: (() => { const sh = model.shape(`${prefix}${n.q}.weight`); return sh.length === 4 ? sh[2] : sh[0]; })(),
	};
}

function disposeAll(value) {
	if (!value) return;
	if (typeof value.dispose === 'function') { value.dispose(); return; }
	if (Array.isArray(value)) { for (const v of value) disposeAll(v); return; }
	if (typeof value === 'object') for (const v of Object.values(value)) if (typeof v === 'object') disposeAll(v);
}

export class FluxVAEDecoder {
	// `model` is an open safetensors (or GGUF) file holding `decoder.*`. With
	// `quantConv`, the file's post_quant_conv (a 1x1 conv) runs ahead of conv_in.
	constructor(model, { quantConv = false } = {}) {
		const m = model;
		const t0 = llm.now();
		const diffusers = m.has('decoder.mid_block.resnets.0.conv1.weight');
		this.postQuant = quantConv && m.has('post_quant_conv.weight') ? conv(m, 'post_quant_conv.', 1) : null;
		this.convIn = conv(m, 'decoder.conv_in.');
		if (diffusers) {
			this.mid1 = resnet(m, 'decoder.mid_block.resnets.0.');
			this.midAttn = attention(m, 'decoder.mid_block.attentions.0.');
			this.mid2 = resnet(m, 'decoder.mid_block.resnets.1.');
		} else {
			this.mid1 = resnet(m, 'decoder.mid.block_1.');
			this.midAttn = attention(m, 'decoder.mid.attn_1.');
			this.mid2 = resnet(m, 'decoder.mid.block_2.');
		}
		// Stages from the latent end: diffusers' up_blocks.0 is the original up.3.
		this.stages = [];
		for (let s = 0; s < 4; s++) {
			const blocks = [];
			for (let i = 0; i < 3; i++) {
				blocks.push(resnet(m, diffusers ? `decoder.up_blocks.${s}.resnets.${i}.` : `decoder.up.${3 - s}.block.${i}.`));
			}
			let upsample = null;
			if (s < 3) upsample = conv(m, diffusers ? `decoder.up_blocks.${s}.upsamplers.0.conv.` : `decoder.up.${3 - s}.upsample.conv.`);
			this.stages.push({ blocks, upsample });
		}
		const norm = diffusers ? 'decoder.conv_norm_out.' : 'decoder.norm_out.';
		this.normOutW = m.upload(`${norm}weight`, 'f32');
		this.normOutB = m.upload(`${norm}bias`, 'f32');
		this.convOut = conv(m, 'decoder.conv_out.');
		this.latentChannels = this.convIn.inC;
		llm.print(`  VAE decoder: ${this.latentChannels} latent channels, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll([this.postQuant, this.convIn, this.mid1, this.midAttn, this.mid2, this.stages, this.normOutW, this.normOutB, this.convOut]);
	}

	// Decode `latent` (Float32Array [C, h, w]) and return [3, 8h, 8w] floats in [0, 1].
	async decode(latent, h, w) {
		const H = h * 8, W = w * 8;
		// The widest stage: 256 channels at full resolution, or 512 at half.
		const size = Math.max(256 * H * W, 512 * (H / 2) * (W / 2), 512 * h * w);
		const a = { x: f32(size), t: f32(size), s: f32(size) };
		const attn = attentionScratch(h * w);
		a.x.buffer.write(latent);
		if (this.postQuant) {
			op.conv2d(this.postQuant, a.x, a.t, h, w);
			copy(a.t, a.x, this.postQuant.outC * h * w * 4);
		}

		let r = op.conv2d(this.convIn, a.x, a.t, h, w);
		copy(a.t, a.x, this.convIn.outC * h * w * 4);
		await this.resnet(this.mid1, a, h, w);
		await this.attention(this.midAttn, a, attn, h, w);
		await this.resnet(this.mid2, a, h, w);
		submit();

		for (const stage of this.stages) {
			for (const b of stage.blocks) {
				await this.resnet(b, a, h, w);
				await breathe(true);
			}
			if (stage.upsample) {
				const ch = stage.blocks[2].outC;
				op.upsample2x(a.x, a.t, ch, h, w);
				h *= 2; w *= 2;
				op.conv2d(stage.upsample, a.t, a.x, h, w);
				await breathe(true);
			}
		}

		const ch = this.stages[3].blocks[2].outC;
		op.groupNorm(a.x, this.normOutW, this.normOutB, a.t, ch, h * w);
		op.silu(a.t, ch * h * w);
		op.conv2d(this.convOut, a.t, a.x, h, w);
		op.scaleShiftClamp(a.x, a.t, 3 * h * w, 0.5, 0.5);
		submit();
		const out = new Float32Array(a.t.buffer.readBytes(0, 3 * h * w * 4).buffer);
		disposeAll([a, attn]);
		return out;
	}

	resnet(r, a, h, w) { return runResnet(r, a, h, w); }

	attention(at, a, t, h, w) { return runAttention(at, a, t, h, w); }
}

// x = x + conv2(silu(norm2(conv1(silu(norm1(x)))))), with a 1x1 shortcut on a
// channel change. `a` is { x, t, s }: the state, and two scratch buffers.
async function runResnet(r, a, h, w) {
	const n = h * w;
	op.groupNorm(a.x, r.norm1w, r.norm1b, a.t, r.inC, n);
	op.silu(a.t, r.inC * n);
	op.conv2d(r.conv1, a.t, a.s, h, w);
	await breathe();
	op.groupNorm(a.s, r.norm2w, r.norm2b, a.t, r.outC, n);
	op.silu(a.t, r.outC * n);
	op.conv2d(r.conv2, a.t, a.s, h, w);
	if (r.shortcut) {
		op.conv2d(r.shortcut, a.x, a.t, h, w);
		op.add(a.s, a.t, r.outC * n);
	} else {
		op.add(a.s, a.x, r.outC * n);
	}
	copy(a.s, a.x, r.outC * n * 4);
}

// Queries per attention dispatch: a window gets a frame between pieces.
const ATTENTION_CHUNK = 512;

// Single-head self-attention over the h*w positions, with a residual.
async function runAttention(at, a, t, h, w) {
	const C = at.channels, S = h * w;
	copy(a.x, t.save, C * S * 4);
	op.groupNorm(a.x, at.normw, at.normb, a.t, C, S);
	op.transposeChannelSpatial(a.t, a.s, C, S, 0);
	op.linearBias(at.qw, at.qb, a.s, t.q, C, C, S);
	op.linearBias(at.kw, at.kb, a.s, t.k, C, C, S);
	op.linearBias(at.vw, at.vb, a.s, t.v, C, C, S);
	for (let start = 0; start < S; start += ATTENTION_CHUNK) {
		op.vaeAttention(t.q, t.k, t.v, t.o, C, S, start, Math.min(ATTENTION_CHUNK, S - start));
		await breathe();
	}
	op.linearBias(at.ow, at.ob, t.o, a.s, C, C, S);
	op.transposeChannelSpatial(a.s, a.x, C, S, 1);
	op.add(a.x, t.save, C * S);
}

function attentionScratch(spatial) {
	return {
		save: f32(512 * spatial), q: f32(512 * spatial), k: f32(512 * spatial),
		v: f32(512 * spatial), o: f32(512 * spatial),
	};
}

// The Flux VAE encoder: image [3, H, W] in [-1, 1] to [2 * C, H/8, W/8] - the
// mean and log-variance of the latent, in that order.
//
// conv_in -> four down stages of two resnets, the first three ending in a
// stride-2 conv -> mid (resnet, attention, resnet) -> GroupNorm, SiLU,
// conv_out. Ported from `flux_vae_encoder.c3`. Both namings are read, as for
// the decoder; the original checkpoint's stages count from the image end too.
export class FluxVAEEncoder {
	// With `quantConv`, the file's quant_conv (a 1x1 conv) runs after conv_out.
	constructor(model, { quantConv = false } = {}) {
		const m = model;
		const t0 = llm.now();
		if (!m.has('encoder.conv_in.weight')) throw new Error(`${m.path} has no VAE encoder`);
		const diffusers = m.has('encoder.mid_block.resnets.0.conv1.weight');
		this.convIn = conv(m, 'encoder.conv_in.');
		this.stages = [];
		for (let s = 0; s < 4; s++) {
			const blocks = [0, 1].map((i) => resnet(m, diffusers ? `encoder.down_blocks.${s}.resnets.${i}.` : `encoder.down.${s}.block.${i}.`));
			const down = s < 3 ? conv(m, diffusers ? `encoder.down_blocks.${s}.downsamplers.0.conv.` : `encoder.down.${s}.downsample.conv.`) : null;
			this.stages.push({ blocks, down });
		}
		const mid = diffusers
			? ['encoder.mid_block.resnets.0.', 'encoder.mid_block.attentions.0.', 'encoder.mid_block.resnets.1.']
			: ['encoder.mid.block_1.', 'encoder.mid.attn_1.', 'encoder.mid.block_2.'];
		this.mid1 = resnet(m, mid[0]);
		this.midAttn = attention(m, mid[1]);
		this.mid2 = resnet(m, mid[2]);
		const norm = diffusers ? 'encoder.conv_norm_out.' : 'encoder.norm_out.';
		this.normOutW = m.upload(`${norm}weight`, 'f32');
		this.normOutB = m.upload(`${norm}bias`, 'f32');
		this.convOut = conv(m, 'encoder.conv_out.');
		this.quant = quantConv && m.has('quant_conv.weight') ? conv(m, 'quant_conv.', 1) : null;
		llm.print(`  VAE encoder: loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll([this.convIn, this.stages, this.mid1, this.midAttn, this.mid2, this.normOutW, this.normOutB, this.convOut, this.quant]);
	}

	// `pixels` is [3, H, W] floats in [-1, 1]; returns [outC, H/8, W/8].
	async encode(pixels, H, W) {
		const size = Math.max(256 * H * W, 3 * H * W);
		const a = { x: f32(size), t: f32(size), s: f32(size) };
		a.x.buffer.write(pixels);
		let h = H, w = W;
		op.conv2d(this.convIn, a.x, a.t, h, w);
		copy(a.t, a.x, this.convIn.outC * h * w * 4);
		submit();
		for (const stage of this.stages) {
			for (const b of stage.blocks) {
				await runResnet(b, a, h, w);
				await breathe(true);
			}
			if (stage.down) {
				op.conv2d(stage.down, a.x, a.t, h, w, 2, 0, { h: h / 2, w: w / 2 });
				h /= 2; w /= 2;
				copy(a.t, a.x, stage.down.outC * h * w * 4);
				await breathe(true);
			}
		}
		const t = attentionScratch(h * w);
		await runResnet(this.mid1, a, h, w);
		await runAttention(this.midAttn, a, t, h, w);
		await runResnet(this.mid2, a, h, w);
		op.groupNorm(a.x, this.normOutW, this.normOutB, a.t, 512, h * w);
		op.silu(a.t, 512 * h * w);
		op.conv2d(this.convOut, a.t, a.x, h, w);
		if (this.quant) {
			op.conv2d(this.quant, a.x, a.t, h, w);
			copy(a.t, a.x, this.quant.outC * h * w * 4);
		}
		submit();
		const outC = this.convOut.outC;
		const out = new Float32Array(a.x.buffer.readBytes(0, outC * h * w * 4).buffer);
		disposeAll([a, t]);
		return { data: out, channels: outC, h, w };
	}
}
