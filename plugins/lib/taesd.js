// TAESD for 16-channel latents (madebyollin/taef1): a fast, rough stand-in for
// the Flux VAE decoder. Latent [16, h, w] to image [3, 8h, 8w] in [0, 1].
//
//   tanh(z / 3) * 3 -> conv(16 -> 64) -> relu ->
//   3 x [3 blocks -> upsample 2x -> conv] -> block -> conv(64 -> 3) -> clamp
//
// A block is conv, relu, conv, relu, conv, plus its input, then relu. The
// layers are `decoder.layers.N`, numbered as in the PyTorch Sequential, so the
// upsamples and ReLUs take indices of their own and have no weights.
//
// Ported from the C3 Z-Image pipeline (`taesd.c3`, since deleted).

import { f32 } from './llm.js';
import * as op from './ops.js';
import { submit, copy } from './gpu.js';

const CH = 64;

function conv(model, prefix) {
	const shape = model.shape(`${prefix}weight`); // [kw, kh, in, out]
	return {
		weight: model.upload(`${prefix}weight`, 'conv'),
		bias: model.has(`${prefix}bias`) ? model.upload(`${prefix}bias`, 'f32') : null,
		inC: shape[2],
		outC: shape[3],
		k: shape[0],
	};
}

function block(model, index) {
	const p = `decoder.layers.${index}.conv.`;
	return [conv(model, `${p}0.`), conv(model, `${p}2.`), conv(model, `${p}4.`)];
}

export class TAESDDecoder {
	constructor(model) {
		this.convIn = conv(model, 'decoder.layers.0.');
		// Stages start at 2, 7 and 12: three blocks, the upsample, its conv.
		this.stages = [2, 7, 12].map((i) => ({
			blocks: [block(model, i), block(model, i + 1), block(model, i + 2)],
			conv: conv(model, `decoder.layers.${i + 4}.`),
		}));
		this.final = block(model, 17);
		this.convOut = conv(model, 'decoder.layers.18.');
	}

	// x -> y, with s as scratch; x is left as it was.
	block(convs, x, y, s, h, w) {
		const n = CH * h * w;
		op.conv2d(convs[0], x, y, h, w);
		op.relu(y, n);
		op.conv2d(convs[1], y, s, h, w);
		op.relu(s, n);
		op.conv2d(convs[2], s, y, h, w);
		op.add(y, x, n);
		op.relu(y, n);
	}

	// Float32Array [16, h, w] -> Float32Array [3, 8h, 8w].
	decode(latent, h, w) {
		const size = CH * h * 8 * w * 8;
		let a = f32(size), b = f32(size);
		const s = f32(size);
		a.buffer.write(latent);
		op.tanhClamp(a, latent.length);
		op.conv2d(this.convIn, a, b, h, w);
		op.relu(b, CH * h * w);
		for (const stage of this.stages) {
			for (const convs of stage.blocks) {
				this.block(convs, b, a, s, h, w);
				[a, b] = [b, a];
			}
			op.upsample2x(b, a, CH, h, w);
			h *= 2; w *= 2;
			op.conv2d(stage.conv, a, b, h, w);
			submit();
		}
		this.block(this.final, b, a, s, h, w);
		op.conv2d(this.convOut, a, b, h, w);
		op.scaleShiftClamp(b, a, 3 * h * w, 1, 0);
		submit();
		const out = new Float32Array(a.buffer.readBytes(0, 3 * h * w * 4).buffer);
		for (const t of [a, b, s]) t.dispose();
		return out;
	}

	dispose() {
		const all = [this.convIn, this.convOut, ...this.final];
		for (const st of this.stages) all.push(st.conv, ...st.blocks.flat());
		for (const c of all) { c.weight.dispose(); if (c.bias) c.bias.dispose(); }
	}
}
