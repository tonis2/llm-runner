// NAF (valeoai/NAF), the feature upsampler Pixal3D samples its fine
// conditioning from: an image encoder turns the picture into per-pixel
// queries, average-pooled into keys on the DINO patch grid, and every pixel's
// feature is its 9x9 neighbourhood cross-attention over the DINO patches.
//
// Only the encoder's maps are made here (queries [s * s, 256] and keys
// [g * g, 256]); `sampleNaf` computes the upsampled features at given points
// alone (naf_sample.shady), never the full s x s map of 1024 channels.
//
// Ported from NAF's `src/model/naf.py` (ImageEncoder, KeyEncoder, NAF),
// `src/layers/convolutions.py` (encoder, EncBlock: no residual, no shortcut)
// and `src/layers/attentions.py` (CrossAttention).

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit } from '/core/gpu.js';
import { openTorch } from '/core/weights/torch.js';
import { axialRopeTables } from '/core/models/dinov3.js';

const DIM = 256, HEADS = 4, HEAD = DIM / HEADS, GROUPS = 8, EPS = 1e-5;
const P = 'image_encoder.';

function conv(m, prefix) {
	const shape = m.shape(`${prefix}weight`); // [kw, kh, in, out]
	return { weight: m.upload(`${prefix}weight`, 'conv'), bias: m.upload(`${prefix}bias`, 'f32'), inC: shape[2], outC: shape[3], k: shape[0] };
}

function disposeAll(v) {
	if (!v) return;
	if (typeof v.dispose === 'function') { v.dispose(); return; }
	if (typeof v === 'object') for (const x of Object.values(v)) disposeAll(x);
}

export class Naf {
	constructor(path) {
		this.path = path;
	}

	load() {
		const m = openTorch(this.path);
		const branch = (name) => ({
			first: conv(m, `${P}${name}.0.`),
			blocks: [1, 2].map((i) => {
				const b = `${P}${name}.${i}.`;
				return {
					n1w: m.upload(`${b}norm1.weight`, 'f32'), n1b: m.upload(`${b}norm1.bias`, 'f32'),
					conv1: conv(m, `${b}conv1.`),
					n2w: m.upload(`${b}norm2.weight`, 'f32'), n2b: m.upload(`${b}norm2.bias`, 'f32'),
					conv2: conv(m, `${b}conv2.`),
				};
			}),
		});
		this.g = { encoder: branch('encoder'), sem: branch('sem_encoder') };
		this.periods = m.floats(`${P}rope.periods`);
		if (this.periods.length !== HEAD / 4) throw new Error(`NAF: ${this.periods.length} RoPE periods, expected ${HEAD / 4}`);
		m.close();
	}

	dispose() {
		disposeAll(this.g);
		this.g = null;
	}

	// One encoder branch over the picture [3, s, s] into `out` [128, s, s]:
	// its first conv, then two blocks of GroupNorm, SiLU, conv twice.
	branch(br, img, out, s, tmp) {
		const n = s * s, C = br.first.outC;
		op.conv2d(br.first, img, out, s, s, 1, null, null, true);
		for (const b of br.blocks) {
			op.groupNorm(out, b.n1w, b.n1b, tmp.a, C, n, GROUPS, EPS);
			op.silu(tmp.a, C * n);
			op.conv2d(b.conv1, tmp.a, tmp.b, s, s, 1, null, null, true);
			op.groupNorm(tmp.b, b.n2w, b.n2b, tmp.a, C, n, GROUPS, EPS);
			op.silu(tmp.a, C * n);
			op.conv2d(b.conv2, tmp.a, out, s, s, 1, null, null, true);
		}
	}

	// The picture `image` [3, size, size] in [0, 1] (not normalised; a Tensor)
	// to the attention's queries [s * s, 256] (RoPE applied) and keys [g * g,
	// 256] (the queries pooled to the DINO grid g). The encoder runs at the
	// picture's size; NAF's adaptive pool then takes its maps down to the
	// output size s (the identity when they are equal).
	encode(image, size, g, s = size) {
		if (size % s || s % g) throw new Error(`NAF: a ${size} picture to ${s} on a grid of ${g}`);
		const N = size * size, n = s * s, half = DIM / 2;
		const map = f32(DIM * N);
		const tmp = { a: f32(half * N), b: f32(half * N) };
		const q = f32(n * DIM), k = f32(g * g * DIM);
		const rope = axialRopeTables(this.periods, s, s);
		let full = null;
		try {
			this.branch(this.g.encoder, image, map.view(0, half * N * 4), size, tmp);
			this.branch(this.g.sem, image, map.view(half * N * 4, half * N * 4), size, tmp);
			submit();
			disposeAll(tmp);
			tmp.a = tmp.b = null;
			if (size === s) {
				op.transposeChannelSpatial(map, q, DIM, n, 0);
			} else {
				full = f32(N * DIM);
				op.transposeChannelSpatial(map, full, DIM, N, 0);
				op.avgPoolRows(full, q, DIM, size, size, size / s);
			}
			op.ropeNeox(q, rope.cos, rope.sin, HEAD, HEADS, n);
			op.avgPoolRows(q, k, DIM, s, s, s / g);
			submit();
		} finally {
			submit();
			disposeAll(tmp);
			disposeAll(rope);
			map.dispose();
			full?.dispose();
		}
		return { q, k, s, g };
	}
}

// The upsampled features at `count` points (float2 [count], in the s x s
// map's pixel units, clamped) into rows of `out` (columns col .. col +
// channels of a `stride`-wide row), from an encoding (Naf.encode; the weights
// are no longer needed). `values` is the DINO patches [g * g, channels] (a
// Tensor or a buffer view).
export function sampleNaf(enc, values, points, out, count, { channels = 1024, stride = channels, col = 0 } = {}) {
	op.nafSample(enc.q, enc.k, values, points, out, enc.s, enc.g, count, channels / HEADS, stride, col, 1 / Math.sqrt(HEAD));
}

export function disposeEncoding(enc) {
	enc?.q?.dispose();
	enc?.k?.dispose();
}
