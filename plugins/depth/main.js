// Depth Anything V2 (ViT-S/14) as a plugin: an image in, a depth map out.
//
//   llm-runner depth model=depth_anything_v2_vits_fp32.safetensors input=photo.jpg output=depth.png
//   llm-runner depth ... height=true albedo=1 flatten=0.7     (a tiling height map)
//
// A DINOv2 encoder with four tapped layers and the DPT head, ported from
// `dependencies/depth_anything.c3l`. Settings are the C3 CLI's options.

import { llm, image, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit, copy } from '../lib/gpu.js';

const PATCH = 14;
const PRETRAIN_GRID = 37;
const TAPS = [2, 5, 8, 11];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

function convOf(m, prefix, { bias = true, transposed = false } = {}) {
	const shape = m.info(`${prefix}weight`).shape; // PyTorch order
	return {
		weight: m.upload(`${prefix}weight`, transposed ? 'f32' : 'conv'),
		bias: bias ? m.upload(`${prefix}bias`, 'f32') : null,
		inC: transposed ? shape[0] : shape[1],
		outC: transposed ? shape[1] : shape[0],
		k: shape[2],
	};
}

function disposeAll(v) {
	if (!v) return;
	if (typeof v.dispose === 'function') { v.dispose(); return; }
	if (typeof v === 'object') for (const x of Object.values(v)) disposeAll(x);
}

class DepthAnything {
	constructor(path) {
		const m = (this.model = llm.open(path));
		const t0 = llm.now();
		this.dim = m.info('pretrained.cls_token').shape.at(-1);
		this.heads = this.dim / 64;
		this.ffn = m.info('pretrained.blocks.0.mlp.fc1.weight').shape[0];
		let n = 0;
		while (m.has(`pretrained.blocks.${n}.norm1.weight`)) n++;
		const f = (name) => m.upload(name, 'f32');
		this.enc = {
			patch: convOf(m, 'pretrained.patch_embed.proj.'),
			cls: f('pretrained.cls_token'),
			normW: f('pretrained.norm.weight'),
			normB: f('pretrained.norm.bias'),
			pos: m.floats('pretrained.pos_embed'),
			blocks: [],
		};
		for (let l = 0; l < n; l++) {
			const p = (s) => `pretrained.blocks.${l}.${s}`;
			this.enc.blocks.push({
				n1w: f(p('norm1.weight')), n1b: f(p('norm1.bias')),
				qkvW: f(p('attn.qkv.weight')), qkvB: f(p('attn.qkv.bias')),
				projW: f(p('attn.proj.weight')), projB: f(p('attn.proj.bias')),
				ls1: f(p('ls1.gamma')),
				n2w: f(p('norm2.weight')), n2b: f(p('norm2.bias')),
				fc1W: f(p('mlp.fc1.weight')), fc1B: f(p('mlp.fc1.bias')),
				fc2W: f(p('mlp.fc2.weight')), fc2B: f(p('mlp.fc2.bias')),
				ls2: f(p('ls2.gamma')),
			});
		}
		const h = 'depth_head.';
		const refine = (i) => {
			const p = `${h}scratch.refinenet${i}.`;
			return {
				r1c1: convOf(m, `${p}resConfUnit1.conv1.`), r1c2: convOf(m, `${p}resConfUnit1.conv2.`),
				r2c1: convOf(m, `${p}resConfUnit2.conv1.`), r2c2: convOf(m, `${p}resConfUnit2.conv2.`),
				out: convOf(m, `${p}out_conv.`),
			};
		};
		this.head = {
			projects: [0, 1, 2, 3].map((i) => convOf(m, `${h}projects.${i}.`)),
			resize0: convOf(m, `${h}resize_layers.0.`, { transposed: true }),
			resize1: convOf(m, `${h}resize_layers.1.`, { transposed: true }),
			resize3: convOf(m, `${h}resize_layers.3.`),
			layerRn: [1, 2, 3, 4].map((i) => convOf(m, `${h}scratch.layer${i}_rn.`, { bias: false })),
			refine: [1, 2, 3, 4].map(refine),
			out1: convOf(m, `${h}scratch.output_conv1.`),
			out2a: convOf(m, `${h}scratch.output_conv2.0.`),
			out2b: convOf(m, `${h}scratch.output_conv2.2.`),
		};
		llm.print(`  Depth Anything: dim ${this.dim}, ${n} blocks, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll(this.enc.blocks);
		disposeAll([this.enc.patch, this.enc.cls, this.enc.normW, this.enc.normB, this.head]);
		this.model.close();
	}

	// Positional embeddings for a gh x gw grid: the pretrained 37 x 37 table,
	// bilinearly resampled (align_corners = False), the CLS row kept.
	positions(gh, gw) {
		const dim = this.dim, src = this.enc.pos, g = PRETRAIN_GRID;
		const out = new Float32Array((1 + gh * gw) * dim);
		out.set(src.subarray(0, dim));
		for (let oy = 0; oy < gh; oy++) {
			let sy = Math.fround(((oy + 0.5) * g) / gh - 0.5);
			if (sy < 0) sy = 0;
			const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, g - 1), dy = sy - y0;
			for (let ox = 0; ox < gw; ox++) {
				let sx = Math.fround(((ox + 0.5) * g) / gw - 0.5);
				if (sx < 0) sx = 0;
				const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, g - 1), dx = sx - x0;
				const t00 = (1 + y0 * g + x0) * dim, t01 = (1 + y0 * g + x1) * dim;
				const t10 = (1 + y1 * g + x0) * dim, t11 = (1 + y1 * g + x1) * dim;
				const d = (1 + oy * gw + ox) * dim;
				for (let c = 0; c < dim; c++) {
					out[d + c] = src[t00 + c] * (1 - dy) * (1 - dx) + src[t01 + c] * (1 - dy) * dx
						+ src[t10 + c] * dy * (1 - dx) + src[t11 + c] * dy * dx;
				}
			}
		}
		return out;
	}

	// image [3, gh*14, gw*14] normalized -> the four tapped, final-normed token
	// sets [1 + gh*gw, dim].
	encode(pixels, gh, gw) {
		const dim = this.dim, np = gh * gw, n = 1 + np;
		const H = gh * PATCH, W = gw * PATCH;
		const img = f32(3 * H * W);
		img.buffer.write(pixels);
		const a = {
			chw: f32(dim * np), hidden: f32(n * dim), norm: f32(n * dim), qkv: f32(n * 3 * dim),
			q: f32(n * dim), k: f32(n * dim), v: f32(n * dim), attn: f32(n * dim),
			branch: f32(n * dim), ffn: f32(n * this.ffn), pos: f32(n * dim),
		};
		a.pos.buffer.write(this.positions(gh, gw));
		const taps = TAPS.map(() => f32(n * dim));

		// Patch embedding: a stride-14 conv, then [dim, np] -> rows 1.. of the tokens.
		op.conv2d(this.enc.patch, img, a.chw, H, W, PATCH, 0);
		op.transposeChannelSpatial(a.chw, a.hidden.view(dim * 4, np * dim * 4), dim, np, 0);
		copy(this.enc.cls, a.hidden, dim * 4);
		op.add(a.hidden, a.pos, n * dim);

		let tap = 0;
		for (let l = 0; l < this.enc.blocks.length; l++) {
			const b = this.enc.blocks[l];
			op.layerNormAffine(a.hidden, b.n1w, b.n1b, a.norm, dim, n);
			op.linearBiasRows(b.qkvW, b.qkvB, a.norm, a.qkv, 3 * dim, dim, n);
			// [n, 3 * dim] -> per-head Q, K, V [heads, n, 64]: the fused output's thirds, transposed.
			for (const [i, t] of [a.q, a.k, a.v].entries()) {
				op.copyRows(a.qkv.view(i * dim * 4), t, n, dim, dim, 0, 3 * dim);
			}
			op.transposeHeads(a.q, a.branch, n, this.heads, 64, 0, true);
			op.transposeHeads(a.k, a.norm, n, this.heads, 64, 0, true);
			op.transposeHeads(a.v, a.attn, n, this.heads, 64, 0);
			op.flashAttention(a.branch, a.norm, a.attn, a.q, this.heads, n, 64);
			op.linearBiasRows(b.projW, b.projB, a.q, a.branch, dim, dim, n);
			op.gatedAdd(a.hidden, b.ls1, a.branch, n * dim, dim, 0);

			op.layerNormAffine(a.hidden, b.n2w, b.n2b, a.norm, dim, n);
			op.linearBiasRows(b.fc1W, b.fc1B, a.norm, a.ffn, this.ffn, dim, n);
			op.gelu(a.ffn, n * this.ffn);
			op.linearBiasRows(b.fc2W, b.fc2B, a.ffn, a.branch, dim, this.ffn, n);
			op.gatedAdd(a.hidden, b.ls2, a.branch, n * dim, dim, 0);
			if (TAPS.includes(l)) {
				op.layerNormAffine(a.hidden, this.enc.normW, this.enc.normB, taps[tap++], dim, n);
			}
			submit();
		}
		disposeAll(a);
		img.dispose();
		return taps;
	}

	// The DPT head: taps -> depth [1, gh*14, gw*14].
	decode(taps, gh, gw) {
		const hd = this.head, dim = this.dim, np = gh * gw;
		const g4 = Math.max(gh, gw) * 4, f8 = g4 * 2;
		const side = Math.max(gh, gw) * PATCH;
		const big = 64 * f8 * f8;
		const d = {
			chw: f32(dim * np), proj: f32(dim * np), resized: f32(48 * g4 * g4),
			rn: [f32(64 * g4 * g4), f32(64 * g4 * g4), f32(64 * np), f32(64 * np)],
			a: f32(big), b: f32(big), cc: f32(big), fused: f32(big), up: f32(big), P: f32(big), Q: f32(big),
			o1: f32(32 * f8 * f8), interp: f32(32 * side * side), convb: f32(32 * side * side),
		};
		const depth = f32(side * side);
		const size = [];
		for (let i = 0; i < 4; i++) {
			// Tokens (CLS skipped) -> [dim, gh, gw] -> project -> resize -> layerN_rn.
			op.transposeChannelSpatial(taps[i].view(dim * 4, np * dim * 4), d.chw, dim, np, 1);
			op.conv2d(hd.projects[i], d.chw, d.proj, gh, gw);
			let h = gh, w = gw;
			if (i === 0) { op.convTranspose(hd.resize0, d.proj, d.resized, gh, gw); h *= 4; w *= 4; }
			else if (i === 1) { op.convTranspose(hd.resize1, d.proj, d.resized, gh, gw); h *= 2; w *= 2; }
			else if (i === 2) { copy(d.proj, d.resized, 192 * np * 4); }
			else ({ h, w } = op.conv2d(hd.resize3, d.proj, d.resized, gh, gw, 2, 1));
			op.conv2d(hd.layerRn[i], d.resized, d.rn[i], h, w);
			size.push([h, w]);
		}
		submit();

		const rcu = (c1, c2, x, dst, h, w) => {
			const n = 64 * h * w;
			copy(x, d.a, n * 4);
			op.relu(d.a, n);
			op.conv2d(c1, d.a, d.b, h, w);
			op.relu(d.b, n);
			op.conv2d(c2, d.b, d.a, h, w);
			if (dst !== x) copy(x, dst, n * 4);
			op.add(dst, d.a, n);
		};
		const fuse = (rn, primary, feat, [h, w], [oh, ow], out) => {
			const n = 64 * h * w;
			copy(primary, d.fused, n * 4);
			if (feat) {
				rcu(rn.r1c1, rn.r1c2, feat, d.cc, h, w);
				op.add(d.fused, d.cc, n);
			}
			rcu(rn.r2c1, rn.r2c2, d.fused, d.fused, h, w);
			op.bilinear(d.fused, d.up, 64, h, w, oh, ow);
			op.conv2d(rn.out, d.up, out, oh, ow);
		};
		fuse(hd.refine[3], d.rn[3], null, size[3], size[2], d.P);
		fuse(hd.refine[2], d.P, d.rn[2], size[2], size[1], d.Q);
		fuse(hd.refine[1], d.Q, d.rn[1], size[1], size[0], d.P);
		fuse(hd.refine[0], d.P, d.rn[0], size[0], [size[0][0] * 2, size[0][1] * 2], d.Q);
		const fh = size[0][0] * 2, fw = size[0][1] * 2;
		const sh = gh * PATCH, sw = gw * PATCH;
		op.conv2d(hd.out1, d.Q, d.o1, fh, fw);
		op.bilinear(d.o1, d.interp, 32, fh, fw, sh, sw);
		op.conv2d(hd.out2a, d.interp, d.convb, sh, sw);
		op.relu(d.convb, 32 * sh * sw);
		op.conv2d(hd.out2b, d.convb, depth, sh, sw);
		op.relu(depth, sh * sw);
		submit();
		disposeAll(d);
		return depth;
	}
}

// A summed-area-table box blur with a clamped window, in doubles.
function boxBlur(src, w, h, r) {
	const sw = w + 1;
	const sat = new Float64Array(sw * (h + 1));
	for (let y = 0; y < h; y++) {
		let row = 0;
		for (let x = 0; x < w; x++) {
			row += src[y * w + x];
			sat[(y + 1) * sw + x + 1] = sat[y * sw + x + 1] + row;
		}
	}
	const out = new Float32Array(w * h);
	for (let y = 0; y < h; y++) {
		const ya = Math.max(y - r, 0), yb = Math.min(y + r, h - 1);
		for (let x = 0; x < w; x++) {
			const xa = Math.max(x - r, 0), xb = Math.min(x + r, w - 1);
			const s = sat[(yb + 1) * sw + xb + 1] - sat[ya * sw + xb + 1] - sat[(yb + 1) * sw + xa] + sat[ya * sw + xa];
			out[y * w + x] = s / ((yb - ya + 1) * (xb - xa + 1));
		}
	}
	return out;
}

function normalize(buf, lo, hi) {
	let mn = buf[0], mx = buf[0];
	for (const v of buf) { if (v < mn) mn = v; if (v > mx) mx = v; }
	let range = mx - mn;
	if (range < 1e-8) range = 1;
	for (let i = 0; i < buf.length; i++) buf[i] = lo + ((buf[i] - mn) / range) * (hi - lo);
	return { mn, mx };
}

function gray(values, width, height) {
	const pixels = new Uint8Array(values.length);
	for (let i = 0; i < values.length; i++) pixels[i] = Math.floor(values[i] * 255 + 0.5);
	return { width, height, channels: 1, pixels };
}

llm.plugin({
	name: 'depth',
	generate(config) {
		const start = llm.now();
		const heightMode = config.height === true || config.height === 'true';
		const res = config.res ?? (heightMode ? 1036 : 518);
		const grid = Math.max(1, Math.floor((res + PATCH / 2) / PATCH));
		const src = image.load(config.input);
		const ow = src.width, oh = src.height;
		const scale = (grid * PATCH) / Math.max(ow, oh);
		const gw = Math.min(grid, Math.max(1, Math.floor((ow * scale) / PATCH + 0.5)));
		const gh = Math.min(grid, Math.max(1, Math.floor((oh * scale) / PATCH + 0.5)));
		llm.print(`=== Depth: ${config.input} (${ow}x${oh}) -> grid ${gw}x${gh}, model input ${gw * PATCH}x${gh * PATCH} ===`);

		const pixels = image.toTensor(image.resize(src, gw * PATCH, gh * PATCH), 'unit');
		const plane = gw * PATCH * gh * PATCH;
		for (let c = 0; c < 3; c++) {
			for (let i = 0; i < plane; i++) pixels[c * plane + i] = (pixels[c * plane + i] - MEAN[c]) / STD[c];
		}

		const model = new DepthAnything(config.model);
		const taps = model.encode(pixels, gh, gw);
		const depth = model.decode(taps, gh, gw);
		for (const t of taps) t.dispose();

		// Back to the source size, then to 8 bits.
		const full = f32(ow * oh);
		op.bilinear(depth, full, 1, gh * PATCH, gw * PATCH, oh, ow);
		submit();
		const values = new Float32Array(full.buffer.readBytes().buffer);
		depth.dispose();
		full.dispose();
		model.dispose();

		const out = config.output ?? 'depth.png';
		if (heightMode) {
			image.savePng(out, gray(heightMap(values, src, ow, oh, config), ow, oh));
		} else {
			const { mn, mx } = normalize(values, 0, 1);
			image.savePng(out, gray(values, ow, oh));
			llm.print(`  depth range ${mn.toFixed(3)}..${mx.toFixed(3)}`);
		}
		llm.print(`=== done in ${llm.since(start)}, saved ${out} ===`);
		return { output: out, width: ow, height: oh };
	},
});

// A tiling height/relief map: the model's depth high-passed to strip the scene
// gradient, blended with the albedo's luminance, plus a fine luminance detail
// pass. `save_height_map` in the C3 pipeline, on the host.
function heightMap(depth, src, w, h, config) {
	const detrend = config.detrend ?? 0.1, detail = config.detail ?? 0.3;
	const albedo = config.albedo ?? 0, flatten = config.flatten ?? 0.7;
	const n = w * h;
	const r = Math.max(1, Math.floor(detrend * Math.max(w, h) + 0.5));
	let base = new Float32Array(n);
	if (albedo < 1) {
		const mean = boxBlur(depth, w, h, r);
		for (let i = 0; i < n; i++) base[i] = depth[i] - mean[i];
		normalize(base, 0, 1);
	}
	let lum = null;
	if (albedo > 0 || detail > 0) {
		const rgb = image.toTensor(image.resize(src, w, h), 'unit');
		lum = new Float32Array(n);
		for (let i = 0; i < n; i++) lum[i] = 0.299 * rgb[i] + 0.587 * rgb[n + i] + 0.114 * rgb[2 * n + i];
	}
	if (albedo > 0) {
		const amean = boxBlur(lum, w, h, r);
		const alb = new Float32Array(n);
		for (let i = 0; i < n; i++) alb[i] = lum[i] - flatten * amean[i];
		normalize(alb, 0, 1);
		for (let i = 0; i < n; i++) base[i] = base[i] * (1 - albedo) + alb[i] * albedo;
	}
	if (detail > 0) {
		const lmean = boxBlur(lum, w, h, Math.max(1, Math.floor(r / 8)));
		const hp = new Float32Array(n);
		for (let i = 0; i < n; i++) hp[i] = lum[i] - lmean[i];
		normalize(hp, -0.5, 0.5);
		for (let i = 0; i < n; i++) base[i] = Math.min(1, Math.max(0, base[i] + detail * hp[i]));
	}
	return base;
}
