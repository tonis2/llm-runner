// The operations a model is written in, over Tensors and views of them.
//
// Each is one kernel dispatch (or two), recorded and not waited for: a model's
// forward pass is a long list of these and one `submit()` at the end. Shapes are
// passed rather than read off the tensors, because the same buffer is reused at
// many shapes - that is what an activation buffer is.
//
// `independent` on the ops that take it leaves out the barrier after the
// dispatch: the next op does not read what this one wrote (Q, K and V are three
// matmuls over one input).

import { GGML } from './llm.js';
import { dispatch, pc, groups, flashDefines } from './gpu.js';

// Whether Q8_0 matmuls may run on the matrix cores, in float16. They do when
// the device has cooperative matrices and 64-wide subgroups (the kernel's
// layout), unless a plugin turns it off with `useMatrixCores(false)`.
const L = three.compute.limits;
let matrixCores = L.cooperativeMatrix && L.subgroupSize === 64;
export function useMatrixCores(on) { matrixCores = on && L.cooperativeMatrix && L.subgroupSize === 64; }
export function matrixCoresOn() { return matrixCores; }

// y[seq, out] = x[seq, in] @ W[rowOffset .. rowOffset + out, :]^T
//
// `exact` keeps a Q8_0 matmul in float32 when the matrix cores are on: their
// operands are float16, so an x that can pass 65504 (the input of a SwiGLU
// down-projection in a model with large activations) has to take the slow path.
export function matmul(w, x, y, out, inDim, seq, rowOffset = 0, independent = false, exact = false) {
	const push = pc('uuuu', out, inDim, seq, rowOffset);
	switch (w.type) {
		case GGML.Q8_0:
			if (matrixCores && !exact && seq >= 16) {
				dispatch('matmul_q8_coop', [w, x, y], [groups(seq, 128) * groups(out, 128)], push, independent);
			} else {
				dispatch('matmul_q8', [w, x, y], [groups(seq, 64) * groups(out, 64)], push, independent);
			}
			return;
		case GGML.F32:
			if (seq <= 4) dispatch('matmul_f32_rows', [w, x, y], [seq, out], push, independent);
			else dispatch('matmul_f32', [w, x, y], [groups(seq, 64) * groups(out, 64)], push, independent);
			return;
		default:
			throw new Error(`matmul: no kernel for weight type ${w.typeName ?? w.type} (${w.name})`);
	}
}

// y = x @ W^T + b, W f32 [out, in].
export function linearBias(w, b, x, y, out, inDim, seq) {
	dispatch('linear_bias', [w, b, x, y], [seq * out], pc('uuu', out, inDim, seq));
}

export function silu(x, n) { dispatch('silu', [x], [groups(n, 256)], pc('u', n)); }
export function mul(a, b, n) { dispatch('elemwise_mul', [a, b], [groups(n, 256)], pc('u', n)); }
export function siluMul(gate, up, n) { dispatch('silu_mul', [gate, up], [groups(n, 256)], pc('u', n)); }
export function add(acc, src, n) { dispatch('residual_add', [acc, src], [groups(n, 256)], pc('u', n)); }

export function rmsNorm(x, weight, y, dim, rows, eps) {
	dispatch('rmsnorm_batch', [x, weight, y], [rows], pc('ufu', dim, eps, rows));
}
export function layerNorm(x, y, dim, rows, eps = 1e-6) {
	dispatch('batch_layernorm', [x, y], [rows], pc('ufuu', dim, eps, rows, 0));
}
// Per-head RMSNorm in place on [rows, heads * headDim]. `headNorm` is a
// workgroup per (row, head); `headNormSmall` a thread per (row, head), for
// small heads where a workgroup is mostly idle.
export function headNorm(x, weight, heads, headDim, rows, eps = 1e-6) {
	dispatch('batch_head_norm', [x, weight], [rows * heads], pc('uuuf', heads, headDim, rows, eps));
}
export function headNormSmall(x, weight, heads, headDim, rows, eps, independent = false) {
	dispatch('head_rmsnorm_batch', [x, weight], [groups(rows * heads, 256)], pc('uuuf', headDim, heads, rows, eps), independent);
}
// The Wan-style VAE norm on [C, spatial]: per pixel, x / |x| * sqrt(C) * gamma,
// then SiLU when `silu`.
export function channelRmsNorm(x, gamma, y, channels, spatial, silu = false) {
	dispatch('channel_rmsnorm', [x, gamma, y], [groups(spatial, 256)], pc('uuu', channels, spatial, silu ? 1 : 0));
}
// The residual VAE's parameter-free shortcuts, added into `y`: AvgDown3D from
// [inC, h*fs, w*fs] to [outC, h, w], and DupUp3D from [inC, h, w] to
// [outC, 2h, 2w]. `ftN` is the temporal factor (1 or 2) of one frame.
export function avgDownAdd(x, y, inC, outC, h, w, ftN, fs) {
	dispatch('avg_down', [x, y], [groups(outC * h * w, 256)], pc('uuuuuu', inC, outC, h, w, ftN, fs));
}
export function dupUpAdd(x, y, inC, outC, h, w, ftN) {
	dispatch('dup_up', [x, y], [groups(outC * h * w * 4, 256)], pc('uuuuu', inC, outC, h, w, ftN));
}
export function groupNorm(x, weight, bias, y, channels, spatial, numGroups = 32, eps = 1e-6) {
	dispatch('group_norm', [x, weight, bias, y], [numGroups], pc('uuuf', channels, spatial, numGroups, eps));
}

// out = x * (1 + mod[scaleAt + d]) + mod[shiftAt + d]
export function modulate(x, mod, y, n, dim, scaleAt, shiftAt) {
	dispatch('adaln_modulate', [x, mod, mod, y], [groups(n, 256)], pc('uuuu', n, dim, scaleAt, shiftAt));
}
// residual += mod[gateAt + d] * x
export function gatedAdd(residual, mod, x, n, dim, gateAt) {
	dispatch('gated_residual_linear', [residual, mod, x], [groups(n, 256)], pc('uuuu', n, dim, gateAt, 0));
}

// Rotate-half RoPE from cos/sin tables [tokens, headDim / 2] on [tokens, heads * headDim].
export function ropeNeox(x, cos, sin, headDim, heads, tokens, independent = false) {
	dispatch('rope_neox', [x, cos, sin], [groups(tokens * heads * (headDim / 2), 128)], pc('uuuu', headDim, heads, tokens, 0), independent);
}
// Rotary from tables on [heads, seq, 128].
export function mrope(x, cos, sin, seq, heads, independent = false) {
	dispatch('mrope', [x, cos, sin], [1, seq, heads], pc('uuuu', seq, heads, 0, 0), independent);
}
export function transposeHeads(x, y, seq, heads, headDim, direction = 0, independent = false) {
	dispatch('transpose_heads', [x, y], [groups(seq * heads * headDim, 256)], pc('uuuu', seq, heads, headDim, direction), independent);
}
export function transposeChannelSpatial(x, y, channels, spatial, direction) {
	dispatch('transpose_channel_spatial', [x, y], [groups(channels * spatial, 256)], pc('uuu', channels, spatial, direction));
}

// Q, K, V [heads, seq, hd] -> out [seq, heads * hd], non-causal.
export function flashAttention(q, k, v, out, heads, seq, hd = 128) {
	if (matrixCores && hd === 128) {
		dispatch('flash_attention_coop', [q, k, v, out], [groups(seq, 64), heads], pc('uuuf', hd, heads, seq, 1 / Math.sqrt(hd)));
		return;
	}
	dispatch('flash_attention', [q, k, v, out], [groups(seq, 16), heads], pc('uuuf', hd, heads, seq, 1 / Math.sqrt(hd)), false, flashDefines(hd));
}
// Q [heads, qLen, 128] over the keys of a cached prefix (k1, v1: [heads, l1,
// 128]) followed by the current ones (k2, v2: [heads, l2, 128]); out [qLen,
// heads * 128]. `limits` (a buffer of uints, one a query row, not falling)
// caps the keys each row sees - a block-causal mask.
export function flashAttentionSplit(q, k1, v1, k2, v2, out, heads, qLen, l1, l2, limits = null) {
	const bindings = [q, l1 > 0 ? k1 : k2, l1 > 0 ? v1 : v2, k2, v2, limits ?? q, out];
	dispatch('flash_attention_split', bindings, [groups(qLen, 64), heads], pc('uuuufu', heads, qLen, l1, l2, 1 / Math.sqrt(128), limits ? 1 : 0));
}
// Causal GQA, Q [n, qHeads * hd], K/V [n, kvHeads * hd].
export function attentionCausal(q, k, v, out, headDim, kvHeads, qHeads, tokens) {
	dispatch('attention_causal', [q, k, v, out], [qHeads, tokens], pc('uuuuf', headDim, kvHeads, qHeads, tokens, 1 / Math.sqrt(headDim)));
}
// The queries `start` .. `start + count` only, so a caller can cut a large one
// up (the Flux VAE's mid block at 1024 is seconds of work).
export function vaeAttention(q, k, v, out, channels, spatial, start = 0, count = spatial) {
	dispatch('vae_attention', [q, k, v, out], [count], pc('uuufuuu', channels, 1, spatial, 1 / Math.sqrt(channels), channels, start, count));
}

export function timestepEmbed(out, dim, timestep) {
	dispatch('timestep_embed', [out], [groups(dim, 256)], pc('uf', dim, timestep));
}
export function patchify(latent, patches, channels, height, width, patchSize = 1) {
	dispatch('patchify', [latent, patches], [groups(channels * height * width, 256)], pc('uuuu', channels, height, width, patchSize));
}
export function unpatchify(patches, latent, channels, height, width, patchSize = 1) {
	dispatch('unpatchify', [patches, latent], [groups(channels * height * width, 256)], pc('uuuu', channels, height, width, patchSize));
}
export function eulerStep(x, v, n, dt) {
	dispatch('flow_euler_step', [x, v], [groups(n, 256)], pc('uf', n, dt));
}
export function concatRows(a, b, out, rows, aCols, bCols) {
	dispatch('concat_rows', [a, b, out], [groups(rows * (aCols + bCols), 256)], pc('uuu', rows, aCols, bCols));
}
export function copyRows(src, dst, rows, cols, dstStride, dstCol, srcStride = cols) {
	dispatch('copy_rows', [src, dst], [groups(rows * cols, 256)], pc('uuuuu', rows, cols, dstStride, dstCol, srcStride));
}
export function upsample2x(x, y, channels, h, w) {
	dispatch('upsample_nearest', [x, y], [groups(channels * h * w * 4, 256)], pc('uuu', channels, h, w));
}
export function scaleShiftClamp(x, y, n, scale, shift) {
	dispatch('scale_shift_clamp', [x, y], [groups(n, 256)], pc('uff', n, scale, shift));
}

// 2D convolution. `conv` is { weight, bias, inC, outC, k }; weights [k, k, in, out].
// A conv with no bias has `bias` null.
// `out` overrides the output size: a stride-2 downsample with diffusers'
// pad-bottom-right is pad 0 and exactly half the input, the missing row and
// column read as zeros.
export function conv2d(conv, x, y, h, w, stride = 1, pad = null, out = null) {
	const k = conv.k;
	const p = pad ?? (k === 3 ? 1 : 0);
	const oh = out ? out.h : Math.floor((h + 2 * p - k) / stride) + 1;
	const ow = out ? out.w : Math.floor((w + 2 * p - k) / stride) + 1;
	const bias = conv.bias ?? conv.weight;
	const push = pc('uuuuuuuuuuuu', conv.inC, conv.outC, h, w, k, k, stride, p, 1, oh, ow, conv.bias ? 1 : 0);
	// A narrow input that is not a multiple of 32 (144) runs a half-empty last chunk.
	const coopIn = conv.inC % 32 === 0 || (conv.inC % 16 === 0 && conv.inC >= 128);
	if (stride === 1 && 2 * p === k - 1 && matrixCores && coopIn && conv.outC % 4 === 0) {
		// A size-keeping conv (3x3 or 1x1) as an implicit GEMM on the matrix cores.
		dispatch('conv2d_coop', [conv.weight, bias, x, y], [groups(oh * ow, 128) * groups(conv.outC, 128)], push);
	} else if (k === 3 && stride === 1 && p === 1) {
		dispatch('conv2d_3x3', [conv.weight, bias, x, y], [groups(ow, 16), groups(oh, 16), groups(conv.outC, 4)], push);
	} else {
		dispatch('conv2d', [conv.weight, bias, x, y], [groups(oh * ow, 256), conv.outC], push);
	}
	return { h: oh, w: ow };
}

// out = x * (1 + mod[scaleAt + d])
export function scaleModulate(x, mod, y, n, dim, scaleAt) {
	dispatch('scale_modulate', [x, mod, y], [groups(n, 256)], pc('uuuu', n, dim, scaleAt, 0));
}
// residual += tanh(mod[gateAt + d]) * x
export function gatedAddTanh(residual, mod, x, n, dim, gateAt) {
	dispatch('gated_residual_tanh', [residual, mod, x], [groups(n, 256)], pc('uuuu', n, dim, gateAt, 0));
}
export function biasAdd(x, bias, n, dim) { dispatch('bias_add', [x, bias], [groups(n, 256)], pc('uu', n, dim)); }
export function fillRows(row, dst, first, rows, dim) {
	if (rows > 0) dispatch('fill_rows', [row, dst], [groups(rows * dim, 256)], pc('uuu', first, rows, dim));
}
export function scale(x, n, s) { dispatch('scale', [x], [groups(n, 256)], pc('uf', n, s)); }

export function layerNormAffine(x, weight, bias, y, dim, rows, eps = 1e-6) {
	dispatch('layernorm_affine', [x, weight, bias, y], [rows], pc('ufuu', dim, eps, rows, 0));
}
export function geluErf(x, n) { dispatch('gelu_erf', [x], [groups(n, 256)], pc('u', n)); }
// [gh * gw, C] tokens to [(gh/2) * (gw/2), 4C], each 2x2 block's side by side.
export function merge2x2(x, y, gh, gw, channels) {
	dispatch('merge_2x2', [x, y], [groups(gh * gw * channels, 256)], pc('uuu', gh, gw, channels));
}
// [seq, heads * hd] to [heads, seq, hdp], zero-padded.
export function padHeads(x, y, seq, heads, hd, hdp) {
	dispatch('pad_heads', [x, y], [groups(heads * seq * hdp, 256)], pc('uuuu', seq, heads, hd, hdp));
}
// Non-causal attention of any head width up to 128 on [heads, seq, hd], the
// width a multiple of 16; `scale` for a head padded out from a narrower one.
export function flashAttentionScalar(q, k, v, out, heads, seq, hd, scale = 1 / Math.sqrt(hd)) {
	dispatch('flash_attention', [q, k, v, out], [groups(seq, 16), heads], pc('uuuf', hd, heads, seq, scale), false, flashDefines(hd));
}
export function gelu(x, n) { dispatch('gelu', [x], [groups(n, 256)], pc('u', n)); }
export function relu(x, n) { dispatch('relu', [x], [groups(n, 256)], pc('u', n)); }
// tanh(x / 3) * 3 in place.
export function tanhClamp(x, n) { dispatch('tanh_clamp', [x], [groups(n, 256)], pc('u', n)); }
// Bilinear resize with align_corners = True.
export function bilinear(x, y, channels, inH, inW, outH, outW) {
	dispatch('bilinear_resize', [x, y], [groups(outH * outW, 256), channels], pc('uuuuu', channels, inH, inW, outH, outW));
}
// Stride == kernel transposed conv; `conv` weight is PyTorch's [in, out, k, k].
export function convTranspose(conv, x, y, h, w) {
	const bias = conv.bias ?? conv.weight;
	dispatch('conv_transpose', [conv.weight, bias, x, y], [groups(h * conv.k * w * conv.k, 256), conv.outC],
		pc('uuuuuu', conv.inC, conv.outC, h, w, conv.k, conv.bias ? 1 : 0));
}

// y = x @ W^T + b over many rows: the matmul, then the bias.
export function linearBiasRows(w, b, x, y, out, inDim, seq) {
	matmul(w, x, y, out, inDim, seq);
	biasAdd(y, b, seq * out, out);
}
