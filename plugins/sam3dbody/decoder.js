// SAM 3D Body's promptable decoders on the GPU (`decoders/promptable_decoder.py`,
// `modules/transformer.py` TransformerDecoderLayer, `SAM3DBody.forward_decoder`
// and `forward_decoder_hand`, `keypoint_token_update_fn` and
// `keypoint3d_token_update_fn`, and their `_hand` twins).
//
// The body decoder and the hand decoder are one design with two sets of
// weights: the hand's are the body's names with `_hand` (`decoder_hand`,
// `ray_cond_emb_hand`, `init_pose_hand` ...) and its image PE is
// `hand_pe_layer`'s; the prompt encoder, `prompt_to_token` and the hand-box
// embedding are shared.
//
// Image tokens: the backbone's [1280, 32, 32] plus the no-mask embedding, the
// 99 ray channels appended, a 1x1 conv back to 1280 and a LayerNorm2d; their
// position embedding is the prompt encoder's random-Fourier grid.
//
// 144 + P tokens of 1024: the pose token (from the CLIFF condition and the
// initial pose and camera), the previous estimate (the initial one again
// without one), P prompt points (one dummy point with no prompt), two
// hand-box tokens, 70 2D keypoint tokens and 70 3D keypoint tokens. Six layers of: self-attention (the token PEs added to the queries
// and keys from the second layer on), cross-attention to the image tokens,
// GELU MLP; 8 heads of 64, LayerNorms at 1e-6, each PE LayerNormed per layer
// (LaPE). After every layer but the last, the pose token (after the final
// norm) goes to the heads (`onLayer`, on the host), and the 70 + 70 keypoint
// tokens take what came back: the 2D keypoints' position embeddings, the image
// features sampled under them, the 3D keypoints' position embeddings.

import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit, copy, breathe, uploadEach } from '../lib/gpu.js';
import { KEYPOINTS } from './heads.js';

const HEAD = 64;
const EPS = 1e-6;
const GRID = 32;

function upload(data, shape = [data.length]) {
	const t = f32(data.length, shape);
	t.buffer.write(data instanceof Float32Array ? data : Float32Array.from(data));
	return t;
}

function disposeAll(v) {
	if (!v) return;
	if (typeof v.dispose === 'function') { v.dispose(); return; }
	if (typeof v === 'object') for (const x of Object.values(v)) disposeAll(x);
}

// W [out, in] row-major with each row padded with zeros to 4 columns.
function padColumns(W, inDim) {
	const rows = W.length / inDim, out = new Float32Array(rows * 4);
	for (let r = 0; r < rows; r++) for (let i = 0; i < inDim; i++) out[r * 4 + i] = W[r * inDim + i];
	return out;
}

// x @ W^T + b on the host, W [out, in] row-major.
function hostLinear(W, b, x, out, inDim) {
	const y = new Float32Array(out);
	for (let o = 0; o < out; o++) {
		let s = b ? b[o] : 0;
		for (let i = 0; i < inDim; i++) s += W[o * inDim + i] * x[i];
		y[o] = s;
	}
	return y;
}

// The prompt encoder's PositionEmbeddingRandom over points in [0, 1]^2:
// sin and cos of 2 pi (2p - 1) @ G, G [2, half].
function randomFourier(G, half, x, y) {
	const out = new Float32Array(2 * half);
	const u = 2 * x - 1, v = 2 * y - 1;
	for (let j = 0; j < half; j++) {
		const a = 2 * Math.PI * (u * G[j] + v * G[half + j]);
		out[j] = Math.sin(a);
		out[half + j] = Math.cos(a);
	}
	return out;
}

// The weights the hand decoder has its own of, by their first name part.
const OWN = new Set(['decoder', 'ray_cond_emb', 'init_to_token_mhr', 'prev_to_token_mhr', 'keypoint_posemb_linear', 'keypoint3d_posemb_linear',
	'keypoint_feat_linear', 'keypoint_embedding', 'keypoint3d_embedding', 'init_pose', 'init_camera']);

// The token layout for `prompts` prompt points.
function layout(prompts) {
	const kp2d = 4 + prompts;
	return { tokens: kp2d + 2 * KEYPOINTS, hands: 2 + prompts, kp2d, kp3d: kp2d + KEYPOINTS };
}

export class BodyDecoder {
	// `linear` is how the layers' matmul weights go up: 'f32' (the default; the
	// decoder is small and feeds its own predictions back five times) or 'q8'.
	// `hand` loads the hand decoder.
	constructor(m, { linear = 'f32', hand = false } = {}) {
		this.m = m;
		this.linear = linear;
		this.hand = hand;
		const n = (name) => {
			const dot = name.indexOf('.');
			return hand && OWN.has(name.slice(0, dot)) ? `${name.slice(0, dot)}_hand${name.slice(dot)}` : name;
		};
		this.name = n;
		const shape = (name) => m.shape(n(name));
		this.dim = shape('decoder.norm_final.weight')[0];
		this.context = shape('ray_cond_emb.norm.weight')[0];
		this.inner = shape('decoder.layers.0.self_attn.q_proj.weight')[1];
		this.heads = this.inner / HEAD;
		this.mlp = shape('decoder.layers.0.ffn.layers.0.0.weight')[1];
		let depth = 0;
		while (m.has(n(`decoder.layers.${depth}.ln1.weight`))) depth++;
		this.depth = depth;
	}

	async load() {
		const m = this.m, D = this.dim, C = this.context, n = this.name;
		const t0 = llm.now();
		const f = (name) => m.upload(n(name), 'f32');
		const w = (name) => m.upload(n(name), this.linear);
		const floats = (name) => m.floats(n(name));
		const N = GRID * GRID;
		// The image tokens' PE: the dense random-Fourier grid, token-major.
		const G = m.floats(this.hand ? 'hand_pe_layer.positional_encoding_gaussian_matrix' : 'prompt_encoder.pe_layer.positional_encoding_gaussian_matrix');
		const pe = new Float32Array(N * C);
		for (let i = 0; i < GRID; i++) {
			for (let j = 0; j < GRID; j++) pe.set(randomFourier(G, C / 2, (j + 0.5) / GRID, (i + 0.5) / GRID), (i * GRID + j) * C);
		}
		// Host-side pieces of the token set-up and the keypoint updates.
		this.host = {
			init: { w: floats('init_to_token_mhr.weight'), b: floats('init_to_token_mhr.bias'), in: m.shape(n('init_to_token_mhr.weight'))[0] },
			prev: { w: floats('prev_to_token_mhr.weight'), b: floats('prev_to_token_mhr.bias'), in: m.shape(n('prev_to_token_mhr.weight'))[0] },
			promptW: m.floats('prompt_to_token.weight'), promptB: m.floats('prompt_to_token.bias'),
			promptG: m.floats('prompt_encoder.pe_layer.positional_encoding_gaussian_matrix'),
			invalidPoint: m.floats('prompt_encoder.invalid_point_embed.weight'),
			notAPoint: m.floats('prompt_encoder.not_a_point_embed.weight'),
			points: new Map(),
			initPose: floats('init_pose.weight'), initCamera: floats('init_camera.weight'),
			// The learned tokens: hand boxes and keypoints.
			handBox: m.floats('hand_box_embedding.weight'),
			keypoints: floats('keypoint_embedding.weight'), keypoints3d: floats('keypoint3d_embedding.weight'),
		};
		this.g = {
			noMask: f('prompt_encoder.no_mask_embed.weight'),
			rayConv: f('ray_cond_emb.conv.weight'),
			rayNormW: f('ray_cond_emb.norm.weight'), rayNormB: f('ray_cond_emb.norm.bias'),
			pe: upload(pe),
			normW: f('decoder.norm_final.weight'), normB: f('decoder.norm_final.bias'),
			// The posemb FFNs' first layers take 2 or 3 inputs: padded to 4.
			pos2d0W: upload(padColumns(floats('keypoint_posemb_linear.layers.0.0.weight'), 2)), pos2d0B: f('keypoint_posemb_linear.layers.0.0.bias'),
			pos3d0W: upload(padColumns(floats('keypoint3d_posemb_linear.layers.0.0.weight'), 3)), pos3d0B: f('keypoint3d_posemb_linear.layers.0.0.bias'),
			pos2dW: w('keypoint_posemb_linear.layers.1.weight'), pos2dB: f('keypoint_posemb_linear.layers.1.bias'),
			pos3dW: w('keypoint3d_posemb_linear.layers.1.weight'), pos3dB: f('keypoint3d_posemb_linear.layers.1.bias'),
			featW: w('keypoint_feat_linear.weight'), featB: f('keypoint_feat_linear.bias'),
		};
		this.layers = [];
		for (let l = 0; l < this.depth; l++) {
			const p = `decoder.layers.${l}.`;
			const layer = await uploadEach({
				pe1W: () => f(`${p}ln_pe_1.weight`), pe1B: () => f(`${p}ln_pe_1.bias`),
				ln1W: () => f(`${p}ln1.weight`), ln1B: () => f(`${p}ln1.bias`),
				sq: () => w(`${p}self_attn.q_proj.weight`), sqB: () => f(`${p}self_attn.q_proj.bias`),
				sk: () => w(`${p}self_attn.k_proj.weight`), skB: () => f(`${p}self_attn.k_proj.bias`),
				sv: () => w(`${p}self_attn.v_proj.weight`), svB: () => f(`${p}self_attn.v_proj.bias`),
				so: () => w(`${p}self_attn.proj.weight`), soB: () => f(`${p}self_attn.proj.bias`),
				ln21W: () => f(`${p}ln2_1.weight`), ln21B: () => f(`${p}ln2_1.bias`),
				ln22W: () => f(`${p}ln2_2.weight`), ln22B: () => f(`${p}ln2_2.bias`),
				cq: () => w(`${p}cross_attn.q_proj.weight`), cqB: () => f(`${p}cross_attn.q_proj.bias`),
				ck: () => w(`${p}cross_attn.k_proj.weight`), ckB: () => f(`${p}cross_attn.k_proj.bias`),
				cv: () => w(`${p}cross_attn.v_proj.weight`), cvB: () => f(`${p}cross_attn.v_proj.bias`),
				co: () => w(`${p}cross_attn.proj.weight`), coB: () => f(`${p}cross_attn.proj.bias`),
				ln3W: () => f(`${p}ln3.weight`), ln3B: () => f(`${p}ln3.bias`),
				fc1: () => w(`${p}ffn.layers.0.0.weight`), fc1B: () => f(`${p}ffn.layers.0.0.bias`),
				fc2: () => w(`${p}ffn.layers.1.weight`), fc2B: () => f(`${p}ffn.layers.1.bias`),
			});
			// The image PE after this layer's ln_pe_2: the same for every image.
			const pe2W = f(`${p}ln_pe_2.weight`), pe2B = f(`${p}ln_pe_2.bias`);
			layer.contextPe = f32(N * C);
			op.layerNormAffine(this.g.pe, pe2W, pe2B, layer.contextPe, C, N, EPS);
			submit();
			pe2W.dispose();
			pe2B.dispose();
			this.layers.push(layer);
		}
		llm.print(`  ${this.hand ? 'hand' : 'body'} decoder: ${this.depth} layers of ${D}, ${this.heads} heads, ${this.linear} weights, loaded in ${llm.since(t0)}`);
	}

	dispose() {
		disposeAll(this.g);
		for (const l of this.layers ?? []) disposeAll(l);
		this.g = null;
		this.layers = null;
	}

	// A prompt point [x, y, label] (x, y in the crop, 0 .. 1) as the prompt
	// encoder embeds it: label -2 is no point (the invalid-point embedding),
	// -1 a wrong one, 0 .. 69 a keypoint (its position's random-Fourier PE
	// plus that keypoint's embedding).
	promptEmbedding([x, y, label]) {
		const h = this.host;
		if (label === -2) return h.invalidPoint;
		if (label === -1) return h.notAPoint;
		if (!h.points.has(label)) h.points.set(label, this.m.floats(`prompt_encoder.point_embeddings.${label}.weight`));
		const e = randomFourier(h.promptG, h.promptG.length / 2, x, y), p = h.points.get(label);
		for (let i = 0; i < e.length; i++) e[i] += p[i];
		return e;
	}

	// The tokens the host makes and their PEs, [2 + P, dim] each: the pose
	// token (no PE), the previous estimate, the prompt points. `condition` is
	// the CLIFF [3], `prev` the previous estimate [522] or null, `prompts` the
	// points.
	firstTokens(condition, prev, prompts) {
		const h = this.host, D = this.dim;
		const init = [...condition, ...h.initPose, ...h.initCamera];
		const tokens = new Float32Array((2 + prompts.length) * D), pe = new Float32Array(tokens.length);
		tokens.set(hostLinear(h.init.w, h.init.b, init, D, h.init.in));
		const p = hostLinear(h.prev.w, h.prev.b, prev ?? init.slice(3), D, h.prev.in);
		tokens.set(p, D);
		pe.set(p, D);
		prompts.forEach((point, i) => {
			const e = this.promptEmbedding(point);
			const t = hostLinear(h.promptW, h.promptB, e, D, e.length);
			tokens.set(t, (2 + i) * D);
			pe.set(t, (2 + i) * D);
		});
		return { tokens, pe };
	}

	// `features` is the backbone's Tensor [1280, 32 * 32] (channel-first),
	// `rays` the ray features [32 * 32, 99], `condition` the CLIFF [3].
	// `onLayer(token, layer)` gets each layer's pose token after the final norm
	// (a Float32Array [dim]) and returns the pose it decodes to with
	// `pred_keypoints_2d_cropped`, `pred_keypoints_2d_depth` and
	// `pred_keypoints_3d` (it may be async). `prompts` are keypoint prompts
	// [[x, y, label]] (see promptEmbedding; one dummy point without), `prev`
	// the previous estimate [522] they refine (pose raw, identity, scale,
	// hands, face, camera). Returns the last layer's pose, `tokens` its tokens
	// up to the hand-box ones after the final norm and `handTokens` those two
	// [2, dim]; `stopAfter` (a layer) returns the raw tokens [144 + P, dim]
	// after it instead.
	//
	// `image` is what prepareImage made of these features and rays, to use
	// again (a second pass over the same crop); without one run makes its own
	// and, with `keepImage`, returns it as `image` for the caller to dispose.
	async run(features, rays, condition, onLayer, { stopAfter = -1, contextOut = null, prompts = null, prev = null, image = null, keepImage = false } = {}) {
		const points = prompts?.length ? prompts : [[0, 0, -2]];
		const at = layout(points.length);
		const D = this.dim, C = this.context, A = this.inner, T = at.tokens, N = GRID * GRID, g = this.g, h = this.host;
		const own = !image;
		if (own) image = this.prepareImage(features, rays, { contextOut });
		const a = {
			at,
			x: f32(T * D), pe: f32(T * D), pen: f32(T * D), h: f32(T * D), qk: f32(T * D),
			q: f32(T * A), sk: f32(T * A), sv: f32(T * A), o: f32(T * A), y: f32(T * D),
			f: f32(T * this.mlp), out: f32(T * D),
			kp2: f32(KEYPOINTS * 4), kp3: f32(KEYPOINTS * 4), kp2H: f32(KEYPOINTS * D), kp3H: f32(KEYPOINTS * D), mask: f32(KEYPOINTS * D), sample: f32(KEYPOINTS * N), sampled: f32(KEYPOINTS * C),
		};
		a.image = image;
		let kept = false;
		try {
			const first = this.firstTokens(condition, prev, points);
			const x = new Float32Array(T * D), pe = new Float32Array(T * D);
			x.set(first.tokens);
			x.set(h.handBox, at.hands * D);
			x.set(h.keypoints, at.kp2d * D);
			x.set(h.keypoints3d, at.kp3d * D);
			pe.set(first.pe);
			a.x.buffer.write(x);
			a.pe.buffer.write(pe);

			let last = null;
			for (const [l, L] of this.layers.entries()) {
				this.layer(L, l, a, N);
				if (l === stopAfter) {
					const t = f32(T * D, [D, T]);
					copy(a.x, t, T * D * 4);
					submit();
					return { tokens: t };
				}
				op.layerNormAffine(a.x, g.normW, g.normB, a.out, D, at.kp2d, EPS);
				submit();
				const token = new Float32Array(a.out.buffer.readBytes(0, D * 4).buffer);
				if (l === this.depth - 1) {
					const tokens = new Float32Array(a.out.buffer.readBytes(0, at.kp2d * D * 4).buffer);
					last = { tokens, handTokens: tokens.subarray(at.hands * D, (at.hands + 2) * D), pose: await onLayer(token, l) };
					if (own && keepImage) { last.image = image; kept = true; }
					break;
				}
				const pose = await onLayer(token, l);
				this.updateKeypoints(pose, a, N);
				await breathe(true);
			}
			return last;
		} finally {
			submit();
			a.image = null;
			disposeAll(a);
			if (own && !kept) image.dispose();
		}
	}

	// The image side of the decoder, which no token changes: the image tokens
	// [N, C] (the backbone's features, the no-mask embedding, the ray features,
	// a 1x1 conv and a LayerNorm2d), channel-first too for sampling under the
	// keypoints, and each layer's cross-attention keys and values. Returns
	// { ctxT, k: [layer], v: [layer], dispose() }.
	prepareImage(features, rays, { contextOut = null } = {}) {
		const C = this.context, A = this.inner, N = GRID * GRID, g = this.g;
		const t = { tok: f32(N * C), cat: f32(N * (C + 99)), raw: f32(N * C), ctx: f32(N * C), ctxN: f32(N * C), ctxK: f32(N * C), rays: upload(rays) };
		const image = {
			ctxT: f32(C * N, [N, C]),
			k: this.layers.map(() => f32(N * A)),
			v: this.layers.map(() => f32(N * A)),
			dispose() { disposeAll(this.ctxT); this.k.forEach((x) => x.dispose()); this.v.forEach((x) => x.dispose()); },
		};
		try {
			// Image tokens, token-major [N, C].
			op.transposeChannelSpatial(features, t.tok, C, N, 0);
			op.biasAdd(t.tok, g.noMask, N * C, C);
			op.concatRows(t.tok, t.rays, t.cat, N, C, 99);
			op.matmul(g.rayConv, t.cat, t.raw, C, C + 99, N);
			op.layerNormAffine(t.raw, g.rayNormW, g.rayNormB, t.ctx, C, N, EPS);
			// Channel-first as well: a matmul weight for sampling under keypoints.
			op.transposeChannelSpatial(t.ctx, image.ctxT, C, N, 1);
			if (contextOut) copy(t.ctx, contextOut, N * C * 4);
			// Each layer's keys (its image PE added) and values.
			for (const [l, L] of this.layers.entries()) {
				op.layerNormAffine(t.ctx, L.ln22W, L.ln22B, t.ctxN, C, N, EPS);
				copy(t.ctxN, t.ctxK, N * C * 4);
				op.add(t.ctxK, L.contextPe, N * C);
				op.linearBiasRows(L.ck, L.ckB, t.ctxK, image.k[l], A, C, N);
				op.linearBiasRows(L.cv, L.cvB, t.ctxN, image.v[l], A, C, N);
			}
		} finally {
			submit();
			disposeAll(t);
		}
		return image;
	}

	// One TransformerDecoderLayer over a.x [144 + P, dim] in place.
	layer(L, l, a, N) {
		const D = this.dim, A = this.inner, nh = this.heads, T = a.at.tokens;
		op.layerNormAffine(a.pe, L.pe1W, L.pe1B, a.pen, D, T, EPS);
		// Self-attention; the first layer leaves the token PEs out of it.
		op.layerNormAffine(a.x, L.ln1W, L.ln1B, a.h, D, T, EPS);
		let qk = a.h;
		if (l > 0) {
			copy(a.h, a.qk, T * D * 4);
			op.add(a.qk, a.pen, T * D);
			qk = a.qk;
		}
		op.linearBiasRows(L.sq, L.sqB, qk, a.q, A, D, T);
		op.linearBiasRows(L.sk, L.skB, qk, a.sk, A, D, T);
		op.linearBiasRows(L.sv, L.svB, a.h, a.sv, A, D, T);
		op.attentionRows(a.q, a.sk, a.sv, a.o, nh, HEAD, { lq: T, lk: T });
		op.linearBiasRows(L.so, L.soB, a.o, a.y, D, A, T);
		op.add(a.x, a.y, T * D);
		// Cross-attention to the image tokens, their PE on the keys.
		op.layerNormAffine(a.x, L.ln21W, L.ln21B, a.qk, D, T, EPS);
		op.add(a.qk, a.pen, T * D);
		op.linearBiasRows(L.cq, L.cqB, a.qk, a.q, A, D, T);
		op.attentionRows(a.q, a.image.k[l], a.image.v[l], a.o, nh, HEAD, { lq: T, lk: N });
		op.linearBiasRows(L.co, L.coB, a.o, a.y, D, A, T);
		op.add(a.x, a.y, T * D);
		// MLP.
		op.layerNormAffine(a.x, L.ln3W, L.ln3B, a.h, D, T, EPS);
		op.linearBiasRows(L.fc1, L.fc1B, a.h, a.f, this.mlp, D, T);
		op.geluErf(a.f, T * this.mlp);
		op.linearBiasRows(L.fc2, L.fc2B, a.f, a.y, D, this.mlp, T);
		op.add(a.x, a.y, T * D);
	}

	// keypoint_token_update_fn (v2) and keypoint3d_token_update_fn, between
	// layers. A 2D keypoint outside the crop or behind the camera gets no PE
	// and no image feature (its token still gets the feature linear's bias).
	updateKeypoints(pose, a, N) {
		const D = this.dim, C = this.context, K = KEYPOINTS, g = this.g;
		const rows = (t, first, count) => t.view(first * D * 4, count * D * 4);
		const kp = pose.pred_keypoints_2d_cropped, depth = pose.pred_keypoints_2d_depth;
		const valid = new Array(K);
		const mask = new Float32Array(K * D);
		// grid_sample(bilinear, zeros, align_corners=False) of the 32 x 32
		// tokens at 2 * kp, as one sparse row of weights per keypoint.
		const sample = new Float32Array(K * N);
		for (let k = 0; k < K; k++) {
			const x = kp[k * 2], y = kp[k * 2 + 1];
			valid[k] = !(x + 0.5 < 0 || x + 0.5 > 1 || y + 0.5 < 0 || y + 0.5 > 1 || depth[k] < 1e-5);
			if (!valid[k]) continue;
			mask.fill(1, k * D, (k + 1) * D);
			const ix = Math.fround(Math.fround(Math.fround(2 * x) + 1) * GRID - 1) / 2, iy = Math.fround(Math.fround(Math.fround(2 * y) + 1) * GRID - 1) / 2;
			const x0 = Math.floor(ix), y0 = Math.floor(iy), fx = ix - x0, fy = iy - y0;
			for (const [dx, dy, wt] of [[0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)], [0, 1, (1 - fx) * fy], [1, 1, fx * fy]]) {
				const cx = x0 + dx, cy = y0 + dy;
				if (cx >= 0 && cy >= 0 && cx < GRID && cy < GRID) sample[k * N + cy * GRID + cx] += wt;
			}
		}
		const kp3 = pose.pred_keypoints_3d;
		const pelvis = [0, 1, 2].map((c) => (kp3[9 * 3 + c] + kp3[10 * 3 + c]) / 2);
		// The posemb FFNs' inputs, 4 wide: the 2D keypoints in the crop, the 3D
		// ones about the pelvis.
		const in2 = new Float32Array(K * 4), in3 = new Float32Array(K * 4);
		for (let k = 0; k < K; k++) {
			in2[k * 4] = kp[k * 2];
			in2[k * 4 + 1] = kp[k * 2 + 1];
			for (let c = 0; c < 3; c++) in3[k * 4 + c] = kp3[k * 3 + c] - pelvis[c];
		}

		// The layer before has been submitted and waited for (run reads its
		// pose token): these buffers are free to write.
		a.mask.buffer.write(mask);
		a.sample.buffer.write(sample);
		a.kp2.buffer.write(in2);
		a.kp3.buffer.write(in3);
		const pe2d = rows(a.pe, a.at.kp2d, K), pe3d = rows(a.pe, a.at.kp3d, K);
		op.linearBiasRows(g.pos2d0W, g.pos2d0B, a.kp2, a.kp2H, D, 4, K);
		op.relu(a.kp2H, K * D);
		op.linearBiasRows(g.pos2dW, g.pos2dB, a.kp2H, pe2d, D, D, K);
		op.mul(pe2d, a.mask, K * D);
		// The features under the keypoints: [K, N] @ ctx [N, C].
		op.matmul(a.image.ctxT, a.sample, a.sampled, C, N, K);
		op.linearBiasRows(g.featW, g.featB, a.sampled, a.y, D, C, K);
		op.add(rows(a.x, a.at.kp2d, K), a.y, K * D);
		op.linearBiasRows(g.pos3d0W, g.pos3d0B, a.kp3, a.kp3H, D, 4, K);
		op.relu(a.kp3H, K * D);
		op.linearBiasRows(g.pos3dW, g.pos3dB, a.kp3H, pe3d, D, D, K);
	}
}
