// UniMate's denoiser (the `graph` attention, `adaln` text model): a skeleton,
// a caption vector and a noisy motion in, the flow's velocity out.
//
// The motion is a grid of tokens, one per (frame, joint), frame-major, with the
// skeleton's rest pose as an extra frame in front (61 x J). Each of the 10
// blocks is spatial attention among the joints of a frame (biased by graph
// distance and edge type, rotated by a spectral RoPE of the skeleton's
// Laplacian), temporal attention along each joint's frames (plain RoPE), and a
// SwiGLU MLP - each behind an RMSNorm and adaLN-modulated by one vector y: the
// timestep, the caption and the pooled rest pose. The last layer reads a root
// feature (whole-body attention from the root token) and a joint feature per
// token.
//
// Everything that depends only on the skeleton - the rest-frame tokens, the
// per-joint embeddings added to every frame, the graph biases, both RoPE tables
// and the pooled rest pose - is made once in `prepare`; a step is then one
// `velocity` per branch (the caption and the empty one, for guidance), which
// differ only in y.
//
// The other released model (`full` attention, `cross_attn` text,
// unimate_uniml3d_f60_v2_full_cross_attn) shares the input layer, the pooled
// rest pose and the last layer, but its blocks differ: one attention over all
// 61 x J tokens at once (a RoPE whose first half turns by the joint's spectral
// angles and second half by the frame), then attention from every token to
// the caption's T5 token states (a learnt null caption stands for the empty
// one), then the MLP. Its y is the timestep and the pooled rest pose only.
// Its caption is the T5 states [tokens, 768], not their mean.
//
// Tokens of padded joints are never made: the reference pads every skeleton to
// 71 joints and masks them out of every attention, which gives the same
// numbers for the real ones.

import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit } from '../lib/gpu.js';

const F = 4;
const D = 512; // width
const H = 8; // heads
const HD = D / H;
const FEAT = 12; // features a joint a frame
const FRAMES = 60;
const F1 = FRAMES + 1; // with the rest-pose frame in front
const TEXT = 768;

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

// erf, to float precision (W. J. Cody's rational approximations via erfc).
function erf(x) {
	const z = Math.abs(x);
	const t = 1 / (1 + 0.5 * z);
	const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
		t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
		t * (-0.82215223 + t * 0.17087277)))))))));
	return x >= 0 ? 1 - r : r - 1;
}
const geluErf = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

// y = W x + b on the host, W [out, in] row-major.
function linear(w, b, x, out, inDim) {
	const y = new Float32Array(out);
	for (let o = 0; o < out; o++) {
		let s = b ? b[o] : 0;
		const row = o * inDim;
		for (let i = 0; i < inDim; i++) s += w[row + i] * x[i];
		y[o] = s;
	}
	return y;
}

export class UniMateDenoiser {
	constructor(ckpt) {
		const t0 = llm.now();
		if (ckpt.has('transformer_blocks.0.s_attn.qkv.weight') && !ckpt.has('transformer_blocks.0.c_attn.q_proj.weight')) this.variant = 'graph';
		else if (ckpt.has('transformer_blocks.0.attn.qkv.weight') && ckpt.has('transformer_blocks.0.c_attn.q_proj.weight')) this.variant = 'full';
		else throw new Error('not a UniMate checkpoint the plugin runs: unimate_uniml3d_f60_v2 (graph attention) or unimate_uniml3d_f60_v2_full_cross_attn');
		if (!ckpt.has('cond_embedder.weight')) throw new Error('a UniMate model without text conditioning');
		// What `velocity` takes as the caption: the pooled T5 vector or its token states.
		this.captionKind = this.variant === 'graph' ? 'pooled' : 'tokens';
		if (ckpt.shape('cond_embedder.weight')[0] !== D) throw new Error(`a UniMate model ${ckpt.shape('cond_embedder.weight')[0]} wide; the plugin runs the ${D}-wide release`);
		const up = (n) => ckpt.upload(n);
		const host = (n) => ckpt.floats(n);
		const lin = (p) => ({ w: up(`${p}.weight`), b: up(`${p}.bias`), rows: ckpt.shape(`${p}.weight`)[0] });
		const mlp = (p) => ({ l0: lin(`${p}.0`), l2: lin(`${p}.2`) });
		// A fused [3D, D] projection as three row ranges, with its bias in thirds.
		const qkv = (w, b) => {
			const bias = host(b);
			return { w: up(w), q: upload(bias.subarray(0, D)), k: upload(bias.subarray(D, 2 * D)), v: upload(bias.subarray(2 * D)) };
		};
		const swiglu = (p) => {
			const hidden = ckpt.shape(`${p}.w12.weight`)[0] / 2;
			const b = host(`${p}.w12.bias`);
			return { hidden, w12: up(`${p}.w12.weight`), b1: upload(b.subarray(0, hidden)), b2: upload(b.subarray(hidden)), w3: lin(`${p}.w3`) };
		};

		this.maxDepth = ckpt.shape('depth_embedding.weight')[0] - 1;
		this.depthTable = host('depth_embedding.weight');
		this.time = { w0: host('time_embedder.mlp.0.weight'), b0: host('time_embedder.mlp.0.bias'),
			w2: host('time_embedder.mlp.2.weight'), b2: host('time_embedder.mlp.2.bias') };
		this.cond = { w: host('cond_embedder.weight'), b: host('cond_embedder.bias') };
		this.input = {
			rootTpos: mlp('input_layer.root_tpos_embedder'), rootX: mlp('input_layer.root_x_embedder'),
			jointTpos: mlp('input_layer.joint_tpos_embedder'), parentTpos: mlp('input_layer.parent_tpos_embedder'),
			fuse: mlp('input_layer.tpos_fuse'), jointX: mlp('input_layer.joint_x_embedder'),
		};
		this.names = lin('joint_name_embedder');
		const P = 'tpos_pool.pool.';
		this.pool = {
			queries: host(`${P}queries`), normQ: up(`${P}norm_q.weight`), normKv: up(`${P}norm_kv.weight`),
			attn: qkv(`${P}cross_attn.in_proj_weight`, `${P}cross_attn.in_proj_bias`), out: lin(`${P}cross_attn.out_proj`),
			normFf: up(`${P}norm_ff.weight`), ffn: swiglu(`${P}ffn`), normOut: up(`${P}norm_out.weight`), proj: lin(`${P}out_proj`),
			heads: 4,
		};
		const S = this.variant === 'graph' ? 'rope_j.spectral_encoder.' : 'rope.spectral_encoder.';
		this.signnet = {
			phi0w: host(`${S}phi.0.weight`), phi0b: host(`${S}phi.0.bias`), phi2w: host(`${S}phi.2.weight`), phi2b: host(`${S}phi.2.bias`),
			rho0w: host(`${S}rho.0.weight`), rho0b: host(`${S}rho.0.bias`), rho2w: host(`${S}rho.2.weight`), rho2b: host(`${S}rho.2.bias`),
			hidden: ckpt.shape(`${S}phi.2.weight`)[0], eig: ckpt.shape(`${S}rho.0.weight`)[1] / ckpt.shape(`${S}phi.2.weight`)[0],
			out: ckpt.shape(`${S}rho.2.weight`)[0],
		};
		this.blocks = [];
		if (this.variant === 'full') this.fullBlocks(ckpt, up, host, lin, qkv, swiglu);
		for (let l = 0; this.variant === 'graph' && ckpt.has(`transformer_blocks.${l}.norm_s.weight`); l++) {
			const p = `transformer_blocks.${l}.`;
			const s = `${p}s_attn.`;
			this.blocks.push({
				normS: up(`${p}norm_s.weight`), normT: up(`${p}norm_t.weight`), normM: up(`${p}norm_mlp.weight`),
				s: { ...qkv(`${s}qkv.weight`, `${s}qkv.bias`), proj: lin(`${s}proj`), qn: up(`${s}q_norm.weight`), kn: up(`${s}k_norm.weight`) },
				t: { ...qkv(`${p}t_attn.qkv.weight`, `${p}t_attn.qkv.bias`), proj: lin(`${p}t_attn.proj`), qn: up(`${p}t_attn.q_norm.weight`), kn: up(`${p}t_attn.k_norm.weight`) },
				mlp: swiglu(`${p}mlp`),
				ada: lin(`${p}adaLN_modulation.1`),
				graph: {
					distEmb: host(`${s}graph_dist_embedding.weight`), distW: host(`${s}graph_dist_proj.weight`), distB: host(`${s}graph_dist_proj.bias`),
					distScale: host(`${s}graph_dist_scale`)[0],
					relEmb: host(`${s}graph_rel_embedding.weight`), relW: host(`${s}graph_rel_proj.weight`), relB: host(`${s}graph_rel_proj.bias`),
					relScale: host(`${s}graph_rel_scale`)[0],
				},
			});
		}
		const L = 'final_layer.';
		this.final = {
			norm: up(`${L}norm_final.weight`), ada: lin(`${L}adaLN_modulation.1`),
			attn: qkv(`${L}root_cross_attn.in_proj_weight`, `${L}root_cross_attn.in_proj_bias`), out: lin(`${L}root_cross_attn.out_proj`),
			normQ: up(`${L}root_cross_norm_q.weight`), normKv: up(`${L}root_cross_norm_kv.weight`),
			rootOut: mlp(`${L}root_out`), jointOut: mlp(`${L}joint_out`), heads: 4,
		};
		this.a = null;
		llm.print(`  [unimate] denoiser (${this.variant} attention): ${this.blocks.length} blocks, resident in ${llm.since(t0)}`);
	}

	// The full-attention blocks. Their RoPE turns the first half of a head
	// (graph: channel i with i + 16) and the second half (time: 32 + i with
	// 48 + i) on its own; the q and k channels are stored with the middle
	// quarters swapped, which makes that one rotation of i with i + 32 over the
	// whole head (ropeNeox) and leaves every q.k as it was.
	fullBlocks(ckpt, up, host, lin, qkv, swiglu) {
		const Q = HD / 4;
		const swap = (c) => (c >= Q && c < 3 * Q ? (c < 2 * Q ? c + Q : c - Q) : c);
		const perm = (src, rows, width) => {
			const out = new Float32Array(src.length);
			for (let r = 0; r < rows; r++) {
				const h = Math.floor(r / HD) * HD, from = h + swap(r - h);
				out.set(src.subarray(from * width, (from + 1) * width), r * width);
			}
			return out;
		};
		const head = (n) => upload(perm(host(n), HD, 1));
		this.nullCaption = host('null_caption');
		for (let l = 0; ckpt.has(`transformer_blocks.${l}.norm1.weight`); l++) {
			const p = `transformer_blocks.${l}.`;
			const w = host(`${p}attn.qkv.weight`), b = host(`${p}attn.qkv.bias`);
			const qk = new Float32Array(w.length);
			qk.set(perm(w.subarray(0, D * D), D, D), 0);
			qk.set(perm(w.subarray(D * D, 2 * D * D), D, D), D * D);
			qk.set(w.subarray(2 * D * D), 2 * D * D);
			const kv = host(`${p}c_attn.kv_proj.bias`);
			this.blocks.push({
				norm1: up(`${p}norm1.weight`), normC: up(`${p}norm_c.weight`), norm2: up(`${p}norm2.weight`),
				attn: {
					w: upload(qk), q: upload(perm(b.subarray(0, D), D, 1)), k: upload(perm(b.subarray(D, 2 * D), D, 1)), v: upload(b.subarray(2 * D)),
					proj: lin(`${p}attn.proj`), qn: head(`${p}attn.q_norm.weight`), kn: head(`${p}attn.k_norm.weight`),
				},
				cross: {
					q: lin(`${p}c_attn.q_proj`), kv: up(`${p}c_attn.kv_proj.weight`), kb: upload(kv.subarray(0, D)), vb: upload(kv.subarray(D)),
					proj: lin(`${p}c_attn.proj`), qn: up(`${p}c_attn.q_norm.weight`), kn: up(`${p}c_attn.k_norm.weight`),
				},
				mlp: swiglu(`${p}mlp`),
				ada: lin(`${p}adaLN_modulation.1`),
			});
		}
		this.condGpu = lin('cond_embedder');
	}

	linear(l, x, y, out, inDim, rows) { op.linearBiasRows(l.w, l.b, x, y, out, inDim, rows); }

	// Linear, SiLU, Linear through `tmp`.
	mlp(m, x, tmp, y, out, inDim, rows) {
		this.linear(m.l0, x, tmp, D, inDim, rows);
		op.silu(tmp, rows * D);
		this.linear(m.l2, tmp, y, out, D, rows);
	}

	swiglu(s, x, a, b, y, rows) {
		op.matmul(s.w12, x, a, s.hidden, D, rows, 0, true);
		op.matmul(s.w12, x, b, s.hidden, D, rows, s.hidden);
		op.biasAdd(a, s.b1, rows * s.hidden, s.hidden);
		op.biasAdd(b, s.b2, rows * s.hidden, s.hidden);
		op.siluMul(a, b, rows * s.hidden);
		this.linear(s.w3, a, y, D, s.hidden, rows);
	}

	// The spectral RoPE angles of each joint: SignNet over its eigenvector values.
	spectralAngles(spectral, J) {
		const s = this.signnet, K = s.eig, Hn = s.hidden, A = s.out;
		const out = new Float32Array(J * A);
		const phi = (v) => {
			const h0 = new Float32Array(Hn);
			for (let i = 0; i < Hn; i++) h0[i] = geluErf(s.phi0w[i] * v + s.phi0b[i]);
			return linear(s.phi2w, s.phi2b, h0, Hn, Hn);
		};
		for (let j = 0; j < J; j++) {
			const h = new Float32Array(K * Hn);
			for (let k = 0; k < K; k++) {
				const v = spectral[j * K + k];
				const a = phi(v), b = phi(-v);
				for (let i = 0; i < Hn; i++) h[k * Hn + i] = a[i] + b[i];
			}
			const r0 = linear(s.rho0w, s.rho0b, h, Hn, K * Hn).map(geluErf);
			out.set(linear(s.rho2w, s.rho2b, r0, A, Hn), j * A);
		}
		return out;
	}

	// Everything a skeleton fixes. `c`: { J, tpos, tposParents ([J, 12], normalised),
	// spectral [J, 8], depths [J], graphDist, relations [J, J], names [J, 768] }.
	prepare(c) {
		this.release();
		const J = c.J, N = F1 * J, R = FRAMES * J;
		const hidden = this.blocks[0].mlp.hidden;
		const a = (this.a = {
			J, N,
			x: f32(N * D), h: f32(N * D), q: f32(N * D), k: f32(N * D), v: f32(N * D), att: f32(N * D), o: f32(N * D),
			ma: f32(N * hidden), mb: f32(N * hidden),
			xin: f32(R * FEAT), xroot: f32(FRAMES * FEAT), rootEmb: f32(F1 * D), rootTmp: f32(F1 * D),
			outJoint: f32(N * FEAT), outRoot: f32(F1 * FEAT),
			y: f32(D), ys: f32(D), mods: this.blocks.map((b) => f32(b.ada.rows)), modF: f32(2 * D),
			text: new Map(),
		});

		// The rest-pose tokens [J, D]: root and joint embedders, the joints'
		// fused with their parents'.
		const tp = upload(c.tpos), tpp = upload(c.tposParents);
		const cat = f32(J * 2 * D);
		const jt = a.q.view(0, J * D * F), pp = a.k.view(0, J * D * F), tpos = f32(J * D), tmp = a.att.view(0, J * 2 * D * F);
		this.mlp(this.input.jointTpos, tp, tmp, jt, D, FEAT, J);
		this.mlp(this.input.parentTpos, tpp, tmp, pp, D, FEAT, J);
		op.concatRows(jt, pp, cat, J, D, D);
		this.mlp(this.input.fuse, cat, tmp, tpos, D, 2 * D, J);
		this.mlp(this.input.rootTpos, tp, tmp, a.v.view(0, J * D * F), D, FEAT, 1);
		op.copyRows(a.v, tpos, 1, D, D, 0);

		// Added to every frame's tokens: depth and joint-name embeddings.
		const depthRows = new Float32Array(J * D);
		for (let j = 0; j < J; j++) {
			const d = Math.min(c.depths[j], this.maxDepth);
			depthRows.set(this.depthTable.subarray(d * D, (d + 1) * D), j * D);
		}
		a.perJoint = upload(depthRows);
		const names = upload(c.names);
		this.linear(this.names, names, a.o.view(0, J * D * F), D, TEXT, J);
		op.add(a.perJoint, a.o, J * D);
		a.restTokens = f32(J * D);
		op.copyRows(tpos, a.restTokens, J, D, D, 0);
		op.add(a.restTokens, a.perJoint, J * D);

		// The pooled rest pose: 4 learnt queries attend the rest tokens.
		const pl = this.pool, Q = 4;
		const queries = upload(pl.queries);
		const qn = f32(Q * D), kv = f32(J * D), pq = f32(Q * D), pk = f32(J * D), pv = f32(J * D), pa = f32(Q * D), po = f32(Q * D);
		op.rmsNorm(queries, pl.normQ, qn, D, Q, 1e-6);
		op.rmsNorm(tpos, pl.normKv, kv, D, J, 1e-6);
		op.matmul(pl.attn.w, qn, pq, D, D, Q, 0);
		op.matmul(pl.attn.w, kv, pk, D, D, J, D);
		op.matmul(pl.attn.w, kv, pv, D, D, J, 2 * D);
		op.biasAdd(pq, pl.attn.q, Q * D, D);
		op.biasAdd(pk, pl.attn.k, J * D, D);
		op.biasAdd(pv, pl.attn.v, J * D, D);
		op.attentionRows(pq, pk, pv, pa, pl.heads, D / pl.heads, { lq: Q, lk: J });
		this.linear(pl.out, pa, po, D, D, Q);
		op.add(queries, po, Q * D);
		op.rmsNorm(queries, pl.normFf, qn, D, Q, 1e-6);
		this.swiglu(pl.ffn, qn, a.ma, a.mb, po, Q);
		op.add(queries, po, Q * D);
		submit();
		const qs = new Float32Array(queries.buffer.readBytes(0, Q * D * F).buffer);
		const mean = new Float32Array(D);
		for (let i = 0; i < Q; i++) for (let d = 0; d < D; d++) mean[d] += qs[i * D + d] / Q;
		const meanT = upload(mean), normed = f32(D), pooled = f32(D);
		op.rmsNorm(meanT, pl.normOut, normed, D, 1, 1e-6);
		this.linear(pl.proj, normed, pooled, D, D, 1);
		submit();
		a.tposVector = new Float32Array(pooled.buffer.readBytes(0, D * F).buffer);
		for (const t of [tp, tpp, cat, tpos, names, queries, qn, kv, pq, pk, pv, pa, po, meanT, normed, pooled]) t.dispose();

		if (this.variant === 'full') {
			this.fullRope(c.spectral, J);
			submit();
			return;
		}

		// Graph biases, one [H, J, J] a block.
		a.graphBias = this.blocks.map((b) => {
			const g = b.graph, E = g.distEmb.length / 6;
			const table = (emb, w, bias, scale, rows) => {
				const t = new Float32Array(rows * H);
				for (let r = 0; r < rows; r++) {
					const y = linear(w, bias, emb.subarray(r * E, (r + 1) * E), H, E);
					for (let h = 0; h < H; h++) t[r * H + h] = y[h] * scale;
				}
				return t;
			};
			const dist = table(g.distEmb, g.distW, g.distB, g.distScale, g.distEmb.length / E);
			const rel = table(g.relEmb, g.relW, g.relB, g.relScale, g.relEmb.length / E);
			const bias = new Float32Array(H * J * J);
			for (let h = 0; h < H; h++) {
				for (let i = 0; i < J; i++) {
					for (let j = 0; j < J; j++) {
						bias[(h * J + i) * J + j] = dist[c.graphDist[i * J + j] * H + h] + rel[c.relations[i * J + j] * H + h];
					}
				}
			}
			return upload(bias);
		});

		// RoPE tables for every token row: spatial by joint (spectral angles),
		// temporal by frame (RopeND: base 200 over the 61 frames).
		const angles = this.spectralAngles(c.spectral, J);
		const half = HD / 2;
		const sc = new Float32Array(N * half), ss = new Float32Array(N * half);
		const tc = new Float32Array(N * half), ts = new Float32Array(N * half);
		const base = (Math.floor(Math.floor(8 * F1 / Math.PI) / 100) + 1) * 100;
		for (let f = 0; f < F1; f++) {
			for (let j = 0; j < J; j++) {
				const n = f * J + j;
				for (let k = 0; k < half; k++) {
					const sa = angles[j * half + k];
					sc[n * half + k] = Math.cos(sa);
					ss[n * half + k] = Math.sin(sa);
					const ta = Math.fround(f * Math.fround(1 / Math.pow(base, (2 * k) / HD)));
					tc[n * half + k] = Math.cos(ta);
					ts[n * half + k] = Math.sin(ta);
				}
			}
		}
		a.spatialCos = upload(sc);
		a.spatialSin = upload(ss);
		a.temporalCos = upload(tc);
		a.temporalSin = upload(ts);
		submit();
	}

	// The full-attention RoPE table, one row of 32 angles a token: the joint's
	// 16 spectral angles, then the frame's 16 (base 200, as RopeND's rule gives
	// for 61 frames) - in the order fullBlocks stored the channels.
	fullRope(spectral, J) {
		const a = this.a, N = a.N, half = HD / 2, G = this.signnet.out, T = half - G;
		const angles = this.spectralAngles(spectral, J);
		const base = (Math.floor(Math.floor(8 * F1 / Math.PI) / 100) + 1) * 100;
		const cos = new Float32Array(N * half), sin = new Float32Array(N * half);
		for (let f = 0; f < F1; f++) {
			for (let j = 0; j < J; j++) {
				const n = (f * J + j) * half;
				for (let k = 0; k < G; k++) {
					cos[n + k] = Math.cos(angles[j * G + k]);
					sin[n + k] = Math.sin(angles[j * G + k]);
				}
				for (let k = 0; k < T; k++) {
					const ta = Math.fround(f * Math.fround(1 / Math.pow(base, (2 * k) / (2 * T))));
					cos[n + G + k] = Math.cos(ta);
					sin[n + G + k] = Math.sin(ta);
				}
			}
		}
		a.ropeCos = upload(cos);
		a.ropeSin = upload(sin);
		a.heads = [f32(N * D), f32(N * D), f32(N * D)];
	}

	// A caption's keys and values for every block's cross-attention, made once
	// a caption: its T5 states [tokens, 768] (or null, the learnt null caption)
	// through the caption embedder and each block's key/value projection.
	textMemory(caption) {
		const a = this.a, key = caption ?? 'null';
		let m = a.text.get(key);
		if (m) return m;
		const tokens = caption ?? this.nullCaption, T = tokens.length / TEXT;
		const raw = upload(tokens), mem = f32(T * D);
		this.linear(this.condGpu, raw, mem, D, TEXT, T);
		m = { T, k: [], v: [] };
		for (const b of this.blocks) {
			const k = f32(T * D), v = f32(T * D), c = b.cross;
			op.matmul(c.kv, mem, k, D, D, T, 0, true);
			op.matmul(c.kv, mem, v, D, D, T, D);
			op.biasAdd(k, c.kb, T * D, D);
			op.biasAdd(v, c.vb, T * D, D);
			op.headNorm(k, c.kn, H, HD, T, 1e-6);
			m.k.push(k);
			m.v.push(v);
		}
		submit();
		raw.dispose();
		mem.dispose();
		a.text.set(key, m);
		return m;
	}

	// The adaLN vector: timestep t in [0, 1], the caption (or null: the empty
	// branch; the full-attention model reads its caption elsewhere) and the
	// pooled rest pose.
	adalnVector(t, caption) {
		const tm = this.time, half = 128;
		const freq = new Float32Array(256);
		for (let i = 0; i < half; i++) {
			const arg = Math.fround(t * Math.fround(Math.exp(-Math.log(10000) * i / half)));
			freq[i] = Math.cos(arg);
			freq[half + i] = Math.sin(arg);
		}
		const h = linear(tm.w0, tm.b0, freq, D, 256).map((v) => v / (1 + Math.exp(-v)));
		const y = linear(tm.w2, tm.b2, h, D, D);
		const c = this.variant === 'full' ? null : caption ? linear(this.cond.w, this.cond.b, caption, D, TEXT) : this.cond.b;
		for (let d = 0; d < D; d++) y[d] += (c ? c[d] : 0) + this.a.tposVector[d];
		return y;
	}

	// One network pass. `x` is the noisy motion as the reference holds it,
	// [J, 12, 60]; the velocity comes back the same way.
	velocity(x, t, caption) {
		const a = this.a, J = a.J, N = a.N, R = FRAMES * J;

		// Modulations of every block from y.
		const y = this.adalnVector(t, caption);
		a.y.buffer.write(y.map((v) => v / (1 + Math.exp(-v))));
		this.blocks.forEach((b, l) => this.linear(b.ada, a.y, a.mods[l], b.ada.rows, D, 1));
		this.linear(this.final.ada, a.y, a.modF, 2 * D, D, 1);

		// Tokens: the rest frame, then the motion's frames (root and joints
		// through their own embedders), each with the per-joint embeddings.
		const xin = new Float32Array(R * FEAT), xroot = new Float32Array(FRAMES * FEAT);
		for (let j = 0; j < J; j++) {
			for (let d = 0; d < FEAT; d++) {
				for (let f = 0; f < FRAMES; f++) xin[(f * J + j) * FEAT + d] = x[(j * FEAT + d) * FRAMES + f];
			}
		}
		for (let f = 0; f < FRAMES; f++) xroot.set(xin.subarray(f * J * FEAT, f * J * FEAT + FEAT), f * FEAT);
		a.xin.buffer.write(xin);
		a.xroot.buffer.write(xroot);
		op.copyRows(a.restTokens, a.x, J, D, D, 0);
		const frames = a.x.view(J * D * F, R * D * F);
		this.mlp(this.input.jointX, a.xin, a.h, frames, D, FEAT, R);
		this.mlp(this.input.rootX, a.xroot, a.rootTmp, a.rootEmb, D, FEAT, FRAMES);
		op.copyRows(a.rootEmb, frames, FRAMES, D, J * D, 0);
		op.biasAdd(frames, a.perJoint, R * D, J * D);

		const spatial = { lq: J, lk: J, sequences: F1, qBs: J, qIs: 1, kvBs: J, kvIs: 1 };
		const temporal = { lq: F1, lk: F1, sequences: J, qBs: 1, qIs: J, kvBs: 1, kvIs: J };
		if (this.variant === 'full') {
			const text = this.textMemory(caption);
			this.blocks.forEach((b, l) => this.fullBlock(b, a.mods[l], text.k[l], text.v[l], text.T));
		}
		if (this.variant === 'graph') this.blocks.forEach((b, l) => {
			const mod = a.mods[l];
			this.attention(b.normS, b.s, mod, 0, a.spatialCos, a.spatialSin, spatial, a.graphBias[l]);
			this.attention(b.normT, b.t, mod, 3, a.temporalCos, a.temporalSin, temporal, null);
			op.rmsNorm(a.x, b.normM, a.h, D, N, 1e-6);
			op.modulate(a.h, mod, a.h, N * D, D, 7 * D, 6 * D);
			this.swiglu(b.mlp, a.h, a.ma, a.mb, a.o, N);
			op.gatedAdd(a.x, mod, a.o, N * D, D, 8 * D);
		});

		// Final layer: modulate, the root token's attention over the frame's
		// other joints, then the root and joint heads.
		const fl = this.final;
		op.rmsNorm(a.x, fl.norm, a.h, D, N, 1e-6);
		op.modulate(a.h, a.modF, a.h, N * D, D, D, 0);
		op.rmsNorm(a.h, fl.normQ, a.o, D, N, 1e-6);
		op.matmul(fl.attn.w, a.o, a.q, D, D, N, 0);
		op.rmsNorm(a.h, fl.normKv, a.o, D, N, 1e-6);
		op.matmul(fl.attn.w, a.o, a.k, D, D, N, D);
		op.matmul(fl.attn.w, a.h, a.v, D, D, N, 2 * D);
		op.biasAdd(a.q, fl.attn.q, N * D, D);
		op.biasAdd(a.k, fl.attn.k, N * D, D);
		op.biasAdd(a.v, fl.attn.v, N * D, D);
		op.attentionRows(a.q, a.k, a.v, a.att, fl.heads, D / fl.heads,
			{ lq: 1, lk: J - 1, sequences: F1, qBs: J, kvOff: 1, kvBs: J, kvIs: 1 });
		op.copyRows(a.att, a.rootTmp, F1, D, D, 0, J * D);
		this.linear(fl.out, a.rootTmp, a.rootEmb, D, D, F1);
		op.copyRows(a.h, a.rootTmp, F1, D, D, 0, J * D);
		op.add(a.rootTmp, a.rootEmb, F1 * D);
		this.mlp(fl.rootOut, a.rootTmp, a.rootEmb, a.outRoot, FEAT, D, F1);
		this.mlp(fl.jointOut, a.h, a.o, a.outJoint, FEAT, D, N);
		submit();

		const joints = new Float32Array(a.outJoint.buffer.readBytes(0, N * FEAT * F).buffer);
		const roots = new Float32Array(a.outRoot.buffer.readBytes(0, F1 * FEAT * F).buffer);
		const v = new Float32Array(J * FEAT * FRAMES);
		for (let f = 0; f < FRAMES; f++) {
			for (let j = 0; j < J; j++) {
				const src = j === 0 ? roots.subarray((f + 1) * FEAT) : joints.subarray(((f + 1) * J + j) * FEAT);
				for (let d = 0; d < FEAT; d++) v[(j * FEAT + d) * FRAMES + f] = src[d];
			}
		}
		return v;
	}

	// A full-attention block: self-attention over every token, attention to
	// the caption's keys and values (added as it is: no modulation, no gate),
	// the MLP.
	fullBlock(b, mod, textK, textV, T) {
		const a = this.a, N = a.N, s = b.attn, c = b.cross;
		const [qh, kh, vh] = a.heads;
		op.rmsNorm(a.x, b.norm1, a.h, D, N, 1e-6);
		op.modulate(a.h, mod, a.h, N * D, D, D, 0);
		op.matmul(s.w, a.h, a.q, D, D, N, 0, true);
		op.matmul(s.w, a.h, a.k, D, D, N, D, true);
		op.matmul(s.w, a.h, a.v, D, D, N, 2 * D);
		op.biasAdd(a.q, s.q, N * D, D);
		op.biasAdd(a.k, s.k, N * D, D);
		op.biasAdd(a.v, s.v, N * D, D);
		op.headNorm(a.q, s.qn, H, HD, N, 1e-6);
		op.headNorm(a.k, s.kn, H, HD, N, 1e-6);
		op.ropeNeox(a.q, a.ropeCos, a.ropeSin, HD, H, N);
		op.ropeNeox(a.k, a.ropeCos, a.ropeSin, HD, H, N);
		op.transposeHeads(a.q, qh, N, H, HD);
		op.transposeHeads(a.k, kh, N, H, HD);
		op.transposeHeads(a.v, vh, N, H, HD);
		op.flashAttentionScalar(qh, kh, vh, a.att, H, N, HD);
		this.linear(s.proj, a.att, a.o, D, D, N);
		op.gatedAdd(a.x, mod, a.o, N * D, D, 2 * D);

		op.rmsNorm(a.x, b.normC, a.h, D, N, 1e-6);
		this.linear(c.q, a.h, a.q, D, D, N);
		op.headNorm(a.q, c.qn, H, HD, N, 1e-6);
		op.attentionRows(a.q, textK, textV, a.att, H, HD, { lq: N, lk: T });
		this.linear(c.proj, a.att, a.o, D, D, N);
		op.add(a.x, a.o, N * D);

		op.rmsNorm(a.x, b.norm2, a.h, D, N, 1e-6);
		op.modulate(a.h, mod, a.h, N * D, D, 4 * D, 3 * D);
		this.swiglu(b.mlp, a.h, a.ma, a.mb, a.o, N);
		op.gatedAdd(a.x, mod, a.o, N * D, D, 5 * D);
	}

	// One attention stage: x += gate * proj(attention(modulate(norm(x)))).
	// `at` is the stage's first chunk of the block's modulation (shift, scale, gate).
	attention(norm, s, mod, at, cos, sin, shape, bias) {
		const a = this.a, N = a.N;
		op.rmsNorm(a.x, norm, a.h, D, N, 1e-6);
		op.modulate(a.h, mod, a.h, N * D, D, (at + 1) * D, at * D);
		op.matmul(s.w, a.h, a.q, D, D, N, 0, true);
		op.matmul(s.w, a.h, a.k, D, D, N, D, true);
		op.matmul(s.w, a.h, a.v, D, D, N, 2 * D);
		op.biasAdd(a.q, s.q, N * D, D);
		op.biasAdd(a.k, s.k, N * D, D);
		op.biasAdd(a.v, s.v, N * D, D);
		op.headNorm(a.q, s.qn, H, HD, N, 1e-6);
		op.headNorm(a.k, s.kn, H, HD, N, 1e-6);
		op.ropeNeox(a.q, cos, sin, HD, H, N);
		op.ropeNeox(a.k, cos, sin, HD, H, N);
		op.attentionRows(a.q, a.k, a.v, a.att, H, HD, shape, bias, 0);
		this.linear(s.proj, a.att, a.o, D, D, N);
		op.gatedAdd(a.x, mod, a.o, N * D, D, (at + 2) * D);
	}

	// Euler along the flow from `noise` (t = 0) to a motion (t = 1), in `steps`
	// even steps, each the empty branch plus `cfg` times the caption's
	// difference from it. [J, 12, 60] in, [J, 12, 60] (normalised) out.
	async sample({ noise, caption, steps = 20, cfg = 3, onStep = null }) {
		const x = Float32Array.from(noise);
		for (let i = 0; i < steps; i++) {
			const t = Math.fround(i / steps), dt = Math.fround((i + 1) / steps) - t;
			const vc = this.velocity(x, t, caption);
			const vu = cfg === 1 ? vc : this.velocity(x, t, null);
			for (let k = 0; k < x.length; k++) x[k] += dt * (vu[k] + cfg * (vc[k] - vu[k]));
			if (onStep) await onStep(i + 1, steps);
		}
		return x;
	}

	release() {
		if (!this.a) return;
		for (const m of this.a.text.values()) [...m.k, ...m.v].forEach((t) => t.dispose());
		for (const v of Object.values(this.a)) {
			if (Array.isArray(v)) v.forEach((t) => t?.dispose?.());
			else if (v?.dispose) v.dispose();
		}
		this.a = null;
	}
}
