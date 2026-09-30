// Kimodo's motion denoiser: a prompt embedding and a noisy motion in, the
// clean motion it predicts out, and the DDIM loop around it.
//
// Two transformers (`root_model.`, `body_model.`) of the same shape. Each reads
// a sequence of 50 text tokens (the embedding through `embed_text` in the first,
// the zero embedding in the rest), a timestep token, a first-heading token and
// one token a frame, with sinusoidal positions over all of it, runs 16 post-norm
// PyTorch TransformerEncoder layers (ReLU-free: GELU with erf) and projects the
// frame tokens out. The root stage predicts the global root trajectory; that is
// turned into local root velocities on the host and handed, with the rest of
// the motion, to the body stage, which predicts the body.
//
// Classifier-free guidance is upstream's separated form over [text,
// constraint, unconditional]. With no constraints the constraint branch is the
// unconditional one exactly (same inputs, same kernels), so it is not run:
// uncond + text_weight (text - uncond) is what the three would give.
//
// Constraints (guide poses, `poses.js`) are upstream's "concat" motion mask:
// each stage reads the frame with the observed values written over the noisy
// ones where the mask is set, followed by the mask itself. Only the constraint
// branch sees them; the text and unconditional branches read a zero mask, as
// they do without constraints.
//
// All weights are float32 and stay resident (about 1.1 GB for both stages).
// They are read from kimodo.cpp's GGUF or straight from NVIDIA's release
// folder; checkpoint.js hides which.

import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { copy, zero, submit } from '../lib/gpu.js';
import { cosineSchedule, ddimStep, globalToLocalRoot } from './motion.js';
import { SKELETONS } from './skeletons.js';
import { openMotionCheckpoint } from './checkpoint.js';

const EMBEDDING = 4096;
const F = 4;

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

// sin at even d, cos at odd, of value * 10000^(-d / width): the positional and
// timestep encodings both.
function sinusoid(out, offset, value, width) {
	for (let d = 0; d < width; d += 2) {
		const v = Math.fround(value * Math.fround(Math.pow(10000, -d / width)));
		out[offset + d] = Math.sin(v);
		out[offset + d + 1] = Math.cos(v);
	}
}

class MotionTransformer {
	constructor(model, prefix, config) {
		const f = (name) => model.upload(prefix + name, 'f32');
		const W = (this.width = config.width);
		this.heads = config.heads;
		this.ffn = config.ffn;
		this.textTokens = config.textTokens;
		this.prefixTokens = config.textTokens + 2;
		this.inDim = model.shape(prefix + 'input_linear.weight')[0];
		this.outDim = model.shape(prefix + 'output_linear.weight')[1];
		this.input = { w: f('input_linear.weight'), b: f('input_linear.bias') };
		this.text = { w: f('embed_text.weight'), b: f('embed_text.bias') };
		this.time0 = { w: f('embed_timestep.time_embed.0.weight'), b: f('embed_timestep.time_embed.0.bias') };
		this.time2 = { w: f('embed_timestep.time_embed.2.weight'), b: f('embed_timestep.time_embed.2.bias') };
		this.heading = { w: f('linear_first_heading_angle.weight'), b: f('linear_first_heading_angle.bias') };
		this.output = { w: f('output_linear.weight'), b: f('output_linear.bias') };
		this.layers = [];
		for (let l = 0; l < config.layers; l++) {
			const p = `seqTransEncoder.layers.${l}.`;
			const qkvBias = model.floats(prefix + p + 'self_attn.in_proj_bias');
			this.layers.push({
				qkv: f(p + 'self_attn.in_proj_weight'),
				qBias: upload(qkvBias.subarray(0, W)),
				kBias: upload(qkvBias.subarray(W, 2 * W)),
				vBias: upload(qkvBias.subarray(2 * W, 3 * W)),
				out: { w: f(p + 'self_attn.out_proj.weight'), b: f(p + 'self_attn.out_proj.bias') },
				norm1: { w: f(p + 'norm1.weight'), b: f(p + 'norm1.bias') },
				ff1: { w: f(p + 'linear1.weight'), b: f(p + 'linear1.bias') },
				ff2: { w: f(p + 'linear2.weight'), b: f(p + 'linear2.bias') },
				norm2: { w: f(p + 'norm2.weight'), b: f(p + 'norm2.bias') },
			});
		}
		this.a = null;
	}

	linear(lin, x, y, out, inDim, rows) {
		op.matmul(lin.w, x, y, out, inDim, rows);
		op.biasAdd(y, lin.b, rows * out, out);
	}

	// Scratch for `batch` sequences of `frames` frames, and the condition tokens
	// for each: `embeddings[b]` a Float32Array(4096) or null (unconditional),
	// every sequence starting from `heading` radians.
	prepare(batch, frames, embeddings, heading) {
		this.release();
		const W = this.width, seq = this.prefixTokens + frames, rows = batch * seq;
		const a = (this.a = {
			batch, frames, seq,
			x: f32(rows * W), h: f32(rows * W), q: f32(rows * W), k: f32(rows * W), v: f32(rows * W),
			qh: f32(rows * W), kh: f32(rows * W), vh: f32(rows * W),
			attn: f32(rows * W), tmp: f32(rows * W), ff: f32(rows * this.ffn),
			prefix: f32(batch * this.prefixTokens * W),
			timeIn: f32(W), time1: f32(W), time: f32(W),
			inputs: [], outputs: [],
		});
		for (let b = 0; b < batch; b++) {
			a.inputs.push(f32(frames * this.inDim));
			a.outputs.push(f32(frames * this.outDim));
		}
		const positions = new Float32Array(seq * W);
		for (let s = 0; s < seq; s++) sinusoid(positions, s * W, s, W);
		a.positions = upload(positions);

		// Per sequence: text tokens, an empty timestep slot, the heading token,
		// each with its position. The timestep token is added per step.
		zero(a.prefix);
		const text = new Float32Array(this.textTokens * EMBEDDING);
		const textIn = f32(text.length);
		const angle = upload(new Float32Array([Math.cos(heading), Math.sin(heading)]));
		for (let b = 0; b < batch; b++) {
			text.fill(0);
			if (embeddings[b]) text.set(embeddings[b]);
			textIn.buffer.write(text);
			const at = b * this.prefixTokens * W;
			const tokens = a.prefix.view(at * F, this.prefixTokens * W * F);
			this.linear(this.text, textIn, tokens, W, EMBEDDING, this.textTokens);
			const headingRow = a.prefix.view((at + (this.textTokens + 1) * W) * F, W * F);
			this.linear(this.heading, angle, headingRow, W, 2, 1);
			op.add(tokens, a.positions, this.prefixTokens * W);
			submit();
		}
		textIn.dispose();
		angle.dispose();
	}

	// Self-attention of each sequence on its own: Q, K and V to [heads, seq,
	// hd] and the float32 flash kernel (the matrix-core one is float16).
	attention(a, seq) {
		const W = this.width, hd = W / this.heads, size = seq * W * F;
		for (let b = 0; b < a.batch; b++) {
			const at = b * size;
			op.transposeHeads(a.q.view(at, size), a.qh.view(at, size), seq, this.heads, hd, 0, true);
			op.transposeHeads(a.k.view(at, size), a.kh.view(at, size), seq, this.heads, hd, 0, true);
			op.transposeHeads(a.v.view(at, size), a.vh.view(at, size), seq, this.heads, hd, 0);
			op.flashAttentionScalar(a.qh.view(at, size), a.kh.view(at, size), a.vh.view(at, size), a.attn.view(at, size), this.heads, seq, hd);
		}
	}

	// One pass: `inputs[b]` the [frames, inDim] rows of each sequence, all at
	// diffusion timestep `timestep`; the [frames, outDim] predictions of each.
	run(inputs, timestep) {
		const a = this.a;
		const W = this.width, P = this.prefixTokens, T = a.frames, seq = a.seq, rows = a.batch * seq;
		const time = new Float32Array(W);
		sinusoid(time, 0, timestep, W);
		a.timeIn.buffer.write(time);
		this.linear(this.time0, a.timeIn, a.time1, W, W, 1);
		op.silu(a.time1, W);
		this.linear(this.time2, a.time1, a.time, W, W, 1);

		for (let b = 0; b < a.batch; b++) {
			a.inputs[b].buffer.write(inputs[b]);
			copy(a.prefix, a.x, P * W * F, b * P * W * F, b * seq * W * F);
			op.add(a.x.view((b * seq + this.textTokens) * W * F, W * F), a.time, W);
			const frameRows = a.x.view((b * seq + P) * W * F, T * W * F);
			this.linear(this.input, a.inputs[b], frameRows, W, this.inDim, T);
			op.add(frameRows, a.positions.view(P * W * F, T * W * F), T * W);
		}

		for (const l of this.layers) {
			op.matmul(l.qkv, a.x, a.q, W, W, rows, 0, true);
			op.matmul(l.qkv, a.x, a.k, W, W, rows, W, true);
			op.matmul(l.qkv, a.x, a.v, W, W, rows, 2 * W);
			op.biasAdd(a.q, l.qBias, rows * W, W);
			op.biasAdd(a.k, l.kBias, rows * W, W);
			op.biasAdd(a.v, l.vBias, rows * W, W);
			this.attention(a, seq);
			this.linear(l.out, a.attn, a.tmp, W, W, rows);
			op.add(a.tmp, a.x, rows * W);
			op.layerNormAffine(a.tmp, l.norm1.w, l.norm1.b, a.h, W, rows, 1e-5);
			this.linear(l.ff1, a.h, a.ff, this.ffn, W, rows);
			op.geluErf(a.ff, rows * this.ffn);
			this.linear(l.ff2, a.ff, a.tmp, W, this.ffn, rows);
			op.add(a.tmp, a.h, rows * W);
			op.layerNormAffine(a.tmp, l.norm2.w, l.norm2.b, a.x, W, rows, 1e-5);
		}

		for (let b = 0; b < a.batch; b++) {
			this.linear(this.output, a.x.view((b * seq + P) * W * F, T * W * F), a.outputs[b], this.outDim, W, T);
		}
		submit();
		return a.outputs.map((t) => new Float32Array(t.buffer.readBytes(0, T * this.outDim * F).buffer));
	}

	release() {
		if (!this.a) return;
		for (const v of Object.values(this.a)) {
			if (Array.isArray(v)) v.forEach((t) => t.dispose());
			else if (v?.dispose) v.dispose();
		}
		this.a = null;
	}
}

export class KimodoDenoiser {
	constructor(path) {
		const c = openMotionCheckpoint(path);
		const m = (this.model = c.model);
		this.name = c.name;
		this.skeletonKey = c.skeletonKey;
		this.skeleton = SKELETONS[this.skeletonKey];
		if (!this.skeleton) {
			m.close();
			throw new Error(`no skeleton table for ${this.skeletonKey}`);
		}
		this.motionDim = c.motionDim;
		this.fps = c.fps;
		this.baseSteps = c.baseSteps;
		if (this.motionDim !== 9 + 12 * this.skeleton.parents.length) {
			m.close();
			throw new Error(`motion_dim ${this.motionDim} does not fit the ${this.skeletonKey} skeleton`);
		}
		const config = c.config;
		this.stats = c.stats;
		const t0 = llm.now();
		this.root = new MotionTransformer(m, c.prefix + 'root_model.', config);
		this.body = new MotionTransformer(m, c.prefix + 'body_model.', config);
		llm.print(`  [kimodo motion] ${this.name} (${this.skeletonKey}) resident in ${llm.since(t0)}`);
	}

	// The clean motion for noisy `motion` [frames, D] at `timestep`, one per
	// batch row the stages were prepared for: [text, unconditional], or [text,
	// constraint, unconditional] with `constraint` ({ observed, mask }) set.
	predict(motion, frames, timestep, constraint = null) {
		const D = this.motionDim, batch = this.root.a.batch;
		const plain = new Float32Array(frames * 2 * D);
		for (let t = 0; t < frames; t++) plain.set(motion.subarray(t * D, (t + 1) * D), t * 2 * D);
		// Each row's frames as the stages read them: the motion, with the
		// constraint row's observed values written in, and that row's mask.
		const frameIns = [], rootIns = [];
		for (let b = 0; b < batch; b++) {
			if (!constraint || b !== 1) {
				frameIns.push(motion);
				rootIns.push(plain);
				continue;
			}
			const { observed, mask } = constraint;
			const x = new Float32Array(frames * D), r = new Float32Array(frames * 2 * D);
			for (let i = 0; i < x.length; i++) x[i] = mask[i] ? observed[i] : motion[i];
			for (let t = 0; t < frames; t++) {
				r.set(x.subarray(t * D, (t + 1) * D), t * 2 * D);
				r.set(mask.subarray(t * D, (t + 1) * D), t * 2 * D + D);
			}
			frameIns.push(x);
			rootIns.push(r);
		}
		const roots = this.root.run(rootIns, timestep);

		const all = new Float32Array(batch * frames * 5);
		roots.forEach((r, b) => all.set(r, b * frames * 5));
		const local = globalToLocalRoot(all, batch, frames, this.stats, this.fps);
		const bodyDim = 2 * D - 1;
		const bodyIns = [];
		for (let b = 0; b < batch; b++) {
			const x = new Float32Array(frames * bodyDim);
			const from = frameIns[b], masked = constraint && b === 1;
			for (let t = 0; t < frames; t++) {
				x.set(local.subarray((b * frames + t) * 4, (b * frames + t + 1) * 4), t * bodyDim);
				x.set(from.subarray(t * D + 5, (t + 1) * D), t * bodyDim + 4);
				if (masked) x.set(constraint.mask.subarray(t * D, (t + 1) * D), t * bodyDim + D - 1);
			}
			bodyIns.push(x);
		}
		const bodies = this.body.run(bodyIns, timestep);

		return roots.map((r, b) => {
			const clean = new Float32Array(frames * D);
			for (let t = 0; t < frames; t++) {
				clean.set(r.subarray(t * 5, (t + 1) * 5), t * D);
				clean.set(bodies[b].subarray(t * (D - 5), (t + 1) * (D - 5)), t * D + 5);
			}
			return clean;
		});
	}

	// DDIM from `noise` [frames, D] to a normalised motion [frames, D].
	// `constraint` ({ observed, mask }, `poses.js` encodePoses) adds the
	// constraint branch, weighted by `constraintWeight`.
	async sample({ embedding, noise, frames, steps = 100, textWeight = 2, heading = 0, constraint = null, constraintWeight = 2, onStep = null }) {
		const D = this.motionDim;
		if (noise.length !== frames * D) throw new Error(`noise is ${noise.length} floats, not ${frames} x ${D}`);
		if (this.root.prefixTokens + frames > 1024) throw new Error(`at most ${1024 - this.root.prefixTokens} frames`);
		if (constraint && constraint.mask.length !== frames * D) throw new Error(`the constraint mask is not ${frames} x ${D}`);
		const schedule = cosineSchedule(this.baseSteps, steps);
		const conditions = constraint ? [embedding, null, null] : [embedding, null];
		this.root.prepare(conditions.length, frames, conditions, heading);
		this.body.prepare(conditions.length, frames, conditions, heading);
		let state = Float32Array.from(noise), next = new Float32Array(state.length);
		const guided = new Float32Array(state.length);
		for (let i = steps - 1; i >= 0; i--) {
			const out = this.predict(state, frames, schedule.timesteps[i], constraint);
			const text = out[0], uncond = out[out.length - 1];
			if (constraint) {
				const posed = out[1];
				for (let j = 0; j < guided.length; j++) {
					guided[j] = uncond[j] + Math.fround(textWeight * (text[j] - uncond[j])) + Math.fround(constraintWeight * (posed[j] - uncond[j]));
				}
			} else {
				for (let j = 0; j < guided.length; j++) guided[j] = uncond[j] + Math.fround(textWeight * (text[j] - uncond[j]));
			}
			ddimStep(schedule, i, state, guided, next);
			[state, next] = [next, state];
			if (onStep) await onStep(steps - i, steps);
		}
		this.root.release();
		this.body.release();
		return state;
	}

	close() {
		this.model.close();
	}
}
