// Kimodo (NVIDIA's text-to-motion model) as a plugin: a prompt in, a skeletal
// animation out.
//
//   llm-runner kimodo text_model=Llama-3-Kimodo-Q8_0.gguf model=kimodo-soma-rp-v1.1-f32.gguf \
//       prompt="a person waves" output=wave
//
// `model` is kimodo.cpp's motion GGUF or NVIDIA's release as downloaded from
// huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1 (the folder, or its .safetensors
// under any name; stats.js carries the statistics for the models it knows,
// others need stats/ beside it; config.yaml is not needed). The text
// encoder is only available as a GGUF (LocalAI-io/Llama-3-Kimodo-GGML).
//
// The text encoder runs first and is dropped before the motion weights load.
// `output` is a directory: motion.glb, motion.json (skeleton, fps, root positions,
// parent-local xyzw rotations and the model's foot contacts a frame), plus the raw float32 streams kimodo.cpp's
// `kmd-sample-embedding` writes (sampling_final_state, root_positions,
// local_rotations_xyzw) and the noise and embedding used, so a run can be
// replayed there.
//
// `glb` names a .glb to write the clip to on its own - the skeleton as nodes and
// one animation, named `name` (the prompt by default). `glb_output` hands the
// same bytes to the program running the job in-process instead, as its output
// "glb" (llm.output), so nothing is written to disk - which is what crig does.
//
// Settings: frames (150, at 30 fps), steps (100), seed (42), text_cfg (2),
// heading (0, radians), progress (false; true prints every step, for a caller
// reading stdout). `poses` are guide keyframes the motion has to pass through
// (see poses.js for their form), weighted by constraint_cfg (2); a pose on frame
// 0 also sets the starting heading. `cleanup` runs upstream's clean-up on the
// result (see cleanup.js): planted feet stop sliding and the guide poses'
// frames land on them exactly. It is on unless `reference` is given, which
// compares the model's own output.
//
// `skeleton_glb` writes one skeleton at rest as a .glb - `skeleton` names it
// (soma30 by default) - and needs no weights: what a caller matches its own rig
// against to say which of its joints a guide pose's names mean. `embedding` reads a 4096-float file instead of running
// the text encoder; `noise` reads the initial noise instead of drawing it from
// the seed. `reference` compares the embedding (text only) or, with `model`, a
// kmd-sample-embedding output directory. Without `model` only the embedding is
// made (`output` is then the file it is written to).

import { llm } from '../lib/llm.js';
import { useMatrixCores } from '../lib/ops.js';
import { profileReport } from '../lib/gpu.js';
import { KimodoTextEncoder } from './encoder.js';
import { KimodoDenoiser } from './denoiser.js';
import { decodeMotion } from './motion.js';
import { writeText } from '../lib/graph/plugins.js';
import { writeMotionGlb, motionGlb } from './glb.js';
import { SKELETONS } from './skeletons.js';
import { restPositions, placePoses, encodePoses } from './poses.js';
import { canCleanUp, cleanUpMotion } from './cleanup.js';

function compare(a, b) {
	let dot = 0, na = 0, nb = 0, maxDiff = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
		maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
	}
	return { cosine: dot / Math.sqrt(na * nb), maxDiff, norm: Math.sqrt(na), referenceNorm: Math.sqrt(nb) };
}

function report(label, c) {
	llm.print(`  ${label} vs reference: cosine ${c.cosine.toFixed(6)}, max |diff| ${c.maxDiff.toExponential(3)}, norm ${c.norm.toFixed(4)} / ${c.referenceNorm.toFixed(4)}`);
}

const floats = (path) => new Float32Array(llm.readBytes(path).buffer.slice(0));

async function embed(config) {
	if (config.embedding) return floats(config.embedding);
	const encoder = new KimodoTextEncoder(config.text_model, config.tokenizer ?? null);
	const prompt = config.prompt ?? '';
	llm.print(`  prompt: ${JSON.stringify(prompt)} (${encoder.tokenizer.encode(prompt).length} tokens)`);
	const embedding = await encoder.encode(prompt);
	encoder.close();
	return embedding;
}

// One skeleton standing at rest, feet on the floor, as a one-frame clip.
function writeRestGlb(path, key) {
	const skeleton = SKELETONS[key];
	if (!skeleton) throw new Error(`no skeleton ${key}; there are ${Object.keys(SKELETONS).join(', ')}`);
	const J = skeleton.parents.length;
	const rotations = new Float32Array(J * 4);
	for (let j = 0; j < J; j++) rotations[j * 4 + 3] = 1;
	const floor = Math.min(...restPositions(skeleton).map((p) => p[1]));
	writeMotionGlb(path, { name: 'rest', skeleton, fps: 30, frames: 1, rotations, root: new Float32Array([0, -floor, 0]) });
	return { glb: path, skeleton: key, joints: J };
}

llm.plugin({
	name: 'kimodo',
	async generate(config) {
		if (config.skeleton_glb) return writeRestGlb(config.skeleton_glb, config.skeleton ?? 'soma30');
		const start = llm.now();
		useMatrixCores(config.matrix_cores ?? true);
		llm.print('\n=== Kimodo ===');
		const embedding = await embed(config);
		if (!config.model) {
			const result = { seconds: (llm.now() - start) / 1000 };
			if (config.output) llm.writeBytes((result.output = config.output), embedding);
			if (config.reference) report('embedding', (result.reference = compare(embedding, floats(config.reference))));
			return result;
		}

		const denoiser = new KimodoDenoiser(config.model);
		const frames = config.frames ?? 150;
		const steps = config.steps ?? 100;
		const seed = config.seed ?? 42;
		const noise = config.noise ? floats(config.noise) : llm.randomNormal(frames * denoiser.motionDim, seed);
		let constraint = null, heading = config.heading ?? 0, keyed = 0, placed = [];
		if (config.poses) {
			placed = placePoses(config.poses, denoiser.skeleton, frames);
			if (placed.length === 0) throw new Error(`none of the guide poses falls inside the ${frames} frames generated`);
			const encoded = encodePoses(placed, denoiser.skeleton, denoiser.stats, frames);
			constraint = { observed: encoded.observed, mask: encoded.mask };
			if (encoded.heading !== null && config.heading === undefined) heading = encoded.heading;
			keyed = placed.length;
			const partial = placed.filter((p) => p.effectors).length;
			llm.print(`  ${keyed} guide pose(s), on frame(s) ${placed.map((p) => p.frame).join(', ')}` +
				(partial ? `; ${partial} holding only ${[...new Set(placed.flatMap((p) => p.effectors ?? []))].join(', ')}` : ''));
		}
		const t0 = llm.now();
		let lastPrint = t0;
		const motion = await denoiser.sample({
			embedding, noise, frames, steps,
			textWeight: config.text_cfg ?? 2,
			heading,
			constraint,
			constraintWeight: config.constraint_cfg ?? 2,
			onStep: (done, total) => {
				if (config.progress || llm.now() - lastPrint > 2000 || done === total) {
					llm.print(`  step ${done}/${total}, ${llm.since(t0)}`);
					lastPrint = llm.now();
				}
			},
		});
		const decoded = decodeMotion(motion, frames, denoiser.skeleton, denoiser.stats);
		const { skeleton, skeletonKey, fps } = denoiser;
		denoiser.close();
		llm.print(`  ${frames} frames, ${steps} steps in ${llm.since(t0)}`);

		let cleaned = null;
		if ((config.cleanup ?? !config.reference) && canCleanUp(skeleton, skeletonKey)) {
			const t1 = llm.now();
			cleaned = cleanUpMotion({ skeleton, key: skeletonKey, frames, ...decoded, keys: placed });
			llm.print(`  cleaned up: feet held on ${cleaned.plantedFrames} frame(s), ${cleaned.keyedFrames} guide pose(s) met, in ${llm.since(t1)}`);
		}

		const result = { frames, fps, steps, seed, skeleton: skeletonKey, poses: keyed, cleaned: cleaned !== null, seconds: (llm.now() - start) / 1000 };
		const clip = { name: config.name ?? config.prompt ?? 'Kimodo', skeleton, fps, frames, rotations: decoded.rotations, root: decoded.root };
		if (config.glb) writeMotionGlb((result.glb = config.glb), clip);
		if (config.glb_output) {
			const bytes = motionGlb(clip);
			llm.output('glb', bytes);
			result.glb_bytes = bytes.length;
		}
		if (config.output) {
			const dir = config.output;
			llm.makeDir(dir);
			llm.writeBytes(`${dir}/sampling_final_state.f32`, motion);
			llm.writeBytes(`${dir}/root_positions.f32`, decoded.root);
			llm.writeBytes(`${dir}/local_rotations_xyzw.f32`, decoded.rotations);
			llm.writeBytes(`${dir}/noise.f32`, noise);
			llm.writeBytes(`${dir}/embedding.f32`, embedding);
			const json = {
				skeleton: skeletonKey, fps, frames, prompt: config.prompt ?? null,
				names: skeleton.names, parents: skeleton.parents, offsets: skeleton.offsets,
				root_positions: Array.from(decoded.root),
				local_rotations_xyzw: Array.from(decoded.rotations),
				foot_contacts: Array.from(decoded.contacts),
			};
			writeText(`${dir}/motion.json`, JSON.stringify(json));
			writeMotionGlb(`${dir}/motion.glb`, clip);
			result.output = dir;
		}
		if (config.reference) {
			const ref = config.reference;
			result.reference = {
				state: compare(motion, floats(`${ref}/sampling_final_state.f32`)),
				root: compare(decoded.root, floats(`${ref}/root_positions.f32`)),
				rotations: compare(decoded.rotations, floats(`${ref}/local_rotations_xyzw.f32`)),
			};
			report('final state', result.reference.state);
			report('root positions', result.reference.root);
			report('rotations', result.reference.rotations);
		}
		if (config.profile) profileReport('kimodo');
		llm.print(`=== done in ${llm.since(start)} ===`);
		return result;
	},
});
