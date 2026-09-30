// UniMate (github.com/Friedrich-M/UniMate, MIT) as a plugin: a rig of any
// shape and a prompt in, two seconds of that rig moving out.
//
//   llm-runner unimate model=checkpoint_step_100000.pt text_model=flan-t5-base/ \
//       rig_glb=character.glb prompt="walks forward" glb=walk.glb
//
// `model` is a released checkpoint as downloaded (huggingface.co/Linzhan/UniMate;
// its EMA weights are read straight from the .pt), either model:
// unimate_uniml3d_f60_v2/checkpoints/checkpoint_step_100000.pt (graph
// attention, the caption as one pooled vector; the recommended one) or
// unimate_uniml3d_f60_v2_full_cross_attn/checkpoints/checkpoint_step_90000.pt
// (full attention, every token attending the caption's words; ~40% slower).
// The checkpoint says which it is. `text_model` is the folder of
// google/flan-t5-base (model.safetensors and tokenizer.json).
//
// The rig is `rig` - { names, parents, positions }: joint names, parent index
// (-1 for the root) and model-space rest positions, Y up, facing +Z - or
// `rig_glb`, a .glb whose first skin is read. 5 to 71 joints.
//
// Every caption the model learnt from reads "An object <does something>": the
// skeleton says what the object is. A prompt is put in that form unless
// `raw_prompt` is set ("walks forward", "A person walks forward" -> "An object
// walks forward").
//
// Settings: steps (20, Euler along the flow), cfg (3), seed (42), stats (auto:
// 'mixamo' for a humanoid, 'truebones' for anything else; or 'objaverse'),
// face ([right, left] joint names that say which way the rig faces; picked
// from the names otherwise), progress (print every step).
//
// Output: `glb` names a .glb for the clip - the rig's joints as nodes (rest
// rotations identity, the root at its rest position) and one animation named
// `name` (the prompt by default), 60 frames at 30 fps, in the rig's own space
// and facing; its extras say so ({ facing: 'as-is' }), for crig not to turn it.
// The result says which joints faced the rig and which statistics were used.

import { llm } from '../lib/llm.js';
import { profileReport } from '../lib/gpu.js';
import { T5Encoder, meanPool } from '../lib/t5.js';
import { writeMotionGlb } from '../kimodo/glb.js';
import { UniMateCheckpoint } from './checkpoint.js';
import { UniMateDenoiser } from './denoiser.js';
import { skeletonCondition } from './skeleton.js';
import { denormalize, decodeCanonical, toRigFrame } from './motion.js';
import { readRigGlb } from './rig.js';

const FRAMES = 60, FPS = 30, FEAT = 12;

// The prompt in the training captions' form.
export function captionOf(prompt) {
	let p = prompt.trim().replace(/\s+/g, ' ');
	if (!p) return p;
	if (/^an object\b/i.test(p)) return 'An object' + p.slice(9);
	const subject = /^(?:a|an|the|this|my)\s+(?:person|man|woman|human|character|creature|animal|figure|model|robot|object|dog|cat|horse|bird|monster|dragon|actor)\b\s*/i;
	if (subject.test(p)) p = p.replace(subject, '');
	else if (/^(?:someone|somebody|he|she|it|they)\b\s*/i.test(p)) p = p.replace(/^\S+\s*/, '');
	if (!p) return 'An object';
	p = 'An object ' + p[0].toLowerCase() + p.slice(1);
	return /[.!?]$/.test(p) ? p : p + '.';
}

// Humanoid statistics for a rig with legs, arms and a head; animal ones else.
function autoStats(clean) {
	const has = (w) => clean.some((n) => n.startsWith('Left ') && n.endsWith(w)) && clean.some((n) => n.startsWith('Right ') && n.endsWith(w));
	const legs = has('Thigh') || has('Upper Leg') || has('Hip') && has('Knee');
	const arms = has('Upper Arm') || has('Shoulder') || has('Forearm') || has('Elbow');
	const head = clean.some((n) => n === 'Head');
	const hands = has('Hand') || has('Wrist') || clean.some((n) => /Finger/.test(n));
	const hind = clean.some((n) => /\b(Hind|Back|Rear|Front|Fore)\b/.test(n) || /Hoof|Paw/.test(n));
	return legs && arms && head && hands && !hind ? 'mixamo' : 'truebones';
}

llm.plugin({
	name: 'unimate',
	async generate(config) {
		const start = llm.now();
		llm.print('\n=== UniMate ===');
		if (!config.model) throw new Error('model= names the UniMate checkpoint (.pt)');
		if (!config.text_model) throw new Error('text_model= names the flan-t5-base folder');
		const rig = config.rig ?? (config.rig_glb ? readRigGlb(config.rig_glb) : null);
		if (!rig) throw new Error('rig= (names, parents, positions) or rig_glb= (a .glb) is the skeleton to animate');

		// Skeleton first: its cleaned names decide the statistics.
		let stats = config.stats ?? 'auto';
		let sk = skeletonCondition(rig, { face: config.face ?? null, stats: stats === 'auto' ? 'objaverse' : stats });
		if (stats === 'auto') {
			stats = autoStats(sk.clean);
			sk = skeletonCondition(rig, { face: config.face ?? null, stats });
		}
		const facing = sk.facing ? `${sk.facing.right} / ${sk.facing.left} (${sk.facing.what})` : 'none found: as the rig stands';
		llm.print(`  ${sk.J} joints, ${stats} statistics, facing from ${facing}`);

		const caption = config.raw_prompt ? (config.prompt ?? '') : captionOf(config.prompt ?? '');
		llm.print(`  prompt: ${JSON.stringify(caption)}`);
		const t5 = new T5Encoder(config.text_model);
		const [words, ...nameStates] = t5.encode([caption, ...sk.clean]);
		const W = t5.dModel;
		t5.close();
		const names = new Float32Array(sk.J * W);
		nameStates.forEach((st, j) => names.set(meanPool(st, W), j * W));

		const ckpt = new UniMateCheckpoint(config.model);
		const net = new UniMateDenoiser(ckpt);
		ckpt.release();
		// The graph model reads the caption's mean, the full one its token states.
		const text = !caption ? null : net.captionKind === 'tokens' ? words : meanPool(words, W);
		net.prepare({ ...sk, names });
		const steps = config.steps ?? 20, seed = config.seed ?? 42, cfg = config.cfg ?? 3;
		const noise = llm.randomNormal(sk.J * FEAT * FRAMES, seed);
		const t0 = llm.now();
		let lastPrint = t0;
		const sample = await net.sample({
			noise, caption: text, steps, cfg,
			onStep: (done, total) => {
				if (config.progress || llm.now() - lastPrint > 2000 || done === total) {
					llm.print(`  step ${done}/${total}, ${llm.since(t0)}`);
					lastPrint = llm.now();
				}
			},
		});
		net.release();
		const clip = toRigFrame(decodeCanonical(denormalize(sample, sk), sk), sk);

		const result = {
			frames: FRAMES, fps: FPS, steps, seed, cfg, joints: sk.J, stats, prompt: caption, attention: net.variant,
			facing: sk.facing, seconds: 0,
		};
		if (config.glb) {
			writeMotionGlb((result.glb = config.glb), {
				name: config.name ?? config.prompt ?? 'UniMate', skeleton: clip.skeleton, fps: FPS, frames: FRAMES,
				rotations: clip.rotations, root: clip.root, rest: rig.positions[rig.parents.indexOf(-1)],
				generator: 'llm-runner unimate', extras: { facing: 'as-is' },
			});
		}
		if (config.profile) profileReport('unimate');
		result.seconds = (llm.now() - start) / 1000;
		llm.print(`=== done in ${llm.since(start)} ===`);
		return result;
	},
});
