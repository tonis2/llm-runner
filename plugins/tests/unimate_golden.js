// `llm-runner tests/unimate_golden.js ckpt=<.pt> golden=<dir from unimate_golden.py>`:
// the UniMate denoiser, one pass, against the reference model's numbers.
import { llm } from '../lib/llm.js';
import { UniMateCheckpoint } from '../unimate/checkpoint.js';
import { UniMateDenoiser } from '../unimate/denoiser.js';
import { skeletonCondition } from '../unimate/skeleton.js';
import { denormalize, decodeCanonical, positionsOf, toRigFrame } from '../unimate/motion.js';

const cfg = globalThis.__llm_config ?? {};
const g = cfg.golden;
const f32 = (n) => new Float32Array(llm.readBytes(`${g}/${n}`).buffer.slice(0));
const i32 = (n) => new Int32Array(llm.readBytes(`${g}/${n}`).buffer.slice(0));
const meta = JSON.parse(llm.readText(`${g}/names.json`));
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
function compare(label, got, want, tol = 2e-3) {
	let maxDiff = 0, maxAbs = 0, dot = 0, na = 0, nb = 0;
	for (let i = 0; i < want.length; i++) {
		maxDiff = Math.max(maxDiff, Math.abs(got[i] - want[i]));
		maxAbs = Math.max(maxAbs, Math.abs(want[i]));
		dot += got[i] * want[i]; na += got[i] * got[i]; nb += want[i] * want[i];
	}
	check(maxDiff <= tol * Math.max(1, maxAbs), `${label}: max |diff| ${maxDiff.toExponential(2)} (max |x| ${maxAbs.toFixed(3)}), cosine ${(dot / Math.sqrt(na * nb)).toFixed(7)}`);
}

// The rig to the model's input, from the rig as it was read.
const sk = skeletonCondition({ names: meta.input_names, parents: meta.input_parents, positions: meta.input_positions },
	{ face: meta.face, stats: meta.stats });
check(JSON.stringify(sk.names) === JSON.stringify(meta.raw), 'joint order (breadth-first) matches');
check(JSON.stringify(sk.clean) === JSON.stringify(meta.clean), 'cleaned names match');
check(JSON.stringify(Array.from(sk.parents)) === JSON.stringify(meta.parents), 'parents match');
check(Math.abs(sk.scale - meta.scale_factor) < 1e-9 * meta.scale_factor, `scale ${sk.scale} vs ${meta.scale_factor}`);
compare('normalised rest pose', sk.tpos, f32('cond_tpos.f32'), 1e-5);
compare('parent rows', sk.tposParents, f32('cond_tpos_parents.f32'), 1e-5);
check(JSON.stringify(Array.from(sk.depths)) === JSON.stringify(Array.from(i32('cond_depths.i32'))), 'depths match');
check(JSON.stringify(Array.from(sk.graphDist)) === JSON.stringify(Array.from(i32('cond_graph_dist.i32'))), 'graph distances match');
check(JSON.stringify(Array.from(sk.relations)) === JSON.stringify(Array.from(i32('cond_relations.i32'))), 'relations match');
{
	// Eigenvectors are unique up to sign (and within repeated eigenvalues).
	const want = f32('cond_spectral.f32'), K = 8, J0 = sk.J;
	// Frequencies whose eigenvalue repeats (a hand's identical finger chains)
	// have no one basis: LAPACK's pick cannot be matched, only the subspace.
	const cos = [];
	for (let k = 0; k < K; k++) {
		let dot = 0;
		for (let j = 0; j < J0; j++) dot += want[j * K + k] * sk.spectral[j * K + k];
		cos.push(Math.abs(dot));
	}
	llm.print(`  eigenvector |cos| per frequency: ${cos.map((c) => c.toFixed(4)).join(' ')}`);
	const distinct = sk.spectralDistinct ?? K;
	check(cos.slice(0, distinct).every((c) => c > 1 - 1e-6), `the ${distinct} non-repeated eigenvectors match up to sign`);
}
{
	// Decoding: the reference's features, forward kinematics against its own.
	const feats = f32('feats.f32');
	const clip = decodeCanonical(feats, sk);
	const pos = positionsOf(clip, sk).flat(2);
	compare('decoded joint positions (canonical frame)', pos, f32('fk.f32'), 1e-4);
	const back = toRigFrame({ rotations: [Array.from({ length: sk.J }, () => [1, 0, 0, 0])], root: [sk.rest[0]] }, sk);
	// The rest pose, taken back, is the rig's own.
	let err = 0;
	const g2 = [], names = back.skeleton.names;
	for (let j = 0; j < sk.J; j++) {
		const p = sk.parents[j] === -1 ? Array.from(back.root.subarray(0, 3)) : g2[sk.parents[j]].map((c, k) => c + back.skeleton.offsets[j][k]);
		g2.push(p);
		const want = meta.input_positions[meta.input_names.indexOf(names[j])];
		err = Math.max(err, ...p.map((c, k) => Math.abs(c - want[k])));
	}
	check(err < 1e-5, `rest pose in the rig's frame matches the rig (max ${err.toExponential(2)})`);
}

const ckpt = new UniMateCheckpoint(cfg.ckpt);
const net = new UniMateDenoiser(ckpt);
const J = meta.J;
const t0 = llm.now();
net.prepare({
	J, tpos: f32('cond_tpos.f32'), tposParents: f32('cond_tpos_parents.f32'), spectral: f32('cond_spectral.f32'),
	depths: i32('cond_depths.i32'), graphDist: i32('cond_graph_dist.i32'), relations: i32('cond_relations.i32'),
	names: f32('names_emb.f32'),
});
llm.print(`  prepared ${J} joints in ${llm.since(t0)}`);
const noise = f32('noise.f32');
// The pooled caption (graph attention) or its T5 states (full attention).
const caption = f32(net.captionKind === 'tokens' ? 'caption_tokens.f32' : 'caption.f32');
const t1 = llm.now();
const vc = net.velocity(noise, 0, caption);
llm.print(`  one pass in ${llm.since(t1)}`);
compare('velocity, caption, t=0', vc, f32('v_cond.f32'));
compare('velocity, empty, t=0', net.velocity(noise, 0, null), f32('v_uncond.f32'));
compare('velocity, caption, t=0.5', net.velocity(noise, 0.5, caption), f32('v_cond_t05.f32'));
const t2 = llm.now();
const sample = await net.sample({ noise, caption, steps: 20, cfg: 3 });
llm.print(`  20 steps in ${llm.since(t2)}`);
compare('20-step Euler sample, cfg 3', sample, f32('sample.f32'), 5e-3);
{
	// The same run with this port's eigenvectors instead of LAPACK's.
	net.prepare({
		J, tpos: sk.tpos, tposParents: sk.tposParents, spectral: sk.spectral,
		depths: sk.depths, graphDist: sk.graphDist, relations: sk.relations, names: f32('names_emb.f32'),
	});
	const own = await net.sample({ noise, caption, steps: 20, cfg: 3 });
	const ref = f32('sample.f32');
	let d = 0, n2 = 0;
	for (let i = 0; i < ref.length; i++) { d += (own[i] - ref[i]) ** 2; n2 += ref[i] ** 2; }
	const clip = decodeCanonical(denormalize(own, sk), sk);
	const p = positionsOf(clip, sk).flat(2), fk = f32('fk.f32');
	let pe = 0;
	for (let i = 0; i < fk.length; i++) pe = Math.max(pe, Math.abs(p[i] - fk[i]));
	llm.print(`  with this port's eigenvectors: sample differs by ${(Math.sqrt(d / n2) * 100).toFixed(2)}% (rms), joints move up to ${pe.toFixed(3)} (skeleton 2 across)`);
}
llm.print(failures ? `${failures} FAILED` : 'all passed');
