// The creature fit on tracks made before: its gradients against finite
// differences, then the whole fit, how well it lands on the tracks, and the
// time it takes. No GPU.
//
//   llm-runner tests/creature_fit.js rig=rig.glb tracks=tracks.json request=request.json \
//       [iterations=1500] [pitch=8] [check=true] [out=fit.json] [save=dir]
//
// `tracks` is track.js's output (tests/creature_track.js out=), `request` the
// JSON with its `markers`; `out` writes the fit's projections beside the
// tracks (image pixels), for drawing; `save` the clip's files (clip.glb,
// clip.bvh, clip.json, animated.glb). Every output is read back and must
// put the joints where the fit did.
import { llm } from '../lib/llm.js';
import { writeText } from '../lib/graph/plugins.js';
import { readRig, parseGlb, qmul, qrotate } from '../creature/rig.js';
import { RigFit, observations, fitRig } from '../creature/fit.js';
import { fittedMotion, clipGlb, clipBvh, clipJson, animatedGlb } from '../creature/output.js';

const cfg = globalThis.__llm_config;
const rig = readRig(cfg.rig);
const tracks = JSON.parse(llm.readText(cfg.tracks));
const request = cfg.request ? JSON.parse(llm.readText(cfg.request)) : {};
const markers = request.markers ?? {};
let failures = 0;

// A small problem at a random state; every parameter group's gradient against
// central differences at a few entries. Once with the one view, once with it
// twice (the second through another camera, its markers shared).
if (cfg.check ?? true) {
	const marked = [];
	const obs = observations(rig, tracks, markers, { marked });
	const T = 6, P = obs.points.length, H = obs.H;
	const small = { ...obs, T, x: obs.x.slice(0, T * P * H), y: obs.y.slice(0, T * P * H), w: obs.w.slice(0, T * P * H) };
	let seed = 1;
	const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
	let worst = 0;
	for (const views of [[small], [small, small]]) {
		const f = new RigFit(rig, views, { k0: 1200, pitch: views.map((_, v) => 8 + 4 * v) });
		for (let i = 0; i < f.theta.length; i++) f.theta[i] = 0.4 * rnd();
		for (let i = 0; i < f.root.length; i++) f.root[i] += 0.02 * rnd();
		for (let i = 0; i < f.marks.length; i++) f.marks[i] = 0.03 * rnd();
		for (let i = 0; i < f.logS.length; i++) { f.logS[i] += 0.05 * rnd(); f.shift[i * 2] = 600 + 30 * rnd(); f.shift[i * 2 + 1] = -500 + 30 * rnd(); }
		for (let v = 0; v < views.length; v++) { f.cam[2 * v] = 1.2 - 1.6 * v; f.cam[2 * v + 1] = 0.2 + 0.1 * v; }
		const tau = 40;
		f.evaluate(tau);
		const grads = Object.fromEntries(Object.entries(f.g).map(([k, v]) => [k, Float64Array.from(v)]));
		const params = { theta: f.theta, root: f.root, marks: f.marks, logS: f.logS, shift: f.shift, cam: f.cam };
		for (const [name, arr] of Object.entries(params)) {
			const eps = name === 'shift' ? 1e-3 : 1e-6;
			for (let n = 0; n < Math.min(8, arr.length); n++) {
				const i = Math.floor(((rnd() + 1) / 2) * arr.length) % arr.length;
				const keep = arr[i];
				arr[i] = keep + eps;
				const up = f.evaluate(tau);
				arr[i] = keep - eps;
				const down = f.evaluate(tau);
				arr[i] = keep;
				const num = (up - down) / (2 * eps), ana = grads[name][i];
				const rel = Math.abs(num - ana) / Math.max(1e-6, Math.abs(num) + Math.abs(ana));
				if (rel > worst) worst = rel;
				if (rel > 1e-4 && Math.abs(num - ana) > 1e-6) llm.print(`  ${views.length} view(s), ${name}[${i}]: analytic ${ana.toExponential(4)}, numeric ${num.toExponential(4)} (rel ${rel.toExponential(2)})`);
			}
		}
	}
	llm.print(`  gradient check (one view and two): worst relative difference ${worst.toExponential(2)}`);
	if (!(worst < 1e-4)) {
		llm.print('FAIL: gradients');
		failures++;
	}
}

const t0 = llm.now();
const fit = fitRig(rig, tracks, { markers, pitch: Number(cfg.pitch ?? request.pitch ?? 8), iterations: Number(cfg.iterations ?? 1500) });
llm.print(`  ${tracks.frames} frames, ${rig.names.length} joints in ${llm.since(t0)}`);
const r = fit.report.reprojection_px;
if (!(r.median <= Number(cfg.max_median ?? 8))) {
	llm.print(`FAIL: reprojection median ${r.median}px`);
	failures++;
}
if (cfg.out) {
	writeText(cfg.out, JSON.stringify({
		points: tracks.points,
		fit: fit.projections.map((row) => row.map((q) => [Math.round(q[0] * 10) / 10, Math.round(-q[1] * 10) / 10])),
		joints: rig.names, parents: rig.parents,
		skeleton: fit.problem.jointProjections().map((row) => row.map((q) => [Math.round(q[0] * 10) / 10, Math.round(-q[1] * 10) / 10])),
		report: fit.report,
	}));
}
// The outputs, read back.
const motion = fittedMotion(fit, rig, { fps: Number(request.fps ?? 24) });
llm.print(`  travel ${motion.travel.toFixed(3)} rig units`);
const J = rig.names.length, T = motion.frames;
// Where the motion puts every joint: [T][J][3].
const truth = [];
for (let t = 0; t < T; t++) {
	const G = [], P = [];
	for (let j = 0; j < J; j++) {
		const q = Array.from(motion.rotations.subarray((t * J + j) * 4, (t * J + j) * 4 + 4)), par = rig.parents[j];
		if (par < 0) { G.push(q); P.push(Array.from(motion.root.subarray(t * 3, t * 3 + 3))); continue; }
		const o = qrotate(G[par], motion.skeleton.offsets[j]);
		G.push(qmul(G[par], q));
		P.push([P[par][0] + o[0], P[par][1] + o[1], P[par][2] + o[2]]);
	}
	truth.push(P);
}
// The fit's own joints (before floor and travel) differ from those by the root's shift only.
{
	fit.problem.forward();
	let worst = 0;
	for (let t = 0; t < T; t++) {
		const shift = [0, 1, 2].map((d) => motion.root[t * 3 + d] - fit.problem.root[t * 3 + d]);
		for (let j = 0; j < J; j++) for (let d = 0; d < 3; d++) worst = Math.max(worst, Math.abs(truth[t][j][d] - shift[d] - fit.problem.pos[(t * J + j) * 3 + d]));
	}
	check('motion against the fit', worst, 1e-4 * rig.height);
}
function check(what, worst, limit) {
	llm.print(`  ${what}: worst ${worst.toExponential(2)} (limit ${limit.toExponential(1)})`);
	if (!(worst <= limit)) {
		llm.print(`FAIL: ${what}`);
		failures++;
	}
}

// BVH: parse, forward kinematics with Z-X-Y Euler angles, against the motion.
const bvh = clipBvh(motion, 100);
{
	const tok = bvh.split(/\s+/).filter(Boolean);
	const joints = [], stack = [];
	let i = 0;
	while (tok[i] !== 'MOTION') {
		const w = tok[i];
		if (w === 'ROOT' || w === 'JOINT') { joints.push({ name: tok[i + 1], parent: stack.length ? stack[stack.length - 1] : -1, end: false }); i += 2; }
		else if (w === 'End') { joints.push({ name: 'end', parent: stack[stack.length - 1], end: true, channels: [] }); i += 2; }
		else if (w === '{') { stack.push(joints.length - 1); i++; }
		else if (w === '}') { stack.pop(); i++; }
		else if (w === 'OFFSET') { joints[joints.length - 1].offset = tok.slice(i + 1, i + 4).map(Number); i += 4; }
		else if (w === 'CHANNELS') { const n = Number(tok[i + 1]); joints[joints.length - 1].channels = tok.slice(i + 2, i + 2 + n); i += 2 + n; }
		else i++;
	}
	const frames = Number(tok[i + 2]), values = tok.slice(i + 6).map(Number), width = values.length / frames;
	const axis = { X: [1, 0, 0], Y: [0, 1, 0], Z: [0, 0, 1] };
	const aa = (a, deg) => { const h = (deg * Math.PI) / 360, s = Math.sin(h); return [a[0] * s, a[1] * s, a[2] * s, Math.cos(h)]; };
	let worst = 0;
	for (let t = 0; t < frames; t++) {
		let k = t * width;
		const G = [], P = [];
		for (const jt of joints) {
			let q = [0, 0, 0, 1];
			const off = jt.offset.slice();
			for (const ch of jt.channels) {
				const v = values[k++];
				if (ch.endsWith('position')) off['XYZ'.indexOf(ch[0])] = v;
				else q = qmul(q, aa(axis[ch[0]], v));
			}
			if (jt.parent < 0) { G.push(q); P.push(off); continue; }
			const o = qrotate(G[jt.parent], off);
			G.push(qmul(G[jt.parent], q));
			P.push([P[jt.parent][0] + o[0], P[jt.parent][1] + o[1], P[jt.parent][2] + o[2]]);
		}
		const named = joints.map((jt, n) => [jt, n]).filter(([jt]) => !jt.end);
		for (const [jt, n] of named) {
			const j = rig.names.indexOf(jt.name);
			for (let d = 0; d < 3; d++) worst = Math.max(worst, Math.abs(P[n][d] / 100 - truth[t][j][d]));
		}
	}
	check('BVH round trip', worst, 5e-3 * rig.height);
}

// The clip .glb: the rig's names and parents.
const glb = clipGlb(motion, request.name ?? 'clip');
{
	const { gltf } = parseGlb(glb);
	const nodes = gltf.skins[0].joints.map((n) => gltf.nodes[n]);
	const names = nodes.map((n) => n.name);
	const parentOf = new Map();
	gltf.nodes.forEach((n, i) => (n.children ?? []).forEach((c) => parentOf.set(c, i)));
	const same = names.every((n, j) => n === rig.names[j] && (rig.parents[j] < 0 ? !parentOf.has(gltf.skins[0].joints[j]) : parentOf.get(gltf.skins[0].joints[j]) === gltf.skins[0].joints[rig.parents[j]]));
	check('clip .glb skeleton', same ? 0 : 1, 0);
	if (gltf.extras?.facing !== 'as-is') { llm.print('FAIL: clip .glb extras.facing'); failures++; }
}

// The animated .glb: its node animation through its own node tree.
let animated = null;
if (rig.glb) {
	animated = animatedGlb(motion, rig, request.name ?? 'clip');
	const { gltf, bin } = parseGlb(animated);
	const anim = gltf.animations[gltf.animations.length - 1];
	const read = (a) => { const acc = gltf.accessors[a], v = gltf.bufferViews[acc.bufferView]; const n = { SCALAR: 1, VEC3: 3, VEC4: 4 }[acc.type]; return new Float32Array(bin.buffer.slice(bin.byteOffset + v.byteOffset, bin.byteOffset + v.byteOffset + acc.count * n * 4)); };
	const tracks = new Map();
	for (const c of anim.channels) tracks.set(`${c.target.node}/${c.target.path}`, read(anim.samplers[c.sampler].output));
	const parentOf = new Map();
	gltf.nodes.forEach((n, i) => (n.children ?? []).forEach((c) => parentOf.set(c, i)));
	let worst = 0;
	for (let t = 0; t < T; t += 7) {
		const world = new Map();
		const worldOf = (i) => {
			if (world.has(i)) return world.get(i);
			const n = gltf.nodes[i];
			const r = tracks.has(`${i}/rotation`) ? Array.from(tracks.get(`${i}/rotation`).subarray(t * 4, t * 4 + 4)) : (n.rotation ?? [0, 0, 0, 1]);
			const tr = tracks.has(`${i}/translation`) ? Array.from(tracks.get(`${i}/translation`).subarray(t * 3, t * 3 + 3)) : (n.translation ?? [0, 0, 0]);
			let w;
			if (parentOf.has(i)) { const p = worldOf(parentOf.get(i)); const o = qrotate(p.r, tr); w = { p: [p.p[0] + o[0], p.p[1] + o[1], p.p[2] + o[2]], r: qmul(p.r, r) }; }
			else w = { p: tr, r };
			world.set(i, w);
			return w;
		};
		rig.glb.nodes.forEach((node, j) => { const p = worldOf(node).p; for (let d = 0; d < 3; d++) worst = Math.max(worst, Math.abs(p[d] - truth[t][j][d])); });
	}
	check('animated .glb node animation', worst, 1e-4 * rig.height);
}
if (cfg.save) {
	llm.makeDir(cfg.save);
	llm.writeBytes(`${cfg.save}/clip.glb`, glb);
	writeText(`${cfg.save}/clip.bvh`, bvh);
	writeText(`${cfg.save}/clip.json`, JSON.stringify(clipJson(motion, request.name ?? 'clip', rig.height)));
	if (animated) llm.writeBytes(`${cfg.save}/animated.glb`, animated);
	llm.print(`  saved ${cfg.save}/{clip.glb,clip.bvh,clip.json${animated ? ',animated.glb' : ''}}`);
}
llm.print(failures ? `${failures} failed` : 'ok');
