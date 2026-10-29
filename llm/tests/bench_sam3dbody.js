// Where a SAM 3D Body frame's time goes, with and without hands, on one image
// (posed once to warm up, then timed).
//
//   llm-runner tests/bench_sam3dbody.js model=build/sam-3d-body-dinov3 [image=...] [bbox=x1,y1,x2,y2] [hands=both|true|false]
//
// Each stage is timed by wrapping it: the image load, the crop inputs (the
// GPU warp, rays), each backbone pass, each decoder run (and in it the heads
// and rig on the host, the rest being the GPU layers and their read-backs),
// and the hand merge.
import { llm } from '/core/llm.js';
import { useMatrixCores } from '/core/ops.js';
import * as crop from '/plugins/sam3dbody/crop.js';
import { Sam3dBody } from '/plugins/sam3dbody/pose.js';

const cfg = globalThis.__llm_config;
const image = cfg.image ?? 'build/sam3d-golden-hands/image.png';
const bbox = String(cfg.bbox ?? '720,107,1530,1457').split(',').map(Number);
const modes = cfg.hands === undefined || cfg.hands === 'both' ? [false, true] : [cfg.hands === true || cfg.hands === 'true'];
useMatrixCores(cfg.matrix_cores ?? true);

const times = new Map();
const add = (label, ms) => times.set(label, (times.get(label) ?? 0) + ms);
// Wraps obj[name] (sync or async) to add its time under `label`.
function time(obj, name, label) {
	const f = obj[name];
	obj[name] = function (...args) {
		const t0 = llm.now();
		const r = f.apply(this, args);
		if (r && typeof r.then === 'function') return r.then((v) => { add(label, llm.now() - t0); return v; });
		add(label, llm.now() - t0);
		return r;
	};
}

const sam = new Sam3dBody(cfg.model ?? 'build/sam-3d-body-dinov3', { hands: modes.includes(true) });
await sam.load();
time(sam.encoder, 'encode', 'backbone');
time(sam.decoder, 'run', 'body decoder');
time(sam.heads, 'decode', '  of which heads + rig (host)');
if (sam.handDecoder) {
	time(sam.handDecoder, 'run', 'hand decoder');
	time(sam.handHeads, 'decode', '  of which hand heads + rig (host)');
}
let t0 = llm.now();
const img = llm.image.load(image);
llm.print(`  image load: ${(llm.now() - t0).toFixed(0)} ms (${img.width}x${img.height})`);
// The crop helpers are module functions: time them through the objects that
// call them where we can, and directly here, as poseImage runs them (the
// frame's bytes up, the warp and normalisation on the GPU, waited for).
const c = crop.boxToCrop(bbox);
t0 = llm.now();
const bytes = crop.uploadImage(img);
crop.warpCropGpu(bytes, img, c.affine).dispose();
bytes.dispose();
llm.print(`  upload and warp alone: ${(llm.now() - t0).toFixed(1)} ms`);
t0 = llm.now();
crop.rayFeatures(c.affine, crop.defaultCamera(img.width, img.height));
llm.print(`  rayFeatures alone: ${(llm.now() - t0).toFixed(1)} ms`);

for (const withHands of modes) {
	await sam.poseImage(img, { bbox, hands: withHands, quiet: true });
	times.clear();
	t0 = llm.now();
	const pose = await sam.poseImage(img, { bbox, hands: withHands, quiet: true });
	const total = llm.now() - t0;
	llm.print(`\n${withHands ? 'with hands' : 'body only'}: ${(total / 1000).toFixed(2)} s${pose.hands_used ? `, hands used ${pose.hands_used}` : ''}`);
	let accounted = 0;
	const order = ['backbone', 'body decoder', '  of which heads + rig (host)', 'hand decoder', '  of which hand heads + rig (host)'];
	for (const label of order.filter((l) => times.has(l))) {
		const ms = times.get(label);
		llm.print(`  ${label.padEnd(36)} ${(ms / 1000).toFixed(3)} s  ${((100 * ms) / total).toFixed(1)}%`);
		if (!label.startsWith(' ')) accounted += ms;
	}
	llm.print(`  ${'rest (crops, merge, glue)'.padEnd(36)} ${((total - accounted) / 1000).toFixed(3)} s  ${((100 * (total - accounted)) / total).toFixed(1)}%`);
}
sam.dispose();
