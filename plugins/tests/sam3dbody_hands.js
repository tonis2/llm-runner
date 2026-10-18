// SAM 3D Body's hands against Meta's code (inference_type="full"), on the
// real weights, stage by stage.
//
//   python tools/sam3dbody/golden_hands.py ... --out build/sam3d-golden-hands      (see that file)
//   llm-runner tests/sam3dbody_hands.js model=build/sam-3d-body-dinov3 golden=build/sam3d-golden-hands
//       [linear=q8|f32] [only=boxes,hands,merge,pose]
//
//   boxes  the body pass's hand boxes in the image, each hand's crop (the
//          left one's cut from the mirrored image) and its pixels
//   hands  the hand decoder and heads on PyTorch's float32 features of each
//          crop, layer by layer
//   merge  from PyTorch's three passes: the checks and the keypoint prompts,
//          the prompted body pass on its features, the hands put in
//   pose   the whole image: our crops, backbone and decoders
import { llm, f32 } from '../lib/llm.js';
import { useMatrixCores } from '../lib/ops.js';
import { openWeights } from '../sam3dbody/weights.js';
import { boxToCrop, warpCrop, defaultCamera, rayFeatures, cliffCondition } from '../sam3dbody/crop.js';
import { BodyHeads } from '../sam3dbody/heads.js';
import { BodyDecoder } from '../sam3dbody/decoder.js';
import { Sam3dBody } from '../sam3dbody/pose.js';
import { handBoxes, handInputs, unmirrorLeft, checkHands, handPrompts, previousEstimate, mergeHands } from '../sam3dbody/hands.js';
import { MHR } from '../sam3dbody/mhr.js';
import { mhrPath } from '../sam3dbody/weights.js';

const cfg = globalThis.__llm_config;
const dir = cfg.golden ?? 'build/sam3d-golden-hands';
const model = cfg.model ?? 'build/sam-3d-body-dinov3';
const only = new Set(String(cfg.only ?? 'boxes,hands,merge,pose').split(','));
useMatrixCores(cfg.matrix_cores ?? true);
const floats = (name) => new Float32Array(llm.readBytes(`${dir}/${name}`).buffer);
const golden = JSON.parse(llm.readText(`${dir}/passes.json`));
const [bodyPass, leftPass, rightPass] = golden.passes;
let failures = 0;

function compare(label, got, ref, maxRel, minCos) {
	if (got.length !== ref.length) throw new Error(`${label}: ${got.length} values against ${ref.length}`);
	let err = 0, dot = 0, na = 0, nb = 0, sq = 0;
	for (let i = 0; i < ref.length; i++) {
		const d = got[i] - ref[i];
		err = Math.max(err, Math.abs(d));
		sq += d * d;
		dot += got[i] * ref[i]; na += got[i] ** 2; nb += ref[i] ** 2;
	}
	const rel = nb ? Math.sqrt(sq / nb) : Math.sqrt(sq), cos = na && nb ? dot / Math.sqrt(na * nb) : 1;
	const ok = rel < maxRel && cos > minCos;
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${label}: relative error ${rel.toExponential(2)}, cosine ${cos.toFixed(6)}, max |diff| ${err.toExponential(2)}`);
	if (!ok) failures++;
	return ok;
}
function points(label, got, ref, dims, bound, unit) {
	let worst = 0, sum = 0;
	const n = ref.length / dims;
	for (let i = 0; i < n; i++) {
		let d = 0;
		for (let k = 0; k < dims; k++) d += (got[i * dims + k] - ref[i * dims + k]) ** 2;
		worst = Math.max(worst, Math.sqrt(d));
		sum += Math.sqrt(d);
	}
	const ok = worst < bound;
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${label}: mean ${(sum / n).toExponential(2)} ${unit}, worst ${worst.toExponential(2)} ${unit}`);
	if (!ok) failures++;
}
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

// A pose as PyTorch recorded it, in the shapes heads.decode gives.
const asPose = (r) => ({
	...r,
	body_pose: Float64Array.from(r.body_pose), shape: Float64Array.from(r.shape), scale: Float64Array.from(r.scale), hand: Float64Array.from(r.hand),
	joint_global_rots: Array.from(r.joint_global_rots),
});

const img = llm.image.load(`${dir}/image.png`);
const cam = defaultCamera(img.width, img.height);
const W = img.width;
const bodyCrop = boxToCrop(golden.bbox);
const bodyView = { crop: bodyCrop, cam };
const batchCrop = (p) => ({ center: p.batch.bbox_center, scale: p.batch.bbox_scale, affine: p.batch.affine_trans.slice(0, 6) });
const mirrorBox = (b) => [W - b[2] - 1, b[1], W - b[0] - 1, b[3]];

if (only.has('boxes')) {
	const [l, r] = handBoxes(bodyPass.hand_box, bodyCrop);
	compare('left hand box (mirrored)', mirrorBox(l), leftPass.batch.bbox, 1e-6, 1 - 1e-12);
	compare('right hand box', r, rightPass.batch.bbox, 1e-6, 1 - 1e-12);
	for (const [name, pass, mirror] of [['left', leftPass, true], ['right', rightPass, false]]) {
		const crop = boxToCrop(pass.batch.bbox, { padding: 0.9 });
		const ref = batchCrop(pass);
		compare(`${name} crop centre, scale, warp`, [...crop.center, ...crop.scale, ...crop.affine], [...ref.center, ...ref.scale, ...ref.affine], 1e-6, 1 - 1e-12);
		const pixels = warpCrop(img, crop.affine, { mirror }), want = floats(`crop_${name === 'left' ? 'lhand' : 'rhand'}.bin`), n = 512 * 512;
		let off = 0, worst = 0;
		for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) {
			const d = Math.abs(pixels[i * 3 + c] - Math.round(want[c * n + i] * 255));
			if (d) off++;
			worst = Math.max(worst, d);
		}
		check(off === 0, `${name} crop pixels: ${off} of ${3 * n} differ from cv2.warpAffine's, by at most ${worst}`);
	}
}

const m = openWeights(model);
const needRig = only.has('hands') || only.has('merge');
const mhr = needRig ? new MHR(mhrPath(model)) : null;
const heads = mhr ? new BodyHeads(m, mhr) : null;
const handHeads = mhr ? new BodyHeads(m, mhr, { hand: true }) : null;
const jointRotation = m.floats('head_pose.joint_rotation');

function checkLayer(label, got, L) {
	compare(`${label} pose head (raw)`, got.pred_pose_raw, L.pred_pose_raw, 1e-4, 1 - 1e-8);
	compare(`${label} camera head`, got.pred_cam, L.pred_cam, 1e-4, 1 - 1e-8);
	compare(`${label} model parameters`, got.mhr_model_params, L.mhr_model_params, 1e-4, 1 - 1e-8);
	points(`${label} 3D keypoints`, got.pred_keypoints_3d, L.pred_keypoints_3d, 3, 1e-4, 'm');
	points(`${label} 2D keypoints`, got.pred_keypoints_2d, L.pred_keypoints_2d, 2, 1e-2, 'px');
}

async function decodeAll(dec, hds, features, view, rays, condition, layers, label, opts = {}) {
	const feat = f32(1280 * 1024, [1024, 1280]);
	feat.buffer.write(features);
	const t0 = llm.now();
	const last = await dec.run(feat, rays, condition, (token, l) => {
		compare(`${label} layer ${l} pose token`, token, layers[l].token, 1e-4, 1 - 1e-8);
		const pose = hds.decode(token, view);
		checkLayer(`${label} layer ${l}`, pose, layers[l]);
		return pose;
	}, opts);
	llm.print(`  ${label}: ${llm.since(t0)}`);
	feat.dispose();
	return last.pose;
}

if (only.has('hands')) {
	const dec = new BodyDecoder(m, { linear: cfg.decoder ?? 'f32', hand: true });
	await dec.load();
	for (const [name, pass, mirror] of [['left', leftPass, true], ['right', rightPass, false]]) {
		const input = handInputs(img, pass.batch.bbox, cam, { mirror });
		await decodeAll(dec, handHeads, floats(`features_${name === 'left' ? 'lhand' : 'rhand'}.bin`), input.view, input.rays, input.condition, pass.layers, `${name} hand`);
	}
	dec.dispose();
}

if (only.has('merge')) {
	const body = asPose(bodyPass.final), left = unmirrorLeft(asPose(leftPass.final), heads), right = asPose(rightPass.final);
	const crops = [boxToCrop(leftPass.batch.bbox, { padding: 0.9 }), boxToCrop(rightPass.batch.bbox, { padding: 0.9 })];
	const c = checkHands(body, left, right, crops, W, jointRotation);
	llm.print(`  checks [left, right]: wrist angles ${c.angle.map((v) => v.toFixed(3)).join(', ')}, boxes ${c.bigEnough}, inside ${c.inside}, near ${c.near}`);
	const prompts = handPrompts(body, c, bodyView);
	compare(`keypoint prompts (${prompts.length})`, prompts.flat(), golden.prompted.keypoint_prompt, 1e-6, 1 - 1e-12);
	const dec = new BodyDecoder(m, { linear: cfg.decoder ?? 'f32' });
	await dec.load();
	const prompted = await decodeAll(dec, heads, floats('features_body.bin'), bodyView, rayFeatures(bodyCrop.affine, cam), cliffCondition(bodyCrop, cam),
		golden.prompted.layers, 'prompted body', { prompts, prev: previousEstimate(body) });
	dec.dispose();
	// Merged from PyTorch's prompted pass, then from ours.
	const M = golden.merged;
	for (const [label, p] of [['merged (torch prompted pass)', asPose(golden.prompted.final)], ['merged (our prompted pass)', prompted]]) {
		const out = mergeHands(heads, p, left, right, c, bodyView, jointRotation);
		llm.print(`  ${label}: hands used ${out.hands_used}`);
		compare(`${label} body pose`, out.body_pose, M.body_pose, 1e-4, 1 - 1e-8);
		compare(`${label} hand pose`, out.hand, M.hand, 1e-4, 1 - 1e-8);
		compare(`${label} scale`, out.scale, M.scale, 1e-4, 1 - 1e-8);
		compare(`${label} shape`, out.shape, M.shape, 1e-4, 1 - 1e-8);
		compare(`${label} model parameters`, out.mhr_model_params, M.mhr_model_params, 1e-4, 1 - 1e-8);
		points(`${label} 3D keypoints`, out.pred_keypoints_3d, M.pred_keypoints_3d, 3, 1e-4, 'm');
		points(`${label} 2D keypoints`, out.pred_keypoints_2d, M.pred_keypoints_2d, 2, 1e-2, 'px');
	}
}

if (only.has('pose')) {
	const linear = cfg.linear ?? 'q8';
	const sam = new Sam3dBody(model, { linear, decoder: cfg.decoder ?? 'f32', hands: true });
	await sam.load();
	const t0 = llm.now();
	const pose = await sam.poseImage(img, { bbox: golden.bbox, quiet: true });
	llm.print(`  image to pose with hands (${linear} backbone): ${llm.since(t0)}, hands used ${pose.hands_used}`);
	const M = golden.merged;
	const bound = linear === 'f32' ? [0.5, 5e-3] : [4, 3e-2];
	points('pose: 2D keypoints', pose.pred_keypoints_2d, M.pred_keypoints_2d, 2, bound[0], 'px');
	points('pose: 2D hand keypoints', pose.pred_keypoints_2d.slice(21 * 2, 63 * 2), M.pred_keypoints_2d.slice(21 * 2, 63 * 2), 2, bound[0], 'px');
	points('pose: 3D keypoints', pose.pred_keypoints_3d, M.pred_keypoints_3d, 3, bound[1], 'm');
	sam.dispose();
}

heads?.dispose();
handHeads?.dispose();
mhr?.close();
m.close();
llm.print(failures ? `${failures} FAILED` : 'all passed');
