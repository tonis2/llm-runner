// SAM 3D Body against Meta's code, on the real weights, stage by stage.
//
//   python tools/sam3dbody/golden.py ... --out build/sam3d-golden      (see that file)
//   llm-runner tests/sam3dbody_golden.js model=build/sam-3d-body-dinov3 golden=build/sam3d-golden
//       [linear=q8|f32] [decoder=f32|q8] [only=crop,backbone,mhr,heads,decoder,pose]
//
//   crop      the box, the warp and the normalised crop golden.py fed the model
//   backbone  DINOv3: the tokens after block 0 and the [1280, 32, 32] features
//             against PyTorch's (backbone widened to float32), and the features
//             against the bf16 run SAM 3D Body itself does; Q8_0 weights (the
//             default) have looser bounds than float32's
//   mhr       the rig alone: each decoder layer's model parameters posed, the
//             joints and keypoints against PyTorch's
//   heads     each layer's pose token through the heads, rig and camera
//   decoder   the decoder on PyTorch's float32 features: its image tokens, the
//             tokens after layer 0, every layer's pose token and prediction
//   pose      the whole image: our crop, our backbone, our decoder
import { llm, f32 } from '/core/llm.js';
import { useMatrixCores } from '/core/ops.js';
import { DinoV3 } from '/core/models/dinov3.js';
import { openWeights, mhrPath } from '/plugins/sam3dbody/weights.js';
import { boxToCrop, warpCrop, normalize, defaultCamera, rayFeatures, cliffCondition } from '/plugins/sam3dbody/crop.js';
import { MHR } from '/plugins/sam3dbody/mhr.js';
import { BodyHeads } from '/plugins/sam3dbody/heads.js';
import { BodyDecoder } from '/plugins/sam3dbody/decoder.js';

const cfg = globalThis.__llm_config;
const dir = cfg.golden ?? 'build/sam3d-golden';
const linear = cfg.linear ?? 'q8';
const only = new Set(String(cfg.only ?? 'crop,backbone,mhr,heads,decoder,pose').split(','));
useMatrixCores(cfg.matrix_cores ?? true);
const floats = (name) => new Float32Array(llm.readBytes(`${dir}/${name}`).buffer);
const golden = JSON.parse(llm.readText(`${dir}/golden.json`));
const layers = JSON.parse(llm.readText(`${dir}/layers.json`));
let failures = 0;

function compare(label, got, ref, maxRel, minCos) {
	if (got.length !== ref.length) throw new Error(`${label}: ${got.length} values against ${ref.length}`);
	let err = 0, scale = 0, dot = 0, na = 0, nb = 0, sq = 0;
	for (let i = 0; i < ref.length; i++) {
		const d = got[i] - ref[i];
		err = Math.max(err, Math.abs(d));
		scale = Math.max(scale, Math.abs(ref[i]));
		sq += d * d;
		dot += got[i] * ref[i]; na += got[i] ** 2; nb += ref[i] ** 2;
	}
	const rel = nb ? Math.sqrt(sq / nb) : Math.sqrt(sq), cos = na && nb ? dot / Math.sqrt(na * nb) : 1;
	const ok = rel < maxRel && cos > minCos;
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${label}: relative error ${rel.toExponential(2)}, cosine ${cos.toFixed(6)}, max |diff| ${err.toExponential(2)} (largest ${scale.toFixed(2)})`);
	if (!ok) failures++;
	return ok;
}
// Largest distance between matching points, against a bound.
function points(label, got, ref, dims, bound, unit) {
	let worst = 0, sum = 0;
	const n = ref.length / dims;
	for (let i = 0; i < n; i++) {
		let d = 0;
		for (let k = 0; k < dims; k++) d += (got[i * dims + k] - ref[i * dims + k]) ** 2;
		d = Math.sqrt(d);
		worst = Math.max(worst, d);
		sum += d;
	}
	const ok = worst < bound;
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${label}: mean ${(sum / n).toExponential(2)} ${unit}, worst ${worst.toExponential(2)} ${unit}`);
	if (!ok) failures++;
}
const flipYZ = (a) => { const o = Float64Array.from(a); for (let i = 0; i < o.length; i += 3) { o[i + 1] = -o[i + 1]; o[i + 2] = -o[i + 2]; } return o; };

const b = golden.batch;
// The image as OpenCV decoded it: JPEG decoders round differently.
const img = llm.image.load(llm.exists(`${dir}/image.png`) ? `${dir}/image.png` : golden.image);
const crop = boxToCrop(golden.bbox);
const cam = defaultCamera(img.width, img.height);
const view = { crop, cam };

if (only.has('crop')) {
	llm.print(`crop: ${img.width}x${img.height}, box ${golden.bbox.map((v) => v.toFixed(1)).join(' ')}`);
	compare('box centre and scale', [...crop.center, ...crop.scale], [...b.bbox_center, ...b.bbox_scale], 1e-7, 0.9999999);
	compare('warp matrix', crop.affine, b.affine_trans, 1e-6, 0.9999999);
	compare('camera', [cam[0], cam[2], cam[1], cam[3]], [b.cam_int[0], b.cam_int[2], b.cam_int[4], b.cam_int[5]], 1e-7, 0.9999999);
	const pixels = warpCrop(img, crop.affine);
	const ref = floats('crop.bin'), n = 512 * 512;
	let off = 0, worst = 0;
	for (let c = 0; c < 3; c++) {
		for (let i = 0; i < n; i++) {
			const d = Math.abs(pixels[i * 3 + c] - Math.round(ref[c * n + i] * 255));
			if (d) off++;
			worst = Math.max(worst, d);
		}
	}
	const ok = off === 0;
	llm.print(`${ok ? 'ok  ' : 'FAIL'} crop pixels: ${off} of ${3 * n} values differ from cv2.warpAffine's, by at most ${worst}`);
	if (!ok) failures++;
	compare('normalised crop', normalize(pixels), floats('input.bin'), 2e-2, 0.9999);
}

const m = openWeights(cfg.model ?? 'build/sam-3d-body-dinov3');
const mhr = (only.has('mhr') || only.has('heads') || only.has('decoder') || only.has('pose')) ? new MHR(mhrPath(cfg.model ?? 'build/sam-3d-body-dinov3')) : null;
const heads = mhr ? new BodyHeads(m, mhr) : null;

if (only.has('mhr')) {
	for (const [l, L] of layers.entries()) {
		const t0 = llm.now();
		const r = heads.pose(L.mhr_model_params, L.shape);
		const ms = llm.now() - t0;
		points(`layer ${l} MHR joints (${ms.toFixed(0)} ms)`, flipYZ(r.rig.joints), L.pred_joint_coords, 3, 1e-5, 'm');
		points(`layer ${l} MHR keypoints`, flipYZ(r.keypoints), L.pred_keypoints_3d, 3, 1e-5, 'm');
	}
	const L = layers[layers.length - 1];
	const t0 = llm.now();
	const r = heads.pose(L.mhr_model_params, L.shape, { allVertices: true });
	points(`final mesh, ${mhr.vertexCount} vertices (${llm.since(t0)})`, flipYZ(r.vertices), floats('vertices.bin'), 3, 1e-5, 'm');
}

// A layer's decoded pose against PyTorch's. `loose` is for Q8_0 decoder
// weights, about 2e-3 off float32 in every token.
function checkPose(label, got, L, loose = false) {
	const v = loose ? [1e-2, 0.9999] : [1e-5, 1 - 1e-9];
	compare(`${label} pose head (raw)`, got.pred_pose_raw, L.pred_pose_raw, ...v);
	compare(`${label} camera head`, got.pred_cam, L.pred_cam, ...v);
	compare(`${label} model parameters`, got.mhr_model_params, L.mhr_model_params, ...v);
	points(`${label} 3D keypoints`, got.pred_keypoints_3d, L.pred_keypoints_3d, 3, loose ? 3e-3 : 1e-5, 'm');
	points(`${label} 2D keypoints`, got.pred_keypoints_2d, L.pred_keypoints_2d, 2, loose ? 0.5 : 1e-3, 'px');
	compare(`${label} camera translation`, got.pred_cam_t, L.pred_cam_t, ...(loose ? v : [1e-6, 1 - 1e-9]));
}

if (only.has('heads')) {
	for (const [l, L] of layers.entries()) checkPose(`layer ${l}`, heads.decode(Float32Array.from(L.token), view), L);
}

let enc = null;
if (only.has('backbone') || only.has('pose')) {
	enc = new DinoV3(m, { linear });
	await enc.load();
}
const pixels = floats('input.bin');
if (only.has('backbone')) {
	// Q8_0 measures 2.4e-2 after block 0, against 1e-2 at the output.
	const bound = linear === 'f32' ? [1e-4, 0.99999999] : [4e-2, 0.9995];
	const tokens = await enc.encode(pixels, 512, 512, { stopAfter: 0 });
	compare(`block 0 tokens (${linear})`, new Float32Array(tokens.buffer.readBytes().buffer), floats('block0.bin'), ...bound);
	tokens.dispose();
	const t0 = llm.now();
	const feat = await enc.encode(pixels, 512, 512);
	llm.print(`  backbone pass: ${llm.since(t0)}`);
	const got = new Float32Array(feat.buffer.readBytes().buffer);
	feat.dispose();
	compare(`features against float32 torch (${linear})`, got, floats('features.bin'), ...(linear === 'f32' ? [1e-3, 0.999999] : [5e-2, 0.999]));
	compare(`features against SAM 3D Body's bf16 run (${linear})`, got, floats('features_bf16.bin'), 1e-1, 0.995);
}

let dec = null;
if (only.has('decoder') || only.has('pose')) {
	dec = new BodyDecoder(m, { linear: cfg.decoder ?? 'f32' });
	await dec.load();
}
const rays = rayFeatures(crop.affine, cam);
const condition = cliffCondition(crop, cam);
const q8Decoder = (cfg.decoder ?? 'f32') === 'q8';
const tokenBound = q8Decoder ? [1e-2, 0.9999] : [1e-4, 1 - 1e-8];

if (only.has('decoder')) {
	const feat = f32(1280 * 1024, [1024, 1280]);
	feat.buffer.write(floats('features.bin'));
	const ctx = f32(1024 * 1280);
	const first = await dec.run(feat, rays, condition, null, { stopAfter: 0, contextOut: ctx });
	// The context is token-major; PyTorch's is channel-first.
	const c = new Float32Array(ctx.buffer.readBytes().buffer), cf = new Float32Array(c.length);
	for (let t = 0; t < 1024; t++) for (let ch = 0; ch < 1280; ch++) cf[ch * 1024 + t] = c[t * 1280 + ch];
	compare('decoder image tokens (features, no-mask, rays, conv, norm)', cf, floats('context.bin'), 1e-4, 1 - 1e-8);
	compare('decoder tokens after layer 0', new Float32Array(first.tokens.buffer.readBytes().buffer), floats('layer0_tokens.bin'), ...tokenBound);
	first.tokens.dispose();
	ctx.dispose();
	const t0 = llm.now();
	const last = await dec.run(feat, rays, condition, (token, l) => {
		compare(`layer ${l} pose token`, token, layers[l].token, ...tokenBound);
		return heads.decode(token, view);
	});
	llm.print(`  decoder, 6 layers with the heads and rig between: ${llm.since(t0)}`);
	feat.dispose();
	checkPose('decoded', last.pose, layers[layers.length - 1], q8Decoder);
	const hb = heads.handBoxes(last.handTokens);
	compare('hand boxes', hb.boxes, layers[layers.length - 1].hand_box, ...tokenBound);
	compare('hand logits', hb.logits, layers[layers.length - 1].hand_logits, ...tokenBound);
}

if (only.has('pose')) {
	const t0 = llm.now();
	const input = normalize(warpCrop(img, crop.affine));
	const feat = await enc.encode(input, 512, 512);
	const last = await dec.run(feat, rays, condition, (token) => heads.decode(token, view));
	feat.dispose();
	llm.print(`  image to pose (${linear} backbone): ${llm.since(t0)}`);
	const L = layers[layers.length - 1];
	// Against the float32 reference; q8 is about as far from it as bf16 is.
	points('pose: 2D keypoints', last.pose.pred_keypoints_2d, L.pred_keypoints_2d, 2, linear === 'f32' ? 0.5 : 4, 'px');
	points('pose: 3D keypoints', last.pose.pred_keypoints_3d, L.pred_keypoints_3d, 3, linear === 'f32' ? 5e-3 : 3e-2, 'm');
	points('pose: camera translation', last.pose.pred_cam_t, L.pred_cam_t, 3, linear === 'f32' ? 5e-3 : 5e-2, 'm');
}

enc?.dispose();
dec?.dispose();
heads?.dispose();
mhr?.close();
m.close();
llm.print(failures ? `${failures} FAILED` : 'all passed');
