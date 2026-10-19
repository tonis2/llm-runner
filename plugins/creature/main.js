// Creature: video frames and a rigged .glb in, an animation of that rig out.
// DINOv3 (SAM 3D Body's backbone) follows points labelled on a keyframe or
// two through the frames, and the rig is fitted to those tracks: its joint
// turns, the root and an orthographic camera (fit.js). No model of the animal
// is involved, so it works for anything a rig can be built for.
//
//   llm-runner creature model=sam-3d-body-dinov3 --server [--port 7860]
//
// `model` is the folder downloaded from huggingface.co/facebook/sam-3d-body-dinov3
// (gated), or its model.ckpt; the backbone stays loaded between requests.
// Requests are JSON, one at a time (docs/creature_server.md):
//
//   POST /animate  { rig, frames, labels, markers?, region?, width?, fps?,
//                    name?, travel?, iterations?, pitch?, outputs?, views? }
//                  -> { name, frames, fps, joints, glb?, clip?, bvh?,
//                       animated_glb?, tracks?, report }
//   POST /track    { frames, labels, region?, width?, draw? }
//                  -> { tracks, previews?, report }
//   POST /fit      { rig, tracks, markers?, ... }  -> as /animate, no GPU
//   GET  /info     what is loaded, the routes and the request's fields
//
//   rig      a rigged .glb (path, or base64), Rigsmith's get_skeleton answer,
//            or { names, parents, positions }
//   frames   a folder of PNG/JPEG frames (name order), or a list of image
//            paths or base64 images (ffmpeg -i clip.mp4 frames/%05d.png)
//   labels   { "<frame>": { "<joint or marker>": [x, y] } }, frame pixels;
//            one keyframe or more, points missing from a keyframe allowed
//   markers  { "<label>": "<joint>" }: a point on a joint's segment, its
//            offset fitted (an eye on the head, the skin over a thigh)
//   region   [x1, y1, x2, y2]: the part of every frame to look at (the
//            animal and room to move); the whole frame without one
//   width    the region resized to this many pixels wide for the backbone,
//            a multiple of 16 (1280)
//   outputs  any of glb (the clip on the rig's skeleton, as Rigsmith and
//            UniMate's clips are), clip (Rigsmith's apply_clip arguments),
//            bvh, animated_glb (the uploaded .glb with the animation added),
//            tracks; ["glb", "clip"] by default. Files come back base64.
//   views    instead of frames/labels/region/width/pitch (or tracks): the
//            same moments from other cameras, [{ frames, labels, region?,
//            width?, pitch? } or { tracks, pitch? }], frame N of each the
//            same instant; the one motion is fitted to all of them (a front
//            and a side clip: what one cannot see along its line of sight,
//            the other sees across). tracks come back one per view.
//
// Rigsmith runs it in-process as a job (llm::pipeline::Job): the job's
// config is the same request plus `model`, the clip comes back as the job's
// outputs `glb`, `clip.json`, `tracks.json` and `animated.glb` (whichever
// `outputs` names), and progress as "  step N/M, ..." lines.

import { llm } from '../lib/llm.js';
import { utf8 } from '../lib/graph/plugins.js';
import { loadBackbone } from './backbone.js';
import { Tracker } from './track.js';
import { frameSource, readLabels, drawTracks, bad } from './request.js';
import { readRig } from './rig.js';
import { fitRig } from './fit.js';
import { fittedMotion, clipGlb, clipBvh, clipJson, animatedGlb } from './output.js';

const OUTPUTS = ['glb', 'clip', 'bvh', 'animated_glb', 'tracks'];
let backbone = null, tracker = null, modelPath = null;

async function ensureBackbone(config) {
	const path = config.model;
	if (backbone && path && path !== modelPath) {
		tracker.dispose();
		backbone.dispose();
		backbone = null;
	}
	if (backbone) return;
	if (!path) throw bad('model: the sam-3d-body-dinov3 folder (start the server with model=..., or give it in the job config)');
	backbone = await loadBackbone(path, { linear: config.linear ?? 'q8', matrixCores: config.matrix_cores ?? true });
	tracker = new Tracker(backbone.encoder, { radius: Number(config.radius ?? 5) });
	modelPath = path;
}

// "  step N/M, what" now and then (Rigsmith reads these).
function stepper() {
	let last = 0;
	return (done, total, what) => {
		const now = llm.now();
		if (done < total && now - last < 500) return;
		last = now;
		llm.print(`  step ${Math.round(done)}/${Math.round(total)}, ${what}`);
	};
}

function list(v, n, what) {
	if (v === undefined || v === null || v === '') return null;
	const out = Array.isArray(v) ? v.map(Number) : String(v).split(',').map(Number);
	if (out.length !== n || out.some((x) => !Number.isFinite(x))) throw bad(`${what} wants ${n} numbers, got ${JSON.stringify(v)}`);
	return out;
}

async function track(request, config, step, labels) {
	await ensureBackbone(config);
	const frames = frameSource(request.frames);
	const region = list(request.region, 4, 'region');
	const width = Number(request.width ?? 1280);
	if (!(width >= 16)) throw bad('width: the backbone\'s input width in pixels, a multiple of 16');
	const tracks = await tracker.track(frames, { labels, region, width, onFrame: (k, n) => step?.(k, n, 'tracking') });
	return { frames, tracks };
}

// The request's views: its `views`, or the request itself as the one view.
function viewsOf(request) {
	if (request.views === undefined) return [request];
	if (!Array.isArray(request.views) || !request.views.length) throw bad('views: a list of views, each { frames, labels } or { tracks }, frame N of each the same moment');
	return request.views;
}

// The whole request: tracks (given, or made), the fit, the outputs asked for.
// Returns { reply (JSON, files in base64), files (name -> bytes) }.
export async function animate(request, config = {}) {
	const t0 = llm.now();
	const outputs = new Set(request.outputs ?? ['glb', 'clip']);
	for (const o of outputs) if (!OUTPUTS.includes(o)) throw bad(`outputs: "${o}" is not one of ${OUTPUTS.join(', ')}`);
	const rig = readRig(request.rig);
	if (outputs.has('animated_glb') && !rig.glb) throw bad('outputs: animated_glb needs the rig as a .glb');
	const name = String(request.name ?? 'creature');
	const fps = Number(request.fps ?? 24);
	const step = stepper();
	let trackSeconds = 0;
	const markers = request.markers ?? {};
	const views = viewsOf(request);
	const fitted = [];
	for (const [i, view] of views.entries()) {
		try {
			let tracks = view.tracks;
			if (tracks) {
				if (typeof tracks === 'string') tracks = JSON.parse(tracks.trim().startsWith('{') ? tracks : llm.readText(tracks));
				if (!Array.isArray(tracks?.hypotheses) || !Array.isArray(tracks.points)) throw bad('tracks: a /track reply\'s tracks ({ points, frames, hypotheses })');
				for (const p of tracks.points) if (!rig.names.includes(p) && !(p in markers)) throw bad(`tracks: "${p}" is neither one of the rig's joints nor a marker (joints: ${rig.names.join(', ')})`);
			} else {
				const { labels } = readLabels({ ...view, markers }, rig.names);
				tracks = (await track(view, config, (k, n, what) => step(i * n + k, views.length * n, what), labels)).tracks;
				trackSeconds += tracks.seconds;
			}
			fitted.push({ tracks, pitch: Number(view.pitch ?? request.pitch ?? 8) });
		} catch (e) {
			if (views.length > 1 && e.status === 400) e.message = `views[${i}]: ${e.message}`;
			throw e;
		}
	}
	const iterations = Number(request.iterations ?? 1500);
	const fit = fitRig(rig, fitted, {
		markers, pitch: Number(request.pitch ?? 8), iterations,
		onStep: (done, total) => step(done, total, 'fitting'),
	});
	const motion = fittedMotion(fit, rig, { fps, travel: request.travel ?? true });
	const report = {
		...fit.report,
		travel: Number(motion.travel.toFixed(4)),
		seconds: { track: Number(trackSeconds.toFixed(1)), fit: fit.report.seconds, total: Number(((llm.now() - t0) / 1000).toFixed(1)) },
	};
	const reply = { name, frames: motion.frames, fps, joints: rig.names, report };
	const files = {};
	if (outputs.has('glb')) files.glb = clipGlb(motion, name);
	if (outputs.has('clip')) reply.clip = clipJson(motion, name, rig.height);
	if (outputs.has('bvh')) reply.bvh = clipBvh(motion, Number(request.bvh_scale ?? 100));
	if (outputs.has('animated_glb')) files['animated.glb'] = animatedGlb(motion, rig, name);
	if (outputs.has('tracks')) reply.tracks = views.length > 1 ? fitted.map((v) => v.tracks) : fitted[0].tracks;
	if (files.glb) reply.glb = llm.base64Encode(files.glb);
	if (files['animated.glb']) reply.animated_glb = llm.base64Encode(files['animated.glb']);
	if (reply.clip) files['clip.json'] = utf8(JSON.stringify(reply.clip));
	if (reply.tracks) files['tracks.json'] = utf8(JSON.stringify(reply.tracks));
	llm.print(`  ${name}: ${motion.frames} frames on ${rig.names.length} joints, travel ${report.travel}, in ${report.seconds.total}s`);
	return { reply, files };
}

const json = (status, value) => llm.respond(status, 'application/json', JSON.stringify(value));

const INFO = {
	plugin: 'creature',
	routes: {
		'POST /animate': 'rig, frames, labels -> the clip (glb, clip, bvh, animated_glb, tracks as asked) and a report',
		'POST /track': 'frames, labels -> tracks (and previews: base64 PNGs of the frames in `draw`)',
		'POST /fit': 'rig, tracks -> as /animate, without the GPU',
		'GET /info': 'this',
	},
	request: {
		rig: 'a rigged .glb (path or base64), Rigsmith get_skeleton JSON, or { names, parents, positions }',
		frames: 'a folder of PNG/JPEG frames, or a list of image paths / base64 images',
		labels: '{ "<frame>": { "<joint or marker>": [x, y] } } in frame pixels',
		markers: '{ "<label>": "<joint>" } points on a joint\'s segment, offsets fitted',
		region: '[x1, y1, x2, y2] in frame pixels (optional)',
		width: 'backbone input width, multiple of 16 (1280)',
		fps: 'frames per second of the clip (24)',
		travel: 'put the walk back from the planted feet (true)',
		iterations: 'fit iterations (1500)',
		pitch: 'the camera\'s expected downward angle in degrees (8)',
		views: '[{ frames, labels, region?, width?, pitch? } or { tracks, pitch? }]: the same moments from other cameras, fitted together',
		outputs: `any of ${OUTPUTS.join(', ')} (["glb", "clip"])`,
	},
};

llm.plugin({
	name: 'creature',

	async load(config) {
		if (config.model) await ensureBackbone(config);
	},

	async handle(req) {
		const path = req.path.split('?')[0].replace(/\/+$/, '') || '/';
		try {
			if (req.method === 'GET' && (path === '/info' || path === '/')) {
				json(200, { ...INFO, model: modelPath, loaded: !!backbone });
				return;
			}
			if (req.method !== 'POST' || !['/animate', '/track', '/fit'].includes(path)) {
				json(404, { error: 'POST /animate, /track or /fit; GET /info' });
				return;
			}
			let body;
			try {
				body = JSON.parse(req.body || '{}');
			} catch (e) {
				json(400, { error: `the body is not JSON: ${e.message}` });
				return;
			}
			const config = globalThis.__llm_config ?? {};
			if (path === '/track') {
				const { labels } = readLabels(body);
				const { frames, tracks } = await track(body, config, stepper(), labels);
				const reply = { tracks, report: { seconds: tracks.seconds } };
				if (Array.isArray(body.draw)) reply.previews = body.draw.map((k) => ({ frame: k, png: llm.base64Encode(drawTracks(frames, tracks, Number(k))) }));
				json(200, reply);
				return;
			}
			if (path === '/fit' && viewsOf(body).some((v) => !v?.tracks)) throw bad('tracks: /fit wants the tracks a /track reply gave (in every view)');
			json(200, (await animate(body, config)).reply);
		} catch (e) {
			llm.print(`  ${req.method} ${path}: ${e.message}`);
			json(e.status ?? 500, { error: String(e.message ?? e) });
		}
	},

	// In-process (Rigsmith's job): the request is the config.
	async generate(config) {
		const { reply, files } = await animate(config, config);
		if (!llm.canOutput()) return reply;
		for (const [name, bytes] of Object.entries(files)) llm.output(name, bytes);
		return { name: reply.name, frames: reply.frames, fps: reply.fps, joints: reply.joints, report: reply.report, outputs: Object.keys(files) };
	},
});
