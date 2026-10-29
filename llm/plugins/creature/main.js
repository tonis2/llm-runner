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
//                    name?, travel?, travel_smooth?, iterations?, pitch?,
//                    outputs?, views?, studio?, finder?, key_every?, flip?,
//                    landmarks?, weights?, sigma?, confidence? }
//                  -> { name, frames, fps, joints, glb?, clip?, bvh?,
//                       animated_glb?, tracks?, report }
//   POST /track    { frames, labels, region?, width?, draw? }
//                  -> { tracks, previews?, report }
//   POST /fit      { rig, tracks, markers?, ... }  -> as /animate, no GPU
//   POST /train    { renders, out, joints?, backgrounds?, epochs?, ... }
//                  -> { finder, report }: a joint finder for one rig, trained
//                     on Rigsmith's render_poses pictures of it (train.js)
//   POST /detect   { finder, frames, region?, key_every?, flip?, draw? } or
//                  { finder, views: [{ frames, region? }] }
//                  -> { tracks, previews?, report }: every joint the finder
//                     knows, on every frame, with no labels (detect.js)
//   GET  /info     what is loaded, the routes and the request's fields
//
//   rig      a rigged .glb (path, or base64), Rigsmith's get_skeleton answer,
//            or { names, parents, positions }
//   frames   a folder of PNG/JPEG frames (name order), or a list of image
//            paths or base64 images (ffmpeg -i clip.mp4 frames/%05d.png)
//   labels   { "<frame>": { "<joint or marker>": [x, y] } }, frame pixels;
//            one keyframe or more, points missing from a keyframe allowed
//   markers  { "<label>": "<joint>" }: a point on a joint's segment, its
//            offset fitted (an eye on the head, the skin over a thigh); or
//            { "<label>": { joint, offset } }, its offset known (a point of
//            the mesh, from the joint at rest)
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
//   studio   Rigsmith render_studio's studio.json: `frames` are a clip made
//            from its picture, both views side by side in every frame; the
//            views are cut out by the layout and their cameras are known
//            (studio.js), so none is searched for and the root moves as the
//            cameras saw it (no travel from the feet)
//   finder   a trained joint finder (/train): a view with neither labels
//            nor tracks has its joints found on every frame instead; the
//            points of the mesh it finds (landmarks) are markers with known
//            offsets
//   weights  { "<term>": weight } over the fit's defaults (fit.js
//            DEFAULT_WEIGHTS), to tune it without code
//   sigma    the reprojection's robust scale in pixels (15)
//   confidence  how a found point's score weighs its pull, weight^confidence;
//            2 for a finder's tracks, 0 (all pull fully) for labels and tracks
//
// Rigsmith runs it in-process as a job (llm::pipeline::Job): the job's
// config is the same request plus `model`, the clip comes back as the job's
// outputs `glb`, `clip.json`, `tracks.json` and `animated.glb` (whichever
// `outputs` names), and progress as "  step N/M, ..." lines. `task: "train"`
// or `"detect"` runs /train's or /detect's request instead.

import { llm } from '/core/llm.js';
import { utf8 } from '/core/graph/plugins.js';
import { loadBackbone } from './backbone.js';
import { Tracker } from './track.js';
import { frameSource, readLabels, drawTracks, bad } from './request.js';
import { readRig } from './rig.js';
import { fitRig, DEFAULT_WEIGHTS } from './fit.js';
import { fittedMotion, clipGlb, clipBvh, clipJson, animatedGlb } from './output.js';
import { Finder, fittedLandmarks } from './detect.js';
import { trainFinder } from './train.js';
import { readStudio, studioViews, calibrateViews, cropFrames } from './studio.js';
import { syncViews } from './sync.js';

const OUTPUTS = ['glb', 'clip', 'bvh', 'animated_glb', 'tracks'];

// The request's fit settings, checked: { weights, sigma, confidence }.
function fitSettings(request) {
	const weights = request.weights ?? null;
	if (weights !== null) {
		if (typeof weights !== 'object' || Array.isArray(weights)) throw bad(`weights: { "<term>": number }, terms ${Object.keys(DEFAULT_WEIGHTS).join(', ')}`);
		for (const [k, v] of Object.entries(weights)) {
			if (!(k in DEFAULT_WEIGHTS)) throw bad(`weights: "${k}" is not one of the fit's terms (${Object.keys(DEFAULT_WEIGHTS).join(', ')})`);
			if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) throw bad(`weights: ${k} is ${JSON.stringify(v)}; it is a number of 0 or more`);
		}
	}
	const number = (key, fallback, least) => {
		if (request[key] === undefined || request[key] === null) return fallback;
		const v = Number(request[key]);
		if (!Number.isFinite(v) || v < least) throw bad(`${key}: ${JSON.stringify(request[key])} is not a number of ${least} or more`);
		return v;
	};
	return { weights, sigma: number('sigma', 15, 0.5), confidence: number('confidence', null, 0) };
}
let backbone = null, tracker = null, modelPath = null, finder = null;

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
	const frames = framesOf(request.frames);
	const region = list(request.region, 4, 'region');
	const width = Number(request.width ?? 1280);
	if (!(width >= 16)) throw bad('width: the backbone\'s input width in pixels, a multiple of 16');
	const tracks = await tracker.track(frames, { labels, region, width, onFrame: (k, n) => step?.(k, n, 'tracking') });
	return { frames, tracks };
}

// The finder at `path`, loaded once and kept until another is asked for.
function finderAt(path) {
	if (!path || typeof path !== 'string') throw bad('finder: the path of a trained joint finder (.safetensors, from POST /train)');
	if (finder && finder.path !== path) {
		finder.dispose();
		finder = null;
	}
	if (!finder) finder = Finder.load(path);
	return finder;
}

// One view's joints found by the request's finder: { frames, tracks }.
async function detect(request, view, config, step) {
	await ensureBackbone(config);
	const f = finderAt(request.finder);
	const frames = framesOf(view.frames);
	const tracks = await f.detect(backbone.encoder, frames, {
		region: list(view.region, 4, 'region'), keyEvery: Number(request.key_every ?? 6), flip: request.flip !== false,
		onFrame: (k, n) => step?.(k, n, 'finding joints'),
	});
	return { frames, tracks };
}

async function train(request, config) {
	await ensureBackbone(config);
	// A finder being rewritten is not the one in memory any more.
	if (finder && finder.path === request.out) {
		finder.dispose();
		finder = null;
	}
	const report = await trainFinder(backbone.encoder, request, stepper());
	return { finder: request.out, report };
}

// `tracks` without the points named in `drop` (landmarks a request leaves out).
function withoutPoints(tracks, drop) {
	if (!drop.size) return tracks;
	const keep = tracks.points.map((n, p) => (drop.has(n) ? -1 : p)).filter((p) => p >= 0);
	const pick = (row) => (row ? keep.map((p) => row[p]) : row);
	return {
		...tracks, points: keep.map((p) => tracks.points[p]),
		hypotheses: tracks.hypotheses.map((h) => ({ ...h, tracks: h.tracks.map(pick), scores: h.scores.map(pick) })),
	};
}

// A view's frames: a frame source already (a studio's cut views), or what
// frameSource reads.
const framesOf = (spec) => (spec && typeof spec.image === 'function' ? spec : frameSource(spec));

// A studio request's views: its frames cut into the layout's two, each with
// its camera, put right for the clip by its first frame against the picture
// it was made from (studio.js calibrateViews). Or tracks found before, with
// the clip's `size` (or its `frames`, which calibrate as well).
function studioViewsOf(request) {
	const layout = readStudio(request.studio);
	const given = request.views !== undefined;
	if (given && (!Array.isArray(request.views) || request.views.length !== 2 || !request.views.every((v) => v?.tracks))) throw bad('studio: frames (the clip), or views [{ tracks }, { tracks }] (front, side) with the clip\'s size [width, height] or frames');
	const frames = request.frames !== undefined ? frameSource(request.frames) : null, first = frames?.image(0);
	const size = first ? [first.width, first.height] : list(request.size, 2, 'size');
	if (!size) throw bad('studio: with views and no frames, size [width, height] of the clip');
	let views = studioViews(layout, size[0], size[1]), calibration = null;
	const reference = first && request.calibrate !== false ? referenceOf(request) : null;
	if (reference) {
		({ views, report: calibration } = calibrateViews(layout, views, reference, first));
		for (const c of calibration) llm.print(`  studio: ${c.name} view scaled ${c.scale}, moved ${c.shift.join(', ')} px against the picture (figures overlap ${c.overlap[0]} -> ${c.overlap[1]})`);
	}
	return views.map((v, i) => ({ ...(given ? request.views[i] : { frames: cropFrames(frames, v.box) }), name: v.name, box: v.box, camera: v.camera, calibration }));
}

// The picture a studio clip was made from: `reference`, or reference.png
// beside the studio.json; null when there is none.
function referenceOf(request) {
	const path = request.reference ?? (typeof request.studio === 'string' && !request.studio.trim().startsWith('{') ? request.studio.replace(/[^/]*$/, 'reference.png') : null);
	if (!path) return null;
	try {
		return llm.image.load(path);
	} catch (e) {
		if (request.reference) throw bad(`reference: ${path} could not be read (${e.message})`);
		return null;
	}
}

// The request's views: its `views`, or the request itself as the one view.
function viewsOf(request) {
	if (request.studio !== undefined) return studioViewsOf(request);
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
	fitSettings(request);
	const name = String(request.name ?? 'creature');
	const fps = Number(request.fps ?? 24);
	const step = stepper();
	let trackSeconds = 0;
	const markers = { ...(request.markers ?? {}) };
	// A finder's landmarks: points of the mesh, markers whose offsets are known.
	// By default all but a mirror pair on one joint (fittedLandmarks);
	// `landmarks` false for none, true for every one, or a list of those to use.
	const unused = new Set();
	if (request.finder) {
		const all = finderAt(request.finder).landmarks;
		const use = request.landmarks ?? fittedLandmarks(all).map((m) => m.name);
		if (use !== true && use !== false && !Array.isArray(use)) throw bad('landmarks: true (all the finder\'s), false, or a list of their names');
		if (Array.isArray(use)) for (const n of use) if (!all.some((m) => m.name === n)) throw bad(`landmarks: the finder has no "${n}" (it has ${all.map((m) => m.name).join(', ') || 'none'})`);
		for (const m of all) {
			if (use === false || (Array.isArray(use) && !use.includes(m.name))) { unused.add(m.name); continue; }
			if (!rig.names.includes(m.joint)) throw bad(`finder: its landmark ${m.name} rides "${m.joint}", which is not one of this rig's joints (trained on another rig?)`);
			markers[m.name] ??= { joint: m.joint, offset: m.offset };
		}
	}
	const views = viewsOf(request);
	const fitted = [];
	for (const [i, view] of views.entries()) {
		try {
			let tracks = view.tracks;
			if (tracks) {
				if (typeof tracks === 'string') tracks = JSON.parse(tracks.trim().startsWith('{') ? tracks : llm.readText(tracks));
				if (!Array.isArray(tracks?.hypotheses) || !Array.isArray(tracks.points)) throw bad('tracks: a /track reply\'s tracks ({ points, frames, hypotheses })');
				tracks = withoutPoints(tracks, unused);
				for (const p of tracks.points) if (!rig.names.includes(p) && !(p in markers)) throw bad(`tracks: "${p}" is neither one of the rig's joints nor a marker (joints: ${rig.names.join(', ')})`);
			} else if (!view.labels && request.finder) {
				tracks = withoutPoints((await detect(request, view, config, (k, n, what) => step(i * n + k, views.length * n, what))).tracks, unused);
				for (const p of tracks.points) if (!rig.names.includes(p) && !(p in markers)) throw bad(`finder: it finds "${p}", which is not one of this rig's joints (trained on another rig?)`);
				trackSeconds += tracks.seconds;
			} else {
				const { labels } = readLabels({ ...view, markers }, rig.names);
				tracks = (await track(view, config, (k, n, what) => step(i * n + k, views.length * n, what), labels)).tracks;
				trackSeconds += tracks.seconds;
			}
			fitted.push({ tracks, pitch: Number(view.pitch ?? request.pitch ?? 8), camera: view.camera });
		} catch (e) {
			if (views.length > 1 && e.status === 400) e.message = `views[${i}]: ${e.message}`;
			throw e;
		}
	}
	// A studio clip's halves in step: measured, and the front re-timed onto the
	// side when that brings their heights well closer (sync.js).
	let sync = null;
	if (request.studio !== undefined && fitted.length === 2) {
		const s = syncViews(fitted[0].tracks, fitted[1].tracks, fitted.map((v) => v.camera), { retime: request.sync !== false });
		fitted[0].tracks = s.front;
		sync = s.report;
		if (sync) llm.print(`  sync: heights ${sync.height_gap_px}px apart on average (worst ${sync.worst_gap_px}px), best lag ${sync.best_lag} frames${sync.retimed ? `; front re-timed, up to ${sync.shift_frames} frames (${sync.retime_gap_px}px apart now)` : ''}`);
	}
	const iterations = Number(request.iterations ?? 1500);
	const fit = fitRig(rig, fitted, {
		markers, pitch: Number(request.pitch ?? 8), iterations, ...fitSettings(request),
		onStep: (done, total) => step(done, total, 'fitting'),
	});
	const motion = fittedMotion(fit, rig, { fps, travel: request.travel ?? true, smooth: Number(request.travel_smooth ?? 25) });
	const report = {
		...fit.report,
		travel: Number(motion.travel.toFixed(4)),
		seconds: { track: Number(trackSeconds.toFixed(1)), fit: fit.report.seconds, total: Number(((llm.now() - t0) / 1000).toFixed(1)) },
	};
	if (request.studio !== undefined) report.studio = { views: views.map((v) => ({ name: v.name, box: v.box })), calibration: views[0].calibration };
	if (sync) report.sync = sync;
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

// /detect: the tracks of each view (one view's alone when there are no views).
async function detectReply(request, config) {
	const step = stepper(), views = viewsOf(request), out = [];
	let seconds = 0;
	for (const [i, view] of views.entries()) {
		if (!view?.frames) throw bad(`${views.length > 1 ? `views[${i}]: ` : ''}frames: a folder of frames, or a list of images`);
		const { frames, tracks } = await detect(request, view, config, (k, n, what) => step(i * n + k, views.length * n, what));
		seconds += tracks.seconds;
		const one = { tracks };
		const draw = view.draw ?? request.draw;
		if (Array.isArray(draw)) one.previews = draw.map((k) => ({ frame: k, png: llm.base64Encode(drawTracks(frames, tracks, Number(k))) }));
		out.push(one);
	}
	const report = { seconds: Math.round(seconds * 10) / 10, finder: finder.path, joints: finder.joints.length };
	if (request.views === undefined) return { ...out[0], report };
	return { tracks: out.map((o) => o.tracks), previews: out.map((o) => o.previews ?? []), report };
}

const json = (status, value) => llm.respond(status, 'application/json', JSON.stringify(value));

const INFO = {
	plugin: 'creature',
	routes: {
		'POST /animate': 'rig, frames, labels -> the clip (glb, clip, bvh, animated_glb, tracks as asked) and a report',
		'POST /track': 'frames, labels -> tracks (and previews: base64 PNGs of the frames in `draw`)',
		'POST /fit': 'rig, tracks -> as /animate, without the GPU',
		'POST /train': 'renders (render_poses folders), out -> a joint finder for that rig, and a report',
		'POST /detect': 'finder, frames (or views) -> tracks of every joint the finder knows (and previews: base64 PNGs of the frames in `draw`)',
		'GET /info': 'this',
	},
	request: {
		rig: 'a rigged .glb (path or base64), Rigsmith get_skeleton JSON, or { names, parents, positions }',
		frames: 'a folder of PNG/JPEG frames, or a list of image paths / base64 images',
		labels: '{ "<frame>": { "<joint or marker>": [x, y] } } in frame pixels',
		markers: '{ "<label>": "<joint>" } points on a joint\'s segment, offsets fitted; or { "<label>": { joint, offset } }, offsets known',
		region: '[x1, y1, x2, y2] in frame pixels (optional)',
		width: 'backbone input width, multiple of 16 (1280)',
		fps: 'frames per second of the clip (24)',
		travel: 'put the walk back from the planted feet (true)',
		travel_smooth: 'frames the travel\'s steps are smoothed over, an odd number; 1 for none (25)',
		iterations: 'fit iterations (1500)',
		pitch: 'the camera\'s expected downward angle in degrees (8)',
		views: '[{ frames, labels, region?, width?, pitch? } or { tracks, pitch? }]: the same moments from other cameras, fitted together',
		studio: 'Rigsmith render_studio\'s studio.json: frames are a clip of its picture (front and side side by side); the views are cut out by its layout and their cameras known',
		reference: 'with studio: the picture the clip was made from (reference.png beside the studio.json by default), its first frame laid on it to put the cameras right',
		calibrate: 'with studio: put the cameras right by the first frame against the reference (true)',
		sync: 'with studio: re-time the front view onto the side\'s when their joints\' heights say they are out of step (true; false to only report)',
		finder: 'a trained joint finder (.safetensors from /train): views without labels or tracks are found with it',
		key_every: 'with a finder: a keyframe for the fit to start from every so many frames (6)',
		flip: 'with a finder: look at each crop mirrored as well, left read as right, and average (true; false to look once)',
		landmarks: 'with a finder: which of its landmarks (points of the mesh) to fit: true for all, false for none, or a list of names (all but a mirror pair on one joint, such as a head\'s two sides)',
		weights: `{ "<term>": weight } over the fit's defaults: ${Object.entries(DEFAULT_WEIGHTS).map(([k, v]) => `${k} ${v}`).join(', ')}`,
		sigma: 'the reprojection\'s robust scale in pixels (15)',
		confidence: 'how a found point\'s score weighs its pull, weight^confidence (2 for a finder\'s tracks, 0 for labels and tracks)',
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
			if (req.method !== 'POST' || !['/animate', '/track', '/fit', '/train', '/detect'].includes(path)) {
				json(404, { error: 'POST /animate, /track, /fit, /train or /detect; GET /info' });
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
			if (path === '/train') {
				json(200, await train(body, config));
				return;
			}
			if (path === '/detect') {
				json(200, await detectReply(body, config));
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
		if (config.task === 'train') return train(config, config);
		if (config.task === 'detect') {
			const reply = await detectReply(config, config);
			if (llm.canOutput()) llm.output('tracks.json', utf8(JSON.stringify(reply.tracks)));
			return reply;
		}
		const { reply, files } = await animate(config, config);
		if (!llm.canOutput()) return reply;
		for (const [name, bytes] of Object.entries(files)) llm.output(name, bytes);
		return { name: reply.name, frames: reply.frames, fps: reply.fps, joints: reply.joints, report: reply.report, outputs: Object.keys(files) };
	},
});
