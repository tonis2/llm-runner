// A request's inputs as the tracker and the fit take them: the frames (a
// folder, or a list of paths and base64 images), the keyframe labels and the
// markers, checked against each other and the rig.

import { llm } from '../lib/llm.js';

const IMAGE = /\.(png|jpe?g)$/i;

// An error in what the client sent: the server answers it with 400.
export function bad(message) {
	const e = new Error(message);
	e.status = 400;
	return e;
}

// A path, or an image or file's bytes in base64 (a data: URL or bare).
export function isInline(text) {
	return text.startsWith('data:') || (text.length > 260 && /^[A-Za-z0-9+/=\s]+$/.test(text.slice(0, 260)));
}

export function bytesOf(text) {
	return isInline(text) ? llm.base64Decode(text) : llm.readBytes(text);
}

// `spec` a folder of PNG/JPEG frames (name order) or a list of paths and
// base64 images, to { count, image(k) }. The last image decoded is kept: a
// sweep asks for each frame once, the tracker its first twice.
export function frameSource(spec) {
	let list = spec;
	if (typeof spec === 'string') {
		const dir = spec.replace(/\/+$/, '');
		list = llm.listDir(dir).filter((e) => !e.dir && IMAGE.test(e.name)).map((e) => `${dir}/${e.name}`).sort();
		if (!list.length) throw bad(`frames: ${spec} holds no .png or .jpg frames`);
	}
	if (!Array.isArray(list) || !list.length) throw bad('frames: a folder, or a list of image paths or base64 images');
	let last = -1, held = null;
	return {
		count: list.length,
		names: list.map((s, k) => (isInline(s) ? `frame ${k}` : s)),
		image(k) {
			if (k !== last) {
				const s = list[k];
				held = isInline(s) ? llm.image.decode(llm.base64Decode(s)) : llm.image.load(s);
				last = k;
			}
			return held;
		},
	};
}

// The labels { frame: { name: [x, y] } } with numeric frames and only
// two-number points; the markers { name: joint }. With `joints` (the rig's
// names) every label must be a joint or a marker on one.
export function readLabels(request, joints = null) {
	const raw = request.labels;
	if (!raw || typeof raw !== 'object') throw bad('labels: { "<frame>": { "<joint or marker>": [x, y] } } on at least one frame');
	const labels = {};
	for (const [frame, points] of Object.entries(raw)) {
		const k = Number(frame);
		if (!Number.isInteger(k) || k < 0) throw bad(`labels: "${frame}" is not a frame number`);
		const clean = {};
		for (const [name, p] of Object.entries(points ?? {})) {
			if (p === null) continue;
			if (!Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite)) throw bad(`labels: ${name} on frame ${k} wants [x, y], got ${JSON.stringify(p)}`);
			clean[name] = [p[0], p[1]];
		}
		if (Object.keys(clean).length) labels[k] = clean;
	}
	if (!Object.keys(labels).length) throw bad('labels: no points');
	const markers = { ...(request.markers ?? {}) };
	if (joints) {
		const known = new Set(joints);
		for (const [m, spec] of Object.entries(markers)) {
			const j = markerJoint(spec);
			if (!known.has(j)) throw bad(`markers: ${m} is on "${j}", which is not one of the rig's joints (${joints.join(', ')})`);
			if (known.has(m)) throw bad(`markers: ${m} is a joint's name; a marker needs a name of its own`);
			if (typeof spec === 'object' && !(Array.isArray(spec.offset) && spec.offset.length === 3 && spec.offset.every(Number.isFinite))) {
				throw bad(`markers: ${m} wants { joint, offset: [x, y, z] } (its place from the joint at rest, in model units), or the joint's name alone`);
			}
		}
		for (const points of Object.values(labels)) {
			for (const name of Object.keys(points)) {
				if (!known.has(name) && !(name in markers)) throw bad(`labels: "${name}" is neither one of the rig's joints nor a marker (joints: ${joints.join(', ')})`);
			}
		}
	}
	return { labels, markers };
}

// The joint a marker is on: `"<joint>"` (its offset fitted) or `{ joint, offset }`
// (its offset known: a point of the mesh, from the joint at rest).
export const markerJoint = (spec) => (spec && typeof spec === 'object' ? spec.joint : spec);

const COLOURS = { L: [230, 40, 40], R: [40, 90, 230], M: [250, 210, 0] };
const side = (name) => (/(_L|_l|\.L|Left|left)$/.test(name) || name.startsWith('l_') ? 'L' : /(_R|_r|\.R|Right|right)$/.test(name) || name.startsWith('r_') ? 'R' : 'M');

// Frame `k` with every hypothesis's points drawn on it (left red, right
// blue, the middle yellow; a point's first hypothesis filled, the rest rings),
// as a PNG's bytes.
export function drawTracks(frames, tracks, k) {
	const img = frames.image(k);
	const { width, height, channels } = img;
	const px = new Uint8Array(width * height * 3);
	for (let i = 0; i < width * height; i++) for (let c = 0; c < 3; c++) px[i * 3 + c] = img.pixels[i * channels + c];
	const dot = (x, y, r, colour, ring) => {
		for (let dy = -r; dy <= r; dy++) {
			for (let dx = -r; dx <= r; dx++) {
				const d2 = dx * dx + dy * dy;
				if (d2 > r * r || (ring && d2 < (r - 2) * (r - 2))) continue;
				const X = Math.round(x + dx), Y = Math.round(y + dy);
				if (X < 0 || Y < 0 || X >= width || Y >= height) continue;
				px.set(colour, (Y * width + X) * 3);
			}
		}
	};
	const r = Math.max(3, Math.round(width / 240));
	tracks.hypotheses.forEach((h, n) => {
		const i = h.frames.indexOf(k);
		if (i < 0) return;
		h.tracks[i].forEach((p, j) => { if (p) dot(p[0], p[1], r, COLOURS[side(tracks.points[j])], n > 0); });
	});
	return llm.image.encodePng({ width, height, channels: 3, pixels: px });
}
