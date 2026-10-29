// A studio clip: one video of the rig from two cameras side by side, front on
// the left and its right side on the right, made from Rigsmith's render_studio
// picture (reference.png) by a video model. studio.json beside that picture is
// the layout: the picture's size, the floor row, each view's columns, and how a
// point p of the model lands in a view — across, the view's left edge plus
// half its width plus (p - origin)·right times pixels_per_unit; down, the floor
// row minus (p - origin)·up times pixels_per_unit. Both cameras are level and
// orthographic, so the fit takes them as known instead of searching for them.
//
//   const layout = readStudio('studio.json');
//   const views = studioViews(layout, frameWidth, frameHeight);   // [{ name, box, camera }]
//   const frames = cropFrames(source, views[0].box);
//
// A clip may be bigger than the picture (MiniMax gave 1920 x 1080 for 1280 x
// 720): the layout is scaled to it, the same both ways.

import { llm } from '../lib/llm.js';
import { bad } from './request.js';

// The layout, from a path or as given, checked.
export function readStudio(spec) {
	let layout = spec;
	if (typeof spec === 'string') {
		try {
			layout = JSON.parse(spec.trim().startsWith('{') ? spec : llm.readText(spec));
		} catch (e) {
			throw bad(`studio: ${spec} is not a studio.json (${e.message})`);
		}
	}
	const ok = layout && layout.width > 0 && layout.height > 0 && layout.pixels_per_unit > 0 && Number.isFinite(layout.ground_row)
		&& Array.isArray(layout.views) && layout.views.length === 2
		&& layout.views.every((v) => v.width > 0 && [v.origin, v.right, v.up].every((a) => Array.isArray(a) && a.length === 3));
	if (!ok) throw bad('studio: Rigsmith render_studio\'s studio.json ({ width, height, ground_row, pixels_per_unit, views: [front, side] })');
	return layout;
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Each view of a `width` x `height` clip of `layout`: { name, box [x1, y1, x2,
 * y2] (whole frame pixels, the divider left out), camera } — the camera as
 * fit.js's RigFit takes it, for points in the box's own pixels: { yaw, pitch,
 * logS, shift [x, y] } (y up).
 */
export function studioViews(layout, width, height) {
	const sx = width / layout.width, sy = height / layout.height;
	if (Math.abs(sx / sy - 1) > 0.02) throw bad(`studio: the clip is ${width} x ${height}, the studio ${layout.width} x ${layout.height}; it is stretched (${sx.toFixed(3)} across, ${sy.toFixed(3)} down), not scaled`);
	const scale = (sx + sy) / 2;
	const gap = Math.ceil(((layout.divider_width ?? 4) / 2 + 1) * scale);
	const ppu = layout.pixels_per_unit * scale;
	return layout.views.map((view, v) => {
		const left = Math.round(view.x * scale), right = Math.round((view.x + view.width) * scale);
		const box = [v === 0 ? left : left + gap, 0, v === layout.views.length - 1 ? Math.min(width, right) : right - gap, height];
		const r = view.right, u = view.up, o = view.origin;
		// The fit's cameras turn about the model's up only: a studio is level,
		// and its up is the model's +Y unless the model was placed tipped over.
		if (Math.abs(u[1] - 1) > 1e-3 || Math.abs(r[1]) > 1e-3) throw bad(`studio: view ${view.name} is not level in the model (up ${u.join(', ')}); render_studio of a model placed upright`);
		// Across: s (p · right) + shift; up: s (p · up) + shift, the box's own pixels.
		return {
			name: view.name, box,
			camera: {
				yaw: Math.atan2(r[2], r[0]),
				pitch: 0,
				logS: Math.log(ppu),
				shift: [(view.x + view.width / 2) * scale - ppu * dot(o, r) - box[0], -layout.ground_row * scale - ppu * dot(o, u) + box[1]],
			},
		};
	});
}

// The figure in one view of `img` (a frame, or the reference picture, read at
// frame pixel (x, y) as its pixel (x / k, y / k)), sampled on a grid of `step`
// frame pixels over `box`: 1 where a pixel is far from the view's own wall and
// floor (their medians, from its top rows and the floor band), else 0.
function figureMask(img, k, box, floorRow, step) {
	const W = Math.floor((box[2] - box[0]) / step), H = Math.floor((box[3] - box[1]) / step);
	const { pixels, channels: C } = img, iw = img.width, ih = img.height;
	const cols = Int32Array.from({ length: W }, (_, j) => Math.min(iw - 1, Math.max(0, Math.round((box[0] + j * step) / k))) * C);
	const row = (y) => Math.min(ih - 1, Math.max(0, Math.round(y / k))) * iw * C;
	const median = (y0, y1) => {
		const vals = [[], [], []];
		for (let y = y0; y < y1; y += step) for (const c0 of cols) { const o = row(y) + c0; for (let c = 0; c < 3; c++) vals[c].push(pixels[o + c]); }
		return vals.map((v) => v.sort((a, b) => a - b)[v.length >> 1] ?? 0);
	};
	const wall = median(box[1] + 4 * step, box[1] + 20 * step), floor = median(floorRow + 6 * step, Math.min(box[3], floorRow + 20 * step));
	const mask = new Uint8Array(W * H);
	for (let i = 0; i < H; i++) {
		const y = box[1] + i * step, bg = y < floorRow ? wall : floor, r = row(y);
		for (let j = 0; j < W; j++) {
			const o = r + cols[j];
			mask[i * W + j] = Math.abs(pixels[o] - bg[0]) + Math.abs(pixels[o + 1] - bg[1]) + Math.abs(pixels[o + 2] - bg[2]) > 40 ? 1 : 0;
		}
	}
	return { mask, W, H };
}

// The cells a mask's figure covers: [i0, i1, j0, j1] (end exclusive), or
// null when it has none.
function extent(m) {
	let i0 = m.H, i1 = 0, j0 = m.W, j1 = 0;
	for (let i = 0; i < m.H; i++) {
		for (let j = 0; j < m.W; j++) {
			if (!m.mask[i * m.W + j]) continue;
			if (i < i0) i0 = i;
			if (i >= i1) i1 = i + 1;
			if (j < j0) j0 = j;
			if (j >= j1) j1 = j + 1;
		}
	}
	return i1 > i0 ? [i0, i1, j0, j1] : null;
}

// How much of the two figures overlap (intersection over union) with the
// reference's figure scaled by k about (cx, cy) and moved by (dx, dy), in
// mask cells. Only cells either figure can reach are counted: the reference's
// extent carried over, and the frame's.
function overlap(ref, vid, k, dx, dy, cx, cy) {
	const re = ref.extent, ve = vid.extent;
	if (!re || !ve) return 0;
	const i0 = Math.max(0, Math.min(ve[0], Math.floor(cy + k * (re[0] - cy) + dy) - 1)), i1 = Math.min(vid.H, Math.max(ve[1], Math.ceil(cy + k * (re[1] - cy) + dy) + 1));
	const j0 = Math.max(0, Math.min(ve[2], Math.floor(cx + k * (re[2] - cx) + dx) - 1)), j1 = Math.min(vid.W, Math.max(ve[3], Math.ceil(cx + k * (re[3] - cx) + dx) + 1));
	// Where each column and row lands in the reference, once (-1 off it).
	const rx = new Int32Array(j1 - j0), rowAt = new Int32Array(i1 - i0);
	for (let j = j0; j < j1; j++) { const x = Math.round(cx + (j - cx - dx) / k); rx[j - j0] = x >= 0 && x < ref.W ? x : -1; }
	for (let i = i0; i < i1; i++) { const y = Math.round(cy + (i - cy - dy) / k); rowAt[i - i0] = y >= 0 && y < ref.H ? y * ref.W : -1; }
	const rm = ref.mask, vm = vid.mask;
	let both = 0, either = 0;
	for (let i = i0; i < i1; i++) {
		const row = rowAt[i - i0], at = i * vid.W;
		for (let j = j0; j < j1; j++) {
			const x = rx[j - j0], a = row >= 0 && x >= 0 ? rm[row + x] : 0, b = vm[at + j];
			both += a & b;
			either += a | b;
		}
	}
	return either ? both / either : 0;
}

/**
 * Each view's camera put right for a clip whose frames are not quite the
 * picture's (a video model may scale or move its first frame a little): the
 * scale about the view's centre and floor, and the shift, that best lay the
 * reference picture's figure on the first frame's, coarse to fine. Returns
 * the views (cameras corrected where that lays the figures closer) and, per
 * view, { name, scale, shift [x, y] (frame pixels), overlap [before, after] }.
 */
export function calibrateViews(layout, views, reference, first) {
	const scale = first.width / layout.width;
	const floorRow = layout.ground_row * scale;
	// The reference at its own size (the layout's), read in frame pixels.
	const refScale = first.width / reference.width;
	const report = [];
	const out = views.map((view) => {
		// Scaled about the box's middle column and the floor row.
		const box = view.box, cxFrame = (box[0] + box[2]) / 2, cyFrame = floorRow;
		let best = { k: 1, dx: 0, dy: 0, o: 0 };
		let before = 0;
		// Three passes: 8-pixel cells searched over 6% and 24 pixels either way,
		// then 4- and 2-pixel cells climbing from the best by single steps while
		// a step lays the figures closer.
		const passes = [[8, 0.02, 3, 3], [4, 0.005, 0, 0], [2, 0.0025, 0, 0]];
		for (const [p, [step, ks, kn, reach]] of passes.entries()) {
			const ref = figureMask(reference, refScale, box, floorRow, step), vid = figureMask(first, 1, box, floorRow, step);
			ref.extent = extent(ref);
			vid.extent = extent(vid);
			const cx = (cxFrame - box[0]) / step, cy = (cyFrame - box[1]) / step;
			if (p === passes.length - 1) before = overlap(ref, vid, 1, 0, 0, cx, cy);
			const at = (n, dx, dy) => ({ n, dx, dy, o: overlap(ref, vid, best.k + n * ks, dx, dy, cx, cy) });
			let here = at(0, Math.round(best.dx / step), Math.round(best.dy / step));
			if (reach) {
				for (let n = -kn; n <= kn; n++) for (let dx = -reach; dx <= reach; dx++) for (let dy = -reach; dy <= reach; dy++) {
					const o = at(n, here.dx + dx, here.dy + dy);
					if (o.o > here.o) here = o;
				}
			} else {
				for (let moved = true, rounds = 0; moved && rounds < 20; rounds++) {
					moved = false;
					for (const [a, b, c] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
						const o = at(here.n + a, here.dx + b, here.dy + c);
						if (o.o > here.o) { here = o; moved = true; }
					}
				}
			}
			best = { k: best.k + here.n * ks, dx: here.dx * step, dy: here.dy * step, o: here.o };
		}
		const one = { name: view.name, scale: Number(best.k.toFixed(4)), shift: [best.dx, best.dy], overlap: [Number(before.toFixed(3)), Number(best.o.toFixed(3))] };
		report.push(one);
		if (!(best.o > before + 0.01)) return view;
		// Frame pixel f becomes c + k (f - c) + d: across in the box's pixels,
		// up as y-up (-f).
		const c = view.camera, k = best.k;
		return {
			...view,
			camera: {
				...c,
				logS: c.logS + Math.log(k),
				shift: [k * (c.shift[0] + box[0] - cxFrame) + cxFrame + best.dx - box[0], k * c.shift[1] + (k - 1) * cyFrame - best.dy],
			},
		};
	});
	return { views: out, report };
}

/**
 * `frames` (request.js frameSource) cut to `box`: each image only the box's
 * pixels, so a view's crops never take in the other view's figure.
 */
export function cropFrames(frames, box) {
	const [x1, y1, x2, y2] = box.map(Math.round);
	let last = -1, held = null;
	return {
		count: frames.count,
		names: frames.names,
		image(k) {
			if (k !== last) {
				const img = frames.image(k), c = img.channels, w = x2 - x1, h = y2 - y1;
				if (x2 > img.width || y2 > img.height) throw bad(`studio: frame ${k} is ${img.width} x ${img.height}, smaller than the first`);
				const pixels = new Uint8Array(w * h * c);
				for (let y = 0; y < h; y++) pixels.set(img.pixels.subarray(((y1 + y) * img.width + x1) * c, ((y1 + y) * img.width + x2) * c), y * w * c);
				held = { width: w, height: h, channels: c, pixels };
				last = k;
			}
			return held;
		},
	};
}
