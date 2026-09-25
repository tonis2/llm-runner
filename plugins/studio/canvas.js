// The graph, drawn: nodes as cards with their ports, wires as curves, and the
// pointer - drag a node to move it, drag from a port to wire it, drag the
// background (or with the middle/right button anywhere) to pan, and the wheel
// zooms around the pointer.
//
// One `Drawing` for the cards, wires and ports, rebuilt on every render - the
// widget layer diffs it into a single patch. A node's settings are real editors
// laid over it (see layout.js and fields.js), sized to the zoom; zoomed far out,
// or under another node, a setting is drawn as its value instead.

import { app } from './app.js';
import { defOf, isSetting, isWire } from './doc.js';
import { layoutOf } from './layout.js';
import { editor } from './fields.js';
import { accepts } from '../lib/graph/types.js';
import {
	THEME, typeColor, NODE_W, NODE_HEAD, PORT_R, LABEL_H, EDIT_ZOOM, ZOOM_STEP,
} from './theme.js';

const { Drawing, Clip, Stack, Anchored } = three.ui;

const TEXT = 12;
const HIT = 9;           // how near a port counts, in points on screen

function measure(text, size) {
	return three.ui.measure(text, { size })[0];
}

// A value as a node shows it: short, one line.
function shown(v) {
	if (v === undefined || v === null) return '';
	if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, '');
	const s = String(v);
	const slash = s.lastIndexOf('/');
	return slash >= 0 && s.length > 24 ? s.slice(slash + 1) : s;
}

const clipCache = new Map();
function clip(text, width, size) {
	const key = `${size}|${width | 0}|${text}`;
	const hit = clipCache.get(key);
	if (hit !== undefined) return hit;
	let out = text;
	if (measure(out, size) > width) {
		while (out.length > 1 && measure(out + '…', size) > width) out = out.slice(0, -1);
		out += '…';
	}
	if (clipCache.size > 4000) clipCache.clear();
	clipCache.set(key, out);
	return out;
}

function layout(n) { return layoutOf(n, app.images[n.id]); }

function nodeBox(n) {
	const L = layout(n);
	return { x: n.pos[0], y: n.pos[1], w: L.w, h: L.h, L };
}

// Where a node's ports are, in graph units; null for a port it does not have.
function inPort(n, name, L = layout(n)) {
	const y = L.inputs.get(name);
	return y === undefined ? null : [n.pos[0], n.pos[1] + y];
}
function outPort(n, name, L = layout(n)) {
	const y = L.outputs.get(name);
	return y === undefined ? null : [n.pos[0] + L.w, n.pos[1] + y];
}

export class GraphCanvas extends three.Widget {
	constructor() {
		super();
		this.size = [800, 600];
		this.view = { x: 0, y: 0, zoom: 1 };
		this.drag = null;       // what the pointer is doing, see pointer()
		this.pointerAt = [0, 0];
		this.hoverPort = null;
		this.version = 0;       // bumped by anything that changes what is drawn
		// Frame the graph whenever the canvas changes size, until somebody
		// pans or zooms it themselves.
		this.autoFrame = true;
	}

	setSize(size) {
		this.size = size;
		if (this.autoFrame) this.frame();
	}

	// Graph units to this drawing's points, and back.
	toScreen([x, y]) { const v = this.view; return [(x - v.x) * v.zoom, (y - v.y) * v.zoom]; }
	toGraph([x, y]) { const v = this.view; return [x / v.zoom + v.x, y / v.zoom + v.y]; }

	// Where a new node goes: the middle of what is on screen.
	center() { return this.toGraph([this.size[0] / 2 - NODE_W / 2, this.size[1] / 3]); }

	// Fit the whole graph on screen.
	frame() {
		const nodes = app.doc.nodes;
		if (nodes.length === 0) { this.view = { x: 0, y: 0, zoom: 1 }; return; }
		let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
		for (const n of nodes) {
			const b = nodeBox(n);
			x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
			x1 = Math.max(x1, b.x + b.w); y1 = Math.max(y1, b.y + b.h);
		}
		const pad = 40;
		const zoom = Math.min(1.2, Math.max(0.25, Math.min(this.size[0] / (x1 - x0 + 2 * pad), this.size[1] / (y1 - y0 + 2 * pad))));
		this.view = {
			zoom,
			x: (x0 + x1) / 2 - this.size[0] / 2 / zoom,
			y: (y0 + y1) / 2 - this.size[1] / 2 / zoom,
		};
	}

	render() {
		// Clipped, or a node dragged past the edge would paint over the panels.
		return new Clip({ size: this.size },
			new Stack({ size: this.size },
				new Drawing({ key: 'graphCanvas', size: this.size, ops: this.ops(), onPointer: (e) => this.pointer(e) }),
				...this.editors()));
	}

	// Whether a node's editors are live widgets right now: close enough to use,
	// and not under a node drawn after it (a widget would paint over that card).
	editable(n, L) {
		if (this.view.zoom < EDIT_ZOOM || L.fields.length === 0) return false;
		const nodes = app.doc.nodes;
		const b = { x0: n.pos[0], y0: n.pos[1], x1: n.pos[0] + L.w, y1: n.pos[1] + L.h };
		for (let k = nodes.indexOf(n) + 1; k < nodes.length; k++) {
			const m = nodes[k];
			const M = layout(m);
			if (m.pos[0] < b.x1 && m.pos[0] + M.w > b.x0 && m.pos[1] < b.y1 && m.pos[1] + M.h > b.y0) return false;
		}
		return true;
	}

	editors() {
		const z = this.view.zoom;
		const [W, H] = this.size;
		const out = [];
		for (const n of app.doc.nodes) {
			const L = layout(n);
			if (!this.editable(n, L)) continue;
			for (const f of L.fields) {
				const [sx, sy] = this.toScreen([n.pos[0] + f.x, n.pos[1] + f.y + LABEL_H]);
				if (sx > W || sy > H || sx + f.w * z < 0 || sy + f.h * z < 0) continue;
				const key = `field:${n.id}:${f.input.name}`;
				out.push(new Anchored({ key: `at:${key}`, h: 'start', v: 'start', margin: [sx, sy] },
					editor(n, f.input, { key, width: f.w * z, height: f.h * z, textSize: 12 * z })));
			}
		}
		return out;
	}

	// ------------------------------------------------------------------
	// Drawing

	ops() {
		const ops = [];
		const z = this.view.zoom;
		const [w, h] = this.size;
		ops.push({ op: 'rect', at: [0, 0], size: [w, h], color: THEME.canvas });

		// A dot grid every 40 units, fading out when zoomed far away.
		if (z > 0.35) {
			const step = 40 * z;
			const ox = -((this.view.x * z) % step), oy = -((this.view.y * z) % step);
			for (let x = ox; x < w; x += step) ops.push({ op: 'line', from: [x, 0], to: [x, h], thickness: 1, color: THEME.grid });
			for (let y = oy; y < h; y += step) ops.push({ op: 'line', from: [0, y], to: [w, y], thickness: 1, color: THEME.grid });
		}

		const doc = app.doc;
		// Wires under the nodes.
		for (const n of doc.nodes) {
			const def = defOf(n.type);
			for (const input of def.inputs) {
				const w0 = doc.wire(n.id, input.name);
				const src = w0 && doc.node(w0.node);
				const from = src && outPort(src, w0.port);
				const to = from && inPort(n, input.name);
				if (to) this.curve(ops, this.toScreen(from), this.toScreen(to), typeColor(input.type), 2.2);
			}
		}
		for (const n of doc.nodes) this.drawNode(ops, n);

		// The wire being dragged.
		const d = this.drag;
		if (d && d.mode === 'wire') {
			const color = typeColor(d.type);
			if (d.fromOut) this.curve(ops, this.toScreen(d.anchor), this.pointerAt, color, 2);
			else this.curve(ops, this.pointerAt, this.toScreen(d.anchor), color, 2);
		}
		if (app.doc.nodes.length === 0) {
			ops.push({ op: 'text', at: [w / 2 - 150, h / 2 - 10], text: 'Add nodes from the list, or open a template', size: 14, color: THEME.dim });
		}
		return ops;
	}

	curve(ops, a, b, color, thickness) {
		const dx = Math.max(40 * this.view.zoom, Math.abs(b[0] - a[0]) * 0.5);
		const c1 = [a[0] + dx, a[1]], c2 = [b[0] - dx, b[1]];
		const steps = 20;
		let prev = a;
		for (let i = 1; i <= steps; i++) {
			const t = i / steps, u = 1 - t;
			const p = [
				u * u * u * a[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * b[0],
				u * u * u * a[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * b[1],
			];
			ops.push({ op: 'line', from: prev, to: p, thickness: thickness * Math.max(this.view.zoom, 0.6), color });
			prev = p;
		}
	}

	drawNode(ops, n) {
		const z = this.view.zoom;
		const b = nodeBox(n);
		const L = b.L;
		const [sx, sy] = this.toScreen([b.x, b.y]);
		const sw = b.w * z, sh = b.h * z;
		if (sx > this.size[0] || sy > this.size[1] || sx + sw < 0 || sy + sh < 0) return;
		const def = L.def;
		const state = app.nodeState[n.id];
		const selected = app.selected === n.id;
		const border = state === 'running' ? THEME.running
			: state === 'error' ? THEME.error
			: selected ? THEME.accent
			: THEME.border;
		const r = 6 * z;

		ops.push({ op: 'shadow', at: [sx, sy + 3 * z], size: [sw, sh], blur: 10 * z, color: [0, 0, 0, 0.45], radius: r });
		ops.push({ op: 'rect', at: [sx, sy], size: [sw, sh], radius: r, color: THEME.node, borderColor: border, borderWidth: selected || state === 'running' || state === 'error' ? 2 : 1 });
		ops.push({ op: 'rect', at: [sx + 1, sy + 1], size: [sw - 2, NODE_HEAD * z - 1], radius: [r, r, 0, 0], color: def.missing ? THEME.error : THEME.nodeHeader });
		const ts = TEXT * z;
		if (z > 0.3) {
			ops.push({ op: 'text', at: [sx + 10 * z, sy + 6 * z], text: clip(def.title, (NODE_W - 70), TEXT) , size: ts + z, color: THEME.text });
			const tag = state === 'cached' ? 'cached' : n.id;
			ops.push({ op: 'text', at: [sx + sw - measure(tag, 10) * z - 8 * z, sy + 8 * z], text: tag, size: 10 * z, color: THEME.dim });
		}

		// Progress along the header's bottom edge while it runs.
		const p = app.progress;
		if (state === 'running' && p && p.node === n.id && p.total) {
			ops.push({ op: 'rect', at: [sx + 1, sy + NODE_HEAD * z - 3 * z], size: [(sw - 2) * p.step / p.total, 3 * z], color: THEME.running });
		}

		const port = (name, at, type, filled) => {
			const [px, py] = this.toScreen(at);
			const color = typeColor(type);
			const hovered = this.hoverPort && this.hoverPort.node === n.id && this.hoverPort.name === name;
			ops.push({ op: 'circle', center: [px, py], radius: (hovered ? PORT_R + 2 : PORT_R) * z, color: filled ? color : [0, 0, 0, 0], borderColor: color, borderWidth: 1.5 * z });
			return [px, py];
		};

		for (const input of L.sockets) {
			const wired = isWire(n.params[input.name]);
			const [px, py] = port(input.name, inPort(n, input.name, L), input.type, wired || !isSetting(input));
			if (z <= 0.3) continue;
			const label = input.name + (input.optional || isSetting(input) ? '' : ' *');
			ops.push({ op: 'text', at: [px + 10 * z, py - ts * 0.62], text: label, size: ts, color: THEME.text });
		}
		for (const output of def.outputs) {
			const [px, py] = port(output.name, outPort(n, output.name, L), output.type, true);
			if (z <= 0.3) continue;
			const label = output.kind ? `${output.name} (${output.kind})` : output.name;
			ops.push({ op: 'text', at: [px - (measure(label, TEXT) + 10) * z, py - ts * 0.62], text: label, size: ts, color: THEME.text });
		}

		// Settings: a port each, a label over the box, and - when the editor is
		// not a live widget - the box and its value drawn here.
		const live = this.editable(n, L);
		for (const f of L.fields) {
			port(f.input.name, inPort(n, f.input.name, L), f.input.type, false);
			if (z <= 0.3) continue;
			const [fx, fy] = this.toScreen([n.pos[0] + f.x, n.pos[1] + f.y]);
			if (f.input.type !== 'BOOL') {
				ops.push({ op: 'text', at: [fx + 2 * z, fy], text: clip(f.input.name, f.w - 4, 10), size: 10 * z, color: THEME.dim });
			}
			if (live) continue;
			const by = fy + LABEL_H * z;
			ops.push({ op: 'rect', at: [fx, by], size: [f.w * z, f.h * z], radius: 4 * z, color: THEME.body, borderColor: THEME.border, borderWidth: 1 });
			const v = n.params[f.input.name] ?? f.input.default;
			const text = f.input.type === 'BOOL' ? `${f.input.name}: ${v ? 'on' : 'off'}` : shown(v);
			ops.push({ op: 'text', at: [fx + 6 * z, by + 6 * z], text: clip(text, f.w - 12, TEXT), size: ts, color: THEME.dim });
		}

		if (L.thumb) {
			const t = L.thumb;
			const img = app.images[n.id];
			const [tx, ty] = this.toScreen([n.pos[0] + t.x, n.pos[1] + t.y]);
			ops.push({ op: 'image', at: [tx, ty], size: [t.w * z, t.h * z], texture: img.tex, radius: 3 * z });
		}
	}

	// ------------------------------------------------------------------
	// Hit testing, in screen points

	portAt(p) {
		const z = this.view.zoom;
		const near = (q) => q && Math.hypot(q[0] - p[0], q[1] - p[1]) <= HIT * Math.max(z, 0.7);
		for (let k = app.doc.nodes.length - 1; k >= 0; k--) {
			const n = app.doc.nodes[k];
			const L = layout(n);
			for (const input of L.def.inputs) {
				const at = inPort(n, input.name, L);
				if (near(at && this.toScreen(at))) return { node: n.id, input: input.name, name: input.name, type: input.type, anchor: at };
			}
			for (const output of L.def.outputs) {
				const at = outPort(n, output.name, L);
				if (near(at && this.toScreen(at))) return { node: n.id, output: output.name, name: output.name, type: output.type, anchor: at };
			}
		}
		return null;
	}

	nodeAt(p) {
		const g = this.toGraph(p);
		for (let k = app.doc.nodes.length - 1; k >= 0; k--) {
			const n = app.doc.nodes[k];
			const b = nodeBox(n);
			if (g[0] >= b.x && g[0] <= b.x + b.w && g[1] >= b.y && g[1] <= b.y + b.h) return n;
		}
		return null;
	}

	// ------------------------------------------------------------------
	// The pointer

	pointer(e) {
		const p = [e.x, e.y];
		this.pointerAt = p;
		switch (e.phase) {
			case 'wheel': return this.zoomAt(p, e.wheel);
			case 'down': return this.press(p, e);
			case 'move': return this.move(p, e);
			case 'up': return this.release(p);
		}
	}

	zoomAt(p, wheel) {
		if (!wheel) return;
		this.autoFrame = false;
		const before = this.toGraph(p);
		// One wheel notch is 1 on X11 but about 15 points on Wayland, so a notch
		// counts as one step whatever its size; a trackpad's small deltas still
		// zoom smoothly in proportion.
		const step = Math.max(-1, Math.min(1, wheel));
		const zoom = Math.min(2.5, Math.max(0.2, this.view.zoom * Math.pow(ZOOM_STEP, step)));
		this.view = { zoom, x: before[0] - p[0] / zoom, y: before[1] - p[1] / zoom };
	}

	press(p, e) {
		const pan = e.button !== 0 || e.buttons & 6;
		if (pan) {
			this.drag = { mode: 'pan', from: p, view: { ...this.view } };
			return;
		}
		const port = this.portAt(p);
		if (port && port.output) {
			this.drag = { mode: 'wire', fromOut: true, node: port.node, port: port.output, type: port.type, anchor: port.anchor };
			return;
		}
		if (port && port.input) {
			const w = app.doc.wire(port.node, port.input);
			if (w) {
				// Picking up a connected input carries its wire away from it.
				const src = app.doc.node(w.node);
				app.doc.disconnect(port.node, port.input);
				app.changed();
				this.drag = { mode: 'wire', fromOut: true, node: w.node, port: w.port, type: port.type, anchor: outPort(src, w.port) };
			} else {
				this.drag = { mode: 'wire', fromOut: false, node: port.node, input: port.input, type: port.type, anchor: port.anchor };
			}
			return;
		}
		const n = this.nodeAt(p);
		if (n) {
			app.select(n.id);
			// Brought to the front, so it draws and hit-tests over the others.
			const doc = app.doc;
			doc.nodes = [...doc.nodes.filter((m) => m !== n), n];
			const g = this.toGraph(p);
			this.drag = { mode: 'move', id: n.id, dx: g[0] - n.pos[0], dy: g[1] - n.pos[1] };
			return;
		}
		app.select(null);
		this.drag = { mode: 'pan', from: p, view: { ...this.view } };
	}

	move(p) {
		const d = this.drag;
		if (!d) {
			const port = this.portAt(p);
			const key = port ? `${port.node}.${port.input ?? port.output}` : null;
			if ((this.hoverPort?.key ?? null) !== key) this.hoverPort = port ? { ...port, key } : null;
			return;
		}
		if (d.mode === 'pan') {
			this.autoFrame = false;
			const z = this.view.zoom;
			this.view = { zoom: z, x: d.view.x - (p[0] - d.from[0]) / z, y: d.view.y - (p[1] - d.from[1]) / z };
		} else if (d.mode === 'move') {
			const n = app.doc.node(d.id);
			if (!n) return;
			const g = this.toGraph(p);
			n.pos = [Math.round(g[0] - d.dx), Math.round(g[1] - d.dy)];
			app.doc.dirty = true;
			this.version++;
		} else if (d.mode === 'wire') {
			const port = this.portAt(p);
			this.hoverPort = port && this.fits(d, port) ? { ...port, key: `${port.node}.${port.input ?? port.output}` } : null;
			this.version++;
		}
	}

	// Whether a wire being dragged may end on `port`.
	fits(d, port) {
		if (d.fromOut) return !!port.input && accepts(port.type, d.type);
		return !!port.output && accepts(d.type, port.type);
	}

	release(p) {
		const d = this.drag;
		this.drag = null;
		this.hoverPort = null;
		if (!d || d.mode !== 'wire') return;
		const port = this.portAt(p);
		if (!port) return;
		const problem = d.fromOut
			? (port.input ? app.doc.connect(d.node, d.port, port.node, port.input) : 'drop a wire on an input')
			: (port.output ? app.doc.connect(port.node, port.output, d.node, d.input) : 'drop a wire on an output');
		if (problem) app.say(problem, true);
		app.changed();
	}
}
