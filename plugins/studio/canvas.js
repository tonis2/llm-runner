// The graph, drawn: nodes as cards with their ports, wires as curves, and the
// pointer - drag a node to move it, drag from a port to wire it, drag the
// background (or with the middle/right button anywhere) to pan, and the wheel
// zooms around the pointer.
//
// One `Drawing` whose ops are rebuilt on every render. The widget layer diffs
// them into a single patch, so moving a node is one list of drawings a frame.

import { app } from './app.js';
import { Doc, defOf, isSetting, isWire } from './doc.js';
import { accepts } from '../lib/graph/types.js';
import {
	THEME, typeColor, NODE_W, NODE_HEAD, NODE_ROW, NODE_PAD, PORT_R,
} from './theme.js';

const { Drawing, Clip } = three.ui;

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

// A node's picture below its rows, if it has one.
function thumbOf(id) {
	const img = app.images[id];
	if (!img) return null;
	const w = NODE_W - 2 * NODE_PAD;
	return { ...img, w, h: Math.min(w * img.height / img.width, 320) };
}

function nodeBox(n) {
	const def = defOf(n.type);
	const thumb = thumbOf(n.id);
	return { x: n.pos[0], y: n.pos[1], w: NODE_W, h: Doc.height(def, thumb ? thumb.h + NODE_PAD : 0), def, thumb };
}

function inPort(n, i) { return [n.pos[0], n.pos[1] + NODE_HEAD + i * NODE_ROW + NODE_ROW / 2]; }
function outPort(n, i) { return [n.pos[0] + NODE_W, n.pos[1] + NODE_HEAD + i * NODE_ROW + NODE_ROW / 2]; }

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
			new Drawing({ key: 'graphCanvas', size: this.size, ops: this.ops(), onPointer: (e) => this.pointer(e) }));
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
			def.inputs.forEach((input, i) => {
				const wv = n.params[input.name];
				if (!isWire(wv)) return;
				const w0 = doc.wire(n.id, input.name);
				const src = doc.node(w0.node);
				if (!src) return;
				const oi = defOf(src.type).outputs.findIndex((o) => o.name === w0.port);
				if (oi < 0) return;
				this.curve(ops, this.toScreen(outPort(src, oi)), this.toScreen(inPort(n, i)), typeColor(input.type), 2.2);
			});
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
		const [sx, sy] = this.toScreen([b.x, b.y]);
		const sw = b.w * z, sh = b.h * z;
		if (sx > this.size[0] || sy > this.size[1] || sx + sw < 0 || sy + sh < 0) return;
		const def = b.def;
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

		def.inputs.forEach((input, i) => {
			const [px, py] = this.toScreen(inPort(n, i));
			const wired = isWire(n.params[input.name]);
			const color = typeColor(input.type);
			const hovered = this.hoverPort && this.hoverPort.node === n.id && this.hoverPort.input === input.name;
			ops.push({ op: 'circle', center: [px, py], radius: (hovered ? PORT_R + 2 : PORT_R) * z, color: wired || !isSetting(input) ? color : [0, 0, 0, 0], borderColor: color, borderWidth: 1.5 * z });
			if (z <= 0.3) return;
			const label = input.name + (input.optional || isSetting(input) ? '' : ' *');
			ops.push({ op: 'text', at: [px + 10 * z, py - ts * 0.62], text: label, size: ts, color: THEME.text });
			if (isSetting(input) && !wired) {
				const v = n.params[input.name] ?? input.default;
				const lw = measure(label, TEXT) + 18;
				const room = NODE_W / 2 - 10 + (def.outputs.length > i ? 0 : NODE_W / 2 - 10) - lw;
				if (room > 20) ops.push({ op: 'text', at: [px + lw * z, py - ts * 0.62], text: clip(shown(v), room, TEXT), size: ts, color: THEME.dim });
			}
		});
		def.outputs.forEach((output, i) => {
			const [px, py] = this.toScreen(outPort(n, i));
			const color = typeColor(output.type);
			ops.push({ op: 'circle', center: [px, py], radius: PORT_R * z, color, borderColor: color, borderWidth: 1.5 * z });
			if (z <= 0.3) return;
			const label = output.kind ? `${output.name} (${output.kind})` : output.name;
			ops.push({ op: 'text', at: [px - (measure(label, TEXT) + 10) * z, py - ts * 0.62], text: label, size: ts, color: THEME.text });
		});

		if (b.thumb) {
			const ty = sy + (NODE_HEAD + Doc.rows(def) * NODE_ROW + NODE_PAD / 2) * z;
			ops.push({ op: 'image', at: [sx + NODE_PAD * z, ty], size: [b.thumb.w * z, b.thumb.h * z], texture: b.thumb.tex, radius: 3 * z });
		}
	}

	// ------------------------------------------------------------------
	// Hit testing, in screen points

	portAt(p) {
		const z = this.view.zoom;
		const near = (q) => Math.hypot(q[0] - p[0], q[1] - p[1]) <= HIT * Math.max(z, 0.7);
		for (let k = app.doc.nodes.length - 1; k >= 0; k--) {
			const n = app.doc.nodes[k];
			const def = defOf(n.type);
			for (let i = 0; i < def.inputs.length; i++) {
				if (near(this.toScreen(inPort(n, i)))) return { node: n.id, input: def.inputs[i].name, type: def.inputs[i].type, anchor: inPort(n, i) };
			}
			for (let i = 0; i < def.outputs.length; i++) {
				if (near(this.toScreen(outPort(n, i)))) return { node: n.id, output: def.outputs[i].name, type: def.outputs[i].type, anchor: outPort(n, i) };
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
		const zoom = Math.min(2.5, Math.max(0.2, this.view.zoom * Math.exp(wheel * 0.12)));
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
				const oi = defOf(src.type).outputs.findIndex((o) => o.name === w.port);
				app.doc.disconnect(port.node, port.input);
				app.changed();
				this.drag = { mode: 'wire', fromOut: true, node: w.node, port: w.port, type: port.type, anchor: outPort(src, oi) };
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
