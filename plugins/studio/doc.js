// The graph being edited: nodes with settings, positions and wires, in the same
// shape a graph file has - so saving is JSON.stringify and running is handing it
// to the Executor.
//
//   { nodes: [ { id, type, params: { input: value | { from: 'node.port' } }, pos: [x, y] } ] }
//
// Everything that changes the graph goes through here, and the checks a wire
// has to pass - types, latent formats, cycles - are made when it is connected
// rather than when the graph runs.

import { llm } from '../lib/llm.js';
import { catalogue } from '../lib/graph/registry.js';
import { accepts, PRIMITIVES } from '../lib/graph/types.js';
import { vaeKind, KIND_FORMAT } from '../lib/latents.js';
import { NODE_W } from './theme.js';

let types = {};
export function refreshTypes() { types = catalogue(); return types; }
export function allTypes() { return types; }

// The definition of a node's type, or a stand-in for one whose plugin is not
// loaded - it keeps its settings and draws as missing.
export function defOf(type) {
	return types[type] ?? { title: `${type} (missing)`, category: 'missing', inputs: [], outputs: [], missing: true };
}

export function isWire(v) {
	return v !== null && typeof v === 'object' && !Array.isArray(v) && typeof v.from === 'string';
}

function splitWire(text) {
	const dot = text.lastIndexOf('.');
	return { node: text.slice(0, dot), port: text.slice(dot + 1) };
}

// The format a VAE file decodes, read from its tensor names once per path.
const vaeFormats = new Map();
function vaeFormatOf(path) {
	if (typeof path !== 'string' || !path) return null;
	if (vaeFormats.has(path)) return vaeFormats.get(path);
	let format = null;
	try {
		if (llm.exists(path)) {
			const m = llm.open(path);
			format = KIND_FORMAT[vaeKind(m)] ?? null;
			m.close();
		}
	} catch { format = null; }
	vaeFormats.set(path, format);
	return format;
}

export class Doc {
	constructor(graph = { nodes: [] }) {
		this.nodes = [];
		this.path = null;
		this.dirty = false;
		this.load(graph);
	}

	load(graph) {
		const list = Array.isArray(graph.nodes)
			? graph.nodes
			: Object.entries(graph.nodes ?? {}).map(([id, n]) => ({ id, ...n }));
		this.nodes = list.map((n) => ({
			id: String(n.id),
			type: n.type,
			params: JSON.parse(JSON.stringify(n.params ?? n.inputs ?? {})),
			pos: Array.isArray(n.pos) ? [n.pos[0], n.pos[1]] : null,
		}));
		for (const e of graph.edges ?? []) {
			const n = this.node(e.to[0]);
			if (n) n.params[e.to[1]] = { from: `${e.from[0]}.${e.from[1]}` };
		}
		if (this.nodes.some((n) => !n.pos)) this.autoLayout();
		this.dirty = false;
	}

	toJSON() {
		return { nodes: this.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params, pos: n.pos.map(Math.round) })) };
	}

	// For the Executor: the same nodes without positions.
	graph() {
		return { nodes: this.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params })) };
	}

	node(id) { return this.nodes.find((n) => n.id === id) ?? null; }

	// A node's input wire as { node, port }, or null.
	wire(id, input) {
		const n = this.node(id);
		const v = n && n.params[input];
		return isWire(v) ? splitWire(v.from) : null;
	}

	freshId(type) {
		const base = (type.split('.').pop() || 'node').replace(/[^a-z0-9_]/gi, '_');
		if (!this.node(base)) return base;
		for (let i = 2; ; i++) if (!this.node(`${base}${i}`)) return `${base}${i}`;
	}

	add(type, pos) {
		const n = { id: this.freshId(type), type, params: {}, pos: [pos[0], pos[1]] };
		this.nodes.push(n);
		this.dirty = true;
		return n;
	}

	remove(id) {
		this.nodes = this.nodes.filter((n) => n.id !== id);
		for (const n of this.nodes) {
			for (const [k, v] of Object.entries(n.params)) {
				if (isWire(v) && splitWire(v.from).node === id) delete n.params[k];
			}
		}
		this.dirty = true;
	}

	// A copy of a node, offset a little, without its wires.
	duplicate(id) {
		const n = this.node(id);
		if (!n) return null;
		const params = {};
		for (const [k, v] of Object.entries(n.params)) if (!isWire(v)) params[k] = JSON.parse(JSON.stringify(v));
		const copy = { id: this.freshId(n.type), type: n.type, params, pos: [n.pos[0] + 30, n.pos[1] + 30] };
		this.nodes.push(copy);
		this.dirty = true;
		return copy;
	}

	setParam(id, input, value) {
		const n = this.node(id);
		if (!n) return;
		if (value === undefined) delete n.params[input];
		else n.params[input] = value;
		this.dirty = true;
	}

	disconnect(id, input) {
		const n = this.node(id);
		if (n && isWire(n.params[input])) {
			delete n.params[input];
			this.dirty = true;
		}
	}

	// Every node `id` reads from, directly or not.
	upstream(id, seen = new Set()) {
		const n = this.node(id);
		if (!n) return seen;
		for (const v of Object.values(n.params)) {
			if (!isWire(v)) continue;
			const src = splitWire(v.from).node;
			if (seen.has(src)) continue;
			seen.add(src);
			this.upstream(src, seen);
		}
		return seen;
	}

	// The latent format a port carries, when it can be known before running: a
	// sampler names its own, a VAE's is read from its file, and an encode
	// makes its VAE's.
	formatOf(id, port) {
		const n = this.node(id);
		if (!n) return null;
		const def = defOf(n.type);
		const out = def.outputs.find((o) => o.name === port);
		if (!out) return null;
		if (out.type === 'LATENT' && out.kind) return out.kind;
		if (out.type === 'VAE') {
			const p = n.params.path;
			return isWire(p) ? null : vaeFormatOf(p);
		}
		if (out.type === 'LATENT') {
			// A latent made from a VAE input (an encode) is in that VAE's format.
			const vaeIn = def.inputs.find((i) => i.type === 'VAE');
			const w = vaeIn && this.wire(id, vaeIn.name);
			return w ? this.formatOf(w.node, w.port) : null;
		}
		return null;
	}

	// Why the wires into `id` disagree about a latent format, or null.
	formatProblem(id) {
		const n = this.node(id);
		if (!n) return null;
		const def = defOf(n.type);
		let vaeFormat = null, latentFormat = null;
		for (const input of def.inputs) {
			const w = this.wire(id, input.name);
			if (!w) continue;
			const f = this.formatOf(w.node, w.port);
			if (!f) continue;
			if (input.type === 'LATENT' && input.kind && input.kind !== f) {
				return `${n.id}.${input.name} takes ${input.kind} latents; ${w.node} makes ${f}`;
			}
			if (input.type === 'VAE') vaeFormat = f;
			if (input.type === 'LATENT') latentFormat = f;
		}
		if (vaeFormat && latentFormat && vaeFormat !== latentFormat) {
			return `this VAE decodes ${vaeFormat} latents, and the latent wired in is ${latentFormat}`;
		}
		return null;
	}

	// Wire `src.port` into `dst.input`. Returns null, or why it cannot be.
	connect(src, port, dst, input) {
		const a = this.node(src), b = this.node(dst);
		if (!a || !b) return 'no such node';
		if (src === dst) return 'a node cannot feed itself';
		const out = defOf(a.type).outputs.find((o) => o.name === port);
		const into = defOf(b.type).inputs.find((i) => i.name === input);
		if (!out || !into) return 'no such port';
		if (!accepts(into.type, out.type)) return `${b.id}.${input} takes ${into.type}, not ${out.type}`;
		if (this.upstream(src).has(dst)) return 'that wire would make a loop';
		const before = b.params[input];
		b.params[input] = { from: `${src}.${port}` };
		// Checked with the wire in place, and every node downstream of it too:
		// connecting a VAE can make an existing latent wire wrong.
		for (const n of [b, ...this.nodes.filter((m) => this.upstream(m.id).has(dst))]) {
			const problem = this.formatProblem(n.id);
			if (problem) {
				if (before === undefined) delete b.params[input];
				else b.params[input] = before;
				return problem;
			}
		}
		this.dirty = true;
		return null;
	}

	// How tall a node is drawn, for spacing a layout; `layout.js` sets it.
	static heightOf = null;

	// Columns by depth - the longest path from something with no inputs - and
	// stacked down each column. For graphs that arrive without positions.
	autoLayout() {
		const depth = new Map();
		const depthOf = (id, trail = new Set()) => {
			if (depth.has(id)) return depth.get(id);
			if (trail.has(id)) return 0;
			trail.add(id);
			let d = 0;
			const n = this.node(id);
			for (const v of Object.values(n?.params ?? {})) {
				if (isWire(v)) d = Math.max(d, depthOf(splitWire(v.from).node, trail) + 1);
			}
			depth.set(id, d);
			return d;
		};
		const columns = [];
		for (const n of this.nodes) {
			const d = depthOf(n.id);
			(columns[d] ??= []).push(n);
		}
		columns.forEach((col, d) => {
			let y = 40;
			for (const n of col ?? []) {
				n.pos = [40 + d * (NODE_W + 70), y];
				y += (Doc.heightOf ? Doc.heightOf(n) : 160) + 40;
			}
		});
	}
}

// Whether an input shows a widget (a setting) rather than only a port.
export function isSetting(input) { return PRIMITIVES.has(input.type); }
