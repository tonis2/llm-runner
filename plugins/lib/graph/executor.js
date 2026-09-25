// Running a graph: nodes wired output to input, evaluated in dependency order.
//
// A graph is JSON:
//
//   {
//     "nodes": [
//       { "id": "te",  "type": "core.text_encoder", "params": { "path": "Qwen3-8B-Q8_0.gguf" } },
//       { "id": "enc", "type": "flux.text_encode",  "params": { "encoder": { "from": "te.encoder" }, "prompt": "a fox" } },
//       ...
//     ],
//     "edges": [ { "from": ["te", "encoder"], "to": ["enc", "encoder"] } ]
//   }
//
// A wire is either an edge or an input written as { "from": "node.port" };
// `nodes` may also be an object keyed by id. Only what the sinks (save,
// preview) need is run.
//
// Every node's result is keyed by its type, its settings and the keys of what
// feeds it. With `keep`, results stay between runs and a node whose key has not
// changed is not run again - a new seed re-samples and re-decodes but does not
// re-encode the prompt or reload the model. Without it, a value is disposed as
// soon as the last node that reads it has run, which is what keeps a one-shot
// run's VRAM down: the DiT is gone before the VAE decodes.

import { llm } from '../llm.js';
import { nodeType } from './registry.js';
import { PRIMITIVES, coerce, accepts } from './types.js';

export class Cancelled extends Error {
	constructor() { super('cancelled'); this.name = 'Cancelled'; }
}

// A wire to `node`'s `port`, for building a graph in code.
export function link(node, port) { return { from: `${node}.${port}` }; }

// cyrb53: a quick 53-bit string hash, as hex.
function hash(text) {
	let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
	for (let i = 0; i < text.length; i++) {
		const c = text.charCodeAt(i);
		h1 = Math.imul(h1 ^ c, 2654435761);
		h2 = Math.imul(h2 ^ c, 1597334677);
	}
	h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
	h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
	return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16);
}

function hashBytes(bytes) {
	let h = 0x811c9dc5;
	for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 16777619);
	return (h >>> 0).toString(16);
}

// A stable text for a setting's value: object keys sorted, byte arrays hashed
// rather than spelled out (an image handed over by the server is one).
function keyText(value) {
	if (value === undefined) return 'u';
	if (value === null || typeof value !== 'object') return JSON.stringify(value);
	if (ArrayBuffer.isView(value)) {
		const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
		return `bytes:${bytes.length}:${hashBytes(bytes)}`;
	}
	if (Array.isArray(value)) return `[${value.map(keyText).join(',')}]`;
	return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${keyText(value[k])}`).join(',')}}`;
}

function isWire(value) {
	return value !== null && typeof value === 'object' && !Array.isArray(value) && typeof value.from === 'string' && Object.keys(value).length === 1;
}

function splitWire(text, where) {
	const dot = text.lastIndexOf('.');
	if (dot <= 0) throw new Error(`${where}: a wire is "node.port", not "${text}"`);
	return { node: text.slice(0, dot), port: text.slice(dot + 1) };
}

// { nodes: Map(id -> { id, type, params, wires: { input: { node, port } } }) }
export function normalizeGraph(graph) {
	if (!graph || typeof graph !== 'object') throw new Error('a graph is an object with nodes');
	const list = Array.isArray(graph.nodes)
		? graph.nodes
		: Object.entries(graph.nodes ?? {}).map(([id, n]) => ({ id, ...n }));
	const nodes = new Map();
	for (const n of list) {
		if (!n.id) throw new Error(`a ${n.type ?? 'node'} has no id`);
		if (nodes.has(n.id)) throw new Error(`two nodes are called ${n.id}`);
		const params = {};
		const wires = {};
		for (const [k, v] of Object.entries(n.params ?? n.inputs ?? {})) {
			if (isWire(v)) wires[k] = splitWire(v.from, `${n.id}.${k}`);
			else params[k] = v;
		}
		nodes.set(n.id, { id: n.id, type: n.type, params, wires, pos: n.pos });
	}
	for (const e of graph.edges ?? []) {
		const [fromNode, fromPort] = e.from;
		const [toNode, toInput] = e.to;
		const n = nodes.get(toNode);
		if (!n) throw new Error(`an edge goes to ${toNode}, which is not a node`);
		n.wires[toInput] = { node: fromNode, port: fromPort };
	}
	return { nodes };
}

function disposeValues(outputs, seen = new Set()) {
	for (const v of Object.values(outputs ?? {})) {
		if (v && typeof v.dispose === 'function' && !seen.has(v)) {
			seen.add(v);
			v.dispose();
		}
	}
}

export class Executor {
	// keep: results stay between runs (a server, the editor). Otherwise each
	// value lives only as long as something still has to read it.
	// yieldFrame: a function returning a promise that settles when the host
	// has drawn a frame (`three.nextFrame` in the studio). Awaited between
	// nodes and at every progress report, so a window stays live through a
	// run; left out, nothing waits.
	constructor({ keep = false, log = (line) => llm.print(line), yieldFrame = null } = {}) {
		this.keep = keep;
		this.log = log;
		this.yieldFrame = yieldFrame;
		this.cache = new Map(); // key -> { outputs, type }
	}

	// Run what `targets` (node ids; the graph's sinks when left out) need.
	// Returns { results: { sinkId: what it returned }, ran: [ids], cached: [ids] }.
	async run(graph, { targets = null, onProgress = null, cancelled = null } = {}) {
		const g = normalizeGraph(graph);
		const defs = new Map();
		for (const n of g.nodes.values()) defs.set(n.id, nodeType(n.type));
		const goal = targets ?? [...g.nodes.values()].filter((n) => defs.get(n.id).output).map((n) => n.id);
		if (goal.length === 0) throw new Error('the graph has nothing to run for: no save or preview node');

		// Dependency order, inputs in the order the node declares them.
		const order = [];
		const state = new Map();
		const visit = (id, from) => {
			const n = g.nodes.get(id);
			if (!n) throw new Error(`${from ?? 'the run'} is wired to ${id}, which is not a node`);
			const s = state.get(id);
			if (s === 'done') return;
			if (s === 'visiting') throw new Error(`the graph has a cycle through ${id}`);
			state.set(id, 'visiting');
			for (const input of defs.get(id).inputs) {
				const w = n.wires[input.name];
				if (w) visit(w.node, `${id}.${input.name}`);
			}
			state.set(id, 'done');
			order.push(id);
		};
		for (const id of goal) visit(id);

		// Inputs checked and settings resolved; then each node's key.
		const plan = new Map();
		for (const id of order) {
			const n = g.nodes.get(id);
			const def = defs.get(id);
			const known = new Set(def.inputs.map((p) => p.name));
			for (const k of [...Object.keys(n.params), ...Object.keys(n.wires)]) {
				if (!known.has(k)) throw new Error(`${id} (${n.type}) has no input called ${k}; it has ${[...known].join(', ') || 'none'}`);
			}
			const values = {};
			const wired = {};
			const parts = [n.type];
			for (const input of def.inputs) {
				const where = `${id}.${input.name}`;
				const w = n.wires[input.name];
				if (w) {
					const src = defs.get(w.node);
					const out = src.outputs.find((o) => o.name === w.port);
					if (!out) throw new Error(`${where} is wired to ${w.node}.${w.port}, but ${src.type} has no output ${w.port}`);
					if (!accepts(input.type, out.type)) throw new Error(`${where} takes ${input.type}, and ${w.node}.${w.port} is ${out.type}`);
					// A latent's format is part of its type where both ends name one.
					if (input.type === 'LATENT' && input.kind && out.kind && input.kind !== out.kind) {
						throw new Error(`${where} takes ${input.kind} latents, and ${w.node}.${w.port} makes ${out.kind}`);
					}
					wired[input.name] = w;
					parts.push(`${input.name}<${plan.get(w.node).key}.${w.port}`);
					continue;
				}
				let v = n.params[input.name];
				if (v === undefined || v === null) v = input.default;
				if (v === undefined) {
					if (!input.optional) throw new Error(`${where} (${input.type}) is not set and not wired`);
					continue;
				}
				if (!PRIMITIVES.has(input.type) && input.type !== 'ANY') {
					throw new Error(`${where} is a ${input.type}: wire it from a node that makes one`);
				}
				values[input.name] = coerce(input, v, where);
				parts.push(`${input.name}=${keyText(values[input.name])}`);
			}
			plan.set(id, { def, values, wired, key: hash(parts.join('|')) });
		}

		// Results of another graph, or of settings since changed, go first - a
		// new LoRA set's DiT should not load beside the old one.
		const wanted = new Set([...plan.values()].map((p) => p.key));
		for (const [key, entry] of this.cache) {
			if (!wanted.has(key)) {
				disposeValues(entry.outputs);
				this.cache.delete(key);
			}
		}

		// How many reads of each result are still to come.
		const reads = new Map();
		for (const p of plan.values()) {
			for (const w of Object.values(p.wired)) {
				const k = plan.get(w.node).key;
				reads.set(k, (reads.get(k) ?? 0) + 1);
			}
		}

		const results = {};
		const ran = [];
		const cached = [];
		const checkCancel = () => { if (cancelled && cancelled()) throw new Cancelled(); };
		const pause = async () => {
			if (this.yieldFrame) await this.yieldFrame();
			checkCancel();
		};
		try {
			for (const id of order) {
				await pause();
				const p = plan.get(id);
				const n = g.nodes.get(id);
				let outputs;
				if (!p.def.output && this.cache.has(p.key)) {
					outputs = this.cache.get(p.key).outputs;
					cached.push(id);
				} else {
					const inputs = { ...p.values };
					for (const [name, w] of Object.entries(p.wired)) {
						const src = this.cache.get(plan.get(w.node).key);
						inputs[name] = src.outputs[w.port];
					}
					const ctx = {
						id,
						type: n.type,
						keep: this.keep,
						log: this.log,
						cancelled: checkCancel,
						// Awaited by a node between steps: reports, lets a frame
						// be drawn, and throws Cancelled if the run was stopped.
						progress: async (step, total, extra = {}) => {
							if (onProgress) await onProgress({ node: id, step, total, ...extra });
							return pause();
						},
					};
					const t0 = llm.now();
					if (onProgress) onProgress({ node: id, start: true });
					try {
						outputs = (await p.def.run(inputs, ctx)) ?? {};
					} catch (e) {
						// Which node failed, for whoever shows the graph.
						if (e && typeof e === 'object' && e.node === undefined) e.node = id;
						throw e;
					}
					ran.push(id);
					if (onProgress) onProgress({ node: id, done: true, ms: llm.now() - t0 });
					if (p.def.output) results[id] = outputs;
					else this.cache.set(p.key, { outputs, type: n.type });
				}
				// This node's reads are done: without `keep`, anything nobody else
				// will read goes now.
				for (const w of Object.values(p.wired)) {
					const k = plan.get(w.node).key;
					const left = reads.get(k) - 1;
					reads.set(k, left);
					if (left === 0 && !this.keep && this.cache.has(k)) {
						disposeValues(this.cache.get(k).outputs);
						this.cache.delete(k);
					}
				}
			}
		} finally {
			if (!this.keep) this.clear();
		}
		return { results, ran, cached };
	}

	// Dispose every kept result.
	clear() {
		for (const entry of this.cache.values()) disposeValues(entry.outputs);
		this.cache.clear();
	}
}
