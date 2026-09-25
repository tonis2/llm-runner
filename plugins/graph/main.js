// Runs a graph file with every installed plugin's nodes.
//
//   llm-runner run flux-t2i.graph.json
//   llm-runner run flux-t2i.graph.json prompt.prompt="a lighthouse" sample.seed=7
//   llm-runner graph list=true                        (the node types there are)
//   llm-runner graph --server --port 7860             (POST /graph, GET /nodes)
//
// `node.input=value` sets a node's setting over the file's. See
// `lib/graph/executor.js` for the file's shape.

import { llm, image } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { Executor } from '../lib/graph/executor.js';
import { catalogue } from '../lib/graph/registry.js';
import { loadPlugins, readSettings } from '../lib/graph/plugins.js';

const state = { plugins: [], settings: {}, server: null };

function readGraph(config) {
	const g = config.graph;
	if (!g) throw new Error('no graph: pass graph=file.json (or run it as `llm-runner run file.json`)');
	const graph = JSON.parse(typeof g === 'string' ? llm.readText(g) : JSON.stringify(g));
	if (!Array.isArray(graph.nodes)) graph.nodes = Object.entries(graph.nodes ?? {}).map(([id, n]) => ({ id, ...n }));
	// node.input=value from the command line (or beside `graph` in a request).
	const byId = new Map(graph.nodes.map((n) => [n.id, n]));
	for (const [key, value] of Object.entries(config)) {
		const dot = key.indexOf('.');
		if (dot <= 0) continue;
		const node = byId.get(key.slice(0, dot));
		if (!node) {
			llm.print(`llm-runner: ${key}: the graph has no node ${key.slice(0, dot)}`);
			continue;
		}
		node.params = { ...(node.params ?? node.inputs ?? {}), [key.slice(dot + 1)]: value };
		delete node.inputs;
	}
	return graph;
}

// Results with any image turned into a size (on the command line) or a PNG
// in base64 (over HTTP).
function describe(results, encode) {
	const out = {};
	for (const [id, r] of Object.entries(results)) {
		out[id] = {};
		for (const [k, v] of Object.entries(r ?? {})) {
			if (v && v.pixels) out[id][k] = encode ? { png: llm.base64Encode(image.encodePng(v)), width: v.width, height: v.height } : { width: v.width, height: v.height };
			else out[id][k] = v;
		}
	}
	return out;
}

function onProgress(e) {
	if (e.done) llm.print(`  [node] ${e.node}: ${(e.ms / 1000).toFixed(2)}s`);
}

function reply(status, body) { llm.respond(status, 'application/json', JSON.stringify(body)); }

llm.plugin({
	name: 'graph',
	async load(config) {
		state.settings = readSettings();
		op.useMatrixCores(config.matrix_cores ?? state.settings.matrix_cores ?? true);
		state.plugins = await loadPlugins(llm.root, state.settings);
		const loaded = state.plugins.filter((p) => p.loaded).map((p) => p.name);
		llm.print(`graph: plugins ${loaded.join(', ') || 'none'}; ${Object.keys(catalogue()).length} node types`);
	},
	async generate(config) {
		if (config.list) {
			for (const [type, d] of Object.entries(catalogue())) {
				const ins = d.inputs.map((i) => `${i.name}: ${i.type}${i.optional ? '?' : ''}${i.default !== undefined ? '=' + JSON.stringify(i.default) : ''}`).join(', ');
				const outs = d.outputs.map((o) => `${o.name}: ${o.type}`).join(', ');
				llm.print(`${type.padEnd(22)} (${ins}) -> (${outs})${d.output ? '  [sink]' : ''}`);
			}
			return { nodes: Object.keys(catalogue()).length };
		}
		const start = llm.now();
		const { results, ran, cached } = await new Executor().run(readGraph(config), { onProgress });
		llm.print(`=== graph done in ${llm.since(start)}: ${ran.length} ran${cached.length ? `, ${cached.length} cached` : ''} ===`);
		return describe(results, false);
	},
	// GET /nodes: the node catalogue. GET /plugins: what is installed.
	// POST /graph with { graph, ...node.input overrides }: the sinks' results,
	// images as base64 PNG. Results stay cached between requests, so a new
	// seed does not reload the model or re-encode the prompt.
	async handle(req) {
		if (req.method === 'GET' && req.path === '/nodes') return reply(200, catalogue());
		if (req.method === 'GET' && req.path === '/plugins') return reply(200, state.plugins);
		if (req.method === 'POST' && req.path === '/graph') {
			state.server = state.server ?? new Executor({ keep: true });
			const body = JSON.parse(req.body || '{}');
			const start = llm.now();
			const { results, ran, cached } = await state.server.run(readGraph(body), { onProgress });
			llm.print(`=== graph done in ${llm.since(start)}: ${ran.length} ran${cached.length ? `, ${cached.length} cached` : ''} ===`);
			return reply(200, { results: describe(results, true), ran, cached });
		}
		reply(404, { error: 'GET /nodes, GET /plugins or POST /graph' });
	},
});
