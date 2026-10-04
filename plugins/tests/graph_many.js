// Inputs that take many wires (`LORA+`): the executor hands the node a list in
// wire order, a graph file may write them as a list or as edges, and the
// studio's Doc adds, removes and unwires them one at a time.
// `llm-runner tests/graph_many.js`.
import { llm } from '../lib/llm.js';
import { defineNodes } from '../lib/graph/registry.js';
import { Executor } from '../lib/graph/executor.js';
import { Doc, refreshTypes } from '../studio/doc.js';
import '../core/nodes.js';

const fails = [];
const check = (ok, what) => { llm.print(`graph_many: ${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) fails.push(what); };

let got = null;
defineNodes('test', {
	'test.sink': {
		title: 'sink', category: 'test', output: true,
		inputs: { lora: 'LORA+?' },
		outputs: {},
		run({ lora }) { got = lora; return {}; },
	},
	'test.one': {
		title: 'one', category: 'test', output: true,
		inputs: { lora: 'LORA?' },
		outputs: {},
		run({ lora }) { got = lora; return {}; },
	},
});

const lora = (id, path, from) => ({ id, type: 'core.lora', params: { path, ...(from ? { lora: { from } } : {}) } });
const paths = () => (got ?? []).map((l) => l.map((x) => x.path).join('>')).join(' | ');
const ex = new Executor({ log: () => {} });

await ex.run({ nodes: [lora('a', 'a'), lora('b', 'b'), lora('c', 'c', 'b.lora'),
	{ id: 's', type: 'test.sink', params: { lora: [{ from: 'a.lora' }, { from: 'c.lora' }] } }] });
check(paths() === 'a | b>c', `a list of wires, in order (${paths()})`);

await ex.run({ nodes: [lora('a', 'a'), { id: 's', type: 'test.sink', params: { lora: { from: 'a.lora' } } }] });
check(paths() === 'a', `one wire still comes as a list (${paths()})`);

got = 'unset';
await ex.run({ nodes: [{ id: 's', type: 'test.sink', params: {} }] });
check(got === undefined, 'no wires: left out');

await ex.run({ nodes: [lora('a', 'a'), lora('b', 'b'), { id: 's', type: 'test.sink', params: {} }],
	edges: [{ from: ['b', 'lora'], to: ['s', 'lora'] }, { from: ['a', 'lora'], to: ['s', 'lora'] }] });
check(paths() === 'b | a', `edges, in order (${paths()})`);

let err = null;
try {
	await ex.run({ nodes: [lora('a', 'a'), lora('b', 'b'), { id: 's', type: 'test.one', params: { lora: [{ from: 'a.lora' }, { from: 'b.lora' }] } }] });
} catch (e) { err = String(e.message); }
check(err && err.includes('takes one wire'), `two wires into a single input refused (${err})`);

// The studio's side.
refreshTypes();
const doc = new Doc({ nodes: [lora('a', 'a'), lora('b', 'b'), lora('c', 'c'), { id: 's', type: 'test.sink', params: {} }, { id: 'o', type: 'test.one', params: {} }] });
check(doc.connect('a', 'lora', 's', 'lora') === null && doc.connect('b', 'lora', 's', 'lora') === null, 'connect two');
check(doc.connect('a', 'lora', 's', 'lora') === null && doc.wires('s', 'lora').length === 2, 'the same wire twice is one');
doc.connect('c', 'lora', 's', 'lora');
check(doc.wires('s', 'lora').map((w) => w.node).join() === 'a,b,c', 'three, in order');
doc.disconnect('s', 'lora', { node: 'b', port: 'lora' });
check(doc.wires('s', 'lora').map((w) => w.node).join() === 'a,c', 'unwire one');
doc.remove('a');
check(JSON.stringify(doc.node('s').params.lora) === '[{"from":"c.lora"}]', 'removing a source drops its wire');
check(doc.upstream('s').has('c'), 'upstream follows a list');
doc.connect('b', 'lora', 'o', 'lora');
doc.connect('c', 'lora', 'o', 'lora');
check(JSON.stringify(doc.node('o').params.lora) === '{"from":"c.lora"}', 'a single input is replaced, not added to');
await ex.run(doc.graph(), { targets: ['s'] });
check(paths() === 'c', `the doc's graph runs (${paths()})`);

llm.print(fails.length ? `graph_many: ${fails.length} failed` : 'graph_many: all ok');
