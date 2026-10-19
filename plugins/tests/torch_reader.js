// lib/torch.js against a checkpoint the standard pickler wrote.
//
//   tools/torch_fixture.py build/torch-fixture
//   llm-runner tests/torch_reader.js dir=build/torch-fixture
//
// Reads every tensor three ways (floats on the host, an f32 upload read back,
// integers through `ints`) and compares them with the values the fixture was
// written from; the non-contiguous one must be left out of the table.
import { llm } from '../lib/llm.js';
import { openTorch, readTorch } from '../lib/torch.js';

const dir = globalThis.__llm_config.dir ?? 'build/torch-fixture';
const expect = JSON.parse(llm.readText(`${dir}/fixture.json`));
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

const { value } = readTorch(`${dir}/fixture.pt`);
check(value.get('epoch') === 3 && value.get('global_step') === 2 ** 33, 'top-level numbers');
check(value.get('hyper_parameters')?.global === 'lightning_stub AttributeDict', 'an unknown class is kept inert');

const m = openTorch(`${dir}/fixture.pt`);
check(same(m.skipped.sort(), ['head.index', 'head.transposed']), `left out of the table: ${m.skipped.join(', ')}`);
for (const [name, e] of Object.entries(expect.tensors)) {
	if (name === 'head.transposed') {
		check(!m.torch.get(name).contiguous, `${name} is seen as not contiguous`);
		continue;
	}
	if (e.dtype === 'I64') {
		check(same(m.ints(name), e.values), `${name} ints`);
		continue;
	}
	const shape = e.shape.length ? e.shape : [1];
	check(same(m.shape(name).slice().reverse(), shape), `${name} shape [${m.shape(name)}]`);
	const host = m.floats(name);
	check(same(Array.from(host), e.values.map(Math.fround)), `${name} ${e.dtype} floats`);
	const t = m.upload(name, 'f32');
	const back = new Float32Array(t.buffer.readBytes().buffer);
	check(same(Array.from(back.subarray(0, e.values.length)), e.values.map(Math.fround)), `${name} uploaded`);
	t.dispose();
}
m.close();
llm.print(failures ? `${failures} FAILED` : 'all passed');
