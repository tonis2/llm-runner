import { llm, GGML_NAME } from '../lib/llm.js';
const m = llm.open('/run/media/tonis/extra/AI models/z-image-turbo-Q8_0.gguf');
const counts = {};
for (const [name, t] of m.tensors) {
	const short = name.replace(/\.\d+\./, '.N.');
	const key = short + ' ' + GGML_NAME[t.type] + ' ' + JSON.stringify(t.shape);
	counts[key] = (counts[key] || 0) + 1;
}
for (const [k, v] of Object.entries(counts)) llm.print(v, k);
llm.print(JSON.stringify(m.metadata));
const v = llm.open('/run/media/tonis/extra/AI models/ae.safetensors');
llm.print('ae tensors', v.tensors.size, [...v.tensors.keys()].filter((k) => k.startsWith('decoder.') && !k.includes('.block.') ).slice(0, 30).join(' '));
