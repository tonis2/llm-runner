import { llm, GGML_NAME } from '../lib/llm.js';
for (const path of [
	'/run/media/tonis/extra/AI models/flux-2-klein-9b-Q8_0.gguf',
	'/run/media/tonis/extra/AI models/Qwen3-8B-Q8_0.gguf',
	'/run/media/tonis/extra/AI models/flux_vae.safetensors',
]) {
	const t0 = llm.now();
	const m = llm.open(path);
	const counts = {};
	const examples = {};
	for (const [name, t] of m.tensors) {
		const key = (m.format === 'gguf' ? GGML_NAME[t.type] : t.type) + ' ' + t.shape.length + 'D';
		counts[key] = (counts[key] || 0) + 1;
		if (!examples[key]) examples[key] = [];
		if (examples[key].length < 4) examples[key].push(name + ' ' + JSON.stringify(t.shape));
	}
	llm.print(path, m.format, 'open', llm.since(t0), m.tensors.size, 'tensors');
	llm.print(JSON.stringify(counts));
	llm.print(JSON.stringify(examples, null, 1));
	const keys = Object.keys(m.metadata).filter((k) => !k.startsWith('tokenizer.ggml.merges'));
	llm.print('meta:', JSON.stringify(Object.fromEntries(keys.slice(0, 40).map((k) => [k, m.metadata[k]]))));
	m.close();
}
