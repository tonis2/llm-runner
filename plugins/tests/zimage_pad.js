import { llm } from '../lib/llm.js';
const m = llm.open('/run/media/tonis/extra/AI models/z-image-turbo-Q8_0.gguf');
for (const [name, t] of m.tensors) if (/pad|cap_emb|norm_out|final_layer\.norm/.test(name)) llm.print(name, t.type, JSON.stringify(t.shape));
