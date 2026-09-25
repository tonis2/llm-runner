// Stand-in Z-Image conditioning for checking the DiT and VAE against the C3
// pipeline when Qwen3-4B is not on disk: Qwen3-8B's second-to-last hidden
// states, cut to the 4B's 2560 columns. Not a meaningful prompt encoding - a
// deterministic one both pipelines read from the same file.
import { llm } from '../lib/llm.js';
import { TextEncoder } from '../lib/qwen3.js';
const c = globalThis.__llm_config;
const enc = new TextEncoder(c.text_model);
const tokens = enc.tokenizer.encode(`<|im_start|>user\n${c.prompt}<|im_end|>\n<|im_start|>assistant\n`, true);
const out = enc.encodeLayers(tokens, [enc.config.nLayers - 2]);
const dim = enc.config.dim, keep = c.dim ?? 2560, n = tokens.length;
const all = new Float32Array(out.buffer.readBytes().buffer);
const file = new Uint8Array(8 + n * keep * 4);
new Uint32Array(file.buffer, 0, 2).set([n, keep]);
const body = new Float32Array(file.buffer, 8, n * keep);
for (let t = 0; t < n; t++) body.set(all.subarray(t * dim, t * dim + keep), t * keep);
llm.writeBytes(c.output, file);
llm.print(`wrote ${c.output}: [${n}, ${keep}]`);
