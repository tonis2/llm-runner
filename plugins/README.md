# Pipelines as plugins

`llm-runner` is one binary. It brings up a headless GPU through three.c3l, opens
the engine's JavaScript runtime and runs a **plugin**: a directory under
`plugins/` whose `main.js` builds a model's graph out of shady kernels and
drives it. A new model is a new directory. The binary does not change.

```
llm-runner flux   --config flux-t2i.json
llm-runner flux   --config flux-kontext.json prompt="make it night" seed=7
llm-runner flux   --config flux-t2i.json keep_dit=true --server --port 7860
llm-runner zimage --config zimage-t2i.json
llm-runner depth  model=depth_anything_v2_vits_fp32.safetensors input=photo.jpg output=depth.png
llm-runner tests/kernels.js          # a script: run once as a module
```

`--config file.json` is read first, then each `key=value` is applied on top of
it. A value that parses as JSON is taken as JSON; anything else is a string.
`--plugins dir` points somewhere other than `./plugins`.

## Layout

```
plugins/
  lib/                 shared by every plugin
    llm.js             the host API: models, tensors, tokenizer, images, noise
    gpu.js             kernels by name, push-block packing, dispatch
    ops.js             the operations models are written in (matmul, norms, attention, conv ...)
    qwen3.js           Qwen-family text encoder (layer-streamed)
    flux_vae.js        Flux VAE encoder and decoder (Flux 1 and Flux 2 namings)
    kernels/*.shady    one kernel per file; common.shady is put in front of each
  flux/                Flux 2 Klein: txt2img, img2img, kontext, LoRA/LoKr, server
  zimage/              Z-Image Turbo
  depth/               Depth Anything V2
  tests/               kernel parity, benchmarks, probes
```

## A plugin

```js
import { llm, image } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit } from '../lib/gpu.js';

llm.plugin({
	name: 'mine',
	load(config) { /* once: compile kernels, maybe keep weights resident */ },
	async generate(config) { /* one job; return a JSON-able result */ },
	async handle(request) { /* server mode, optional: llm.respond(status, type, body) */ },
});
```

The runner evaluates `main.js`, calls `load(config)`, and then calls
`generate(config)` once. With `--server` it calls `handle({ method, path, body })`
for each request instead. A plugin without `handle` gets `generate({ ...config,
...jsonBody })` and its result is sent back as JSON. Buffers and kernels persist
between requests, so weights a plugin keeps are loaded once.

`console.log` output is collected and printed when a call finishes. Use
`llm.print` for progress you want to see while it runs.

## The host API (`lib/llm.js`)

| | |
|---|---|
| `llm.open(path)` | A GGUF or safetensors file, memory-mapped: `.tensors`, `.metadata`, `.meta(key)`, `.shape(name)` (GGUF order), `.type(name)` |
| `model.upload(name, as)` | A `Tensor` on the GPU. `as` is `'raw'` (file bytes), `'f32'` (dequantised or widened), `'conv'` (f32, laid out `[kh, kw, in, out]`), `'q8'` (requantised to Q8_0), or `'auto'` (raw for the types matmul reads natively, otherwise f32). The bytes go from the mapping straight to VRAM in 64 MB pieces. |
| `model.floats(name)` | A small tensor on the host, as a Float32Array |
| `model.embedRows(name, ids, into)` | Embedding lookup on the host, written into a GPU buffer |
| `model.tokenizer()` | `.encode(text, specials)`, `.find(special)`, `.decode(ids)` |
| `llm.randomNormal(n, seed)` | The same noise the C3 pipelines drew for a seed |
| `image.load / decode / savePng / encodePng / cropTo / fit16 / resize / toTensor / fromTensor` | Images as `{ width, height, channels, pixels }` |
| `llm.readText / readBytes / writeBytes / exists / base64Encode / base64Decode` | Files |
| `llm.print(...)`, `llm.now()`, `llm.since(t0)` | Progress |

GPU work goes through `three.compute` (see `lib/three.c3l/docs/functions.md`,
section `three.compute`). `gpu.js` and `ops.js` wrap it. Every op records a
dispatch, and `submit()` runs everything recorded and waits.
`independent: true` on an op leaves out the barrier after it, for work the next
op does not read, such as Q, K and V over one input.

## Kernels

A kernel is `lib/kernels/<name>.shady`, compiled at runtime by the shady
compiler inside three.c3l and cached on disk (`build/shader-cache`). Shady reads
like C3. The language reference is `lib/three.c3l/lib/shady.c3l/LANGUAGE.md`.
Keep one entry point per file: a kernel's bindings are its module's.

A kernel can be sized from JS. It spells a size `$NAME`, and
`kernel(name, { NAME: value })` compiles one pipeline per set of values. This is
how `flash_attention` serves head widths of 64 and 128.

Shady has no `half` and no integer dot product yet. So f16 and bf16 are decoded
by hand (`common.shady`), and the matmuls accumulate in f32 from f32 tiles.

## Checking a port

`tests/kernels.js` runs each shady kernel next to the Slang kernel it replaced,
on the same random inputs: `llm-runner tests/kernels.js [only=name]`. Most agree
bit for bit. The Q8_0 matmul and flash attention agree to fp16 precision, because
Slang kept those tiles in fp16. The whole pipelines were compared image to image
against the C3 builds:

| | against the C3 pipeline |
|---|---|
| Flux txt2img, 512² | 53.6 dB PSNR |
| Flux kontext edit | 66.3 dB |
| Flux img2img | 63.6 dB |
| Flux kontext + LoKr + LoRA | 38.1 dB (requantised merge) |
| Depth Anything, depth / height map | 92.3 / 91.5 dB |
| Z-Image DiT, first step | velocity correlation 0.999998 |

Z-Image goes further apart from the second step on. The C3 build's Q8_0 matmul
converts activations to fp16, and some of Z-Image's FFN activations exceed
65,504, so the C3 build overflows (to inf, then its 1e10 clamp). The f32 tiles
here don't. See `tests/bench_matmul.js` for the matmul speed comparison.
