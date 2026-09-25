# llm-runner

GPU inference engine in [C3](https://c3-lang.org), running on Vulkan compute.

## llm-runner: pipelines as JavaScript plugins

`llm-runner` is one binary built on [three.c3l](lib/three.c3l). It opens the
engine's headless GPU and JavaScript runtime and runs a pipeline written as a
plugin under `plugins/`: JS drives the graph and the kernels are `.shady`
compiled at runtime. The C3 side supplies what JS can't do cheaply: memory-mapped
GGUF and safetensors weights streamed straight to VRAM, the tokenizer, and
images. A new model is a new plugin directory, with no rebuild.

```sh
c3c build llm-runner
./build/llm-runner flux --config flux-t2i.json
./build/llm-runner flux --config flux-kontext.json
./build/llm-runner flux --config flux-t2i.json keep_dit=true --server --port 7860
./build/llm-runner zimage --config zimage-t2i.json
./build/llm-runner zimage --config zimage-t2i.json input=photo.png strength=0.6
./build/llm-runner depth model=depth_anything_v2_vits_fp32.safetensors input=photo.jpg
```

Ported: Flux 2 Klein (txt2img, img2img, kontext, LoRA/LoKr, the A1111-style
server), Z-Image Turbo (txt2img, img2img, LoRA, TAESD), and Depth Anything V2.
The C3 pipelines they replaced, and their Slang shaders, are gone; SkinTokens
was dropped. See [plugins/README.md](plugins/README.md) for the plugin API and
for how each port was checked.

On a GPU with `VK_KHR_cooperative_matrix` and 64-wide subgroups (RDNA3), the
Q8_0 matmuls, head-128 attention and the VAE's convolutions run on the matrix
cores in float16 with float32 accumulation. `matrix_cores=false` keeps
everything in float32.

## Building

Requires [c3c](https://github.com/c3lang/c3c) 0.8.2+.

```sh
git submodule update --init --recursive
c3c build llm-runner
```

Kernels are compiled from `plugins/lib/kernels/*.shady` when first used and
cached in `build/shader-cache`. The cache key is the source, not the compiler, so
clear it after changing shady itself.

### macOS

Nothing extra to do — `c3c build llm-runner` is the whole story.

macOS has no system Vulkan, so on arm64 `vk` uses the Khronos loader and the
KosmicKrisp Metal driver that `lib/three.c3l/lib/vulkan.c3l` fetches
(`fetch-dylibs.sh`; see that library's README). A system SDK in
`/usr/local/lib`, if present, is preferred.

Apple GPUs expose no `VK_KHR_cooperative_matrix`, so the engine falls back to
the tiled GEMM path automatically.

To distribute a build, copy both `.dylib`s next to the executable — the
compiled-in source path means nothing on another machine, but `@executable_path`
is checked first.

## Tests

```sh
c3c test --trust=full                  # host code; tokenizer_test needs test/data/, not checked in
./build/llm-runner tests/matmul_coop.js  # matrix-core kernels against the float32 ones
./build/llm-runner tests/bench_attention.js
./build/llm-runner tests/conv_coop.js
```
