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
./build/llm-runner flux --config flux-kontext-lora.json
./build/llm-runner flux --config flux-t2i.json keep_dit=true --server --port 7860
./build/llm-runner zimage --config zimage-t2i.json
./build/llm-runner depth model=depth_anything_v2_vits_fp32.safetensors input=photo.jpg
```

Ported so far: Flux 2 Klein (txt2img, img2img, kontext, LoRA/LoKr, the
A1111-style server), Z-Image Turbo, and Depth Anything V2. See
[plugins/README.md](plugins/README.md) for the plugin API and for how each port
was checked against the C3 pipeline it replaces.

## The C3 pipelines

Targets: `zimage`, `rig`. SkinTokens (`rig`) exists only here so far. The C3
Z-Image pipeline stays as a reference until the plugin has run with its real text
encoder (Qwen3-4B). The C3 Flux and Depth Anything pipelines were deleted once
their plugins matched them, and the old Qwen-Image-Edit experiment (`vit-test`)
was dropped; Qwen-Image 2.1 is planned as a plugin.

## Building

Requires [c3c](https://github.com/c3lang/c3c) 0.8.2+ and, to rebuild shaders,
[slangc](https://github.com/shader-slang/slang).

```sh
git submodule update --init            # dependencies/vulkan.c3l, dependencies/image.c3l
c3c build llm-runner                   # or zimage / rig
```

Shaders are checked in as `.spv`; recompile them only after editing a `.slang`:

```sh
c3c build shaders --trust=full
```

### macOS

Nothing extra to do — `c3c build llm-runner` is the whole story.

macOS has no system Vulkan, so `vk` ships its own loader and a Metal driver
(KosmicKrisp) under `dependencies/vulkan.c3l/macos-aarch64/`. It locates them
relative to its own source path, so they are found whether or not a LunarG SDK
is installed. A system SDK in `/usr/local/lib`, if present, is preferred.

Apple GPUs expose no `VK_KHR_cooperative_matrix`, so the engine falls back to
the tiled GEMM path automatically.

To distribute a build, copy both `.dylib`s next to the executable — the
compiled-in source path means nothing on another machine, but `@executable_path`
is checked first.

## Tests

```sh
c3c test --trust=full
```

`inference_test` and `tokenizer_test` need fixtures under `test/models/` and
`test/data/`, which are not checked in.
