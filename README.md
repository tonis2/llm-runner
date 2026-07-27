# llm-runner

GPU inference engine in [C3](https://c3-lang.org), running on Vulkan compute.

Targets: `flux`, `zimage`, `vit-test`, `depth`.

## Building

Requires [c3c](https://github.com/c3lang/c3c) 0.8.2+ and, to rebuild shaders,
[slangc](https://github.com/shader-slang/slang).

```sh
git submodule update --init            # dependencies/vulkan.c3l, dependencies/image.c3l
c3c build flux                         # or zimage / vit-test / depth
```

Shaders are checked in as `.spv`; recompile them only after editing a `.slang`:

```sh
c3c build shaders --trust=full
```

### macOS

Nothing extra to do — `c3c build flux` is the whole story.

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
