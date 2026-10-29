# llm-runner

A GPU inference engine written in [C3](https://c3-lang.org) that runs on Vulkan compute.

Each model is a **plugin**: a folder of JavaScript that builds the model's graph out of
`.shady` compute kernels and drives it. `llm-runner` is one binary that brings up a
headless GPU through [three.c3l](lib/three.c3l), opens its JavaScript runtime and runs
the plugin you name. Plugins are read from disk when they run, so you can add or edit a
model without rebuilding anything.

The C3 side does the work JavaScript can't do cheaply:
- it memory-maps GGUF, safetensors and PyTorch weights and streams them straight to VRAM;
- it runs the tokenizer;
- it handles images and files.

## Models

| Plugin | What it does |
|---|---|
| `flux` | Flux 2 Klein: text to image, img2img, kontext editing, LoRA/LoKr, an A1111-style server |
| `zimage` | Z-Image Turbo: text to image, img2img, CFG, LoRA, TAESD |
| `qwenimage` | Qwen-Image 2.1: text to image, img2img, editing with up to three reference images, LoRA |
| `depth` | Depth Anything V2: depth maps and tiling height maps |
| `kimodo` | Kimodo text-to-motion: a prompt to a skeletal animation (SOMA, SMPL-X or G1 skeleton) |
| `unimate` | UniMate text-to-motion for any rig: a rig and a prompt to an animation of that rig |
| `sam3dbody` | SAM 3D Body: a person in an image to a posed rig, or video frames to a motion clip (GLB, BVH) |
| `creature` | Video frames and a rigged `.glb` to an animation of that rig, for any animal (DINOv3 point tracking and a rig fit) |

Weights are used as they are downloaded, with no conversion step. Where to get each
model's weights, and how each port was checked against its reference, is in
[llm/README.md](llm/README.md).

## Building

You need [c3c](https://github.com/c3lang/c3c) 0.8.2 or newer and a Vulkan driver.

```sh
git submodule update --init --recursive
c3c build llm-runner
```

Kernels are compiled from `llm/core/kernels/*.shady` the first time they're used and
cached in `llm/build/shader-cache`. The cache is keyed on each kernel's source, not on
the compiler, so clear it after you update shady itself.

**macOS.** `c3c build llm-runner` is all you need. macOS has no system Vulkan, so on
arm64 the build uses the Khronos loader and the KosmicKrisp Metal driver that
`lib/three.c3l/lib/vulkan.c3l` fetches (see that library's README). If a Vulkan SDK is
installed in `/usr/local/lib`, it is used instead. To hand a build to someone else, put
both `.dylib`s next to the executable. Apple GPUs have no cooperative-matrix support, so
the float32 kernels are used.

## Running

```sh
./build/llm-runner <plugin> [--config file.json] [key=value ...]
```

`--config` is read first, then each `key=value` is applied on top of it. A value that
parses as JSON is taken as JSON; anything else is a string.

```sh
./build/llm-runner flux --config flux-t2i.json
./build/llm-runner flux --config flux-kontext.json prompt="make it night" seed=7
./build/llm-runner zimage --config zimage-t2i.json input=photo.png strength=0.6
./build/llm-runner qwenimage --config qwenimage.json prompt="a neon sign that says OPEN"
./build/llm-runner depth model=depth_anything_v2_vits_fp32.safetensors input=photo.jpg output=depth.png
./build/llm-runner kimodo model=Kimodo-SOMA-RP-v1.2 text_model=Llama-3-Kimodo-Q8_0.gguf prompt="a person waves"
```

The JSON files in the repository root are examples; their model paths point at the
author's machine, so change them to point at your own downloads.

### As a server

`--server` loads the model once and answers HTTP requests. Each request's JSON body is
applied over the config.

```sh
./build/llm-runner flux --config flux-t2i.json keep_dit=true --server --port 7860
./build/llm-runner creature model=sam-3d-body-dinov3 --server --port 7860
```

`--bind` sets the address (default `127.0.0.1`), and `--port` sets the port (default
`7860`). The routes are described in [docs/flux_server.md](docs/flux_server.md) and
[docs/creature_server.md](docs/creature_server.md).

### Node graphs

The image models are also sets of typed nodes: loaders, prompt encoders, samplers, and
shared VAE, LoRA and image nodes. A graph file wires them together the way ComfyUI does,
so you can swap the VAE, chain LoRAs or reuse a prompt without touching a plugin's code.

```sh
./build/llm-runner run llm/apps/graph/templates/flux-t2i.json sample.seed=7
./build/llm-runner run llm/apps/graph/templates/qwenimage-edit.json input0.path=photo.png prompt.prompt="make it winter"
./build/llm-runner graph list=true                  # the node types there are
./build/llm-runner graph --server --port 7860       # GET /nodes, GET /plugins, POST /graph
```

The graph server keeps results between requests, so a new seed re-samples without
reloading the model or re-encoding the prompt.

### The studio

```sh
./build/llm-runner ui [--size 1600x1000]
```

`ui` opens a node editor in a window (`llm/apps/studio/`). Generations run between
frames, so the window keeps drawing while they run.

### Scripts

A path ending in `.js` is run once as a module. The kernel checks and benchmarks run this
way:

```sh
./build/llm-runner tests/matmul_coop.js
```

## The llm folder

Everything `llm-runner` runs lives in `llm/`:

```
llm/
  core/      what llm-runner provides: the host API (llm.js), GPU ops and kernels,
             weight readers, VAEs, text and vision encoders, motion writers, the graph machinery
  apps/      llm-runner's own programs: the graph runner (with templates/) and the studio
  plugins/   the models, one folder each, with a plugin.json
  tests/     kernel checks against float32, model goldens, benchmarks
```

A plugin imports only core, never another plugin. Imports are rooted at the llm folder,
as `import { llm } from '/core/llm.js'`, so a plugin linked in from anywhere else
resolves them the same way. How to write a plugin, the host API, port types and kernels
are all covered in [llm/README.md](llm/README.md).

### Where the folder is found

`llm-runner` uses the first of these that holds `core/` (`lib/pipeline/plugins.c3`,
`plugins_root`):

1. `--plugins <dir>`;
2. `llm` beside the executable, where a release ships it. Symlinks are followed, so this
   works wherever the release is installed;
3. `./llm` in the current directory;
4. `llm` beside the library's own sources. The library knows that location from where
   c3c compiled it, so a checkout's build finds its folder from any directory.

### Adding a plugin

Link its folder into `plugins/`:

```sh
ln -s ~/src/my-plugin llm/plugins/my-plugin
```

The next run uses whatever is in the linked folder. A plugin that needs a newer core says
so with `"requires_core": n` in its `plugin.json`, and is then refused with a message
saying what is out of date. To hide plugins without removing them, list them under
`disabled_plugins` in `~/.config/llm-runner/settings.json`.

Plugins are trusted code: they read and write files and run kernels on your GPU.

## Making a release

```sh
tools/package.py dist kimodo flux       # dist/llm-runner and dist/llm/{core, apps, plugins/kimodo, plugins/flux}
```

The release is the binary with its `llm` folder beside it. It finds that folder from
wherever it is installed, and users add plugins by linking them into `dist/llm/plugins`.

## Using it from another C3 program

The engine is also a C3 library, `llm`, that runs a plugin in-process on a renderer you
already have. Rigsmith uses it this way to run Kimodo and
Creature jobs:

```c3
llm::pipeline::Job* job = mem::new(llm::pipeline::Job);
String root = llm::pipeline::plugins_root(mem, setting, "llm");
job.start(&renderer, "kimodo", config_json, root)!;
while (job.pump()) { /* draw a frame */ }
if (job.state == DONE) use(job.result());   // the plugin's result, as JSON
char[] glb = job.output("glb");              // bytes the plugin handed over
job.close();
```

`pump` runs a slice of the job and returns, so a program that calls it once a frame
keeps drawing while the job runs.

To vendor the library, pack it into a single file:

```sh
tools/pack_library.py ../my-program/libs/llm.c3l kimodo creature
```

The zip holds the C3 sources, plus `llm/core` and the plugins you name as data. c3c
unpacks it to `build/unpacked_c3l/llm.c3l`, and `plugins_root` finds the `llm` folder
there without any path from the program. When you package your program, copy that
folder into the package, because plugins are read from disk.

## GPU notes

On a GPU with `VK_KHR_cooperative_matrix` and 64-wide subgroups (RDNA3, for example),
the heavy kernels run on the matrix cores in float16 with float32 accumulation:
- Q8_0 matmuls;
- head-128 attention;
- the VAE's convolutions.

On an RX 7800 XT that is about 2-4x faster per step. Pass `matrix_cores=false` to keep
everything in float32.

## Tests

```sh
c3c test                                   # the C3 side; tokenizer_test needs test/data/, not checked in
./build/llm-runner tests/matmul_coop.js    # matrix-core kernels against the float32 ones
./build/llm-runner tests/conv_coop.js
./build/llm-runner tests/bench_attention.js
```

`llm/tests/` also holds golden checks against the reference implementations. Some of
them need numbers that the scripts in `tools/` write with PyTorch.

## License

Apache 2.0, see [LICENSE](LICENSE). Model weights carry their own licenses; check each
model's release before you use what it makes.
