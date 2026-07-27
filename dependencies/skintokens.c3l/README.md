# skintokens.c3l

Auto-rigging: a triangle mesh in, a skeleton out. A port of
[VAST-AI's SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) onto the
`llm` core — a Michelangelo/3DShape2VecSet shape encoder feeding a Qwen3
transformer that emits a rig as a token sequence.

```c3
import skintokens;

TriMesh cow = skintokens::load_gltf(mem, "cow.glb")!;
RigResult rig = skintokens::generate_rig(mem, &ctx, &encoder, &model, cow)!;
skintokens::export_rig_glb(mem, "cow_rigged.glb", cow, &rig)!;
```

`examples/rig` in the parent project is a working CLI over exactly this.

## The pipeline

| stage | what it does |
| --- | --- |
| `prepare_shape_input` | area-weighted surface sampling, 512 FPS queries, 54-dim Fourier embedding |
| `MeshEncoder.encode` | 8 self-attention layers + 1 cross-attention → 512 × 896 |
| `generate_rig` | prefills those as embeddings, then samples under `RigGrammar` |
| `detokenize_skeleton` | token sequence → joints and parents |
| `export_rig_glb` | rigged `.glb`, via `gltf.c3l` |

## Weights

One GGUF holds all three networks. Build it from the upstream checkpoint with
the parent project's converter:

```
python3 tools/convert_ckpt.py grpo_1400.ckpt --gguf out.gguf --hparams-from last.ckpt
```

The transformer is renamed to llama.cpp's scheme and quantised Q8_0, so the
stock Qwen3 loader in `llm_text` reads it with no special case; the mesh
encoder and VAE keep their own names at F32. The converter permutes the q/k
RoPE rows — the engine rotates adjacent pairs where HF pairs `(i, i+head_dim/2)`
— and getting that wrong yields logits that are finite, plausible and wrong.

## What is not ported

**Skin weights.** The transformer emits skin tokens after the skeleton in one
sequence, and decoding them needs the checkpoint's FSQ-CVAE. `RigGrammar` stops
at the skeleton's terminator, and `export_rig_glb` substitutes a distance bind
(`bind_geometric`) so the exported file is a valid glTF skin. The skeleton is
the model's prediction; the weights are not.

**Joint names.** Joints are named positionally. The checkpoint ships vroid and
mixamo skeleton templates with real names, but matching a generated skeleton
against one is a separate problem.

## Sampling

Decoding is stochastic and the settings are load-bearing: greedy decoding
collapses onto the coordinate bin nearest zero, and the repetition penalty is
what makes a sequence terminate at all. The model also wanders when a sequence
runs long — the reference implementation does the same — so generate a few
seeds and pick between them with `RigResult.off_mesh_fraction` rather than
taking the first that parses.
