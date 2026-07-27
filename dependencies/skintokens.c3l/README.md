# skintokens.c3l

Auto-rigging: a triangle mesh in, a skinned skeleton out. A port of
[VAST-AI's SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) onto the
`llm` core — a Michelangelo/3DShape2VecSet shape encoder feeding a Qwen3
transformer that emits a rig as a token sequence, and an FSQ-CVAE that turns
the tail of that sequence back into skin weights.

```c3
import skintokens;

TriMesh cow = skintokens::load_gltf(mem, "cow.glb")!;
ShapeEncoderInput shape = skintokens::prepare_shape_input(mem, cow)!;
float[] prefix = encoder.encode(mem, &shape)!;

RigResult rig = skintokens::generate_rig_from_prefix(
    mem, &ctx, &model, prefix, shape.normalization)!;
float[] skin = decoder.decode_rig(mem, &shape, &rig, cow)!;

skintokens::export_rig_glb(mem, "cow_rigged.glb", cow, &rig, skin: skin)!;
```

`examples/rig` in the parent project is a working CLI over exactly this.

## The pipeline

| stage | what it does |
| --- | --- |
| `prepare_shape_input` | area-weighted surface sampling, 512 FPS queries, 54-dim Fourier embedding |
| `MeshEncoder.encode` | 8 self-attention layers + 1 cross-attention → 512 × 896 |
| `generate_rig` | prefills those as embeddings, then samples under `RigGrammar` |
| `detokenize_skeleton` | token sequence → joints and parents |
| `SkinDecoder.decode_rig` | four codes per bone → a weight per vertex per bone |
| `export_rig_glb` | rigged `.glb`, via `gltf.c3l` |

One sequence carries both halves: the skeleton, the tokenizer's terminator,
then four skin codes per bone, then the model's own terminator. Nothing in the
logits marks the boundary — `RigGrammar` is what keeps sampling on the right
side of it, and what counts the codes out.

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

`--drop-training` leaves out the VAE's own encoder, its `quant` and the FSQ's
`project_in`. Those exist to turn ground-truth weights into tokens; inference
only ever goes the other way.

## Where this departs from the reference

**Skin is evaluated at the mesh's vertices.** The decoder's last block is a
cross-attention from a query point into a bone's context, and queries never see
each other — so a weight depends on nothing but the point and the bone. The
reference evaluates that field at its 54000 sampled points and then carries the
result to the vertices with an eight-neighbour inverse-distance blend. Reading
it at the vertices directly is the same field without the resampling, and for
any mesh under 54000 vertices it is also less work.

**The skin block is four codes per bone, not four minus one.** The reference's
logits processor compares `length - where` against `J * tokens_per_skin`, and
at the step just after the skeleton's terminator that difference is already 1 —
so it forces the terminator after `J*4 - 1` codes. `decode` then slices `J*4`
values, the last of which is the terminator itself; `33035 - 267` is 32768,
outside a 32768-entry codebook, and the mixed-radix unpacking quietly turns it
into entry 0. The last bone's fourth latent is wrong there. Training writes the
full `J*4`, so counting the whole block is both what the model was taught and
what leaves every bone intact.

**Sampling, not beam search.** `rig_beam.c3` implements a beam search, but a
deterministic one; the reference passes `num_beams` alongside `do_sample`,
which is beam *sampling*. Generating a few seeds and choosing between them with
`RigResult.off_mesh_fraction` is the cheaper substitute, and it is what
`examples/rig` does.

**Joint names.** Joints are named positionally. The checkpoint ships vroid and
mixamo skeleton templates with real names, but matching a generated skeleton
against one is a separate problem.

## Sampling

Decoding is stochastic and the settings are load-bearing: greedy decoding
collapses onto the coordinate bin nearest zero, and the repetition penalty is
what makes a sequence terminate at all. The model also wanders when a sequence
runs long — the reference implementation does the same — so generate a few
seeds and pick between them with `RigResult.off_mesh_fraction` rather than
taking the first that parses. Search with `predict_skin` off and decode the
winner once more with it on: the seed reproduces the skeleton exactly, so the
skin codes cost one pass rather than one per seed.

## Checking it against the reference

`tools/compare_skin_reference.py` runs the VAE decoder in torch against the
same inputs, fed by `test/zz_skin_dump.c3`:

```
c3c test --trust=full --test-filter test_zz_dump_skin_reference
venv/bin/python tools/compare_skin_reference.py --ref path/to/SkinTokens
```

Three things in this network are wrong in ways that stay finite and plausible,
which is why it is worth checking rather than eyeballing: the PMPE phase term
in the embedder, `norm_cross` landing on the queries instead of the keys, and
the head split over the fused q/k/v projection — the reference concatenates
`to_q`, `to_k` and `to_v` and splits the result into heads *afterwards*, so
head 4's queries come out of `to_k`.

