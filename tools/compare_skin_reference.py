"""
Check the skin VAE decoder against the reference implementation.

The kernels have CPU references in test/mesh_encoder_test.c3. What those cannot
catch is a misreading of the architecture, and this decoder has three places
where a wrong reading stays finite and plausible: the PMPE embedder's phase
term, `norm_cross` landing on the queries instead of the keys, and the head
split over the fused q/k/v projection, which the reference performs *after*
concatenating them so that head 4's queries come out of `to_k`.

So both sides are fed identical bytes and the answers compared:

    c3c test --trust=full --test-filter test_zz_dump_skin_reference
    venv/bin/python tools/compare_skin_reference.py

Needs torch, numpy, einops, diffusers, safetensors, and a clone of
https://github.com/VAST-AI-Research/SkinTokens whose path is given with --ref.
Not the transformer or the shape encoder — only the VAE, whose weights come
from the split safetensors rather than the GGUF.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default=os.path.join(HERE, "SkinTokens"),
                help="clone of VAST-AI-Research/SkinTokens")
ap.add_argument("--weights", default=os.path.join(
    ROOT, "test/models/skintokens/safetensors/grpo_1400.vae.safetensors"))
ap.add_argument("--dump", default=os.path.join(ROOT, "build"),
                help="where test_zz_dump_skin_reference wrote its .f32 files")
args = ap.parse_args()

sys.path.insert(0, args.ref)
# The reference asks the GPU its name at import time to decide about flash
# attention 3. There is no CUDA here and the answer is "no" either way.
torch.cuda.get_device_name = lambda *a, **k: "cpu"

from src.model.skin_vae.autoencoders.skin_fsq_cvae_model import SkinFSQCVAEModel

# --- the model, from the checkpoint's own hyper-parameters ------------------
#
# Every one of these is a `skintokens.*` key in the converted GGUF; the shapes
# in the checkpoint confirm them independently (cond_quant is [768, 512], so
# width_encoder is 768 and latent_channels 512; FSQ.project_out is [5, 512], so
# the codebook is five dimensions wide).

model = SkinFSQCVAEModel(
    in_channels=4, cond_channels=3, latent_channels=512,
    num_attention_heads=12, width_encoder=768, width_decoder=768,
    num_layers_encoder=2, num_layers_decoder=10,
    embedding_type="frequency", embed_frequency=8, embed_include_pi=True,
    sample_tokens=32, is_learned_queries=True, use_pmpe=True,
    FSQ_dict={"levels": [8, 8, 8, 8, 8], "dim": 512},
).float().eval()

from safetensors.torch import load_file

state = load_file(args.weights)
prefix = "vae.model."
sub = {k[len(prefix):]: v.float() for k, v in state.items() if k.startswith(prefix)}
missing, unexpected = model.load_state_dict(sub, strict=False)
print(f"  weights: loaded {len(sub)}, missing {len(missing)}, unexpected {len(unexpected)}")

# The converter drops the VAE's own encoder, its quant and the FSQ's project_in:
# all three exist to turn ground-truth weights into tokens, which inference
# never does. Anything else missing would be a real gap.
allowed = ("encoder.", "quant.", "FSQ.project_in.")
stray = [k for k in missing if not any(k.startswith(a) for a in allowed)]
if stray:
    print(f"  !! missing decode-path weights: {stray[:8]}")
if unexpected:
    print(f"  !! unexpected: {unexpected[:8]}")


def load(name, dtype=np.float32, cols=None):
    path = os.path.join(args.dump, name)
    if not os.path.exists(path):
        sys.exit(f"missing {path}; run the dump test first")
    a = np.fromfile(path, dtype=dtype)
    return a.reshape(-1, cols) if cols else a


cloud_pos = load("skin_cloud_pos.f32", cols=3)
cloud_nrm = load("skin_cloud_nrm.f32", cols=3)
cond_idx = load("skin_cond_idx.u32", dtype=np.uint32).astype(np.int64)
codes = load("skin_codes.u32", dtype=np.uint32).astype(np.int64)
query_pos = load("skin_query_pos.f32", cols=3)
query_nrm = load("skin_query_nrm.f32", cols=3)

TOKENS_PER_SKIN = 4
bones = len(codes) // TOKENS_PER_SKIN
queries = len(query_pos)
mine = load("skin_mine.f32").reshape(queries, bones)

print(f"  {len(cloud_pos)} cloud points, {len(cond_idx)} conditioning tokens, "
      f"{bones} bones, {queries} queries")

cond = torch.from_numpy(np.concatenate([cloud_pos, cloud_nrm], axis=-1)).unsqueeze(0)
points = torch.from_numpy(np.concatenate([query_pos, query_nrm], axis=-1)).unsqueeze(0)

with torch.no_grad():
    # `_encode` would pick the conditioning points itself, by a random draw and
    # then farthest point sampling. Substituting ours is the whole point: two
    # different point sets would disagree for a reason that is not a bug.
    def embed(x):
        return torch.cat([model.embedder(x[..., :3]), x[..., 3:]], dim=-1)

    cond_kv = embed(cond)
    cond_q = embed(cond[:, cond_idx])
    cond_latents = model.cond_quant(model.cond_encoder(cond_q, cond_kv))
    print(f"  cond latents {tuple(cond_latents.shape)} "
          f"mean {cond_latents.mean():+.5f} std {cond_latents.std():.5f}")

    ref = np.zeros((queries, bones), dtype=np.float32)
    for b in range(bones):
        indices = torch.from_numpy(codes[b * TOKENS_PER_SKIN:(b + 1) * TOKENS_PER_SKIN])
        z = model.FSQ.indices_to_codes(indices.unsqueeze(0)).reshape(1, TOKENS_PER_SKIN, -1)
        logits = model._decode(z=z, cond=cond_latents, sampled_points=points)
        ref[:, b] = logits.reshape(-1).numpy()

d = np.abs(ref - mine)
print(f"  ref   mean {ref.mean():+.5f} std {ref.std():.5f} "
      f"min {ref.min():+.4f} max {ref.max():+.4f}")
print(f"  mine  mean {mine.mean():+.5f} std {mine.std():.5f} "
      f"min {mine.min():+.4f} max {mine.max():+.4f}")
print(f"  |diff| mean {d.mean():.6f} max {d.max():.6f}")
print(f"  relative error {np.linalg.norm(ref - mine) / np.linalg.norm(ref):.6f}")
print(f"  correlation {np.corrcoef(ref.ravel(), mine.ravel())[0, 1]:.6f}")

# Per bone as well as overall: one bone disagreeing while the others match
# points at the FSQ unpacking, and all of them drifting together points at the
# conditioning encoder they share.
for b in range(bones):
    col = np.abs(ref[:, b] - mine[:, b])
    print(f"    bone {b}: ref mean {ref[:, b].mean():.5f} "
          f"mine {mine[:, b].mean():.5f}  |diff| max {col.max():.6f}")
