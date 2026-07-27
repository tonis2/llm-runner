"""
Check the mesh encoder against the reference implementation.

Run after `test/zz_dump*` style dumps, or adapt the paths: the point is that
both sides see byte-identical input, so any disagreement is ours. This is what
caught the RoPE permutation — every logit was finite, plausible and wrong.

    python3 -m venv venv && venv/bin/pip install torch numpy einops omegaconf lightning transformers
    venv/bin/python tools/compare_encoder_reference.py

Needs a clone of https://github.com/VAST-AI-Research/SkinTokens next to it for
the encoder comparison; the transformer one only needs transformers.
"""
import sys, os, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "SkinTokens"))

# The reference asks the GPU its name at import time to decide about flash
# attention 3. There is no CUDA here and the answer is "no" either way.
torch.cuda.get_device_name = lambda *a, **k: "cpu"

from src.model.michelangelo.models.tsal.sal_perceiver import ShapeAsLatentPerceiverEncoder

cfg = dict(device=None, dtype=None, num_latents=256, point_feats=3, embed_dim=64,
           num_freqs=8, include_pi=False, width=512, heads=8, num_encoder_layers=8,
           init_scale=0.25, qkv_bias=False, use_ln_post=True, use_checkpoint=False,
           supervision_type="occupancy", query_method=False, token_num=512, flash=False)
enc = ShapeAsLatentPerceiverEncoder(**cfg).float().eval()
output_proj = nn.Sequential(nn.Linear(512, 896), nn.RMSNorm(896)).float().eval()

# Weights out of the checkpoint, via the torch-free reader already in the repo.
sys.path.insert(0, "/Users/tonis/Documents/c3/llm-runner/tools")
import convert_ckpt as C
z, prefix, obj = C.open_ckpt("/Users/tonis/Documents/ai_models/grpo_1400.ckpt")
tensors = C.collect_tensors(obj)
reader = C.StorageReader(z, prefix)
items = [(n, t) for n, t in tensors.items()]
reader.plan([t for _, t in items])

def load(prefix_name, module):
    sd = {}
    for i, (name, t) in enumerate(items):
        clean = name[len("state_dict."):] if name.startswith("state_dict.") else name
        if not clean.startswith(prefix_name):
            continue
        raw = reader.read(t, i)
        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        f32 = arr.view(np.float32).reshape(t.shape)
        sd[clean[len(prefix_name):]] = torch.from_numpy(f32.copy())
    missing, unexpected = module.load_state_dict(sd, strict=False)
    print(f"  {prefix_name}: loaded {len(sd)}, missing {len(missing)}, unexpected {len(unexpected)}")
    if missing: print("    missing:", missing[:5])
    if unexpected: print("    unexpected:", unexpected[:5])

load("mesh_encoder.", enc)
load("output_proj.", output_proj)
z.close()

def f32(name, shape):
    return torch.from_numpy(np.fromfile(os.path.join(HERE, name), dtype=np.float32).reshape(shape))

pos = f32("cow_pos.f32", (1, 54000, 3))
nrm = f32("cow_nrm.f32", (1, 54000, 3))
qidx = np.fromfile(os.path.join(HERE, "cow_qidx.f32"), dtype=np.float32).astype(np.int64)
mine = np.fromfile(os.path.join(HERE, "cow_mine.f32"), dtype=np.float32).reshape(512, 896)

# The reference encoder forward, with our query selection substituted for its
# own FPS so both sides see identical inputs.
with torch.no_grad():
    e = enc.encoder
    data = e.fourier_embedder(pos)
    data = torch.cat([data, nrm], dim=-1)
    print("  input dim", data.shape)
    data = e.input_proj(data)
    query = data[:, qidx, :]
    lat = e.cross_attn(query, data)
    lat = e.self_attn(lat)
    lat = e.ln_post(lat)
    ref = output_proj(lat)[0].numpy()

d = np.abs(ref - mine)
print(f"  ref   mean {ref.mean():+.5f} std {ref.std():.5f} min {ref.min():+.4f} max {ref.max():+.4f}")
print(f"  mine  mean {mine.mean():+.5f} std {mine.std():.5f} min {mine.min():+.4f} max {mine.max():+.4f}")
print(f"  |diff| mean {d.mean():.6f} max {d.max():.6f}")
den = np.linalg.norm(ref)
print(f"  relative error {np.linalg.norm(ref - mine)/den:.6f}")
c = np.corrcoef(ref.ravel(), mine.ravel())[0, 1]
print(f"  correlation {c:.6f}")
