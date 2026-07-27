"""
Check the transformer against the reference implementation.

Run after `test/zz_dump*` style dumps, or adapt the paths: the point is that
both sides see byte-identical input, so any disagreement is ours. This is what
caught the RoPE permutation — every logit was finite, plausible and wrong.

    python3 -m venv venv && venv/bin/pip install torch numpy einops omegaconf lightning transformers
    venv/bin/python tools/compare_transformer_reference.py

Needs a clone of https://github.com/VAST-AI-Research/SkinTokens next to it for
the encoder comparison; the transformer one only needs transformers.
"""
import sys, os, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/tonis/Documents/c3/llm-runner/tools")
import convert_ckpt as C
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B",
    hidden_size=896, vocab_size=33036, n_positions=3192, max_position_embeddings=3192,
    tie_word_embeddings=False)
cfg.dtype = torch.float32
model = AutoModelForCausalLM.from_config(cfg).float().eval()
rope = getattr(cfg, "rope_theta", None) or getattr(cfg, "rope_parameters", None)
print("  layers", cfg.num_hidden_layers, "heads", cfg.num_attention_heads,
      "kv", cfg.num_key_value_heads, "head_dim", cfg.head_dim,
      "ffn", cfg.intermediate_size, "rope", rope, "tie", cfg.tie_word_embeddings)

z, prefix_name, obj = C.open_ckpt("/Users/tonis/Documents/ai_models/grpo_1400.ckpt")
tensors = C.collect_tensors(obj)
reader = C.StorageReader(z, prefix_name)
items = [(n, t) for n, t in tensors.items()]
reader.plan([t for _, t in items])

sd = {}
for i, (name, t) in enumerate(items):
    clean = name[len("state_dict."):] if name.startswith("state_dict.") else name
    if not clean.startswith("transformer."):
        continue
    raw = reader.read(t, i)
    arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    sd[clean[len("transformer."):]] = torch.from_numpy(arr.view(np.float32).reshape(t.shape).copy())
z.close()
# The checkpoint ships a separate lm_head, so tying would make one of the two
# overwrite the other on load.
head = sd["lm_head.weight"]
tokens = sd["model.embed_tokens.weight"]
print(f"  lm_head vs embed identical: {torch.equal(head, tokens)}")
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  loaded {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
if missing: print("    missing:", missing[:6])
if unexpected: print("    unexpected:", unexpected[:6])

prefix = torch.from_numpy(np.fromfile(os.path.join(HERE, "cow_mine.f32"), dtype=np.float32).reshape(1, 512, 896))
BOS, CLS_RIGNET = 257, 264
emb = model.get_input_embeddings()
start = emb(torch.tensor([[BOS, CLS_RIGNET]]))
inputs = torch.cat([prefix, start], dim=1)
print("  inputs_embeds", tuple(inputs.shape))

with torch.no_grad():
    out = model(inputs_embeds=inputs).logits[0, -1].numpy()

mine = np.fromfile(os.path.join(HERE, "cow_logits_mine_f32.f32"), dtype=np.float32)
d = np.abs(out - mine)
print(f"  ref  logits: mean {out.mean():+.4f} std {out.std():.4f} argmax {out.argmax()} max {out.max():.4f}")
print(f"  mine logits: mean {mine.mean():+.4f} std {mine.std():.4f} argmax {mine.argmax()} max {mine.max():.4f}")
print(f"  |diff| mean {d.mean():.5f} max {d.max():.5f}   corr {np.corrcoef(out, mine)[0,1]:.6f}")
print("  ref  top10:", np.argsort(-out)[:10].tolist())
print("  mine top10:", np.argsort(-mine)[:10].tolist())
