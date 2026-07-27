"""
Run the reference's *generation* path in torch, on our encoder prefix.

`compare_transformer_reference.py` pins one forward pass. This pins the whole
decode: real checkpoint weights in HF, the reference's own logits processor and
`generate_kwargs`, fed the same 512x896 mesh prefix our engine produced. If the
reference also comes back with a near-linear skeleton, the sparse branching is
the model's, not ours.

    rigvenv/bin/python tools/compare_generate_reference.py [--cls articulation]

Needs a clone of https://github.com/VAST-AI-Research/SkinTokens for the
tokenizer (the grammar the logits processor enforces).
"""
import argparse, os, sys, json
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = "/private/tmp/claude-501/-Users-tonis-Documents-c3-llm-runner/2e73097e-df9f-4807-990a-0be65a0f4fab/scratchpad"
REF = os.path.join(SCRATCH, "SkinTokens")
sys.path.insert(0, HERE)
sys.path.insert(0, REF)

import convert_ckpt as C
from transformers import AutoConfig, AutoModelForCausalLM
from src.tokenizer.parse import get_tokenizer

# `src.model.tokenrig` drags in flash_attn and the VAE, neither of which will
# import on this machine. The logits processor is self-contained, so lift its
# source out of the reference verbatim rather than paraphrasing the grammar.
def _lift(path, start, end):
    src = open(path).read()
    a = src.index(start)
    b = src.index(end, a)
    return src[a:b]

_ns = {"torch": torch, "Tensor": torch.Tensor, "FloatTensor": torch.FloatTensor}
from transformers import LogitsProcessor, LogitsProcessorList
_ns.update(LogitsProcessor=LogitsProcessor, LogitsProcessorList=LogitsProcessorList,
           Tokenizer=object)
_TR = os.path.join(REF, "src/model/tokenrig.py")
exec(_lift(_TR, "class VocabSwitchingLogitsProcessor", "class TokenRig"), _ns)
exec(_lift(_TR, "def get_logits_processor", "@torch.no_grad()"), _ns)
get_logits_processor = _ns["get_logits_processor"]

ap = argparse.ArgumentParser()
ap.add_argument("--cls", default="articulation",
                help="articulation | rignet | vroid | none")
ap.add_argument("--beams", type=int, default=10)
ap.add_argument("--max-new", type=int, default=400)
ap.add_argument("--seeds", type=int, default=1, help="run seeds 0..N-1")
ap.add_argument("--q8", action="store_true",
                help="round-trip the 2D weights through Q8_0, as the shipped GGUF does")
ap.add_argument("--prefix", default=os.path.join(SCRATCH, "cow_mine.f32"))
args = ap.parse_args()

# --- tokenizer, straight from the checkpoint's own hyper_parameters ---------
tok = get_tokenizer(**{
    "__target__": "tokenizer_part",
    "num_discrete": 256,
    "continuous_range": [-1, 1],
    "cls_token_id": {"rignet": 0, "vroid": 1, "articulation": 2},
    "parts_token_id": {"body": 0, "hand": 1},
})
VAE_VOCAB = 32768
vocab_size = tok.vocab_size + VAE_VOCAB + 1
EOS = vocab_size - 1
print(f"  tokenizer vocab {tok.vocab_size}  total {vocab_size}  eos {EOS}")
print(f"  bos {tok.token_id_bos}  skel-eos {tok.token_id_eos}  branch {tok.token_id_branch}")
print(f"  cls tokens {tok.cls_token_id}  cls_none {tok.token_id_cls_none}")

# --- weights ----------------------------------------------------------------
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B",
    hidden_size=896, vocab_size=vocab_size, n_positions=3192,
    max_position_embeddings=3192, tie_word_embeddings=False)
cfg.dtype = torch.float32
model = AutoModelForCausalLM.from_config(cfg).float().eval()

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
if args.q8:
    # What the shipped GGUF actually holds: 32-element blocks down each row,
    # an fp16 scale of max|x|/127, round-half-away-from-zero. The RoPE
    # permutation reorders rows, not columns, so block boundaries are the same
    # either side of it and this is a faithful stand-in.
    n = 0
    for k, w in sd.items():
        if w.dim() < 2 or w.shape[-1] % 32:
            continue
        x = w.reshape(-1, 32)
        d = (x.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8).half().float()
        q = torch.sign(x / d) * torch.floor((x / d).abs() + 0.5)
        sd[k] = (q.clamp(-128, 127) * d).reshape(w.shape)
        n += 1
    print(f"  Q8_0 round-tripped {n} tensors")

missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  loaded {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
assert not missing and not unexpected

# --- prompt -----------------------------------------------------------------
prefix = torch.from_numpy(np.fromfile(args.prefix, dtype=np.float32).reshape(1, -1, 896))
cls_token = tok.token_id_cls_none if args.cls == "none" else tok.cls_name_to_token(args.cls)
start_tokens = torch.tensor([[tok.token_id_bos, cls_token]])
start_embed = model.get_input_embeddings()(start_tokens)
inputs = torch.cat([prefix, start_embed], dim=1)
print(f"  prefix {tuple(prefix.shape)}  start {start_tokens.tolist()}  cls={args.cls}")

for seed in range(args.seeds):
  torch.manual_seed(seed)
  with torch.no_grad():
    out = model.generate(
        inputs_embeds=inputs,
        bos_token_id=tok.token_id_bos,
        eos_token_id=EOS,
        pad_token_id=tok.token_id_pad,
        logits_processor=get_logits_processor(
            tokenizer=tok, eos=EOS, tokens_per_skin=4, start_tokens=start_tokens[0]),
        max_new_tokens=args.max_new,
        num_return_sequences=1,
        num_beams=args.beams,
        do_sample=True,
        top_k=10,
        top_p=0.95,
        temperature=1.5,
        repetition_penalty=2.0,
    )

  ids = out[0].tolist()
  full = start_tokens[0].tolist() + ids

  # --- what came out --------------------------------------------------------
  skel_eos = tok.token_id_eos
  cut = full.index(skel_eos) + 1 if skel_eos in full else len(full)
  skel = full[:cut]
  branches = sum(1 for t in skel if t == tok.token_id_branch)
  line = (f"  cls={args.cls} beams={args.beams} seed={seed}: {len(skel)} tokens, "
          f"{branches} branch tokens, terminated={skel_eos in full}")
  try:
      d = tok.detokenize(np.array(skel))
      joints = np.asarray(d.joints)
      parents = list(d.parents)
      roots = sum(1 for p in parents if p is None or p < 0)
      kids = {}
      for i, p in enumerate(parents):
          if p is not None and p >= 0:
              kids[p] = kids.get(p, 0) + 1
      forks = sum(1 for v in kids.values() if v > 1)
      off = int(((joints < -1.0) | (joints > 1.0)).any(axis=1).sum())
      # The cow normalised the way AugmentAffine does it — uniform scale by the
      # largest extent — occupies x +/-0.339, y +/-0.746, z +/-1.0. Joints
      # outside that are off the mesh even though they are inside the cube,
      # which is the weaker test.
      half = np.array([0.339, 0.746, 1.0])
      offmesh = int((np.abs(joints) > half + 0.02).any(axis=1).sum())
      print(f"{line}, {len(joints)} joints, {roots} roots, {forks} forking joints, "
            f"{off} outside cube, {offmesh} off mesh, "
            f"x {joints[:,0].min():+.2f}..{joints[:,0].max():+.2f}")
  except Exception as e:
      print(f"{line} -- detokenize failed: {type(e).__name__}: {e}")
