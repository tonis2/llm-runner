#!/usr/bin/env python3
"""
Convert a PyTorch Lightning .ckpt to safetensors or GGUF, without torch or numpy.

A .ckpt is a ZIP holding one pickle that describes every tensor plus one
entry per storage holding its raw bytes. Nothing in that needs torch to read:
the pickle is unpickled with the torch classes stubbed out, and the storage
bytes are already in the layout safetensors wants.

    python3 tools/convert_ckpt.py model.ckpt -o out/model.safetensors
    python3 tools/convert_ckpt.py model.ckpt -o out/ --split
    python3 tools/convert_ckpt.py model.ckpt --gguf -o out/model.gguf
    python3 tools/convert_ckpt.py model.ckpt --dry-run

For safetensors, dtypes are preserved exactly — bf16 stays bf16, and
llm::upload_safetensor_f32 converts on upload. Selected hyperparameters are
copied into the __metadata__ block so the dimensions travel with the weights.

--gguf writes all three SkinTokens networks into one file instead, renaming the
transformer to llama.cpp's scheme so the engine's existing Qwen3 loader reads it
with no special case, and quantising its 2D weights to Q8_0. The mesh encoder
and VAE keep their names and stay F32, since their kernels are F32-only. The
hyperparameters become `skintokens.*` metadata keys.
"""

import argparse
import collections
import io
import json
import os
import pickle
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_writer

# Storage class name (or dtype name) -> (safetensors dtype, bytes per element)
DTYPES = {
    "FloatStorage": ("F32", 4),
    "HalfStorage": ("F16", 2),
    "BFloat16Storage": ("BF16", 2),
    "DoubleStorage": ("F64", 8),
    "LongStorage": ("I64", 8),
    "IntStorage": ("I32", 4),
    "ShortStorage": ("I16", 2),
    "CharStorage": ("I8", 1),
    "ByteStorage": ("U8", 1),
    "BoolStorage": ("BOOL", 1),
    "float32": ("F32", 4),
    "float16": ("F16", 2),
    "bfloat16": ("BF16", 2),
    "float64": ("F64", 8),
    "int64": ("I64", 8),
    "int32": ("I32", 4),
    "int16": ("I16", 2),
    "int8": ("I8", 1),
    "uint8": ("U8", 1),
    "bool": ("BOOL", 1),
}

# Weights only the training graph uses. The VAE's encoder turns ground-truth
# skin weights into tokens, which inference never does — it goes the other way,
# from tokens the transformer produced. Dropping these is opt-in because being
# wrong about one means a silently broken model rather than a loud failure.
TRAINING_ONLY = (
    "vae.model.encoder.",
    "vae.model.quant.",
    "vae.model.FSQ.project_in.",
)

# Prefix -> output file stem, so a caller can bring up one network without
# faulting in the other 900 MB.
COMPONENTS = (
    ("mesh_encoder.", "mesh_encoder"),
    ("output_proj.", "mesh_encoder"),
    ("transformer.", "transformer"),
    ("vae.", "vae"),
)

# Hyperparameters worth carrying alongside the weights: everything needed to
# build the graph. Matched as a suffix of the dotted hparams path.
KEEP_HPARAMS = (
    "model_config.hidden_size",
    "model_config.vae_decoder_dim",
    "model_config.tokens_per_skin",
    "model_config.tokens_skin_cond",
    "model_config.use_rope",
    "model_config.encode_repeat",
    "model_config.mesh_encoder_layers",
    "model_config.mesh_encoder_attention_heads",
    "model_config.mesh_encoder_dim",
    "mesh_encoder.num_latents",
    "mesh_encoder.embed_dim",
    "mesh_encoder.point_feats",
    "mesh_encoder.num_freqs",
    "mesh_encoder.include_pi",
    "mesh_encoder.heads",
    "mesh_encoder.width",
    "mesh_encoder.num_encoder_layers",
    "mesh_encoder.use_ln_post",
    "mesh_encoder.qkv_bias",
    "mesh_encoder.token_num",
    "llm.pretrained_model_name_or_path",
    "llm.n_positions",
    "llm.max_position_embeddings",
    "llm.hidden_size",
    "model.in_channels",
    "model.cond_channels",
    "model.latent_channels",
    "model.num_attention_heads",
    "model.width_encoder",
    "model.width_decoder",
    "model.num_layers_encoder",
    "model.num_layers_decoder",
    "model.embedding_type",
    "model.embed_frequency",
    "model.embed_include_pi",
    "model.is_learned_queries",
    "model.use_pmpe",
    "model.FSQ_dict.levels",
    "model.FSQ_dict.dim",
    "sample.cond_tokens",
    "sample.sample_tokens",
    "sample.compress_tokens",
    "generate_kwargs.max_new_tokens",
    "generate_kwargs.num_beams",
    "generate_kwargs.top_k",
    "generate_kwargs.top_p",
    "generate_kwargs.repetition_penalty",
    "generate_kwargs.temperature",
    "sampler.num_samples",
    "sampler.num_vertex_samples",
    "sampler.num_skin_samples",
)


class Storage:
    """A reference to one `data/N` entry in the archive."""

    __slots__ = ("key", "dtype", "numel")

    def __init__(self, key, dtype, numel):
        self.key, self.dtype, self.numel = key, dtype, numel


class Tensor:
    """A view onto a Storage: what `_rebuild_tensor_v2` would have produced."""

    __slots__ = ("storage", "offset", "shape", "stride")

    def __init__(self, storage, offset, shape, stride):
        self.storage, self.offset = storage, offset
        self.shape, self.stride = tuple(shape), tuple(stride)

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def dtype(self):
        return DTYPES[self.storage.dtype][0]

    @property
    def itemsize(self):
        return DTYPES[self.storage.dtype][1]

    @property
    def nbytes(self):
        return self.numel * self.itemsize

    def contiguous_stride(self):
        stride, acc = [], 1
        for d in reversed(self.shape):
            stride.append(acc)
            acc *= d
        return tuple(reversed(stride))


class _Stub:
    """Stands in for any torch class the pickle names."""

    def __init__(self, name):
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        return None


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *rest):
    return Tensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, *rest):
    return data


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if name in ("_rebuild_parameter", "_rebuild_parameter_with_state"):
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        return _Stub(name)

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise ValueError(f"unexpected persistent id: {pid!r}")
        _, storage_type, key, _location, numel = pid
        dtype = getattr(storage_type, "__name__", str(storage_type))
        dtype = dtype.rsplit(".", 1)[-1]
        if dtype not in DTYPES:
            raise ValueError(f"unknown storage dtype {dtype!r}")
        return Storage(key, dtype, numel)


def open_ckpt(path):
    """Return (zipfile, archive_prefix, unpickled_object)."""
    z = zipfile.ZipFile(path)
    pkl = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
    if pkl is None:
        raise ValueError(f"{path}: no data.pkl — not a torch.save archive")
    prefix = pkl[: -len("data.pkl")]

    order = prefix + "byteorder"
    if order in z.namelist():
        bo = z.read(order).decode().strip()
        if bo != "little":
            raise ValueError(f"{path}: {bo}-endian archive, only little-endian is handled")

    return z, prefix, _Unpickler(io.BytesIO(z.read(pkl))).load()


def collect_tensors(obj, prefix=""):
    """Flatten the object graph to {dotted name: Tensor}, in encounter order."""
    out = {}
    if isinstance(obj, Tensor):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(collect_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(collect_tensors(v, f"{prefix}[{i}]"))
    return out


def collect_hparams(obj, prefix="", out=None, depth=0):
    """Flatten scalar leaves of the hyper_parameters tree."""
    if out is None:
        out = {}
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            collect_hparams(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj
    elif isinstance(obj, (list, tuple)) and len(obj) <= 32:
        if all(isinstance(x, (str, int, float, bool)) or x is None for x in obj):
            out[prefix] = list(obj)
    return out


def build_metadata(sources):
    """Merge selected hyperparameters from one or more (object, path) pairs.

    Earlier sources win. The fusion checkpoint describes the transformer and
    shape encoder but not the skin VAE's own architecture — that lives in the
    VAE checkpoint it was trained from, so --hparams-from can supply it.
    """
    meta = {"converted_by": "tools/convert_ckpt.py"}
    meta["source"] = ", ".join(os.path.basename(p) for _, p in sources)
    for obj, _ in sources:
        hp = collect_hparams(obj.get("hyper_parameters", {}) if isinstance(obj, dict) else {})
        for path, value in hp.items():
            for wanted in KEEP_HPARAMS:
                if path == wanted or path.endswith("." + wanted):
                    key = wanted.replace(".", "_")
                    if key not in meta:
                        meta[key] = json.dumps(value) if isinstance(value, list) else str(value)
                    break
    return meta


class StorageReader:
    """Reads storage entries, holding each only as long as it is still needed."""

    def __init__(self, z, prefix):
        self.z, self.prefix = z, prefix
        self.cache = {}
        self.last_use = {}

    def plan(self, tensors):
        """Record, for each storage, the index of the final tensor that reads it."""
        for i, t in enumerate(tensors):
            self.last_use[t.storage.key] = i

    def read(self, tensor, index):
        key = tensor.storage.key
        buf = self.cache.get(key)
        if buf is None:
            name = f"{self.prefix}data/{key}"
            buf = self.z.read(name)
            expected = tensor.storage.numel * tensor.itemsize
            if len(buf) != expected:
                raise ValueError(
                    f"storage {key}: archive holds {len(buf)} bytes, "
                    f"pickle declares {expected}"
                )
            if self.last_use.get(key, index) > index:
                self.cache[key] = buf
        elif self.last_use.get(key, index) <= index:
            del self.cache[key]

        start = tensor.offset * tensor.itemsize
        return buf[start : start + tensor.nbytes]


def write_safetensors(path, named, reader, metadata, quiet=False):
    """Write [8-byte header length][JSON header][tensor data] to path."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for name, t in named:
        header[name] = {
            "dtype": t.dtype,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + t.nbytes],
        }
        offset += t.nbytes

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad so the data section starts 8-byte aligned: the C3 loader hands F32
    # tensors to the GPU straight off the mmap, without a realigning copy.
    blob += b" " * (-len(blob) % 8)

    reader.plan([t for _, t in named])
    total = offset
    written = 0

    tmp = path + ".partial"
    try:
        with open(tmp, "wb") as f:
            f.write(len(blob).to_bytes(8, "little"))
            f.write(blob)
            for i, (name, t) in enumerate(named):
                if t.stride != t.contiguous_stride() and t.numel > 1:
                    raise ValueError(
                        f"{name}: non-contiguous (stride {t.stride}, "
                        f"expected {t.contiguous_stride()}) — would need a gather"
                    )
                data = reader.read(t, i)
                if len(data) != t.nbytes:
                    raise ValueError(f"{name}: got {len(data)} bytes, expected {t.nbytes}")
                f.write(data)
                written += len(data)
                if not quiet and (i % 25 == 0 or i == len(named) - 1):
                    pct = 100.0 * written / total if total else 100.0
                    sys.stderr.write(f"\r  {path}: {pct:5.1f}%  ({i + 1}/{len(named)})")
                    sys.stderr.flush()
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    if not quiet:
        sys.stderr.write("\n")
    return 8 + len(blob) + total


def component_of(name):
    for prefix, stem in COMPONENTS:
        if name.startswith(prefix):
            return stem
    return "misc"


# --- GGUF ---------------------------------------------------------------
#
# The transformer is stock Qwen3 at hidden 896 with a 33036 vocab, so renaming
# its tensors to llama.cpp's scheme is enough for the engine's existing
# `load_llm_model` to read them: `configs/qwen3.json` is data driven, and its
# `attention.key_length` covers the one awkward case here, a head_dim of 128
# against a dim/heads of 56.
#
# The mesh encoder and skin VAE keep their own names. Their prefixes do not
# collide with anything, and their loaders look tensors up by name.

GGUF_TOP_LEVEL = {
    "transformer.model.embed_tokens.weight": "token_embd.weight",
    "transformer.model.norm.weight": "output_norm.weight",
    "transformer.lm_head.weight": "output.weight",
}

GGUF_PER_LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

LAYER_RE = re.compile(r"^transformer\.model\.layers\.(\d+)\.(.+)$")


def permute_rope_rows(raw, itemsize, out_dim, n_head):
    """
    Reorder a projection's output rows from HF's rotate_half pairing to the
    adjacent-pair one GGUF readers use.

    HF rotates dimension j against j + head_dim/2; llama.cpp — and this
    engine's `rope` kernel, which indexes `head*dim + pair*2` — rotates j
    against j+1. The two are the same rotation on differently ordered rows, so
    the fix belongs in the weights and not in the kernel. Getting it wrong does
    not fail loudly: every logit stays finite and plausible, and the model just
    stops meaning anything.

    Row `h*hd + 2b + a` takes what was row `h*hd + a*hd/2 + b`, which is
    llama.cpp's `.reshape(n_head, 2, hd//2, ...).swapaxes(1, 2)`.
    """
    head_dim = out_dim // n_head
    if head_dim % 2:
        raise ValueError(f"odd head_dim {head_dim} cannot be paired")

    half = head_dim // 2
    row_bytes = len(raw) // out_dim
    out = bytearray(len(raw))

    for h in range(n_head):
        for a in range(2):
            for b in range(half):
                src = (h * head_dim + a * half + b) * row_bytes
                dst = (h * head_dim + 2 * b + a) * row_bytes
                out[dst:dst + row_bytes] = raw[src:src + row_bytes]
    return bytes(out)


# Which transformer tensors need it, and over how many heads. The norms are
# applied per head over head_dim, before RoPE, so their elements ride the same
# permutation as the rows they scale.
ROPE_PERMUTED = {
    "attn_q.weight": "heads",
    "attn_k.weight": "kv_heads",
    "attn_q_norm.weight": "one",
    "attn_k_norm.weight": "one",
}


def gguf_name(name):
    """Rename a checkpoint tensor for GGUF, or return it unchanged."""
    if name in GGUF_TOP_LEVEL:
        return GGUF_TOP_LEVEL[name]
    m = LAYER_RE.match(name)
    if m:
        suffix = GGUF_PER_LAYER.get(m.group(2))
        if suffix is None:
            raise ValueError(f"no GGUF name for transformer tensor {name}")
        return f"blk.{int(m.group(1))}.{suffix}"
    if name.startswith("transformer."):
        raise ValueError(f"no GGUF name for transformer tensor {name}")
    return name


def gguf_type_for(name, tensor, transformer_quant):
    """
    Quantise the transformer's 2D weights and leave everything else F32.

    Only the LLM path has kernels for anything but F32 — `dispatch_matmul`
    selects on Q8_0 and the K-quants and falls through to the F32 kernel
    otherwise — so the mesh encoder and VAE, whose kernels this repo wrote,
    stay F32. Norms are 1D and stay F32 everywhere; they are a rounding error
    in the total and quantising them is where accuracy actually goes.
    """
    if not name.startswith("transformer."):
        return gguf_writer.GGML_F32
    if len(tensor.shape) < 2 or transformer_quant == "f32":
        return gguf_writer.GGML_F32
    # Q8_0 works down a row at a time, so the fastest-varying axis has to
    # divide into blocks of 32. Every Qwen3 weight here does.
    if tensor.shape[-1] % gguf_writer.Q8_0_BLOCK:
        return gguf_writer.GGML_F32
    return gguf_writer.GGML_Q8_0


def gguf_architecture_kv(by_name, hparams):
    """
    Derive the config from tensor shapes, which cannot disagree with the
    weights, rather than from the hyperparameters, which describe a training
    run. Only the context length and rope base come from elsewhere.
    """
    def shape(name):
        t = by_name.get(name)
        if t is None:
            raise ValueError(f"cannot describe the transformer: {name} is missing")
        return t.shape

    layers = 1 + max(
        int(m.group(1))
        for m in (LAYER_RE.match(n) for n in by_name)
        if m
    )
    vocab, dim = shape("transformer.model.embed_tokens.weight")
    head_dim = shape("transformer.model.layers.0.self_attn.q_norm.weight")[0]
    n_heads = shape("transformer.model.layers.0.self_attn.q_proj.weight")[0] // head_dim
    n_kv_heads = shape("transformer.model.layers.0.self_attn.k_proj.weight")[0] // head_dim
    ffn_dim = shape("transformer.model.layers.0.mlp.gate_proj.weight")[0]

    context = int(hparams.get("llm_n_positions", 3192))
    kvs = [
        gguf_writer.kv("general.architecture", gguf_writer.T_STRING, "qwen3"),
        gguf_writer.kv("general.name", gguf_writer.T_STRING, "SkinTokens"),
        gguf_writer.kv("general.alignment", gguf_writer.T_UINT32, gguf_writer.GGUF_ALIGNMENT),
        gguf_writer.kv("qwen3.block_count", gguf_writer.T_UINT32, layers),
        gguf_writer.kv("qwen3.embedding_length", gguf_writer.T_UINT32, dim),
        gguf_writer.kv("qwen3.feed_forward_length", gguf_writer.T_UINT32, ffn_dim),
        gguf_writer.kv("qwen3.attention.head_count", gguf_writer.T_UINT32, n_heads),
        gguf_writer.kv("qwen3.attention.head_count_kv", gguf_writer.T_UINT32, n_kv_heads),
        gguf_writer.kv("qwen3.attention.key_length", gguf_writer.T_UINT32, head_dim),
        gguf_writer.kv("qwen3.attention.value_length", gguf_writer.T_UINT32, head_dim),
        gguf_writer.kv("qwen3.context_length", gguf_writer.T_UINT32, context),
        gguf_writer.kv("qwen3.vocab_size", gguf_writer.T_UINT32, vocab),
        # Neither is overridden by the training config, so both are Qwen3-0.6B's
        # own defaults; the checkpoint only ever names the model it started from.
        gguf_writer.kv("qwen3.rope.freq_base", gguf_writer.T_FLOAT32, 1000000.0),
        gguf_writer.kv("qwen3.attention.layer_norm_rms_epsilon",
                       gguf_writer.T_FLOAT32, 1e-6),
    ]
    summary = dict(
        layers=layers, dim=dim, heads=n_heads, kv_heads=n_kv_heads,
        head_dim=head_dim, ffn=ffn_dim, vocab=vocab, context=context,
    )
    return kvs, summary


def write_gguf_file(path, named, reader, hparams, transformer_quant, quiet=False):
    by_name = dict(named)
    kvs, summary = gguf_architecture_kv(by_name, hparams)

    head_counts = {
        "heads": summary["heads"],
        "kv_heads": summary["kv_heads"],
        "one": 1,
    }

    # Everything the other two components need to be self describing travels
    # alongside, so one file is enough to rebuild the whole pipeline.
    for key in sorted(hparams):
        kvs.append(gguf_writer.kv(f"skintokens.{key}",
                                  gguf_writer.T_STRING, str(hparams[key])))

    entries = []
    permutes = {}
    for name, t in named:
        if t.stride != t.contiguous_stride() and t.numel > 1:
            raise ValueError(f"{name}: non-contiguous, would need a gather")

        target = gguf_name(name)
        for suffix, which in ROPE_PERMUTED.items():
            if target.endswith(suffix) and target.startswith("blk."):
                permutes[target] = (head_counts[which], t.shape[0])
                break

        entries.append((
            target,
            list(t.shape),
            t.dtype,
            gguf_type_for(name, t, transformer_quant),
            t.numel,
            t,
        ))

    def read_raw(tensor, index):
        raw = reader.read(tensor, index)
        entry = entries[index]
        permute = permutes.get(entry[0])
        if permute is None:
            return raw
        n_head, out_dim = permute
        itemsize = len(raw) // tensor.numel
        return permute_rope_rows(raw, itemsize, out_dim, n_head)

    reader.plan([t for _, t in named])
    total = gguf_writer.write(path, entries, kvs, read_raw=read_raw, quiet=quiet)
    print(f"  RoPE-permuted {len(permutes)} attention tensors")
    return total, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="path to the .ckpt")
    ap.add_argument("-o", "--output", help="output file, or directory when --split")
    ap.add_argument("--split", action="store_true",
                    help="one file per component (mesh_encoder, transformer, vae)")
    ap.add_argument("--drop-training", action="store_true",
                    help="omit weights only the training graph uses")
    ap.add_argument("--keep-prefix", action="store_true",
                    help="keep the leading 'state_dict.' on every name")
    ap.add_argument("--hparams-from", metavar="CKPT", action="append", default=[],
                    help="also read hyperparameters from this .ckpt (weights ignored); "
                         "repeatable, and the input's own values take precedence")
    ap.add_argument("--gguf", action="store_true",
                    help="write one GGUF instead of safetensors; the transformer is "
                         "renamed to llama.cpp's scheme so the engine's Qwen3 loader "
                         "reads it unchanged")
    ap.add_argument("--transformer-quant", choices=("q8_0", "f32"), default="q8_0",
                    help="dtype for the transformer's 2D weights under --gguf "
                         "(default q8_0; everything else stays F32)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written and stop")
    args = ap.parse_args()

    z, prefix, obj = open_ckpt(args.input)
    tensors = collect_tensors(obj)
    if not tensors:
        sys.exit(f"{args.input}: no tensors found")

    named = []
    dropped = []
    for name, t in tensors.items():
        clean = name
        if not args.keep_prefix and clean.startswith("state_dict."):
            clean = clean[len("state_dict."):]
        if args.drop_training and clean.startswith(TRAINING_ONLY):
            dropped.append((clean, t))
            continue
        named.append((clean, t))

    if args.gguf and args.split:
        sys.exit("--gguf writes a single file; drop --split")

    groups = collections.OrderedDict()
    if args.split:
        for name, t in named:
            groups.setdefault(component_of(name), []).append((name, t))
    else:
        groups["all"] = named

    stem = os.path.basename(args.input)
    if stem.endswith(".ckpt"):
        stem = stem[: -len(".ckpt")]

    suffix = "gguf" if args.gguf else "safetensors"
    if args.split:
        outdir = args.output or os.path.dirname(args.input) or "."
        paths = {g: os.path.join(outdir, f"{stem}.{g}.{suffix}") for g in groups}
    else:
        outdir = os.path.dirname(args.output) if args.output else ""
        paths = {"all": args.output or os.path.join(
            os.path.dirname(args.input) or ".", f"{stem}.{suffix}")}

    print(f"{args.input}: {len(tensors)} tensors")
    if dropped:
        freed = sum(t.nbytes for _, t in dropped)
        print(f"  dropping {len(dropped)} training-only tensors ({freed / 1e6:.1f} MB)")
        for name, _ in dropped:
            print(f"    - {name}")
    for g, items in groups.items():
        size = sum(t.nbytes for _, t in items)
        kinds = sorted({t.dtype for _, t in items})
        print(f"  {paths[g]}: {len(items)} tensors, {size / 1e6:.1f} MB, {'/'.join(kinds)}")

    if args.dry_run:
        return

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    sources = [(obj, args.input)]
    for extra in args.hparams_from:
        _z, _p, extra_obj = open_ckpt(extra)
        sources.append((extra_obj, extra))
        _z.close()

    metadata = build_metadata(sources)
    reader = StorageReader(z, prefix)

    if args.gguf:
        path = paths["all"]
        total, summary = write_gguf_file(
            path, named, reader, metadata, args.transformer_quant)
        print(f"  wrote {path} ({total / 1e6:.1f} MB)")
        print("  qwen3: {layers} layers, dim {dim}, {heads} heads / {kv_heads} kv, "
              "head_dim {head_dim}, ffn {ffn}, vocab {vocab}, ctx {context}".format(**summary))
        return

    for g, items in groups.items():
        total = write_safetensors(paths[g], items, reader, metadata)
        print(f"  wrote {paths[g]} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
