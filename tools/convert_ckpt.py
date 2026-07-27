#!/usr/bin/env python3
"""
Convert a PyTorch Lightning .ckpt to safetensors, without torch or numpy.

A .ckpt is a ZIP holding one pickle that describes every tensor plus one
entry per storage holding its raw bytes. Nothing in that needs torch to read:
the pickle is unpickled with the torch classes stubbed out, and the storage
bytes are already in the layout safetensors wants.

    python3 tools/convert_ckpt.py model.ckpt -o out/model.safetensors
    python3 tools/convert_ckpt.py model.ckpt -o out/ --split
    python3 tools/convert_ckpt.py model.ckpt --dry-run

Dtypes are preserved exactly — bf16 stays bf16, and llm::upload_safetensor_f32
converts on upload. Selected hyperparameters are copied into the safetensors
__metadata__ block so the model dimensions travel with the weights.
"""

import argparse
import collections
import io
import json
import os
import pickle
import sys
import zipfile

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

    groups = collections.OrderedDict()
    if args.split:
        for name, t in named:
            groups.setdefault(component_of(name), []).append((name, t))
    else:
        groups["all"] = named

    stem = os.path.basename(args.input)
    if stem.endswith(".ckpt"):
        stem = stem[: -len(".ckpt")]

    if args.split:
        outdir = args.output or os.path.dirname(args.input) or "."
        paths = {g: os.path.join(outdir, f"{stem}.{g}.safetensors") for g in groups}
    else:
        outdir = os.path.dirname(args.output) if args.output else ""
        paths = {"all": args.output or os.path.join(
            os.path.dirname(args.input) or ".", f"{stem}.safetensors")}

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
    for g, items in groups.items():
        total = write_safetensors(paths[g], items, reader, metadata)
        print(f"  wrote {paths[g]} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
