"""
Write a GGUF file, without torch or numpy.

The engine's LLM path loads GGUF, and its qwen3 config is data driven: the
`params` block maps config fields to GGUF metadata keys and the `weights` block
maps logical names to tensor names. So a SkinTokens checkpoint written out with
llama.cpp's naming loads through `load_llm_model` with no engine changes at all
— the transformer really is stock Qwen3, only at hidden 896 with a 33036 vocab.

Two conventions matter and are easy to get wrong:

  - GGUF stores dimensions fastest-varying first, so a PyTorch [out, in]
    row-major weight is described as ne = [in, out]. The bytes do not move.
  - The engine aligns the data section to a hard-coded 32 bytes rather than
    reading `general.alignment`, so 32 it is.

Dtype is chosen per tensor by the caller. The one to avoid is bf16: `GGML_BF16`
exists as a type but no kernel consumes it, and `dispatch_matmul` falls through
to the F32 path for anything it does not recognise — a bf16 weight would be
read as F32 and produce noise rather than an error.
"""

import math
import os
import struct
import sys

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF metadata value types.
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10

# ggml tensor types, matching lib/gguf/types.c3.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_BF16 = 30

Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34  # fp16 scale + 32 int8


def type_size(ggml_type, numel):
    if ggml_type == GGML_F32:
        return numel * 4
    if ggml_type in (GGML_F16, GGML_BF16):
        return numel * 2
    if ggml_type == GGML_Q8_0:
        if numel % Q8_0_BLOCK:
            raise ValueError(f"Q8_0 needs a multiple of {Q8_0_BLOCK} elements, got {numel}")
        return numel // Q8_0_BLOCK * Q8_0_BLOCK_BYTES
    raise ValueError(f"unsupported ggml type {ggml_type}")


def _string(s):
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(vtype, value):
    if vtype == T_STRING:
        return _string(value)
    if vtype == T_UINT32:
        return struct.pack("<I", value)
    if vtype == T_INT32:
        return struct.pack("<i", value)
    if vtype == T_FLOAT32:
        return struct.pack("<f", value)
    if vtype == T_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if vtype == T_UINT64:
        return struct.pack("<Q", value)
    if vtype == T_ARRAY:
        elem_type, items = value
        out = struct.pack("<IQ", elem_type, len(items))
        for item in items:
            out += _value(elem_type, item)
        return out
    raise ValueError(f"unsupported metadata type {vtype}")


def kv(key, vtype, value):
    return _string(key) + struct.pack("<I", vtype) + _value(vtype, value)


# --- dtype conversion ---------------------------------------------------
#
# Every conversion starts from bf16, which is exactly the top 16 bits of an
# F32. Widening is therefore a byte interleave and not arithmetic: the slice
# assignments below run in C, where a per-element Python loop over 230M
# parameters would not finish in reasonable time.


def bf16_to_f32(raw):
    n = len(raw) // 2
    out = bytearray(n * 4)
    out[2::4] = raw[0::2]
    out[3::4] = raw[1::2]
    return bytes(out)


def to_f32(raw, src_dtype):
    if src_dtype == "F32":
        return raw
    if src_dtype == "BF16":
        return bf16_to_f32(raw)
    if src_dtype == "F16":
        n = len(raw) // 2
        return struct.pack(f"<{n}f", *struct.unpack(f"<{n}e", raw))
    raise ValueError(f"cannot widen {src_dtype} to F32")


def quantize_q8_0(f32_raw):
    """Block of 32: one fp16 scale then 32 int8, matching llama.cpp and
    llm::dequant_q8_0_block."""
    numel = len(f32_raw) // 4
    if numel % Q8_0_BLOCK:
        raise ValueError(f"Q8_0 needs a multiple of {Q8_0_BLOCK} elements, got {numel}")

    blocks = numel // Q8_0_BLOCK
    out = bytearray(blocks * Q8_0_BLOCK_BYTES)
    unpack = struct.Struct("<32f").unpack_from
    pack_scale = struct.Struct("<e").pack_into
    pack_qs = struct.Struct("<32b").pack_into

    for b in range(blocks):
        values = unpack(f32_raw, b * 128)
        amax = max(map(abs, values))
        d = amax / 127.0
        if d < 1e-8:
            d = 1e-8
        inv = 1.0 / d

        base = b * Q8_0_BLOCK_BYTES
        pack_scale(out, base, d)

        # Round half away from zero, as llama.cpp's roundf does. Python's
        # round() is half-to-even and would bias the low bit.
        qs = []
        for v in values:
            q = math.floor(v * inv + 0.5) if v >= 0 else -math.floor(-v * inv + 0.5)
            if q > 127:
                q = 127
            elif q < -127:
                q = -127
            qs.append(q)
        pack_qs(out, base + 2, *qs)

    return bytes(out)


def convert(raw, src_dtype, dst_type):
    if dst_type == GGML_F32:
        return to_f32(raw, src_dtype)
    if dst_type == GGML_Q8_0:
        return quantize_q8_0(to_f32(raw, src_dtype))
    if dst_type == GGML_BF16 and src_dtype == "BF16":
        return raw
    if dst_type == GGML_F16 and src_dtype == "F16":
        return raw
    raise ValueError(f"no conversion from {src_dtype} to ggml type {dst_type}")


def _pad(n, alignment=GGUF_ALIGNMENT):
    return (-n) % alignment


def write(path, entries, metadata, read_raw, quiet=False):
    """
    entries: list of (gguf_name, pytorch_shape, src_dtype, ggml_type, numel, token)
             `token` is handed back to read_raw to fetch the bytes.
    metadata: list of pre-encoded KV blocks, from kv().
    read_raw: (token, index) -> raw source bytes for that tensor.
    """
    header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(entries), len(metadata))
    header += b"".join(metadata)

    # Tensor sizes follow from the element count alone, so the whole info
    # block can be laid out before a single byte is converted.
    infos = bytearray()
    offset = 0
    sizes = []
    for name, shape, _src, ggml_type, numel, _token in entries:
        size = type_size(ggml_type, numel)
        sizes.append(size)
        # ne is the reverse of the PyTorch shape: fastest-varying first.
        dims = list(reversed(shape)) or [1]
        infos += _string(name)
        infos += struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<IQ", ggml_type, offset)
        offset += size + _pad(size)

    body_start = len(header) + len(infos)
    lead = _pad(body_start)
    total = offset

    tmp = path + ".partial"
    written = 0
    try:
        with open(tmp, "wb") as f:
            f.write(header)
            f.write(infos)
            f.write(b"\0" * lead)

            for i, (name, _shape, src_dtype, ggml_type, numel, token) in enumerate(entries):
                raw = read_raw(token, i)
                data = convert(raw, src_dtype, ggml_type)
                if len(data) != sizes[i]:
                    raise ValueError(
                        f"{name}: converted to {len(data)} bytes, expected {sizes[i]}"
                    )
                f.write(data)
                f.write(b"\0" * _pad(len(data)))
                written += sizes[i]
                if not quiet and (i % 10 == 0 or i == len(entries) - 1):
                    pct = 100.0 * written / total if total else 100.0
                    sys.stderr.write(
                        f"\r  {os.path.basename(path)}: {pct:5.1f}%  ({i + 1}/{len(entries)})"
                    )
                    sys.stderr.flush()
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    if not quiet:
        sys.stderr.write("\n")
    return body_start + lead + total
