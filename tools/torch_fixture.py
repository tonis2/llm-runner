#!/usr/bin/env python3
"""Writes a small PyTorch-format checkpoint for tests/torch_reader.js, without torch.

    tools/torch_fixture.py <dir>      # writes <dir>/fixture.pt and <dir>/fixture.json

The file is what torch.save writes: a zip of stored entries, `archive/data.pkl`
(protocol 2) whose tensors are `torch._utils._rebuild_tensor_v2` over storages
named by persistent ids `('storage', torch.<T>Storage, key, 'cpu', numel)`, and
one `archive/data/<key>` per storage. Stand-in `torch` modules let the standard
pickler write those globals. The checkpoint is shaped like a Lightning one: a
`state_dict` beside other entries, one an object of an unknown class.

fixture.json holds every tensor's dtype, shape and values (as floats, after
bf16/f16 rounding) for the test to compare against.
"""
import collections
import json
import os
import pickle
import struct
import sys
import types
import zipfile

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

# --- stand-in torch -------------------------------------------------------------

torch = types.ModuleType("torch")
utils = types.ModuleType("torch._utils")
sys.modules["torch"] = torch
sys.modules["torch._utils"] = utils
torch._utils = utils


def storage_class(name, fmt, size):
    cls = type(name, (), {"fmt": fmt, "size": size})
    cls.__module__ = "torch"
    setattr(torch, name, cls)
    return cls


FloatStorage = storage_class("FloatStorage", "f", 4)
HalfStorage = storage_class("HalfStorage", "e", 2)
BFloat16Storage = storage_class("BFloat16Storage", "bf16", 2)
LongStorage = storage_class("LongStorage", "q", 8)


def _rebuild_tensor_v2(*args):
    raise RuntimeError("never called")


_rebuild_tensor_v2.__module__ = "torch._utils"
utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def _rebuild_parameter(*args):
    raise RuntimeError("never called")


_rebuild_parameter.__module__ = "torch._utils"
utils._rebuild_parameter = _rebuild_parameter

lightning = types.ModuleType("lightning_stub")
sys.modules["lightning_stub"] = lightning


class AttributeDict(dict):
    pass


AttributeDict.__module__ = "lightning_stub"
lightning.AttributeDict = AttributeDict


class Storage:
    def __init__(self, cls, key, values):
        self.cls, self.key, self.values = cls, key, values

    def data(self):
        if self.cls is BFloat16Storage:
            return b"".join(struct.pack("<f", v)[2:] for v in self.values)
        return struct.pack("<%d%s" % (len(self.values), self.cls.fmt), *self.values)

    def stored(self):
        # What the reader sees after the type's rounding.
        if self.cls is BFloat16Storage:
            return [struct.unpack("<f", b"\0\0" + struct.pack("<f", v)[2:])[0] for v in self.values]
        return list(struct.unpack("<%d%s" % (len(self.values), self.cls.fmt), self.data()))


class Tensor:
    def __init__(self, storage, offset, shape, stride=None, parameter=False):
        if stride is None:
            stride, s = [], 1
            for d in reversed(shape):
                stride.insert(0, s)
                s *= d
        self.storage, self.offset, self.shape, self.stride, self.parameter = storage, offset, shape, stride, parameter

    def __reduce__(self):
        t = (_rebuild_tensor_v2, (self.storage, self.offset, tuple(self.shape), tuple(self.stride), False, collections.OrderedDict()))
        if self.parameter:
            return (_rebuild_parameter, (TensorRef(self), True, collections.OrderedDict()))
        return t


class TensorRef:
    def __init__(self, t):
        self.t = t

    def __reduce__(self):
        t = self.t
        return (_rebuild_tensor_v2, (t.storage, t.offset, tuple(t.shape), tuple(t.stride), False, collections.OrderedDict()))


class Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Storage):
            return ("storage", obj.cls, obj.key, "cpu", len(obj.values))
        return None


# --- the checkpoint ---------------------------------------------------------------

def ramp(n, scale, bias=0.0):
    return [bias + scale * ((i * 37) % 101 - 50) for i in range(n)]


storages = [
    Storage(FloatStorage, "0", ramp(24, 0.125)),
    Storage(BFloat16Storage, "1", ramp(40, 0.0371, 0.5)),
    Storage(HalfStorage, "2", ramp(16, 0.25)),
    Storage(LongStorage, "3", [3, -1, 1 << 40, 7]),
    Storage(FloatStorage, "4", ramp(12, 1.5)),
]
s0, s1, s2, s3, s4 = storages
state = collections.OrderedDict()
state["encoder.weight"] = Tensor(s0, 0, [4, 6])
state["encoder.bias"] = Tensor(s1, 8, [2, 4, 4], parameter=True)      # a view at an offset, as a Parameter
state["encoder.scale"] = Tensor(s1, 0, [8])                           # shares a storage with the one above
state["head.weight"] = Tensor(s2, 0, [16])
state["head.index"] = Tensor(s3, 0, [4])
state["head.transposed"] = Tensor(s4, 0, [3, 4], stride=[1, 3])        # not contiguous: left out of the table
state["scalar"] = Tensor(s0, 5, [])

hyper = AttributeDict(lr=0.001, layers=[1, 2, 3], name="fixture")
saved = {"epoch": 3, "global_step": 1 << 33, "state_dict": state, "hyper_parameters": hyper, "flags": {True, False}}

body = []


class Out:
    def write(self, b):
        body.append(b)


Pickler(Out(), protocol=2).dump(saved)
pkl = b"".join(body)

path = os.path.join(out_dir, "fixture.pt")
with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
    z.writestr("archive/data.pkl", pkl)
    for s in storages:
        z.writestr("archive/data/" + s.key, s.data())
    z.writestr("archive/version", "3\n")

expect = {}
for name, t in state.items():
    vals = t.storage.stored()
    idx = []

    def walk(dim, at):
        if dim == len(t.shape):
            idx.append(at)
            return
        for i in range(t.shape[dim]):
            walk(dim + 1, at + i * t.stride[dim])

    walk(0, t.offset)
    dtype = {FloatStorage: "F32", HalfStorage: "F16", BFloat16Storage: "BF16", LongStorage: "I64"}[t.storage.cls]
    expect[name] = {"dtype": dtype, "shape": t.shape, "values": [vals[i] for i in idx]}
json.dump({"tensors": expect, "epoch": 3}, open(os.path.join(out_dir, "fixture.json"), "w"))
print(f"wrote {path} ({len(state)} tensors)")
