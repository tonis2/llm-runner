# A real mesh as LATO.2's topo dataset prepares it (quantize_mesh_clustering,
# dedup_quantized_mesh, extract_active_voxels), in numpy without trimesh:
# vertices on the 1024 grid, their offsets, faces, edges, and the 64^3 voxels
# the faces touch (sampled rather than o_voxel's exact dual grid).
#   python3 tools/lato2_mesh_prep.py mesh.glb out.safetensors
#
# For tests/lato2_gt_topology.js and tests/lato2_scaffold.js; LATO.2 bundles
# three such meshes in assets/example_mesh/.
import json
import struct
import sys

import numpy as np

RES, VOX = 1024, 64


def quat_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def node_matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if 'scale' in n:
        m = np.diag(list(n['scale']) + [1.0]) @ m
    if 'rotation' in n:
        r = np.eye(4); r[:3, :3] = quat_mat(n['rotation']); m = r @ m
    if 'translation' in n:
        t = np.eye(4); t[:3, 3] = n['translation']; m = t @ m
    return m


def load_glb(path):
    b = open(path, 'rb').read()
    jl = struct.unpack('<I', b[12:16])[0]
    g = json.loads(b[20:20 + jl])
    binc = b[20 + jl + 8:]
    if g.get('extensionsUsed'):
        print('  extensions:', g['extensionsUsed'])

    def acc(i):
        a = g['accessors'][i]
        bv = g['bufferViews'][a['bufferView']]
        dt = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[a['componentType']]
        c = {'VEC3': 3, 'SCALAR': 1}[a['type']]
        size = np.dtype(dt).itemsize * c
        stride = bv.get('byteStride', size)
        off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        raw = np.frombuffer(binc, np.uint8, stride * (a['count'] - 1) + size, off)
        rows = np.lib.stride_tricks.as_strided(raw, (a['count'], size), (stride, 1))
        return np.ascontiguousarray(rows).view(dt).reshape(a['count'], c)

    verts, faces = [], []
    base = 0

    def walk(ni, parent):
        nonlocal base
        n = g['nodes'][ni]
        m = parent @ node_matrix(n)
        if 'mesh' in n:
            for p in g['meshes'][n['mesh']]['primitives']:
                if p.get('mode', 4) != 4:
                    continue
                v = acc(p['attributes']['POSITION']).astype(np.float64)
                v = (np.c_[v, np.ones(len(v))] @ m.T)[:, :3]
                f = acc(p['indices']).reshape(-1, 3).astype(np.int64) if 'indices' in p else np.arange(len(v)).reshape(-1, 3)
                verts.append(v); faces.append(f + base); base += len(v)
        for c in n.get('children', []):
            walk(c, m)

    for ni in g['scenes'][g.get('scene', 0)]['nodes']:
        walk(ni, np.eye(4))
    return np.concatenate(verts), np.concatenate(faces)


def quantize(vertices, faces):
    lo, hi = vertices.min(0), vertices.max(0)
    center = (lo + hi) / 2
    nv = (vertices - center) / max(float((hi - lo).max()), 1e-7) + 0.5
    grid = np.clip(np.floor(nv * RES), 0, RES - 1).astype(np.int64)
    h = (grid[:, 0] * RES + grid[:, 1]) * RES + grid[:, 2]
    uniq, inv = np.unique(h, return_inverse=True)
    vsum = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(vsum, inv, nv); np.add.at(cnt, inv, 1)
    vmean = vsum / cnt[:, None]
    vint = np.clip(np.floor(vmean * RES), 0, RES - 1).astype(np.int64)
    off = np.clip((vmean - (vint + 0.5) / RES) * 2 * RES, -1, 1)
    f = inv[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    return vint, off, f


def dedup(vint, off, f):
    # merge_vertices on the int coords, then unique non-degenerate faces, then
    # drop unreferenced vertices (trimesh's order: sorted unique rows)
    h = (vint[:, 0] * RES + vint[:, 1]) * RES + vint[:, 2]
    uniq, first, inv = np.unique(h, return_index=True, return_inverse=True)
    v2 = vint[first]
    osum = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(osum, inv, off); np.add.at(cnt, inv, 1)
    o2 = osum / cnt[:, None]
    f = inv[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    _, keep = np.unique(np.sort(f, 1), axis=0, return_index=True)
    f = f[np.sort(keep)]
    used = np.unique(f)
    remap = -np.ones(len(v2), np.int64); remap[used] = np.arange(len(used))
    return v2[used], o2[used], remap[f]


def voxels(vint, f):
    q = np.clip(vint / (RES - 1.0) - 0.5, -0.5 + 1e-6, 0.5 - 1e-6) * 0.99999
    tri = q[f]
    cells = set()
    for t in tri:
        span = np.abs(t.max(0) - t.min(0)).max()
        k = max(2, int(np.ceil(span * VOX * 4)) + 1)
        a, b = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k))
        m = (a + b) <= 1
        a, b = a[m], b[m]
        p = t[0] + a[:, None] * (t[1] - t[0]) + b[:, None] * (t[2] - t[0])
        c = np.clip(np.floor((p + 0.5) * VOX), 0, VOX - 1).astype(np.int64)
        cells.update(map(tuple, c))
    c = np.array(sorted(cells, key=lambda x: (x[0] * VOX + x[1]) * VOX + x[2]), np.int32)
    return c


def save(path, tensors):
    header, blobs, pos = {}, [], 0
    for k, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {np.dtype('int32'): 'I32', np.dtype('float32'): 'F32'}[a.dtype]
        header[k] = {'dtype': dt, 'shape': list(a.shape), 'data_offsets': [pos, pos + a.nbytes]}
        blobs.append(a.tobytes()); pos += a.nbytes
    h = json.dumps(header).encode()
    h += b' ' * ((8 - len(h) % 8) % 8)
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(h))); fh.write(h)
        for blob in blobs:
            fh.write(blob)


v, f = load_glb(sys.argv[1])
print(f'  raw: {len(v)} vertices, {len(f)} faces')
vint, off, f = quantize(v, f)
vint, off, f = dedup(vint, off, f)
e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
vox = voxels(vint, f)
print(f'  quantized: {len(vint)} vertices, {len(f)} faces, {len(e)} edges, {len(vox)} voxels of 64^3')
save(sys.argv[2], {
    'verts': vint.astype(np.int32), 'offsets': off.astype(np.float32),
    'faces': f.astype(np.int32), 'edges': e.astype(np.int32), 'voxels': vox,
})
