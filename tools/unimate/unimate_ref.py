"""UniMate reference run on a rig, bypassing UniMate's dataset loader: what the
llm-runner plugin (plugins/unimate) was checked against. Needs PyTorch (CPU is
enough), the UniMate repo and its dependencies (see ../../plugins/unimate).

  python unimate_ref.py jobs.json out_dir

jobs.json: [{"rig": "path.glb", "tag": "humanoid_walk", "prompt": "An object walks forward.",
             "face": ["thigh_r", "thigh_l"], "stats": "mixamo", "seed": 10}, ...]

Per job it writes <tag>.glb (Kimodo-convention motion glb, rig joint names),
<tag>.npy (denormalized (60, J, 12) features), <tag>_golden.npz (cond, noise,
raw sample) and <tag>.png (skeleton contact sheet from FK of the features).
"""

import json
import os
import resource
import struct
import sys
import time

# A clone of github.com/Friedrich-M/UniMate and the folder holding
# unimate_uniml3d_f60_v2/ (config.json, dataset_stats.npy, checkpoints/).
# UNIMATE_VARIANT picks another model folder there (e.g.
# unimate_uniml3d_f60_v2_full_cross_attn) and UNIMATE_CHECKPOINT a .pt that is
# not its checkpoints/checkpoint_step_100000.pt.
REPO = os.environ.get('UNIMATE_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UniMate'))
MODELS = os.environ.get('UNIMATE_MODELS', '.')
EXP = os.path.join(MODELS, os.environ.get('UNIMATE_VARIANT', 'unimate_uniml3d_f60_v2'))
CHECKPOINT = os.environ.get('UNIMATE_CHECKPOINT', os.path.join(EXP, 'checkpoints/checkpoint_step_100000.pt'))
os.environ.setdefault('HF_HOME', os.path.join(MODELS, 'hf'))
sys.path.insert(0, REPO)

import numpy as np
import torch
import pygltflib

from Quaternions import Quaternions
from Animation import positions_global, offsets_from_positions

from unimate.configs.schema import MainConfig
from unimate.models.factory import create_model
from unimate.models.text_encoder.factory import create_text_encoder
from unimate.inference.sample import _build_diffusion, _load_checkpoint
from unimate.inference.generate import generate_samples, ClassifierFreeSampleModel
from unimate.utils.text_emb_cache import pool, sequences_from_hidden
from unimate.dataset.transforms import apply_normalization, build_parent_features
from unimate.dataset.mixture.collate import mixture_batch_collate
from unimate.utils.topology_utils import compute_edge_indexs
from unimate.utils.motion_utils import (
    recover_unimate_anim_from_rot, recover_unimate_joint_pos_from_rot,
)
from data_process.utils.motion_features import process_tpose, build_topology_cond
from data_process.joint_annotation.names_clean_rule import clean_joint_name, post_process


# ---------------------------------------------------------------- rig input

def _node_matrix(n):
    if n.matrix:
        return np.array(n.matrix, dtype=np.float64).reshape(4, 4).T
    t = np.array(n.translation or [0, 0, 0], dtype=np.float64)
    x, y, z, w = n.rotation or [0, 0, 0, 1]
    s = np.array(n.scale or [1, 1, 1], dtype=np.float64)
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r * s[None]
    m[:3, 3] = t
    return m


def read_rig(path):
    """Skin joints of a glb: names, parents (parents first), world rest positions."""
    g = pygltflib.GLTF2().load(path)
    parent_of = {}
    for i, n in enumerate(g.nodes):
        for c in n.children or []:
            parent_of[c] = i
    world = {}

    def world_of(i):
        if i not in world:
            m = _node_matrix(g.nodes[i])
            world[i] = world_of(parent_of[i]) @ m if i in parent_of else m
        return world[i]

    joints = list(g.skins[0].joints)
    joint_set = set(joints)

    def joint_parent(i):
        p = parent_of.get(i)
        while p is not None and p not in joint_set:
            p = parent_of.get(p)
        return p

    # Depth-first from the roots so every parent comes before its children.
    kids = {j: [] for j in joints}
    roots = []
    for j in joints:
        p = joint_parent(j)
        (kids[p] if p is not None else roots).append(j)
    if len(roots) != 1:
        raise ValueError(f'{path}: {len(roots)} root joints; UniMate needs one')
    order = []
    stack = [roots[0]]
    while stack:
        j = stack.pop()
        order.append(j)
        stack.extend(reversed(kids[j]))
    index = {j: k for k, j in enumerate(order)}
    names = [g.nodes[j].name or f'joint{k}' for k, j in enumerate(order)]
    parents = np.array([-1 if joint_parent(j) is None else index[joint_parent(j)] for j in order])
    pos = np.array([world_of(j)[:3, 3] for j in order])
    return names, parents, pos


# ------------------------------------------------------------- conditioning

def build_cond(names, parents, pos, face, object_type):
    J = len(names)
    offsets = pos.copy()
    offsets[1:] = pos[1:] - pos[parents[1:]]
    tpos_data = {
        'names': np.array(names), 'parents': parents,
        'rest_local_pos': offsets,
        'rest_local_rot': np.tile([1.0, 0, 0, 0], (J, 1)),  # identity, wxyz
        'fps': np.array(30),
    }
    face_joints = None
    if face:
        face_joints = {'r_hip': {'raw': face[0]}, 'l_hip': {'raw': face[1]}}
    clean = [post_process(clean_joint_name(n, object_type)) for n in names]
    (canon, offs, scale, _gh, cparents, names_bfs, _fps, bfs, face_idxs, baxis) = process_tpose(
        tpos_data, face_joints=face_joints, target_diameter=2.0)
    if face and face_idxs == [-1, -1]:
        raise ValueError(f'face joints {face} not found')
    clean_bfs = [clean[i] for i in bfs]
    cond = build_topology_cond(
        object_type, cparents, offs, names_bfs, clean_bfs,
        positions_global(canon)[0], canon.rotations.qs[0],
        Quaternions(canon.rotations.qs[0]).qs,  # identity rest: global == local
        face_joint_idxs=face_idxs, body_axis=baxis, scale_factor=scale)
    # The facing rotation canonicalization put on the root.
    cond['facing_quat'] = canon.rotations.qs[0][0]
    return cond


def encode(encoder, texts):
    with torch.no_grad():
        inputs = encoder.tokenize(texts)
        hidden = encoder(inputs)
    return sequences_from_hidden(hidden.cpu(), inputs['attention_mask'].cpu())


def make_batch(config, cond, stats, names_emb, caption, caption_tokens):
    parents = np.asarray(cond['parents'])
    tpos = np.asarray(cond['tpos_first_frame']).copy()
    tpos[:, 1] -= tpos[:, 1].min()
    offsets = offsets_from_positions(tpos, parents)
    J = len(parents)
    F = config.dataset.feature_len
    pad = np.zeros((J, F - 3))
    pad[:, :6] = Quaternions.id(1).rotation_matrix(cont6d=True)[0]
    tpos12 = np.concatenate([tpos, pad], axis=-1)
    mean = np.zeros((J, F))
    std = np.zeros((J, F))
    mean[0], std[0] = stats['mean_root'], stats['std_root']
    mean[1:], std[1:] = stats['mean_local'], stats['std_local']
    tpos_n = apply_normalization(tpos12, mean, std)
    T = config.dataset.max_motion_length
    batch = {
        'motion': np.zeros((T, J, F)),
        'max_motion_length': T,
        'motion_length': T,
        'max_joints': config.dataset.max_joints,
        'parents': parents,
        'edge_indexs': compute_edge_indexs(parents),
        'tpos_first_frame': tpos_n,
        'tpos_first_frame_parents': build_parent_features(tpos_n, parents)['tpos_first_frame_parents'],
        'offsets': offsets,
        'joint_graph_dist': np.asarray(cond['joint_graph_dists']),
        'joint_relations': np.asarray(cond['joint_relations']),
        'joint_depths': np.asarray(cond['joint_depths']),
        'spectral_feats': np.asarray(cond['spectral_feats']),
        'joint_names_emb': names_emb,
        'object_type': cond['object_type'],
        'start_idx': 0,
        'mean': mean,
        'std': std,
        'split_tag': 'test',
        'caption': caption,
        'caption_emb': pool(caption_tokens),
        'caption_tokens': caption_tokens,
    }
    _, c = mixture_batch_collate([batch])
    return c, offsets


# ------------------------------------------------------------------ output

def write_motion_glb(path, name, names, parents, offsets, rot_xyzw, root, fps=30):
    frames, J = rot_xyzw.shape[:2]
    blob = bytearray()
    views, accessors = [], []

    def add(arr, typ, minmax=False):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': a.nbytes})
        acc = {'bufferView': len(views) - 1, 'componentType': 5126, 'count': frames, 'type': typ}
        if minmax:
            acc['min'] = [float(a.min())]
            acc['max'] = [float(a.max())]
        accessors.append(acc)
        blob.extend(a.tobytes())
        return len(accessors) - 1

    inp = add(np.arange(frames) / fps, 'SCALAR', True)
    samplers = [{'input': inp, 'output': add(root, 'VEC3'), 'interpolation': 'LINEAR'}]
    channels = [{'sampler': 0, 'target': {'node': 0, 'path': 'translation'}}]
    for j in range(J):
        samplers.append({'input': inp, 'output': add(rot_xyzw[:, j], 'VEC4'), 'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1, 'target': {'node': j, 'path': 'rotation'}})
    nodes = [{'name': n, 'translation': [float(v) for v in (root[0] if j == 0 else offsets[j])]}
             for j, n in enumerate(names)]
    for j in range(1, J):
        nodes[parents[j]].setdefault('children', []).append(j)
    gltf = {
        'asset': {'version': '2.0', 'generator': 'unimate_ref.py'},
        'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': nodes,
        'skins': [{'joints': list(range(J)), 'skeleton': 0}],
        'animations': [{'name': name, 'samplers': samplers, 'channels': channels}],
        'accessors': accessors, 'bufferViews': views, 'buffers': [{'byteLength': len(blob)}],
    }
    js = json.dumps(gltf).encode()
    js += b' ' * (-len(js) % 4)
    blob += b'\0' * (-len(blob) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(blob), 0x004E4942) + bytes(blob))


def contact_sheet(path, title, pos, parents, frames=(0, 10, 20, 30, 40, 50, 59)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lo, hi = pos.min((0, 1)), pos.max((0, 1))
    fig, axes = plt.subplots(2, len(frames), figsize=(2.2 * len(frames), 5.2))
    for col, t in enumerate(frames):
        for row, (a, b, lab) in enumerate([(0, 1, 'front X/Y'), (2, 1, 'side Z/Y')]):
            ax = axes[row, col]
            p = pos[t]
            for j, q in enumerate(parents):
                if q >= 0:
                    ax.plot([p[j, a], p[q, a]], [p[j, b], p[q, b]], 'b-', lw=1.2)
            ax.plot(p[0, a], p[0, b], 'ro', ms=3)
            ax.set_xlim(lo[a] - .1, hi[a] + .1)
            ax.set_ylim(lo[b] - .1, hi[b] + .1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'f{t}', fontsize=8)
            if col == 0:
                ax.set_ylabel(lab, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# -------------------------------------------------------------------- main

def main():
    jobs = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    t0 = time.time()
    config = MainConfig.from_json(os.path.join(EXP, 'config.json'))
    config.sampling.device = 'cpu'
    model = create_model(dataset_config=config.dataset, model_config=config.model)
    diffusion, gen_diffusion = _build_diffusion(config)
    _load_checkpoint(model, CHECKPOINT, config)
    model.eval()
    print(f'denoiser: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params', flush=True)
    encoder = create_text_encoder(encoder_type='t5', encoder_version=config.model.text_encoder_version,
                                  device='cpu', pool=False)
    all_stats = np.load(os.path.join(EXP, 'dataset_stats.npy'), allow_pickle=True).item()
    print(f'loaded in {time.time() - t0:.1f}s; stats sets: {list(all_stats)}', flush=True)

    for job in jobs:
        t1 = time.time()
        tag = job['tag']
        names, parents, pos = read_rig(os.path.expanduser(job['rig']))
        cond = build_cond(names, parents, pos, job.get('face'), tag.replace('-', '_'))
        J = len(cond['parents'])
        clean = cond['clean_joint_names']
        name_rows = encode(encoder, list(clean))
        names_emb = np.stack([pool(r) for r in name_rows])
        cap_tokens = encode(encoder, [job['prompt']])[0]
        c, offsets = make_batch(config, cond, all_stats[job.get('stats', 'objaverse')],
                                names_emb, job['prompt'], cap_tokens)
        shape = (1, config.dataset.max_joints, config.dataset.feature_len, config.dataset.max_motion_length)
        seed = job.get('seed', 10)
        torch.manual_seed(seed)
        noise = torch.randn(shape)
        torch.manual_seed(seed)
        steps = job.get('steps', int(os.environ.get('UNIMATE_STEPS', '0')))
        with torch.no_grad():
            if steps:
                # Fixed-step Euler over steps+1 grid points (t=0 noise -> t=1 data).
                cfg = ClassifierFreeSampleModel(model, cfg_scale=config.sampling.cfg_scale)
                fn = gen_diffusion.sample_ode(sampling_method='euler', num_steps=steps + 1)
                sample = fn(noise.clone(), cfg, cond=c)[-1]
            else:
                sample = generate_samples(model, c, shape, 'flow', diffusion, gen_diffusion,
                                          device=torch.device('cpu'), cfg_scale=config.sampling.cfg_scale)
        gen_s = time.time() - t1

        mean = c['mean'][0, :J].numpy()
        std = c['std'][0, :J].numpy()
        feats = sample[0, :J].permute(2, 0, 1).numpy() * std + mean  # (T, J, 12)
        np.save(os.path.join(out, f'{tag}.npy'), feats)
        np.savez(os.path.join(out, f'{tag}_golden.npz'), noise=noise.numpy(), sample=sample.numpy(),
                 **{k: v.numpy() for k, v in c.items() if torch.is_tensor(v)})

        cparents = np.asarray(cond['parents'])
        fk = recover_unimate_joint_pos_from_rot(feats, cparents, offsets)
        contact_sheet(os.path.join(out, f'{tag}.png'), f"{tag}: {job['prompt']}", fk, cparents)

        # Back out of the canonical frame: undo the facing turn and the scale.
        anim = recover_unimate_anim_from_rot(feats, cparents, offsets)
        face_inv = -Quaternions(np.asarray(cond['facing_quat'])[None])
        rots = anim.rotations.copy()
        rots[:, 0] = face_inv * rots[:, 0]
        root = (face_inv * anim.positions[:, 0]) / cond['scale_factor']
        offs = offsets / cond['scale_factor']
        q = rots.qs  # wxyz
        write_motion_glb(os.path.join(out, f'{tag}.glb'), tag, list(cond['joint_names']), cparents,
                         offs, np.concatenate([q[..., 1:], q[..., :1]], -1), root)

        foot_y = fk[:, :, 1].min(1)
        print(f"{tag}: J={J} gen {gen_s:.1f}s  travel {np.linalg.norm(fk[-1, 0, [0, 2]] - fk[0, 0, [0, 2]]):.2f} "
              f"(diam 2)  min-y range [{foot_y.min():.3f},{foot_y.max():.3f}]  clean names: {clean[:8]}...",
              flush=True)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'peak RSS {rss:.0f} MB, total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
