"""Reference numbers for the llm-runner UniMate port, as raw little-endian files.

  python unimate_golden.py out_dir [rig.glb face_r face_l stats prompt]

Writes into out_dir:
  tokens.json            T5 token ids of a few test strings and of each joint name
  t5_<i>.f32             last hidden state (T, 768) of test string i
  names.json             raw + cleaned joint names (BFS order), parents, depths, ...
  cond_*.f32 / .i32      conditioning arrays the network reads (J = real joints)
  caption.f32            pooled caption embedding (768)
  caption_tokens.f32     the caption's T5 states (T, 768), what cross-attention reads
  names_emb.f32          pooled joint-name embeddings (J, 768)
  noise.f32              (J, 12, 60) initial noise for the real joints only
  v_cond.f32 / v_uncond.f32   one network pass at t = 0 on the noise
  sample.f32             20-step Euler result, CFG 3 (J, 12, 60), normalised
  feats.f32              denormalised features (60, J, 12)
  fk.f32                 their joint positions (60, J, 3), canonical frame
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unimate_ref as R  # noqa: E402  (sets up paths and imports)

import numpy as np
import torch

from unimate.configs.schema import MainConfig
from unimate.models.factory import create_model
from unimate.models.text_encoder.factory import create_text_encoder
from unimate.inference.sample import _build_diffusion, _load_checkpoint
from unimate.inference.generate import ClassifierFreeSampleModel
from unimate.utils.text_emb_cache import pool
from unimate.utils.motion_utils import recover_unimate_joint_pos_from_rot


def w(path, arr, dtype=np.float32):
    np.ascontiguousarray(np.asarray(arr), dtype=dtype).tofile(path)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rig = sys.argv[2] if len(sys.argv) > 2 else '/var/home/tonis/Documents/c3/crig/assets/animations/humanoid.glb'
    face = sys.argv[3:5] if len(sys.argv) > 4 else ['thigh_R', 'thigh_L']
    stats_key = sys.argv[5] if len(sys.argv) > 5 else 'mixamo'
    prompt = sys.argv[6] if len(sys.argv) > 6 else 'An object walks forward.'
    torch.set_num_threads(os.cpu_count())

    config = MainConfig.from_json(os.path.join(R.EXP, 'config.json'))
    model = create_model(dataset_config=config.dataset, model_config=config.model)
    diffusion, gen = _build_diffusion(config)
    _load_checkpoint(model, R.CHECKPOINT, config)
    model.eval()
    enc = create_text_encoder(encoder_type='t5', encoder_version=config.model.text_encoder_version,
                              device='cpu', pool=False)

    # T5 on its own: token ids and hidden states of a few strings.
    tests = [prompt, 'Left Thigh', 'An object gallops, then stops!', 'Right Index Finger End', 'x']
    tok_ids = {}
    for i, s in enumerate(tests):
        inputs = enc.tokenize(s)
        with torch.no_grad():
            hidden = enc(inputs)[0].numpy()
        tok_ids[s] = inputs['input_ids'][0].tolist()
        w(f'{out}/t5_{i}.f32', hidden)

    names, parents, pos = R.read_rig(rig)
    cond = R.build_cond(names, parents, pos, face, 'rig')
    clean = list(cond['clean_joint_names'])
    for s in clean:
        tok_ids.setdefault(s, enc.tokenize(s)['input_ids'][0].tolist())
    json.dump({'tests': tests, 'ids': tok_ids}, open(f'{out}/tokens.json', 'w'), indent=0)

    name_rows = R.encode(enc, clean)
    names_emb = np.stack([pool(r) for r in name_rows])
    cap = R.encode(enc, [prompt])[0]
    c, offsets = R.make_batch(config, cond, np.load(os.path.join(R.EXP, 'dataset_stats.npy'), allow_pickle=True).item()[stats_key],
                              names_emb, prompt, cap)
    J = len(cond['parents'])
    json.dump({
        'rig': rig, 'face': face, 'stats': stats_key, 'prompt': prompt,
        'raw': list(cond['joint_names']), 'clean': clean,
        'input_names': names, 'input_parents': parents.tolist(), 'input_positions': pos.tolist(),
        'parents': np.asarray(cond['parents']).tolist(),
        'scale_factor': float(cond['scale_factor']), 'facing_quat': np.asarray(cond['facing_quat']).tolist(),
        'J': J,
    }, open(f'{out}/names.json', 'w'), indent=0)

    w(f'{out}/cond_tpos.f32', c['tpos_first_frame'][0, :J])
    w(f'{out}/cond_tpos_parents.f32', c['tpos_first_frame_parents'][0, :J])
    w(f'{out}/cond_tpos_raw.f32', np.asarray(cond['tpos_first_frame']))
    w(f'{out}/cond_offsets.f32', offsets)
    w(f'{out}/cond_spectral.f32', c['spectral_feats'][0, :J])
    w(f'{out}/cond_depths.i32', c['joint_depths'][0, :J], np.int32)
    w(f'{out}/cond_graph_dist.i32', c['graph_dist'][0, :J, :J], np.int32)
    w(f'{out}/cond_relations.i32', c['joint_relations'][0, :J, :J], np.int32)
    w(f'{out}/cond_mean.f32', c['mean'][0, :J])
    w(f'{out}/cond_std.f32', c['std'][0, :J])
    w(f'{out}/caption.f32', c['caption_emb'][0])
    w(f'{out}/caption_tokens.f32', cap)
    w(f'{out}/names_emb.f32', names_emb)

    shape = (1, config.dataset.max_joints, config.dataset.feature_len, config.dataset.max_motion_length)
    torch.manual_seed(10)
    noise = torch.randn(shape)
    noise[:, J:] = 0
    w(f'{out}/noise.f32', noise[0, :J])
    with torch.no_grad():
        t = torch.zeros(1)
        w(f'{out}/v_cond.f32', model(noise, t, c)[0, :J])
        w(f'{out}/v_uncond.f32', model(noise, t, c, force_mask=True)[0, :J])
        t = torch.full((1,), 0.5)
        w(f'{out}/v_cond_t05.f32', model(noise, t, c)[0, :J])
        cfg = ClassifierFreeSampleModel(model, cfg_scale=config.sampling.cfg_scale)
        fn = gen.sample_ode(sampling_method='euler', num_steps=21)
        sample = fn(noise.clone(), cfg, cond=c)[-1]
    w(f'{out}/sample.f32', sample[0, :J])
    mean = c['mean'][0, :J].numpy()
    std = c['std'][0, :J].numpy()
    feats = sample[0, :J].permute(2, 0, 1).numpy() * std + mean
    w(f'{out}/feats.f32', feats)
    fk = recover_unimate_joint_pos_from_rot(feats.astype(np.float32).astype(np.float64),
                                            np.asarray(cond['parents']), offsets.astype(np.float32).astype(np.float64))
    w(f'{out}/fk.f32', fk)
    print('golden written:', out, 'J =', J)


if __name__ == '__main__':
    main()
