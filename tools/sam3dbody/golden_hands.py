#!/usr/bin/env python3
"""Reference numbers for llm/tests/sam3dbody_hands.js: SAM 3D Body's full
inference (`inference_type="full"`), body decoder, both hands and the keypoint
prompted second body pass, from Meta's own code.

    python tools/sam3dbody/golden_hands.py --repo sam-3d-body --ckpt sam-3d-body-dinov3/model.ckpt \
        --mhr sam-3d-body-dinov3/assets/mhr_model.pt --image person.jpg [--bbox x0 y0 x1 y1] --out build/sam3d-golden-hands

Runs on the CPU with the backbone widened to float32, as golden.py does. The
three passes are `body` (the person's box), `lhand` (the left hand's box on the
image mirrored left to right) and `rhand`. Writes, float32 little-endian:

    image.png             the image as OpenCV decoded it
    crop_<pass>.bin       [3, 512, 512]   each pass's crop in [0, 1]
    features_<pass>.bin   [1280, 32, 32]  each pass's backbone output
    passes.json           each pass's batch (box centre and scale, warp, camera)
                          and every decoder layer's pose token and prediction,
                          as forward_step returned them (the left hand not yet
                          unflipped); the second body pass's keypoint prompt and
                          layers; the hand boxes; the merged result
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--mhr", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--bbox", type=float, nargs=4)
ap.add_argument("--out", default="build/sam3d-golden-hands")
args = ap.parse_args()

sys.path.insert(0, args.repo)
from sam_3d_body.build_models import load_sam_3d_body  # noqa: E402
from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator  # noqa: E402
from sam_3d_body.data.utils.prepare_batch import prepare_batch  # noqa: E402
import sam_3d_body.models.meta_arch.sam3d_body as meta  # noqa: E402
import cv2  # noqa: E402

# The model moves tensors to CUDA; this runs on the CPU.
torch.Tensor.cuda = lambda self, *a, **k: self
meta.recursive_to = lambda x, device: x

os.makedirs(args.out, exist_ok=True)
torch.set_grad_enabled(False)
model, cfg = load_sam_3d_body(args.ckpt, device="cpu", mhr_path=args.mhr)
est = SAM3DBodyEstimator(model, cfg)
model.backbone = model.backbone.float()
model.backbone_dtype = torch.float32

img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
cv2.imwrite(os.path.join(args.out, "image.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
box = np.array(args.bbox if args.bbox else [0, 0, w, h], dtype=np.float32).reshape(1, 4)
batch = prepare_batch(img, est.transform, box)


def save(name, t):
    t.detach().float().cpu().contiguous().numpy().astype("<f4").tofile(os.path.join(args.out, name))


def plain(v):
    if isinstance(v, torch.Tensor):
        return v.detach().double().cpu().numpy().reshape(-1).tolist()
    return np.asarray(v, dtype=np.float64).reshape(-1).tolist()


KEYS = ("pred_pose_raw", "global_rot", "body_pose", "shape", "scale", "hand", "pred_cam",
        "mhr_model_params", "joint_global_rots", "pred_keypoints_3d", "pred_keypoints_2d",
        "pred_keypoints_2d_cropped", "pred_keypoints_2d_depth", "pred_cam_t")


def record(o):
    return {k: plain(o[k][0]) for k in KEYS if o.get(k) is not None} | {"focal_length": float(o["focal_length"][0])}


NAMES = ["body", "lhand", "rhand"]
passes = []
layers = []
tokens = []
features = []

hooks = [
    model.backbone.register_forward_hook(lambda m, i, o: features.append(o[-1] if isinstance(o, tuple) else o)),
    model.head_pose.register_forward_hook(lambda m, i, o: tokens.append(i[0])),
    model.head_pose_hand.register_forward_hook(lambda m, i, o: tokens.append(i[0])),
]
project, project_hand = model.camera_project, model.camera_project_hand


def on_project(fn):
    def wrapped(pose_output, b):
        out = fn(pose_output, b)
        layers.append(record(out | {"pred_keypoints_2d_cropped": model._full_to_crop(b, out["pred_keypoints_2d"])}))
        return out
    return wrapped


model.camera_project = on_project(project)
model.camera_project_hand = on_project(project_hand)

step = model.forward_step


def forward_step(b, decoder_type="body"):
    first = len(layers)
    out = step(b, decoder_type=decoder_type)
    name = NAMES[len(passes)]
    save(f"crop_{name}.bin", b["img"][0, 0])
    save(f"features_{name}.bin", features[-1][0])
    key = "mhr" if decoder_type == "body" else "mhr_hand"
    entry = {
        "name": name,
        "batch": {k: plain(b[k]) for k in ("bbox", "bbox_center", "bbox_scale", "affine_trans", "cam_int", "ori_img_size", "img_size")},
        "layers": [L | {"token": plain(t[0])} for L, t in zip(layers[first:], tokens[first:])],
        "final": record(out[key]),
    }
    if decoder_type == "body":
        entry["hand_box"] = plain(out["mhr"]["hand_box"][0])
        entry["hand_logits"] = plain(out["mhr"]["hand_logits"][0])
    passes.append(entry)
    return out


model.forward_step = forward_step

prompted = {}
run_prompt = model.run_keypoint_prompt


def run_keypoint_prompt(b, output, keypoint_prompt):
    first = len(layers)
    out, kp = run_prompt(b, output, keypoint_prompt)
    prompted["keypoint_prompt"] = plain(keypoint_prompt[0])
    prompted["layers"] = [L | {"token": plain(t[0])} for L, t in zip(layers[first:], tokens[first:])]
    prompted["final"] = record(out["mhr"])
    return out, kp


model.run_keypoint_prompt = run_keypoint_prompt

model._initialize_batch(batch)
pose_output, batch_lhand, batch_rhand, lhand_output, rhand_output = model.run_inference(
    img, batch, inference_type="full", transform_hand=est.transform_hand, thresh_wrist_angle=est.thresh_wrist_angle)
for hk in hooks:
    hk.remove()

doc = {
    "image": os.path.abspath(args.image),
    "bbox": box[0].tolist(),
    "torch": torch.__version__,
    "thresh_wrist_angle": est.thresh_wrist_angle,
    "passes": passes,
    "prompted": prompted,
    "merged": record(pose_output["mhr"]),
}
json.dump(doc, open(os.path.join(args.out, "passes.json"), "w"))
print(f"wrote {args.out}: passes {[p['name'] for p in passes]}, prompt {prompted.get('keypoint_prompt')}")
