#!/usr/bin/env python3
"""Reference numbers for llm/tests/sam3dbody_golden.js, from Meta's own code.

    python tools/sam3dbody/golden.py --repo sam-3d-body --ckpt sam-3d-body-dinov3/model.ckpt \
        --mhr sam-3d-body-dinov3/assets/mhr_model.pt --image person.jpg [--bbox x0 y0 x1 y1] --out build/sam3d-golden

Needs PyTorch, a clone of github.com/facebookresearch/sam-3d-body (with its
requirements) and the gated weights from huggingface.co/facebook/sam-3d-body-dinov3.
Runs on the CPU, with the body decoder only (`inference_type="body"`: no hand
crops, no keypoint prompts).

The crop is made by the estimator's own transform (GetBBoxCenterScale,
TopdownAffine to 512x512, ToTensor) and normalised by the model's
data_preprocess, exactly as process_one_image does; with no --bbox the whole
image is the box, as there with no detector. Writes, all float32 little-endian:

    input.bin        [3, 512, 512]   the normalised crop the backbone reads
    block0.bin       [1029, 1280]    tokens after block 0, backbone in float32
    features.bin     [1280, 32, 32]  backbone output, in float32
    features_bf16.bin [1280, 32, 32] backbone output as SAM 3D Body runs it (bf16)
    image.png        the image as OpenCV decoded it (decoders differ on JPEGs)
    crop.bin         [3, 512, 512]   the crop before normalising, in [0, 1]
    context.bin      [1280, 32, 32]  the decoder's image tokens: features, no-mask
                                     embedding and ray conditioning
    layer0_tokens.bin [145, 1024]    the decoder's tokens after its first layer
    vertices.bin     [18439, 3]      the final mesh
    layers.json      every decoder layer's pose token (after norm_final), head
                     outputs, MHR model parameters, joints and keypoints
    golden.json      shapes, the crop's box and camera, the torch version

The decoder half runs on the float32 backbone's features, as everything after
the backbone runs in float32 in SAM 3D Body itself.
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
ap.add_argument("--out", default="build/sam3d-golden")
args = ap.parse_args()

sys.path.insert(0, args.repo)
from sam_3d_body.build_models import load_sam_3d_body  # noqa: E402
from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator  # noqa: E402
from sam_3d_body.data.utils.prepare_batch import prepare_batch  # noqa: E402
import cv2  # noqa: E402

# The model moves a meshgrid to CUDA; this runs on the CPU.
torch.Tensor.cuda = lambda self, *a, **k: self

os.makedirs(args.out, exist_ok=True)
torch.set_grad_enabled(False)
model, cfg = load_sam_3d_body(args.ckpt, device="cpu", mhr_path=args.mhr)
est = SAM3DBodyEstimator(model, cfg)

img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
cv2.imwrite(os.path.join(args.out, "image.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
box = np.array(args.bbox if args.bbox else [0, 0, w, h], dtype=np.float32).reshape(1, 4)
batch = prepare_batch(img, est.transform, box)
x = model.data_preprocess(batch["img"][0].float())  # [1, 3, 512, 512]


def save(name, t):
    a = t.detach().float().cpu().contiguous().numpy().astype("<f4")
    a.tofile(os.path.join(args.out, name))
    return list(a.shape)


shapes = {"input.bin": save("input.bin", x[0]), "crop.bin": save("crop.bin", batch["img"][0, 0])}

# As the model runs it: bf16 weights and activations, LayerNorms included.
feat = model.backbone(x.type(model.backbone_dtype))
shapes["features_bf16.bin"] = save("features_bf16.bin", feat[0])

# The same weights widened to float32, the cleaner target for the port.
enc = model.backbone.float()
seen = {}
hook = enc.encoder.blocks[0].register_forward_hook(lambda m, i, o: seen.update(block0=o[0] if isinstance(o, (list, tuple)) else o))
feat32 = enc(x.float())
hook.remove()
shapes["block0.bin"] = save("block0.bin", seen["block0"][0])
shapes["features.bin"] = save("features.bin", feat32[0])

# The body decoder, every intermediate prediction recorded as it is made.
# (A forward hook that returns something replaces the module's output.)
seen = {"tokens": []}
hooks = [
    model.ray_cond_emb.register_forward_hook(lambda m, i, o: seen.update(context=o)),
    model.decoder.layers[0].register_forward_hook(lambda m, i, o: seen.update(layer0=o[0])),
    model.head_pose.register_forward_hook(lambda m, i, o: seen["tokens"].append(i[0])),
]
outputs = []
project = model.camera_project
model.camera_project = lambda pose_output, b: outputs.append(project(pose_output, b)) or outputs[-1]
model.backbone_dtype = torch.float32  # the backbone was widened above
model._initialize_batch(batch)
out = model.forward_step(batch, decoder_type="body")
for h in hooks:
    h.remove()
model.camera_project = project
shapes["context.bin"] = save("context.bin", seen["context"][0])
shapes["layer0_tokens.bin"] = save("layer0_tokens.bin", seen["layer0"][0])
shapes["vertices.bin"] = save("vertices.bin", out["mhr"]["pred_vertices"][0])


def plain(v):
    return v.detach().double().cpu().numpy().reshape(-1).tolist()


layers = []
for token, o in zip(seen["tokens"], outputs):
    layers.append({"token": plain(token[0])} | {k: plain(o[k][0]) for k in (
        "pred_pose_raw", "global_rot", "body_pose", "shape", "scale", "hand", "pred_cam",
        "mhr_model_params", "pred_joint_coords", "joint_global_rots", "pred_keypoints_3d",
        "pred_keypoints_2d", "pred_keypoints_2d_cropped", "pred_keypoints_2d_depth", "pred_cam_t")}
        | {"focal_length": float(o["focal_length"][0])})
layers[-1]["hand_box"] = plain(out["mhr"]["hand_box"][0])
layers[-1]["hand_logits"] = plain(out["mhr"]["hand_logits"][0])
json.dump(layers, open(os.path.join(args.out, "layers.json"), "w"))

json.dump({
    "shapes": shapes,
    "bbox": box[0].tolist(),
    "batch": {k: batch[k].double().reshape(-1).tolist() for k in
              ("bbox_center", "bbox_scale", "affine_trans", "cam_int", "ori_img_size", "img_size")},
    "image": os.path.abspath(args.image),
    "torch": torch.__version__,
}, open(os.path.join(args.out, "golden.json"), "w"), indent=1)
print(f"wrote {args.out}: {shapes}")
