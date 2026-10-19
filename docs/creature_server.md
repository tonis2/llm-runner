## Creature server

The creature plugin turns video frames and a rigged `.glb` into an animation
of that rig. It needs no model of the animal, so it works for anything a rig
can be built for:

1. **Labels.** You mark the rig's joints on one or more frames.
2. **Tracking.** DINOv3 (SAM 3D Body's backbone) follows those points through
   the clip on the GPU.
3. **Fit.** The rig is posed to match the tracks. Per frame it fits the joint
   turns and the root, together with one orthographic camera. Bone lengths are
   the rig's own; unlabelled joints follow their chain; the legs stay in their
   plane; the feet stay on the floor; every joint stays inside the rig's
   rotation limits, which Rigsmith's player enforces on any clip it plays.
4. **Travel.** The walk's forward movement is recovered from the planted feet.

The backbone stays loaded between requests.

### Run

```sh
c3c build llm-runner
./build/llm-runner creature model=sam-3d-body-dinov3 --server --port 7860
```

`model` is the folder downloaded from
[facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
(gated), or its `model.ckpt`.

The server takes `--bind` and `--port` as every plugin server does. It handles
one request at a time; there is one GPU.

### `POST /animate`

| Field        | Type | Notes |
|--------------|------|-------|
| `rig`        | string or object | One of: a rigged `.glb` (a path, or base64 with or without a `data:` prefix); Rigsmith's `get_skeleton` answer; or `{ names, parents, positions }`. Joint positions are in model space, +Y up. |
| `frames`     | string or string[] | A folder of PNG/JPEG frames, taken in name order (`ffmpeg -i clip.mp4 frames/%05d.png`), or a list of image paths and base64 images. |
| `labels`     | object | `{ "<frame>": { "<joint or marker>": [x, y] } }` in frame pixels. Give one keyframe or more; a keyframe may leave points out. A second keyframe where the legs cross helps a lot, because DINOv3 tells left from right poorly. |
| `markers`    | object | `{ "<label>": "<joint>" }`. A marker is a point on the segment that starts at that joint, at an offset the fit learns: an eye on `head`, or the skin over a thigh on `hip_R`. Use one for anything that is not exactly at a joint. |
| `region`     | [x1, y1, x2, y2] | The part of every frame to look at: the animal plus room to move. Defaults to the whole frame. |
| `width`      | number | The region is resized to this width for the backbone; a multiple of 16. Default 1280. |
| `fps`        | number | The clip's frame rate. Default 24. |
| `name`       | string | The clip's name. Default `creature`. |
| `travel`     | bool | Put the walk's travel back on the root. Default true. Without it the animal walks in place. |
| `iterations` | number | Fit iterations. Default 1500. |
| `pitch`      | number | The camera's expected downward angle, in degrees. Default 8. Raise it for a camera that stands close or looks down on the animal (it sees the feet from above); otherwise the fit leans the body to make up the difference. |
| `outputs`    | string[] | Any of `glb`, `clip`, `bvh`, `animated_glb`, `tracks`. Default `["glb", "clip"]`. |

What each output is:

- `glb` is the clip on the rig's skeleton, with no mesh. Rest turns are
  identity and the root is in model space, with `extras.facing: "as-is"`, as
  Rigsmith's and UniMate's clips are.
- `clip` is the arguments for Rigsmith's `apply_clip`. Every frame is a
  keyframe holding each joint's turn against its parent, and the root's slide
  from rest as a fraction of rig height.
- `bvh` is in rig units times 100, so metres become centimetres.
- `animated_glb` is the uploaded `.glb` with the animation added to its own
  nodes. It plays on the mesh in any glTF viewer or Blender.
- `tracks` is the tracks the fit was made from. Pass them to `/fit` to re-fit
  without the GPU.

The reply looks like this. Files are base64.

```json
{ "name": "walk", "frames": 124, "fps": 24, "joints": ["pelvis", "..."],
  "glb": "<base64>", "clip": { "name": "walk", "keyframes": ["..."] },
  "report": { "reprojection_px": { "median": 6.6, "p90": 27.2, "over_30": 0.087 },
              "camera": { "yaw": 77.9, "pitch": 8.0 }, "limits": { "joints": 51, "worst_deg": 5.5 },
              "travel": 0.35,
              "seconds": { "track": 225, "fit": 72, "total": 297 } } }
```

`limits` (for a rig that has them) says how far past its rotation limits the
clip goes at worst, in degrees; Rigsmith plays it held at them.

`reprojection_px` is each tracked point's distance from where the fitted rig
puts it. Points far off in many frames usually mean one of three things: a
label on a different part of the body than the joint it names (a marker is the
fix), a track that swapped legs (add a keyframe there), or a rig whose
proportions differ from the animal's.

One view sees nothing along its line of sight, so a side or three-quarter
view of a walk works best: it shows the legs' swing. Head-on, the fit cannot
tell a limb reaching toward the camera from one hanging back when both draw
alike, and it may pick the wrong one (arms held forward, say).

Errors come back as a 400 (something wrong in the request: an unknown joint, a
glb with no skin, and so on) or a 500, with `{"error": "..."}`. The server
stays up either way.

### `POST /track` and `POST /fit`

`/track` takes `frames`, `labels`, `region`, `width`, and optionally `draw`, a
list of frame numbers. It returns `{ tracks, previews?, report }`. A preview is
a base64 PNG of that frame with every track drawn on it: left points red, right
blue, middle yellow. Look at these before fitting, so you catch a leg swap
early.

`/fit` takes `rig`, the `tracks` from a `/track` reply, `markers` and the rest
of `/animate`'s fields. It returns `/animate`'s reply in about a minute, with
no GPU work.

### `GET /info`

Returns what is loaded, the routes, and the request's fields.

### Example

```sh
ffmpeg -i walk.mp4 frames/%05d.png
curl -s -X POST http://127.0.0.1:7860/animate -d @- <<'EOF' | jq -r .glb | base64 -d > walk.glb
{ "rig": "dinosaur_rigged.glb", "frames": "frames", "fps": 24, "name": "walk",
  "region": [40, 61, 1240, 736],
  "labels": { "0": { "head_tip": [1015, 305], "neck": [755, 295], "knee_L": [645, 470],
                     "ankle_L": [585, 595], "ball_L": [630, 695], "toe_L": [725, 702],
                     "eye": [870, 202] } },
  "markers": { "eye": "head" } }
EOF
```

### From Rigsmith

Rigsmith stores rigs the way this plugin reads and writes them: rest turns
identity, +Y up, facing +Z. Today, without any Rigsmith code:

1. Rig the model in Rigsmith and save it. The saved `<name>_copy.glb` is the
   `rig`. Alternatively, send Rigsmith's `get_skeleton` answer as the `rig`.
2. Label the frames with the rig's joint names, as `get_skeleton` lists them.
3. Bring the result in. Either call `apply_clip` with the reply's `clip`, or
   save the `glb` into Rigsmith's animations library. The clip keeps the rig's
   joint names because of `extras.facing: "as-is"`.
4. Clean up with Rigsmith's own tools: `pin_feet`, `smooth_clip`, `make_loop`.

Rigsmith can also run the plugin in its own process, the way it runs Kimodo,
through `llm::pipeline::Job` once `creature` is bundled
(`tools/bundle_plugins.py`, which bundles kimodo and creature by default).

- **Config:** the job's config is the same JSON as an `/animate` request, plus
  `model`.
- **Outputs:** the clip comes back as the job's outputs, named `glb`,
  `clip.json`, `tracks.json` and `animated.glb`, whichever `outputs` asks for.
- **Progress:** printed as `  step N/M, tracking|fitting` lines.
- **Frames:** pass them as paths, so the config JSON stays small.
