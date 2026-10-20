## Creature server

The creature plugin turns video frames and a rigged `.glb` into an animation
of that rig. It needs no model of the animal, so it works for anything a rig
can be built for:

1. **Labels.** You mark the rig's joints on one or more frames.
2. **Tracking.** DINOv3 (SAM 3D Body's backbone) follows those points through
   the clip on the GPU.
3. **Fit.** The rig is posed to match the tracks. Per frame it fits the joint
   turns and the root, together with an orthographic camera for each view
   (one clip, or several of the same moments). Bone lengths are
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
| `markers`    | object | `{ "<label>": "<joint>" }`. A marker is a point on the segment that starts at that joint, at an offset the fit learns: an eye on `head`, or the skin over a thigh on `hip_R`. Use one for anything that is not exactly at a joint. Or `{ "<label>": { "joint": "<joint>", "offset": [x, y, z] } }` when the offset is known (the point's place from the joint at rest, in model units): it is held, and it shows the joint's turn about its own bone. A finder's landmarks come in this way by themselves. |
| `region`     | [x1, y1, x2, y2] | The part of every frame to look at: the animal plus room to move. Defaults to the whole frame. |
| `width`      | number | The region is resized to this width for the backbone; a multiple of 16. Default 1280. |
| `fps`        | number | The clip's frame rate. Default 24. |
| `name`       | string | The clip's name. Default `creature`. |
| `travel`     | bool | Put the walk's travel back on the root. Default true. Without it the animal walks in place. |
| `travel_smooth` | number | The travel's steps are smoothed over this many frames, an odd number; 1 for none. Default 25, about a second: a walk's speed rises and falls with every step in a fit, more than a body's does. |
| `iterations` | number | Fit iterations. Default 1500. |
| `pitch`      | number | The camera's expected downward angle, in degrees. Default 8. Raise it for a camera that stands close or looks down on the animal (it sees the feet from above); otherwise the fit leans the body to make up the difference. |
| `outputs`    | string[] | Any of `glb`, `clip`, `bvh`, `animated_glb`, `tracks`. Default `["glb", "clip"]`. |
| `views`      | object[] | Instead of `frames`, `labels`, `region`, `width` and `pitch`: the same moments filmed by several cameras. See [Two views](#two-views). |
| `finder`     | string | A trained joint finder (`/train`). Views with neither `labels` nor `tracks` have their joints found with it. See [A trained finder](#a-trained-finder-no-labels). |
| `key_every`  | number | With a finder: a keyframe for the fit to start from every so many frames. Default 6. |
| `landmarks`  | bool or string[] | With a finder: which of its landmarks to fit: all (true), none (false), or those listed. By default all but a mirror pair on one joint (a head's two sides, from a finder drawn before render_poses stopped drawing them): each is the other's mirror image, and the finder takes one for the other, which tips the head over. |
| `flip`       | bool | With a finder: look at each crop mirrored as well, each joint's scores there read as its partner's on the other side (`thigh_L` as `thigh_R`), and average the two. Twice the finding time (about 40 s more for a two-view clip), and on the Zidane exam the joints found 10% nearer and the head a degree nearer. A character that is strongly lopsided (a sword in one hand) may do better with `false`. Default true. |
| `weights`    | object | `{ "<term>": weight }` over the fit's defaults, to tune it without code. `GET /info` lists the terms and their defaults. Unknown terms are refused. |
| `sigma`      | number | The reprojection's robust scale in pixels: an error much past it barely pulls. Default 15. |
| `confidence` | number | How a found point's score weighs its pull: its weight to this power. Default 2 for a finder's tracks, 0 (every point pulls fully) for labels and tracked points. |

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
alike, and it may pick the wrong one (arms held forward, say). Two views
settle it.

### Two views

`views` fits one motion to several clips of the same moments, a front and a
side say. Each view has its own camera, and every view's points project the
same pose. What one view leaves along its line of sight, the other sees
across it.

```json
{ "rig": "zidane.glb", "fps": 24, "name": "walk",
  "markers": { "eye_R": "head", "nose": "head", "belt": "hips" },
  "views": [
    { "frames": "front/", "labels": { "5": { "head_top": [381, 60], "...": [0, 0] } } },
    { "frames": "side/",  "labels": { "5": { "head_top": [397, 63], "...": [0, 0] } } } ] }
```

- **One timeline.** Frame N of each view must be the same instant. Cut and
  retime the clips first, if need be, so they step together. `frames` as a
  list of paths may skip or repeat frames to do it. The first view's frame
  count is the clip's length.
- **Each view** is `{ frames, labels, region?, width?, pitch? }`, tracked
  as `/animate` tracks one clip, or `{ tracks, pitch? }` with a `/track`
  reply's tracks. `pitch` defaults to the request's own. Two views need far
  less of it than one: the second shows a lean the first would invent, so 8
  to 10 suits a close front camera that one view alone needed 25 for.
- **Joint names and `markers`** are shared by every view. Label each view
  only with what it shows. The side view leaves out the far limbs where they
  are hidden; the front view leaves out feet that go off the frame.
- **Labels on the same keyframes** in every view work best.
- **Label at least one joint on the head** (`head_top`, say), not only
  markers. A marker's offset is fitted, so eyes and a nose alone cannot tell
  a head held level from one bowed, and the fit may bow it.
- **The reply** is as `/animate`'s. `report.views` gives each view's
  `reprojection_px` and `camera`; the top-level ones pool every view and give
  the first view's camera. `outputs: ["tracks"]` returns a list, one per view.

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

### A trained finder: no labels

Labels and tracking are where a fit usually goes wrong: a track drifts or
swaps legs, and a keyframe misses the joints a view hides. A **finder** does
without them. It is a joint detector made for one rig, the way the "train
DINO on the character's own mesh" method does it:

1. **Draw the rig.** Rigsmith's `render_poses` MCP tool draws the rigged
   model in thousands of poses inside its own joint limits: random ones, and
   frames of the clips you give it, nudged. Each pose is drawn from the
   cameras you give (front and side, say), jittered, under a light of its
   own, on a transparent background. Every joint's pixel is written beside
   each picture, and with `landmarks` so are points of the mesh: the face and
   the back of the head, the chest, the hips, the heels. The labels cost nothing, and
   every one is exact.
2. **Train.** `POST /train` lays each picture over a background with jittered
   colours, encodes it with DINOv3, and fits a small head on the patch
   features. The head scores each patch for each joint, and its softmax over
   the picture is fitted to a Gaussian at the joint's pixel.
3. **Find.** `POST /detect` runs the finder over a clip's frames and returns
   every joint on every frame, in `/track`'s format. Or give `/animate` a
   `finder` and views with no labels: it finds the joints in each view, then
   fits as usual.

The finder only knows the rig it was trained on, and the joints it was trained
on: by default every joint whose bone is at least 4% of the rig's height
(fingers are too small for 16-pixel patches), and the landmarks it was drawn
with. A joint sits inside the body, where nothing in the picture marks it; a
nose or a heel shows, and shows which way the head or the foot is turned,
which no joint can: the head's turn about its own neck leaves both its ends
where they were. The fit takes a finder's landmarks as markers with known
offsets.

#### Drawing the training set

`render_poses` is hidden from Rigsmith's tool list; call it by name (or
`crig.ops.renderPoses` from `run_script`). One call draws `count` poses from
index `first`. The same seed and index always draw the same picture, so a set
is drawn a few hundred poses at a time:

```sh
for first in 0 500 1000; do
  curl -s -X POST http://127.0.0.1:8808/mcp -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,
    "method":"tools/call","params":{"name":"render_poses","arguments":{"out":"/data/zidane_renders",
    "count":500,"first":'$first',"clips":"all","landmarks":"auto",
    "cameras":[{"name":"front","yaw":0,"yaw_jitter":20},{"name":"left","yaw":90,"yaw_jitter":20},{"name":"right","yaw":-90,"yaw_jitter":20}]}}}'
done
```

| Argument | Default | Notes |
|----------|---------|-------|
| `out` | | The folder; made when missing. |
| `count`, `first`, `seed` | 100, 0, 1 | Which poses this call draws. |
| `size` | 448 | Square pictures, a multiple of 16. |
| `cameras` | front (yaw 0), side (yaw 90) | `{ name, yaw, pitch? (8), distance? (4), yaw_jitter? (15), pitch_jitter? (8) }`. Yaw is in degrees from in front of the rig (0 looks at its face, 90 at its left side). `distance` is in rig sizes, and only sets how strong the perspective is: the picture is zoomed to the framing. |
| `fill` | [0.45, 0.75] | The share of the picture the joints' box fills, along its longer side. |
| `shift`, `spread`, `heading`, `tilt` | 0.08, 20, 30, 10 | Framing offset; the turn of a joint with no limits; the root's turn and tip, in degrees. |
| `blur` | 0.3 | The share of pictures motion-blurred. |
| `clips`, `clip_share`, `clip_noise` | none, 0.5, 12 | Clips a share of the poses are drawn from (names as `get_scene` lists them, `"pack/clip"` where two packs share a name, or `"all"` for every clip that plays on the rig), each joint nudged by up to `clip_noise` degrees inside its limits and the root turned as a random pose's is. A finder trained mostly on poses a body takes finds them better, and still sees the far reaches of the limits. Leave the clip you examine with out. |
| `light_jitter` | 1 | How far each picture's light strays from the scene's: the sun from anywhere above, brighter or dimmer, warmer or cooler, the shadowed side lifted more or less. 0 for the scene's own light. An exam keeps one light. |
| `landmarks` | none | `"auto"`: follow points of the mesh as well: the front and back of the root, of the joint the shoulders branch from and of the head, and each heel. Each is the vertex furthest that way among those the joint's skin carries. Not the head's two sides: each is the other's mirror image. |
| `motion`, `fps` | false, 24 | Draw an exam instead (below). |
| `clip`, `clip_start`, `clip_speed`, `travel` | | An exam of a library clip instead of a made-up movement, from `clip_start` seconds at `clip_speed`; `travel` keeps its slide across the floor (default false). |

Each call writes `poses_<first>.json`: the joints, their parents and rest
positions; the landmarks (`{ name, joint, vertex, offset }`: the joint each
rides and its place from that joint at rest); for each picture its file, its
camera, every joint's `[x, y, depth, seen]` and every landmark's in `marks`
(`seen` is 1 when the point is inside the picture with no part of the body in
front of it); and every pose's joint turns. The document is not touched.

#### `POST /train`

| Field | Type | Notes |
|-------|------|-------|
| `renders` | string or string[] | `render_poses` folders. |
| `out` | string | Where to write the finder, a `.safetensors`. |
| `joints` | string[] | The joints to find. Default: those big enough to see. |
| `landmarks` | bool | Find the set's landmarks too. Default true. Where a picture shows a landmark it is learnt at its pixel; where it hides it (a nose seen from behind), its heatmap is taught to be flat, so its score there is low and the fit hardly listens to it. |
| `hidden_weight` | number | How much a hidden landmark's flat heatmap counts against a shown one's. Default 0.3. |
| `layers` | number[] | DINOv3 blocks (0 to 30) whose patches are laid beside the last block's before the PCA: a middle block knows better where things are, the last what they are. Default none. |
| `backgrounds` | string | A folder of images to crop backgrounds from, alongside made-up ones (flat, sky and ground, tiles, colour fields, and a pale studio wall and floor). |
| `epochs` | number | Passes over the pictures. Default 40. |
| `rank`, `hidden` | number | The PCA's size and the head's hidden width. Defaults 384 and 256. |
| `batch`, `lr`, `sigma` | number | 32 pictures a step; Adam's rate 0.001; the target Gaussian's width in patches, 1. |
| `holdout`, `seed`, `limit` | number | The share held out to score (0.05); the seed (1); a cap on the pictures used. |

The reply is `{ finder, report }`. The report gives `held_out_px`, the
distance from each held-out picture's joints to where the finder puts them
(`median`, `mean`, `p90`, and `within`, the share inside `near` pixels). It
also gives `worst_joints`, the loss over the epochs, and with landmarks
`landmark_score`: the median score of a shown landmark and of a hidden one,
which should be far apart. Encoding the pictures takes most of the time,
about 0.3 s each on an RX 7800 XT. The head itself trains in a few minutes.
The request blocks until training is done (about 25 minutes for 4500
pictures), so run it with no client timeout.

What to expect, from a 70-joint humanoid with a tail (4500 pictures, front,
left and right cameras, 40 joints found):
- Held out: a median of 7.5 px on 448-pixel pictures. The worst joints are
  the finger roots, where the hands are mistaken for each other.
- An exam from two cameras: joints found a median of 6 px from where they
  were drawn, and fitted a median of 4% of the rig's height from the truth.
- A real two-camera walk: the same body lean and foot spacing as SAM 3D
  Body's two-view fusion gives for that clip, within a degree and a
  centimetre.

#### `POST /detect`

`{ finder, frames, region?, key_every?, flip?, draw? }`, or `{ finder, views:
[{ frames, region? }] }`, returns `{ tracks, previews?, report }`. The tracks
are a list when `views` is given.

Each frame is looked at twice. The first look is at the whole frame (or
`region`), to find the figure. The second is a square crop around the joints
the first look found, framed the way the training pictures were, with the
crop's centre and size smoothed over the clip (its size over about a second,
so it does not follow the toes up and down every step). A joint's place
between the 16-pixel patches comes from the parabola through the log scores
around its peak, which is exact for the Gaussian the finder was trained on: a
joint moving across patches moves smoothly instead of jumping from one
patch's centre to the next. A joint's score is how much of its heatmap lies
near its peak. In the fit, a point pulls as hard as its score allows (see
`confidence`). A second peak far from the first is kept as a
second hypothesis, and the fit takes whichever one its motion agrees with: a
left knee's heatmap often lights the right knee too. A keyframe falls every
`key_every` frames (6); the fit starts from those.

With `finder` in an `/animate` request, every view without `labels` or
`tracks` is found this way.

#### The exam

`render_poses` with `"motion": true` draws one smooth movement inside the
limits, one frame per index, from cameras held still. With `"clip"` it plays
a library clip instead, as Rigsmith's player does (inside the limits, on the
floor), on the spot unless `"travel": true`: a walk filmed the way a video
films one. Its true turns are in the JSON. `tests/creature_exam.js` lays the
frames over a background, finds the joints, fits the rig and grades it:

```sh
# Walk_slow from the video's two cameras, framed as tightly as the video frames it
sh mcp.sh render_poses '{"out":"/data/zidane_exam_walk","clip":"Walk_slow","count":96,"size":768,
  "fill":[0.87,0.87],"shift":0,"cameras":[{"name":"front","yaw":0,"pitch":10,"yaw_jitter":0,"pitch_jitter":0},
  {"name":"side","yaw":78,"pitch":10,"yaw_jitter":0,"pitch_jitter":0}]}'
./build/llm-runner tests/creature_exam.js model=sam-3d-body-dinov3 rig=zidane.glb \
    finder=zidane.finder.safetensors exam=/data/zidane_exam_walk
```

It reports:
- how far the found joints are from the drawn ones, in pixels (passes under
  `found_px`, 12);
- how far each fitted joint is from the truth, measured from the root as a
  share of the rig's height (under `place_pct`, 6%);
- along `chain` (the root to the head by default), the turns a viewer sees:
  each joint's turn against its parent, against the truth's (the worst joint
  under `chain_deg`, 10°); the head's turn in the root's frame (under
  `head_deg`, 12°); wobble, how far each joint strays from its own mean turn
  against how far the truth's does (under `wobble`, 2×); and jitter, the
  change of each joint's frame-to-frame turn against the truth's.

A fit can put every joint near its place and still fold the spine: the chest
bent forward and the neck back, which only the turns show.
`noise=<camera>:<px>` adds that much seeded noise to one camera's found
joints before the fit, since a real video's side view is found less well
than renders are.

Bone directions in degrees are printed too. Short bones (a finger's root, a
toe's tip) turn tens of degrees on a few pixels, so those numbers show where
the fit goes wrong rather than how wrong the clip looks. `tracks_out=` keeps
what was found, and `tracks_in=` fits it again without the GPU.

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
