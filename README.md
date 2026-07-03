# CapAI

I made CapAI because I felt like I never got enough feedback on my game.
Water polo is a pretty small sport compared to something like basketball or
soccer, so there's just not much out there to help you improve, especially
since half of what you're doing is underwater where nobody can see it anyway.

CapAI looks at a photo or video of you playing and shows your skeleton, your
joint angles (elbows, knees), your shoulder tilt, and where the ball is.

The frontend lives at [cap-ai.netlify.app](https://cap-ai.netlify.app). This
repo is the backend that does the analysis, plus the tools I'm using to train
better water-polo-specific models.

## How it works

The backend is a FastAPI service. You send it footage, it sends back JSON —
skeleton landmarks, joint angles, ball positions. All the drawing happens in
the frontend.

### `GET /health`
Returns `{"status": "ok"}` whenever the service is awake.

### `POST /analyze`
`multipart/form-data`:

- `footage` — the uploaded file (MP4, MOV, JPG, PNG; max 200 MB)
- `consent` — string, must be `"accepted"`

Response:

```json
{
  "ok": true,
  "source": {"filename": "...", "type": "video/mp4", "width": 1920,
             "height": 1080, "frames_analyzed": 42},
  "frames": [
    {
      "index": 0,
      "landmarks": [[0.49, 0.41, 0.96], "...33 entries..."],
      "angles": {"elbow_l": 157.0, "elbow_r": 142.3,
                 "knee_l": 96.2, "knee_r": 110.5, "shoulder_tilt": 4.1},
      "visibility": {"tracked": 20, "partial": 7, "estimated": 6},
      "balls": [{"x": 0.51, "y": 0.48, "r": 0.04}],
      "others": [{"landmarks": [[0.3, 0.2, 0.91], "...33 entries..."]}],
      "kinematics_ms": 0.42
    }
  ]
}
```

Invalid input returns HTTP 400 with `{"ok": false, "error": "..."}`.

Angles use the shoulder–elbow–wrist and hip–knee–ankle triplets plus the
shoulder-line tilt from horizontal. Visibility tiers: tracked (>= 0.75),
partial (0.30–0.74), estimated (< 0.30) — that last tier matters a lot in
water polo, because your legs are underwater and the model is basically
guessing. CapAI is honest about that instead of showing you confident numbers
for joints it can't actually see.

The `balls` array holds the detected ball (a YOLOv8 model I trained on water
polo footage, with a color-based fallback). The `others` array holds up to 4
other players in frame so the whole scene can be rendered. Uploaded footage is
processed in a temp directory and deleted immediately after analysis — nothing
is stored.

## Why pool footage is hard (and what I did about it)

Off-the-shelf pose models are trained on land sports. Pools break them:
glare off the water, splash that looks like limbs, and legs that just aren't
visible. The backend does a bunch of work to compensate — deglaring and
contrast enhancement tuned for pool footage, a tiled fallback pass so distant
players in wide shots still get detected, a crop-and-refine pass on the main
athlete, and locking onto the most prominent player instead of whoever the
model finds first.

The longer-term fix is a model that's actually trained on water polo. That's
in progress — see `tools/`:

- **`tools/annotator/`** — a browser-based annotation editor I built for
  labeling pose keypoints on real pool frames. It auto-places points, then you
  drag to fix them and mark which joints are underwater.
- **`tools/synthgen/`** — a Blender script that generates synthetic training
  data: a rigged swimmer in randomized shooting and eggbeater poses, with
  randomized cameras, lighting, cap/suit colors, and a waterline that
  determines per-joint visibility. Renders images with YOLO-pose labels,
  no hand-labeling needed.

## Run it locally

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The pose model (`pose_landmarker.task`) downloads automatically on first start.

## Deploy

The service ships a `Dockerfile` so the MediaPipe runtime gets the GL system
libraries it needs (`libgl1`, `libegl1`, `libgles2`, `libglib2.0-0`). It runs
on a Hugging Face Space (Docker SDK, port 7860); `render.yaml` is also included
if you'd rather deploy on Render.

CORS is locked to `https://cap-ai.netlify.app` and `http://localhost:8000`
(for testing purposes). The `/analyze` endpoint is rate-limited per IP and can
require API keys (set the `API_KEYS` env var) so the backend isn't an open
free-compute faucet.

## Roadmap

- **Water-polo-trained pose model** — fine-tuning YOLO pose on the dataset
  coming out of the annotator + synthetic generator, to replace the generic
  MediaPipe model that was never trained on swimmers.
- **Server-side super-resolution** (needs a GPU tier) — upscaling low-res
  footage with Real-ESRGAN before analysis. Not viable on the current
  CPU-only free tier (~2–5 s per 1080p frame on CPU would blow the request
  timeout on video). For now, image enhancement happens client-side in the
  frontend via transformers.js.

---

I built this because I wanted it to exist. If you play water polo and try it
out, let me know what you think.
