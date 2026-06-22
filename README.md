# CapAI Backend

FastAPI service for water polo biomechanical analysis. Detects the 33-point
MediaPipe Pose skeleton and computes per-frame joint kinematics for uploaded
images or video.

## Endpoints

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
shoulder-line tilt from horizontal, with normalized x scaled by the image
aspect ratio so angles reflect true geometry. Visibility tiers: tracked
(>= 0.75), partial (0.30–0.74), estimated (< 0.30). The `balls` array holds
detected ball candidates as normalized center `x`/`y` and radius `r` (fraction
of the longer image side); it is empty when no ball is found. The `others`
array holds up to 4 non-primary athlete skeletons (landmarks only, no angles)
for whole-scene rendering. Uploaded footage is processed in a temp directory
and deleted immediately after analysis.

This service returns JSON only — all skeleton/ball rendering lives in the
frontend (CapAI-Frontend, deployed to cap-ai.netlify.app).

## Run locally

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The pose model (`pose_landmarker.task`) downloads automatically on first start.

## Deploy (Render, Docker)

The service ships a `Dockerfile` so the MediaPipe Tasks runtime gets the GL
system libraries it needs (`libgl1`, `libegl1`, `libgles2`, `libglib2.0-0`).
`render.yaml` builds from `backend/Dockerfile`. Render auto-deploys on push to
the connected branch.

CORS is locked to `https://cap-ai.netlify.app` and `http://localhost:8000`.

## Roadmap

### Server-side super-resolution (planned, GPU tier required)

Currently the backend returns JSON only — no image bytes. All rendering happens
client-side in the frontend. Super-resolution (upscaling low-res source images
via Real-ESRGAN or a similar model) was evaluated but is not viable on the
current CPU-only free tier:

- Real-ESRGAN 2× on a 1080p frame takes ~2–5 s on CPU; video (up to 60
  frames) would blow the 90 s request timeout entirely.
- Returning upscaled frames as base64 adds 8–15 MB per frame to the response.

Once the backend moves to a GPU-backed instance (e.g. Hugging Face Spaces Pro,
Render GPU, or a self-hosted machine with an NVIDIA card), server-side SR
becomes practical:

1. After `_analyze_image` / `_analyze_video` runs, upscale each source frame
   with Real-ESRGAN (2× recommended for speed/quality balance).
2. Encode the upscaled frame as JPEG base64 and add it to the frame payload
   (`"image_b64": "..."`).
3. Update `source.width` / `source.height` to the upscaled dimensions.
4. The frontend swaps its locally-held source image for the server-returned
   one and renders the canvas at the new resolution.

In the meantime, image quality improvement is handled client-side in the
frontend via browser-based SR (transformers.js).
