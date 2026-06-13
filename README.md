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
      "kinematics_ms": 0.42
    }
  ]
}
```

Invalid input returns HTTP 400 with `{"ok": false, "error": "..."}`.

Angles use the shoulder–elbow–wrist and hip–knee–ankle triplets plus the
shoulder-line tilt from horizontal, with normalized x scaled by the image
aspect ratio so angles reflect true geometry. Visibility tiers: tracked
(>= 0.75), partial (0.30–0.74), estimated (< 0.30). Uploaded footage is
processed in a temp directory and deleted immediately after analysis.

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
