# CapAI Pose Annotator

A local pipeline to build a custom water-polo pose dataset:
extract frames → auto-place the 33 MediaPipe points → drag-to-correct in a
webpage → export a COCO-keypoints dataset.

You never click points from scratch — MediaPipe pre-places all 33 on every
frame, and you just nudge the wrong ones (especially underwater joints).

## Setup

```bash
cd tools/annotator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg is also needed for frame extraction (mac: brew install ffmpeg)
```

## The easy path — do everything in the browser

```bash
python server.py --project ./myproject
```
Open <http://localhost:8000>. Because the project is empty, you get an
**ingest screen**: drop a video file (or paste a YouTube / video URL), pick a
frame rate, and click **Extract & annotate**. It runs ffmpeg + MediaPipe and
drops you straight into the editor. Re-open the same `--project` later and you
can ingest more footage — already-reviewed frames are preserved, and only the
new ones get auto-annotated.

`--fps 0.5` (the default in the form) = one frame every 2 seconds. Use
close-up footage so joints are big enough to place accurately.

That's it — skip to **Correcting them** below. The manual CLI steps are still
available if you prefer them:

<details>
<summary>Manual CLI pipeline (optional)</summary>

### 1. Extract frames
```bash
python extract_frames.py --videos ./videos --out ./frames --fps 0.5
# or:  python extract_frames.py --urls urls.txt --out ./frames --fps 0.5
```
### 2. Auto-place the 33 points
```bash
python auto_annotate.py --frames ./frames
```
### 3. Open the editor on the resulting file
```bash
python server.py --data ./frames/annotations.json
```
</details>

## Correcting them in the browser

Once frames are loaded, the editor opens automatically.

- **Drag** a point to move it.
- **Click** a point to cycle its state: green (visible) → orange (underwater /
  occluded) → grey (not labelled). Use orange for joints you can see roughly
  where they are but are submerged; the model still learns the body geometry.
- **A / D** (or arrows) move between frames. **R** marks the frame reviewed and
  jumps to the next one needing a decision. **E** cycles the selected point's
  state.
- **Decline** (button, or **X**) rejects a bad frame — blurry, wide-angle,
  no clear athlete — so it's dropped from the export without being deleted.
  This is the manual filter; no CLIP or auto-classifier needed.
- Edits autosave to `annotations.json`, so you can stop and resume anytime.

## Export

Click **Export COCO** (only *reviewed* frames are exported). You get:

```
tools/annotator/export/
  images/
  annotations.coco.json
```

Drag that folder into Roboflow (Keypoint Detection project), or train
YOLO-Pose directly against the COCO file.

## Notes

- This is a local dev tool — it is not part of the deployed backend and has no
  secrets. Nothing here is uploaded anywhere; export stays on your machine.
- The 33 keypoint names and the skeleton are MediaPipe BlazePose's, so the
  dataset is drop-in compatible with anything expecting that layout.
