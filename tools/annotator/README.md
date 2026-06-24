# CapAI Pose Annotator

A local, four-step pipeline to build a custom water-polo pose dataset:
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

## 1. Extract frames

From local videos:
```bash
python extract_frames.py --videos ./videos --out ./frames --fps 0.5
```
From YouTube URLs (one per line in `urls.txt`):
```bash
python extract_frames.py --urls urls.txt --out ./frames --fps 0.5
```
`--fps 0.5` = one frame every 2 seconds. Use close-up footage so joints are
big enough to place accurately.

## 2. Auto-place the 33 points

```bash
python auto_annotate.py --frames ./frames
```
Writes `frames/annotations.json` with every landmark pre-placed and a
visibility flag per point (visible / occluded / not-labelled). Frames where
MediaPipe found nobody are flagged so you can spot them in the editor.

## 3. Correct them in the browser

```bash
python server.py --data ./frames/annotations.json
```
Open <http://localhost:8000>.

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

## 4. Export

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
