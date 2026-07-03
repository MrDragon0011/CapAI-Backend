"""Step 2 — auto-place the 17 COCO pose points on every frame with MediaPipe.

  python auto_annotate.py --frames ./frames

Writes annotations.json next to the frames (or wherever --out points). Each
frame gets all 17 landmarks pre-placed so you only DRAG to correct in the
editor instead of clicking every point from scratch.

MediaPipe's BlazePose model always detects 33 landmarks internally (it can't
be told to only find 17) — we keep that detection (it's still the strongest
free detector available for bootstrapping labels) and then keep only the 17
that correspond to the standard COCO keypoint set, discarding the extra face
(eye corners, mouth) and hand/foot detail (fingers, heel, foot-index) points
that COCO doesn't have and this project doesn't analyze.

Visibility flag per point (COCO convention):
  2 = visible      (MediaPipe confident)
  1 = occluded     (low confidence — e.g. underwater; position is a guess)
  0 = not labelled (off-frame / no signal)

Frames where MediaPipe finds nobody still get an entry (17 points parked in
the centre, flagged no_detection) so you can place them by hand or skip them.
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

from upscale import Upscaler

mp_pose = mp.solutions.pose

# The 17 standard COCO keypoint names, in COCO index order.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Index into MediaPipe's 33-point BlazePose output for each COCO point above,
# in the same order as KEYPOINT_NAMES. BlazePose has three points per eye
# (inner/centre/outer); we take the centre one to match COCO's single point.
BLAZEPOSE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Standard COCO skeleton, as 0-indexed pairs into KEYPOINT_NAMES.
SKELETON = sorted([sorted(pair) for pair in [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]])

VIS_HI = 0.5    # >= this -> visible (2)
VIS_LO = 0.1    # >= this -> occluded (1); below -> not labelled (0)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _vflag(visibility: float) -> int:
    if visibility >= VIS_HI:
        return 2
    if visibility >= VIS_LO:
        return 1
    return 0


def annotate_frame(pose, path: Path, upscaler=None):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Super-resolve only the pixels fed to the detector. MediaPipe returns
    # normalized coords, so we keep the ORIGINAL w/h below and the points still
    # land on the real frame — no rescaling needed.
    if upscaler is not None:
        rgb = upscaler(rgb)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        # Park all 17 points in a vertical line down the centre so they're easy
        # to grab and drag if you decide to label this frame by hand.
        kp = [[round(w * 0.5, 1), round(h * (0.2 + 0.6 * i / 16), 1), 0]
              for i in range(17)]
        return {"file": path.name, "width": w, "height": h,
                "keypoints": kp, "reviewed": False, "no_detection": True}

    landmarks = res.pose_landmarks.landmark
    kp = []
    for bp_i in BLAZEPOSE_TO_COCO:
        lm = landmarks[bp_i]
        x = min(max(lm.x, 0.0), 1.0) * w
        y = min(max(lm.y, 0.0), 1.0) * h
        kp.append([round(x, 1), round(y, 1), _vflag(lm.visibility)])
    return {"file": path.name, "width": w, "height": h,
            "keypoints": kp, "reviewed": False, "no_detection": False}


def annotate_dir(frames_dir, complexity=2, progress=None, only=None,
                 upscale=False, min_side=640) -> dict:
    """Run MediaPipe over images in frames_dir and return the data dict.

    Pure function — does not write to disk. `progress(done, total)` is called
    periodically if supplied (the web UI uses it; the CLI passes a printer).
    `only` is an optional set of filenames to restrict to — used for
    incremental ingest so already-annotated frames aren't redone.
    `upscale` super-resolves low-res frames (shorter side < `min_side`) before
    detection — helps a lot on blurry/compressed footage.
    Raises ValueError if there are no images to process.
    """
    frames_dir = Path(frames_dir)
    files = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() in IMG_EXT)
    if only is not None:
        files = [f for f in files if f.name in only]
    if not files:
        raise ValueError(f"No images found in {frames_dir}")

    upscaler = Upscaler(min_side=min_side) if upscale else None

    images = []
    detected = 0
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=complexity,
                      min_detection_confidence=0.3) as pose:
        for i, f in enumerate(files, 1):
            entry = annotate_frame(pose, f, upscaler)
            if entry is None:
                continue
            images.append(entry)
            if not entry["no_detection"]:
                detected += 1
            if progress and (i % 25 == 0 or i == len(files)):
                progress(i, len(files))

    return {
        "keypoint_names": KEYPOINT_NAMES,
        "skeleton": SKELETON,
        "frames_dir": str(frames_dir.resolve()),
        "images": images,
        "_detected": detected,
    }


def main():
    ap = argparse.ArgumentParser(description="Auto-place 17 pose points per frame.")
    ap.add_argument("--frames", required=True, help="folder of extracted frames")
    ap.add_argument("--out", help="annotations.json path (default: <frames>/annotations.json)")
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2],
                    help="MediaPipe model_complexity (2 = most accurate, default)")
    ap.add_argument("--upscale", action="store_true",
                    help="super-resolve low-res frames before detection "
                         "(helps on blurry/compressed footage)")
    ap.add_argument("--min-side", type=int, default=640,
                    help="only upscale frames whose shorter side is below this "
                         "(default 640)")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        raise SystemExit(f"Not a folder: {frames_dir}")
    out_path = Path(args.out) if args.out else frames_dir / "annotations.json"

    try:
        data = annotate_dir(frames_dir, args.complexity,
                            progress=lambda d, t: print(f"  {d}/{t} frames"),
                            upscale=args.upscale, min_side=args.min_side)
    except ValueError as exc:
        raise SystemExit(str(exc))

    detected = data.pop("_detected")
    out_path.write_text(json.dumps(data, indent=1))
    print(f"\nWrote {out_path}")
    print(f"  {len(data['images'])} frames, pose found in {detected}, "
          f"{len(data['images']) - detected} need manual placement")
    print("Next: python server.py --data", out_path)


if __name__ == "__main__":
    main()
