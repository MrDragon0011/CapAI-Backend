"""Step 2 — auto-place the 33 pose points on every frame with MediaPipe.

  python auto_annotate.py --frames ./frames

Writes annotations.json next to the frames (or wherever --out points). Each
frame gets all 33 landmarks pre-placed so you only DRAG to correct in the
editor instead of clicking every point from scratch.

Visibility flag per point (COCO convention):
  2 = visible      (MediaPipe confident)
  1 = occluded     (low confidence — e.g. underwater; position is a guess)
  0 = not labelled (off-frame / no signal)

Frames where MediaPipe finds nobody still get an entry (33 points parked in
the centre, flagged no_detection) so you can place them by hand or skip them.
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# The 33 MediaPipe BlazePose landmark names, in index order.
KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Bone connections (0-indexed pairs) so the editor can draw the skeleton.
SKELETON = sorted([sorted(pair) for pair in mp_pose.POSE_CONNECTIONS])

VIS_HI = 0.5    # >= this -> visible (2)
VIS_LO = 0.1    # >= this -> occluded (1); below -> not labelled (0)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _vflag(visibility: float) -> int:
    if visibility >= VIS_HI:
        return 2
    if visibility >= VIS_LO:
        return 1
    return 0


def annotate_frame(pose, path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        # Park all 33 points in a vertical line down the centre so they're easy
        # to grab and drag if you decide to label this frame by hand.
        kp = [[round(w * 0.5, 1), round(h * (0.2 + 0.6 * i / 32), 1), 0]
              for i in range(33)]
        return {"file": path.name, "width": w, "height": h,
                "keypoints": kp, "reviewed": False, "no_detection": True}

    kp = []
    for lm in res.pose_landmarks.landmark:
        x = min(max(lm.x, 0.0), 1.0) * w
        y = min(max(lm.y, 0.0), 1.0) * h
        kp.append([round(x, 1), round(y, 1), _vflag(lm.visibility)])
    return {"file": path.name, "width": w, "height": h,
            "keypoints": kp, "reviewed": False, "no_detection": False}


def main():
    ap = argparse.ArgumentParser(description="Auto-place 33 pose points per frame.")
    ap.add_argument("--frames", required=True, help="folder of extracted frames")
    ap.add_argument("--out", help="annotations.json path (default: <frames>/annotations.json)")
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2],
                    help="MediaPipe model_complexity (2 = most accurate, default)")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        raise SystemExit(f"Not a folder: {frames_dir}")
    out_path = Path(args.out) if args.out else frames_dir / "annotations.json"

    files = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No images found in {frames_dir}")

    images = []
    detected = 0
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=args.complexity,
                      min_detection_confidence=0.3) as pose:
        for i, f in enumerate(files, 1):
            entry = annotate_frame(pose, f)
            if entry is None:
                continue
            images.append(entry)
            if not entry["no_detection"]:
                detected += 1
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)} frames")

    data = {
        "keypoint_names": KEYPOINT_NAMES,
        "skeleton": SKELETON,
        "frames_dir": str(frames_dir.resolve()),
        "images": images,
    }
    out_path.write_text(json.dumps(data, indent=1))
    print(f"\nWrote {out_path}")
    print(f"  {len(images)} frames, pose found in {detected}, "
          f"{len(images) - detected} need manual placement")
    print("Next: python server.py --data", out_path)


if __name__ == "__main__":
    main()
