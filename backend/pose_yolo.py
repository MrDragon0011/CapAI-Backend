"""Local YOLO26-pose engine — the water-polo pose model, without Roboflow.

The "Water Polo Pose Estimation" Roboflow workflow is, under the hood, a stock
COCO-pretrained `yolo26m-pose` model plus a little post-processing (a filter
that drops ball detections sitting on a player's head/cap). Nothing was
fine-tuned, so there's no reason to call Roboflow's serverless API at runtime —
we run the identical model locally with Ultralytics. Weights auto-download on
first use, exactly like the MediaPipe and ball-detector models already do.

Selected by POSE_ENGINE=yolo (see main.py). The default MediaPipe path is left
completely untouched, so flipping the env var is a safe, reversible switch.

YOLO26-pose emits the 17 COCO keypoints. The rest of the backend (ANGLE_JOINTS,
shoulder tilt) and the frontend skeleton speak 33-point BlazePose indexing, so
we scatter the 17 COCO points into their BlazePose slots and park the other 16
slots at the body centre with visibility 0 — the exact inverse of the
annotator's BLAZEPOSE_TO_COCO mapping. Downstream code needs no changes and the
`landmarks` payload stays 33 entries long.
"""

import logging
import math
import os

logger = logging.getLogger("uvicorn.error")

# Keep Ultralytics' settings/cache in a writable dir regardless of host.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

# COCO keypoint i -> BlazePose slot (inverse of the annotator's BLAZEPOSE_TO_COCO).
COCO_TO_BLAZEPOSE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
BLAZEPOSE_POINTS = 33
FILLER_SLOTS = [s for s in range(BLAZEPOSE_POINTS) if s not in set(COCO_TO_BLAZEPOSE)]
# COCO head keypoints (nose, eyes, ears) -> BlazePose slots, for ball suppression.
HEAD_SLOTS = [COCO_TO_BLAZEPOSE[i] for i in range(5)]

WEIGHTS = os.environ.get("YOLO_POSE_WEIGHTS", "yolo26m-pose.pt")
# Person-box confidence: low so partially-occluded swimmers (half underwater,
# back-facing) still get detected. Keypoint confidence: also permissive, so
# joints the model has a weak-but-real read on (partly submerged limbs) still
# get placed rather than dropped entirely.
POSE_CONF = float(os.environ.get("YOLO_POSE_CONF", "0.25"))   # min person-box confidence
KPT_CONF = float(os.environ.get("YOLO_KPT_CONF", "0.15"))     # min keypoint conf to place a joint

_pose_model = None  # lazy singleton; False = load failed


class _Landmark:
    """Minimal stand-in for a MediaPipe NormalizedLandmark (x, y, visibility)."""

    __slots__ = ("x", "y", "visibility")

    def __init__(self, x=0.0, y=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def get_pose_model():
    """Lazily load YOLO26m-pose. Returns the model or None if it can't load."""
    global _pose_model
    if _pose_model is None:
        try:
            from ultralytics import YOLO
            _pose_model = YOLO(WEIGHTS)  # weights auto-download on first use
            logger.info("[pose-yolo] Loaded %s", WEIGHTS)
        except Exception as exc:
            logger.error("[pose-yolo] Failed to load %s: %s", WEIGHTS, exc)
            _pose_model = False
    return _pose_model or None


def warm() -> bool:
    """Load the model now so the first user request doesn't pay the download +
    load cost inside its (short) timeout window. Returns True if it's ready."""
    return get_pose_model() is not None


def _person_to_landmarks(kpts_xy, kpts_conf, width: int, height: int):
    """Scatter 17 COCO keypoints into a 33-slot BlazePose-shaped landmark list.

    Placed slots carry the model's per-keypoint confidence as visibility; the
    other 16 slots are parked at the body centre with visibility 0 so they stay
    out of the angle math and render as 'estimated'. Parking them at the centre
    (rather than 0,0) keeps the full-list bbox used by _pose_score correct.
    """
    lm = [_Landmark() for _ in range(BLAZEPOSE_POINTS)]
    xs, ys = [], []
    for coco_i, slot in enumerate(COCO_TO_BLAZEPOSE):
        conf = float(kpts_conf[coco_i]) if kpts_conf is not None else 1.0
        nx = float(kpts_xy[coco_i][0]) / width
        ny = float(kpts_xy[coco_i][1]) / height
        lm[slot].x = nx
        lm[slot].y = ny
        # Drop keypoints the model isn't confident about (visibility 0 -> the
        # frontend skips them), so uncertain/underwater joints aren't drawn as
        # stray points instead of silently trusting a bad guess.
        if conf >= KPT_CONF:
            lm[slot].visibility = round(conf, 4)
            xs.append(nx)
            ys.append(ny)

    cx = sum(xs) / len(xs) if xs else 0.5
    cy = sum(ys) / len(ys) if ys else 0.5
    for slot in FILLER_SLOTS:
        lm[slot].x = cx
        lm[slot].y = cy
    return lm


def detect_poses(bgr_image, width: int, height: int):
    """Run YOLO26-pose on a BGR frame; return a list of 33-slot landmark lists.

    Returns [] on any failure so the endpoint degrades gracefully instead of
    breaking (mirrors the ball detector's fail-open behaviour).
    """
    model = get_pose_model()
    if model is None:
        return []
    try:
        res = model.predict(bgr_image, conf=POSE_CONF, verbose=False)[0]
    except Exception as exc:
        logger.error("[pose-yolo] inference failed: %s", exc)
        return []

    kp = getattr(res, "keypoints", None)
    if kp is None or kp.data is None or len(kp.data) == 0:
        return []
    data = kp.data.cpu().numpy()  # (num_people, 17, 3) -> x_px, y_px, conf
    poses = []
    for person in data:
        poses.append(_person_to_landmarks(person[:, :2], person[:, 2], width, height))
    return poses


def suppress_head_balls(balls, poses, width: int, height: int):
    """Drop ball detections sitting on a player's head/cap.

    Ports the workflow's Head_Suppressed_Ball_Filter: water-polo caps are round
    and read as balls, so any ball centre landing on a detected head keypoint is
    a false positive. `balls` are normalized {x, y, r} dicts; returns the kept
    subset (same shape).
    """
    if not balls or not poses:
        return balls
    heads = []
    for lm in poses:
        for slot in HEAD_SLOTS:
            p = lm[slot]
            if p.visibility > 0.05:
                heads.append((p.x * width, p.y * height))
    if not heads:
        return balls

    max_wh = max(width, height)
    kept = []
    for b in balls:
        bx, by = b["x"] * width, b["y"] * height
        br = b["r"] * max_wh
        if any(math.hypot(bx - hx, by - hy) <= max(28.0, br * 1.35) for hx, hy in heads):
            continue  # sitting on a head — drop it
        kept.append(b)
    return kept
