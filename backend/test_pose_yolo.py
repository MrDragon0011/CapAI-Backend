"""Smoke test for the local YOLO26-pose engine.

Runs with plain `python test_pose_yolo.py` (no test framework needed). The
deterministic checks (COCO->BlazePose scatter, head-ball suppression) run
without the model. The live inference check downloads yolo26m-pose weights and
runs a real forward pass — it's skipped automatically if ultralytics/torch or
network aren't available, so the test still passes in a bare environment.
"""

import sys

import numpy as np

import pose_yolo
from main import (ANGLE_JOINTS, _detect_ball, _frame_result, _others_payload,
                  _select_best_pose)

EXPECTED_FRAME_KEYS = {
    "index", "pose_detected", "landmarks", "angles", "visibility",
    "kinematics_ms", "balls", "others",
}


def test_scatter_shape_and_angles():
    """17 COCO keypoints must land in their BlazePose slots, filling the joints
    the kinematics code reads and leaving a full 33-length landmark list."""
    w, h = 640, 480
    # Fake one person: place every COCO point at a distinct spot, high confidence.
    xy = np.array([[i * 10 + 50, i * 5 + 50] for i in range(17)], dtype=float)
    conf = np.ones(17, dtype=float)
    lm = pose_yolo._person_to_landmarks(xy, conf, w, h)

    assert len(lm) == pose_yolo.BLAZEPOSE_POINTS, "expected a 33-slot landmark list"
    # Every joint used for angles must be a placed (visible) COCO point.
    for name, idxs in ANGLE_JOINTS.items():
        for i in idxs:
            assert lm[i].visibility > 0.0, f"{name} joint slot {i} not placed"
    print("ok: scatter produces 33 slots with angle joints placed")


def test_frame_payload_keys():
    """A YOLO-derived pose flows through the shared frame builder unchanged."""
    w, h = 640, 480
    xy = np.array([[i * 10 + 50, i * 5 + 50] for i in range(17)], dtype=float)
    lm = pose_yolo._person_to_landmarks(xy, np.ones(17), w, h)
    frame = _frame_result(lm, w, h, 0, balls=[], others=[])
    assert set(frame) == EXPECTED_FRAME_KEYS, f"unexpected keys: {set(frame)}"
    assert frame["pose_detected"] is True
    assert len(frame["landmarks"]) == pose_yolo.BLAZEPOSE_POINTS
    print("ok: frame payload has the expected keys and 33 landmarks")


def test_head_ball_suppression():
    """A ball centred on a head keypoint is dropped; a far ball is kept."""
    w, h = 640, 480
    xy = np.array([[i * 10 + 50, i * 5 + 50] for i in range(17)], dtype=float)
    poses = [pose_yolo._person_to_landmarks(xy, np.ones(17), w, h)]
    nose = poses[0][pose_yolo.HEAD_SLOTS[0]]  # nose slot
    on_head = {"x": nose.x, "y": nose.y, "r": 0.02}
    far_away = {"x": 0.95, "y": 0.95, "r": 0.02}

    kept = pose_yolo.suppress_head_balls([on_head, far_away], poses, w, h)
    assert on_head not in kept, "ball on the head should be suppressed"
    assert far_away in kept, "ball far from any head should be kept"
    print("ok: head-ball suppression drops caps, keeps real balls")


def test_live_inference_optional():
    """Load the real model and run one forward pass. Skips if unavailable."""
    model = pose_yolo.get_pose_model()
    if model is None:
        print("skip: yolo26m-pose weights/ultralytics unavailable (expected offline)")
        return
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = pose_yolo.detect_poses(img, 640, 480)
    assert isinstance(poses, list), "detect_poses must return a list"
    print(f"ok: live inference ran, {len(poses)} pose(s) on a blank frame")


if __name__ == "__main__":
    test_scatter_shape_and_angles()
    test_frame_payload_keys()
    test_head_ball_suppression()
    test_live_inference_optional()
    print("\nAll pose_yolo smoke checks passed.")
    sys.exit(0)
