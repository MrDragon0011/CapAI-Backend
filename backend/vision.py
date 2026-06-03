import json
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).parent / "holistic_landmarker.task"
OUTPUT_PATH = Path(__file__).parent / "landmarks_sequence.json"

FRAME_SAMPLE_RATE = int(os.getenv("FRAME_SAMPLE_RATE", "3"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "500"))


def _landmark_to_dict(lm):
    return {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, "visibility", None)}


def extract_landmarks(video_path: str, output_path: str | None = None) -> None:
    out = Path(output_path) if output_path else OUTPUT_PATH

    t_model = time.monotonic()
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    print(f"[vision] Model options created in {time.monotonic()-t_model:.2f}s", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[vision] Video opened: fps={fps:.1f} total_frames={total_video_frames}", flush=True)

    landmarks_sequence = []
    frame_index = 0
    processed = 0

    t_proc = time.monotonic()
    with mp_vision.HolisticLandmarker.create_from_options(options) as landmarker:
        print(f"[vision] Landmarker created in {time.monotonic()-t_proc:.2f}s", flush=True)

        while processed < MAX_FRAMES:
            if frame_index % FRAME_SAMPLE_RATE != 0:
                if not cap.grab():
                    break
                frame_index += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_index / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_data = {
                "frame": frame_index,
                "timestamp_ms": timestamp_ms,
                "pose_landmarks": [_landmark_to_dict(lm) for lm in result.pose_landmarks] if result.pose_landmarks else [],
                "pose_world_landmarks": [_landmark_to_dict(lm) for lm in result.pose_world_landmarks] if result.pose_world_landmarks else [],
                "left_hand_landmarks": [_landmark_to_dict(lm) for lm in result.left_hand_landmarks] if result.left_hand_landmarks else [],
                "right_hand_landmarks": [_landmark_to_dict(lm) for lm in result.right_hand_landmarks] if result.right_hand_landmarks else [],
            }
            landmarks_sequence.append(frame_data)
            frame_index += 1
            processed += 1

            if processed % 50 == 0:
                elapsed = time.monotonic() - t_proc
                print(f"[vision] Processed {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} fps)", flush=True)

    cap.release()

    t_write = time.monotonic()
    with open(out, "w") as f:
        json.dump({"fps": fps, "total_frames": processed, "frames": landmarks_sequence}, f)
    print(f"[vision] Saved {processed} frames to {out} in {time.monotonic()-t_write:.2f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vision.py <video_path>")
        sys.exit(1)
    extract_landmarks(sys.argv[1])
