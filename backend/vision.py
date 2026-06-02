import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).parent / "holistic_landmarker.task"
OUTPUT_PATH = Path(__file__).parent / "landmarks_sequence.json"


def _landmark_to_dict(lm):
    return {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, "visibility", None)}


def extract_landmarks(video_path: str) -> None:
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarks_sequence = []

    with mp_vision.HolisticLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
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
                "face_landmarks": [_landmark_to_dict(lm) for lm in result.face_landmarks] if result.face_landmarks else [],
            }
            landmarks_sequence.append(frame_data)
            frame_index += 1

    cap.release()

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"fps": fps, "total_frames": len(landmarks_sequence), "frames": landmarks_sequence}, f)

    print(f"Saved {len(landmarks_sequence)} frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vision.py <video_path>")
        sys.exit(1)
    extract_landmarks(sys.argv[1])
