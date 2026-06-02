import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

LABELS = ["shooting", "passing", "swimming", "goalie"]
SEQUENCE_PATH = Path(__file__).parent / "landmarks_sequence.json"
MODEL_PATH = Path(__file__).parent / "classifier_model.joblib"

POSE_INDICES = list(range(33))
WINDOW_SIZE = 30

POSE_KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
}


def _lm_xyz(landmark_list, index):
    if not landmark_list or index >= len(landmark_list):
        return np.zeros(3)
    lm = landmark_list[index]
    return np.array([lm["x"], lm["y"], lm["z"]])


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _frame_features(frame):
    pose = frame.get("pose_landmarks", [])
    pose_world = frame.get("pose_world_landmarks", [])
    lh = frame.get("left_hand_landmarks", [])
    rh = frame.get("right_hand_landmarks", [])

    kp = {name: _lm_xyz(pose, idx) for name, idx in POSE_KEYPOINTS.items()}
    kp_w = {name: _lm_xyz(pose_world, idx) for name, idx in POSE_KEYPOINTS.items()}

    r_elbow_angle = _angle(kp["right_shoulder"], kp["right_elbow"], kp["right_wrist"])
    l_elbow_angle = _angle(kp["left_shoulder"], kp["left_elbow"], kp["left_wrist"])
    r_shoulder_angle = _angle(kp["right_hip"], kp["right_shoulder"], kp["right_elbow"])
    l_shoulder_angle = _angle(kp["left_hip"], kp["left_shoulder"], kp["left_elbow"])
    r_knee_angle = _angle(kp["right_hip"], kp["right_knee"], kp["right_ankle"])
    l_knee_angle = _angle(kp["left_hip"], kp["left_knee"], kp["left_ankle"])

    mid_shoulder = (kp["left_shoulder"] + kp["right_shoulder"]) / 2
    mid_hip = (kp["left_hip"] + kp["right_hip"]) / 2
    torso_vec = mid_shoulder - mid_hip
    torso_lean = float(np.arctan2(torso_vec[0], torso_vec[1] + 1e-8))

    rw = kp["right_wrist"]
    lw = kp["left_wrist"]
    rw_w = kp_w["right_wrist"]
    lw_w = kp_w["left_wrist"]

    wrist_height_diff = float(rw[1] - lw[1])
    wrist_separation = float(np.linalg.norm(rw - lw))

    rh_centroid = np.mean([np.array([l["x"], l["y"], l["z"]]) for l in rh], axis=0) if rh else np.zeros(3)
    lh_centroid = np.mean([np.array([l["x"], l["y"], l["z"]]) for l in lh], axis=0) if lh else np.zeros(3)
    hand_spread_r = float(np.linalg.norm(rh_centroid - kp["right_wrist"])) if rh else 0.0
    hand_spread_l = float(np.linalg.norm(lh_centroid - kp["left_wrist"])) if lh else 0.0

    pose_flat = np.concatenate([_lm_xyz(pose, i) for i in POSE_INDICES])

    features = np.concatenate([
        pose_flat,
        rw_w, lw_w,
        [r_elbow_angle, l_elbow_angle, r_shoulder_angle, l_shoulder_angle, r_knee_angle, l_knee_angle],
        [torso_lean, wrist_height_diff, wrist_separation, hand_spread_r, hand_spread_l],
    ])
    return features


def _sequence_features(frames):
    per_frame = np.array([_frame_features(f) for f in frames])
    mean = per_frame.mean(axis=0)
    std = per_frame.std(axis=0)
    delta = np.diff(per_frame, axis=0).mean(axis=0) if len(per_frame) > 1 else np.zeros_like(mean)
    return np.concatenate([mean, std, delta])


def load_sequence(path: str = None):
    src = Path(path) if path else SEQUENCE_PATH
    with open(src) as f:
        data = json.load(f)
    return data["frames"]


def _sliding_windows(frames, window_size=WINDOW_SIZE, step=15):
    windows = []
    for start in range(0, max(1, len(frames) - window_size + 1), step):
        windows.append(frames[start:start + window_size])
    return windows


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)),
    ])


def train(training_data: list[tuple[str, str]], model_save_path: str = None):
    X, y = [], []
    le = LabelEncoder()
    le.fit(LABELS)

    for label, sequence_path in training_data:
        frames = load_sequence(sequence_path)
        for window in _sliding_windows(frames):
            X.append(_sequence_features(window))
            y.append(label)

    X = np.array(X)
    y_enc = le.transform(y)

    pipeline = build_pipeline()
    pipeline.fit(X, y_enc)

    save_path = Path(model_save_path) if model_save_path else MODEL_PATH
    joblib.dump({"pipeline": pipeline, "label_encoder": le}, save_path)
    print(f"Model saved to {save_path}")
    return pipeline, le


def detect_action(sequence_path: str = None, model_path: str = None) -> str:
    artifact = joblib.load(Path(model_path) if model_path else MODEL_PATH)
    pipeline = artifact["pipeline"]
    le = artifact["label_encoder"]

    frames = load_sequence(sequence_path)
    windows = _sliding_windows(frames)
    X = np.array([_sequence_features(w) for w in windows])

    predictions = pipeline.predict(X)
    counts = np.bincount(predictions, minlength=len(le.classes_))
    majority = int(np.argmax(counts))
    label = le.inverse_transform([majority])[0]
    return label


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classifier.py detect [sequence.json] [model.joblib]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "detect":
        seq = sys.argv[2] if len(sys.argv) > 2 else None
        mdl = sys.argv[3] if len(sys.argv) > 3 else None
        action = detect_action(seq, mdl)
        print(f"Detected action: {action}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
