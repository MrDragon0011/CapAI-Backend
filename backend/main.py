import asyncio
import logging
import math
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="CapAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cap-ai.netlify.app",
        "http://localhost:8000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK = 1024 * 1024
SAMPLE_FPS = 3
MAX_FRAMES = 60

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".3gp", ".ts"}

TRACKED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.30

# Water polo ball: yellow/white — HSV ranges for detection
BALL_HSV_RANGES = [
    # Yellow
    ((20, 80, 80), (35, 255, 255)),
    # White / light
    ((0, 0, 180), (180, 60, 255)),
]
BALL_MIN_RADIUS_PX = 8    # ~3 cm ball at ~3 m away, 1080p
BALL_MAX_RADIUS_PX = 120  # close-up shot

ANGLE_JOINTS = {
    "elbow_l": (11, 13, 15),
    "elbow_r": (12, 14, 16),
    "knee_l": (23, 25, 27),
    "knee_r": (24, 26, 28),
}

MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

# Reject low-confidence detections so we don't emit a degenerate centre-clustered skeleton
MIN_POSE_DETECTION_CONFIDENCE = 0.6
MIN_POSE_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6


def _ensure_model() -> bool:
    if MODEL_PATH.exists():
        return True
    try:
        logger.info("[startup] Downloading pose_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info(f"[startup] Pose model saved to {MODEL_PATH}")
        return True
    except Exception as exc:
        logger.error(f"[startup] Failed to download pose model: {exc}")
        return False


@app.on_event("startup")
def _startup():
    _ensure_model()


def _error(message: str, status: int = 400):
    return JSONResponse(status_code=status, content={"ok": False, "error": message})


def _angle(a, b, c) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _shoulder_tilt(lm, aspect: float) -> float:
    dx = (lm[12].x - lm[11].x) * aspect
    dy = lm[12].y - lm[11].y
    tilt = math.degrees(math.atan2(dy, dx))
    if tilt > 90:
        tilt -= 180
    elif tilt < -90:
        tilt += 180
    return tilt


def _compute_kinematics(lm, aspect: float):
    def point(i):
        return (lm[i].x * aspect, lm[i].y)

    angles = {}
    for name, (a, b, c) in ANGLE_JOINTS.items():
        angles[name] = round(_angle(point(a), point(b), point(c)), 1)
    angles["shoulder_tilt"] = round(_shoulder_tilt(lm, aspect), 1)

    tracked = partial = estimated = 0
    for landmark in lm:
        v = landmark.visibility
        if v >= TRACKED_THRESHOLD:
            tracked += 1
        elif v >= PARTIAL_THRESHOLD:
            partial += 1
        else:
            estimated += 1

    visibility = {"tracked": tracked, "partial": partial, "estimated": estimated}
    return angles, visibility


def _detect_ball(bgr_image, width: int, height: int):
    """Return list of {x, y, r} dicts (normalized 0..1) for detected balls."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in BALL_HSV_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(BALL_MIN_RADIUS_PX * 2, 20),
        param1=50,
        param2=18,
        minRadius=BALL_MIN_RADIUS_PX,
        maxRadius=BALL_MAX_RADIUS_PX,
    )
    if circles is None:
        return []
    balls = []
    for (cx, cy, r) in circles[0]:
        balls.append({
            "x": round(float(cx) / width, 4),
            "y": round(float(cy) / height, 4),
            "r": round(float(r) / max(width, height), 4),
        })
    return balls


def _landmarks_payload(lm):
    # lm = result.pose_landmarks[0] — image-normalized (0..1), NOT world landmarks (metres)
    return [[round(p.x, 4), round(p.y, 4), round(p.visibility, 4)] for p in lm]


def _frame_result(pose_landmarks, width, height, index, bgr_image=None):
    lm = pose_landmarks[0]
    aspect = width / height if height else 1.0

    t0 = time.perf_counter()
    angles, visibility = _compute_kinematics(lm, aspect)
    kinematics_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    balls = _detect_ball(bgr_image, width, height) if bgr_image is not None else []

    return {
        "index": index,
        "landmarks": _landmarks_payload(lm),
        "angles": angles,
        "visibility": visibility,
        "kinematics_ms": kinematics_ms,
        "balls": balls,
    }


def _image_landmarker():
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _video_landmarker():
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _analyze_image(path, filename, content_type):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frames = []
    with _image_landmarker() as landmarker:
        result = landmarker.detect(mp_image)
    if result.pose_landmarks:
        # Use pose_landmarks (image-normalized 0..1), never pose_world_landmarks (metres)
        lm0 = result.pose_landmarks[0]
        logger.info("[image] landmark[0] x=%.4f y=%.4f (expect spread across 0..1)", lm0[0].x, lm0[0].y)
        frames.append(_frame_result(result.pose_landmarks, width, height, 0, image))

    return _build_response(filename, content_type, width, height, frames)


def _analyze_video(path, filename, content_type):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every = max(1, int(round(fps / SAMPLE_FPS)))

    frames = []
    frame_index = 0

    with _video_landmarker() as landmarker:
        while len(frames) < MAX_FRAMES:
            if not cap.grab():
                break
            if frame_index % sample_every != 0:
                frame_index += 1
                continue
            ok, image = cap.retrieve()
            if not ok:
                break
            if width == 0 or height == 0:
                height, width = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_index / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.pose_landmarks:
                # Use pose_landmarks (image-normalized 0..1), never pose_world_landmarks (metres)
                if len(frames) == 0:
                    lm0 = result.pose_landmarks[0]
                    logger.info("[video] landmark[0] x=%.4f y=%.4f (expect spread across 0..1)", lm0[0].x, lm0[0].y)
                frames.append(_frame_result(result.pose_landmarks, width, height, frame_index, image))
            frame_index += 1

    cap.release()
    return _build_response(filename, content_type, width, height, frames)


def _build_response(filename, content_type, width, height, frames):
    return {
        "ok": True,
        "source": {
            "filename": filename,
            "type": content_type,
            "width": width,
            "height": height,
            "frames_analyzed": len(frames),
        },
        "frames": frames,
    }


@app.get("/")
def root():
    return {"service": "CapAI Backend", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    footage: UploadFile = File(...),
    consent: str = Form(...),
):
    if consent != "accepted":
        return _error("Consent was not accepted.")

    filename = footage.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix in IMAGE_EXT:
        is_video = False
    elif suffix in VIDEO_EXT:
        is_video = True
    else:
        return _error(
            "Unsupported file type. "
            "Images: JPG, PNG, GIF, BMP, WEBP, TIFF, HEIC. "
            "Videos: MP4, MOV, AVI, MKV, WEBM, M4V, WMV, FLV, 3GP."
        )

    if not MODEL_PATH.exists() and not _ensure_model():
        return _error("Pose model is unavailable on the server.", 503)

    tmp_dir = Path(tempfile.mkdtemp(prefix="capai_"))
    tmp_path = str(tmp_dir / f"footage{suffix}")

    try:
        written = 0
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await footage.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_BYTES:
                    return _error("File exceeds the 200 MB limit.")
                out.write(chunk)

        if written == 0:
            return _error("Uploaded file is empty.")

        loop = asyncio.get_running_loop()
        worker = _analyze_video if is_video else _analyze_image
        content_type = footage.content_type or ""
        try:
            result = await loop.run_in_executor(
                None, worker, tmp_path, filename, content_type
            )
        except ValueError as exc:
            return _error(str(exc))

        return JSONResponse(content=result)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
