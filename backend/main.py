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

ANGLE_JOINTS = {
    "elbow_l": (11, 13, 15),
    "elbow_r": (12, 14, 16),
    "knee_l": (23, 25, 27),
    "knee_r": (24, 26, 28),
}

MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


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


def _landmarks_payload(lm):
    return [[round(p.x, 4), round(p.y, 4), round(p.visibility, 4)] for p in lm]


def _frame_result(pose_landmarks, width, height, index):
    lm = pose_landmarks[0]
    aspect = width / height if height else 1.0

    t0 = time.perf_counter()
    angles, visibility = _compute_kinematics(lm, aspect)
    kinematics_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    return {
        "index": index,
        "landmarks": _landmarks_payload(lm),
        "angles": angles,
        "visibility": visibility,
        "kinematics_ms": kinematics_ms,
    }


def _image_landmarker():
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _video_landmarker():
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
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
        frames.append(_frame_result(result.pose_landmarks, width, height, 0))

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
                frames.append(_frame_result(result.pose_landmarks, width, height, frame_index))
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
