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

# Water polo ball is a saturated yellow. We deliberately do NOT match white,
# because pool footage is full of white splash, foam and caps that would all
# read as "ball". Tight, saturated yellow only.
BALL_HSV_LO = (22, 120, 120)
BALL_HSV_HI = (33, 255, 255)
BALL_MIN_AREA_FRAC = 0.0002   # ignore tiny yellow specks
BALL_MAX_AREA_FRAC = 0.08     # ignore huge yellow regions (lane gear, banners)
BALL_MIN_CIRCULARITY = 0.65   # 1.0 = perfect circle; reject ragged splash blobs
BALL_MAX_RESULTS = 1          # only the single best candidate

# Trained YOLOv8 ball detector (preferred over the HSV colour heuristic).
# Falls back to HSV automatically if the weights or ultralytics are missing.
BALL_MODEL_PATH = Path(__file__).parent / "ball_detector.pt"
BALL_CONF = 0.25              # min confidence for a YOLO ball detection
_ball_model = None            # lazily loaded singleton; False = load failed

ANGLE_JOINTS = {
    "elbow_l": (11, 13, 15),
    "elbow_r": (12, 14, 16),
    "knee_l": (23, 25, 27),
    "knee_r": (24, 26, 28),
}

# Detect several candidates then lock onto the most prominent athlete
NUM_POSES = 5
# Pad the person crop before the high-res refinement pass (fraction of bbox)
CROP_PAD = 0.25

MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

# Reject low-confidence detections so we don't emit a degenerate centre-clustered skeleton
MIN_POSE_DETECTION_CONFIDENCE = 0.6
MIN_POSE_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# Preprocessing tuned for pool footage (glare + low wet-skin contrast)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
DEGLARE_THRESHOLD = 230  # pixels brighter than this are blown-out pool highlights
DEGLARE_CAP = 200        # tone them down to this so they stop fooling the detector


def _preprocess(bgr_image):
    """Enhance a pool frame for pose detection: tame glare, then boost contrast.

    Returns a NEW BGR image. The original is kept untouched so colour-based
    ball detection still sees the true pixels.
    """
    img = bgr_image.copy()

    # 1. Deglare: pull down specular highlights off the water surface
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glare = gray > DEGLARE_THRESHOLD
    if glare.any():
        img[glare] = np.minimum(img[glare], DEGLARE_CAP).astype(img.dtype)

    # 2. CLAHE on the L channel: recover wet-skin / underwater contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


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
    _get_ball_model()  # warm the YOLO detector so the first request isn't slow


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


def _get_ball_model():
    """Lazily load the trained YOLO ball detector. Returns the model or None."""
    global _ball_model
    if _ball_model is None:
        if not BALL_MODEL_PATH.exists():
            logger.info("[ball] No ball_detector.pt; using HSV fallback.")
            _ball_model = False
        else:
            try:
                from ultralytics import YOLO
                _ball_model = YOLO(str(BALL_MODEL_PATH))
                logger.info("[ball] Loaded YOLO ball detector from %s", BALL_MODEL_PATH)
            except Exception as exc:
                logger.error("[ball] Failed to load YOLO model (%s); using HSV.", exc)
                _ball_model = False
    return _ball_model or None


def _detect_ball_yolo(model, bgr_image, width: int, height: int):
    """Run the trained YOLO detector and return up to BALL_MAX_RESULTS balls."""
    res = model.predict(bgr_image, conf=BALL_CONF, verbose=False)[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    order = confs.argsort()[::-1][:BALL_MAX_RESULTS]

    balls = []
    max_wh = float(max(width, height))
    for i in order:
        x0, y0, x1, y1 = xyxy[i]
        cx = (float(x0) + float(x1)) / 2.0
        cy = (float(y0) + float(y1)) / 2.0
        r = max(float(x1) - float(x0), float(y1) - float(y0)) / 2.0
        balls.append({
            "x": round(cx / width, 4),
            "y": round(cy / height, 4),
            "r": round(r / max_wh, 4),
        })
    return balls


def _detect_ball(bgr_image, width: int, height: int):
    """Detect the ball, preferring the trained YOLO model over HSV colour.

    Tries the YOLO detector first; on any failure (or if weights are absent)
    falls back to the saturated-yellow heuristic so the endpoint never breaks.
    """
    model = _get_ball_model()
    if model is not None:
        try:
            return _detect_ball_yolo(model, bgr_image, width, height)
        except Exception as exc:
            logger.error("[ball] YOLO inference failed (%s); using HSV.", exc)
    return _detect_ball_hsv(bgr_image, width, height)


def _detect_ball_hsv(bgr_image, width: int, height: int):
    """Return up to BALL_MAX_RESULTS {x, y, r} dicts (normalized 0..1).

    Strategy: mask saturated yellow, then keep only blobs that are actually
    round (high circularity) and a sensible size. This rejects the splash,
    foam and white caps that wrecked the naive colour+Hough approach.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(BALL_HSV_LO), np.array(BALL_HSV_HI))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(width * height)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < BALL_MIN_AREA_FRAC * frame_area or area > BALL_MAX_AREA_FRAC * frame_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perim * perim)
        if circularity < BALL_MIN_CIRCULARITY:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        # Score: prefer rounder and larger blobs
        candidates.append((circularity * area, cx, cy, r))

    candidates.sort(key=lambda t: t[0], reverse=True)
    balls = []
    for _, cx, cy, r in candidates[:BALL_MAX_RESULTS]:
        balls.append({
            "x": round(float(cx) / width, 4),
            "y": round(float(cy) / height, 4),
            "r": round(float(r) / max(width, height), 4),
        })
    return balls


def _pose_score(lm):
    """Rank a pose by prominence: bbox area weighted by mean visibility.

    Picks the biggest, most-confident body in the frame so we lock onto the
    athlete taking the shot instead of a distant/partial swimmer.
    """
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    area = max(1e-6, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    mean_vis = sum(p.visibility for p in lm) / len(lm)
    return area * mean_vis


def _select_best_pose(pose_landmarks_list):
    """Return the index of the most prominent athlete among detected poses."""
    best_i, best_score = 0, -1.0
    for i, lm in enumerate(pose_landmarks_list):
        s = _pose_score(lm)
        if s > best_score:
            best_i, best_score = i, s
    return best_i


def _bbox_px(lm, width, height, pad):
    """Padded pixel bounding box (x0, y0, x1, y1) around a pose."""
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0 = int(max(0, (x0 - px) * width))
    y0 = int(max(0, (y0 - py) * height))
    x1 = int(min(1.0, (x1 + px)) * width)
    y1 = int(min(1.0, (y1 + py)) * height)
    return x0, y0, x1, y1


def _refine_pose(landmarker, bgr_image, lm, width, height):
    """Re-run pose on a crop around the athlete for higher landmark precision.

    Returns (refined_landmarks, ok). Landmarks are remapped to full-frame
    normalized coords. Falls back to the original pose if refinement fails.
    """
    x0, y0, x1, y1 = _bbox_px(lm, width, height, CROP_PAD)
    cw, ch = x1 - x0, y1 - y0
    # Skip if the athlete already fills most of the frame (nothing to gain)
    if cw < 32 or ch < 32 or (cw * ch) >= 0.6 * width * height:
        return lm, False

    crop = bgr_image[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        res = landmarker.detect(mp_crop)
    except Exception:
        return lm, False
    if not res.pose_landmarks:
        return lm, False

    refined = res.pose_landmarks[_select_best_pose(res.pose_landmarks)]
    # Map crop-normalized landmarks back to full-frame normalized coords
    for p in refined:
        p.x = (x0 + p.x * cw) / width
        p.y = (y0 + p.y * ch) / height
    return refined, True


def _landmarks_payload(lm):
    # lm = result.pose_landmarks[0] — image-normalized (0..1), NOT world landmarks (metres)
    return [[round(p.x, 4), round(p.y, 4), round(p.visibility, 4)] for p in lm]


def _frame_result(lm, width, height, index, bgr_image=None):
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
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _video_landmarker():
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
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
    enhanced = _preprocess(image)  # for pose detection only
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frames = []
    with _image_landmarker() as landmarker:
        result = landmarker.detect(mp_image)
        if result.pose_landmarks:
            # Lock onto the most prominent athlete, then refine on a crop
            best = result.pose_landmarks[_select_best_pose(result.pose_landmarks)]
            best, refined = _refine_pose(landmarker, enhanced, best, width, height)
            logger.info("[image] poses=%d refined=%s", len(result.pose_landmarks), refined)
            # Ball detection uses the ORIGINAL frame (true colours)
            frames.append(_frame_result(best, width, height, 0, image))

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

    with _video_landmarker() as landmarker, _image_landmarker() as refiner:
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
            enhanced = _preprocess(image)  # for pose detection only
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_index / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.pose_landmarks:
                # Lock onto the most prominent athlete, then refine on a crop
                best = result.pose_landmarks[_select_best_pose(result.pose_landmarks)]
                best, _ = _refine_pose(refiner, enhanced, best, width, height)
                # Ball detection uses the ORIGINAL frame (true colours)
                frames.append(_frame_result(best, width, height, frame_index, image))
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
