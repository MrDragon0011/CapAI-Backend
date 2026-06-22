import asyncio
import base64
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
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="CapAI Backend")

# CORS is the FIRST middleware added so it is the OUTERMOST layer: every
# response — including 4xx/5xx and rate-limit rejections — gets CORS headers,
# otherwise the browser surfaces a backend error as a misleading CORS failure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cap-ai.netlify.app",
        "http://localhost:8000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _client_ip(request: Request) -> str:
    """Real client IP behind HF Spaces' reverse proxy.

    request.client.host is the proxy, so prefer the first hop in
    X-Forwarded-For. Falls back to the socket peer for direct/local calls.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


# Per-IP rate limiting. The default handler returns HTTP 429 with Retry-After.
limiter = Limiter(key_func=_client_ip)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"ok": False, "error": "Too many requests. Please slow down."},
        headers={"Retry-After": "60"},
    )


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    # Never leak a stack trace to the client; log the full detail server-side.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "Internal server error."},
    )


MAX_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK = 1024 * 1024
SAMPLE_FPS = 3
MAX_FRAMES = 60

# Resource bounds so a small upload can't generate unbounded work / OOM.
MAX_IMAGE_PIXELS = 50_000_000      # 50 MP decoded ceiling (decompression bombs)
MAX_VIDEO_DIM = 4096               # reject >4K-on-the-long-side clips
MAX_VIDEO_SECONDS = 120            # reject very long clips up front
REQUEST_TIMEOUT_S = 90             # abort a single /analyze that runs too long
MAX_CONCURRENT_ANALYSES = 2        # global cap so one client can't saturate CPU
_analysis_sem = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".3gp", ".ts"}

TRACKED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.30
# A joint angle is only meaningful if the landmarks forming it are actually
# visible. Underwater legs come back as low-visibility ESTIMATES, so without
# this gate we'd report confident-looking knee angles for joints we can't see.
ANGLE_MIN_VISIBILITY = 0.5

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
# How many of the non-primary athletes to also return per frame (context only,
# no kinematics) so the UI can draw every player in the scene.
MAX_OTHER_PLAYERS = 4
# Ball temporal carry-forward: even a trained detector drops the ball on
# motion-blur / occlusion frames. Rather than flicker the marker out, carry the
# last known position forward for a few frames so it reads as continuous.
BALL_CARRY_MAX = 8

# Super-resolution (opt-in via sr=accepted, images only). Real-ESRGAN x2plus
# runs on CPU but is slow — input is capped so the upscale finishes within the
# request timeout. On a GPU-backed instance the cap can be raised significantly.
SR_INPUT_CAP = 400       # cap long edge before upscaling (→ max 800 px output)
_sr_model = None         # lazy singleton; False = load attempted and failed

MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

# Reject low-confidence detections so we don't emit a degenerate centre-clustered
# skeleton. Water polo bodies are inherently low-confidence (legs submerged, lots
# of back-facing shots), so we sit lower than the land-footage default of ~0.5.
MIN_POSE_DETECTION_CONFIDENCE = 0.3
MIN_POSE_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.4

# Tiled fallback for wide shots. When whole-frame detection finds nobody, split
# the frame into an overlapping grid and detect per-tile so small, distant
# players appear large enough for MediaPipe's person detector to fire. Overlap
# stops a body that straddles a tile seam from being missed by both tiles.
TILE_GRID = (2, 2)
TILE_OVERLAP = 0.15

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
    # Ball model is loaded lazily on first request to avoid OOM at startup


def _error(message: str, status: int = 400):
    return JSONResponse(status_code=status, content={"ok": False, "error": message})


def _sniff_kind(path: str):
    """Identify the real media type from magic bytes, ignoring the filename.

    Returns "image", "video", or None (unknown / not a media file). This is the
    source of truth for routing — a .jpg-renamed .zip or a text file sniffs as
    None and is rejected before any expensive decode work.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(32)
    except OSError:
        return None
    if len(head) < 12:
        return None

    # ISO base media (mp4/mov/m4v/3gp + heic/heif) share the ftyp box.
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        heif_brands = {b"heic", b"heix", b"hevc", b"heim", b"heis",
                       b"mif1", b"msf1", b"avif"}
        return "image" if brand in heif_brands else "video"

    # Images
    if head[:3] == b"\xff\xd8\xff":
        return "image"                                   # jpeg
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"                                   # png
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image"                                   # gif
    if head[:2] == b"BM":
        return "image"                                   # bmp
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"                                   # webp
    if head[:2] in (b"II", b"MM"):
        return "image"                                   # tiff

    # Videos
    if head[:4] == b"\x1aE\xdf\xa3":
        return "video"                                   # matroska / webm
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "video"                                   # avi
    if head[:3] == b"FLV":
        return "video"                                   # flv
    if head[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"):
        return "video"                                   # mpeg ps / ts
    if head[:4] == b"OggS":
        return "video"                                   # ogg
    return None


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


def _visible(lm, *idxs):
    """True only if every listed landmark is confidently visible."""
    return all(lm[i].visibility >= ANGLE_MIN_VISIBILITY for i in idxs)


def _compute_kinematics(lm, aspect: float):
    def point(i):
        return (lm[i].x * aspect, lm[i].y)

    # Report an angle only when its joints are actually visible; otherwise null
    # so the UI greys it out instead of showing a bogus underwater estimate.
    angles = {}
    for name, (a, b, c) in ANGLE_JOINTS.items():
        if _visible(lm, a, b, c):
            angles[name] = round(_angle(point(a), point(b), point(c)), 1)
        else:
            angles[name] = None
    angles["shoulder_tilt"] = (
        round(_shoulder_tilt(lm, aspect), 1) if _visible(lm, 11, 12) else None
    )

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


def _empty_angles():
    a = {name: None for name in ANGLE_JOINTS}
    a["shoulder_tilt"] = None
    return a


def _others_payload(pose_landmarks_list, best_index):
    """Landmarks for every detected athlete except the primary one.

    Context only — no angles or refinement — so the UI can render the whole
    scene without the cost of full kinematics on each extra player.
    """
    others = []
    for i, lm in enumerate(pose_landmarks_list):
        if i == best_index:
            continue
        others.append({"landmarks": _landmarks_payload(lm)})
        if len(others) >= MAX_OTHER_PLAYERS:
            break
    return others


class _BallTracker:
    """Per-video ball smoother: carry the last detection forward over gaps.

    A frame's raw ball detection (0 or 1 ball) goes in; the ball to actually
    emit comes out. When the detector drops the ball we re-emit the last known
    position for up to BALL_CARRY_MAX frames, tagged carried=True so the UI can
    fade it. Stateless across videos — one instance per analysis.
    """

    def __init__(self):
        self._last = None
        self._age = 0

    def update(self, balls):
        if balls:
            self._last = balls[0]
            self._age = 0
            return [{**self._last, "carried": False}]
        if self._last is not None and self._age < BALL_CARRY_MAX:
            self._age += 1
            return [{**self._last, "carried": True}]
        return []


def _frame_result(lm, width, height, index, bgr_image=None, balls=None, others=None):
    """Build one frame's payload.

    `lm` may be None when no pose was detected — we still run ball detection
    and return a frame so the ball (and the image itself) shows up. In that
    case landmarks are empty, angles are null, and pose_detected is False.

    `balls` lets the caller supply already-tracked detections (video carry-
    forward); when None we detect on this frame directly. `others` carries the
    non-primary athletes for whole-scene rendering.
    """
    aspect = width / height if height else 1.0

    t0 = time.perf_counter()
    if lm is not None:
        angles, visibility = _compute_kinematics(lm, aspect)
        landmarks = _landmarks_payload(lm)
        pose_detected = True
    else:
        angles = _empty_angles()
        visibility = {"tracked": 0, "partial": 0, "estimated": 0}
        landmarks = []
        pose_detected = False
    kinematics_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    if balls is None:
        balls = _detect_ball(bgr_image, width, height) if bgr_image is not None else []

    return {
        "index": index,
        "pose_detected": pose_detected,
        "landmarks": landmarks,
        "angles": angles,
        "visibility": visibility,
        "kinematics_ms": kinematics_ms,
        "balls": balls,
        "others": others or [],
    }


def _mp_rgb(bgr_image):
    """Wrap a BGR frame as an RGB MediaPipe Image."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _detect_tiled(detect_image, bgr_image, width, height):
    """Last-resort wide-shot detection: split into overlapping tiles, detect on
    each (so small/distant players are large enough to fire), and remap every
    landmark back to full-frame normalized coords.

    Overlapping tiles can detect the same player twice; that's harmless —
    _select_best_pose still picks the biggest body and _others_payload caps the
    rest, so at worst a duplicate context skeleton is drawn.
    """
    rows, cols = TILE_GRID
    tw, th = width // cols, height // rows
    if tw < 32 or th < 32:
        return []
    ox, oy = int(tw * TILE_OVERLAP), int(th * TILE_OVERLAP)
    found = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * tw - ox)
            y0 = max(0, r * th - oy)
            x1 = min(width, (c + 1) * tw + ox)
            y1 = min(height, (r + 1) * th + oy)
            cw, ch = x1 - x0, y1 - y0
            crop = bgr_image[y0:y1, x0:x1]
            try:
                res = detect_image(_mp_rgb(crop))
            except Exception:
                continue
            for lm in res.pose_landmarks:
                for p in lm:
                    p.x = (x0 + p.x * cw) / width
                    p.y = (y0 + p.y * ch) / height
                found.append(lm)
    return found


def _detect_with_fallback(detect_image, bgr_image, width, height):
    """Detect poses on the enhanced frame, falling back to the raw frame, then
    to a tiled pass for wide shots.

    Pool preprocessing (CLAHE/deglare) usually helps, but on some frames it
    suppresses the athlete entirely. Try enhanced first; if nothing is found,
    retry on the untouched frame; if still nothing, tile the frame so small
    distant players become detectable. Returns (poses, image_those_poses_came
    _from) so the caller can refine on the matching image. `detect_image` is a
    single-argument IMAGE-mode detect callable.
    """
    enhanced = _preprocess(bgr_image)
    res = detect_image(_mp_rgb(enhanced))
    if res.pose_landmarks:
        return res.pose_landmarks, enhanced
    res_raw = detect_image(_mp_rgb(bgr_image))
    if res_raw.pose_landmarks:
        logger.info("[pose] found on raw frame after enhanced failed")
        return res_raw.pose_landmarks, bgr_image
    tiled = _detect_tiled(detect_image, bgr_image, width, height)
    if tiled:
        logger.info("[pose] found %d via tiled fallback", len(tiled))
        return tiled, bgr_image
    return [], enhanced


def _get_sr_model():
    """Lazily load Real-ESRGAN x2plus. Returns the upsampler or None."""
    global _sr_model
    if _sr_model is not None:
        return _sr_model or None
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch
        _sr_device = "cuda" if torch.cuda.is_available() else "cpu"
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                      num_block=23, num_grow_ch=32, scale=2)
        _sr_model = RealESRGANer(
            scale=2,
            model_path=(
                "https://github.com/xinntao/Real-ESRGAN/releases/download"
                "/v0.2.1/RealESRGAN_x2plus.pth"
            ),
            model=net,
            tile=128,        # tiled inference keeps peak RAM predictable on CPU
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),  # float16 only on GPU
            device=_sr_device,
        )
        logger.info("[sr] Real-ESRGAN x2plus loaded on %s", _sr_device)
    except Exception as exc:
        logger.error("[sr] Failed to load SR model (%s); SR disabled.", exc)
        _sr_model = False
    return _sr_model or None


def _upscale(bgr_image):
    """Return a 2x upscaled BGR image, or None if SR is unavailable/fails."""
    h, w = bgr_image.shape[:2]
    long_edge = max(h, w)
    if long_edge > SR_INPUT_CAP:
        scale = SR_INPUT_CAP / long_edge
        bgr_image = cv2.resize(
            bgr_image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    model = _get_sr_model()
    if model is None:
        return None
    try:
        out, _ = model.enhance(bgr_image, outscale=2)
        return out
    except Exception as exc:
        logger.error("[sr] enhance failed: %s", exc)
        return None


def _to_data_uri(bgr_image):
    """Encode a BGR image as a PNG data URI string."""
    ok, buf = cv2.imencode(".png", bgr_image)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/png;base64,{b64}"


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


def _analyze_image(path, filename, content_type, run_sr=False):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    height, width = image.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Image resolution exceeds the 50 MP limit.")

    frames = []
    with _image_landmarker() as landmarker:
        # Detect on the enhanced frame, falling back to the raw frame if pool
        # preprocessing hid the athlete. `src` is whichever image won.
        poses, src = _detect_with_fallback(landmarker.detect, image, width, height)
        best = None
        others = []
        if poses:
            # Lock onto the most prominent athlete, keep the rest as context,
            # then refine the primary one on a crop.
            best_i = _select_best_pose(poses)
            others = _others_payload(poses, best_i)
            best = poses[best_i]
            best, refined = _refine_pose(landmarker, src, best, width, height)
            logger.info("[image] poses=%d refined=%s", len(poses), refined)
        else:
            logger.info("[image] no pose detected (enhanced+raw); ball only")
        # Always emit a frame so the ball (and image) shows even with no pose.
        # Ball detection uses the ORIGINAL frame (true colours).
        frames.append(_frame_result(best, width, height, 0, image, others=others))

    upscaled_uri = None
    if run_sr:
        t0 = time.perf_counter()
        upscaled = _upscale(image)
        if upscaled is not None:
            upscaled_uri = _to_data_uri(upscaled)
            logger.info("[sr] upscaled in %.1fs", time.perf_counter() - t0)
        else:
            logger.warning("[sr] upscale unavailable; omitting upscaled_image")

    return _build_response(filename, content_type, width, height, frames, upscaled_uri)


def _analyze_video(path, filename, content_type, run_sr=False):  # run_sr ignored for video
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max(width, height) > MAX_VIDEO_DIM:
        cap.release()
        raise ValueError("Video resolution exceeds the 4K limit.")
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps > 0 and frame_count > 0 and (frame_count / fps) > MAX_VIDEO_SECONDS:
        cap.release()
        raise ValueError("Video is longer than the 120 second limit.")

    sample_every = max(1, int(round(fps / SAMPLE_FPS)))

    frames = []
    frame_index = 0
    ball_tracker = _BallTracker()

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
            timestamp_ms = int((frame_index / fps) * 1000)
            result = landmarker.detect_for_video(_mp_rgb(enhanced), timestamp_ms)
            poses, src = result.pose_landmarks, enhanced
            if not poses:
                # Preprocessing may have hidden the athlete; retry on the raw
                # frame with the IMAGE-mode refiner (no timestamp ordering).
                raw = refiner.detect(_mp_rgb(image))
                if raw.pose_landmarks:
                    poses, src = raw.pose_landmarks, image
            best = None
            others = []
            if poses:
                # Lock onto the most prominent athlete, keep the rest as context,
                # then refine the primary one on a crop.
                best_i = _select_best_pose(poses)
                others = _others_payload(poses, best_i)
                best = poses[best_i]
                best, _ = _refine_pose(refiner, src, best, width, height)
            # Ball detection uses the ORIGINAL frame (true colours); carry the
            # last position forward over frames where the detector drops it.
            balls = ball_tracker.update(_detect_ball(image, width, height))
            # Always emit a frame so we sample at SAMPLE_FPS regardless of pose.
            frames.append(
                _frame_result(best, width, height, frame_index, image,
                              balls=balls, others=others)
            )
            frame_index += 1

    cap.release()
    return _build_response(filename, content_type, width, height, frames)


def _build_response(filename, content_type, width, height, frames, upscaled_image=None):
    resp = {
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
    if upscaled_image:
        resp["upscaled_image"] = upscaled_image
    return resp


@app.get("/")
def root():
    return {"service": "CapAI Backend", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
@limiter.limit("6/minute")
async def analyze(
    request: Request,
    footage: UploadFile = File(...),
    consent: str = Form(...),
    sr: str = Form(default=""),
):
    if consent != "accepted":
        return _error("Consent was not accepted.")

    # Reject oversized bodies up front using the declared length, before we
    # stream a single byte to disk. The streaming guard below is the real
    # enforcement (the header can lie), this just fails fast on honest clients.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        return _error("File exceeds the 200 MB limit.", 413)

    filename = footage.filename or "upload"
    # Only ever derive a temp-file suffix from a whitelisted extension; never
    # build a filesystem path from the raw client filename (path traversal).
    ext = Path(filename).suffix.lower()
    if ext not in IMAGE_EXT and ext not in VIDEO_EXT:
        ext = ""

    if not MODEL_PATH.exists() and not _ensure_model():
        return _error("Pose model is unavailable on the server.", 503)

    tmp_dir = Path(tempfile.mkdtemp(prefix="capai_"))
    tmp_path = str(tmp_dir / f"footage{ext}")

    try:
        written = 0
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await footage.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_BYTES:
                    return _error("File exceeds the 200 MB limit.", 413)
                out.write(chunk)

        if written == 0:
            return _error("Uploaded file is empty.")

        # Trust the bytes, not the extension or Content-Type, to route.
        kind = _sniff_kind(tmp_path)
        if kind is None:
            return _error(
                "Unsupported or unrecognized file. "
                "Upload a real image (JPG, PNG, GIF, BMP, WEBP, TIFF, HEIC) "
                "or video (MP4, MOV, AVI, MKV, WEBM, FLV, 3GP)."
            )
        is_video = kind == "video"
        run_sr = (sr == "accepted") and not is_video

        loop = asyncio.get_running_loop()
        worker = _analyze_video if is_video else _analyze_image
        content_type = footage.content_type or ""
        try:
            async with _analysis_sem:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, worker, tmp_path, filename, content_type, run_sr
                    ),
                    timeout=REQUEST_TIMEOUT_S,
                )
        except asyncio.TimeoutError:
            return _error("Analysis timed out. Try a shorter or smaller clip.", 504)
        except ValueError as exc:
            return _error(str(exc))

        return JSONResponse(content=result)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
