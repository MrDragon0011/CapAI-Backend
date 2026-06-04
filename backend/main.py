import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vision import extract_landmarks, MODEL_PATH as HOLISTIC_MODEL_PATH
from classifier import detect_action, load_model, MODEL_PATH as CLASSIFIER_MODEL_PATH
from analysis import analyse

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "150"))
MAX_LANDMARK_FRAMES = 200
UPLOAD_CHUNK = 1024 * 1024

_classifier_artifact: dict | None = None


def _try_download_holistic_model() -> bool:
    if HOLISTIC_MODEL_PATH.exists():
        return True
    script = Path(__file__).parent / "download_models.sh"
    if not script.exists():
        return False
    try:
        logger.info("[startup] holistic_landmarker.task missing — running download_models.sh")
        result = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info(f"[startup] download_models.sh succeeded:\n{result.stdout.strip()}")
            return HOLISTIC_MODEL_PATH.exists()
        logger.error(f"[startup] download_models.sh failed (exit {result.returncode}):\n{result.stderr.strip()}")
        return False
    except Exception as exc:
        logger.error(f"[startup] download_models.sh raised: {exc}")
        return False


@app.on_event("startup")
def _preload_models():
    global _classifier_artifact

    if _try_download_holistic_model():
        logger.info(f"[startup] Holistic model ready at {HOLISTIC_MODEL_PATH}")
    else:
        logger.error(
            "[startup] holistic_landmarker.task not found and could not be downloaded. "
            "Place the file in the backend directory or ensure download_models.sh can reach "
            "storage.googleapis.com at build time. The /analyze endpoint will return 503."
        )

    if CLASSIFIER_MODEL_PATH.exists():
        try:
            t0 = time.monotonic()
            _classifier_artifact = load_model()
            logger.info(f"[startup] Classifier model loaded in {time.monotonic()-t0:.2f}s")
        except Exception as exc:
            logger.error(
                f"[startup] classifier_model.joblib exists but failed to load: {exc}. "
                "Re-generate it by running classifier.train() with labelled data."
            )
    else:
        logger.warning(
            f"[startup] classifier_model.joblib not found at {CLASSIFIER_MODEL_PATH}. "
            "The /analyze endpoint will return 503 until a trained model is present."
        )


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception):
    detail = f"{type(exc).__name__}: {exc}"
    logger.error(f"[unhandled] {request.method} {request.url}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": detail})


@app.get("/")
def read_root():
    return {"message": "Hello from CapAI backend"}


@app.get("/health")
def health():
    issues = []
    if not HOLISTIC_MODEL_PATH.exists():
        issues.append(
            f"holistic_landmarker.task not found at {HOLISTIC_MODEL_PATH}. "
            "It is downloaded automatically at startup — check Render build logs."
        )
    if not CLASSIFIER_MODEL_PATH.exists():
        issues.append(
            f"classifier_model.joblib not found at {CLASSIFIER_MODEL_PATH}. "
            "Run classifier.train() with labelled landmark sequences to generate it."
        )
    return {
        "status": "ok" if not issues else "degraded",
        "issues": issues,
        "classifier_preloaded": _classifier_artifact is not None,
        "holistic_model_present": HOLISTIC_MODEL_PATH.exists(),
    }


def _stream_upload_to_disk(file_obj, dest: str) -> int:
    written = 0
    with open(dest, "wb") as out:
        while True:
            chunk = file_obj.read(UPLOAD_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
    return written


async def _run_pipeline(tmp_video: str, landmarks_path: str, req_id: str):
    loop = asyncio.get_running_loop()

    if not HOLISTIC_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "holistic_landmarker.task is missing on the server. "
                "The model is downloaded automatically at startup — check Render build logs. "
                "You can also manually place the file in the backend directory."
            ),
        )

    logger.info(f"[{req_id}] Stage 1/3: starting landmark extraction")
    t0 = time.monotonic()
    try:
        await loop.run_in_executor(None, extract_landmarks, tmp_video, landmarks_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Stage 1/3 FAILED after {time.monotonic()-t0:.1f}s:\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Landmark extraction failed: {type(e).__name__}: {e}")
    logger.info(f"[{req_id}] Stage 1/3 done in {time.monotonic()-t0:.1f}s")

    if _classifier_artifact is None and not CLASSIFIER_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Classifier model not found at {CLASSIFIER_MODEL_PATH}. "
                "Run classifier.train() with labelled data to generate classifier_model.joblib."
            ),
        )

    logger.info(f"[{req_id}] Stage 2/3: classifying action")
    t1 = time.monotonic()
    try:
        action = await loop.run_in_executor(
            None,
            lambda: detect_action(
                sequence_path=landmarks_path,
                artifact=_classifier_artifact,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Stage 2/3 FAILED after {time.monotonic()-t1:.1f}s:\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Action classification failed: {type(e).__name__}: {e}")
    logger.info(f"[{req_id}] Stage 2/3 done in {time.monotonic()-t1:.1f}s — action={action}")

    logger.info(f"[{req_id}] Stage 3/3: biomechanical analysis")
    t2 = time.monotonic()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: analyse(action, landmarks_path),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Stage 3/3 FAILED after {time.monotonic()-t2:.1f}s:\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Biomechanical analysis failed: {type(e).__name__}: {e}")
    logger.info(f"[{req_id}] Stage 3/3 done in {time.monotonic()-t2:.1f}s")

    return action, result


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    req_id = os.urandom(4).hex()
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    content_type = file.content_type or ""
    logger.info(
        f"[{req_id}] /analyze — filename={file.filename!r} "
        f"content_type={content_type!r} suffix={suffix!r}"
    )
    if suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file extension {suffix!r}. Upload an MP4, MOV, or AVI clip.",
        )

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_video = str(tmp_dir / f"upload{suffix}")
    landmarks_path = str(tmp_dir / "landmarks.json")

    try:
        loop = asyncio.get_running_loop()

        logger.info(f"[{req_id}] Streaming upload to disk (chunk={UPLOAD_CHUNK//1024}KB)")
        t_upload = time.monotonic()
        try:
            bytes_written = await loop.run_in_executor(
                None, _stream_upload_to_disk, file.file, tmp_video
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to save uploaded file: {type(e).__name__}: {e}")
        logger.info(f"[{req_id}] Upload saved: {bytes_written/1024/1024:.2f} MB in {time.monotonic()-t_upload:.2f}s")

        if bytes_written == 0:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        logger.info(f"[{req_id}] Starting pipeline (timeout={PIPELINE_TIMEOUT}s)")
        t_pipeline = time.monotonic()
        try:
            action, result = await asyncio.wait_for(
                _run_pipeline(tmp_video, landmarks_path, req_id),
                timeout=PIPELINE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"[{req_id}] Pipeline timed out after {time.monotonic()-t_pipeline:.1f}s")
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Analysis exceeded the {PIPELINE_TIMEOUT}s time limit. "
                    "Try uploading a shorter clip (under 30 seconds)."
                ),
            )
        logger.info(f"[{req_id}] Pipeline complete in {time.monotonic()-t_pipeline:.1f}s")

        try:
            with open(landmarks_path) as f:
                seq_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read landmark data: {type(e).__name__}: {e}")

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    shutil.rmtree(tmp_dir, ignore_errors=True)

    all_frames = [fr for fr in seq_data["frames"] if fr.get("pose_landmarks")]
    sampled = all_frames[:MAX_LANDMARK_FRAMES]

    landmarks_payload = {
        "fps": seq_data["fps"],
        "frames": [
            {
                "t": fr["timestamp_ms"],
                "lm": [
                    [lm["x"], lm["y"], lm.get("visibility") or 1.0]
                    for lm in fr["pose_landmarks"]
                ],
            }
            for fr in sampled
        ],
    }

    logger.info(f"[{req_id}] Responding — frames={len(sampled)} action={action}")
    return JSONResponse(content={
        "action": result["action"],
        "label": result["label"],
        "overall_elite_score_pct": result["overall_elite_score_pct"],
        "priority_focus": result["priority_focus"],
        "total_frames_analysed": result["total_frames_analysed"],
        "metrics": result["metrics"],
        "landmarks": landmarks_payload,
    })
