import asyncio
import json
import logging
import os
import shutil
import tempfile
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

_classifier_artifact: dict | None = None


@app.on_event("startup")
def _preload_models():
    global _classifier_artifact
    if CLASSIFIER_MODEL_PATH.exists():
        try:
            _classifier_artifact = load_model()
            logger.info("Classifier model loaded at startup.")
        except Exception as exc:
            logger.warning(f"Could not preload classifier model: {exc}")
    else:
        logger.warning(f"Classifier model not found at {CLASSIFIER_MODEL_PATH}; will fail at request time.")

    if not HOLISTIC_MODEL_PATH.exists():
        logger.warning(f"Holistic landmark model not found at {HOLISTIC_MODEL_PATH}.")


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception):
    detail = f"{type(exc).__name__}: {exc}"
    logger.error(f"Unhandled exception on {request.method} {request.url}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": detail})


@app.get("/")
def read_root():
    return {"message": "Hello from CapAI backend"}


@app.get("/health")
def health():
    issues = []
    if not HOLISTIC_MODEL_PATH.exists():
        issues.append(
            f"Missing holistic_landmarker.task at {HOLISTIC_MODEL_PATH}. "
            "Download from the MediaPipe model repository and place in the backend directory."
        )
    if not CLASSIFIER_MODEL_PATH.exists():
        issues.append(
            f"Missing classifier_model.joblib at {CLASSIFIER_MODEL_PATH}. "
            "Run classifier.train() with labelled data to generate it."
        )
    return {"status": "ok" if not issues else "degraded", "issues": issues}


async def _run_pipeline(tmp_video: str, landmarks_path: str):
    loop = asyncio.get_running_loop()

    if not HOLISTIC_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Holistic landmark model not found at {HOLISTIC_MODEL_PATH}. "
                "Download holistic_landmarker.task from the MediaPipe model repository "
                "and place it in the backend directory."
            ),
        )

    try:
        await loop.run_in_executor(None, extract_landmarks, tmp_video, landmarks_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Landmark extraction failed: {traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Landmark extraction failed: {type(e).__name__}: {e}")

    if _classifier_artifact is None and not CLASSIFIER_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Classifier model not found at {CLASSIFIER_MODEL_PATH}. "
                "Run classifier.train() with labelled data to generate classifier_model.joblib."
            ),
        )

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
        logger.error(f"Action classification failed: {traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Action classification failed: {type(e).__name__}: {e}")

    try:
        result = await loop.run_in_executor(
            None,
            lambda: analyse(action, landmarks_path),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Biomechanical analysis failed: {traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Biomechanical analysis failed: {type(e).__name__}: {e}")

    return action, result


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".mp4"

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_video = str(tmp_dir / f"upload{suffix}")
    landmarks_path = str(tmp_dir / "landmarks.json")

    try:
        loop = asyncio.get_running_loop()
        raw = await file.read()
        await loop.run_in_executor(None, lambda: Path(tmp_video).write_bytes(raw))

        try:
            action, result = await asyncio.wait_for(
                _run_pipeline(tmp_video, landmarks_path),
                timeout=PIPELINE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Analysis exceeded the {PIPELINE_TIMEOUT}s time limit. "
                    "Try uploading a shorter clip (under 30 seconds)."
                ),
            )

        try:
            with open(landmarks_path) as f:
                seq_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read landmark data: {type(e).__name__}: {e}")

    except Exception as e:
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

    return JSONResponse(content={
        "action": result["action"],
        "label": result["label"],
        "overall_elite_score_pct": result["overall_elite_score_pct"],
        "priority_focus": result["priority_focus"],
        "total_frames_analysed": result["total_frames_analysed"],
        "metrics": result["metrics"],
        "landmarks": landmarks_payload,
    })
