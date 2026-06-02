import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vision import extract_landmarks, OUTPUT_PATH as LANDMARKS_PATH
from classifier import detect_action, MODEL_PATH as CLASSIFIER_MODEL_PATH
from analysis import analyse

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


@app.get("/")
def read_root():
    return {"message": "Hello from CapAI backend"}


@app.get("/health")
def health():
    issues = []
    from vision import MODEL_PATH as HOLISTIC_MODEL_PATH
    if not HOLISTIC_MODEL_PATH.exists():
        issues.append(f"Missing holistic_landmarker.task at {HOLISTIC_MODEL_PATH}")
    if not CLASSIFIER_MODEL_PATH.exists():
        issues.append(f"Missing classifier_model.joblib at {CLASSIFIER_MODEL_PATH}")
    return {"status": "ok" if not issues else "degraded", "issues": issues}


async def _run_pipeline(tmp_path: str):
    loop = asyncio.get_event_loop()

    from vision import MODEL_PATH as HOLISTIC_MODEL_PATH
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
        await loop.run_in_executor(None, extract_landmarks, tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Landmark extraction failed: {e}")

    if not CLASSIFIER_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Classifier model not found at {CLASSIFIER_MODEL_PATH}. "
                "Run classifier.train() with labelled data to generate classifier_model.joblib."
            ),
        )

    try:
        action = await loop.run_in_executor(None, detect_action, str(LANDMARKS_PATH))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Action classification failed: {e}")

    try:
        result = await loop.run_in_executor(None, analyse, action, str(LANDMARKS_PATH))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Biomechanical analysis failed: {e}")

    return action, result


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        try:
            action, result = await asyncio.wait_for(
                _run_pipeline(tmp_path),
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
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    try:
        with open(LANDMARKS_PATH) as f:
            seq_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read landmark data: {e}")

    all_frames = [fr for fr in seq_data["frames"][::3] if fr.get("pose_landmarks")]
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
