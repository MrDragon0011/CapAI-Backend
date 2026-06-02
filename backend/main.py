import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from vision import extract_landmarks, OUTPUT_PATH as LANDMARKS_PATH
from classifier import detect_action
from analysis import analyse

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello from CapAI backend"}


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        extract_landmarks(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Landmark extraction failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    try:
        action = detect_action(str(LANDMARKS_PATH))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Action classification failed: {e}")

    try:
        result = analyse(action, str(LANDMARKS_PATH))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Biomechanical analysis failed: {e}")

    return JSONResponse(content={
        "action": result["action"],
        "label": result["label"],
        "overall_elite_score_pct": result["overall_elite_score_pct"],
        "priority_focus": result["priority_focus"],
        "total_frames_analysed": result["total_frames_analysed"],
        "metrics": result["metrics"],
    })
