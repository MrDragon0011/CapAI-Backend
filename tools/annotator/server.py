"""Step 3 — the annotation editor server.

  # Start an empty project and ingest video from the browser:
  python server.py --project ./myproject

  # ...or open an existing dataset directly:
  python server.py --data ./frames/annotations.json

Open http://localhost:8000. If the project is empty you get an ingest screen
— drop a video file or paste a YouTube URL and it extracts frames and
auto-places the 17 points for you, then drops into the editor. Drag points to
fix them, click a point to cycle visibility (visible -> underwater -> off),
Decline bad frames, and Export when done. Export writes a COCO-keypoints
dataset to <project>/export/.

Edits autosave to annotations.json as you go.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from auto_annotate import KEYPOINT_NAMES, SKELETON, annotate_dir
from extract_frames import download_urls, extract_one

HERE = Path(__file__).parent
app = FastAPI(title="CapAI Pose Annotator")

# Populated in main() before the server starts.
DATA_PATH: Path = None
FRAMES_DIR: Path = None
DATA: dict = None


def _empty_data() -> dict:
    return {"keypoint_names": KEYPOINT_NAMES, "skeleton": SKELETON,
            "frames_dir": str(FRAMES_DIR.resolve()), "images": []}


def _load():
    global DATA, FRAMES_DIR
    if DATA_PATH.is_file():
        DATA = json.loads(DATA_PATH.read_text())
        # Prefer the frames_dir baked into the file; fall back to the json's folder.
        fd = DATA.get("frames_dir")
        if fd and Path(fd).is_dir():
            FRAMES_DIR = Path(fd)
    else:
        # Empty project — the UI shows the ingest screen instead of the editor.
        DATA = _empty_data()


def _save():
    DATA_PATH.write_text(json.dumps(DATA, indent=1))


class SaveBody(BaseModel):
    index: int
    keypoints: list[list[float]]
    reviewed: bool
    declined: bool = False


@app.get("/favicon.ico")
@app.get("/favicon.png")
def favicon():
    return FileResponse(HERE / "static" / "favicon.png", media_type="image/png")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/data")
def get_data():
    reviewed = sum(1 for im in DATA["images"] if im.get("reviewed"))
    declined = sum(1 for im in DATA["images"] if im.get("declined"))
    return {
        "keypoint_names": DATA["keypoint_names"],
        "skeleton": DATA["skeleton"],
        "count": len(DATA["images"]),
        "reviewed": reviewed,
        "declined": declined,
        "images": DATA["images"],
    }


@app.get("/frames/{name}")
def frame(name: str):
    # Guard against path traversal — only serve a plain filename from FRAMES_DIR.
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, "bad name")
    p = FRAMES_DIR / name
    if not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.post("/api/save")
def save(body: SaveBody):
    if not (0 <= body.index < len(DATA["images"])):
        raise HTTPException(400, "index out of range")
    im = DATA["images"][body.index]
    im["keypoints"] = [[round(p[0], 1), round(p[1], 1), int(p[2])]
                       for p in body.keypoints]
    im["reviewed"] = body.reviewed
    im["declined"] = body.declined
    _save()
    reviewed = sum(1 for x in DATA["images"] if x.get("reviewed"))
    declined = sum(1 for x in DATA["images"] if x.get("declined"))
    return {"ok": True, "reviewed": reviewed, "declined": declined}


def _bbox(kps):
    """Tight bbox around all labelled (v>0) points, with a little padding."""
    pts = [(x, y) for x, y, v in kps if v > 0]
    if not pts:
        return [0, 0, 0, 0]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    pad_x = (x1 - x0) * 0.1 + 5
    pad_y = (y1 - y0) * 0.1 + 5
    return [round(x0 - pad_x, 1), round(y0 - pad_y, 1),
            round((x1 - x0) + 2 * pad_x, 1), round((y1 - y0) + 2 * pad_y, 1)]


@app.post("/api/export")
def export():
    """Write a COCO-keypoints dataset to ./export (images/ + annotations.coco.json)."""
    out_dir = HERE / "export"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    names = DATA["keypoint_names"]
    skeleton_1 = [[a + 1, b + 1] for a, b in DATA["skeleton"]]  # COCO is 1-indexed

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": names,
            "skeleton": skeleton_1,
        }],
    }

    ann_id = 1
    exported = 0
    for img_id, im in enumerate(DATA["images"], 1):
        if im.get("declined"):
            continue  # explicitly rejected as a bad frame
        if not im.get("reviewed"):
            continue  # only export frames you've actually checked
        src = FRAMES_DIR / im["file"]
        if not src.is_file():
            continue
        shutil.copy2(src, img_out / im["file"])

        kps = im["keypoints"]
        flat = []
        num = 0
        for x, y, v in kps:
            flat.extend([round(x, 1), round(y, 1), int(v)])
            if v > 0:
                num += 1

        coco["images"].append({
            "id": img_id,
            "file_name": im["file"],
            "width": im["width"],
            "height": im["height"],
        })
        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": flat,
            "num_keypoints": num,
            "bbox": _bbox(kps),
            "area": round(_bbox(kps)[2] * _bbox(kps)[3], 1),
            "iscrowd": 0,
        })
        ann_id += 1
        exported += 1

    (out_dir / "annotations.coco.json").write_text(json.dumps(coco, indent=1))
    return JSONResponse({
        "ok": True,
        "exported": exported,
        "path": str(out_dir.resolve()),
    })


@app.post("/api/ingest")
async def ingest(video: UploadFile = File(None), url: str = Form(""),
                 fps: float = Form(0.5)):
    """Extract frames from an uploaded video or a URL, auto-annotate the NEW
    frames, and append them to the project — preserving any already-reviewed
    work. Returns the new total frame count.
    """
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    before = {f.name for f in FRAMES_DIR.glob("frame_*.jpg")}
    start = len(before) + 1

    try:
        if video is not None and video.filename:
            suffix = Path(video.filename).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(video.file, tmp)
                tmp_path = Path(tmp.name)
            try:
                extract_one(tmp_path, FRAMES_DIR, fps, start)
            finally:
                tmp_path.unlink(missing_ok=True)
        elif url.strip():
            dl_dir = FRAMES_DIR.parent / "_downloads"
            dl_dir.mkdir(exist_ok=True)
            urls_file = dl_dir / "urls.txt"
            urls_file.write_text(url.strip())
            idx = start
            for v in download_urls(urls_file, dl_dir):
                idx = extract_one(v, FRAMES_DIR, fps, idx)
        else:
            raise HTTPException(400, "Provide a video file or a URL.")
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, f"Frame extraction failed: {exc}")
    except FileNotFoundError as exc:
        raise HTTPException(500, f"Missing tool (ffmpeg / yt-dlp?): {exc}")

    new_names = {f.name for f in FRAMES_DIR.glob("frame_*.jpg")} - before
    if not new_names:
        raise HTTPException(400, "No frames were extracted from that source.")

    added = annotate_dir(FRAMES_DIR, only=new_names)
    added.pop("_detected", None)
    DATA["images"].extend(added["images"])
    _save()
    return {"ok": True, "added": len(new_names), "count": len(DATA["images"])}


def main():
    global DATA_PATH, FRAMES_DIR
    ap = argparse.ArgumentParser(description="Run the pose annotation editor.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--project", help="project folder (frames + annotations live here; "
                                       "lets you ingest video from the browser)")
    src.add_argument("--data", help="open an existing annotations.json directly")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.project:
        proj = Path(args.project)
        proj.mkdir(parents=True, exist_ok=True)
        FRAMES_DIR = proj / "frames"
        DATA_PATH = proj / "annotations.json"
    else:
        DATA_PATH = Path(args.data)
        if not DATA_PATH.is_file():
            raise SystemExit(f"No such file: {DATA_PATH}")
        FRAMES_DIR = DATA_PATH.parent  # overridden by frames_dir in _load() if set

    _load()
    n = len(DATA["images"])
    print(f"{'Loaded ' + str(n) + ' frames' if n else 'Empty project — ingest from the browser'}")
    print(f"Open http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
