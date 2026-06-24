"""Step 3 — the annotation editor server.

  python server.py --data ./frames/annotations.json

Then open http://localhost:8000 in your browser. Drag the points to fix
them, click a point to cycle its visibility (visible -> underwater -> off),
and hit Export when you're done. Export writes a COCO-keypoints dataset to
./export/ that you can drag into Roboflow or train YOLO-Pose on directly.

Edits autosave to annotations.json as you go, so you can close the tab and
come back later without losing progress.
"""

import argparse
import json
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
app = FastAPI(title="CapAI Pose Annotator")

# Populated in main() before the server starts.
DATA_PATH: Path = None
FRAMES_DIR: Path = None
DATA: dict = None


def _load():
    global DATA, FRAMES_DIR
    DATA = json.loads(DATA_PATH.read_text())
    # Prefer the frames_dir baked into the file; fall back to the json's folder.
    fd = DATA.get("frames_dir")
    FRAMES_DIR = Path(fd) if fd and Path(fd).is_dir() else DATA_PATH.parent


def _save():
    DATA_PATH.write_text(json.dumps(DATA, indent=1))


class SaveBody(BaseModel):
    index: int
    keypoints: list[list[float]]
    reviewed: bool
    declined: bool = False


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


def main():
    global DATA_PATH
    ap = argparse.ArgumentParser(description="Run the pose annotation editor.")
    ap.add_argument("--data", required=True, help="annotations.json from auto_annotate.py")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    DATA_PATH = Path(args.data)
    if not DATA_PATH.is_file():
        raise SystemExit(f"No such file: {DATA_PATH}")
    _load()
    print(f"Loaded {len(DATA['images'])} frames from {DATA_PATH}")
    print(f"Open http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
