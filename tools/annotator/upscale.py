"""Optional super-resolution for low-quality frames.

MediaPipe's pose detector does much better when the athlete occupies more
pixels. On compressed or low-res pool footage — especially wide shots where a
player is small and blurry — upscaling the frame before detection recovers
keypoints that would otherwise be missed entirely.

We use OpenCV's FSRCNN super-resolution model: tiny (~40 KB), fast on CPU, and
it ships with the `cv2.dnn_superres` module in opencv-contrib-python (already
pulled in by mediapipe). The model file downloads once and is cached in
`.sr_models/` next to this script. If it can't be fetched (offline, or OpenCV
was built without the contrib module) we fall back to plain Lanczos
interpolation, which still gives the detector more pixels to chew on.

Only frames whose shorter side is below `min_side` get upscaled — sharp,
high-res frames are left untouched so we don't waste time or soften them.

Important: this only ever touches the pixels handed to the detector. MediaPipe
returns normalized (0..1) coordinates, so the keypoints map straight back onto
the original frame — callers keep using the original width/height and need to
do nothing special.
"""

import urllib.request
from pathlib import Path

import cv2

HERE = Path(__file__).parent
MODEL_DIR = HERE / ".sr_models"
# FSRCNN weights, the reference source OpenCV's own docs point at.
FSRCNN_URL = ("https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/"
              "models/FSRCNN_x{scale}.pb")


class Upscaler:
    """Callable that super-resolves an image if it's below the size threshold.

    Load once, call per frame: `up = Upscaler(); rgb = up(rgb)`.
    """

    def __init__(self, scale: int = 2, min_side: int = 640):
        self.scale = scale
        self.min_side = min_side
        self._sr = self._load_model(scale)
        self.backend = "fsrcnn" if self._sr is not None else "lanczos"

    def _load_model(self, scale: int):
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
        except AttributeError:
            # OpenCV built without the contrib dnn_superres module.
            return None
        path = MODEL_DIR / f"FSRCNN_x{scale}.pb"
        if not path.is_file():
            # Download to a temp file and rename on success so an interrupted
            # download can't leave a partial .pb that is_file() then treats as
            # the cached model forever (silently degrading to Lanczos).
            tmp = path.with_name(path.name + ".tmp")
            try:
                MODEL_DIR.mkdir(exist_ok=True)
                urllib.request.urlretrieve(FSRCNN_URL.format(scale=scale), tmp)
                tmp.replace(path)
            except Exception:
                tmp.unlink(missing_ok=True)
                return None  # offline / blocked — caller falls back to Lanczos
        try:
            sr.readModel(str(path))
            sr.setModel("fsrcnn", scale)
            return sr
        except Exception:
            return None

    def __call__(self, img):
        h, w = img.shape[:2]
        if min(h, w) >= self.min_side:
            return img  # already big enough — leave it alone
        if self._sr is not None:
            try:
                return self._sr.upsample(img)
            except Exception:
                pass  # fall through to interpolation on any runtime hiccup
        return cv2.resize(img, (w * self.scale, h * self.scale),
                          interpolation=cv2.INTER_LANCZOS4)
