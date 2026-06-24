"""Step 1 — pull frames out of videos.

Two ways to feed it:

  # A) from local video files in a folder
  python extract_frames.py --videos ./videos --out ./frames --fps 0.5

  # B) from a list of YouTube URLs (one per line in urls.txt)
  python extract_frames.py --urls urls.txt --out ./frames --fps 0.5

--fps 0.5 means "one frame every 2 seconds". Bump it up for fast action,
down for long matches. Frames land in --out as frame_000001.jpg, numbered
continuously across every video so nothing overwrites.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


def _have(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None


def extract_one(video: Path, out_dir: Path, fps: float, start_index: int) -> int:
    """Extract frames from a single video; return the next free frame index."""
    # ffmpeg can't continue a global counter across calls, so write each video
    # to a temp pattern then rename into the shared, continuously-numbered set.
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "f_%06d.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",                  # high JPEG quality
            pattern,
        ]
        subprocess.run(cmd, check=True)
        produced = sorted(Path(tmp).glob("f_*.jpg"))
        idx = start_index
        for f in produced:
            dest = out_dir / f"frame_{idx:06d}.jpg"
            f.replace(dest)
            idx += 1
    print(f"  {video.name}: {idx - start_index} frames")
    return idx


def download_urls(urls_file: Path, dest: Path) -> list[Path]:
    """Download each YouTube URL to dest/ with yt-dlp; return the saved files."""
    if not _have("yt-dlp"):
        sys.exit("yt-dlp not found. pip install yt-dlp (or use --videos instead).")
    urls = [u.strip() for u in urls_file.read_text().splitlines() if u.strip()]
    saved = []
    for i, url in enumerate(urls):
        out_tmpl = str(dest / f"dl_{i:03d}.%(ext)s")
        print(f"Downloading {url}")
        subprocess.run(
            ["yt-dlp", "-f", "mp4/best", "-o", out_tmpl, url], check=True
        )
        saved.extend(sorted(dest.glob(f"dl_{i:03d}.*")))
    return saved


def main():
    ap = argparse.ArgumentParser(description="Extract frames from videos.")
    ap.add_argument("--videos", help="folder of local video files")
    ap.add_argument("--urls", help="text file of YouTube URLs, one per line")
    ap.add_argument("--out", required=True, help="output folder for frames")
    ap.add_argument("--fps", type=float, default=0.5,
                    help="frames per second to sample (default 0.5 = every 2s)")
    args = ap.parse_args()

    if not _have("ffmpeg"):
        sys.exit("ffmpeg not found. Install it (mac: brew install ffmpeg).")
    if not args.videos and not args.urls:
        sys.exit("Give me --videos <folder> or --urls <file>.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos: list[Path] = []
    if args.urls:
        dl_dir = out_dir / "_downloads"
        dl_dir.mkdir(exist_ok=True)
        videos.extend(download_urls(Path(args.urls), dl_dir))
    if args.videos:
        for f in sorted(Path(args.videos).iterdir()):
            if f.suffix.lower() in VIDEO_EXT:
                videos.append(f)

    if not videos:
        sys.exit("No videos found to process.")

    print(f"Processing {len(videos)} video(s) at {args.fps} fps -> {out_dir}")
    idx = 1
    for v in videos:
        idx = extract_one(v, out_dir, args.fps, idx)
    print(f"\nDone. {idx - 1} frames in {out_dir}")
    print("Next: python auto_annotate.py --frames", out_dir)


if __name__ == "__main__":
    main()
