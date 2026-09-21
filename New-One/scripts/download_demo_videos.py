#!/usr/bin/env python3
"""Optional: download higher-quality ANPR sample clips for gate demo.

Run this only if you want real traffic footage instead of the bundled/synthetic
clips created by prepare_demo_videos.py.

  python scripts/download_demo_videos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo_videos"
SOURCE_OUT = DEMO_DIR / "_source_traffic.mp4"

# Public ANPR tutorial sample (YOLOv8 ANPR projects)
GDRIVE_ID = "1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta"


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("Install gdown first: pip install gdown")
        sys.exit(1)

    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    print(f"Downloading source traffic clip to {SOURCE_OUT} ...")
    gdown.download(url, str(SOURCE_OUT), quiet=False)
    if not SOURCE_OUT.is_file():
        print("Download failed.")
        sys.exit(1)

    print("Download complete. Now run:")
    print("  python scripts/prepare_demo_videos.py --force")


if __name__ == "__main__":
    main()
