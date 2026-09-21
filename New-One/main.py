#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.pipeline import PathwayMonitorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCTV pathway obstruction monitor")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--rtsp",
        default=None,
        help="Override RTSP URL from config",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview window (press q to quit)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = AppConfig.load(args.config)
    if args.rtsp:
        config.rtsp_url = args.rtsp

    pipeline = PathwayMonitorPipeline(config, show_preview=args.preview)
    pipeline.run()


if __name__ == "__main__":
    main()
