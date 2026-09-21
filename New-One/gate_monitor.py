#!/usr/bin/env python3
"""Run front/back gate ANPR monitors for vehicle in/out tracking."""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.gate_pipeline import GatePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate ANPR vehicle monitor")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--preview", action="store_true", help="Show OpenCV preview windows")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = AppConfig.load(args.config)
    if not config.gate_vehicle_enabled:
        logging.error("gate_vehicle.enabled is false in config.yaml")
        sys.exit(1)

    gate_cameras = config.gate_cameras()
    if not gate_cameras:
        logging.error("No cameras with role gate_in or gate_out found in config.yaml")
        sys.exit(1)

    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    for camera in gate_cameras:
        pipeline = GatePipeline(config, camera)

        def _run(pipe: GatePipeline = pipeline) -> None:
            pipe.run(show_preview=args.preview, should_stop=stop_event.is_set)

        thread = threading.Thread(
            target=_run,
            name=f"gate-{camera.id}",
            daemon=True,
        )
        threads.append(thread)
        thread.start()
        logging.info("Started gate monitor for %s (%s)", camera.name, camera.role)

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logging.info("Stopping gate monitors...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
