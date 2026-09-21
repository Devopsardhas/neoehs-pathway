#!/usr/bin/env python3
"""Prepare front/back gate demo videos for the web upload demo.

Uses a bundled or downloaded traffic clip when available, otherwise generates
simple synthetic gate footage locally (no network required).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo_videos"
FRONT_OUT = DEMO_DIR / "front_gate_demo.mp4"
BACK_OUT = DEMO_DIR / "back_gate_demo.mp4"
SOURCE_CANDIDATES = [
    DEMO_DIR / "_source_traffic.mp4",
    DEMO_DIR / "_back_raw.mp4",
    DEMO_DIR / "Traffic IP Camera video.mp4",
]


def _open_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {path}")
    return writer


def _overlay_gate_ui(frame: np.ndarray, title: str, direction: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.putText(
        out,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    badge = "ENTRY" if direction == "in" else "EXIT"
    color = (40, 180, 80) if direction == "in" else (40, 120, 255)
    cv2.rectangle(out, (w - 120, 8), (w - 8, 32), color, -1)
    cv2.putText(
        out,
        badge,
        (w - 108, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(out, (w // 2 - 80, h - 28), (w // 2 + 80, h - 8), (180, 180, 180), 2)
    cv2.putText(
        out,
        "GATE DEMO",
        (w // 2 - 58, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def _split_source(source: Path, front_out: Path, back_out: Path, max_seconds: float = 20.0) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(max_seconds * fps)

    front_writer = _open_writer(front_out, width, height, fps)
    back_writer = _open_writer(back_out, width, height, fps)

    frames: list[np.ndarray] = []
    while len(frames) < max_frames * 2:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 30:
        front_writer.release()
        back_writer.release()
        raise RuntimeError(f"Source video too short: {source}")

    mid = len(frames) // 2
    front_frames = frames[:mid]
    back_frames = list(reversed(frames[mid:]))

    for frame in front_frames:
        front_writer.write(_overlay_gate_ui(frame, "Front Gate Camera", "in"))
    for frame in back_frames:
        back_writer.write(_overlay_gate_ui(frame, "Back Gate Camera", "out"))

    front_writer.release()
    back_writer.release()
    print(f"Created {front_out.name} ({len(front_frames)} frames)")
    print(f"Created {back_out.name} ({len(back_frames)} frames, reversed segment)")


def _draw_car(frame: np.ndarray, x: int, y: int, scale: float, plate_text: str) -> None:
    w = int(220 * scale)
    h = int(90 * scale)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 85, 180), -1)
    cv2.rectangle(frame, (x + 20, y - 35), (x + w - 20, y), (70, 110, 210), -1)
    cv2.rectangle(frame, (x + 30, y - 28), (x + 75, y - 8), (180, 220, 255), -1)
    cv2.rectangle(frame, (x + w - 75, y - 28), (x + w - 30, y - 8), (180, 220, 255), -1)
    cv2.circle(frame, (x + 45, y + h), int(22 * scale), (25, 25, 25), -1)
    cv2.circle(frame, (x + w - 45, y + h), int(22 * scale), (25, 25, 25), -1)
    plate_w = int(120 * scale)
    plate_h = int(28 * scale)
    px = x + (w - plate_w) // 2
    py = y + h - plate_h - 8
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (240, 240, 240), -1)
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (20, 20, 20), 2)
    cv2.putText(
        frame,
        plate_text,
        (px + 8, py + plate_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * scale,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )


def _generate_synthetic(out_path: Path, direction: str, plate_text: str, seconds: float = 12.0) -> None:
    width, height, fps = 1280, 720, 25
    total_frames = int(seconds * fps)
    writer = _open_writer(out_path, width, height, fps)
    title = "Front Gate Camera" if direction == "in" else "Back Gate Camera"

    for index in range(total_frames):
        frame = np.full((height, width, 3), 45, dtype=np.uint8)
        cv2.rectangle(frame, (0, height // 2 + 40), (width, height), (28, 28, 28), -1)
        cv2.line(frame, (0, height // 2 + 40), (width, height // 2 + 40), (220, 220, 100), 4)
        for lane_x in range(120, width, 180):
            cv2.line(
                frame,
                (lane_x, height // 2 + 120),
                (lane_x, height - 40),
                (90, 90, 90),
                2,
            )

        progress = index / max(total_frames - 1, 1)
        if direction == "in":
            car_x = int(-260 + progress * (width + 320))
        else:
            car_x = int(width + 260 - progress * (width + 320))
        car_y = height // 2 + 70
        _draw_car(frame, car_x, car_y, 1.0, plate_text)
        frame = _overlay_gate_ui(frame, title, direction)
        writer.write(frame)

    writer.release()
    print(f"Created synthetic {out_path.name}")


def find_source() -> Path | None:
    for candidate in SOURCE_CANDIDATES:
        if candidate.is_file() and candidate.stat().st_size > 100_000:
            return candidate
    return None


def prepare(force: bool = False) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    if FRONT_OUT.exists() and BACK_OUT.exists() and not force:
        print(f"Demo videos already exist:\n  {FRONT_OUT}\n  {BACK_OUT}")
        return

    source = find_source()
    if source is not None:
        print(f"Using source clip: {source.name}")
        _split_source(source, FRONT_OUT, BACK_OUT)
        return

    print("No source traffic clip found — generating synthetic demo videos.")
    _generate_synthetic(FRONT_OUT, "in", "TN01AB1234")
    _generate_synthetic(BACK_OUT, "out", "TN01AB1234")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gate demo sample videos")
    parser.add_argument("--force", action="store_true", help="Regenerate even if outputs exist")
    args = parser.parse_args()
    try:
        prepare(force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
