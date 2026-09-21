#!/usr/bin/env python3
"""
Interactive tool to draw pathway and mobile-usage zone polygons on a CCTV frame.

Usage:
  python tools/define_pathway.py --source rtsp://user:pass@ip/stream --type pathway
  python tools/define_pathway.py --source sample.jpg --type mobile_usage
  python tools/define_pathway.py --source rtsp://... --type pathway --web
  python tools/define_pathway.py --source sample.jpg --points "120,80 640,80 640,360 120,360"
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pathway import ZONE_COLORS, ZoneType
from src.preview_gui import destroy_all, gui_available, imshow, wait_key
from src.rtsp_capture import RTSPCapture

DRAWER_HTML = Path(__file__).with_name("pathway_drawer.html")
WEB_WORK_DIR = ROOT / "data" / "pathway_drawer"


class PathwayDrawer:
    def __init__(self, frame: np.ndarray, output_file: Path, zone_type: ZoneType):
        self.frame = frame.copy()
        self.output_file = output_file
        self.zone_type = zone_type
        self.color = ZONE_COLORS[zone_type]
        self.points: list[tuple[int, int]] = []
        self.window = f"Define {zone_type.value} zone - click points, Enter=save, r=reset, q=quit"

    def _mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

    def run(self) -> None:
        if not gui_available():
            raise RuntimeError(_gui_unavailable_message())

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._mouse_callback)

        while True:
            canvas = self._render_canvas()

            if not imshow(self.window, canvas):
                raise RuntimeError(_gui_unavailable_message())

            key = wait_key(20)
            if key == ord("q"):
                break
            if key == ord("r"):
                self.points.clear()
            if key in (13, 10) and len(self.points) >= 3:
                self._save()
                break

        destroy_all()

    def _render_canvas(self) -> np.ndarray:
        canvas = self.frame.copy()
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=self.color, thickness=2)
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color=self.color)
            canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
            cv2.polylines(canvas, [pts], isClosed=True, color=self.color, thickness=2)

        for idx, (x, y) in enumerate(self.points):
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(idx + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        hint = f"Zone type: {self.zone_type.value} | Enter=save | r=reset | q=quit"
        cv2.putText(canvas, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return canvas

    def _save(self) -> None:
        default_id = "main_pathway" if self.zone_type == ZoneType.PATHWAY else "mobile_zone_1"
        default_name = (
            "Main_pathway"
            if self.zone_type == ZoneType.PATHWAY
            else "Mobile allowed area"
        )

        zone_id = input(f"Zone ID [{default_id}]: ").strip() or default_id
        zone_name = input(f"Zone name [{default_name}]: ").strip() or default_name
        save_zone(
            self.output_file,
            zone_id=zone_id,
            zone_name=zone_name,
            zone_type=self.zone_type,
            points=self.points,
        )


def _gui_unavailable_message() -> str:
    return (
        "OpenCV GUI is unavailable on this system (common with opencv-python-headless).\n"
        "Options:\n"
        "  1) Web drawer:  python tools/define_pathway.py --source <url> --type pathway --web\n"
        "  2) CLI points:  python tools/define_pathway.py --source <url> --points \"x1,y1 x2,y2 x3,y3\"\n"
        "  3) Install GUI:   pip uninstall opencv-python-headless -y && pip install opencv-python"
    )


def save_zone(
    output_file: Path,
    zone_id: str,
    zone_name: str,
    zone_type: ZoneType,
    points: list[tuple[int, int]],
) -> None:
    data = {"pathways": []}
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

    data["pathways"] = [p for p in data.get("pathways", []) if p.get("id") != zone_id]
    data["pathways"].append(
        {
            "id": zone_id,
            "name": zone_name,
            "type": zone_type.value,
            "points": [[int(x), int(y)] for x, y in points],
        }
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {zone_type.value} zone '{zone_name}' to {output_file}")


def parse_points(raw: str) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for chunk in raw.split():
        if not chunk.strip():
            continue
        x_str, y_str = chunk.split(",", 1)
        points.append((int(x_str), int(y_str)))
    if len(points) < 3:
        raise ValueError("At least 3 points are required")
    return points


def load_frame(source: str) -> np.ndarray:
    if source.lower().startswith("rtsp://"):
        capture = RTSPCapture(source)
        frame = capture.read_one()
        capture.release()
        if frame is None:
            raise RuntimeError(f"Could not read frame from {source}")
        return frame

    frame = cv2.imread(source)
    if frame is None:
        raise RuntimeError(f"Could not read image {source}")
    return frame


def run_web_drawer(
    frame: np.ndarray,
    output_file: Path,
    zone_type: ZoneType,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    if not DRAWER_HTML.is_file():
        raise RuntimeError(f"Missing web drawer template: {DRAWER_HTML}")

    WEB_WORK_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = WEB_WORK_DIR / "frame.jpg"
    html_path = WEB_WORK_DIR / "pathway_drawer.html"
    cv2.imwrite(str(frame_path), frame)
    html_path.write_text(DRAWER_HTML.read_text(encoding="utf-8"), encoding="utf-8")

    handler_cls = partial(_DrawerHandler, output_file=output_file, work_dir=WEB_WORK_DIR)
    server = ThreadingHTTPServer((host, port), handler_cls)
    url = (
        f"http://{host}:{port}/pathway_drawer.html"
        f"?type={quote(zone_type.value)}&frame={quote('frame.jpg')}"
    )

    print(f"Pathway web drawer running at {url}")
    print("Click points in the browser, then press Save Zone.")
    print("Press Ctrl+C here when finished.")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped pathway web drawer.")
    finally:
        server.server_close()


class _DrawerHandler(BaseHTTPRequestHandler):
    output_file: Path
    work_dir: Path

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        rel_path = self.path.split("?", 1)[0].lstrip("/") or "pathway_drawer.html"
        file_path = (self.work_dir / rel_path).resolve()
        if not str(file_path).startswith(str(self.work_dir.resolve())):
            self.send_error(403)
            return
        if not file_path.is_file():
            self.send_error(404)
            return

        content_type = "text/html"
        if file_path.suffix.lower() in {".jpg", ".jpeg"}:
            content_type = "image/jpeg"
        elif file_path.suffix.lower() == ".png":
            content_type = "image/png"

        payload = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/save":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
            zone_id = str(payload["id"]).strip()
            zone_name = str(payload["name"]).strip()
            zone_type = ZoneType(str(payload["type"]))
            points = [(int(x), int(y)) for x, y in payload["points"]]
            if len(points) < 3:
                raise ValueError("At least 3 points are required")
            save_zone(
                self.output_file,
                zone_id=zone_id,
                zone_name=zone_name,
                zone_type=zone_type,
                points=points,
            )
        except Exception as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            message = json.dumps({"detail": str(exc)}).encode("utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
            return

        response = json.dumps(
            {"ok": True, "name": zone_name, "output": str(self.output_file)}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw pathway or mobile-usage zone polygons")
    parser.add_argument("--source", required=True, help="RTSP URL or image path")
    parser.add_argument(
        "--type",
        choices=[ZoneType.PATHWAY.value, ZoneType.MOBILE_USAGE.value],
        default=ZoneType.PATHWAY.value,
        help="Zone type: pathway (blocking) or mobile_usage (no phone zone)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "pathways.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Open browser-based drawer (works without OpenCV GUI)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --web mode (default: 8765)",
    )
    parser.add_argument(
        "--points",
        help='Headless save, e.g. --points "120,80 640,80 640,360 120,360"',
    )
    parser.add_argument("--zone-id", help="Zone ID for --points mode")
    parser.add_argument("--zone-name", help="Zone name for --points mode")
    args = parser.parse_args()

    frame = load_frame(args.source)
    output_file = Path(args.output)
    zone_type = ZoneType(args.type)

    if args.points:
        points = parse_points(args.points)
        zone_id = args.zone_id or (
            "main_pathway" if zone_type == ZoneType.PATHWAY else "mobile_zone_1"
        )
        zone_name = args.zone_name or (
            "Main_pathway" if zone_type == ZoneType.PATHWAY else "Mobile allowed area"
        )
        save_zone(output_file, zone_id, zone_name, zone_type, points)
        return

    if args.web or not gui_available():
        if not args.web and not gui_available():
            print(_gui_unavailable_message())
            print("\nStarting web drawer instead...\n")
        run_web_drawer(frame, output_file, zone_type, port=args.port)
        return

    drawer = PathwayDrawer(frame, output_file, zone_type)
    drawer.run()


if __name__ == "__main__":
    main()
