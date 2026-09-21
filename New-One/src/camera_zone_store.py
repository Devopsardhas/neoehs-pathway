from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.pathway import ZoneType

ALLOWED_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class CameraZoneStore:
    """Per-camera reference frames and pathway / mobile-free zone definitions."""

    def __init__(self, project_root: Path):
        self.root = (project_root / "data" / "camera_zones").resolve()
        self.frames_dir = self.root / "frames"
        self.videos_dir = self.root / "videos"
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def zone_file(self, camera_id: str) -> Path:
        return self.root / f"{camera_id}.json"

    def frame_path(self, camera_id: str) -> Path | None:
        for ext in ALLOWED_FRAME_EXTENSIONS:
            path = self.frames_dir / f"{camera_id}{ext}"
            if path.is_file():
                return path
        return None

    def video_path(self, camera_id: str) -> Path | None:
        for ext in ALLOWED_VIDEO_EXTENSIONS:
            path = self.videos_dir / f"{camera_id}{ext}"
            if path.is_file():
                return path
        return None

    def load_data(self, camera_id: str) -> dict:
        zone_file = self.zone_file(camera_id)
        if not zone_file.exists():
            return {"camera_id": camera_id, "pathways": []}
        with zone_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_data(self, camera_id: str, data: dict) -> None:
        data["camera_id"] = camera_id
        zone_file = self.zone_file(camera_id)
        zone_file.parent.mkdir(parents=True, exist_ok=True)
        with zone_file.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def list_zones(self, camera_id: str) -> list[dict]:
        return list(self.load_data(camera_id).get("pathways", []))

    def save_frame(self, camera_id: str, filename: str, data: bytes) -> Path:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_FRAME_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{suffix}'. "
                f"Use: {', '.join(sorted(ALLOWED_FRAME_EXTENSIONS))}"
            )

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode uploaded reference image")

        dest, _zones_cleared = self.save_bgr_frame(camera_id, frame, source="image")
        return dest

    def save_video(self, camera_id: str, filename: str, data: bytes) -> Path:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{suffix}'. "
                f"Use: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
            )

        for existing in self.videos_dir.glob(f"{camera_id}.*"):
            if existing.is_file():
                existing.unlink()

        dest = self.videos_dir / f"{camera_id}{suffix}"
        dest.write_bytes(data)

        zone_data = self.load_data(camera_id)
        zone_data["demo_video"] = dest.name
        size = self.video_size(dest)
        if size is not None:
            zone_data["video_width"] = size[0]
            zone_data["video_height"] = size[1]
        self.save_data(camera_id, zone_data)
        return dest

    def save_bgr_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        source: str = "image",
    ) -> tuple[Path, bool]:
        if frame is None or frame.size == 0:
            raise ValueError("Could not decode camera frame")

        old_size = self.reference_size(camera_id)
        zone_data = self.load_data(camera_id)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        new_size = (width, height)
        zones_cleared = False
        if old_size is not None and old_size != new_size and zone_data.get("pathways"):
            zone_data["pathways"] = []
            zones_cleared = True

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise ValueError("Could not encode camera frame")

        for existing in self.frames_dir.glob(f"{camera_id}.*"):
            if existing.is_file():
                existing.unlink()

        dest = self.frames_dir / f"{camera_id}.jpg"
        dest.write_bytes(encoded.tobytes())

        zone_data["reference_frame"] = dest.name
        zone_data["frame_width"] = width
        zone_data["frame_height"] = height
        zone_data["frame_source"] = source
        self.save_data(camera_id, zone_data)
        return dest, zones_cleared

    def extract_video_frame(self, camera_id: str, time_sec: float = 0.0) -> dict:
        video_path = self.video_path(camera_id)
        if video_path is None:
            raise ValueError("Upload a video before capturing a drawing frame")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video")

        frame = None
        try:
            if time_sec > 0:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)

            for _ in range(45):
                ok, candidate = capture.read()
                if not ok or candidate is None or candidate.size == 0:
                    continue
                if float(np.std(candidate)) < 2.0:
                    continue
                frame = candidate
                break

            if frame is None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    raise ValueError("Could not read a frame from the uploaded video")
        finally:
            capture.release()

        if frame is None:
            raise ValueError("Could not read a frame from the uploaded video")

        dest, zones_cleared = self.save_bgr_frame(camera_id, frame, source="video")
        width, height = int(frame.shape[1]), int(frame.shape[0])
        zone_data = self.load_data(camera_id)
        zone_data["video_width"] = width
        zone_data["video_height"] = height
        zone_data["frame_time_sec"] = max(0.0, float(time_sec))
        self.save_data(camera_id, zone_data)
        return {
            "path": str(dest),
            "width": width,
            "height": height,
            "time_sec": max(0.0, float(time_sec)),
            "zones_cleared": zones_cleared,
            "from_video": True,
        }

    def save_zone(
        self,
        camera_id: str,
        zone_id: str,
        zone_name: str,
        zone_type: ZoneType | str,
        points: list[tuple[int, int] | list[int]],
    ) -> dict:
        if isinstance(zone_type, str):
            zone_type = ZoneType(zone_type)

        data = self.load_data(camera_id)
        frame_path = self.frame_path(camera_id)
        if frame_path is not None:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                data["frame_width"] = int(frame.shape[1])
                data["frame_height"] = int(frame.shape[0])
                data["reference_frame"] = frame_path.name

        normalized_points = [[int(x), int(y)] for x, y in points]
        pathways = [item for item in data.get("pathways", []) if item.get("id") != zone_id]
        pathways.append(
            {
                "id": zone_id,
                "name": zone_name,
                "type": zone_type.value,
                "points": normalized_points,
            }
        )
        data["pathways"] = pathways
        self.save_data(camera_id, data)
        return data

    def load_reference_frame(self, camera_id: str) -> np.ndarray | None:
        frame_path = self.frame_path(camera_id)
        if frame_path is None:
            return None
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None
        return frame

    def reference_size(self, camera_id: str) -> tuple[int, int] | None:
        data = self.load_data(camera_id)
        width = data.get("frame_width")
        height = data.get("frame_height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height

        frame_path = self.frame_path(camera_id)
        if frame_path is None:
            return None

        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None
        return int(frame.shape[1]), int(frame.shape[0])

    def video_size(self, video_path: Path) -> tuple[int, int] | None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return None
        ok, frame = capture.read()
        prop_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        prop_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        if ok and frame is not None and frame.size > 0:
            return int(frame.shape[1]), int(frame.shape[0])
        if prop_w > 0 and prop_h > 0:
            return prop_w, prop_h
        return None

    def zone_scale_for_video(self, camera_id: str, video_path: Path) -> tuple[float, float]:
        ref_size = self.reference_size(camera_id)
        video_size = self.video_size(video_path)
        if ref_size is None or video_size is None:
            return 1.0, 1.0

        ref_w, ref_h = ref_size
        vid_w, vid_h = video_size
        if ref_w == vid_w and ref_h == vid_h:
            return 1.0, 1.0
        return vid_w / ref_w, vid_h / ref_h

    def camera_status(self, camera_id: str) -> dict:
        frame = self.frame_path(camera_id)
        video = self.video_path(camera_id)
        zones = self.list_zones(camera_id)
        ref_size = self.reference_size(camera_id)
        data = self.load_data(camera_id)
        video_size = None
        if video is not None:
            stored_w = data.get("video_width")
            stored_h = data.get("video_height")
            if isinstance(stored_w, int) and isinstance(stored_h, int) and stored_w > 0 and stored_h > 0:
                video_size = (stored_w, stored_h)
            else:
                video_size = self.video_size(video)
        frame_matches_video = bool(
            ref_size
            and video_size
            and ref_size[0] == video_size[0]
            and ref_size[1] == video_size[1]
        )
        return {
            "camera_id": camera_id,
            "has_frame": frame is not None,
            "frame_name": frame.name if frame else None,
            "frame_width": ref_size[0] if ref_size else None,
            "frame_height": ref_size[1] if ref_size else None,
            "frame_source": data.get("frame_source") or ("image" if frame else None),
            "frame_time_sec": data.get("frame_time_sec"),
            "has_video": video is not None,
            "video_name": video.name if video else None,
            "video_width": video_size[0] if video_size else None,
            "video_height": video_size[1] if video_size else None,
            "frame_matches_video": frame_matches_video,
            "zone_count": len(zones),
            "pathway_zones": sum(1 for zone in zones if zone.get("type") == ZoneType.PATHWAY.value),
            "mobile_zones": sum(
                1 for zone in zones if zone.get("type") == ZoneType.MOBILE_USAGE.value
            ),
            "zones": zones,
        }

    def clear_camera(self, camera_id: str) -> None:
        for existing in self.frames_dir.glob(f"{camera_id}.*"):
            if existing.is_file():
                existing.unlink()
        for existing in self.videos_dir.glob(f"{camera_id}.*"):
            if existing.is_file():
                existing.unlink()
        zone_file = self.zone_file(camera_id)
        if zone_file.is_file():
            zone_file.unlink()
