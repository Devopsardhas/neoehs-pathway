from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.camera_zone_store import CameraZoneStore
from src.config import AppConfig, CameraConfig
from src.detection_settings import DetectionSettings
from src.pipeline import PathwayMonitorPipeline

logger = logging.getLogger(__name__)

DEFAULT_DEMO_CAMERAS: list[tuple[str, str]] = [
    ("camera_1", "CCTV Camera 1"),
    ("camera_2", "CCTV Camera 2"),
    ("camera_3", "CCTV Camera 3"),
    ("camera_4", "CCTV Camera 4"),
]


@dataclass
class CameraZoneDemoStatus:
    state: str = "idle"
    message: str = "Upload a video, draw zones on a native video frame, then run detection."
    current_camera: str = ""
    current_camera_name: str = ""
    progress_pct: float = 0.0
    events_detected: int = 0
    last_run_at: str | None = None
    error: str | None = None
    results: list[dict] = field(default_factory=list)
    cameras: list[dict] = field(default_factory=list)


class CameraZoneDemoProcessor:
    """Image-based zone setup and full violation detection on uploaded videos."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.store = CameraZoneStore(config.project_root)
        self._lock = threading.Lock()
        self._status = CameraZoneDemoStatus()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._preview_jpeg: bytes = b""
        self._refresh_status()
        self._preview_jpeg = self._placeholder_jpeg(
            "Detection review",
            "Run detection to watch the video with pathway mapping.",
        )

    def demo_cameras(self) -> list[CameraConfig]:
        configured = {
            camera.id: camera
            for camera in self.config.cameras
            if camera.role == "monitor"
        }
        cameras: list[CameraConfig] = []
        for camera_id, name in DEFAULT_DEMO_CAMERAS:
            if camera_id in configured:
                cameras.append(configured[camera_id])
            else:
                cameras.append(
                    CameraConfig(
                        id=camera_id,
                        name=name,
                        rtsp_url=self.config.rtsp_url,
                        role="monitor",
                    )
                )
        return cameras

    def _refresh_status(self) -> None:
        self._status.cameras = [
            self.store.camera_status(camera.id) | {"name": camera.name}
            for camera in self.demo_cameras()
        ]

    def refresh(self) -> None:
        with self._lock:
            self._refresh_status()

    def status(self) -> dict:
        with self._lock:
            payload = asdict(self._status)
            payload["demo_cameras"] = [
                {"id": camera.id, "name": camera.name} for camera in self.demo_cameras()
            ]
            return payload

    def is_running(self) -> bool:
        with self._lock:
            return self._status.state == "running"

    @staticmethod
    def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
        max_width = 1280
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (max_width, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return encoded.tobytes() if ok else b""

    def _placeholder_jpeg(self, title: str, detail: str = "") -> bytes:
        image = np.zeros((420, 760, 3), dtype=np.uint8)
        image[:] = (28, 32, 40)
        cv2.putText(
            image,
            title,
            (36, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (210, 220, 230),
            2,
            cv2.LINE_AA,
        )
        if detail:
            cv2.putText(
                image,
                detail[:80],
                (36, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (150, 160, 175),
                1,
                cv2.LINE_AA,
            )
        return self._encode_jpeg(image, quality=75)

    def _annotate_review(self, frame: np.ndarray, label: str) -> np.ndarray:
        review = frame.copy()
        height, width = review.shape[:2]
        bar_h = max(28, height // 28)
        cv2.rectangle(review, (0, height - bar_h), (width, height), (32, 36, 44), -1)
        cv2.putText(
            review,
            f"{label}  |  Yellow = pathway  |  Magenta = mobile zone  |  Boxes = detections",
            (12, height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return review

    def _publish_preview(self, frame: np.ndarray, label: str) -> None:
        jpeg = self._encode_jpeg(self._annotate_review(frame, label))
        if not jpeg:
            return
        with self._lock:
            self._preview_jpeg = jpeg

    def get_preview_jpeg(self) -> bytes:
        with self._lock:
            return self._preview_jpeg or self._placeholder_jpeg(
                "Detection review",
                "Run detection to watch the video with pathway mapping.",
            )

    async def review_mjpeg_stream(self) -> Any:
        boundary = b"frame"
        while True:
            frame = self.get_preview_jpeg()
            yield (
                b"--" + boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            await asyncio.sleep(0.10)

    def save_frame(self, camera_id: str, filename: str, data: bytes) -> Path:
        self._validate_camera_id(camera_id)
        path = self.store.save_frame(camera_id, filename, data)
        self._refresh_status()
        return path

    def save_video(self, camera_id: str, filename: str, data: bytes) -> dict:
        self._validate_camera_id(camera_id)
        self.store.save_video(camera_id, filename, data)
        extracted = self.store.extract_video_frame(camera_id, time_sec=0.0)
        self._refresh_status()
        return extracted

    def capture_video_frame(self, camera_id: str, time_sec: float = 0.0) -> dict:
        self._validate_camera_id(camera_id)
        extracted = self.store.extract_video_frame(camera_id, time_sec=time_sec)
        self._refresh_status()
        return extracted

    def save_zone(self, camera_id: str, payload: dict) -> dict:
        self._validate_camera_id(camera_id)
        zone_id = str(payload.get("id", "")).strip()
        zone_name = str(payload.get("name", "")).strip()
        zone_type = str(payload.get("type", "pathway")).strip()
        points = payload.get("points") or []
        if not zone_id or not zone_name:
            raise ValueError("Zone id and name are required")
        if len(points) < 3:
            raise ValueError("At least 3 polygon points are required")

        normalized = [(int(point[0]), int(point[1])) for point in points]
        data = self.store.save_zone(camera_id, zone_id, zone_name, zone_type, normalized)
        self._refresh_status()
        return {
            "camera_id": camera_id,
            "name": zone_name,
            "output": str(self.store.zone_file(camera_id)),
            "zone_count": len(data.get("pathways", [])),
        }

    def clear_camera(self, camera_id: str) -> None:
        if self.is_running():
            raise RuntimeError("Cannot clear camera data while processing is running")
        self._validate_camera_id(camera_id)
        self.store.clear_camera(camera_id)
        self._refresh_status()

    def start(self, camera_ids: list[str] | None = None) -> None:
        with self._lock:
            if self._status.state == "running":
                raise RuntimeError("Camera demo processing is already running")

            cameras = self.demo_cameras()
            allowed_ids = {camera.id for camera in cameras}
            if camera_ids is None:
                selected = [
                    camera.id
                    for camera in cameras
                    if self.store.video_path(camera.id) is not None
                ]
            else:
                selected = [camera_id for camera_id in camera_ids if camera_id in allowed_ids]

            if not selected:
                raise RuntimeError("Upload at least one demo video before running detection")

            missing_video = [
                camera_id for camera_id in selected if self.store.video_path(camera_id) is None
            ]
            if missing_video:
                raise RuntimeError(
                    f"Missing demo video for: {', '.join(missing_video)}"
                )

            missing_frame = [
                camera_id
                for camera_id in selected
                if self.store.frame_path(camera_id) is None
            ]
            if missing_frame:
                raise RuntimeError(
                    "Capture a drawing frame from the uploaded video before running detection for: "
                    + ", ".join(missing_frame)
                )

            missing_zones = [
                camera_id
                for camera_id in selected
                if not self.store.list_zones(camera_id)
            ]
            if missing_zones:
                raise RuntimeError(
                    "Define pathway or mobile-free zones before running detection for: "
                    + ", ".join(missing_zones)
                )

            self._stop.clear()
            self._preview_jpeg = self._placeholder_jpeg(
                "Starting detection review…",
                "Pathway overlay will appear on the first processed frame.",
            )
            self._status = CameraZoneDemoStatus(
                state="running",
                message="Starting camera demo detection…",
                cameras=self._status.cameras,
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(selected,),
                name="camera-zone-demo-processor",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _settings_for_demo(self) -> DetectionSettings:
        settings = DetectionSettings.from_config(self.config)
        settings.pathway_block = True
        settings.mobile_usage = True
        settings.ppe_violation = True
        settings.fire_detection = True
        settings.smoke_detection = True
        settings.object_fall = True
        settings.near_miss = True
        settings.oil_spillage = True
        settings.wet_floor = True
        settings.low_visibility = True
        return settings

    def _run(self, camera_ids: list[str]) -> None:
        results: list[dict] = []

        try:
            demo_settings = self._settings_for_demo()
            demo_settings.apply_to(self.config)

            for index, camera_id in enumerate(camera_ids):
                if self._stop.is_set():
                    break

                camera = next(
                    (item for item in self.demo_cameras() if item.id == camera_id),
                    None,
                )
                label = camera.name if camera else camera_id
                video_path = self.store.video_path(camera_id)
                if video_path is None:
                    continue

                zone_scale = self.store.zone_scale_for_video(camera_id, video_path)
                zone_file = self.store.zone_file(camera_id)

                with self._lock:
                    self._status.current_camera = camera_id
                    self._status.current_camera_name = label
                    self._status.message = f"Reviewing {label} with pathway overlay…"
                    self._status.progress_pct = (index / len(camera_ids)) * 100.0
                    self._preview_jpeg = self._placeholder_jpeg(
                        f"Loading {label}…",
                        "Pathway overlay starts with the first processed frame.",
                    )

                def on_observation(observation, cam_id=camera_id, cam_label=label) -> None:
                    results.append(
                        {
                            "camera_id": cam_id,
                            "camera_name": cam_label,
                            "violation_type": observation.violation_type,
                            "object_class": observation.object_class,
                            "severity": observation.severity,
                            "summary": observation.summary,
                            "observation_id": observation.observation_id,
                        }
                    )
                    with self._lock:
                        self._status.events_detected += 1
                        self._status.results = results[-100:]

                def on_progress(ratio: float, idx=index, total=len(camera_ids)) -> None:
                    with self._lock:
                        base = idx / total
                        self._status.progress_pct = min(
                            99.0,
                            (base + ratio / total) * 100.0,
                        )

                def on_frame(preview: np.ndarray, cam_label=label) -> None:
                    self._publish_preview(preview, cam_label)

                pipeline = PathwayMonitorPipeline(
                    self.config,
                    persist_observations=True,
                    video_path=str(video_path),
                    lock_settings=True,
                    pathway_config_override=str(zone_file),
                    zone_scale=zone_scale,
                    camera_id_override=camera_id,
                )
                pipeline.run(
                    on_observation=on_observation,
                    progress_callback=on_progress,
                    frame_callback=on_frame,
                    should_stop=self._stop.is_set,
                )
                logger.info(
                    "Camera demo %s finished with scale %s",
                    camera_id,
                    zone_scale,
                )

            with self._lock:
                if self._stop.is_set():
                    self._status.state = "idle"
                    self._status.message = "Camera demo processing was stopped."
                else:
                    self._status.state = "completed"
                    self._status.progress_pct = 100.0
                    self._status.message = (
                        f"Camera demo complete — {len(results)} observation(s) recorded."
                    )
                    self._status.last_run_at = datetime.now(timezone.utc).isoformat()
                    self._status.results = results[-100:]
                self._status.current_camera = ""
                self._status.current_camera_name = ""
                self._refresh_status()

        except Exception as exc:
            logger.exception("Camera zone demo processing failed")
            with self._lock:
                self._status.state = "error"
                self._status.error = str(exc)
                self._status.message = "Camera demo processing failed."
                self._preview_jpeg = self._placeholder_jpeg(
                    "Detection failed",
                    str(exc)[:80],
                )
                self._refresh_status()

    def _validate_camera_id(self, camera_id: str) -> None:
        allowed = {camera.id for camera in self.demo_cameras()}
        if camera_id not in allowed:
            raise ValueError(f"Unknown demo camera: {camera_id}")
