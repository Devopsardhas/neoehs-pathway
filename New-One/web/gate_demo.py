from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import AppConfig, CameraConfig
from src.gate_pipeline import GatePipeline
from src.vehicle_manager import VehicleManager

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ROLE_FILES = {
    "gate_in": "front_gate_demo",
    "gate_out": "back_gate_demo",
}


@dataclass
class DemoStatus:
    state: str = "idle"
    message: str = "Upload sample videos and run demo detection."
    current_step: str = ""
    progress_pct: float = 0.0
    events_detected: int = 0
    front_video: str | None = None
    back_video: str | None = None
    last_run_at: str | None = None
    error: str | None = None
    results: list[dict] = field(default_factory=list)


class GateDemoProcessor:
    """Upload and process sample gate videos for demo ANPR without live cameras."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.demo_dir = (config.project_root / "data" / "demo_videos").resolve()
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._status = DemoStatus()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._refresh_uploaded_files()

    def _refresh_uploaded_files(self) -> None:
        front = self._video_path("gate_in")
        back = self._video_path("gate_out")
        self._status.front_video = front.name if front.exists() else None
        self._status.back_video = back.name if back.exists() else None

    def refresh_uploads(self) -> None:
        with self._lock:
            self._refresh_uploaded_files()

    def _video_path(self, role: str) -> Path:
        prefix = ROLE_FILES[role]
        for path in sorted(self.demo_dir.glob(f"{prefix}.*")):
            if path.suffix.lower() in ALLOWED_EXTENSIONS:
                return path
        return self.demo_dir / f"{prefix}.mp4"

    def status(self) -> dict:
        with self._lock:
            return asdict(self._status)

    def is_running(self) -> bool:
        with self._lock:
            return self._status.state == "running"

    def save_upload(self, role: str, filename: str, data: bytes) -> Path:
        if role not in ROLE_FILES:
            raise ValueError(f"Unknown gate role: {role}")

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{suffix}'. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        prefix = ROLE_FILES[role]
        for existing in self.demo_dir.glob(f"{prefix}.*"):
            if existing.is_file():
                existing.unlink()

        dest = self.demo_dir / f"{prefix}{suffix}"
        dest.write_bytes(data)
        self._refresh_uploaded_files()
        logger.info("Saved demo video for %s: %s", role, dest.name)
        return dest

    def _camera_for_role(self, role: str) -> CameraConfig:
        for camera in self.config.gate_cameras():
            if camera.role == role:
                return camera

        if role == "gate_in":
            return CameraConfig(
                id="demo_gate_front",
                name="Front Gate (Demo)",
                rtsp_url="",
                role="gate_in",
                gate_id="main_gate",
                gate_name="Front Gate",
            )
        return CameraConfig(
            id="demo_gate_back",
            name="Back Gate (Demo)",
            rtsp_url="",
            role="gate_out",
            gate_id="main_gate",
            gate_name="Back Gate",
        )

    def start(self, roles: list[str] | None = None) -> None:
        with self._lock:
            if self._status.state == "running":
                raise RuntimeError("Demo processing is already running")

            if roles is None:
                roles = []
                if self._video_path("gate_in").exists():
                    roles.append("gate_in")
                if self._video_path("gate_out").exists():
                    roles.append("gate_out")
            if not roles:
                raise RuntimeError("Upload at least one demo video before running")

            missing = [role for role in roles if not self._video_path(role).exists()]
            if missing:
                labels = ", ".join(
                    "front gate" if role == "gate_in" else "back gate" for role in missing
                )
                raise RuntimeError(f"Missing demo video for: {labels}")

            self._stop.clear()
            self._status = DemoStatus(
                state="running",
                message="Starting demo detection…",
                front_video=self._status.front_video,
                back_video=self._status.back_video,
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(roles,),
                name="gate-demo-processor",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self, roles: list[str]) -> None:
        total_events = 0
        results: list[dict] = []
        vehicles = VehicleManager(self.config)

        try:
            for index, role in enumerate(roles):
                if self._stop.is_set():
                    break

                label = "Front gate (entry)" if role == "gate_in" else "Back gate (exit)"
                video_path = self._video_path(role)
                camera = self._camera_for_role(role)

                with self._lock:
                    self._status.current_step = role
                    self._status.message = f"Processing {label}…"
                    self._status.progress_pct = (index / len(roles)) * 100.0

                def on_event(event) -> None:
                    results.append(
                        {
                            "plate_number": event.plate_number,
                            "direction": event.direction,
                            "vehicle_type": event.vehicle_type,
                            "vehicle_brand": event.vehicle_brand,
                            "confidence": round(event.confidence, 2),
                            "gate_name": event.gate_name,
                        }
                    )
                    with self._lock:
                        self._status.events_detected += 1

                def on_progress(ratio: float) -> None:
                    with self._lock:
                        base = index / len(roles)
                        self._status.progress_pct = min(
                            99.0,
                            (base + ratio / len(roles)) * 100.0,
                        )

                pipeline = GatePipeline(
                    self.config,
                    camera,
                    video_path=str(video_path),
                    shared_vehicle_manager=vehicles,
                )
                detected = pipeline.run(
                    on_event=on_event,
                    progress_callback=on_progress,
                    should_stop=self._stop.is_set,
                )
                total_events += detected
                logger.info("Demo %s finished with %d events", role, detected)

            with self._lock:
                if self._stop.is_set():
                    self._status.state = "idle"
                    self._status.message = "Demo processing was stopped."
                else:
                    self._status.state = "completed"
                    self._status.progress_pct = 100.0
                    self._status.message = (
                        f"Demo complete — {len(results)} vehicle event(s) recorded."
                    )
                    self._status.last_run_at = datetime.now(timezone.utc).isoformat()
                    self._status.results = results[-50:]
                self._status.current_step = "done"
                self._refresh_uploaded_files()

        except Exception as exc:
            logger.exception("Demo gate processing failed")
            with self._lock:
                self._status.state = "error"
                self._status.error = str(exc)
                self._status.message = "Demo processing failed."
        finally:
            vehicles.close()

    def clear_videos(self) -> None:
        if self.is_running():
            raise RuntimeError("Cannot clear videos while demo is running")

        for role in ROLE_FILES:
            prefix = ROLE_FILES[role]
            for existing in self.demo_dir.glob(f"{prefix}.*"):
                if existing.is_file():
                    existing.unlink()

        with self._lock:
            self._status.front_video = None
            self._status.back_video = None
