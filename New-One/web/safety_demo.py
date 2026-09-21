from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import AppConfig
from src.detection_settings import DetectionSettings
from src.pipeline import PathwayMonitorPipeline

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEMO_SCENARIOS: dict[str, dict[str, str]] = {
    "fire_detection": {
        "prefix": "fire_demo",
        "label": "Fire Detection",
        "result_link": "/violations/fire_detection",
    },
    "object_fall": {
        "prefix": "object_fall_demo",
        "label": "Object Fall",
        "result_link": "/violations/object_fall",
    },
    "near_miss": {
        "prefix": "near_miss_demo",
        "label": "Near Miss",
        "result_link": "/violations/near_miss",
    },
}


@dataclass
class SafetyDemoStatus:
    state: str = "idle"
    message: str = "Upload sample videos and run safety demo detection."
    current_step: str = ""
    progress_pct: float = 0.0
    events_detected: int = 0
    fire_video: str | None = None
    object_fall_video: str | None = None
    near_miss_video: str | None = None
    last_run_at: str | None = None
    error: str | None = None
    results: list[dict] = field(default_factory=list)


class SafetyDemoProcessor:
    """Upload and process sample videos for fire, object fall, and near miss demos."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.demo_dir = (config.project_root / "data" / "demo_videos").resolve()
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._status = SafetyDemoStatus()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._refresh_uploaded_files()

    def _refresh_uploaded_files(self) -> None:
        self._status.fire_video = self._video_name("fire_detection")
        self._status.object_fall_video = self._video_name("object_fall")
        self._status.near_miss_video = self._video_name("near_miss")

    def refresh_uploads(self) -> None:
        with self._lock:
            self._refresh_uploaded_files()

    def _video_path(self, scenario: str) -> Path:
        prefix = DEMO_SCENARIOS[scenario]["prefix"]
        for path in sorted(self.demo_dir.glob(f"{prefix}.*")):
            if path.suffix.lower() in ALLOWED_EXTENSIONS:
                return path
        return self.demo_dir / f"{prefix}.mp4"

    def _video_name(self, scenario: str) -> str | None:
        path = self._video_path(scenario)
        return path.name if path.exists() else None

    def status(self) -> dict:
        with self._lock:
            payload = asdict(self._status)
            payload["scenarios"] = DEMO_SCENARIOS
            return payload

    def is_running(self) -> bool:
        with self._lock:
            return self._status.state == "running"

    def save_upload(self, scenario: str, filename: str, data: bytes) -> Path:
        if scenario not in DEMO_SCENARIOS:
            raise ValueError(f"Unknown demo scenario: {scenario}")

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{suffix}'. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        prefix = DEMO_SCENARIOS[scenario]["prefix"]
        for existing in self.demo_dir.glob(f"{prefix}.*"):
            if existing.is_file():
                existing.unlink()

        dest = self.demo_dir / f"{prefix}{suffix}"
        dest.write_bytes(data)
        self._refresh_uploaded_files()
        logger.info("Saved safety demo video for %s: %s", scenario, dest.name)
        return dest

    def _settings_for_scenario(self, scenario: str) -> DetectionSettings:
        base = DetectionSettings.from_config(self.config)
        base.pathway_block = False
        base.mobile_usage = False
        base.ppe_violation = False
        base.fire_detection = scenario == "fire_detection"
        base.smoke_detection = scenario == "fire_detection"
        base.object_fall = scenario == "object_fall"
        base.near_miss = scenario == "near_miss"
        base.oil_spillage = False
        base.wet_floor = False
        base.low_visibility = False
        return base

    def start(self, scenarios: list[str] | None = None) -> None:
        with self._lock:
            if self._status.state == "running":
                raise RuntimeError("Demo processing is already running")

            if scenarios is None:
                scenarios = [
                    key
                    for key in DEMO_SCENARIOS
                    if self._video_path(key).exists()
                ]
            if not scenarios:
                raise RuntimeError("Upload at least one safety demo video before running")

            missing = [key for key in scenarios if not self._video_path(key).exists()]
            if missing:
                labels = ", ".join(DEMO_SCENARIOS[key]["label"] for key in missing)
                raise RuntimeError(f"Missing demo video for: {labels}")

            self._stop.clear()
            self._status = SafetyDemoStatus(
                state="running",
                message="Starting safety demo detection…",
                fire_video=self._status.fire_video,
                object_fall_video=self._status.object_fall_video,
                near_miss_video=self._status.near_miss_video,
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(scenarios,),
                name="safety-demo-processor",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self, scenarios: list[str]) -> None:
        results: list[dict] = []

        try:
            for index, scenario in enumerate(scenarios):
                if self._stop.is_set():
                    break

                label = DEMO_SCENARIOS[scenario]["label"]
                video_path = self._video_path(scenario)

                with self._lock:
                    self._status.current_step = scenario
                    self._status.message = f"Processing {label}…"
                    self._status.progress_pct = (index / len(scenarios)) * 100.0

                demo_settings = self._settings_for_scenario(scenario)
                demo_settings.apply_to(self.config)

                def on_observation(observation) -> None:
                    results.append(
                        {
                            "violation_type": observation.violation_type,
                            "violation_label": label,
                            "object_class": observation.object_class,
                            "severity": observation.severity,
                            "summary": observation.summary,
                            "observation_id": observation.observation_id,
                        }
                    )
                    with self._lock:
                        self._status.events_detected += 1

                def on_progress(ratio: float) -> None:
                    with self._lock:
                        base = index / len(scenarios)
                        self._status.progress_pct = min(
                            99.0,
                            (base + ratio / len(scenarios)) * 100.0,
                        )

                pipeline = PathwayMonitorPipeline(
                    self.config,
                    persist_observations=True,
                    video_path=str(video_path),
                    lock_settings=True,
                )
                pipeline.run(
                    on_observation=on_observation,
                    progress_callback=on_progress,
                    should_stop=self._stop.is_set,
                )
                logger.info("Safety demo %s finished with %d events", scenario, len(results))

            with self._lock:
                if self._stop.is_set():
                    self._status.state = "idle"
                    self._status.message = "Demo processing was stopped."
                else:
                    self._status.state = "completed"
                    self._status.progress_pct = 100.0
                    self._status.message = (
                        f"Demo complete — {len(results)} observation(s) recorded."
                    )
                    self._status.last_run_at = datetime.now(timezone.utc).isoformat()
                    self._status.results = results[-50:]
                self._status.current_step = "done"
                self._refresh_uploaded_files()

        except Exception as exc:
            logger.exception("Safety demo processing failed")
            with self._lock:
                self._status.state = "error"
                self._status.error = str(exc)
                self._status.message = "Demo processing failed."

    def clear_videos(self) -> None:
        if self.is_running():
            raise RuntimeError("Cannot clear videos while demo is running")

        for scenario in DEMO_SCENARIOS:
            prefix = DEMO_SCENARIOS[scenario]["prefix"]
            for existing in self.demo_dir.glob(f"{prefix}.*"):
                if existing.is_file():
                    existing.unlink()

        with self._lock:
            self._status.fire_video = None
            self._status.object_fall_video = None
            self._status.near_miss_video = None
