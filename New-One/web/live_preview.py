from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import cv2
import numpy as np

from src.config import AppConfig
from src.pipeline import PathwayMonitorPipeline
from src.rtsp_capture import RTSPCapture

logger = logging.getLogger(__name__)

MJPEG_BOUNDARY = b"frame"
PREVIEW_MAX_WIDTH = 1280


class LivePreviewManager:
    """Runs camera preview with optional detection overlays for the dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, bytes] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._errors: dict[str, str] = {}
        self._status: dict[str, str] = {}
        self._detection_active: dict[str, bool] = {}
        self._warmup_started = False
        self._warmup_lock = threading.Lock()

    @staticmethod
    def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes | None:
        if PREVIEW_MAX_WIDTH and frame.shape[1] > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(
                frame,
                (PREVIEW_MAX_WIDTH, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return encoded.tobytes() if ok else None

    def _build_status_frame(self, title: str, detail: str = "") -> bytes:
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        image[:] = (24, 28, 36)
        cv2.putText(
            image,
            title,
            (36, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 190, 210),
            2,
            cv2.LINE_AA,
        )
        if detail:
            cv2.putText(
                image,
                detail[:72],
                (36, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (140, 150, 170),
                1,
                cv2.LINE_AA,
            )
        return self._encode_jpeg(image, quality=75) or b""

    def _set_status(self, camera_id: str, status: str) -> None:
        with self._lock:
            self._status[camera_id] = status
            if camera_id not in self._frames or not self._detection_active.get(camera_id):
                self._frames[camera_id] = self._build_status_frame(status)

    def list_cameras(self, config: AppConfig) -> list[dict[str, str]]:
        return [{"id": camera.id, "name": camera.name} for camera in config.cameras]

    def get_status(self, camera_id: str) -> dict[str, str | None]:
        with self._lock:
            return {
                "status": self._status.get(camera_id, "idle"),
                "error": self._errors.get(camera_id),
                "streaming": str(camera_id in self._frames and camera_id not in self._errors).lower(),
            }

    def get_frame(self, camera_id: str) -> bytes:
        with self._lock:
            if camera_id in self._errors:
                return self._build_status_frame("Preview unavailable", self._errors[camera_id])
            if camera_id in self._frames:
                return self._frames[camera_id]
            return self._build_status_frame("Connecting to camera...")

    def warmup(self, config: AppConfig) -> None:
        with self._warmup_lock:
            if self._warmup_started:
                return
            self._warmup_started = True

        def _load() -> None:
            try:
                logger.info("Pre-loading detection model for dashboard preview...")
                PathwayMonitorPipeline(config, show_preview=False, persist_observations=False)
                logger.info("Detection model ready for dashboard preview")
            except Exception:
                logger.exception("Dashboard preview model warmup failed")

        threading.Thread(target=_load, name="live-preview-warmup", daemon=True).start()

    def ensure_camera(self, config: AppConfig, camera_id: str) -> None:
        with self._lock:
            thread = self._threads.get(camera_id)
            if thread is not None and thread.is_alive():
                return

            for other_id, stop_event in list(self._stop_events.items()):
                if other_id != camera_id:
                    stop_event.set()

            stop_event = threading.Event()
            self._stop_events[camera_id] = stop_event
            self._errors.pop(camera_id, None)
            self._detection_active.pop(camera_id, None)
            self._status[camera_id] = "Starting preview..."
            self._frames[camera_id] = self._build_status_frame("Connecting to camera...")

            thread = threading.Thread(
                target=self._run_camera,
                args=(config, camera_id, stop_event),
                name=f"live-preview-{camera_id}",
                daemon=True,
            )
            self._threads[camera_id] = thread
            thread.start()

    def _raw_rtsp_loop(
        self,
        camera_config: AppConfig,
        camera_id: str,
        stop_event: threading.Event,
    ) -> None:
        capture = RTSPCapture(
            camera_config.rtsp_url,
            camera_config.reconnect_delay_sec,
            camera_config.max_read_failures,
        )
        try:
            for frame in capture.frames():
                if stop_event.is_set():
                    break
                with self._lock:
                    if self._detection_active.get(camera_id):
                        continue
                jpeg = self._encode_jpeg(frame)
                if not jpeg:
                    continue
                with self._lock:
                    if not self._detection_active.get(camera_id):
                        self._frames[camera_id] = jpeg
                        self._status[camera_id] = "Live (loading detections...)"
        except Exception:
            logger.exception("Raw RTSP preview failed for camera %s", camera_id)
        finally:
            capture.release()

    def _run_camera(
        self,
        config: AppConfig,
        camera_id: str,
        stop_event: threading.Event,
    ) -> None:
        try:
            camera = config.get_camera(camera_id)
        except KeyError as exc:
            with self._lock:
                self._errors[camera_id] = str(exc)
            return

        camera_config = config.with_camera(camera)
        persist = config.web_live_preview_persist

        raw_thread = threading.Thread(
            target=self._raw_rtsp_loop,
            args=(camera_config, camera_id, stop_event),
            name=f"live-preview-raw-{camera_id}",
            daemon=True,
        )
        raw_thread.start()

        try:
            self._set_status(camera_id, "Loading detection model...")
            pipeline = PathwayMonitorPipeline(
                camera_config,
                show_preview=False,
                persist_observations=persist,
            )
        except Exception as exc:
            logger.exception("Live preview init failed for camera %s", camera_id)
            with self._lock:
                self._errors[camera_id] = f"Init failed: {exc}"
            stop_event.set()
            return

        def on_frame(preview: np.ndarray) -> None:
            if stop_event.is_set():
                return
            jpeg = self._encode_jpeg(preview)
            if not jpeg:
                return
            with self._lock:
                self._detection_active[camera_id] = True
                self._frames[camera_id] = jpeg
                self._status[camera_id] = "Live with detections"

        self._set_status(camera_id, "Starting detections...")
        try:
            pipeline.run(frame_callback=on_frame, should_stop=stop_event.is_set)
        except Exception as exc:
            logger.exception("Live preview failed for camera %s", camera_id)
            with self._lock:
                if camera_id not in self._detection_active:
                    self._errors[camera_id] = str(exc)
                else:
                    self._status[camera_id] = f"Detection stopped: {exc}"
        finally:
            stop_event.set()
            raw_thread.join(timeout=2.0)
            with self._lock:
                self._frames.pop(camera_id, None)
                self._threads.pop(camera_id, None)
                self._stop_events.pop(camera_id, None)
                self._detection_active.pop(camera_id, None)
                if camera_id not in self._errors:
                    self._status[camera_id] = "Preview stopped"

    async def mjpeg_stream(self, config: AppConfig, camera_id: str) -> Any:
        self.ensure_camera(config, camera_id)
        idle_checks = 0
        while True:
            with self._lock:
                thread = self._threads.get(camera_id)
                thread_alive = thread is not None and thread.is_alive()
                has_error = camera_id in self._errors

            if not thread_alive and not has_error:
                idle_checks += 1
                if idle_checks >= 25:
                    idle_checks = 0
                    self.ensure_camera(config, camera_id)
            else:
                idle_checks = 0

            frame = self.get_frame(camera_id)
            yield (
                b"--" + MJPEG_BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            await asyncio.sleep(0.08)
