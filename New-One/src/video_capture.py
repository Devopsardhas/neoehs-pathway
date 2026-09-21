from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoFileCapture:
    """Reads frames from a local video file once, then stops."""

    def __init__(self, path: str | Path, fallback_fps: float = 25.0):
        self.path = str(path)
        self.fallback_fps = fallback_fps if fallback_fps > 1e-3 else 25.0
        self._cap: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0.0

    def _open(self) -> bool:
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            logger.error("Failed to open video file: %s", self.path)
            return False

        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 1e-3 else 0.0
        logger.info(
            "Opened demo video %s (%d frames, %.2f fps, %.1fs)",
            self.path,
            self.total_frames,
            self.effective_fps,
            self.duration_seconds or 0.0,
        )
        return True

    @property
    def effective_fps(self) -> float:
        return self.fps if self.fps > 1e-3 else self.fallback_fps

    @property
    def duration_seconds(self) -> float | None:
        if self.total_frames <= 0:
            return None
        return self.total_frames / self.effective_fps

    def current_time_sec(self) -> float:
        """Playback time of the current frame, not wall-clock processing time."""
        msec = 0.0
        if self._cap is not None:
            msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if msec > 0:
            seconds = msec / 1000.0
        elif self.frame_count > 0:
            seconds = (self.frame_count - 1) / self.effective_fps
        else:
            seconds = 0.0

        limit = self.duration_seconds
        if limit is not None:
            seconds = min(seconds, limit)
        return max(0.0, seconds)

    def frames(self) -> Generator[np.ndarray, None, None]:
        if not self._open():
            return

        self.frame_count = 0
        while self._cap is not None and self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok or frame is None or frame.size == 0:
                break
            self.frame_count += 1
            yield frame

    def progress_ratio(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, self.frame_count / self.total_frames)

    def read_one(self) -> np.ndarray | None:
        for frame in self.frames():
            return frame
        return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
