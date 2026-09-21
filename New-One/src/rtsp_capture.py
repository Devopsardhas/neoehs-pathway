from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Error-resilient RTSP decode: TCP transport, discard corrupt NAL units, ignore decode errors.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer+discardcorrupt|"
    "flags;low_delay|"
    "err_detect;ignore_err|"
    "max_delay;500000|"
    "reorder_queue_size;0"
)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


@contextlib.contextmanager
def _quiet_ffmpeg_stderr() -> Generator[None, None, None]:
    """Hide libav h264 decode spam for known-bad CCTV frames."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull)


def _is_valid_frame(frame: np.ndarray | None) -> bool:
    if frame is None or frame.size == 0:
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    height, width = frame.shape[:2]
    if height < 16 or width < 16:
        return False
    # Corrupt h264 decode often returns flat gray/black partial frames.
    if float(np.std(frame)) < 2.0:
        return False
    return True


class RTSPCapture:
    """Reads frames from an RTSP stream with automatic reconnection."""

    def __init__(
        self,
        url: str,
        reconnect_delay_sec: int = 5,
        max_read_failures: int = 8,
    ):
        self.url = url
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_read_failures = max_read_failures
        self._cap: cv2.VideoCapture | None = None
        self._corrupt_frame_count = 0

    def _connect(self) -> bool:
        if self._cap is not None:
            self._cap.release()

        with _quiet_ffmpeg_stderr():
            self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            logger.error("Failed to open RTSP stream: %s", self.url)
            return False

        logger.info("Connected to RTSP stream")
        return True

    def _discard_buffered_frames(self, count: int = 2) -> None:
        if self._cap is None:
            return
        with _quiet_ffmpeg_stderr():
            for _ in range(count):
                self._cap.grab()

    def discard_stale_frames(self, max_grabs: int = 4) -> None:
        """Drop buffered RTSP frames so inference works on the latest image."""
        if self._cap is None:
            return
        with _quiet_ffmpeg_stderr():
            for _ in range(max_grabs):
                if not self._cap.grab():
                    break

    def _read_valid_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None

        with _quiet_ffmpeg_stderr():
            ok, frame = self._cap.read()

        if ok and _is_valid_frame(frame):
            return True, frame

        if self._cap is None:
            return False, None

        with _quiet_ffmpeg_stderr():
            grabbed = self._cap.grab()
            if not grabbed:
                return False, None
            ok, frame = self._cap.retrieve()

        if ok and _is_valid_frame(frame):
            return True, frame

        return False, None

    def frames(self) -> Generator[np.ndarray, None, None]:
        consecutive_failures = 0

        while True:
            if self._cap is None or not self._cap.isOpened():
                if not self._connect():
                    time.sleep(self.reconnect_delay_sec)
                    continue
                consecutive_failures = 0
                self._discard_buffered_frames()

            ok, frame = self._read_valid_frame()
            if not ok or frame is None:
                consecutive_failures += 1
                self._corrupt_frame_count += 1
                if consecutive_failures == 1:
                    logger.debug("Skipped corrupt RTSP frame, trying next frame")
                elif consecutive_failures == self.max_read_failures:
                    logger.debug(
                        "Skipped %d corrupt RTSP frames in a row",
                        consecutive_failures,
                    )

                if consecutive_failures < self.max_read_failures:
                    time.sleep(0.02)
                    continue

                logger.warning(
                    "RTSP stream stalled after %d bad frames, reconnecting...",
                    consecutive_failures,
                )
                if self._cap is not None:
                    self._cap.release()
                self._cap = None
                consecutive_failures = 0
                time.sleep(self.reconnect_delay_sec)
                continue

            consecutive_failures = 0
            yield frame

    def read_one(self) -> np.ndarray | None:
        for frame in self.frames():
            return frame
        return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
