from __future__ import annotations

from collections import defaultdict

import numpy as np
import supervision as sv

from src.config import AppConfig
from src.severity import VIOLATION_OIL_SPILLAGE, VIOLATION_WET_FLOOR


def _center(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class FloorHazardMonitor:
    """Detects oil spillage and wet floor hazards anywhere in the camera view."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._streak: dict[tuple[str, str, int, int], int] = defaultdict(int)

    def _spatial_bucket(
        self,
        violation_type: str,
        object_class: str,
        cx: float,
        cy: float,
    ) -> tuple[str, str, int, int]:
        bucket = int(self.config.floor_hazard_spatial_bucket_px)
        return (violation_type, object_class, int(cx // bucket), int(cy // bucket))

    def _process_detections(
        self,
        detections: sv.Detections,
        violation_type: str,
        object_class: str,
        min_frames: int,
    ) -> tuple[list[tuple[str, np.ndarray, float, int | None, str]], set[tuple[str, str, float, float]]]:
        confirmed: list[tuple[str, np.ndarray, float, int | None, str]] = []
        active_spatial: set[tuple[str, str, float, float]] = set()
        seen_buckets: set[tuple[str, str, int, int]] = set()
        active_buckets: set[tuple[str, str, int, int]] = set()

        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            confidence = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else 0.0
            )
            track_id = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                and detections.tracker_id[i] is not None
                else None
            )
            cx, cy = _center(xyxy)
            bucket = self._spatial_bucket(violation_type, object_class, cx, cy)
            active_buckets.add(bucket)
            active_spatial.add((violation_type, object_class, cx, cy))

            self._streak[bucket] += 1
            if self._streak[bucket] >= min_frames and bucket not in seen_buckets:
                confirmed.append((violation_type, xyxy, confidence, track_id, object_class))
                seen_buckets.add(bucket)

        for bucket in [key for key in self._streak if key[0] == violation_type and key not in active_buckets]:
            self._streak.pop(bucket, None)

        return confirmed, active_spatial

    def detect_violations(
        self,
        oil_detections: sv.Detections,
        wet_floor_detections: sv.Detections,
        oil_enabled: bool = True,
        wet_floor_enabled: bool = True,
    ) -> tuple[
        list[tuple[str, np.ndarray, float, int | None, str]],
        list[tuple[str, np.ndarray, float, int | None, str]],
        set[tuple[str, str, float, float]],
    ]:
        oil_confirmed: list[tuple[str, np.ndarray, float, int | None, str]] = []
        wet_floor_confirmed: list[tuple[str, np.ndarray, float, int | None, str]] = []
        active_spatial: set[tuple[str, str, float, float]] = set()

        if oil_enabled:
            oil_confirmed, oil_active = self._process_detections(
                oil_detections,
                VIOLATION_OIL_SPILLAGE,
                "oil_spillage",
                self.config.oil_spillage_min_detection_frames,
            )
            active_spatial |= oil_active
        else:
            self._clear_streaks(VIOLATION_OIL_SPILLAGE)

        if wet_floor_enabled:
            wet_floor_confirmed, wet_active = self._process_detections(
                wet_floor_detections,
                VIOLATION_WET_FLOOR,
                "wet_floor",
                self.config.wet_floor_min_detection_frames,
            )
            active_spatial |= wet_active
        else:
            self._clear_streaks(VIOLATION_WET_FLOOR)

        return oil_confirmed, wet_floor_confirmed, active_spatial

    def _clear_streaks(self, violation_type: str) -> None:
        for bucket in [key for key in self._streak if key[0] == violation_type]:
            self._streak.pop(bucket, None)
