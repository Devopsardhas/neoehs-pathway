from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import supervision as sv

from src.config import AppConfig


class StaticObjectFilter:
    """Marks tracked objects as static when their anchor point barely moves over time."""

    LARGE_OBJECT_AREA_PX = 20_000

    def __init__(self, config: AppConfig):
        self.config = config
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=config.history_frames)
        )
        self._static_streak: dict[int, int] = defaultdict(int)

    def _track_point(self, xyxy: np.ndarray) -> tuple[float, float]:
        x1, y1, x2, y2 = map(float, xyxy)
        if (x2 - x1) * (y2 - y1) >= self.LARGE_OBJECT_AREA_PX:
            # Large machinery bboxes jitter at the top — track floor contact instead.
            return ((x1 + x2) / 2.0, y2)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _max_displacement_for_bbox(self, xyxy: np.ndarray) -> float:
        x1, y1, x2, y2 = map(float, xyxy)
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        diagonal = float(np.hypot(width, height))
        scaled = diagonal * 0.05
        return max(float(self.config.max_displacement_px), scaled)

    def _is_stationary(self, track_id: int, xyxy: np.ndarray) -> bool:
        points = self._history.get(track_id)
        if not points or len(points) < self.config.min_static_frames:
            return False

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        displacement = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
        )
        return displacement <= self._max_displacement_for_bbox(xyxy)

    def annotate(self, detections: sv.Detections) -> tuple[sv.Detections, np.ndarray]:
        if detections.tracker_id is None or len(detections) == 0:
            return detections, np.array([], dtype=bool)

        static_mask = np.zeros(len(detections), dtype=bool)

        for i, track_id in enumerate(detections.tracker_id):
            if track_id is None:
                continue

            tid = int(track_id)
            anchor = self._track_point(detections.xyxy[i])
            self._history[tid].append(anchor)

            if self._is_stationary(tid, detections.xyxy[i]):
                self._static_streak[tid] += 1
            else:
                self._static_streak[tid] = 0

            static_mask[i] = self._static_streak[tid] >= self.config.min_static_frames

        static_detections = detections[static_mask]
        return static_detections, static_mask

    def prune(self, active_track_ids: set[int]) -> None:
        stale = [tid for tid in self._history if tid not in active_track_ids]
        for tid in stale:
            self._history.pop(tid, None)
            self._static_streak.pop(tid, None)
