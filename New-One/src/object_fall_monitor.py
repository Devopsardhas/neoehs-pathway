from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import supervision as sv

from src.config import AppConfig
from src.pathway import PathwayZone, ZoneRef, zone_ref_for_object, zone_ref_for_person

FALL_KIND_OBJECT = "object_fall"
FALL_KIND_PERSON = "person_fall"


def _motion_points(xyxy: np.ndarray) -> tuple[float, float]:
    foot_y = float(xyxy[3])
    center_y = float((xyxy[1] + xyxy[3]) / 2.0)
    return foot_y, center_y


class ObjectFallMonitor:
    """Detects falling objects or persons from downward motion anywhere in the camera view."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=config.object_fall_history_frames)
        )
        self._fall_streak: dict[int, int] = defaultdict(int)

    def _zone_label(
        self,
        xyxy: np.ndarray,
        label_zones: list[PathwayZone],
        camera_id: str,
        fall_kind: str,
    ) -> ZoneRef:
        if fall_kind == FALL_KIND_PERSON:
            return zone_ref_for_person(
                xyxy,
                label_zones,
                camera_id,
                self.config.object_fall_person_min_overlap,
            )
        return zone_ref_for_object(
            xyxy,
            label_zones,
            camera_id,
            min_overlap_ratio=self.config.pathway_min_overlap_ratio,
            min_overlap_area_px=self.config.pathway_min_overlap_area_px,
            use_foot_point=self.config.pathway_use_foot_point,
        )

    @staticmethod
    def _series_delta(values: list[float]) -> tuple[float, float, float]:
        if len(values) < 2:
            return 0.0, 0.0, 0.0

        total_delta = values[-1] - values[0]
        steps = len(values) - 1
        velocity = total_delta / max(steps, 1)
        step_deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
        max_step = max(step_deltas) if step_deltas else 0.0
        return total_delta, velocity, max_step

    def _is_falling(self, track_id: int, foot_y: float, center_y: float) -> bool:
        history = self._history[track_id]
        history.append((foot_y, center_y))

        if len(history) < 2:
            return False

        recent_count = max(2, int(self.config.object_fall_recent_frames))
        recent = list(history)[-recent_count:]
        foot_values = [sample[0] for sample in recent]
        center_values = [sample[1] for sample in recent]
        combined_values = [max(foot, center) for foot, center in recent]

        recent_delta, recent_velocity, recent_max_step = self._series_delta(combined_values)
        total_delta, total_velocity, total_max_step = self._series_delta(
            [max(foot, center) for foot, center in history]
        )

        sudden_drop = max(recent_max_step, total_max_step) >= self.config.object_fall_sudden_drop_px
        recent_fall = (
            recent_delta >= self.config.object_fall_recent_min_distance_px
            and recent_velocity >= self.config.object_fall_min_velocity_px
        )
        sustained_fall = (
            total_delta >= self.config.object_fall_min_distance_px
            and total_velocity >= self.config.object_fall_min_velocity_px
        )

        # Ignore objects that are clearly being lifted (recent upward motion dominates).
        if recent_delta <= -self.config.object_fall_recent_min_distance_px:
            return False

        return sudden_drop or recent_fall or sustained_fall

    def detect_falls(
        self,
        detections: sv.Detections,
        label_zones: list[PathwayZone],
        camera_id: str,
        class_name_fn,
        fall_kind: str,
    ) -> tuple[list[tuple[ZoneRef, int, np.ndarray, str, str]], set[int]]:
        if len(detections) == 0 or detections.tracker_id is None:
            return [], set()

        confirmed: list[tuple[ZoneRef, int, np.ndarray, str, str]] = []
        active_tracks: set[int] = set()
        seen_tracks: set[int] = set()
        seen_this_frame: set[int] = set()

        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            if track_id is None:
                continue

            tid = int(track_id)
            seen_this_frame.add(tid)
            xyxy = detections.xyxy[i]
            class_name = class_name_fn(int(detections.class_id[i]))
            zone = self._zone_label(xyxy, label_zones, camera_id, fall_kind)

            foot_y, center_y = _motion_points(xyxy)
            if self._is_falling(tid, foot_y, center_y):
                self._fall_streak[tid] += 1
                active_tracks.add(tid)
                if (
                    self._fall_streak[tid] >= self.config.object_fall_min_frames
                    and tid not in seen_tracks
                ):
                    confirmed.append((zone, tid, xyxy, class_name, fall_kind))
                    seen_tracks.add(tid)
            else:
                self._fall_streak[tid] = 0

        for tid in [tid for tid in self._history if tid not in seen_this_frame]:
            self._history.pop(tid, None)
            self._fall_streak.pop(tid, None)

        return confirmed, active_tracks

    def is_track_falling(self, track_id: int) -> bool:
        return self._fall_streak.get(int(track_id), 0) > 0

    def prune_tracks(self, active_track_ids: set[int]) -> None:
        for tid in list(self._history):
            if tid not in active_track_ids:
                self._history.pop(tid, None)
                self._fall_streak.pop(tid, None)
