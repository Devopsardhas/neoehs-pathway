from __future__ import annotations

from collections import defaultdict

import numpy as np
import supervision as sv

from src.config import AppConfig
from src.pathway import PathwayZone, ZoneRef, zone_ref_for_person


def _center(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class NearMissMonitor:
    """Detects close proximity between a person and a moving hazard anywhere in the camera view."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._proximity_streak: dict[tuple[int, str], int] = defaultdict(int)

    def _is_hazard_class(self, class_name: str) -> bool:
        lowered = class_name.lower()
        return any(
            term in lowered or lowered in term
            for term in self.config.near_miss_hazard_classes
        )

    def detect_violations(
        self,
        person_detections: sv.Detections,
        hazard_detections: sv.Detections,
        hazard_class_names: list[str],
        label_zones: list[PathwayZone],
        camera_id: str,
    ) -> tuple[
        list[tuple[ZoneRef, int, np.ndarray, str, int, np.ndarray]],
        set[tuple[int, str]],
    ]:
        """
        Returns violations and active (person_track, hazard_class) pairs.
        """
        if not self.config.near_miss_enabled:
            return [], set()

        if (
            person_detections.tracker_id is None
            or hazard_detections.tracker_id is None
            or len(person_detections) == 0
            or len(hazard_detections) == 0
        ):
            return [], set()

        confirmed: list[tuple[ZoneRef, int, np.ndarray, str, int, np.ndarray]] = []
        active_pairs: set[tuple[int, str]] = set()
        seen_pairs: set[tuple[int, str]] = set()

        for p_idx in range(len(person_detections)):
            person_track = person_detections.tracker_id[p_idx]
            if person_track is None:
                continue

            person_tid = int(person_track)
            person_xyxy = person_detections.xyxy[p_idx]
            zone = zone_ref_for_person(
                person_xyxy,
                label_zones,
                camera_id,
                self.config.near_miss_person_min_overlap,
            )
            person_center = _center(person_xyxy)

            for h_idx in range(len(hazard_detections)):
                hazard_track = hazard_detections.tracker_id[h_idx]
                if hazard_track is None:
                    continue

                hazard_tid = int(hazard_track)
                if hazard_tid == person_tid:
                    continue

                hazard_xyxy = hazard_detections.xyxy[h_idx]
                hazard_class = hazard_class_names[h_idx] if h_idx < len(hazard_class_names) else "hazard"
                hazard_center = _center(hazard_xyxy)

                if _distance(person_center, hazard_center) > self.config.near_miss_max_distance_px:
                    continue

                pair = (person_tid, hazard_class)
                active_pairs.add(pair)
                self._proximity_streak[pair] += 1

                if (
                    self._proximity_streak[pair] >= self.config.near_miss_min_frames
                    and pair not in seen_pairs
                ):
                    confirmed.append(
                        (zone, person_tid, person_xyxy, hazard_class, hazard_tid, hazard_xyxy)
                    )
                    seen_pairs.add(pair)

        for pair in [key for key in self._proximity_streak if key not in active_pairs]:
            self._proximity_streak.pop(pair, None)

        return confirmed, active_pairs

    def prune_tracks(self, active_person_tracks: set[int]) -> None:
        stale = [
            pair
            for pair in self._proximity_streak
            if pair[0] not in active_person_tracks
        ]
        for pair in stale:
            self._proximity_streak.pop(pair, None)
