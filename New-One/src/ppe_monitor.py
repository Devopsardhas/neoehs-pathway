from __future__ import annotations

from collections import defaultdict

import numpy as np
import supervision as sv

from src.config import AppConfig
from src.pathway import PathwayZone, ZoneRef, zone_ref_for_person
from src.ppe_items import PPEItemConfig


def _center(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _region_xyxy(person_xyxy: np.ndarray, top_ratio: float, bottom_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = person_xyxy
    height = y2 - y1
    return np.array([x1, y1 + height * top_ratio, x2, y1 + height * bottom_ratio], dtype=float)


def _bbox_contains_point(xyxy: np.ndarray, x: float, y: float, padding: float = 0.0) -> bool:
    x1, y1, x2, y2 = xyxy
    return (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding)


class PPEMonitor:
    """
    Detects persons anywhere in the camera view without required PPE items.
    Produces one violation per person with all missing items grouped together.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._violation_streak: dict[int, int] = defaultdict(int)

    def _ppe_near_region(
        self,
        region_xyxy: np.ndarray,
        ppe_detections: sv.Detections,
    ) -> bool:
        if len(ppe_detections) == 0:
            return False

        region_center = _center(region_xyxy)
        pad = self.config.ppe_match_padding_px

        for i in range(len(ppe_detections)):
            ppe_xyxy = ppe_detections.xyxy[i]
            ppe_center = _center(ppe_xyxy)

            if _distance(region_center, ppe_center) <= self.config.ppe_match_distance_px:
                return True

            px, py = ppe_center
            if _bbox_contains_point(region_xyxy, px, py, padding=pad):
                return True

        return self._any_ppe_overlaps_region(region_xyxy, ppe_detections, pad)

    @staticmethod
    def _any_ppe_overlaps_region(
        region_xyxy: np.ndarray,
        ppe_detections: sv.Detections,
        padding: float,
    ) -> bool:
        rx1, ry1, rx2, ry2 = region_xyxy
        rx1 -= padding
        ry1 -= padding
        rx2 += padding
        ry2 += padding

        for i in range(len(ppe_detections)):
            px1, py1, px2, py2 = ppe_detections.xyxy[i]
            if px1 <= rx2 and px2 >= rx1 and py1 <= ry2 and py2 >= ry1:
                return True
        return False

    def _missing_items(
        self,
        person_xyxy: np.ndarray,
        item_detections: dict[str, sv.Detections],
        active_items: list[PPEItemConfig],
    ) -> list[str]:
        missing: list[str] = []
        for item in active_items:
            region = _region_xyxy(person_xyxy, item.region_top, item.region_bottom)
            detections = item_detections.get(item.key, sv.Detections.empty())
            if not self._ppe_near_region(region, detections):
                missing.append(item.missing_key)
        return missing

    def detect_violations(
        self,
        person_detections: sv.Detections,
        item_detections: dict[str, sv.Detections],
        label_zones: list[PathwayZone],
        camera_id: str,
    ) -> tuple[list[tuple[ZoneRef, int, np.ndarray, list[str]]], set[int]]:
        if not self.config.ppe_enabled:
            return [], set()

        active_items = self.config.enabled_ppe_items()
        if not active_items:
            return [], set()

        if person_detections.tracker_id is None or len(person_detections) == 0:
            return [], set()

        confirmed: list[tuple[ZoneRef, int, np.ndarray, list[str]]] = []
        active_track_ids: set[int] = set()

        for i in range(len(person_detections)):
            track_id = person_detections.tracker_id[i]
            if track_id is None:
                continue

            tid = int(track_id)
            person_xyxy = person_detections.xyxy[i]
            zone = zone_ref_for_person(
                person_xyxy,
                label_zones,
                camera_id,
                self.config.ppe_person_min_overlap,
            )

            missing_items = self._missing_items(person_xyxy, item_detections, active_items)
            if not missing_items:
                self._violation_streak.pop(tid, None)
                continue

            active_track_ids.add(tid)
            self._violation_streak[tid] += 1
            if self._violation_streak[tid] >= self.config.ppe_min_violation_frames:
                confirmed.append((zone, tid, person_xyxy, missing_items))

        for tid in [tid for tid in self._violation_streak if tid not in active_track_ids]:
            self._violation_streak.pop(tid, None)

        return confirmed, active_track_ids

    def prune_track(self, active_track_ids: set[int]) -> None:
        stale = [tid for tid in self._violation_streak if tid not in active_track_ids]
        for tid in stale:
            self._violation_streak.pop(tid, None)
