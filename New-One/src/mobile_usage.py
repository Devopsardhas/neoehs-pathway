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


def _bbox_contains_point(xyxy: np.ndarray, x: float, y: float, padding: float = 0.0) -> bool:
    x1, y1, x2, y2 = xyxy
    return (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding)


def _upper_body_xyxy(person_xyxy: np.ndarray, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = person_xyxy
    height = y2 - y1
    return np.array([x1, y1, x2, y1 + height * ratio], dtype=float)


class MobileUsageMonitor:
    """
    Mobile phone use is ALLOWED inside magenta (mobile_usage) zones only.
    Violation when a person uses a phone anywhere else in the camera view.

    Deduplication is track-scoped: one person roaming the frame produces a
    single continuous violation episode.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._usage_streak: dict[int, int] = defaultdict(int)
        self._phone_miss: dict[int, int] = defaultdict(int)

    def _phone_near_person(
        self,
        person_xyxy: np.ndarray,
        phone_detections: sv.Detections,
    ) -> bool:
        if len(phone_detections) == 0:
            return False

        person_center = _center(person_xyxy)
        upper_body = _upper_body_xyxy(person_xyxy, self.config.mobile_upper_body_ratio)
        pad = self.config.mobile_person_phone_padding_px

        for i in range(len(phone_detections)):
            phone_xyxy = phone_detections.xyxy[i]
            phone_center = _center(phone_xyxy)

            if _distance(person_center, phone_center) <= self.config.mobile_person_phone_distance_px:
                return True

            px, py = phone_center
            if _bbox_contains_point(person_xyxy, px, py, padding=pad):
                return True

            if _bbox_contains_point(upper_body, px, py, padding=pad):
                return True

        return False

    def _person_in_any_zone(
        self,
        person_xyxy: np.ndarray,
        zones: list[PathwayZone],
    ) -> bool:
        return any(
            zone.person_in_zone(person_xyxy, self.config.mobile_person_min_overlap)
            for zone in zones
        )

    def _is_using_phone(
        self,
        person_xyxy: np.ndarray,
        phone_detections: sv.Detections,
        label_zones: list[PathwayZone],
    ) -> bool:
        if self._phone_near_person(person_xyxy, phone_detections):
            return True

        if not self.config.mobile_allow_phone_in_zone_match:
            return False

        for i in range(len(phone_detections)):
            phone_xyxy = phone_detections.xyxy[i]
            for zone in label_zones:
                if zone.person_in_zone(phone_xyxy, self.config.mobile_person_min_overlap):
                    return True
        return False

    def _reset_track_state(self, tid: int) -> None:
        self._usage_streak.pop(tid, None)
        self._phone_miss.pop(tid, None)

    def detect_violations(
        self,
        person_detections: sv.Detections,
        phone_detections: sv.Detections,
        label_zones: list[PathwayZone],
        mobile_allowed_zones: list[PathwayZone],
        camera_id: str,
    ) -> tuple[list[tuple[ZoneRef, int, np.ndarray]], set[int]]:
        """
        Returns confirmed violations and track IDs currently using a phone
        outside allowed mobile zones (for stale resolution).
        """
        if not self.config.mobile_usage_enabled:
            return [], set()

        if person_detections.tracker_id is None or len(person_detections) == 0:
            return [], set()

        confirmed: list[tuple[ZoneRef, int, np.ndarray]] = []
        seen_tracks: set[int] = set()
        active_track_ids: set[int] = set()
        seen_this_frame: set[int] = set()

        for i in range(len(person_detections)):
            track_id = person_detections.tracker_id[i]
            if track_id is None:
                continue

            tid = int(track_id)
            seen_this_frame.add(tid)
            person_xyxy = person_detections.xyxy[i]

            if self._person_in_any_zone(person_xyxy, mobile_allowed_zones):
                self._phone_miss[tid] += 1
                if self._phone_miss[tid] >= self.config.mobile_phone_miss_frames:
                    self._usage_streak[tid] = 0
                continue

            if not self._is_using_phone(person_xyxy, phone_detections, label_zones):
                self._phone_miss[tid] += 1
                if self._phone_miss[tid] >= self.config.mobile_phone_miss_frames:
                    self._usage_streak[tid] = 0
                continue

            self._phone_miss.pop(tid, None)
            active_track_ids.add(tid)
            self._usage_streak[tid] += 1

            if (
                self._usage_streak[tid] >= self.config.mobile_min_usage_frames
                and tid not in seen_tracks
            ):
                zone = zone_ref_for_person(
                    person_xyxy,
                    label_zones,
                    camera_id,
                    self.config.mobile_person_min_overlap,
                )
                confirmed.append((zone, tid, person_xyxy))
                seen_tracks.add(tid)

        for tid in list(self._usage_streak):
            if tid not in seen_this_frame:
                self._reset_track_state(tid)

        return confirmed, active_track_ids

    def prune_track(self, active_track_ids: set[int]) -> None:
        stale = [tid for tid in self._usage_streak if tid not in active_track_ids]
        for tid in stale:
            self._reset_track_state(tid)
