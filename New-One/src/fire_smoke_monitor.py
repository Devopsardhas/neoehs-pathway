from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from src.config import AppConfig
from src.fire_temporal import FireTemporalValidator
from src.smoke_temporal import SmokeTemporalValidator
from src.severity import VIOLATION_FIRE, VIOLATION_SMOKE

logger = logging.getLogger(__name__)


def _center(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


@dataclass
class _FireCluster:
    indices: list[int] = field(default_factory=list)
    cx: float = 0.0
    cy: float = 0.0
    xyxy: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    confidence: float = 0.0
    subtype: str = "structural_fire"
    track_id: int | None = None


def _cluster_fire_detections(
    detections: sv.Detections,
    subtypes: list[str],
    merge_px: float,
) -> list[_FireCluster]:
    """Group nearby flame boxes into one fire incident (single-linkage by center distance)."""
    if len(detections) == 0:
        return []

    clusters: list[_FireCluster] = []
    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        cx, cy = _center(xyxy)
        confidence = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 0.0
        )
        subtype = subtypes[i] if i < len(subtypes) else "structural_fire"
        track_id = (
            int(detections.tracker_id[i])
            if detections.tracker_id is not None
            and detections.tracker_id[i] is not None
            else None
        )

        target: _FireCluster | None = None
        for cluster in clusters:
            for member_idx in cluster.indices:
                member_xyxy = detections.xyxy[member_idx]
                mx, my = _center(member_xyxy)
                if _distance(mx, my, cx, cy) <= merge_px:
                    target = cluster
                    break
            if target is not None:
                break

        if target is None:
            clusters.append(
                _FireCluster(
                    indices=[i],
                    cx=cx,
                    cy=cy,
                    xyxy=xyxy.astype(float).copy(),
                    confidence=confidence,
                    subtype=subtype,
                    track_id=track_id,
                )
            )
            continue

        target.indices.append(i)
        member_count = len(target.indices)
        target.cx = ((target.cx * (member_count - 1)) + cx) / member_count
        target.cy = ((target.cy * (member_count - 1)) + cy) / member_count
        target.xyxy[0] = min(float(target.xyxy[0]), float(xyxy[0]))
        target.xyxy[1] = min(float(target.xyxy[1]), float(xyxy[1]))
        target.xyxy[2] = max(float(target.xyxy[2]), float(xyxy[2]))
        target.xyxy[3] = max(float(target.xyxy[3]), float(xyxy[3]))
        if confidence >= target.confidence:
            target.confidence = confidence
            target.subtype = subtype
            target.track_id = track_id

    return clusters


class FireSmokeMonitor:
    """Detects fire (by subtype) and smoke in the camera view."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._streak: dict[tuple[str, int, int], int] = defaultdict(int)
        self._bucket_miss: dict[tuple[str, int, int], int] = defaultdict(int)
        self._bucket_centers: dict[tuple[str, int, int], tuple[float, float]] = {}
        self._smoke_bucket_centers: dict[tuple[str, int, int], tuple[float, float]] = {}
        self._temporal = FireTemporalValidator(config)
        self._smoke_temporal = SmokeTemporalValidator(config)

    def _spatial_bucket(self, violation_type: str, object_class: str, cx: float, cy: float) -> tuple[str, str, int, int]:
        bucket = int(self.config.fire_smoke_spatial_bucket_px)
        return (violation_type, object_class, int(cx // bucket), int(cy // bucket))

    def _fire_incident_bucket(self, cx: float, cy: float) -> tuple[str, int, int]:
        merge_px = float(self.config.fire_merge_distance_px)
        for bucket, (bx, by) in self._bucket_centers.items():
            if bucket[0] != VIOLATION_FIRE:
                continue
            if _distance(cx, cy, bx, by) <= merge_px:
                self._bucket_centers[bucket] = (cx, cy)
                return bucket

        grid = int(merge_px)
        bucket = (VIOLATION_FIRE, int(cx // grid), int(cy // grid))
        self._bucket_centers[bucket] = (cx, cy)
        return bucket

    def _smoke_incident_bucket(self, cx: float, cy: float) -> tuple[str, int, int]:
        merge_px = float(self.config.smoke_merge_distance_px)
        for bucket, (bx, by) in self._smoke_bucket_centers.items():
            if bucket[0] != VIOLATION_SMOKE:
                continue
            if _distance(cx, cy, bx, by) <= merge_px:
                self._smoke_bucket_centers[bucket] = (cx, cy)
                return bucket

        grid = int(merge_px)
        bucket = (VIOLATION_SMOKE, int(cx // grid), int(cy // grid))
        self._smoke_bucket_centers[bucket] = (cx, cy)
        return bucket

    def _process_fire_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        subtypes: list[str],
        min_frames: int,
    ) -> tuple[list[tuple[str, np.ndarray, float, int | None, str]], set[tuple[str, str, float, float]]]:
        confirmed: list[tuple[str, np.ndarray, float, int | None, str]] = []
        active_spatial: set[tuple[str, str, float, float]] = set()
        seen_buckets: set[tuple[str, int, int]] = set()
        active_buckets: set[tuple[str, int, int]] = set()

        merge_px = self.config.fire_merge_distance_px
        clusters = _cluster_fire_detections(detections, subtypes, merge_px)

        for cluster in clusters:
            bucket = self._fire_incident_bucket(cluster.cx, cluster.cy)
            active_buckets.add(bucket)
            active_spatial.add((VIOLATION_FIRE, cluster.subtype, cluster.cx, cluster.cy))
            self._bucket_miss[bucket] = 0
            self._temporal.record(bucket, frame, cluster.xyxy)

            required_streak = (
                self.config.fire_trusted_min_frames
                if cluster.confidence >= self.config.fire_trusted_confidence
                else self.config.fire_min_detection_frames
            )
            self._streak[bucket] += 1
            streak_ready = self._streak[bucket] >= required_streak
            temporal_ready = self._temporal.is_confirmed(bucket, cluster.confidence)
            if streak_ready and temporal_ready and bucket not in seen_buckets:
                confirmed.append(
                    (
                        VIOLATION_FIRE,
                        cluster.xyxy,
                        cluster.confidence,
                        cluster.track_id,
                        cluster.subtype,
                    )
                )
                seen_buckets.add(bucket)
            elif streak_ready and not temporal_ready and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Fire candidate waiting for temporal confirm | streak=%d bucket=%s conf=%.2f",
                    self._streak[bucket],
                    bucket,
                    cluster.confidence,
                )

        miss_limit = max(1, int(self.config.fire_streak_miss_frames))
        for bucket in list(self._streak):
            if bucket[0] != VIOLATION_FIRE or bucket in active_buckets:
                continue
            self._bucket_miss[bucket] += 1
            if self._bucket_miss[bucket] <= miss_limit:
                continue
            self._streak.pop(bucket, None)
            self._bucket_miss.pop(bucket, None)
            self._bucket_centers.pop(bucket, None)
            self._temporal.clear_bucket(bucket)

        return confirmed, active_spatial

    def _process_smoke_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        min_frames: int,
    ) -> tuple[list[tuple[str, np.ndarray, float, int | None, str]], set[tuple[str, str, float, float]]]:
        confirmed: list[tuple[str, np.ndarray, float, int | None, str]] = []
        active_spatial: set[tuple[str, str, float, float]] = set()
        seen_buckets: set[tuple[str, int, int]] = set()
        active_buckets: set[tuple[str, int, int]] = set()
        subtype = "smoke"

        merge_px = self.config.smoke_merge_distance_px
        subtypes = [subtype] * len(detections)
        clusters = _cluster_fire_detections(detections, subtypes, merge_px)

        for cluster in clusters:
            bucket = self._smoke_incident_bucket(cluster.cx, cluster.cy)
            active_buckets.add(bucket)
            active_spatial.add((VIOLATION_SMOKE, subtype, cluster.cx, cluster.cy))
            self._smoke_temporal.record(bucket, frame, cluster.xyxy)

            required_streak = (
                self.config.smoke_trusted_min_frames
                if cluster.confidence >= self.config.smoke_trusted_confidence
                else min_frames
            )
            self._streak[bucket] += 1
            streak_ready = self._streak[bucket] >= required_streak
            temporal_ready = self._smoke_temporal.is_confirmed(bucket, cluster.confidence)
            if streak_ready and temporal_ready and bucket not in seen_buckets:
                confirmed.append(
                    (VIOLATION_SMOKE, cluster.xyxy, cluster.confidence, cluster.track_id, subtype)
                )
                seen_buckets.add(bucket)

        for bucket in [key for key in self._streak if key[0] == VIOLATION_SMOKE and key not in active_buckets]:
            self._streak.pop(bucket, None)
            self._smoke_bucket_centers.pop(bucket, None)
            self._smoke_temporal.clear_bucket(bucket)

        return confirmed, active_spatial

    def detect_violations(
        self,
        frame: np.ndarray,
        fire_detections: sv.Detections,
        fire_subtypes: list[str],
        smoke_detections: sv.Detections,
        fire_enabled: bool = True,
        smoke_enabled: bool = True,
    ) -> tuple[
        list[tuple[str, np.ndarray, float, int | None, str]],
        list[tuple[str, np.ndarray, float, int | None, str]],
        set[tuple[str, str, float, float]],
    ]:
        fire_confirmed: list[tuple[str, np.ndarray, float, int | None, str]] = []
        smoke_confirmed: list[tuple[str, np.ndarray, float, int | None, str]] = []
        active_spatial: set[tuple[str, str, float, float]] = set()

        if fire_enabled:
            fire_confirmed, fire_active = self._process_fire_detections(
                frame,
                fire_detections,
                fire_subtypes,
                self.config.fire_min_detection_frames,
            )
            active_spatial |= fire_active
        else:
            self._clear_streaks(VIOLATION_FIRE)

        if smoke_enabled:
            smoke_confirmed, smoke_active = self._process_smoke_detections(
                frame,
                smoke_detections,
                self.config.smoke_min_detection_frames,
            )
            active_spatial |= smoke_active
        else:
            self._clear_streaks(VIOLATION_SMOKE)

        return fire_confirmed, smoke_confirmed, active_spatial

    def _clear_streaks(self, violation_type: str) -> None:
        for bucket in [key for key in self._streak if key[0] == violation_type]:
            self._streak.pop(bucket, None)
            self._bucket_miss.pop(bucket, None)
            self._bucket_centers.pop(bucket, None)
        if violation_type == VIOLATION_FIRE:
            self._temporal.clear_fire_history()
        elif violation_type == VIOLATION_SMOKE:
            self._smoke_temporal.clear_smoke_history()
            self._smoke_bucket_centers = {
                key: value
                for key, value in self._smoke_bucket_centers.items()
                if key[0] != VIOLATION_SMOKE
            }
