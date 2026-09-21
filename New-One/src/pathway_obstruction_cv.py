from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import AppConfig
from src.pathway import PathwayZone, polygon_parts


@dataclass
class PathwayObstruction:
    xyxy: np.ndarray
    zone: PathwayZone
    track_id: int
    area_px: float


def zone_mask(zone: PathwayZone, frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for part in polygon_parts(zone.polygon):
        pts = np.array(part.exterior.coords[:-1], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class PathwayObstructionMonitor:
    """
    Detect new static obstructions in pathway zones by comparing the live frame
    with the uploaded reference CCTV image.
    """

    def __init__(self, config: AppConfig, reference_bgr: np.ndarray | None):
        self.config = config
        self.reference_bgr = reference_bgr
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=config.pathway_block_cv_history_frames)
        )
        self._static_streak: dict[int, int] = defaultdict(int)
        self._next_track_id = 1
        self._active_tracks: dict[int, np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return (
            self.config.pathway_block_cv_enabled
            and self.reference_bgr is not None
        )

    def _resize_reference(self, frame_shape: tuple[int, ...]) -> np.ndarray | None:
        if self.reference_bgr is None:
            return None
        height, width = frame_shape[:2]
        ref = self.reference_bgr
        if ref.shape[0] != height or ref.shape[1] != width:
            ref = cv2.resize(ref, (width, height), interpolation=cv2.INTER_LINEAR)
        return ref

    def _prepare_gray(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reference = self._resize_reference(frame.shape)
        if reference is None:
            raise RuntimeError("Reference frame is missing")

        blur = self.config.pathway_block_cv_blur_kernel
        if blur % 2 == 0:
            blur += 1

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (blur, blur), 0)
        ref_gray = cv2.GaussianBlur(ref_gray, (blur, blur), 0)
        return frame_gray, ref_gray

    def _overlaps_person(
        self,
        xyxy: np.ndarray,
        person_boxes: list[np.ndarray],
    ) -> bool:
        pad = self.config.pathway_block_cv_person_padding_px
        for person_xyxy in person_boxes:
            padded = np.array(
                [
                    person_xyxy[0] - pad,
                    person_xyxy[1] - pad,
                    person_xyxy[2] + pad,
                    person_xyxy[3] + pad,
                ],
                dtype=float,
            )
            if _bbox_iou(xyxy, padded) >= 0.08:
                return True
        return False

    def _person_mask(
        self,
        frame_shape: tuple[int, ...],
        person_boxes: list[np.ndarray],
    ) -> np.ndarray:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pad = int(self.config.pathway_block_cv_person_padding_px)
        for person_xyxy in person_boxes:
            x1 = max(0, int(person_xyxy[0] - pad))
            y1 = max(0, int(person_xyxy[1] - pad))
            x2 = min(frame_shape[1], int(person_xyxy[2] + pad))
            y2 = min(frame_shape[0], int(person_xyxy[3] + pad))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def _is_large_object(self, xyxy: np.ndarray, area: float) -> bool:
        return area >= self.config.pathway_block_large_object_area_px

    def _match_track(self, center: tuple[float, float]) -> int:
        best_id = None
        best_dist = float(self.config.pathway_block_cv_match_distance_px)
        for track_id, previous in self._active_tracks.items():
            dist = float(np.hypot(center[0] - previous[0], center[1] - previous[1]))
            if dist <= best_dist:
                best_dist = dist
                best_id = track_id

        if best_id is not None:
            return best_id

        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _is_stationary(self, track_id: int, xyxy: np.ndarray) -> bool:
        points = self._history.get(track_id)
        if not points or len(points) < self.config.pathway_block_cv_min_static_frames:
            return False

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        displacement = max(max(xs) - min(xs), max(ys) - min(ys))
        width = max(float(xyxy[2] - xyxy[0]), 1.0)
        height = max(float(xyxy[3] - xyxy[1]), 1.0)
        diagonal = float(np.hypot(width, height))
        allowed = max(
            float(self.config.pathway_block_cv_max_displacement_px),
            diagonal * 0.05,
        )
        return displacement <= allowed

    def _dedupe_candidates(
        self,
        candidates: list[tuple[np.ndarray, PathwayZone, float]],
    ) -> list[tuple[np.ndarray, PathwayZone, float]]:
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda item: item[2], reverse=True)
        kept: list[tuple[np.ndarray, PathwayZone, float]] = []
        for xyxy, zone, area in ordered:
            center = (
                (float(xyxy[0]) + float(xyxy[2])) / 2.0,
                (float(xyxy[1]) + float(xyxy[3])) / 2.0,
            )
            if any(_bbox_iou(xyxy, existing[0]) >= 0.35 for existing in kept):
                continue
            if any(
                float(np.hypot(center[0] - ((e[0][0] + e[0][2]) / 2.0), center[1] - ((e[0][1] + e[0][3]) / 2.0)))
                <= self.config.pathway_block_cv_match_distance_px
                for e in kept
            ):
                continue
            kept.append((xyxy, zone, area))
        return kept

    def _best_walkway_zone(
        self,
        walkway_zones: list[PathwayZone],
        xyxy: np.ndarray,
    ) -> PathwayZone | None:
        best_zone: PathwayZone | None = None
        best_area = 0.0
        for zone in walkway_zones:
            area = zone.intersection_area(xyxy)
            if area > best_area:
                best_area = area
                best_zone = zone
        return best_zone

    def detect(
        self,
        frame: np.ndarray,
        walkway_zones: list[PathwayZone],
        person_boxes: list[np.ndarray],
    ) -> list[PathwayObstruction]:
        if not self.enabled or not walkway_zones:
            return []

        frame_gray, ref_gray = self._prepare_gray(frame)
        diff = cv2.absdiff(frame_gray, ref_gray)

        kernel_size = self.config.pathway_block_cv_morph_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1

        matched_tracks: set[int] = set()
        confirmed: list[PathwayObstruction] = []
        raw_candidates: list[tuple[np.ndarray, PathwayZone, float]] = []

        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for zone in walkway_zones:
            combined_mask = cv2.bitwise_or(combined_mask, zone_mask(zone, frame.shape))

        masked_diff = cv2.bitwise_and(diff, diff, mask=combined_mask)
        _, thresh = cv2.threshold(
            masked_diff,
            self.config.pathway_block_cv_diff_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        # Small open kernel preserves thin floor objects; a wide close
        # reconnects long horizontal plates that break into fragments.
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
        if self.config.pathway_block_cv_morph_kernel > 1:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(9, kernel_size * 2), 3),
            )
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        if person_boxes:
            person_mask = self._person_mask(frame.shape, person_boxes)
            thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(person_mask))

        frame_width = frame.shape[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 200:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            wide_flat = w >= max(h, 1) * self.config.small_object_wide_aspect
            min_width = (
                max(16, self.config.pathway_block_cv_min_width_px // 2)
                if wide_flat
                else self.config.pathway_block_cv_min_width_px
            )
            min_height = (
                max(self.config.small_object_min_height_px, 6)
                if wide_flat
                else self.config.pathway_block_cv_min_height_px
            )
            min_area = 350.0 if wide_flat else self.config.pathway_block_cv_min_area_px
            if w < min_width or h < min_height or area < min_area:
                continue
            xyxy = np.array([x, y, x + w, y + h], dtype=float)
            large = self._is_large_object(xyxy, area)
            if w >= frame_width * 0.55 and self._overlaps_person(xyxy, person_boxes) and not large:
                continue

            zone = self._best_walkway_zone(walkway_zones, xyxy)
            if zone is None:
                continue

            raw_candidates.append((xyxy, zone, area))

        deduped = self._dedupe_candidates(raw_candidates)
        for xyxy, zone, area in deduped:
            center = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)
            track_id = self._match_track(center)
            matched_tracks.add(track_id)
            self._active_tracks[track_id] = center
            self._history[track_id].append(center)

            if self._is_stationary(track_id, xyxy):
                self._static_streak[track_id] += 1
            else:
                self._static_streak[track_id] = 0

            large = self._is_large_object(xyxy, area)
            width = max(float(xyxy[2] - xyxy[0]), 1.0)
            height = max(float(xyxy[3] - xyxy[1]), 1.0)
            low_profile = height <= 56.0 or (
                width / height >= self.config.small_object_wide_aspect and height <= 90.0
            )
            moving_large_ready = (
                self.config.pathway_block_include_moving_large
                and large
                and len(self._history[track_id]) >= 2
            )
            low_profile_ready = (
                self.config.small_object_enabled
                and low_profile
                and len(self._history[track_id]) >= 2
            )
            if (
                self._static_streak[track_id] >= self.config.pathway_block_cv_min_static_frames
                or moving_large_ready
                or low_profile_ready
            ):
                confirmed.append(
                    PathwayObstruction(
                        xyxy=xyxy,
                        zone=zone,
                        track_id=track_id,
                        area_px=area,
                    )
                )

        stale = [track_id for track_id in self._active_tracks if track_id not in matched_tracks]
        for track_id in stale:
            self._active_tracks.pop(track_id, None)
            self._history.pop(track_id, None)
            self._static_streak.pop(track_id, None)

        return confirmed
