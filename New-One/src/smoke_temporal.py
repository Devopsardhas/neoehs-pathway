from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import AppConfig


@dataclass
class _SmokeSample:
    cx: float
    cy: float
    area: float
    gray_std: float
    edge_ratio: float
    mask_fill: float
    solidity: float
    rectangularity: float


def _clamp_xyxy(xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(width, int(xyxy[2]))
    y2 = min(height, int(xyxy[3]))
    return x1, y1, x2, y2


def analyze_smoke_region(frame: np.ndarray, xyxy: np.ndarray) -> _SmokeSample:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_xyxy(xyxy, width, height)
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    area = float(max(0, x2 - x1) * max(0, y2 - y1))

    if x2 <= x1 or y2 <= y1:
        return _SmokeSample(
            cx=cx,
            cy=cy,
            area=area,
            gray_std=0.0,
            edge_ratio=0.0,
            mask_fill=0.0,
            solidity=0.0,
            rectangularity=0.0,
        )

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    smoke_mask = (
        (saturation < 45)
        & (gray > 70)
        & (gray < 210)
    ).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 50, 150)

    mask_fill = 0.0
    solidity = 0.0
    rectangularity = 0.0
    roi_area = float(smoke_mask.size)
    contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        mask_fill = contour_area / roi_area if roi_area else 0.0
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0
        _, _, rect_w, rect_h = cv2.boundingRect(contour)
        rect_area = float(max(1, rect_w * rect_h))
        rectangularity = contour_area / rect_area

    return _SmokeSample(
        cx=cx,
        cy=cy,
        area=area,
        gray_std=float(np.std(gray)),
        edge_ratio=float(np.count_nonzero(edges)) / float(edges.size),
        mask_fill=mask_fill,
        solidity=solidity,
        rectangularity=rectangularity,
    )


def is_smoke_like_region(sample: _SmokeSample, config: AppConfig) -> bool:
    """Reject rectangular office surfaces (monitors, panels) that match gray smoke color."""
    if sample.gray_std < config.fire_smoke_cv_smoke_min_texture_std:
        return False
    if sample.edge_ratio > config.fire_smoke_cv_smoke_max_edge_ratio:
        return False
    if sample.mask_fill > config.fire_smoke_cv_smoke_max_mask_fill:
        return False
    if sample.solidity > config.fire_smoke_cv_smoke_max_solidity:
        return False
    if sample.rectangularity > config.fire_smoke_cv_smoke_max_rectangularity:
        return False
    return True


def passes_smoke_shape_filters(
    frame: np.ndarray,
    xyxy: np.ndarray,
    frame_area: float,
    config: AppConfig,
) -> bool:
    box_area = float(max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1]))
    if box_area / frame_area > config.fire_smoke_cv_smoke_max_area_ratio:
        return False

    x1, y1, x2, y2 = [int(v) for v in xyxy]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    frame_h, frame_w = frame.shape[:2]
    if max(box_w, box_h) / max(1, min(box_w, box_h)) > config.fire_smoke_cv_smoke_max_aspect_ratio:
        return False

    # Cubicle edges / ceiling strips often touch the frame border.
    edge_margin = max(8, int(min(frame_h, frame_w) * 0.04))
    if y1 <= edge_margin and box_h > box_w * 1.2:
        return False
    if x1 <= edge_margin and box_w < box_h * 0.8:
        return False

    return is_smoke_like_region(analyze_smoke_region(frame, xyxy), config)


class SmokeTemporalValidator:
    """Reject static white/shadow CV regions; allow YOLO smoke and drifting plumes."""

    def __init__(self, config: AppConfig):
        self.config = config
        maxlen = max(3, int(config.smoke_min_detection_frames))
        self._history: dict[tuple[str, int, int], deque[_SmokeSample]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )

    def record(self, bucket: tuple[str, int, int], frame: np.ndarray, xyxy: np.ndarray) -> None:
        self._history[bucket].append(analyze_smoke_region(frame, xyxy))

    def _samples(self, bucket: tuple[str, int, int]) -> list[_SmokeSample]:
        history = self._history.get(bucket)
        if not history:
            return []
        return list(history)

    def _metrics(self, samples: list[_SmokeSample]) -> dict[str, float]:
        areas = np.array([sample.area for sample in samples], dtype=float)
        gray_stds = np.array([sample.gray_std for sample in samples], dtype=float)
        centers = np.array([(sample.cx, sample.cy) for sample in samples], dtype=float)

        texture_std = float(np.std(gray_stds))
        area_std = float(np.std(areas))
        mean_area = float(np.mean(areas))
        total_drift = float(np.linalg.norm(centers[-1] - centers[0])) if len(centers) >= 2 else 0.0
        frame_drifts = [
            float(np.linalg.norm(centers[index] - centers[index - 1]))
            for index in range(1, len(centers))
        ]
        mean_jitter = float(np.mean(frame_drifts)) if frame_drifts else 0.0
        texture_flicker = max(texture_std / 255.0, area_std / max(mean_area, 1.0))
        return {
            "mean_gray_std": float(np.mean(gray_stds)),
            "texture_flicker": texture_flicker,
            "total_drift": total_drift,
            "mean_jitter": mean_jitter,
        }

    def is_confirmed(self, bucket: tuple[str, int, int], yolo_confidence: float = 0.0) -> bool:
        samples = self._samples(bucket)
        if len(samples) < self.config.smoke_min_detection_frames:
            return False

        if yolo_confidence >= self.config.smoke_trusted_confidence:
            return True

        metrics = self._metrics(samples)
        if metrics["mean_gray_std"] < self.config.fire_smoke_cv_smoke_min_texture_std:
            return False

        if metrics["total_drift"] >= self.config.smoke_temporal_min_drift_px:
            return True

        if (
            metrics["total_drift"] <= self.config.smoke_temporal_static_max_drift_px
            and metrics["texture_flicker"] < self.config.smoke_temporal_min_texture_std
        ):
            return False

        return metrics["texture_flicker"] >= self.config.smoke_temporal_min_texture_std

    def clear_bucket(self, bucket: tuple[str, int, int]) -> None:
        self._history.pop(bucket, None)

    def clear_smoke_history(self) -> None:
        for bucket in list(self._history):
            self.clear_bucket(bucket)
