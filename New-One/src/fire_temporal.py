from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import AppConfig
from src.severity import VIOLATION_FIRE


@dataclass
class _FlameSample:
    cx: float
    cy: float
    area: float
    density: float
    brightness_std: float
    flame_value_std: float


def _clamp_xyxy(xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(width, int(xyxy[2]))
    y2 = min(height, int(xyxy[3]))
    return x1, y1, x2, y2


def _fire_hsv_mask(hsv: np.ndarray) -> np.ndarray:
    lower_red_a = np.array([0, 80, 80], dtype=np.uint8)
    upper_red_a = np.array([20, 255, 255], dtype=np.uint8)
    lower_red_b = np.array([160, 80, 80], dtype=np.uint8)
    upper_red_b = np.array([179, 255, 255], dtype=np.uint8)
    lower_orange = np.array([8, 100, 120], dtype=np.uint8)
    upper_orange = np.array([35, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red_a, upper_red_a),
        cv2.inRange(hsv, lower_red_b, upper_red_b),
    )
    return cv2.bitwise_or(mask, cv2.inRange(hsv, lower_orange, upper_orange))


def analyze_flame_region(frame: np.ndarray, xyxy: np.ndarray) -> _FlameSample:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_xyxy(xyxy, width, height)
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    area = float(max(0, x2 - x1) * max(0, y2 - y1))

    if x2 <= x1 or y2 <= y1:
        return _FlameSample(cx=cx, cy=cy, area=area, density=0.0, brightness_std=0.0, flame_value_std=0.0)

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    fire_mask = _fire_hsv_mask(hsv)
    density = float(np.count_nonzero(fire_mask)) / float(fire_mask.size)
    brightness_std = float(np.std(hsv[:, :, 2]))
    flame_values = hsv[:, :, 2][fire_mask > 0]
    flame_value_std = float(np.std(flame_values)) if flame_values.size else 0.0
    return _FlameSample(
        cx=cx,
        cy=cy,
        area=area,
        density=density,
        brightness_std=brightness_std,
        flame_value_std=flame_value_std,
    )


class FireTemporalValidator:
    """Block static red/yellow objects; allow YOLO fire and flickering CV flame regions."""

    def __init__(self, config: AppConfig):
        self.config = config
        maxlen = max(3, int(config.fire_trusted_min_frames))
        self._history: dict[tuple[str, int, int], deque[_FlameSample]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )

    def record(self, bucket: tuple[str, int, int], frame: np.ndarray, xyxy: np.ndarray) -> None:
        self._history[bucket].append(analyze_flame_region(frame, xyxy))

    def _samples(self, bucket: tuple[str, int, int]) -> list[_FlameSample]:
        history = self._history.get(bucket)
        if not history:
            return []
        return list(history)

    def _metrics(self, samples: list[_FlameSample]) -> dict[str, float]:
        areas = np.array([sample.area for sample in samples], dtype=float)
        densities = np.array([sample.density for sample in samples], dtype=float)
        brightness = np.array([sample.brightness_std for sample in samples], dtype=float)
        flame_values = np.array([sample.flame_value_std for sample in samples], dtype=float)
        centers = np.array([(sample.cx, sample.cy) for sample in samples], dtype=float)

        density_std = float(np.std(densities))
        brightness_std = float(np.std(brightness))
        area_std = float(np.std(areas))
        frame_drifts = [
            float(np.linalg.norm(centers[index] - centers[index - 1]))
            for index in range(1, len(centers))
        ]
        total_drift = float(np.linalg.norm(centers[-1] - centers[0])) if len(centers) >= 2 else 0.0
        mean_jitter = float(np.mean(frame_drifts)) if frame_drifts else 0.0
        mean_area = float(np.mean(areas))
        texture_flicker = max(density_std, brightness_std / 255.0)
        return {
            "mean_area": mean_area,
            "mean_density": float(np.mean(densities)),
            "mean_brightness": float(np.mean(brightness)),
            "mean_flame_value_std": float(np.mean(flame_values)),
            "density_std": density_std,
            "total_drift": total_drift,
            "mean_jitter": mean_jitter,
            "texture_flicker": texture_flicker,
        }

    def _is_static_colored_object(self, metrics: dict[str, float], yolo_confidence: float) -> bool:
        """Static red bag, warehouse label, yellow box — no frame-to-frame flame activity."""
        if yolo_confidence >= self.config.fire_trusted_confidence:
            return False
        return (
            metrics["total_drift"] <= self.config.fire_temporal_static_max_drift_px
            and metrics["texture_flicker"] < self.config.fire_temporal_min_density_std
        )

    def _is_flat_color_sign(self, metrics: dict[str, float], yolo_confidence: float) -> bool:
        return (
            yolo_confidence < self.config.fire_trusted_confidence
            and metrics["total_drift"] <= self.config.fire_temporal_static_max_drift_px
            and metrics["mean_density"] > self.config.fire_temporal_flat_density_min
            and metrics["mean_brightness"] < self.config.fire_temporal_flame_brightness_min
        )

    def _is_bulk_motion_false_positive(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["total_drift"] >= self.config.fire_temporal_bulk_motion_min_drift_px
            and metrics["texture_flicker"] < self.config.fire_temporal_min_density_std
            and metrics["mean_brightness"] < self.config.fire_temporal_flame_brightness_min
        )

    def _is_moving_colored_object(self, metrics: dict[str, float], yolo_confidence: float) -> bool:
        """Red pouch/bag carried by a person — box motion must not bypass static checks."""
        if yolo_confidence >= self.config.fire_trusted_confidence:
            return False
        return (
            metrics["total_drift"] > self.config.fire_temporal_static_max_drift_px
            and metrics["mean_flame_value_std"] < self.config.fire_temporal_flame_value_std_min
        )

    def _has_live_flame_signature(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["mean_density"] >= self.config.fire_temporal_flame_density_min
            and metrics["mean_flame_value_std"] >= self.config.fire_temporal_flame_value_std_min
        )

    def _has_temporal_flicker(self, metrics: dict[str, float]) -> bool:
        # Pixel-level variation only — box jitter from walking people is not flame flicker.
        return metrics["texture_flicker"] >= self.config.fire_temporal_min_density_std

    def is_confirmed(self, bucket: tuple[str, int, int], yolo_confidence: float = 0.0) -> bool:
        samples = self._samples(bucket)
        if len(samples) < self.config.fire_trusted_min_frames:
            return False

        metrics = self._metrics(samples)

        if self._is_flat_color_sign(metrics, yolo_confidence):
            return False

        if self._is_static_colored_object(metrics, yolo_confidence):
            return False

        if self._is_moving_colored_object(metrics, yolo_confidence):
            return False

        if yolo_confidence < self.config.fire_trusted_confidence and self._is_bulk_motion_false_positive(
            metrics
        ):
            return False

        # Strong YOLO fire/flame score — fast confirm.
        if yolo_confidence >= self.config.fire_trusted_confidence:
            return True

        # CV fallback or weak YOLO — must flicker over time (blocks static red bags/labels).
        if not self._has_live_flame_signature(metrics):
            return False
        return self._has_temporal_flicker(metrics)

    def clear_bucket(self, bucket: tuple[str, int, int]) -> None:
        self._history.pop(bucket, None)

    def clear_fire_history(self) -> None:
        stale = [bucket for bucket in self._history if bucket[0] == VIOLATION_FIRE]
        for bucket in stale:
            self.clear_bucket(bucket)
