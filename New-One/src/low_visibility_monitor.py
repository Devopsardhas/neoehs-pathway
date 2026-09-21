from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import AppConfig
from src.severity import VIOLATION_LOW_VISIBILITY


@dataclass(frozen=True)
class VisibilityMetrics:
    visibility_score: float
    laplacian_variance: float
    contrast: float
    brightness: float
    edge_density: float


class LowVisibilityMonitor:
    """Detects fog, haze, glare, and under-exposure that reduce scene visibility."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._streak = 0

    def analyze_frame(self, frame: np.ndarray) -> VisibilityMetrics:
        stride = max(1, int(self.config.low_visibility_sample_stride_px))
        height, width = frame.shape[:2]
        sample_width = max(160, width // stride)
        sample_height = max(120, height // stride)
        sample = cv2.resize(frame, (sample_width, sample_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        brightness = float(gray.mean())
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)

        lap_score = min(1.0, laplacian_variance / max(self.config.low_visibility_laplacian_reference, 1.0))
        contrast_score = min(1.0, contrast / max(self.config.low_visibility_contrast_reference, 1.0))
        edge_score = min(1.0, edge_density / max(self.config.low_visibility_edge_reference, 0.001))

        visibility_score = lap_score * 0.4 + contrast_score * 0.35 + edge_score * 0.25

        if brightness < self.config.low_visibility_brightness_min:
            visibility_score *= max(
                0.25,
                brightness / max(self.config.low_visibility_brightness_min, 1.0),
            )
        if brightness > self.config.low_visibility_brightness_max:
            visibility_score *= max(
                0.25,
                (255.0 - brightness)
                / max(255.0 - self.config.low_visibility_brightness_max, 1.0),
            )

        return VisibilityMetrics(
            visibility_score=max(0.0, min(1.0, visibility_score)),
            laplacian_variance=laplacian_variance,
            contrast=contrast,
            brightness=brightness,
            edge_density=edge_density,
        )

    def detect_violation(
        self,
        frame: np.ndarray,
        enabled: bool = True,
    ) -> tuple[bool, float, np.ndarray | None, VisibilityMetrics | None]:
        if not enabled:
            self._streak = 0
            return False, 0.0, None, None

        metrics = self.analyze_frame(frame)
        is_low = metrics.visibility_score < self.config.low_visibility_score_threshold
        confidence = max(0.0, min(1.0, 1.0 - metrics.visibility_score))

        if is_low:
            self._streak += 1
        else:
            self._streak = 0

        if self._streak < self.config.low_visibility_min_detection_frames:
            return False, confidence, None, metrics

        height, width = frame.shape[:2]
        xyxy = np.array([0.0, 0.0, float(width - 1), float(height - 1)], dtype=np.float32)
        return True, confidence, xyxy, metrics

    @staticmethod
    def violation_type() -> str:
        return VIOLATION_LOW_VISIBILITY

    @staticmethod
    def object_class() -> str:
        return "low_visibility"
