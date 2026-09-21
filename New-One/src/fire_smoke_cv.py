from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from src.config import AppConfig
from src.fire_temporal import analyze_flame_region
from src.smoke_temporal import analyze_smoke_region, passes_smoke_shape_filters


def _boxes_from_mask(
    mask: np.ndarray,
    min_area: int,
    class_id: int,
    confidence: float,
) -> sv.Detections:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[float]] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
        confidences.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


def detect_fire_regions(frame: np.ndarray, config: AppConfig) -> sv.Detections:
    """HSV color fallback for flame-like orange/red regions."""
    if not config.fire_smoke_cv_fallback_enabled:
        return sv.Detections.empty()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_orange, upper_orange))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    detections = _boxes_from_mask(
        mask,
        min_area=config.fire_smoke_cv_fire_min_area,
        class_id=0,
        confidence=config.fire_smoke_cv_fire_confidence,
    )
    if len(detections) == 0:
        return detections

    keep_indices: list[int] = []
    min_flame_std = config.fire_temporal_flame_value_std_min
    for index in range(len(detections)):
        sample = analyze_flame_region(frame, detections.xyxy[index])
        if sample.flame_value_std >= min_flame_std:
            keep_indices.append(index)

    if not keep_indices:
        return sv.Detections.empty()
    return detections[keep_indices]


def detect_smoke_regions(frame: np.ndarray, config: AppConfig) -> sv.Detections:
    """Low-saturation gray-region fallback for smoke plumes."""
    if not config.fire_smoke_cv_fallback_enabled:
        return sv.Detections.empty()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (
        (saturation < config.fire_smoke_cv_smoke_max_saturation)
        & (gray > config.fire_smoke_cv_smoke_min_brightness)
        & (gray < config.fire_smoke_cv_smoke_max_brightness)
        & (value > config.fire_smoke_cv_smoke_min_brightness)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    detections = _boxes_from_mask(
        mask,
        min_area=config.fire_smoke_cv_smoke_min_area,
        class_id=0,
        confidence=config.fire_smoke_cv_smoke_confidence,
    )
    if len(detections) == 0:
        return detections

    frame_area = float(frame.shape[0] * frame.shape[1])
    keep_indices: list[int] = []
    for index in range(len(detections)):
        if passes_smoke_shape_filters(frame, detections.xyxy[index], frame_area, config):
            keep_indices.append(index)

    if not keep_indices:
        return sv.Detections.empty()
    return detections[keep_indices]


def _core_detections(detections: sv.Detections) -> sv.Detections:
    """Strip extra metadata so YOLO and CV boxes can be merged safely."""
    if len(detections) == 0:
        return detections

    kwargs: dict = {"xyxy": detections.xyxy}
    if detections.confidence is not None:
        kwargs["confidence"] = detections.confidence
    if detections.class_id is not None:
        kwargs["class_id"] = detections.class_id
    return sv.Detections(**kwargs)


def merge_detections(primary: sv.Detections, fallback: sv.Detections) -> sv.Detections:
    if len(primary) == 0:
        return fallback
    if len(fallback) == 0:
        return primary

    primary_core = _core_detections(primary)
    fallback_core = _core_detections(fallback)
    return sv.Detections.merge([primary_core, fallback_core])
