from __future__ import annotations

import numpy as np
import supervision as sv

VEHICLE_TYPE_ALIASES: dict[str, str] = {
    "car": "Car",
    "sedan": "Car",
    "hatchback": "Car",
    "suv": "SUV",
    "jeep": "SUV",
    "crossover": "SUV",
    "van": "Van",
    "minivan": "Van",
    "mpv": "Van",
    "motorcycle": "Bike",
    "bike": "Bike",
    "bicycle": "Bike",
    "scooter": "Bike",
    "moped": "Bike",
    "truck": "Lorry",
    "lorry": "Lorry",
    "pickup truck": "Pickup",
    "pickup": "Pickup",
    "bus": "Bus",
    "coach": "Bus",
    "auto": "Auto",
    "autorickshaw": "Auto",
    "three wheeler": "Auto",
    "vehicle": "Vehicle",
}


def normalize_vehicle_type(raw_class: str) -> str:
    key = raw_class.strip().lower()
    if key in VEHICLE_TYPE_ALIASES:
        return VEHICLE_TYPE_ALIASES[key]
    for alias, label in VEHICLE_TYPE_ALIASES.items():
        if alias in key or key in alias:
            return label
    return raw_class.strip().title() or "Unknown"


def format_brand_name(raw_brand: str) -> str:
    cleaned = " ".join(raw_brand.strip().split())
    if not cleaned:
        return "Unknown"
    return cleaned.title()


def _center(xyxy: np.ndarray) -> tuple[float, float]:
    return float((xyxy[0] + xyxy[2]) / 2.0), float((xyxy[1] + xyxy[3]) / 2.0)


def _point_in_box(point: tuple[float, float], xyxy: np.ndarray, padding: float = 0.0) -> bool:
    x, y = point
    return (
        float(xyxy[0]) - padding <= x <= float(xyxy[2]) + padding
        and float(xyxy[1]) - padding <= y <= float(xyxy[3]) + padding
    )


def detect_vehicle_brand(
    vehicle_xyxy: np.ndarray,
    brand_detections: sv.Detections,
    class_name_fn,
    brand_class_names: set[str],
    min_confidence: float = 0.20,
) -> tuple[str, float]:
    if len(brand_detections) == 0:
        return "Unknown", 0.0

    best_brand = "Unknown"
    best_conf = 0.0
    best_score = -1.0

    for idx in range(len(brand_detections)):
        class_name = class_name_fn(int(brand_detections.class_id[idx])).lower()
        if class_name not in brand_class_names:
            continue
        confidence = (
            float(brand_detections.confidence[idx])
            if brand_detections.confidence is not None
            else 0.0
        )
        if confidence < min_confidence:
            continue

        brand_xyxy = brand_detections.xyxy[idx]
        brand_center = _center(brand_xyxy)
        if not _point_in_box(brand_center, vehicle_xyxy, padding=12.0):
            continue

        score = confidence
        if score > best_score:
            best_score = score
            best_brand = format_brand_name(class_name)
            best_conf = confidence

    return best_brand, best_conf
