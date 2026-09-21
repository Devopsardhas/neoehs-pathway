from __future__ import annotations

from src.config import AppConfig

VIOLATION_PATHWAY_BLOCK = "pathway_block"
VIOLATION_MOBILE_USAGE = "mobile_usage"
VIOLATION_PPE = "ppe_violation"
VIOLATION_FIRE = "fire_detection"
VIOLATION_SMOKE = "smoke_detection"
VIOLATION_OBJECT_FALL = "object_fall"
VIOLATION_NEAR_MISS = "near_miss"
VIOLATION_OIL_SPILLAGE = "oil_spillage"
VIOLATION_WET_FLOOR = "wet_floor"
VIOLATION_LOW_VISIBILITY = "low_visibility"

CAMERA_WIDE_HAZARD_VIOLATIONS = (
    VIOLATION_FIRE,
    VIOLATION_SMOKE,
    VIOLATION_OIL_SPILLAGE,
    VIOLATION_WET_FLOOR,
    VIOLATION_LOW_VISIBILITY,
)

STATUS_ACTIVE = "active"
STATUS_RESOLVED = "resolved"


def compute_severity(duration_sec: float, config: AppConfig) -> str:
    if duration_sec < config.brief_threshold_sec:
        return "low"
    if duration_sec < config.moderate_threshold_sec:
        return "medium"
    return "high"


def compute_fire_severity(
    duration_sec: float,
    config: AppConfig,
    fire_type_key: str | None = None,
) -> str:
    if fire_type_key:
        for fire_type in config.fire_types:
            if fire_type.key == fire_type_key and fire_type.immediate_high:
                return "high"
    if config.fire_immediate_high and fire_type_key in (None, "structural_fire"):
        return "high"
    return compute_severity(duration_sec, config)


def compute_near_miss_severity(duration_sec: float, config: AppConfig) -> str:
    if config.near_miss_immediate_high:
        return "high"
    return compute_severity(duration_sec, config)


def compute_object_fall_severity(duration_sec: float, config: AppConfig) -> str:
    if config.object_fall_immediate_high:
        return "high"
    return compute_severity(duration_sec, config)


def compute_smoke_severity(duration_sec: float, config: AppConfig) -> str:
    if config.smoke_immediate_high:
        return "high"
    return compute_severity(duration_sec, config)


def compute_oil_spillage_severity(duration_sec: float, config: AppConfig) -> str:
    if config.oil_spillage_immediate_high:
        return "high"
    return compute_severity(duration_sec, config)


def compute_wet_floor_severity(duration_sec: float, config: AppConfig) -> str:
    if config.wet_floor_immediate_high:
        return "high"
    return compute_severity(duration_sec, config)


def compute_low_visibility_severity(duration_sec: float, config: AppConfig) -> str:
    if config.low_visibility_immediate_high:
        return "high"
    return compute_severity(duration_sec, config)
