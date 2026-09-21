from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import AppConfig
from src.ppe_items import (
    PPE_ITEM_BOOTS,
    PPE_ITEM_GLOVES,
    PPE_ITEM_HELMET,
    PPE_ITEM_VEST,
)
from src.severity import (
    VIOLATION_FIRE,
    VIOLATION_MOBILE_USAGE,
    VIOLATION_NEAR_MISS,
    VIOLATION_OBJECT_FALL,
    VIOLATION_OIL_SPILLAGE,
    VIOLATION_PATHWAY_BLOCK,
    VIOLATION_PPE,
    VIOLATION_SMOKE,
    VIOLATION_WET_FLOOR,
    VIOLATION_LOW_VISIBILITY,
)

SETTINGS_FILENAME = "data/detection_settings.json"

VIOLATION_SETTING_KEYS = {
    VIOLATION_PATHWAY_BLOCK: "pathway_block",
    VIOLATION_MOBILE_USAGE: "mobile_usage",
    VIOLATION_PPE: "ppe_violation",
    VIOLATION_FIRE: "fire_detection",
    VIOLATION_SMOKE: "smoke_detection",
    VIOLATION_OBJECT_FALL: "object_fall",
    VIOLATION_NEAR_MISS: "near_miss",
    VIOLATION_OIL_SPILLAGE: "oil_spillage",
    VIOLATION_WET_FLOOR: "wet_floor",
    VIOLATION_LOW_VISIBILITY: "low_visibility",
}

SETTING_LABELS = {
    "pathway_block": "Pathway Block",
    "mobile_usage": "Mobile Usage",
    "ppe_violation": "PPE Violation",
    "fire_detection": "Fire Detection",
    "smoke_detection": "Smoke Detection",
    "object_fall": "Object Fall",
    "near_miss": "Near Miss",
    "oil_spillage": "Oil Spillage",
    "wet_floor": "Wet Floor",
    "low_visibility": "Low Visibility",
}


@dataclass
class DetectionSettings:
    pathway_block: bool = True
    mobile_usage: bool = True
    ppe_violation: bool = True
    fire_detection: bool = True
    smoke_detection: bool = True
    object_fall: bool = True
    near_miss: bool = True
    oil_spillage: bool = True
    wet_floor: bool = True
    low_visibility: bool = True
    ppe_helmet: bool = True
    ppe_vest: bool = True
    ppe_gloves: bool = False
    ppe_boots: bool = False

    @classmethod
    def from_config(cls, config: AppConfig) -> DetectionSettings:
        return cls(
            pathway_block=config.pathway_block_enabled,
            mobile_usage=config.mobile_usage_enabled,
            ppe_violation=config.ppe_enabled,
            fire_detection=config.fire_detection_enabled,
            smoke_detection=config.smoke_detection_enabled,
            object_fall=config.object_fall_enabled,
            near_miss=config.near_miss_enabled,
            oil_spillage=config.oil_spillage_enabled,
            wet_floor=config.wet_floor_enabled,
            low_visibility=config.low_visibility_enabled,
            ppe_helmet=config.is_ppe_item_enabled(PPE_ITEM_HELMET),
            ppe_vest=config.is_ppe_item_enabled(PPE_ITEM_VEST),
            ppe_gloves=config.is_ppe_item_enabled(PPE_ITEM_GLOVES),
            ppe_boots=config.is_ppe_item_enabled(PPE_ITEM_BOOTS),
        )

    def apply_to(self, config: AppConfig) -> None:
        config.pathway_block_enabled = self.pathway_block
        config.mobile_usage_enabled = self.mobile_usage
        config.ppe_enabled = self.ppe_violation
        config.fire_detection_enabled = self.fire_detection
        config.smoke_detection_enabled = self.smoke_detection
        config.object_fall_enabled = self.object_fall
        config.near_miss_enabled = self.near_miss
        config.oil_spillage_enabled = self.oil_spillage
        config.wet_floor_enabled = self.wet_floor
        config.low_visibility_enabled = self.low_visibility
        config.ppe_item_enabled = {
            PPE_ITEM_HELMET: self.ppe_helmet,
            PPE_ITEM_VEST: self.ppe_vest,
            PPE_ITEM_GLOVES: self.ppe_gloves,
            PPE_ITEM_BOOTS: self.ppe_boots,
        }

    def is_enabled(self, violation_type: str) -> bool:
        key = VIOLATION_SETTING_KEYS.get(violation_type)
        if key is None:
            return True
        return bool(getattr(self, key, True))


class DetectionSettingsStore:
    """Persists per-violation enable flags for web UI and live pipeline reload."""

    def __init__(self, config: AppConfig):
        self.path = config.resolve(SETTINGS_FILENAME)

    def load(self, config: AppConfig | None = None) -> DetectionSettings:
        defaults = DetectionSettings.from_config(config) if config else DetectionSettings()
        if not self.path.exists():
            return defaults

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return defaults

        return DetectionSettings(
            pathway_block=bool(raw.get("pathway_block", defaults.pathway_block)),
            mobile_usage=bool(raw.get("mobile_usage", defaults.mobile_usage)),
            ppe_violation=bool(raw.get("ppe_violation", defaults.ppe_violation)),
            fire_detection=bool(raw.get("fire_detection", defaults.fire_detection)),
            smoke_detection=bool(raw.get("smoke_detection", defaults.smoke_detection)),
            object_fall=bool(raw.get("object_fall", defaults.object_fall)),
            near_miss=bool(raw.get("near_miss", defaults.near_miss)),
            oil_spillage=bool(raw.get("oil_spillage", defaults.oil_spillage)),
            wet_floor=bool(raw.get("wet_floor", defaults.wet_floor)),
            low_visibility=bool(raw.get("low_visibility", defaults.low_visibility)),
            ppe_helmet=bool(raw.get("ppe_helmet", defaults.ppe_helmet)),
            ppe_vest=bool(raw.get("ppe_vest", defaults.ppe_vest)),
            ppe_gloves=bool(raw.get("ppe_gloves", defaults.ppe_gloves)),
            ppe_boots=bool(raw.get("ppe_boots", defaults.ppe_boots)),
        )

    def save(self, settings: DetectionSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(settings), indent=2) + "\n",
            encoding="utf-8",
        )

    def ensure_exists(self, config: AppConfig) -> DetectionSettings:
        settings = self.load(config)
        if not self.path.exists():
            self.save(settings)
        return settings
