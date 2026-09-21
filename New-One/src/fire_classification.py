from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import supervision as sv

if TYPE_CHECKING:
    from src.config import AppConfig


@dataclass(frozen=True)
class FireType:
    key: str
    label: str
    classes: tuple[str, ...]
    alert: bool = True
    immediate_high: bool = False


DEFAULT_FIRE_TYPES: tuple[FireType, ...] = (
    FireType(
        key="structural_fire",
        label="Structural Fire",
        classes=("fire", "flame"),
        alert=True,
        immediate_high=True,
    ),
    FireType(
        key="welding",
        label="Welding",
        classes=("welding", "welding spark", "welding arc", "welding sparks"),
        alert=True,
        immediate_high=False,
    ),
    FireType(
        key="electrical_spark",
        label="Electrical Spark",
        classes=("electrical spark", "electrical sparks", "spark", "electrical arc", "sparks"),
        alert=True,
        immediate_high=False,
    ),
)


class FireClassifier:
    """Maps YOLO class names to configured fire subtypes."""

    def __init__(self, fire_types: list[FireType]):
        self.fire_types = fire_types
        self._class_map: dict[str, FireType] = {}
        for fire_type in fire_types:
            for class_name in fire_type.classes:
                self._class_map[class_name.lower()] = fire_type

    @classmethod
    def from_config(cls, config: AppConfig) -> FireClassifier:
        return cls(config.fire_types)

    @property
    def detection_classes(self) -> list[str]:
        classes: list[str] = []
        seen: set[str] = set()
        for fire_type in self.fire_types:
            if not fire_type.alert:
                continue
            for class_name in fire_type.classes:
                lowered = class_name.lower()
                if lowered not in seen:
                    seen.add(lowered)
                    classes.append(class_name)
        return classes

    def classify(self, class_name: str) -> FireType | None:
        lowered = class_name.lower()
        if lowered in self._class_map:
            return self._class_map[lowered]

        for term, fire_type in self._class_map.items():
            if term in lowered or lowered in term:
                return fire_type
        return None

    def classify_detection(self, class_name: str) -> FireType | None:
        fire_type = self.classify(class_name)
        if fire_type is None or not fire_type.alert:
            return None
        return fire_type

    def filter_alerting_detections(
        self,
        detections: sv.Detections,
        class_name_fn,
    ) -> tuple[sv.Detections, list[str]]:
        if len(detections) == 0:
            return detections, []

        keep: list[int] = []
        subtypes: list[str] = []
        for i, class_id in enumerate(detections.class_id):
            fire_type = self.classify_detection(class_name_fn(int(class_id)))
            if fire_type is None:
                continue
            keep.append(i)
            subtypes.append(fire_type.key)

        if not keep:
            return sv.Detections.empty(), []

        return detections[keep], subtypes

    def get_type(self, key: str) -> FireType | None:
        for fire_type in self.fire_types:
            if fire_type.key == key:
                return fire_type
        return None

    def label_for(self, key: str) -> str:
        fire_type = self.get_type(key)
        return fire_type.label if fire_type else key.replace("_", " ").title()
