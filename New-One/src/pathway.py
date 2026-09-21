from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, box


def normalize_polygon(geometry) -> Polygon:
    """Ensure zone geometry is a single Polygon (buffer(0) can yield MultiPolygon)."""
    if isinstance(geometry, Polygon):
        return geometry

    if isinstance(geometry, MultiPolygon):
        if not geometry.geoms:
            raise ValueError("Empty MultiPolygon in pathway config")
        return max(geometry.geoms, key=lambda part: part.area)

    polygons = [
        part for part in getattr(geometry, "geoms", [])
        if isinstance(part, Polygon)
    ]
    if polygons:
        return max(polygons, key=lambda part: part.area)

    raise ValueError(f"Unsupported pathway geometry type: {type(geometry).__name__}")


def polygon_parts(polygon: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(polygon, MultiPolygon):
        return list(polygon.geoms)
    return [polygon]


@dataclass(frozen=True)
class ZoneRef:
    """Lightweight zone label for observations (pathway zone or full camera view)."""
    id: str
    name: str


CAMERA_VIEW_NAME = "Camera View"


def zone_ref_for_person(
    xyxy: np.ndarray,
    zones: list[PathwayZone],
    camera_id: str,
    min_overlap: float,
) -> ZoneRef:
    for zone in zones:
        if zone.person_in_zone(xyxy, min_overlap):
            return ZoneRef(zone.id, zone.name)
    return ZoneRef(camera_id, CAMERA_VIEW_NAME)


def zone_ref_for_object(
    xyxy: np.ndarray,
    zones: list[PathwayZone],
    camera_id: str,
    min_overlap_ratio: float,
    min_overlap_area_px: float,
    use_foot_point: bool = True,
) -> ZoneRef:
    for zone in zones:
        if zone.overlaps_detection(
            xyxy,
            min_overlap_ratio,
            min_overlap_area_px,
            use_foot_point,
        ):
            return ZoneRef(zone.id, zone.name)
    return ZoneRef(camera_id, CAMERA_VIEW_NAME)


class ZoneType(str, Enum):
    PATHWAY = "pathway"
    MOBILE_USAGE = "mobile_usage"


ZONE_COLORS = {
    ZoneType.PATHWAY: (0, 255, 255),       # yellow
    ZoneType.MOBILE_USAGE: (255, 0, 255),  # magenta
}


@dataclass
class PathwayZone:
    id: str
    name: str
    polygon: Polygon
    zone_type: ZoneType = ZoneType.PATHWAY

    @property
    def is_mobile_allowed(self) -> bool:
        """Magenta zone — mobile phone use is permitted here."""
        return self.zone_type == ZoneType.MOBILE_USAGE

    @property
    def is_mobile_usage(self) -> bool:
        return self.is_mobile_allowed

    @property
    def counts_for_blocking(self) -> bool:
        """Mobile usage zones are inside the pathway and also block checks."""
        return True

    def overlaps_detection(
        self,
        xyxy: np.ndarray,
        min_overlap_ratio: float,
        min_overlap_area_px: float,
        use_foot_point: bool = True,
        large_object_area_px: float = 25_000.0,
        footprint_height_ratio: float = 0.35,
    ) -> bool:
        """
        True when the object's floor contact sits on the pathway.

        High-angle CCTV bboxes are tall, so a machine parked beside the walkway
        often clips the yellow zone. Only the bottom band of the box (wheels /
        pallet) is used for the occupancy test.
        """
        x1, y1, x2, y2 = map(float, xyxy)
        if x2 <= x1 or y2 <= y1:
            return False

        width = x2 - x1
        height = y2 - y1
        low_profile = height <= 48.0 or (width / max(height, 1.0) >= 2.2)
        foot_y1 = y1 if low_profile else y2 - (height * footprint_height_ratio)
        footprint = box(x1, foot_y1, x2, y2)
        if footprint.area <= 0 or not footprint.intersects(self.polygon):
            return False

        intersection = footprint.intersection(self.polygon)
        if intersection.is_empty:
            return False

        overlap_ratio = intersection.area / footprint.area
        required_area = min(min_overlap_area_px, footprint.area * 0.55)
        if overlap_ratio < min_overlap_ratio or intersection.area < required_area:
            return False

        if use_foot_point and not low_profile:
            contact = Point((x1 + x2) / 2.0, y2)
            if not self.polygon.contains(contact) and overlap_ratio < max(min_overlap_ratio, 0.35):
                # Foot is off the path and only a sliver of the base is inside.
                return False

        return True

    def intersection_area(self, xyxy: np.ndarray) -> float:
        x1, y1, x2, y2 = map(float, xyxy)
        bbox = box(x1, y1, x2, y2)
        if bbox.area <= 0 or not bbox.intersects(self.polygon):
            return 0.0
        intersection = bbox.intersection(self.polygon)
        return float(intersection.area) if not intersection.is_empty else 0.0

    def person_in_zone(self, xyxy: np.ndarray, min_body_overlap: float = 0.03) -> bool:
        """
        Relaxed check for people — center, foot, head, or small overlap counts.
        Used for mobile usage zones where strict object rules are too tight.
        """
        x1, y1, x2, y2 = map(float, xyxy)
        bbox = box(x1, y1, x2, y2)
        if bbox.area <= 0 or not bbox.intersects(self.polygon):
            return False

        for point in (
            Point((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            Point((x1 + x2) / 2.0, y2),
            Point((x1 + x2) / 2.0, y1),
        ):
            if self.polygon.contains(point):
                return True

        intersection = bbox.intersection(self.polygon)
        if intersection.is_empty:
            return False
        return (intersection.area / bbox.area) >= min_body_overlap

    def contains_point(self, x: float, y: float) -> bool:
        return self.polygon.contains(Point(x, y))

    def draw(
        self,
        frame: np.ndarray,
        fill_alpha: float = 0.0,
        thickness: int = 2,
    ) -> None:
        color = ZONE_COLORS[self.zone_type]
        label = self.name
        if self.is_mobile_allowed:
            label = f"{self.name} (mobile OK)"

        parts = []
        for part in polygon_parts(self.polygon):
            pts = np.array(part.exterior.coords[:-1], dtype=np.int32)
            if len(pts) >= 3:
                parts.append(pts)
        if not parts:
            return

        if fill_alpha > 0:
            overlay = frame.copy()
            cv2.fillPoly(overlay, parts, color)
            cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

        cv2.polylines(frame, parts, isClosed=True, color=color, thickness=thickness)
        cv2.putText(
            frame,
            label,
            tuple(parts[0][0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


class PathwayManager:
    def __init__(
        self,
        config_file: Path,
        min_overlap_ratio: float = 0.12,
        min_overlap_area_px: float = 500.0,
        use_foot_point: bool = True,
        zone_scale: tuple[float, float] | None = None,
    ):
        self.min_overlap_ratio = min_overlap_ratio
        self.min_overlap_area_px = min_overlap_area_px
        self.use_foot_point = use_foot_point
        self.zone_scale = zone_scale or (1.0, 1.0)
        self.zones: list[PathwayZone] = []
        self.load(config_file)

    def load(self, config_file: Path) -> None:
        self.zones.clear()
        if not config_file.exists():
            return

        with config_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        scale_x, scale_y = self.zone_scale
        for item in data.get("pathways", []):
            points = [
                [float(point[0]) * scale_x, float(point[1]) * scale_y]
                for point in item["points"]
            ]
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            polygon = normalize_polygon(polygon)

            zone_type = ZoneType(item.get("type", ZoneType.PATHWAY.value))
            self.zones.append(
                PathwayZone(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    polygon=polygon,
                    zone_type=zone_type,
                )
            )

    @property
    def blocking_zones(self) -> list[PathwayZone]:
        """Pathway + mobile usage zones — both checked for obstructions."""
        return [zone for zone in self.zones if zone.counts_for_blocking]

    @property
    def mobile_allowed_zones(self) -> list[PathwayZone]:
        """Magenta zones where mobile phone use is allowed."""
        return [zone for zone in self.zones if zone.is_mobile_allowed]

    @property
    def mobile_usage_zones(self) -> list[PathwayZone]:
        return self.mobile_allowed_zones

    @property
    def walkway_zones(self) -> list[PathwayZone]:
        """Yellow main pathway zones — mobile banned outside magenta sub-zones."""
        zones = [zone for zone in self.zones if zone.zone_type == ZoneType.PATHWAY]
        return zones if zones else self.blocking_zones

    def zones_for_detection(self, xyxy: np.ndarray) -> list[PathwayZone]:
        return [
            zone
            for zone in self.blocking_zones
            if zone.overlaps_detection(
                xyxy,
                self.min_overlap_ratio,
                self.min_overlap_area_px,
                self.use_foot_point,
            )
        ]

    def best_walkway_zone_for_detection(self, xyxy: np.ndarray) -> PathwayZone | None:
        best_zone: PathwayZone | None = None
        best_area = 0.0
        for zone in self.walkway_zones:
            if not zone.overlaps_detection(
                xyxy,
                self.min_overlap_ratio,
                self.min_overlap_area_px,
                self.use_foot_point,
            ):
                continue
            area = zone.intersection_area(xyxy)
            if area > best_area:
                best_area = area
                best_zone = zone
        return best_zone

    def draw_all(self, frame: np.ndarray) -> None:
        for zone in self.zones:
            zone.draw(frame)

    def draw_reference_pathways(
        self,
        frame: np.ndarray,
        highlight_zone_id: str | None = None,
    ) -> None:
        """Draw walkway polygons for evidence — thicker fill on the blocked zone."""
        for zone in self.walkway_zones:
            highlighted = highlight_zone_id is not None and zone.id == highlight_zone_id
            zone.draw(
                frame,
                fill_alpha=0.22 if highlighted else 0.12,
                thickness=3 if highlighted else 2,
            )

