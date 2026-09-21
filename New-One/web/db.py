from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.config import AppConfig
from src.ppe_items import PPE_MISSING_LABELS
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


@dataclass
class ObservationRecord:
    observation_id: str
    camera_id: str
    zone_id: str
    zone_name: str
    track_id: int | None
    object_class: str
    first_seen: str
    last_seen: str
    duration_seconds: float
    severity: str
    status: str
    summary: str
    image_path: str
    violation_type: str
    confidence_score: float


VIOLATION_LABELS = {
    VIOLATION_PATHWAY_BLOCK: "Pathway Block",
    VIOLATION_MOBILE_USAGE: "Mobile Usage",
    VIOLATION_PPE: "PPE Violation",
    VIOLATION_FIRE: "Fire Detection",
    VIOLATION_SMOKE: "Smoke Detection",
    VIOLATION_OBJECT_FALL: "Object Fall",
    VIOLATION_NEAR_MISS: "Near Miss",
    VIOLATION_OIL_SPILLAGE: "Oil Spillage",
    VIOLATION_WET_FLOOR: "Wet Floor",
    VIOLATION_LOW_VISIBILITY: "Low Visibility",
}

OBJECT_CLASS_LABELS = {
    "structural_fire": "Structural Fire",
    "welding": "Welding",
    "electrical_spark": "Electrical Spark",
    "smoke": "Smoke",
    "oil_spillage": "Oil Spillage",
    "wet_floor": "Wet Floor",
    "low_visibility": "Low Visibility",
    **PPE_MISSING_LABELS,
    "mobile_usage": "Mobile Usage",
    "person_fall": "Person Fall",
    "forklift": "Forklift",
    "truck": "Truck",
    "vehicle": "Vehicle",
    "cart": "Cart",
    "trolley": "Trolley",
    "machine": "Machine",
    "equipment": "Equipment",
    "pathway_obstruction": "Pathway Obstruction",
}


class VisionDatabase:
    TABLE = "vision_observation"

    def __init__(self, config: AppConfig):
        self.config = config
        self.db_path = config.resolve(config.db_path)
        self.capture_dir = config.resolve(config.capture_dir)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_record(self, row: sqlite3.Row) -> ObservationRecord:
        return ObservationRecord(
            observation_id=row["observation_id"],
            camera_id=row["camera_id"],
            zone_id=row["zone_id"],
            zone_name=row["zone_name"],
            track_id=row["track_id"],
            object_class=row["object_class"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            duration_seconds=float(row["duration_seconds"]),
            severity=row["severity"],
            status=row["status"],
            summary=row["summary"],
            image_path=row["image_path"] or "",
            violation_type=row["violation_type"],
            confidence_score=float(row["confidence_score"]),
        )

    def get_observation(self, observation_id: str) -> ObservationRecord | None:
        if not self.db_path.exists():
            return None
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {self.TABLE} WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_observations(
        self,
        violation_type: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        camera_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ObservationRecord]:
        if not self.db_path.exists():
            return []

        query = f"SELECT * FROM {self.TABLE} WHERE 1=1"
        params: list = []

        if violation_type:
            query += " AND violation_type = ?"
            params.append(violation_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        query += " ORDER BY last_seen DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count_observations(
        self,
        violation_type: str | None = None,
        status: str | None = None,
        severity: str | None = None,
    ) -> int:
        if not self.db_path.exists():
            return 0

        query = f"SELECT COUNT(*) FROM {self.TABLE} WHERE 1=1"
        params: list = []

        if violation_type:
            query += " AND violation_type = ?"
            params.append(violation_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        if severity:
            query += " AND severity = ?"
            params.append(severity)

        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()[0])

    def dashboard_stats(self) -> dict:
        stats = {
            "total": 0,
            "active": 0,
            "resolved": 0,
            "by_violation": {},
            "by_severity": {"low": 0, "medium": 0, "high": 0},
            "by_camera": {},
        }
        if not self.db_path.exists():
            return stats

        with self._connect() as conn:
            stats["total"] = int(conn.execute(f"SELECT COUNT(*) FROM {self.TABLE}").fetchone()[0])
            stats["active"] = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {self.TABLE} WHERE status = 'active'"
                ).fetchone()[0]
            )
            stats["resolved"] = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {self.TABLE} WHERE status = 'resolved'"
                ).fetchone()[0]
            )

            for row in conn.execute(
                f"""
                SELECT violation_type, COUNT(*) AS count
                FROM {self.TABLE}
                GROUP BY violation_type
                """
            ):
                stats["by_violation"][row["violation_type"]] = row["count"]

            for row in conn.execute(
                f"""
                SELECT severity, COUNT(*) AS count
                FROM {self.TABLE}
                GROUP BY severity
                """
            ):
                stats["by_severity"][row["severity"]] = row["count"]

            for row in conn.execute(
                f"""
                SELECT camera_id, COUNT(*) AS count
                FROM {self.TABLE}
                GROUP BY camera_id
                """
            ):
                stats["by_camera"][row["camera_id"]] = row["count"]

        return stats

    def resolve_image_path(self, image_path: str) -> Path | None:
        if not image_path:
            return None
        path = Path(image_path)
        if path.exists():
            return path
        candidate = self.capture_dir / path.name
        if candidate.exists():
            return candidate
        project_relative = self.config.project_root / image_path
        if project_relative.exists():
            return project_relative
        return None
