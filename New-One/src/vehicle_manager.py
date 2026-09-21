from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig, CameraConfig
from src.vehicle_details import normalize_vehicle_type


@dataclass
class VehicleEvent:
    event_id: str
    visit_id: str
    plate_number: str
    direction: str
    gate_id: str
    gate_name: str
    camera_id: str
    camera_name: str
    vehicle_class: str
    vehicle_type: str
    vehicle_brand: str
    event_time: str
    confidence: float
    image_path: str
    duration_seconds: float | None = None


@dataclass
class VehicleInside:
    plate_number: str
    visit_id: str
    entry_time: str
    entry_gate_id: str
    entry_gate_name: str
    entry_camera_id: str
    vehicle_class: str
    vehicle_type: str
    vehicle_brand: str
    duration_seconds: float


@dataclass
class VehicleStats:
    inside_count: int
    entries_today: int
    exits_today: int
    avg_visit_duration_sec: float
    total_visits_today: int


class VehicleManager:
    EVENT_TABLE = "vehicle_event"
    PRESENCE_TABLE = "vehicle_presence"

    def __init__(self, config: AppConfig):
        self.config = config
        self.db_path = str(config.resolve(config.gate_vehicle_db_path))
        self.capture_dir = config.resolve(config.gate_vehicle_capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        self._recent_events: dict[tuple[str, str, str], datetime] = {}

    @contextmanager
    def _connect(self):
        self._db.row_factory = sqlite3.Row
        try:
            yield self._db
        finally:
            self._db.commit()

    def close(self) -> None:
        if hasattr(self, "_db"):
            self._db.close()

    def _init_db(self) -> None:
        self._db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.EVENT_TABLE} (
                event_id TEXT PRIMARY KEY,
                visit_id TEXT NOT NULL,
                plate_number TEXT NOT NULL,
                direction TEXT NOT NULL,
                gate_id TEXT NOT NULL,
                gate_name TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                camera_name TEXT NOT NULL,
                vehicle_class TEXT NOT NULL,
                vehicle_type TEXT NOT NULL DEFAULT 'Unknown',
                vehicle_brand TEXT NOT NULL DEFAULT 'Unknown',
                event_time TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT NOT NULL,
                duration_seconds REAL
            )
            """
        )
        self._db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.PRESENCE_TABLE} (
                plate_number TEXT PRIMARY KEY,
                visit_id TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_gate_id TEXT NOT NULL,
                entry_gate_name TEXT NOT NULL,
                entry_camera_id TEXT NOT NULL,
                vehicle_class TEXT NOT NULL,
                vehicle_type TEXT NOT NULL DEFAULT 'Unknown',
                vehicle_brand TEXT NOT NULL DEFAULT 'Unknown'
            )
            """
        )
        self._db.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.EVENT_TABLE}_plate_time
            ON {self.EVENT_TABLE} (plate_number, event_time DESC)
            """
        )
        self._db.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.EVENT_TABLE}_direction_time
            ON {self.EVENT_TABLE} (direction, event_time DESC)
            """
        )
        self._db.commit()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        event_columns = {
            row[1] for row in self._db.execute(f"PRAGMA table_info({self.EVENT_TABLE})").fetchall()
        }
        presence_columns = {
            row[1] for row in self._db.execute(f"PRAGMA table_info({self.PRESENCE_TABLE})").fetchall()
        }
        if "vehicle_type" not in event_columns:
            self._db.execute(
                f"ALTER TABLE {self.EVENT_TABLE} ADD COLUMN vehicle_type TEXT NOT NULL DEFAULT 'Unknown'"
            )
        if "vehicle_brand" not in event_columns:
            self._db.execute(
                f"ALTER TABLE {self.EVENT_TABLE} ADD COLUMN vehicle_brand TEXT NOT NULL DEFAULT 'Unknown'"
            )
        if "vehicle_type" not in presence_columns:
            self._db.execute(
                f"ALTER TABLE {self.PRESENCE_TABLE} ADD COLUMN vehicle_type TEXT NOT NULL DEFAULT 'Unknown'"
            )
        if "vehicle_brand" not in presence_columns:
            self._db.execute(
                f"ALTER TABLE {self.PRESENCE_TABLE} ADD COLUMN vehicle_brand TEXT NOT NULL DEFAULT 'Unknown'"
            )
        self._db.commit()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _is_duplicate(self, plate_number: str, direction: str, gate_id: str) -> bool:
        key = (plate_number, direction, gate_id)
        last_seen = self._recent_events.get(key)
        if last_seen is None:
            return False
        elapsed = (self._now() - last_seen).total_seconds()
        return elapsed < self.config.gate_duplicate_cooldown_sec

    def _mark_recent(self, plate_number: str, direction: str, gate_id: str) -> None:
        self._recent_events[(plate_number, direction, gate_id)] = self._now()

    def _save_capture(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        plate_number: str,
        direction: str,
        vehicle_type: str,
        vehicle_brand: str,
    ) -> str:
        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]
        pad = 20
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        annotated = frame.copy()
        color = (0, 180, 0) if direction == "in" else (0, 140, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{plate_number} {direction.upper()}"
        detail = vehicle_type
        if vehicle_brand != "Unknown":
            detail = f"{vehicle_type} · {vehicle_brand}"
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            detail,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        timestamp = self._now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plate_number}_{direction}_{timestamp}.jpg"
        path = self.capture_dir / filename
        cv2.imwrite(str(path), annotated)
        return str(path)

    def record_gate_detection(
        self,
        frame: np.ndarray,
        plate_number: str,
        vehicle_class: str,
        vehicle_type: str,
        vehicle_brand: str,
        confidence: float,
        xyxy: np.ndarray,
        camera: CameraConfig,
    ) -> VehicleEvent | None:
        if camera.role not in {"gate_in", "gate_out"}:
            return None

        direction = "in" if camera.role == "gate_in" else "out"
        if self._is_duplicate(plate_number, direction, camera.gate_id):
            return None

        resolved_type = vehicle_type or normalize_vehicle_type(vehicle_class)
        resolved_brand = vehicle_brand or "Unknown"
        now = self._now()
        event_id = str(uuid.uuid4())
        image_path = self._save_capture(
            frame,
            xyxy,
            plate_number,
            direction,
            resolved_type,
            resolved_brand,
        )

        with self._connect() as conn:
            if direction == "in":
                visit_id = str(uuid.uuid4())
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.PRESENCE_TABLE} (
                        plate_number, visit_id, entry_time, entry_gate_id,
                        entry_gate_name, entry_camera_id, vehicle_class,
                        vehicle_type, vehicle_brand
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plate_number,
                        visit_id,
                        now.isoformat(),
                        camera.gate_id,
                        camera.gate_name,
                        camera.id,
                        vehicle_class,
                        resolved_type,
                        resolved_brand,
                    ),
                )
                duration_seconds = None
            else:
                row = conn.execute(
                    f"SELECT * FROM {self.PRESENCE_TABLE} WHERE plate_number = ?",
                    (plate_number,),
                ).fetchone()
                if row is None:
                    visit_id = str(uuid.uuid4())
                    duration_seconds = None
                else:
                    visit_id = row["visit_id"]
                    entry_time = datetime.fromisoformat(row["entry_time"])
                    duration_seconds = (now - entry_time).total_seconds()
                    conn.execute(
                        f"DELETE FROM {self.PRESENCE_TABLE} WHERE plate_number = ?",
                        (plate_number,),
                    )

            conn.execute(
                f"""
                INSERT INTO {self.EVENT_TABLE} (
                    event_id, visit_id, plate_number, direction, gate_id, gate_name,
                    camera_id, camera_name, vehicle_class, vehicle_type, vehicle_brand,
                    event_time, confidence, image_path, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    visit_id,
                    plate_number,
                    direction,
                    camera.gate_id,
                    camera.gate_name,
                    camera.id,
                    camera.name,
                    vehicle_class,
                    resolved_type,
                    resolved_brand,
                    now.isoformat(),
                    confidence,
                    image_path,
                    duration_seconds,
                ),
            )

        self._mark_recent(plate_number, direction, camera.gate_id)
        return VehicleEvent(
            event_id=event_id,
            visit_id=visit_id,
            plate_number=plate_number,
            direction=direction,
            gate_id=camera.gate_id,
            gate_name=camera.gate_name,
            camera_id=camera.id,
            camera_name=camera.name,
            vehicle_class=vehicle_class,
            vehicle_type=resolved_type,
            vehicle_brand=resolved_brand,
            event_time=now.isoformat(),
            confidence=confidence,
            image_path=image_path,
            duration_seconds=duration_seconds,
        )

    def list_inside(self) -> list[VehicleInside]:
        now = self._now()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM {self.PRESENCE_TABLE}
                ORDER BY entry_time DESC
                """
            ).fetchall()

        inside: list[VehicleInside] = []
        for row in rows:
            entry_time = datetime.fromisoformat(row["entry_time"])
            inside.append(
                VehicleInside(
                    plate_number=row["plate_number"],
                    visit_id=row["visit_id"],
                    entry_time=row["entry_time"],
                    entry_gate_id=row["entry_gate_id"],
                    entry_gate_name=row["entry_gate_name"],
                    entry_camera_id=row["entry_camera_id"],
                    vehicle_class=row["vehicle_class"],
                    vehicle_type=row["vehicle_type"] if "vehicle_type" in row.keys() else normalize_vehicle_type(row["vehicle_class"]),
                    vehicle_brand=row["vehicle_brand"] if "vehicle_brand" in row.keys() else "Unknown",
                    duration_seconds=(now - entry_time).total_seconds(),
                )
            )
        return inside

    def list_events(self, limit: int = 100, plate_number: str | None = None) -> list[VehicleEvent]:
        with self._connect() as conn:
            if plate_number:
                rows = conn.execute(
                    f"""
                    SELECT * FROM {self.EVENT_TABLE}
                    WHERE plate_number = ?
                    ORDER BY event_time DESC LIMIT ?
                    """,
                    (plate_number.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT * FROM {self.EVENT_TABLE}
                    ORDER BY event_time DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: str) -> VehicleEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {self.EVENT_TABLE} WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_event(row) if row else None

    def stats(self) -> VehicleStats:
        now = self._now()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        with self._connect() as conn:
            inside_count = conn.execute(
                f"SELECT COUNT(*) FROM {self.PRESENCE_TABLE}"
            ).fetchone()[0]
            entries_today = conn.execute(
                f"""
                SELECT COUNT(*) FROM {self.EVENT_TABLE}
                WHERE direction = 'in' AND event_time >= ?
                """,
                (start_of_day,),
            ).fetchone()[0]
            exits_today = conn.execute(
                f"""
                SELECT COUNT(*) FROM {self.EVENT_TABLE}
                WHERE direction = 'out' AND event_time >= ?
                """,
                (start_of_day,),
            ).fetchone()[0]
            avg_row = conn.execute(
                f"""
                SELECT AVG(duration_seconds) FROM {self.EVENT_TABLE}
                WHERE direction = 'out' AND duration_seconds IS NOT NULL AND event_time >= ?
                """,
                (start_of_day,),
            ).fetchone()[0]

        return VehicleStats(
            inside_count=int(inside_count or 0),
            entries_today=int(entries_today or 0),
            exits_today=int(exits_today or 0),
            avg_visit_duration_sec=float(avg_row or 0.0),
            total_visits_today=int(entries_today or 0),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> VehicleEvent:
        keys = row.keys()
        vehicle_class = row["vehicle_class"]
        return VehicleEvent(
            event_id=row["event_id"],
            visit_id=row["visit_id"],
            plate_number=row["plate_number"],
            direction=row["direction"],
            gate_id=row["gate_id"],
            gate_name=row["gate_name"],
            camera_id=row["camera_id"],
            camera_name=row["camera_name"],
            vehicle_class=vehicle_class,
            vehicle_type=row["vehicle_type"] if "vehicle_type" in keys else normalize_vehicle_type(vehicle_class),
            vehicle_brand=row["vehicle_brand"] if "vehicle_brand" in keys else "Unknown",
            event_time=row["event_time"],
            confidence=row["confidence"],
            image_path=row["image_path"],
            duration_seconds=row["duration_seconds"],
        )
