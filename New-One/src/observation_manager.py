from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig
from src.fire_classification import FireClassifier
from src.pathway import PathwayManager
from src.severity import (
    STATUS_ACTIVE,
    STATUS_RESOLVED,
    VIOLATION_FIRE,
    VIOLATION_MOBILE_USAGE,
    VIOLATION_NEAR_MISS,
    VIOLATION_OBJECT_FALL,
    VIOLATION_PATHWAY_BLOCK,
    VIOLATION_PPE,
    VIOLATION_SMOKE,
    VIOLATION_OIL_SPILLAGE,
    VIOLATION_WET_FLOOR,
    VIOLATION_LOW_VISIBILITY,
    compute_fire_severity,
    compute_low_visibility_severity,
    compute_near_miss_severity,
    compute_object_fall_severity,
    compute_oil_spillage_severity,
    compute_severity,
    compute_smoke_severity,
    compute_wet_floor_severity,
)
from src.ppe_items import encode_missing_ppe
from src.summary import (
    build_fire_summary,
    build_low_visibility_summary,
    build_mobile_summary,
    build_near_miss_summary,
    build_object_fall_summary,
    build_oil_spillage_summary,
    build_ppe_summary,
    build_smoke_summary,
    build_summary,
    build_wet_floor_summary,
)


@dataclass
class VisionObservation:
    observation_id: str
    camera_id: str
    zone_id: str
    zone_name: str
    track_id: int | None
    object_class: str
    first_seen: datetime
    last_seen: datetime
    duration_seconds: float
    severity: str
    status: str
    summary: str
    image_path: str
    violation_type: str
    confidence_score: float


@dataclass
class _SpatialRecord:
    observation_id: str
    zone_id: str
    object_class: str
    center_x: float
    center_y: float


@dataclass
class _PendingPathwayBlock:
    zone_id: str
    zone_name: str
    object_class: str
    track_id: int | None
    center_x: float
    center_y: float
    xyxy: np.ndarray
    first_seen: datetime
    last_seen: datetime
    confidence_score: float
    frame: np.ndarray


class ObservationManager:
    """Persists violations to the vision_observation table."""

    TABLE_NAME = "vision_observation"

    def __init__(
        self,
        config: AppConfig,
        persist: bool = True,
        pathways: PathwayManager | None = None,
        reference_frame: np.ndarray | None = None,
    ):
        self.config = config
        self.persist = persist
        self.pathways = pathways
        self.reference_frame = reference_frame
        if persist:
            self.db_path = str(config.resolve(config.db_path))
            self.capture_dir = config.resolve(config.capture_dir)
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            preview_dir = config.resolve("data")
            preview_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str((preview_dir / f"preview_{config.camera_id}.db").resolve())
            self.capture_dir = preview_dir / "preview_captures"
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        self._last_db_update: dict[str, float] = {}
        self._spatial_index: list[_SpatialRecord] = []
        self._mobile_last_active: dict[int, datetime] = {}
        self._ppe_last_active: dict[int, datetime] = {}
        self._object_fall_last_active: dict[int, datetime] = {}
        self._near_miss_last_active: dict[tuple[int, str], datetime] = {}
        self._hazard_last_active: dict[str, datetime] = {}
        self._pending_pathway_blocks: dict[str, _PendingPathwayBlock] = {}
        self._media_origin: datetime | None = None
        self._media_now: datetime | None = None
        self._max_media_duration: float | None = None
        self._load_spatial_index()

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
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                observation_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                zone_name TEXT NOT NULL,
                track_id INTEGER,
                object_class TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                image_path TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                confidence_score REAL NOT NULL
            )
            """
        )
        self._db.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_active
            ON {self.TABLE_NAME} (status, violation_type, zone_id)
            """
        )
        self._db.commit()

    def _load_spatial_index(self) -> None:
        """In-memory spatial helpers for pathway dedup (not stored in DB)."""
        self._spatial_index.clear()

    def bind_video_timeline(
        self,
        duration_seconds: float | None = None,
        origin: datetime | None = None,
    ) -> None:
        """Use the source video clock so durations match playback, not processing time."""
        self._media_origin = origin or datetime.now(timezone.utc)
        self._media_now = self._media_origin
        if duration_seconds is not None and duration_seconds > 0:
            self._max_media_duration = duration_seconds
        else:
            self._max_media_duration = None

    def set_media_time(self, seconds: float) -> None:
        if self._media_origin is None:
            return
        clamped = max(0.0, float(seconds))
        if self._max_media_duration is not None:
            clamped = min(clamped, self._max_media_duration)
        self._media_now = self._media_origin + timedelta(seconds=clamped)

    def _now(self) -> datetime:
        if self._media_now is not None:
            return self._media_now
        return datetime.now(timezone.utc)

    @staticmethod
    def _as_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def _elapsed_seconds(
        self,
        start: datetime,
        end: datetime | None = None,
        *,
        cap_to_video: bool = True,
    ) -> float:
        finish = end if end is not None else self._now()
        elapsed = max(
            0.0,
            (self._as_aware(finish) - self._as_aware(start)).total_seconds(),
        )
        if cap_to_video and self._max_media_duration is not None:
            elapsed = min(elapsed, self._max_media_duration)
        return elapsed

    def _distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return float(np.hypot(x1 - x2, y1 - y2))

    def _register_spatial(
        self,
        observation_id: str,
        zone_id: str,
        object_class: str,
        center_x: float,
        center_y: float,
    ) -> None:
        self._spatial_index = [
            row for row in self._spatial_index if row.observation_id != observation_id
        ]
        self._spatial_index.append(
            _SpatialRecord(observation_id, zone_id, object_class, center_x, center_y)
        )

    def _find_active_pathway_match(
        self,
        zone_id: str,
        object_class: str,
        center_x: float,
        center_y: float,
        track_id: int | None,
    ) -> VisionObservation | None:
        if track_id is not None:
            match = self._fetch_active(
                violation_type=VIOLATION_PATHWAY_BLOCK,
                zone_id=zone_id,
                track_id=track_id,
            )
            if match:
                return match

        for row in self._spatial_index:
            if row.zone_id != zone_id:
                continue
            if self._distance(row.center_x, row.center_y, center_x, center_y) > self.config.spatial_match_radius_px:
                continue
            obs = self._fetch_by_id(row.observation_id)
            if obs and obs.status == STATUS_ACTIVE and obs.violation_type == VIOLATION_PATHWAY_BLOCK:
                return obs
        return None

    def _find_pending_pathway_match(
        self,
        zone_id: str,
        center_x: float,
        center_y: float,
        track_id: int | None,
    ) -> tuple[str, _PendingPathwayBlock] | tuple[None, None]:
        if track_id is not None:
            for key, pending in self._pending_pathway_blocks.items():
                if pending.zone_id == zone_id and pending.track_id == track_id:
                    return key, pending

        best_key = None
        best_pending = None
        best_distance = float(self.config.spatial_match_radius_px)
        for key, pending in self._pending_pathway_blocks.items():
            if pending.zone_id != zone_id:
                continue
            distance = self._distance(pending.center_x, pending.center_y, center_x, center_y)
            if distance <= best_distance:
                best_distance = distance
                best_key = key
                best_pending = pending
        if best_key is None or best_pending is None:
            return None, None
        return best_key, best_pending

    def _promote_pending_pathway_block(
        self,
        pending: _PendingPathwayBlock,
        now: datetime,
        duration_seconds: float,
    ) -> VisionObservation:
        observation_id = str(uuid.uuid4())
        image_path = self._save_pathway_block_capture(
            pending.frame,
            pending.xyxy,
            observation_id,
            zone_id=pending.zone_id,
        )
        summary = build_summary(
            pending.object_class,
            duration_seconds,
            pending.zone_name,
            self.config,
        )
        obs = self._insert_observation(
            observation_id=observation_id,
            zone_id=pending.zone_id,
            zone_name=pending.zone_name,
            object_class=pending.object_class,
            track_id=pending.track_id,
            first_seen=pending.first_seen,
            duration_seconds=duration_seconds,
            summary=summary,
            image_path=image_path,
            violation_type=VIOLATION_PATHWAY_BLOCK,
            confidence_score=pending.confidence_score,
        )
        self._register_spatial(
            observation_id,
            pending.zone_id,
            pending.object_class,
            pending.center_x,
            pending.center_y,
        )
        return obs

    def _find_active_mobile_match(
        self,
        track_id: int,
        center_x: float,
        center_y: float,
    ) -> VisionObservation | None:
        match = self._fetch_active_mobile_by_track(track_id)
        if match:
            return match

        for row in self._spatial_index:
            if row.object_class != "mobile_usage":
                continue
            if self._distance(row.center_x, row.center_y, center_x, center_y) <= self.config.spatial_match_radius_px:
                obs = self._fetch_by_id(row.observation_id)
                if obs and obs.status == STATUS_ACTIVE:
                    return obs
        return None

    def _fetch_active_mobile_by_track(self, track_id: int) -> VisionObservation | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ? AND track_id = ?
                ORDER BY last_seen DESC LIMIT 1
                """,
                (STATUS_ACTIVE, VIOLATION_MOBILE_USAGE, track_id),
            ).fetchone()
        return self._row_to_observation(row) if row else None

    def _fetch_active_ppe_by_track(self, track_id: int) -> VisionObservation | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ? AND track_id = ?
                ORDER BY last_seen DESC LIMIT 1
                """,
                (STATUS_ACTIVE, VIOLATION_PPE, track_id),
            ).fetchone()
        return self._row_to_observation(row) if row else None

    def _find_active_ppe_match(
        self,
        track_id: int,
        center_x: float,
        center_y: float,
    ) -> VisionObservation | None:
        match = self._fetch_active_ppe_by_track(track_id)
        if match:
            return match

        for row in self._spatial_index:
            if not row.object_class.startswith("missing_"):
                continue
            if self._distance(row.center_x, row.center_y, center_x, center_y) <= self.config.spatial_match_radius_px:
                obs = self._fetch_by_id(row.observation_id)
                if obs and obs.status == STATUS_ACTIVE and obs.violation_type == VIOLATION_PPE:
                    return obs
        return None

    def _find_active_hazard_match(
        self,
        violation_type: str,
        object_class: str,
        center_x: float,
        center_y: float,
        track_id: int | None,
    ) -> VisionObservation | None:
        if violation_type == VIOLATION_FIRE:
            return self._find_active_fire_match(center_x, center_y, track_id)

        if track_id is not None:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    f"""
                    SELECT * FROM {self.TABLE_NAME}
                    WHERE status = ? AND violation_type = ? AND track_id = ?
                    ORDER BY last_seen DESC LIMIT 1
                    """,
                    (STATUS_ACTIVE, violation_type, track_id),
                ).fetchone()
            if row:
                return self._row_to_observation(row)

        for row in self._spatial_index:
            if row.object_class != object_class:
                continue
            if self._distance(row.center_x, row.center_y, center_x, center_y) <= self.config.spatial_match_radius_px:
                obs = self._fetch_by_id(row.observation_id)
                if obs and obs.status == STATUS_ACTIVE and obs.violation_type == violation_type:
                    return obs
        return None

    def _find_active_fire_match(
        self,
        center_x: float,
        center_y: float,
        track_id: int | None,
    ) -> VisionObservation | None:
        if track_id is not None:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    f"""
                    SELECT * FROM {self.TABLE_NAME}
                    WHERE status = ? AND violation_type = ? AND track_id = ?
                    ORDER BY last_seen DESC LIMIT 1
                    """,
                    (STATUS_ACTIVE, VIOLATION_FIRE, track_id),
                ).fetchone()
            if row:
                return self._row_to_observation(row)

        merge_px = self.config.fire_merge_distance_px
        for row in self._spatial_index:
            if self._distance(row.center_x, row.center_y, center_x, center_y) <= merge_px:
                obs = self._fetch_by_id(row.observation_id)
                if obs and obs.status == STATUS_ACTIVE and obs.violation_type == VIOLATION_FIRE:
                    return obs
        return None

    def _fetch_active_by_track(
        self,
        violation_type: str,
        track_id: int,
    ) -> VisionObservation | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ? AND track_id = ?
                ORDER BY last_seen DESC LIMIT 1
                """,
                (STATUS_ACTIVE, violation_type, track_id),
            ).fetchone()
        return self._row_to_observation(row) if row else None

    def _fetch_active_near_miss(
        self,
        person_track_id: int,
        hazard_class: str,
    ) -> VisionObservation | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ? AND track_id = ? AND object_class = ?
                ORDER BY last_seen DESC LIMIT 1
                """,
                (STATUS_ACTIVE, VIOLATION_NEAR_MISS, person_track_id, hazard_class),
            ).fetchone()
        return self._row_to_observation(row) if row else None

    def _find_active_fall_match(
        self,
        track_id: int,
        center_x: float,
        center_y: float,
    ) -> VisionObservation | None:
        match = self._fetch_active_by_track(VIOLATION_OBJECT_FALL, track_id)
        if match:
            return match

        for row in self._spatial_index:
            if row.object_class not in ("object_fall", "person_fall") and not row.object_class.endswith("_fall"):
                continue
            if self._distance(row.center_x, row.center_y, center_x, center_y) <= self.config.spatial_match_radius_px:
                obs = self._fetch_by_id(row.observation_id)
                if obs and obs.status == STATUS_ACTIVE and obs.violation_type == VIOLATION_OBJECT_FALL:
                    return obs
        return None

    def _find_active_near_miss_match(
        self,
        person_track_id: int,
        hazard_class: str,
        center_x: float,
        center_y: float,
    ) -> VisionObservation | None:
        match = self._fetch_active_near_miss(person_track_id, hazard_class)
        if match:
            return match

        for row in self._spatial_index:
            if row.object_class != hazard_class:
                continue
            if self._distance(row.center_x, row.center_y, center_x, center_y) <= self.config.spatial_match_radius_px:
                obs = self._fetch_by_id(row.observation_id)
                if obs and obs.status == STATUS_ACTIVE and obs.violation_type == VIOLATION_NEAR_MISS:
                    return obs
        return None

    def _fetch_by_id(self, observation_id: str) -> VisionObservation | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"SELECT * FROM {self.TABLE_NAME} WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        return self._row_to_observation(row) if row else None

    def _fetch_active(
        self,
        violation_type: str,
        zone_id: str,
        track_id: int | None = None,
        object_class: str | None = None,
    ) -> VisionObservation | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if track_id is not None:
                row = conn.execute(
                    f"""
                    SELECT * FROM {self.TABLE_NAME}
                    WHERE status = ? AND violation_type = ? AND zone_id = ? AND track_id = ?
                    ORDER BY last_seen DESC LIMIT 1
                    """,
                    (STATUS_ACTIVE, violation_type, zone_id, track_id),
                ).fetchone()
            elif object_class is not None:
                row = conn.execute(
                    f"""
                    SELECT * FROM {self.TABLE_NAME}
                    WHERE status = ? AND violation_type = ? AND zone_id = ? AND object_class = ?
                    ORDER BY last_seen DESC LIMIT 1
                    """,
                    (STATUS_ACTIVE, violation_type, zone_id, object_class),
                ).fetchone()
            else:
                return None
        return self._row_to_observation(row) if row else None

    def _row_to_observation(self, row: sqlite3.Row) -> VisionObservation:
        return VisionObservation(
            observation_id=row["observation_id"],
            camera_id=row["camera_id"],
            zone_id=row["zone_id"],
            zone_name=row["zone_name"],
            track_id=row["track_id"],
            object_class=row["object_class"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            duration_seconds=row["duration_seconds"],
            severity=row["severity"],
            status=row["status"],
            summary=row["summary"],
            image_path=row["image_path"],
            violation_type=row["violation_type"],
            confidence_score=row["confidence_score"],
        )

    def _padded_bbox(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        pad: int = 20,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return x1, y1, x2, y2

    def _resize_reference_to_frame(self, frame: np.ndarray) -> np.ndarray | None:
        if self.reference_frame is None:
            return None
        height, width = frame.shape[:2]
        reference = self.reference_frame
        if reference.shape[0] != height or reference.shape[1] != width:
            reference = cv2.resize(
                reference,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        return reference

    def _draw_banner(
        self,
        frame: np.ndarray,
        title: str,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        panel = frame.copy()
        height, width = panel.shape[:2]
        bar_h = max(36, height // 22)
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_h), color, -1)
        cv2.addWeighted(overlay, 0.82, panel, 0.18, 0, panel)
        cv2.putText(
            panel,
            title,
            (12, bar_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return panel

    def _overlay_reference_pathways(
        self,
        frame: np.ndarray,
        zone_id: str | None = None,
    ) -> None:
        if self.pathways is None:
            return
        self.pathways.draw_reference_pathways(frame, highlight_zone_id=zone_id)

    def _compose_pathway_block_evidence(
        self,
        live_frame: np.ndarray,
        xyxy: np.ndarray,
        zone_id: str | None = None,
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        live = live_frame.copy()
        self._overlay_reference_pathways(live, zone_id)
        x1, y1, x2, y2 = self._padded_bbox(live, xyxy)
        cv2.rectangle(live, (x1, y1), (x2, y2), color, 2)

        reference = self._resize_reference_to_frame(live_frame)
        if reference is None:
            return self._draw_banner(live, "Pathway Block + Reference Pathway", (0, 140, 180))

        reference_panel = reference.copy()
        self._overlay_reference_pathways(reference_panel, zone_id)
        left = self._draw_banner(reference_panel, "Reference Pathway", (0, 140, 180))
        right = self._draw_banner(live, "Detected Block", (0, 0, 180))
        gap = np.full((left.shape[0], 8, 3), (36, 36, 36), dtype=np.uint8)
        return np.hstack([left, gap, right])

    def _save_capture(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        observation_id: str,
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> str:
        if not self.persist:
            return ""
        x1, y1, x2, y2 = self._padded_bbox(frame, xyxy)

        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        timestamp = self._now().strftime("%Y%m%d_%H%M%S")
        full_path = self.capture_dir / f"{observation_id}_{timestamp}_full.jpg"
        crop_path = self.capture_dir / f"{observation_id}_{timestamp}_crop.jpg"
        crop = frame[y1:y2, x1:x2].copy()

        cv2.imwrite(str(full_path), annotated)
        cv2.imwrite(str(crop_path), crop)
        return str(full_path)

    def _save_pathway_block_capture(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        observation_id: str,
        zone_id: str | None = None,
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> str:
        if not self.persist:
            return ""
        x1, y1, x2, y2 = self._padded_bbox(frame, xyxy)
        annotated = self._compose_pathway_block_evidence(frame, xyxy, zone_id, color)

        timestamp = self._now().strftime("%Y%m%d_%H%M%S")
        full_path = self.capture_dir / f"{observation_id}_{timestamp}_full.jpg"
        crop_path = self.capture_dir / f"{observation_id}_{timestamp}_crop.jpg"
        crop = frame[y1:y2, x1:x2].copy()

        cv2.imwrite(str(full_path), annotated)
        cv2.imwrite(str(crop_path), crop)
        return str(full_path)

    def _insert_observation(
        self,
        observation_id: str,
        zone_id: str,
        zone_name: str,
        object_class: str,
        track_id: int | None,
        first_seen: datetime,
        duration_seconds: float,
        summary: str,
        image_path: str,
        violation_type: str,
        confidence_score: float,
        severity: str | None = None,
    ) -> VisionObservation:
        resolved_severity = severity if severity is not None else compute_severity(duration_seconds, self.config)
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (
                    observation_id, camera_id, zone_id, zone_name, track_id,
                    object_class, first_seen, last_seen, duration_seconds,
                    severity, status, summary, image_path, violation_type,
                    confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    self.config.camera_id,
                    zone_id,
                    zone_name,
                    track_id,
                    object_class,
                    first_seen.isoformat(),
                    first_seen.isoformat(),
                    duration_seconds,
                    resolved_severity,
                    STATUS_ACTIVE,
                    summary,
                    image_path,
                    violation_type,
                    confidence_score,
                ),
            )

        return VisionObservation(
            observation_id=observation_id,
            camera_id=self.config.camera_id,
            zone_id=zone_id,
            zone_name=zone_name,
            track_id=track_id,
            object_class=object_class,
            first_seen=first_seen,
            last_seen=first_seen,
            duration_seconds=duration_seconds,
            severity=resolved_severity,
            status=STATUS_ACTIVE,
            summary=summary,
            image_path=image_path,
            violation_type=violation_type,
            confidence_score=confidence_score,
        )

    def process_pathway_block(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        object_class: str,
        zone_id: str,
        zone_name: str,
        confidence_score: float = 0.0,
        track_id: int | None = None,
    ) -> tuple[VisionObservation | None, bool]:
        center_x = float((xyxy[0] + xyxy[2]) / 2.0)
        center_y = float((xyxy[1] + xyxy[3]) / 2.0)
        now = self._now()
        min_duration = self.config.pathway_block_min_duration_sec

        existing = self._find_active_pathway_match(
            zone_id, object_class, center_x, center_y, track_id
        )

        if existing is not None:
            self._register_spatial(existing.observation_id, zone_id, object_class, center_x, center_y)
            updated = self._update_existing(
                existing,
                now,
                confidence_score,
                summary_builder=lambda duration: build_summary(
                    object_class, duration, zone_name, self.config
                ),
            )
            return updated, False

        pending_key, pending = self._find_pending_pathway_match(
            zone_id, center_x, center_y, track_id
        )
        if pending is not None:
            pending.last_seen = now
            pending.center_x = center_x
            pending.center_y = center_y
            pending.confidence_score = max(pending.confidence_score, confidence_score)
            pending.frame = frame.copy()
            pending.xyxy = xyxy.copy()
            duration_seconds = self._elapsed_seconds(pending.first_seen, now)
            if duration_seconds < min_duration:
                return None, False

            promoted = self._promote_pending_pathway_block(pending, now, duration_seconds)
            if pending_key is not None:
                self._pending_pathway_blocks.pop(pending_key, None)
            return promoted, True

        self._pending_pathway_blocks[f"{zone_id}:{track_id or f'{int(center_x)}:{int(center_y)}'}"] = (
            _PendingPathwayBlock(
                zone_id=zone_id,
                zone_name=zone_name,
                object_class=object_class,
                track_id=track_id,
                center_x=center_x,
                center_y=center_y,
                xyxy=xyxy.copy(),
                first_seen=now,
                last_seen=now,
                confidence_score=confidence_score,
                frame=frame.copy(),
            )
        )
        return None, False

    def process_mobile_usage(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        zone_id: str,
        zone_name: str,
        track_id: int,
        confidence_score: float = 0.0,
    ) -> VisionObservation:
        center_x = float((xyxy[0] + xyxy[2]) / 2.0)
        center_y = float((xyxy[1] + xyxy[3]) / 2.0)
        now = self._now()
        object_class = "mobile_usage"

        existing = self._find_active_mobile_match(track_id, center_x, center_y)

        if existing is None:
            observation_id = str(uuid.uuid4())
            image_path = self._save_capture(frame, xyxy, observation_id, color=(255, 0, 255))
            summary = build_mobile_summary(0.0, zone_name, self.config)
            obs = self._insert_observation(
                observation_id=observation_id,
                zone_id=zone_id,
                zone_name=zone_name,
                object_class=object_class,
                track_id=track_id,
                first_seen=now,
                duration_seconds=0.0,
                summary=summary,
                image_path=image_path,
                violation_type=VIOLATION_MOBILE_USAGE,
                confidence_score=confidence_score,
            )
            self._register_spatial(observation_id, zone_id, object_class, center_x, center_y)
            self._mobile_last_active[track_id] = now
            return obs

        self._register_spatial(existing.observation_id, zone_id, object_class, center_x, center_y)
        self._mobile_last_active[track_id] = now
        if existing.track_id != track_id:
            existing.track_id = track_id
        return self._update_existing(
            existing,
            now,
            confidence_score,
            zone_id=zone_id,
            zone_name=zone_name,
            track_id=track_id,
            summary_builder=lambda duration: build_mobile_summary(duration, zone_name, self.config),
        )

    def process_ppe_violation(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        zone_id: str,
        zone_name: str,
        track_id: int,
        missing_items: list[str],
        confidence_score: float = 0.0,
    ) -> VisionObservation:
        center_x = float((xyxy[0] + xyxy[2]) / 2.0)
        center_y = float((xyxy[1] + xyxy[3]) / 2.0)
        now = self._now()
        object_class = encode_missing_ppe(missing_items)

        existing = self._find_active_ppe_match(track_id, center_x, center_y)

        if existing is None:
            observation_id = str(uuid.uuid4())
            image_path = self._save_capture(frame, xyxy, observation_id, color=(0, 165, 255))
            summary = build_ppe_summary(missing_items, 0.0, zone_name, self.config)
            obs = self._insert_observation(
                observation_id=observation_id,
                zone_id=zone_id,
                zone_name=zone_name,
                object_class=object_class,
                track_id=track_id,
                first_seen=now,
                duration_seconds=0.0,
                summary=summary,
                image_path=image_path,
                violation_type=VIOLATION_PPE,
                confidence_score=confidence_score,
            )
            self._register_spatial(observation_id, zone_id, object_class, center_x, center_y)
            self._ppe_last_active[track_id] = now
            return obs

        self._register_spatial(existing.observation_id, zone_id, object_class, center_x, center_y)
        self._ppe_last_active[track_id] = now
        if existing.track_id != track_id:
            existing.track_id = track_id
        return self._update_existing(
            existing,
            now,
            confidence_score,
            zone_id=zone_id,
            zone_name=zone_name,
            track_id=track_id,
            object_class=object_class,
            summary_builder=lambda duration: build_ppe_summary(
                missing_items, duration, zone_name, self.config
            ),
        )

    def process_object_fall(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        zone_id: str,
        zone_name: str,
        track_id: int,
        object_class: str,
        fall_kind: str,
        confidence_score: float = 0.0,
    ) -> VisionObservation:
        center_x = float((xyxy[0] + xyxy[2]) / 2.0)
        center_y = float((xyxy[1] + xyxy[3]) / 2.0)
        now = self._now()
        stored_class = fall_kind if fall_kind == "person_fall" else object_class

        existing = self._find_active_fall_match(track_id, center_x, center_y)

        if existing is None:
            observation_id = str(uuid.uuid4())
            image_path = self._save_capture(frame, xyxy, observation_id, color=(0, 128, 255))
            summary = build_object_fall_summary(stored_class, fall_kind, 0.0, zone_name, self.config)
            obs = self._insert_observation(
                observation_id=observation_id,
                zone_id=zone_id,
                zone_name=zone_name,
                object_class=stored_class,
                track_id=track_id,
                first_seen=now,
                duration_seconds=0.0,
                summary=summary,
                image_path=image_path,
                violation_type=VIOLATION_OBJECT_FALL,
                confidence_score=confidence_score,
                severity=compute_object_fall_severity(0.0, self.config),
            )
            self._register_spatial(observation_id, zone_id, stored_class, center_x, center_y)
            self._object_fall_last_active[track_id] = now
            return obs

        self._register_spatial(existing.observation_id, zone_id, stored_class, center_x, center_y)
        self._object_fall_last_active[track_id] = now
        return self._update_existing(
            existing,
            now,
            confidence_score,
            zone_id=zone_id,
            zone_name=zone_name,
            track_id=track_id,
            severity_override=compute_object_fall_severity(
                self._elapsed_seconds(existing.first_seen, now), self.config
            ),
            summary_builder=lambda duration: build_object_fall_summary(
                stored_class, fall_kind, duration, zone_name, self.config
            ),
        )

    def process_near_miss(
        self,
        frame: np.ndarray,
        person_xyxy: np.ndarray,
        zone_id: str,
        zone_name: str,
        person_track_id: int,
        hazard_class: str,
        confidence_score: float = 0.0,
    ) -> VisionObservation:
        center_x = float((person_xyxy[0] + person_xyxy[2]) / 2.0)
        center_y = float((person_xyxy[1] + person_xyxy[3]) / 2.0)
        now = self._now()
        active_key = (person_track_id, hazard_class)

        existing = self._find_active_near_miss_match(
            person_track_id, hazard_class, center_x, center_y
        )

        if existing is None:
            observation_id = str(uuid.uuid4())
            image_path = self._save_capture(frame, person_xyxy, observation_id, color=(255, 128, 0))
            summary = build_near_miss_summary(hazard_class, 0.0, zone_name, self.config)
            obs = self._insert_observation(
                observation_id=observation_id,
                zone_id=zone_id,
                zone_name=zone_name,
                object_class=hazard_class,
                track_id=person_track_id,
                first_seen=now,
                duration_seconds=0.0,
                summary=summary,
                image_path=image_path,
                violation_type=VIOLATION_NEAR_MISS,
                confidence_score=confidence_score,
                severity=compute_near_miss_severity(0.0, self.config),
            )
            self._register_spatial(observation_id, zone_id, hazard_class, center_x, center_y)
            self._near_miss_last_active[active_key] = now
            return obs

        self._register_spatial(existing.observation_id, zone_id, hazard_class, center_x, center_y)
        self._near_miss_last_active[active_key] = now
        return self._update_existing(
            existing,
            now,
            confidence_score,
            zone_id=zone_id,
            zone_name=zone_name,
            track_id=person_track_id,
            severity_override=compute_near_miss_severity(
                self._elapsed_seconds(existing.first_seen, now), self.config
            ),
            summary_builder=lambda duration: build_near_miss_summary(
                hazard_class, duration, zone_name, self.config
            ),
        )

    def process_hazard_detection(
        self,
        frame: np.ndarray,
        xyxy: np.ndarray,
        violation_type: str,
        object_class: str,
        confidence_score: float = 0.0,
        track_id: int | None = None,
    ) -> VisionObservation:
        center_x = float((xyxy[0] + xyxy[2]) / 2.0)
        center_y = float((xyxy[1] + xyxy[3]) / 2.0)
        now = self._now()
        zone_id = self.config.camera_id
        zone_name = "Camera View"

        existing = self._find_active_hazard_match(
            violation_type, object_class, center_x, center_y, track_id
        )

        if violation_type == VIOLATION_FIRE:
            fire_type_label = FireClassifier.from_config(self.config).label_for(object_class)
            severity_fn = lambda duration, key=object_class, label=fire_type_label: compute_fire_severity(
                duration, self.config, key
            )
            summary_fn = lambda duration, label=fire_type_label: build_fire_summary(
                duration, label, self.config
            )
            color = (0, 0, 255) if object_class == "structural_fire" else (0, 140, 255)
        elif violation_type == VIOLATION_SMOKE:
            severity_fn = lambda duration: compute_smoke_severity(duration, self.config)
            summary_fn = lambda duration: build_smoke_summary(duration, self.config)
            color = (128, 128, 128)
        elif violation_type == VIOLATION_OIL_SPILLAGE:
            severity_fn = lambda duration: compute_oil_spillage_severity(duration, self.config)
            summary_fn = lambda duration: build_oil_spillage_summary(duration, self.config)
            color = (0, 165, 255)
        elif violation_type == VIOLATION_WET_FLOOR:
            severity_fn = lambda duration: compute_wet_floor_severity(duration, self.config)
            summary_fn = lambda duration: build_wet_floor_summary(duration, self.config)
            color = (255, 200, 0)
        elif violation_type == VIOLATION_LOW_VISIBILITY:
            severity_fn = lambda duration: compute_low_visibility_severity(duration, self.config)
            summary_fn = lambda duration: build_low_visibility_summary(duration, self.config)
            color = (80, 80, 80)
        else:
            severity_fn = lambda duration: compute_smoke_severity(duration, self.config)
            summary_fn = lambda duration: build_smoke_summary(duration, self.config)
            color = (128, 128, 128)

        if existing is None:
            observation_id = str(uuid.uuid4())
            image_path = self._save_capture(frame, xyxy, observation_id, color=color)
            summary = summary_fn(0.0)
            obs = self._insert_observation(
                observation_id=observation_id,
                zone_id=zone_id,
                zone_name=zone_name,
                object_class=object_class,
                track_id=track_id,
                first_seen=now,
                duration_seconds=0.0,
                summary=summary,
                image_path=image_path,
                violation_type=violation_type,
                confidence_score=confidence_score,
                severity=severity_fn(0.0),
            )
            self._register_spatial(observation_id, zone_id, object_class, center_x, center_y)
            self._hazard_last_active[obs.observation_id] = now
            return obs

        self._register_spatial(existing.observation_id, zone_id, object_class, center_x, center_y)
        self._hazard_last_active[existing.observation_id] = now
        update_track_id = track_id if track_id is not None else existing.track_id
        return self._update_existing(
            existing,
            now,
            confidence_score,
            zone_id=zone_id,
            zone_name=zone_name,
            track_id=update_track_id,
            severity_override=severity_fn(self._elapsed_seconds(existing.first_seen, now)),
            summary_builder=lambda duration: summary_fn(duration),
        )

    def _update_existing(
        self,
        existing: VisionObservation,
        now: datetime,
        confidence_score: float,
        summary_builder,
        zone_id: str | None = None,
        zone_name: str | None = None,
        track_id: int | None = None,
        object_class: str | None = None,
        severity_override: str | None = None,
    ) -> VisionObservation:
        duration_seconds = self._elapsed_seconds(existing.first_seen, now)
        last_update = self._last_db_update.get(existing.observation_id, 0.0)
        if duration_seconds - last_update < self.config.min_duration_for_update_sec:
            return existing

        summary = summary_builder(duration_seconds)
        severity = severity_override if severity_override is not None else compute_severity(duration_seconds, self.config)
        merged_confidence = max(existing.confidence_score, confidence_score)
        self._last_db_update[existing.observation_id] = duration_seconds

        update_zone_id = zone_id if zone_id is not None else existing.zone_id
        update_zone_name = zone_name if zone_name is not None else existing.zone_name
        update_track_id = track_id if track_id is not None else existing.track_id
        update_object_class = object_class if object_class is not None else existing.object_class

        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE {self.TABLE_NAME}
                SET last_seen = ?, duration_seconds = ?, severity = ?,
                    summary = ?, confidence_score = ?, zone_id = ?,
                    zone_name = ?, track_id = ?, object_class = ?
                WHERE observation_id = ?
                """,
                (
                    now.isoformat(),
                    duration_seconds,
                    severity,
                    summary,
                    merged_confidence,
                    update_zone_id,
                    update_zone_name,
                    update_track_id,
                    update_object_class,
                    existing.observation_id,
                ),
            )

        existing.last_seen = now
        existing.duration_seconds = duration_seconds
        existing.severity = severity
        existing.summary = summary
        existing.confidence_score = merged_confidence
        existing.zone_id = update_zone_id
        existing.zone_name = update_zone_name
        existing.track_id = update_track_id
        existing.object_class = update_object_class
        return existing

    def resolve_stale_pathway_blocks(
        self,
        active_keys: set[tuple[str, str, float, float]],
    ) -> None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ?
                """,
                (STATUS_ACTIVE, VIOLATION_PATHWAY_BLOCK),
            ).fetchall()

            for row in rows:
                obs = self._row_to_observation(row)
                spatial = next(
                    (s for s in self._spatial_index if s.observation_id == obs.observation_id),
                    None,
                )
                if spatial is None:
                    continue
                matched = any(
                    obs.zone_id == zone_id
                    and self._distance(spatial.center_x, spatial.center_y, cx, cy)
                    <= self.config.spatial_match_radius_px
                    for zone_id, _object_class, cx, cy in active_keys
                )
                if not matched:
                    conn.execute(
                        f"UPDATE {self.TABLE_NAME} SET status = ? WHERE observation_id = ?",
                        (STATUS_RESOLVED, obs.observation_id),
                    )
                    self._spatial_index = [
                        s for s in self._spatial_index if s.observation_id != obs.observation_id
                    ]

        stale_pending_keys: list[str] = []
        for key, pending in self._pending_pathway_blocks.items():
            matched = any(
                pending.zone_id == zone_id
                and self._distance(pending.center_x, pending.center_y, cx, cy)
                <= self.config.spatial_match_radius_px
                for zone_id, _object_class, cx, cy in active_keys
            )
            if not matched:
                stale_pending_keys.append(key)
        for key in stale_pending_keys:
            self._pending_pathway_blocks.pop(key, None)

    def resolve_stale_mobile_usage(self, active_track_ids: set[int]) -> None:
        now = self._now()
        grace = self.config.mobile_resolve_grace_sec

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ?
                """,
                (STATUS_ACTIVE, VIOLATION_MOBILE_USAGE),
            ).fetchall()

            for row in rows:
                obs = self._row_to_observation(row)
                if obs.track_id is None:
                    continue

                if obs.track_id in active_track_ids:
                    self._mobile_last_active[obs.track_id] = now
                    continue

                last_active = self._mobile_last_active.get(obs.track_id, obs.last_seen)
                if self._elapsed_seconds(last_active, now, cap_to_video=False) < grace:
                    continue

                conn.execute(
                    f"UPDATE {self.TABLE_NAME} SET status = ? WHERE observation_id = ?",
                    (STATUS_RESOLVED, obs.observation_id),
                )
                self._spatial_index = [
                    s for s in self._spatial_index if s.observation_id != obs.observation_id
                ]
                self._mobile_last_active.pop(obs.track_id, None)

    def resolve_stale_object_falls(self, active_track_ids: set[int]) -> None:
        now = self._now()
        grace = self.config.object_fall_resolve_grace_sec

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ?
                """,
                (STATUS_ACTIVE, VIOLATION_OBJECT_FALL),
            ).fetchall()

            for row in rows:
                obs = self._row_to_observation(row)
                if obs.track_id is None:
                    continue

                if obs.track_id in active_track_ids:
                    self._object_fall_last_active[obs.track_id] = now
                    continue

                last_active = self._object_fall_last_active.get(obs.track_id, obs.last_seen)
                if self._elapsed_seconds(last_active, now, cap_to_video=False) < grace:
                    continue

                conn.execute(
                    f"UPDATE {self.TABLE_NAME} SET status = ? WHERE observation_id = ?",
                    (STATUS_RESOLVED, obs.observation_id),
                )
                self._spatial_index = [
                    s for s in self._spatial_index if s.observation_id != obs.observation_id
                ]
                self._object_fall_last_active.pop(obs.track_id, None)

    def resolve_stale_near_miss(self, active_keys: set[tuple[int, str]]) -> None:
        now = self._now()
        grace = self.config.near_miss_resolve_grace_sec

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ?
                """,
                (STATUS_ACTIVE, VIOLATION_NEAR_MISS),
            ).fetchall()

            for row in rows:
                obs = self._row_to_observation(row)
                if obs.track_id is None:
                    continue

                key = (obs.track_id, obs.object_class)
                if key in active_keys:
                    self._near_miss_last_active[key] = now
                    continue

                last_active = self._near_miss_last_active.get(key, obs.last_seen)
                if self._elapsed_seconds(last_active, now, cap_to_video=False) < grace:
                    continue

                conn.execute(
                    f"UPDATE {self.TABLE_NAME} SET status = ? WHERE observation_id = ?",
                    (STATUS_RESOLVED, obs.observation_id),
                )
                self._spatial_index = [
                    s for s in self._spatial_index if s.observation_id != obs.observation_id
                ]
                self._near_miss_last_active.pop(key, None)

    def resolve_stale_ppe_violations(self, active_track_ids: set[int]) -> None:
        now = self._now()
        grace = self.config.ppe_resolve_grace_sec

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type = ?
                """,
                (STATUS_ACTIVE, VIOLATION_PPE),
            ).fetchall()

            for row in rows:
                obs = self._row_to_observation(row)
                if obs.track_id is None:
                    continue

                if obs.track_id in active_track_ids:
                    self._ppe_last_active[obs.track_id] = now
                    continue

                last_active = self._ppe_last_active.get(obs.track_id, obs.last_seen)
                if self._elapsed_seconds(last_active, now, cap_to_video=False) < grace:
                    continue

                conn.execute(
                    f"UPDATE {self.TABLE_NAME} SET status = ? WHERE observation_id = ?",
                    (STATUS_RESOLVED, obs.observation_id),
                )
                self._spatial_index = [
                    s for s in self._spatial_index if s.observation_id != obs.observation_id
                ]
                self._ppe_last_active.pop(obs.track_id, None)

    def resolve_stale_hazard_detections(
        self,
        active_spatial: set[tuple[str, str, float, float]],
    ) -> None:
        now = self._now()
        grace_by_type = {
            VIOLATION_FIRE: self.config.fire_smoke_resolve_grace_sec,
            VIOLATION_SMOKE: self.config.fire_smoke_resolve_grace_sec,
            VIOLATION_OIL_SPILLAGE: self.config.floor_hazard_resolve_grace_sec,
            VIOLATION_WET_FLOOR: self.config.floor_hazard_resolve_grace_sec,
            VIOLATION_LOW_VISIBILITY: self.config.low_visibility_resolve_grace_sec,
        }

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ", ".join("?" for _ in grace_by_type)
            rows = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ? AND violation_type IN ({placeholders})
                """,
                (STATUS_ACTIVE, *grace_by_type.keys()),
            ).fetchall()

            for row in rows:
                obs = self._row_to_observation(row)
                spatial = next(
                    (s for s in self._spatial_index if s.observation_id == obs.observation_id),
                    None,
                )
                if spatial is None:
                    continue

                matched = any(
                    obs.violation_type == violation_type
                    and obs.object_class == object_class
                    and self._distance(spatial.center_x, spatial.center_y, cx, cy)
                    <= self.config.spatial_match_radius_px
                    for violation_type, object_class, cx, cy in active_spatial
                )
                if matched:
                    self._hazard_last_active[obs.observation_id] = now
                    continue

                grace = grace_by_type.get(obs.violation_type, 5.0)
                last_active = self._hazard_last_active.get(obs.observation_id, obs.last_seen)
                if self._elapsed_seconds(last_active, now, cap_to_video=False) < grace:
                    continue

                conn.execute(
                    f"UPDATE {self.TABLE_NAME} SET status = ? WHERE observation_id = ?",
                    (STATUS_RESOLVED, obs.observation_id),
                )
                self._spatial_index = [
                    s for s in self._spatial_index if s.observation_id != obs.observation_id
                ]
                self._hazard_last_active.pop(obs.observation_id, None)

    def list_active(self) -> list[VisionObservation]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM {self.TABLE_NAME}
                WHERE status = ?
                ORDER BY last_seen DESC
                """,
                (STATUS_ACTIVE,),
            ).fetchall()
        return [self._row_to_observation(row) for row in rows]
