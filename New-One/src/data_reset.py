from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataResetResult:
    cleared_databases: list[str] = field(default_factory=list)
    removed_captures: int = 0
    removed_demo_videos: int = 0
    removed_zone_videos: int = 0
    removed_zone_files: int = 0

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.cleared_databases:
            parts.append(
                f"cleared {len(self.cleared_databases)} database(s): "
                + ", ".join(self.cleared_databases)
            )
        if self.removed_captures:
            parts.append(f"removed {self.removed_captures} capture image(s)")
        if self.removed_demo_videos:
            parts.append(f"removed {self.removed_demo_videos} demo video(s)")
        if self.removed_zone_videos:
            parts.append(f"removed {self.removed_zone_videos} camera demo video(s)")
        if self.removed_zone_files:
            parts.append(f"removed {self.removed_zone_files} zone file(s)")
        return "; ".join(parts) if parts else "nothing to clear"


def _clear_sqlite_tables(db_path: Path) -> list[str]:
    if not db_path.is_file():
        return []

    conn = sqlite3.connect(db_path)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
        return tables
    finally:
        conn.close()


def _remove_files(folder: Path, *, keep_names: set[str] | None = None) -> int:
    if not folder.is_dir():
        return 0

    keep = keep_names or set()
    removed = 0
    for path in folder.iterdir():
        if not path.is_file() or path.name in keep:
            continue
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def reset_demo_data(project_root: Path, *, keep_zones: bool = True) -> DataResetResult:
    root = project_root.resolve()
    result = DataResetResult()

    for pattern in ("data/observations.db", "data/vehicles.db", "data/preview_*.db"):
        for db_path in root.glob(pattern):
            tables = _clear_sqlite_tables(db_path)
            if tables:
                result.cleared_databases.append(db_path.name)

    captures = root / "captures"
    if captures.is_dir():
        for path in captures.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                path.unlink(missing_ok=True)
                result.removed_captures += 1

    vehicle_captures = root / "captures" / "vehicles"
    if vehicle_captures.is_dir():
        for path in vehicle_captures.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
                result.removed_captures += 1

    result.removed_demo_videos = _remove_files(
        root / "data" / "demo_videos",
        keep_names={"README.md"},
    )
    result.removed_zone_videos = _remove_files(root / "data" / "camera_zones" / "videos")

    if not keep_zones:
        zone_root = root / "data" / "camera_zones"
        for path in zone_root.glob("*.json"):
            path.unlink(missing_ok=True)
            result.removed_zone_files += 1
        result.removed_zone_files += _remove_files(zone_root / "frames")

    return result
