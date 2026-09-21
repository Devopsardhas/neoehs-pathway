#!/usr/bin/env python3
"""List records from the vision_observation table."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Query vision_observation records")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--status", default=None, help="Filter by status: active, resolved")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    db_path = config.resolve(config.db_path)

    if not db_path.exists():
        print(f"No database found at {db_path}")
        return

    query = """
        SELECT observation_id, camera_id, zone_id, zone_name, track_id, object_class,
               first_seen, last_seen, duration_seconds, severity, status, summary,
               image_path, violation_type, confidence_score
        FROM vision_observation
    """
    params: list = []
    if args.status:
        query += " WHERE status = ?"
        params.append(args.status)
    query += " ORDER BY last_seen DESC LIMIT ?"
    params.append(args.limit)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    if not rows:
        print("No records found.")
        return

    for row in rows:
        print("-" * 80)
        print(f"observation_id   : {row['observation_id']}")
        print(f"camera_id        : {row['camera_id']}")
        print(f"zone_id          : {row['zone_id']}")
        print(f"zone_name        : {row['zone_name']}")
        print(f"track_id         : {row['track_id']}")
        print(f"object_class     : {row['object_class']}")
        print(f"first_seen       : {row['first_seen']}")
        print(f"last_seen        : {row['last_seen']}")
        print(f"duration_seconds : {row['duration_seconds']:.0f}")
        print(f"severity         : {row['severity']}")
        print(f"status           : {row['status']}")
        print(f"violation_type   : {row['violation_type']}")
        print(f"confidence_score : {row['confidence_score']:.2f}")
        print(f"image_path       : {row['image_path']}")
        print(f"summary          : {row['summary']}")


if __name__ == "__main__":
    main()
