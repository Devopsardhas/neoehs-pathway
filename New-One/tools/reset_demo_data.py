#!/usr/bin/env python3
"""Clear observations, captures, and demo videos for a fresh test run."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_reset import reset_demo_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear demo data and observations")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root folder",
    )
    parser.add_argument(
        "--also-zones",
        action="store_true",
        help="Also remove camera zone JSON and reference images",
    )
    args = parser.parse_args()

    result = reset_demo_data(Path(args.root).resolve(), keep_zones=not args.also_zones)
    print(f"Reset complete — {result.summary}")


if __name__ == "__main__":
    main()
