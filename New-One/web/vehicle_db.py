from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.vehicle_manager import VehicleEvent, VehicleInside, VehicleManager, VehicleStats


class VehicleDatabase:
    def __init__(self, config: AppConfig):
        self.config = config
        self.manager = VehicleManager(config)

    def stats(self) -> VehicleStats:
        return self.manager.stats()

    def list_inside(self) -> list[VehicleInside]:
        return self.manager.list_inside()

    def list_events(self, limit: int = 100, plate_number: str | None = None) -> list[VehicleEvent]:
        return self.manager.list_events(limit=limit, plate_number=plate_number)

    def get_event(self, event_id: str):
        return self.manager.get_event(event_id)

    def resolve_image_path(self, image_path: str) -> Path | None:
        path = Path(image_path)
        if path.is_file():
            return path
        candidate = self.config.project_root / image_path
        if candidate.is_file():
            return candidate
        return None

    def close(self) -> None:
        self.manager.close()
