from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from src.config import AppConfig
from src.plate_format import format_plate_display, parse_and_correct_plate, plates_similar
from src.plate_ocr import read_plate
from src.vehicle_details import detect_vehicle_brand, normalize_vehicle_type


@dataclass
class GateDetection:
    plate_number: str
    vehicle_type: str
    vehicle_brand: str
    vehicle_class: str
    confidence: float
    xyxy: np.ndarray
    track_id: int | None


@dataclass
class _TrackState:
    plate_votes: dict[str, int] = field(default_factory=dict)
    vehicle_class: str = "vehicle"
    vehicle_type: str = "Unknown"
    vehicle_brand: str = "Unknown"
    best_confidence: float = 0.0
    last_xyxy: np.ndarray | None = None
    emitted: bool = False


class GateAnprMonitor:
    """Confirms plate reads across consecutive frames before emitting gate events."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._tracks: dict[int, _TrackState] = {}
        self._brand_class_names = {name.lower() for name in config.gate_brand_classes}

    def prune_tracks(self, active_track_ids: set[int]) -> None:
        stale = [track_id for track_id in self._tracks if track_id not in active_track_ids]
        for track_id in stale:
            del self._tracks[track_id]

    @staticmethod
    def _center(xyxy: np.ndarray) -> tuple[float, float]:
        return float((xyxy[0] + xyxy[2]) / 2.0), float((xyxy[1] + xyxy[3]) / 2.0)

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _nearest_vehicle(
        self,
        plate_xyxy: np.ndarray,
        vehicle_detections: sv.Detections,
        class_name_fn,
    ) -> tuple[np.ndarray | None, str, float]:
        if len(vehicle_detections) == 0:
            return None, "vehicle", 0.0

        plate_center = self._center(plate_xyxy)
        best_class = "vehicle"
        best_conf = 0.0
        best_distance = float("inf")
        best_xyxy: np.ndarray | None = None

        for idx in range(len(vehicle_detections)):
            vehicle_xyxy = vehicle_detections.xyxy[idx]
            vehicle_center = self._center(vehicle_xyxy)
            distance = self._distance(plate_center, vehicle_center)
            if distance > self.config.gate_plate_match_distance_px:
                continue
            confidence = (
                float(vehicle_detections.confidence[idx])
                if vehicle_detections.confidence is not None
                else 0.0
            )
            if distance < best_distance:
                best_distance = distance
                best_xyxy = vehicle_xyxy.copy()
                best_class = class_name_fn(int(vehicle_detections.class_id[idx]))
                best_conf = confidence

        return best_xyxy, best_class, best_conf

    def _vote_plate(self, state: _TrackState, plate_text: str) -> str:
        canonical = parse_and_correct_plate(plate_text) or plate_text.upper()
        for existing in list(state.plate_votes):
            if plates_similar(existing, canonical):
                canonical = existing
                break
        state.plate_votes[canonical] = state.plate_votes.get(canonical, 0) + 1
        return canonical

    def process_frame(
        self,
        frame: np.ndarray,
        plate_detections: sv.Detections,
        vehicle_detections: sv.Detections,
        brand_detections: sv.Detections,
        class_name_fn,
    ) -> list[GateDetection]:
        confirmed: list[GateDetection] = []
        active_track_ids: set[int] = set()

        if plate_detections.tracker_id is None:
            return confirmed

        for idx in range(len(plate_detections)):
            track_id = plate_detections.tracker_id[idx]
            if track_id is None:
                continue
            track_id = int(track_id)
            active_track_ids.add(track_id)

            xyxy = plate_detections.xyxy[idx]
            x1, y1, x2, y2 = map(int, xyxy)
            h, w = frame.shape[:2]
            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            plate_text, ocr_conf = read_plate(crop, enabled=self.config.gate_ocr_enabled)
            if not plate_text:
                continue

            plate_text = parse_and_correct_plate(plate_text) or plate_text

            vehicle_xyxy, vehicle_class, vehicle_conf = self._nearest_vehicle(
                xyxy,
                vehicle_detections,
                class_name_fn,
            )
            vehicle_type = normalize_vehicle_type(vehicle_class)
            vehicle_brand = "Unknown"
            brand_conf = 0.0
            if vehicle_xyxy is not None:
                vehicle_brand, brand_conf = detect_vehicle_brand(
                    vehicle_xyxy,
                    brand_detections,
                    class_name_fn,
                    self._brand_class_names,
                    min_confidence=self.config.gate_brand_confidence,
                )

            detection_conf = max(
                float(plate_detections.confidence[idx]) if plate_detections.confidence is not None else 0.0,
                ocr_conf,
                vehicle_conf,
                brand_conf,
            )

            state = self._tracks.setdefault(track_id, _TrackState())
            voted_plate = self._vote_plate(state, plate_text)
            state.vehicle_class = vehicle_class
            state.vehicle_type = vehicle_type
            if vehicle_brand != "Unknown":
                state.vehicle_brand = vehicle_brand
            state.best_confidence = max(state.best_confidence, detection_conf)
            state.last_xyxy = xyxy.copy()

            if state.emitted:
                continue

            if state.plate_votes[voted_plate] >= self.config.gate_min_confirm_frames:
                state.emitted = True
                confirmed.append(
                    GateDetection(
                        plate_number=voted_plate,
                        vehicle_type=state.vehicle_type,
                        vehicle_brand=state.vehicle_brand,
                        vehicle_class=state.vehicle_class,
                        confidence=state.best_confidence,
                        xyxy=xyxy.copy(),
                        track_id=track_id,
                    )
                )

        self.prune_tracks(active_track_ids)
        return confirmed

    def preview_labels(self, plate_detections: sv.Detections) -> list[str]:
        labels: list[str] = []
        if plate_detections.tracker_id is None:
            return labels
        for track_id in plate_detections.tracker_id:
            if track_id is None:
                labels.append("plate")
                continue
            state = self._tracks.get(int(track_id))
            if state and state.plate_votes:
                plate = max(state.plate_votes, key=state.plate_votes.get)
                plate_label = format_plate_display(plate)
                detail = state.vehicle_type
                if state.vehicle_brand != "Unknown":
                    detail = f"{detail} · {state.vehicle_brand}"
                labels.append(f"{plate_label} ({detail})")
            else:
                labels.append("scanning")
        return labels
