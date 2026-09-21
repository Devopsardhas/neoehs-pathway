from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import supervision as sv

from src.config import AppConfig, CameraConfig
from src.detector import ObjectDetector
from src.gate_anpr_monitor import GateAnprMonitor
from src.preview_gui import destroy_all as preview_destroy_all
from src.preview_gui import imshow as preview_imshow
from src.preview_gui import wait_key as preview_wait_key
from src.rtsp_capture import RTSPCapture
from src.video_capture import VideoFileCapture
from src.vehicle_manager import VehicleManager

logger = logging.getLogger(__name__)


class GatePipeline:
    """ANPR pipeline for a single gate camera (entry or exit)."""

    def __init__(
        self,
        config: AppConfig,
        camera: CameraConfig,
        video_path: str | Path | None = None,
        shared_vehicle_manager: VehicleManager | None = None,
    ):
        self.config = config.with_camera(camera)
        self.camera = camera
        self.video_path = str(video_path) if video_path else None
        if self.video_path:
            self.capture = VideoFileCapture(self.video_path)
        else:
            self.capture = RTSPCapture(
                self.config.rtsp_url,
                self.config.reconnect_delay_sec,
                self.config.max_read_failures,
            )
        self._owns_vehicle_manager = shared_vehicle_manager is None
        self.detector = ObjectDetector(self.config)
        if self.detector._world_mode:
            gate_classes = list(
                dict.fromkeys(
                    [
                        *self.config.custom_classes,
                        *self.config.gate_vehicle_classes,
                        *self.config.gate_plate_classes,
                        *self.config.gate_brand_classes,
                    ]
                )
            )
            self.detector.model.set_classes(gate_classes)
            self.detector.names = {index: name for index, name in enumerate(gate_classes)}
        self.monitor = GateAnprMonitor(self.config)
        self.vehicles = shared_vehicle_manager or VehicleManager(self.config)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.minimum_matching_threshold,
            frame_rate=self.config.frame_rate,
        )
        self.vehicle_tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.minimum_matching_threshold,
            frame_rate=self.config.frame_rate,
        )
        self.plate_box = sv.BoxAnnotator(color=sv.Color.from_hex("#00C2FF"))
        self.vehicle_box = sv.BoxAnnotator(color=sv.Color.from_hex("#FFB000"))
        self.label_annotator = sv.LabelAnnotator()
        self._vehicle_class_ids = self._resolve_class_ids(self.config.gate_vehicle_classes)
        self._plate_class_ids = self._resolve_class_ids(self.config.gate_plate_classes)
        self._brand_class_ids = self._resolve_class_ids(self.config.gate_brand_classes)

    def _resolve_class_ids(self, class_names: list[str]) -> set[int]:
        lowered = {name.lower() for name in class_names}
        return {
            class_id
            for class_id, name in self.detector.names.items()
            if name.lower() in lowered
        }

    def _filter_classes(
        self,
        detections: sv.Detections,
        class_ids: set[int],
        confidence: float,
    ) -> sv.Detections:
        if len(detections) == 0 or not class_ids:
            return sv.Detections.empty()

        mask = []
        for idx in range(len(detections)):
            class_id = int(detections.class_id[idx])
            score = (
                float(detections.confidence[idx])
                if detections.confidence is not None
                else 0.0
            )
            mask.append(class_id in class_ids and score >= confidence)
        return detections[np.array(mask)] if any(mask) else sv.Detections.empty()

    def _render_preview(
        self,
        frame: np.ndarray,
        vehicle_detections: sv.Detections,
        plate_detections: sv.Detections,
        labels: list[str] | None = None,
    ) -> np.ndarray:
        preview = frame.copy()
        if len(vehicle_detections) > 0:
            preview = self.vehicle_box.annotate(preview, vehicle_detections)
        if len(plate_detections) > 0:
            preview = self.plate_box.annotate(preview, plate_detections)
            if labels:
                preview = self.label_annotator.annotate(preview, plate_detections, labels)
        role_label = "ENTRY" if self.camera.role == "gate_in" else "EXIT"
        cv2.putText(
            preview,
            f"{self.camera.name} · {role_label}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def run(
        self,
        on_event: Callable | None = None,
        frame_callback: Callable[[np.ndarray], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        show_preview: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> int:
        frame_index = 0
        events_detected = 0
        try:
            for frame in self.capture.frames():
                if should_stop and should_stop():
                    break

                frame_index += 1
                if isinstance(self.capture, VideoFileCapture) and progress_callback:
                    progress_callback(self.capture.progress_ratio())

                if frame_index % self.config.frame_skip != 0:
                    continue

                all_detections = self.detector.detect_all(frame)
                vehicle_detections = self._filter_classes(
                    all_detections,
                    self._vehicle_class_ids,
                    self.config.gate_vehicle_confidence,
                )
                plate_detections = self._filter_classes(
                    all_detections,
                    self._plate_class_ids,
                    self.config.gate_plate_confidence,
                )
                brand_detections = self._filter_classes(
                    all_detections,
                    self._brand_class_ids,
                    self.config.gate_brand_confidence,
                )

                vehicle_detections = self.vehicle_tracker.update_with_detections(vehicle_detections)
                plate_detections = self.tracker.update_with_detections(plate_detections)

                confirmed = self.monitor.process_frame(
                    frame,
                    plate_detections,
                    vehicle_detections,
                    brand_detections,
                    self.detector.class_name,
                )

                for detection in confirmed:
                    event = self.vehicles.record_gate_detection(
                        frame=frame,
                        plate_number=detection.plate_number,
                        vehicle_class=detection.vehicle_class,
                        vehicle_type=detection.vehicle_type,
                        vehicle_brand=detection.vehicle_brand,
                        confidence=detection.confidence,
                        xyxy=detection.xyxy,
                        camera=self.camera,
                    )
                    if event is None:
                        continue

                    logger.info(
                        "Gate %s | %s | plate=%s | type=%s | brand=%s | confidence=%.2f",
                        event.direction.upper(),
                        self.camera.name,
                        event.plate_number,
                        event.vehicle_type,
                        event.vehicle_brand,
                        event.confidence,
                    )
                    if on_event:
                        on_event(event)
                    events_detected += 1

                if show_preview or frame_callback:
                    preview = self._render_preview(
                        frame,
                        vehicle_detections,
                        plate_detections,
                        self.monitor.preview_labels(plate_detections) or None,
                    )
                    if frame_callback:
                        frame_callback(preview)
                    if show_preview:
                        if preview_imshow(f"Gate Monitor · {self.camera.name}", preview):
                            if preview_wait_key(1) == ord("q"):
                                break
        finally:
            self.capture.release()
            if self._owns_vehicle_manager:
                self.vehicles.close()
            if show_preview:
                preview_destroy_all()
        return events_detected
