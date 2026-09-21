from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import supervision as sv

from src.config import AppConfig
from src.detection_settings import DetectionSettingsStore
from src.detector import ObjectDetector
from src.fire_classification import FireClassifier
from src.fire_smoke_cv import (
    detect_fire_regions,
    detect_smoke_regions,
    merge_detections,
)
from src.fire_smoke_monitor import FireSmokeMonitor
from src.floor_hazard_monitor import FloorHazardMonitor
from src.low_visibility_monitor import LowVisibilityMonitor
from src.mobile_usage import MobileUsageMonitor
from src.near_miss_monitor import NearMissMonitor
from src.object_fall_monitor import FALL_KIND_OBJECT, FALL_KIND_PERSON, ObjectFallMonitor
from src.observation_manager import ObservationManager
from src.pathway import PathwayManager
from src.pathway_obstruction_cv import PathwayObstructionMonitor
from src.ppe_monitor import PPEMonitor
from src.preview_gui import destroy_all as preview_destroy_all
from src.preview_gui import imshow as preview_imshow
from src.preview_gui import wait_key as preview_wait_key
from src.rtsp_capture import RTSPCapture
from src.static_filter import StaticObjectFilter
from src.video_capture import VideoFileCapture

logger = logging.getLogger(__name__)


class PathwayMonitorPipeline:
    def __init__(
        self,
        config: AppConfig,
        show_preview: bool = False,
        persist_observations: bool = True,
        video_path: str | Path | None = None,
        lock_settings: bool = False,
        pathway_config_override: str | Path | None = None,
        zone_scale: tuple[float, float] | None = None,
        camera_id_override: str | None = None,
        reference_frame: np.ndarray | None = None,
    ):
        self.config = config
        self.show_preview = show_preview
        self.persist_observations = persist_observations
        self.video_path = str(video_path) if video_path else None
        if camera_id_override:
            self.config.camera_id = camera_id_override

        if self.video_path:
            self.capture = VideoFileCapture(
                self.video_path,
                fallback_fps=float(config.frame_rate),
            )
        else:
            self.capture = RTSPCapture(
                config.rtsp_url,
                config.reconnect_delay_sec,
                config.max_read_failures,
            )
        self.detector = ObjectDetector(config)
        self.object_tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )
        self.person_tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )
        self.static_filter = StaticObjectFilter(config)
        pathway_file = (
            Path(pathway_config_override)
            if pathway_config_override
            else config.resolve(config.pathway_config_file)
        )
        self.pathways = PathwayManager(
            pathway_file,
            min_overlap_ratio=config.pathway_min_overlap_ratio,
            min_overlap_area_px=config.pathway_min_overlap_area_px,
            use_foot_point=config.pathway_use_foot_point,
            zone_scale=zone_scale,
        )
        if reference_frame is None:
            reference_frame = self._load_reference_frame(pathway_file)
        self.pathway_obstruction_monitor = PathwayObstructionMonitor(config, reference_frame)
        self.mobile_monitor = MobileUsageMonitor(config)
        self.ppe_monitor = PPEMonitor(config)
        self.object_fall_monitor = ObjectFallMonitor(config)
        self.near_miss_monitor = NearMissMonitor(config)
        self.fire_smoke_monitor = FireSmokeMonitor(config)
        self.floor_hazard_monitor = FloorHazardMonitor(config)
        self.low_visibility_monitor = LowVisibilityMonitor(config)
        self.fire_classifier = FireClassifier.from_config(config)
        self.observations = ObservationManager(
            config,
            persist=persist_observations,
            pathways=self.pathways,
            reference_frame=reference_frame,
        )
        self.settings_store = DetectionSettingsStore(config)
        initial_settings = self.settings_store.ensure_exists(config)
        initial_settings.apply_to(self.config)
        self._settings_reload_interval = 0 if lock_settings else 15

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.mobile_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF00FF"))
        self.phone_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00FF00"))
        self.ppe_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF8800"))
        self.fire_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF0000"))
        self.smoke_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#888888"))
        self.oil_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#CC7700"))
        self.wet_floor_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00CCCC"))
        self.fire_tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )
        self.smoke_tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )
        self.oil_tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )
        self.wet_floor_tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )
        self.hazard_tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )
        self.fall_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#0088FF"))
        self.near_miss_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF4444"))

    def _load_reference_frame(self, pathway_file: Path) -> np.ndarray | None:
        from src.camera_zone_store import CameraZoneStore

        store = CameraZoneStore(self.config.project_root)
        frame = store.load_reference_frame(self.config.camera_id)
        if frame is not None:
            return frame

        zone_data_path = pathway_file.with_suffix(".json")
        if zone_data_path.is_file():
            frame = store.load_reference_frame(zone_data_path.stem)
            if frame is not None:
                return frame
        return None

    @staticmethod
    def _bbox_center(xyxy: np.ndarray) -> tuple[float, float]:
        return float((xyxy[0] + xyxy[2]) / 2.0), float((xyxy[1] + xyxy[3]) / 2.0)

    @staticmethod
    def _center_distance(a: np.ndarray, b: np.ndarray) -> float:
        ax, ay = PathwayMonitorPipeline._bbox_center(a)
        bx, by = PathwayMonitorPipeline._bbox_center(b)
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _center_inside_bbox(self, inner: np.ndarray, outer: np.ndarray, padding_ratio: float = 0.08) -> bool:
        cx, cy = self._bbox_center(inner)
        pad_x = (outer[2] - outer[0]) * padding_ratio
        pad_y = (outer[3] - outer[1]) * padding_ratio
        return (
            float(outer[0]) - pad_x <= cx <= float(outer[2]) + pad_x
            and float(outer[1]) - pad_y <= cy <= float(outer[3]) + pad_y
        )

    def _attached_to_person(self, object_xyxy: np.ndarray, person_boxes: list[np.ndarray]) -> bool:
        # Floor plates sit near a worker's feet on high-angle CCTV and must
        # not be treated as worn/carried PPE.
        if self._is_low_profile_object(object_xyxy):
            return False

        obj_w = max(float(object_xyxy[2] - object_xyxy[0]), 1.0)
        obj_h = max(float(object_xyxy[3] - object_xyxy[1]), 1.0)
        obj_area = obj_w * obj_h

        for person_xyxy in person_boxes:
            person_w = max(float(person_xyxy[2] - person_xyxy[0]), 1.0)
            person_h = max(float(person_xyxy[3] - person_xyxy[1]), 1.0)
            person_area = person_w * person_h

            # Only skip small worn/carried items (helmet, phone). Keep large machinery/carts.
            if obj_area > person_area * 0.45 and max(obj_w, obj_h) > person_h * 0.40:
                continue

            if self._bbox_iou(object_xyxy, person_xyxy) >= 0.10:
                return True
            if self._center_inside_bbox(object_xyxy, person_xyxy):
                return True
        return False

    def _filter_person_attached_blocking(
        self,
        detections: sv.Detections,
        person_boxes: list[np.ndarray],
    ) -> sv.Detections:
        if len(detections) == 0 or not person_boxes:
            return detections

        keep = [
            idx
            for idx in range(len(detections))
            if not self._attached_to_person(detections.xyxy[idx], person_boxes)
        ]
        return detections[keep]

    def _bbox_area(self, xyxy: np.ndarray) -> float:
        return max(float(xyxy[2] - xyxy[0]), 0.0) * max(float(xyxy[3] - xyxy[1]), 0.0)

    def _is_large_pathway_object(self, xyxy: np.ndarray) -> bool:
        return self._bbox_area(xyxy) >= self.config.pathway_block_large_object_area_px

    def _is_low_profile_object(self, xyxy: np.ndarray) -> bool:
        width = max(float(xyxy[2] - xyxy[0]), 1.0)
        height = max(float(xyxy[3] - xyxy[1]), 1.0)
        return height <= 56.0 or (
            width / height >= self.config.small_object_wide_aspect and height <= 90.0
        )

    def _walkway_detections(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections
        keep = [
            idx
            for idx in range(len(detections))
            if self.pathways.best_walkway_zone_for_detection(detections.xyxy[idx]) is not None
        ]
        if not keep:
            return sv.Detections.empty()
        return detections[keep]

    def _person_boxes(self, all_detections: sv.Detections) -> list[np.ndarray]:
        person_detections = self.detector.filter_by_class_names(
            all_detections,
            [self.config.mobile_person_class],
        )
        if len(person_detections) == 0:
            return []
        return [person_detections.xyxy[idx] for idx in range(len(person_detections))]

    def _overlaps_falling_object(
        self,
        xyxy: np.ndarray,
        falling_boxes: list[np.ndarray],
        merge_px: float,
    ) -> bool:
        for fall_xyxy in falling_boxes:
            if self._center_distance(xyxy, fall_xyxy) <= merge_px:
                return True
        return False

    def _needs_full_detection(self) -> bool:
        """Full YOLO pass is only needed for pathway/person/object violations."""
        return (
            self.config.pathway_block_enabled
            or (
                self.config.object_fall_enabled
                and self.config.object_fall_detect_object
            )
            or self.config.mobile_usage_enabled
            or self.config.ppe_enabled
            or self.config.near_miss_enabled
            or (
                self.config.object_fall_enabled
                and self.config.object_fall_detect_person
            )
        )

    def _build_labels(self, detections: sv.Detections) -> list[str]:
        labels = []
        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            tracker_ids = [None] * len(detections)
        for class_id, track_id in zip(detections.class_id, tracker_ids, strict=False):
            name = self.detector.class_name(int(class_id))
            if track_id is None:
                labels.append(name)
            else:
                labels.append(f"#{int(track_id)} {name}")
        return labels

    def _filter_smoke_person_overlap(
        self,
        smoke_detections: sv.Detections,
        person_detections: sv.Detections,
        frame: np.ndarray,
    ) -> sv.Detections:
        if len(smoke_detections) == 0 or len(person_detections) == 0:
            return smoke_detections

        from src.smoke_temporal import analyze_smoke_region

        max_overlap = self.config.smoke_person_max_overlap
        keep: list[int] = []
        for index in range(len(smoke_detections)):
            sample = analyze_smoke_region(frame, smoke_detections.xyxy[index])
            # Wispy plumes near a person's hand are irregular; solid blobs on clothing are not.
            if (
                sample.solidity <= self.config.fire_smoke_cv_smoke_max_solidity
                and sample.mask_fill <= self.config.fire_smoke_cv_smoke_max_mask_fill
            ):
                keep.append(index)
                continue

            sx1, sy1, sx2, sy2 = smoke_detections.xyxy[index]
            smoke_area = max(1.0, float((sx2 - sx1) * (sy2 - sy1)))
            overlaps_person = False
            for person_index in range(len(person_detections)):
                px1, py1, px2, py2 = person_detections.xyxy[person_index]
                ix1 = max(float(sx1), float(px1))
                iy1 = max(float(sy1), float(py1))
                ix2 = min(float(sx2), float(px2))
                iy2 = min(float(sy2), float(py2))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / smoke_area >= max_overlap:
                    overlaps_person = True
                    break
            if not overlaps_person:
                keep.append(index)

        return smoke_detections[keep] if keep else sv.Detections.empty()

    def _cv_detections_as_class(
        self,
        detections: sv.Detections,
        class_name: str,
    ) -> sv.Detections:
        if len(detections) == 0:
            return detections

        target_id = next(
            (
                class_id
                for class_id, name in self.detector.names.items()
                if name.lower() == class_name.lower()
                or class_name.lower() in name.lower()
            ),
            None,
        )
        if target_id is None:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=np.full(len(detections), int(target_id), dtype=int),
        )

    def _render_preview_frame(
        self,
        frame: np.ndarray,
        *,
        static_detections: sv.Detections,
        mobile_violation_detections: sv.Detections,
        phone_detections: sv.Detections,
        ppe_violation_detections: sv.Detections,
        fire_violation_detections: sv.Detections,
        smoke_violation_detections: sv.Detections,
        oil_violation_detections: sv.Detections,
        wet_floor_violation_detections: sv.Detections,
        object_fall_violation_detections: sv.Detections,
        person_fall_violation_detections: sv.Detections,
        near_miss_violation_detections: sv.Detections,
        pathway_occupancy_detections: sv.Detections | None = None,
        cv_obstruction_boxes: list[np.ndarray] | None = None,
        low_visibility_active: bool = False,
        low_visibility_score: float | None = None,
    ) -> np.ndarray:
        preview = frame.copy()
        self.pathways.draw_all(preview)

        occupancy = pathway_occupancy_detections if pathway_occupancy_detections is not None else sv.Detections.empty()
        if len(occupancy) > 0:
            occupancy_labels = self._build_labels(occupancy)
            preview = self.box_annotator.annotate(preview, occupancy)
            preview = self.label_annotator.annotate(preview, occupancy, occupancy_labels)

        if len(static_detections) > 0:
            labels = self._build_labels(static_detections)
            preview = self.box_annotator.annotate(preview, static_detections)
            preview = self.label_annotator.annotate(preview, static_detections, labels)

        if cv_obstruction_boxes:
            for xyxy in cv_obstruction_boxes:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    preview,
                    "pathway object",
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

        if len(mobile_violation_detections) > 0:
            mobile_labels = self._build_labels(mobile_violation_detections)
            preview = self.mobile_box_annotator.annotate(preview, mobile_violation_detections)
            preview = self.label_annotator.annotate(
                preview, mobile_violation_detections, mobile_labels
            )

        if len(phone_detections) > 0:
            preview = self.phone_box_annotator.annotate(preview, phone_detections)

        if len(ppe_violation_detections) > 0:
            ppe_labels = self._build_labels(ppe_violation_detections)
            preview = self.ppe_box_annotator.annotate(preview, ppe_violation_detections)
            preview = self.label_annotator.annotate(
                preview, ppe_violation_detections, ppe_labels
            )

        if len(fire_violation_detections) > 0:
            fire_labels = self._build_labels(fire_violation_detections)
            preview = self.fire_box_annotator.annotate(preview, fire_violation_detections)
            preview = self.label_annotator.annotate(
                preview, fire_violation_detections, fire_labels
            )

        if len(smoke_violation_detections) > 0:
            smoke_labels = self._build_labels(smoke_violation_detections)
            preview = self.smoke_box_annotator.annotate(preview, smoke_violation_detections)
            preview = self.label_annotator.annotate(
                preview, smoke_violation_detections, smoke_labels
            )

        if len(oil_violation_detections) > 0:
            oil_labels = self._build_labels(oil_violation_detections)
            preview = self.oil_box_annotator.annotate(preview, oil_violation_detections)
            preview = self.label_annotator.annotate(
                preview, oil_violation_detections, oil_labels
            )

        if len(wet_floor_violation_detections) > 0:
            wet_floor_labels = self._build_labels(wet_floor_violation_detections)
            preview = self.wet_floor_box_annotator.annotate(preview, wet_floor_violation_detections)
            preview = self.label_annotator.annotate(
                preview, wet_floor_violation_detections, wet_floor_labels
            )

        if len(object_fall_violation_detections) > 0:
            fall_labels = self._build_labels(object_fall_violation_detections)
            preview = self.fall_box_annotator.annotate(preview, object_fall_violation_detections)
            preview = self.label_annotator.annotate(
                preview, object_fall_violation_detections, fall_labels
            )

        if len(person_fall_violation_detections) > 0:
            person_fall_labels = self._build_labels(person_fall_violation_detections)
            preview = self.fall_box_annotator.annotate(preview, person_fall_violation_detections)
            preview = self.label_annotator.annotate(
                preview, person_fall_violation_detections, person_fall_labels
            )

        if len(near_miss_violation_detections) > 0:
            near_miss_labels = self._build_labels(near_miss_violation_detections)
            preview = self.near_miss_box_annotator.annotate(preview, near_miss_violation_detections)
            preview = self.label_annotator.annotate(
                preview, near_miss_violation_detections, near_miss_labels
            )

        if low_visibility_active:
            banner = "LOW VISIBILITY"
            if low_visibility_score is not None:
                banner = f"LOW VISIBILITY ({low_visibility_score * 100:.0f}% clarity)"
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 42), (40, 40, 40), -1)
            cv2.putText(
                preview,
                banner,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

        return preview

    def _sync_video_clock(self) -> None:
        if not isinstance(self.capture, VideoFileCapture):
            return
        if not getattr(self, "_video_clock_bound", False):
            duration = self.capture.duration_seconds
            self.observations.bind_video_timeline(duration_seconds=duration)
            self._video_clock_bound = True
            logger.info(
                "Observation times use video timeline (%.1fs at %.2f fps)",
                duration or 0.0,
                self.capture.effective_fps,
            )
        self.observations.set_media_time(self.capture.current_time_sec())

    def run(
        self,
        on_observation: Callable | None = None,
        frame_callback: Callable[[np.ndarray], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> int:
        frame_index = 0
        self._video_clock_bound = False

        try:
            for frame in self.capture.frames():
                if should_stop and should_stop():
                    break
                frame_index += 1
                self._sync_video_clock()

                if isinstance(self.capture, VideoFileCapture) and progress_callback:
                    progress_callback(self.capture.progress_ratio())

                if frame_index % self.config.frame_skip != 0:
                    continue

                if (
                    self._settings_reload_interval > 0
                    and frame_index % self._settings_reload_interval == 0
                ):
                    runtime_settings = self.settings_store.load(self.config)
                    runtime_settings.apply_to(self.config)

                if isinstance(self.capture, RTSPCapture):
                    self.capture.discard_stale_frames()

                all_detections = sv.Detections.empty()
                if self._needs_full_detection():
                    detect_confidence = self.config.confidence
                    if self.config.pathway_block_enabled:
                        detect_confidence = min(
                            detect_confidence,
                            self.config.pathway_block_confidence,
                        )
                    small_object_rois = [
                        zone.polygon.bounds for zone in self.pathways.walkway_zones
                    ]
                    all_detections = self.detector.detect_all(
                        frame,
                        confidence=detect_confidence,
                        rois=small_object_rois or None,
                    )
                walkway_zones = self.pathways.walkway_zones
                label_zones = self.pathways.zones
                camera_id = self.config.camera_id

                # --- Tracked non-person objects (pathway block + object fall) ---
                static_detections = sv.Detections.empty()
                blocking_detections = sv.Detections.empty()
                pathway_occupancy_detections = sv.Detections.empty()
                cv_obstruction_boxes: list[np.ndarray] = []
                need_blocking = (
                    self.config.pathway_block_enabled
                    or (
                        self.config.object_fall_enabled
                        and self.config.object_fall_detect_object
                    )
                )

                blocking_raw = sv.Detections.empty()
                if need_blocking:
                    blocking_raw = self.detector.exclude_classes(
                        all_detections,
                        self.detector._blocking_exclude_ids,
                    )
                    blocking_detections = self.object_tracker.update_with_detections(blocking_raw)

                    active_object_tracks = set()
                    if blocking_detections.tracker_id is not None:
                        active_object_tracks = {
                            int(tid) for tid in blocking_detections.tracker_id if tid is not None
                        }
                    self.static_filter.prune(active_object_tracks)
                    self.object_fall_monitor.prune_tracks(active_object_tracks)

                if self.config.pathway_block_enabled:
                    static_detections, _ = self.static_filter.annotate(blocking_detections)
                    person_boxes = self._person_boxes(all_detections)
                    static_detections = self._filter_person_attached_blocking(
                        static_detections,
                        person_boxes,
                    )
                    pathway_active_keys: set[tuple[str, str, float, float]] = set()
                    reported_boxes: list[np.ndarray] = []

                    def _report_pathway_block(
                        xyxy: np.ndarray,
                        zone,
                        object_class: str,
                        confidence: float,
                        track_id: int | None,
                    ) -> None:
                        for previous in reported_boxes:
                            if self._bbox_iou(xyxy, previous) >= 0.30:
                                return

                        observation, newly_confirmed = self.observations.process_pathway_block(
                            frame=frame,
                            xyxy=xyxy,
                            object_class=object_class,
                            zone_id=zone.id,
                            zone_name=zone.name,
                            confidence_score=confidence,
                            track_id=track_id,
                        )

                        center_x = float((xyxy[0] + xyxy[2]) / 2.0)
                        center_y = float((xyxy[1] + xyxy[3]) / 2.0)
                        pathway_active_keys.add((zone.id, object_class, center_x, center_y))
                        reported_boxes.append(xyxy)

                        if observation is None:
                            return

                        logger.info(
                            "Pathway block %s | %s on %s | %.0fs | severity=%s | %s",
                            observation.observation_id[:8],
                            object_class,
                            zone.name,
                            observation.duration_seconds,
                            observation.severity,
                            observation.summary,
                        )

                        if on_observation and newly_confirmed:
                            on_observation(observation)

                    for i in range(len(static_detections)):
                        xyxy = static_detections.xyxy[i]
                        class_name = self.detector.class_name(int(static_detections.class_id[i]))
                        confidence = (
                            float(static_detections.confidence[i])
                            if static_detections.confidence is not None
                            else 0.0
                        )
                        track_id = (
                            int(static_detections.tracker_id[i])
                            if static_detections.tracker_id is not None
                            and static_detections.tracker_id[i] is not None
                            else None
                        )
                        zone = self.pathways.best_walkway_zone_for_detection(xyxy)
                        if zone is None:
                            continue

                        _report_pathway_block(
                            xyxy,
                            zone,
                            class_name,
                            confidence,
                            track_id,
                        )

                    occupancy_source = self._walkway_detections(blocking_raw)
                    for i in range(len(occupancy_source)):
                        xyxy = occupancy_source.xyxy[i]
                        reportable = (
                            self.config.pathway_block_include_moving_large
                            and self._is_large_pathway_object(xyxy)
                        ) or (
                            self.config.small_object_enabled
                            and self._is_low_profile_object(xyxy)
                        )
                        if not reportable:
                            continue
                        zone = self.pathways.best_walkway_zone_for_detection(xyxy)
                        if zone is None:
                            continue
                        class_name = self.detector.class_name(int(occupancy_source.class_id[i]))
                        confidence = (
                            float(occupancy_source.confidence[i])
                            if occupancy_source.confidence is not None
                            else 0.0
                        )
                        track_id = (
                            int(occupancy_source.tracker_id[i])
                            if occupancy_source.tracker_id is not None
                            and occupancy_source.tracker_id[i] is not None
                            else None
                        )
                        _report_pathway_block(
                            xyxy,
                            zone,
                            class_name,
                            confidence,
                            track_id,
                        )

                    cv_obstructions = self.pathway_obstruction_monitor.detect(
                        frame,
                        walkway_zones,
                        person_boxes,
                    )
                    for obstruction in cv_obstructions:
                        xyxy = obstruction.xyxy
                        zone = obstruction.zone
                        _report_pathway_block(
                            xyxy,
                            zone,
                            "pathway_obstruction",
                            0.75,
                            obstruction.track_id,
                        )

                    pathway_occupancy_detections = occupancy_source
                    cv_obstruction_boxes = [item.xyxy for item in cv_obstructions]

                    self.observations.resolve_stale_pathway_blocks(pathway_active_keys)
                else:
                    self.observations.resolve_stale_pathway_blocks(set())

                # --- Object fall detection (tracked objects) ---
                object_fall_violation_detections = sv.Detections.empty()
                person_fall_violation_detections = sv.Detections.empty()
                object_fall_active_tracks: set[int] = set()
                falling_object_boxes: list[np.ndarray] = []

                if (
                    self.config.object_fall_enabled
                    and self.config.object_fall_detect_object
                    and len(blocking_detections) > 0
                ):
                    object_falls, object_fall_active_tracks = self.object_fall_monitor.detect_falls(
                        blocking_detections,
                        label_zones,
                        camera_id,
                        self.detector.class_name,
                        FALL_KIND_OBJECT,
                    )
                    fall_indices = []
                    for zone, track_id, xyxy, class_name, fall_kind in object_falls:
                        confidence = 0.0
                        for idx in range(len(blocking_detections)):
                            if blocking_detections.tracker_id[idx] == track_id:
                                if blocking_detections.confidence is not None:
                                    confidence = float(blocking_detections.confidence[idx])
                                fall_indices.append(idx)
                                break

                        stored_class = class_name
                        if class_name.lower() in ("fire", "flame"):
                            stored_class = "falling object"

                        falling_object_boxes.append(xyxy)
                        observation = self.observations.process_object_fall(
                            frame=frame,
                            xyxy=xyxy,
                            zone_id=zone.id,
                            zone_name=zone.name,
                            track_id=track_id,
                            object_class=stored_class,
                            fall_kind=fall_kind,
                            confidence_score=confidence,
                        )

                        logger.info(
                            "Object fall %s | %s on %s | %.0fs | severity=%s | %s",
                            observation.observation_id[:8],
                            class_name,
                            zone.name,
                            observation.duration_seconds,
                            observation.severity,
                            observation.summary,
                        )

                        if on_observation:
                            on_observation(observation)

                    if fall_indices:
                        object_fall_violation_detections = blocking_detections[fall_indices]

                for idx in range(len(blocking_detections)):
                    if blocking_detections.tracker_id is None:
                        break
                    track_id = blocking_detections.tracker_id[idx]
                    if track_id is None:
                        continue
                    if int(track_id) in object_fall_active_tracks:
                        falling_object_boxes.append(blocking_detections.xyxy[idx])

                # --- Person-based violations: mobile usage, PPE, person fall, near miss ---
                mobile_violation_detections = sv.Detections.empty()
                ppe_violation_detections = sv.Detections.empty()
                near_miss_violation_detections = sv.Detections.empty()
                mobile_active_track_ids: set[int] = set()
                ppe_active_track_ids: set[int] = set()
                near_miss_active_keys: set[tuple[int, str]] = set()
                phone_detections = sv.Detections.empty()
                person_detections = sv.Detections.empty()

                mobile_allowed_zones = self.pathways.mobile_allowed_zones
                need_person_pass = (
                    self.config.mobile_usage_enabled
                    or self.config.ppe_enabled
                    or self.config.near_miss_enabled
                    or (
                        self.config.object_fall_enabled
                        and self.config.object_fall_detect_person
                    )
                )

                if need_person_pass:
                    person_classes = list({
                        self.config.mobile_person_class,
                        self.config.ppe_person_class,
                        self.config.near_miss_person_class,
                    })
                    person_detections = self.detector.filter_by_class_names(
                        all_detections,
                        person_classes,
                    )
                    person_detections = self.person_tracker.update_with_detections(person_detections)

                    active_person_tracks = set()
                    if person_detections.tracker_id is not None:
                        active_person_tracks = {
                            int(tid) for tid in person_detections.tracker_id if tid is not None
                        }
                    self.mobile_monitor.prune_track(active_person_tracks)
                    self.ppe_monitor.prune_track(active_person_tracks)
                    self.object_fall_monitor.prune_tracks(active_person_tracks)
                    self.near_miss_monitor.prune_tracks(active_person_tracks)

                if self.config.mobile_usage_enabled:
                    if len(person_detections) > 0:
                        phone_detections = self.detector.detect_phones(frame)

                        violations, mobile_active_track_ids = self.mobile_monitor.detect_violations(
                            person_detections=person_detections,
                            phone_detections=phone_detections,
                            label_zones=label_zones,
                            mobile_allowed_zones=mobile_allowed_zones,
                            camera_id=camera_id,
                        )
                    else:
                        violations = []
                        mobile_active_track_ids = set()
                        phone_detections = sv.Detections.empty()

                    violation_indices = []
                    for zone, track_id, person_xyxy in violations:
                        person_confidence = 0.0
                        for idx in range(len(person_detections)):
                            if person_detections.tracker_id[idx] == track_id:
                                if person_detections.confidence is not None:
                                    person_confidence = float(person_detections.confidence[idx])
                                violation_indices.append(idx)
                                break

                        observation = self.observations.process_mobile_usage(
                            frame=frame,
                            xyxy=person_xyxy,
                            zone_id=zone.id,
                            zone_name=zone.name,
                            track_id=track_id,
                            confidence_score=person_confidence,
                        )

                        logger.info(
                            "Mobile usage %s | person #%s outside allowed zone on %s | %.0fs | severity=%s | %s",
                            observation.observation_id[:8],
                            track_id,
                            zone.name,
                            observation.duration_seconds,
                            observation.severity,
                            observation.summary,
                        )

                        if on_observation:
                            on_observation(observation)

                    if violation_indices:
                        mobile_violation_detections = person_detections[violation_indices]

                    self.observations.resolve_stale_mobile_usage(mobile_active_track_ids)
                else:
                    self.observations.resolve_stale_mobile_usage(set())
                    phone_detections = sv.Detections.empty()

                if self.config.ppe_enabled and len(person_detections) > 0:
                    ppe_detections = self.detector.detect_ppe(frame)
                    item_detections = {
                        item.key: self.detector.filter_ppe_item(ppe_detections, item.key)
                        for item in self.config.enabled_ppe_items()
                    }

                    ppe_violations, ppe_active_track_ids = self.ppe_monitor.detect_violations(
                        person_detections=person_detections,
                        item_detections=item_detections,
                        label_zones=label_zones,
                        camera_id=camera_id,
                    )

                    ppe_violation_indices = []
                    for zone, track_id, person_xyxy, missing_items in ppe_violations:
                        person_confidence = 0.0
                        for idx in range(len(person_detections)):
                            if person_detections.tracker_id[idx] == track_id:
                                if person_detections.confidence is not None:
                                    person_confidence = float(person_detections.confidence[idx])
                                ppe_violation_indices.append(idx)
                                break

                        observation = self.observations.process_ppe_violation(
                            frame=frame,
                            xyxy=person_xyxy,
                            zone_id=zone.id,
                            zone_name=zone.name,
                            track_id=track_id,
                            missing_items=missing_items,
                            confidence_score=person_confidence,
                        )

                        logger.info(
                            "PPE violation %s | person #%s missing %s on %s | %.0fs | severity=%s | %s",
                            observation.observation_id[:8],
                            track_id,
                            ", ".join(missing_items),
                            zone.name,
                            observation.duration_seconds,
                            observation.severity,
                            observation.summary,
                        )

                        if on_observation:
                            on_observation(observation)

                    if ppe_violation_indices:
                        ppe_violation_detections = person_detections[ppe_violation_indices]

                    self.observations.resolve_stale_ppe_violations(ppe_active_track_ids)
                else:
                    self.observations.resolve_stale_ppe_violations(set())

                # --- Person fall detection ---
                if (
                    self.config.object_fall_enabled
                    and self.config.object_fall_detect_person
                    and len(person_detections) > 0
                ):
                    person_falls, person_fall_active = self.object_fall_monitor.detect_falls(
                        person_detections,
                        label_zones,
                        camera_id,
                        self.detector.class_name,
                        FALL_KIND_PERSON,
                    )
                    object_fall_active_tracks |= person_fall_active
                    person_fall_indices = []
                    for zone, track_id, xyxy, class_name, fall_kind in person_falls:
                        confidence = 0.0
                        for idx in range(len(person_detections)):
                            if person_detections.tracker_id[idx] == track_id:
                                if person_detections.confidence is not None:
                                    confidence = float(person_detections.confidence[idx])
                                person_fall_indices.append(idx)
                                break

                        observation = self.observations.process_object_fall(
                            frame=frame,
                            xyxy=xyxy,
                            zone_id=zone.id,
                            zone_name=zone.name,
                            track_id=track_id,
                            object_class=class_name,
                            fall_kind=fall_kind,
                            confidence_score=confidence,
                        )

                        logger.info(
                            "Person fall %s | person #%s on %s | %.0fs | severity=%s | %s",
                            observation.observation_id[:8],
                            track_id,
                            zone.name,
                            observation.duration_seconds,
                            observation.severity,
                            observation.summary,
                        )

                        if on_observation:
                            on_observation(observation)

                    if person_fall_indices:
                        person_fall_violation_detections = person_detections[person_fall_indices]

                if self.config.object_fall_enabled:
                    self.observations.resolve_stale_object_falls(object_fall_active_tracks)
                else:
                    self.observations.resolve_stale_object_falls(set())

                # --- Near miss detection (person + moving hazard) ---
                if self.config.near_miss_enabled and len(person_detections) > 0:
                    hazard_indices = []
                    for idx in range(len(all_detections)):
                        class_name = self.detector.class_name(int(all_detections.class_id[idx]))
                        if self.near_miss_monitor._is_hazard_class(class_name):
                            hazard_indices.append(idx)

                    if hazard_indices:
                        hazard_detections = all_detections[hazard_indices]
                        hazard_detections = self.hazard_tracker.update_with_detections(hazard_detections)
                        hazard_class_names = [
                            self.detector.class_name(int(class_id))
                            for class_id in hazard_detections.class_id
                        ]

                        near_miss_violations, near_miss_active_keys = (
                            self.near_miss_monitor.detect_violations(
                                person_detections=person_detections,
                                hazard_detections=hazard_detections,
                                hazard_class_names=hazard_class_names,
                                label_zones=label_zones,
                                camera_id=camera_id,
                            )
                        )

                        near_miss_indices = []
                        for (
                            zone,
                            person_track_id,
                            person_xyxy,
                            hazard_class,
                            _hazard_track_id,
                            _hazard_xyxy,
                        ) in near_miss_violations:
                            person_confidence = 0.0
                            for idx in range(len(person_detections)):
                                if person_detections.tracker_id[idx] == person_track_id:
                                    if person_detections.confidence is not None:
                                        person_confidence = float(person_detections.confidence[idx])
                                    near_miss_indices.append(idx)
                                    break

                            observation = self.observations.process_near_miss(
                                frame=frame,
                                person_xyxy=person_xyxy,
                                zone_id=zone.id,
                                zone_name=zone.name,
                                person_track_id=person_track_id,
                                hazard_class=hazard_class,
                                confidence_score=person_confidence,
                            )

                            logger.info(
                                "Near miss %s | person #%s vs %s on %s | %.0fs | severity=%s | %s",
                                observation.observation_id[:8],
                                person_track_id,
                                hazard_class,
                                zone.name,
                                observation.duration_seconds,
                                observation.severity,
                                observation.summary,
                            )

                            if on_observation:
                                on_observation(observation)

                        if near_miss_indices:
                            near_miss_violation_detections = person_detections[near_miss_indices]

                    self.observations.resolve_stale_near_miss(near_miss_active_keys)
                else:
                    self.observations.resolve_stale_near_miss(set())

                # --- Camera-wide hazard detection: fire, smoke, oil spillage, wet floor ---
                fire_violation_detections = sv.Detections.empty()
                smoke_violation_detections = sv.Detections.empty()
                oil_violation_detections = sv.Detections.empty()
                wet_floor_violation_detections = sv.Detections.empty()
                hazard_active_spatial: set[tuple[str, str, float, float]] = set()

                fire_enabled = self.config.fire_detection_enabled
                smoke_enabled = self.config.smoke_detection_enabled
                oil_enabled = self.config.oil_spillage_enabled
                wet_floor_enabled = self.config.wet_floor_enabled
                low_visibility_enabled = self.config.low_visibility_enabled
                low_visibility_active = False
                low_visibility_score: float | None = None

                if fire_enabled or smoke_enabled:
                    hazard_detections = self.detector.detect_fire_smoke(frame)
                    if self.config.fire_smoke_cv_fallback_enabled:
                        if self.config.fire_smoke_cv_fire_fallback_enabled:
                            cv_fire = detect_fire_regions(frame, self.config)
                            hazard_detections = merge_detections(
                                hazard_detections,
                                self._cv_detections_as_class(cv_fire, "fire"),
                            )
                        cv_smoke = detect_smoke_regions(frame, self.config)
                        if self.config.fire_smoke_cv_smoke_fallback_enabled:
                            hazard_detections = merge_detections(
                                hazard_detections,
                                self._cv_detections_as_class(cv_smoke, "smoke"),
                            )

                    fire_detections = sv.Detections.empty()
                    fire_subtypes: list[str] = []
                    smoke_detections = sv.Detections.empty()

                    if fire_enabled:
                        fire_candidates = self.detector.filter_fire_classes(hazard_detections)
                        if len(fire_candidates) > 0 and fire_candidates.confidence is not None:
                            keep = [
                                index
                                for index, score in enumerate(fire_candidates.confidence)
                                if float(score) >= self.config.fire_min_confidence
                            ]
                            fire_candidates = fire_candidates[keep] if keep else sv.Detections.empty()
                        fire_detections, fire_subtypes = self.fire_classifier.filter_alerting_detections(
                            fire_candidates,
                            self.detector.class_name,
                        )
                        if len(fire_detections) == 0 and len(fire_candidates) > 0:
                            fire_detections = fire_candidates
                            fire_subtypes = []
                            for class_id in fire_detections.class_id:
                                fire_type = self.fire_classifier.classify_detection(
                                    self.detector.class_name(int(class_id))
                                )
                                fire_subtypes.append(
                                    fire_type.key if fire_type else "structural_fire"
                                )
                        fire_detections = self.fire_tracker.update_with_detections(fire_detections)
                        fire_subtypes = []
                        for class_id in fire_detections.class_id:
                            fire_type = self.fire_classifier.classify_detection(
                                self.detector.class_name(int(class_id))
                            )
                            fire_subtypes.append(
                                fire_type.key if fire_type else "structural_fire"
                            )
                    if smoke_enabled:
                        smoke_detections = self.detector.filter_smoke_classes(hazard_detections)
                        smoke_detections = self._filter_smoke_person_overlap(
                            smoke_detections,
                            person_detections,
                            frame,
                        )
                        if len(smoke_detections) > 0:
                            tracked_smoke = self.smoke_tracker.update_with_detections(smoke_detections)
                            # ByteTrack silently drops boxes below an internal high-confidence cutoff.
                            if len(tracked_smoke) > 0:
                                smoke_detections = tracked_smoke

                    if logger.isEnabledFor(logging.DEBUG) and (
                        len(fire_detections) > 0 or len(smoke_detections) > 0
                    ):
                        logger.debug(
                            "Fire/smoke candidates | fire=%d smoke=%d",
                            len(fire_detections),
                            len(smoke_detections),
                        )

                    fire_violations, smoke_violations, hazard_active_spatial = (
                        self.fire_smoke_monitor.detect_violations(
                            frame,
                            fire_detections,
                            fire_subtypes,
                            smoke_detections,
                            fire_enabled=fire_enabled,
                            smoke_enabled=smoke_enabled,
                        )
                    )

                    if (
                        fire_enabled
                        and len(fire_detections) > 0
                        and len(fire_violations) == 0
                        and frame_index % 45 == 0
                    ):
                        logger.info(
                            "Fire candidates detected (%d) — building confirmation on live stream "
                            "(needs %d frames + flicker check)",
                            len(fire_detections),
                            self.config.fire_min_detection_frames,
                        )

                    fire_indices = []
                    merge_px = self.config.fire_merge_distance_px
                    if fire_enabled:
                        for violation_type, xyxy, confidence, track_id, fire_subtype in fire_violations:
                            if falling_object_boxes and self._overlaps_falling_object(
                                xyxy,
                                falling_object_boxes,
                                self.config.fire_merge_distance_px,
                            ):
                                logger.debug(
                                    "Skipping fire alert — overlapping falling object motion detected"
                                )
                                continue
                            if track_id is not None and self.object_fall_monitor.is_track_falling(track_id):
                                logger.debug(
                                    "Skipping fire alert — track #%s is in falling motion",
                                    track_id,
                                )
                                continue

                            observation = self.observations.process_hazard_detection(
                                frame=frame,
                                xyxy=xyxy,
                                violation_type=violation_type,
                                object_class=fire_subtype,
                                confidence_score=confidence,
                                track_id=track_id,
                            )
                            vcx = float((xyxy[0] + xyxy[2]) / 2.0)
                            vcy = float((xyxy[1] + xyxy[3]) / 2.0)
                            for idx in range(len(fire_detections)):
                                if idx in fire_indices:
                                    continue
                                dxy = fire_detections.xyxy[idx]
                                dcx = float((dxy[0] + dxy[2]) / 2.0)
                                dcy = float((dxy[1] + dxy[3]) / 2.0)
                                if ((vcx - dcx) ** 2 + (vcy - dcy) ** 2) ** 0.5 <= merge_px:
                                    fire_indices.append(idx)

                            fire_label = self.fire_classifier.label_for(fire_subtype)
                            logger.info(
                                "Fire detection %s | %s | %.0fs | severity=%s | %s",
                                observation.observation_id[:8],
                                fire_label,
                                observation.duration_seconds,
                                observation.severity,
                                observation.summary,
                            )
                            if on_observation:
                                on_observation(observation)

                    smoke_indices = []
                    if smoke_enabled:
                        for violation_type, xyxy, confidence, track_id, _subtype in smoke_violations:
                            observation = self.observations.process_hazard_detection(
                                frame=frame,
                                xyxy=xyxy,
                                violation_type=violation_type,
                                object_class="smoke",
                                confidence_score=confidence,
                                track_id=track_id,
                            )
                            for idx in range(len(smoke_detections)):
                                if track_id is not None and smoke_detections.tracker_id[idx] == track_id:
                                    smoke_indices.append(idx)
                                    break

                            logger.info(
                                "Smoke detection %s | %.0fs | severity=%s | %s",
                                observation.observation_id[:8],
                                observation.duration_seconds,
                                observation.severity,
                                observation.summary,
                            )
                            if on_observation:
                                on_observation(observation)

                    if fire_indices:
                        fire_violation_detections = fire_detections[fire_indices]
                    elif self.show_preview and len(fire_detections) > 0:
                        fire_violation_detections = fire_detections
                    if smoke_indices:
                        smoke_violation_detections = smoke_detections[smoke_indices]

                if oil_enabled or wet_floor_enabled:
                    floor_hazard_detections = self.detector.detect_floor_hazards(frame)
                    oil_detections = sv.Detections.empty()
                    wet_floor_detections = sv.Detections.empty()

                    if oil_enabled:
                        oil_detections = self.detector.filter_oil_spillage_classes(floor_hazard_detections)
                        oil_detections = self.oil_tracker.update_with_detections(oil_detections)
                    if wet_floor_enabled:
                        wet_floor_detections = self.detector.filter_wet_floor_classes(floor_hazard_detections)
                        wet_floor_detections = self.wet_floor_tracker.update_with_detections(wet_floor_detections)

                    oil_violations, wet_floor_violations, floor_active_spatial = (
                        self.floor_hazard_monitor.detect_violations(
                            oil_detections,
                            wet_floor_detections,
                            oil_enabled=oil_enabled,
                            wet_floor_enabled=wet_floor_enabled,
                        )
                    )
                    hazard_active_spatial |= floor_active_spatial

                    oil_indices = []
                    if oil_enabled:
                        for violation_type, xyxy, confidence, track_id, _subtype in oil_violations:
                            observation = self.observations.process_hazard_detection(
                                frame=frame,
                                xyxy=xyxy,
                                violation_type=violation_type,
                                object_class="oil_spillage",
                                confidence_score=confidence,
                                track_id=track_id,
                            )
                            for idx in range(len(oil_detections)):
                                if track_id is not None and oil_detections.tracker_id[idx] == track_id:
                                    oil_indices.append(idx)
                                    break

                            logger.info(
                                "Oil spillage %s | %.0fs | severity=%s | %s",
                                observation.observation_id[:8],
                                observation.duration_seconds,
                                observation.severity,
                                observation.summary,
                            )
                            if on_observation:
                                on_observation(observation)

                    wet_floor_indices = []
                    if wet_floor_enabled:
                        for violation_type, xyxy, confidence, track_id, _subtype in wet_floor_violations:
                            observation = self.observations.process_hazard_detection(
                                frame=frame,
                                xyxy=xyxy,
                                violation_type=violation_type,
                                object_class="wet_floor",
                                confidence_score=confidence,
                                track_id=track_id,
                            )
                            for idx in range(len(wet_floor_detections)):
                                if track_id is not None and wet_floor_detections.tracker_id[idx] == track_id:
                                    wet_floor_indices.append(idx)
                                    break

                            logger.info(
                                "Wet floor %s | %.0fs | severity=%s | %s",
                                observation.observation_id[:8],
                                observation.duration_seconds,
                                observation.severity,
                                observation.summary,
                            )
                            if on_observation:
                                on_observation(observation)

                    if oil_indices:
                        oil_violation_detections = oil_detections[oil_indices]
                    if wet_floor_indices:
                        wet_floor_violation_detections = wet_floor_detections[wet_floor_indices]

                if low_visibility_enabled:
                    confirmed, confidence, xyxy, metrics = self.low_visibility_monitor.detect_violation(
                        frame,
                        enabled=True,
                    )
                    if metrics is not None:
                        low_visibility_score = metrics.visibility_score
                    if confirmed and xyxy is not None:
                        low_visibility_active = True
                        center_x = float((xyxy[0] + xyxy[2]) / 2.0)
                        center_y = float((xyxy[1] + xyxy[3]) / 2.0)
                        hazard_active_spatial.add(
                            (
                                self.low_visibility_monitor.violation_type(),
                                self.low_visibility_monitor.object_class(),
                                center_x,
                                center_y,
                            )
                        )
                        observation = self.observations.process_hazard_detection(
                            frame=frame,
                            xyxy=xyxy,
                            violation_type=self.low_visibility_monitor.violation_type(),
                            object_class=self.low_visibility_monitor.object_class(),
                            confidence_score=confidence,
                        )
                        logger.info(
                            "Low visibility | score=%.2f | confidence=%.2f | severity=%s",
                            metrics.visibility_score if metrics else 0.0,
                            confidence,
                            observation.severity,
                        )
                        if on_observation:
                            on_observation(observation)

                if (
                    fire_enabled
                    or smoke_enabled
                    or oil_enabled
                    or wet_floor_enabled
                    or low_visibility_enabled
                ):
                    self.observations.resolve_stale_hazard_detections(hazard_active_spatial)
                else:
                    self.observations.resolve_stale_hazard_detections(set())

                if self.show_preview or frame_callback:
                    preview = self._render_preview_frame(
                        frame,
                        static_detections=static_detections,
                        mobile_violation_detections=mobile_violation_detections,
                        phone_detections=phone_detections,
                        ppe_violation_detections=ppe_violation_detections,
                        fire_violation_detections=fire_violation_detections,
                        smoke_violation_detections=smoke_violation_detections,
                        oil_violation_detections=oil_violation_detections,
                        wet_floor_violation_detections=wet_floor_violation_detections,
                        object_fall_violation_detections=object_fall_violation_detections,
                        person_fall_violation_detections=person_fall_violation_detections,
                        near_miss_violation_detections=near_miss_violation_detections,
                        pathway_occupancy_detections=pathway_occupancy_detections,
                        cv_obstruction_boxes=cv_obstruction_boxes,
                        low_visibility_active=low_visibility_active,
                        low_visibility_score=low_visibility_score,
                    )
                    if frame_callback:
                        frame_callback(preview)
                    if self.show_preview:
                        if preview_imshow("CCTV Pathway Monitor", preview):
                            if preview_wait_key(1) == ord("q"):
                                break

        finally:
            self.capture.release()
            self.observations.close()
            if self.show_preview:
                preview_destroy_all()
        return frame_index
