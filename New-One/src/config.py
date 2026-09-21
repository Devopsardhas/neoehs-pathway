from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.fire_classification import DEFAULT_FIRE_TYPES, FireType
from src.ppe_items import PPEItemConfig, load_ppe_items


@dataclass
class CameraConfig:
    id: str
    name: str
    rtsp_url: str
    role: str = "monitor"
    gate_id: str = "main_gate"
    gate_name: str = "Main Gate"


@dataclass
class AppConfig:
    rtsp_url: str
    reconnect_delay_sec: int
    frame_skip: int
    max_read_failures: int
    model: str
    detection_mode: str
    custom_classes: list[str]
    confidence: float
    iou: float
    imgsz: int
    small_object_enabled: bool
    small_object_tile_size: int
    small_object_tile_overlap: float
    small_object_imgsz: int
    small_object_max_tiles: int
    small_object_min_height_px: int
    small_object_wide_aspect: float
    blocking_exclude_classes: list[str]
    track_activation_threshold: float
    lost_track_buffer: int
    minimum_matching_threshold: float
    frame_rate: int
    history_frames: int
    max_displacement_px: float
    min_static_frames: int
    pathway_config_file: str
    pathway_min_overlap_ratio: float
    pathway_min_overlap_area_px: float
    pathway_use_foot_point: bool
    db_path: str
    capture_dir: str
    spatial_match_radius_px: float
    min_duration_for_update_sec: float
    brief_threshold_sec: int
    moderate_threshold_sec: int
    pathway_block_enabled: bool
    pathway_block_min_duration_sec: float
    pathway_block_cv_enabled: bool
    pathway_block_cv_diff_threshold: int
    pathway_block_cv_min_area_px: float
    pathway_block_cv_min_width_px: int
    pathway_block_cv_min_height_px: int
    pathway_block_cv_blur_kernel: int
    pathway_block_cv_morph_kernel: int
    pathway_block_cv_history_frames: int
    pathway_block_cv_min_static_frames: int
    pathway_block_cv_max_displacement_px: float
    pathway_block_cv_match_distance_px: float
    pathway_block_cv_person_padding_px: float
    pathway_block_confidence: float
    pathway_block_include_moving_large: bool
    pathway_block_large_object_area_px: float
    mobile_usage_enabled: bool
    mobile_person_class: str
    mobile_phone_classes: list[str]
    mobile_phone_confidence: float
    mobile_person_phone_distance_px: float
    mobile_person_phone_padding_px: float
    mobile_upper_body_ratio: float
    mobile_person_min_overlap: float
    mobile_allow_phone_in_zone_match: bool
    mobile_min_usage_frames: int
    mobile_phone_miss_frames: int
    mobile_pathway_miss_frames: int
    mobile_resolve_grace_sec: float
    ppe_enabled: bool
    ppe_person_class: str
    ppe_items: list[PPEItemConfig]
    ppe_item_enabled: dict[str, bool]
    ppe_confidence: float
    ppe_match_distance_px: float
    ppe_match_padding_px: float
    ppe_person_min_overlap: float
    ppe_min_violation_frames: int
    ppe_resolve_grace_sec: float
    fire_smoke_enabled: bool
    fire_detection_enabled: bool
    smoke_detection_enabled: bool
    fire_types: list[FireType]
    fire_classes: list[str]
    smoke_classes: list[str]
    fire_smoke_confidence: float
    fire_min_detection_frames: int
    fire_streak_miss_frames: int
    smoke_min_detection_frames: int
    smoke_trusted_confidence: float
    smoke_trusted_min_frames: int
    smoke_merge_distance_px: float
    smoke_person_max_overlap: float
    smoke_temporal_min_drift_px: float
    smoke_temporal_min_texture_std: float
    smoke_temporal_static_max_drift_px: float
    smoke_temporal_min_jitter_px: float
    fire_smoke_resolve_grace_sec: float
    fire_smoke_spatial_bucket_px: float
    fire_merge_distance_px: float
    fire_min_confidence: float
    fire_trusted_confidence: float
    fire_trusted_min_frames: int
    fire_temporal_min_frames: int
    fire_temporal_min_area_px: float
    fire_temporal_min_area_ratio: float
    fire_temporal_min_density_std: float
    fire_temporal_static_max_drift_px: float
    fire_temporal_bulk_motion_min_drift_px: float
    fire_temporal_min_jitter_px: float
    fire_temporal_flame_density_min: float
    fire_temporal_flame_brightness_min: float
    fire_temporal_flame_value_std_min: float
    fire_temporal_flat_density_min: float
    fire_immediate_high: bool
    smoke_immediate_high: bool
    fire_smoke_cv_fallback_enabled: bool
    fire_smoke_cv_fire_fallback_enabled: bool
    fire_smoke_cv_smoke_fallback_enabled: bool
    fire_smoke_cv_fire_min_area: int
    fire_smoke_cv_smoke_min_area: int
    fire_smoke_cv_fire_confidence: float
    fire_smoke_cv_smoke_confidence: float
    fire_smoke_cv_smoke_max_saturation: int
    fire_smoke_cv_smoke_min_brightness: int
    fire_smoke_cv_smoke_max_brightness: int
    fire_smoke_cv_smoke_max_area_ratio: float
    fire_smoke_cv_smoke_min_texture_std: float
    fire_smoke_cv_smoke_max_edge_ratio: float
    fire_smoke_cv_smoke_max_mask_fill: float
    fire_smoke_cv_smoke_max_solidity: float
    fire_smoke_cv_smoke_max_rectangularity: float
    fire_smoke_cv_smoke_max_aspect_ratio: float
    floor_hazard_enabled: bool
    oil_spillage_enabled: bool
    wet_floor_enabled: bool
    oil_spillage_classes: list[str]
    wet_floor_classes: list[str]
    floor_hazard_confidence: float
    oil_spillage_min_detection_frames: int
    wet_floor_min_detection_frames: int
    floor_hazard_resolve_grace_sec: float
    floor_hazard_spatial_bucket_px: float
    oil_spillage_immediate_high: bool
    wet_floor_immediate_high: bool
    low_visibility_enabled: bool
    low_visibility_min_detection_frames: int
    low_visibility_resolve_grace_sec: float
    low_visibility_score_threshold: float
    low_visibility_laplacian_reference: float
    low_visibility_contrast_reference: float
    low_visibility_edge_reference: float
    low_visibility_brightness_min: float
    low_visibility_brightness_max: float
    low_visibility_sample_stride_px: int
    low_visibility_immediate_high: bool
    object_fall_enabled: bool
    object_fall_detect_person: bool
    object_fall_detect_object: bool
    object_fall_history_frames: int
    object_fall_min_distance_px: float
    object_fall_min_velocity_px: float
    object_fall_min_frames: int
    object_fall_recent_frames: int
    object_fall_recent_min_distance_px: float
    object_fall_sudden_drop_px: float
    object_fall_resolve_grace_sec: float
    object_fall_person_min_overlap: float
    object_fall_immediate_high: bool
    near_miss_enabled: bool
    near_miss_person_class: str
    near_miss_hazard_classes: list[str]
    near_miss_max_distance_px: float
    near_miss_min_frames: int
    near_miss_resolve_grace_sec: float
    near_miss_person_min_overlap: float
    near_miss_immediate_high: bool
    camera_id: str
    cameras: list[CameraConfig]
    web_host: str
    web_port: int
    web_secret_key: str
    web_users: dict[str, str]
    web_live_preview_enabled: bool
    web_live_preview_persist: bool
    gate_vehicle_enabled: bool
    gate_vehicle_classes: list[str]
    gate_plate_classes: list[str]
    gate_brand_classes: list[str]
    gate_vehicle_confidence: float
    gate_plate_confidence: float
    gate_brand_confidence: float
    gate_min_confirm_frames: int
    gate_duplicate_cooldown_sec: float
    gate_plate_match_distance_px: float
    gate_vehicle_db_path: str
    gate_vehicle_capture_dir: str
    gate_ocr_enabled: bool
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    def is_ppe_item_enabled(self, item_key: str) -> bool:
        return bool(self.ppe_item_enabled.get(item_key, False))

    def enabled_ppe_items(self) -> list[PPEItemConfig]:
        return [item for item in self.ppe_items if self.is_ppe_item_enabled(item.key)]

    def all_ppe_classes(self) -> list[str]:
        classes: list[str] = []
        for item in self.ppe_items:
            classes.extend(item.classes)
        return classes

    def get_camera(self, camera_id: str) -> CameraConfig:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        raise KeyError(f"Unknown camera: {camera_id}")

    def with_camera(self, camera: CameraConfig) -> "AppConfig":
        from dataclasses import replace

        return replace(self, camera_id=camera.id, rtsp_url=camera.rtsp_url)

    def gate_cameras(self) -> list[CameraConfig]:
        return [camera for camera in self.cameras if camera.role in {"gate_in", "gate_out"}]

    def is_gate_camera(self, camera_id: str) -> bool:
        try:
            camera = self.get_camera(camera_id)
        except KeyError:
            return False
        return camera.role in {"gate_in", "gate_out"}

    def resolve_model_path(self) -> str:
        raw = Path(self.model)
        if raw.is_file():
            return str(raw.resolve())
        project_model = self.project_root / self.model
        if project_model.is_file():
            return str(project_model.resolve())
        return self.model

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "AppConfig":
        root = Path(__file__).resolve().parents[1]
        path = Path(config_path) if config_path else root / "config.yaml"
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        rtsp = raw["rtsp"]
        detection = raw["detection"]
        small_object = detection.get("small_object", {})
        tracking = raw["tracking"]
        static_filter = raw["static_filter"]
        pathway = raw["pathway"]
        observations = raw["observations"]
        summary = raw["summary"]
        mobile = raw.get("mobile_usage", {})
        pathway_block = raw.get("pathway_block", {})
        ppe = raw.get("ppe_violation", {})
        fire_smoke = raw.get("fire_smoke", {})
        floor_hazard = raw.get("floor_hazard", {})
        low_visibility = raw.get("low_visibility", {})
        object_fall = raw.get("object_fall", {})
        near_miss = raw.get("near_miss", {})
        gate_vehicle = raw.get("gate_vehicle", {})
        camera = raw.get("camera", {})
        web = raw.get("web", {})
        cameras_raw = raw.get("cameras")
        if cameras_raw:
            cameras = [
                CameraConfig(
                    id=entry["id"],
                    name=entry.get("name", entry["id"]),
                    rtsp_url=entry["rtsp_url"],
                    role=entry.get("role", "monitor"),
                    gate_id=entry.get("gate_id", "main_gate"),
                    gate_name=entry.get("gate_name", entry.get("name", entry["id"])),
                )
                for entry in cameras_raw
            ]
        else:
            cameras = [
                CameraConfig(
                    id=camera.get("id", "camera_1"),
                    name=camera.get("name", "Camera 1"),
                    rtsp_url=rtsp["url"],
                )
            ]

        ppe_items = load_ppe_items(ppe)
        return cls(
            rtsp_url=rtsp["url"],
            reconnect_delay_sec=rtsp["reconnect_delay_sec"],
            frame_skip=rtsp["frame_skip"],
            max_read_failures=rtsp.get("max_read_failures", 8),
            model=detection["model"],
            detection_mode=detection.get("mode", "coco").lower(),
            custom_classes=detection.get("custom_classes", []),
            confidence=detection["confidence"],
            iou=detection["iou"],
            imgsz=detection.get("imgsz", 640),
            small_object_enabled=bool(small_object.get("enabled", True)),
            small_object_tile_size=int(small_object.get("tile_size", 640)),
            small_object_tile_overlap=float(small_object.get("tile_overlap", 0.25)),
            small_object_imgsz=int(small_object.get("imgsz", 640)),
            small_object_max_tiles=int(small_object.get("max_tiles", 12)),
            small_object_min_height_px=int(small_object.get("min_height_px", 6)),
            small_object_wide_aspect=float(small_object.get("wide_aspect", 2.2)),
            blocking_exclude_classes=[
                c.lower() for c in detection.get("blocking_exclude_classes", ["person"])
            ],
            track_activation_threshold=tracking["track_activation_threshold"],
            lost_track_buffer=tracking["lost_track_buffer"],
            minimum_matching_threshold=tracking["minimum_matching_threshold"],
            frame_rate=tracking["frame_rate"],
            history_frames=static_filter["history_frames"],
            max_displacement_px=static_filter["max_displacement_px"],
            min_static_frames=static_filter["min_static_frames"],
            pathway_config_file=pathway["config_file"],
            pathway_min_overlap_ratio=pathway.get("min_overlap_ratio", 0.12),
            pathway_min_overlap_area_px=pathway.get("min_overlap_area_px", 500),
            pathway_use_foot_point=pathway.get("use_foot_point", True),
            db_path=observations["db_path"],
            capture_dir=observations["capture_dir"],
            spatial_match_radius_px=observations["spatial_match_radius_px"],
            min_duration_for_update_sec=observations["min_duration_for_update_sec"],
            brief_threshold_sec=summary["brief_threshold_sec"],
            moderate_threshold_sec=summary["moderate_threshold_sec"],
            pathway_block_enabled=pathway_block.get("enabled", True),
            pathway_block_min_duration_sec=float(pathway_block.get("min_duration_sec", 5)),
            pathway_block_cv_enabled=pathway_block.get("cv_fallback_enabled", True),
            pathway_block_cv_diff_threshold=int(pathway_block.get("cv_diff_threshold", 22)),
            pathway_block_cv_min_area_px=float(pathway_block.get("cv_min_area_px", 2500)),
            pathway_block_cv_min_width_px=int(pathway_block.get("cv_min_width_px", 35)),
            pathway_block_cv_min_height_px=int(pathway_block.get("cv_min_height_px", 35)),
            pathway_block_cv_blur_kernel=int(pathway_block.get("cv_blur_kernel", 5)),
            pathway_block_cv_morph_kernel=int(pathway_block.get("cv_morph_kernel", 5)),
            pathway_block_cv_history_frames=int(pathway_block.get("cv_history_frames", 20)),
            pathway_block_cv_min_static_frames=int(pathway_block.get("cv_min_static_frames", 4)),
            pathway_block_cv_max_displacement_px=float(pathway_block.get("cv_max_displacement_px", 18)),
            pathway_block_cv_match_distance_px=float(pathway_block.get("cv_match_distance_px", 90)),
            pathway_block_cv_person_padding_px=float(pathway_block.get("cv_person_padding_px", 35)),
            pathway_block_confidence=float(pathway_block.get("confidence", 0.12)),
            pathway_block_include_moving_large=bool(
                pathway_block.get("include_moving_large_objects", True)
            ),
            pathway_block_large_object_area_px=float(
                pathway_block.get("large_object_area_px", 12_000)
            ),
            mobile_usage_enabled=mobile.get("enabled", True),
            mobile_person_class=mobile.get("person_class", "person"),
            mobile_phone_classes=mobile.get(
                "phone_classes",
                ["cell phone", "mobile phone", "smartphone", "phone"],
            ),
            mobile_phone_confidence=mobile.get("phone_confidence", 0.15),
            mobile_person_phone_distance_px=mobile.get("person_phone_distance_px", 180),
            mobile_person_phone_padding_px=mobile.get("person_phone_padding_px", 50),
            mobile_upper_body_ratio=mobile.get("upper_body_ratio", 0.65),
            mobile_person_min_overlap=mobile.get("person_min_overlap", 0.03),
            mobile_allow_phone_in_zone_match=mobile.get("allow_phone_in_zone_match", True),
            mobile_min_usage_frames=mobile.get("min_usage_frames", 5),
            mobile_phone_miss_frames=mobile.get("phone_miss_frames", 4),
            mobile_pathway_miss_frames=mobile.get("pathway_miss_frames", 4),
            mobile_resolve_grace_sec=mobile.get("resolve_grace_sec", 8.0),
            ppe_enabled=ppe.get("enabled", True),
            ppe_person_class=ppe.get("person_class", "person"),
            ppe_items=ppe_items,
            ppe_item_enabled={item.key: item.default_enabled for item in ppe_items},
            ppe_confidence=ppe.get("confidence", 0.20),
            ppe_match_distance_px=ppe.get("match_distance_px", 120),
            ppe_match_padding_px=ppe.get("match_padding_px", 40),
            ppe_person_min_overlap=ppe.get("person_min_overlap", 0.03),
            ppe_min_violation_frames=ppe.get("min_violation_frames", 5),
            ppe_resolve_grace_sec=ppe.get("resolve_grace_sec", 3.0),
            fire_smoke_enabled=fire_smoke.get("enabled", True),
            fire_detection_enabled=fire_smoke.get(
                "fire_enabled",
                fire_smoke.get("enabled", True),
            ),
            smoke_detection_enabled=fire_smoke.get(
                "smoke_enabled",
                fire_smoke.get("enabled", True),
            ),
            fire_types=cls._load_fire_types(fire_smoke),
            fire_classes=fire_smoke.get("fire_classes", ["fire", "flame"]),
            smoke_classes=fire_smoke.get("smoke_classes", ["smoke", "smoke plume"]),
            fire_smoke_confidence=fire_smoke.get("confidence", 0.20),
            fire_min_detection_frames=fire_smoke.get("fire_min_detection_frames", 3),
            fire_streak_miss_frames=fire_smoke.get("streak_miss_frames", 4),
            smoke_min_detection_frames=fire_smoke.get("smoke_min_detection_frames", 5),
            smoke_trusted_confidence=fire_smoke.get("smoke_trusted_confidence", 0.40),
            smoke_trusted_min_frames=fire_smoke.get("smoke_trusted_min_frames", 3),
            smoke_merge_distance_px=fire_smoke.get("smoke_merge_distance_px", 150),
            smoke_person_max_overlap=fire_smoke.get("smoke_person_max_overlap", 0.20),
            smoke_temporal_min_drift_px=fire_smoke.get("smoke_temporal_min_drift_px", 6.0),
            smoke_temporal_min_texture_std=fire_smoke.get("smoke_temporal_min_texture_std", 0.008),
            smoke_temporal_static_max_drift_px=fire_smoke.get("smoke_temporal_static_max_drift_px", 8),
            smoke_temporal_min_jitter_px=fire_smoke.get("smoke_temporal_min_jitter_px", 1.5),
            fire_smoke_resolve_grace_sec=fire_smoke.get("resolve_grace_sec", 5.0),
            fire_smoke_spatial_bucket_px=fire_smoke.get("spatial_bucket_px", 80),
            fire_merge_distance_px=fire_smoke.get("fire_merge_distance_px", 200),
            fire_min_confidence=fire_smoke.get("fire_min_confidence", 0.30),
            fire_trusted_confidence=fire_smoke.get("fire_trusted_confidence", 0.40),
            fire_trusted_min_frames=fire_smoke.get("fire_trusted_min_frames", 3),
            fire_temporal_min_frames=fire_smoke.get("fire_temporal_min_frames", 3),
            fire_temporal_min_area_px=fire_smoke.get("fire_temporal_min_area_px", 250),
            fire_temporal_min_area_ratio=fire_smoke.get("fire_temporal_min_area_ratio", 0.0002),
            fire_temporal_min_density_std=fire_smoke.get("fire_temporal_min_density_std", 0.012),
            fire_temporal_static_max_drift_px=fire_smoke.get("fire_temporal_static_max_drift_px", 5),
            fire_temporal_bulk_motion_min_drift_px=fire_smoke.get(
                "fire_temporal_bulk_motion_min_drift_px", 55
            ),
            fire_temporal_min_jitter_px=fire_smoke.get("fire_temporal_min_jitter_px", 1.5),
            fire_temporal_flame_density_min=fire_smoke.get("fire_temporal_flame_density_min", 0.14),
            fire_temporal_flame_brightness_min=fire_smoke.get(
                "fire_temporal_flame_brightness_min", 15.0
            ),
            fire_temporal_flame_value_std_min=fire_smoke.get(
                "fire_temporal_flame_value_std_min", 38.0
            ),
            fire_temporal_flat_density_min=fire_smoke.get("fire_temporal_flat_density_min", 0.42),
            fire_immediate_high=fire_smoke.get("fire_immediate_high", True),
            smoke_immediate_high=fire_smoke.get("smoke_immediate_high", True),
            fire_smoke_cv_fallback_enabled=fire_smoke.get("cv_fallback_enabled", True),
            fire_smoke_cv_fire_fallback_enabled=fire_smoke.get(
                "cv_fire_fallback_enabled",
                fire_smoke.get("cv_fallback_enabled", False),
            ),
            fire_smoke_cv_smoke_fallback_enabled=fire_smoke.get(
                "cv_smoke_fallback_enabled",
                True,
            ),
            fire_smoke_cv_fire_min_area=fire_smoke.get("cv_fire_min_area", 900),
            fire_smoke_cv_smoke_min_area=fire_smoke.get("cv_smoke_min_area", 2500),
            fire_smoke_cv_fire_confidence=fire_smoke.get("cv_fire_confidence", 0.35),
            fire_smoke_cv_smoke_confidence=fire_smoke.get("cv_smoke_confidence", 0.35),
            fire_smoke_cv_smoke_max_saturation=fire_smoke.get("cv_smoke_max_saturation", 45),
            fire_smoke_cv_smoke_min_brightness=fire_smoke.get("cv_smoke_min_brightness", 70),
            fire_smoke_cv_smoke_max_brightness=fire_smoke.get("cv_smoke_max_brightness", 210),
            fire_smoke_cv_smoke_max_area_ratio=fire_smoke.get("cv_smoke_max_area_ratio", 0.10),
            fire_smoke_cv_smoke_min_texture_std=fire_smoke.get("cv_smoke_min_texture_std", 35.0),
            fire_smoke_cv_smoke_max_edge_ratio=fire_smoke.get("cv_smoke_max_edge_ratio", 0.20),
            fire_smoke_cv_smoke_max_mask_fill=fire_smoke.get("cv_smoke_max_mask_fill", 0.50),
            fire_smoke_cv_smoke_max_solidity=fire_smoke.get("cv_smoke_max_solidity", 0.65),
            fire_smoke_cv_smoke_max_rectangularity=fire_smoke.get("cv_smoke_max_rectangularity", 0.50),
            fire_smoke_cv_smoke_max_aspect_ratio=fire_smoke.get("cv_smoke_max_aspect_ratio", 4.0),
            floor_hazard_enabled=floor_hazard.get("enabled", True),
            oil_spillage_enabled=floor_hazard.get(
                "oil_spillage_enabled",
                floor_hazard.get("enabled", True),
            ),
            wet_floor_enabled=floor_hazard.get(
                "wet_floor_enabled",
                floor_hazard.get("enabled", True),
            ),
            oil_spillage_classes=floor_hazard.get(
                "oil_classes",
                ["oil spill", "oil spillage", "oil puddle", "oil leak", "spilled oil"],
            ),
            wet_floor_classes=floor_hazard.get(
                "wet_floor_classes",
                ["wet floor", "water spill", "puddle", "wet surface", "slippery floor"],
            ),
            floor_hazard_confidence=floor_hazard.get("confidence", 0.20),
            oil_spillage_min_detection_frames=floor_hazard.get("oil_min_detection_frames", 4),
            wet_floor_min_detection_frames=floor_hazard.get("wet_floor_min_detection_frames", 4),
            floor_hazard_resolve_grace_sec=floor_hazard.get("resolve_grace_sec", 5.0),
            floor_hazard_spatial_bucket_px=floor_hazard.get("spatial_bucket_px", 80),
            oil_spillage_immediate_high=floor_hazard.get("oil_immediate_high", False),
            wet_floor_immediate_high=floor_hazard.get("wet_floor_immediate_high", False),
            low_visibility_enabled=low_visibility.get("enabled", True),
            low_visibility_min_detection_frames=low_visibility.get("min_detection_frames", 8),
            low_visibility_resolve_grace_sec=low_visibility.get("resolve_grace_sec", 8.0),
            low_visibility_score_threshold=low_visibility.get("score_threshold", 0.42),
            low_visibility_laplacian_reference=low_visibility.get("laplacian_reference", 120.0),
            low_visibility_contrast_reference=low_visibility.get("contrast_reference", 45.0),
            low_visibility_edge_reference=low_visibility.get("edge_reference", 0.045),
            low_visibility_brightness_min=low_visibility.get("brightness_min", 25.0),
            low_visibility_brightness_max=low_visibility.get("brightness_max", 220.0),
            low_visibility_sample_stride_px=low_visibility.get("sample_stride_px", 4),
            low_visibility_immediate_high=low_visibility.get("immediate_high", False),
            object_fall_enabled=object_fall.get("enabled", True),
            object_fall_detect_person=object_fall.get("detect_person_fall", True),
            object_fall_detect_object=object_fall.get("detect_object_fall", True),
            object_fall_history_frames=object_fall.get("history_frames", 8),
            object_fall_min_distance_px=object_fall.get("min_fall_distance_px", 40),
            object_fall_min_velocity_px=object_fall.get("min_fall_velocity_px", 8),
            object_fall_min_frames=object_fall.get("min_fall_frames", 3),
            object_fall_recent_frames=object_fall.get("recent_frames", 4),
            object_fall_recent_min_distance_px=object_fall.get("recent_min_fall_distance_px", 24),
            object_fall_sudden_drop_px=object_fall.get("sudden_drop_px", 18),
            object_fall_resolve_grace_sec=object_fall.get("resolve_grace_sec", 5.0),
            object_fall_person_min_overlap=object_fall.get("person_min_overlap", 0.03),
            object_fall_immediate_high=object_fall.get("immediate_high", True),
            near_miss_enabled=near_miss.get("enabled", True),
            near_miss_person_class=near_miss.get("person_class", "person"),
            near_miss_hazard_classes=near_miss.get(
                "hazard_classes",
                ["forklift", "truck", "cart", "trolley", "vehicle", "machine", "equipment"],
            ),
            near_miss_max_distance_px=near_miss.get("max_distance_px", 120),
            near_miss_min_frames=near_miss.get("min_frames", 4),
            near_miss_resolve_grace_sec=near_miss.get("resolve_grace_sec", 5.0),
            near_miss_person_min_overlap=near_miss.get("person_min_overlap", 0.03),
            near_miss_immediate_high=near_miss.get("immediate_high", True),
            camera_id=camera.get("id", cameras[0].id),
            cameras=cameras,
            web_host=web.get("host", "0.0.0.0"),
            web_port=web.get("port", 8080),
            web_secret_key=web.get("secret_key", "change-this-secret-key"),
            web_users=web.get("users", {"admin": "admin123"}),
            web_live_preview_enabled=web.get("live_preview_enabled", True),
            web_live_preview_persist=web.get("live_preview_persist", False),
            gate_vehicle_enabled=gate_vehicle.get("enabled", True),
            gate_vehicle_classes=gate_vehicle.get(
                "vehicle_classes",
                ["car", "truck", "bus", "van", "motorcycle", "vehicle", "auto"],
            ),
            gate_plate_classes=gate_vehicle.get(
                "plate_classes",
                ["license plate", "number plate", "registration plate", "licence plate"],
            ),
            gate_brand_classes=gate_vehicle.get(
                "brand_classes",
                [
                    "toyota", "honda", "hyundai", "maruti", "suzuki", "tata", "mahindra",
                    "ford", "bmw", "mercedes", "audi", "kia", "nissan", "volkswagen",
                    "skoda", "renault", "chevrolet", "jeep", "isuzu", "ashok leyland",
                ],
            ),
            gate_vehicle_confidence=gate_vehicle.get("vehicle_confidence", 0.30),
            gate_plate_confidence=gate_vehicle.get("plate_confidence", 0.25),
            gate_brand_confidence=gate_vehicle.get("brand_confidence", 0.22),
            gate_min_confirm_frames=gate_vehicle.get("min_confirm_frames", 3),
            gate_duplicate_cooldown_sec=gate_vehicle.get("duplicate_cooldown_sec", 45.0),
            gate_plate_match_distance_px=gate_vehicle.get("plate_match_distance_px", 220.0),
            gate_vehicle_db_path=gate_vehicle.get("db_path", "data/vehicles.db"),
            gate_vehicle_capture_dir=gate_vehicle.get("capture_dir", "captures/vehicles"),
            gate_ocr_enabled=gate_vehicle.get("ocr_enabled", True),
            project_root=root,
        )

    def resolve(self, relative: str) -> Path:
        return self.project_root / relative

    @staticmethod
    def _load_fire_types(fire_smoke: dict[str, Any]) -> list[FireType]:
        raw_types = fire_smoke.get("fire_types")
        if not raw_types:
            legacy_classes = fire_smoke.get("fire_classes", ["fire", "flame"])
            return [
                FireType(
                    key="structural_fire",
                    label="Structural Fire",
                    classes=tuple(legacy_classes),
                    alert=True,
                    immediate_high=fire_smoke.get("fire_immediate_high", True),
                ),
                *DEFAULT_FIRE_TYPES[1:],
            ]

        fire_types: list[FireType] = []
        items = raw_types.items() if isinstance(raw_types, dict) else enumerate(raw_types)
        for key, cfg in items:
            if isinstance(cfg, str):
                continue
            type_key = key if isinstance(key, str) else cfg.get("key", f"fire_type_{key}")
            fire_types.append(
                FireType(
                    key=type_key,
                    label=cfg.get("label", type_key.replace("_", " ").title()),
                    classes=tuple(cfg.get("classes", [])),
                    alert=cfg.get("alert", True),
                    immediate_high=cfg.get("immediate_high", False),
                )
            )
        return fire_types or list(DEFAULT_FIRE_TYPES)
