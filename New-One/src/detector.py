from __future__ import annotations

import logging

import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.config import AppConfig

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, YOLO] = {}

# Focused YOLO-World prompts for tiled inference. A short list keeps the
# open-vocabulary head on floor-level industrial items instead of people/vehicles.
FLOOR_OBJECT_PROMPTS = [
    "metal plate",
    "steel plate",
    "floor plate",
    "metal sheet",
    "low pallet",
    "pallet",
    "wooden pallet",
    "metal pallet",
    "pipe",
    "pipe section",
    "valve",
    "barrier",
    "safety barrier",
    "container",
    "crate",
    "toolbox",
    "material on floor",
    "flat object",
    "object on floor",
    "obstruction",
    "industrial equipment",
    "metal frame",
    "yellow frame",
    "carton",
    "box",
]


def load_yolo_model(model_path: str) -> YOLO:
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached
    model = YOLO(model_path)
    _MODEL_CACHE[model_path] = model
    return model


class ObjectDetector:
    """YOLO object detector. Supports COCO (default) or YOLO-World (custom classes)."""

    def __init__(self, config: AppConfig):
        self.config = config
        model_path = config.resolve_model_path()
        self.model = load_yolo_model(model_path)
        self._world_mode = config.detection_mode == "world"

        if self._world_mode:
            world_classes = list(
                dict.fromkeys(
                    [
                        *config.custom_classes,
                        *FLOOR_OBJECT_PROMPTS,
                        *config.fire_classes,
                        *config.smoke_classes,
                    ]
                )
            )
            self._world_class_names = world_classes
            self.model.set_classes(world_classes)
            self.names = {i: name for i, name in enumerate(world_classes)}
            logger.info("YOLO-World mode with %d custom classes", len(world_classes))
        else:
            self._world_class_names = []
            self.names = self.model.names
            logger.info("COCO mode with %d classes", len(self.names))

        self._blocking_exclude_ids = self._resolve_blocking_exclude_ids()

    def _resolve_blocking_exclude_ids(self) -> set[int]:
        """Classes never treated as pathway obstructions (people, worn PPE, phones, fire/smoke)."""
        exclude_names = list(self.config.blocking_exclude_classes)
        exclude_names.extend(self.config.all_ppe_classes())
        exclude_names.extend(self.config.mobile_phone_classes)
        exclude_names.extend(self.config.fire_classes)
        exclude_names.extend(self.config.smoke_classes)
        deduped = list(dict.fromkeys(name.lower() for name in exclude_names))
        return self._resolve_exclude_ids(deduped)

    def _resolve_exclude_ids(self, exclude_names: list[str]) -> set[int]:
        exclude = set()
        lowered = {name.lower() for name in exclude_names}
        for class_id, name in self.names.items():
            if name.lower() in lowered:
                exclude.add(class_id)
        return exclude

    def detect_all(
        self,
        frame,
        confidence: float | None = None,
        rois: list[tuple[float, float, float, float]] | None = None,
    ) -> sv.Detections:
        conf = self.config.confidence if confidence is None else confidence
        detections = self._predict_frame(frame, conf, self.config.imgsz)
        if self.config.small_object_enabled and rois:
            tiled = self._predict_tiles(frame, conf, rois)
            detections = self._merge_keep_small(detections, tiled)
        return detections

    def _predict_frame(self, frame, confidence: float, imgsz: int) -> sv.Detections:
        results = self.model.predict(
            frame,
            conf=confidence,
            iou=self.config.iou,
            imgsz=imgsz,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(results)

    def _tile_windows(
        self,
        frame_shape: tuple[int, ...],
        rois: list[tuple[float, float, float, float]] | None,
    ) -> list[tuple[int, int, int, int]]:
        height, width = frame_shape[:2]
        tile = max(320, int(self.config.small_object_tile_size))
        overlap = min(0.5, max(0.05, float(self.config.small_object_tile_overlap)))
        step = max(64, int(tile * (1.0 - overlap)))
        max_tiles = max(1, int(self.config.small_object_max_tiles))

        regions: list[tuple[int, int, int, int]]
        if rois:
            regions = []
            pad = 48
            for x1, y1, x2, y2 in rois:
                rx1 = max(0, int(x1) - pad)
                ry1 = max(0, int(y1) - pad)
                rx2 = min(width, int(x2) + pad)
                ry2 = min(height, int(y2) + pad)
                if rx2 - rx1 >= 16 and ry2 - ry1 >= 16:
                    regions.append((rx1, ry1, rx2, ry2))
            if not regions:
                regions = [(0, 0, width, height)]
        else:
            regions = [(0, 0, width, height)]

        windows: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for rx1, ry1, rx2, ry2 in regions:
            region_w = rx2 - rx1
            region_h = ry2 - ry1
            if region_w <= tile and region_h <= tile:
                key = (rx1, ry1, rx2, ry2)
                if key not in seen:
                    seen.add(key)
                    windows.append(key)
                continue
            y = ry1
            while y < ry2:
                x = rx1
                y2 = min(ry2, y + tile)
                if y2 - y < tile and ry2 - tile >= ry1:
                    y = max(ry1, ry2 - tile)
                    y2 = ry2
                while x < rx2:
                    x2 = min(rx2, x + tile)
                    if x2 - x < tile and rx2 - tile >= rx1:
                        x = max(rx1, rx2 - tile)
                        x2 = rx2
                    key = (x, y, x2, y2)
                    if key not in seen:
                        seen.add(key)
                        windows.append(key)
                    if x2 >= rx2:
                        break
                    x += step
                if y2 >= ry2:
                    break
                y += step

        if len(windows) > max_tiles:
            windows = self._spread_windows(windows, max_tiles)
        return windows

    @staticmethod
    def _spread_windows(
        windows: list[tuple[int, int, int, int]],
        max_tiles: int,
    ) -> list[tuple[int, int, int, int]]:
        """Keep a spatially spread subset so distant floor objects are not dropped."""
        if len(windows) <= max_tiles:
            return windows

        centers = [
            ((win[0] + win[2]) / 2.0, (win[1] + win[3]) / 2.0) for win in windows
        ]
        areas = [
            (idx, (win[2] - win[0]) * (win[3] - win[1]))
            for idx, win in enumerate(windows)
        ]
        selected = [max(areas, key=lambda item: item[1])[0]]
        while len(selected) < max_tiles:
            best_idx = None
            best_dist = -1.0
            for idx in range(len(windows)):
                if idx in selected:
                    continue
                dist = min(
                    (centers[idx][0] - centers[other][0]) ** 2
                    + (centers[idx][1] - centers[other][1]) ** 2
                    for other in selected
                )
                if dist > best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
        return [windows[idx] for idx in selected]

    def _predict_tiles(
        self,
        frame,
        confidence: float,
        rois: list[tuple[float, float, float, float]] | None,
    ) -> sv.Detections:
        windows = self._tile_windows(frame.shape, rois)
        if not windows:
            return sv.Detections.empty()

        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
        if not crops or any(crop.size == 0 for crop in crops):
            windows = [
                window
                for window, crop in zip(windows, crops, strict=False)
                if crop.size > 0
            ]
            crops = [crop for crop in crops if crop.size > 0]
        if not crops:
            return sv.Detections.empty()

        used_floor_prompts = False
        if self._world_mode and self._world_class_names:
            self.model.set_classes(FLOOR_OBJECT_PROMPTS)
            used_floor_prompts = True
        try:
            results = self.model.predict(
                crops,
                conf=confidence,
                iou=max(0.35, self.config.iou - 0.10),
                imgsz=self.config.small_object_imgsz,
                verbose=False,
            )
        finally:
            if used_floor_prompts:
                self.model.set_classes(self._world_class_names)

        name_to_id = {name.lower(): class_id for class_id, name in self.names.items()}
        fallback_id = (
            name_to_id.get("object on floor")
            or name_to_id.get("obstruction")
            or next(iter(self.names), 0)
        )
        tile_names = (
            {idx: name for idx, name in enumerate(FLOOR_OBJECT_PROMPTS)}
            if used_floor_prompts
            else self.names
        )

        boxes: list[np.ndarray] = []
        scores: list[float] = []
        class_ids: list[int] = []
        min_h = max(1, int(self.config.small_object_min_height_px))
        wide_aspect = max(1.2, float(self.config.small_object_wide_aspect))
        for window, result in zip(windows, results, strict=False):
            ox, oy = window[0], window[1]
            tile_det = sv.Detections.from_ultralytics(result)
            for i in range(len(tile_det)):
                x1, y1, x2, y2 = map(float, tile_det.xyxy[i])
                width = max(x2 - x1, 1.0)
                height = max(y2 - y1, 1.0)
                if height < min_h and width / height < wide_aspect:
                    continue
                raw_class = int(tile_det.class_id[i]) if tile_det.class_id is not None else 0
                class_name = str(tile_names.get(raw_class, "object on floor")).lower()
                boxes.append(np.array([x1 + ox, y1 + oy, x2 + ox, y2 + oy], dtype=float))
                scores.append(
                    float(tile_det.confidence[i]) if tile_det.confidence is not None else 0.0
                )
                class_ids.append(int(name_to_id.get(class_name, fallback_id)))

        if not boxes:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.vstack(boxes),
            confidence=np.array(scores, dtype=float),
            class_id=np.array(class_ids, dtype=int),
        )

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _merge_keep_small(
        self,
        primary: sv.Detections,
        extra: sv.Detections,
        iou_thresh: float = 0.55,
    ) -> sv.Detections:
        if len(extra) == 0:
            return primary
        if len(primary) == 0:
            return extra

        keep: list[int] = []
        for i in range(len(extra)):
            box = extra.xyxy[i]
            extra_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            duplicate = False
            for j in range(len(primary)):
                iou = self._box_iou(box, primary.xyxy[j])
                if iou < iou_thresh:
                    continue
                primary_area = max(0.0, float(primary.xyxy[j][2] - primary.xyxy[j][0])) * max(
                    0.0, float(primary.xyxy[j][3] - primary.xyxy[j][1])
                )
                # Keep a distinctly smaller/low box even if it overlaps a large one.
                if extra_area > 0 and extra_area < primary_area * 0.45 and iou < 0.75:
                    continue
                duplicate = True
                break
            if not duplicate:
                keep.append(i)

        if not keep:
            return primary
        extra = extra[keep]
        return sv.Detections(
            xyxy=np.vstack([primary.xyxy, extra.xyxy]),
            confidence=(
                np.concatenate([primary.confidence, extra.confidence])
                if primary.confidence is not None and extra.confidence is not None
                else None
            ),
            class_id=np.concatenate([primary.class_id, extra.class_id]),
        )

    def detect_phones(self, frame) -> sv.Detections:
        """Extra-sensitive pass for small phones on CCTV."""
        detections = self.detect_all(frame, confidence=self.config.mobile_phone_confidence)
        return self.filter_phone_classes(detections)

    def detect_ppe(self, frame) -> sv.Detections:
        detections = self.detect_all(frame, confidence=self.config.ppe_confidence)
        return self.filter_ppe_classes(detections)

    def detect_fire_smoke(self, frame) -> sv.Detections:
        detections = self.detect_all(frame, confidence=self.config.fire_smoke_confidence)
        fire_smoke_terms = list(
            dict.fromkeys([*self.config.fire_classes, *self.config.smoke_classes])
        )
        return self.filter_by_terms(detections, fire_smoke_terms)

    def detect_floor_hazards(self, frame) -> sv.Detections:
        return self.detect_all(frame, confidence=self.config.floor_hazard_confidence)

    def filter_oil_spillage_classes(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_terms(detections, self.config.oil_spillage_classes)

    def filter_wet_floor_classes(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_terms(detections, self.config.wet_floor_classes)

    def filter_fire_classes(self, detections: sv.Detections) -> sv.Detections:
        return self._filter_by_terms(detections, self.config.fire_classes)

    def filter_smoke_classes(self, detections: sv.Detections) -> sv.Detections:
        return self._filter_by_terms(detections, self.config.smoke_classes)

    def filter_helmet_classes(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_terms(detections, self._classes_for_ppe_item("helmet"))

    def filter_vest_classes(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_terms(detections, self._classes_for_ppe_item("vest"))

    def filter_ppe_item(self, detections: sv.Detections, item_key: str) -> sv.Detections:
        return self.filter_by_terms(detections, self._classes_for_ppe_item(item_key))

    def _classes_for_ppe_item(self, item_key: str) -> list[str]:
        for item in self.config.ppe_items:
            if item.key == item_key:
                return item.classes
        return []

    def detect_blocking(self, frame) -> sv.Detections:
        """Sensitive pass for pathway obstructions (boxes, carts, machinery)."""
        blocking_confidence = min(self.config.confidence, 0.20)
        detections = self.detect_all(frame, confidence=blocking_confidence)
        return self.exclude_classes(detections, self._blocking_exclude_ids)

    def exclude_classes(self, detections: sv.Detections, exclude_ids: set[int]) -> sv.Detections:
        if len(detections) == 0 or not exclude_ids:
            return detections

        keep = [
            i for i, class_id in enumerate(detections.class_id)
            if int(class_id) not in exclude_ids
        ]
        return detections[keep]

    def filter_by_class_names(
        self,
        detections: sv.Detections,
        class_names: list[str],
    ) -> sv.Detections:
        if len(detections) == 0:
            return detections

        wanted = {name.lower() for name in class_names}
        keep = [
            i
            for i, class_id in enumerate(detections.class_id)
            if self.class_name(int(class_id)).lower() in wanted
        ]
        return detections[keep]

    def filter_phone_classes(self, detections: sv.Detections) -> sv.Detections:
        return self._filter_by_terms(detections, self.config.mobile_phone_classes)

    def filter_ppe_classes(self, detections: sv.Detections) -> sv.Detections:
        return self.filter_by_terms(detections, self.config.all_ppe_classes())

    def filter_by_terms(self, detections: sv.Detections, terms: list[str]) -> sv.Detections:
        return self._filter_by_terms(detections, terms)

    def _filter_by_terms(self, detections: sv.Detections, terms: list[str]) -> sv.Detections:
        if len(detections) == 0:
            return detections

        wanted = [term.lower() for term in terms]
        keep = [
            i
            for i, class_id in enumerate(detections.class_id)
            if self._matches_term(self.class_name(int(class_id)), wanted)
        ]
        return detections[keep]

    @staticmethod
    def _matches_term(name: str, terms: list[str]) -> bool:
        lowered = name.lower()
        return any(term in lowered or lowered in term for term in terms)

    def class_name(self, class_id: int) -> str:
        return self.names.get(int(class_id), "unknown")
