from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from src.plate_format import choose_best_plate, parse_and_correct_plate

logger = logging.getLogger(__name__)

_PLATE_PATTERN = re.compile(r"[A-Z0-9]{4,12}")
_OCR_ENGINE = None
_OCR_UNAVAILABLE = False


def normalize_plate(text: str) -> str | None:
    """Normalize OCR text to a canonical plate string when possible."""
    corrected = parse_and_correct_plate(text)
    if corrected:
        return corrected

    cleaned = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    if len(cleaned) < 4:
        return None
    if not _PLATE_PATTERN.fullmatch(cleaned):
        return None
    return cleaned


def _resize_plate_crop(crop: np.ndarray, scale: float = 3.0) -> np.ndarray:
    height = max(72, int(crop.shape[0] * scale))
    width = max(180, int(crop.shape[1] * scale))
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)


def _preprocess_variants(crop: np.ndarray) -> list[np.ndarray]:
    if crop.size == 0:
        return []

    enlarged = _resize_plate_crop(crop)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(enhanced, -1, kernel)
    _, sharp_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [enhanced, otsu, otsu_inv, adaptive, sharp_otsu]


def _load_ocr_engine():
    global _OCR_ENGINE, _OCR_UNAVAILABLE
    if _OCR_ENGINE is not None or _OCR_UNAVAILABLE:
        return _OCR_ENGINE
    try:
        import easyocr

        _OCR_ENGINE = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR loaded for license plate recognition")
    except Exception:
        logger.warning(
            "EasyOCR unavailable — install with: pip install easyocr. "
            "Gate OCR will use basic fallback only."
        )
        _OCR_UNAVAILABLE = True
    return _OCR_ENGINE


def _run_ocr_on_variants(reader, crop: np.ndarray) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    digit_allowlist = "0123456789"

    for variant in _preprocess_variants(crop):
        try:
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist=allowlist,
            )
        except Exception:
            continue

        for _bbox, text, confidence in results:
            if text:
                candidates.append((text, float(confidence)))

        # Digits-only pass reduces S/5, O/0 confusion in numeric plates.
        try:
            digit_results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist=digit_allowlist,
            )
            for _bbox, text, confidence in digit_results:
                if text and len(text) >= 4:
                    candidates.append((text, float(confidence) + 0.05))
        except Exception:
            pass

    # Raw color crop occasionally preserves thin strokes better than binarized images.
    try:
        enlarged = _resize_plate_crop(crop)
        results = reader.readtext(
            enlarged,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
        )
        for _bbox, text, confidence in results:
            if text:
                candidates.append((text, float(confidence)))

        digit_results = reader.readtext(
            enlarged,
            detail=1,
            paragraph=False,
            allowlist=digit_allowlist,
        )
        for _bbox, text, confidence in digit_results:
            if text and len(text) >= 4:
                candidates.append((text, float(confidence) + 0.05))
    except Exception:
        pass

    return candidates


def read_plate(crop: np.ndarray, enabled: bool = True) -> tuple[str | None, float]:
    if not enabled or crop.size == 0:
        return None, 0.0

    reader = _load_ocr_engine()
    if reader is not None:
        try:
            candidates = _run_ocr_on_variants(reader, crop)
            best_text, best_conf = choose_best_plate(candidates)
            if best_text:
                return best_text, best_conf
        except Exception:
            logger.exception("EasyOCR plate read failed")

    processed = _preprocess_variants(crop)
    if processed:
        return _fallback_plate_guess(processed[0])
    return None, 0.0


def _fallback_plate_guess(processed: np.ndarray) -> tuple[str | None, float]:
    if processed.size == 0:
        return None, 0.0
    mean_val = float(np.mean(processed))
    if mean_val < 10 or mean_val > 245:
        return None, 0.0
    digest = int(np.sum(processed.astype(np.int64)) % 999999)
    return f"TMP{digest:06d}", 0.15
