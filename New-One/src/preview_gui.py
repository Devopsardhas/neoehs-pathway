from __future__ import annotations

import logging

import cv2

logger = logging.getLogger(__name__)

_GUI_OK: bool | None = None


def gui_available() -> bool:
    global _GUI_OK
    if _GUI_OK is not None:
        return _GUI_OK
    try:
        cv2.namedWindow("__preview_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__preview_probe__")
        _GUI_OK = True
    except cv2.error:
        _GUI_OK = False
        logger.warning(
            "OpenCV GUI is unavailable. Local --preview windows are disabled. "
            "Install GUI support with: pip install opencv-python "
            "(not opencv-python-headless), or use the web dashboard live preview."
        )
    return _GUI_OK


def imshow(window_title: str, frame) -> bool:
    global _GUI_OK
    if not gui_available():
        return False
    try:
        cv2.imshow(window_title, frame)
        return True
    except cv2.error:
        _GUI_OK = False
        logger.warning("OpenCV imshow failed; disabling local preview window.")
        return False


def wait_key(delay_ms: int = 1) -> int:
    if not gui_available():
        return -1
    try:
        return cv2.waitKey(delay_ms) & 0xFF
    except cv2.error:
        return -1


def destroy_all() -> None:
    if not gui_available():
        return
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
