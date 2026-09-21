import cv2
import numpy as np

from src.config import AppConfig
from src.fire_temporal import FireTemporalValidator, analyze_flame_region
from src.fire_smoke_cv import detect_fire_regions

BAG = r"C:\Users\Ardhas-Dev\.cursor\projects\c-Users-Ardhas-Dev-cctv-pathway-monitor\assets\c__Users_Ardhas-Dev_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_58afd779-433d-441c-839f-e428ab9ec0b0_20260807_145012_full-7cb26bc5-feca-4abb-98bc-fcebf969446e.png"
FIRE = r"C:\Users\Ardhas-Dev\.cursor\projects\c-Users-Ardhas-Dev-cctv-pathway-monitor\assets\c__Users_Ardhas-Dev_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_e85d5282-169e-485a-83ee-f1c3c640153b_20260807_141235_full-1176141c-a178-4d3e-9e9e-0a4cc2b06038.png"


def test(name: str, path: str, n: int = 5) -> None:
    frame = cv2.imread(path)
    cv = detect_fire_regions(frame, cfg)
    validator = FireTemporalValidator(cfg)
    bucket = ("fire", 100, 200)
    for index in range(n):
        for i in range(len(cv)):
            validator.record(bucket, frame, cv.xyxy[i])
        ok = validator.is_confirmed(bucket, 0.0)
        metrics = validator._metrics(validator._samples(bucket))
        print(
            f"{name} frame {index + 1}: confirmed={ok} "
            f"flicker={metrics['flicker']:.4f} drift={metrics['total_drift']:.1f} "
            f"flame_std={metrics['mean_flame_value_std']:.1f} "
            f"static={validator._is_static_colored_object(metrics, 0.0)} "
            f"flat={validator._is_flat_color_sign(metrics, 0.0)} "
            f"sig={validator._has_live_flame_signature(metrics)}"
        )


if __name__ == "__main__":
    cfg = AppConfig.load("config.yaml")
    test("bag", BAG)
    test("fire", FIRE)
