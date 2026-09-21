from __future__ import annotations

from src.config import AppConfig
from src.ppe_items import encode_missing_ppe


def build_summary(class_name: str, duration_sec: float, pathway_name: str, config: AppConfig) -> str:
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    if duration_sec < config.brief_threshold_sec:
        label = "Pathway obstruction" if class_name == "pathway_obstruction" else f"Object ({class_name})"
        return (
            f"{label} is blocking {pathway_name}. "
            f"Detected for {duration_text}."
        )

    if duration_sec < config.moderate_threshold_sec:
        return (
            f"Ongoing obstruction: {class_name} on {pathway_name} "
            f"for {duration_text}. Pathway remains blocked."
        )

    return (
        f"Persistent violation: {class_name} has blocked {pathway_name} "
        f"for {duration_text}. Immediate clearance recommended."
    )


def build_mobile_summary(duration_sec: float, pathway_name: str, config: AppConfig) -> str:
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    if duration_sec < config.brief_threshold_sec:
        return (
            f"Mobile phone usage on {pathway_name} outside the allowed mobile zone. "
            f"Detected for {duration_text}."
        )

    if duration_sec < config.moderate_threshold_sec:
        return (
            f"Ongoing mobile violation on {pathway_name} outside allowed zone "
            f"for {duration_text}. Phone use is restricted on this pathway."
        )

    return (
        f"Persistent mobile violation on {pathway_name} outside allowed zone "
        f"for {duration_text}. Repeated phone use on restricted pathway area."
    )


def build_ppe_summary(
    missing_items: list[str],
    duration_sec: float,
    zone_name: str,
    config: AppConfig,
) -> str:
    from src.ppe_items import format_missing_ppe_list

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    items_label = format_missing_ppe_list(missing_items)

    if duration_sec < config.brief_threshold_sec:
        return (
            f"PPE violation on {zone_name}: {items_label}. "
            f"Person without required equipment for {duration_text}."
        )

    if duration_sec < config.moderate_threshold_sec:
        return (
            f"Ongoing PPE violation on {zone_name}: {items_label} "
            f"for {duration_text}. Required safety equipment not worn."
        )

    return (
        f"Persistent PPE violation on {zone_name}: {items_label} "
        f"for {duration_text}. Immediate compliance action recommended."
    )


def build_fire_summary(duration_sec: float, fire_type_label: str, config: AppConfig) -> str:
    if duration_sec < config.brief_threshold_sec:
        return f"{fire_type_label} detected in camera view. Review source and assess risk."

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    if fire_type_label == "Structural Fire":
        return (
            f"Active structural fire detected in camera view for {duration_text}. "
            "Emergency response may be required."
        )
    if fire_type_label == "Welding":
        return (
            f"Welding activity detected in camera view for {duration_text}. "
            "Verify hot work permits and nearby combustible materials."
        )
    if fire_type_label == "Electrical Spark":
        return (
            f"Electrical sparking detected in camera view for {duration_text}. "
            "Inspect electrical equipment and nearby fire risk."
        )

    return (
        f"Active {fire_type_label.lower()} detected in camera view for {duration_text}. "
        "Investigate source and assess safety risk."
    )


def build_smoke_summary(duration_sec: float, config: AppConfig) -> str:
    if duration_sec < config.brief_threshold_sec:
        return "Smoke detected in camera view. Check for fire or hazardous conditions."

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"Smoke present in camera view for {duration_text}. "
        "Investigate source and assess air quality risk."
    )


def build_oil_spillage_summary(duration_sec: float, config: AppConfig) -> str:
    if duration_sec < config.brief_threshold_sec:
        return "Oil spillage detected in camera view. Mark area and arrange cleanup."

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"Oil spillage present in camera view for {duration_text}. "
        "Contain spill, post warning signs, and clean affected floor area."
    )


def build_wet_floor_summary(duration_sec: float, config: AppConfig) -> str:
    if duration_sec < config.brief_threshold_sec:
        return "Wet floor detected in camera view. Check for slip hazard and post caution."

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"Wet floor present in camera view for {duration_text}. "
        "Inspect source, dry the area, and prevent pedestrian slip risk."
    )


def build_low_visibility_summary(duration_sec: float, config: AppConfig) -> str:
    if duration_sec < config.brief_threshold_sec:
        return (
            "Low visibility detected in camera view. "
            "Check for fog, haze, glare, dust, or camera obstruction."
        )

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"Low visibility present in camera view for {duration_text}. "
        "Inspect environmental conditions, lighting, and camera lens cleanliness."
    )


def build_object_fall_summary(
    object_class: str,
    fall_kind: str,
    duration_sec: float,
    zone_name: str,
    config: AppConfig,
) -> str:
    subject = "Person fall" if fall_kind == "person_fall" else f"{object_class} fall"
    if duration_sec < config.brief_threshold_sec:
        return f"{subject} detected on {zone_name}. Immediate area inspection recommended."

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"{subject} event on {zone_name} observed for {duration_text}. "
        "Review struck-by risk and secure overhead work areas."
    )


def build_near_miss_summary(
    hazard_class: str,
    duration_sec: float,
    zone_name: str,
    config: AppConfig,
) -> str:
    if duration_sec < config.brief_threshold_sec:
        return (
            f"Near miss: person and {hazard_class} in close proximity on {zone_name}. "
            "Review pedestrian and vehicle separation."
        )

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"Near miss between person and {hazard_class} on {zone_name} "
        f"for {duration_text}. Investigate traffic management and blind spots."
    )
