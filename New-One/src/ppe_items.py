from __future__ import annotations

from dataclasses import dataclass

PPE_ITEM_HELMET = "helmet"
PPE_ITEM_VEST = "vest"
PPE_ITEM_GLOVES = "gloves"
PPE_ITEM_BOOTS = "boots"

PPE_ITEM_KEYS = (PPE_ITEM_HELMET, PPE_ITEM_VEST, PPE_ITEM_GLOVES, PPE_ITEM_BOOTS)

PPE_ITEM_SETTING_KEYS = {
    PPE_ITEM_HELMET: "ppe_helmet",
    PPE_ITEM_VEST: "ppe_vest",
    PPE_ITEM_GLOVES: "ppe_gloves",
    PPE_ITEM_BOOTS: "ppe_boots",
}

PPE_ITEM_SETTING_LABELS = {
    "ppe_helmet": "Helmet / Hard Hat",
    "ppe_vest": "Safety Vest",
    "ppe_gloves": "Gloves",
    "ppe_boots": "Safety Boots",
}

PPE_ITEM_SETTING_DESCRIPTIONS = {
    "ppe_helmet": "Flag persons without a helmet or hard hat.",
    "ppe_vest": "Flag persons without a high-visibility safety vest.",
    "ppe_gloves": "Flag persons without safety gloves.",
    "ppe_boots": "Flag persons without safety boots or shoes.",
}

PPE_MISSING_LABELS = {
    "missing_helmet": "Missing Helmet",
    "missing_vest": "Missing Vest",
    "missing_gloves": "Missing Gloves",
    "missing_boots": "Missing Boots",
}


@dataclass(frozen=True)
class PPEItemConfig:
    key: str
    missing_key: str
    label: str
    classes: list[str]
    region_top: float
    region_bottom: float
    default_enabled: bool


def missing_key_for_item(item_key: str) -> str:
    return f"missing_{item_key}"


def ppe_item_label(missing_key: str) -> str:
    return PPE_MISSING_LABELS.get(missing_key, missing_key.replace("_", " ").title())


def encode_missing_ppe(missing_keys: list[str]) -> str:
    return ",".join(sorted(missing_keys))


def decode_missing_ppe(object_class: str) -> list[str]:
    if not object_class or object_class == "missing_ppe":
        return []
    return [part.strip() for part in object_class.split(",") if part.strip()]


def format_missing_ppe_list(missing_keys: list[str]) -> str:
    labels = [ppe_item_label(key) for key in sorted(missing_keys)]
    if not labels:
        return "Missing PPE"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


DEFAULT_PPE_ITEMS: dict[str, dict] = {
    PPE_ITEM_HELMET: {
        "classes": ["helmet", "hard hat", "safety helmet", "construction helmet"],
        "region_top": 0.0,
        "region_bottom": 0.35,
        "enabled": True,
    },
    PPE_ITEM_VEST: {
        "classes": [
            "safety vest",
            "high visibility vest",
            "reflective vest",
            "hi-vis vest",
        ],
        "region_top": 0.35,
        "region_bottom": 0.80,
        "enabled": True,
    },
    PPE_ITEM_GLOVES: {
        "classes": ["gloves", "safety gloves", "work gloves", "protective gloves"],
        "region_top": 0.40,
        "region_bottom": 0.72,
        "enabled": False,
    },
    PPE_ITEM_BOOTS: {
        "classes": [
            "boots",
            "safety boots",
            "steel toe boots",
            "safety shoes",
            "work boots",
        ],
        "region_top": 0.72,
        "region_bottom": 1.0,
        "enabled": False,
    },
}


def load_ppe_items(raw: dict) -> list[PPEItemConfig]:
    items_raw = raw.get("items")
    if items_raw is None:
        items_raw = _legacy_ppe_items(raw)

    items: list[PPEItemConfig] = []
    for key in PPE_ITEM_KEYS:
        defaults = DEFAULT_PPE_ITEMS[key]
        item_raw = items_raw.get(key, {})
        items.append(
            PPEItemConfig(
                key=key,
                missing_key=missing_key_for_item(key),
                label=key.replace("_", " ").title(),
                classes=item_raw.get("classes", defaults["classes"]),
                region_top=float(item_raw.get("region_top", defaults["region_top"])),
                region_bottom=float(item_raw.get("region_bottom", defaults["region_bottom"])),
                default_enabled=bool(item_raw.get("enabled", defaults["enabled"])),
            )
        )
    return items


def _legacy_ppe_items(raw: dict) -> dict:
    return {
        PPE_ITEM_HELMET: {
            "classes": raw.get(
                "helmet_classes",
                DEFAULT_PPE_ITEMS[PPE_ITEM_HELMET]["classes"],
            ),
            "region_top": 0.0,
            "region_bottom": raw.get("head_region_ratio", 0.35),
            "enabled": raw.get("require_helmet", True),
        },
        PPE_ITEM_VEST: {
            "classes": raw.get(
                "vest_classes",
                DEFAULT_PPE_ITEMS[PPE_ITEM_VEST]["classes"],
            ),
            "region_top": raw.get("head_region_ratio", 0.35),
            "region_bottom": raw.get("head_region_ratio", 0.35) + raw.get(
                "torso_region_ratio", 0.45
            ),
            "enabled": raw.get("require_vest", False),
        },
        PPE_ITEM_GLOVES: DEFAULT_PPE_ITEMS[PPE_ITEM_GLOVES],
        PPE_ITEM_BOOTS: DEFAULT_PPE_ITEMS[PPE_ITEM_BOOTS],
    }
