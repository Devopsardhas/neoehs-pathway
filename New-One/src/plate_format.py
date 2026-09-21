from __future__ import annotations

import re
from difflib import SequenceMatcher

# Common Indian RTO state/UT codes (2 letters)
INDIAN_STATE_CODES = frozenset(
    {
        "AN",
        "AP",
        "AR",
        "AS",
        "BR",
        "CG",
        "CH",
        "DD",
        "DL",
        "GA",
        "GJ",
        "HP",
        "HR",
        "JH",
        "JK",
        "KA",
        "KL",
        "LA",
        "LD",
        "MH",
        "ML",
        "MN",
        "MP",
        "MZ",
        "NL",
        "OD",
        "OR",
        "PB",
        "PY",
        "RJ",
        "SK",
        "TN",
        "TR",
        "TS",
        "UK",
        "UP",
        "WB",
    }
)

# OCR often confuses these in the letter regions of Indian plates.
LETTER_CONFUSIONS: dict[str, str] = {
    "0": "ODQ",
    "1": "IL",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
    "B": "8",
    "D": "0",
    "G": "6",
    "H": "M",
    "I": "1",
    "K": "MNH",
    "L": "1",
    "M": "KNHW",
    "N": "MH",
    "O": "0",
    "Q": "0",
    "S": "5",
    "W": "M",
    "Z": "2",
}

DIGIT_CONFUSIONS: dict[str, str] = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

_PLATE_CORE = re.compile(r"^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$")
_TRAILING_ARTIFACTS = re.compile(r"(IN|IND|COM|NET|ORG|WWW)$")
_DIGIT_LIKE = re.compile(r"^[0-9OILSZGBQ]+$")
_PREFIX_NUMERIC = re.compile(r"^([A-Z]{1,4})([0-9OILSZGBQ]{3,})$")


def clean_plate_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    if not cleaned:
        return ""

    # Drop trailing web/country OCR noise (e.g. ".in" -> IN)
    for _ in range(2):
        trimmed = _TRAILING_ARTIFACTS.sub("", cleaned)
        if trimmed == cleaned or len(trimmed) < 4:
            break
        if _matches_indian_plate(trimmed) or _correct_generic_plate(trimmed):
            cleaned = trimmed
        else:
            break
    return cleaned


def _correct_generic_plate(cleaned: str) -> str | None:
    """Fix letter/digit confusions for numeric and prefix+numeric plates."""
    if not cleaned:
        return None

    if _DIGIT_LIKE.fullmatch(cleaned):
        fixed = _fix_digit_section(cleaned)
        if fixed.isdigit() and len(fixed) >= 4:
            return fixed

    prefix_match = _PREFIX_NUMERIC.fullmatch(cleaned)
    if prefix_match:
        prefix, tail = prefix_match.groups()
        fixed_tail = _fix_digit_section(tail)
        if fixed_tail.isdigit():
            return prefix + fixed_tail

    # Digits with one trailing confused letter (e.g. 8254S -> 82545)
    trailing_match = re.fullmatch(r"(\d{3,})([OILSZGBQ])", cleaned)
    if trailing_match:
        digits, last = trailing_match.groups()
        fixed_last = DIGIT_CONFUSIONS.get(last, last)
        if fixed_last.isdigit():
            return digits + fixed_last

    # Fix any letter sitting inside an otherwise numeric run.
    if re.search(r"[OILSZGBQ]", cleaned) and re.search(r"\d", cleaned):
        fixed = _fix_digit_section(cleaned)
        if fixed.isdigit() and len(fixed) >= 4:
            return fixed
        prefix_match = _PREFIX_NUMERIC.fullmatch(fixed) if not fixed.isdigit() else None
        if prefix_match:
            prefix, tail = prefix_match.groups()
            if tail.isdigit():
                return prefix + tail

    return None


def _matches_indian_plate(plate: str) -> bool:
    match = _PLATE_CORE.fullmatch(plate)
    if not match:
        return False
    state = match.group(1)
    return state in INDIAN_STATE_CODES


def _fix_digit_section(value: str) -> str:
    return "".join(DIGIT_CONFUSIONS.get(char, char) for char in value)


def _fix_letter_section(value: str) -> str:
    digit_in_letters = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    fixed = []
    for char in value:
        if char.isdigit():
            fixed.append(digit_in_letters.get(char, char))
        else:
            fixed.append(char)
    return "".join(fixed)


# Common 2-letter series OCR misreads seen on Indian plates.
_SERIES_OCR_FIXES: dict[str, str] = {
    "MK": "MM",
    "KM": "MM",
    "NH": "MN",
    "HN": "MN",
    "VV": "VN",
    "NV": "MN",
}


def _refine_series_letters(series: str) -> str:
    if len(series) == 2:
        return _SERIES_OCR_FIXES.get(series, series)
    return series


def _apply_component_fixes(state: str, district: str, series: str, number: str) -> str | None:
    district = _fix_digit_section(district)
    number = _fix_digit_section(number)
    series = _fix_letter_section(series)
    series = re.sub(r"[^A-Z]", "", series)
    series = _refine_series_letters(series)
    if len(series) < 1 or len(series) > 3:
        return None
    candidate = f"{state}{district}{series}{number}"
    return candidate if _matches_indian_plate(candidate) else None


def _generate_letter_variants(series: str, max_variants: int = 24) -> list[str]:
    """Generate likely series variants for single-character OCR errors."""
    if not series:
        return [""]

    variants: set[str] = {series}
    queue = [series]
    while queue and len(variants) < max_variants:
        current = queue.pop(0)
        for index, char in enumerate(current):
            for replacement in LETTER_CONFUSIONS.get(char, ""):
                if replacement == char:
                    continue
                candidate = current[:index] + replacement + current[index + 1 :]
                if candidate not in variants:
                    variants.add(candidate)
                    queue.append(candidate)
                if len(variants) >= max_variants:
                    break
    return sorted(variants)


def parse_and_correct_plate(text: str) -> str | None:
    cleaned = clean_plate_text(text)
    if not cleaned:
        return None

    if _matches_indian_plate(cleaned):
        match = _PLATE_CORE.fullmatch(cleaned)
        if match:
            fixed = _apply_component_fixes(*match.groups())
            if fixed:
                return fixed
        return cleaned

    match = _PLATE_CORE.fullmatch(cleaned)
    if match:
        fixed = _apply_component_fixes(*match.groups())
        if fixed:
            return fixed

    # Try to locate a valid plate substring inside noisy OCR output.
    for length in (10, 11, 12):
        if len(cleaned) < length:
            continue
        for start in range(0, len(cleaned) - length + 1):
            chunk = cleaned[start : start + length]
            chunk_match = _PLATE_CORE.fullmatch(chunk)
            if not chunk_match:
                continue
            fixed = _apply_component_fixes(*chunk_match.groups())
            if fixed:
                return fixed
            state, district, series, number = chunk_match.groups()
            if state not in INDIAN_STATE_CODES:
                continue
            for series_variant in _generate_letter_variants(series):
                fixed = _apply_component_fixes(state, district, series_variant, number)
                if fixed:
                    return fixed

    generic = _correct_generic_plate(cleaned)
    if generic:
        return generic

    return None


def format_plate_display(plate: str) -> str:
    plate = plate.upper()
    match = _PLATE_CORE.fullmatch(plate)
    if match:
        state, district, series, number = match.groups()
        return f"{state} {district} {series} {number}"

    prefix_match = _PREFIX_NUMERIC.fullmatch(plate)
    if prefix_match:
        prefix, tail = prefix_match.groups()
        return f"{prefix} {tail}"

    if plate.isdigit() and len(plate) >= 4:
        return plate
    return plate


def plates_similar(a: str, b: str, max_letter_edits: int = 1) -> bool:
    a = a.upper()
    b = b.upper()
    if a == b:
        return True
    if len(a) != len(b):
        return False

    match_a = _PLATE_CORE.fullmatch(a)
    match_b = _PLATE_CORE.fullmatch(b)
    if not match_a or not match_b:
        return SequenceMatcher(None, a, b).ratio() >= 0.9

    parts_a = match_a.groups()
    parts_b = match_b.groups()
    if parts_a[0] != parts_b[0] or parts_a[1] != parts_b[1] or parts_a[3] != parts_b[3]:
        return False

    return SequenceMatcher(None, parts_a[2], parts_b[2]).ratio() >= (
        1.0 - max_letter_edits / max(len(parts_a[2]), 1)
    )


def choose_best_plate(candidates: list[tuple[str, float]]) -> tuple[str | None, float]:
    """Pick the best plate from OCR candidates using confidence + format score."""
    if not candidates:
        return None, 0.0

    clustered: dict[str, float] = {}
    for raw_text, confidence in candidates:
        corrected = parse_and_correct_plate(raw_text)
        if not corrected:
            continue

        bonus = 0.15 if _matches_indian_plate(corrected) else 0.0
        if corrected.isdigit():
            bonus += 0.08
        score = float(confidence) + bonus

        merged_key = corrected
        for existing in list(clustered):
            if plates_similar(existing, corrected):
                merged_key = existing
                break
        clustered[merged_key] = clustered.get(merged_key, 0.0) + score

    if not clustered:
        return None, 0.0

    best_plate, best_score = max(clustered.items(), key=lambda item: item[1])
    return best_plate, best_score
