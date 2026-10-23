"""Score the standard 60-item BFI-2 in its documented item order."""
from pathlib import Path
import json
import math
from numbers import Real

ROOT = Path(__file__).resolve().parents[1]
KEY = json.loads((ROOT / "config/bfi2_scoring.json").read_text())
DOMAIN_ITEMS = KEY["domains"]
REVERSE_ITEMS = frozenset(KEY["reverse_items"])
DOMAIN_LABELS = KEY["domain_labels"]
LABELS = {
    "very inaccurate": 1.0,
    "moderately inaccurate": 2.0,
    "neither accurate nor inaccurate": 3.0,
    "moderately accurate": 4.0,
    "very accurate": 5.0,
}

def coerce_answer(value):
    """Accept numeric item means or the five configured response labels."""
    if isinstance(value, bool):
        raise ValueError("Boolean responses are not valid BFI-2 scores")
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in LABELS:
            return LABELS[text.casefold()]
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"Unknown BFI-2 response label: {text!r}") from exc
    elif isinstance(value, Real):
        number = float(value)
    else:
        raise ValueError(f"Invalid BFI-2 response type: {type(value).__name__}")
    if not math.isfinite(number) or not 1 <= number <= 5:
        raise ValueError("BFI-2 responses must be finite values between 1 and 5")
    return number

def score_bfi2(answers):
    """Require Q1--Q60, reverse as 6-x, and average all 12 items per domain."""
    required = [f"Q{i}" for i in range(1, 61)]
    missing = [item for item in required if item not in answers or answers[item] is None or answers[item] == ""]
    if missing:
        raise ValueError("Missing BFI-2 responses: " + ", ".join(missing))
    values = {}
    for item in required:
        try:
            value = coerce_answer(answers[item])
        except ValueError as exc:
            raise ValueError(f"{item}: {exc}") from exc
        index = int(item[1:])
        values[index] = 6.0 - value if index in REVERSE_ITEMS else value
    return {domain: sum(values[index] for index in items) / len(items) for domain, items in DOMAIN_ITEMS.items()}
