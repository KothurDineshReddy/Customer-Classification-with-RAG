"""Record normalisation before classification."""

import re
from typing import Any

# Suffixes stripped from business names before classification
_LEGAL_SUFFIXES = re.compile(
    r"\b(LLC|L\.L\.C|Inc|Inc\.|Corp|Corp\.|Co\.|Ltd|Ltd\.|LP|L\.P\.|"
    r"LLP|L\.L\.P|DBA|d/b/a|doing business as)\b\.?",
    re.IGNORECASE,
)

# Collapse runs of whitespace and strip leading/trailing
_WHITESPACE = re.compile(r"\s{2,}")

# Unit/suite tokens in addresses we want to normalise.
# # is a non-word char so \b won't anchor before it; handle it separately.
_SUITE_VARIANTS = re.compile(
    r"(?:\b(?:ste|suite|apt|unit)\b|#)\s*\.?\s*([a-z0-9]+)\b",
    re.IGNORECASE,
)


def _normalise_name(name: str) -> str:
    name = name.strip()
    name = _LEGAL_SUFFIXES.sub("", name)
    name = _WHITESPACE.sub(" ", name).strip().rstrip(",").strip()
    return name


def _normalise_address(address: str) -> str:
    address = address.strip()
    # Standardise Suite/Ste/#/Unit → "Suite X"
    address = _SUITE_VARIANTS.sub(lambda m: f"Suite {m.group(1).upper()}", address)
    address = _WHITESPACE.sub(" ", address).strip()
    return address


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a new record dict with normalised name, address, city, and state.

    All string fields are stripped; customer_name has legal suffixes removed;
    address has suite tokens standardised; city and state are title/upper-cased.
    Non-string values are passed through unchanged.
    """
    cleaned = dict(record)
    cleaned["customer_name"] = _normalise_name(str(record.get("customer_name", "") or ""))
    cleaned["address"] = _normalise_address(str(record.get("address", "") or ""))
    cleaned["city"] = str(record.get("city", "") or "").strip().title()
    cleaned["state"] = str(record.get("state", "") or "").strip().upper()
    cleaned["zip"] = str(record.get("zip", "") or "").strip()
    return cleaned
