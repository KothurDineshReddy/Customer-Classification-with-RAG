"""LLM-based classification of customer accounts against the taxonomy."""

import json
import logging
import os
import random
import time
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    national_regional: str = Field(..., description="National or Regional")
    unit_type: str = Field(..., description="Multi-unit or Single-unit")
    premise: str = Field(..., description="On-Premise or Off-Premise")
    channel: str = Field(..., description="Top-level channel (e.g. Restaurant, Bar, Grocery)")
    establishment_type: str = Field(
        ..., description="Specific establishment type within the channel"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence score 0–1")
    reason: str = Field(..., description="Short justification for the classification")

    @field_validator("national_regional")
    @classmethod
    def _valid_national_regional(cls, v: str) -> str:
        allowed = {"National", "Regional"}
        if v not in allowed:
            raise ValueError(f"national_regional must be one of {allowed}, got {v!r}")
        return v

    @field_validator("unit_type")
    @classmethod
    def _valid_unit_type(cls, v: str) -> str:
        allowed = {"Multi-unit", "Single-unit"}
        if v not in allowed:
            raise ValueError(f"unit_type must be one of {allowed}, got {v!r}")
        return v

    @field_validator("premise")
    @classmethod
    def _valid_premise(cls, v: str) -> str:
        allowed = {"On-Premise", "Off-Premise"}
        if v not in allowed:
            raise ValueError(f"premise must be one of {allowed}, got {v!r}")
        return v


class ClassificationError(Exception):
    """Raised when a record cannot be classified."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a beverage-industry account classification assistant for a wholesale distributor.
Your job is to classify a customer account into the company taxonomy using ONLY the \
taxonomy entries provided — never invent new categories.

Return a single JSON object with these exact keys:
  national_regional  — "National" or "Regional"
  unit_type          — "Multi-unit" or "Single-unit"
  premise            — "On-Premise" or "Off-Premise"
  channel            — the channel value from the best matching taxonomy entry
  establishment_type — the establishment_type from the best matching taxonomy entry
  confidence         — a float from 0.0 to 1.0 representing your certainty
  reason             — one or two sentences explaining the match

Rules:
- Output valid JSON only. No markdown, no surrounding text, no code fences.
- channel and establishment_type MUST exactly match one of the provided entries.
- If the account is ambiguous, choose the closest match and lower the confidence score.
"""


def _build_prompt(record: dict[str, Any], retrieved: list[dict[str, Any]]) -> str:
    entries_text = "\n".join(
        f"  - [{e['channel']}] {e['establishment_type']} "
        f"({e['national_regional']}, {e['unit_type']}, {e['premise']}): "
        f"{e['definition']}"
        for e in retrieved
    )
    return (
        f"Customer account:\n"
        f"  Name: {record.get('customer_name', '')}\n"
        f"  Address: {record.get('address', '')}, "
        f"{record.get('city', '')}, {record.get('state', '')}\n\n"
        f"Taxonomy entries to choose from (use ONLY these):\n"
        f"{entries_text}\n\n"
        f"Classify this account and return JSON."
    )


# ---------------------------------------------------------------------------
# LLM provider abstraction
# ---------------------------------------------------------------------------


class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the raw text response from the model."""
        ...


class OpenAILLMProvider:
    """Calls OpenAI chat completions with exponential-backoff retry."""

    MODEL = "gpt-4o-mini"
    MAX_ATTEMPTS = 3
    BASE_DELAY = 1.0  # seconds

    def __init__(self, model: str = MODEL) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        from openai import RateLimitError

        last_exc: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                return response.choices[0].message.content or ""
            except RateLimitError as exc:
                last_exc = exc
                delay = self.BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    self.MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)

        raise ClassificationError(
            f"Rate limit exceeded after {self.MAX_ATTEMPTS} attempts"
        ) from last_exc


# ---------------------------------------------------------------------------
# Keyword rules for the mock — ordered most-specific first
# ---------------------------------------------------------------------------

_KEYWORD_RULES: list[tuple[list[str], dict[str, Any]]] = [
    # Off-Premise chains — check before generic bar/restaurant terms
    (
        ["costco", "sam's club", "bj's wholesale"],
        {
            "national_regional": "National",
            "unit_type": "Multi-unit",
            "premise": "Off-Premise",
            "channel": "Warehouse Club",
            "establishment_type": "Membership Warehouse Club",
        },
    ),
    (
        [
            "total wine",
            "bevmo",
            "spec's",
            "binny's",
            "abc fine wine",
            "liquor barn",
            "keg liquors",
            "marketview",
        ],
        {
            "national_regional": "National",
            "unit_type": "Multi-unit",
            "premise": "Off-Premise",
            "channel": "Liquor Store",
            "establishment_type": "Chain Liquor Superstore",
        },
    ),
    (
        ["liquor", "spirits", "bottle shop", "wine & spirits"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "Off-Premise",
            "channel": "Liquor Store",
            "establishment_type": "Specialty / Independent Liquor Store",
        },
    ),
    (
        [
            "kroger",
            "heb",
            "safeway",
            "albertsons",
            "publix",
            "meijer",
            "wegmans",
            "whole foods",
            "trader joe",
            "aldi",
            "lidl",
            "winn-dixie",
            "food lion",
            "giant eagle",
            "stop & shop",
            "smart & final",
            "piggly wiggly",
            "hy-vee",
            "stater bros",
            "raley's",
            "price chopper",
            "winco",
        ],
        {
            "national_regional": "National",
            "unit_type": "Multi-unit",
            "premise": "Off-Premise",
            "channel": "Grocery",
            "establishment_type": "Conventional Supermarket",
        },
    ),
    (
        ["grocery", "supermarket", "market", "food mart"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "Off-Premise",
            "channel": "Grocery",
            "establishment_type": "Conventional Supermarket",
        },
    ),
    (
        [
            "7-eleven",
            "circle k",
            "speedway",
            "wawa",
            "sheetz",
            "casey's",
            "ampm",
            "kwik trip",
            "pilot flying j",
            "love's travel",
        ],
        {
            "national_regional": "National",
            "unit_type": "Multi-unit",
            "premise": "Off-Premise",
            "channel": "Convenience",
            "establishment_type": "Urban Convenience Store",
        },
    ),
    (
        ["convenience", "c-store", "gas station"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "Off-Premise",
            "channel": "Convenience",
            "establishment_type": "Gas Station Convenience Store",
        },
    ),
    (
        ["walgreens", "cvs", "rite aid"],
        {
            "national_regional": "National",
            "unit_type": "Multi-unit",
            "premise": "Off-Premise",
            "channel": "Drug Store",
            "establishment_type": "National Drug Store Chain",
        },
    ),
    # On-Premise — Hotels before Bar/Restaurant to avoid substring collisions
    (
        [
            "hilton",
            "marriott",
            "hyatt",
            "holiday inn",
            "hampton inn",
            "courtyard",
            "doubletree",
            "sheraton",
            "westin",
            "omni",
            "loews",
            "ritz-carlton",
            "four seasons",
            "radisson",
            "wyndham",
            "best western",
            "comfort inn",
            "quality inn",
            "days inn",
            "super 8",
            "la quinta",
            "ramada",
            "staybridge",
            "residence inn",
            "fairfield inn",
            "springhill",
            "homewood suites",
        ],
        {
            "national_regional": "National",
            "unit_type": "Multi-unit",
            "premise": "On-Premise",
            "channel": "Hotel",
            "establishment_type": "Full-Service Hotel Bar",
        },
    ),
    (
        ["hotel", "inn", "resort", "suites", "lodge"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Hotel",
            "establishment_type": "Full-Service Hotel Bar",
        },
    ),
    (
        ["stadium", "arena", "ballpark", "amphitheater", "amphitheatre", "racetrack"],
        {
            "national_regional": "National",
            "unit_type": "Multi-unit",
            "premise": "On-Premise",
            "channel": "Stadium",
            "establishment_type": "Professional Sports Arena",
        },
    ),
    (
        ["nightclub", "night club", "club ", "lounge"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Nightclub",
            "establishment_type": "Dance Club",
        },
    ),
    (
        ["catering", "caterers", "caterer", "banquet"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Catering",
            "establishment_type": "Full-Service Catering Company",
        },
    ),
    (
        ["cafe", "café", "coffee", "bakery", "tea house", "patisserie"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Cafe",
            "establishment_type": "Coffee & Wine Cafe",
        },
    ),
    (
        [
            "taproom",
            "tap room",
            "brewing",
            "brewery",
            "brewpub",
            "brew pub",
            "taphouse",
            "tap house",
            "alehouse",
            "ale house",
            "pub",
            "tavern",
            "bar & grill",
            "bar and grill",
            "sports bar",
            "cocktail",
            "whiskey bar",
            "wine bar",
            " bar",
            "bar ",
        ],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Bar",
            "establishment_type": "Beer Bar / Taproom",
        },
    ),
    (
        [
            "applebee's",
            "chili's",
            "olive garden",
            "tgi friday",
            "red lobster",
            "outback",
            "cracker barrel",
            "ihop",
            "denny's",
            "waffle house",
            "buffalo wild wings",
            "cheesecake factory",
            "p.f. chang",
            "panera",
            "chipotle",
            "five guys",
            "shake shack",
            "wingstop",
            "raising cane",
            "texas roadhouse",
            "longhorn steakhouse",
            "golden corral",
            "steak 'n shake",
            "sonic",
            "whataburger",
            "in-n-out",
        ],
        {
            "national_regional": "National",
            "unit_type": "Multi-unit",
            "premise": "On-Premise",
            "channel": "Restaurant",
            "establishment_type": "Casual Dining",
        },
    ),
    (
        ["steakhouse", "steak house", "chophouse"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Restaurant",
            "establishment_type": "Steakhouse",
        },
    ),
    (
        ["pizza", "pizzeria", "pizzas"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Restaurant",
            "establishment_type": "Pizza Restaurant",
        },
    ),
    (
        ["sushi", "ramen", "izakaya"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Restaurant",
            "establishment_type": "Sushi / Japanese",
        },
    ),
    (
        ["bbq", "barbeque", "barbecue", "smokehouse"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Restaurant",
            "establishment_type": "BBQ / Smokehouse",
        },
    ),
    (
        ["seafood", "fish house", "oyster", "crab"],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Restaurant",
            "establishment_type": "Seafood Restaurant",
        },
    ),
    # Default: Restaurant / Casual Dining
    (
        [
            "grill",
            "grille",
            "kitchen",
            "bistro",
            "diner",
            "eatery",
            "restaurant",
            "dining",
            "food",
            "mama",
            "casa",
            "papa",
            "house",
            "place",
        ],
        {
            "national_regional": "Regional",
            "unit_type": "Single-unit",
            "premise": "On-Premise",
            "channel": "Restaurant",
            "establishment_type": "Casual Dining",
        },
    ),
]

_DEFAULT_CLASSIFICATION = {
    "national_regional": "Regional",
    "unit_type": "Single-unit",
    "premise": "On-Premise",
    "channel": "Restaurant",
    "establishment_type": "Casual Dining",
}


class MockLLMProvider:
    """Keyword-based deterministic classifier — no API calls."""

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        name_line = next((ln for ln in user.splitlines() if "Name:" in ln), "")
        name = name_line.split("Name:", 1)[-1].strip().lower()

        classification = _DEFAULT_CLASSIFICATION.copy()
        for keywords, attrs in _KEYWORD_RULES:
            if any(kw in name for kw in keywords):
                classification = attrs.copy()
                break

        # Derive confidence from a hash of the customer name so the value is
        # deterministic per record regardless of call order or thread scheduling.
        import hashlib

        seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        confidence = round(rng.uniform(0.70, 0.95), 2)
        classification["confidence"] = confidence
        classification["reason"] = (
            f"Customer name '{name_line.split('Name:', 1)[-1].strip()}' "
            f"matched keyword rules for {classification['channel']} / "
            f"{classification['establishment_type']}."
        )
        return json.dumps(classification)


def get_default_llm_provider() -> LLMProvider:
    use_mock = os.getenv("USE_MOCK_LLM", "true").lower() not in ("false", "0", "no")
    if use_mock:
        logger.info("Using MockLLMProvider")
        return MockLLMProvider()
    logger.info("Using OpenAILLMProvider")
    return OpenAILLMProvider()


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------


def classify_record(
    record: dict[str, Any],
    retrieved_taxonomy: list[dict[str, Any]],
    llm_provider: LLMProvider,
) -> ClassificationResult:
    """Classify a single customer record against retrieved taxonomy entries.

    Args:
        record: Row dict with at least customer_name, address, city, state.
        retrieved_taxonomy: Top-k entries from the FAISS retrieval step.
        llm_provider: LLM backend to use (real or mock).

    Returns:
        Validated ClassificationResult.

    Raises:
        ClassificationError: If the LLM response cannot be parsed or validated.
    """
    system = _SYSTEM_PROMPT
    user = _build_prompt(record, retrieved_taxonomy)

    try:
        raw = llm_provider.complete(system, user)
    except ClassificationError:
        raise
    except Exception as exc:
        raise ClassificationError(f"LLM call failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClassificationError(f"LLM returned non-JSON: {raw!r}") from exc

    try:
        return ClassificationResult(**data)
    except Exception as exc:
        raise ClassificationError(f"Response failed validation: {exc}\nRaw: {raw!r}") from exc
