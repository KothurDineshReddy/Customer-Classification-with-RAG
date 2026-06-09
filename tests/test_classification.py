"""Tests for pipeline/classification.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.classification import (
    ClassificationError,
    ClassificationResult,
    MockLLMProvider,
    OpenAILLMProvider,
    classify_record,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_TAXONOMY = [
    {
        "id": "TAX0001",
        "national_regional": "Regional",
        "unit_type": "Single-unit",
        "premise": "On-Premise",
        "channel": "Bar",
        "establishment_type": "Beer Bar / Taproom",
        "definition": "Bars centred on draft beer, often craft-focused or brewery-owned.",
        "inclusion_examples": ["craft beer taprooms", "brewery taprooms"],
        "exclusion_examples": ["sports bars", "nightclubs"],
    },
    {
        "id": "TAX0002",
        "national_regional": "National",
        "unit_type": "Multi-unit",
        "premise": "Off-Premise",
        "channel": "Liquor Store",
        "establishment_type": "Chain Liquor Superstore",
        "definition": "Large-format chain liquor retailers with high SKU counts.",
        "inclusion_examples": ["Total Wine & More", "BevMo"],
        "exclusion_examples": ["independent bottle shops"],
    },
    {
        "id": "TAX0003",
        "national_regional": "National",
        "unit_type": "Multi-unit",
        "premise": "On-Premise",
        "channel": "Hotel",
        "establishment_type": "Full-Service Hotel Bar",
        "definition": "Lobby bar within a full-service hotel catering to guests.",
        "inclusion_examples": ["hotel lobby bars", "resort bars"],
        "exclusion_examples": ["motel common rooms"],
    },
    {
        "id": "TAX0004",
        "national_regional": "Regional",
        "unit_type": "Single-unit",
        "premise": "On-Premise",
        "channel": "Restaurant",
        "establishment_type": "Casual Dining",
        "definition": "Full-service restaurants with table service and a casual atmosphere.",
        "inclusion_examples": ["family restaurants", "diners"],
        "exclusion_examples": ["fast food", "fine dining"],
    },
    {
        "id": "TAX0005",
        "national_regional": "National",
        "unit_type": "Multi-unit",
        "premise": "Off-Premise",
        "channel": "Grocery",
        "establishment_type": "Conventional Supermarket",
        "definition": "Full-service grocery chains with a beer and wine aisle.",
        "inclusion_examples": ["Kroger", "Safeway"],
        "exclusion_examples": ["warehouse clubs"],
    },
]


def _valid_result_dict(**overrides) -> dict:
    base = {
        "national_regional": "Regional",
        "unit_type": "Single-unit",
        "premise": "On-Premise",
        "channel": "Restaurant",
        "establishment_type": "Casual Dining",
        "confidence": 0.85,
        "reason": "Matches casual dining pattern.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ClassificationResult model
# ---------------------------------------------------------------------------


class TestClassificationResult:
    def test_valid_construction(self):
        result = ClassificationResult(**_valid_result_dict())
        assert result.channel == "Restaurant"
        assert result.confidence == 0.85

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ClassificationResult(**_valid_result_dict(confidence=1.5))
        with pytest.raises(Exception):
            ClassificationResult(**_valid_result_dict(confidence=-0.1))

    def test_invalid_national_regional(self):
        with pytest.raises(Exception):
            ClassificationResult(**_valid_result_dict(national_regional="Unknown"))

    def test_invalid_unit_type(self):
        with pytest.raises(Exception):
            ClassificationResult(**_valid_result_dict(unit_type="Chain"))

    def test_invalid_premise(self):
        with pytest.raises(Exception):
            ClassificationResult(**_valid_result_dict(premise="Hybrid"))


# ---------------------------------------------------------------------------
# MockLLMProvider — keyword routing
# ---------------------------------------------------------------------------


class TestMockLLMProvider:
    def _classify(self, name: str) -> ClassificationResult:
        provider = MockLLMProvider()
        record = {"customer_name": name, "address": "123 Main St", "city": "Austin", "state": "TX"}
        return classify_record(record, SAMPLE_TAXONOMY, provider)

    def test_bar_keyword(self):
        result = self._classify("The Rusty Nail Bar")
        assert result.channel == "Bar"
        assert result.premise == "On-Premise"

    def test_taproom_keyword(self):
        result = self._classify("Austin Brewing Co. Taproom")
        assert result.channel == "Bar"

    def test_liquor_store_chain(self):
        result = self._classify("Total Wine & More")
        assert result.channel == "Liquor Store"
        assert result.premise == "Off-Premise"
        assert result.unit_type == "Multi-unit"

    def test_liquor_keyword(self):
        result = self._classify("Smith's Liquor Store")
        assert result.channel == "Liquor Store"

    def test_hotel_brand(self):
        result = self._classify("Hilton Garden Inn Dallas")
        assert result.channel == "Hotel"
        assert result.premise == "On-Premise"

    def test_hotel_keyword(self):
        result = self._classify("Sunset Inn & Suites")
        assert result.channel == "Hotel"

    def test_grocery_chain(self):
        result = self._classify("Kroger #1042")
        assert result.channel == "Grocery"
        assert result.premise == "Off-Premise"

    def test_grocery_keyword(self):
        result = self._classify("Downtown Food Market")
        assert result.channel == "Grocery"

    def test_convenience_chain(self):
        result = self._classify("7-Eleven Store #4521")
        assert result.channel == "Convenience"

    def test_pizza_restaurant(self):
        result = self._classify("Mama Rosa's Pizzeria")
        assert result.channel == "Restaurant"
        assert result.establishment_type == "Pizza Restaurant"

    def test_bbq_restaurant(self):
        result = self._classify("Texas Smokehouse BBQ")
        assert result.establishment_type == "BBQ / Smokehouse"

    def test_steakhouse(self):
        result = self._classify("Capital Chophouse & Steakhouse")
        assert result.establishment_type == "Steakhouse"

    def test_default_falls_back_to_restaurant(self):
        result = self._classify("Joe's Place")
        assert result.channel == "Restaurant"

    def test_confidence_in_valid_range(self):
        result = self._classify("Some Random Grill")
        assert 0.70 <= result.confidence <= 0.95

    def test_reason_is_non_empty(self):
        result = self._classify("Corner Bistro")
        assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# JSON parsing in classify_record
# ---------------------------------------------------------------------------


class TestClassifyRecordParsing:
    def _provider_returning(self, raw: str) -> MagicMock:
        provider = MagicMock()
        provider.complete.return_value = raw
        return provider

    def test_valid_json_returns_result(self):
        provider = self._provider_returning(json.dumps(_valid_result_dict()))
        result = classify_record(
            {"customer_name": "Test", "address": "", "city": "", "state": ""},
            SAMPLE_TAXONOMY,
            provider,
        )
        assert isinstance(result, ClassificationResult)
        assert result.channel == "Restaurant"

    def test_malformed_json_raises_classification_error(self):
        provider = self._provider_returning("not json at all {{{")
        with pytest.raises(ClassificationError, match="non-JSON"):
            classify_record(
                {"customer_name": "Test", "address": "", "city": "", "state": ""},
                SAMPLE_TAXONOMY,
                provider,
            )

    def test_json_missing_required_field_raises(self):
        bad = _valid_result_dict()
        del bad["establishment_type"]
        provider = self._provider_returning(json.dumps(bad))
        with pytest.raises(ClassificationError, match="validation"):
            classify_record(
                {"customer_name": "Test", "address": "", "city": "", "state": ""},
                SAMPLE_TAXONOMY,
                provider,
            )

    def test_invalid_enum_field_raises(self):
        bad = _valid_result_dict(national_regional="EMEA")
        provider = self._provider_returning(json.dumps(bad))
        with pytest.raises(ClassificationError, match="validation"):
            classify_record(
                {"customer_name": "Test", "address": "", "city": "", "state": ""},
                SAMPLE_TAXONOMY,
                provider,
            )

    def test_llm_exception_wrapped_as_classification_error(self):
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("network down")
        with pytest.raises(ClassificationError, match="LLM call failed"):
            classify_record(
                {"customer_name": "Test", "address": "", "city": "", "state": ""},
                SAMPLE_TAXONOMY,
                provider,
            )


# ---------------------------------------------------------------------------
# OpenAI retry logic
# ---------------------------------------------------------------------------


class TestOpenAIRetry:
    def _make_provider(self) -> OpenAILLMProvider:
        provider = OpenAILLMProvider.__new__(OpenAILLMProvider)
        provider.model = "gpt-4o-mini"
        provider._client = MagicMock()
        return provider

    def test_succeeds_on_first_attempt(self):

        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_valid_result_dict())
        provider._client.chat.completions.create.return_value = mock_response

        with patch("time.sleep"):
            result = provider.complete("sys", "user")

        assert json.loads(result)["channel"] == "Restaurant"
        provider._client.chat.completions.create.assert_called_once()

    def test_retries_on_rate_limit_then_succeeds(self):
        from openai import RateLimitError

        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_valid_result_dict())

        rate_limit_exc = RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        provider._client.chat.completions.create.side_effect = [
            rate_limit_exc,
            rate_limit_exc,
            mock_response,
        ]

        with patch("time.sleep") as mock_sleep:
            result = provider.complete("sys", "user")

        assert json.loads(result)["channel"] == "Restaurant"
        assert provider._client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2
        # Exponential backoff: first sleep=1s, second sleep=2s
        assert mock_sleep.call_args_list[0][0][0] == pytest.approx(1.0)
        assert mock_sleep.call_args_list[1][0][0] == pytest.approx(2.0)

    def test_raises_after_max_attempts(self):
        from openai import RateLimitError

        provider = self._make_provider()
        rate_limit_exc = RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        provider._client.chat.completions.create.side_effect = rate_limit_exc

        with patch("time.sleep"):
            with pytest.raises(ClassificationError, match="Rate limit exceeded"):
                provider.complete("sys", "user")

        assert provider._client.chat.completions.create.call_count == 3
