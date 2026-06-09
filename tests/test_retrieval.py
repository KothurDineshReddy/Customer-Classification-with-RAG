"""Tests for pipeline/retrieval.py."""

import json
from pathlib import Path

import faiss
import numpy as np
import pytest
import yaml

from pipeline.retrieval import MockEmbeddingProvider, build_index, retrieve

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_ENTRIES = [
    {
        "id": "TAX0001",
        "national_regional": "National",
        "unit_type": "Multi-unit",
        "premise": "On-Premise",
        "channel": "Restaurant",
        "establishment_type": "Fine Dining",
        "definition": "Upscale restaurants with formal table service and premium menus.",
        "inclusion_examples": ["white-tablecloth restaurants", "tasting menus"],
        "exclusion_examples": ["fast food", "cafeterias"],
    },
    {
        "id": "TAX0002",
        "national_regional": "Regional",
        "unit_type": "Single-unit",
        "premise": "On-Premise",
        "channel": "Bar",
        "establishment_type": "Cocktail Lounge",
        "definition": "Bars emphasizing craft cocktails and premium spirits.",
        "inclusion_examples": ["speakeasies", "craft cocktail bars"],
        "exclusion_examples": ["dive bars", "nightclubs"],
    },
    {
        "id": "TAX0003",
        "national_regional": "National",
        "unit_type": "Multi-unit",
        "premise": "Off-Premise",
        "channel": "Grocery",
        "establishment_type": "Conventional Supermarket",
        "definition": "Full-service grocery chains with a dedicated beer and wine aisle.",
        "inclusion_examples": ["Kroger", "Safeway"],
        "exclusion_examples": ["warehouse clubs", "convenience stores"],
    },
    {
        "id": "TAX0004",
        "national_regional": "Regional",
        "unit_type": "Single-unit",
        "premise": "Off-Premise",
        "channel": "Liquor Store",
        "establishment_type": "Specialty / Independent Liquor Store",
        "definition": "Independent retailers with curated spirits, wine, and beer.",
        "inclusion_examples": ["boutique bottle shops", "natural wine shops"],
        "exclusion_examples": ["chain liquor superstores"],
    },
    {
        "id": "TAX0005",
        "national_regional": "National",
        "unit_type": "Multi-unit",
        "premise": "On-Premise",
        "channel": "Hotel",
        "establishment_type": "Full-Service Hotel Bar",
        "definition": "Lobby bar within a full-service hotel catering to guests and locals.",
        "inclusion_examples": ["hotel lobby bars", "resort bars"],
        "exclusion_examples": ["motel common rooms"],
    },
    {
        "id": "TAX0006",
        "national_regional": "Regional",
        "unit_type": "Single-unit",
        "premise": "Off-Premise",
        "channel": "Convenience",
        "establishment_type": "Urban Convenience Store",
        "definition": "Non-fuel convenience retailers in dense urban areas with cold beer and wine.",  # noqa: E501
        "inclusion_examples": ["7-Eleven", "Wawa", "Sheetz"],
        "exclusion_examples": ["gas station c-stores", "grocery stores"],
    },
]


@pytest.fixture()
def taxonomy_yaml(tmp_path: Path) -> Path:
    """Write a small taxonomy YAML fixture to a temp file."""
    data = {
        "taxonomy_version": "v1",
        "total_entries": len(FIXTURE_ENTRIES),
        "entries": FIXTURE_ENTRIES,
    }
    p = tmp_path / "v1.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture()
def mock_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()


@pytest.fixture()
def built_index(taxonomy_yaml: Path, mock_provider: MockEmbeddingProvider):
    """Build a FAISS index from the fixture taxonomy."""
    return build_index(taxonomy_yaml, mock_provider)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_returns_index_and_metadata(self, built_index):
        index, metadata = built_index
        assert isinstance(index, faiss.IndexFlatIP)
        assert isinstance(metadata, list)

    def test_index_contains_all_entries(self, built_index):
        index, metadata = built_index
        assert index.ntotal == len(FIXTURE_ENTRIES)
        assert len(metadata) == len(FIXTURE_ENTRIES)

    def test_vectors_are_normalised(self, built_index):
        index, _ = built_index
        # Reconstruct stored vectors and check unit norm
        vecs = index.reconstruct_n(0, index.ntotal)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_index_cached_on_disk(self, taxonomy_yaml: Path, mock_provider: MockEmbeddingProvider):
        index_path = taxonomy_yaml.with_suffix(".index")
        meta_path = taxonomy_yaml.with_suffix(".meta.json")

        build_index(taxonomy_yaml, mock_provider, index_path, meta_path)
        assert index_path.exists()
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert len(meta) == len(FIXTURE_ENTRIES)

    def test_second_call_loads_from_cache(
        self, taxonomy_yaml: Path, mock_provider: MockEmbeddingProvider
    ):
        index_path = taxonomy_yaml.with_suffix(".index")
        meta_path = taxonomy_yaml.with_suffix(".meta.json")

        index1, _ = build_index(taxonomy_yaml, mock_provider, index_path, meta_path)
        index2, _ = build_index(taxonomy_yaml, mock_provider, index_path, meta_path)

        assert index1.ntotal == index2.ntotal


class TestRetrieve:
    def test_returns_k_results(self, built_index, mock_provider):
        index, metadata = built_index
        results = retrieve("craft cocktails bar spirits", index, metadata, mock_provider, k=3)
        assert len(results) == 3

    def test_results_have_score_key(self, built_index, mock_provider):
        index, metadata = built_index
        results = retrieve("grocery store wine", index, metadata, mock_provider, k=2)
        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.01  # cosine similarity, allow tiny fp slack

    def test_results_preserve_metadata_fields(self, built_index, mock_provider):
        index, metadata = built_index
        results = retrieve("fine dining restaurant", index, metadata, mock_provider, k=1)
        required = {"id", "channel", "establishment_type", "definition", "score"}
        assert required.issubset(results[0].keys())

    def test_k_capped_at_index_size(self, built_index, mock_provider):
        index, metadata = built_index
        # Requesting more than ntotal should just return all entries
        results = retrieve("anything", index, metadata, mock_provider, k=100)
        assert len(results) == len(FIXTURE_ENTRIES)

    def test_consistent_results_for_same_query(self, built_index, mock_provider):
        index, metadata = built_index
        r1 = retrieve("hotel lobby bar", index, metadata, mock_provider, k=3)
        r2 = retrieve("hotel lobby bar", index, metadata, mock_provider, k=3)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]


class TestMockEmbeddingProvider:
    def test_output_shape(self):
        provider = MockEmbeddingProvider(dim=64)
        vecs = provider.embed(["hello", "world", "test"])
        assert vecs.shape == (3, 64)

    def test_deterministic(self):
        provider = MockEmbeddingProvider()
        v1 = provider.embed(["hello world"])
        v2 = provider.embed(["hello world"])
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_differ(self):
        provider = MockEmbeddingProvider()
        v1 = provider.embed(["restaurant fine dining"])
        v2 = provider.embed(["liquor store spirits"])
        assert not np.allclose(v1, v2)
