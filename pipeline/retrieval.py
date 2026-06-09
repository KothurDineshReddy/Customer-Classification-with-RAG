"""FAISS-based retrieval for taxonomy entries."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np
import yaml

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536  # matches text-embedding-3-small


# ---------------------------------------------------------------------------
# Embedding provider protocol
# ---------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim)."""
        ...


class MockEmbeddingProvider:
    """Deterministic random embeddings keyed on text hash — no API calls."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vecs[i] = rng.standard_normal(self.dim).astype(np.float32)
        return vecs


class OpenAIEmbeddingProvider:
    """Real embeddings via the OpenAI API (text-embedding-3-small)."""

    MODEL = "text-embedding-3-small"
    BATCH_SIZE = 256

    def __init__(self) -> None:
        from openai import OpenAI  # lazy import to keep mock path dependency-free

        self._client = OpenAI()

    def embed(self, texts: list[str]) -> np.ndarray:
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            response = self._client.embeddings.create(input=batch, model=self.MODEL)
            batch_vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
            all_vecs.append(batch_vecs)
            logger.debug("Embedded batch %d–%d", start, start + len(batch) - 1)
        return np.vstack(all_vecs)


def get_default_provider() -> EmbeddingProvider:
    """Return the provider selected by the USE_MOCK_LLM env var."""
    use_mock = os.getenv("USE_MOCK_LLM", "true").lower() not in ("false", "0", "no")
    if use_mock:
        logger.info(
            "Using MockEmbeddingProvider (USE_MOCK_LLM=%s)", os.getenv("USE_MOCK_LLM", "true")
        )
        return MockEmbeddingProvider()
    logger.info("Using OpenAIEmbeddingProvider")
    return OpenAIEmbeddingProvider()


# ---------------------------------------------------------------------------
# Text representation for each taxonomy entry
# ---------------------------------------------------------------------------


def _entry_text(entry: dict[str, Any]) -> str:
    parts = [
        f"{entry['channel']} - {entry['establishment_type']}: {entry['definition']}",
        f"premise: {entry['premise']}",
        f"inclusions: {', '.join(entry.get('inclusion_examples', []))}",
    ]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def build_index(
    taxonomy_yaml_path: Path,
    embedding_provider: EmbeddingProvider,
    index_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    """Build (or load from cache) a FAISS IndexFlatIP over taxonomy entries.

    Saves the index + metadata alongside the YAML if paths are not provided.
    On subsequent calls with the same paths, loads from disk instead of
    re-embedding.
    """
    taxonomy_yaml_path = Path(taxonomy_yaml_path)
    if index_path is None:
        index_path = taxonomy_yaml_path.with_suffix(".index")
    if meta_path is None:
        meta_path = taxonomy_yaml_path.with_suffix(".meta.json")

    if index_path.exists() and meta_path.exists():
        logger.info("Loading cached index from %s", index_path)
        index = faiss.read_index(str(index_path))
        with meta_path.open() as fh:
            metadata = json.load(fh)
        logger.info("Loaded %d entries from cache", len(metadata))
        return index, metadata

    logger.info("Building index from %s", taxonomy_yaml_path)
    with taxonomy_yaml_path.open() as fh:
        taxonomy = yaml.safe_load(fh)

    entries: list[dict[str, Any]] = taxonomy["entries"]
    texts = [_entry_text(e) for e in entries]

    logger.info("Embedding %d entries…", len(texts))
    vecs = embedding_provider.embed(texts)

    # Normalise for cosine similarity via inner product
    faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, str(index_path))
    with meta_path.open("w") as fh:
        json.dump(entries, fh)

    logger.info("Index saved to %s (%d entries)", index_path, len(entries))
    return index, entries


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    query_text: str,
    index: faiss.IndexFlatIP,
    metadata: list[dict[str, Any]],
    embedding_provider: EmbeddingProvider,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-k taxonomy entries closest to query_text.

    Each result dict is the original metadata entry augmented with a
    ``score`` key (cosine similarity, 0–1).
    """
    vec = embedding_provider.embed([query_text])
    faiss.normalize_L2(vec)

    scores, indices = index.search(vec, k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(score)
        results.append(entry)

    return results
