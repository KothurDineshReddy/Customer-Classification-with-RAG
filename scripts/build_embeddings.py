"""Build the FAISS index from data/taxonomy/v1.yaml.

Usage:
    python scripts/build_embeddings.py [--mock]

Flags:
    --mock   Force MockEmbeddingProvider regardless of USE_MOCK_LLM env var.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.retrieval import (
    MockEmbeddingProvider,
    build_index,
    get_default_provider,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS taxonomy index.")
    parser.add_argument("--mock", action="store_true", help="Force mock embeddings")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("data/taxonomy/v1.yaml"),
        help="Path to taxonomy YAML (default: data/taxonomy/v1.yaml)",
    )
    args = parser.parse_args()

    provider = MockEmbeddingProvider() if args.mock else get_default_provider()
    index, metadata = build_index(args.taxonomy, provider)
    print(f"Done. Index contains {index.ntotal} vectors.")


if __name__ == "__main__":
    main()
