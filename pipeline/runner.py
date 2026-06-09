"""Pipeline orchestrator: validates → retrieves → classifies → exports."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from pipeline.classification import (
    ClassificationError,
    classify_record,
    get_default_llm_provider,
)
from pipeline.confidence import compute_confidence
from pipeline.export import write_output_chunk
from pipeline.preprocessing import clean_record
from pipeline.retrieval import (
    EmbeddingProvider,
    build_index,
    get_default_provider,
    retrieve,
)
from pipeline.validation import validate_csv_schema

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
MAX_WORKERS = 10
TAXONOMY_DIR = Path(os.getenv("DATA_DIR", "./data")) / "taxonomy"
TAXONOMY_YAML = TAXONOMY_DIR / "v1.yaml"


# ---------------------------------------------------------------------------
# Job tracker protocol — decoupled from FastAPI so runner is testable standalone
# ---------------------------------------------------------------------------


class JobTracker(Protocol):
    def set_progress(self, job_id: str, records_processed: int) -> None: ...
    def set_completed(self, job_id: str) -> None: ...
    def set_failed(self, job_id: str, error: str) -> None: ...


class _NoOpTracker:
    """Used when no tracker is provided (e.g. in tests or CLI runs)."""

    def set_progress(self, job_id: str, records_processed: int) -> None:
        pass

    def set_completed(self, job_id: str) -> None:
        pass

    def set_failed(self, job_id: str, error: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Single-record worker (runs inside a thread)
# ---------------------------------------------------------------------------


def _classify_one(
    record: dict[str, Any],
    index: Any,
    metadata: list[dict[str, Any]],
    embedding_provider: EmbeddingProvider,
    llm_provider: Any,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Preprocess → retrieve → classify → blend confidence for one record."""
    cleaned = clean_record(record)

    query = f"{cleaned['customer_name']} {cleaned['address']} {cleaned['city']} {cleaned['state']}"
    hits = retrieve(query, index, metadata, embedding_provider, k=5)

    top_retrieval_score = hits[0]["score"] if hits else 0.0

    try:
        result = classify_record(cleaned, hits, llm_provider)
    except ClassificationError as exc:
        logger.warning("Classification failed for %s: %s", record.get("customer_id"), exc)
        return {
            **cleaned,
            "national_regional": "",
            "unit_type": "",
            "premise": "",
            "channel": "",
            "establishment_type": "",
            "confidence": 0.0,
            "needs_review": True,
            "reason": "",
            "error": str(exc),
        }

    final_confidence, needs_review = compute_confidence(
        llm_confidence=result.confidence,
        retrieval_score=top_retrieval_score,
        threshold=confidence_threshold,
    )

    return {
        **cleaned,
        "national_regional": result.national_regional,
        "unit_type": result.unit_type,
        "premise": result.premise,
        "channel": result.channel,
        "establishment_type": result.establishment_type,
        "confidence": final_confidence,
        "needs_review": needs_review,
        "reason": result.reason,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    job_id: str,
    input_path: Path,
    output_path: Path,
    taxonomy_yaml: Path = TAXONOMY_YAML,
    confidence_threshold: float = 0.75,
    job_tracker: JobTracker | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: Any | None = None,
) -> int:
    """Run the full classification pipeline for one job.

    Args:
        job_id: Identifier for logging and job tracking.
        input_path: Path to the uploaded customer CSV.
        output_path: Path where the classified output CSV will be written.
        taxonomy_yaml: Override the default taxonomy YAML location.
        confidence_threshold: Records below this score get needs_review=True.
        job_tracker: Optional tracker; defaults to no-op.
        embedding_provider: Override the embedding backend (for testing).
        llm_provider: Override the LLM backend (for testing).

    Returns:
        Total number of records processed.

    Raises:
        Exception: Re-raised after marking the job as failed.
    """
    tracker = job_tracker or _NoOpTracker()
    log = logging.LoggerAdapter(logger, {"job_id": job_id})

    try:
        log.info("Validating input %s", input_path)
        n_rows = validate_csv_schema(input_path)
        log.info("Input valid — %d rows", n_rows)

        emb_provider = embedding_provider or get_default_provider()
        llm = llm_provider or get_default_llm_provider()

        log.info("Building/loading FAISS index from %s", taxonomy_yaml)
        index, metadata = build_index(taxonomy_yaml, emb_provider)
        log.info("Index ready — %d entries", len(metadata))

        records_processed = 0
        is_first_chunk = True

        for chunk_df in pd.read_csv(input_path, chunksize=CHUNK_SIZE, dtype=str):
            chunk_df = chunk_df.fillna("")
            chunk_records = chunk_df.to_dict(orient="records")

            output_rows: list[dict[str, Any]] = [None] * len(chunk_records)  # type: ignore[list-item]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                future_to_idx = {
                    pool.submit(
                        _classify_one,
                        rec,
                        index,
                        metadata,
                        emb_provider,
                        llm,
                        confidence_threshold,
                    ): i
                    for i, rec in enumerate(chunk_records)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        output_rows[idx] = future.result()
                    except Exception as exc:
                        log.error("Unhandled error on record %d: %s", idx, exc)
                        rec = chunk_records[idx]
                        output_rows[idx] = {
                            **rec,
                            "national_regional": "",
                            "unit_type": "",
                            "premise": "",
                            "channel": "",
                            "establishment_type": "",
                            "confidence": 0.0,
                            "needs_review": True,
                            "reason": "",
                            "error": f"Unhandled: {exc}",
                        }

            write_output_chunk(output_rows, output_path, is_first_chunk)
            is_first_chunk = False
            records_processed += len(chunk_records)

            tracker.set_progress(job_id, records_processed)
            log.info("Chunk done — %d / %d rows processed", records_processed, n_rows)

        tracker.set_completed(job_id)
        log.info("Pipeline complete — %d records written to %s", records_processed, output_path)
        return records_processed

    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        tracker.set_failed(job_id, str(exc))
        raise
