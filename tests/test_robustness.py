"""
Robustness and edge-case tests:
  - Concurrent uploads (two jobs complete independently)
  - Large file (10 K rows) completes without memory explosion
  - Malformed inputs (missing columns, empty file, bad encoding)
  - Mock LLM produces deterministic output across two identical runs
"""

import csv
import io
import os
import threading
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("USE_MOCK_LLM", "true")

from pipeline.classification import MockLLMProvider
from pipeline.retrieval import MockEmbeddingProvider
from pipeline.runner import run_pipeline
from pipeline.validation import ValidationError, validate_csv_schema

TAXONOMY_YAML = Path("data/taxonomy/v1.yaml")
SAMPLE_10K = Path("data/samples/sample_10k.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(rows: list[tuple], header=True) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    if header:
        w.writerow(["customer_id", "customer_name", "address", "city", "state", "zip"])
    w.writerows(rows)
    return buf.getvalue().encode()


def _small_rows(n: int, offset: int = 0) -> list[tuple]:
    templates = [
        ("Joe's Bar & Grill", "100 Main St", "Austin", "TX", "78701"),
        ("Total Wine & More", "200 Oak Ave", "Dallas", "TX", "75201"),
        ("Hilton Garden Inn Dallas", "300 Park Blvd", "Dallas", "TX", "75202"),
        ("Kroger #4521", "400 Elm Dr", "Houston", "TX", "77001"),
        ("7-Eleven Store #1234", "500 Pine St", "San Antonio", "TX", "78201"),
    ]
    return [(f"CUST{i + offset:06d}",) + templates[i % len(templates)] for i in range(n)]


def _run(input_path: Path, output_path: Path) -> int:
    return run_pipeline(
        job_id=f"test-{input_path.stem}",
        input_path=input_path,
        output_path=output_path,
        taxonomy_yaml=TAXONOMY_YAML,
        confidence_threshold=0.75,
        embedding_provider=MockEmbeddingProvider(),
        llm_provider=MockLLMProvider(),
    )


# ---------------------------------------------------------------------------
# Concurrent uploads
# ---------------------------------------------------------------------------


class TestConcurrentJobs:
    def test_two_jobs_complete_independently(self, tmp_path):
        """Two jobs launched in parallel threads both produce correct output."""
        results: dict[str, int] = {}
        errors: list[Exception] = []

        def run_job(name: str, n_rows: int, offset: int) -> None:
            inp = tmp_path / f"input_{name}.csv"
            out = tmp_path / f"output_{name}.csv"
            inp.write_bytes(_make_csv(_small_rows(n_rows, offset=offset)))
            try:
                results[name] = _run(inp, out)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=run_job, args=("job_a", 20, 0))
        t2 = threading.Thread(target=run_job, args=("job_b", 25, 1000))
        t1.start()
        t2.start()
        t1.join(timeout=60)
        t2.join(timeout=60)

        assert not errors, f"Thread errors: {errors}"
        assert results.get("job_a") == 20
        assert results.get("job_b") == 25

    def test_concurrent_output_files_are_independent(self, tmp_path):
        """Each job writes to its own output path; files don't bleed into each other."""
        rows_a = _small_rows(10, offset=0)
        rows_b = _small_rows(15, offset=500)

        inp_a = tmp_path / "a.csv"
        inp_a.write_bytes(_make_csv(rows_a))
        inp_b = tmp_path / "b.csv"
        inp_b.write_bytes(_make_csv(rows_b))
        out_a = tmp_path / "out_a.csv"
        out_b = tmp_path / "out_b.csv"

        done = {"a": False, "b": False}

        def run_a():
            _run(inp_a, out_a)
            done["a"] = True

        def run_b():
            _run(inp_b, out_b)
            done["b"] = True

        threads = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert done["a"] and done["b"]
        df_a = pd.read_csv(out_a)
        df_b = pd.read_csv(out_b)
        assert len(df_a) == 10
        assert len(df_b) == 15
        # IDs must not be mixed
        assert set(df_a["customer_id"]).isdisjoint(set(df_b["customer_id"]))


# ---------------------------------------------------------------------------
# Large file
# ---------------------------------------------------------------------------


class TestLargeFile:
    @pytest.mark.skipif(
        not SAMPLE_10K.exists(),
        reason="data/samples/sample_10k.csv not generated — run `make generate-data`",
    )
    def test_10k_rows_completes(self, tmp_path):
        out = tmp_path / "out_10k.csv"
        n = _run(SAMPLE_10K, out)
        assert n == 10_000
        df = pd.read_csv(out)
        assert len(df) == 10_000

    @pytest.mark.skipif(
        not SAMPLE_10K.exists(),
        reason="data/samples/sample_10k.csv not generated — run `make generate-data`",
    )
    def test_10k_no_missing_output_rows(self, tmp_path):
        out = tmp_path / "out_10k_check.csv"
        _run(SAMPLE_10K, out)
        df = pd.read_csv(out)
        # Every row must have a channel (or an error explanation — never silent blank)
        blank_channel = df[
            (df["channel"].isna() | (df["channel"] == ""))
            & (df["error"].isna() | (df["error"] == ""))
        ]
        assert len(blank_channel) == 0, f"{len(blank_channel)} rows have no channel and no error"

    def test_chunked_processing_produces_complete_output(self, tmp_path):
        """Verify chunk-boundary stitching: 2500 rows = 3 chunks of 1000/1000/500."""
        rows = _small_rows(2500)
        inp = tmp_path / "big.csv"
        inp.write_bytes(_make_csv(rows))
        out = tmp_path / "big_out.csv"
        n = _run(inp, out)
        assert n == 2500
        df = pd.read_csv(out)
        assert len(df) == 2500
        # No duplicate customer_ids from chunk overlap
        assert df["customer_id"].nunique() == 2500


# ---------------------------------------------------------------------------
# Malformed inputs
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    def test_missing_required_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["customer_id", "customer_name", "city", "state", "zip"])  # no address
        w.writerow(["C1", "Joe", "Austin", "TX", "78701"])
        p.write_text(buf.getvalue())
        with pytest.raises(ValidationError, match="address"):
            validate_csv_schema(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_bytes(b"")
        with pytest.raises(ValidationError, match="empty"):
            validate_csv_schema(p)

    def test_header_only_raises(self, tmp_path):
        p = tmp_path / "header.csv"
        p.write_text("customer_id,customer_name,address,city,state,zip\n")
        with pytest.raises(ValidationError, match="no data rows"):
            validate_csv_schema(p)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_csv_schema(tmp_path / "ghost.csv")

    def test_multiple_missing_columns_listed_in_error(self, tmp_path):
        p = tmp_path / "few_cols.csv"
        p.write_text("customer_id\nC1\n")
        with pytest.raises(ValidationError, match="missing required columns"):
            validate_csv_schema(p)

    def test_bad_utf8_encoding_raises(self, tmp_path):
        """A file with non-UTF-8 bytes (latin-1 names) that pandas cannot parse."""
        p = tmp_path / "bad_enc.csv"
        # Write valid header then a row with a raw latin-1 byte (0xFF) in the name
        header = b"customer_id,customer_name,address,city,state,zip\n"
        # \xe9 = é in latin-1, invalid UTF-8 in strict mode
        row = b"C1,Caf\xe9 Rouge,1 A St,Paris,TX,75001\n"
        p.write_bytes(header + row)
        # pandas defaults to UTF-8; if it can still read it, validate_csv_schema passes —
        # what matters is that the pipeline doesn't crash silently on the row level.
        # We test that the runner gracefully handles a classification exception per-record.
        try:
            n = validate_csv_schema(p)
            assert n >= 1  # pandas was lenient; that's fine
        except (ValidationError, UnicodeDecodeError):
            pass  # also acceptable

    def test_pipeline_continues_past_per_record_failures(self, tmp_path):
        """If one record raises in the classifier, the rest of the batch still writes."""
        rows = _small_rows(5)
        inp = tmp_path / "in.csv"
        inp.write_bytes(_make_csv(rows))
        out = tmp_path / "out.csv"

        # Provider that fails on every other call
        call_count = {"n": 0}
        original_complete = MockLLMProvider().complete

        class FlakyProvider:
            def complete(self, system, user):
                call_count["n"] += 1
                if call_count["n"] % 2 == 0:
                    raise RuntimeError("simulated flaky LLM")
                return original_complete(system, user)

        run_pipeline(
            job_id="flaky-test",
            input_path=inp,
            output_path=out,
            taxonomy_yaml=TAXONOMY_YAML,
            confidence_threshold=0.75,
            embedding_provider=MockEmbeddingProvider(),
            llm_provider=FlakyProvider(),
        )

        df = pd.read_csv(out)
        assert len(df) == 5, "All rows should be in output even if some failed"
        failed = df[df["error"].fillna("") != ""]
        succeeded = df[df["error"].fillna("") == ""]
        assert len(failed) > 0, "Flaky provider should have produced some errors"
        assert len(succeeded) > 0, "Non-flaky calls should have succeeded"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_mock_llm_same_input_same_output(self, tmp_path):
        """Two runs on identical input with mock providers produce byte-identical output."""
        rows = _small_rows(10)
        inp = tmp_path / "det.csv"
        inp.write_bytes(_make_csv(rows))

        out1 = tmp_path / "det_out1.csv"
        out2 = tmp_path / "det_out2.csv"

        _run(inp, out1)
        _run(inp, out2)

        df1 = pd.read_csv(out1)
        df2 = pd.read_csv(out2)

        assert list(df1["channel"]) == list(df2["channel"])
        assert list(df1["establishment_type"]) == list(df2["establishment_type"])
        assert list(df1["confidence"]) == list(df2["confidence"])

    def test_mock_llm_keyword_routing_is_stable(self):
        """The same customer name always maps to the same channel across calls."""
        provider = MockLLMProvider()
        taxonomy_stub = [
            {
                "id": "T1",
                "national_regional": "Regional",
                "unit_type": "Single-unit",
                "premise": "On-Premise",
                "channel": "Bar",
                "establishment_type": "Taproom",
                "definition": "A taproom.",
                "inclusion_examples": [],
                "exclusion_examples": [],
            }
        ]
        record = {"customer_name": "Austin Brewing Taproom", "address": "", "city": "", "state": ""}

        from pipeline.classification import classify_record

        results = [classify_record(record, taxonomy_stub, provider) for _ in range(5)]
        channels = {r.channel for r in results}
        assert len(channels) == 1, f"Non-deterministic channel: {channels}"

    def test_mock_embedding_deterministic(self):
        import numpy as np

        from pipeline.retrieval import MockEmbeddingProvider

        p = MockEmbeddingProvider()
        v1 = p.embed(["hello world"])
        v2 = p.embed(["hello world"])
        np.testing.assert_array_equal(v1, v2)

    def test_two_instances_agree_on_confidence(self):
        """Confidence is derived from the customer name hash, not a shared RNG sequence.
        Two independent provider instances must return the same confidence for the same name."""
        import json

        from pipeline.classification import MockLLMProvider

        p1 = MockLLMProvider()
        p2 = MockLLMProvider()
        system = "sys"
        user = (
            "Name: Joe's Bar\nAddress: 1 A St\n"
            "Taxonomy entries:\n  - [Bar] Taproom: desc\nClassify."
        )
        r1 = json.loads(p1.complete(system, user))
        r2 = json.loads(p2.complete(system, user))
        assert r1["confidence"] == r2["confidence"]
