"""FastAPI endpoint tests using TestClient."""

import csv
import io
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Force mock mode before importing the app so no OpenAI calls are made
os.environ.setdefault("USE_MOCK_LLM", "true")

from app.jobs import job_store
from app.main import app

TAXONOMY_YAML = Path("data/taxonomy/v1.yaml")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(rows: list[tuple], *, include_header: bool = True) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    if include_header:
        writer.writerow(["customer_id", "customer_name", "address", "city", "state", "zip"])
    writer.writerows(rows)
    return buf.getvalue().encode()


SAMPLE_ROWS = [
    ("CUST000001", "Joe's Bar & Grill", "100 Main St", "Austin", "TX", "78701"),
    ("CUST000002", "Total Wine & More", "200 Oak Ave", "Dallas", "TX", "75201"),
    ("CUST000003", "Hilton Garden Inn", "300 Park Blvd", "Dallas", "TX", "75202"),
    ("CUST000004", "Kroger #4521", "400 Elm Dr", "Houston", "TX", "77001"),
    ("CUST000005", "7-Eleven Store #1234", "500 Pine St", "San Antonio", "TX", "78201"),
]


@pytest.fixture()
def client():
    """Fresh TestClient; resets job_store between tests."""
    job_store._jobs.clear()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _upload(client: TestClient, rows=None) -> str:
    """Upload a CSV and return the job_id."""
    data = _make_csv(rows or SAMPLE_ROWS)
    resp = client.post("/upload", files={"file": ("sample.csv", data, "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()["job_id"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class TestUpload:
    def test_valid_csv_returns_200_with_job_id(self, client):
        data = _make_csv(SAMPLE_ROWS)
        resp = client.post("/upload", files={"file": ("test.csv", data, "text/csv")})
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "pending"
        assert len(body["job_id"]) > 0

    def test_upload_creates_job_in_store(self, client):
        data = _make_csv(SAMPLE_ROWS)
        resp = client.post("/upload", files={"file": ("test.csv", data, "text/csv")})
        job_id = resp.json()["job_id"]
        assert job_store.get_job(job_id) is not None

    def test_upload_records_row_count(self, client):
        data = _make_csv(SAMPLE_ROWS)
        resp = client.post("/upload", files={"file": ("test.csv", data, "text/csv")})
        job_id = resp.json()["job_id"]
        assert job_store.get_job(job_id).records_total == len(SAMPLE_ROWS)

    def test_upload_rejects_missing_column(self, client):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["customer_id", "customer_name", "city", "state", "zip"])  # address missing
        w.writerow(["C1", "Joe", "Austin", "TX", "78701"])
        data = buf.getvalue().encode()
        resp = client.post("/upload", files={"file": ("bad.csv", data, "text/csv")})
        assert resp.status_code == 400
        assert "address" in resp.json()["detail"].lower()

    def test_upload_rejects_non_csv_extension(self, client):
        data = b"not,a,csv\n"
        resp = client.post("/upload", files={"file": ("data.txt", data, "text/plain")})
        assert resp.status_code == 400

    def test_upload_rejects_empty_file(self, client):
        resp = client.post("/upload", files={"file": ("empty.csv", b"", "text/csv")})
        assert resp.status_code == 400

    def test_upload_saves_file_to_disk(self, client):
        data = _make_csv(SAMPLE_ROWS)
        resp = client.post("/upload", files={"file": ("test.csv", data, "text/csv")})
        job_id = resp.json()["job_id"]
        job = job_store.get_job(job_id)
        assert job.input_path.exists()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_returns_status_response(self, client):
        job_id = _upload(client)
        resp = client.post("/run", json={"job_id": job_id})
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] in ("pending", "running")

    def test_run_unknown_job_returns_404(self, client):
        resp = client.post("/run", json={"job_id": "does-not-exist"})
        assert resp.status_code == 404

    def test_run_invalid_threshold_rejected(self, client):
        job_id = _upload(client)
        resp = client.post("/run", json={"job_id": job_id, "confidence_threshold": 1.5})
        assert resp.status_code == 422

    def test_run_already_running_returns_409(self, client):
        job_id = _upload(client)
        job_store.set_status(job_id, "running")
        resp = client.post("/run", json={"job_id": job_id})
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_pending_after_upload(self, client):
        job_id = _upload(client)
        resp = client.get(f"/status/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "pending"
        assert body["records_total"] == len(SAMPLE_ROWS)

    def test_status_unknown_job_returns_404(self, client):
        resp = client.get("/status/no-such-job")
        assert resp.status_code == 404

    def test_status_has_all_required_fields(self, client):
        job_id = _upload(client)
        body = client.get(f"/status/{job_id}").json()
        required = {
            "job_id",
            "status",
            "records_processed",
            "records_total",
            "created_at",
            "started_at",
            "completed_at",
            "error_message",
        }
        assert required.issubset(body.keys())


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class TestDownload:
    def test_download_returns_404_when_pending(self, client):
        job_id = _upload(client)
        resp = client.get(f"/download/{job_id}")
        assert resp.status_code == 404

    def test_download_returns_404_when_running(self, client):
        job_id = _upload(client)
        job_store.set_status(job_id, "running")
        resp = client.get(f"/download/{job_id}")
        assert resp.status_code == 404

    def test_download_returns_400_when_failed(self, client):
        job_id = _upload(client)
        job_store.set_failed(job_id, "something broke")
        resp = client.get(f"/download/{job_id}")
        assert resp.status_code == 400

    def test_download_unknown_job_returns_404(self, client):
        resp = client.get("/download/ghost-job")
        assert resp.status_code == 404

    def test_download_returns_csv_when_completed(self, client, tmp_path):
        job_id = _upload(client)
        job = job_store.get_job(job_id)
        # Write a fake output file and mark the job completed
        job.output_path.write_text("customer_id,channel\nC1,Bar\n")
        job_store.set_status(job_id, "completed")

        resp = client.get(f"/download/{job_id}")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "customer_id" in resp.text


# ---------------------------------------------------------------------------
# Happy-path E2E: upload → run (background) → poll → download
# ---------------------------------------------------------------------------


class TestE2EHappyPath:
    def test_full_pipeline_with_mock_llm(self, client):

        # Upload
        rows = [
            ("CUST000001", "Joe's Bar & Grill", "100 Main St", "Austin", "TX", "78701"),
            ("CUST000002", "Total Wine & More", "200 Oak Ave", "Dallas", "TX", "75201"),
            ("CUST000003", "Hilton Garden Inn Dallas", "300 Park Blvd", "Dallas", "TX", "75202"),
            ("CUST000004", "Kroger #4521", "400 Elm Dr", "Houston", "TX", "77001"),
            ("CUST000005", "7-Eleven Store #1234", "500 Pine St", "San Antonio", "TX", "78201"),
            ("CUST000006", "Mama Rosa's Pizzeria", "601 Cedar Ln", "Chicago", "IL", "60601"),
            ("CUST000007", "Austin Brewing Taproom", "702 Walnut Way", "Austin", "TX", "78702"),
            ("CUST000008", "Walgreens", "803 Maple Ave", "Denver", "CO", "80201"),
            ("CUST000009", "Outback Steakhouse", "904 Chestnut Pkwy", "Phoenix", "AZ", "85001"),
            ("CUST000010", "Sam's Club #842", "1005 Birch Rd", "Las Vegas", "NV", "89101"),
        ]
        data = _make_csv(rows)
        up = client.post("/upload", files={"file": ("e2e.csv", data, "text/csv")})
        assert up.status_code == 200
        job_id = up.json()["job_id"]

        # Patch providers so the background task uses mocks and the cached index
        with (
            patch(
                "app.routes.classification._run_pipeline_task",
                wraps=lambda jid: _run_with_mocks(jid),
            ),
        ):
            run_resp = client.post("/run", json={"job_id": job_id})
            assert run_resp.status_code == 200

        # Poll until done (background task runs synchronously inside TestClient)
        deadline = time.time() + 30
        while time.time() < deadline:
            st = client.get(f"/status/{job_id}").json()
            if st["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert st["status"] == "completed", f"Job failed: {st.get('error_message')}"
        assert st["records_processed"] == len(rows)

        # Download
        dl = client.get(f"/download/{job_id}")
        assert dl.status_code == 200
        assert "text/csv" in dl.headers["content-type"]

        # Parse and validate output
        import pandas as pd

        df = pd.read_csv(io.StringIO(dl.text))
        assert len(df) == len(rows)

        required_cols = {
            "customer_id",
            "customer_name",
            "channel",
            "establishment_type",
            "confidence",
            "needs_review",
        }
        assert required_cols.issubset(set(df.columns))
        assert (df["confidence"] >= 0).all()
        assert (df["confidence"] <= 1).all()


def _run_with_mocks(job_id: str) -> None:
    """Thin wrapper used by the E2E test to inject mock providers."""
    from pipeline.classification import MockLLMProvider
    from pipeline.retrieval import MockEmbeddingProvider
    from pipeline.runner import run_pipeline

    job = job_store.get_job(job_id)
    if job is None:
        return

    job_store.set_status(job_id, "running")
    taxonomy_yaml = Path("data/taxonomy/v1.yaml")

    try:
        run_pipeline(
            job_id=job_id,
            input_path=job.input_path,
            output_path=job.output_path,
            taxonomy_yaml=taxonomy_yaml,
            confidence_threshold=job.confidence_threshold,
            job_tracker=job_store,
            embedding_provider=MockEmbeddingProvider(),
            llm_provider=MockLLMProvider(),
        )
    except Exception as exc:
        job_store.set_failed(job_id, str(exc))
