"""Thread-safe in-memory job tracker."""

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class Job:
    job_id: str
    status: JobStatus
    input_path: Path
    output_path: Path
    taxonomy_version: str
    confidence_threshold: float
    records_processed: int = 0
    records_total: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


class JobStore:
    """Thread-safe dict-backed job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        job_id: str,
        input_path: Path,
        output_path: Path,
        taxonomy_version: str = "v1",
        confidence_threshold: float = 0.75,
        records_total: int = 0,
    ) -> Job:
        job = Job(
            job_id=job_id,
            status="pending",
            input_path=input_path,
            output_path=output_path,
            taxonomy_version=taxonomy_version,
            confidence_threshold=confidence_threshold,
            records_total=records_total,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            if status == "running" and job.started_at is None:
                job.started_at = datetime.now(UTC)
            elif status in ("completed", "failed"):
                job.completed_at = datetime.now(UTC)

    def set_progress(self, job_id: str, records_processed: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.records_processed = records_processed

    def set_completed(self, job_id: str) -> None:
        self.set_status(job_id, "completed")

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error_message = error
            job.completed_at = datetime.now(UTC)

    # ------------------------------------------------------------------
    # JobTracker protocol — satisfies pipeline.runner.JobTracker
    # ------------------------------------------------------------------

    def set_error(self, job_id: str, error: str) -> None:
        self.set_failed(job_id, error)


# Module-level singleton shared across the app process
job_store = JobStore()
