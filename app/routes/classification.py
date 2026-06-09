"""Classification API routes."""

import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.jobs import job_store
from app.schemas.requests import (
    RunRequest,
    StatusResponse,
    UploadResponse,
)
from pipeline.validation import ValidationError, validate_csv_schema

logger = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
TAXONOMY_DIR = DATA_DIR / "taxonomy"


def _job_dir(job_id: str) -> Path:
    return DATA_DIR / "jobs" / job_id


def _status_response(job_id: str) -> StatusResponse:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        records_processed=job.records_processed,
        records_total=job.records_total,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


def _run_pipeline_task(job_id: str) -> None:
    """Background task that runs the pipeline and updates job state."""
    from pipeline.runner import run_pipeline

    job = job_store.get_job(job_id)
    if job is None:
        logger.error("Background task: job %s not found", job_id)
        return

    job_store.set_status(job_id, "running")
    taxonomy_yaml = TAXONOMY_DIR / f"{job.taxonomy_version}.yaml"

    try:
        run_pipeline(
            job_id=job_id,
            input_path=job.input_path,
            output_path=job.output_path,
            taxonomy_yaml=taxonomy_yaml,
            confidence_threshold=job.confidence_threshold,
            job_tracker=job_store,
        )
    except Exception as exc:
        logger.error("Pipeline failed for job %s: %s", job_id, exc)
        # run_pipeline already called job_tracker.set_failed; log only


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    """Accept a CSV file, validate its schema, persist it, and create a job."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    job_id = str(uuid.uuid4())
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.csv"
    contents = await file.read()
    input_path.write_bytes(contents)

    try:
        n_rows = validate_csv_schema(input_path)
    except ValidationError as exc:
        input_path.unlink(missing_ok=True)
        job_dir.rmdir()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_path = job_dir / "output.csv"
    job_store.create_job(
        job_id=job_id,
        input_path=input_path,
        output_path=output_path,
        records_total=n_rows,
    )

    logger.info("Uploaded job %s — %d rows", job_id, n_rows)
    return UploadResponse(
        job_id=job_id,
        status="pending",
        message=f"Accepted {n_rows} rows. POST /run to start classification.",
    )


@router.post("/run", response_model=StatusResponse)
async def run(req: RunRequest, background_tasks: BackgroundTasks) -> StatusResponse:
    """Start the pipeline for an existing job as a background task."""
    job = job_store.get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id!r} not found")
    if job.status not in ("pending",):
        raise HTTPException(
            status_code=409,
            detail=f"Job {req.job_id!r} is already {job.status}",
        )

    # Persist run-time parameters back onto the job before the task starts
    job.taxonomy_version = req.taxonomy_version
    job.confidence_threshold = req.confidence_threshold

    background_tasks.add_task(_run_pipeline_task, req.job_id)
    # Status is still "pending" at this instant; the background task flips it
    return _status_response(req.job_id)


@router.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str) -> StatusResponse:
    return _status_response(job_id)


@router.get("/download/{job_id}")
async def download(job_id: str) -> FileResponse:
    """Stream the output CSV once the job is completed."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.status == "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id!r} failed: {job.error_message}",
        )
    if job.status != "completed":
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id!r} output is not yet available (status: {job.status})",
        )
    if not job.output_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Output file missing for completed job {job_id!r}",
        )
    return FileResponse(
        path=str(job.output_path),
        media_type="text/csv",
        filename=f"classification_{job_id}.csv",
    )
