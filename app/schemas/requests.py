"""Pydantic request/response models for the classification API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    status: str
    message: str


class RunRequest(BaseModel):
    job_id: str
    taxonomy_version: str = "v1"
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class StatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    records_processed: int
    records_total: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None


class ErrorResponse(BaseModel):
    detail: str
