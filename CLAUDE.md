# SGWS Customer Classification Replica

## Purpose

This is a working replica of an internal batch classification tool. The 
goal is to demonstrate the architecture end-to-end for interview prep.

## What it does

Analysts upload a CSV of customer accounts (customer_name, address). The 
system classifies each account into a 5-level business taxonomy 
(National/Regional, Multi/Single-unit, Premise, Channel, Establishment 
Type), returns confidence scores and manual-review flags, and outputs a 
CSV the analyst downloads.

## Architecture

- **Frontend**: Streamlit (file upload, configuration, status polling, download)
- **Backend**: FastAPI (upload, run, status, download endpoints)
- **Pipeline**: Python batch processor running as background task
- **Retrieval**: FAISS in-memory index over ~4000 taxonomy entries
- **LLM**: OpenAI API (with mock mode for offline testing)
- **Storage**: Local filesystem under ./data/jobs/{job_id}/

## Key design decisions

- FAISS in-memory (not a vector DB service) — taxonomy is small and static
- Background tasks in FastAPI (not Celery) — internal tool, low concurrency
- In-memory job tracking (not a database) — restart resilience accepted
- Mock LLM mode for offline testing (controlled by env var)
- Chunked processing (1000 rows at a time) with ThreadPoolExecutor (10 workers)

## Tech stack

- Python 3.11+
- FastAPI, uvicorn, pydantic
- Streamlit
- pandas
- faiss-cpu
- openai
- pytest, pytest-asyncio
- ruff

## Coding standards

- Type hints on all function signatures
- Pydantic models for all API requests/responses
- Structured logging with job_id in every log line
- No hardcoded credentials — use environment variables
- Tests for all deterministic logic; mock external APIs

## Directory structure

```
sgws-classification-replica/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry
│   ├── routes/
│   │   └── classification.py
│   ├── schemas/
│   │   └── requests.py      # Pydantic models
│   └── jobs.py              # In-memory job tracking
├── frontend/
│   └── streamlit_app.py
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── retrieval.py         # FAISS
│   ├── classification.py    # LLM
│   ├── confidence.py
│   ├── export.py
│   └── runner.py            # Orchestrates the pipeline
├── data/
│   ├── taxonomy/
│   │   └── v1.yaml          # Reference taxonomy
│   └── jobs/                # Per-job input/output (gitignored)
├── scripts/
│   ├── generate_taxonomy.py
│   ├── generate_input.py
│   └── build_embeddings.py
└── tests/
    ├── test_validation.py
    ├── test_preprocessing.py
    ├── test_pipeline.py
    └── test_api.py
```