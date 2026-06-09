# Customer Classification Replica

A working replica of an internal batch classification tool used at a major
beverage distributor. Analysts upload a CSV of customer accounts; the system
classifies each into a 5-level business taxonomy using FAISS-based retrieval
and an LLM, then returns a downloadable CSV with confidence scores and
manual-review flags.

---

## Quick Start (5 commands)

```bash
git clone <this-repo> && cd sgws-classification-replica
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # USE_MOCK_LLM=true by default — no API key needed
make generate-data build-index run-api   # terminal 1
# open a second terminal:
source venv/bin/activate && make run-ui
```

Open **http://localhost:8501** and upload `data/samples/sample_100.csv`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  :8501                           │
│  [Upload CSV] → [Configure] → [Run] → [Progress bar] → [Download]   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  HTTP (httpx)
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI backend  :8000                          │
│                                                                      │
│  POST /upload ──► validate schema ──► save input.csv                │
│  POST /run    ──► BackgroundTask ──► run_pipeline()                  │
│  GET  /status ──► in-memory JobStore (thread-safe dict)              │
│  GET  /download ──► FileResponse(output.csv)                         │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Pipeline (background thread)                     │
│                                                                      │
│  validate_csv_schema()                                               │
│       │                                                              │
│       ▼                                                              │
│  pandas.read_csv(chunksize=1000)                                     │
│       │                                                              │
│       ▼  per chunk                                                   │
│  ThreadPoolExecutor(max_workers=10)                                  │
│       │                                                              │
│       ▼  per record                                                  │
│  clean_record()   ← preprocessing (strip, normalize, de-suffix)     │
│       │                                                              │
│       ▼                                                              │
│  retrieve()       ← FAISS IndexFlatIP  (cosine similarity, top-5)   │
│       │              └── v1.index  (4 052 taxonomy entries)         │
│       │                                                              │
│       ▼                                                              │
│  classify_record() ← OpenAI gpt-4o-mini  (or MockLLMProvider)       │
│       │              prompt: customer info + top-5 taxonomy entries  │
│       │              response_format=json_object, temperature=0.1    │
│       │                                                              │
│       ▼                                                              │
│  compute_confidence()  ← 0.7×llm + 0.3×retrieval_score              │
│       │                                                              │
│       ▼                                                              │
│  write_output_chunk()  ← appends CSV rows after each chunk          │
└──────────────────────────────────────────────────────────────────────┘

Storage layout
  data/
  ├── taxonomy/
  │   ├── v1.yaml          4 052 reference taxonomy entries
  │   ├── v1.index         FAISS binary index (float32, dim=1536)
  │   └── v1.meta.json     taxonomy metadata parallel to index
  ├── jobs/
  │   └── {uuid}/
  │       ├── input.csv    uploaded file
  │       └── output.csv   classified results
  └── samples/
      ├── sample_100.csv   quick test
      ├── sample_1k.csv    integration test
      └── sample_10k.csv   stress test
```

---

## How to Demo This

> Assumes `make generate-data build-index` has been run and both servers are
> running (`make run-api` / `make run-ui`).

**Step 1 — Upload**

1. Open http://localhost:8501
2. Click **Browse files** and select `data/samples/sample_100.csv`
3. A 5-row preview appears: `customer_id`, `customer_name`, `address`, …
4. Click **⬆️ Upload to API** → green banner: *"Uploaded 100 rows — Job ID: …"*

**Step 2 — Configure**

5. Leave **Taxonomy version** as `v1`
6. Drag **Confidence threshold** slider — default `0.75` is a good demo value
7. Click **▶️ Run Classification**

**Step 3 — Watch progress**

8. A progress bar appears and auto-updates every 3 seconds
9. Status cycles: `PENDING → RUNNING → COMPLETED`
10. Mock mode is fast (~5 s for 100 rows); live OpenAI adds ~30 s

**Step 4 — Download & inspect**

11. **⬇️ Download Output CSV** button appears — click to save
12. The 10-row preview shows:
    - `channel` (Restaurant, Bar, Hotel, Grocery, …)
    - `establishment_type` (Casual Dining, Taproom, Chain Liquor Superstore, …)
    - `confidence` (0.52–0.95 in mock mode; higher with real embeddings)
    - Amber-highlighted rows where `needs_review = True`

**Step 5 — Explore the API directly**

```bash
open http://localhost:8000/docs   # interactive Swagger UI
```

---

## Design Decisions

### FAISS in-memory index (not a vector DB service)

The taxonomy has ~4 000 entries and is updated at most a few times a year.
A cloud vector DB (Pinecone, Weaviate) would add a network hop, an API key,
and a monthly bill for a dataset that fits in 25 MB of RAM.
`IndexFlatIP` over normalised vectors gives exact cosine similarity with
sub-millisecond query latency. The index is built once with
`make build-index` and loaded from `v1.index` on every API startup.

### FastAPI background tasks (not Celery)

This is an internal tool with low concurrency — typically one analyst at a
time. FastAPI's `BackgroundTasks` runs the pipeline in a thread on the same
process with no infrastructure overhead (no broker, no worker pool to operate).
The accepted trade-off is that restarting the API server loses in-flight job
state, which is acceptable for an internal tool.

### Mock LLM mode (`USE_MOCK_LLM=true`)

The mock provider uses MD5-seeded random vectors for embeddings and
keyword-based routing for classification. Both are **fully deterministic per
record** — the same customer name always produces the same channel and
confidence score regardless of call order or thread scheduling. This lets the
test suite run in CI with zero API cost and zero flakiness.

### Chunked processing with `ThreadPoolExecutor`

`pandas.read_csv(chunksize=1000)` keeps memory constant for arbitrarily large
files — only one chunk of 1 000 rows is in memory at a time. Within each
chunk, 10 threads run `classify_one_record` concurrently, saturating OpenAI's
per-request latency while respecting the rate limit. Per-record exceptions
are caught, written to the `error` column, and the batch continues — no
silent data loss.

---

## Makefile reference

| Target | What it does |
|---|---|
| `make install` | Create venv, install deps |
| `make test` | ruff lint + format check + pytest |
| `make lint` | ruff check only |
| `make format` | ruff format (writes changes) |
| `make generate-data` | Generate taxonomy YAML + sample CSVs |
| `make build-index` | Build FAISS index from v1.yaml |
| `make run-api` | Start FastAPI on :8000 (mock mode) |
| `make run-ui` | Start Streamlit on :8501 |
| `make clean` | Remove caches, pyc, job data |

---

## Input format

| Column | Required | Description |
|---|---|---|
| `customer_id` | ✓ | Unique identifier |
| `customer_name` | ✓ | Business name |
| `address` | ✓ | Street address |
| `city` | ✓ | City |
| `state` | ✓ | Two-letter state code |
| `zip` | ✓ | ZIP code |

## Output format

All input columns are preserved. Added columns:

| Column | Description |
|---|---|
| `national_regional` | `National` or `Regional` |
| `unit_type` | `Multi-unit` or `Single-unit` |
| `premise` | `On-Premise` or `Off-Premise` |
| `channel` | Top-level channel (Restaurant, Bar, Grocery, …) |
| `establishment_type` | Specific type within channel |
| `confidence` | Blended score 0–1 (0.7×LLM + 0.3×retrieval) |
| `needs_review` | `True` if confidence < threshold |
| `reason` | One-sentence justification from the LLM |
| `error` | Non-empty if the record failed to classify |

---

## Running tests

```bash
make test          # full gate: lint + format + pytest
pytest -v          # pytest only
pytest tests/test_robustness.py -v   # robustness suite only
```

Total: **116 tests** across 6 test modules.
