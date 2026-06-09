.PHONY: install test lint format run-api run-ui generate-data build-index clean

PYTHON   ?= python3
PIP      ?= pip
VENV     := venv
ACTIVATE := . $(VENV)/bin/activate &&

# ── Environment ──────────────────────────────────────────────────────────────

install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) $(PIP) install --upgrade pip
	$(ACTIVATE) $(PIP) install -r requirements.txt
	@echo ""
	@echo "✅  Environment ready. Run: source venv/bin/activate"

# ── Quality gate ─────────────────────────────────────────────────────────────

lint:
	ruff check .

format:
	ruff format .

test:
	bash scripts/run_all_tests.sh

# ── Data generation ──────────────────────────────────────────────────────────

generate-data:
	$(PYTHON) scripts/generate_taxonomy.py
	$(PYTHON) scripts/generate_input.py --rows 100   --output data/samples/sample_100.csv  --seed 1
	$(PYTHON) scripts/generate_input.py --rows 1000  --output data/samples/sample_1k.csv   --seed 2
	$(PYTHON) scripts/generate_input.py --rows 10000 --output data/samples/sample_10k.csv  --seed 3
	@echo "✅  Taxonomy + sample CSVs written."

build-index:
	$(PYTHON) scripts/build_embeddings.py --mock
	@echo "✅  FAISS index built at data/taxonomy/v1.index"

# ── Run services ─────────────────────────────────────────────────────────────

run-api:
	USE_MOCK_LLM=true uvicorn app.main:app --reload --port 8000

run-ui:
	streamlit run frontend/streamlit_app.py

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache htmlcov .coverage
	rm -rf data/jobs/*
	@echo "✅  Cleaned."
