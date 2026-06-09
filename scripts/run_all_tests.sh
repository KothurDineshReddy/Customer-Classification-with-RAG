#!/usr/bin/env bash
# Run the full quality gate: lint → format-check → tests.
# Exit code is 0 only when every step passes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0
ERRORS=()

step() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "▶  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
ok()   { echo "✅  $*"; ((PASS++)) || true; }
fail() { echo "❌  $*"; ((FAIL++)) || true; ERRORS+=("$*"); }

# ── 1. Ruff lint ────────────────────────────────────────────────────────────
step "ruff check (lint)"
if ruff check .; then ok "ruff lint"; else fail "ruff lint"; fi

# ── 2. Ruff format (check only, no writes) ──────────────────────────────────
step "ruff format --check"
if ruff format --check .; then ok "ruff format"; else fail "ruff format"; fi

# ── 3. Pytest ───────────────────────────────────────────────────────────────
step "pytest"
if python -m pytest -v --tb=short; then ok "pytest"; else fail "pytest"; fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: ${PASS} passed  ${FAIL} failed"
if [ ${#ERRORS[@]} -gt 0 ]; then
  echo "  Failed steps:"
  for e in "${ERRORS[@]}"; do echo "    • $e"; done
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ "$FAIL" -eq 0 ]
