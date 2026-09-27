# Developer entry points for the Meher Sweets & Namkeen agent.
#
# Every variable can be overridden on the command line:
#   make test
#   make eval CASES=evals/seed_cases.jsonl REPEATS=1
#   make serve PORT=9000 MODEL=llama-3.3-70b-versatile
#   make package
#
# `serve` must be running before `eval`; the runner polls GET /health for up
# to 60s and fails if the service never answers.
#
# SHELL is POSIX sh; the recipes are POSIX, not cmd.exe.

SHELL := /bin/sh

PY         ?= python3
HOST       ?= 127.0.0.1
PORT       ?= 8000
BASE_URL   ?= http://127.0.0.1:8000
CASES      ?= evals/cases.jsonl
SEED_CASES ?= evals/seed_cases.jsonl
REPEATS    ?= 3
MODEL      ?= qwen2.5:7b-instruct
PACKAGE    ?= dhanur-task-yashi-gupta.zip

# `src` layout: the editable install puts meher_agent on the path, and the
# fallback keeps `make test` working in a bare checkout with no install.
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))
export LLM_MODEL := $(MODEL)

.DEFAULT_GOAL := help
.PHONY: help install serve test eval eval-seed lint format clean report package cases check-cases

help:
	@printf 'targets: install serve test eval eval-seed lint format clean report package cases check-cases\n'
	@printf 'overridable: PY HOST PORT BASE_URL CASES SEED_CASES REPEATS MODEL PACKAGE\n'

install:
	$(PY) -m pip install -e '.[dev]'

serve:
	$(PY) -m uvicorn meher_agent.api.app:app --host $(HOST) --port $(PORT)

test:
	$(PY) -m pytest tests -q

eval:
	$(PY) -m evals.runner $(CASES) --repeats $(REPEATS) --base-url $(BASE_URL)

eval-seed:
	$(PY) -m evals.runner $(SEED_CASES) --repeats $(REPEATS) --base-url $(BASE_URL)

# No linter is pinned in pyproject.toml; compileall is the stdlib syntax gate.
lint:
	$(PY) -m compileall -q src evals scripts tests

# ruff is the intended formatter (pip install ruff); black is accepted too.
format:
	@if $(PY) -m ruff --version >/dev/null 2>&1; then $(PY) -m ruff format src evals scripts tests; elif $(PY) -m black --version >/dev/null 2>&1; then $(PY) -m black src evals scripts tests; else echo "error: no formatter installed (pip install ruff)" >&2; exit 1; fi

report:
	@if [ -f reports/summary.md ]; then cat reports/summary.md; else echo "error: reports/summary.md not found - run 'make eval' first" >&2; exit 1; fi

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f $(PACKAGE) ../$(PACKAGE)

package:
	$(PY) scripts/package_submission.py --out ../$(PACKAGE)

# The oracle owns every derived rupee figure in the case file, so these two
# targets are the only sanctioned way to edit cases.jsonl.
cases:
	$(PY) scripts/compute_expected_totals.py --cases $(CASES) --write

check-cases:
	$(PY) scripts/compute_expected_totals.py --cases $(CASES) --check
