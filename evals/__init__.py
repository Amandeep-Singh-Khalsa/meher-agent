"""Offline evaluation harness for the Meher Sweets agent.

This is a standalone top-level package. `python -m evals.runner ...` does not
put `src/` on `sys.path` by itself, and the reviewer is explicitly not required
to export `PYTHONPATH`, so the package installs it here instead of relying on
the invocation. It is appended (not prepended) so a reviewer-provided
`PYTHONPATH` still wins.

Nothing in here imports `meher_agent.api`: the harness talks to the service over
HTTP only, so it can be pointed at any OpenAI-compatible or Meher endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["REPO_ROOT", "SRC_DIR", "DATA_DIR", "DEFAULT_CASES", "DEFAULT_OUT_DIR"]

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CASES = REPO_ROOT / "evals" / "cases.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "reports"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
