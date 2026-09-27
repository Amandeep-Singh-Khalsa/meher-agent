"""Builds ``dhanur-task-yashi-gupta.zip`` from the repository tree.

The tree is walked, not git-archived, so the packager works on any machine and
the output is byte-reproducible: entries are sorted, every ZipInfo carries a
fixed timestamp and fixed permissions, and the deflate level is pinned. Two runs
over an unchanged tree produce the same SHA-256.

    python scripts/package_submission.py            # ../dhanur-task-yashi-gupta.zip
    python scripts/package_submission.py --check    # report what is missing; build nothing

Hard failures (non-zero exit, nothing shipped):

* a required file is missing (README.md, TECHNICAL_REPORT.md, Makefile,
  pyproject.toml, the data files, ...);
* credential-shaped text (``sk-...``, ``AKIA...``) is found in anything that
  would be packaged.

``.env`` is never packaged; ``.env.example`` is. Partial eval output
(``reports/partial-*.json``) and build noise (``.venv``, ``__pycache__``,
``.git``, ``*.egg-info``, ``*.pyc``) are excluded.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = [
    "MissingFiles",
    "SecretFound",
    "PackageResult",
    "build_package",
    "check_tree",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "dhanur-task-yashi-gupta.zip"
DEFAULT_OUT = REPO_ROOT.parent / DEFAULT_NAME

#: Fixed zip metadata: two runs over an unchanged tree must hash identically.
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_EXTERNAL_ATTR = 0o100644 << 16
FIXED_COMPRESS_LEVEL = 9

#: Directory names never worth shipping.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        "build",
        "dist",
        ".idea",
        ".vscode",
    }
)
#: File names never worth shipping.
EXCLUDED_FILES: frozenset[str] = frozenset({".DS_Store", "Thumbs.db"})
#: Glob patterns never worth shipping (matched against the file name).
EXCLUDED_GLOBS: tuple[str, ...] = ("*.pyc", "*.pyo", "*.log", "*.tmp")

#: Files the graded submission cannot exist without.
REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "TECHNICAL_REPORT.md",
    "Makefile",
    "pyproject.toml",
    "config.toml",
    ".gitattributes",
    ".gitignore",
    ".env.example",
    "data/business.md",
    "data/policies.md",
    "data/prices.csv",
    "docs/INTERNAL_CONTRACT.md",
    "evals/cases.jsonl",
    "evals/seed_cases.jsonl",
    "evals/runner.py",
    "evals/report.py",
    "evals/checks.py",
    "scripts/package_submission.py",
    "scripts/compute_expected_totals.py",
    "src/meher_agent/api/app.py",
)

#: A key that must never leave the machine, whatever file it is hiding in.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai-style key", re.compile(r"sk-[A-Za-z0-9]{16,}")),
    ("aws access key id", re.compile(r"AKIA[0-9A-Z]{16}")),
)
#: Archive member suffixes scanned as text; anything else is skipped rather
#: than decoded, so a stray binary cannot fail the run or slow it down.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {".py", ".md", ".toml", ".cfg", ".ini", ".txt", ".json", ".jsonl", ".csv", ".yml", ".yaml", ".sh", ".example", ""}
)
SCAN_LIMIT_BYTES = 4 * 1024 * 1024


class MissingFiles(RuntimeError):
    """A required file is absent; the package would be broken."""


class SecretFound(RuntimeError):
    """Something in the tree must not be packaged."""


@dataclass
class PackageResult:
    path: Path
    entries: list[str] = field(default_factory=list)
    uncompressed_bytes: int = 0
    sha256: str = ""
    skipped: list[str] = field(default_factory=list)

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size


# --------------------------------------------------------------------------
# Required files
# --------------------------------------------------------------------------


def check_tree(repo_root: Path) -> list[str]:
    """What is missing from the tree, as a sorted list of path strings."""
    missing = [rel for rel in REQUIRED_FILES if not (repo_root / rel).is_file()]
    tests_dir = repo_root / "tests"
    if not tests_dir.is_dir() or not any(tests_dir.glob("test_*.py")):
        missing.append("tests/ (no test_*.py files)")
    return sorted(missing)


# --------------------------------------------------------------------------
# File selection
# --------------------------------------------------------------------------


def _is_excluded(rel: Path) -> str:
    parts = rel.parts
    name = rel.name
    if any(part in EXCLUDED_DIRS for part in parts[:-1]):
        return "excluded directory"
    if any(part.endswith(".egg-info") for part in parts[:-1]):
        return "build metadata (*.egg-info)"
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment file (never packaged)"
    if name in EXCLUDED_FILES:
        return "excluded file"
    if parts[0] == "reports" and fnmatch.fnmatch(name, "partial-*.json"):
        return "partial eval output"
    if name == DEFAULT_NAME:
        return "the package itself"
    if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_GLOBS):
        return "excluded pattern"
    return ""


def walk_tree(repo_root: Path) -> tuple[list[Path], list[str]]:
    """(files to package, notes about what was left out)."""
    files: list[Path] = []
    notes: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)
        reason = _is_excluded(rel)
        if reason:
            notes.append(f"skipped {rel.as_posix()} ({reason})")
            continue
        files.append(path)
    return files, notes


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------


def _is_texty(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name.endswith(".example")


def _scan_text(text: str, where: str) -> list[str]:
    hits: list[str] = []
    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{label} at {where}:{line}")
    return hits


def scan_for_secrets(path: Path) -> list[str]:
    """Every secret-looking match in one file, as short descriptions."""
    try:
        if path.stat().st_size > SCAN_LIMIT_BYTES or not _is_texty(path):
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return _scan_text(text, path.as_posix())


def scan_tree(files: Sequence[Path]) -> list[str]:
    findings: list[str] = []
    for path in files:
        findings.extend(scan_for_secrets(path))
    return findings


# --------------------------------------------------------------------------
# Deterministic zip
# --------------------------------------------------------------------------


def _fixed_info(arcname: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(arcname, date_time=FIXED_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.external_attr = FIXED_EXTERNAL_ATTR
    return info


def write_zip(repo_root: Path, files: Sequence[Path], out_path: Path) -> tuple[list[str], int]:
    """Write the zip with fixed metadata; returns (sorted entries, uncompressed bytes)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    uncompressed = 0
    with zipfile.ZipFile(
        out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=FIXED_COMPRESS_LEVEL
    ) as archive:
        for path in files:
            arcname = path.relative_to(repo_root).as_posix()
            data = path.read_bytes()
            archive.writestr(_fixed_info(arcname), data)
            entries.append(arcname)
            uncompressed += len(data)
    return sorted(entries), uncompressed


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_package(repo_root: Path, out_path: Path) -> PackageResult:
    missing = check_tree(repo_root)
    if missing:
        raise MissingFiles(
            "refusing to package: required files are missing:\n  - " + "\n  - ".join(missing)
        )

    files, notes = walk_tree(repo_root)
    findings = scan_tree(files)
    if findings:
        raise SecretFound(
            "refusing to package: credential-shaped content found\n  - "
            + "\n  - ".join(findings)
        )

    entries, uncompressed = write_zip(repo_root, files, out_path)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return PackageResult(
        path=out_path,
        entries=entries,
        uncompressed_bytes=uncompressed,
        sha256=digest,
        skipped=notes,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="package_submission.py",
        description=f"Build {DEFAULT_NAME} (deterministic; refuses secrets and broken trees).",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="repository root")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"output zip path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build nothing; only report which required files are missing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        print(f"error: repository root not found: {repo_root}", file=sys.stderr)
        return 2

    if args.check:
        missing = check_tree(repo_root)
        if missing:
            print(
                f"error: {len(missing)} required file(s) missing from {repo_root}:",
                file=sys.stderr,
            )
            for rel in missing:
                print(f"  - {rel}", file=sys.stderr)
            return 1
        print(f"tree complete: all {len(REQUIRED_FILES)} required files present in {repo_root}")
        return 0

    try:
        result = build_package(repo_root, Path(args.out).resolve())
    except MissingFiles as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SecretFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: packaging failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for note in result.skipped:
        print(f"  {note}")
    print(f"package:      {result.path}")
    print(f"entries:      {len(result.entries)}")
    print(f"uncompressed: {result.uncompressed_bytes:,} bytes")
    print(f"zip size:     {result.size_bytes:,} bytes")
    print(f"sha256:       {result.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
