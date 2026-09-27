"""Replays a case file against a live ``POST /chat`` service and scores it.

Runnable both ways, from the repo root:

    .\\.venv\\Scripts\\python.exe -m evals.runner --cases evals/cases.jsonl --repeats 3 \\
        --base-url http://127.0.0.1:8000
    .\\.venv\\Scripts\\python.exe -m evals.runner evals/cases.jsonl

Design notes the reviewer needs to know:

* **Sequential by default.** ``--concurrency`` exists but defaults to 1: a local
  7B model degrades badly under concurrency, and a 3x repeat comparison is only
  meaningful if the load on the box is identical every run.
* **Fresh conversation per case.** The id is ``eval-<repeat>-<case_id>``, so no
  turn ever sees a previous case's history and G1 genuinely measures a first
  reply rather than the fourth one.
* **Tokens are never fabricated.** They are read from response headers (or a
  ``usage`` object in the body) when the service exposes them, and otherwise
  ``usage_source`` stays ``"unknown"`` with zero counts, which the report states
  explicitly rather than quietly dividing by nothing.
* **The ``actions`` array is read as "what the model requested".** The public
  action shape carries no origin flag, so an escalate the loop generated on step
  exhaustion is only excluded from that set when the service marks it
  auto-generated; otherwise it is treated as a real call. See
  ``checks._is_implicit_escalation``.
* **A partial JSON is rewritten after every case** so an interrupted 40-minute
  run is not lost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx

from evals import DEFAULT_BASE_URL, DEFAULT_CASES, DEFAULT_OUT_DIR
from evals.checks import CaseResult, RunSummary, check_case
from evals.report import write_reports

__all__ = [
    "run_cases",
    "run_once",
    "load_cases",
    "wait_for_ready",
    "replay_case",
    "main",
]

#: Response headers a service might use to report token usage. Any OpenAI-shaped
#: convention is accepted so the harness works against an unknown endpoint.
_PROMPT_TOKEN_HEADERS = (
    "x-prompt-tokens",
    "x-input-tokens",
    "x-eval-prompt-tokens",
    "x-usage-prompt-tokens",
    "openai-prompt-tokens",
    "x-total-prompt-tokens",
    "prompt-tokens",
)
_COMPLETION_TOKEN_HEADERS = (
    "x-completion-tokens",
    "x-output-tokens",
    "x-eval-completion-tokens",
    "x-usage-completion-tokens",
    "openai-completion-tokens",
    "x-total-completion-tokens",
    "completion-tokens",
)
_READY_TIMEOUT_S = 60.0
_READY_POLL_S = 1.0


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------
# Case loading
# --------------------------------------------------------------------------


def load_cases(path: Path) -> tuple[list[dict], list[str]]:
    """Read a JSONL case file. Returns (cases, warnings); never raises."""
    cases: list[dict] = []
    warnings: list[str] = []
    try:
        raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read the case file {path}: {exc}") from exc
    for lineno, line in enumerate(raw_lines, start=1):
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            warnings.append(f"{path}:{lineno}: not valid JSON ({exc.msg}); line skipped")
            continue
        if not isinstance(parsed, dict):
            warnings.append(
                f"{path}:{lineno}: expected a JSON object, got {type(parsed).__name__}; line skipped"
            )
            continue
        if "id" in parsed:
            parsed["id"] = str(parsed["id"])
        else:
            parsed["id"] = f"line-{lineno}"
            warnings.append(f"{path}:{lineno}: case has no id; using {parsed['id']!r}")
        cases.append(parsed)
    return cases, warnings


def filter_cases(
    cases: Sequence[dict], *, only: str | None = None, category: str | None = None
) -> list[dict]:
    """``--only`` keeps ids starting with the given prefix, ``--category`` is exact."""
    out = list(cases)
    if only:
        out = [c for c in out if str(c.get("id", "")).startswith(only)]
    if category:
        wanted = category.strip().casefold()
        out = [c for c in out if str(c.get("category", "")).strip().casefold() == wanted]
    return out


# --------------------------------------------------------------------------
# Service interaction
# --------------------------------------------------------------------------


def wait_for_ready(
    client: httpx.Client, base_url: str, *, timeout_s: float = _READY_TIMEOUT_S
) -> bool:
    """Poll ``GET /health`` until the service answers or ``timeout_s`` elapses."""
    url = f"{base_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_s
    last = "no attempt made"
    while True:
        try:
            response = client.get(url, timeout=10.0)
            if response.status_code < 500:
                return True
            last = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            print(
                f"error: {url} did not become ready within {timeout_s:.0f}s (last: {last}). "
                f"Start the service first, e.g. "
                f".\\.venv\\Scripts\\python.exe -m uvicorn meher_agent.api.app:app --port 8000",
                file=sys.stderr,
                flush=True,
            )
            return False
        time.sleep(_READY_POLL_S)


def _header_token(headers: Any, names: Sequence[str]) -> int | None:
    lowered = {str(k).lower(): v for k, v in dict(headers).items()}
    for name in names:
        if name in lowered:
            try:
                return int(float(str(lowered[name]).strip()))
            except ValueError:
                continue
    return None


def _body_usage(body: Any) -> tuple[int | None, int | None]:
    if not isinstance(body, dict):
        return None, None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        meta = body.get("meta")
        usage = meta.get("usage") if isinstance(meta, dict) else None
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    try:
        return (int(prompt) if prompt is not None else None, int(completion) if completion is not None else None)
    except (TypeError, ValueError):
        return None, None


def _case_turns(case: dict) -> list[str]:
    turns = case.get("turns")
    if isinstance(turns, str):
        return [turns]
    if not isinstance(turns, (list, tuple)):
        return []
    return [str(t) for t in turns if t is not None and not isinstance(t, (dict, list))]


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _excerpt(text: str, limit: int = 200) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"


def replay_case(
    client: httpx.Client,
    case: dict,
    *,
    base_url: str,
    repeat: int,
    timeout_s: float,
    run_id: str,
) -> CaseResult:
    """Send every turn of one case, in order, on a fresh conversation id."""
    url = f"{base_url.rstrip('/')}/chat"
    turns = _case_turns(case)
    # The run id keeps repeat runs of the same case isolated. Without it the
    # store remembers the earlier run, every case replays as a *second* turn,
    # and the first-turn "AI" disclosure check fails for reasons that have
    # nothing to do with the agent.
    conversation_id = f"eval-{run_id}-{repeat}-{case.get('id', 'case')}"
    result = CaseResult(
        case_id=str(case.get("id", "")),
        category=str(case.get("category", "") or ""),
        turns=list(turns),
        replies=[],
        sources=[],
        actions=[],
        handoff=[],
        latencies_s=[],
        prompt_tokens=[],
        completion_tokens=[],
        usage_source="unknown",
        case=case,
    )
    if not turns:
        result.error = "case has no usable 'turns' to replay"
        result.checks = check_case(case, result)
        return result

    for index, message in enumerate(turns):
        payload = {"conversation_id": conversation_id, "message": message}
        started = time.perf_counter()
        try:
            response = client.post(url, json=payload, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            result.error = f"turn {index + 1} timed out after {timeout_s:.0f}s: {exc}"
            break
        except httpx.HTTPError as exc:
            result.error = f"turn {index + 1} failed: {type(exc).__name__}: {exc}"
            break
        elapsed = time.perf_counter() - started
        if response.status_code >= 400:
            body = _excerpt(response.text)
            result.error = f"turn {index + 1} returned HTTP {response.status_code}: {body}"
            break
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            result.error = f"turn {index + 1} did not return JSON: {_excerpt(response.text)}"
            break
        if not isinstance(body, dict):
            result.error = f"turn {index + 1} returned {type(body).__name__}, expected a JSON object"
            break

        result.latencies_s.append(elapsed)
        result.replies.append(str(body.get("reply") or ""))
        result.sources.append([str(s) for s in _as_sequence(body.get("sources"))])
        result.actions.append([a for a in _as_sequence(body.get("actions")) if isinstance(a, dict)])
        result.handoff.append(bool(body.get("handoff")))

        prompt, completion = _header_token(response.headers, _PROMPT_TOKEN_HEADERS), _header_token(
            response.headers, _COMPLETION_TOKEN_HEADERS
        )
        if prompt is None or completion is None:
            body_prompt, body_completion = _body_usage(body)
            prompt = prompt if prompt is not None else body_prompt
            completion = completion if completion is not None else body_completion
        if prompt is not None:
            result.prompt_tokens.append(prompt)
            result.completion_tokens.append(completion or 0)
            result.usage_source = "endpoint"
        else:
            result.prompt_tokens.append(0)
            result.completion_tokens.append(0)

    result.checks = check_case(case, result)
    return result


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _progress_line(repeat: int, repeats: int, done: int, total: int, result: CaseResult) -> str:
    graded = [c for c in result.checks if not c.skipped]
    passed = sum(1 for c in graded if c.passed)
    failed = [c.name for c in graded if not c.passed]
    seconds = sum(result.latencies_s)
    tokens = f"tok={sum(result.prompt_tokens)}/{sum(result.completion_tokens)}" if result.usage_source == "endpoint" else "tok=n/a"
    verdict = f"{passed}/{len(graded)}" if graded else "no-checks"
    suffix = ""
    if failed:
        suffix = f" FAIL[{','.join(failed)}]"
    if result.error:
        suffix += f" ERROR[{_excerpt(result.error, 120)}]"
    return (
        f"[run {repeat}/{repeats} {done:>3}/{total}] {result.case_id:<16} {verdict:<9} "
        f"{seconds:6.1f}s  {tokens}{suffix}"
    )


def run_once(
    cases: Sequence[dict],
    client: httpx.Client,
    *,
    base_url: str,
    repeat: int,
    repeats: int,
    timeout_s: float,
    partial_path: Path,
    cases_path: Path,
    concurrency: int = 1,
    run_id: str = "0",
) -> RunSummary:
    started_at = _now()
    run = RunSummary(
        index=repeat,
        base_url=base_url,
        cases_path=str(cases_path),
        started_at=started_at,
        finished_at=started_at,
    )
    total = len(cases)

    def record(result: CaseResult) -> None:
        run.results.append(result)
        print(_progress_line(repeat, repeats, len(run.results), total, result), flush=True)
        _write_json(
            partial_path,
            {
                "generated_at": _now(),
                "partial": True,
                "run": run.index,
                "repeats": repeats,
                "completed": len(run.results),
                "of": total,
                "results": [r.to_dict() for r in run.results],
            },
        )

    if concurrency <= 1:
        for case in cases:
            record(
                replay_case(
                    client, case, base_url=base_url, repeat=repeat, timeout_s=timeout_s, run_id=run_id
                )
            )
        return _finish(run)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                replay_case,
                client,
                case,
                base_url=base_url,
                repeat=repeat,
                timeout_s=timeout_s,
                run_id=run_id,
            )
            for case in cases
        ]
        for future in futures:
            record(future.result())
    return _finish(run)


def _finish(run: RunSummary) -> RunSummary:
    run.finished_at = _now()
    return run


def run_cases(
    cases_path: Path,
    *,
    base_url: str,
    repeats: int = 3,
    out_dir: Path = DEFAULT_OUT_DIR,
    timeout_s: float = 180.0,
    concurrency: int = 1,
    only: str | None = None,
    category: str | None = None,
    wait_s: float = _READY_TIMEOUT_S,
) -> list[RunSummary]:
    """Replay ``cases_path`` ``repeats`` times and write the reports.

    Returns one :class:`RunSummary` per repeat, in repeat order.
    """
    cases_path = Path(cases_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_cases, warnings = load_cases(cases_path)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr, flush=True)
    cases = filter_cases(all_cases, only=only, category=category)
    if not cases:
        print(
            f"error: no cases selected from {cases_path}"
            + (f" with --only {only!r}" if only else "")
            + (f" --category {category!r}" if category else ""),
            file=sys.stderr,
            flush=True,
        )
        return []
    print(
        f"cases: {len(cases)} of {len(all_cases)} from {cases_path}; "
        f"repeats={repeats}; base_url={base_url}; concurrency={concurrency}",
        flush=True,
    )

    partial_path = out_dir / f"partial-run-{_stamp()}.json"
    print(f"partial results: {partial_path}", flush=True)

    # One id for the whole invocation, so the three repeats stay independent of
    # each other *and* of every earlier invocation against the same service.
    run_id = _stamp()

    runs: list[RunSummary] = []
    interrupted = False
    with httpx.Client(headers={"accept": "application/json"}) as client:
        if not wait_for_ready(client, base_url, timeout_s=wait_s):
            raise RuntimeError(f"service at {base_url} never became ready")
        for repeat in range(1, max(1, repeats) + 1):
            print(f"--- run {repeat}/{repeats} ---", flush=True)
            try:
                runs.append(
                    run_once(
                        cases,
                        client,
                        base_url=base_url,
                        repeat=repeat,
                        repeats=repeats,
                        timeout_s=timeout_s,
                        partial_path=partial_path,
                        cases_path=cases_path,
                        concurrency=concurrency,
                        run_id=run_id,
                    )
                )
            except KeyboardInterrupt:
                interrupted = True
                print(
                    "\ninterrupted: writing reports for the runs completed so far", flush=True
                )
                break
    json_path, md_path = write_reports(runs, out_dir)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return runs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.runner",
        description="Replay an evaluation case file against a running Meher agent service.",
    )
    parser.add_argument("cases_positional", nargs="?", default=None, metavar="CASES",
                        help="path to the JSONL case file (same as --cases)")
    parser.add_argument("--cases", default=None, help="path to the JSONL case file")
    parser.add_argument("--base-url", default=None,
                        help=f"service base URL (default: $EVAL_BASE_URL or {DEFAULT_BASE_URL})")
    parser.add_argument("--repeats", type=int, default=3, help="full passes over the case file")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where reports are written")
    parser.add_argument("--only", default=None, help="debug: keep only case ids with this prefix")
    parser.add_argument("--category", default=None, help="debug: keep only this category")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="in-flight cases; 1 (sequential) is the reproducible default")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request timeout in seconds")
    parser.add_argument("--wait-s", type=float, default=_READY_TIMEOUT_S,
                        help="how long to wait for GET /health to answer")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cases_path = Path(args.cases or args.cases_positional or DEFAULT_CASES)
    base_url = (
        args.base_url
        or os.getenv("EVAL_BASE_URL", "").strip()
        or DEFAULT_BASE_URL
    )
    try:
        run_cases(
            cases_path,
            base_url=base_url,
            repeats=args.repeats,
            out_dir=Path(args.out_dir),
            timeout_s=args.timeout,
            concurrency=max(1, args.concurrency),
            only=args.only,
            category=args.category,
            wait_s=args.wait_s,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
