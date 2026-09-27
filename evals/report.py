"""Turns ``list[RunSummary]`` into the two artefacts the report is graded on.

* ``<out_dir>/run-<timestamp>.json`` - every case, every run, every check, with
  pass/fail and detail, plus the per-run numerators and denominators behind each
  rate. Any number quoted in the technical report has to be traceable here.
* ``<out_dir>/summary.md`` - the same numbers in prose and tables.

Both are additive: a previous ``summary.md`` is snapshotted to
``summary-<timestamp>.md`` before being replaced, so nothing is ever lost.

Every rate is reported as the **mean of the runs** and the **worst run**, because
sampling at temperature 0 on a local 7B is not bit-reproducible. Latency and cost
are worst = max; pass rates and accuracy are worst = min.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from evals.checks import (
    CaseResult,
    RunSummary,
    allowed_rupee_amounts,
    valid_source_ids,
)
from meher_agent.config import load_config

__all__ = [
    "CostSettings",
    "RateMetric",
    "RunMetrics",
    "Summary",
    "cost_settings",
    "cost_inr_per_100_conversations",
    "percentile",
    "aggregate",
    "summarise_run",
    "summarise",
    "render_summary_md",
    "write_reports",
]

#: Check name -> whether the graded report denominator is the set of cases that
#: set that field. Kept explicit so a change in checks.py cannot silently move a
#: denominator.
_CHECKS = ("G1", "G2", "G3", "G4", "must_include", "must_include_any", "must_not_include",
            "expect_action", "expect_lead")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CostSettings:
    inr_per_usd: float
    input_cost_per_mtok_usd: float
    output_cost_per_mtok_usd: float
    source: str = "config.toml"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cost_settings(
    config_path: Path | None = None,
    *,
    inr_per_usd: float | None = None,
    input_cost_per_mtok_usd: float | None = None,
    output_cost_per_mtok_usd: float | None = None,
) -> CostSettings:
    """Read the cost block from config.toml; the keyword overrides exist so a
    reviewer (or a test) can price a run without editing the config."""
    cost = load_config(config_path).cost
    return CostSettings(
        inr_per_usd=cost.inr_per_usd if inr_per_usd is None else float(inr_per_usd),
        input_cost_per_mtok_usd=(
            cost.input_cost_per_mtok_usd if input_cost_per_mtok_usd is None else float(input_cost_per_mtok_usd)
        ),
        output_cost_per_mtok_usd=(
            cost.output_cost_per_mtok_usd if output_cost_per_mtok_usd is None else float(output_cost_per_mtok_usd)
        ),
        source=str(config_path) if config_path else "config.toml",
    )


def cost_inr_per_100_conversations(
    *,
    prompt_tokens_per_message: float,
    completion_tokens_per_message: float,
    messages_per_conversation: float,
    inr_per_usd: float,
    input_cost_per_mtok_usd: float,
    output_cost_per_mtok_usd: float,
) -> float:
    """Rupees per 100 conversations.

    usd_per_message = (in_tok * in_usd_per_mtok + out_tok * out_usd_per_mtok) / 1e6
    usd_per_conversation = usd_per_message * messages_per_conversation
    inr_per_100_conversations = usd_per_conversation * inr_per_usd * 100
    """
    usd_per_message = (
        prompt_tokens_per_message * input_cost_per_mtok_usd
        + completion_tokens_per_message * output_cost_per_mtok_usd
    ) / 1_000_000.0
    usd_per_conversation = usd_per_message * messages_per_conversation
    return usd_per_conversation * inr_per_usd * 100.0


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, no interpolation.

    The sample is sorted ascending and the result is the element at index
    ``ceil(q * n) - 1`` (clamped to the sample). With n messages that is the
    smallest value at or above the q-th fraction of observations, so p95 of ten
    messages is the slowest message, and the result is always an observed value
    that appears in the JSON - which is what makes the number checkable.
    """
    sample = sorted(float(v) for v in values)
    if not sample:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    index = max(0, min(len(sample) - 1, math.ceil(q * len(sample)) - 1))
    return sample[index]


@dataclass(frozen=True)
class RateMetric:
    """One rate, across every run."""

    label: str
    numerator: int
    denominator: int
    per_run: tuple[float, ...]
    mean: float
    worst: float
    best: float
    higher_is_better: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "per_run": list(self.per_run),
            "mean": self.mean,
            "worst": self.worst,
            "best": self.best,
            "higher_is_better": self.higher_is_better,
        }


def aggregate(
    label: str,
    per_run: Sequence[float],
    *,
    higher_is_better: bool = True,
    numerator: int = 0,
    denominator: int = 0,
) -> RateMetric:
    values = [float(v) for v in per_run]
    if not values:
        return RateMetric(label, 0, 0, (), 0.0, 0.0, 0.0, higher_is_better)
    mean = sum(values) / len(values)
    worst = min(values) if higher_is_better else max(values)
    best = max(values) if higher_is_better else min(values)
    return RateMetric(
        label=label,
        numerator=numerator,
        denominator=denominator,
        per_run=tuple(values),
        mean=mean,
        worst=worst,
        best=best,
        higher_is_better=higher_is_better,
    )


# --------------------------------------------------------------------------
# Per-run metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunMetrics:
    index: int
    started_at: str
    finished_at: str
    cases: int
    errors: int
    messages: int
    messages_with_usage: int
    case_passed: int
    per_category: dict[str, tuple[int, int]]
    per_check: dict[str, tuple[int, int]]
    latencies_s: list[float]
    p50_s: float
    p95_s: float
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_per_message: float
    completion_tokens_per_message: float
    cost_inr_per_100_conversations: float
    usage_sources: dict[str, int]

    def rate(self, numerator: int, denominator: int) -> float:
        return (numerator / denominator) if denominator else 0.0

    def check_rate(self, name: str) -> float:
        return self.rate(*self.per_check.get(name, (0, 0)))

    def category_rate(self, name: str) -> float:
        return self.rate(*self.per_category.get(name, (0, 0)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["latencies_s"] = list(self.latencies_s)
        return payload


def _count_check(result: CaseResult, name: str) -> tuple[int, int, int]:
    """(passed, failed, graded) for one check name on one case."""
    passed = failed = graded = 0
    for outcome in result.checks:
        if outcome.name != name or outcome.skipped:
            continue
        graded += 1
        if outcome.passed:
            passed += 1
        else:
            failed += 1
    return passed, failed, graded


def summarise_run(run: RunSummary, cost: CostSettings) -> RunMetrics:
    """Reduce one run to the numbers the report quotes.

    Denominators come from the check outcomes themselves, which is why a case
    that never set ``expect_action`` contributes no ``expect_action`` outcome and
    therefore cannot dilute the action-accuracy figure.
    """
    per_category: dict[str, tuple[int, int]] = {}
    per_check: dict[str, tuple[int, int]] = {}
    case_passed = 0
    latencies: list[float] = []
    prompt_total = 0
    completion_total = 0
    messages = 0
    messages_with_usage = 0
    usage_sources: dict[str, int] = {}
    errors = 0

    for result in run.results:
        if result.error:
            errors += 1
        graded = [c for c in result.checks if not c.skipped]
        passed = all(c.passed for c in graded)
        category = (result.category or "").strip() or "<uncategorised>"
        prev_p, prev_t = per_category.get(category, (0, 0))
        per_category[category] = (prev_p + int(passed), prev_t + 1)
        if passed:
            case_passed += 1
        for name in _CHECKS:
            p, _f, g = _count_check(result, name)
            if g:
                prev_p, prev_t = per_check.get(name, (0, 0))
                per_check[name] = (prev_p + p, prev_t + g)
        latencies.extend(result.latencies_s)
        messages += len(result.latencies_s)
        prompt_total += sum(result.prompt_tokens)
        completion_total += sum(result.completion_tokens)
        if result.usage_source == "endpoint":
            messages_with_usage += len(result.latencies_s)
        usage_sources[result.usage_source] = usage_sources.get(result.usage_source, 0) + 1

    cases = len(run.results)
    messages_per_conversation = (messages / cases) if cases else 0.0
    prompt_per_message = (prompt_total / messages) if messages else 0.0
    completion_per_message = (completion_total / messages) if messages else 0.0
    return RunMetrics(
        index=run.index,
        started_at=run.started_at,
        finished_at=run.finished_at,
        cases=cases,
        errors=errors,
        messages=messages,
        messages_with_usage=messages_with_usage,
        case_passed=case_passed,
        per_category=dict(per_category),
        per_check=dict(per_check),
        latencies_s=list(latencies),
        p50_s=percentile(latencies, 0.50),
        p95_s=percentile(latencies, 0.95),
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        prompt_tokens_per_message=prompt_per_message,
        completion_tokens_per_message=completion_per_message,
        cost_inr_per_100_conversations=cost_inr_per_100_conversations(
            prompt_tokens_per_message=prompt_per_message,
            completion_tokens_per_message=completion_per_message,
            messages_per_conversation=messages_per_conversation,
            inr_per_usd=cost.inr_per_usd,
            input_cost_per_mtok_usd=cost.input_cost_per_mtok_usd,
            output_cost_per_mtok_usd=cost.output_cost_per_mtok_usd,
        ),
        usage_sources=usage_sources,
    )


@dataclass
class Summary:
    generated_at: str
    base_url: str
    cases_path: str
    runs: int
    repeats: int
    cost: CostSettings
    per_run: list[RunMetrics]
    rates: dict[str, RateMetric] = field(default_factory=dict)

    def rate(self, key: str) -> RateMetric:
        return self.rates.get(key) or RateMetric(key, 0, 0, (), 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "base_url": self.base_url,
            "cases_path": self.cases_path,
            "runs": self.runs,
            "cost": self.cost.to_dict(),
            "per_run": [m.to_dict() for m in self.per_run],
            "rates": {k: v.to_dict() for k, v in self.rates.items()},
        }


# --------------------------------------------------------------------------
# Aggregation across runs
# --------------------------------------------------------------------------


def _case_rate(m: RunMetrics) -> float:
    return m.rate(m.case_passed, m.cases)


def _check_rate(m: RunMetrics, name: str) -> float:
    return m.check_rate(name)


def summarise(runs: Sequence[RunSummary], cost: CostSettings) -> Summary:
    per_run = [summarise_run(run, cost) for run in runs]
    summary = Summary(
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        base_url=runs[0].base_url if runs else "",
        cases_path=runs[0].cases_path if runs else "",
        runs=len(runs),
        repeats=len(runs),
        cost=cost,
        per_run=per_run,
    )
    rates: dict[str, RateMetric] = {}

    def add(
        key: str,
        label: str,
        values: Sequence[float],
        *,
        higher_is_better: bool = True,
        counts: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        metric = aggregate(label, values, higher_is_better=higher_is_better)
        if counts:
            metric = replace(
                metric,
                numerator=sum(p for p, _ in counts),
                denominator=sum(t for _, t in counts),
            )
        rates[key] = metric

    def check_counts(name: str) -> list[tuple[int, int]]:
        return [m.per_check.get(name, (0, 0)) for m in per_run]

    add("overall_pass_rate", "Overall pass rate (all checks of a case)", [_case_rate(m) for m in per_run],
        counts=[(m.case_passed, m.cases) for m in per_run])
    add("check_pass_rate", "All graded check outcomes", [
        m.rate(sum(p for p, _ in m.per_check.values()), sum(t for _, t in m.per_check.values()))
        for m in per_run
    ])
    add("invented_amount_rate", "Invented-amount rate (G2 failures / replies checked)",
        [1.0 - _check_rate(m, "G2") for m in per_run], higher_is_better=False,
        counts=[(t - p, t) for p, t in check_counts("G2")])
    add("ai_disclosure_rate", "AI-disclosure rate (G1 pass / first replies)",
        [_check_rate(m, "G1") for m in per_run], counts=check_counts("G1"))
    add("action_accuracy", "Action accuracy (cases that set expect_action)",
        [_check_rate(m, "expect_action") for m in per_run], counts=check_counts("expect_action"))
    add("g3_pass_rate", "Reply-length pass rate (G3)", [_check_rate(m, "G3") for m in per_run],
        counts=check_counts("G3"))
    add("g4_pass_rate", "Source-citation pass rate (G4, six categories only)",
        [_check_rate(m, "G4") for m in per_run], counts=check_counts("G4"))
    add("must_include_rate", "must_include pass rate", [_check_rate(m, "must_include") for m in per_run],
        counts=check_counts("must_include"))
    add("must_include_any_rate", "must_include_any pass rate",
        [_check_rate(m, "must_include_any") for m in per_run], counts=check_counts("must_include_any"))
    add("must_not_include_rate", "must_not_include pass rate",
        [_check_rate(m, "must_not_include") for m in per_run], counts=check_counts("must_not_include"))
    add("expect_lead_rate", "Lead-field match rate (expect_lead)",
        [_check_rate(m, "expect_lead") for m in per_run], counts=check_counts("expect_lead"))

    categories = sorted({c for m in per_run for c in m.per_category})
    for category in categories:
        add(f"category:{category}", f"Pass rate: {category}",
            [m.category_rate(category) for m in per_run],
            counts=[m.per_category.get(category, (0, 0)) for m in per_run])

    add("latency_p50_s", "Latency p50 per message (s)", [m.p50_s for m in per_run], higher_is_better=False)
    add("latency_p95_s", "Latency p95 per message (s)", [m.p95_s for m in per_run], higher_is_better=False)
    add("prompt_tokens_per_message", "Prompt tokens per message",
        [m.prompt_tokens_per_message for m in per_run], higher_is_better=False)
    add("completion_tokens_per_message", "Completion tokens per message",
        [m.completion_tokens_per_message for m in per_run], higher_is_better=False)
    add("cost_inr_per_100_conversations", "Cost in rupees per 100 conversations",
        [m.cost_inr_per_100_conversations for m in per_run], higher_is_better=False)
    add("messages_per_conversation", "Messages per conversation",
        [(m.messages / m.cases) if m.cases else 0.0 for m in per_run], higher_is_better=False)
    add("error_rate", "Case error rate (request failed)", [m.rate(m.errors, m.cases) for m in per_run],
        higher_is_better=False, counts=[(m.errors, m.cases) for m in per_run])

    summary.rates = rates
    return summary


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    return f"{value:,.2f}"


def _table_header(metrics: list[RunMetrics]) -> tuple[str, str]:
    columns = " | ".join(f"run {m.index}" for m in metrics) if metrics else "(no runs)"
    header = f"| Metric | {columns} | mean | worst | best |"
    return header, "|" + "---|" * (len(metrics) + 4)


def _row(label: str, metric: RateMetric, metrics: list[RunMetrics], fmt: Callable[[float], str]) -> str:
    cells = [fmt(v) for v in metric.per_run]
    while len(cells) < len(metrics):
        cells.append("n/a")
    return "| " + label + " | " + " | ".join(cells) + f" | {fmt(metric.mean)} | {fmt(metric.worst)} | {fmt(metric.best)} |"


def render_summary_md(summary: Summary, *, partial_note: str = "", json_name: str = "") -> str:
    metrics = summary.per_run
    lines: list[str] = []
    add = lines.append

    add("# Evaluation summary")
    add("")
    add(f"- generated: {summary.generated_at}")
    add(f"- endpoint: `{summary.base_url}`")
    add(f"- cases: `{summary.cases_path}`")
    add(f"- evidence: `{json_name or 'run-<timestamp>.json'}` (every case, run and check behind these numbers)")
    add(f"- runs: {summary.runs} (the full set is replayed {summary.runs}x; models are not deterministic, "
        f"so every rate below is the mean of the {summary.runs} runs and the worst single run)")
    total_cases = sum(m.cases for m in metrics)
    total_messages = sum(m.messages for m in metrics)
    total_errors = sum(m.errors for m in metrics)
    add(f"- graded: {total_cases} case runs, {total_messages} messages, {total_errors} failed requests")
    if partial_note:
        add(f"- note: {partial_note}")
    add("")

    add("## Overall and per-category pass rate")
    add("")
    add("A case passes when every applicable check on it passes. Checks that do not apply "
        "to a case (for example G4 on a `complaint` case) are excluded from the denominator.")
    add("")
    header, separator = _table_header(metrics)
    add(header)
    add(separator)
    add(_row("Overall pass rate (all checks of a case)", summary.rate("overall_pass_rate"), metrics, _pct))
    add(_row("All graded check outcomes", summary.rate("check_pass_rate"), metrics, _pct))
    for key in sorted(k for k in summary.rates if k.startswith("category:")):
        add(_row(summary.rate(key).label, summary.rate(key), metrics, _pct))
    add("")
    if metrics:
        counts = "; ".join(f"run {m.index}: {m.case_passed}/{m.cases}" for m in metrics)
        add(f"Passing cases per run - {counts}.")
    add("")

    add("## Graded guardrails")
    add("")
    add(header)
    add(separator)
    for key in (
        "invented_amount_rate",
        "ai_disclosure_rate",
        "action_accuracy",
        "g3_pass_rate",
        "g4_pass_rate",
        "must_include_rate",
        "must_include_any_rate",
        "must_not_include_rate",
        "expect_lead_rate",
        "error_rate",
    ):
        metric = summary.rate(key)
        add(_row(metric.label, metric, metrics, _pct))
    add("")
    if metrics:
        g2_total = sum(m.per_check.get("G2", (0, 0))[1] for m in metrics)
        g2_bad = sum(m.per_check.get("G2", (0, 0))[1] - m.per_check.get("G2", (0, 0))[0] for m in metrics)
        g1_total = sum(m.per_check.get("G1", (0, 0))[1] for m in metrics)
        g1_ok = sum(m.per_check.get("G1", (0, 0))[0] for m in metrics)
        act_total = sum(m.per_check.get("expect_action", (0, 0))[1] for m in metrics)
        act_ok = sum(m.per_check.get("expect_action", (0, 0))[0] for m in metrics)
        add(f"- invented amounts: {g2_bad} G2 failures over {g2_total} replies checked")
        add(f"- AI disclosure: {g1_ok} first replies disclosed AI out of {g1_total} first replies")
        add(f"- action accuracy: {act_ok} correct out of {act_total} cases that set `expect_action`")
    add("")

    add("## Latency, tokens and cost")
    add("")
    add(header)
    add(separator)
    for key in (
        "latency_p50_s",
        "latency_p95_s",
        "prompt_tokens_per_message",
        "completion_tokens_per_message",
        "messages_per_conversation",
        "cost_inr_per_100_conversations",
    ):
        metric = summary.rate(key)
        add(_row(metric.label, metric, metrics, _num))
    add("")
    if metrics:
        add("Per run:")
        add("")
        for m in metrics:
            add(f"- run {m.index}: {m.messages} messages, p50 {m.p50_s:.2f}s, p95 {m.p95_s:.2f}s, "
                f"prompt {m.prompt_tokens} tok, completion {m.completion_tokens} tok, "
                f"{m.messages_with_usage}/{m.messages} messages with reported usage, "
                f"cost Rs {m.cost_inr_per_100_conversations:.2f} per 100 conversations")
        add("")
    usage = sorted({s for m in metrics for s in m.usage_sources})
    if usage == ["unknown"]:
        add("**Token counts are 0 because the service reported no token usage.** The harness never "
            "estimates tokens, so the cost line is derived from measured tokens only and must be read "
            "as \"not measured\" rather than \"free\". Re-run against an endpoint that exposes usage "
            "headers (or a `usage` object in the response body) to populate it.")
    elif "unknown" in usage:
        add("Some requests did not report token usage; those messages contribute 0 tokens.")
    add("")

    add("## How the numbers are computed")
    add("")
    add(f"- Cost config from `{summary.cost.source}`: inr_per_usd={summary.cost.inr_per_usd}, "
        f"input=${summary.cost.input_cost_per_mtok_usd}/Mtok, output=${summary.cost.output_cost_per_mtok_usd}/Mtok. "
        "A local model has 0.0 prices, so the cost line reads Rs 0.00 while the token counts stay real.")
    add("- Cost per 100 conversations = "
        "`((prompt_tok_per_msg * input_usd_per_mtok + completion_tok_per_msg * output_usd_per_mtok) / 1e6) "
        "* messages_per_conversation * inr_per_usd * 100`.")
    add("- p50/p95 are nearest-rank percentiles with no interpolation: the sorted per-message latencies "
        "of a run, taking the value at index `ceil(q * n) - 1`. Every value quoted is an observed "
        "latency recorded in the run JSON.")
    add("- Latency and cost are measured around the `POST /chat` call with `time.perf_counter()`, one "
        "measurement per message.")
    add("- G2 allowed amounts: every `price_inr` in data/prices.csv, the policy amounts 60, 999 and 5000, "
        "plus the case's `allowed_amounts`.")
    add("- G4 applies only to the fact, price, arithmetic, policy, hindi and hinglish categories.")
    add("")
    add("Every case, run and check behind these numbers is in the run JSON named at the top of this file.")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Timestamped name, disambiguated when two runs land in the same second."""
    stamp = _timestamp()
    target = directory / f"{stem}-{stamp}{suffix}"
    counter = 2
    while target.exists():
        target = directory / f"{stem}-{stamp}-{counter}{suffix}"
        counter += 1
    return target


def _snapshot(previous: Path, out_dir: Path) -> Path | None:
    if not previous.exists() or previous.stat().st_size == 0:
        return None
    target = _unique_path(out_dir, "summary", ".md")
    shutil.copy2(previous, target)
    return target


def _write_json_document(summary: Summary, runs: Sequence[RunSummary], out_path: Path) -> Path:
    flat: list[dict[str, Any]] = []
    for run in runs:
        for result in run.results:
            payload = result.to_dict()
            payload["run"] = run.index
            flat.append(payload)
    document = {
        "generated_at": summary.generated_at,
        "harness": "evals",
        "runs": summary.runs,
        "base_url": summary.base_url,
        "cases_path": summary.cases_path,
        "cost_config": summary.cost.to_dict(),
        "g2_allowed_rupee_amounts": sorted(allowed_rupee_amounts()),
        "valid_source_ids": sorted(valid_source_ids()),
        "percentile_method": "nearest-rank, index = ceil(q * n) - 1, no interpolation",
        "per_run": [m.to_dict() for m in summary.per_run],
        "summary": summary.to_dict(),
        "run_index": [run.to_dict() for run in runs],
        "cases": flat,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(out_path)
    return out_path


def write_reports(
    runs: list[RunSummary],
    out_dir: Path,
    *,
    config_path: Path | None = None,
    inr_per_usd: float | None = None,
    input_cost_per_mtok_usd: float | None = None,
    output_cost_per_mtok_usd: float | None = None,
) -> tuple[Path, Path]:
    """Write ``run-<timestamp>.json`` and ``summary.md``. Returns (json, md).

    A pre-existing ``summary.md`` is copied to ``summary-<timestamp>.md`` first, so
    earlier runs stay on disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = cost_settings(
        config_path,
        inr_per_usd=inr_per_usd,
        input_cost_per_mtok_usd=input_cost_per_mtok_usd,
        output_cost_per_mtok_usd=output_cost_per_mtok_usd,
    )
    summary = summarise(runs, cost)
    json_path = _write_json_document(summary, runs, _unique_path(out_dir, "run", ".json"))

    md_path = out_dir / "summary.md"
    archived = _snapshot(md_path, out_dir)
    note = f"the previous summary.md was archived as {archived.name}" if archived else ""
    md_path.write_text(
        render_summary_md(summary, partial_note=note, json_name=json_path.name), encoding="utf-8"
    )
    return json_path, md_path
