"""Report tests: the metrics and the cost arithmetic, offline and synthetic.

These build ``RunSummary`` objects by hand rather than replaying anything, so
report.py is exercised without a service, a model, or a case file. The numbers
the technical report quotes are all produced here, so the denominators are worth
asserting explicitly: an action case that never set ``expect_action`` and a
second turn of an existing conversation both have to stay out of the way.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evals.checks import CaseResult, CheckOutcome, RunSummary
from evals.report import (
    CostSettings,
    aggregate,
    cost_inr_per_100_conversations,
    cost_settings,
    percentile,
    render_summary_md,
    summarise,
    summarise_run,
    write_reports,
)

PASS = "pass"
FAIL = "fail"
SKIP = "skip"


def outcome(name: str, passed: bool = True, skipped: bool = False, detail: str = "") -> CheckOutcome:
    return CheckOutcome(
        name=name,
        passed=passed,
        detail=detail or ("ok" if passed else "not ok"),
        skipped=skipped,
    )


def make_result(
    case_id: str,
    category: str,
    checks: list[CheckOutcome],
    *,
    turns: int = 1,
    latencies: list[float] | None = None,
    prompt: list[int] | None = None,
    completion: list[int] | None = None,
    usage_source: str = "endpoint",
    error: str | None = None,
) -> CaseResult:
    result = CaseResult(
        case_id=case_id,
        category=category,
        turns=["q"] * turns,
        replies=[AI] * turns,
        sources=[["business.md#about"]] * turns,
        actions=[[]] * turns,
        handoff=[False] * turns,
        latencies_s=latencies if latencies is not None else [1.0] * turns,
        prompt_tokens=prompt if prompt is not None else [1000] * turns,
        completion_tokens=completion if completion is not None else [200] * turns,
        usage_source=usage_source,
        error=error,
    )
    result.checks = checks
    return result


AI = "I am the Meher AI assistant."


def default_checks(
    *,
    g1: bool = True,
    g2: bool = True,
    g3: bool = True,
    g4: bool | None = None,
    action: CheckOutcome | None = None,
    must_include: CheckOutcome | None = None,
) -> list[CheckOutcome]:
    checks = [
        must_include or outcome("must_include", skipped=True),
        outcome("G1", g1),
        outcome("G2", g2),
        outcome("G3", g3),
    ]
    checks.append(outcome("G4", True, skipped=g4 is None) if g4 is None else outcome("G4", g4))
    if action is not None:
        checks.append(action)
    return checks


def make_run(index: int, results: list[CaseResult], *, base_url: str = "http://test") -> RunSummary:
    return RunSummary(
        index=index,
        base_url=base_url,
        cases_path="evals/cases.jsonl",
        started_at="2026-09-27T10:00:00+05:30",
        finished_at="2026-09-27T10:30:00+05:30",
        results=results,
    )


LOCAL = CostSettings(inr_per_usd=87.0, input_cost_per_mtok_usd=0.0, output_cost_per_mtok_usd=0.0)
PRICED = CostSettings(inr_per_usd=87.0, input_cost_per_mtok_usd=0.5, output_cost_per_mtok_usd=1.5)


# --------------------------------------------------------------------------
# Percentiles
# --------------------------------------------------------------------------


def test_percentile_is_nearest_rank_without_interpolation() -> None:
    values = [float(v) for v in range(1, 11)]
    assert percentile(values, 0.50) == 5.0
    assert percentile(values, 0.95) == 10.0


def test_percentile_indices_follow_ceil_q_times_n() -> None:
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 0.50) == 10.0
    assert percentile(values, 0.95) == 19.0


def test_percentile_of_a_single_sample() -> None:
    assert percentile([2.5], 0.50) == 2.5
    assert percentile([2.5], 0.95) == 2.5


def test_percentile_ignores_input_order() -> None:
    # sorted -> [1, 3, 5, 9]; nearest rank at q=0.5 is ceil(0.5*4) = 2 -> 3.0
    assert percentile([9.0, 1.0, 5.0, 3.0], 0.50) == 3.0


def test_percentile_of_nothing_is_zero() -> None:
    assert percentile([], 0.95) == 0.0


def test_p50_and_p95_pool_every_message_in_the_run() -> None:
    results = [
        make_result(f"c{i}", "fact", default_checks(), latencies=[1.0, 2.0]) for i in range(5)
    ]
    metrics = summarise_run(make_run(1, results), LOCAL)
    assert metrics.messages == 10
    # sorted -> [1,1,1,1,1,2,2,2,2,2]; ceil(0.5*10) = 5 and ceil(0.95*10) = 10
    assert metrics.p50_s == pytest.approx(1.0)
    assert metrics.p95_s == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Mean of runs and worst run
# --------------------------------------------------------------------------


def test_mean_and_worst_run_for_a_rate() -> None:
    metric = aggregate("x", [0.5, 1.0, 0.75], higher_is_better=True)
    assert metric.per_run == (0.5, 1.0, 0.75)
    assert metric.mean == pytest.approx(0.75)
    assert metric.worst == pytest.approx(0.5)
    assert metric.best == pytest.approx(1.0)


def test_worst_run_for_a_cost_is_the_most_expensive() -> None:
    metric = aggregate("x", [1.0, 3.0, 2.0], higher_is_better=False)
    assert metric.worst == pytest.approx(3.0)
    assert metric.best == pytest.approx(1.0)


def test_overall_pass_rate_is_the_mean_and_worst_of_the_runs() -> None:
    runs = [
        make_run(1, [make_result(f"a{i}", "fact", default_checks(g1=i != 0)) for i in range(4)]),
        make_run(2, [make_result(f"a{i}", "fact", default_checks()) for i in range(4)]),
        make_run(3, [make_result(f"a{i}", "fact", default_checks(g1=i != 3)) for i in range(4)]),
    ]
    summary = summarise(runs, LOCAL)
    rate = summary.rate("overall_pass_rate")
    assert rate.per_run == (0.75, 1.0, 0.75)
    assert rate.mean == pytest.approx(0.8333333, rel=1e-5)
    assert rate.worst == pytest.approx(0.75)


def test_latency_p95_worst_run_is_the_slowest_run() -> None:
    runs = [
        make_run(1, [make_result("a", "fact", default_checks(), latencies=[1.0])]),
        make_run(2, [make_result("a", "fact", default_checks(), latencies=[9.0])]),
        make_run(3, [make_result("a", "fact", default_checks(), latencies=[2.0])]),
    ]
    summary = summarise(runs, LOCAL)
    p95 = summary.rate("latency_p95_s")
    assert p95.per_run == (1.0, 9.0, 2.0)
    assert p95.worst == pytest.approx(9.0)
    assert p95.mean == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Guardrail rates and their denominators
# --------------------------------------------------------------------------


def test_invented_amount_rate_is_g2_failures_over_replies_checked() -> None:
    results = [
        make_result("a", "fact", default_checks()),
        make_result("b", "price", default_checks(g2=False)),
        make_result("c", "price", default_checks()),
        make_result("d", "arithmetic", default_checks(g2=False)),
    ]
    summary = summarise([make_run(1, results)], LOCAL)
    rate = summary.rate("invented_amount_rate")
    assert rate.per_run == (0.5,)
    assert rate.denominator == 4
    assert rate.numerator == 2
    assert rate.worst == pytest.approx(0.5)


def test_invented_amount_rate_worst_run_is_the_worst_rate() -> None:
    runs = [
        make_run(1, [make_result("a", "fact", default_checks())]),
        make_run(2, [make_result("a", "fact", default_checks(g2=False))]),
    ]
    rate = summarise(runs, LOCAL).rate("invented_amount_rate")
    assert rate.worst == pytest.approx(1.0)
    assert rate.best == pytest.approx(0.0)


def test_action_accuracy_counts_only_cases_that_set_expect_action() -> None:
    results = [
        make_result("a", "lead", default_checks(action=outcome("expect_action", True))),
        make_result("b", "fact", default_checks()),
        make_result("c", "fact", default_checks()),
        make_result("d", "complaint", default_checks(action=outcome("expect_action", False))),
    ]
    summary = summarise([make_run(1, results)], LOCAL)
    rate = summary.rate("action_accuracy")
    assert rate.denominator == 2
    assert rate.per_run == (0.5,)


def test_a_skipped_action_check_is_out_of_the_denominator() -> None:
    results = [
        make_result("a", "lead", default_checks(action=outcome("expect_action", True))),
        make_result("b", "lead", default_checks(action=outcome("expect_action", skipped=True))),
    ]
    rate = summarise([make_run(1, results)], LOCAL).rate("action_accuracy")
    assert rate.denominator == 1
    assert rate.per_run == (1.0,)


def test_ai_disclosure_rate_is_over_first_replies_only() -> None:
    results = [
        make_result("a", "fact", default_checks(g1=True), turns=3),
        make_result("b", "fact", default_checks(g1=False), turns=3),
        make_result("c", "hindi", default_checks(g1=True), turns=1),
    ]
    summary = summarise([make_run(1, results)], LOCAL)
    rate = summary.rate("ai_disclosure_rate")
    assert rate.denominator == 3
    assert rate.per_run == (pytest.approx(2 / 3),)
    assert summary.per_run[0].messages == 7


def test_per_category_pass_rates() -> None:
    results = [
        make_result("a", "price", default_checks()),
        make_result("b", "price", default_checks(g2=False)),
        make_result("c", "lead", default_checks()),
    ]
    summary = summarise([make_run(1, results)], LOCAL)
    assert summary.rate("category:price").per_run == (0.5,)
    assert summary.rate("category:lead").per_run == (1.0,)


def test_g4_skips_do_not_dilute_the_citation_rate() -> None:
    results = [
        make_result("a", "price", default_checks(g4=True)),
        make_result("b", "lead", default_checks(g4=None)),
    ]
    rate = summarise([make_run(1, results)], LOCAL).rate("g4_pass_rate")
    assert rate.denominator == 1


def test_error_rate_counts_failed_requests() -> None:
    results = [
        make_result("a", "fact", default_checks()),
        make_result("b", "fact", default_checks(g2=False, g3=False), error="HTTP 500"),
    ]
    rate = summarise([make_run(1, results)], LOCAL).rate("error_rate")
    assert rate.per_run == (0.5,)


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def test_cost_per_100_conversations_matches_the_hand_computation() -> None:
    # ((10000*0.5 + 2000*1.5)/1e6) * 87 * 100 = 69.6
    value = cost_inr_per_100_conversations(
        prompt_tokens_per_message=10_000,
        completion_tokens_per_message=2_000,
        messages_per_conversation=1,
        inr_per_usd=87,
        input_cost_per_mtok_usd=0.5,
        output_cost_per_mtok_usd=1.5,
    )
    assert value == pytest.approx(69.6)


def test_cost_scales_with_the_number_of_messages_per_conversation() -> None:
    single = cost_inr_per_100_conversations(
        prompt_tokens_per_message=10_000,
        completion_tokens_per_message=2_000,
        messages_per_conversation=1,
        inr_per_usd=87,
        input_cost_per_mtok_usd=0.5,
        output_cost_per_mtok_usd=1.5,
    )
    double = cost_inr_per_100_conversations(
        prompt_tokens_per_message=10_000,
        completion_tokens_per_message=2_000,
        messages_per_conversation=2,
        inr_per_usd=87,
        input_cost_per_mtok_usd=0.5,
        output_cost_per_mtok_usd=1.5,
    )
    assert double == pytest.approx(2 * single)


def test_run_cost_uses_the_measured_tokens() -> None:
    results = [
        make_result("a", "fact", default_checks(), turns=1, prompt=[10_000], completion=[2_000])
    ]
    metrics = summarise_run(make_run(1, results), PRICED)
    assert metrics.messages == 1
    assert metrics.prompt_tokens_per_message == pytest.approx(10_000)
    assert metrics.completion_tokens_per_message == pytest.approx(2_000)
    assert metrics.cost_inr_per_100_conversations == pytest.approx(69.6)


def test_a_local_model_costs_zero_but_still_reports_tokens() -> None:
    results = [
        make_result("a", "fact", default_checks(), turns=2, prompt=[10_000, 10_000],
                    completion=[2_000, 2_000])
    ]
    metrics = summarise_run(make_run(1, results), LOCAL)
    assert metrics.cost_inr_per_100_conversations == 0.0
    assert metrics.prompt_tokens == 20_000
    assert metrics.completion_tokens == 4_000
    summary = summarise([make_run(1, results)], LOCAL)
    assert "Rs 0.00" in render_summary_md(summary)
    assert "20,000" in render_summary_md(summary).replace("20000", "20,000")


def test_unreported_usage_stays_zero_and_is_said_out_loud() -> None:
    results = [
        make_result("a", "fact", default_checks(), prompt=[0], completion=[0],
                    usage_source="unknown")
    ]
    summary = summarise([make_run(1, results)], PRICED)
    assert summary.per_run[0].prompt_tokens == 0
    assert summary.per_run[0].messages_with_usage == 0
    text = render_summary_md(summary)
    assert "never estimates tokens" in text


def test_cost_settings_are_read_from_config_not_hardcoded() -> None:
    settings = cost_settings(Path("config.toml"))
    assert settings.inr_per_usd == 87.0
    assert settings.input_cost_per_mtok_usd == 0.0
    assert settings.output_cost_per_mtok_usd == 0.0


def test_cost_settings_accept_overrides() -> None:
    settings = cost_settings(inr_per_usd=87, input_cost_per_mtok_usd=0.5, output_cost_per_mtok_usd=1.5)
    assert settings.to_dict() == {
        "inr_per_usd": 87.0,
        "input_cost_per_mtok_usd": 0.5,
        "output_cost_per_mtok_usd": 1.5,
        "source": "config.toml",
    }


# --------------------------------------------------------------------------
# The written artefacts
# --------------------------------------------------------------------------


def three_runs() -> list[RunSummary]:
    return [
        make_run(1, [make_result("a", "price", default_checks(), latencies=[1.0])]),
        make_run(2, [make_result("a", "price", default_checks(g2=False), latencies=[3.0])]),
        make_run(3, [make_result("a", "price", default_checks(), latencies=[2.0])]),
    ]


def test_write_reports_writes_a_timestamped_json_and_a_summary(tmp_path: Path) -> None:
    json_path, md_path = write_reports(three_runs(), tmp_path)
    assert json_path.parent == tmp_path
    assert re.fullmatch(r"run-\d{8}-\d{6}\.json", json_path.name)
    assert md_path == tmp_path / "summary.md"
    assert json_path.exists() and md_path.exists()


def test_run_json_carries_every_case_run_and_check(tmp_path: Path) -> None:
    json_path, _ = write_reports(three_runs(), tmp_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["runs"] == 3
    assert len(document["cases"]) == 3
    for entry in document["cases"]:
        assert entry["case_id"] == "a"
        assert {c["name"] for c in entry["checks"]} >= {"G1", "G2", "G3", "G4"}
        assert all("detail" in c and "passed" in c for c in entry["checks"])
    assert document["summary"]["rates"]["overall_pass_rate"]["per_run"] == [1.0, 0.0, 1.0]
    assert document["g2_allowed_rupee_amounts"] == sorted(document["g2_allowed_rupee_amounts"])
    assert 999 in document["g2_allowed_rupee_amounts"]
    assert "prices.csv#KK-1000" in document["valid_source_ids"]


def test_summary_reports_mean_and_worst_for_every_rate(tmp_path: Path) -> None:
    _, md_path = write_reports(three_runs(), tmp_path)
    text = md_path.read_text(encoding="utf-8")
    for needle in (
        "Overall pass rate",
        "Invented-amount rate",
        "AI-disclosure rate",
        "Action accuracy",
        "Latency p50 per message",
        "Latency p95 per message",
        "Prompt tokens per message",
        "Completion tokens per message",
        "Cost in rupees per 100 conversations",
        "mean",
        "worst",
    ):
        assert needle in text, needle
    assert "66.7%" in text


def test_summary_states_the_denominators(tmp_path: Path) -> None:
    runs = [
        make_run(
            1,
            [
                make_result("a", "lead", default_checks(action=outcome("expect_action", True))),
                make_result("b", "fact", default_checks(g1=False)),
            ],
        )
    ]
    _, md_path = write_reports(runs, tmp_path)
    text = md_path.read_text(encoding="utf-8")
    assert "0 G2 failures over 2 replies checked" in text
    assert "1 first replies disclosed AI out of 2 first replies" in text
    assert "1 correct out of 1 cases that set `expect_action`" in text


def test_a_previous_summary_is_archived_not_deleted(tmp_path: Path) -> None:
    first_json, md_path = write_reports(three_runs(), tmp_path)
    original = md_path.read_text(encoding="utf-8")
    second_json, rewritten = write_reports(three_runs(), tmp_path)
    assert rewritten == md_path
    assert md_path.exists() and "Evaluation summary" in md_path.read_text(encoding="utf-8")
    archived = sorted(p.name for p in tmp_path.glob("summary-*.md"))
    assert len(archived) == 1
    assert (tmp_path / archived[0]).read_text(encoding="utf-8") == original
    assert first_json != second_json


def test_no_runs_still_writes_both_artefacts(tmp_path: Path) -> None:
    json_path, md_path = write_reports([], tmp_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["cases"] == []
    assert "no runs" in md_path.read_text(encoding="utf-8")


def test_run_metrics_serialise(tmp_path: Path) -> None:
    metrics = summarise_run(make_run(1, [make_result("a", "fact", default_checks())]), LOCAL)
    payload = json.loads(json.dumps(metrics.to_dict()))
    assert payload["cases"] == 1
    assert payload["latencies_s"] == [1.0]
