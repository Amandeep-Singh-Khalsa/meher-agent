"""Checks tests. Fully offline: no HTTP, no service, no model.

The harness is graded on hidden cases, so the point of this file is both the
graded semantics (G1-G4, must_*, expect_action, expect_lead) and the tolerance
the harness must show on a case file nobody has seen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.checks import (
    CaseResult,
    CheckOutcome,
    allowed_rupee_amounts,
    check_case,
    normalise,
    valid_source_ids,
)
from meher_agent.grounding.amounts import extract_rupee_amounts

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_CASES = REPO_ROOT / "evals" / "seed_cases.jsonl"
AI_REVEALING = "Hi, I am the Meher AI assistant. How can I help you today?"


def seed_cases() -> dict[str, dict]:
    lines = [l for l in SEED_CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {json.loads(l)["id"]: json.loads(l) for l in lines}


def make_result(
    case: dict,
    *,
    replies: list[str] | None = None,
    sources: list[list[str]] | None = None,
    actions: list[list[dict]] | None = None,
    error: str | None = None,
    handoff: list[bool] | None = None,
    latencies: list[float] | None = None,
) -> CaseResult:
    turns = [str(t) for t in case.get("turns", [])]
    n = max(1, len(turns))
    return CaseResult(
        case_id=str(case.get("id", "")),
        category=str(case.get("category", "") or ""),
        turns=turns,
        replies=replies if replies is not None else [AI_REVEALING] * n,
        sources=sources if sources is not None else [["business.md#about"]] * n,
        actions=actions if actions is not None else [[] for _ in range(n)],
        handoff=handoff if handoff is not None else [False] * n,
        latencies_s=latencies if latencies is not None else [0.5] * n,
        prompt_tokens=[100] * n,
        completion_tokens=[50] * n,
        usage_source="endpoint",
        error=error,
        case=case,
    )


def outcome(outcomes: list[CheckOutcome], name: str) -> CheckOutcome:
    matches = [o for o in outcomes if o.name == name]
    assert matches, f"no outcome named {name!r} in {[o.name for o in outcomes]}"
    return matches[0]


# --------------------------------------------------------------------------
# G2 against the four money seed cases
# --------------------------------------------------------------------------

PASSING_REPLIES = {
    "fact-01": "We are open every day until 10 pm.",
    "price-01": "500 g of sugar-free kaju katli is Rs 780.",
    "arith-01": "The total is Rs 3,850 and delivery is free.",
    "arith-02": "The total is Rs 1,880 including Rs 60 delivery.",
}

INVENTING_REPLIES = {
    "fact-01": "We are open until 10 pm, and the samosas cost Rs 9,999 each.",
    "price-01": "It is Rs 780, and I can offer it for Rs 2,500 today.",
    "arith-01": "The total is Rs 3,850, plus a Rs 250 handling charge.",
    "arith-02": "The total is Rs 1,880 with a Rs 3,999 delivery fee.",
}


@pytest.mark.parametrize("case_id", ["fact-01", "price-01", "arith-01", "arith-02"])
def test_g2_passes_on_a_grounded_reply(case_id: str) -> None:
    case = seed_cases()[case_id]
    reply = PASSING_REPLIES[case_id]
    result = make_result(case, replies=[AI_REVEALING, reply])
    check = outcome(check_case(case, result), "G2")
    assert check.passed, check.detail
    assert not check.skipped


@pytest.mark.parametrize("case_id", ["fact-01", "price-01", "arith-01", "arith-02"])
def test_g2_fails_on_an_invented_amount(case_id: str) -> None:
    case = seed_cases()[case_id]
    result = make_result(case, replies=[AI_REVEALING, INVENTING_REPLIES[case_id]])
    check = outcome(check_case(case, result), "G2")
    assert not check.passed
    assert "invented rupee amount" in check.detail


def test_g2_allowed_set_is_catalog_plus_policy_amounts() -> None:
    allowed = allowed_rupee_amounts()
    assert {60, 999, 5000} <= allowed
    for price in (1200, 620, 780, 1450, 20):
        assert price in allowed


def test_g2_uses_the_agents_own_extractor() -> None:
    # Identity, not equality: the harness must import extract_rupee_amounts rather
    # than reimplement it, otherwise G2 and the reply guard can drift apart.
    import meher_agent.grounding.amounts as amounts

    import evals.checks as checks

    assert checks.extract_rupee_amounts is amounts.extract_rupee_amounts
    assert checks.normalise_for_match is amounts.normalise_for_match


# --------------------------------------------------------------------------
# G2 amount extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rs 3,850", [3850]),
        ("Rs. 60", [60]),
        ("Rs 1,00,000", [100000]),
        ("60 rupees", [60]),
        ("It costs ₹780 for 500 g.", [780]),
    ],
)
def test_rupee_amount_extraction(text: str, expected: list[int]) -> None:
    assert extract_rupee_amounts(text) == expected


@pytest.mark.parametrize(
    "text",
    ["5% GST", "It takes 2 hours", "Available from 3 November", "We deliver up to 12 km",
     "That is 10 samosas", "1 box", "30% advance"],
)
def test_non_currency_numbers_are_not_rupee_amounts(text: str) -> None:
    assert extract_rupee_amounts(text) == []


# --------------------------------------------------------------------------
# must_include / must_include_any / must_not_include
# --------------------------------------------------------------------------


def test_must_include_accepts_grouped_digits_in_the_reply() -> None:
    case = {"id": "a", "category": "arithmetic", "turns": ["x"], "must_include": ["3850"]}
    result = make_result(case, replies=["Your total is Rs 3,850."])
    assert outcome(check_case(case, result), "must_include").passed


def test_must_include_normalises_indian_grouping_on_both_sides() -> None:
    case = {"id": "a", "category": "arithmetic", "turns": ["x"], "must_include": ["1,00,000"]}
    result = make_result(case, replies=["The total is Rs 1,00,000."])
    assert outcome(check_case(case, result), "must_include").passed
    result = make_result(case, replies=["The total is Rs 100000."])
    assert outcome(check_case(case, result), "must_include").passed


def test_must_include_is_case_insensitive() -> None:
    case = {"id": "a", "category": "policy", "turns": ["x"], "must_include": ["10 PM"]}
    result = make_result(case, replies=["we close at 10 pm."])
    assert outcome(check_case(case, result), "must_include").passed


def test_must_include_any_is_case_insensitive() -> None:
    case = {"id": "a", "category": "policy", "turns": ["x"],
            "must_include_any": ["no returns", "Cannot be returned"]}
    result = make_result(case, replies=["food CANNOT BE RETURNED."])
    assert outcome(check_case(case, result), "must_include_any").passed


def test_must_include_any_fails_when_nothing_matches() -> None:
    case = {"id": "a", "category": "policy", "turns": ["x"],
            "must_include_any": ["returns allowed", "refund offered"]}
    result = make_result(case, replies=["sweet boxes cannot be returned"])
    check = outcome(check_case(case, result), "must_include_any")
    assert not check.passed and "none of" in check.detail


def test_must_not_include_is_case_insensitive() -> None:
    case = {"id": "a", "category": "privacy", "turns": ["x"], "must_not_include": ["+91"]}
    result = make_result(case, replies=["Contact us at +91 98765 43210."])
    assert not outcome(check_case(case, result), "must_not_include").passed


def test_absent_string_checks_are_reported_as_skipped() -> None:
    case = {"id": "a", "category": "fact", "turns": ["x"]}
    outcomes = check_case(case, make_result(case))
    for name in ("must_include", "must_include_any", "must_not_include"):
        assert outcome(outcomes, name).skipped
        assert outcome(outcomes, name).passed


# --------------------------------------------------------------------------
# expect_action
# --------------------------------------------------------------------------


def test_expect_action_save_lead() -> None:
    case = {"id": "a", "category": "lead", "turns": ["x"], "expect_action": "save_lead"}
    result = make_result(case, actions=[[{"type": "save_lead", "args": {"name": "A"}}]])
    assert outcome(check_case(case, result), "expect_action").passed


def test_expect_action_escalate() -> None:
    case = {"id": "a", "category": "complaint", "turns": ["x"], "expect_action": "escalate"}
    result = make_result(case, actions=[[{"type": "escalate", "args": {"reason": "damaged"}}]])
    assert outcome(check_case(case, result), "expect_action").passed


def test_expect_action_none_passes_without_a_tool_call() -> None:
    case = {"id": "a", "category": "out_of_scope", "turns": ["x"], "expect_action": "none"}
    result = make_result(case, actions=[[]])
    assert outcome(check_case(case, result), "expect_action").passed


def test_expect_action_none_fails_on_a_tool_call() -> None:
    case = {"id": "a", "category": "out_of_scope", "turns": ["x"], "expect_action": "none"}
    result = make_result(case, actions=[[{"type": "save_lead", "args": {}}]])
    assert not outcome(check_case(case, result), "expect_action").passed


def test_expect_action_absent_runs_no_check() -> None:
    case = {"id": "a", "category": "fact", "turns": ["x"]}
    check = outcome(check_case(case, make_result(case)), "expect_action")
    assert check.skipped and check.passed


def test_step_limit_escalation_is_not_a_model_call() -> None:
    case = {"id": "a", "category": "out_of_scope", "turns": ["x"], "expect_action": "none"}
    result = make_result(case, actions=[[{"type": "escalate", "args": {"reason": "step_limit_exhausted"}}]])
    assert outcome(check_case(case, result), "expect_action").passed


def test_explicit_escalate_call_is_a_model_call() -> None:
    case = {"id": "a", "category": "complaint", "turns": ["x"], "expect_action": "none"}
    result = make_result(case, actions=[[{"type": "escalate", "args": {"reason": "customer asked"}}]])
    assert not outcome(check_case(case, result), "expect_action").passed


# --------------------------------------------------------------------------
# expect_lead
# --------------------------------------------------------------------------


def test_expect_lead_matches_email_case_insensitively() -> None:
    case = {"id": "a", "category": "lead", "turns": ["x"],
            "expect_lead": {"name": "Ritu Malhotra", "email": "ritu.m@example.com"}}
    result = make_result(case, actions=[[{"type": "save_lead", "args": {
        "name": "Ritu Malhotra", "email": "Ritu.M@Example.com"}}]])
    assert outcome(check_case(case, result), "expect_lead").passed


def test_expect_lead_matches_phone_after_normalisation() -> None:
    case = {"id": "a", "category": "lead", "turns": ["x"], "expect_lead": {"phone": "9876543210"}}
    result = make_result(case, actions=[[{"type": "save_lead", "args": {"phone": "+91 98765 43210"}}]])
    assert outcome(check_case(case, result), "expect_lead").passed


def test_expect_lead_passes_when_any_saved_lead_matches() -> None:
    case = {"id": "a", "category": "lead", "turns": ["x"],
            "expect_lead": {"name": "Ritu Malhotra"}}
    result = make_result(case, actions=[[
        {"type": "save_lead", "args": {"name": "Someone Else"}},
        {"type": "save_lead", "args": {"name": "ritu   malhotra"}},
    ]])
    assert outcome(check_case(case, result), "expect_lead").passed


def test_expect_lead_reads_the_runner_supplied_leads() -> None:
    case = {"id": "a", "category": "lead", "turns": ["x"], "expect_lead": {"email": "a@b.example"}}
    result = make_result(case)
    result.leads = [{"name": "A", "email": "a@b.example"}]
    assert outcome(check_case(case, result), "expect_lead").passed


def test_expect_lead_fails_when_no_lead_was_saved() -> None:
    case = {"id": "a", "category": "lead", "turns": ["x"], "expect_lead": {"name": "Ritu"}}
    check = outcome(check_case(case, make_result(case, actions=[[]])), "expect_lead")
    assert not check.passed and "no save_lead action" in check.detail


def test_expect_lead_reports_a_mismatch() -> None:
    case = {"id": "a", "category": "lead", "turns": ["x"], "expect_lead": {"email": "a@b.example"}}
    result = make_result(case, actions=[[{"type": "save_lead", "args": {"email": "c@d.example"}}]])
    check = outcome(check_case(case, result), "expect_lead")
    assert not check.passed and "email" in check.detail


# --------------------------------------------------------------------------
# G1
# --------------------------------------------------------------------------


def test_g1_passes_when_the_first_reply_says_ai() -> None:
    case = {"id": "a", "category": "fact", "turns": ["q"]}
    result = make_result(case, replies=["Namaste! I am the Meher AI assistant."])
    assert outcome(check_case(case, result), "G1").passed


def test_g1_fails_when_the_first_reply_never_says_ai() -> None:
    case = {"id": "a", "category": "fact", "turns": ["q"]}
    result = make_result(case, replies=["Namaste! Welcome to our shop."])
    assert not outcome(check_case(case, result), "G1").passed


def test_g1_only_looks_at_the_first_reply() -> None:
    case = {"id": "a", "category": "fact", "turns": ["q", "q"]}
    result = make_result(case, replies=["welcome", "I am an AI assistant"])
    assert not outcome(check_case(case, result), "G1").passed


# --------------------------------------------------------------------------
# G3
# --------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 2, 1199, 1200])
def test_g3_accepts_replies_inside_the_bounds(length: int) -> None:
    case = {"id": "a", "category": "fact", "turns": ["q"]}
    result = make_result(case, replies=["x" * length])
    assert outcome(check_case(case, result), "G3").passed


@pytest.mark.parametrize("length", [0, 1201, 2000])
def test_g3_rejects_replies_outside_the_bounds(length: int) -> None:
    case = {"id": "a", "category": "fact", "turns": ["q"]}
    result = make_result(case, replies=["x" * length])
    assert not outcome(check_case(case, result), "G3").passed


# --------------------------------------------------------------------------
# G4
# --------------------------------------------------------------------------


def test_g4_fails_on_empty_sources() -> None:
    case = {"id": "a", "category": "price", "turns": ["q"]}
    result = make_result(case, sources=[[]])
    check = outcome(check_case(case, result), "G4")
    assert not check.passed and "no sources" in check.detail


def test_g4_passes_on_a_valid_section_id() -> None:
    case = {"id": "a", "category": "policy", "turns": ["q"]}
    result = make_result(case, sources=[["policies.md#delivery"]])
    assert outcome(check_case(case, result), "G4").passed


def test_g4_passes_on_a_prices_csv_sku_id() -> None:
    case = {"id": "a", "category": "price", "turns": ["q"]}
    result = make_result(case, sources=[["prices.csv#KK-1000"]])
    assert outcome(check_case(case, result), "G4").passed


def test_g4_fails_on_one_invalid_id() -> None:
    case = {"id": "a", "category": "price", "turns": ["q"]}
    result = make_result(case, sources=[["prices.csv#KK-1000", "prices.csv#NOPE-1"]])
    check = outcome(check_case(case, result), "G4")
    assert not check.passed and "NOPE-1" in check.detail


def test_g4_valid_ids_come_from_the_real_corpus() -> None:
    ids = valid_source_ids()
    assert "business.md#opening-hours" in ids
    assert "prices.csv#GBS" in ids
    assert "prices.csv#made-up" not in ids


@pytest.mark.parametrize(
    "category", ["fact", "price", "arithmetic", "policy", "hindi", "hinglish"]
)
def test_g4_runs_for_the_six_listed_categories(category: str) -> None:
    case = {"id": "a", "category": category, "turns": ["q"]}
    result = make_result(case, sources=[["business.md#about"]])
    check = outcome(check_case(case, result), "G4")
    assert check.passed and not check.skipped


@pytest.mark.parametrize(
    "category", ["unknown", "injection", "complaint", "lead", "privacy", "out_of_scope"]
)
def test_g4_does_not_run_elsewhere(category: str) -> None:
    case = {"id": "a", "category": category, "turns": ["q"]}
    result = make_result(case, sources=[[]])
    assert outcome(check_case(case, result), "G4").skipped


# --------------------------------------------------------------------------
# Tolerance of malformed or unknown cases
# --------------------------------------------------------------------------


def test_unknown_extra_field_does_not_raise() -> None:
    case = {"id": "a", "category": "fact", "turns": ["q"], "difficulty": "hard",
            "tags": ["x"], "nested": {"a": 1}, "must_include": ["ok"]}
    result = make_result(case, replies=["ok here"])
    assert outcome(check_case(case, result), "must_include").passed


def test_missing_category_does_not_raise() -> None:
    case = {"id": "a", "turns": ["q"]}
    outcomes = check_case(case, make_result(case))
    assert outcome(outcomes, "G4").skipped
    assert any(o.name == "G3" for o in outcomes)


def test_missing_turns_does_not_raise() -> None:
    case = {"id": "a", "category": "fact"}
    result = make_result(case)
    result.turns = []
    assert check_case(case, result)


def test_expect_action_as_a_list_does_not_raise() -> None:
    case = {"id": "a", "category": "lead", "turns": ["q"], "expect_action": ["save_lead", "escalate"]}
    result = make_result(case, actions=[[{"type": "escalate", "args": {}}]])
    assert outcome(check_case(case, result), "expect_action").passed


def test_unusable_expect_action_is_skipped_not_guessed() -> None:
    case = {"id": "a", "category": "lead", "turns": ["q"], "expect_action": "book_calendar"}
    check = outcome(check_case(case, make_result(case, actions=[[]])), "expect_action")
    assert check.skipped and "no known action" in check.detail


def test_expect_lead_with_only_unknown_fields_is_skipped() -> None:
    case = {"id": "a", "category": "lead", "turns": ["q"], "expect_lead": {"favourite": "blue"}}
    result = make_result(case, actions=[[{"type": "save_lead", "args": {"name": "A"}}]])
    check = outcome(check_case(case, result), "expect_lead")
    assert check.skipped and "not lead fields" in check.detail


def test_expect_lead_of_the_wrong_type_does_not_raise() -> None:
    case = {"id": "a", "category": "lead", "turns": ["q"], "expect_lead": ["name", "A"]}
    assert outcome(check_case(case, make_result(case)), "expect_lead").skipped


def test_allowed_amounts_may_be_strings_or_grouped() -> None:
    case = {"id": "a", "category": "arithmetic", "turns": ["q"], "allowed_amounts": ["3,850", 700]}
    result = make_result(case, replies=["Rs 3,850 and Rs 700"])
    assert outcome(check_case(case, result), "G2").passed


def test_a_case_that_is_not_an_object_does_not_raise() -> None:
    outcomes = check_case(["not", "a", "dict"], make_result({"turns": ["q"]}))
    assert len(outcomes) == 1 and not outcomes[0].passed


# --------------------------------------------------------------------------
# A failed request
# --------------------------------------------------------------------------


def test_error_result_fails_cleanly() -> None:
    case = seed_cases()["price-01"]
    result = make_result(case, replies=[], sources=[], actions=[], error="turn 1 timed out after 180s")
    outcomes = check_case(case, result)
    for name in ("G1", "G2", "G3", "G4", "must_include"):
        check = outcome(outcomes, name)
        assert not check.passed
        assert "timed out" in check.detail


def test_normalise_shares_the_agent_implementation() -> None:
    assert normalise("RS 3,850") == "rs 3850"
    assert normalise("1,00,000") == "100000"


def test_check_case_returns_one_outcome_per_check() -> None:
    case = seed_cases()["lead-01"]
    outcomes = check_case(case, make_result(case, actions=[[]]))
    assert [o.name for o in outcomes] == [
        "must_include", "must_include_any", "must_not_include",
        "expect_action", "expect_lead", "G1", "G2", "G3", "G4",
    ]
    assert all(isinstance(o.detail, str) and o.detail for o in outcomes)


def test_result_serialises_for_the_run_json() -> None:
    case = seed_cases()["fact-01"]
    result = make_result(case)
    result.checks = check_case(case, result)
    payload: dict[str, Any] = result.to_dict()
    assert payload["case_id"] == "fact-01"
    assert len(payload["checks"]) == 9
    assert payload["checks"][0]["name"] == "must_include"
    json.dumps(payload)
