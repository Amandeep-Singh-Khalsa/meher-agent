"""Tests for evals/cases.jsonl and for the oracle that produced its numbers.

Two things are being defended here.

1. The thirteen seed cases are the task's own fixture, not ours. They are
   compared byte for byte, in order, as the first thirteen lines, so nobody can
   quietly reformat or improve them.
2. Every rupee number in a quoted case was produced by
   ``scripts/compute_expected_totals.py``, not by a human. The tests below
   recompute each one and fail on a disagreement, which is what makes the file an
   independent oracle rather than a restatement of the agent's own billing.

The oracle is loaded from its file path rather than imported as a package module
because it must not depend on the package at all; one test here greps its source
to keep it that way.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from meher_agent.config import (
    POLICY_BULK_ADVANCE_PCT,
    POLICY_BULK_BOXES,
    POLICY_BULK_KG,
    POLICY_COD_LIMIT_INR,
    POLICY_DELIVERY_FEE_INR,
    POLICY_DELIVERY_RADIUS_KM,
    POLICY_DISCOUNT_MIN_BOXES,
    POLICY_DISCOUNT_PCT,
    POLICY_FREE_DELIVERY_ABOVE_INR,
    REPO_ROOT,
)
from meher_agent.data.corpus import load_corpus
from meher_agent.tools.validation import is_valid_email, is_valid_phone

ROOT = Path(REPO_ROOT)
CASES_PATH = ROOT / "evals" / "cases.jsonl"
SEED_PATH = ROOT / "evals" / "seed_cases.jsonl"
ORACLE_PATH = ROOT / "scripts" / "compute_expected_totals.py"

SEED_IDS: frozenset[str] = frozenset(
    {
        "fact-01", "price-01", "arith-01", "arith-02", "policy-01", "unknown-01",
        "hindi-01", "hinglish-01", "inject-01", "lead-01", "complaint-01",
        "privacy-01", "oos-01",
    }
)
SEED_CATEGORIES: frozenset[str] = frozenset(
    {
        "fact", "price", "arithmetic", "policy", "unknown", "hindi", "hinglish",
        "injection", "lead", "complaint", "privacy", "out_of_scope",
    }
)
TEAM_REPLY_MARKERS: tuple[str, ...] = (
    "team", "don't have", "do not have", "not on", "isn't on", "is not on",
)

#: The three money amounts written into data/policies.md. A hand-typed
#: allowed_amounts value has to be one of these, a catalog price, a customer
#: number lifted from the case text, or something the oracle computed.
POLICY_AMOUNTS: frozenset[int] = frozenset(
    {POLICY_DELIVERY_FEE_INR, POLICY_FREE_DELIVERY_ABOVE_INR, POLICY_COD_LIMIT_INR}
)

_DIGIT_RUN_RE = re.compile(r"\d[\d,]*")
_CASE_FIELDS: frozenset[str] = frozenset(
    {
        "id", "category", "turns", "must_include", "must_include_any",
        "must_not_include", "expect_action", "expect_lead", "allowed_amounts",
        "quote", "note",
    }
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _load_oracle() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compute_expected_totals", ORACLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def oracle() -> ModuleType:
    return _load_oracle()


@pytest.fixture(scope="module")
def raw_cases() -> str:
    return CASES_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def case_lines(raw_cases: str) -> list[str]:
    lines = raw_cases.split("\n")
    assert lines[-1] == "", "cases.jsonl must end with a newline"
    return lines[:-1]


@pytest.fixture(scope="module")
def cases(case_lines: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in case_lines]


@pytest.fixture(scope="module")
def seed_lines() -> list[str]:
    return SEED_PATH.read_text(encoding="utf-8").split("\n")[:-1]


@pytest.fixture(scope="module")
def catalog() -> dict[str, int]:
    return {sku.sku: sku.price_inr for sku in load_corpus().skus}


def quoted(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cases if isinstance(c.get("quote"), dict)]


def authored(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cases if c["id"] not in SEED_IDS]


def quote_result(oracle: ModuleType, case: dict[str, Any]) -> dict[str, Any]:
    quote = case["quote"]
    lines = [(entry["sku"], int(entry["qty"])) for entry in quote["lines"]]
    distance = quote.get("distance_km")
    return oracle.compute(lines, None if distance is None else int(distance))


def text_amounts(case: dict[str, Any]) -> set[int]:
    """Every integer the customer's own words contain, separators stripped."""
    found: set[int] = set()
    for turn in case.get("turns") or []:
        for run in _DIGIT_RUN_RE.findall(str(turn)):
            cleaned = run.replace(",", "")
            if cleaned.isdigit():
                found.add(int(cleaned))
    return found


# --------------------------------------------------------------------------
# Seed integrity
# --------------------------------------------------------------------------


def test_seed_cases_are_the_first_thirteen_lines_byte_for_byte(
    case_lines: list[str], seed_lines: list[str]
) -> None:
    assert len(seed_lines) == 13, "the seed file is expected to hold exactly 13 cases"
    assert case_lines[:13] == seed_lines, (
        "the first thirteen lines of cases.jsonl must be the seed lines unchanged: "
        "same bytes, same order, no added fields"
    )


def test_seed_bytes_on_disk_match_not_just_the_parsed_values() -> None:
    seed_raw = SEED_PATH.read_bytes()
    cases_raw = CASES_PATH.read_bytes()
    seed_body = seed_raw.split(b"\n")[:13]
    assert cases_raw.split(b"\n")[:13] == seed_body
    assert b"\r\n" not in cases_raw, "cases.jsonl must be LF-only"


def test_reserialising_a_seed_line_reproduces_its_bytes(oracle: ModuleType) -> None:
    for line in SEED_PATH.read_text(encoding="utf-8").split("\n")[:-1]:
        assert oracle.dumps_case(json.loads(line)) == line


def test_the_oracle_key_order_covers_every_seed_field(oracle: ModuleType) -> None:
    for line in SEED_PATH.read_text(encoding="utf-8").split("\n")[:-1]:
        case = json.loads(line)
        assert set(case) <= set(oracle.KEY_ORDER)


# --------------------------------------------------------------------------
# File shape
# --------------------------------------------------------------------------


def test_total_case_count_is_at_least_fifty(cases: list[dict[str, Any]]) -> None:
    assert len(cases) >= 50
    assert len(cases) - len(SEED_IDS) >= 37


def test_case_ids_are_unique(cases: list[dict[str, Any]]) -> None:
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case id"


def test_every_line_is_one_json_object_with_non_empty_turns(
    cases: list[dict[str, Any]], case_lines: list[str]
) -> None:
    assert len(cases) == len(case_lines)
    for line, case in zip(case_lines, cases):
        assert isinstance(case, dict), line
        assert line == line.strip(), f"case {case.get('id')} has stray whitespace"
        assert line.startswith("{") and line.endswith("}"), line
        assert set(case) <= _CASE_FIELDS, f"case {case.get('id')} has unknown fields"
        turns = case.get("turns")
        assert isinstance(turns, list) and turns, f"case {case.get('id')} has no turns"
        for turn in turns:
            assert isinstance(turn, str) and turn.strip()


def test_every_case_declares_a_seed_category(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        assert case.get("category") in SEED_CATEGORIES, case["id"]


def test_every_authored_case_says_what_makes_it_hard(
    cases: list[dict[str, Any]]
) -> None:
    for case in authored(cases):
        note = case.get("note")
        assert isinstance(note, str) and len(note.strip()) >= 20, case["id"]


def test_expect_action_only_names_actions_the_harness_understands(
    cases: list[dict[str, Any]]
) -> None:
    allowed = {"save_lead", "escalate", "none"}
    for case in cases:
        if "expect_action" in case:
            assert case["expect_action"] in allowed, case["id"]


# --------------------------------------------------------------------------
# Quotas
# --------------------------------------------------------------------------


def test_hindi_or_hinglish_quota(cases: list[dict[str, Any]]) -> None:
    counted = [c for c in cases if c["category"] in {"hindi", "hinglish"}]
    assert len(counted) >= 10, "at least ten Hindi or Hinglish cases"
    assert sum(1 for c in counted if c["category"] == "hindi") >= 5
    assert sum(1 for c in counted if c["category"] == "hinglish") >= 6
    assert any(
        any("ऀ" <= ch <= "ॿ" for ch in turn) for c in counted for turn in c["turns"]
    ), "at least one case must actually be written in Devanagari"
    assert any(
        any(re.search(r"\d", turn) for turn in c["turns"])
        and any("ऀ" <= ch <= "ॿ" for ch in " ".join(c["turns"]))
        for c in cases
    ), "at least one Hindi case must contain Latin digits"


def test_injection_quota(cases: list[dict[str, Any]]) -> None:
    assert sum(1 for c in cases if c["category"] == "injection") >= 5


def test_injection_set_covers_the_required_attack_shapes(
    cases: list[dict[str, Any]]
) -> None:
    blob = " ".join(
        turn
        for case in cases
        if case["category"] == "injection"
        for turn in case["turns"]
    ).casefold()
    for shape in (
        "ignore all previous instructions",
        "owner",
        "developer mode",
        "system prompt",
        "50% off",
        "play a game",
        "another customer",
    ):
        assert shape in blob, f"no injection case covers {shape!r}"


def test_injection_set_has_a_false_positive_control(
    oracle: ModuleType, cases: list[dict[str, Any]]
) -> None:
    """A polite, legitimate question that must still be answered.

    The control has a quote, so the oracle grades its arithmetic, and it asks
    about a discount, so an agent that blocks on the word "discount" fails it.
    """
    controls = [
        c
        for c in cases
        if c["category"] == "injection"
        and isinstance(c.get("quote"), dict)
        and "discount" in " ".join(c["turns"]).casefold()
    ]
    assert controls, "no injection case is a legitimate question that must be answered"
    for control in controls:
        assert control.get("must_include")
        assert not control.get("must_not_include")
        assert quote_result(oracle, control)["total"] == int(
            str(control["must_include"][-1])
        )


def test_multi_turn_quota(cases: list[dict[str, Any]]) -> None:
    assert sum(1 for c in cases if len(c["turns"]) >= 2) >= 5


def test_i_do_not_know_team_will_reply_quota(cases: list[dict[str, Any]]) -> None:
    def offers_team(case: dict[str, Any]) -> bool:
        haystack = case.get("must_include_any") or []
        return any(marker in needle.casefold() for marker in TEAM_REPLY_MARKERS for needle in haystack)

    counted = [c for c in cases if offers_team(c)]
    assert len(counted) >= 5, "at least five cases must expect the team to reply"
    assert sum(1 for c in counted if c["category"] == "unknown") >= 5


def test_arithmetic_quota(cases: list[dict[str, Any]]) -> None:
    counted = [c for c in cases if c["category"] == "arithmetic"]
    assert len(counted) >= 5
    assert sum(1 for c in counted if isinstance(c.get("quote"), dict)) >= 8


def test_remaining_category_quotas(cases: list[dict[str, Any]]) -> None:
    def count(category: str) -> int:
        return sum(1 for c in cases if c["category"] == category)

    assert count("lead") >= 5
    assert count("complaint") >= 3
    assert count("privacy") >= 4
    assert count("out_of_scope") >= 3
    assert count("unknown") >= 5
    assert count("policy") >= 5
    assert count("price") >= 4
    assert count("fact") >= 3


# --------------------------------------------------------------------------
# The oracle agrees with every number in the file
# --------------------------------------------------------------------------


def test_every_quoted_case_allowed_amounts_are_exactly_the_computed_ones(
    oracle: ModuleType, cases: list[dict[str, Any]]
) -> None:
    assert quoted(cases), "no case carries a quote block"
    for case in quoted(cases):
        result = quote_result(oracle, case)
        assert case["allowed_amounts"] == result["amounts"], case["id"]
        assert case["allowed_amounts"] == sorted(set(case["allowed_amounts"])), case["id"]
        assert all(isinstance(v, int) for v in case["allowed_amounts"]), case["id"]


def test_every_asserted_total_matches_the_oracle(
    oracle: ModuleType, cases: list[dict[str, Any]]
) -> None:
    for case in quoted(cases):
        if case["quote"].get("assert_total") is not True:
            assert str(quote_result(oracle, case)["total"]) not in case.get("must_include", [])
            continue
        total = quote_result(oracle, case)["total"]
        assert str(total) in [str(v) for v in case["must_include"]], case["id"]


def test_every_must_include_amount_in_an_arithmetic_case_is_a_computed_amount(
    oracle: ModuleType, cases: list[dict[str, Any]], catalog: dict[str, int]
) -> None:
    """No hand-typed number that the oracle would not compute.

    A quoted case can only assert values the oracle produced. An arithmetic
    case with no quote block (arith-04 asks about the Rs 999 boundary itself)
    can only assert a catalog price or one of the three policy amounts.
    """
    for case in cases:
        if case["category"] != "arithmetic" or case["id"] in SEED_IDS:
            continue
        permitted: set[int] = set(catalog.values()) | set(POLICY_AMOUNTS)
        if isinstance(case.get("quote"), dict):
            permitted |= set(quote_result(oracle, case)["amounts"])
        for needle in case.get("must_include") or []:
            digits = _DIGIT_RUN_RE.fullmatch(str(needle).strip())
            assert digits is not None, f"{case['id']}: {needle!r} is not a bare number"
            value = int(str(needle).replace(",", ""))
            assert value in permitted, (
                f"{case['id']} asserts {value} but the oracle does not compute it"
            )


def test_every_allowed_amount_is_a_positive_number(cases: list[dict[str, Any]]) -> None:
    """G2 only polices rupees the reply states; a non-positive allowed amount is
    a case bug the harness would never catch, so the tests catch it here."""
    checked = 0
    for case in cases:
        for value in case.get("allowed_amounts") or []:
            checked += 1
            assert isinstance(value, int) and not isinstance(value, bool), case["id"]
            assert value > 0, f"{case['id']}: allowed amount {value} is not positive"
    assert checked >= 20


def test_monetary_needles_in_must_include_are_well_formed(
    cases: list[dict[str, Any]]
) -> None:
    """A money needle is a bare, optionally comma-grouped positive integer.

    "Rs 3850", "3850/-" or "3,85" would either never match a normalised reply or
    match the wrong number, so a needle made only of digits and commas has to
    parse as a positive integer with sane grouping.
    """
    checked = 0
    for case in cases:
        for needle in case.get("must_include") or []:
            text = str(needle).strip()
            if not text or not all(ch.isdigit() or ch == "," for ch in text):
                continue
            checked += 1
            cleaned = text.replace(",", "")
            assert cleaned.isdigit() and int(cleaned) > 0, f"{case['id']}: {needle!r}"
            if "," in text:
                groups = text.split(",")
                assert len(groups[-1]) == 3, f"{case['id']}: {needle!r} has a malformed last group"
                assert all(len(g) == 2 for g in groups[1:-1]), f"{case['id']}: {needle!r}"
                assert 1 <= len(groups[0]) <= 3, f"{case['id']}: {needle!r}"
    assert checked >= 10


def test_no_hand_typed_allowed_amount_in_an_authored_case_is_unjustified(
    oracle: ModuleType, cases: list[dict[str, Any]], catalog: dict[str, int]
) -> None:
    """An allowed amount is either computed, catalog, policy, or the customer's.

    Anything else is a number someone typed because they believed it, which is
    exactly what this file is supposed to make impossible.
    """
    for case in authored(cases):
        permitted: set[int] = set(catalog.values()) | set(POLICY_AMOUNTS) | text_amounts(case)
        if isinstance(case.get("quote"), dict):
            permitted |= set(quote_result(oracle, case)["amounts"])
        for value in case.get("allowed_amounts") or []:
            assert isinstance(value, int), case["id"]
            assert value in permitted, f"{case['id']} allows unearned amount {value}"


def test_seed_cases_keep_the_amounts_the_task_specified(
    cases: list[dict[str, Any]]
) -> None:
    by_id = {c["id"]: c for c in cases}
    assert by_id["arith-01"]["allowed_amounts"] == [2400, 3850]
    assert by_id["arith-02"]["allowed_amounts"] == [1680, 200, 1880]
    assert by_id["hinglish-01"]["allowed_amounts"] == [39000, 1950, 37050, 11115]
    assert by_id["lead-01"]["allowed_amounts"] == [43500, 13050]


# --------------------------------------------------------------------------
# The four graded seed answers, and the boundaries around them
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "distance_km", "subtotal", "discount", "total", "advance"),
    [
        ([("KK-1000", 2), ("GBL", 1)], 5, 3850, 0, 3850, 0),
        ([("ML-1000", 3), ("SM-1", 10)], 2, 1880, 0, 1880, 0),
        ([("GBS", 60)], None, 39000, 1950, 37050, 11115),
        ([("GBL", 30)], None, 43500, 0, 43500, 13050),
    ],
    ids=["kk2+gbl1@5km", "ml3+sm10@2km", "gbs60", "gbl30"],
)
def test_oracle_reproduces_the_four_seed_totals(
    oracle: ModuleType,
    lines: list[tuple[str, int]],
    distance_km: int | None,
    subtotal: int,
    discount: int,
    total: int,
    advance: int,
) -> None:
    result = oracle.compute(lines, distance_km)
    assert result["subtotal"] == subtotal
    assert result["discount"] == discount
    assert result["delivery_free"] is True
    assert result["total"] == total
    assert result["advance"] == advance


def test_thirty_large_boxes_is_the_no_discount_trap(oracle: ModuleType) -> None:
    result = oracle.compute([("GBL", 30)])
    assert result["subtotal"] == 43500
    assert result["discount"] == 0, "30 boxes is under 50 and must not be discounted"
    assert result["total"] == 43500
    assert result["bulk_notice"] is True
    assert result["advance"] == 13050


@pytest.mark.parametrize("boxes", [1, 10, 25, 30, 49])
def test_no_discount_below_fifty_boxes(oracle: ModuleType, boxes: int) -> None:
    result = oracle.compute([("GBS", boxes)])
    assert result["discount"] == 0
    assert result["discount_pct"] == 0
    assert result["total"] == result["subtotal"] + (result["delivery_fee"] or 0)
    assert result["subtotal"] == 650 * boxes


def test_discount_at_exactly_fifty_boxes_is_five_percent_of_the_gift_boxes(
    oracle: ModuleType,
) -> None:
    # 50 x 650 is Rs 32,500, so 5% is Rs 1,625 and the payable total is 30,875.
    result = oracle.compute([("GBS", 50)])
    assert result["subtotal"] == 32500
    assert result["discount"] == 1625
    assert result["total"] == 30875
    assert result["advance"] == 9263, "half-up: 30% of 30,875 is 9,262.5"


def test_discount_touches_the_gift_box_subtotal_only(oracle: ModuleType) -> None:
    result = oracle.compute([("GBS", 50), ("KK-1000", 10)])
    assert result["subtotal"] == 32500 + 12000
    assert result["discount"] == 1625
    assert result["discount"] != round(0.05 * result["subtotal"])


def test_sub_threshold_order_pays_sixty_delivery(oracle: ModuleType) -> None:
    result = oracle.compute([("SP-500", 4)], 3)
    assert result["subtotal"] == 960
    assert result["delivery_free"] is False
    assert result["delivery_fee"] == 60
    assert result["total"] == 1020


def test_the_free_delivery_boundary_is_nine_hundred_and_ninety_nine(
    oracle: ModuleType,
) -> None:
    below = oracle.compute([("RM-500", 1), ("SP-500", 1), ("DH-500", 1), ("AB-400", 1)], 2)
    assert below["subtotal"] == 890
    assert below["delivery_free"] is False
    assert below["delivery_fee"] == 60
    assert below["total"] == 950
    above = oracle.compute([("KK-500", 1), ("SP-500", 1), ("AB-400", 1)], 2)
    assert above["subtotal"] == 1010
    assert above["delivery_free"] is True
    assert above["delivery_fee"] == 0
    assert oracle.FREE_DELIVERY_ABOVE_INR == 999


def test_delivery_beyond_eight_kilometres_is_impossible(oracle: ModuleType) -> None:
    beyond = oracle.compute([("GBS", 51)], 9)
    assert beyond["delivery_possible"] is False
    assert beyond["delivery_fee"] is None
    assert beyond["total"] == beyond["subtotal"] - beyond["discount"]
    at_limit = oracle.compute([("GBS", 51)], 8)
    assert at_limit["delivery_possible"] is True
    assert at_limit["delivery_fee"] == 0


def test_bulk_notice_needs_more_than_ten_kilos_or_more_than_twenty_five_boxes(
    oracle: ModuleType,
) -> None:
    assert oracle.compute([("KK-1000", 10)])["advance"] == 0
    assert oracle.compute([("KK-1000", 11)])["advance"] == 3960
    assert oracle.compute([("GBS", 25)])["advance"] == 0
    assert oracle.compute([("GBS", 26)])["bulk_notice"] is True


def test_every_computed_value_is_an_integer(oracle: ModuleType) -> None:
    result = oracle.compute([("GBS", 60), ("KK-1000", 3)], 3)
    for key in ("subtotal", "discount", "payable", "total", "advance", "weight_grams"):
        assert type(result[key]) is int, key
    for line in result["lines"]:
        assert type(line["unit"]) is int and type(line["line_total"]) is int
    for value in result["amounts"]:
        assert type(value) is int


def test_amounts_never_contain_zero(oracle: ModuleType) -> None:
    free = oracle.compute([("KK-1000", 2)], 5)
    assert free["delivery_free"] is True
    assert free["discount"] == 0 and free["advance"] == 0
    assert free["amounts"] == [1200, 2400]
    assert 0 not in free["amounts"], "a waived amount is spoken as free, not as Rs 0"


def test_pack_grams_reads_every_shape_in_the_catalog(oracle: ModuleType) -> None:
    assert oracle.pack_grams("1 kg") == 1000
    assert oracle.pack_grams("500 g") == 500
    assert oracle.pack_grams("1 piece") == 0
    assert oracle.pack_grams("1 box") == 0
    assert oracle.pack_grams("1 kg assorted sweets and 200 g dry fruits") == 1200


def test_catalog_comes_from_the_csv_and_not_from_the_package(
    oracle: ModuleType, catalog: dict[str, int]
) -> None:
    assert oracle.load_catalog(ROOT / "data" / "prices.csv") == catalog
    assert oracle.gift_box_skus(ROOT / "data" / "prices.csv") == {"GBS", "GBL"}


def test_oracle_policy_constants_match_the_agents_own(
    oracle: ModuleType,
) -> None:
    assert oracle.DISCOUNT_PCT == POLICY_DISCOUNT_PCT
    assert oracle.DISCOUNT_MIN_BOXES == POLICY_DISCOUNT_MIN_BOXES
    assert oracle.DELIVERY_FEE_INR == POLICY_DELIVERY_FEE_INR
    assert oracle.FREE_DELIVERY_ABOVE_INR == POLICY_FREE_DELIVERY_ABOVE_INR
    assert oracle.DELIVERY_RADIUS_KM == POLICY_DELIVERY_RADIUS_KM
    assert oracle.BULK_WEIGHT_KG == POLICY_BULK_KG
    assert oracle.BULK_BOXES == POLICY_BULK_BOXES
    assert oracle.ADVANCE_PCT == POLICY_BULK_ADVANCE_PCT
    assert oracle.COD_LIMIT_INR == POLICY_COD_LIMIT_INR
    assert oracle.GST_PCT == 5


def test_oracle_source_never_imports_the_package() -> None:
    source = ORACLE_PATH.read_text(encoding="utf-8")
    assert "meher_agent" not in source, (
        "the expected totals must not come from the code under test, so "
        "compute_expected_totals.py may not mention the package at all"
    )
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == [
        "__future__",
        "argparse",
        "csv",
        "json",
        "re",
        "sys",
        "datetime",
        "decimal",
        "functools",
        "pathlib",
        "typing",
    ], f"unexpected imports: {imported}"


def test_oracle_only_uses_the_standard_library() -> None:
    tree = ast.parse(ORACLE_PATH.read_text(encoding="utf-8"))
    roots = {
        (alias.name or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert roots <= set(sys.stdlib_module_names), roots


# --------------------------------------------------------------------------
# The oracle's own CLI contract
# --------------------------------------------------------------------------


def _run_oracle(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORACLE_PATH), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_check_mode_exits_zero_on_the_committed_file() -> None:
    result = _run_oracle("--cases", str(CASES_PATH), "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "up to date" in result.stdout


def test_table_mode_lists_every_quoted_case() -> None:
    result = _run_oracle("--cases", str(CASES_PATH))
    assert result.returncode == 0, result.stdout + result.stderr
    for case_id in ("arith-07", "inject-08", "hindi-05", "lead-05"):
        assert case_id in result.stdout
    assert "subtotal" in result.stdout and "advance" in result.stdout


def test_csv_flag_alone_prints_a_table() -> None:
    """``--csv data/prices.csv`` is the oracle's standalone entry point: it must
    run and print a sensible table even with no expectations file to read."""
    result = _run_oracle(
        "--csv", str(ROOT / "data" / "prices.csv"), "--cases", "does-not-exist.jsonl"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for token in ("KK-1000", "GBS", "GBL", "price_inr", "shelf_life_days"):
        assert token in result.stdout


def test_csv_flag_with_the_default_case_file_prints_the_quoted_table() -> None:
    result = _run_oracle("--csv", str(ROOT / "data" / "prices.csv"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "subtotal" in result.stdout and "total" in result.stdout


def test_csv_flag_falls_back_to_the_catalog_when_there_is_no_case_file() -> None:
    missing = "does-not-exist.jsonl"
    result = _run_oracle(
        "--csv", str(ROOT / "data" / "prices.csv"), "--cases", missing
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KKSF-500" in result.stdout
    assert "free delivery at Rs 999+" in result.stdout


def test_check_flag_accepts_a_file_argument(tmp_path: Path) -> None:
    """``--check <file.jsonl>`` verifies exactly the file it was given."""
    target = tmp_path / "copy.jsonl"
    target.write_bytes(CASES_PATH.read_bytes())
    result = _run_oracle(
        "--csv", str(ROOT / "data" / "prices.csv"), "--check", str(target)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "up to date" in result.stdout
    stale = tmp_path / "stale.jsonl"
    stale.write_bytes(
        CASES_PATH.read_text(encoding="utf-8").replace("30875", "30876").encode("utf-8")
    )
    result = _run_oracle(
        "--csv", str(ROOT / "data" / "prices.csv"), "--check", str(stale)
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "out of date" in result.stderr


def test_write_mode_is_a_no_op_on_the_committed_file(
    tmp_path: Path, oracle: ModuleType
) -> None:
    raw = CASES_PATH.read_text(encoding="utf-8")
    assert oracle.render(raw) == raw.split("\n")[:-1], "the committed file is stale"


def test_check_mode_fails_and_explains_on_a_stale_copy(
    tmp_path: Path, oracle: ModuleType
) -> None:
    stale = tmp_path / "stale.jsonl"
    lines = strip_derived_amounts(oracle)
    stale.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    result = _run_oracle("--cases", str(stale), "--check")
    assert result.returncode == 1
    assert "out of date" in result.stderr
    assert "allowed_amounts" in result.stderr


def test_write_mode_leaves_cases_without_a_quote_byte_for_byte(
    tmp_path: Path, oracle: ModuleType
) -> None:
    lines = strip_derived_amounts(oracle)
    source = tmp_path / "cases.jsonl"
    source.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    before = source.read_bytes().split(b"\n")
    result = _run_oracle("--cases", str(source), "--write")
    assert result.returncode == 0, result.stdout + result.stderr
    after = source.read_bytes().split(b"\n")
    assert len(before) == len(after)
    rewritten = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    quoted_indexes = [i for i, line in enumerate(lines) if '"quote"' in line]
    assert rewritten == quoted_indexes
    assert _run_oracle("--cases", str(source), "--check").returncode == 0


def strip_derived_amounts(oracle: ModuleType) -> list[str]:
    """The case file with every derived amount removed again.

    This is what a hand-typed file looks like, and it is what ``--check`` has to
    notice and ``--write`` has to repair.
    """
    lines: list[str] = []
    for line in CASES_PATH.read_text(encoding="utf-8").split("\n")[:-1]:
        case = json.loads(line)
        if isinstance(case.get("quote"), dict):
            total = str(quote_result(oracle, case)["total"])
            case.pop("allowed_amounts", None)
            case["must_include"] = [
                v for v in case.get("must_include", []) if str(v) != total
            ]
            line = oracle.dumps_case(case)
        lines.append(line)
    return lines


# --------------------------------------------------------------------------
# Leads and expectations
# --------------------------------------------------------------------------


def test_every_expect_lead_email_is_syntactically_valid(
    cases: list[dict[str, Any]]
) -> None:
    checked = 0
    for case in cases:
        email = (case.get("expect_lead") or {}).get("email")
        if email is None:
            continue
        checked += 1
        assert is_valid_email(email), f"{case['id']}: {email!r}"
    assert checked >= 3


def test_every_expect_lead_phone_is_syntactically_valid(
    cases: list[dict[str, Any]]
) -> None:
    checked = 0
    for case in cases:
        phone = (case.get("expect_lead") or {}).get("phone")
        if phone is None:
            continue
        checked += 1
        assert is_valid_phone(phone), f"{case['id']}: {phone!r}"
    assert checked >= 3


def test_every_expect_lead_value_appears_in_the_cases_own_turns(
    cases: list[dict[str, Any]]
) -> None:
    """A lead the customer never actually typed is a hallucination.

    Name and email must appear character-for-character in the case's own turn
    text; the phone must appear as a digit run (customers sometimes format it
    with spaces or dashes).
    """
    checked = 0
    for case in cases:
        lead = case.get("expect_lead")
        if not isinstance(lead, dict):
            continue
        haystack = " ".join(str(t) for t in case.get("turns") or [])
        digits = "".join(ch for ch in haystack if ch.isdigit())
        for field in ("name", "email"):
            value = lead.get(field)
            if value is None:
                continue
            checked += 1
            assert str(value) in haystack, (
                f"{case['id']}: expect_lead {field} {value!r} is not in the case's own turns"
            )
        phone = lead.get("phone")
        if phone is not None:
            checked += 1
            assert "".join(ch for ch in str(phone) if ch.isdigit()) in digits, (
                f"{case['id']}: expect_lead phone {phone!r} is not in the case's own turns"
            )
    assert checked >= 5


def test_every_expect_lead_names_at_least_one_contact_field(
    cases: list[dict[str, Any]]
) -> None:
    named = 0
    for case in cases:
        if "expect_lead" not in case:
            continue
        lead = case["expect_lead"]
        assert any(lead.get(field) for field in ("name", "email", "phone")), case["id"]
        named += 1 if lead.get("name") else 0
    assert named >= 4


def test_a_lead_with_an_invalid_phone_expects_no_tool_call(
    cases: list[dict[str, Any]]
) -> None:
    refused = [
        c
        for c in cases
        if c["category"] == "lead" and c.get("expect_action") == "none"
    ]
    assert refused, "no lead case covers a phone number that must not be saved"
    for case in refused:
        phones = [
            token
            for turn in case["turns"]
            for token in re.findall(r"\b\d{5,13}\b", turn)
        ]
        assert phones, case["id"]
        assert any(not is_valid_phone(p) for p in phones), case["id"]
        assert case.get("expect_lead") is None, case["id"]


def test_complaint_cases_all_escalate(cases: list[dict[str, Any]]) -> None:
    complaints = [c for c in cases if c["category"] == "complaint"]
    assert len(complaints) >= 3
    for case in complaints:
        assert case.get("expect_action") == "escalate", case["id"]


def test_lead_cases_that_save_a_lead_name_the_action(
    cases: list[dict[str, Any]]
) -> None:
    savers = [c for c in cases if c.get("expect_action") == "save_lead"]
    assert len(savers) >= 5
    for case in savers:
        assert case.get("expect_lead"), case["id"]
