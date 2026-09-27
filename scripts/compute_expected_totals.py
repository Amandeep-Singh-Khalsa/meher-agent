"""Independent oracle for the expected totals quoted in evals/cases.jsonl.

Why this file is a re-implementation
------------------------------------
``evals/checks.py`` grades every rupee amount a reply states against the case's
``allowed_amounts``, and this script is what puts those numbers there.  If they
came from the billing engine under test, the harness would be checking the agent
against itself and a regression in that engine would silently move the goalposts.
So this module reads data/prices.csv with the stdlib ``csv`` module, applies
data/policies.md directly, and imports nothing from the agent package: the only
assertion needed is the test in tests/test_cases.py that greps this file's source
and fails on any import of the package.

The policy constants below are transcribed by hand from data/policies.md.  The
arithmetic that uses them is written here and nowhere else.  tests/test_cases.py
cross-checks the constants against the agent's own ``POLICY_*`` values, which
keeps a policy edit from silently making the two disagree.

    Usage
    -----
    python scripts/compute_expected_totals.py --csv data/prices.csv
    python scripts/compute_expected_totals.py --cases evals/cases.jsonl --write
    python scripts/compute_expected_totals.py --cases evals/cases.jsonl --check
    python scripts/compute_expected_totals.py --csv data/prices.csv --check evals/cases.jsonl
    python scripts/compute_expected_totals.py --cases evals/cases.jsonl

    ``--csv`` is the prices file (``--prices`` is accepted as an alias).  With no
    ``--cases`` file to read, the oracle prints the catalog table it computed
    straight from the CSV.  ``--check`` takes an optional FILE: with one, only
    that file is verified; without one, the ``--cases`` file is.

``--write`` recomputes every case that carries a ``quote`` block and writes the
derived numbers back: ``allowed_amounts`` becomes exactly the oracle's
``amounts``, and ``assert_total: true`` pushes the total into ``must_include``.
A case without a ``quote`` block is copied through byte for byte, which is what
keeps the thirteen seed lines identical to evals/seed_cases.jsonl.

``--check`` is the same computation without the write: it exits 0 only when the
file on disk is exactly what the oracle would produce.

The ``quote`` convention
------------------------
    "quote": {"lines": [{"sku": "KK-1000", "qty": 2},
                        {"sku": "GBL", "qty": 1}],
              "distance_km": 5, "assert_total": true}

``distance_km`` may be omitted (or null) for "no distance was stated", which
still gets a delivery fee; ``assert_total`` is what turns the computed total into
a hard ``must_include`` requirement.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "REPO_ROOT",
    "load_catalog",
    "pack_grams",
    "compute",
    "round_half_up",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "cases.jsonl"
DEFAULT_PRICES_PATH = REPO_ROOT / "data" / "prices.csv"

# -- data/policies.md, transcribed ------------------------------------------
DISCOUNT_PCT = 5
DISCOUNT_MIN_BOXES = 50
DELIVERY_FEE_INR = 60
FREE_DELIVERY_ABOVE_INR = 999
DELIVERY_RADIUS_KM = 8
BULK_WEIGHT_KG = 10
BULK_BOXES = 25
ADVANCE_PCT = 30
COD_LIMIT_INR = 5000
GST_PCT = 5
GIFT_BOX_TYPE = "gift box"
PREORDER_LAST_DAY = date(2026, 11, 5)

# -- the shape of one cases.jsonl record ------------------------------------
#: Serialisation order.  It is not alphabetical: it is the order that reproduces
#: the thirteen seed lines byte for byte, so a run of --write can never
#: restyle a line it did not have to touch.
KEY_ORDER: tuple[str, ...] = (
    "id",
    "category",
    "turns",
    "must_include",
    "must_not_include",
    "expect_action",
    "expect_lead",
    "must_include_any",
    "allowed_amounts",
    "quote",
    "note",
)
SEPARATORS = (", ", ": ")

_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g)\b", re.IGNORECASE)
_GRAM_MULTIPLIER = {"kg": 1000, "g": 1}


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_rows(csv_path: str = "") -> tuple[dict[str, str], ...]:
    """Every prices.csv row, in file order, as plain strings."""
    path = Path(csv_path) if csv_path else DEFAULT_PRICES_PATH
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return tuple(
            {(k or "").strip(): (v or "").strip() for k, v in row.items() if k is not None}
            for row in rows
            if (row.get("sku") or "").strip()
        )


def load_catalog(csv_path: Path | str = DEFAULT_PRICES_PATH) -> dict[str, int]:
    """sku -> price_inr, straight from the CSV and nothing else."""
    return {row["sku"]: int(row["price_inr"]) for row in load_rows(str(csv_path))}


def pack_for(csv_path: Path | str = DEFAULT_PRICES_PATH) -> dict[str, str]:
    return {row["sku"]: row["pack"] for row in load_rows(str(csv_path))}


def gift_box_skus(csv_path: Path | str = DEFAULT_PRICES_PATH) -> frozenset[str]:
    """Whichever SKUs prices.csv marks as ``type = "gift box"`` (GBS and GBL).

    Read from the data rather than hardcoded, so a future gift box added to
    prices.csv is discounted by this oracle without a code change.
    """
    return frozenset(
        row["sku"]
        for row in load_rows(str(csv_path))
        if row.get("type", "").strip().lower() == GIFT_BOX_TYPE
    )


def pack_grams(pack: str) -> int:
    """Grams in a pack description, summing compound ones.

    "1 kg" -> 1000, "500 g" -> 500, "1 box" -> 0, "1 piece" -> 0,
    "1 kg assorted sweets and 200 g dry fruits" -> 1200.
    """
    total = 0
    for value, unit in _WEIGHT_RE.findall(pack or ""):
        grams = Decimal(value) * _GRAM_MULTIPLIER[unit.lower()]
        total += int(grams)
    return total


def round_half_up(numerator: int, denominator: int) -> int:
    """floor(numerator / denominator + 0.5) with integer maths only.

    Half-up rather than banker's rounding, so 5% of an odd gift-box subtotal is
    the same rupee on every platform and in every language runtime.
    """
    return (2 * numerator + denominator) // (2 * denominator)


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


def compute(
    lines: Sequence[tuple[str, int]],
    distance_km: int | None = None,
    *,
    today: date | None = None,
    prices_path: Path | str = DEFAULT_PRICES_PATH,
) -> dict[str, Any]:
    """Every derived rupee value for one order, as integers.

    ``lines`` is ``[(sku, qty), ...]``; ``distance_km`` is the delivery distance
    or ``None`` when the customer never said.  Rules, all from data/policies.md:
    line totals are ``qty * price``; the 5% discount applies to the gift-box
    subtotal only and only from 50 boxes; delivery is free from a payable total
    of 999, costs 60 below that and is impossible beyond 8 km; an order over
    10 kg or over 25 gift boxes needs 3 days' notice and a 30% advance; GST is
    already inside every price and is never added.
    """
    prices = load_catalog(prices_path)
    packs = pack_for(prices_path)
    gift_skus = gift_box_skus(prices_path)

    computed_lines: list[dict[str, Any]] = []
    subtotal = 0
    gift_qty = 0
    gift_subtotal = 0
    weight_grams = 0
    for raw_sku, raw_qty in lines:
        sku = str(raw_sku).strip()
        if sku not in prices:
            raise KeyError(f"unknown sku {sku!r}; prices.csv has {sorted(prices)}")
        qty = int(raw_qty)
        if qty < 0:
            raise ValueError(f"negative quantity for {sku}: {qty}")
        unit_price = prices[sku]
        line_total = qty * unit_price
        subtotal += line_total
        weight_grams += pack_grams(packs[sku]) * qty
        if sku in gift_skus:
            gift_qty += qty
            gift_subtotal += line_total
        computed_lines.append(
            {"sku": sku, "qty": qty, "unit": unit_price, "line_total": line_total}
        )

    discount_pct = 0
    discount = 0
    if gift_qty >= DISCOUNT_MIN_BOXES and gift_subtotal > 0:
        discount_pct = DISCOUNT_PCT
        discount = round_half_up(gift_subtotal * DISCOUNT_PCT, 100)
    payable = subtotal - discount

    if not computed_lines:
        delivery_fee: int | None = None
        delivery_free = False
        delivery_possible = True
    elif distance_km is not None and distance_km > DELIVERY_RADIUS_KM:
        delivery_fee = None
        delivery_free = False
        delivery_possible = False
    elif payable >= FREE_DELIVERY_ABOVE_INR:
        delivery_fee = 0
        delivery_free = True
        delivery_possible = True
    else:
        delivery_fee = DELIVERY_FEE_INR
        delivery_free = False
        delivery_possible = True

    total = payable + (delivery_fee or 0)

    bulk_notice = (
        weight_grams > BULK_WEIGHT_KG * 1000 or gift_qty > BULK_BOXES
    )
    advance_pct = ADVANCE_PCT if bulk_notice else 0
    advance = round_half_up(total * advance_pct, 100) if bulk_notice else 0

    preorder_open = (today <= PREORDER_LAST_DAY) if today is not None else None

    # Zero is dropped on purpose: a free delivery, a waived discount and a
    # zero advance are spoken as "free", "none" and "no advance", never as
    # "Rs 0", and the seed cases' allowed_amounts lists carry no 0 either.
    amounts = sorted(
        {
            value
            for value in (
                *(line["unit"] for line in computed_lines),
                *(line["line_total"] for line in computed_lines),
                subtotal,
                discount,
                delivery_fee,
                total,
                advance,
            )
            if value
        }
    )

    notes: list[str] = []
    if delivery_possible and delivery_free:
        notes.append(f"delivery free: payable {payable} >= {FREE_DELIVERY_ABOVE_INR}")
    elif delivery_fee is not None:
        notes.append(f"delivery {DELIVERY_FEE_INR}: payable {payable} < {FREE_DELIVERY_ABOVE_INR}")
    if not delivery_possible:
        notes.append(f"no delivery beyond {DELIVERY_RADIUS_KM} km; pickup or own courier")
    if gift_qty and not discount:
        notes.append(
            f"no discount: {gift_qty} gift box(es) is below the {DISCOUNT_MIN_BOXES}-box "
            f"minimum"
        )
    if bulk_notice:
        notes.append(f"bulk: {ADVANCE_PCT}% advance of {advance} and 3 days' notice")
    if total > COD_LIMIT_INR:
        notes.append(f"cash on delivery unavailable above {COD_LIMIT_INR}")
    if preorder_open is False:
        notes.append(f"gift-box pre-orders closed on {PREORDER_LAST_DAY.isoformat()}")

    return {
        "lines": computed_lines,
        "subtotal": subtotal,
        "discount": discount,
        "discount_pct": discount_pct,
        "payable": payable,
        "delivery_fee": delivery_fee,
        "delivery_free": delivery_free,
        "delivery_possible": delivery_possible,
        "total": total,
        "advance": advance,
        "advance_pct": advance_pct,
        "bulk_notice": bulk_notice,
        "gift_qty": gift_qty,
        "gift_subtotal": gift_subtotal,
        "weight_grams": weight_grams,
        "preorder_open": preorder_open,
        "amounts": amounts,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# The case file
# --------------------------------------------------------------------------


class CaseFileError(RuntimeError):
    """The case file cannot be read, parsed, or made self-consistent."""


def dumps_case(case: dict[str, Any]) -> str:
    """One line, one compact JSON object, keys in KEY_ORDER, LF-friendly."""
    ordered: dict[str, Any] = {key: case[key] for key in KEY_ORDER if key in case}
    for key in case:
        if key not in ordered:
            ordered[key] = case[key]
    return json.dumps(ordered, ensure_ascii=False, separators=SEPARATORS)


def quote_lines(case: dict[str, Any]) -> list[tuple[str, int]] | None:
    """The ``(sku, qty)`` pairs of a case's quote block, or None if it has none."""
    quote = case.get("quote")
    if not isinstance(quote, dict):
        return None
    lines: list[tuple[str, int]] = []
    for entry in quote.get("lines") or []:
        if not isinstance(entry, dict) or "sku" not in entry:
            raise CaseFileError(f"case {case.get('id')!r} has a malformed quote line {entry!r}")
        lines.append((str(entry["sku"]), int(entry["qty"])))
    return lines


def expected_for(
    case: dict[str, Any], prices_path: Path | str = DEFAULT_PRICES_PATH
) -> dict[str, Any] | None:
    """The oracle's result for a case, or None when the case is not quoted."""
    lines = quote_lines(case)
    if lines is None:
        return None
    quote = case["quote"]
    distance = quote.get("distance_km")
    return compute(
        lines,
        None if distance is None else int(distance),
        prices_path=prices_path,
    )


def apply_quote(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Return the case with the oracle's numbers written into it."""
    updated = dict(case)
    updated["allowed_amounts"] = list(result["amounts"])
    quote = updated.get("quote") or {}
    if quote.get("assert_total") is True:
        total = str(result["total"])
        must_include = [str(v) for v in updated.get("must_include") or []]
        if total not in must_include:
            must_include.append(total)
        updated["must_include"] = must_include
    return updated


def split_lines(raw: str) -> list[str]:
    """The file's lines, without the single trailing empty element."""
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def render(raw: str, prices_path: Path | str = DEFAULT_PRICES_PATH) -> list[str]:
    """What the file should contain, given what it currently contains.

    A line that is blank, unparseable, or a case without a ``quote`` block is
    returned unchanged, so the rest of the file is never restyled.
    """
    out: list[str] = []
    for line in split_lines(raw):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            out.append(line)
            continue
        try:
            case = json.loads(stripped)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if not isinstance(case, dict):
            out.append(line)
            continue
        result = expected_for(case, prices_path)
        if result is None:
            out.append(line)
            continue
        out.append(dumps_case(apply_quote(case, result)))
    return out


def format_issues(raw: str) -> list[str]:
    """Whole-file problems -- the ones a per-case diff cannot see."""
    issues: list[str] = []
    if b"\r\n" in raw.encode("utf-8", "surrogateescape"):
        issues.append("file contains CRLF line endings; evals/*.jsonl must be LF")
    if raw and not raw.endswith("\n"):
        issues.append("file does not end with a newline")
    return issues


def case_diff(stored: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """A human-readable summary of what --write would change in one case."""
    diffs: list[str] = []
    stored_amounts = set(stored.get("allowed_amounts") or [])
    expected_amounts = set(expected.get("allowed_amounts") or [])
    if added := sorted(expected_amounts - stored_amounts):
        diffs.append(f"allowed_amounts missing {added}")
    if removed := sorted(stored_amounts - expected_amounts):
        diffs.append(f"allowed_amounts has stale {removed}")
    quote = stored.get("quote")
    if isinstance(quote, dict) and quote.get("assert_total") is True:
        total = str(expected.get("total"))
        if total not in {str(v) for v in stored.get("must_include") or []}:
            diffs.append(f"must_include missing the asserted total {total}")
    return diffs


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError as exc:
        raise CaseFileError(f"case file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise CaseFileError(f"case file is not valid UTF-8: {path}: {exc}") from exc


def mode_write(cases_path: Path, prices_path: Path) -> int:
    raw = read_text(cases_path)
    rendered = render(raw, prices_path)
    payload = "\n".join(rendered) + "\n"
    changed = [i for i, (old, new) in enumerate(zip(split_lines(raw), rendered), 1) if old != new]
    if len(split_lines(raw)) != len(rendered):
        changed.append(0)
    if payload != raw:
        cases_path.write_bytes(payload.encode("utf-8"))
    quoted = sum(1 for line in rendered if '"quote"' in line)
    print(f"wrote {cases_path} ({len(rendered)} cases, {quoted} quoted)")
    print(f"lines rewritten: {len(changed)}")
    return 0


def mode_check(cases_path: Path, prices_path: Path) -> int:
    raw = read_text(cases_path)
    current = split_lines(raw)
    rendered = render(raw, prices_path)
    problems: list[str] = list(format_issues(raw))

    if len(current) != len(rendered):
        problems.append(
            f"line count changed: file has {len(current)}, oracle expects {len(rendered)}"
        )

    quoted = 0
    stale = 0
    for index, (old, new) in enumerate(zip(current, rendered), start=1):
        if new == old:
            quoted += 1 if '"quote"' in new else 0
            continue
        try:
            stored = json.loads(old)
        except json.JSONDecodeError:
            problems.append(f"line {index}: not valid JSON, so --write would rewrite it")
            stale += 1
            continue
        if not isinstance(stored, dict) or not isinstance(stored.get("quote"), dict):
            problems.append(
                f"line {index}: case {stored.get('id', '?')!r} is not in canonical form"
            )
            stale += 1
            continue
        result = expected_for(stored, prices_path)
        if result is None:
            continue
        quoted += 1
        diffs = case_diff(stored, apply_quote(stored, result))
        if not diffs:
            diffs = ["serialisation is not canonical (key order or spacing)"]
        problems.append(f"line {index}: case {stored.get('id', '?')} is out of date: {'; '.join(diffs)}")
        stale += 1

    if not problems:
        print(f"ok: {cases_path} is up to date ({len(current)} cases, {quoted} quoted)")
        return 0
    print(f"FAIL: {cases_path} is out of date ({stale} case(s) need rewriting)", file=sys.stderr)
    for problem in problems[:40]:
        print(f"  - {problem}", file=sys.stderr)
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
    print(
        f"  run: python scripts/compute_expected_totals.py --cases {cases_path} --write",
        file=sys.stderr,
    )
    return 1


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def mode_catalog(prices_path: Path) -> int:
    """A small table of what the oracle read out of the CSV and nothing else.

    This is the ``--csv data/prices.csv`` mode: with no expectations file to
    verify or tabulate, the oracle still shows its work - every SKU, pack,
    price, type and shelf life, plus which SKUs earn the gift-box discount.
    """
    rows = load_rows(str(prices_path))
    if not rows:
        print(f"error: no usable rows in {prices_path}", file=sys.stderr)
        return 2
    table: list[tuple[str, ...]] = [
        ("sku", "item", "pack", "price_inr", "type", "shelf_life_days")
    ]
    for row in rows:
        table.append(
            (
                row.get("sku", ""),
                row.get("item", ""),
                row.get("pack", ""),
                row.get("price_inr", ""),
                row.get("type", ""),
                row.get("shelf_life_days", ""),
            )
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    for position, row in enumerate(table):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if position == 0:
            print("  ".join("-" * width for width in widths))
    gift = ", ".join(sorted(gift_box_skus(prices_path))) or "none"
    print(f"\n{len(rows)} sku(s) in {prices_path}")
    print(f"gift boxes (the 5% discount applies to these only): {gift}")
    print(f"free delivery at Rs {FREE_DELIVERY_ABOVE_INR}+, else Rs {DELIVERY_FEE_INR}; "
          f"delivery radius {DELIVERY_RADIUS_KM} km")
    return 0


def mode_table(cases_path: Path, prices_path: Path) -> int:
    raw = read_text(cases_path)
    quoted: list[tuple[str, int | None, dict[str, Any]]] = []
    for index, line in enumerate(split_lines(raw), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        case = json.loads(stripped)
        result = expected_for(case, prices_path)
        if result is None:
            continue
        distance = (case.get("quote") or {}).get("distance_km")
        quoted.append((str(case.get("id", f"line-{index}")), distance, result))
    if not quoted:
        print(f"no quoted cases in {cases_path}")
        return 0

    table: list[tuple[str, ...]] = [
        ("case", "order", "km", "subtotal", "discount", "delivery", "total", "advance", "bulk")
    ]
    for case_id, distance, result in quoted:
        order = " + ".join(f"{line['qty']}x{line['sku']}" for line in result["lines"])
        table.append(
            (
                case_id,
                order,
                _fmt(distance),
                _fmt(result["subtotal"]),
                _fmt(result["discount"]),
                "free" if result["delivery_free"] else _fmt(result["delivery_fee"]),
                _fmt(result["total"]),
                _fmt(result["advance"]),
                _fmt(result["bulk_notice"]),
            )
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    for position, row in enumerate(table):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if position == 0:
            print("  ".join("-" * width for width in widths))
    print(f"\n{len(quoted)} quoted case(s) in {cases_path}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compute_expected_totals.py",
        description=(
            "Independent oracle for evals/cases.jsonl: writes the expected rupee "
            "totals back into every case that has a 'quote' block, checks them, or "
            "prints them as a table."
        ),
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="path to the JSONL case file (default: evals/cases.jsonl)",
    )
    parser.add_argument(
        "--csv",
        "--prices",
        dest="prices",
        default=str(DEFAULT_PRICES_PATH),
        help="path to prices.csv (default: data/prices.csv)",
    )
    parser.add_argument(
        "--write", action="store_true", help="write the computed totals into the case file"
    )
    parser.add_argument(
        "--check",
        nargs="?",
        const=True,
        default=None,
        metavar="FILE",
        help=(
            "exit non-zero if the expectations file is out of date, and print what "
            "differs; with no FILE the --cases file is checked, with one only that file is"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases_path = Path(args.cases)
    prices_path = Path(args.prices)
    if not prices_path.is_file():
        print(f"error: prices file not found: {prices_path}", file=sys.stderr)
        return 2
    try:
        if args.write and args.check:
            print("error: choose --write or --check, not both", file=sys.stderr)
            return 2
        if args.write:
            return mode_write(cases_path, prices_path)
        if args.check:
            target = cases_path if args.check is True else Path(str(args.check))
            return mode_check(target, prices_path)
        if not cases_path.is_file():
            return mode_catalog(prices_path)
        return mode_table(cases_path, prices_path)
    except CaseFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
