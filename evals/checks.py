"""Per-case checks for the Meher Sweets evaluation harness.

Semantics fixed by the task spec and implemented literally:

* every reply-level check runs against the reply to the **last** turn;
* matching is **case-insensitive** and thousands separators (including Indian
  grouping, so `1,00,000` -> `100000`) are stripped from numbers in the reply
  *before* comparison, via ``meher_agent.grounding.amounts.normalise_for_match``
  so the harness and the reply guard can never disagree;
* G1 the first reply of the conversation contains the word "AI";
* G2 every rupee amount in the checked reply is allowed
  (catalog prices + policy amounts 60/999/5000 + the case's ``allowed_amounts``);
* G3 the reply is 1..1200 characters;
* G4 for fact/price/arithmetic/policy/hindi/hinglish the sources are non-empty
  and every id exists in ``data/``.

Design rules for a harness that has to survive hidden cases:

1. Nothing raises. A malformed case produces a *failed* or *skipped*
   ``CheckOutcome`` with a detail explaining why, never an exception.
2. A check that does not apply is reported as ``skipped=True`` with
   ``passed=True``, so a caller that only looks at ``passed`` still works while
   a caller that wants correct denominators can exclude it.
3. A failed request (``CaseResult.error``) fails every applicable check with the
   error in the detail, instead of the checks silently passing on an empty reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence, TypeVar

from meher_agent.config import (
    POLICY_COD_LIMIT_INR,
    POLICY_DELIVERY_FEE_INR,
    POLICY_FREE_DELIVERY_ABOVE_INR,
)
from meher_agent.data.corpus import load_corpus
from meher_agent.grounding.amounts import (
    extract_rupee_amounts,
    normalise_for_match,
    to_int,
)
from meher_agent.tools.validation import (
    normalise_date,
    normalise_email,
    normalise_phone,
)

__all__ = [
    "CheckOutcome",
    "CaseResult",
    "RunSummary",
    "check_case",
    "case_passed",
    "failed_checks",
    "normalise",
    "allowed_rupee_amounts",
    "valid_source_ids",
    "G4_CATEGORIES",
    "MAX_REPLY_CHARS",
    "MIN_REPLY_CHARS",
    "POLICY_AMOUNTS",
]

T = TypeVar("T")

#: G3 bounds, fixed by the task spec.
MIN_REPLY_CHARS = 1
MAX_REPLY_CHARS = 1200

#: G4 only applies to these categories; everywhere else citing is optional.
G4_CATEGORIES: frozenset[str] = frozenset(
    {"fact", "price", "arithmetic", "policy", "hindi", "hinglish"}
)

#: The three money amounts written into data/policies.md. Part of the shop's
#: data rather than a tunable, so a reply may always state them.
POLICY_AMOUNTS: frozenset[int] = frozenset(
    {POLICY_DELIVERY_FEE_INR, POLICY_FREE_DELIVERY_ABOVE_INR, POLICY_COD_LIMIT_INR}
)
assert POLICY_AMOUNTS == frozenset({60, 999, 5000})

#: Action expectations the harness understands. Anything else is reported as
#: an unrecognised expectation rather than silently treated as a pass.
_NO_ACTION_ALIASES = frozenset(
    {"none", "no", "null", "nothing", "no_action", "no-tool", "no_tool", "notool"}
)
_ACTION_NAMES = frozenset({"save_lead", "escalate"})

#: Args whose *value* being true means "the loop did this, not the model".
_AUTO_FLAG_KEYS = ("auto", "automatic", "implicit", "is_implicit", "system")
#: Args whose value has to *mention* a step limit before an escalate is treated
#: as loop-generated. A model that writes "reason": "step limit exhausted" is
#: describing the same event, so matching on the text is safe.
_REASON_KEYS = ("reason", "note", "cause", "trigger", "triggered_by", "source", "origin")
_STEP_LIMIT_MARKERS = (
    "step_limit",
    "step-limit",
    "step limit",
    "max_steps",
    "max-steps",
    "max steps",
    "step_exhausted",
    "steps exhausted",
)

_WS_RE = re.compile(r"\s+")
_AI_RE = re.compile(r"\bai\b", re.IGNORECASE)
_EXCERPT = 160


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class CheckOutcome:
    """One graded assertion about one case.

    ``skipped`` is not part of the three graded attributes; it exists so the
    report can keep "not applicable" checks out of its denominators.
    """

    name: str
    passed: bool
    detail: str
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "skipped": self.skipped,
            "detail": self.detail,
        }


@dataclass
class CaseResult:
    """Everything the runner observed while replaying one case."""

    case_id: str
    category: str
    turns: list[str]
    replies: list[str]
    sources: list[list[str]]
    actions: list[list[dict]]
    handoff: list[bool]
    latencies_s: list[float]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    usage_source: str
    error: str | None = None
    #: Leads as the runner saw them (from GET /leads, when it collected them).
    leads: list[dict] = field(default_factory=list)
    #: The case definition, so the report JSON is self-supporting.
    case: dict[str, Any] | None = None
    checks: list[CheckOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "turns": list(self.turns),
            "replies": list(self.replies),
            "sources": [list(s) for s in self.sources],
            "actions": [list(a) for a in self.actions],
            "handoff": list(self.handoff),
            "latencies_s": list(self.latencies_s),
            "prompt_tokens": list(self.prompt_tokens),
            "completion_tokens": list(self.completion_tokens),
            "usage_source": self.usage_source,
            "error": self.error,
            "leads": list(self.leads),
            "case": self.case,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class RunSummary:
    """One full pass of the case file against the service."""

    index: int
    base_url: str
    cases_path: str
    started_at: str
    finished_at: str
    results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "base_url": self.base_url,
            "cases_path": self.cases_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [r.to_dict() for r in self.results],
        }


# --------------------------------------------------------------------------
# Corpus-derived sets
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def allowed_rupee_amounts(data_dir: str | None = None) -> frozenset[int]:
    """Every rupee amount G2 accepts with no per-case help: catalog + policy."""
    corpus = load_corpus(data_dir)
    return frozenset(corpus.catalog_prices()) | POLICY_AMOUNTS


@lru_cache(maxsize=4)
def valid_source_ids(data_dir: str | None = None) -> frozenset[str]:
    """Every source id that exists: section ids from the two .md files, plus
    ``prices.csv#<SKU>`` for each SKU row."""
    return frozenset(load_corpus(data_dir).source_ids)


@lru_cache(maxsize=4)
def _valid_source_ids_folded(data_dir: str | None = None) -> frozenset[str]:
    # An id is a case-insensitive handle in practice, so folding is tolerance,
    # not a second source of truth: an unknown id still fails.
    return frozenset(s.casefold() for s in valid_source_ids(data_dir))


def normalise(text: str) -> str:
    """Normalise for must_include / must_not_include comparison.

    A thin alias of ``meher_agent.grounding.amounts.normalise_for_match`` so the
    harness and the agent share one implementation.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return normalise_for_match(text)


# --------------------------------------------------------------------------
# Tolerant coercion helpers
# --------------------------------------------------------------------------


def _skip(name: str, why: str) -> CheckOutcome:
    return CheckOutcome(name, True, f"skipped: {why}", skipped=True)


def _fail(name: str, why: str) -> CheckOutcome:
    return CheckOutcome(name, False, why)


def _pass(name: str, why: str) -> CheckOutcome:
    return CheckOutcome(name, True, why)


def _excerpt(text: str, limit: int = _EXCERPT) -> str:
    flat = _WS_RE.sub(" ", text or "").strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"


def _as_str_list(value: Any) -> list[str]:
    """Coerce a case field into a list of non-empty strings, dropping junk."""
    if value is None:
        return []
    if isinstance(value, str):
        items: list[Any] = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, (dict, list, tuple, set, frozenset)):
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _as_int_list(value: Any) -> list[int]:
    """Coerce a case field into ints, tolerating strings and '3,850'."""
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            out.append(int(item))
            continue
        try:
            out.append(to_int(str(item)))
        except ValueError:
            continue
    return out


def _case_category(case: Mapping[str, Any]) -> str:
    raw = case.get("category")
    if raw is None:
        return ""
    if isinstance(raw, (list, dict)):
        return ""
    return str(raw).strip().casefold()


def _at(seq: Sequence[T], index: int) -> T | None:
    if 0 <= index < len(seq):
        return seq[index]
    return None


def _last_index(result: CaseResult) -> int:
    return (
        max(
            len(result.replies),
            len(result.sources),
            len(result.actions),
            len(result.handoff),
            len(result.latencies_s),
        )
        - 1
    )


def _note(error: str | None) -> str:
    return f" [case failed: {error}]" if error else ""


# --------------------------------------------------------------------------
# Reply-level checks
# --------------------------------------------------------------------------


def _check_must_include(raw: Any, reply: str, note: str) -> CheckOutcome:
    needles = _as_str_list(raw)
    if not needles:
        return _skip("must_include", "case sets no must_include strings")
    hay = normalise(reply)
    missing = [n for n in needles if normalise(n) not in hay]
    if missing:
        return _fail(
            "must_include",
            f"{len(missing)} of {len(needles)} required string(s) absent: "
            f"{missing}; reply: \"{_excerpt(reply)}\"{note}",
        )
    return _pass("must_include", f"all {len(needles)} required string(s) present{note}")


def _check_must_include_any(raw: Any, reply: str, note: str) -> CheckOutcome:
    needles = _as_str_list(raw)
    if not needles:
        return _skip("must_include_any", "case sets no must_include_any strings")
    hay = normalise(reply)
    hits = [n for n in needles if normalise(n) in hay]
    if not hits:
        return _fail(
            "must_include_any",
            f"none of {len(needles)} alternative(s) present: {needles}; "
            f"reply: \"{_excerpt(reply)}\"{note}",
        )
    return _pass("must_include_any", f"matched alternative(s) {hits} of {needles}{note}")


def _check_must_not_include(raw: Any, reply: str, note: str) -> CheckOutcome:
    needles = _as_str_list(raw)
    if not needles:
        return _skip("must_not_include", "case sets no must_not_include strings")
    hay = normalise(reply)
    hits = [n for n in needles if normalise(n) in hay]
    if hits:
        return _fail(
            "must_not_include",
            f"forbidden string(s) present: {hits}; reply: \"{_excerpt(reply)}\"{note}",
        )
    return _pass("must_not_include", f"none of {len(needles)} forbidden string(s) present{note}")


def _check_g1(reply: str | None, note: str) -> CheckOutcome:
    if reply is None:
        return _fail("G1", f"no first reply was captured, so AI disclosure cannot be shown{note}")
    if _AI_RE.search(reply):
        return _pass("G1", f"first reply discloses AI: \"{_excerpt(reply)}\"{note}")
    return _fail("G1", f"first reply does not contain the word \"AI\": \"{_excerpt(reply)}\"{note}")


def _check_g2(reply: str | None, case_amounts: list[int], note: str) -> CheckOutcome:
    if reply is None:
        return _fail("G2", f"no reply was captured, so no amount can be shown to be allowed{note}")
    allowed = set(allowed_rupee_amounts()) | set(case_amounts)
    found = extract_rupee_amounts(reply)
    invented = [v for v in found if v not in allowed]
    if invented:
        return _fail(
            "G2",
            f"invented rupee amount(s) {invented} not in the allowed set "
            f"{sorted(allowed)}; reply: \"{_excerpt(reply)}\"{note}",
        )
    if not found:
        return _pass("G2", f"reply states no rupee amount{note}")
    return _pass("G2", f"all rupee amounts {found} are allowed{note}")


def _check_g3(reply: str | None, note: str) -> CheckOutcome:
    if reply is None:
        return _fail("G3", f"no reply was captured, so its length is unknown{note}")
    length = len(reply)
    if MIN_REPLY_CHARS <= length <= MAX_REPLY_CHARS:
        return _pass("G3", f"reply is {length} characters (allowed {MIN_REPLY_CHARS}..{MAX_REPLY_CHARS}){note}")
    return _fail(
        "G3",
        f"reply is {length} characters, outside {MIN_REPLY_CHARS}..{MAX_REPLY_CHARS}{note}",
    )


def _check_g4(sources: list[str] | None, category: str, note: str) -> CheckOutcome:
    if category not in G4_CATEGORIES:
        shown = category or "<none>"
        return _skip("G4", f"category {shown} is not one of {sorted(G4_CATEGORIES)}")
    if not sources:
        return _fail("G4", f"the last turn cited no sources{note}")
    valid = valid_source_ids()
    folded = _valid_source_ids_folded()
    invalid = [s for s in sources if s.strip() not in valid and s.strip().casefold() not in folded]
    if invalid:
        return _fail(
            "G4",
            f"{len(invalid)} of {len(sources)} cited source id(s) do not exist in data/: "
            f"{invalid}; valid ids are the business.md/policies.md section ids and "
            f"prices.csv#<SKU>{note}",
        )
    return _pass("G4", f"all {len(sources)} cited source id(s) exist in data/; {sources}{note}")


# --------------------------------------------------------------------------
# Action / lead checks
# --------------------------------------------------------------------------


def _action_type(action: Any) -> str:
    if isinstance(action, str):
        return action.strip()
    if isinstance(action, Mapping):
        for key in ("type", "name", "tool", "action"):
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _is_implicit_escalation(action: Any) -> bool:
    """True for a loop-generated escalate (step limit), not a model tool call.

    The public action shape is ``{"type", "args"}`` with no origin flag, so the
    only defensible reading is: the ``actions`` array holds what the model
    requested, and an escalate the *loop* generated is excluded only when the
    service marks it as such - an explicit auto flag, or a reason that names the
    step limit. Anything unmarked is counted as a real call.
    """
    if not isinstance(action, Mapping):
        return False
    if _action_type(action).casefold() != "escalate":
        return False
    args = action.get("args")
    if not isinstance(args, Mapping):
        return False
    for key in _AUTO_FLAG_KEYS:
        if key in args and str(args[key]).strip().casefold() in {"true", "yes", "1"}:
            return True
    for key in _REASON_KEYS:
        if key not in args:
            continue
        text = str(args[key]).strip().casefold()
        if any(marker in text for marker in _STEP_LIMIT_MARKERS):
            return True
    return False


def _model_called_actions(actions: list[Any]) -> list[str]:
    called: list[str] = []
    for action in actions or []:
        if _is_implicit_escalation(action):
            continue
        name = _action_type(action)
        if name:
            called.append(name)
    return called


def _parse_expect_action(raw: Any) -> tuple[list[str], list[str]]:
    """Return (understandable expectations, unusable raw values)."""
    raw_items = _as_str_list(raw)
    good: list[str] = []
    bad: list[str] = []
    for item in raw_items:
        token = item.strip().casefold()
        if token in _NO_ACTION_ALIASES:
            good.append("none")
        elif token in _ACTION_NAMES:
            good.append(token)
        else:
            bad.append(item)
    return good, bad


def _check_expect_action(raw: Any, actions: list[Any], note: str) -> CheckOutcome:
    if raw is None:
        return _skip("expect_action", "case sets no expect_action")
    wanted, unusable = _parse_expect_action(raw)
    if not wanted:
        return _skip(
            "expect_action",
            f"case sets expect_action to {raw!r}, which names no known action "
            f"(expected one of {sorted(_ACTION_NAMES)} or 'none')",
        )
    called = _model_called_actions(actions)
    suffix = f"; ignored unusable expect_action value(s) {unusable}" if unusable else ""
    if "none" in wanted:
        if called:
            return _fail(
                "expect_action",
                f"case expects no tool call, but the last turn called {sorted(set(called))}{note}{suffix}",
            )
        return _pass("expect_action", f"case expects no tool call and none was made{note}{suffix}")
    for name in wanted:
        if name != "none" and name in called:
            detail = f"last turn called {name} (all calls: {called})"
            return _pass("expect_action", f"{detail}{note}{suffix}")
    return _fail(
        "expect_action",
        f"expected one of {wanted}, last turn called {called or 'nothing'}{note}{suffix}",
    )


def _norm_text(value: Any) -> str:
    return _WS_RE.sub(" ", str(value)).strip().casefold()


def _norm_phone(value: Any) -> str:
    try:
        return normalise_phone(value)
    except Exception:  # an unparsable value still has to compare against itself
        return _norm_text(value)


def _norm_email(value: Any) -> str:
    try:
        return normalise_email(value)
    except Exception:  # as above
        return _norm_text(value)


def _norm_date(value: Any) -> str:
    try:
        return normalise_date(value)
    except Exception:  # as above
        return _norm_text(value)


#: Field -> normaliser. Unknown fields are reported, never guessed at.
_LEAD_FIELDS: dict[str, Callable[[Any], str]] = {
    "phone": _norm_phone,
    "email": _norm_email,
    "date": _norm_date,
    "name": _norm_text,
    "need": _norm_text,
    "quantity": _norm_text,
    "conversation_id": _norm_text,
    "created_at": _norm_text,
}


def _lead_candidates(result: CaseResult, actions: list[Any]) -> list[dict]:
    candidates: list[dict] = [c for c in result.leads if isinstance(c, Mapping)]
    for action in actions or []:
        if not isinstance(action, Mapping):
            continue
        if _action_type(action).casefold() != "save_lead":
            continue
        args = action.get("args")
        if isinstance(args, Mapping):
            candidates.append(dict(args))
    return candidates


def _check_expect_lead(raw: Any, result: CaseResult, actions: list[Any], note: str) -> CheckOutcome:
    if raw is None:
        return _skip("expect_lead", "case sets no expect_lead")
    if not isinstance(raw, Mapping):
        return _skip(
            "expect_lead",
            f"case sets expect_lead to {type(raw).__name__}, not an object of lead fields",
        )
    unknown = [k for k in raw if k not in _LEAD_FIELDS]
    comparable = {k: v for k, v in raw.items() if k in _LEAD_FIELDS}
    candidates = _lead_candidates(result, actions)
    if not candidates:
        return _fail(
            "expect_lead",
            f"expected a saved lead with {comparable}, but no save_lead action was taken{note}",
        )
    if not comparable:
        return _skip(
            "expect_lead",
            f"expect_lead names only fields that are not lead fields ({unknown}), so nothing is comparable",
        )
    best_detail = ""
    best_score = -1
    for candidate in candidates:
        diffs: list[str] = []
        for field_name, expected in comparable.items():
            actual = candidate.get(field_name)
            if actual is None:
                diffs.append(f"{field_name}: missing (expected {expected!r})")
                continue
            if _LEAD_FIELDS[field_name](expected) != _LEAD_FIELDS[field_name](actual):
                diffs.append(f"{field_name}: expected {expected!r}, got {actual!r}")
        score = len(comparable) - len(diffs)
        if score > best_score:
            best_score = score
            best_detail = "; ".join(diffs) if diffs else "exact match on every field"
        if not diffs:
            extra = f"; ignored unknown expect_lead field(s) {unknown}" if unknown else ""
            return _pass(
                "expect_lead",
                f"saved lead matches on {sorted(comparable)} (normalised){note}{extra}",
            )
    return _fail("expect_lead", f"no saved lead matches; closest differs: {best_detail}{note}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _guarded(name: str, run: Callable[[], CheckOutcome]) -> CheckOutcome:
    try:
        outcome = run()
    except Exception as exc:  # a harness bug must not be a silent pass
        return _fail(name, f"harness error while running {name}: {exc!r}")
    if not isinstance(outcome, CheckOutcome):
        return _fail(name, f"harness error: {name} produced {outcome!r}")
    if not outcome.name:
        outcome.name = name
    return outcome


def check_case(case: dict, result: CaseResult) -> list[CheckOutcome]:
    """Every check for one case, in a fixed order.

    ``case`` is read field by field with tolerant coercion: an unknown extra
    field is ignored, a missing ``category`` only disables G4, a missing
    ``turns`` is irrelevant here (the runner replays them), an ``expect_action``
    given as a list is read as "any of these", and an ``expect_lead`` field that
    is not a lead field is reported and skipped. None of them raise.
    """
    if not isinstance(case, Mapping):
        return [_fail("case", f"case must be a JSON object, got {type(case).__name__}")]
    if not isinstance(result, CaseResult):
        return [_fail("case", f"result must be a CaseResult, got {type(result).__name__}")]

    outcomes: list[CheckOutcome] = []
    category = _case_category(case)
    last = _last_index(result)
    reply = _at(result.replies, last)
    sources = _at(result.sources, last)
    actions = _at(result.actions, last)
    note = _note(result.error)

    def string_list_field(key: str) -> list[str]:
        return _as_str_list(case.get(key))

    outcomes.append(
        _guarded("must_include", lambda: _check_must_include(string_list_field("must_include"), reply or "", note))
    )
    outcomes.append(
        _guarded(
            "must_include_any",
            lambda: _check_must_include_any(string_list_field("must_include_any"), reply or "", note),
        )
    )
    outcomes.append(
        _guarded(
            "must_not_include",
            lambda: _check_must_not_include(string_list_field("must_not_include"), reply or "", note),
        )
    )
    outcomes.append(_guarded("expect_action", lambda: _check_expect_action(case.get("expect_action"), actions or [], note)))
    outcomes.append(
        _guarded("expect_lead", lambda: _check_expect_lead(case.get("expect_lead"), result, actions or [], note))
    )
    outcomes.append(_guarded("G1", lambda: _check_g1(_at(result.replies, 0), note)))
    outcomes.append(
        _guarded("G2", lambda: _check_g2(reply, _as_int_list(case.get("allowed_amounts")), note))
    )
    outcomes.append(_guarded("G3", lambda: _check_g3(reply, note)))
    outcomes.append(
        _guarded("G4", lambda: _check_g4(list(sources) if sources is not None else None, category, note))
    )
    return outcomes


def failed_checks(result: CaseResult) -> list[CheckOutcome]:
    """Graded failures only; skipped checks never count as failures."""
    return [c for c in result.checks if not c.passed]


def case_passed(result: CaseResult) -> bool:
    return not failed_checks(result)
