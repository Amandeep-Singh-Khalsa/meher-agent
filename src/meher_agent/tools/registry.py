"""The tool layer: the only place where the model can change anything.

Every argument arrives from a 7B model, so all of it is treated as hostile input.
`ToolRegistry.run` never raises: a bad call comes back as `ToolResult(ok=False)`
whose `content` is a JSON instruction the model can act on, and is recorded as an
`Action` for the API response and the eval harness. A validation error is a normal
turn, not a 500.

Two rules decided here and relied on elsewhere:

* `Action.args` holds the **post-validation** values on success (the harness
  compares them against `expect_lead`) and the **raw, truncated** values on
  failure, so a rejected call is visible in the response.
* A successful `escalate` sets the handoff flag. The agent loop can read it from
  `registry.handoff`, from `handoff_for(conversation_id)`, or - most robustly -
  from `any(a.type == "escalate" and a.ok for a in actions_for(cid))`.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..config import Config
from ..data.corpus import Corpus
from ..types import Action, Lead, ToolResult
from .store import LeadStore
from .validation import (
    ValidationError,
    clean_optional_text,
    normalise_date,
    normalise_email,
    normalise_name,
    normalise_phone,
)

__all__ = ["ToolRegistry", "TOOL_NAMES"]

log = logging.getLogger(__name__)

TOOL_NAMES: tuple[str, ...] = ("save_lead", "escalate")

#: Anything longer is a pasted message, not a field value. Checked before the
#: validation regexes run, so a 10 MB string cannot make them allocate a 10M-item
#: digit list.
_MAX_FIELD_CHARS = 1000
#: Recorded `Action.args` stay small: the response is logged and graded.
_MAX_RECORDED_CHARS = 200
_MAX_RECORDED_ITEMS = 20
_MAX_RECORDED_DEPTH = 4
_MAX_KEY_CHARS = 50

_HINT_PHONE = (
    "Ask the customer to re-read their number, or ask for an email instead. "
    "Do not invent a number."
)
_HINT_EMAIL = (
    "Ask the customer to re-read the address, or ask for a phone number instead. "
    "Do not invent an email."
)
_HINT_CONTACT = (
    "Ask the customer for a phone number or an email address so the team can reach "
    "them. Do not invent contact details."
)
_HINT_NAME = "Ask the customer to spell the name again. Do not guess at a spelling."
_HINT_NEED = "Ask what the order is for, in one short sentence, then save the lead."
_HINT_DATE = "Ask the customer to repeat the date. Omit the field if they are unsure."
_HINT_QUANTITY = "Ask how many they need, or leave the field out."
_HINT_REASON = "Say what went wrong in the customer's own words, then escalate again."
_HINT_ARGS = (
    "Call the tool again with a JSON object of named fields, for example "
    '{"name": "Ritu Malhotra", "need": "30 gift boxes", "phone": "9876543210"}.'
)
_HINT_UNKNOWN = "Call one of the available tools, or answer the customer directly."
_HINT_INTERNAL = "Try once more with the same details; if it fails again, escalate with the reason."

_HINT_BY_FIELD: dict[str, str] = {
    "name": _HINT_NAME,
    "need": _HINT_NEED,
    "phone": _HINT_PHONE,
    "email": _HINT_EMAIL,
    "quantity": _HINT_QUANTITY,
    "date": _HINT_DATE,
    "reason": _HINT_REASON,
}


class _ArgError(ValidationError):
    """A validation failure that already knows what the model should do next."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _hint(field: str) -> str:
    return _HINT_BY_FIELD.get(field, _HINT_ARGS)


def _require_text(raw: object, field: str) -> str:
    """Insist on a JSON string.

    A number where a name belongs is the model having invented or mangled
    something, and the recovery is to make it ask again, not to coerce silently.
    """
    if not isinstance(raw, str):
        raise _ArgError(
            f"{field} is required and must be a string, got {type(raw).__name__}", _hint(field)
        )
    if len(raw) > _MAX_FIELD_CHARS:
        raise _ArgError(
            f"{field} is {len(raw)} characters long, which is a pasted message rather than a {field}",
            _hint(field),
        )
    return raw


def _optional_text(raw: object, field: str) -> str | None:
    return None if raw is None else _require_text(raw, field)


def _optional_quantity(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise _ArgError("quantity must be a number or a short string, got bool", _HINT_QUANTITY)
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not math.isfinite(raw):
            raise _ArgError("quantity must be a finite number or a short string", _HINT_QUANTITY)
        return _require_text(str(raw), "quantity")
    return _require_text(raw, "quantity")


def _normalise(fn: Callable[[object], str], value: str, field: str) -> str:
    try:
        return fn(value)
    except ValidationError as exc:
        raise _ArgError(str(exc), _hint(field)) from exc


def _type_marker(value: object) -> str:
    try:
        return f"{type(value).__name__}(len={len(value)})"  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - this runs on hostile input, it must not raise
        return type(value).__name__


def _safe_value(value: object, depth: int = 0) -> Any:
    """A JSON-safe, size-bounded view of an arbitrary value, for the action log."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) < 2**53 else _clip(str(value), _MAX_RECORDED_CHARS)
    if isinstance(value, float):
        return value if math.isfinite(value) else _clip(str(value), _MAX_RECORDED_CHARS)
    if isinstance(value, str):
        return _clip(value, _MAX_RECORDED_CHARS)
    if depth >= _MAX_RECORDED_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_value(item, depth + 1) for item in list(value)[:_MAX_RECORDED_ITEMS]]
    if isinstance(value, dict):
        return {
            _clip(str(k), _MAX_KEY_CHARS): _safe_value(v, depth + 1)
            for k, v in list(value.items())[:_MAX_RECORDED_ITEMS]
        }
    return f"<{type(value).__name__}>"


def _safe_conversation_id(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _split_args(args: object) -> tuple[dict[str, Any], dict[str, Any], _ArgError | None]:
    """Return (raw args for validation, recorded-safe args, error).

    Validation reads the untouched values so that a real field is never judged on
    a truncated copy; the action log gets the bounded copy so one 10 MB argument
    cannot land in the HTTP response.
    """
    if not isinstance(args, dict):
        marker = {"__args__": _type_marker(args)}
        reason = f"tool arguments must be a JSON object of named fields, got {type(args).__name__}"
        return {}, marker, _ArgError(reason, _HINT_ARGS)

    raw: dict[str, Any] = {}
    safe: dict[str, Any] = {}
    for key, value in args.items():
        skey = _clip(str(key), _MAX_KEY_CHARS)
        raw[skey] = value
        safe[skey] = _safe_value(value)
    return raw, safe, None


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


class ToolRegistry:
    def __init__(
        self,
        corpus: Corpus,
        config: Config,
        *,
        leads: LeadStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        # corpus/config are held so tool messages can be grounded in shop data.
        # `leads` is injectable so the API layer can share one LeadStore with the
        # registry instead of silently splitting the leads across two objects.
        self._corpus = corpus
        self._config = config
        self._leads = LeadStore() if leads is None else leads
        self._clock = clock if clock is not None else _utc_now
        self._lock = threading.Lock()
        self._log: list[tuple[str, Action]] = []
        self._handoff = False
        self._handoff_conversation: str | None = None
        self._lead_seq = 0

    # -- introspection -----------------------------------------------------

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def config(self) -> Config:
        return self._config

    @property
    def leads(self) -> LeadStore:
        return self._leads

    @property
    def handoff(self) -> bool:
        """True when a successful escalate has been recorded since the last reset."""
        with self._lock:
            return self._handoff

    def handoff_for(self, conversation_id: str) -> bool:
        with self._lock:
            return self._handoff and self._handoff_conversation == conversation_id

    def names(self) -> list[str]:
        return list(TOOL_NAMES)

    def actions(self) -> list[Action]:
        """Every call recorded since the last `reset`, successes and failures."""
        with self._lock:
            return [action for _, action in self._log]

    def actions_for(self, conversation_id: str) -> list[Action]:
        """The subset belonging to one conversation.

        `actions()` is the whole turn; a shared registry serving several
        conversations concurrently should use this one instead.
        """
        with self._lock:
            return [action for cid, action in self._log if cid == conversation_id]

    def reset(self) -> None:
        """Clear the per-turn action log and handoff flag. Call at turn start."""
        with self._lock:
            self._log.clear()
            self._handoff = False
            self._handoff_conversation = None

    # -- execution ---------------------------------------------------------

    def run(self, name: str, args: object, *, conversation_id: str) -> ToolResult:
        """Execute one tool call. Returns a ToolResult for every possible input."""
        tool_name = name if isinstance(name, str) else _type_marker(name)
        raw_args: dict[str, Any] = {}
        safe_args: dict[str, Any] = {}
        args_error: _ArgError | None = None

        ok = False
        error: str | None = None
        hint = _HINT_ARGS
        recorded: dict[str, Any] = safe_args
        payload: dict[str, Any] = {}

        try:
            conv_id = _safe_conversation_id(conversation_id)
            raw_args, safe_args, args_error = _split_args(args)
            recorded = safe_args
            if tool_name not in TOOL_NAMES:
                raise _ArgError(
                    f"unknown tool '{_clip(tool_name, 60)}'; available tools are "
                    f"{', '.join(TOOL_NAMES)}",
                    _HINT_UNKNOWN,
                )
            if args_error is not None:
                raise args_error
            if tool_name == "save_lead":
                payload, recorded = self._save_lead(raw_args, conv_id)
            else:
                payload, recorded = self._escalate(raw_args, conv_id)
            ok = True
        except _ArgError as exc:
            error, hint = str(exc) or "invalid arguments", exc.hint
        except Exception as exc:  # noqa: BLE001 - a bad tool call must not kill the turn
            log.exception("tool %r failed unexpectedly", tool_name)
            error = f"internal error in tool '{tool_name}': {type(exc).__name__}"
            hint = _HINT_INTERNAL

        if not ok:
            payload = {"ok": False, "error": error, "hint": hint}
            recorded = safe_args

        self._record(_safe_conversation_id(conversation_id), Action(
            type=tool_name, args=recorded, ok=ok, error=error
        ))
        return ToolResult(ok=ok, content=_dumps(payload), error=error)

    # -- tools -------------------------------------------------------------

    def _save_lead(self, args: dict[str, Any], conversation_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        name = _normalise(normalise_name, _require_text(args.get("name"), "name"), "name")

        need = clean_optional_text(_require_text(args.get("need"), "need"), max_len=_MAX_FIELD_CHARS)
        if not need:
            raise _ArgError("need is required and must describe what the customer wants", _HINT_NEED)

        phone_raw = _optional_text(args.get("phone"), "phone")
        email_raw = _optional_text(args.get("email"), "email")
        if phone_raw is None and email_raw is None:
            raise _ArgError(
                "save_lead needs at least one contact detail: a phone number or an email "
                "address, so the team can call or write back",
                _HINT_CONTACT,
            )

        phone = None if phone_raw is None else _normalise(normalise_phone, phone_raw, "phone")
        email = None if email_raw is None else _normalise(normalise_email, email_raw, "email")

        quantity = clean_optional_text(_optional_quantity(args.get("quantity")))
        date_raw = _optional_text(args.get("date"), "date")
        when = None if date_raw is None else _normalise(normalise_date, date_raw, "date")

        lead = Lead(
            name=name,
            need=need,
            phone=phone,
            email=email,
            quantity=quantity,
            date=when,
            conversation_id=conversation_id,
            created_at=self._now_iso(),
        )
        self._leads.add(lead)
        with self._lock:
            self._lead_seq += 1
            lead_id = f"lead-{self._lead_seq:05d}"

        if phone and email:
            channel = "by phone or email"
        elif phone:
            channel = "by phone"
        else:
            channel = "by email"

        recorded: dict[str, Any] = {"name": name, "need": need}
        if phone:
            recorded["phone"] = phone
        if email:
            recorded["email"] = email
        if quantity:
            recorded["quantity"] = quantity
        if when:
            recorded["date"] = when

        payload = {
            "ok": True,
            "lead_id": lead_id,
            "message": f"Lead saved. The team will follow up {channel}.",
        }
        return payload, recorded

    def _escalate(self, args: dict[str, Any], conversation_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        reason = clean_optional_text(
            _require_text(args.get("reason"), "reason"), max_len=_MAX_FIELD_CHARS
        )
        if not reason:
            raise _ArgError("reason is required and must say what needs a human", _HINT_REASON)

        with self._lock:
            self._handoff = True
            self._handoff_conversation = conversation_id

        payload = {
            "ok": True,
            "handoff": True,
            "message": (
                "Handed to a human teammate. Tell the customer the team will get back to "
                "them, apologise once if they are unhappy, and ask for anything still missing."
            ),
        }
        return payload, {"reason": reason}

    # -- internals ---------------------------------------------------------

    def _now_iso(self) -> str:
        moment = self._clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _record(self, conversation_id: str, action: Action) -> None:
        with self._lock:
            self._log.append((conversation_id, action))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
