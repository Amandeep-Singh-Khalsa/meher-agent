"""The hand-written tool-calling loop: the only place that talks to the model.

Every other layer is deterministic. Retrieval decides what is true, billing
decides what the money is, the guard decides what may be said, the registry
decides what a tool call is allowed to do. This file decides only two things:
how many times to ask the model, and which draft to keep.

Three invariants hold no matter what comes back from the endpoint:

* at most ``config.llm.max_steps`` model calls per customer turn, repair calls
  included. The budget is a counter taken *before* each call, so an exception,
  an empty answer or a runaway tool loop can never exceed it.
* a reply always exists, always fits the length limit, and always discloses that
  it is an AI on the first turn.
* no exception escapes. A turn that fails somewhere unexpected still records its
  history, hands off to the shop team, and logs a masked traceback.

* no customer ever reads a tool call. A 7B model asked to call a tool often writes
  `escalate {"reason": "..."}` into its answer instead; the loop runs those calls
  through the registry unchanged and removes them from the text.

The budget is spent in a fixed order, because the order is the design:

1. the answer itself,
2. tool round trips (a lead or a complaint is worth more than a second draft),
3. the intent nudge, for when a 7B model serialised the tool call as prose,
4. the guard repair, the only one that can be skipped without losing the turn.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from typing import Any

from ..grounding.amounts import ascii_digits
from ..grounding.guard import (
    INTENT_COMPLAINT,
    INTENT_ESCALATION,
    INTENT_RETURNS,
    GuardVerdict,
    extract_phones,
    normalise_language,
)
from ..llm.client import LLMResponse
from ..llm.prompts import NUDGE_TOOL, build_messages, fence_customer_message, tool_schemas
from ..tools.validation import normalise_email, normalise_phone
from ..types import Action, AgentOutcome, Briefing, Quote, ToolResult, TurnRecord
from .conversation import Conversation
from .services import Services

__all__ = ["run_turn", "run_message", "MAX_SOURCES"]

logger = logging.getLogger(__name__)

#: The response `sources` array is read by a human and by the eval harness, so it
#: stays as small as the evidence allows: the model's own citations first, then
#: the ids retrieval actually returned.
MAX_SOURCES = 4

#: Tool calls the model wrote into its answer instead of the tool_calls channel,
#: recovered per turn. The cap keeps a pathological answer from turning into a
#: loop of rescues; two is already far more than a real model produces.
_MAX_RESCUED_PROSE_CALLS = 2

#: Appended to a first-turn model reply that never said it was an AI. Short, and
#: it must carry the literal substring "AI" in every language, so the Hindi tag
#: spells it in Latin script inside a Devanagari sentence.
_DISCLOSURE = {
    "english": "(AI assistant)",
    "hinglish": "(AI assistant)",
    "hindi": "(AI सहायक)",
}

#: Deterministic lead signal, checked only when the guard's own intent detection
#: did not already fire: asking to order, or naming a bulk, wedding or custom
#: order in any of the three scripts.
_ORDER_WORDS: tuple[str, ...] = (
    "order", "orders", "ordered", "ordering",
    "bulk", "wholesale", "wedding", "custom", "customise", "customize",
    "ऑर्डर", "थोक", "शादी", "कस्टम",
)

#: How a customer introduces themselves. Paired with a phone number or an email
#: address from the same message, this is the "name plus contact" the tool
#: schema asks for.
_SELF_INTRO_WORDS: tuple[str, ...] = (
    "my name is", "i am", "i'm", "this is", "myself", "here is", "speaking",
    "मेरा नाम", "मैं हूँ",
)

_COMPLAINT_INTENTS = frozenset({INTENT_COMPLAINT, INTENT_RETURNS})
_LEAD_INTENTS = frozenset({INTENT_ESCALATION})

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
#: Word-bounded on purpose: a bare `"ai" in reply` is satisfied by "daily" and
#: "favourite", which would let a first reply through with no disclosure at all.
_AI_TOKEN_RE = re.compile(r"\bai\b", re.IGNORECASE)
#: A cited id is only ever accepted in the exact form data/ uses, so a model
#: cannot invent a citation that merely looks right.
_SOURCE_RE = re.compile(r"(?:business\.md|prices\.csv|policies\.md)#[A-Za-z0-9_-]+")
#: Used only when the shared PII scrubber is not importable yet.
_PII_REDACT_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z.]{2,}|\+?\d[\d \-]{7,}\d")

_REPAIR_INSTRUCTION = (
    "Your last reply was rejected by an automated check. Rewrite it once, fixing exactly "
    "these problems:\n{problems}\nKeep only rupee figures that appear in the COMPUTED QUOTE "
    "or PERMITTED RUPEE AMOUNTS blocks, keep the same language and script, do not add "
    "headings, and output only the corrected reply text."
)

#: Used when even the guard's own templates fail. Must satisfy every rule the
#: eval harness checks: non-empty, no invented amounts, states it is an AI, and
#: tells the customer a human is taking over.
_HARD_FALLBACK = (
    "I am the AI assistant for Meher Sweets & Namkeen. I could not finish that request just "
    "now, so I have passed it to the shop team. They will follow up with you by email within "
    "one working day."
)


# --------------------------------------------------------------------------
# Small text helpers
# --------------------------------------------------------------------------


def _clamp(text: str, limit: int) -> str:
    """Cut to `limit` characters on a word boundary. Never returns more."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space].rstrip()
    return cut


def _clean(text: str) -> str:
    """A model's answer without the markdown fence some 7B models wrap it in."""
    body = (text or "").strip()
    if body.startswith("```") and body.count("```") >= 2:
        body = body.split("```", 2)[1].strip()
    return body


def _with_disclosure(text: str, language: str, is_first_turn: bool, limit: int) -> str:
    """Clamp first, then append the AI disclosure.

    The reverse order can push a reply back over the length limit, which is one
    of the things the eval harness fails on. Doing the disclosure here rather
    than through the guard also keeps a missing "AI" from costing a model call:
    it is a deterministic fix, not a model failure.
    """
    body = _clamp(text.strip(), limit)
    if not body or not is_first_turn or _AI_TOKEN_RE.search(body):
        return body
    tag = _DISCLOSURE.get(language, _DISCLOSURE["english"])
    return f"{_clamp(body, max(0, limit - len(tag) - 1))}\n{tag}".strip()


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    folded = (text or "").casefold()
    return any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", folded) for word in words)


def _kept_enough(draft: str, repaired: str) -> bool:
    """Did the guard's rewrite leave an answer, or only a fragment of one?

    `ReplyGuard.repair` drops the sentences that carry an invented amount and
    keeps the rest, which is right for most of a reply and useless when the
    invented amount was the only sentence: what is left is a citation and little
    else. In that case the customer gets the computed template instead.
    """
    said = len((draft or "").split())
    kept = len((repaired or "").split())
    return kept * 2 >= said


def _looks_like_lead(question: str) -> bool:
    """A lead the guard's keyword intents may have missed.

    Two signals only, both cheap and both deterministic: an ordering word, or a
    self-introduction together with a phone number or an email address.
    """
    text = ascii_digits(question or "")
    if _has_word(text, _ORDER_WORDS):
        return True
    has_contact = bool(extract_phones(text)) or bool(_EMAIL_RE.search(text))
    return has_contact and _has_word(text, _SELF_INTRO_WORDS)


#: Name straight after a self-introduction, Latin or Devanagari, up to four words.
_NAME_AFTER_INTRO_RE = re.compile(
    # The prefix is matched case-insensitively, the name case-sensitively, so
    # "I am" matches while the name still has to look like a name.
    r"(?i:my name is|i am|i'm|this is|myself)\s+"
    r"((?:[A-Z][\w'’-]+|[\u0900-\u097F][\u0900-\u097F'’]*)(?:\s+(?:[A-Z][\w'’-]+|[\u0900-\u097F][\u0900-\u097F'’]*)){0,3})",
    re.UNICODE,
)

#: Cap on the "need" the loop writes down when it reads one out of the message.
_MAX_NEED_CHARS = 400


def _scrub(text: str) -> str:
    """PII scrub for a log line, preferring the shared safety module."""
    try:
        from ..safety.pii import mask_text
    except Exception:  # noqa: BLE001 - the safety layer may not be present yet
        return _PII_REDACT_RE.sub("[redacted]", text)
    try:
        return str(mask_text(text))
    except Exception:  # noqa: BLE001 - never let logging be the failure
        return _PII_REDACT_RE.sub("[redacted]", text)


# --------------------------------------------------------------------------
# The model-call budget
# --------------------------------------------------------------------------


class _Budget:
    """The hard cap on model calls, taken before each one.

    Repair and intent calls are taken from the same pool as the answer, which is
    the only way the cap means anything.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self.spent = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.spent += 1
        return True


# --------------------------------------------------------------------------
# One turn
# --------------------------------------------------------------------------


class _Run:
    """Mutable state of a single ``run_turn`` call."""

    def __init__(self, message: str, conversation_id: str, services: Services) -> None:
        self.services = services
        self.message = message or ""
        self.conversation_id = conversation_id
        self.conversation: Conversation | None = None
        self.briefing: Briefing | None = None
        self.quote: Quote | None = None
        self.is_first_turn = True
        self.budget = _Budget(services.config.llm.max_steps)
        self.tool_errors: list[str] = []
        self.guard_repairs = 0
        self.intent_repairs = 0
        self.step_limit_hit = False
        self.used_fallback_reply = False
        self.handoff = False
        #: Set when *the loop* escalated rather than the model, which makes the
        #: guard's own handoff text the reply instead of the model's sentence.
        self.loop_escalated = False
        self.tool_calls_seen = 0
        self.tool_attempted = 0
        self.tool_calls_succeeded = 0
        self.lead_saved = False

    # -- entry points -------------------------------------------------------

    def execute(self) -> AgentOutcome:
        self._prepare()
        draft = self._model_loop()
        reply = self._ground(draft)
        sources = self._sources(reply)
        actions = self._ok_actions()
        self._record(reply)
        return self._outcome(reply, sources, actions)

    def safe_outcome(self) -> AgentOutcome:
        """The turn failed somewhere unexpected. Never raise out of here either."""
        try:
            return self._safe_outcome()
        except Exception:  # noqa: BLE001 - last line of defence
            logger.error("run_turn could not even build a safe outcome for %r", self.conversation_id)
            return AgentOutcome(
                reply=_HARD_FALLBACK,
                handoff=True,
                model_calls=self.budget.spent,
                prompt_tokens=self.budget.prompt_tokens,
                completion_tokens=self.budget.completion_tokens,
                tool_errors=list(self.tool_errors),
                guard_repairs=self.guard_repairs,
                intent_repairs=self.intent_repairs,
                step_limit_hit=self.step_limit_hit,
                used_fallback_reply=True,
            )

    # -- preparation --------------------------------------------------------

    def _prepare(self) -> None:
        services = self.services
        self.conversation = services.conversations.get(self.conversation_id)
        self.is_first_turn = self.conversation.reply_count == 0
        # The action log lives on the registry, and one registry is shared by
        # every conversation, so it is cleared here or the next turn inherits it.
        services.tools.reset()
        briefing = services.retrieval.brief(self.message, self.conversation.turns)
        self.briefing = briefing
        if briefing.resolved.items:
            self.quote = services.billing.quote(briefing.resolved)

    def _history(self) -> list[TurnRecord]:
        return list(self.conversation.turns) if self.conversation is not None else []

    def _context_messages(self) -> list[dict[str, Any]]:
        services = self.services
        return build_messages(
            self.briefing,
            self._history(),
            quote=self.quote,
            max_history_turns=services.config.runtime.max_history_turns,
        )

    # -- the loop -----------------------------------------------------------

    def _deterministic_escalate(self) -> str | None:
        """Escalate a complaint without waiting for the model to call the tool.

        A complaint is the one thing that must never be missed: the customer has
        a problem and needs a human. The guard detects it deterministically from
        the message text, so the loop escalates immediately — the same way a lead
        is captured deterministically. This also saves a model call, because the
        guard writes the apology and handoff text itself.
        """
        if self.briefing is None:
            return None
        try:
            intent = self.services.guard.detect_intent(self.briefing, self.quote)
        except Exception:  # noqa: BLE001 - a detector failure must not skip the answer
            return None
        if intent not in _COMPLAINT_INTENTS and intent not in _LEAD_INTENTS:
            return None
        if self.loop_escalated or self.handoff:
            return None
        return self._escalate_and_reply("customer raised a complaint")

    def _model_loop(self) -> str:
        """Ask the model until it produces a clean draft, or the budget runs out."""
        schemas = tool_schemas()
        messages = self._context_messages()
        wants_nudge = self._wants_tool_repair()

        while True:
            response = self._complete(messages, schemas)
            if response is None:
                return self._step_limit_reply()
            if response.tool_calls:
                self._run_tools(response, messages)
                continue
            rescued = self._rescue_prose_calls(response.content)
            # The model answered but did not escalate. The guard already detected
            # the complaint, so the loop escalates on its behalf — the same way a
            # lead is captured when the model forgets the tool. The model's reply
            # is discarded because the guard writes the apology and handoff.
            if not self.loop_escalated and not self.handoff:
                fallback = self._deterministic_escalate()
                if fallback is not None:
                    return fallback
            # The model made no attempt to call a tool at all: it answered and
            # forgot. The details are all in the message, so the loop saves the
            # lead itself, which costs no model call. A model that *did* call a
            # tool is left alone, even if the call was rejected: the loop asks
            # the customer to correct a bad phone rather than guessing at one.
            if not self.tool_attempted and not self.lead_saved and self._deterministic_lead():
                return rescued or response.content
            if self.tool_calls_succeeded and rescued.strip():
                if rescued != response.content:
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": NUDGE_TOOL})
                return rescued
            if wants_nudge and not self.lead_saved and not self.loop_escalated and self.budget.remaining:
                wants_nudge = False
                self.intent_repairs += 1
                if response.content.strip():
                    messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": NUDGE_TOOL})
                continue
            return rescued or response.content

    def _complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> LLMResponse | None:
        """The only function in the codebase that calls the model, so it is the
        only one that spends the budget. None means the cap is already reached.
        """
        if not self.budget.take():
            return None
        response = self.services.llm.complete(messages, tools=tools)
        usage = response.usage
        self.budget.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.budget.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return response

    def _run_tools(self, response: LLMResponse, messages: list[dict[str, Any]]) -> None:
        """Execute every requested tool and hand the results back to the model.

        A rejected call is a normal iteration: its error goes back as the tool
        message *and* into the diagnostics, so the model gets a second attempt
        inside the same turn.
        """
        messages.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in response.tool_calls
                ],
            }
        )
        for call in response.tool_calls:
            self.tool_calls_seen += 1
            self.tool_attempted += 1
            result = self._run_tool(
                call.name, call.arguments, conversation_id=self.conversation_id
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result.content}
            )
            if not result.ok:
                self.tool_errors.append(f"{call.name}: {result.error or 'tool call rejected'}")
            else:
                self.tool_calls_succeeded += 1
                if call.name == "escalate":
                    self.handoff = True
                elif call.name == "save_lead":
                    self.lead_saved = True

    def _wants_tool_repair(self) -> bool:
        if not self.services.config.agent.allow_intent_repair or self.briefing is None:
            return False
        try:
            intent = self.services.guard.detect_intent(self.briefing, self.quote)
        except Exception:  # noqa: BLE001 - a detector failure must not skip the answer
            return False
        if intent in _COMPLAINT_INTENTS or intent in _LEAD_INTENTS:
            return True
        return _looks_like_lead(self.briefing.question)

    def _ground_contact_args(self, name: str, args: Any) -> Any:
        """Drop any contact detail the customer never actually gave.

        The model fills a missing field rather than leaving it out: asked to save
        a lead from a message carrying only an email, it will happily invent a
        ten-digit phone number. A fabricated number in a CRM is worse than a
        missing one, because the team will call it. Anything that cannot be found
        in the customer's own message is removed here, and the tool then rejects
        the call outright if nothing valid is left.
        """
        if name != "save_lead" or not isinstance(args, dict) or self.briefing is None:
            return args
        source = self.briefing.question or ""
        email_source = source.casefold()
        source_digits = re.sub(r"\D", "", ascii_digits(source))
        cleaned = dict(args)
        for field, normalise in (("phone", normalise_phone), ("email", normalise_email)):
            value = cleaned.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            # Only a *plausible* detail can be a hallucination. One that is
            # already invalid is not the model inventing something, and the
            # registry's specific complaint ("10 digits", "not a valid email")
            # is far more useful to the customer than a silent omission.
            try:
                normalise(value)
            except Exception:  # noqa: BLE001 - invalid, so leave it to the registry
                continue
            if field == "email":
                present = value.strip().casefold() in email_source
            else:
                digits = re.sub(r"\D", "", value)
                present = bool(digits) and digits in source_digits
            if not present:
                # Not an error: the call still succeeds, so the correction stays
                # out of tool_errors and the customer is never asked about a
                # detail they did not give.
                cleaned.pop(field, None)
        return cleaned

    def _run_tool(self, name: str, args: Any, *, conversation_id: str) -> Any:
        args = self._ground_contact_args(name, args)
        if name == "escalate" and self.briefing is not None:
            try:
                intent = self.services.guard.detect_intent(self.briefing, self.quote)
            except Exception:  # noqa: BLE001
                intent = None
            if intent not in _COMPLAINT_INTENTS and intent not in _LEAD_INTENTS:
                return ToolResult(
                    ok=False,
                    content='{"error": "not a complaint"}',
                    error=(
                        "escalate rejected: this is not a complaint or escalation request. "
                        "Answer the customer's question directly with the information you have."
                    ),
                )
        return self.services.tools.run(name, args, conversation_id=conversation_id)

    def _escalate_blocked(self) -> bool:
        """True when the model tried to escalate a message that is not a complaint."""
        return any("escalate rejected" in (e or "") for e in self.tool_errors)

    def _deterministic_lead(self) -> bool:
        """Write the lead down from the message itself, with no model involved.

        A lead is the one thing this agent must never drop, and a 7B model asked
        to answer *and* call a tool forgets the call often enough to matter: it
        quotes the customer's details back and never saves them. Every field is
        surfaceable from the text, so the loop reads them itself, cheaply and
        identically every time, and leaves the model the job it is good at,
        which is writing the reply. The tool still validates and normalises
        everything, so a bad extraction is rejected, not stored.
        """
        if self.briefing is None or getattr(self.services, "leads", None) is None:
            return False
        question = self.briefing.question or ""
        if not _looks_like_lead(question):
            return False
        name_match = _NAME_AFTER_INTRO_RE.search(question)
        if not name_match:
            return False
        name = name_match.group(1).strip(" .,;:")
        if not name:
            return False

        digits = ascii_digits(question)
        phones = list(extract_phones(digits))
        email_match = _EMAIL_RE.search(question)
        if not phones and not email_match:
            return False

        # The need is the request itself, with the contact details taken back out
        # so the team reads a requirement rather than a line of PII.
        need = _EMAIL_RE.sub(" ", question)
        need = _PII_REDACT_RE.sub(" ", need)
        need = re.sub(r"\s+", " ", need).strip(" .,;:")[:_MAX_NEED_CHARS]

        args: dict[str, Any] = {"name": name, "need": need or question[:_MAX_NEED_CHARS]}
        if phones:
            args["phone"] = phones[0]
        if email_match:
            args["email"] = email_match.group(0)

        result = self.services.tools.run("save_lead", args, conversation_id=self.conversation_id)
        if not result.ok:
            self.tool_errors.append(f"save_lead: {result.error or 'call rejected'}")
            return False
        self.tool_calls_seen += 1
        self.lead_saved = True
        return True

    def _step_limit_reply(self) -> str:
        """The budget is gone and there is no clean reply: hand the turn to the team.

        Failing silently would leave a customer with nothing, so the loop escalates
        on its own and answers with the guard's handoff text — unless the model was
        trying to escalate a message that is not a complaint, in which case it gets
        a normal answer instead.
        """
        if self._escalate_blocked():
            return self._clean_fallback_reply()
        self.step_limit_hit = True
        return self._escalate_and_reply("step limit reached for this message")

    def _clean_fallback_reply(self) -> str:
        """A normal answer when the model wasted its budget on rejected escalations."""
        self.step_limit_hit = True
        self.used_fallback_reply = True
        return self._handoff_reply()

    def _escalate_and_reply(self, reason: str) -> str:
        """Escalate on the model's behalf and let the guard write the reply.

        The model is not the one asking for a human here, so its sentence is not
        the last word: the guard's escalation template is the only text that
        reliably states the handoff and the policy behind it.
        """
        self.handoff = True
        self.loop_escalated = True
        self.used_fallback_reply = True
        result = self.services.tools.run(
            "escalate", {"reason": reason}, conversation_id=self.conversation_id
        )
        if not result.ok:
            self.tool_errors.append(f"escalate: {result.error or 'implicit escalate rejected'}")
        return self._handoff_reply()

    def _handoff_reply(self) -> str:
        verdict = GuardVerdict(
            ok=False,
            violations=["escalated: the agent asked for a human on the customer's behalf"],
            details={"escalated": True},
        )
        return self.services.guard.repair(
            "", verdict, briefing=self.briefing, quote=self.quote
        )

    def _rescue_prose_calls(self, content: str) -> str:
        """Run the tool calls the model wrote into its answer, and take them out.

        Asked to call a tool, a 7B model often answers `escalate {"reason": "..."}`
        in plain prose. Left alone the customer reads a JSON blob, the complaint
        is never flagged for a human, and a lead the model believes it saved is
        never written. The arguments are handed to the registry unchanged, so
        validation, the action log and the rule that only successful calls reach
        the response are exactly the same as for a structured call.
        """
        names = self.services.tools.names()
        if not names or not content:
            return content
        pattern = re.compile(
            rf"(?<![\w.])({'|'.join(re.escape(name) for name in names)})\s*([{{\[])",
            re.IGNORECASE,
        )
        decoder = json.JSONDecoder()
        cleaned = content
        for _ in range(_MAX_RESCUED_PROSE_CALLS):
            match = pattern.search(cleaned)
            if match is None:
                break
            name = match.group(1)
            start = match.start()
            # The model did try to call a tool, however badly. Count the attempt
            # before validating, so a malformed call stays a visible tool error
            # instead of being quietly replaced by the deterministic fallback.
            self.tool_attempted += 1
            try:
                arguments, end = decoder.raw_decode(cleaned, match.start(2))
            except ValueError:
                stop = cleaned.find("\n", start)
                cleaned = cleaned[:start] + cleaned[stop if stop != -1 else len(cleaned):]
                self.tool_errors.append(
                    f"{name}: the model wrote the call as text and the arguments could not be read"
                )
                continue
            cleaned = cleaned[:start] + cleaned[end:]
            self.tool_calls_seen += 1
            if not isinstance(arguments, dict):
                self.tool_errors.append(
                    f"{name}: the model wrote the call as text with arguments that are not an object"
                )
                continue
            result = self._run_tool(
                name, arguments, conversation_id=self.conversation_id
            )
            if not result.ok:
                self.tool_errors.append(f"{name}: {result.error or 'tool call rejected'}")
            elif name == "escalate":
                self.handoff = True
                self.loop_escalated = True
            elif name == "save_lead":
                self.lead_saved = True
        return cleaned

    # -- grounding ----------------------------------------------------------

    def _ground(self, draft: str) -> str:
        guard = self.services.guard
        language = self._language()
        limit = self.services.config.agent.max_reply_chars

        if self.step_limit_hit:
            return self._finalise(draft, language)

        text = _clean(self._rescue_prose_calls(draft))
        if self.loop_escalated:
            # The loop escalated on the model's behalf, so the model's sentence is
            # not the customer's answer; the guard's handoff text is.
            self.used_fallback_reply = True
            return self._finalise(self._handoff_reply(), language)

        text = _with_disclosure(text, language, self.is_first_turn, limit)
        if not text:
            self.used_fallback_reply = True
            return self._finalise(guard.repair("", None, briefing=self.briefing, quote=self.quote), language)

        verdict = guard.inspect(
            text, briefing=self.briefing, quote=self.quote, is_first_turn=self.is_first_turn
        )
        if verdict.ok:
            return self._finalise(text, language)
        return self._repair_reply(text, verdict, language, limit)

    def _repair_reply(self, text: str, verdict: GuardVerdict, language: str, limit: int) -> str:
        guard = self.services.guard
        if self.services.config.agent.allow_reply_repair and self.budget.remaining:
            self.guard_repairs = 1
            messages = self._context_messages()
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {"role": "user", "content": _REPAIR_INSTRUCTION.format(
                    problems="; ".join(verdict.violations)
                )}
            )
            # No tools on the repair call: the tool phase is over, and a repair
            # turn that started a new one could loop until the budget was gone.
            response = self._complete(messages, None)
            candidate = (
                _with_disclosure(
                    _clean(self._rescue_prose_calls(response.content)),
                    language,
                    self.is_first_turn,
                    limit,
                )
                if response is not None
                else ""
            )
            if self.loop_escalated:
                self.used_fallback_reply = True
                return self._finalise(self._handoff_reply(), language)
            if candidate:
                second = guard.inspect(
                    candidate,
                    briefing=self.briefing,
                    quote=self.quote,
                    is_first_turn=self.is_first_turn,
                )
                if second.ok:
                    return self._finalise(candidate, language)
                self.used_fallback_reply = True
                return self._finalise(self._deterministic(candidate, second), language)
        self.used_fallback_reply = True
        return self._finalise(self._deterministic(text, verdict), language)

    def _deterministic(self, draft: str, verdict: GuardVerdict) -> str:
        """The guard's safe rewrite, or the full template when it is only a stub."""
        repaired = self.services.guard.repair(
            draft, verdict, briefing=self.briefing, quote=self.quote
        )
        if _kept_enough(draft, repaired):
            return repaired
        return self.services.guard.repair(
            "", verdict, briefing=self.briefing, quote=self.quote
        )

    def _finalise(self, reply: str, language: str) -> str:
        """Last gate: length, disclosure, and one more inspection.

        Cheap, deterministic, and the reason a reply can always be returned.
        """
        limit = self.services.config.agent.max_reply_chars
        text = _with_disclosure(_clean(reply), language, self.is_first_turn, limit) or _HARD_FALLBACK
        verdict = self.services.guard.inspect(
            text, briefing=self.briefing, quote=self.quote, is_first_turn=self.is_first_turn
        )
        if verdict.ok:
            return text
        self.used_fallback_reply = True
        return (
            _with_disclosure(
                _clean(self._deterministic(text, verdict)), language, self.is_first_turn, limit
            )
            or _HARD_FALLBACK
        )

    def _language(self) -> str:
        return normalise_language(self.briefing.language if self.briefing is not None else None)

    # -- output -------------------------------------------------------------

    def _sources(self, reply: str) -> list[str]:
        """Valid citations, the model's own first, then what retrieval returned.

        Every id is checked against ``corpus.source_ids``; an id that data/ does
        not contain is dropped rather than passed on.
        """
        valid = self.services.corpus.source_ids
        canonical = {source_id.casefold(): source_id for source_id in valid}
        found: list[str] = []

        def add(candidate: str) -> None:
            if candidate in valid and candidate not in found:
                found.append(candidate)

        for raw in _SOURCE_RE.findall(reply or ""):
            add(raw if raw in valid else canonical.get(raw.casefold(), ""))

        if self.briefing is not None:
            resolved = self.briefing.resolved
            for hit in resolved.sku_hits:
                add(hit.sku.source_id)
            for section in resolved.sections:
                add(section.section.source_id)
        return found[:MAX_SOURCES]

    def _ok_actions(self) -> list[Action]:
        """Only successful calls are public; a rejected one is a diagnostic."""
        try:
            logged = self.services.tools.actions_for(self.conversation_id)
        except Exception:  # noqa: BLE001 - the response must still be built
            return []
        return [action for action in logged if action.ok]

    def _record(self, reply: str) -> None:
        conversation = self.conversation
        if conversation is None:
            return
        conversation.add_turn("user", self.message)
        conversation.add_turn("assistant", reply)
        conversation.reply_count += 1
        if self.briefing is not None:
            conversation.last_language = self.briefing.language
        if self.handoff:
            conversation.handoff = True

    def _outcome(
        self, reply: str, sources: list[str], actions: list[Action]
    ) -> AgentOutcome:
        return AgentOutcome(
            reply=reply,
            sources=sources,
            actions=actions,
            handoff=self.handoff,
            model_calls=self.budget.spent,
            prompt_tokens=self.budget.prompt_tokens,
            completion_tokens=self.budget.completion_tokens,
            tool_errors=list(self.tool_errors),
            guard_repairs=self.guard_repairs,
            intent_repairs=self.intent_repairs,
            step_limit_hit=self.step_limit_hit,
            used_fallback_reply=self.used_fallback_reply,
        )

    # -- the failure path ---------------------------------------------------

    def _safe_outcome(self) -> AgentOutcome:
        logger.error(
            "run_turn failed for conversation %r: %s\n%s",
            self.conversation_id,
            _scrub(str(sys.exc_info()[1])),
            _scrub(traceback.format_exc()),
        )
        briefing = self.briefing
        if briefing is None:
            try:
                briefing = self.services.retrieval.brief(self.message, [])
                self.briefing = briefing
            except Exception:  # noqa: BLE001 - the guard can cope without one
                briefing = None
        try:
            reply = self.services.guard.repair(
                "", None, briefing=briefing, quote=self.quote
            )
        except Exception:  # noqa: BLE001 - the guard itself is broken
            reply = _HARD_FALLBACK
        limit = self.services.config.agent.max_reply_chars
        reply = (
            _with_disclosure(_clean(reply), normalise_language(
                briefing.language if briefing is not None else None
            ), self.is_first_turn, limit)
            or _HARD_FALLBACK
        )
        self.used_fallback_reply = True
        self.handoff = True
        try:
            sources = self._sources(reply)
        except Exception:  # noqa: BLE001
            sources = []
        actions = self._ok_actions()
        try:
            self._record(reply)
        except Exception:  # noqa: BLE001 - never lose the reply over bookkeeping
            logger.error("could not record turns for conversation %r", self.conversation_id)
        return self._outcome(reply, sources, actions)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def run_turn(message: str, conversation_id: str, services: Services) -> AgentOutcome:
    """Answer one customer turn. The single entry point the HTTP layer calls.

    Always returns an outcome: on an unexpected failure the reply comes from
    the guard's templates, the turn is handed to the shop team, and a masked
    traceback is logged.
    """
    run = _Run(message=message, conversation_id=conversation_id, services=services)
    try:
        return run.execute()
    except Exception:  # noqa: BLE001 - a chat turn must never 500
        return run.safe_outcome()


def run_message(message: str, conversation_id: str, services: Services) -> AgentOutcome:
    """Alias for :func:`run_turn` for callers that speak in messages."""
    return run_turn(message, conversation_id, services)
