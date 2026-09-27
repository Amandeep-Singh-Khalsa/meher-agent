"""The graded requirement: a tool call with invalid arguments.

A 7B model produces malformed tool calls constantly. Every one of them has to
come back as a `tool` error the model can read and recover from, inside the same
turn, without a 500 and without the bad call ever reaching the public `actions`
array of the response.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from meher_agent.agent.conversation import ConversationStore
from meher_agent.agent.loop import run_turn
from meher_agent.agent.services import Services
from meher_agent.config import Config, load_config
from meher_agent.data.corpus import Corpus, load_corpus
from meher_agent.grounding.billing import BillingEngine
from meher_agent.grounding.guard import ReplyGuard
from meher_agent.llm.client import LLMResponse, ToolCallRequest, Usage, _parse_arguments
from meher_agent.retrieval.pipeline import RetrievalPipeline
from meher_agent.tools.registry import ToolRegistry
from meher_agent.tools.store import LeadStore
from meher_agent.types import AgentOutcome

LEAD = "We need 30 large gift boxes for our office Diwali party on 3 November. I'm Ritu Malhotra, ritu.m@example.com"
VALID = {"name": "Ritu Malhotra", "need": "30 large gift boxes", "email": "ritu.m@example.com"}
FINAL = "Noted, our team will call you back to confirm the order within one working day."


def _response(content: str, calls: list[ToolCallRequest] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=calls or [],
        usage=Usage(prompt_tokens=50, completion_tokens=10, source="endpoint"),
        finish_reason="tool_calls" if calls else "stop",
        raw={},
    )


def _tool(name: str, arguments: Any, call_id: str = "call_1") -> LLMResponse:
    return _response("", [ToolCallRequest(id=call_id, name=name, arguments=arguments)])


def broken_arguments(raw: str) -> dict[str, Any]:
    """What LLMClient hands the registry when the model emits broken JSON."""
    arguments, problem = _parse_arguments(raw)
    assert problem is not None
    assert arguments == {}
    return arguments


class ScriptedLLM:
    """A model that does exactly what the script says, and records the requests."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict], *, tools=None, tool_choice=None) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        if not self.script:
            raise AssertionError("the model was called more times than the script allows")
        return self.script.pop(0)

    def tool_results(self, index: int) -> list[dict[str, Any]]:
        return [
            json.loads(message["content"])
            for message in self.calls[index]
            if message.get("role") == "tool"
        ]


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def corpus(config: Config) -> Corpus:
    return load_corpus(config.runtime.data_dir)


def run_with(
    script: list[LLMResponse],
    config: Config,
    corpus: Corpus,
    *,
    message: str = LEAD,
) -> tuple[AgentOutcome, ScriptedLLM, Services]:
    llm = ScriptedLLM(script)
    leads = LeadStore()
    services = Services(
        config=config,
        corpus=corpus,
        retrieval=RetrievalPipeline(corpus, config),
        billing=BillingEngine(corpus, config),
        guard=ReplyGuard(corpus, config),
        llm=llm,
        tools=ToolRegistry(corpus, config, leads=leads),
        leads=leads,
        conversations=ConversationStore(max_conversations=8, max_history_turns=12),
    )
    return run_turn(message, "c1", services), llm, services


# --------------------------------------------------------------------------
# The graded cases
# --------------------------------------------------------------------------

INVALID_CALLS: list[tuple[str, str, dict[str, Any]]] = [
    ("no_contact", "contact", {"name": "X", "need": "Y"}),
    (
        "phone_too_short",
        "10 digits",
        {"name": "Ritu Malhotra", "need": "30 gift boxes", "phone": "12345"},
    ),
    (
        "phone_bad_prefix",
        "starting with 6, 7, 8 or 9",
        {"name": "Ritu Malhotra", "need": "30 gift boxes", "phone": "1234567890"},
    ),
    (
        "email_malformed",
        "not a valid email",
        {"name": "Ritu Malhotra", "need": "30 gift boxes", "email": "not-an-email"},
    ),
    (
        "date_not_a_calendar_date",
        "not a real calendar date",
        {
            "name": "Ritu Malhotra",
            "need": "30 gift boxes",
            "email": "ritu.m@example.com",
            "date": "31/02/2026",
        },
    ),
    (
        "name_is_a_number",
        "name is required and must be a string",
        {"name": 12345, "need": "30 gift boxes", "email": "ritu.m@example.com"},
    ),
    (
        "need_is_whitespace",
        "need is required",
        {"name": "Ritu Malhotra", "need": "   ", "email": "ritu.m@example.com"},
    ),
    ("arguments_were_malformed_json", "name is required", broken_arguments('{"name": "X", "need":')),
    ("arguments_were_a_list", "must be a JSON object", ["Ritu Malhotra", "30 gift boxes"]),
]


@pytest.mark.parametrize(
    ("name", "expected_error", "arguments"),
    INVALID_CALLS,
    ids=[case[0] for case in INVALID_CALLS],
)
def test_invalid_call_is_reported_to_the_model_and_then_recovered(
    config: Config, corpus: Corpus, name: str, expected_error: str, arguments: Any
) -> None:
    outcome, llm, services = run_with(
        [
            _tool("save_lead", arguments),
            _tool("save_lead", VALID, call_id="call_2"),
            _response(FINAL),
        ],
        config,
        corpus,
    )

    assert len(outcome.tool_errors) == 1, name
    assert expected_error in outcome.tool_errors[0], name
    assert outcome.tool_errors[0].startswith("save_lead:"), name

    results = llm.tool_results(2)
    assert [result["ok"] for result in results] == [False, True], name
    assert results[0]["hint"], name
    assert "error" in results[0], name

    assert [action.type for action in outcome.actions] == ["save_lead"], name
    assert outcome.actions[0].args == {"name": "Ritu Malhotra", "need": "30 large gift boxes", "email": "ritu.m@example.com"}
    assert outcome.actions[0].ok is True

    assert outcome.reply.strip(), name
    assert "AI" in outcome.reply, name
    assert outcome.handoff is False, name
    assert outcome.model_calls == 3, name
    assert outcome.step_limit_hit is False, name
    assert outcome.used_fallback_reply is False, name

    conversation = services.conversations.get("c1")
    assert conversation.reply_count == 1, name
    assert len(conversation.turns) == 2, name


@pytest.mark.parametrize(
    ("name", "expected_error", "arguments"),
    INVALID_CALLS,
    ids=[case[0] for case in INVALID_CALLS],
)
def test_invalid_call_never_reaches_the_public_actions(
    config: Config, corpus: Corpus, name: str, expected_error: str, arguments: Any
) -> None:
    outcome, _, services = run_with(
        [
            _tool("save_lead", arguments),
            _tool("save_lead", VALID, call_id="call_2"),
            _response(FINAL),
        ],
        config,
        corpus,
    )

    public = [action.to_public() for action in outcome.actions]
    assert public == [
        {"type": "save_lead", "args": VALID},
    ], name
    for action in outcome.actions:
        assert action.ok is True, name

    logged = services.tools.actions()
    assert any(action.ok is False for action in logged), name
    assert len(logged) == 2, name
    assert [lead.email for lead in services.leads.all()] == ["ritu.m@example.com"], name


def test_unknown_tool_name_is_a_tool_error(config: Config, corpus: Corpus) -> None:
    outcome, llm, _ = run_with(
        [
            _tool("book_table", {"date": "2026-11-03"}),
            _tool("save_lead", VALID, call_id="call_2"),
            _response(FINAL),
        ],
        config,
        corpus,
    )

    assert len(outcome.tool_errors) == 1
    assert "unknown tool" in outcome.tool_errors[0]
    assert outcome.tool_errors[0].startswith("book_table:")
    assert llm.tool_results(1)[0]["ok"] is False
    assert [action.type for action in outcome.actions] == ["save_lead"]
    assert outcome.handoff is False


def test_escalate_with_a_blank_reason_is_a_tool_error(config: Config, corpus: Corpus) -> None:
    outcome, llm, _ = run_with(
        [
            _tool("escalate", {"reason": "   "}),
            _tool("escalate", {"reason": "the customer wants a human"}, call_id="call_2"),
            _response("I have passed this to the shop team."),
        ],
        config,
        corpus,
        message="The gift box arrived crushed and I want a refund.",
    )

    assert len(outcome.tool_errors) == 1
    assert "reason is required" in outcome.tool_errors[0]
    assert llm.tool_results(1)[0]["ok"] is False
    assert [action.type for action in outcome.actions] == ["escalate"]
    assert outcome.handoff is True


def test_two_invalid_calls_then_a_good_one_stays_inside_the_budget(
    config: Config, corpus: Corpus
) -> None:
    outcome, llm, _ = run_with(
        [
            _tool("save_lead", {"name": "Ritu Malhotra", "need": "30 gift boxes"}),
            _tool("save_lead", {"name": "Ritu Malhotra", "need": "30 gift boxes", "phone": "12345"}, call_id="call_2"),
            _tool("save_lead", VALID, call_id="call_3"),
            _response(FINAL),
        ],
        config,
        corpus,
    )

    assert len(llm.calls) == config.llm.max_steps
    assert outcome.model_calls == config.llm.max_steps
    assert len(outcome.tool_errors) == 2
    assert [action.type for action in outcome.actions] == ["save_lead"]
    assert outcome.step_limit_hit is False
    assert outcome.reply.strip()


def test_a_turn_of_nothing_but_invalid_calls_ends_in_a_handoff(
    config: Config, corpus: Corpus
) -> None:
    """Four rejected calls and no budget left: the turn is escalated, not lost."""
    bad = _tool("save_lead", {"name": "Ritu Malhotra", "need": "30 gift boxes"}, call_id="c")
    outcome, llm, _ = run_with([bad, bad, bad, bad], config, corpus)

    assert len(llm.calls) == config.llm.max_steps
    assert outcome.model_calls == config.llm.max_steps
    assert outcome.step_limit_hit is True
    assert outcome.handoff is True
    assert [action.type for action in outcome.actions] == ["escalate"]
    assert "AI" in outcome.reply
    assert 1 <= len(outcome.reply) <= 1200
    assert len(outcome.tool_errors) == config.llm.max_steps
    assert outcome.used_fallback_reply is True
