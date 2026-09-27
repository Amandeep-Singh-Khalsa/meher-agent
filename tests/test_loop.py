"""The agent loop: budget, tool round trips, intent repair, guard repair, state.

Every test drives a scripted fake model, so the suite is offline and fast. The
properties under test are the ones the rest of the system assumes: the
``max_steps`` cap holds, a reply always exists and always discloses the AI on the
first turn, a rejected tool call never reaches the public ``actions`` array, and
the conversation advances.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from meher_agent.agent.conversation import ConversationStore
from meher_agent.agent.loop import run_turn
from meher_agent.agent.services import Services
from meher_agent.config import Config, load_config
from meher_agent.data.corpus import Corpus, load_corpus
from meher_agent.grounding.billing import BillingEngine
from meher_agent.grounding.guard import ReplyGuard
from meher_agent.llm.client import LLMResponse, ToolCallRequest, Usage
from meher_agent.llm.prompts import NUDGE_TOOL
from meher_agent.retrieval.pipeline import RetrievalPipeline
from meher_agent.tools.registry import ToolRegistry
from meher_agent.tools.store import LeadStore
from meher_agent.types import AgentOutcome

# --------------------------------------------------------------------------
# Fake model
# --------------------------------------------------------------------------


def text_reply(text: str, *, prompt: int = 100, completion: int = 20) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion, source="endpoint"),
        finish_reason="stop",
        raw={},
    )


def tool_call(
    name: str, arguments: dict[str, Any], *, call_id: str = "call_1", prompt: int = 100, completion: int = 5
) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion, source="endpoint"),
        finish_reason="tool_calls",
        raw={},
    )


class FakeLLM:
    """Returns a scripted queue of responses and records every request."""

    def __init__(self, responses: list[LLMResponse] | None = None, *, repeat: LLMResponse | None = None) -> None:
        self._queue = list(responses or [])
        self._repeat = repeat
        self.calls: list[list[dict[str, Any]]] = []
        self.tools_seen: list[Any] = []

    def complete(self, messages: list[dict], *, tools=None, tool_choice=None) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        self.tools_seen.append(tools)
        if self._queue:
            return self._queue.pop(0)
        if self._repeat is not None:
            return self._repeat
        return text_reply("EXHAUSTED: the fake had no scripted response left", prompt=0, completion=0)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def roles(self, index: int) -> list[str]:
        return [str(message.get("role")) for message in self.calls[index]]


class ExplodingLLM:
    """A model that fails the way a dead endpoint does."""

    def complete(self, messages: list[dict], *, tools=None, tool_choice=None) -> LLMResponse:
        raise RuntimeError("connection refused: http://localhost:11434/v1/chat/completions")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def corpus(config: Config) -> Corpus:
    return load_corpus(config.runtime.data_dir)


def make_services(llm: Any, config: Config, corpus: Corpus) -> Services:
    leads = LeadStore()
    return Services(
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


def turn(
    message: str,
    responses: list[LLMResponse],
    *,
    config: Config,
    corpus: Corpus,
    conversation_id: str = "c1",
    repeat: LLMResponse | None = None,
) -> tuple[AgentOutcome, FakeLLM, Services]:
    llm = FakeLLM(responses, repeat=repeat)
    services = make_services(llm, config, corpus)
    return run_turn(message, conversation_id, services), llm, services


PRICE_QUESTION = "How much is 500 g of sugar-free kaju katli?"
GOOD_PRICE_REPLY = (
    "Sugar-free Kaju Katli, 500 g, is Rs 780. That pack keeps for 10 days, so tell me the "
    "quantity you need and I can work out the total. prices.csv#KKSF-500"
)
COMPLAINT = "The gift box you delivered 30 minutes ago is completely crushed. Very disappointed."
LEAD = "We need 30 large gift boxes for our office Diwali party on 3 November. I'm Ritu Malhotra, ritu.m@example.com"
#: A contact the deterministic capture deliberately will not own: it has contact
#: details but no self-introduction and no ordering word, so the loop cannot read a
#: name out of it. Used by the tests that are about the *model's* tool channel.
LEAD_NO_CAPTURE = "Kindly get back to Ritu Malhotra at ritu.m@example.com about 30 gift boxes for 3 November."


# --------------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------------


def test_english_price_question_answers_in_one_call(config, corpus) -> None:
    outcome, llm, _ = turn(PRICE_QUESTION, [text_reply(GOOD_PRICE_REPLY)], config=config, corpus=corpus)

    assert llm.call_count == 1
    assert outcome.model_calls == 1
    assert "780" in outcome.reply
    assert "AI" in outcome.reply
    assert outcome.sources, "a price answer must cite something"
    assert set(outcome.sources) <= corpus.source_ids
    assert outcome.actions == []
    assert outcome.handoff is False
    assert outcome.used_fallback_reply is False
    assert outcome.guard_repairs == 0
    assert outcome.intent_repairs == 0
    assert outcome.step_limit_hit is False
    assert outcome.tool_errors == []
    assert 1 <= len(outcome.reply) <= 1200


def test_lead_turn_saves_one_lead_with_post_validation_args(config, corpus) -> None:
    outcome, llm, services = turn(
        LEAD,
        [
            tool_call(
                "save_lead",
                {
                    "name": "Ritu Malhotra",
                    "need": "30 large gift boxes for the office Diwali party",
                    "email": "Ritu.M@Example.COM",
                    "quantity": "30",
                },
            ),
            text_reply(
                "Thank you, I have noted your order and the team will call you back to confirm "
                "the 3 days' notice and the 30% advance."
            ),
        ],
        config=config,
        corpus=corpus,
    )

    assert llm.call_count == 2
    assert [action.type for action in outcome.actions] == ["save_lead"]
    assert outcome.actions[0].args["email"] == "ritu.m@example.com"
    assert outcome.actions[0].to_public() == {
        "type": "save_lead",
        "args": {
            "name": "Ritu Malhotra",
            "need": "30 large gift boxes for the office Diwali party",
            "email": "ritu.m@example.com",
            "quantity": "30",
        },
    }
    assert outcome.handoff is False
    assert outcome.tool_errors == []
    assert [lead.email for lead in services.leads.all()] == ["ritu.m@example.com"]


def test_complaint_turn_escalates_and_says_the_team_will_reply(config, corpus) -> None:
    outcome, llm, _ = turn(
        COMPLAINT,
        [
            tool_call("escalate", {"reason": "the gift box arrived crushed"}),
            text_reply(
                "I am very sorry about the crushed box. Please send a photo of the box as it "
                "arrived, with your name and the delivery time, and our team will follow up with "
                "you by email within one working day."
            ),
        ],
        config=config,
        corpus=corpus,
    )

    assert [action.type for action in outcome.actions] == ["escalate"]
    assert outcome.handoff is True
    assert "email" in outcome.reply
    assert outcome.tool_errors == []
    assert outcome.model_calls == 2


def test_price_question_calls_no_tool(config, corpus) -> None:
    """The graded oos-01 / price shape: a plain question, and no action at all."""
    outcome, llm, _ = turn(
        "What is the price of 1 kg motichoor laddoo?",
        [text_reply("Motichoor Laddoo, 1 kg, is Rs 560. prices.csv#ML-1000")],
        config=config,
        corpus=corpus,
    )

    assert outcome.actions == []
    assert outcome.model_calls == 1
    assert outcome.intent_repairs == 0
    assert "560" in outcome.reply
    assert "prices.csv#ML-1000" in outcome.sources


# --------------------------------------------------------------------------
# The cap
# --------------------------------------------------------------------------


def test_step_cap_stops_at_max_steps_and_escalates(config, corpus) -> None:
    """A model that only ever calls a tool must not be able to spend forever."""
    loop = tool_call("save_lead", {"name": "X", "need": "Y"})
    outcome, llm, _ = turn(
        PRICE_QUESTION,
        [],
        config=config,
        corpus=corpus,
        repeat=loop,
    )

    assert llm.call_count == config.llm.max_steps
    assert outcome.model_calls == config.llm.max_steps
    assert outcome.step_limit_hit is True
    assert outcome.handoff is True
    assert [action.type for action in outcome.actions] == ["escalate"]
    assert outcome.used_fallback_reply is True
    assert outcome.reply
    assert len(outcome.tool_errors) == config.llm.max_steps


def test_model_calls_never_exceed_max_steps(config, corpus) -> None:
    budget = config.llm.max_steps
    scripts: list[tuple[str, list[LLMResponse], LLMResponse | None]] = [
        (PRICE_QUESTION, [text_reply(GOOD_PRICE_REPLY)], None),
        (PRICE_QUESTION, [text_reply("It costs Rs 9999, about 2 hours ago.")] * 4, None),
        (LEAD, [text_reply("Sure, who is this for?")] * 4, None),
        (COMPLAINT, [], tool_call("escalate", {"reason": "again"})),
        (PRICE_QUESTION, [text_reply("")], None),
    ]
    for message, responses, repeat in scripts:
        outcome, _, _ = turn(message, responses, config=config, corpus=corpus, repeat=repeat)
        assert 1 <= outcome.model_calls <= budget, message
        assert outcome.reply, message


def test_budget_is_shared_with_the_repair_calls(config, corpus) -> None:
    """Tool round trips and the guard repair all draw on the same four calls."""
    outcome, llm, _ = turn(
        PRICE_QUESTION,
        [
            tool_call("escalate", {"reason": "not needed"}),
            tool_call("escalate", {"reason": "still not needed"}),
            text_reply("It is Rs 7,00,000 for that, roughly."),
            text_reply("Actually it is Rs 850."),
        ],
        config=config,
        corpus=corpus,
    )

    assert outcome.model_calls == config.llm.max_steps
    assert outcome.guard_repairs == 1
    assert "7,00,000" not in outcome.reply
    assert "850" not in outcome.reply


# --------------------------------------------------------------------------
# Tool errors
# --------------------------------------------------------------------------


def test_failed_tool_call_then_a_successful_retry(config, corpus) -> None:
    outcome, llm, _ = turn(
        LEAD,
        [
            tool_call("save_lead", {"name": "Ritu Malhotra", "need": "30 gift boxes"}),
            tool_call(
                "save_lead",
                {
                    "name": "Ritu Malhotra",
                    "need": "30 gift boxes",
                    "email": "ritu.m@example.com",
                },
                call_id="call_2",
            ),
            text_reply("Noted, the team will call you back to confirm the order."),
        ],
        config=config,
        corpus=corpus,
    )

    assert len(outcome.tool_errors) == 1
    assert "contact" in outcome.tool_errors[0]
    assert [action.type for action in outcome.actions] == ["save_lead"]
    assert outcome.actions[0].args["email"] == "ritu.m@example.com"
    assert outcome.model_calls == 3

    tool_messages = [m for m in llm.calls[2] if m.get("role") == "tool"]
    assert [json.loads(m["content"])["ok"] for m in tool_messages] == [False, True]


def test_rejected_tool_call_never_reaches_the_public_actions(config, corpus) -> None:
    outcome, _, _ = turn(
        LEAD,
        [
            tool_call("save_lead", {"name": "Ritu", "need": "boxes", "phone": "12345"}),
            text_reply("Could you read your phone number back to me?"),
        ],
        config=config,
        corpus=corpus,
    )

    assert outcome.actions == []
    assert len(outcome.tool_errors) == 1
    assert "phone" in outcome.tool_errors[0]


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


def test_guard_violation_triggers_exactly_one_repair_call(config, corpus) -> None:
    outcome, llm, _ = turn(
        PRICE_QUESTION,
        [
            text_reply("Sugar-free Kaju Katli 500 g is Rs 7,00,000, my favourite. prices.csv#KKSF-500"),
            text_reply("Sugar-free Kaju Katli 500 g is Rs 780. prices.csv#KKSF-500"),
        ],
        config=config,
        corpus=corpus,
    )

    assert outcome.guard_repairs == 1
    assert llm.call_count == 2
    assert outcome.model_calls == 2
    assert outcome.used_fallback_reply is False
    assert "7,00,000" not in outcome.reply
    assert "780" in outcome.reply
    assert "invented_amount" in llm.calls[1][-1]["content"]


def test_second_guard_failure_falls_back_to_the_template(config, corpus) -> None:
    outcome, llm, _ = turn(
        PRICE_QUESTION,
        [
            text_reply("That is Rs 7,00,000 for sure."),
            text_reply("Actually it is Rs 850, honestly."),
        ],
        config=config,
        corpus=corpus,
    )

    assert llm.call_count == 2
    assert outcome.guard_repairs == 1
    assert outcome.used_fallback_reply is True
    assert "7,00,000" not in outcome.reply
    assert "850" not in outcome.reply
    assert "AI" in outcome.reply
    assert 1 <= len(outcome.reply) <= 1200


def test_overlong_reply_is_clamped_to_the_limit(config, corpus) -> None:
    outcome, llm, _ = turn(
        PRICE_QUESTION,
        [text_reply("Sugar-free Kaju Katli is Rs 780. " + "We also stock namkeen. " * 90)],
        config=config,
        corpus=corpus,
    )

    assert len(outcome.reply) <= config.agent.max_reply_chars
    assert "AI" in outcome.reply
    assert llm.call_count <= config.llm.max_steps


def test_reply_repair_can_be_switched_off(config, corpus) -> None:
    strict = replace(
        config, agent=replace(config.agent, allow_reply_repair=False)
    )
    outcome, llm, _ = turn(
        PRICE_QUESTION,
        [text_reply("Sugar-free Kaju Katli 500 g is Rs 7,00,000.")],
        config=strict,
        corpus=corpus,
    )

    assert llm.call_count == 1
    assert outcome.guard_repairs == 0
    assert outcome.used_fallback_reply is True
    assert "7,00,000" not in outcome.reply
    assert "780" in outcome.reply


def test_empty_model_reply_falls_back_to_the_template(config, corpus) -> None:
    outcome, llm, _ = turn(PRICE_QUESTION, [text_reply("   ")], config=config, corpus=corpus)

    assert llm.call_count == 1
    assert outcome.reply.strip()
    assert "AI" in outcome.reply
    assert outcome.used_fallback_reply is True
    assert len(outcome.reply) <= config.agent.max_reply_chars


# --------------------------------------------------------------------------
# The AI disclosure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "draft", "expected"),
    [
        (
            "What time do you close today?",
            "We close at 10 pm every day, at 14 Central Market, Road No. 7, Jaipur.",
            "(AI assistant)",
        ),
        (
            "क्या आप 12 किलोमीटर दूर डिलीवरी करते हैं?",
            "हम दुकान से 8 किमी के भीतर डिलीवरी करते हैं।",
            "AI सहायक",
        ),
        (
            "Bhaiya kaju katli ka price kya hai?",
            "Bhaiya 1 kg Kaju Katli ka price Rs 1,200 hai, aur 8 km ke andar delivery hoti hai.",
            "(AI assistant)",
        ),
    ],
)
def test_first_reply_always_discloses_that_it_is_an_ai(
    config, corpus, message: str, draft: str, expected: str
) -> None:
    outcome, _, _ = turn(message, [text_reply(draft)], config=config, corpus=corpus)

    assert "AI" in outcome.reply
    assert expected in outcome.reply
    assert outcome.model_calls == 1
    assert len(outcome.reply) <= config.agent.max_reply_chars


def test_ai_is_not_appended_to_every_turn(config, corpus) -> None:
    services = make_services(
        FakeLLM([text_reply("It is Rs 780, the same as before.")]), config, corpus
    )
    run_turn(PRICE_QUESTION, "c1", services)
    services.llm = FakeLLM([text_reply("It is Rs 780, unchanged.")])
    second = run_turn("And with delivery?", "c1", services)

    assert "(AI assistant)" not in second.reply
    assert "AI" not in second.reply


def test_overlong_first_reply_keeps_the_disclosure_inside_the_limit(config, corpus) -> None:
    outcome, _, _ = turn(
        PRICE_QUESTION,
        [text_reply("Sugar-free Kaju Katli is Rs 780. " + "Namkeen too. " * 120)],
        config=config,
        corpus=corpus,
    )

    assert len(outcome.reply) <= config.agent.max_reply_chars
    assert "AI" in outcome.reply


# --------------------------------------------------------------------------
# Intent repair
# --------------------------------------------------------------------------


def test_intent_repair_spends_one_call_when_no_tool_was_called(config, corpus) -> None:
    """A lead the model tried to save but got wrong is nudged once.

    The nudge path is exercised on a lead where the model attempts a tool call
    that the registry rejects. A complaint the model ignored is now escalated
    deterministically, which is covered by
    ``test_a_complaint_the_model_ignores_is_escalated_deterministically``.
    """
    outcome, llm, _ = turn(
        LEAD,
        [
            tool_call("save_lead", {"name": "Ritu", "need": "boxes", "phone": "12345"}),
            text_reply("Let me note that down."),
            tool_call(
                "save_lead",
                {"name": "Ritu Malhotra", "need": "30 gift boxes", "email": "ritu.m@example.com"},
                call_id="call_nudge",
            ),
            text_reply("Noted. Our team will confirm the 3 days' notice and the advance by email."),
        ],
        config=config,
        corpus=corpus,
    )

    assert outcome.intent_repairs == 1
    assert outcome.model_calls == 4
    assert [action.type for action in outcome.actions] == ["save_lead"]
    assert llm.calls[2][-1]["content"] == NUDGE_TOOL


def test_a_complaint_the_model_ignores_is_escalated_deterministically(config, corpus) -> None:
    """The model answered but never escalated. The guard detected the complaint,
    so the loop escalates on its behalf — the same way a lead is captured when
    the model forgets the tool. The model's reply is discarded because the guard
    writes the apology and handoff.
    """
    outcome, llm, services = turn(
        COMPLAINT,
        [text_reply("I am sorry to hear that. Let me check with the team.")],
        config=config,
        corpus=corpus,
    )

    assert [action.type for action in outcome.actions] == ["escalate"]
    assert outcome.handoff is True
    assert services.tools.handoff_for("c1") is True
    assert outcome.model_calls == 1
    assert outcome.intent_repairs == 0
    assert "team" in outcome.reply.lower() or "call" in outcome.reply.lower()
    assert outcome.used_fallback_reply is True


def test_a_lead_the_model_forgets_is_saved_deterministically(config, corpus) -> None:
    """The model answered and never wrote the lead down, so the loop saves it.

    This is the path a 7B model actually takes: it quotes the customer's details
    back and forgets the tool. The lead must not be lost, so the loop reads the
    name and contact out of the message itself.
    """
    outcome, llm, services = turn(
        LEAD,
        [text_reply("Ritu, I have your order of 30 large gift boxes. The total is Rs 43,500.")],
        config=config,
        corpus=corpus,
    )

    assert [action.type for action in outcome.actions] == ["save_lead"]
    leads = services.leads.all()
    assert [lead.name for lead in leads] == ["Ritu Malhotra"]
    assert [lead.email for lead in leads] == ["ritu.m@example.com"]
    # No model call is spent on it, and the customer is not asked again.
    assert outcome.model_calls == 1
    assert outcome.intent_repairs == 0
    assert "Rs 43,500" in outcome.reply
    # The contact details are written to the lead, not echoed into the "need".
    assert "ritu.m@example.com" not in leads[0].need


def test_a_rejected_lead_call_asks_rather_than_guessing(config, corpus) -> None:
    """A call the registry rejected is *not* quietly replaced by a guessed one.

    The loop asks the customer to read the number back instead. Inventing a lead
    the customer never gave would be worse than losing one turn.
    """
    outcome, _, services = turn(
        LEAD,
        [
            tool_call("save_lead", {"name": "Ritu", "need": "boxes", "phone": "12345"}),
            text_reply("Could you read your phone number back to me?"),
        ],
        config=config,
        corpus=corpus,
    )

    assert outcome.actions == []
    assert services.leads.all() == []
    assert len(outcome.tool_errors) == 1
    assert "phone" in outcome.tool_errors[0]


def test_a_malformed_call_on_a_non_capturable_lead_is_reported(config, corpus) -> None:
    """With no name in the text to fall back on, the error is all there is.

    ``LEAD_NO_CAPTURE`` has contact details but no self-introduction, so the loop
    cannot invent a name and the rejected call is what the turn reports.
    """
    outcome, _, services = turn(
        LEAD_NO_CAPTURE,
        [text_reply('Noted. save_lead {"name": "Ritu Malhotra", "need": "30 gift boxes", "phone": "12345"}')],
        config=config,
        corpus=corpus,
    )

    assert outcome.actions == []
    assert services.leads.all() == []
    assert len(outcome.tool_errors) == 1
    assert "10 digits" in outcome.tool_errors[0]


def test_intent_repair_is_skipped_when_a_tool_already_fired(config, corpus) -> None:
    outcome, llm, _ = turn(
        COMPLAINT,
        [
            tool_call("escalate", {"reason": "crushed gift box"}),
            text_reply("I am sorry, and the team will follow up by email today."),
        ],
        config=config,
        corpus=corpus,
    )

    assert outcome.intent_repairs == 0
    assert outcome.model_calls == 2
    assert llm.call_count == 2


def test_intent_repair_can_be_switched_off(config, corpus) -> None:
    strict = replace(config, agent=replace(config.agent, allow_intent_repair=False))
    outcome, llm, _ = turn(
        LEAD, [text_reply("Sure, who should I ask for?")], config=strict, corpus=corpus
    )

    assert outcome.intent_repairs == 0
    assert llm.call_count == 1


# --------------------------------------------------------------------------
# Conversation state, history and tokens
# --------------------------------------------------------------------------


def test_multi_turn_history_reaches_the_second_turn(config, corpus) -> None:
    services = make_services(
        FakeLLM([text_reply("Motichoor Laddoo, 1 kg, is Rs 560. prices.csv#ML-1000")]),
        config,
        corpus,
    )
    run_turn("What is the price of 1 kg motichoor laddoo?", "c1", services)

    llm = FakeLLM([text_reply("The total is Rs 1,880. prices.csv#ML-1000")])
    services.llm = llm
    outcome = run_turn(
        "Make it 3 kg, and add 10 samosas. Deliver to 2 km away. Total?", "c1", services
    )

    joined = "\n".join(str(message.get("content")) for message in llm.calls[0])
    assert "What is the price of 1 kg motichoor laddoo?" in joined
    assert "Motichoor Laddoo, 1 kg, is Rs 560" in joined
    assert "1,880" in joined, "the computed quote must be in the second turn's context"
    assert "ML-1000" in joined
    assert "SM-1" in joined
    assert outcome.model_calls == 1
    assert "1,880" in outcome.reply


def test_conversation_state_advances(config, corpus) -> None:
    services = make_services(FakeLLM(), config, corpus)
    outcome = run_turn("क्या आप 12 किलोमीटर दूर डिलीवरी करते हैं?", "c1", services)
    conversation = services.conversations.get("c1")

    assert conversation.reply_count == 1
    assert conversation.last_language == "hindi"
    assert [turn_record.role for turn_record in conversation.turns] == ["user", "assistant"]
    assert conversation.turns[0].content == "क्या आप 12 किलोमीटर दूर डिलीवरी करते हैं?"
    assert conversation.turns[1].content == outcome.reply
    assert conversation.handoff is False


def test_handoff_is_recorded_on_the_conversation(config, corpus) -> None:
    services = make_services(
        FakeLLM(
            [
                tool_call("escalate", {"reason": "the gift box arrived crushed"}),
                text_reply("I am sorry, and the team will follow up by email today."),
            ]
        ),
        config,
        corpus,
    )
    run_turn(COMPLAINT, "c1", services)

    assert services.conversations.get("c1").handoff is True


def test_tokens_accumulate_across_every_call(config, corpus) -> None:
    outcome, llm, _ = turn(
        LEAD,
        [
            tool_call(
                "save_lead",
                {"name": "Ritu Malhotra", "need": "30 gift boxes", "email": "ritu.m@example.com"},
                prompt=100,
                completion=20,
            ),
            text_reply("Noted, the team will call you back.", prompt=250, completion=35),
        ],
        config=config,
        corpus=corpus,
    )

    assert llm.call_count == 2
    assert outcome.model_calls == 2
    assert outcome.prompt_tokens == 350
    assert outcome.completion_tokens == 55


def test_registry_is_reset_between_turns(config, corpus) -> None:
    services = make_services(
        FakeLLM(
            [
                tool_call(
                    "save_lead",
                    {"name": "Ritu Malhotra", "need": "boxes", "email": "ritu.m@example.com"},
                ),
                text_reply("Noted, the team will call you back."),
            ]
        ),
        config,
        corpus,
    )
    first = run_turn(LEAD, "c1", services)
    assert [action.type for action in first.actions] == ["save_lead"]

    services.llm = FakeLLM([text_reply("It is Rs 780. prices.csv#KKSF-500")])
    second = run_turn(PRICE_QUESTION, "c1", services)

    assert second.actions == []
    assert second.handoff is False
    assert second.tool_errors == []
    assert services.tools.actions() == []


# --------------------------------------------------------------------------
# Nothing escapes
# --------------------------------------------------------------------------


def test_llm_exception_does_not_escape(config, corpus) -> None:
    services = make_services(ExplodingLLM(), config, corpus)
    outcome = run_turn(PRICE_QUESTION, "c1", services)

    assert outcome.reply.strip()
    assert "AI" in outcome.reply
    assert outcome.handoff is True
    assert outcome.used_fallback_reply is True
    assert outcome.model_calls == 1  # the counter is taken before the call
    conversation = services.conversations.get("c1")
    assert conversation.reply_count == 1
    assert len(conversation.turns) == 2


def test_llm_exception_mid_tool_loop_does_not_escape(config, corpus) -> None:
    class HalfDead:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict], *, tools=None, tool_choice=None) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return tool_call(
                    "save_lead",
                    {"name": "Ritu Malhotra", "need": "b", "email": "ritu.m@example.com"},
                )
            raise RuntimeError("connection reset by peer")

    services = make_services(HalfDead(), config, corpus)
    outcome = run_turn(LEAD, "c1", services)

    assert outcome.reply.strip()
    assert outcome.handoff is True
    assert [action.type for action in outcome.actions] == ["save_lead"]


def test_broken_registry_does_not_500(config, corpus) -> None:
    class BrokenRegistry(ToolRegistry):
        def run(self, name, args, *, conversation_id):  # type: ignore[override]
            raise ZeroDivisionError("registry exploded")

    services = make_services(FakeLLM([tool_call("escalate", {"reason": "x"})]), config, corpus)
    services.tools = BrokenRegistry(corpus, config)

    outcome = run_turn(COMPLAINT, "c1", services)

    assert outcome.reply.strip()
    assert outcome.handoff is True
    assert "AI" in outcome.reply


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def test_sources_never_invent_an_id(config, corpus) -> None:
    outcome, _, _ = turn(
        PRICE_QUESTION,
        [text_reply("It is Rs 780. prices.csv#KKSF-500, prices.csv#NOPE-999, policies.md#discounts")],
        config=config,
        corpus=corpus,
    )

    assert set(outcome.sources) <= corpus.source_ids
    assert "prices.csv#NOPE-999" not in outcome.sources
    assert "policies.md#discounts" not in outcome.sources
    assert "prices.csv#KKSF-500" in outcome.sources


def test_sources_fall_back_to_the_briefing_evidence(config, corpus) -> None:
    outcome, _, _ = turn(
        PRICE_QUESTION, [text_reply("Sugar-free Kaju Katli 500 g is Rs 780.")],
        config=config, corpus=corpus,
    )

    assert outcome.sources
    assert "prices.csv#KKSF-500" in outcome.sources
    assert len(outcome.sources) <= 4


def test_model_citations_come_first(config, corpus) -> None:
    outcome, _, _ = turn(
        PRICE_QUESTION,
        [text_reply("It is Rs 780. policies.md#delivery, prices.csv#KKSF-500")],
        config=config,
        corpus=corpus,
    )

    assert outcome.sources[0] == "policies.md#delivery"
    assert "prices.csv#KKSF-500" in outcome.sources


def test_fallback_reply_still_cites_evidence(config, corpus) -> None:
    outcome, _, _ = turn(PRICE_QUESTION, [text_reply("   ")], config=config, corpus=corpus)

    assert outcome.used_fallback_reply is True
    assert outcome.sources
    assert set(outcome.sources) <= corpus.source_ids


# --------------------------------------------------------------------------
# Tool calls the model wrote into its answer instead of calling the tool
# --------------------------------------------------------------------------


def test_prose_escalate_is_run_and_never_shown_to_the_customer(config, corpus) -> None:
    """A complaint the model wrote as prose is escalated deterministically.

    The model wrote the escalation as text instead of calling the tool. The guard
    detected the complaint, so the loop escalates on its behalf. The model's
    prose is discarded because the guard writes the apology and handoff.
    """
    outcome, _, services = turn(
        COMPLAINT,
        [
            text_reply(
                'I am sorry about that. escalate {"reason": "the gift box arrived crushed"}'
            ),
        ],
        config=config,
        corpus=corpus,
    )

    assert outcome.model_calls == 1
    assert [action.type for action in outcome.actions] == ["escalate"]
    assert outcome.handoff is True
    assert services.tools.handoff_for("c1") is True
    assert "escalate {" not in outcome.reply
    assert "{" not in outcome.reply
    assert outcome.reply.strip()
    assert "AI" in outcome.reply
    assert outcome.used_fallback_reply is True
    assert outcome.step_limit_hit is False


def test_prose_save_lead_is_rescued_so_the_lead_is_really_written(config, corpus) -> None:
    outcome, _, services = turn(
        LEAD_NO_CAPTURE,
        [
            text_reply(
                'Noted, the team will call you back. save_lead {"name": "Ritu Malhotra", '
                '"need": "30 large gift boxes", "email": "ritu.m@example.com"}'
            ),
        ],
        config=config,
        corpus=corpus,
    )

    assert [action.type for action in outcome.actions] == ["save_lead"]
    assert [lead.email for lead in services.leads.all()] == ["ritu.m@example.com"]
    assert "save_lead" not in outcome.reply
    assert "Noted, the team will call you back." in outcome.reply
    assert outcome.tool_errors == []
    assert outcome.handoff is False


def test_prose_call_with_unreadable_json_is_stripped_and_reported(config, corpus) -> None:
    outcome, _, _ = turn(
        PRICE_QUESTION,
        [
            text_reply('It is Rs 780. save_lead {"name": '),
            text_reply("It is Rs 780."),
        ],
        config=config,
        corpus=corpus,
    )

    assert "save_lead" not in outcome.reply
    assert "{" not in outcome.reply
    assert outcome.reply.strip()
    assert len(outcome.tool_errors) == 1
    assert "could not be read" in outcome.tool_errors[0]
    assert outcome.actions == []
    assert outcome.handoff is False


def test_prose_call_with_non_object_arguments_is_rejected(config, corpus) -> None:
    outcome, _, services = turn(
        LEAD_NO_CAPTURE,
        [text_reply('Noted. save_lead ["Ritu Malhotra", "30 gift boxes"]')],
        config=config,
        corpus=corpus,
    )

    assert "save_lead" not in outcome.reply
    assert outcome.actions == []
    assert services.leads.all() == []
    assert len(outcome.tool_errors) == 1
    assert "not an object" in outcome.tool_errors[0]


def test_prose_rescue_still_goes_through_registry_validation(config, corpus) -> None:
    """A rescued call is an ordinary call: a bad phone is still rejected."""
    outcome, _, services = turn(
        LEAD_NO_CAPTURE,
        [text_reply('Noted. save_lead {"name": "Ritu Malhotra", "need": "30 gift boxes", "phone": "12345"}')],
        config=config,
        corpus=corpus,
    )

    assert outcome.actions == []
    assert services.leads.all() == []
    assert len(outcome.tool_errors) == 1
    assert "10 digits" in outcome.tool_errors[0]
    assert outcome.reply.strip()
