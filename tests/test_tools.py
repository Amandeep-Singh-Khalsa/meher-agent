"""Tool layer: validation, the hostile-input contract, and the action log."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from meher_agent.config import load_config
from meher_agent.data.corpus import load_corpus
from meher_agent.tools.registry import ToolRegistry
from meher_agent.tools.store import LeadStore

TEN_MB = 10 * 1024 * 1024
CREATED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def corpus(config):
    return load_corpus(config.runtime.data_dir)


@pytest.fixture
def registry(corpus, config):
    return ToolRegistry(corpus, config)


def payload_of(result) -> dict:
    return json.loads(result.content)


# --------------------------------------------------------------------------
# Surface
# --------------------------------------------------------------------------


def test_names_lists_both_tools(registry: ToolRegistry) -> None:
    assert registry.names() == ["save_lead", "escalate"]


def test_names_returns_a_copy(registry: ToolRegistry) -> None:
    first = registry.names()
    first.append("rm -rf")
    assert registry.names() == ["save_lead", "escalate"]


def test_content_is_json_with_ok_flag(registry: ToolRegistry) -> None:
    result = registry.run(
        "save_lead",
        {"name": "Ritu Malhotra", "need": "30 gift boxes", "email": "ritu.m@example.com"},
        conversation_id="c1",
    )
    assert result.ok is True
    assert result.error is None
    body = payload_of(result)
    assert body["ok"] is True
    assert body["lead_id"].startswith("lead-")
    assert "follow up by email" in body["message"]


# --------------------------------------------------------------------------
# save_lead happy paths
# --------------------------------------------------------------------------


def test_save_lead_with_email_only(registry: ToolRegistry) -> None:
    result = registry.run(
        "save_lead",
        {"name": "Ritu Malhotra", "need": "30 large gift boxes", "email": "Ritu.M@Example.COM"},
        conversation_id="c1",
    )
    assert result.ok is True
    leads = registry.leads.all()
    assert len(leads) == 1
    lead = leads[0]
    assert lead.name == "Ritu Malhotra"
    assert lead.email == "ritu.m@example.com"
    assert lead.phone is None
    assert lead.conversation_id == "c1"
    assert CREATED_AT_RE.match(lead.created_at), lead.created_at


def test_save_lead_with_phone_only(registry: ToolRegistry) -> None:
    result = registry.run(
        "save_lead",
        {"name": "Amit  Verma ", "need": "5 kg kaju katli", "phone": "+91 98765-43210"},
        conversation_id="c1",
    )
    assert result.ok is True
    assert registry.leads.all()[0].phone == "9876543210"
    assert registry.leads.all()[0].name == "Amit Verma"
    assert "follow up by phone" in payload_of(result)["message"]


def test_save_lead_with_both_contacts(registry: ToolRegistry) -> None:
    result = registry.run(
        "save_lead",
        {
            "name": "Sunita Rao",
            "need": "60 small gift boxes for Diwali",
            "phone": "09876543210",
            "email": "sunita@example.co.in",
        },
        conversation_id="c1",
    )
    assert result.ok is True
    lead = registry.leads.all()[0]
    assert lead.phone == "9876543210"
    assert lead.email == "sunita@example.co.in"
    assert "phone or email" in payload_of(result)["message"]


def test_save_lead_records_post_validation_args(registry: ToolRegistry) -> None:
    registry.run(
        "save_lead",
        {
            "name": "  Ritu   Malhotra ",
            "need": "30 large gift boxes for an office Diwali party",
            "email": "RITU.M@Example.com",
            "date": "3 November 2026",
            "quantity": " 30   boxes ",
        },
        conversation_id="seed-01",
    )
    action = registry.actions()[0]
    assert action.type == "save_lead"
    assert action.ok is True
    assert action.error is None
    assert action.args == {
        "name": "Ritu Malhotra",
        "need": "30 large gift boxes for an office Diwali party",
        "email": "ritu.m@example.com",
        "quantity": "30 boxes",
        "date": "2026-11-03",
    }


def test_action_public_shape_is_type_and_args(registry: ToolRegistry) -> None:
    registry.run(
        "save_lead",
        {"name": "Ritu", "need": "boxes", "email": "ritu.m@example.com"},
        conversation_id="c1",
    )
    public = registry.actions()[0].to_public()
    assert set(public) == {"type", "args"}
    assert public["type"] == "save_lead"
    assert public["args"]["email"] == "ritu.m@example.com"


def test_tool_result_does_not_echo_pii(registry: ToolRegistry) -> None:
    """The model only needs to know the save worked, not the value that was stored."""
    result = registry.run(
        "save_lead",
        {"name": "Ritu", "need": "boxes", "email": "ritu.m@example.com", "phone": "9876543210"},
        conversation_id="c1",
    )
    assert "9876543210" not in result.content
    assert "ritu.m@example.com" not in result.content
    assert "Lead saved" in result.content


def test_created_at_uses_an_injectable_utc_clock(corpus, config) -> None:
    fixed = datetime(2026, 9, 28, 10, 14, 3, tzinfo=timezone.utc)
    registry = ToolRegistry(corpus, config, clock=lambda: fixed)
    registry.run(
        "save_lead", {"name": "Ritu", "need": "boxes", "email": "r@example.com"}, conversation_id="c1"
    )
    assert registry.leads.all()[0].created_at == "2026-09-28T10:14:03Z"


def test_naive_clock_is_treated_as_utc(corpus, config) -> None:
    registry = ToolRegistry(corpus, config, clock=lambda: datetime(2026, 9, 28, 10, 14, 3))
    registry.run(
        "save_lead", {"name": "Ritu", "need": "boxes", "email": "r@example.com"}, conversation_id="c1"
    )
    assert registry.leads.all()[0].created_at == "2026-09-28T10:14:03Z"


def test_leads_can_be_shared_with_the_api_layer(corpus, config) -> None:
    shared = LeadStore()
    registry = ToolRegistry(corpus, config, leads=shared)
    registry.run(
        "save_lead", {"name": "Ritu", "need": "boxes", "email": "r@example.com"}, conversation_id="c1"
    )
    assert len(shared.all()) == 1
    assert registry.leads is shared


def test_leads_are_filterable_by_conversation(registry: ToolRegistry) -> None:
    for cid, email in (("a", "a@example.com"), ("b", "b@example.com"), ("a", "c@example.com")):
        registry.run("save_lead", {"name": "Ritu", "need": "boxes", "email": email}, conversation_id=cid)
    assert len(registry.leads.all()) == 3
    assert [lead.email for lead in registry.leads.for_conversation("a")] == [
        "a@example.com",
        "c@example.com",
    ]
    assert registry.leads.for_conversation("nope") == []


# --------------------------------------------------------------------------
# escalate
# --------------------------------------------------------------------------


def test_escalate_sets_handoff(registry: ToolRegistry) -> None:
    result = registry.run(
        "escalate", {"reason": "the gift box arrived crushed"}, conversation_id="c1"
    )
    assert result.ok is True
    body = payload_of(result)
    assert body["handoff"] is True
    assert registry.handoff is True
    assert registry.handoff_for("c1") is True
    assert registry.handoff_for("c2") is False
    assert registry.actions()[0].type == "escalate"
    assert registry.actions()[0].ok is True
    assert registry.actions()[0].args == {"reason": "the gift box arrived crushed"}


def test_escalate_without_reason_fails(registry: ToolRegistry) -> None:
    result = registry.run("escalate", {}, conversation_id="c1")
    assert result.ok is False
    assert result.error
    assert registry.handoff is False


def test_escalate_with_whitespace_reason_fails(registry: ToolRegistry) -> None:
    result = registry.run("escalate", {"reason": "  \n\t "}, conversation_id="c1")
    assert result.ok is False
    assert result.error
    assert registry.handoff is False


def test_escalate_with_non_string_reason_fails(registry: ToolRegistry) -> None:
    result = registry.run("escalate", {"reason": 42}, conversation_id="c1")
    assert result.ok is False
    assert "must be a string" in result.error


def test_failed_escalate_is_recorded_but_does_not_hand_off(registry: ToolRegistry) -> None:
    registry.run("escalate", {"reason": ""}, conversation_id="c1")
    assert len(registry.actions()) == 1
    assert registry.actions()[0].ok is False
    assert registry.handoff is False


# --------------------------------------------------------------------------
# The hostile-input contract
# --------------------------------------------------------------------------

HOSTILE: list[tuple[str, object, object]] = [
    ("args is a list", "save_lead", ["name", "need"]),
    ("args is a string", "save_lead", "name=Ritu need=boxes"),
    ("args is None", "save_lead", None),
    ("args is an int", "save_lead", 7),
    ("args is a 10 MB string", "save_lead", "x" * TEN_MB),
    ("tool name is a number", 123, {"name": "Ritu", "need": "boxes", "email": "a@b.com"}),
    ("tool name is None", None, {}),
    ("name is a number", "save_lead", {"name": 42, "need": "boxes", "email": "a@b.com"}),
    ("name is missing", "save_lead", {"need": "boxes", "email": "a@b.com"}),
    ("name is a dict", "save_lead", {"name": {"first": "Ritu"}, "need": "b", "email": "a@b.com"}),
    ("phone is an int", "save_lead", {"name": "Ritu", "need": "b", "phone": 9876543210}),
    ("phone is None", "save_lead", {"name": "Ritu", "need": "b", "phone": None}),
    ("phone is empty", "save_lead", {"name": "Ritu", "need": "b", "phone": "   "}),
    ("phone starts with 5", "save_lead", {"name": "Ritu", "need": "b", "phone": "5123456789"}),
    ("need is whitespace", "save_lead", {"name": "Ritu", "need": "  \n\t ", "email": "a@b.com"}),
    ("need is missing", "save_lead", {"name": "Ritu", "email": "a@b.com"}),
    ("no contact detail", "save_lead", {"name": "Ritu", "need": "30 boxes"}),
    ("date is 31/02/2026", "save_lead", {"name": "R", "need": "b", "email": "a@b.com", "date": "31/02/2026"}),
    ("date is junk", "save_lead", {"name": "R", "need": "b", "email": "a@b.com", "date": "tomorrow-ish"}),
    ("email is not-an-email", "save_lead", {"name": "R", "need": "b", "email": "not-an-email"}),
    ("email has no TLD", "save_lead", {"name": "R", "need": "b", "email": "a@b"}),
    ("email is a list", "save_lead", {"name": "R", "need": "b", "email": ["a@b.com"]}),
    ("unknown tool", "delete_everything", {}),
    ("unknown tool with args", "send_email", {"to": "attacker@evil.example"}),
    ("deeply nested junk", "save_lead", {"name": {"a": {"b": {"c": [{"d": {"e": "x"}}]}}}, "need": "b", "email": "a@b.com"}),
    ("non-string keys", "save_lead", {5: "Ritu", "need": "b", "email": "a@b.com"}),
    ("args is a 10 MB string field", "save_lead", {"name": "x" * TEN_MB, "need": "b", "email": "a@b.com"}),
    ("args is a 10 MB need field", "save_lead", {"name": "R", "need": "x" * TEN_MB, "email": "a@b.com"}),
    ("escalate with a list reason", "escalate", {"reason": ["crushed", "box"]}),
]


@pytest.mark.parametrize("label,tool,args", HOSTILE, ids=[case[0] for case in HOSTILE])
def test_hostile_inputs_never_raise_and_always_fail(
    registry: ToolRegistry, label: str, tool: object, args: object
) -> None:
    result = registry.run(tool, args, conversation_id="c-hostile")  # type: ignore[arg-type]
    assert result.ok is False, label
    assert result.error, label
    body = payload_of(result)
    assert body["ok"] is False
    assert body["error"]
    assert body["hint"]
    assert len(registry.actions()) == 1
    assert registry.actions()[0].ok is False
    assert registry.leads.all() == []
    assert registry.handoff is False


def test_hostile_args_are_truncated_in_the_action_log(registry: ToolRegistry) -> None:
    result = registry.run(
        "save_lead",
        {"name": "Ritu", "need": "boxes", "phone": "1234567890", "note": "x" * TEN_MB},
        conversation_id="c1",
    )
    assert result.ok is False
    recorded = registry.actions()[0].args
    assert len(json.dumps(recorded)) < 2000
    assert recorded["note"].endswith("...(truncated)")


def test_a_non_string_conversation_id_is_coerced_not_fatal(registry: ToolRegistry) -> None:
    result = registry.run(
        "save_lead",
        {"name": "Ritu", "need": "boxes", "email": "a@b.com"},
        conversation_id=None,  # type: ignore[arg-type]
    )
    assert result.ok is True
    assert registry.leads.all()[0].conversation_id == "None"
    assert len(registry.leads.for_conversation("None")) == 1


def test_unknown_args_shape_is_visible_in_the_action_log(registry: ToolRegistry) -> None:
    registry.run("save_lead", ["a", "b"], conversation_id="c1")
    assert registry.actions()[0].args == {"__args__": "list(len=2)"}


def test_missing_contact_error_asks_for_one(registry: ToolRegistry) -> None:
    result = registry.run("save_lead", {"name": "Ritu", "need": "30 boxes"}, conversation_id="c1")
    assert result.ok is False
    assert "phone" in result.error.lower()
    assert "email" in result.error.lower()
    assert "phone" in payload_of(result)["hint"].lower()


def test_phone_error_hint_matches_the_specified_wording(registry: ToolRegistry) -> None:
    result = registry.run(
        "save_lead", {"name": "R", "need": "b", "phone": "1234567890"}, conversation_id="c1"
    )
    assert result.ok is False
    body = payload_of(result)
    assert body["error"] == "phone must be an Indian mobile number starting with 6, 7, 8 or 9"
    assert body["hint"] == (
        "Ask the customer to re-read their number, or ask for an email instead. "
        "Do not invent a number."
    )


# --------------------------------------------------------------------------
# Action log
# --------------------------------------------------------------------------


def test_every_call_records_an_action(registry: ToolRegistry) -> None:
    registry.run("save_lead", {"name": "R", "need": "b", "email": "a@b.com"}, conversation_id="c1")
    registry.run("save_lead", {"name": "R", "need": "b", "phone": "bad"}, conversation_id="c1")
    registry.run("nope", {}, conversation_id="c1")
    registry.run("escalate", {"reason": "damaged box"}, conversation_id="c1")
    actions = registry.actions()
    assert [a.type for a in actions] == ["save_lead", "save_lead", "nope", "escalate"]
    assert [a.ok for a in actions] == [True, False, False, True]
    assert actions[1].error and actions[2].error
    assert actions[0].error is None and actions[3].error is None


def test_failed_call_records_the_raw_unvalidated_args(registry: ToolRegistry) -> None:
    registry.run(
        "save_lead",
        {"name": "Ritu  ", "need": "boxes", "email": "NOT-An-Email"},
        conversation_id="c1",
    )
    args = registry.actions()[0].args
    assert args["email"] == "NOT-An-Email"
    assert args["name"] == "Ritu  "


def test_actions_is_a_copy(registry: ToolRegistry) -> None:
    registry.run("escalate", {"reason": "x"}, conversation_id="c1")
    snapshot = registry.actions()
    snapshot.clear()
    assert len(registry.actions()) == 1


def test_actions_for_filters_by_conversation(registry: ToolRegistry) -> None:
    registry.run("escalate", {"reason": "a"}, conversation_id="c1")
    registry.run("escalate", {"reason": "b"}, conversation_id="c2")
    assert len(registry.actions()) == 2
    assert [a.args["reason"] for a in registry.actions_for("c2")] == ["b"]


def test_reset_clears_actions_and_handoff(registry: ToolRegistry) -> None:
    registry.run("escalate", {"reason": "a"}, conversation_id="c1")
    assert registry.handoff is True
    registry.reset()
    assert registry.actions() == []
    assert registry.handoff is False


def test_unknown_tool_error_lists_the_available_tools(registry: ToolRegistry) -> None:
    result = registry.run("save_lead_v2", {}, conversation_id="c1")
    assert result.ok is False
    assert "save_lead" in result.error and "escalate" in result.error
    assert registry.actions()[0].type == "save_lead_v2"


def test_registry_is_thread_safe(corpus, config) -> None:
    registry = ToolRegistry(corpus, config)

    def save(index: int) -> None:
        registry.run(
            "save_lead",
            {"name": f"Customer {index}", "need": "boxes", "email": f"c{index}@example.com"},
            conversation_id=f"c{index % 5}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(50)))

    assert len(registry.actions()) == 50
    assert len(registry.leads.all()) == 50
    assert sum(len(registry.actions_for(f"c{i}")) for i in range(5)) == 50
    assert sum(len(registry.leads.for_conversation(f"c{i}")) for i in range(5)) == 50
