"""HTTP surface, driven entirely by stubs.

No real model call happens here. `run_turn` is replaced on the module so the
endpoint's own translation, validation and failure handling are what is under
test, and `Services` is supplied through the dependency override rather than
through `app.state`, so the lifespan that builds the real object graph never
runs.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from meher_agent.agent.conversation import ConversationStore
from meher_agent.api import app as api_app
from meher_agent.config import load_config
from meher_agent.tools.store import LeadStore
from meher_agent.types import Action, AgentOutcome, ChatResponse, Lead

RunTurn = Callable[[str, str, Any], AgentOutcome]


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


class StubLLM:
    def __init__(self, *, reachable: bool = True, model: str = "stub-model-7b") -> None:
        self.reachable = reachable
        self.model = model
        self.health_calls = 0

    def health(self) -> bool:
        self.health_calls += 1
        return self.reachable


class StubServices:
    """Enough of the composition root for the HTTP layer and nothing more."""

    def __init__(self, *, llm: StubLLM | None = None) -> None:
        self.config = load_config()
        self.leads = LeadStore()
        self.conversations = ConversationStore(max_conversations=64, max_history_turns=12)
        self.llm = llm or StubLLM()


def make_outcome(**fields: Any) -> AgentOutcome:
    base: dict[str, Any] = {
        "reply": "Kaju katli 1 kg is Rs 1,200.",
        "sources": ["prices.csv#KK-1000"],
        "actions": [],
        "handoff": False,
    }
    base.update(fields)
    return AgentOutcome(**base)


def make_run_turn(outcome: AgentOutcome | None = None, **fields: Any) -> RunTurn:
    resolved = outcome if outcome is not None else make_outcome(**fields)

    def fake(message: str, conversation_id: str, services: Any) -> AgentOutcome:
        return resolved

    return fake


@contextlib.contextmanager
def patched_services(services: Any) -> Iterator[None]:
    api_app.app.dependency_overrides[api_app.get_services_dep] = lambda: services
    try:
        yield
    finally:
        api_app.app.dependency_overrides.pop(api_app.get_services_dep, None)


@contextlib.contextmanager
def state_services(services: Any) -> Iterator[None]:
    # Starlette's State exposes no way to unset a key, and monkeypatch cannot
    # record an absent one, so the backing dict is saved and restored directly.
    state = api_app.app.state
    had = "services" in state._state
    saved = state._state.get("services")
    state.services = services
    try:
        yield
    finally:
        if had:
            state.services = saved
        else:
            state._state.pop("services", None)


@pytest.fixture(autouse=True)
def clean_app() -> Iterator[None]:
    api_app.reset_health_cache()
    yield
    api_app.app.dependency_overrides.clear()
    api_app.reset_health_cache()


@pytest.fixture
def services() -> StubServices:
    return StubServices()


@pytest.fixture
def client(services: StubServices) -> Iterator[TestClient]:
    with patched_services(services):
        yield TestClient(api_app.app)


@pytest.fixture
def lenient_client(services: StubServices) -> Iterator[TestClient]:
    with patched_services(services):
        yield TestClient(api_app.app, raise_server_exceptions=False)


def post(client: TestClient, message: str, conversation_id: str = "c-205") -> Any:
    return client.post("/chat", json={"conversation_id": conversation_id, "message": message})


def add_lead(store: Any, **fields: Any) -> Lead:
    lead = Lead(
        name=fields.pop("name", "Ritu Malhotra"),
        need=fields.pop("need", "30 large gift boxes for the office Diwali party"),
        conversation_id=fields.pop("conversation_id", "c-205"),
        created_at=fields.pop("created_at", "2026-09-28T10:14:03Z"),
        **fields,
    )
    return store.add(lead)


# --------------------------------------------------------------------------
# POST /chat
# --------------------------------------------------------------------------


def test_chat_returns_exactly_the_four_contracted_keys(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_app, "_run_turn", make_run_turn())
    body = post(client, "How much is 1 kg kaju katli?").json()
    assert set(body) == {"reply", "sources", "actions", "handoff"}
    assert isinstance(body["reply"], str) and body["reply"]
    assert isinstance(body["sources"], list)
    assert isinstance(body["actions"], list)
    assert isinstance(body["handoff"], bool)
    assert body["sources"] == ["prices.csv#KK-1000"]


def test_chat_serialises_actions_as_type_and_args(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = make_outcome(
        actions=[
            Action(type="save_lead", args={"name": "Ritu Malhotra", "email": "ritu.m@example.com"})
        ]
    )
    monkeypatch.setattr(api_app, "_run_turn", make_run_turn(outcome))
    body = post(client, "30 gift boxes for our office party").json()
    assert body["actions"] == [
        {
            "type": "save_lead",
            "args": {"name": "Ritu Malhotra", "email": "ritu.m@example.com"},
        }
    ]
    assert set(body["actions"][0]) == {"type", "args"}


def test_chat_drops_actions_that_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = make_outcome(
        actions=[
            Action(type="save_lead", args={"name": "Ritu"}, ok=True),
            Action(type="save_lead", args={"name": ""}, ok=False, error="need is empty"),
        ]
    )
    monkeypatch.setattr(api_app, "_run_turn", make_run_turn(outcome))
    body = post(client, "save my details").json()
    assert [action["type"] for action in body["actions"]] == ["save_lead"]
    assert body["actions"][0]["args"] == {"name": "Ritu"}


def test_chat_passes_handoff_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_app, "_run_turn", make_run_turn(handoff=True))
    assert post(client, "the box arrived crushed").json()["handoff"] is True


def test_chat_reports_diagnostics_in_headers_not_in_the_body(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = make_outcome(model_calls=3, tool_errors=["save_lead: need is empty"])
    monkeypatch.setattr(api_app, "_run_turn", make_run_turn(outcome))
    response = post(client, "what is the total?")
    assert response.headers["X-Model-Calls"] == "3"
    assert response.headers["X-Tool-Errors"] == "1"
    assert set(response.json()) == {"reply", "sources", "actions", "handoff"}


def test_empty_sources_round_trip(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_app, "_run_turn", make_run_turn(sources=[]))
    assert post(client, "hello").json()["sources"] == []


def test_a_reply_of_exactly_1200_chars_round_trips(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "x" * 1200
    monkeypatch.setattr(api_app, "_run_turn", make_run_turn(reply=reply))
    response = post(client, "give me the long answer")
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == reply
    assert len(body["reply"]) == 1200


def test_an_empty_reply_round_trips(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_app, "_run_turn", make_run_turn(reply=""))
    response = post(client, "anything?")
    assert response.status_code == 200
    assert response.json()["reply"] == ""


@pytest.mark.parametrize(
    "message",
    ["नमस्ते 🙏 दो किलो काजू कटली", "こんにちは 🎉 \U0001f9f0", "café — naïve 😀"],
)
def test_unicode_round_trips(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    seen: list[str] = []

    def fake(raw: str, conversation_id: str, services: Any) -> AgentOutcome:
        seen.append(raw)
        return make_outcome(reply=f"reply to {message}")

    monkeypatch.setattr(api_app, "_run_turn", fake)
    body = post(client, message).json()
    assert seen == [message]
    assert body["reply"] == f"reply to {message}"


def test_a_raising_run_turn_never_returns_a_traceback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(message: str, conversation_id: str, services: Any) -> AgentOutcome:
        raise RuntimeError("ollama refused the connection")

    monkeypatch.setattr(api_app, "_run_turn", boom)
    response = post(client, "what is the price?")
    assert response.status_code == 200
    assert response.headers["X-Turn-Error"] == "1"
    assert "Traceback" not in response.text
    assert "ollama refused" not in response.text
    body = response.json()
    assert set(body) == {"reply", "sources", "actions", "handoff"}
    assert body["reply"]
    assert body["sources"] == [] and body["actions"] == []


def test_a_missing_loop_module_still_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This is what the lazy import in _run_turn buys: with the loop unreachable
    # the endpoint degrades to a normal reply instead of failing to import.
    monkeypatch.setitem(sys.modules, "meher_agent.agent.loop", None)
    response = post(client, "are you there?")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"reply", "sources", "actions", "handoff"}
    assert body["reply"]
    assert body["handoff"] is True


# --------------------------------------------------------------------------
# POST /chat validation
# --------------------------------------------------------------------------


def test_missing_conversation_id_is_422(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 422
    assert "conversation_id" in response.json()["detail"]


def test_missing_message_is_422(client: TestClient) -> None:
    response = client.post("/chat", json={"conversation_id": "c-205"})
    assert response.status_code == 422
    assert "message" in response.json()["detail"]


def test_blank_message_is_422(client: TestClient) -> None:
    response = client.post("/chat", json={"conversation_id": "c-205", "message": "   "})
    assert response.status_code == 422
    assert response.json()["detail"] == api_app._BLANK_MESSAGE


def test_blank_conversation_id_is_422(client: TestClient) -> None:
    response = client.post("/chat", json={"conversation_id": "  ", "message": "hello"})
    assert response.status_code == 422
    assert response.json()["detail"] == api_app._BLANK_CONVERSATION_ID


def test_empty_message_is_422(client: TestClient) -> None:
    response = client.post("/chat", json={"conversation_id": "c-205", "message": ""})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# GET /leads
# --------------------------------------------------------------------------


def test_leads_mask_the_phone_and_email_and_null_the_absent_one(
    client: TestClient, services: StubServices
) -> None:
    add_lead(services.leads, email="ritu.m@example.com", phone=None)
    assert client.get("/leads").json() == [
        {
            "name": "Ritu Malhotra",
            "email": "r*****@example.com",
            "phone": None,
            "need": "30 large gift boxes for the office Diwali party",
            "conversation_id": "c-205",
            "created_at": "2026-09-28T10:14:03Z",
        }
    ]


def test_leads_mask_a_phone_only_lead(client: TestClient, services: StubServices) -> None:
    add_lead(services.leads, name="Amit Verma", phone="9876543210", email=None)
    response = client.get("/leads")
    lead = response.json()[0]
    assert lead["phone"] == "******3210"
    assert lead["email"] is None
    assert "9876543210" not in response.text


def test_leads_are_newest_last(client: TestClient, services: StubServices) -> None:
    add_lead(services.leads, name="First", created_at="2026-09-28T09:00:00Z")
    add_lead(services.leads, name="Second", created_at="2026-09-28T10:14:03Z")
    add_lead(services.leads, name="Third", created_at="2026-09-28T11:00:00Z")
    body = client.get("/leads").json()
    assert [lead["name"] for lead in body] == ["First", "Second", "Third"]
    assert set(body[0]) == {
        "name",
        "email",
        "phone",
        "need",
        "conversation_id",
        "created_at",
    }


def test_leads_is_an_empty_list_when_nothing_was_saved(client: TestClient) -> None:
    response = client.get("/leads")
    assert response.status_code == 200
    assert response.json() == []


# --------------------------------------------------------------------------
# GET /health
# --------------------------------------------------------------------------


def test_health_reports_ok_with_the_model_name(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == load_config().llm.model
    assert body["llm_reachable"] is True
    assert body["leads"] == 0


def test_health_survives_an_unreachable_model(
    client: TestClient, services: StubServices
) -> None:
    services.llm.reachable = False
    api_app.reset_health_cache()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["llm_reachable"] is False


def test_health_survives_an_llm_that_raises(client: TestClient, services: StubServices) -> None:
    def broken() -> bool:
        raise ConnectionError("connection refused")

    services.llm.health = broken  # type: ignore[method-assign]
    api_app.reset_health_cache()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["llm_reachable"] is False


def test_health_caches_the_llm_check(client: TestClient, services: StubServices) -> None:
    for _ in range(4):
        assert client.get("/health").status_code == 200
    assert services.llm.health_calls == 1


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_index_serves_the_chat_page(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Meher Sweets" in resp.text


def test_services_dependency_reads_app_state() -> None:
    stub = StubServices()
    with state_services(stub):
        assert api_app.get_services_dep() is stub


def test_endpoints_report_503_until_startup_finishes() -> None:
    with state_services(None):
        response = TestClient(api_app.app).get("/health")
    assert response.status_code == 503
    assert "starting up" in response.json()["detail"]


def test_unhandled_errors_become_a_json_500_without_a_traceback(
    lenient_client: TestClient, services: StubServices
) -> None:
    class ExplodingStore:
        def all(self) -> list[Lead]:
            raise RuntimeError("store died holding ritu.m@example.com / 9876543210")

    services.leads = ExplodingStore()
    response = lenient_client.get("/leads")
    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "ritu.m@example.com" not in response.text
    assert "9876543210" not in response.text
    assert "Traceback" not in response.text


# --------------------------------------------------------------------------
# Conversation isolation, through the real ConversationStore
# --------------------------------------------------------------------------


def test_two_conversation_ids_keep_separate_histories(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, services: StubServices
) -> None:
    def history_aware(message: str, conversation_id: str, store: Any) -> AgentOutcome:
        conversation = store.conversations.get(conversation_id)
        seen_before = len(conversation.turns) // 2
        conversation.add_turn("user", message)
        conversation.add_turn("assistant", "acknowledged")
        return AgentOutcome(
            reply=f"turn {seen_before + 1} of {conversation.conversation_id}", sources=[]
        )

    monkeypatch.setattr(api_app, "_run_turn", history_aware)

    first = post(client, "what is the price of 1 kg kaju katli?", "c-101").json()
    second = post(client, "and 2 kg?", "c-101").json()
    other = post(client, "what is the price of 1 kg kaju katli?", "c-202").json()

    assert first["reply"] == "turn 1 of c-101"
    assert second["reply"] == "turn 2 of c-101"
    assert other["reply"] == "turn 1 of c-202"
    assert services.conversations.conversation_ids() == ["c-101", "c-202"]


def test_the_response_model_is_the_contracted_shape() -> None:
    assert set(ChatResponse.model_fields) == {"reply", "sources", "actions", "handoff"}
