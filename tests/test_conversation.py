"""ConversationStore: history cap, LRU eviction, and thread safety."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from meher_agent.agent.conversation import Conversation, ConversationStore
from meher_agent.types import TurnRecord


@pytest.fixture
def store() -> ConversationStore:
    return ConversationStore(max_conversations=3, max_history_turns=4)


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------


def test_get_creates_on_first_sight() -> None:
    store = ConversationStore(max_conversations=5, max_history_turns=5)
    conversation = store.get("c1")
    assert isinstance(conversation, Conversation)
    assert conversation.conversation_id == "c1"
    assert conversation.turns == []
    assert conversation.handoff is False
    assert conversation.reply_count == 0
    assert conversation.last_language == "english"
    assert store.count == 1


def test_get_returns_the_same_live_object() -> None:
    store = ConversationStore(max_conversations=5, max_history_turns=5)
    first = store.get("c1")
    first.add_turn("user", "hello")
    assert store.get("c1") is first
    assert [t.content for t in store.get("c1").turns] == ["hello"]
    assert store.count == 1


def test_add_turn_returns_the_record() -> None:
    conversation = Conversation(conversation_id="c1")
    record = conversation.add_turn("user", "hi")
    assert isinstance(record, TurnRecord)
    assert (record.role, record.content) == ("user", "hi")
    assert conversation.turns == [record]


def test_history_is_capped_oldest_first_dropped() -> None:
    store = ConversationStore(max_conversations=5, max_history_turns=3)
    conversation = store.get("c1")
    for i in range(6):
        conversation.add_turn("user", f"m{i}")
    assert [t.content for t in conversation.turns] == ["m3", "m4", "m5"]


def test_history_cap_of_zero_keeps_nothing() -> None:
    store = ConversationStore(max_conversations=5, max_history_turns=0)
    conversation = store.get("c1")
    conversation.add_turn("user", "m0")
    assert conversation.turns == []


def test_recent_window() -> None:
    store = ConversationStore(max_conversations=5, max_history_turns=6)
    conversation = store.get("c1")
    for i in range(5):
        conversation.add_turn("user", f"m{i}")
    assert [t.content for t in conversation.recent(2)] == ["m3", "m4"]
    assert len(conversation.recent()) == 5


def test_conversation_fields_are_settable() -> None:
    conversation = Conversation(conversation_id="c1")
    conversation.handoff = True
    conversation.reply_count = 3
    conversation.last_language = "hinglish"
    assert conversation.handoff is True
    assert conversation.reply_count == 3
    assert conversation.last_language == "hinglish"


def test_clear_empties_the_store() -> None:
    store = ConversationStore(max_conversations=5, max_history_turns=5)
    store.get("c1").add_turn("user", "hi")
    store.get("c2")
    assert store.count == 2
    store.clear()
    assert store.count == 0
    assert store.get("c1").turns == []


# --------------------------------------------------------------------------
# LRU eviction
# --------------------------------------------------------------------------


def test_evicts_least_recently_used_when_over_the_cap() -> None:
    store = ConversationStore(max_conversations=3, max_history_turns=4)
    for cid in ("a", "b", "c"):
        store.get(cid)
    assert store.conversation_ids() == ["a", "b", "c"]

    store.get("d")
    assert store.conversation_ids() == ["b", "c", "d"]
    assert store.count == 3


def test_reading_a_conversation_makes_it_recent() -> None:
    store = ConversationStore(max_conversations=3, max_history_turns=4)
    for cid in ("a", "b", "c"):
        store.get(cid)
    store.get("a")
    assert store.conversation_ids() == ["b", "c", "a"]

    store.get("d")
    assert store.conversation_ids() == ["c", "a", "d"]


def test_evicted_conversation_starts_fresh() -> None:
    store = ConversationStore(max_conversations=2, max_history_turns=4)
    store.get("a").add_turn("user", "remembered")
    store.get("b")
    store.get("c")
    assert store.get("a").turns == []
    assert store.count == 2


def test_count_never_exceeds_the_cap() -> None:
    store = ConversationStore(max_conversations=5, max_history_turns=4)
    for i in range(50):
        store.get(f"c{i}")
        assert store.count <= 5
    assert store.count == 5


def test_a_cap_of_zero_is_clamped_to_one() -> None:
    store = ConversationStore(max_conversations=0, max_history_turns=4)
    store.get("a")
    assert store.count == 1


# --------------------------------------------------------------------------
# Thread safety
# --------------------------------------------------------------------------


def test_concurrent_get_and_add_turn_is_consistent() -> None:
    ids = [f"c{i}" for i in range(50)]
    turns_each = 4
    calls = 200
    store = ConversationStore(max_conversations=len(ids), max_history_turns=turns_each)

    def work(index: int) -> tuple[str, int]:
        conversation_id = ids[index % len(ids)]
        conversation = store.get(conversation_id)
        conversation.add_turn("user", f"message {index}")
        return conversation_id, len(conversation.turns)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, range(calls)))

    assert len(results) == calls
    assert store.count == len(ids)
    for conversation_id in ids:
        assert len(store.get(conversation_id).turns) == turns_each
    assert sorted(store.conversation_ids()) == sorted(ids)


def test_concurrent_eviction_keeps_the_cap() -> None:
    store = ConversationStore(max_conversations=10, max_history_turns=2)

    def work(index: int) -> int:
        conversation = store.get(f"c{index % 40}")
        conversation.add_turn("user", "x")
        return store.count

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(work, range(200)))

    assert max(counts) <= 10
    assert store.count <= 10
