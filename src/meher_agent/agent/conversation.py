"""Per-conversation state with LRU eviction.

`config.runtime.max_conversations` bounds memory on a long-lived server and
`config.runtime.max_history_turns` bounds the context window we replay to the
model, so both are enforced here rather than by each caller.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from pydantic import BaseModel, Field, PrivateAttr

from ..types import TurnRecord

__all__ = ["Conversation", "ConversationStore"]


class Conversation(BaseModel):
    conversation_id: str
    turns: list[TurnRecord] = Field(default_factory=list)
    handoff: bool = False
    reply_count: int = 0
    last_language: str = "english"

    _max_turns: int = PrivateAttr(default=12)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def add_turn(self, role: str, content: str) -> TurnRecord:
        """Append one turn, dropping the oldest once the history cap is exceeded.

        The lock is per conversation, not global: two conversations may be served
        in parallel, and a caller may mutate the object it got from `get()`.
        """
        record = TurnRecord(role=role, content=content)
        with self._lock:
            self.turns.append(record)
            if self._max_turns >= 0:
                excess = len(self.turns) - self._max_turns
                if excess > 0:
                    del self.turns[:excess]
        return record

    def recent(self, max_turns: int | None = None) -> list[TurnRecord]:
        """Newest-last window of history, optionally shorter than the cap."""
        limit = self._max_turns if max_turns is None else max(0, max_turns)
        with self._lock:
            return list(self.turns[-limit:]) if limit else []


class ConversationStore:
    def __init__(self, *, max_conversations: int, max_history_turns: int) -> None:
        self._lock = threading.RLock()
        self._max_conversations = max(1, max_conversations)
        self._max_history_turns = max(0, max_history_turns)
        self._conversations: "OrderedDict[str, Conversation]" = OrderedDict()

    def get(self, conversation_id: str) -> Conversation:
        """Fetch, creating on first sight, and mark most-recently-used.

        The live object is returned, not a copy: the agent loop appends turns to
        it across several model calls within one request.
        """
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                conversation = Conversation(conversation_id=conversation_id)
                conversation._max_turns = self._max_history_turns
                self._conversations[conversation_id] = conversation
                self._evict_locked()
            else:
                self._conversations.move_to_end(conversation_id)
            return conversation

    def clear(self) -> None:
        with self._lock:
            self._conversations.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._conversations)

    def conversation_ids(self) -> list[str]:
        """Least-recently-used first. Used by tests and diagnostics only."""
        with self._lock:
            return list(self._conversations)

    def _evict_locked(self) -> None:
        while len(self._conversations) > self._max_conversations:
            self._conversations.popitem(last=False)
