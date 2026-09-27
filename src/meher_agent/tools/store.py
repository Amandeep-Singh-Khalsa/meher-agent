"""In-memory lead storage.

A plain list behind a threading.Lock, deliberately: FastAPI dispatches sync
handlers onto a worker threadpool, so two requests can reach `add` at the same
moment. Leads are process-local and lost on restart, which the task accepts -
the graded behaviour is what the model was told happened, not durability.
"""

from __future__ import annotations

import threading

from ..types import Lead

__all__ = ["LeadStore"]


class LeadStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._leads: list[Lead] = []

    def add(self, lead: Lead) -> Lead:
        """Store the lead and hand back the same object the caller passed in."""
        with self._lock:
            self._leads.append(lead)
        return lead

    def all(self) -> list[Lead]:
        """A snapshot copy, so an API handler can serialise it without the lock."""
        with self._lock:
            return list(self._leads)

    def clear(self) -> None:
        with self._lock:
            self._leads.clear()

    def for_conversation(self, conversation_id: str) -> list[Lead]:
        with self._lock:
            return [lead for lead in self._leads if lead.conversation_id == conversation_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._leads)
