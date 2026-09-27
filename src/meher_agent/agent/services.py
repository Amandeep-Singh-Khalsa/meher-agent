"""Composition root.

Every cross-layer dependency is declared here as a container so that each module
can be developed and unit-tested in isolation. Nothing else in the codebase
imports a concrete implementation from another layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from ..config import Config, get_config
from ..data.corpus import Corpus, load_corpus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..agent.conversation import ConversationStore
    from ..grounding.billing import BillingEngine
    from ..grounding.guard import ReplyGuard
    from ..llm.client import LLMClient
    from ..retrieval.pipeline import RetrievalPipeline
    from ..tools.registry import ToolRegistry
    from ..tools.store import LeadStore


@dataclass
class Services:
    config: Config
    corpus: Corpus
    retrieval: "RetrievalPipeline"
    billing: "BillingEngine"
    guard: "ReplyGuard"
    llm: "LLMClient"
    tools: "ToolRegistry"
    leads: "LeadStore"
    conversations: "ConversationStore"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.config.llm.model,
            "base_url": self.config.llm.base_url,
            "temperature": self.config.llm.temperature,
            "max_steps": self.config.llm.max_steps,
            "skus": len(self.corpus.skus),
            "sections": len(self.corpus.sections),
        }


def build_services(config: Config | None = None) -> Services:
    """Wire the object graph. Called once at startup; cheap enough to repeat."""
    from ..agent.conversation import ConversationStore
    from ..grounding.billing import BillingEngine
    from ..grounding.guard import ReplyGuard
    from ..llm.client import LLMClient
    from ..retrieval.pipeline import RetrievalPipeline
    from ..tools.registry import ToolRegistry
    from ..tools.store import LeadStore

    cfg = config or get_config()
    corpus = load_corpus(cfg.runtime.data_dir)
    leads = LeadStore()
    return Services(
        config=cfg,
        corpus=corpus,
        retrieval=RetrievalPipeline(corpus, cfg),
        billing=BillingEngine(corpus, cfg),
        guard=ReplyGuard(corpus, cfg),
        llm=LLMClient(cfg),
        tools=ToolRegistry(corpus, cfg, leads=leads),
        leads=leads,
        conversations=ConversationStore(
            max_conversations=cfg.runtime.max_conversations,
            max_history_turns=cfg.runtime.max_history_turns,
        ),
    )


@lru_cache(maxsize=1)
def get_services() -> Services:
    return build_services()
