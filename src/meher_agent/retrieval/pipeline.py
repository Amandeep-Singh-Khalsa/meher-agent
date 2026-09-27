"""The one entry point the agent calls: message plus history to a Briefing.

Everything the answer layer is allowed to know comes from here: the sections of
data/ that were retrieved, the price rows that were retrieved, the order lines
that could be read out of the customer's own words, the language to answer in,
and the only rupee figures a reply may quote. Nothing else in the system reads
data/ directly, so this is also the place where cross-turn continuity lives: a
turn that says "make it 3 kg" resolves against the items the previous turns were
already discussing.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..config import Config
from ..data.corpus import Corpus
from ..types import Briefing, Resolution, RetrievedSKU, TurnRecord
from ..text.lexicon import tokenize
from ..text.translit import detect_language
from .resolver import OrderResolver
from .retriever import Retriever

__all__ = ["RetrievalPipeline"]

#: Fixed rupee figures the policy itself defines, quoted in data/policies.md.
#: They are not catalog prices, but a correct reply may state them, so the
#: reply guard has to be told about them.
_POLICY_AMOUNTS: frozenset[int] = frozenset({60, 999, 5000})

#: How many earlier user turns to re-read when the current one names no product.
#: Two is enough for "2 kg kaju katli" followed by "how much is that with
#: delivery?", three gives a little slack without re-reading the whole history.
_CARRY_LOOKBACK = 3

#: Words that mean "add this to what I already asked for", in any of the three
#: scripts the shop answers in. A bare "aur" / "और" is not enough: it is also the
#: word that joins two products inside a single new order.
_ADDITIVE_WORDS: frozenset[str] = frozenset(
    {"add", "also", "too", "aswell", "along", "besides", "further", "साथ", "अलावा"}
)

#: Subjects that have nothing to do with a sweets shop. Presence of one of these,
#: together with no grounding at all in data/, marks a turn as out of scope.
_UNRELATED_SUBJECTS: tuple[str, ...] = (
    "assignment", "essay", "homework", "thesis", "resume", "cv", "cover letter",
    "poem", "story", "joke", "photosynthesis", "translate this", "medical advice",
    "diagnose", "symptom", "prescription", "loan", "credit card", "crypto",
    "bitcoin", "stock price", "share price", "election", "movie review",
    "python script", "java code", "debug my", "write a program", "astrology",
    "horoscope", "relationship advice", "weather", "cricket score", "flight status",
    "train ticket", "university admission",
)


class RetrievalPipeline:
    """Retrieve, resolve and package one turn of conversation."""

    def __init__(self, corpus: Corpus, config: Config) -> None:
        self.corpus = corpus
        self.config = config
        self.retriever = Retriever(corpus, config.retrieval)
        self.resolver = OrderResolver(corpus, config.retrieval)
        self.allowed_amounts: list[int] = sorted(
            set(corpus.catalog_prices()) | _POLICY_AMOUNTS
        )

    # -- public ------------------------------------------------------------
    def brief(self, message: str, history: Sequence[TurnRecord] | None = None) -> Briefing:
        """Ground one customer turn.

        ``history`` is the conversation so far, oldest first. Only the previous
        *user* turns are re-read, and only to carry an unfinished order forward.
        """
        question = (message or "").strip()
        prior = self._prior_resolution(question, history or ())

        resolution = self.resolver.resolve(question, carry=prior.items)
        if not resolution.items and prior.items:
            # "How much is that?" names nothing, so the order under discussion is
            # still the order. Carry it, and say so, rather than quoting zero.
            resolution.items = list(prior.items)
            resolution.unresolved.append(
                "No product named in this turn; kept the order already under "
                "discussion: "
                + ", ".join(f"{item.qty} x {item.item}" for item in prior.items)
            )
        if resolution.items and prior.items and self._is_additive(question):
            # "...and add 10 samosas": the turn named products of its own, but the
            # order already under discussion is still part of it. Items this turn
            # restated are not added twice, only the ones it left alone.
            named = {item.sku for item in resolution.items}
            kept = [item for item in prior.items if item.sku not in named]
            if kept:
                resolution.items = kept + resolution.items
                resolution.unresolved.append(
                    "Kept from earlier in this conversation: "
                    + ", ".join(f"{item.qty} x {item.item}" for item in kept)
                )
        if resolution.distance_km is None and prior.distance_km is not None:
            resolution.distance_km = prior.distance_km

        resolution.sections = self.retriever.sections(question)
        resolution.sku_hits = self._sku_hits(question, resolution)

        return Briefing(
            question=question,
            language=detect_language(question),
            resolved=resolution,
            allowed_amounts=list(self.allowed_amounts),
            out_of_scope_hint=self._out_of_scope(question, resolution),
        )

    # -- history -----------------------------------------------------------
    @staticmethod
    def _is_additive(question: str) -> bool:
        """True when the turn adds to the order instead of replacing it.

        Only an explicit additive word counts. Without one, a turn that names
        products is read as a new order, because quoting a stale line the customer
        never asked for again is worse than quoting a short order.
        """
        words = set(tokenize(question))
        return bool(words & _ADDITIVE_WORDS)

    def _prior_resolution(
        self, question: str, history: Sequence[TurnRecord]
    ) -> Resolution:
        """The most recent earlier turn that actually contained an order.

        Reading history back is cheap (the indexes are already built) and it is
        the only way "make it 3 kg" can know which product the customer means.
        The current turn is skipped, so a question cannot carry its own items
        into itself and double them.
        """
        seen = 0
        for record in reversed(list(history)):
            if getattr(record, "role", "") != "user":
                continue
            text = (getattr(record, "content", "") or "").strip()
            if not text or text == question:
                continue
            seen += 1
            if seen > _CARRY_LOOKBACK:
                break
            resolution = self.resolver.resolve(text)
            if resolution.items:
                return resolution
        return Resolution()

    # -- evidence ----------------------------------------------------------
    def _sku_hits(self, question: str, resolution: Resolution) -> list[RetrievedSKU]:
        """Catalog rows this turn could be talking about, best evidence first.

        The fuzzy retriever is asked first, then any product the resolver settled
        on is added: a resolved line must always be backed by a catalog row, even
        when the retriever ranked its neighbours higher.
        """
        hits = list(self.retriever.skus(question))
        seen = {hit.sku.sku for hit in hits}
        for item in resolution.items:
            if item.sku in seen:
                continue
            sku = self.corpus.sku(item.sku)
            if sku is None:
                continue
            seen.add(sku.sku)
            quoted = repr(item.matched_text) if item.matched_text else "the order"
            hits.append(
                RetrievedSKU(sku=sku, score=1.0, reason=f"resolved from {quoted}")
            )
        return hits

    # -- scope -------------------------------------------------------------
    def _out_of_scope(self, question: str, resolution: Resolution) -> bool:
        """True when the message is plainly not about the shop.

        This is a hint, not a verdict: the reply guard runs its own, wider
        detection. The bar is deliberately high, because the guard checks this
        flag *first*. A message that matched nothing but still asks the shop
        something ("do you make rabri?") is in scope and unknown, not out of
        scope, so the flag needs positive evidence of another subject.
        """
        if not question:
            return False
        if resolution.items or resolution.sku_hits:
            return False
        if any(section.score > 0.0 for section in resolution.sections):
            return False
        folded = question.casefold()
        return any(marker in folded for marker in _UNRELATED_SUBJECTS)
