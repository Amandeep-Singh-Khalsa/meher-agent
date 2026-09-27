"""Turns one customer message into a checkable order reading.

This is the only layer that decides *what* the customer asked for and *how many*.
It never computes money: quantities and catalog SKUs are the output, and every
rupee figure in a reply is produced downstream from data/prices.csv. That split is
what lets the reply guard treat any number the model invents as a violation.

Two rules shape the design:

* Single message. "Make it 3 kg" only means something next to "1 kg motichoor
  laddoo", so resolving each turn alone keeps this layer trivially auditable; the
  optional ``carry`` argument is how :mod:`retrieval.pipeline` hands in the items
  the conversation was already discussing.
* Never guess silently. A weight that no pack divides, a pack size that was left
  unsaid, a rupee amount that looks like a count: each one becomes an
  ``approximated`` flag or an ``unresolved`` note, so the model is told exactly
  what was assumed instead of inventing a quantity itself.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..config import RetrievalConfig
from ..data.corpus import Corpus
from ..types import Resolution, ResolvedOrderItem, SKU
from ..text.lexicon import STOPWORDS, tokenize
from ..text.translit import fold_devanagari
from .retriever import SKU_ALIASES, SKU_QUALIFIERS, best_alias_window, sku_alias_index

__all__ = ["OrderResolver"]

#: How far from a product mention a number may sit and still count as its
#: quantity: "2 kg kaju katli", "kaju katli 2 kg", "500 g of the kaju katli".
_QUANTITY_RADIUS = 5
#: ...of which at most this many words may carry meaning in between.
_MAX_BRIDGE_WORDS = 2
#: Filler that can sit between a number and a product name without breaking the
#: link: "2 kg **of** the kaju katli", "2 **more** kg **of** moti laddoo".
#: "and" is deliberately absent: it separates two orders, so a number after it
#: belongs to the next product and a number before it to the previous one.
_BRIDGE_WORDS = frozenset(
    {
        "of", "for", "the", "a", "an", "please", "more", "extra", "additional",
        "total", "worth", "in", "into", "to", "pack", "packs", "packed",
        "packing", "order", "bhi", "hi", "ka", "ki", "ke", "hai", "hain",
        "dena", "lena", "kar", "karo", "want", "need", "chahiye", "chahie",
    }
)
#: How far an ordering verb may sit from a product mention before a mention with
#: no quantity of its own stops counting as an order.
_INTENT_RADIUS = 6
#: A bare number this large is a rupee figure that lost its symbol, not a count.
_MAX_BARE_COUNT = 200
#: A qualifier the customer actually typed ("bada", "sugar free") outweighs a
#: shorter mention that happens to name the same family ("gift box").
_QUALIFIER_BONUS = 0.5

_UNIT_KG = frozenset({"kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"})
_UNIT_G = frozenset({"g", "gm", "gms", "gram", "grams"})
_UNIT_COUNT = frozenset(
    {
        "box", "boxes", "bx", "packet", "packets", "pkt", "pkts", "piece",
        "pieces", "pc", "pcs", "unit", "units", "nos", "plate", "plates",
        "tray", "trays", "parcel", "parcels", "bora", "bori",
    }
)
#: Folds of "किलोमीटर"/"किमी" land here too, so "12 किलोमीटर दूर" reads as a
#: distance and never as a 12 kg order.
_UNIT_KM = frozenset({"km", "kms", "kilm", "kimi", "kim", "kilomitar", "kilomiter"})

_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "pair": 2, "dozen": 12,
    # Hinglish number words, only ever read next to a product mention.
    "ek": 1, "do": 2, "teen": 13, "char": 4, "paanch": 5, "chhe": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "bees": 20, "tees": 30,
    "pachaas": 50, "sath": 60, "saath": 60, "sau": 100,
}

#: Marks a number that belongs to a date, so "3 November" is never "3 boxes".
_MONTHS = frozenset(
    {
        "jan", "january", "feb", "february", "mar", "march", "apr", "april",
        "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
        "september", "oct", "october", "nov", "november", "dec", "december",
        "janavari", "febvari", "aprail", "aapril", "mei", "julai", "monday",
        "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "week", "weekend", "tomorrow", "yesterday",
    }
)
#: "10 pm", "6 o'clock": a time of day, not an amount.
_TIME_WORDS = frozenset({"pm", "am", "oclock", "hrs"})
#: Elapsed durations. A number next to one of these measures time, not a count of
#: items: "delivered 30 minutes ago" is a single damaged box, not 30 gift boxes.
_ELAPSED_UNITS = frozenset(
    {
        "min", "mins", "minute", "minutes", "hr", "hrs", "hour", "hours",
        "sec", "secs", "second", "seconds", "day", "days", "week", "weeks",
        "month", "months", "year", "years", "मिनट", "घंटे", "घंटा", "दिन",
    }
)
#: "... 2 hours ago" - the number is a duration even when the unit is dropped.
_AGO_NEAR = re.compile(r"^\s*(?:[a-z\u0900-\u097F]+\s+){0,2}ago\b", re.IGNORECASE)
#: Rupee amounts and percentages are money, not counts.
_MONEY_WORDS = frozenset({"rs", "inr", "rupees", "rupaiya", "rupaye", "lakh", "lacs"})
_CURRENCY_BEFORE = re.compile(r"(?:\u20b9|rs\.?|inr)\s*$", re.IGNORECASE)

#: Ordering intent. Without one of these, a bare product mention is a question
#: about the product ("is the kaju katli vegetarian?") and must not become an
#: order line, or the model would quote a total nobody asked for.
_ORDER_INTENT = frozenset(
    {
        "want", "wanna", "wants", "need", "needs", "required", "order", "book",
        "booked", "reserve", "send", "dispatch", "courier", "get", "give",
        "pack", "packed", "packing", "add", "make", "please", "deliver",
        "chahiye", "chahie", "chahye", "chahiyega", "chaahie", "lena", "lijiye",
        "mangwa", "mangwao", "mangwana", "mangwaye", "bhejo", "bhej", "bhejna",
        "bana", "banwa", "banwana", "banwaye", "rakh", "rakhna", "deni",
        "dedo", "dena", "pakka", "confirm", "dewa", "dijiye", "chahiyega",
    }
)
#: Asking what something costs is intent too: the answer is a pack price.
_QUOTE_INTENT = frozenset(
    {
        "price", "prices", "pricing", "rate", "rates", "cost", "costs", "total",
        "amount", "much", "kitna", "kitni", "kitne", "paisa", "paise", "daam",
        "dam", "kharcha", "bhav", "mrp", "bill", "invoice", "estimate",
        "quotation", "budget",
    }
)
#: "I want to know if it is vegetarian" asks, it does not order.
_QUESTION_MARKERS = frozenset(
    {"know", "check", "verify", "explain", "inform", "wonder", "curious", "mean", "means"}
)
#: "2 more kg" adds to the order already discussed; "make it 3 kg" replaces it.
_MORE_WORDS = frozenset({"more", "additional", "extra", "aur", "another", "plus"})

#: Word characters plus the combining marks Devanagari hangs off its letters.
#: Python's ``\w`` does not include those marks, so without this "क्या" would come
#: apart as "क" + "्या" and no Hindi phrase would ever match an alias.
_TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)*|[^\W\d_][\w\u0900-\u0dff\u0300-\u036f\u200c\u200d\u0964\u0965]*",
    re.UNICODE,
)
_PERCENT = re.compile(r"\s*%")


@dataclass(frozen=True)
class _Tok:
    """One token, keeping the position it occupies in the original message."""

    key: str
    raw: str
    start: int
    end: int
    number: float | None


@dataclass(frozen=True)
class _Quantity:
    """A number found near a product mention, with the unit it carries."""

    value: float
    unit: str  # "kg" | "g" | "count" | ""
    index: int
    more: bool = False

    @property
    def grams(self) -> int | None:
        if self.unit == "kg":
            return int(round(self.value * 1000))
        if self.unit == "g":
            return int(round(self.value))
        return None

    def describe(self) -> str:
        if self.unit == "kg":
            return f"{self.value:g} kg"
        if self.unit == "g":
            return f"{self.value:g} g"
        return f"{self.value:g}"


@dataclass
class _Window:
    """One product mention in the message, and the SKUs that can satisfy it."""

    start: int
    end: int
    text: str
    cands: list[tuple[float, int, str]] = field(default_factory=list)
    quantity: _Quantity | None = None
    notes: list[str] = field(default_factory=list)


class OrderResolver:
    """Reads quantities, product mentions and delivery distance out of a message."""

    def __init__(self, corpus: Corpus, config: RetrievalConfig) -> None:
        self.corpus = corpus
        self.config = config
        self._aliases = sku_alias_index(corpus)
        self._qualifiers: dict[str, set[str]] = {
            sku.sku: {
                token
                for token in tokenize(
                    " ".join((sku.item, *SKU_ALIASES.get(sku.sku, ())))
                )
                if token in SKU_QUALIFIERS
            }
            for sku in corpus.skus
        }
        # A mention has to name the product: without this, the alias
        # "rasmalai 500 g" would match a bare "500 g" and read it as rasmalai.
        self._name_tokens: dict[str, frozenset[str]] = {
            sku.sku: frozenset(
                token
                for token in tokenize(sku.item)
                if token not in STOPWORDS
                and token not in _UNIT_KG
                and token not in _UNIT_G
                and token not in _UNIT_COUNT
                and not token.isdigit()
            )
            | frozenset(
                token
                for token in tokenize(f"{sku.sku} {sku.sku.replace('-', ' ')}")
                if token not in STOPWORDS and not token.isdigit()
            )
            for sku in corpus.skus
        }

    # -- public ------------------------------------------------------------
    def resolve(
        self,
        message: str,
        *,
        carry: Sequence[ResolvedOrderItem] | None = None,
    ) -> Resolution:
        """Read ``message`` into items, a distance and any unresolved notes.

        ``carry`` holds items the conversation was already discussing. A quantity
        in this message that no product mention can claim is applied to them, so
        "Make it 3 kg" updates the motichoor laddoo from the previous turn instead
        of resolving to nothing.
        """
        text = message or ""
        tokens = _tokenize(text)
        keys = [token.key for token in tokens]
        resolution = Resolution(distance_km=self._distance_km(tokens))

        windows = self._select_windows(keys, tokens, text)
        taken: set[int] = set()
        for window in windows:
            taken.update(range(window.start, window.end))
        self._assign_quantities(windows, tokens, text, taken)

        for window in windows:
            self._fill(window, tokens, text, keys, resolution)

        self._carry_over(windows, tokens, text, carry or (), resolution)
        return resolution

    # -- product mentions --------------------------------------------------
    def _select_windows(
        self, keys: list[str], tokens: list[_Tok], text: str
    ) -> list[_Window]:
        """Every product mention in the message, left to right, without overlap."""
        found: dict[tuple[int, int], _Window] = {}
        for sku in self.corpus.skus:
            quality, span, start, end = best_alias_window(
                self._aliases.get(sku.sku, ()),
                keys,
                required=self._name_tokens.get(sku.sku),
            )
            if quality <= 0.0:
                continue
            window = found.setdefault(
                (start, end),
                _Window(
                    start=start,
                    end=end,
                    text=text[tokens[start].start:tokens[end - 1].end].strip(),
                ),
            )
            window.cands.append((quality, span, sku.sku))
        for window in found.values():
            window.cands.sort(key=lambda item: (-item[0], -item[1], item[2]))

        # Strongest mention first: "kaju katli" (quality 1.0) must claim its tokens
        # before a weaker "kg kaju katli" mention can stretch across them. A
        # qualifier the customer typed ("bada", "sugar free") outranks a shorter
        # unqualified mention, so "bada diwali gift box" is the large box and not
        # the small one that a bare "gift box" would also match.
        def rank(window: _Window) -> tuple[float, int, int]:
            quality = max(cand[0] for cand in window.cands)
            near = set(keys[max(0, window.start - 3):window.end + 3])
            if any(
                near & self._qualifiers[sku_id] for _, _, sku_id in window.cands
            ):
                quality += _QUALIFIER_BONUS
            return (-quality, window.start, -window.end)

        chosen: list[_Window] = []
        used: set[int] = set()
        for window in sorted(found.values(), key=rank):
            span = set(range(window.start, window.end))
            if span & used:
                continue
            used |= span
            chosen.append(window)
        chosen.sort(key=lambda w: w.start)

        # A bare qualifier in front of a product is part of that product's
        # mention: "sugar free 1 kg kaju katli" is one order, not a qualifier and
        # a separate phrase, and folding them together lets the qualifier choose
        # the SKU.
        joined = True
        while joined and len(chosen) > 1:
            joined = False
            for first, second in zip(chosen, chosen[1:]):
                if first.end != second.start:
                    continue
                span = range(first.start, first.end)
                if not all(keys[index] in SKU_QUALIFIERS for index in span):
                    continue
                first.end = second.end
                first.cands = sorted(
                    first.cands + second.cands,
                    key=lambda item: (-item[0], -item[1], item[2]),
                )
                first.text = text[tokens[first.start].start:tokens[first.end - 1].end].strip()
                chosen.remove(second)
                joined = True
                break
        return chosen

    def _assign_quantities(
        self,
        windows: list[_Window],
        tokens: list[_Tok],
        text: str,
        taken: set[int],
    ) -> None:
        """Give every number to the mention it belongs to, and to it alone.

        "2 kg kaju katli and 500 g soan papdi" has three numbers and two
        mentions; "2 kg kaju katli and one large gift box" has a "one" that is
        closer to the gift box than to the katli. Each number therefore goes to
        its single nearest mention, and a mention may use a number inside its own
        window because "500 g soan papdi" is one phrase.
        """
        claims: list[tuple[int, int, int, _Quantity]] = []
        for position, window in enumerate(windows):
            blocked = taken - set(range(window.start, window.end))
            for distance, side, quantity in self._nearby_numbers(
                window, tokens, text, blocked
            ):
                claims.append((distance, side, position, quantity))
        claims.sort(key=lambda claim: (claim[0], claim[1], claim[2]))
        claimed: set[int] = set()
        for _, _, position, quantity in claims:
            if quantity.index in claimed or windows[position].quantity is not None:
                continue
            windows[position].quantity = quantity
            claimed.add(quantity.index)

    def _nearby_numbers(
        self, window: _Window, tokens: list[_Tok], text: str, blocked: set[int]
    ) -> list[tuple[int, int, _Quantity]]:
        """Quantities reachable from ``window``, as (distance, side, quantity).

        Distance counts only the words that carry meaning, so "2 kg kaju katli"
        and "2 more kg of moti laddoo" are both one word away, while a number
        three nouns away belongs to a different part of the sentence.
        """
        found: list[tuple[int, int, _Quantity]] = []
        low = max(0, window.start - _QUANTITY_RADIUS)
        high = min(len(tokens), window.end + _QUANTITY_RADIUS)
        for index in range(low, high):
            if index in blocked:
                continue
            quantity = self._number_at(tokens, index, text)
            if quantity is None:
                continue
            if index < window.start:
                side, between = 0, range(index + 1, window.start)
            elif index >= window.end:
                side, between = 1, range(window.end, index)
            else:
                # "500 g soan papdi" and "1 kg gulab jamun": the size is part of
                # the phrase the alias matched, so it belongs to this mention.
                side, between = 0, ()
            distance = sum(
                1 for position in between if tokens[position].key not in _BRIDGE_WORDS
            )
            if distance > _MAX_BRIDGE_WORDS:
                continue
            found.append((distance, side, quantity))
        return found

    def _number_at(self, tokens: list[_Tok], index: int, text: str) -> _Quantity | None:
        """Parse a quantity at ``index``, or None when the number is not one."""
        token = tokens[index]
        if token.number is not None:
            value: float | None = token.number
        elif token.key in _NUMBER_WORDS:
            value = float(_NUMBER_WORDS[token.key])
        else:
            return None
        if value <= 0:
            return None

        after = tokens[index + 1].key if index + 1 < len(tokens) else ""
        before = tokens[index - 1].key if index else ""
        # "3 November", "November 3", "10 pm" and a bare year are not amounts ordered.
        if after in _MONTHS or before in _MONTHS or after in _TIME_WORDS:
            return None
        # An elapsed duration is not a count: "delivered 30 minutes ago" is one
        # crushed box, not thirty gift boxes. Without this, a complaint becomes a
        # 30-box order and the guard quotes a total the customer never asked for.
        if after in _ELAPSED_UNITS or before in _ELAPSED_UNITS:
            return None
        if _AGO_NEAR.match(text[token.end:token.end + 12]):
            return None
        if value == int(value) and 1900 <= value <= 2100 and not after:
            return None
        # "Rs 5000", "₹1,200" and "50%" are money, never counts.
        if (
            before in _MONEY_WORDS
            or _CURRENCY_BEFORE.search(text[max(0, token.start - 5):token.start])
            or _PERCENT.match(text[token.end:token.end + 2])
        ):
            return None

        unit = self._unit_of(after)
        if unit == "km":
            return None
        if not unit and value > _MAX_BARE_COUNT:
            return None
        return _Quantity(value=value, unit=unit, index=index, more=before in _MORE_WORDS)

    @staticmethod
    def _unit_of(after: str) -> str:
        if after in _UNIT_KG:
            return "kg"
        if after in _UNIT_G:
            return "g"
        if after in _UNIT_COUNT:
            return "count"
        if after in _UNIT_KM:
            return "km"
        if after.startswith("kilo") and "m" in after:
            return "km"  # kilometer, kilomitar (folded किलोमीटर)
        if after.startswith("kilo"):
            return "kg"
        if "dozen" in after:
            return "count"
        return ""

    # -- distance ----------------------------------------------------------
    def _distance_km(self, tokens: list[_Tok]) -> int | None:
        """First stated distance in km, or None when the customer gave none."""
        for index, token in enumerate(tokens):
            if token.number is None:
                continue
            after = tokens[index + 1].key if index + 1 < len(tokens) else ""
            if self._unit_of(after) != "km":
                continue
            return max(0, int(round(token.number)))
        return None

    # -- items -------------------------------------------------------------
    def _fill(
        self,
        window: _Window,
        tokens: list[_Tok],
        text: str,
        keys: list[str],
        resolution: Resolution,
    ) -> None:
        sku, choice_notes = self._choose_sku(window, keys)
        if sku is None:
            resolution.unresolved.extend(choice_notes)
            return
        qty, approximated, extra = self._quantity_for_sku(sku, window, keys)
        resolution.unresolved.extend(extra)
        if qty <= 0:
            # A question about the product ("is the kaju katli vegetarian?") is not
            # an order, so the pack choice never had to be made.
            return
        resolution.unresolved.extend(choice_notes)

        item = ResolvedOrderItem(
            sku=sku.sku,
            item=sku.item,
            pack=sku.pack,
            qty=qty,
            unit_price_inr=sku.price_inr,
            matched_text=self._matched_text(window, tokens, text),
            approximated=approximated,
        )
        existing = next((i for i in resolution.items if i.sku == sku.sku), None)
        if existing is None:
            resolution.items.append(item)
        else:
            existing.qty += qty
            existing.approximated = existing.approximated or approximated
            existing.matched_text = f"{existing.matched_text} + {item.matched_text}"

    def _matched_text(self, window: _Window, tokens: list[_Tok], text: str) -> str:
        """The phrase the item was read from, quantity included."""
        start, end = window.start, window.end
        quantity = window.quantity
        if quantity is not None:
            start = min(start, quantity.index)
            if quantity.unit or quantity.index < window.start:
                end = max(end, quantity.index + 1)
        return text[tokens[start].start:tokens[end - 1].end].strip()

    def _choose_sku(
        self, window: _Window, keys: list[str]
    ) -> tuple[SKU | None, list[str]]:
        """Best SKU for this mention, plus notes about anything assumed."""
        notes: list[str] = []
        cands = list(window.cands)

        # A qualifier beside the mention ("sugar free", "bada") outranks a longer
        # alias match, so "sugar free kaju katli" cannot land on plain katli.
        near = set(keys[max(0, window.start - 3):min(len(keys), window.end + 3)])
        qualified = [
            (len(near & self._qualifiers[sku_id]), quality, span, sku_id)
            for quality, span, sku_id in cands
        ]
        best_hits = max(hits for hits, _, _, _ in qualified)
        if best_hits:
            cands = [
                (quality, span, sku_id)
                for hits, quality, span, sku_id in qualified
                if hits == best_hits
            ]

        grams = window.quantity.grams if window.quantity is not None else None
        ranked: list[tuple[tuple[int, int, float, int, str], SKU]] = []
        for quality, span, sku_id in cands:
            sku = self.corpus.sku(sku_id)
            if sku is None:
                continue
            if grams is not None:
                if sku.pack_grams is None:
                    continue
                ratio = grams / sku.pack_grams
                exact = 0 if abs(ratio - round(ratio)) < 1e-9 else 1
                packs = max(1, math.ceil(ratio - 1e-9))
            else:
                exact, packs = 0, 1
            # An exact pack fit first, then the fewest packs, then the best
            # phrase match: "500 g kaju katli" is KK-500, "2 kg kaju katli" is
            # two KK-1000 rather than four KK-500.
            key = (exact, packs, -quality, -span, sku_id)
            ranked.append((key, sku))
        if not ranked:
            if grams is not None:
                names = ", ".join(
                    self.corpus.sku(sku_id).item
                    for _, _, sku_id in window.cands[:2]
                    if self.corpus.sku(sku_id)
                )
                notes.append(
                    f"'{window.text}' is sold as {names} by the box, so "
                    f"{window.quantity.describe()} cannot be priced from the catalog."
                )
            return None, notes

        ranked.sort(key=lambda entry: entry[0])
        best_key, best = ranked[0]
        rivals = [
            sku
            for key, sku in ranked[1:]
            if key[0] == best_key[0] and key[1] == best_key[1]
            and key[4] != best_key[4]
            and sku.pack != best.pack
        ]
        if rivals:
            options = ", ".join(f"{sku.sku} ({sku.pack})" for sku in rivals[:2])
            notes.append(
                f"pack size not stated: using {best.sku} ({best.pack}); "
                f"also available as {options}"
            )
        return best, notes

    def _quantity_for_sku(
        self, sku: SKU, window: _Window, keys: Sequence[str] = ()
    ) -> tuple[int, bool, list[str]]:
        """How many packs of ``sku`` this mention means, and what was assumed."""
        quantity = window.quantity
        if quantity is None:
            if not self._has_intent(window, keys):
                return 0, False, [
                    f"{sku.item} was mentioned without a quantity, so no order "
                    f"line was created."
                ]
            return 1, False, []

        grams = quantity.grams
        if grams is None:
            count = int(round(quantity.value))
            if count <= 0:
                return 0, False, []
            if sku.pack_grams is not None:
                return count, False, [
                    f"{sku.item} is sold in packs of {sku.pack}; read "
                    f"{quantity.describe()} as {count} pack(s)."
                ]
            return count, False, []

        if sku.pack_grams is None:
            return 0, False, [
                f"{quantity.describe()} cannot be priced: {sku.item} ({sku.pack}) "
                f"is not sold by weight."
            ]
        ratio = grams / sku.pack_grams
        if abs(ratio - round(ratio)) < 1e-9:
            return max(1, int(round(ratio))), False, []
        packs = max(1, math.ceil(ratio - 1e-9))
        return packs, True, [
            f"{quantity.describe()} of {sku.item} is not a whole number of "
            f"{sku.pack} packs; rounded up to {packs} pack(s)."
        ]

    @staticmethod
    def _has_intent(window: _Window, keys: list[str]) -> bool:
        """Whether the message reads like an order, not a question about a product."""
        everywhere = set(keys)
        if _QUESTION_MARKERS & everywhere:
            return False
        if _QUOTE_INTENT & everywhere:
            return True
        near = set(
            keys[max(0, window.start - _INTENT_RADIUS):window.end + _INTENT_RADIUS]
        )
        return bool(_ORDER_INTENT & near)

    # -- cross-turn --------------------------------------------------------
    def _carry_over(
        self,
        windows: list[_Window],
        tokens: list[_Tok],
        text: str,
        carry: Iterable[ResolvedOrderItem],
        resolution: Resolution,
    ) -> None:
        """Apply a product-less quantity to an item from an earlier turn."""
        previous = list(carry)
        if not previous:
            return
        used = {w.quantity.index for w in windows if w.quantity is not None}
        free = [
            quantity
            for index in range(len(tokens))
            if index not in used
            for quantity in (self._number_at(tokens, index, text),)
            if quantity is not None
        ]
        if not free:
            return

        for item in previous:
            if not free or any(i.sku == item.sku for i in resolution.items):
                continue
            sku = self.corpus.sku(item.sku)
            if sku is None:
                continue
            position = _pick_carried(free, sku)
            if position is None:
                continue
            quantity = free.pop(position)
            probe = _Window(start=0, end=0, text="", quantity=quantity)
            qty, approximated, notes = self._quantity_for_sku(sku, probe)
            if qty <= 0:
                continue
            if quantity.more:
                qty += item.qty
            resolution.items.append(
                ResolvedOrderItem(
                    sku=sku.sku,
                    item=sku.item,
                    pack=sku.pack,
                    qty=qty,
                    unit_price_inr=sku.price_inr,
                    matched_text=f"{quantity.describe()} of the {sku.item} under discussion",
                    approximated=approximated or item.approximated,
                )
            )
            resolution.unresolved.append(
                f"{quantity.describe()} was given without naming a product; applied "
                f"to the {sku.item} ({sku.sku}) being discussed."
            )
            resolution.unresolved.extend(notes)


def _pick_carried(free: list[_Quantity], sku: SKU) -> int | None:
    """Index of the free quantity that suits ``sku``: weight first, then count."""
    for wanted in ("kg", "g", ""):
        for index, quantity in enumerate(free):
            if quantity.unit != wanted:
                continue
            if wanted in ("kg", "g") and sku.pack_grams is None:
                continue
            return index
    if sku.pack_grams is None:
        for index, quantity in enumerate(free):
            if quantity.unit == "count":
                return index
    return None


def _tokenize(text: str) -> list[_Tok]:
    """Tokenise with positions, keeping numbers attached to what follows.

    Positions refer to the message as it arrived, so matched text quotes the
    customer's own words; each token is folded separately for matching, which is
    why "काजू कटली" and "kaju katli" land on the same aliases.
    """
    tokens: list[_Tok] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group()
        key = re.sub(r"[\s_]+", " ", fold_devanagari(raw).casefold()).strip()
        if not key:
            continue
        digits = key.replace(",", "").replace(" ", "")
        number: float | None = None
        if re.fullmatch(r"\d+(?:\.\d+)?", digits):
            number = float(digits)
        tokens.append(
            _Tok(key=key, raw=raw, start=match.start(), end=match.end(), number=number)
        )
    return tokens
