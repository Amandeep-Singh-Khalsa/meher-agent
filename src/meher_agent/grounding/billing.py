"""Deterministic rupee arithmetic.

The model never produces money, it only verbalises what this module computed.
Every value is an ``int``: percentages go through :func:`round_half_up`, which
is exact integer maths, so a 5% discount on an odd gift-box subtotal
(1950 -> 97.5 -> 98) cannot depend on binary floating point.

Policy constants come from data/policies.md via meher_agent.config; nothing
about the pricing rules is hardcoded here beyond which SKU type earns the
discount.
"""

from __future__ import annotations

import re
from datetime import date

from ..config import (
    POLICY_BULK_ADVANCE_PCT,
    POLICY_BULK_BOXES,
    POLICY_BULK_KG,
    POLICY_COD_LIMIT_INR,
    POLICY_DELIVERY_FEE_INR,
    POLICY_DELIVERY_RADIUS_KM,
    POLICY_DISCOUNT_MIN_BOXES,
    POLICY_DISCOUNT_PCT,
    POLICY_FREE_DELIVERY_ABOVE_INR,
    Config,
)
from ..data.corpus import Corpus
from ..grounding.amounts import extract_rupee_amounts, format_inr
from ..types import LineItem, Quote, Resolution, ResolvedOrderItem, SKU

#: data/prices.csv marks the Diwali boxes with this value in its "type" column.
GIFT_BOX_TYPE = "gift box"

#: "Gift boxes can be pre-ordered until 5 November 2026."
GIFT_BOX_PREORDER_LAST_DAY = date(2026, 11, 5)

_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u0964])\s+|\n+")

_WEIGHT_MULTIPLIER = {"kg": 1000, "g": 1}


def inr(amount: int) -> str:
    """Customer-facing rupee text, via grounding.amounts.format_inr.

    format_inr currently leaves a trailing thousands separator on every value
    of 1000 or more ("3,850,"), which reads as a typo in a sentence, so the
    stray separator is stripped here rather than in customer text.
    """
    return format_inr(amount).rstrip(",")


def round_half_up(numerator: int, denominator: int) -> int:
    """floor(numerator / denominator + 0.5) using integers only.

    Half-up rather than banker's rounding so that a 5% discount on a gift-box
    subtotal of 1950 is 98 rupees, every time, on every platform.
    """
    return (2 * numerator + denominator) // (2 * denominator)


def pack_weight_grams(pack: str) -> int:
    """Total grams in a pack description, summing compound ones such as
    "1 kg assorted sweets and 200 g dry fruits". Pieces and boxes weigh 0."""
    total = 0
    for value, unit in _WEIGHT_RE.findall(pack or ""):
        grams = int(float(value) * _WEIGHT_MULTIPLIER[unit.lower()])
        total += grams
    return total


def is_gift_box(sku: SKU | None, item_name: str = "") -> bool:
    """True for a Diwali gift box line. Falls back to the item name for a
    resolution the catalog does not know, so the discount is never silently
    skipped because of a SKU typo."""
    if sku is not None and sku.type.strip().lower() == GIFT_BOX_TYPE:
        return True
    return GIFT_BOX_TYPE in (item_name or "").lower()


class BillingEngine:
    """Turns a deterministic :class:`Resolution` into a :class:`Quote`."""

    def __init__(self, corpus: Corpus, config: Config) -> None:
        self._corpus = corpus
        self._config = config

    def quote(self, resolution: Resolution, *, today: date | None = None) -> Quote:
        lines = [self._line(item) for item in resolution.items]
        subtotal = sum(line.line_total_inr for line in lines)

        gift_lines = [
            line
            for line in lines
            if is_gift_box(self._corpus.sku(line.sku), line.item)
        ]
        gift_qty = sum(line.qty for line in gift_lines)
        gift_subtotal = sum(line.line_total_inr for line in gift_lines)

        discount_pct, discount_inr = self._discount(gift_qty, gift_subtotal)
        payable = subtotal - discount_inr

        delivery_fee, delivery_free, delivery_possible = self._delivery(
            payable, has_lines=bool(lines), distance_km=resolution.distance_km
        )
        total = payable + (delivery_fee or 0)

        weight_grams = self._weight_grams(lines)
        requires_bulk_notice = (
            weight_grams > POLICY_BULK_KG * 1000 or gift_qty > POLICY_BULK_BOXES
        )
        advance_pct = POLICY_BULK_ADVANCE_PCT if requires_bulk_notice else 0
        advance_inr = (
            round_half_up(total * advance_pct, 100) if requires_bulk_notice else 0
        )

        quote = Quote(
            lines=lines,
            subtotal_inr=subtotal,
            discount_pct=discount_pct,
            discount_inr=discount_inr,
            delivery_fee_inr=delivery_fee,
            delivery_free=delivery_free,
            delivery_possible=delivery_possible,
            total_inr=total,
            advance_pct=advance_pct,
            advance_inr=advance_inr,
            requires_bulk_notice=requires_bulk_notice,
        )
        quote.notes = self._notes(
            resolution,
            quote,
            gift_qty=gift_qty,
            gift_subtotal=gift_subtotal,
            weight_grams=weight_grams,
            today=today,
        )
        return quote

    # -- pieces ------------------------------------------------------------

    def _line(self, item: ResolvedOrderItem) -> LineItem:
        sku = self._corpus.sku(item.sku)
        # prices.csv is the only source of prices, so it wins over whatever the
        # resolver carried on the item.
        unit_price = sku.price_inr if sku is not None else item.unit_price_inr
        qty = max(0, item.qty)
        return LineItem(
            sku=item.sku,
            item=item.item or (sku.item if sku is not None else ""),
            pack=item.pack or (sku.pack if sku is not None else ""),
            qty=qty,
            unit_price_inr=unit_price,
            line_total_inr=qty * unit_price,
        )

    def _discount(self, gift_qty: int, gift_subtotal: int) -> tuple[int, int]:
        """The only discount in the system: 5% of the gift-box subtotal, and
        only from 50 boxes. Never on the whole order, never on other items."""
        if gift_qty < POLICY_DISCOUNT_MIN_BOXES or gift_subtotal <= 0:
            return 0, 0
        return POLICY_DISCOUNT_PCT, round_half_up(gift_subtotal * POLICY_DISCOUNT_PCT, 100)

    def _delivery(
        self, payable: int, *, has_lines: bool, distance_km: int | None
    ) -> tuple[int | None, bool, bool]:
        if not has_lines:
            return None, False, True
        if distance_km is not None and distance_km > POLICY_DELIVERY_RADIUS_KM:
            return None, False, False
        if payable >= POLICY_FREE_DELIVERY_ABOVE_INR:
            return 0, True, True
        return POLICY_DELIVERY_FEE_INR, False, True

    def _weight_grams(self, lines: list[LineItem]) -> int:
        total = 0
        for line in lines:
            sku = self._corpus.sku(line.sku)
            pack = line.pack or (sku.pack if sku is not None else "")
            total += pack_weight_grams(pack) * line.qty
        return total

    # -- narration ---------------------------------------------------------

    def _notes(
        self,
        resolution: Resolution,
        quote: Quote,
        *,
        gift_qty: int,
        gift_subtotal: int,
        weight_grams: int,
        today: date | None,
    ) -> list[str]:
        notes: list[str] = []

        if quote.discount_inr > 0:
            notes.append(
                f"{POLICY_DISCOUNT_PCT}% discount of Rs {inr(quote.discount_inr)} "
                f"applied to the gift-box subtotal of Rs {inr(gift_subtotal)} only "
                f"({gift_qty} boxes)."
            )
        elif gift_qty > 0:
            notes.append(
                f"No discount: the {POLICY_DISCOUNT_PCT}% gift-box discount starts at "
                f"{POLICY_DISCOUNT_MIN_BOXES} boxes and this order has {gift_qty}."
            )

        if not quote.delivery_possible:
            notes.append(
                f"We do not deliver beyond {POLICY_DELIVERY_RADIUS_KM} km; the customer can "
                f"pick up from the shop or book their own courier."
            )
        elif quote.delivery_fee_inr is None:
            pass
        elif quote.delivery_free:
            notes.append(
                f"Delivery is free: the payable total of Rs {inr(quote.total_inr)} is at or "
                f"above Rs {inr(POLICY_FREE_DELIVERY_ABOVE_INR)}."
            )
        else:
            notes.append(
                f"Delivery fee Rs {inr(quote.delivery_fee_inr)} because the payable total is "
                f"below Rs {inr(POLICY_FREE_DELIVERY_ABOVE_INR)}."
            )

        if quote.requires_bulk_notice:
            notes.append(
                f"Bulk order: a {POLICY_BULK_ADVANCE_PCT}% advance of Rs "
                f"{inr(quote.advance_inr)} and 3 days' notice are needed."
            )
        if quote.total_inr > POLICY_COD_LIMIT_INR:
            notes.append(
                f"Cash on delivery is only available up to Rs "
                f"{inr(POLICY_COD_LIMIT_INR)}; this order is settled by UPI, card or "
                f"advance."
            )

        if gift_qty > 0:
            if today is not None and today > GIFT_BOX_PREORDER_LAST_DAY:
                notes.append(
                    f"Gift-box pre-orders closed on {GIFT_BOX_PREORDER_LAST_DAY.day} "
                    f"{GIFT_BOX_PREORDER_LAST_DAY.strftime('%B %Y')}; check stock before promising."
                )
            else:
                notes.append(
                    f"Gift-box pre-orders close on {GIFT_BOX_PREORDER_LAST_DAY.day} "
                    f"{GIFT_BOX_PREORDER_LAST_DAY.strftime('%B %Y')}."
                )

        notes.extend(_safe_notes(resolution.unresolved, quote))
        return notes


def _safe_notes(raw: list[str], quote: Quote) -> list[str]:
    """Drop unresolved-note sentences that carry a rupee amount billing never
    computed. The notes are shown to the model, so they must not become a
    back door around the guard."""
    allowed = set(quote.all_amounts())
    kept: list[str] = []
    for note in raw:
        for sentence in _SENTENCE_SPLIT.split(note or ""):
            sentence = sentence.strip()
            if not sentence:
                continue
            if any(value not in allowed for value in extract_rupee_amounts(sentence)):
                continue
            kept.append(sentence)
    return kept
