"""Billing tests: the four graded seed cases plus every pricing rule.

Amounts are asserted as ``int`` throughout, because a float anywhere in the
chain is a rupee the shop cannot actually charge.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from meher_agent.config import (
    POLICY_BULK_ADVANCE_PCT,
    POLICY_COD_LIMIT_INR,
    POLICY_DELIVERY_FEE_INR,
    POLICY_DELIVERY_RADIUS_KM,
    POLICY_DISCOUNT_PCT,
    POLICY_FREE_DELIVERY_ABOVE_INR,
    REPO_ROOT,
    load_config,
)
from meher_agent.data.corpus import Corpus
from meher_agent.grounding.billing import (
    BillingEngine,
    pack_weight_grams,
    round_half_up,
)
from meher_agent.types import Quote, Resolution, ResolvedOrderItem

DATA_DIR = Path(REPO_ROOT) / "data"


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus(DATA_DIR)


@pytest.fixture(scope="module")
def engine(corpus: Corpus) -> BillingEngine:
    return BillingEngine(corpus, load_config())


def resolution(
    corpus: Corpus, lines: list[tuple[str, int]], distance_km: int | None = None
) -> Resolution:
    items = []
    for sku_id, qty in lines:
        sku = corpus.sku(sku_id)
        assert sku is not None, sku_id
        items.append(
            ResolvedOrderItem(
                sku=sku.sku,
                item=sku.item,
                pack=sku.pack,
                qty=qty,
                unit_price_inr=sku.price_inr,
            )
        )
    return Resolution(items=items, distance_km=distance_km)


def quote_for(
    engine: BillingEngine,
    corpus: Corpus,
    lines: list[tuple[str, int]],
    distance_km: int | None = None,
) -> Quote:
    return engine.quote(resolution(corpus, lines, distance_km))


# --------------------------------------------------------------------------
# The four graded seed cases
# --------------------------------------------------------------------------


def test_seed_case_1_kaju_katli_and_large_gift_box(
    engine: BillingEngine, corpus: Corpus
) -> None:
    # "I want 2 kg kaju katli and one large Diwali gift box, delivered 5 km from
    # your shop. What is the total including delivery?"
    quote = quote_for(engine, corpus, [("KK-1000", 2), ("GBL", 1)], 5)

    assert [line.line_total_inr for line in quote.lines] == [2400, 1450]
    assert quote.subtotal_inr == 3850
    assert quote.discount_inr == 0
    assert quote.discount_pct == 0
    assert quote.delivery_fee_inr == 0
    assert quote.delivery_free is True
    assert quote.delivery_possible is True
    assert quote.total_inr == 3850
    assert quote.requires_bulk_notice is False
    assert quote.advance_inr == 0


def test_seed_case_2_motichoor_laddoo_and_samosas(
    engine: BillingEngine, corpus: Corpus
) -> None:
    # "What is the price of 1 kg motichoor laddoo?" then "Make it 3 kg, and add
    # 10 samosas. Deliver to 2 km away. Total?"
    single = quote_for(engine, corpus, [("ML-1000", 1)])
    assert single.subtotal_inr == 560
    assert single.total_inr == 620  # 560 plus the 60 delivery fee

    order = quote_for(engine, corpus, [("ML-1000", 3), ("SM-1", 10)], 2)
    assert [line.line_total_inr for line in order.lines] == [1680, 200]
    assert order.subtotal_inr == 1880
    assert order.discount_inr == 0
    assert order.delivery_fee_inr == 0
    assert order.total_inr == 1880


def test_seed_case_3_sixty_small_gift_boxes_get_five_percent(
    engine: BillingEngine, corpus: Corpus
) -> None:
    # "Bhaiya office ke liye 60 small gift box chahiye 2 November tak. Kitna
    # padega, koi discount milega?"
    quote = quote_for(engine, corpus, [("GBS", 60)])

    assert quote.subtotal_inr == 39000
    assert quote.discount_pct == 5
    assert quote.discount_inr == 1950
    assert quote.delivery_fee_inr == 0
    assert quote.total_inr == 37050
    assert quote.requires_bulk_notice is True
    assert quote.advance_pct == 30
    assert quote.advance_inr == 11115


def test_seed_case_4_thirty_large_gift_boxes_get_no_discount(
    engine: BillingEngine, corpus: Corpus
) -> None:
    # "We need 30 large gift boxes for our office Diwali party on 3 November."
    # The trap: 30 is not 50, so the 5% discount must not appear.
    quote = quote_for(engine, corpus, [("GBL", 30)])

    assert quote.subtotal_inr == 43500
    assert quote.discount_pct == 0
    assert quote.discount_inr == 0
    assert quote.delivery_fee_inr == 0
    assert quote.total_inr == 43500
    assert quote.requires_bulk_notice is True
    assert quote.advance_inr == 13050


# --------------------------------------------------------------------------
# Discount threshold
# --------------------------------------------------------------------------


@pytest.mark.parametrize("boxes", [1, 25, 30, 49])
def test_no_discount_below_fifty_boxes(
    engine: BillingEngine, corpus: Corpus, boxes: int
) -> None:
    quote = quote_for(engine, corpus, [("GBS", boxes)])
    assert quote.discount_inr == 0
    assert quote.discount_pct == 0
    assert quote.total_inr == quote.subtotal_inr + (quote.delivery_fee_inr or 0)


def test_discount_applies_at_exactly_fifty_boxes(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("GBS", 50)])
    assert quote.discount_pct == POLICY_DISCOUNT_PCT
    assert quote.discount_inr == 1625
    assert quote.total_inr == 30875


def test_discount_counts_both_box_sizes_together(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("GBS", 30), ("GBL", 20)])
    gift_subtotal = 650 * 30 + 1450 * 20
    assert gift_subtotal == 48500
    assert quote.discount_inr == 2425
    assert quote.total_inr == 48500 - 2425


def test_discount_is_computed_on_the_gift_box_subtotal_only(
    engine: BillingEngine, corpus: Corpus
) -> None:
    # Rs 32,500 of gift boxes plus Rs 12,000 of sweets: only the boxes are cut.
    quote = quote_for(engine, corpus, [("GBS", 50), ("KK-1000", 10)])

    assert quote.subtotal_inr == 32500 + 12000
    assert quote.discount_inr == 1625
    assert quote.total_inr == 44500 - 1625
    assert quote.discount_inr != round_half_up(quote.subtotal_inr * POLICY_DISCOUNT_PCT, 100)


def test_discount_ignores_non_gift_box_lines(
    engine: BillingEngine, corpus: Corpus
) -> None:
    # Rs 1,50,000 of sweets is worth no discount at all.
    quote = quote_for(engine, corpus, [("KK-1000", 125)])
    assert quote.subtotal_inr == 150000
    assert quote.discount_inr == 0
    assert quote.discount_pct == 0


def test_discount_rounds_half_up_on_an_odd_subtotal(
    engine: BillingEngine, corpus: Corpus
) -> None:
    # The documented case: 3 x 650 = 1950 and 5% of that is 97.5, which must
    # round to 98. Three boxes are under the threshold, so the engine applies no
    # discount; 51 boxes produce the same half-rupee through the engine.
    assert round_half_up(1950 * POLICY_DISCOUNT_PCT, 100) == 98
    assert round_half_up(650 * 51 * POLICY_DISCOUNT_PCT, 100) == 1658

    under_threshold = quote_for(engine, corpus, [("GBS", 3)])
    assert under_threshold.subtotal_inr == 1950
    assert under_threshold.discount_inr == 0

    over_threshold = quote_for(engine, corpus, [("GBS", 51)])
    assert over_threshold.discount_inr == 1658
    assert over_threshold.total_inr == 650 * 51 - 1658


def test_only_ever_five_or_zero_percent_discount(
    engine: BillingEngine, corpus: Corpus
) -> None:
    for boxes in (3, 49, 50, 60, 120):
        quote = quote_for(engine, corpus, [("GBS", boxes)])
        assert quote.discount_pct in {0, POLICY_DISCOUNT_PCT}


def test_round_half_up_is_integer_only() -> None:
    assert round_half_up(1, 2) == 1
    assert round_half_up(1, 3) == 0
    assert round_half_up(2, 3) == 1
    assert round_half_up(650, 20) == 33
    assert round_half_up(1950, 20) == 98
    assert round_half_up(3075, 100) == 31
    assert type(round_half_up(1950, 20)) is int


# --------------------------------------------------------------------------
# Delivery
# --------------------------------------------------------------------------


def test_delivery_fee_below_the_free_threshold(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("SP-500", 4)], 3)  # 4 x 240 = 960
    assert quote.subtotal_inr == 960
    assert quote.delivery_free is False
    assert quote.delivery_fee_inr == POLICY_DELIVERY_FEE_INR
    assert quote.total_inr == 1020


def test_delivery_is_free_at_or_above_the_threshold(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("KK-500", 1), ("SP-500", 1), ("AB-400", 1)], 1)
    assert quote.subtotal_inr == 620 + 240 + 150
    assert quote.delivery_free is True
    assert quote.delivery_fee_inr == 0
    assert quote.total_inr == quote.subtotal_inr


def test_delivery_threshold_boundary(engine: BillingEngine) -> None:
    # Every catalog price is even, so no order can total exactly Rs 999; the
    # boundary is therefore asserted on the rule itself.
    assert engine._delivery(999, has_lines=True, distance_km=2) == (0, True, True)
    assert engine._delivery(998, has_lines=True, distance_km=2) == (60, False, True)
    assert engine._delivery(1000, has_lines=True, distance_km=2) == (0, True, True)


def test_delivery_is_impossible_beyond_eight_km(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("KK-1000", 1)], 9)
    assert quote.delivery_possible is False
    assert quote.delivery_fee_inr is None
    assert quote.delivery_free is False
    assert quote.total_inr == 1200
    assert any(str(POLICY_DELIVERY_RADIUS_KM) in note for note in quote.notes)


def test_delivery_at_exactly_eight_km_is_possible(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("KK-1000", 1)], POLICY_DELIVERY_RADIUS_KM)
    assert quote.delivery_possible is True
    assert quote.delivery_free is True


# --------------------------------------------------------------------------
# Bulk notice and advance
# --------------------------------------------------------------------------


def test_bulk_notice_at_eleven_kilograms(engine: BillingEngine, corpus: Corpus) -> None:
    quote = quote_for(engine, corpus, [("KK-1000", 11)])
    assert quote.requires_bulk_notice is True
    assert quote.advance_pct == POLICY_BULK_ADVANCE_PCT
    assert quote.advance_inr == 3960  # 30% of 13200


def test_no_bulk_notice_at_exactly_ten_kilograms(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("KK-1000", 10)])
    assert quote.requires_bulk_notice is False
    assert quote.advance_pct == 0
    assert quote.advance_inr == 0


def test_bulk_notice_at_twenty_six_boxes(engine: BillingEngine, corpus: Corpus) -> None:
    quote = quote_for(engine, corpus, [("GBS", 26)])
    assert quote.requires_bulk_notice is True
    assert quote.advance_inr == 5070  # 30% of 16900


def test_no_bulk_notice_at_exactly_twenty_five_boxes(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("GBS", 25)])
    assert quote.requires_bulk_notice is False
    assert quote.advance_inr == 0


def test_advance_is_thirty_percent_of_the_payable_total_after_discount(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("GBS", 60)], 4)
    payable = quote.subtotal_inr - quote.discount_inr
    assert payable == 37050
    assert quote.total_inr == payable
    assert quote.advance_inr == 11115
    assert quote.advance_inr == payable * 30 // 100


def test_advance_comes_from_the_payable_total_not_the_subtotal(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("GBS", 50)], 4)
    payable = 32500 - 1625
    assert quote.total_inr == payable
    # half-up, so not floor division: 9262.5 becomes 9263
    assert quote.advance_inr == 9263
    assert quote.advance_inr == round_half_up(payable * POLICY_BULK_ADVANCE_PCT, 100)
    assert quote.advance_inr != 32500 * 30 // 100


# --------------------------------------------------------------------------
# Edge cases and integer discipline
# --------------------------------------------------------------------------


def test_empty_resolution_is_a_zero_quote_with_no_delivery_fee(
    engine: BillingEngine,
) -> None:
    quote = engine.quote(Resolution())
    assert quote.lines == []
    assert quote.subtotal_inr == 0
    assert quote.total_inr == 0
    assert quote.discount_inr == 0
    assert quote.delivery_fee_inr is None
    assert quote.delivery_free is False
    assert quote.delivery_possible is True
    assert quote.requires_bulk_notice is False
    assert quote.advance_inr == 0
    assert quote.all_amounts() == []


def test_every_quote_value_is_an_int(engine: BillingEngine, corpus: Corpus) -> None:
    quote = quote_for(engine, corpus, [("GBS", 60), ("KK-1000", 3)], 3)
    values: list[int] = [
        quote.subtotal_inr,
        quote.discount_pct,
        quote.discount_inr,
        quote.delivery_fee_inr,
        quote.total_inr,
        quote.advance_pct,
        quote.advance_inr,
    ]
    values.extend(line.unit_price_inr for line in quote.lines)
    values.extend(line.line_total_inr for line in quote.lines)
    for value in values:
        assert type(value) is int, value
    for value in quote.all_amounts():
        assert type(value) is int


def test_catalog_price_wins_over_a_resolver_price(
    engine: BillingEngine, corpus: Corpus
) -> None:
    tampered = Resolution(
        items=[
            ResolvedOrderItem(
                sku="ML-1000",
                item="Motichoor Laddoo",
                pack="1 kg",
                qty=2,
                unit_price_inr=1,
            )
        ]
    )
    quote = engine.quote(tampered)
    assert quote.lines[0].unit_price_inr == 560
    assert quote.subtotal_inr == 1120


def test_compound_pack_weight_is_summed() -> None:
    assert pack_weight_grams("1 kg") == 1000
    assert pack_weight_grams("500 g") == 500
    assert pack_weight_grams("1 box") == 0
    assert pack_weight_grams("1 piece") == 0
    assert pack_weight_grams("1 kg assorted sweets and 200 g dry fruits") == 1200


def test_notes_never_carry_an_uncomputed_rupee_amount(
    engine: BillingEngine, corpus: Corpus
) -> None:
    polluted = resolution(corpus, [("GBS", 60)])
    polluted.unresolved = ["Could not price the wedding cake, maybe Rs 9,999.", "Check the date."]
    quote = engine.quote(polluted)
    assert not any("9,999" in note for note in quote.notes)
    assert any("Check the date." in note for note in quote.notes)


def test_gift_box_preorder_note_tracks_today(
    engine: BillingEngine, corpus: Corpus
) -> None:
    open_order = engine.quote(resolution(corpus, [("GBL", 1)]), today=date(2026, 10, 1))
    closed_order = engine.quote(resolution(corpus, [("GBL", 1)]), today=date(2026, 11, 6))
    assert any("close on" in note for note in open_order.notes)
    assert any("closed on" in note for note in closed_order.notes)
    assert open_order.total_inr == closed_order.total_inr == 1450


def test_cash_on_delivery_note_appears_above_the_limit(
    engine: BillingEngine, corpus: Corpus
) -> None:
    small = quote_for(engine, corpus, [("ML-1000", 1)], 1)
    big = quote_for(engine, corpus, [("GBS", 60)], 1)
    assert not any("5,000" in note for note in small.notes)
    assert any("5,000" in note for note in big.notes)
    assert POLICY_COD_LIMIT_INR == 5000


def test_quote_is_independent_of_input_resolution_mutation(
    engine: BillingEngine, corpus: Corpus
) -> None:
    source = resolution(corpus, [("KK-1000", 2)], 5)
    quote = engine.quote(source)
    snapshot = (quote.subtotal_inr, quote.total_inr)
    source.items[0].qty = 99
    assert (quote.subtotal_inr, quote.total_inr) == snapshot


def test_line_item_exposes_the_computed_values(
    engine: BillingEngine, corpus: Corpus
) -> None:
    quote = quote_for(engine, corpus, [("KK-1000", 2), ("GBS", 1)])
    first, second = quote.lines
    assert (first.sku, first.qty, first.unit_price_inr, first.line_total_inr) == (
        "KK-1000",
        2,
        1200,
        2400,
    )
    assert (second.sku, second.line_total_inr) == ("GBS", 650)
    assert set(quote.all_amounts()) >= {2400, 650, 3050, 1200}
