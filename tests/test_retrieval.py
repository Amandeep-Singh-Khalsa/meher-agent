"""Retrieval tests: transliteration, lexicon, section/SKU retrieval, order
resolution, and the briefing the agent actually calls.

The numbers asserted here come from data/ (prices.csv, business.md,
policies.md) and from the policy the resolver is allowed to apply: pack multiples
and the 8 km delivery radius. Nothing is invented; if a price in data/ changes,
these tests are supposed to fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meher_agent.config import REPO_ROOT, RetrievalConfig, load_config
from meher_agent.data.corpus import Corpus
from meher_agent.retrieval.pipeline import RetrievalPipeline
from meher_agent.retrieval.resolver import OrderResolver
from meher_agent.retrieval.retriever import Retriever
from meher_agent.text.lexicon import SYNONYMS, expand, tokenize
from meher_agent.text.translit import detect_language, devanagari_ratio, fold_devanagari
from meher_agent.types import TurnRecord

DATA_DIR = Path(REPO_ROOT) / "data"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> RetrievalConfig:
    return load_config().retrieval


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus(DATA_DIR)


@pytest.fixture(scope="module")
def retriever(corpus: Corpus, config: RetrievalConfig) -> Retriever:
    return Retriever(corpus, config)


@pytest.fixture(scope="module")
def resolver(corpus: Corpus, config: RetrievalConfig) -> OrderResolver:
    return OrderResolver(corpus, config)


@pytest.fixture(scope="module")
def pipeline(corpus: Corpus) -> RetrievalPipeline:
    return RetrievalPipeline(corpus, load_config())


def skus(resolver: OrderResolver, message: str) -> list[tuple[str, int]]:
    return [(item.sku, item.qty) for item in resolver.resolve(message).items]


def section_ids(retriever: Retriever, query: str) -> list[str]:
    return [hit.section.source_id for hit in retriever.sections(query)]


def sku_ids(retriever: Retriever, query: str) -> list[str]:
    return [hit.sku.sku for hit in retriever.skus(query)]


# --------------------------------------------------------------------------
# text.translit
# --------------------------------------------------------------------------


def test_fold_devanagari_maps_product_names_onto_the_catalog() -> None:
    assert "kaju" in fold_devanagari("काजू कटली")
    assert "katali" in fold_devanagari("काजू कटली")
    assert "gulab" in fold_devanagari("गुलाब जामुन")
    # Folded spelling is literal: "समोसे" is "samose". The "samosa" spelling the
    # catalog uses is a lexical variant, and lives in the lexicon and aliases.
    assert fold_devanagari("समोसे") == "samose"


def test_fold_devanagari_drops_final_schwa_and_normalises_digits() -> None:
    assert fold_devanagari("दाम") == "dam"
    assert fold_devanagari("१२ किलो").split()[0] == "12"


def test_devanagari_ratio_separates_scripts() -> None:
    assert devanagari_ratio("काजू कटली") == 1.0
    assert devanagari_ratio("kaju katli") == 0.0
    assert 0.5 < devanagari_ratio("2 kg काजू कटली चाहिए") < 1.0


def test_detect_language_picks_exactly_one_tag() -> None:
    assert detect_language("क्या आप 12 किलोमीटर दूर डिलीवरी करते हैं?") == "hindi"
    assert detect_language("Bhaiya office ke liye 60 gift box chahiye") == "hinglish"
    assert detect_language("I want 10 samosas, please") == "english"


# --------------------------------------------------------------------------
# text.lexicon
# --------------------------------------------------------------------------


def test_expand_reaches_the_rest_of_the_product_family() -> None:
    expanded = expand("kaju")
    assert "katli" in expanded
    assert "katali" in expanded
    assert "kaju" in expanded


def test_expand_stops_at_one_hop() -> None:
    # "laddoo" may be motichoor or besan, so both are fair game, but a family that
    # never mentioned the token must not be dragged in through one of them.
    assert {"motichoor", "besan"} <= expand("laddoo")
    assert "samosa" not in expand("laddoo")


def test_expand_of_a_function_word_stays_small() -> None:
    assert len(expand("kitna")) < len(expand("kaju"))
    assert "dam" in expand("kitna")


def test_lexicon_covers_the_products_the_data_ships() -> None:
    folded = " ".join(fold_devanagari(text) for text in SYNONYMS)
    for product in ("kaju", "soan", "papdi", "gulab", "jamun", "laddoo", "samosa"):
        assert product in folded


def test_tokenize_keeps_a_decimal_together() -> None:
    assert tokenize("2.5 kg kaju katli") == ["2.5", "kg", "kaju", "katli"]
    assert tokenize("30 boxes") == ["30", "boxes"]
    # A full stop that is not between digits is not part of anything.
    assert tokenize("Rs.500") == ["rs", "500"]


# --------------------------------------------------------------------------
# Retriever: policy and business sections
# --------------------------------------------------------------------------


def test_section_retrieval_finds_the_asked_about_policy(retriever: Retriever) -> None:
    assert "policies.md#delivery" in section_ids(retriever, "do you deliver to my house?")
    assert "business.md#opening-hours" in section_ids(retriever, "what time do you close today?")
    assert "policies.md#payment" in section_ids(retriever, "can I pay by UPI?")
    assert "policies.md#returns-and-damaged-deliveries" in section_ids(
        retriever, "can I return the sweets if I do not like them?"
    )


def test_section_retrieval_anchors_below_minimum_score(retriever: Retriever) -> None:
    # "ajnabi" appears nowhere in data/, so only the anchor can bring these in.
    hits = retriever.sections("ajnabi ke liye 3 din rakhna hai, pickup milega?")
    assert "business.md#how-to-order" in [hit.section.source_id for hit in hits]


def test_section_retrieval_respects_top_k(retriever: Retriever) -> None:
    assert len(retriever.sections("kaju katli")) <= retriever.config.section_top_k


def test_a_stocked_product_is_answered_from_the_catalog(retriever: Retriever) -> None:
    # The catalog is the grounding for a product name, not a policy section: a
    # product-only message must not fall back to "how to order" with a zero score.
    assert "business.md#how-to-order" not in section_ids(retriever, "moti laddoo")
    assert retriever.skus("moti laddoo")[0].sku.sku == "ML-1000"


def test_an_unstocked_product_gets_no_catalog_row(retriever: Retriever) -> None:
    assert retriever.skus("banana chips") == []


# --------------------------------------------------------------------------
# Retriever: the price catalog
# --------------------------------------------------------------------------


def test_plain_product_name_ranks_both_packs_equally(retriever: Retriever) -> None:
    assert sku_ids(retriever, "kaju katli")[:2] == ["KK-1000", "KK-500"]
    assert {hit.sku.sku for hit in retriever.skus("kaju katli")} == {
        "KK-1000",
        "KK-500",
        "KKSF-500",
    }


def test_qualifier_picks_the_sugar_free_row(retriever: Retriever) -> None:
    assert sku_ids(retriever, "sugar free kaju katli")[0] == "KKSF-500"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("bada diwali gift box", "GBL"),
        ("large gift box", "GBL"),
        ("chhota gift box", "GBS"),
        ("small gift box", "GBS"),
        ("motichoor laddoo", "ML-1000"),
        ("besan laddoo", "BL-1000"),
        ("soan papdi", "SP-500"),
        ("gulab jamun", "GJ-1000"),
        ("rasmalai", "RM-500"),
        ("khaman dhokla", "DH-500"),
        ("aloo bhujia", "AB-400"),
        ("samosa", "SM-1"),
    ],
)
def test_one_message_resolves_to_one_catalog_row(
    retriever: Retriever, query: str, expected: str
) -> None:
    assert sku_ids(retriever, query)[0] == expected


def test_retrieved_rows_carry_catalog_prices(retriever: Retriever, corpus: Corpus) -> None:
    hit = retriever.skus("how much is 500 g of sugar-free kaju katli?")[0]
    assert hit.sku.sku == "KKSF-500"
    assert hit.sku.price_inr == 780
    assert hit.reason


def test_quantity_alone_never_invents_a_product(retriever: Retriever) -> None:
    assert retriever.skus("60 small gift box chahiye")[0].sku.sku == "GBS"
    assert sku_ids(retriever, "what is the weather like in Delhi?") == []


# --------------------------------------------------------------------------
# OrderResolver
# --------------------------------------------------------------------------


def test_resolver_reads_two_products_and_a_distance(resolver: OrderResolver) -> None:
    resolution = resolver.resolve(
        "I want 2 kg kaju katli and one large Diwali gift box, delivered 5 km "
        "from your shop. What is the total including delivery?"
    )
    assert [(item.sku, item.qty) for item in resolution.items] == [
        ("KK-1000", 2),
        ("GBL", 1),
    ]
    assert resolution.distance_km == 5
    assert not any(item.approximated for item in resolution.items)


def test_resolver_keeps_a_qualifier_with_its_product(resolver: OrderResolver) -> None:
    assert skus(resolver, "How much is 500 g of sugar-free kaju katli?") == [
        ("KKSF-500", 1)
    ]
    assert skus(resolver, "1 kg sugar free kaju katli and 2 small gift boxes") == [
        ("KKSF-500", 2),
        ("GBS", 2),
    ]


def test_resolver_uses_box_packs_for_boxes(resolver: OrderResolver) -> None:
    assert skus(resolver, "30 large gift boxes for our office") == [("GBL", 30)]
    assert skus(resolver, "Bhaiya office ke liye 60 small gift box chahiye") == [("GBS", 60)]


def test_resolver_understands_devanagari(resolver: OrderResolver) -> None:
    assert skus(resolver, "मुझे 3 बड़े डायवाली गिफ्ट बॉक्स चाहिए") == [("GBL", 3)]
    assert skus(resolver, "2 किलो काजू कटली और 1 बड़ा डायवाली गिफ्ट बॉक्स") == [
        ("KK-1000", 2),
        ("GBL", 1),
    ]
    assert resolver.resolve("क्या आप 12 किलोमीटर दूर डिलीवरी करते हैं?").distance_km == 12


def test_resolver_does_not_read_delivery_distance_as_a_weight(
    resolver: OrderResolver,
) -> None:
    resolution = resolver.resolve("Send 5 boxes of soan papdi to 3 km away")
    assert resolution.distance_km == 3
    assert [(item.sku, item.qty) for item in resolution.items] == [("SP-500", 5)]


def test_resolver_rounds_a_non_exact_weight_up_and_says_so(
    resolver: OrderResolver,
) -> None:
    resolution = resolver.resolve("2.5 kg motichoor laddoo")
    assert [(item.sku, item.qty) for item in resolution.items] == [("ML-1000", 3)]
    assert resolution.items[0].approximated is True
    assert any("2.5" in note for note in resolution.unresolved)


def test_resolver_leaves_a_qualifier_to_the_quote_owner(resolver: OrderResolver) -> None:
    # Money is never computed here: prices come from the catalog on the item.
    resolution = resolver.resolve("How much is 500 g of sugar-free kaju katli?")
    assert resolution.items[0].unit_price_inr == 780


def test_resolver_notes_a_product_with_no_quantity(resolver: OrderResolver) -> None:
    resolution = resolver.resolve("bada gift box")
    assert resolution.items == []
    assert any("without a quantity" in note for note in resolution.unresolved)


def test_resolver_carries_a_bare_quantity_from_an_earlier_turn(
    resolver: OrderResolver,
) -> None:
    first = resolver.resolve("2 kg kaju katli")
    second = resolver.resolve("Make it 3 kg", carry=first.items)
    assert [(item.sku, item.qty) for item in second.items] == [("KK-1000", 3)]
    assert any("without naming a product" in note for note in second.unresolved)


# --------------------------------------------------------------------------
# RetrievalPipeline
# --------------------------------------------------------------------------


def test_brief_sets_everything_the_answer_layer_needs(
    pipeline: RetrievalPipeline, corpus: Corpus
) -> None:
    briefing = pipeline.brief("I want 2 kg kaju katli and one large Diwali gift box", [])
    assert briefing.language == "english"
    assert [(item.sku, item.qty) for item in briefing.resolved.items] == [
        ("KK-1000", 2),
        ("GBL", 1),
    ]
    assert briefing.allowed_amounts == sorted(set(corpus.catalog_prices()) | {60, 999, 5000})
    assert {hit.sku.sku for hit in briefing.resolved.sku_hits} >= {"KK-1000", "GBL"}


def test_brief_keeps_the_order_under_discussion(pipeline: RetrievalPipeline) -> None:
    history = [TurnRecord(role="user", content="What is the price of 1 kg motichoor laddoo?")]
    briefing = pipeline.brief("Make it 3 kg, and add 10 samosas.", history)
    assert sorted((item.sku, item.qty) for item in briefing.resolved.items) == [
        ("ML-1000", 3),
        ("SM-1", 10),
    ]


def test_brief_adds_to_the_order_when_the_customer_says_so(
    pipeline: RetrievalPipeline,
) -> None:
    history = [TurnRecord(role="user", content="2 kg kaju katli and 500 g soan papdi")]
    briefing = pipeline.brief("Add 1 kg gulab jamun too.", history)
    assert [(item.sku, item.qty) for item in briefing.resolved.items] == [
        ("KK-1000", 2),
        ("SP-500", 1),
        ("GJ-1000", 1),
    ]


def test_brief_does_not_repeat_a_restated_line(pipeline: RetrievalPipeline) -> None:
    history = [TurnRecord(role="user", content="2 kg kaju katli")]
    briefing = pipeline.brief("Also make it 3 kg of kaju katli.", history)
    assert [(item.sku, item.qty) for item in briefing.resolved.items] == [("KK-1000", 3)]


def test_brief_replaces_the_order_when_nothing_says_otherwise(
    pipeline: RetrievalPipeline,
) -> None:
    history = [TurnRecord(role="user", content="2 kg kaju katli")]
    briefing = pipeline.brief("How much is 500 g sugar-free kaju katli?", history)
    assert [(item.sku, item.qty) for item in briefing.resolved.items] == [
        ("KKSF-500", 1)
    ]


def test_brief_follows_up_without_repeating_the_product(
    pipeline: RetrievalPipeline,
) -> None:
    history = [
        TurnRecord(
            role="user",
            content="I want 2 kg kaju katli and one large Diwali gift box, 5 km away",
        )
    ]
    briefing = pipeline.brief("What is the total?", history)
    assert [item.sku for item in briefing.resolved.items] == ["KK-1000", "GBL"]
    assert briefing.resolved.distance_km == 5


def test_brief_language_follows_the_customer(pipeline: RetrievalPipeline) -> None:
    assert pipeline.brief("Bhaiya 60 small gift box chahiye", []).language == "hinglish"
    assert pipeline.brief("मुझे 2 किलो मोतीचूर लड्डू चाहिए", []).language == "hindi"


def test_every_resolved_line_is_backed_by_a_catalog_row(
    pipeline: RetrievalPipeline,
) -> None:
    for question in (
        "chhota gift box aur 2 kg kaju katli",
        "2.5 kg motichoor laddoo",
        "a dozen samosas",
        "1 kg besan laddoo and 3 rasmalai",
    ):
        briefing = pipeline.brief(question, [])
        hit_ids = {hit.sku.sku for hit in briefing.resolved.sku_hits}
        assert {item.sku for item in briefing.resolved.items} <= hit_ids


def test_a_blank_turn_retrieves_nothing(pipeline: RetrievalPipeline) -> None:
    briefing = pipeline.brief("   ", [])
    assert briefing.resolved.sections == []
    assert briefing.resolved.sku_hits == []
    assert briefing.resolved.items == []


def test_brief_flags_an_unrelated_question(pipeline: RetrievalPipeline) -> None:
    assert pipeline.brief(
        "Can you write my college assignment on photosynthesis?", []
    ).out_of_scope_hint is True
    # An unstocked product is unknown, not out of scope.
    assert pipeline.brief("Do you make rabri?", []).out_of_scope_hint is False
