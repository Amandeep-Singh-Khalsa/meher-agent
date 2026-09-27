"""Two small indexes over data/: policy sections, and the price catalog.

Section retrieval is BM25 plus explicit intent anchors. The anchors matter more
than the raw term scores: a customer asking "koi discount milega?" writes none of
the words that appear in the Diwali discounts section, so an anchor guarantees
the right policy reaches the model while BM25 fills in the rest of the context.

SKU retrieval is a fuzzy matcher, not BM25: it combines alias-phrase matching,
token overlap against the synonym-expanded query, and character-trigram
containment, so that "sugar free kaju katli" lands on KKSF-500 and "moti laddoo"
lands on ML-1000 without a stemmer or an embedding model.

The alias index and the window matcher live at module level because the order
resolver matches the same phrases; it needs the token positions, not just a score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Container, Iterable

from ..config import RetrievalConfig
from ..data.corpus import Corpus
from ..types import RetrievedSKU, RetrievedSection, Section, SKU
from ..text.lexicon import STOPWORDS, expand, tokenize
from ..text.translit import fold_devanagari

__all__ = [
    "Retriever",
    "SKU_ALIASES",
    "SKU_QUALIFIERS",
    "best_alias_window",
    "sku_alias_index",
]

#: Added to an anchored section so it outranks a merely lexically similar one,
#: while still letting a strongly matching BM25 section finish above it. Anchored
#: sections are included whatever their BM25 score, so this only sets the order.
_ANCHOR_BONUS = 1.0

#: Fraction of an alias that must be present in a window of the message before
#: the window counts as a match. Below this, matching gets lucky on the pack or
#: unit word alone ("much for 1 kg" versus the alias "1 kg motichoor laddoo").
_MIN_ALIAS_HIT = 0.6

#: A SKU with no alias match is still listed when at least this much of the
#: message (or of the phrase that nearly matched) appears in its catalog line.
_MIN_CONTAINMENT = 0.5

#: Tokens that select one SKU over another within the same item family. A
#: qualifier that some SKU carries and the message mentions removes every SKU
#: that does not: "sugar free" cannot mean plain kaju katli.
SKU_QUALIFIERS: frozenset[str] = frozenset(
    {
        "sugar", "free", "sugarfree", "sugarless", "diet", "small", "large",
        "big", "bada", "bade", "badi", "bad", "bara", "bare", "chhota",
        "chota", "chhote", "chote", "chhoti", "choti", "mini", "premium",
        "jumbo", "dry", "fruit", "fruits",
    }
)

#: Every surface form that should retrieve the SKU. Devanagari is folded at build
#: time, so "काजू कटली" and "kaju katli" reach the same alias.
SKU_ALIASES: dict[str, tuple[str, ...]] = {
    "KK-1000": (
        "kaju katli", "kaju katly", "kaju katlee", "kaju kathli", "kaju katali",
        "kaju", "katli", "kaju katli 1 kg", "1 kg kaju katli", "whole kg kaju katli",
        "full kg kaju katli", "kaju katli big", "big kaju katli", "kaju katli diamond",
        "cashew katli", "kk1000", "kk 1000", "काजू कटली", "काजू", "कटली",
    ),
    "KK-500": (
        "kaju katli", "kaju katly", "kaju katlee", "kaju kathli", "kaju katali",
        "kaju", "katli", "kaju katli 500 g", "500 g kaju katli", "half kg kaju katli",
        "kaju katli half", "small kaju katli", "mini kaju katli", "kk500", "kk 500",
        "काजू कटली", "काजू", "कटली",
    ),
    "KKSF-500": (
        "sugar free kaju katli", "sugarfree kaju katli", "sugar-free kaju katli",
        "kaju katli sugar free", "sugar free katli", "sugar free kaju",
        "no sugar kaju katli", "diet kaju katli", "sugarless kaju katli",
        "sugar free", "sugarfree", "sugar free kaju katli 500 g", "kksf",
        "kksf500", "सुगर फ्री काजू कटली", "शुगर फ्री काजू कटली", "सुगर फ्री", "शुगर फ्री",
    ),
    "ML-1000": (
        "motichoor laddoo", "motichur laddoo", "motichoor ladoo", "moti laddoo",
        "moti ladoo", "moti laddu", "moti laddus", "motichoor", "moti", "moori",
        "mukhi", "pearl laddoo", "boondi laddoo", "moti laddoo 1 kg",
        "1 kg motichoor laddoo", "ml1000", "ml 1000", "मोतीचूर लड्डू", "मोतीचोर लड्डू",
        "मोतीचूर", "मोतीचोर", "मोती",
    ),
    "BL-1000": (
        "besan laddoo", "besan ladoo", "besan laddu", "besan laddus", "besan",
        "basan laddoo", "besan ke laddoo", "gram flour laddoo", "besan laddoo 1 kg",
        "1 kg besan laddoo", "bl1000", "bl 1000", "बेसन लड्डू", "बेसन",
    ),
    "SP-500": (
        "soan papdi", "soan papri", "son papdi", "son papadi", "soan papada",
        "soan papde", "papdi", "papri", "papadi", "soan papdi 500 g",
        "sp500", "sp 500", "सोन पापड़ी", "सोन पापडी",
    ),
    "GJ-1000": (
        "gulab jamun", "gulaab jamun", "golab jamun", "jamun", "gulab",
        "gulab jamoons", "gulab jamun 1 kg", "gj1000", "gj 1000",
        "गुलाब जामुन", "जामुन", "गुलाब",
    ),
    "RM-500": (
        "rasmalai", "ras malai", "rasa malai", "rasmlai", "rasmai", "malai",
        "rasmalai 500 g", "rm500", "rm 500", "रसमलाई", "रसमलाई", "रस मलाई",
    ),
    "NM-400": (
        "mixed namkeen", "mix namkeen", "mixture namkeen", "namkeen", "namkean",
        "mixed snack", "mixed namkeen 400 g", "nm400", "nm 400", "नमकीन",
        "मिक्स नमकीन",
    ),
    "AB-400": (
        "aloo bhujia", "alu bhujia", "aloo bhujiya", "aloo bhujiya", "bhujia",
        "bhujiya", "potato bhujia", "aloo bhujia 400 g", "ab400", "ab 400",
        "आलू भुजिया", "भुजिया", "आलू",
    ),
    "SM-1": (
        "samosa", "samosas", "samosay", "samoosa", "samoose", "singara", "samosa piece",
        "one samosa", "sm1", "sm 1", "समोसा", "समोसे",
    ),
    "DH-500": (
        "khaman dhokla", "dhokla", "dhokle", "khaman", "handvo",
        "khaman dhokla 500 g", "dh500", "dh 500", "खमन धोकला", "धोकला", "धोकले",
    ),
    "GBS": (
        "diwali gift box small", "small gift box", "small diwali gift box",
        "small diwali box", "small box", "small gift", "chhota gift box",
        "chota gift box", "chhoti gift box", "chhote gift box", "mini gift box",
        "gbs", "छोटा गिफ्ट बॉक्स", "छोटे गिफ्ट बॉक्स", "गिफ्ट बॉक्स छोटा",
        "gift box", "giftbox", "gifth box", "gift bx", "hamper", "hamper box",
        "gift hamper", "gift basket", "combo box", "boks", "boxs", "बक्सा",
        "गिफ्ट बॉक्स", "उपहार", "टोकरी", "assorted sweets box", "500 g assorted sweets",
    ),
    "GBL": (
        "diwali gift box large", "large gift box", "big gift box", "bada gift box",
        "badi gift box", "bade gift box", "large diwali box", "big diwali box",
        "bada diwali box", "premium gift box", "dry fruit gift box", "gbl",
        "big gift hamper", "बड़ा गिफ्ट बॉक्स", "बड़े गिफ्ट बॉक्स", "गिफ्ट बॉक्स बड़ा",
        "1 kg assorted sweets",
    ),
}

#: (regex, source ids) applied to the folded, casefolded message. These are the
#: intents where a synonym-blind lexical score reliably picks the wrong section.
_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"\bdiscount|\bdiscounts|\bcoupon|\bcheap|\bcheaper|\boff\b|\boffer|\boffers|\bconcessi|"
     r"\bsale\b|\bdeal\b|\bbachat|\bsasta|\bchhoot|\bchhut|\brebate|\bvoucher|\bsubsid",
     ("policies.md#diwali-2026-gift-boxes-and-discounts",)),
    (r"\bdeliver|\bdelivery|\bshipping|\bcourier|\bdistance|\bkm\b|\bkilometer|\bkilometre|"
     r"\bdilivari|\bdilavari|\bbhej|\bbhejo|\bpahuch|\bdoor\b|\bradius|\bsame day",
     ("policies.md#delivery",)),
    (r"\breturn|\brefund|\brefunds|\bdamaged|\bbroken|\bbroken|\bspoiled|\bspoilt|"
     r"\bcrushed|\btorn\b|\breplace|\breplacement|\bvapas|\bwapas|\bexchange",
     ("policies.md#returns-and-damaged-deliveries",)),
    (r"\bcomplaint|\bcomplain|\bunhappy|\bdisappointed|\bdisappoint|\bshocked|"
     r"\bnot happy|\bbad service|\bpoor service|\brude|\bfurious|\bangry|\bshikaya",
     ("policies.md#complaints", "policies.md#returns-and-damaged-deliveries")),
    (r"\bbulk|\bwholesale|\bwhole sale|\badvance|\bnotice|\bprepay|\bpre payment|"
     r"\bcorporate|\boffice order|\bparty order|\bfunction|\bevent\b|\bthora",
     ("policies.md#bulk-orders",)),
    (r"\bpay\b|\bpayment|\bupi|\bneft|\bcard\b|\bcod\b|\bcash\b|\bcash on delivery\b|"
     r"\bmoney\b|\bgst\b|\bgstin|\binvoice|\bbill\b|\breceipt|\bvat\b|\btax\b|"
     r"\bdebit|\bcredit\b|\bkaidun|\bbank\b|\bnet banking",
     ("policies.md#payment",)),
    (r"\bgst\b|\bgstin|\binvoice|\bbill\b|\breceipt|\btax\b|\bvat\b|\bprabhashan",
     ("policies.md#prices-and-gst",)),
    (r"\bprice\b|\bprices\b|\bpricing\b|\brate\b|\brates\b|\bcost\b|\bhow much\b|"
     r"\bkitna\b|\bkitne\b|\bkimat\b|\bdaam\b|\bdam\b|\bpaisa\b|\bpaise\b|"
     r"\bmrp\b|\brate list|\bprice list",
     ("policies.md#prices-and-gst",)),
    (r"\bwedding|\bcustom|\bbespoke|\bcake|\bbarat|\bshaadi|\bsagan|\bmehndi|\bhaldi|"
     r"\breception|\bcatering|\bthali\b|\bbanket|\bbanquet|\bparty order",
     ("policies.md#wedding-and-custom-orders",)),
    (r"\ballerg|\ballergy|\ballergic|\bnut\b|\bnuts\b|\bcashew|\bpeanut|\bpeanuts|"
     r"\balmond|\bmilk\b|\bdairy|\bwheat\b|\bgluten|\bsoy\b|\btrace|\btraces|"
     r"\bveg\b|\bvegetarian|\bnonveg\b|\bnon veg\b|\begg\b|\beggs\b|\bpuran veg|"
     r"\bcashew allergy|\bhas cashew|\bsafe for",
     ("policies.md#ingredients-and-allergens",)),
    (r"\bstorage\b|\bstore\b|\bfridge|\brefrigerat|\bshelf life|\bshelflife|"
     r"\bexpiry|\bexpir|\bexpire|\bhow long|\bhow many days|\beat within|"
     r"\bbest before|\buse by|\broom temperature|\bfresh for",
     ("policies.md#storage",)),
    (r"\bhours\b|\bhour\b|\btiming|\btime\b|\bwaqt\b|\bsamay\b|\bopen\b|\bopens\b|"
     r"\bopening|\bclose\b|\bcloses\b|\bclosing|\bkhula|\bbanda|\bband hai|\bholi|"
     r"\bholiday|\bclosed\b|\bshut\b|\baaj\b|\btoday\b|\bkitne baje|\bsubah|\bshaam|"
     r"\braat\b",
     ("business.md#opening-hours",)),
    (r"\baddress|\blocation|\bwhere\b|\bdirection|\bnear\b|\bnearby|\blandmark|\bmap\b|"
     r"\blocated|\bsituated|\bkahan|\bkahan hai|\bkahan par|\bpata\b|\bpatа|\brajouri|"
     r"\bcentral market|\bnew delhi|\bpin ?code|\bpostcode",
     ("business.md#address",)),
    (r"\babout\b|\byour shop\b|\bwho are you|\bhistory\b|\bsince\b|\bfamily shop",
     ("business.md#about", "business.md#meher-sweets-namkeen")),
    (r"\bcontact|\bphone\b|\bcall\b|\bmobile|\bnumber\b|\bwhatsapp|\bwhats app|"
     r"\bemail\b|\bmail\b|\bowner|\bmanager|\bstaff\b|\breach you|\bspeak to",
     ("business.md#contact", "business.md#how-to-order")),
    (r"\bhow (?:do i|to) order|\border process|\border kaise|\bordering process|"
     r"\bkaise order",
     ("business.md#how-to-order",)),
    (r"\bpickup|\bpick up|\btakeaway|\btake away|\bcollect|\bcollection|\bself pickup",
     ("business.md#how-to-order", "policies.md#delivery")),
    (r"\blanguage|\blanguages|\bhindi\b|\bhinglish|\benglish\b|\bbhasha",
     ("business.md#languages",)),
    (r"\bsame day|\b4:00|\b4 pm|\bsame-day|\bajaj|\baj",
     ("policies.md#delivery",)),
)

#: Last-resort grounding when nothing clears ``min_section_score`` (an unknown
#: product name, say). Who we are and how to order is the safest context to hand
#: the model when the question itself is unanswerable from data/.
_FALLBACK_SECTION_IDS: tuple[str, ...] = (
    "business.md#how-to-order",
    "business.md#about",
)

_COMPILED_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(pattern), source_ids) for pattern, source_ids in _INTENT_RULES
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _trigrams(text: str) -> set[str]:
    squashed = f"  {text.strip()}  "
    return {squashed[i:i + 3] for i in range(len(squashed) - 2)}


def _levenshtein_within(a: str, b: str, limit: int = 1) -> bool:
    """True when ``a`` and ``b`` are within ``limit`` edits. Short strings must
    be equal, otherwise "g" and "kg" would look like a typo of each other."""
    if abs(len(a) - len(b)) > limit:
        return False
    if len(a) < 4 or len(b) < 4:
        return a == b
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        best = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
            best = min(best, current[-1])
        if best > limit:
            return False
        previous = current
    return previous[-1] <= limit


def _singular(token: str) -> str:
    """'boxes' -> 'box', 'laddoos' -> 'laddoo'; unchanged otherwise."""
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens_match(query_token: str, alias_token: str) -> float:
    if query_token == alias_token:
        return 1.0
    # "3 boxes" against the alias "gift box": the same word, so a full match.
    if _singular(query_token) == _singular(alias_token):
        return 1.0
    if _levenshtein_within(query_token, alias_token):
        return 0.8
    return 0.0


def _window_pairs(alias: tuple[str, ...], window: list[str]) -> list[tuple[float, str]]:
    """Greedy one-to-one match of alias tokens to window tokens.

    One-to-one matters: without it the window "laddoo laddoo" would score twice
    against a single "laddoo" in the alias and outrank the real match.
    """
    used: set[int] = set()
    pairs: list[tuple[float, str]] = []
    for token in window:
        best_gain = 0.0
        best_index = -1
        for index, alias_token in enumerate(alias):
            if index in used:
                continue
            gain = _tokens_match(token, alias_token)
            if gain > best_gain:
                best_gain, best_index = gain, index
        if best_gain > 0.0 and best_index >= 0:
            used.add(best_index)
            pairs.append((best_gain, alias[best_index]))
    return pairs


def sku_alias_index(corpus: Corpus) -> dict[str, list[tuple[str, ...]]]:
    """Folded alias tuples per SKU, longest first. Shared with the resolver.

    The item name itself is an alias, and Devanagari forms are folded at build
    time, so "काजू कटली" and "kaju katli" reach the same entry.
    """
    index: dict[str, list[tuple[str, ...]]] = {}
    for sku in corpus.skus:
        aliases = {
            tokens
            for form in (sku.item, *SKU_ALIASES.get(sku.sku, ()))
            for tokens in [tuple(tokenize(form))]
            if tokens
        }
        index[sku.sku] = sorted(aliases, key=lambda alias: (-len(alias), alias))
    return index


def best_alias_window(
    aliases: Iterable[tuple[str, ...]],
    query_tokens: list[str],
    *,
    max_span: int = 4,
    required: Container[str] | None = None,
) -> tuple[float, int, int, int]:
    """Best contiguous window of ``query_tokens`` matching any alias.

    Returns ``(quality, span, start, end)`` with ``end`` exclusive, and quality 0
    when nothing matched. Aliases match in any word order but only inside one
    contiguous window of raw tokens, so "large diwali box" reaches the alias
    "diwali gift box large" while a phrase stitched across "and" does not, and
    stopwords inside the window lower the quality instead of hiding the match.

    ``required`` names tokens a window must actually hit, which is how the
    resolver tells a product mention ("2 kg rasmalai") from a pack size on its
    own ("500 g"), where the alias "rasmalai 500 g" would otherwise match on the
    size words alone.
    """
    best = (0.0, 0, 0, 0)
    if not query_tokens:
        return best
    for alias in aliases:
        span = len(alias)
        if span > max_span or span > len(query_tokens) + 1:
            continue
        for width in (span, span + 1):
            for start in range(0, len(query_tokens) - width + 1):
                window = query_tokens[start:start + width]
                pairs = _window_pairs(alias, window)
                hit = sum(gain for gain, _ in pairs)
                if hit < _MIN_ALIAS_HIT * span:
                    continue
                if required is not None and not any(
                    gain > 0.0 and token in required for gain, token in pairs
                ):
                    continue
                quality = (hit / span) * (span / width)
                if quality > best[0] + 1e-9 or (
                    abs(quality - best[0]) <= 1e-9 and span > best[1]
                ):
                    best = (quality, span, start, start + width)
    return best


class _BM25:
    """Classic Robertson/Sparck-Jones BM25 over a handful of documents.

    Scores are divided by the total idf mass of the query, which turns them into
    "fraction of the query's evidence that this document matches" (roughly
    0..1). Without that normalisation a synonym-expanded query of a hundred terms
    would score in the hundreds, and ``min_section_score`` would be meaningless.
    """

    def __init__(self, documents: list[list[str]], k1: float, b: float) -> None:
        self.k1 = k1
        self.b = b
        self.counts = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.frequency: dict[str, int] = {}
        for doc in documents:
            for term in set(doc):
                self.frequency[term] = self.frequency.get(term, 0) + 1
        self.n = len(documents)

    def _idf(self, term: str) -> float:
        df = self.frequency.get(term)
        if not df:
            return 0.0
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str]) -> list[float]:
        if not self.n or not self.avg_length:
            return [0.0] * self.n
        terms = sorted(set(query_terms))
        weights = {term: self._idf(term) for term in terms}
        total = sum(weights.values())
        if total <= 0.0:
            return [0.0] * self.n
        scores = [0.0] * self.n
        for term in terms:
            idf = weights[term]
            if idf <= 0.0:
                continue
            for i, counts in enumerate(self.counts):
                tf = counts.get(term)
                if not tf:
                    continue
                norm = 1.0 - self.b + self.b * (self.lengths[i] / self.avg_length)
                scores[i] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return [value / total for value in scores]


def _content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS]


def _expand_terms(tokens: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in STOPWORDS:
            continue
        for term in sorted(expand(token)):
            if term not in seen:
                seen.add(term)
                out.append(term)
    return out


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------


class Retriever:
    """Policy-section and price-catalog retrieval over a loaded corpus."""

    def __init__(self, corpus: Corpus, config: RetrievalConfig) -> None:
        self.corpus = corpus
        self.config = config
        self._sections: list[Section] = list(corpus.sections)
        self._section_index = _BM25(
            [_expand_terms(_content_tokens(s.text)) for s in self._sections],
            config.bm25_k1,
            config.bm25_b,
        )
        self._skus: list[SKU] = list(corpus.skus)
        self._sku_aliases: dict[str, list[tuple[str, ...]]] = sku_alias_index(corpus)
        self._sku_blob: dict[str, set[str]] = {}
        self._sku_trigrams: dict[str, set[str]] = {}
        for sku in self._skus:
            blob_text = self._sku_blob_text(sku)
            self._sku_blob[sku.sku] = set(tokenize(blob_text))
            self._sku_trigrams[sku.sku] = _trigrams(blob_text)

    # -- construction ------------------------------------------------------
    @staticmethod
    def _sku_blob_text(sku: SKU) -> str:
        bits = [sku.item, sku.pack, sku.type, ", ".join(sku.allergens)]
        bits.extend(SKU_ALIASES.get(sku.sku, ()))
        bits.append(sku.sku)
        bits.append(sku.sku.replace("-", ""))
        bits.append(sku.sku.replace("-", " "))
        return " ".join(bits)

    # -- sections ----------------------------------------------------------
    def _anchors(self, query: str) -> tuple[str, ...]:
        haystack = fold_devanagari(query or "").casefold()
        hits: list[str] = []
        for pattern, source_ids in _COMPILED_RULES:
            if pattern.search(haystack) is None:
                continue
            for source_id in source_ids:
                if source_id not in hits:
                    hits.append(source_id)
        return tuple(hits)

    def sections(self, query: str, top_k: int | None = None) -> list[RetrievedSection]:
        """Best-matching policy and business sections, anchors always included."""
        limit = self.config.section_top_k if top_k is None else max(0, int(top_k))
        if not self._sections or limit == 0:
            return []
        query_terms = _expand_terms(_content_tokens(query))
        if not query_terms:
            # Nothing was said, so nothing can be retrieved. Handing back a
            # zero-scored fallback here would only give the reply layer context
            # it has no reason to quote.
            return []
        scores = self._section_index.score(query_terms)
        anchored = set(self._anchors(query))
        known = {s.source_id for s in self._sections}

        ranked: list[tuple[float, str]] = []
        for section, score in zip(self._sections, scores):
            final = score + (_ANCHOR_BONUS if section.source_id in anchored else 0.0)
            if section.source_id in anchored or final >= self.config.min_section_score:
                ranked.append((round(final, 6), section.source_id))

        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        keep_anchors = [pair for pair in ranked if pair[1] in anchored]
        others = [pair for pair in ranked if pair[1] not in anchored]
        others = others[: max(0, limit - len(keep_anchors))]

        out: list[RetrievedSection] = []
        selected = keep_anchors + others
        if not selected:
            best = max(scores) if scores else 0.0
            if best > 0.0:
                selected = [
                    (round(float(value), 6), section.source_id)
                    for section, value in zip(self._sections, scores)
                    if value == best
                ][:1]
            elif not self.skus(query):
                # Nothing in data/ matches at all. Who we are and how to order is
                # the only honest context to hand over for an unanswerable
                # question; a product name the catalog does not carry is already
                # grounded by the empty sku list, so it gets nothing.
                selected = [(0.0, source_id) for source_id in _FALLBACK_SECTION_IDS]
        for score, source_id in selected:
            if source_id not in known:
                continue
            section = self.corpus.section(source_id)
            if section is None:
                continue
            out.append(RetrievedSection(section=section, score=float(score)))
        return out

    # -- skus --------------------------------------------------------------
    def _best_alias_match(
        self, sku: SKU, query_tokens: list[str]
    ) -> tuple[float, int, str]:
        """Best (quality, matched alias length, matched text) for one SKU."""
        quality, span, start, end = best_alias_window(
            self._sku_aliases.get(sku.sku, ()), query_tokens
        )
        return quality, span, " ".join(query_tokens[start:end]) if quality else ""

    def skus(self, query: str, top_k: int | None = None) -> list[RetrievedSKU]:
        """Price-catalog rows that could satisfy the message, best first."""
        limit = self.config.sku_top_k if top_k is None else max(0, int(top_k))
        if not self._skus or limit == 0:
            return []
        raw_tokens = tokenize(query)
        query_tokens = _content_tokens(query)
        query_text = " ".join(query_tokens)
        query_trigrams = _trigrams(query_text)
        query_set = set(query_tokens)
        expanded = set(_expand_terms(query_tokens))
        qualifiers = query_set & SKU_QUALIFIERS

        scored: list[tuple[float, str, str]] = []
        for sku in self._skus:
            blob = self._sku_blob[sku.sku]
            alias_quality, alias_len, matched = self._best_alias_match(sku, raw_tokens)
            coverage = len(query_set & blob) / len(query_set) if query_set else 0.0
            blob_trigrams = self._sku_trigrams[sku.sku]
            probe = _trigrams(matched) if matched else query_trigrams
            containment = (
                len(probe & blob_trigrams) / len(probe) if probe else 0.0
            )
            if alias_quality <= 0.0 and containment < _MIN_CONTAINMENT:
                continue
            # The synonym-expanded overlap only counts when no alias matched:
            # otherwise it is decided by how long a pack's alias list happens to
            # be, which would split identical products such as KK-1000 and
            # KK-500 on nothing but spelling and stop the tie breaking cleanly.
            expanded_hits = 0.0
            if alias_quality <= 0.0 and expanded:
                expanded_hits = len(expanded & blob) / len(expanded)
            score = (
                4.0 * alias_quality
                + 1.5 * coverage
                + 0.5 * expanded_hits
                + 2.0 * containment
            )
            if score > 0.5:
                scored.append((score, sku.sku, matched))

        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], item[1]))
        top = scored[0][0]
        plausible = {sku for score, sku, _ in scored if score >= 0.6 * top}

        if qualifiers:
            keepers: set[str] = set(plausible)
            for qualifier in sorted(qualifiers):
                holders = {sku for sku in plausible if qualifier in self._sku_blob[sku]}
                if holders:
                    keepers &= holders
            if keepers:
                scored = [item for item in scored if item[1] in keepers]

        out: list[RetrievedSKU] = []
        for score, sku_id, matched in scored[:limit]:
            sku = self.corpus.sku(sku_id)
            if sku is None:
                continue
            out.append(
                RetrievedSKU(
                    sku=sku,
                    score=round(float(score), 6),
                    reason=self._reason(sku, matched),
                )
            )
        return out

    @staticmethod
    def _reason(sku: SKU, matched: str) -> str:
        label = f"via '{matched}'" if matched else "via fuzzy token overlap"
        return (
            f"sku:{sku.sku} {sku.item} ({sku.pack}) {label} "
            f"— type: {sku.type}, contains: {sku.allergens or 'none'}"
        )
