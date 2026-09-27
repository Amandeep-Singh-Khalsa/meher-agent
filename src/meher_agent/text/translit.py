"""Devanagari -> Latin folding and a three-way language tag.

Folding strategy: a *literal* transliteration. Every consonant is written as
"core + inherent vowel", a vowel matra replaces that inherent vowel, a virama
suppresses it, and one conservative schwa rule is applied: a word-final inherent
"a" is dropped (Hindi final-schwa deletion, e.g. daam -> dam).

No medial-schwa deletion is attempted, because in Hindi it is lexical rather than
script-determined: the written form cannot distinguish the "katli" in kaTli from
the "kaatali" in kaTaaLi. Guessing it corrupts ordinary shop text. The residual
variation is absorbed downstream instead, by the synonym lexicon in
text.lexicon (which carries the folded Devanagari spelling of every product) and
by the trigram matcher in retrieval.retriever.

Digits are folded to ASCII because every downstream regex, and the reply guard,
only understands 0-9.
"""

from __future__ import annotations

import re

__all__ = ["fold_devanagari", "devanagari_ratio", "detect_language"]


# --------------------------------------------------------------------------
# Devanagari tables
# --------------------------------------------------------------------------

#: Independent vowels.
_VOWELS: dict[str, str] = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "ri", "ॠ": "ri", "ऌ": "li", "ॡ": "li",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऍ": "a", "ऎ": "e", "ऑ": "o", "ऒ": "o",
}

#: Vowel signs (matras). Short forms only, so that the matra already carries the
#: vowel: kaju must fold to "kaju", never to "kajju" or "kaaju".
_MATRAS: dict[str, str] = {
    "ा": "a", "ि": "i", "ी": "i",
    "ु": "u", "ू": "u",
    "ृ": "ri", "ॄ": "ri",
    "ॢ": "li", "ॣ": "li",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॊ": "o", "ॅ": "e", "ॆ": "e",
}

#: Consonants, written as the consonant *without* its inherent vowel.
_CONSONANTS: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "ऩ": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ऱ": "r", "ळ": "l", "ऴ": "l",
    "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # Nukta consonants, precomposed.
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r",
    "ढ़": "rh", "फ़": "f", "य़": "y",
    # Rare Devanagari letters that appear in loanwords.
    "ॹ": "k", "ॺ": "g", "ॻ": "j", "ॼ": "d", "ॽ": "t", "ॾ": "n",
    "ऺ": "th", "ऻ": "d", "ऽ": "",
}

_VIRAMA = "्"
_ANUSVARA = "ं"
_CHANDRABINDU = "ँ"
_VISARGA = "ः"
_NUKTA = "़"
_ABJA = "ऺ"

#: V + a-matra is "wa", not "va": Hindi writes "diwali" as diwaali, so a purely
#: literal fold yields "divali" and misses the Latin product name. Keyed on the
#: *folded* consonant, because that is what the matra branch has in hand.
_MATRA_LIGATURES: dict[tuple[str, str], str] = {
    ("v", "ा"): "wa",
    ("v", "ो"): "wo",
    ("v", "ौ"): "wau",
    ("v", "ॉ"): "wo",
}

#: A consonant immediately before a virama loses its "a" and may take a
#: non-default core: the core-then-virama pair is what the fix is keyed on.
_CONJUNCT_FIXES: dict[tuple[str, str], str] = {
    ("फ", _VIRAMA): "f",
    ("ष", _VIRAMA): "sh",
    ("ह", _VIRAMA): "h",
    ("ज", _VIRAMA): "j",
}

_JA_NYA = "ज" + _VIRAMA + "ञ"
_GYA = "ग" + _VIRAMA + "य"

_DIGIT_FOLD = str.maketrans({
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
})

_PUNCT_FOLD = str.maketrans({
    "।": ". ", "॥": ". ", "“": '"', "”": '"', "‘": "'",
    "’": "'", "–": "-", "—": "-", " ": " ",
})

#: Whole conjuncts whose core-by-core fold is badly wrong: Hindi writes
#: gyaan, not jnyan, so jnya is folded to gya before the word loop runs.
_LIGATURE_PRE_PASS: tuple[tuple[str, str], ...] = ((_JA_NYA, _GYA),)

#: Characters that are part of a Devanagari syllable.
_SYLLABLE_CHARS = frozenset(_CONSONANTS) | frozenset(_VOWELS) | frozenset(_MATRAS) | {_VIRAMA}
#: Characters that attach to the syllable in progress rather than ending it.
_TRAILING_CHARS = frozenset({_ANUSVARA, _CHANDRABINDU, _VISARGA, _NUKTA, _ABJA})


# --------------------------------------------------------------------------
# Folding
# --------------------------------------------------------------------------


class _Unit:
    """One folded syllable. ``inherent`` records whether a trailing "a" is the
    consonant's own schwa rather than a written matra, because only a schwa may
    be replaced by a matra, suppressed by a virama, or dropped at word end."""

    __slots__ = ("text", "inherent", "letter")

    def __init__(self, text: str, inherent: bool, letter: str = "") -> None:
        self.text = text
        self.inherent = inherent
        self.letter = letter

    def core(self) -> str:
        if self.inherent and self.text.endswith("a"):
            return self.text[:-1]
        return self.text


def _fold_word(word: str) -> str:
    units: list[_Unit] = []
    pending: _Unit | None = None
    i = 0
    n = len(word)

    while i < n:
        ch = word[i]

        if ch == _VIRAMA:
            if pending is not None:
                fix = _CONJUNCT_FIXES.get((pending.letter, _VIRAMA))
                pending.text = fix if fix is not None else pending.core()
            i += 1
            continue

        if ch in _MATRAS:
            if pending is not None:
                core = pending.core()
                ligature = _MATRA_LIGATURES.get((core, ch))
                pending.text = ligature if ligature else core + _MATRAS[ch]
                pending.inherent = False
            i += 1
            continue

        if pending is not None:
            units.append(pending)
            pending = None

        if ch in _CONSONANTS:
            core = _CONSONANTS[ch]
            j = i + 1
            for mark in (_NUKTA, _ABJA):
                if j < n and word[j] == mark and ch + mark in _CONSONANTS:
                    core = _CONSONANTS[ch + mark]
                    j += 1
            pending = _Unit(core + "a", True, ch)
            i = j
            continue

        if ch in _VOWELS:
            pending = _Unit(_VOWELS[ch], False)
            i += 1
            continue

        if ch in (_ANUSVARA, _CHANDRABINDU):
            if pending is not None:
                pending.text += "n"
            i += 1
            continue

        if ch == _VISARGA:
            if pending is not None:
                pending.text += "h"
            i += 1
            continue

        i += 1

    if pending is not None:
        units.append(pending)
    if units and units[-1].inherent and units[-1].text.endswith("a"):
        units[-1].text = units[-1].text[:-1]
    return "".join(u.text for u in units)


def fold_devanagari(text: str) -> str:
    """Transliterate Devanagari to Latin, leaving non-Devanagari text untouched.

    >>> fold_devanagari("काजू कटली का दाम")
    'kaju katali ka dam'
    """
    if not text:
        return ""
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(_fold_word("".join(buf)))
            buf.clear()

    folded_input = text.translate(_DIGIT_FOLD).translate(_PUNCT_FOLD)
    for source, target in _LIGATURE_PRE_PASS:
        folded_input = folded_input.replace(source, target)

    for ch in folded_input:
        if ch in _SYLLABLE_CHARS or ch in _TRAILING_CHARS:
            buf.append(ch)
        else:
            flush()
            out.append(ch)
    flush()
    return "".join(out)


def devanagari_ratio(text: str) -> float:
    """Share of *letters* that are Devanagari code points, in 0.0-1.0.

    Digits, punctuation and whitespace are excluded from the denominator, so
    "12 km" is not treated as mostly Devanagari and "क्या 12" is fully Devanagari.
    """
    if not text:
        return 0.0
    total = 0
    devanagari = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        if "ऀ" <= ch <= "ॿ":
            devanagari += 1
    if total == 0:
        return 0.0
    return devanagari / total


# --------------------------------------------------------------------------
# Language tag
# --------------------------------------------------------------------------

#: Romanised Hindi function words, particles and kinship terms. None of these is
#: an English word, so one hit is decisive evidence of Hinglish.
_HINGLISH_MARKERS: tuple[str, ...] = (
    "kitna padega", "kitna padegi", "kitna hoga", "kitna lagega", "kitna lagta",
    "kitne paise", "kitna milega", "kya hoga", "kya milega", "kya milegi",
    "kitna", "kitne", "kitni", "kya", "kyu", "kyun", "kahan", "kaise",
    "kaisa", "kaisi", "hai na", "ka kya", "ka dam", "ka rate", "ki kimat",
    "ke liye", "hai", "hain", "ho", "hoon", "hun", "chahiye", "chaiye",
    "batao", "bata", "bataiye", "dena", "deno", "dijiye", "lelo", "lena",
    "mangwao", "mangwayo", "mangwana", "bhejo", "bhej", "bhejiye", "dekhlo",
    "dekhna", "karo", "karna", "karke", "kardo", "dene", "lagao", "rakhna",
    "banwao", "banaye", "banana", "chuna", "chuno", "khana", "khaye", "peena",
    "piyo", "aao", "aana", "de do", "dede", "kar do", "bana do",
    "aap", "aapka", "aapki", "aapke", "aapko", "tum", "tumhe", "tumhari",
    "mujhe", "mujhko", "mera", "meri", "hamara", "hamari",
    "bhaiya", "bhai", "yaar", "didi", "aunty", "uncle",
    "kripya", "krupya", "namaste", "jiji", "jii",
    "thoda", "thodi", "zyada", "bahut", "bohot", "sirf", "accha", "acha",
    "ab", "abhi", "jaldi", "foran", "turant", "waqt", "kal", "parso", "aaj",
    "raat", "shaam", "shaam tak", "subah", "dopahar", "ghar", "yahan", "wahan",
    "idhar", "udhar", "pahunch", "pahunchao", "chalo", "chalega", "chalegi",
    "mein", "paisa", "paise", "rupaiya", "rupaye", "rupaiye", "rupay",
    "aur", "phir", "fir", "baad", "wala", "wale", "wali", "waala",
    "sab", "sabko", "koi", "kuch", "nahi", "nahin", "haan", "han",
    "shukriya", "dhanyavaad", "jaldi se", "abhi tak", "bataiye kya",
    "kitna rupees", "kitne rupees", "khareedna", "khareed", "mangwana",
)

#: Romanised shop vocabulary that is also ordinary English, so it may never on
#: its own mark a message as Hinglish: "I want 10 samosas" is English.
_SOFT_MARKERS: tuple[str, ...] = (
    "discount", "delivery", "deliver", "diwali", "deepawali", "samosa",
    "laddoo", "ladoo", "namkeen", "kaju", "katli", "halwai", "courier",
    "order", "booking", "bhandara", "sher", "thali", "pakora", "chai",
    "paratha", "biryani", "raita", "lassi", "papad", "papri", "dulhan",
    "shaadi", "barat", "mehndi", "halwa", "kheer", "barfi",
)

_WORD_CACHE: dict[str, re.Pattern[str]] = {}


def _word_present(haystack: str, marker: str) -> bool:
    pattern = _WORD_CACHE.get(marker)
    if pattern is None:
        pattern = re.compile(
            r"(?<![a-z0-9])" + re.escape(marker).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        )
        _WORD_CACHE[marker] = pattern
    return pattern.search(haystack) is not None


def _hinglish_hits(text: str) -> int:
    haystack = fold_devanagari(text).casefold()
    if not haystack:
        return 0
    hard = sum(1 for marker in _HINGLISH_MARKERS if _word_present(haystack, marker))
    if hard:
        return hard
    soft = any(_word_present(haystack, marker) for marker in _SOFT_MARKERS)
    return 1 if soft and re.search(r"(?<![a-z0-9])(?:se|ko|ka|ki|ke)(?![a-z0-9])", haystack) else 0


def detect_language(text: str) -> str:
    """Classify ``text`` as exactly one of ``hindi``, ``hinglish``, ``english``.

    ``hindi`` requires real Devanagari script; ``hinglish`` requires a Romanised
    Hindi function word. Product names never count on their own, so "I want 10
    samosas" is still English.
    """
    if not text or not text.strip():
        return "english"
    if devanagari_ratio(text) >= 0.15:
        return "hindi"
    return "hinglish" if _hinglish_hits(text) > 0 else "english"
