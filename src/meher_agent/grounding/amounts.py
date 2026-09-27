"""Rupee-amount extraction and matching.

This module is the single implementation of two things that must never diverge:

1. The global check **G2** ("every rupee amount in the reply is allowed").
2. The reply guard that enforces the same rule at inference time.

Both the agent and the evaluation harness import from here, so a number that the
guard lets through is by construction a number the harness will accept.

G2 definition (from the task spec): a rupee amount is a number appearing after
"Rs", "Rs.", "Rs" or the rupee sign, or before the word "rupees". Matching for
must_include / must_not_include is case-insensitive and thousands separators
(including Indian grouping such as 1,00,000) are removed before comparing.
"""

from __future__ import annotations

import re

# A currency marker, immediately before the number.
# "Rs" uses a lookbehind rather than \b so that "Rs. 60" and "Rs60" both match:
# a trailing \b would fail against the period in "Rs.".
_PREFIX = r"(?:₹|(?<![A-Za-z])Rs\.?|\bINR\b|रु\.?|रूपये|रुपये)"
# A currency marker, immediately after the number.
_SUFFIX = r"(?:rupees|rupaye|रुपये|रूपये)"

# A number, optionally carrying thousands separators. The grouped form REQUIRES at
# least one comma group: without the (?!\d) boundary "\d{1,3}" would match a
# prefix of a longer run and silently truncate it ("Rs 1200" -> 120).
_NUM = r"(?:\d{1,3}(?:,\d{2,3})+(?!\d)|\d+)"

# Longest markers first so "Rs." wins over "Rs" and "रूपये" over "रु".
_RE_AMOUNT = re.compile(
    rf"(?P<pre>{_PREFIX})\s*(?P<n1>{_NUM})"
    rf"|(?P<n2>{_NUM})\s*(?P<post>{_SUFFIX})",
    re.IGNORECASE | re.UNICODE,
)

# A number that carries a thousands separator. Applied to text before matching.
_RE_GROUPED_NUMBER = re.compile(rf"(?<![\d.])(?P<n>{_NUM})(?![\d])")

# Matches a bare digit run, used to force ASCII digits in output.
_RE_ANY_DIGITS = re.compile(r"\d+")


def strip_thousands(raw: str) -> str:
    """'3,850' -> '3850'; '1,00,000' -> '100000'."""
    return raw.replace(",", "")


def to_int(raw: str) -> int:
    """Parse a possibly comma-grouped integer string. Raises ValueError on junk."""
    cleaned = strip_thousands(raw.strip())
    if not cleaned.isdigit():
        raise ValueError(f"not an integer: {raw!r}")
    return int(cleaned)


def extract_rupee_amounts(text: str) -> list[int]:
    """Every rupee amount in `text`, in order of appearance, de-duplicated.

    Only amounts carrying an explicit currency marker are returned, so '5% GST',
    '2 hours', '3 November', '12 km' and '10 samosas' are correctly ignored.
    """
    if not text:
        return []
    found: list[int] = []
    for m in _RE_AMOUNT.finditer(text):
        raw = m.group("n1") or m.group("n2")
        if raw is None:
            continue
        try:
            value = to_int(raw)
        except ValueError:
            continue
        found.append(value)
    return list(dict.fromkeys(found))


def has_rupee_amount_outside(text: str, allowed: set[int]) -> list[int]:
    """The subset of rupee amounts in `text` that is not in `allowed`."""
    return [v for v in extract_rupee_amounts(text) if v not in allowed]


def normalise_for_match(text: str) -> str:
    """Normalise a reply for must_include / must_not_include comparison.

    Case-insensitive, and all thousands separators inside numbers are removed so
    that a reply saying "3,850" satisfies must_include "3850".
    """
    if not text:
        return ""

    def _strip_group(m: re.Match[str]) -> str:
        return strip_thousands(m.group("n"))

    return _RE_GROUPED_NUMBER.sub(_strip_group, text).casefold()


def contains_any(text: str, needles: list[str]) -> bool:
    """True when any needle appears in `text` under normalise_for_match rules."""
    norm = normalise_for_match(text)
    return any(normalise_for_match(n) in norm for n in needles)


def contains_all(text: str, needles: list[str]) -> bool:
    norm = normalise_for_match(text)
    return all(normalise_for_match(n) in norm for n in needles)


def contains_none(text: str, needles: list[str]) -> bool:
    norm = normalise_for_match(text)
    return not any(normalise_for_match(n) in norm for n in needles)


# --------------------------------------------------------------------------
# Digit normalisation
# --------------------------------------------------------------------------

_DEVANAGARI_DIGITS = str.maketrans(
    {
        "\u0966": "0", "\u0967": "1", "\u0968": "2", "\u0969": "3",
        "\u096a": "4", "\u096b": "5", "\u096c": "6", "\u096d": "7",
        "\u096e": "8", "\u096f": "9",
        "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3",
        "\u0664": "4", "\u0665": "5", "\u0666": "6", "\u0667": "7",
        "\u0668": "8", "\u0669": "9",
    }
)


def ascii_digits(text: str) -> str:
    """Convert Devanagari and Arabic-Indic digits to ASCII 0-9.

    The reply guard applies this to every outgoing reply: the seed case
    `hindi-01` requires the literal ASCII digit "8" in a Hindi answer, which a
    model would otherwise render as the Devanagari "८".
    """
    return text.translate(_DEVANAGARI_DIGITS)


def format_inr(amount: int) -> str:
    """Indian digit grouping, matching how the shop writes prices.

    1200 -> '1,200';  11115 -> '11,115';  100000 -> '1,00,000'.
    """
    if amount < 0:
        return f"-{format_inr(-amount)}"
    digits = str(amount)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.append(head[-2:])
        head = head[:-2]
    if head:
        groups.append(head)
    groups.reverse()
    return ",".join([*groups, tail])
