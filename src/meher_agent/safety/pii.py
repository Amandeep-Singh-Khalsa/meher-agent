"""PII masking.

One rule set with three callers: the HTTP response, the logging filter, and any
other place that has to show a contact detail to a human without disclosing it.
The two shapes the task grades on are reproduced byte for byte:

    "ritu.m@example.com"  ->  "r*****@example.com"
    "9876543210"          ->  "******3210"

Two conventions hold everywhere:

* A blank input is *absent*, not masked, so ``None``, ``""`` and whitespace all
  come back as ``None``. That is what lets the API render JSON ``null`` without a
  second "is it empty" check at every call site.
* Masking is idempotent, so re-masking a log record that a root filter and a
  handler filter both touch cannot corrupt an already-masked value.
"""

from __future__ import annotations

import re

__all__ = [
    "PHONE_VISIBLE_DIGITS",
    "mask_email",
    "mask_phone",
    "mask_text",
    "mask_value",
]

#: The last four digits survive: enough for a human to recognise which number is
#: being discussed, far too little to place a call.
PHONE_VISIBLE_DIGITS = 4

_EMAIL_RE = re.compile(
    r"(?P<local>[A-Za-z0-9._%+\-]+)@(?P<domain>[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+)"
)

#: An Indian mobile, optionally country- and space-prefixed. The lookarounds are
#: what keep this conservative: a longer digit run (an order number, a timestamp
#: in nanoseconds) is left alone rather than half-masked.
_PHONE_RE = re.compile(
    r"(?<![\d+])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
)

#: A value that is *only* a number, possibly decorated, is a phone-shaped field
#: rather than prose.
_PHONE_SHAPE_RE = re.compile(r"^[\d][\d\s+\-()]*$")

_DIGITS_RE = re.compile(r"\D")


def _mask_local(local: str) -> str:
    """First character plus one star per hidden character, never bare."""
    hidden = max(1, len(local) - 1)
    return f"{local[0]}{'*' * hidden}"


def _mask_digits(digits: str) -> str:
    return f"{'*' * max(0, len(digits) - PHONE_VISIBLE_DIGITS)}{digits[-PHONE_VISIBLE_DIGITS:]}"


def _redact_email(match: re.Match[str]) -> str:
    return f"{_mask_local(match.group('local'))}@{match.group('domain')}"


def _redact_phone(match: re.Match[str]) -> str:
    digits = _DIGITS_RE.sub("", match.group(0))
    return _mask_digits(digits[-10:])


def _strip_country_code(digits: str) -> str:
    """Reduce +91 / 091 / a trunk 0 prefix, longest plausible form first."""
    if len(digits) == 13 and digits.startswith("091"):
        return digits[3:]
    if len(digits) == 12 and digits.startswith("91"):
        return digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        return digits[1:]
    return digits


def mask_email(value: str | None) -> str | None:
    """Mask an address, or ``None`` when the value holds no address at all."""
    if value is None:
        return None
    text = value.strip()
    if not text or _EMAIL_RE.search(text) is None:
        return None
    return mask_text(text)


def mask_phone(value: str | None) -> str | None:
    """Mask a phone number, or ``None`` when it is not a 10-digit mobile.

    Anything that is not exactly one mobile after the country code comes off is
    reported as absent rather than echoed back: a half-understood number is not
    safe to show, and not showing it is the cheapest failure mode.
    """
    if value is None:
        return None
    digits = _strip_country_code(_DIGITS_RE.sub("", value))
    if len(digits) != 10:
        return None
    return _mask_digits(digits)


def mask_text(text: str) -> str:
    """Scrub every address and every phone number out of free text.

    Addresses go first: a local part may itself be digit-heavy, and masking it
    out of the text before the phone pass avoids a second match inside an
    address that has already been rewritten.
    """
    if not text:
        return text
    return _PHONE_RE.sub(_redact_phone, _EMAIL_RE.sub(_redact_email, text))


def mask_value(value: str | None) -> str | None:
    """Mask a value of unknown shape: address, number, or prose."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if _EMAIL_RE.search(text) is not None:
        return mask_email(text)
    if _PHONE_SHAPE_RE.match(text) is not None:
        return mask_phone(text)
    return mask_text(text)
