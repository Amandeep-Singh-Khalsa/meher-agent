"""Normalisation and validation for tool arguments.

The task spec fixes these rules, and the evaluation harness has to normalise
identically when it checks `expect_lead`. One implementation, imported by both,
so the harness can never disagree with the agent about what a valid phone or
email is.

Phone:  normalised to 10 digits, must be an Indian mobile starting 6-9.
        Accepts "+91", a leading "0", spaces and dashes.
Email:  lower-cased and RFC-valid.
Date:   "YYYY-MM-DD".
"""

from __future__ import annotations

import re
from datetime import date, timedelta

__all__ = [
    "ValidationError",
    "normalise_phone",
    "is_valid_phone",
    "normalise_email",
    "is_valid_email",
    "normalise_date",
    "is_valid_date",
    "normalise_name",
    "is_valid_name",
    "clean_optional_text",
]


class ValidationError(ValueError):
    """Raised when a tool argument fails validation.

    The agent loop turns this into a `tool` error message back to the model; it
    must never propagate to the HTTP layer as a crash.
    """


# --------------------------------------------------------------------------
# Phone
# --------------------------------------------------------------------------

_DIGITS_RE = re.compile(r"\d")


def normalise_phone(raw: object) -> str:
    """Return the bare 10-digit Indian mobile number, or raise ValidationError."""
    if raw is None:
        raise ValidationError("phone is required")
    text = str(raw).strip()
    if not text:
        raise ValidationError("phone is empty")

    digits = "".join(_DIGITS_RE.findall(text))

    # Strip an international or trunk prefix, longest sensible form first.
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    elif len(digits) == 13 and digits.startswith("091"):
        digits = digits[3:]

    if len(digits) != 10:
        raise ValidationError(
            f"phone must be 10 digits after removing a +91 or leading 0, got {len(digits)} digits"
        )
    if digits[0] not in "6789":
        raise ValidationError("phone must be an Indian mobile number starting with 6, 7, 8 or 9")
    return digits


def is_valid_phone(raw: object) -> bool:
    try:
        normalise_phone(raw)
    except ValidationError:
        return False
    return True


# --------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------

# Pragmatic RFC 5322 subset: a dot-atom local part, a dotted domain, no spaces.
_EMAIL_RE = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)"
    r"@(?P<domain>[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+)$"
)


def normalise_email(raw: object) -> str:
    """Return the lower-cased address, or raise ValidationError."""
    if raw is None:
        raise ValidationError("email is required")
    text = str(raw).strip()
    if not text:
        raise ValidationError("email is empty")
    if len(text) > 254:
        raise ValidationError("email is longer than 254 characters")
    text = text.lower()
    m = _EMAIL_RE.match(text)
    if not m:
        raise ValidationError(f"'{str(raw)[:60]}' is not a valid email address")
    if len(m.group("local")) > 64:
        raise ValidationError("email local part is longer than 64 characters")
    if len(m.group("domain")) > 253:
        raise ValidationError("email domain is longer than 253 characters")
    return text


def is_valid_email(raw: object) -> bool:
    try:
        normalise_email(raw)
    except ValidationError:
        return False
    return True


# --------------------------------------------------------------------------
# Date
# --------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_RE_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_RE_DMY_SLASH = re.compile(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$")
_RE_DMY_WORD = re.compile(rf"^(\d{{1,2}})\s*([A-Za-z]+)\.?\s*(\d{{4}})?$", re.IGNORECASE)


def normalise_date(raw: object, *, today: date | None = None) -> str:
    """Return 'YYYY-MM-DD' or raise ValidationError.

    Accepts ISO, DD/MM/YYYY, DD.MM.YYYY, and "3 November 2026" / "3 Nov" /
    "2 November tak". A missing year resolves to the next occurrence of that
    month/day, so "2 November" in September 2026 becomes 2026-11-02.
    """
    if raw is None:
        raise ValidationError("date is required")
    text = str(raw).strip()
    if not text:
        raise ValidationError("date is empty")

    today = today or date.today()

    m = _RE_ISO.match(text)
    if m:
        return _build(int(m.group(1)), int(m.group(2)), int(m.group(3)), text)

    m = _RE_DMY_SLASH.match(text)
    if m:
        # Indian convention: day first.
        return _build(int(m.group(3)), int(m.group(2)), int(m.group(1)), text)

    m = _RE_DMY_WORD.match(text)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        if month_name not in _MONTHS:
            raise ValidationError(f"'{text}' has an unrecognised month name")
        month = _MONTHS[month_name]
        year_raw = m.group(3)
        if year_raw:
            return _build(int(year_raw), month, day, text)
        return _next_occurrence(day, month, today, text)

    raise ValidationError(
        f"'{text[:40]}' is not a recognised date; use YYYY-MM-DD, DD/MM/YYYY or '3 November 2026'"
    )


def _build(year: int, month: int, day: int, original: str) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise ValidationError(f"'{original}' is not a real calendar date: {exc}") from exc


def _next_occurrence(day: int, month: int, today: date, original: str) -> str:
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError as exc:
            raise ValidationError(f"'{original}' is not a real calendar date: {exc}") from exc
        if candidate >= today:
            return candidate.isoformat()
    raise ValidationError(f"'{original}' is not a real calendar date")


def is_valid_date(raw: object) -> bool:
    try:
        normalise_date(raw)
    except ValidationError:
        return False
    return True


# --------------------------------------------------------------------------
# Text fields
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def clean_optional_text(raw: object, *, max_len: int = 500) -> str | None:
    """Collapse whitespace and drop empties. Returns None for empty input."""
    if raw is None:
        return None
    text = _WS_RE.sub(" ", str(raw)).strip()
    if not text:
        return None
    return text[:max_len]


def normalise_name(raw: object) -> str:
    """Return a display name with collapsed whitespace, or raise ValidationError."""
    if raw is None:
        raise ValidationError("name is required")
    text = _WS_RE.sub(" ", str(raw)).strip()
    if not text:
        raise ValidationError("name is empty")
    if len(text) > 200:
        raise ValidationError("name is longer than 200 characters")
    if not re.search(r"[A-Za-z\u0900-\u097F]", text):
        raise ValidationError("name must contain letters")
    return text


def is_valid_name(raw: object) -> bool:
    try:
        normalise_name(raw)
    except ValidationError:
        return False
    return True
