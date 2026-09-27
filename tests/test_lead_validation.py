"""normalise_phone / normalise_email / normalise_date / normalise_name, and the
seed-01 lead end to end.

The normalisers are the graded surface: the eval harness imports the same
functions, so an edge case that is right here is right there.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from meher_agent.config import load_config
from meher_agent.data.corpus import load_corpus
from meher_agent.tools.registry import ToolRegistry
from meher_agent.tools.validation import (
    ValidationError,
    clean_optional_text,
    is_valid_date,
    is_valid_email,
    is_valid_name,
    is_valid_phone,
    normalise_date,
    normalise_email,
    normalise_name,
    normalise_phone,
)

SEED_01_ARGS = {
    "name": "Ritu Malhotra",
    "need": "30 large gift boxes for an office Diwali party",
    "email": "ritu.m@example.com",
    "date": "3 November 2026",
    "quantity": "30 boxes",
}


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def corpus(config):
    return load_corpus(config.runtime.data_dir)


@pytest.fixture
def registry(corpus, config):
    return ToolRegistry(corpus, config)


def raises(fn, *args, **kwargs) -> str:
    with pytest.raises(ValidationError) as excinfo:
        fn(*args, **kwargs)
    return str(excinfo.value)


# --------------------------------------------------------------------------
# phone
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("9876543210", "9876543210"),
        ("+91 98765 43210", "9876543210"),
        ("+919876543210", "9876543210"),
        ("09876543210", "9876543210"),
        ("091 98765-43210", "9876543210"),
        ("98765-43210", "9876543210"),
        ("  98765 43210  ", "9876543210"),
        ("6-1234-56789", "6123456789"),
        ("+91-9876543210", "9876543210"),
        (9876543210, "9876543210"),
    ],
)
def test_normalise_phone_accepts(raw: object, expected: str) -> None:
    assert normalise_phone(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        None,
        "1234567890",
        "5123456789",
        "987654321",
        "98765432101",
        "abcdefghij",
        "+91 1234",
        98765.0,
        True,
    ],
)
def test_normalise_phone_rejects(raw: object) -> None:
    assert raises(normalise_phone, raw)
    assert is_valid_phone(raw) is False


def test_normalise_phone_error_names_the_rule() -> None:
    assert "starting with 6, 7, 8 or 9" in raises(normalise_phone, "1234567890")
    assert "10 digits" in raises(normalise_phone, "12345")


# --------------------------------------------------------------------------
# email
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ritu.m@example.com", "ritu.m@example.com"),
        ("RITU.M@Example.COM", "ritu.m@example.com"),
        ("  ritu.m@example.com  ", "ritu.m@example.com"),
        ("a+b@example.co.in", "a+b@example.co.in"),
        ("Ritu_Malhotra@sub.example.com", "ritu_malhotra@sub.example.com"),
    ],
)
def test_normalise_email_accepts(raw: str, expected: str) -> None:
    assert normalise_email(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        None,
        "not-an-email",
        "a@b",
        "a@b.",
        "@example.com",
        "ritu@example",
        "ritu mail@example.com",
        "ritu@@example.com",
        "ritu.m@example..com",
        ".ritu@example.com",
        "ritu.@example.com",
    ],
)
def test_normalise_email_rejects(raw: object) -> None:
    assert raises(normalise_email, raw)
    assert is_valid_email(raw) is False


def test_normalise_email_length_limits() -> None:
    assert raises(normalise_email, "a" * 65 + "@example.com")
    assert raises(normalise_email, "a@" + ("sub." * 64) + "example.com")
    assert raises(normalise_email, "a" * 250 + "@example.com")
    assert is_valid_email("a" * 64 + "@example.com") is True


# --------------------------------------------------------------------------
# date
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-11-03", "2026-11-03"),
        ("2026-11-3", "2026-11-03"),
        ("3 November 2026", "2026-11-03"),
        ("3 Nov. 2026", "2026-11-03"),
        ("3nov2026", "2026-11-03"),
        ("3/11/2026", "2026-11-03"),
        ("03.11.2026", "2026-11-03"),
        ("  2026-11-03  ", "2026-11-03"),
    ],
)
def test_normalise_date_accepts(raw: str, expected: str) -> None:
    assert normalise_date(raw) == expected


def test_bare_day_and_month_resolves_to_the_next_occurrence() -> None:
    today = date(2026, 9, 27)
    assert normalise_date("3 November", today=today) == "2026-11-03"
    assert normalise_date("2 November", today=today) == "2026-11-02"
    assert normalise_date("3 Nov", today=today) == "2026-11-03"


def test_bare_day_and_month_rolls_over_a_past_date() -> None:
    assert normalise_date("3 January", today=date(2026, 9, 27)) == "2027-01-03"
    assert normalise_date("3 November", today=date(2026, 11, 3)) == "2026-11-03"


def test_bare_day_and_month_defaults_to_today() -> None:
    today = date.today()
    month_day = date(today.year if today.month <= 11 else today.year + 1, 11, 3)
    assert normalise_date("3 November") == month_day.isoformat()


@pytest.mark.parametrize(
    "raw", ["31/02/2026", "2026-13-01", "2026-02-30", "", "   ", None, "tomorrow", "3 Smarch 2026", "2026-11"]
)
def test_normalise_date_rejects(raw: object) -> None:
    assert raises(normalise_date, raw)
    assert is_valid_date(raw) is False


def test_normalise_date_error_shows_the_offending_value() -> None:
    assert "31/02/2026" in raises(normalise_date, "31/02/2026")
    assert "calendar date" in raises(normalise_date, "31/02/2026")


# --------------------------------------------------------------------------
# name and free text
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ritu Malhotra", "Ritu Malhotra"),
        ("  Ritu   Malhotra  ", "Ritu Malhotra"),
        ("Ritu\tMalhotra", "Ritu Malhotra"),
        ("रितु मल्होत्रा", "रितु मल्होत्रा"),
        ("Ritu Malhotra\n", "Ritu Malhotra"),
    ],
)
def test_normalise_name_accepts(raw: str, expected: str) -> None:
    assert normalise_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "12345", "!!", "..."])
def test_normalise_name_rejects(raw: object) -> None:
    assert raises(normalise_name, raw)
    assert is_valid_name(raw) is False


def test_normalise_name_length_limit() -> None:
    assert raises(normalise_name, "R" * 201)
    assert is_valid_name("R" * 200) is True


def test_clean_optional_text() -> None:
    assert clean_optional_text("  30   boxes  ") == "30 boxes"
    assert clean_optional_text("  \n ") is None
    assert clean_optional_text(None) is None
    assert clean_optional_text("x" * 20, max_len=5) == "xxxxx"
    assert clean_optional_text(30) == "30"


# --------------------------------------------------------------------------
# seed-01 end to end
# --------------------------------------------------------------------------


def test_seed_01_lead_saves_exactly_the_expected_values(registry: ToolRegistry) -> None:
    result = registry.run("save_lead", SEED_01_ARGS, conversation_id="seed-01")

    assert result.ok is True, result.error
    assert registry.leads.all()[0].model_dump() == {
        "name": "Ritu Malhotra",
        "need": "30 large gift boxes for an office Diwali party",
        "phone": None,
        "email": "ritu.m@example.com",
        "quantity": "30 boxes",
        "date": "2026-11-03",
        "conversation_id": "seed-01",
        "created_at": registry.leads.all()[0].created_at,
    }
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", registry.leads.all()[0].created_at)


def test_seed_01_recorded_action_args_are_post_validation(registry: ToolRegistry) -> None:
    registry.run("save_lead", SEED_01_ARGS, conversation_id="seed-01")
    action = registry.actions()[0]

    assert action.type == "save_lead"
    assert action.ok is True
    assert action.args["name"] == "Ritu Malhotra"
    assert action.args["email"] == "ritu.m@example.com"
    assert action.args["date"] == "2026-11-03"
    assert action.args["quantity"] == "30 boxes"
    assert action.args["need"] == SEED_01_ARGS["need"]
    assert "phone" not in action.args


def test_seed_01_as_the_model_would_write_it(registry: ToolRegistry) -> None:
    result = registry.run(
        "save_lead",
        {
            "name": "Ritu Malhotra",
            "need": "30 large gift boxes for the office Diwali party on 3 November",
            "phone": "+91 98765 43210",
            "quantity": "30 large gift boxes",
            "date": "3 Nov 2026",
        },
        conversation_id="seed-01",
    )
    assert result.ok is True, result.error
    lead = registry.leads.all()[0]
    assert lead.phone == "9876543210"
    assert lead.date == "2026-11-03"
    assert lead.email is None
