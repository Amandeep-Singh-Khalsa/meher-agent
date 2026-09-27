"""PII masking and the log filter that enforces it.

The two graded shapes are asserted byte for byte. The logging test is graded in
its own right: a record carrying a raw address and a raw number must come out of
a real `StreamHandler` with neither one in it.
"""

from __future__ import annotations

import io
import logging

import pytest

from meher_agent.logging_utils import (
    HANDLER_NAME,
    PiiMaskingFilter,
    RedactingAdapter,
    log_safe,
    setup_logging,
)
from meher_agent.safety.pii import mask_email, mask_phone, mask_text, mask_value

RAW_EMAIL = "ritu.m@example.com"
RAW_PHONE = "9876543210"
MASKED_EMAIL = "r*****@example.com"
MASKED_PHONE = "******3210"


# --------------------------------------------------------------------------
# The two graded examples
# --------------------------------------------------------------------------


def test_graded_email_example() -> None:
    assert mask_email(RAW_EMAIL) == MASKED_EMAIL


def test_graded_phone_example() -> None:
    assert mask_phone(RAW_PHONE) == MASKED_PHONE


def test_graded_examples_hold_through_mask_text() -> None:
    assert mask_text(RAW_EMAIL) == MASKED_EMAIL
    assert mask_text(RAW_PHONE) == MASKED_PHONE


# --------------------------------------------------------------------------
# Absent values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mask", [mask_email, mask_phone, mask_value])
def test_none_stays_none(mask) -> None:
    assert mask(None) is None


@pytest.mark.parametrize("blank", ["", " ", "\t\n", "   "])
def test_blank_is_reported_as_absent(blank: str) -> None:
    assert mask_email(blank) is None
    assert mask_phone(blank) is None
    assert mask_value(blank) is None


def test_blank_text_masks_to_blank() -> None:
    assert mask_text("") == ""


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------


def test_single_character_local_part() -> None:
    assert mask_email("a@b.com") == "a*@b.com"


def test_long_local_part_keeps_one_star_per_hidden_character() -> None:
    local = "a" * 40
    masked = mask_email(f"{local}@example.com")
    assert masked is not None
    assert masked == "a" + "*" * 39 + "@example.com"


def test_address_is_masked_not_lowercased() -> None:
    assert mask_email("Ritu.M@Example.COM") == "R*****@Example.COM"


def test_address_with_display_name_keeps_the_surrounding_text() -> None:
    assert mask_email("Ritu Malhotra <ritu.m@example.com>") == (
        f"Ritu Malhotra <{MASKED_EMAIL}>"
    )


def test_value_without_an_address_is_absent() -> None:
    assert mask_email("Ritu Malhotra") is None
    assert mask_email("orders@meher-sweets") is None


def test_digit_heavy_local_part_is_not_also_read_as_a_phone() -> None:
    assert mask_email("9876543210@example.com") == "9*********@example.com"


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


def test_twelve_digit_number_normalises_to_the_last_ten() -> None:
    assert mask_phone("919876543210") == MASKED_PHONE


def test_country_code_with_spaces() -> None:
    assert mask_phone("+91 98765 43210") == MASKED_PHONE


def test_country_code_with_a_hyphen() -> None:
    assert mask_phone("+91-98765-43210") == MASKED_PHONE


def test_leading_trunk_zero() -> None:
    assert mask_phone("09876543210") == MASKED_PHONE


def test_double_prefix() -> None:
    assert mask_phone("0919876543210") == MASKED_PHONE


def test_surrounding_whitespace_is_ignored() -> None:
    assert mask_phone("  9876543210  ") == MASKED_PHONE


@pytest.mark.parametrize("value", ["987654321", "98765432101", "12345", "not a phone"])
def test_a_number_that_is_not_a_mobile_is_absent(value: str) -> None:
    assert mask_phone(value) is None


def test_a_longer_digit_run_is_not_half_masked() -> None:
    text = "order 98765432109876 confirmed"
    assert mask_text(text) == text


# --------------------------------------------------------------------------
# Dispatch and free text
# --------------------------------------------------------------------------


def test_mask_value_dispatches_on_an_address() -> None:
    assert mask_value(RAW_EMAIL) == MASKED_EMAIL


def test_mask_value_dispatches_on_a_number() -> None:
    assert mask_value(RAW_PHONE) == MASKED_PHONE
    assert mask_value("+91 98765 43210") == MASKED_PHONE


def test_mask_value_dispatches_on_prose() -> None:
    assert mask_value(f"reach Ritu on {RAW_PHONE} or {RAW_EMAIL}") == (
        f"reach Ritu on {MASKED_PHONE} or {MASKED_EMAIL}"
    )


def test_mask_text_scrubs_every_item_and_keeps_the_words() -> None:
    text = (
        "Ritu Malhotra asked for 30 gift boxes. Email ritu.m@example.com, "
        "call 9876543210 or 8123456789, or write to amit.v@meher.example."
    )
    assert mask_text(text) == (
        "Ritu Malhotra asked for 30 gift boxes. Email r*****@example.com, "
        "call ******3210 or ******6789, or write to a*****@meher.example."
    )


def test_mask_text_leaves_clean_prose_untouched() -> None:
    text = "The shop closes at 10 pm and we deliver within 8 km."
    assert mask_text(text) == text


def test_mask_text_is_idempotent() -> None:
    once = mask_text(f"{RAW_EMAIL} and {RAW_PHONE}")
    assert mask_text(once) == once
    assert mask_text(mask_text(mask_text(once))) == once


def test_mask_text_leaves_non_latin_scripts_readable() -> None:
    text = "मेहर स्वीट्स 10 pm तक खुली है — 9876543210 पर संपर्क करें 🎉"
    masked = mask_text(text)
    assert masked == "मेहर स्वीट्स 10 pm तक खुली है — ******3210 पर संपर्क करें 🎉"
    assert "मेहर स्वीट्स" in masked


def test_mask_text_does_not_swallow_trailing_punctuation() -> None:
    assert mask_text(f"Write to {RAW_EMAIL}.") == f"Write to {MASKED_EMAIL}."


# --------------------------------------------------------------------------
# Log safety
# --------------------------------------------------------------------------


class _CapturingLogger:
    def __init__(self, name: str) -> None:
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.addFilter(PiiMaskingFilter())
        self.logger = logging.getLogger(name)
        self.logger.handlers = [self.handler]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def restore(self) -> None:
        self.logger.handlers = []
        self.logger.propagate = True

    @property
    def text(self) -> str:
        return self.stream.getvalue()


@pytest.fixture
def captured() -> _CapturingLogger:
    handle = _CapturingLogger("meher_agent.tests.pii_masking")
    try:
        yield handle
    finally:
        handle.restore()


def test_logs_never_contain_a_raw_email_or_a_raw_phone(captured: _CapturingLogger) -> None:
    captured.logger.info("lead saved for %s on %s", RAW_EMAIL, RAW_PHONE)
    text = captured.text
    assert RAW_EMAIL not in text
    assert RAW_PHONE not in text
    assert MASKED_EMAIL in text
    assert MASKED_PHONE in text


def test_log_filter_masks_a_fully_formatted_message(captured: _CapturingLogger) -> None:
    captured.logger.info(f"customer {RAW_EMAIL} called {RAW_PHONE}")
    text = captured.text
    assert RAW_EMAIL not in text
    assert RAW_PHONE not in text


def test_log_filter_masks_dict_style_args(captured: _CapturingLogger) -> None:
    captured.logger.info("lead %(email)s / %(phone)s", {"email": RAW_EMAIL, "phone": RAW_PHONE})
    text = captured.text
    assert RAW_EMAIL not in text
    assert RAW_PHONE not in text
    assert MASKED_EMAIL in text
    assert MASKED_PHONE in text


def test_log_filter_tolerates_args_none(captured: _CapturingLogger) -> None:
    captured.logger.info("nothing to mask here")
    assert "nothing to mask here" in captured.text


def test_log_filter_keeps_non_string_args_usable(captured: _CapturingLogger) -> None:
    captured.logger.info("%s boxes for %d rupees", "thirty", 60)
    assert "thirty boxes for 60 rupees" in captured.text


def test_log_filter_masks_a_non_string_message(captured: _CapturingLogger) -> None:
    captured.logger.info(ValueError(f"bad contact {RAW_EMAIL} / {RAW_PHONE}"))
    text = captured.text
    assert RAW_EMAIL not in text
    assert RAW_PHONE not in text
    assert MASKED_EMAIL in text


def test_log_filter_masks_a_rendered_traceback(captured: _CapturingLogger) -> None:
    try:
        raise ValueError(f"rejected contact {RAW_EMAIL} on {RAW_PHONE}")
    except ValueError:
        captured.logger.exception("save_lead blew up")
    text = captured.text
    assert RAW_EMAIL not in text
    assert RAW_PHONE not in text
    assert MASKED_PHONE in text
    assert "Traceback" in text
    assert "ValueError" in text


def test_log_safe_masks_before_emission(captured: _CapturingLogger) -> None:
    log_safe(captured.logger, logging.WARNING, "escalated %s for %s", RAW_EMAIL, RAW_PHONE)
    text = captured.text
    assert RAW_EMAIL not in text
    assert RAW_PHONE not in text


def test_redacting_adapter_masks_message_and_extra(captured: _CapturingLogger) -> None:
    adapter = RedactingAdapter(captured.logger, {})
    adapter.info("escalating %s", RAW_EMAIL, extra={"phone": RAW_PHONE, "boxes": 30})
    text = captured.text
    assert RAW_EMAIL not in text
    assert RAW_PHONE not in text


# --------------------------------------------------------------------------
# setup_logging
# --------------------------------------------------------------------------


@pytest.fixture
def root_logger() -> logging.Logger:
    root = logging.getLogger()
    handlers = list(root.handlers)
    filters = list(root.filters)
    level = root.level
    try:
        yield root
    finally:
        root.handlers = handlers
        root.filters = filters
        root.setLevel(level)


def _installed(root: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in root.handlers if handler.get_name() == HANDLER_NAME]


def test_setup_logging_is_idempotent(root_logger: logging.Logger) -> None:
    setup_logging("INFO")
    setup_logging("DEBUG")
    assert len(_installed(root_logger)) == 1
    assert root_logger.level == logging.DEBUG


def test_setup_logging_installs_the_filter_on_the_handler_and_the_root(
    root_logger: logging.Logger,
) -> None:
    setup_logging("INFO")
    installed = _installed(root_logger)
    assert len(installed) == 1
    assert any(isinstance(f, PiiMaskingFilter) for f in installed[0].filters)
    assert any(isinstance(f, PiiMaskingFilter) for f in root_logger.filters)
    assert root_logger.level == logging.INFO


def test_setup_logging_masks_a_record_reaching_the_root_handler(
    root_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    setup_logging("INFO")
    logger = logging.getLogger("meher_agent.tests.pii_setup")
    logger.propagate = True
    logger.setLevel(logging.INFO)
    logger.info("escalate %s now", RAW_PHONE)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    captured = capsys.readouterr()
    assert RAW_PHONE not in captured.err
    assert MASKED_PHONE in captured.err


def test_setup_logging_falls_back_to_info_for_an_unknown_level(
    root_logger: logging.Logger,
) -> None:
    setup_logging("VERBOSE-ISH")
    assert root_logger.level == logging.INFO
