"""The authoring primitive: the one place a pattern-shaped check is built.

Everything here is about a check that would check LESS than it was configured
to. That is the failure this package exists to prevent, and a primitive handed
to contributors and to users writing runtime rules is where a new instance of
it would arrive.
"""

from __future__ import annotations

import pytest

from jamjet_guardrails.authoring import Limits, _limit_spans


def test_content_at_the_limit_is_not_over_it() -> None:
    """The boundary in the direction that fails open. `>=` here would report a
    limit on content that respected it."""
    assert _limit_spans("x" * 40, Limits(max_chars=40)) == []


def test_content_one_character_past_the_limit_spans_the_excess() -> None:
    assert _limit_spans("x" * 41, Limits(max_chars=40)) == [("LENGTH_LIMIT", (40, 41))]


def test_the_span_runs_from_the_first_excess_character_to_the_end() -> None:
    assert _limit_spans("x" * 50, Limits(max_chars=40)) == [("LENGTH_LIMIT", (40, 50))]


def test_bytes_are_counted_as_utf8_not_as_characters() -> None:
    """Four characters, twelve bytes. A limit of 8 bytes is exceeded at the
    third character, and a character count would not notice at all."""
    content = "你好世界"
    assert _limit_spans(content, Limits(max_bytes=8)) == [("LENGTH_LIMIT", (2, 4))]
    assert _limit_spans(content, Limits(max_chars=8)) == []


def test_the_byte_span_starts_at_the_character_that_crossed_the_limit() -> None:
    assert _limit_spans("abécd", Limits(max_bytes=3)) == [("LENGTH_LIMIT", (2, 5))]


def test_lines_are_counted_and_the_span_starts_at_the_first_excess_line() -> None:
    assert _limit_spans("a\nb\nc", Limits(max_lines=2)) == [("LENGTH_LIMIT", (4, 5))]


def test_content_at_the_line_limit_is_not_over_it() -> None:
    assert _limit_spans("a\nb", Limits(max_lines=2)) == []


def test_a_trailing_newline_does_not_open_a_line_that_is_not_there() -> None:
    """ "a\\nb\\n" is two lines, not three. Counting newlines rather than line
    starts reports a third line with nothing in it, and the span would be
    empty, which the chain refuses."""
    assert _limit_spans("a\nb\n", Limits(max_lines=2)) == []


def test_several_limits_report_one_finding_from_the_earliest_breach() -> None:
    """One finding, not three. Three overlapping spans would merge into one
    placeholder anyway, and the audit record would claim three detections of
    one fact."""
    assert _limit_spans("x" * 50, Limits(max_chars=40, max_bytes=45)) == [
        ("LENGTH_LIMIT", (40, 50))
    ]


def test_limits_that_select_nothing_are_refused() -> None:
    with pytest.raises(ValueError, match="selects nothing"):
        Limits()


@pytest.mark.parametrize("field", ["max_chars", "max_bytes", "max_lines"])
def test_a_limit_below_one_is_refused(field: str) -> None:
    """A limit of zero denies every input including the empty one, which is a
    check that is on rather than a check that is configured."""
    with pytest.raises(ValueError, match="at least 1"):
        Limits(**{field: 0})
