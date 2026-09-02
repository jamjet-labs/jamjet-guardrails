"""The authoring primitive: the one place a pattern-shaped check is built.

Everything here is about a check that would check LESS than it was configured
to. That is the failure this package exists to prevent, and a primitive handed
to contributors and to users writing runtime rules is where a new instance of
it would arrive.
"""

from __future__ import annotations

import re

import pytest

from jamjet_guardrails.authoring import Limits, _limit_spans, _nests_unbounded_repeats


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


def test_byte_limit_boundary_exactly_at_the_limit() -> None:
    """A character whose cumulative byte total exactly equals the limit separates
    the two boundary forms. The test 'abécd' with max_bytes=3 jumps from 2 bytes
    to 4 bytes at the third character, so neither `>` nor `>=` sees the boundary.
    This case lands exactly on 4 bytes, where `>` and `>=` disagree: `>` fires at
    the next character (index 3), while `>=` fires at the character that reached
    the limit (index 2). The incorrect form truncates a character the limit did
    not object to."""
    assert _limit_spans("aébc", Limits(max_bytes=4)) == [("LENGTH_LIMIT", (3, 4))]


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


# Each row was run against the implementation on 3.10.20 and 3.14.5 and agreed.
# The `False` rows matter more than the `True` ones: a guard that refuses an
# ordinary pattern is a guard that gets worked around.
_NESTING = [
    (r"(a+)+b", True),
    (r"(a*)*", True),
    (r"([a-z]+)*", True),
    (r"(?:\w+\s?)*", True),
    (r"(?=(a+)+)", True),
    (r"^(a+)+$", True),
    (r"((a+))+", True),
    (r"(x|y+)*", True),
    (r"(a|aa)+$", False),
    (r"\bJIRA-\d{4,}\b", False),
    (r"[a-z0-9-]+\.corp\.example", False),
    (r"a{2,3}b+", False),
    (r"(a{1,3})+", False),
    (r"(?:ab)+c", False),
    (r"(a+){2}", False),
    (r"\d+", False),
    (r"(?:[a-f0-9]{2})+", False),
    (r"(a?)*", False),
]


@pytest.mark.parametrize(("pattern", "nests"), _NESTING, ids=[p for p, _ in _NESTING])
def test_the_nested_repeat_guard_agrees_with_the_recorded_table(pattern: str, nests: bool) -> None:
    assert _nests_unbounded_repeats(pattern) is nests


def test_an_alternation_of_bounded_repeats_is_not_nesting() -> None:
    """`(a|aa)+` is the textbook catastrophic pattern that this guard does NOT
    catch, and the docstring says so. Pinned here so the claim in the docstring
    is a measurement rather than a hedge: if the guard is ever widened to catch
    it, this test fails and the docstring is corrected in the same commit."""
    assert _nests_unbounded_repeats(r"(a|aa)+$") is False


def test_a_compiled_pattern_is_accepted_and_its_flags_are_used() -> None:
    assert _nests_unbounded_repeats(re.compile(r"(a+)+", re.IGNORECASE)) is True
