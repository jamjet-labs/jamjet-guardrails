"""Folded views, and the offsets that map a match back to what produced it.

A check that matches over a transformed view of the content has to report a
span into the ORIGINAL content, because that is what the chain rewrites and
what a corpus labels. The transformation is not length-preserving in either
direction: casefolding the German sharp s produces two characters from one,
and stripping a zero-width space produces none from one. Both directions are
here, because a mapping that is right for one is silently wrong for the other.

The four tests at the end arrived with the UTS #39 skeleton, which is the
first caller that needs a map to compose and the first that produces one that
is not non-decreasing. Each was watched to FAIL, `__pycache__` cleared
between runs, against: `span` reading the first and last entries of the map
rather than its minimum and maximum; `span` dropping the plus one from the
end; `compose` returning the inner view unchanged; and `compose` dropping the
check that the two views meet.
"""

from __future__ import annotations

import pytest

from jamjet_guardrails._fold import _Folded, casefold_view, compose, fold


def test_an_identity_fold_leaves_every_span_where_it_was() -> None:
    view = fold("hello world", lambda ch: ch)
    assert view.text == "hello world"
    assert view.span(6, 11) == (6, 11)


def test_casefolding_lowers_the_view_and_keeps_the_source_span() -> None:
    view = casefold_view("see PROJECT BLUEBIRD here")
    start = view.text.index("project bluebird")
    assert view.span(start, start + len("project bluebird")) == (4, 20)


def test_a_fold_that_expands_one_character_maps_every_product_to_its_source() -> None:
    """The sharp s casefolds to two characters, so the view is longer than the
    source and every index after it is shifted. A span read off the view
    without the map would run past the end of the word."""
    source = "Straße BLUEBIRD"
    view = casefold_view(source)
    assert view.text == "strasse bluebird"
    start = view.text.index("strasse")
    assert view.span(start, start + len("strasse")) == (0, 6)
    assert source[slice(*view.span(start, start + len("strasse")))] == "Straße"


def test_a_match_that_starts_inside_an_expanded_product_still_finds_its_source() -> None:
    """The existing expansion test starts its match before the sharp s and
    ends after it, so the product's two view characters are read only from
    the interior of a longer match: ``start`` and ``end`` both land on plain
    one-to-one characters either side of it. This one's match starts ON the
    product itself. ``origin`` for "Straße" is (0, 1, 2, 3, 4, 4, 5): view
    offset 4 is the first of the two view characters the sharp s produces,
    and reading ``origin[start]`` from inside that run of duplicate values,
    rather than from a plain one-to-one character before it, has never been
    exercised before."""
    source = "Straße"
    view = casefold_view(source)
    assert view.origin == (0, 1, 2, 3, 4, 4, 5)
    start = view.text.index("sse")
    span = view.span(start, start + len("sse"))
    assert span == (4, 6)
    assert source[slice(*span)] == "ße"


def test_a_match_starting_on_the_second_half_of_a_product_still_covers_it_whole() -> None:
    """A different boundary from the match above, whose match starts on the
    product's FIRST view character. This one's match is only the SECOND of
    the sharp s's two view characters, plus the letter after it, and the
    source span still comes back as the WHOLE source character the product
    came from rather than half of one, because half a source character
    cannot be redacted."""
    source = "Straße"
    view = casefold_view(source)
    span = view.span(5, 7)
    assert span == (4, 6)
    assert source[slice(*span)] == "ße"


def test_a_fold_that_deletes_a_character_spans_the_run_that_carried_it() -> None:
    """A marker laundered with a zero-width space matches in the view, and the
    span has to cover the zero-width space too: a redaction that left it
    standing would leave the launderer's byte in content reported as rewritten."""
    source = "<|im_\u200bstart|>"
    view = fold(source, lambda ch: "" if ch == "\u200b" else ch)
    assert view.text == "<|im_start|>"
    start = view.text.index("<|im_start|>")
    span = view.span(start, start + len("<|im_start|>"))
    assert span == (0, 13)
    assert source[slice(*span)] == source


def test_a_match_ending_immediately_before_a_deleted_character_stops_at_it() -> None:
    """The end is the last matched character's source index PLUS ONE, never
    the source index of the character after the match, and here that is not
    a detail: source index 5 is the deleted zero-width space, missing from
    ``origin``, so the map jumps straight from 4 to 6. Reading ``origin[end]``
    instead of ``origin[end - 1] + 1`` would read 6, where the match's last
    character maps to 4, and swallow the deleted character into a span the
    match never touched. A redaction over that span would remove a byte
    nothing detected, which is the defect class this module exists to
    prevent."""
    source = "<|im_\u200bstart|>"
    view = fold(source, lambda ch: "" if ch == "\u200b" else ch)
    assert view.origin == (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)
    span = view.span(0, 5)
    assert span == (0, 5)
    assert source[slice(*span)] == "<|im_"


def test_every_character_of_the_view_has_exactly_one_source_index() -> None:
    view = casefold_view("Straße")
    assert len(view.origin) == len(view.text)
    assert view.source_length == 6


def test_an_empty_span_is_refused_rather_than_returning_a_zero_width_one() -> None:
    """A zero-width span is a non-detection wearing a detection's clothes, and
    the chain refuses one. Refusing here names the caller's mistake instead."""
    view = casefold_view("abc")
    with pytest.raises(ValueError, match="empty"):
        view.span(1, 1)


def test_a_span_past_the_view_is_refused() -> None:
    view = casefold_view("abc")
    with pytest.raises(ValueError, match="does not index"):
        view.span(0, 9)


def test_a_fold_that_deletes_everything_produces_an_empty_view() -> None:
    """Not an error. A content of nothing but zero-width characters folds to
    nothing, and the caller finds no match rather than being handed an
    exception from a helper."""
    view = fold("\u200b\u200b", lambda ch: "")
    assert view.text == ""
    assert view.origin == ()


def test_a_span_over_a_reordering_map_covers_the_whole_source_run() -> None:
    """The case ``span`` takes a minimum and a maximum for.

    Every fold ``fold`` builds produces a NON-DECREASING map, and the UTS #39
    skeleton's canonical ordering step does not: it permutes combining marks by
    class, so a map of (0, 2, 1) is what the caller is handed. Reading the
    first and last entries of the matched range returns (0, 2) there, one
    character short of the run the match covered, and a redaction over that
    span leaves a character standing inside content reported as rewritten.

    Built literally rather than through ``jamjet_guardrails._unicode``, because
    this is a property of the map and not of normalisation, and a test that
    reached for the skeleton would fail for whichever of the two broke.
    """
    view = _Folded(text="abc", origin=(0, 2, 1), source_length=3)
    assert view.span(0, 3) == (0, 3)
    assert view.span(1, 3) == (1, 3)
    assert view.span(1, 2) == (2, 3)


def test_a_span_over_a_non_decreasing_map_is_what_it_always_was() -> None:
    """The widening above must not have moved any answer that was already right.

    Minimum and maximum over a non-decreasing map are its first and last
    entries, so every existing caller sees the same spans. Asserted rather than
    argued, over the contracting and the expanding fold both.
    """
    expanding = casefold_view("Stra\u00dfe")
    assert expanding.span(4, 7) == (expanding.origin[4], expanding.origin[6] + 1)
    contracting = fold("<|im_\u200bstart|>", lambda ch: "" if ch == "\u200b" else ch)
    assert contracting.span(0, 5) == (contracting.origin[0], contracting.origin[4] + 1)


def test_composing_two_views_gives_one_map_back_to_the_original_source() -> None:
    """Three folds deep, one map. The skeleton is what needs this.

    The alternative is a caller holding three views and walking them by hand at
    every match site, which is how the two copies of the span arithmetic in
    ``_spans`` began.
    """
    source = "Stra\u00dfe \u200bX"
    stripped = fold(source, lambda ch: "" if ch == "\u200b" else ch)
    lowered = casefold_view(stripped.text)
    both = compose(stripped, lowered)

    assert both.text == "strasse x"
    assert both.source_length == len(source)
    start = both.text.index("x")
    assert both.span(start, start + 1) == (8, 9)
    assert source[slice(*both.span(start, start + 1))] == "X"
    # The sharp s expanded in the second fold and the zero-width space was
    # deleted in the first, so the composed map has to survive both directions
    # at once: "sse" is one source character plus one.
    sse = both.text.index("sse")
    assert both.span(sse, sse + 3) == (4, 6)
    assert source[slice(*both.span(sse, sse + 3))] == "\u00dfe"


def test_composing_views_that_do_not_meet_is_refused() -> None:
    """A silent wrong span is the one failure this module exists to prevent.

    Without the check the composition indexes ``first.origin`` with offsets
    computed against some other string. Where the second view is shorter that
    returns spans that are simply wrong, with nothing raised and nothing to
    notice.
    """
    first = casefold_view("abc")
    second = casefold_view("a much longer string")
    with pytest.raises(ValueError, match="cannot compose"):
        compose(first, second)
