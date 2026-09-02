"""Folded views, and the offsets that map a match back to what produced it.

A check that matches over a transformed view of the content has to report a
span into the ORIGINAL content, because that is what the chain rewrites and
what a corpus labels. The transformation is not length-preserving in either
direction: casefolding the German sharp s produces two characters from one,
and stripping a zero-width space produces none from one. Both directions are
here, because a mapping that is right for one is silently wrong for the other.
"""

from __future__ import annotations

import pytest

from jamjet_guardrails._fold import casefold_view, fold


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
