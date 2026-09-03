"""Folded views of content, and the offsets that lead back from one.

A detector that matches over a TRANSFORMED view of the content still has to
report a span into the content the chain was given: that is the string
``saw`` hashes, the string every other detector's spans index into, and the
string the corpus labels. Nothing else in this package needs that, because
every detector so far matches the content directly.

Two folds this package will use are not length-preserving, in opposite
directions, and a naive index is wrong for both:

- casefolding expands, so the German sharp s becomes two characters and every
  index after it is shifted forward in the view;
- stripping default-ignorable code points contracts, so a marker laundered
  with a zero-width space matches in the view and its source run is longer.

A third fold does not change the length at all and is still not the identity
on offsets: the canonical ordering step inside NFD PERMUTES combining marks by
combining class, so the map it produces is the first one here that is not
non-decreasing. ``span`` below is written for that case and says what breaks
without it.

So a view carries, per character it produced, the index of the source
character that produced it. A span in the view maps back to the first source
index of its first character through the last source index of its last, which
is the run of source characters that produced the match. On a contracting fold
that run INCLUDES the deleted characters inside the match, which is the
behaviour a redaction needs: leaving the launderer's zero-width space standing
inside content reported as rewritten is the defect this module exists to
avoid.

Private by name and on purpose. Nothing here is re-exported from the package
root, and a port is free to reach the same verdicts by any other means; the
conformance document says so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Folded:
    """One transformed view of a string, with the way back."""

    text: str
    """The folded view. A match is searched for in this."""

    origin: tuple[int, ...]
    """Per character of ``text``, the index in the source that produced it.

    Always the same length as ``text``. A source character that produced
    several view characters appears several times; one that produced none does
    not appear at all, which is why the map is needed in both directions.
    """

    source_length: int
    """``len(source)``, kept so a span can be checked against the source."""

    def span(self, start: int, end: int) -> tuple[int, int]:
        """The source span covering the half-open view span ``[start, end)``.

        The end is the last matched character's source index PLUS ONE, not the
        source index of the character after the match. Those differ whenever
        the fold expanded a character: with ``origin`` reading (0, 0, 1) for a
        two-character product, a match over the first two view characters must
        come back as source (0, 1), and reading the map at ``end`` would give
        (0, 1) only by luck and would run off the end at the last character.

        MINIMUM AND MAXIMUM over the matched range, not the first and last
        entries, and the difference is a defect rather than a refinement. Every
        fold built by ``fold`` below produces a non-decreasing map, and on one
        of those the two spellings are identical. The canonical ordering step
        inside NFD is not one of those: it REORDERS combining marks by
        combining class, so a base plus a mark of class 230 plus a mark of
        class 220 comes out of ``jamjet_guardrails._unicode`` with ``origin``
        reading (0, 2, 1). Reading the first and last entries there returns
        (0, 2), which stops one character short of the run the match covered,
        and a redaction over it leaves the reordered mark standing inside
        content reported as rewritten. That is the same defect the contracting
        fold above already refuses, arriving from the other direction.

        The scan is over the matched range only, so it costs what the match is
        long rather than what the content is long.
        """
        if start >= end:
            raise ValueError(
                f"empty folded span ({start}, {end}); a zero-width span is a "
                "non-detection wearing a detection's clothes"
            )
        if start < 0 or end > len(self.text):
            raise ValueError(
                f"folded span ({start}, {end}) does not index into the "
                f"{len(self.text)} characters of this view"
            )
        covered = self.origin[start:end]
        return (min(covered), max(covered) + 1)


def fold(source: str, per_char: Callable[[str], str]) -> _Folded:
    """Build a view by replacing each character with what ``per_char`` returns.

    Per CHARACTER, deliberately. A fold defined over the whole string could
    reorder or merge across characters, and then no offset map exists at all.
    Every fold this package needs is per character: casefolding, dropping a
    class of code points, and mapping a confusable to its prototype are all
    one-character rules. A fold that needs context is a different mechanism and
    does not belong here.

    A character that produces nothing appends an empty string and extends by
    an empty list, so it contributes to neither the view nor the map without
    needing to be special-cased.
    """
    out: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(source):
        produced = per_char(char)
        out.append(produced)
        origin.extend([index] * len(produced))
    return _Folded(text="".join(out), origin=tuple(origin), source_length=len(source))


def compose(first: _Folded, second: _Folded) -> _Folded:
    """One view over the source, from a view of the source and a view of THAT.

    Folds are composable and each is opt-in per caller, which means the callers
    that take two or three of them need a way to end up holding ONE map. The
    UTS #39 skeleton is three folds deep (decompose, map through confusables,
    decompose again) and a caller that kept three maps and walked them by hand
    at every match site would be writing this function once per call site,
    which is how the two copies of the span arithmetic in ``_spans`` began.

    The composition is a lookup rather than an arithmetic: ``second.origin``
    indexes into ``first.text``, and ``first.origin`` turns each of those into
    a source index. Non-decreasing maps compose to a non-decreasing map and
    reordering ones do not, which is why ``span`` above may not assume it.

    Refuses a ``second`` that was not built over ``first.text``. Without the
    check the composition silently indexes with offsets from some other string
    and returns spans that are wrong rather than raising, and a span that is
    wrong is the one failure mode this module exists to prevent.
    """
    if second.source_length != len(first.text):
        raise ValueError(
            f"cannot compose: the second view was built over {second.source_length} "
            f"characters and the first produced {len(first.text)}"
        )
    return _Folded(
        text=second.text,
        origin=tuple(first.origin[index] for index in second.origin),
        source_length=first.source_length,
    )


def casefold_view(source: str) -> _Folded:
    """Case-insensitive view, per character.

    ``str.casefold`` on the whole string is not the same operation: it is
    defined over the string and may in principle map a sequence, so the
    per-character form is what keeps the offset map exact. The two agree on
    every input this package has measured, and this one is the one that carries
    a map.
    """
    return fold(source, str.casefold)
