"""Build a pattern-shaped check without writing a detector.

This module is two things wearing one implementation. A user configuring the
``rules`` check at runtime instantiates it with their own patterns; a
contributor adding a check to this repository instantiates it with a corpus
beside it. Building those separately would produce two engines that disagree
about spans, about merging and about what a configuration that selects nothing
means, and the second one would not be corpus-gated.

What it owns: collecting spans, handing them to the shared merge, constructing
the ``Verdict``, and refusing at construction every configuration that would
check less than it was asked to. What it does not own: deciding what is worth
matching. That is the caller's, and for a registered check it is the corpus's.

A check built here is a ``constraint``. Its findings carry no confidence,
because a pattern matches or it does not.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

# `re._parser` on 3.11 and later; `sre_parse` on 3.10, where `re._parser` does
# not exist at all. Measured: on 3.10.20 the first import raises
# ModuleNotFoundError("No module named 're._parser'; 're' is not a package"),
# and `sre_parse` emits a DeprecationWarning from 3.10 onward, which is
# suppressed here rather than shown to a caller who did not ask for it. Both
# modules produce identical parse trees for every pattern in the recorded
# table in tests/test_authoring.py, checked on 3.10.20 and 3.14.5.
#
# A private standard-library module, deliberately, and the alternative was
# worse: the only other way to see a pattern's STRUCTURE is to re-parse the
# regex syntax by hand, and a second regex parser in a security library is a
# larger liability than a pinned dependence on the one that is already there.
# The guard degrades honestly if this ever breaks: `_nests_unbounded_repeats`
# raising is a construction-time failure naming the pattern, not a check that
# silently stops guarding.
try:
    from re import _parser as _regex_parser  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - exercised on 3.10 only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import sre_parse as _regex_parser

_MAXREPEAT = _regex_parser.MAXREPEAT

_LINE = re.compile("\n")


@dataclass(frozen=True, slots=True)
class Limits:
    """Size ceilings, in units this package can actually count.

    Characters, bytes and lines. NOT tokens: a token count needs a tokenizer,
    this package carries none and will not estimate one, and a limit that is
    approximately a token limit is a limit nobody can reason about. A caller
    who needs a token ceiling derives a character ceiling from a ratio they
    measured on their own traffic, which is a number they can check.
    """

    max_chars: int | None = None
    max_bytes: int | None = None
    max_lines: int | None = None

    def __post_init__(self) -> None:
        values = (self.max_chars, self.max_bytes, self.max_lines)
        if all(value is None for value in values):
            raise ValueError(
                "Limits selects nothing; pass at least one of max_chars, max_bytes or max_lines"
            )
        for name, value in zip(("max_chars", "max_bytes", "max_lines"), values):
            if value is not None and value < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {value}; a limit of zero denies "
                    "every input including the empty one"
                )


def _limit_spans(content: str, limits: Limits) -> list[tuple[str, tuple[int, int]]]:
    """At most one finding, spanning from the first excess character to the end.

    ONE finding rather than one per breached limit. Three limits breached is
    one fact about one piece of content; three overlapping spans would collapse
    into one placeholder anyway and leave an audit record claiming three
    detections. The earliest breach wins, so the span covers everything any
    limit objects to.

    The span is never empty: it opens at an index strictly inside the content,
    because each branch below fires only when the content is longer than the
    limit it tests.
    """
    starts: list[int] = []

    if limits.max_chars is not None and len(content) > limits.max_chars:
        starts.append(limits.max_chars)

    if limits.max_bytes is not None and len(content.encode("utf-8")) > limits.max_bytes:
        # Walked rather than sliced. `content.encode()[:n].decode()` truncates
        # mid-character and raises, and a slice of the string is a character
        # count, which is the thing this branch exists not to be.
        total = 0
        for index, char in enumerate(content):
            total += len(char.encode("utf-8"))
            if total > limits.max_bytes:
                starts.append(index)
                break

    if limits.max_lines is not None:
        # LINE STARTS, not newline count. "a\nb\n" is two lines and carries two
        # newlines, so counting newlines reports a third line beginning at the
        # end of the content, whose span would be empty and which the chain
        # refuses. A line start at len(content) is the end of the content and
        # opens nothing.
        starts_of_lines = [0] + [match.end() for match in _LINE.finditer(content)]
        if len(starts_of_lines) > limits.max_lines and starts_of_lines[limits.max_lines] < len(
            content
        ):
            starts.append(starts_of_lines[limits.max_lines])

    if not starts:
        return []
    return [("LENGTH_LIMIT", (min(starts), len(content)))]


def _nests_unbounded_repeats(pattern: str | re.Pattern[str]) -> bool:
    """Whether an unbounded repeat encloses another unbounded repeat.

    WHAT THIS IS AND IS NOT, because the difference decides how it may be
    described in documentation. It catches the textbook nested-quantifier
    shape: ``(a+)+``, ``(a*)*``, ``(?:\\w+\\s?)*``, where the number of ways to
    split a failing input across the two repeats grows exponentially with the
    input. It is NOT a proof that a pattern cannot backtrack catastrophically.
    ``(a|aa)+`` is exponential and passes this guard, and a test pins that so
    the sentence stays true.

    It is not a timeout either, and cannot be: pure-Python ``re`` offers no way
    to bound a match's running time, so a caller's own pattern is a caller's
    own risk. Saying that plainly is the honest position. Saying "ReDoS
    protected" on the strength of this walk would be the claim this project
    does not make.

    The walk descends into groups, branches and lookarounds, carrying whether
    it is already inside an unbounded repeat. A repeat is unbounded when its
    maximum is MAXREPEAT, which is what ``+``, ``*`` and ``{n,}`` all parse to;
    ``{1,3}`` is bounded and nests freely.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    return _walk_for_nesting(_regex_parser.parse(compiled.pattern, compiled.flags), False)


def _walk_for_nesting(sequence: Any, inside_unbounded: bool) -> bool:
    """Depth-first over one parsed subpattern.

    The opcodes are compared by NAME rather than imported as constants,
    because the constants live in the same private module the import above
    already leans on and the names are stable across every version in the
    matrix. An opcode this walk does not know is descended into where it
    carries a subpattern and otherwise ignored, which is the safe direction:
    an unknown opcode makes the guard miss a nesting, never invent one.
    """
    for opcode, argument in sequence:
        name = str(opcode)
        if name in ("MAX_REPEAT", "MIN_REPEAT"):
            unbounded = argument[1] is _MAXREPEAT
            if unbounded and inside_unbounded:
                return True
            if _walk_for_nesting(argument[2], inside_unbounded or unbounded):
                return True
        elif name == "SUBPATTERN":
            if _walk_for_nesting(argument[3], inside_unbounded):
                return True
        elif name == "BRANCH":
            for alternative in argument[1]:
                if _walk_for_nesting(alternative, inside_unbounded):
                    return True
        elif name in ("ASSERT", "ASSERT_NOT"):
            if _walk_for_nesting(argument[1], inside_unbounded):
                return True
        elif name == "ATOMIC_GROUP":
            if _walk_for_nesting(argument, inside_unbounded):
                return True
    return False
