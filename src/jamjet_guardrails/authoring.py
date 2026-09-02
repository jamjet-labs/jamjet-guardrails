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
from dataclasses import dataclass

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
