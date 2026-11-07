"""What a reader sees, for the checks that match on it, and the way back.

One U+200B inside a GitHub token, or between two groups of a card number, used
to get `allow` from every check in the output chain. `secrets` and `pii` matched
the bytes, the bytes no longer spelled the credential, and `injection-structural`
lets up to four unexplained invisible characters through by design. A fullwidth
`ＡＫＩＡ`, a `%5F` for the underscore of `ghp_`, and `alice%40corp.com` all did
the same. Every one of those reads as the credential to a person, to a model,
and to whatever decodes the URL, so a detector that matches the bytes and not
the reading is answering a question nobody asked.

So the checks that match text match it over a NORMALISED VIEW, and report spans
into the original content, which is what the chain rewrites, what `saw` hashes
and what a corpus labels. Three steps, in this order, each one defined by a
published property rather than by a list somebody keeps:

1. **Percent-decoding, one level, where it is safe.** A `%XX` triplet becomes the
   character it encodes when that is printable ASCII (space through tilde), and a
   run of triplets becomes one character when it is a well-formed UTF-8
   encoding of a code point at or above U+00A0. Control characters, C0 and C1,
   stay encoded: `%0A` is a line break to a PEM walker and `%02d` is a format
   string, and decoding either changes a structure rather than revealing a
   reading. ONE level: the decoded text is never decoded again, so `%2541` reads
   `%41` and not `A`. That is the rule `_decode` keeps for the same reason.
2. **Default-ignorable code points are removed.** Default_Ignorable_Code_Point is
   Unicode's own name for "a conforming renderer draws this as nothing", and it
   is the table `injection-structural` already counts from, so the two cannot
   disagree about what is invisible. It covers the zero-width characters, the
   bidi embedding, override and isolate controls, the directional marks, the
   word joiner and invisible operators, the variation selectors, the Hangul
   fillers, the soft hyphen and the tag block.
3. **NFKC, one code point at a time.** Compatibility decomposition is what folds
   fullwidth and mathematical letters and digits, ideographic and no-break
   spaces, and the fullwidth `＠` back to what they are compatible with.

Per CODE POINT, and the difference from string NFKC is stated rather than hidden:
canonical COMPOSITION across two code points is not performed, so `e` followed
by U+0301 stays two code points here while string NFKC makes one. Every built-in
pattern this view feeds matches ASCII or `\\w` plus combining marks, which read
the two spellings alike; a caller's rule that depends on composed text still
matches the content as written, which is scanned too. In exchange the offset map
is exact and needs no reordering.

**Confusable folding is not a step here, and the reason is the definition.** The
UTS #39 skeleton maps ASCII to ASCII as well: `I` and `1` fold to `l`, `0` to
`O`, `m` to `rn`. A regex written against ASCII credentials stops matching its
own text over a skeleton, which is why `PatternGuardrail` keeps it behind
`fold_confusables` and applies it only to banned substrings, where the needle is
folded the same way. That option builds its skeleton over this view as well as
over the content.

**Normalising never takes a finding away.** Each caller scans the raw content
as well as this view (`matched_both_ways`), so a pattern that matches the bytes
as written, an invisible character a rule names on purpose for instance, still
matches. A raw finding is replaced only by a longer one of its own type over
the same credential. When nothing in the content needs normalising the view IS
the content and the second scan does not happen, which is the common case:
ASCII without a `%` costs one `str.isascii` and one substring test.

**`injection-structural` does not read this view and must not.** Its job is to
report the very characters step 2 removes. What it shares is the table, so one
definition of "renders as nothing" serves both: nothing removed here is
invisible to that check.

Private by name. Nothing here is re-exported, and a port may reach the same
verdicts by other means; `docs/conformance.md` states what a port must see.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

# What a check hands `matched_both_ways`: a function from a string to the
# `(type, span)` pairs it claims in that string.
Scan = Callable[[str], Iterable[tuple[str, tuple[int, int]]]]

# A maximal run of percent triplets, and a maximal run of non-ASCII code points.
# Neither can overlap the other, because a triplet is ASCII, so the edits the two
# produce never contend for one source character.
#
# The run is spelled with its first triplet outside the repeat on purpose. As
# `(?:%XX)+` the pattern has no literal prefix, so the engine tried it at every
# position and cost 13.7 ms on a megabyte with no `%` in it; led by a literal
# `%` it is found by a fast scan for that character, 0.28 ms on the same input.
_PERCENT_RUN = re.compile(r"%[0-9A-Fa-f]{2}(?:%[0-9A-Fa-f]{2})*")
_NON_ASCII_RUN = re.compile(r"[^\x00-\x7f]+")

# Printable ASCII, the single-byte triplets that are decoded. Space is in, so a
# card written `4111%201111%201111%201111` reads as the card it is; every control
# character is out.
_ASCII_LOW = 0x20
_ASCII_HIGH = 0x7E
# U+00A0, the first code point above the C1 controls. A multi-byte sequence that
# decodes below it is a C1 control and stays encoded, for the reason the C0
# controls do.
_FIRST_DECODED = 0xA0


@lru_cache(maxsize=1)
def _ignorable() -> frozenset[str]:
    """Every Default_Ignorable_Code_Point, from the table `injection-structural` owns.

    Imported here rather than at module import because the detectors package
    imports this module, and `injection_structural` lives inside that package:
    a top-level import would make `authoring`, which `rules` imports, reach the
    detectors package while it is still being built. After the first call it is a
    cache hit.
    """
    from jamjet_guardrails.detectors.injection_structural import _DEFAULT_IGNORABLE

    return frozenset(
        chr(point) for low, high in _DEFAULT_IGNORABLE for point in range(low, high + 1)
    )


@lru_cache(maxsize=1)
def _ignorable_pattern() -> re.Pattern[str]:
    """One character class over the same table, for the fast path's test."""
    from jamjet_guardrails.detectors.injection_structural import _DEFAULT_IGNORABLE

    ranges = "".join(
        f"{re.escape(chr(low))}-{re.escape(chr(high))}" for low, high in _DEFAULT_IGNORABLE
    )
    return re.compile(f"[{ranges}]")


@lru_cache(maxsize=4096)
def fold_character(character: str) -> str:
    """What one code point reads as: nothing if it renders as nothing, else its NFKC.

    The NFKC result is filtered as well, and today that filter changes nothing:
    swept over the whole code space, no code point outside the table has an NFKC
    that contains one inside it (U+FFA0 maps to U+1160, and both are in the
    table, so the first test already returns). It stays because the table and
    the interpreter's normalisation data are versioned separately, and a mapping
    added later would otherwise put an invisible character back into the view.
    `tests/test_normalise.py` holds the sweep.

    Memoised because the answer depends on the code point alone and prose in any
    one script asks about the same few hundred of them.
    """
    ignorable = _ignorable()
    if character in ignorable:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKC", character) if c not in ignorable)


@dataclass(frozen=True, slots=True)
class _Edit:
    """One stretch of the source that the view spells differently."""

    source_start: int
    source_end: int
    view_start: int
    view_end: int


@dataclass(frozen=True, slots=True)
class Normalised:
    """The view, and the way from any span in it back to the source.

    Stored as the EDITS rather than as an index per character. Between two edits
    the view and the source agree character for character, offset by a constant,
    so a map needs one entry per change rather than one per character, and a
    megabyte of ordinary text costs a megabyte of view and nothing else.
    """

    text: str
    source_length: int
    edits: tuple[_Edit, ...]
    _view_starts: tuple[int, ...]

    @property
    def identity(self) -> bool:
        """Whether the view is the source, so a second scan would find nothing new."""
        return not self.edits

    def _source_of(self, index: int) -> tuple[int, int]:
        """The source run that produced the view character at ``index``.

        The LAST edit starting at or before ``index``. Several edits can start at
        one view offset, because a deletion occupies no view characters: a
        zero-width space followed by a fullwidth letter is a deletion at view 5
        and a replacement at view 5. `bisect_right` picks the later one, and the
        later one is the one that produced the character there.
        """
        position = bisect_right(self._view_starts, index) - 1
        if position < 0:
            return index, index + 1
        edit = self.edits[position]
        if index < edit.view_end:
            return edit.source_start, edit.source_end
        source = edit.source_end + (index - edit.view_end)
        return source, source + 1

    def span(self, start: int, end: int) -> tuple[int, int]:
        """The source span covering the half-open view span ``[start, end)``.

        From the first source character of the first view character through the
        last source character of the last one. A deleted character INSIDE the
        match therefore lies inside the span, which is what a redaction needs: a
        token reported as rewritten must not leave the launderer's zero-width
        space standing in the middle of the placeholder's neighbours. A deleted
        character just outside the match is outside the span.

        A view character produced by several source characters, a percent
        triplet or a multi-byte sequence of them, maps to all of them, so a span
        never cuts `%5F` into `%5` and a stray `F`.
        """
        if start >= end:
            raise ValueError(
                f"empty normalised span ({start}, {end}); a zero-width span is a "
                "non-detection wearing a detection's clothes"
            )
        if start < 0 or end > len(self.text):
            raise ValueError(
                f"normalised span ({start}, {end}) does not index into the "
                f"{len(self.text)} characters of this view"
            )
        return self._source_of(start)[0], self._source_of(end - 1)[1]


def _decoded_at(raw: bytes, index: int) -> tuple[int, str] | None:
    """The width in bytes and the reading of the sequence starting at ``index``.

    ``None`` where it stays encoded: a control character, a lone continuation
    byte, a truncated or overlong sequence, or a C1 control.
    """
    byte = raw[index]
    if _ASCII_LOW <= byte <= _ASCII_HIGH:
        return 1, chr(byte)
    if not 0xC2 <= byte <= 0xF4:
        return None
    width = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
    try:
        text = raw[index : index + width].decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(text) != 1 or ord(text) < _FIRST_DECODED:
        return None
    return width, fold_character(text)


def _percent_edits(content: str, run: re.Match[str]) -> list[tuple[int, int, str]]:
    """The triplets in one run that decode safely, each with what it reads as."""
    raw = bytes.fromhex(run.group().replace("%", ""))
    base = run.start()
    edits: list[tuple[int, int, str]] = []
    index = 0
    while index < len(raw):
        decoded = _decoded_at(raw, index)
        if decoded is None:
            index += 1
            continue
        width, reading = decoded
        edits.append((base + 3 * index, base + 3 * (index + width), reading))
        index += width
    return edits


def normalise(content: str) -> Normalised:
    """The view of ``content`` the module docstring describes, with its map."""
    if content.isascii() and "%" not in content:
        return Normalised(content, len(content), (), ())

    changes: list[tuple[int, int, str]] = []
    for run in _PERCENT_RUN.finditer(content):
        changes += _percent_edits(content, run)
    ignorable = _ignorable_pattern()
    for run in _NON_ASCII_RUN.finditer(content):
        text = run.group()
        # Whole-run NFKC stability implies per-code-point stability: a code point
        # whose own NFKC differs is changed by string NFKC too, because
        # recomposition never produces a compatibility character, a singleton or
        # a composition exclusion. So a run that passes both tests has nothing in
        # it to fold, and prose in any script that is already NFKC is skipped at
        # the speed of the two C calls.
        if unicodedata.is_normalized("NFKC", text) and ignorable.search(text) is None:
            continue
        for offset, character in enumerate(text, run.start()):
            folded = fold_character(character)
            if folded != character:
                changes.append((offset, offset + 1, folded))
    if not changes:
        return Normalised(content, len(content), (), ())
    changes.sort()

    pieces: list[str] = []
    edits: list[_Edit] = []
    cursor = 0
    view = 0
    for source_start, source_end, replacement in changes:
        pieces.append(content[cursor:source_start])
        view += source_start - cursor
        pieces.append(replacement)
        edits.append(_Edit(source_start, source_end, view, view + len(replacement)))
        view += len(replacement)
        cursor = source_end
    pieces.append(content[cursor:])
    return Normalised(
        text="".join(pieces),
        source_length=len(content),
        edits=tuple(edits),
        _view_starts=tuple(edit.view_start for edit in edits),
    )


def _containment(spans: list[tuple[int, int]]) -> Callable[[tuple[int, int]], bool]:
    """A test for "some span in ``spans`` contains this one", answered by bisect.

    Sorted by start with the running furthest end beside it, so the question is
    one bisect rather than a scan of every span: a container exists exactly when
    one of the spans starting at or before this one ends at or after it. The
    linear form of this test is the quadratic that `url-exfiltration` already
    had to remove once.
    """
    ordered = sorted(spans)
    starts = [start for start, _ in ordered]
    furthest: list[int] = []
    reach = -1
    for _, end in ordered:
        reach = max(reach, end)
        furthest.append(reach)

    def contains(span: tuple[int, int]) -> bool:
        prefix = bisect_right(starts, span[0])
        return bool(prefix) and furthest[prefix - 1] >= span[1]

    return contains


def matched_both_ways(content: str, scan: Scan) -> list[tuple[str, tuple[int, int]]]:
    """Every span ``scan`` claims over the content AND over its normalised view.

    ``scan`` maps a string to ``(type, span)`` pairs. It runs over the raw
    content, and, when normalising changed anything, over the view with each
    span mapped back to the content.

    The raw findings are kept exactly as the scan reported them, so nothing a
    check said about content before this module existed changes. A view finding
    is added only when no raw finding of the same type already contains it, and
    a raw finding is then dropped only when an ADDED view finding of its type
    strictly contains it: a token the raw scan matched up to a zero-width space
    and the view matched whole is one finding over the whole token, not two, so
    the audit record does not claim two detections for one credential.

    Sorted by span, which is the precondition `_spans._merge` states.
    """
    raw = list(scan(content))
    view = normalise(content)
    if view.identity:
        return sorted(raw, key=lambda pair: pair[1])
    mapped = dict.fromkeys((type_name, view.span(*span)) for type_name, span in scan(view.text))

    raw_by_type: dict[str, list[tuple[int, int]]] = {}
    for type_name, span in raw:
        raw_by_type.setdefault(type_name, []).append(span)
    raw_contains = {name: _containment(spans) for name, spans in raw_by_type.items()}
    added = [
        (type_name, span)
        for type_name, span in mapped
        if type_name not in raw_contains or not raw_contains[type_name](span)
    ]

    added_by_type: dict[str, list[tuple[int, int]]] = {}
    for type_name, span in added:
        added_by_type.setdefault(type_name, []).append(span)
    added_contains = {name: _containment(spans) for name, spans in added_by_type.items()}
    kept = [
        (type_name, span)
        for type_name, span in raw
        if type_name not in added_contains or not added_contains[type_name](span)
    ]
    return sorted(kept + added, key=lambda pair: pair[1])
