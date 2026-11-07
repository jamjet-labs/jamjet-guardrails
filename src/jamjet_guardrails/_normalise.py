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
3. **NFKC, one code point at a time, then three folds a reader makes.** Every
   dash, Dash_Punctuation or U+2212 MINUS SIGN, reads as `-`; an ideographic
   full stop between two ASCII letters or digits reads as `.`; and, for the
   checks that ask for it, every combining mark is dropped after NFKD instead
   of NFKC, so an accented letter reads as its base letter and a mark stacked
   on every character of a token reads as nothing. `fold_character` says which
   checks ask and why `rules` does not.

Per CODE POINT, and the difference from string NFKC is stated rather than hidden:
canonical COMPOSITION across two code points is not performed. With marks kept,
`e` followed by U+0301 stays two code points here while string NFKC makes one;
`rules` keeps marks, and a caller's pattern that depends on composed text still
matches the content as written, which is scanned too. With marks dropped, which
is what `secrets` and `pii` read, the mark is gone and the question does not
arise. In exchange the offset map is exact and needs no reordering.

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
the same credential.

**What takes the slow path.** Content that is ASCII with no `%` in it is its own
view, and costs one `str.isascii` and one substring test. Anything else is
walked: one regex pass for runs of non-ASCII code points and percent triplets,
then each run's distinct code points folded once. That includes content that
needs nothing changed, an emoji carrying U+FE0F or a ZWJ, any `%20`, any
accented letter, so the slow path is the ordinary one for most text that is not
plain ASCII. It is linear, the map holds only edits that change length, and
the view is built once per piece of content and shared by every check in a
chain run. Content over `MAX_CHARS` is refused rather than read, by every check
that reads the view.

**What still passes.** A token split by a line break is two lines to this view,
because joining lines would also join every line of a PEM body and every
wrapped paragraph; the corpora carry the case as a labelled miss. Confusable
letters inside a credential, for the reason above.

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
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

from jamjet_guardrails.errors import GuardrailUnavailableError

# What a check hands `matched_both_ways`: a function from a string to the
# `(type, span)` pairs it claims in that string.
Scan = Callable[[str], Iterable[tuple[str, tuple[int, int]]]]

# The largest content a view is built for. Above it the checks that read the
# view refuse rather than allow: a check that cannot inspect content must not
# report it clean. Sixteen million code points is four times the largest
# context window any model this package is used with accepts, and it keeps
# every offset in the map inside a 32-bit integer.
MAX_CHARS = 16_000_000

# One pass finds both things that can change: a run of percent triplets, or a
# run of non-ASCII code points. They cannot overlap, because a triplet is ASCII,
# and finding them with one pattern hands them over in source order, which is
# the order the offset map is built in.
#
# The percent alternative is spelled with its first triplet outside the repeat
# on purpose. As `(?:%XX)+` it had no literal prefix and cost 13.7 ms on a
# megabyte with no `%` in it; led by a literal it is found by a fast scan.
_CHANGEABLE = re.compile(r"%[0-9A-Fa-f]{2}(?:%[0-9A-Fa-f]{2})*|[^\x00-\x7f]+")

# Printable ASCII, the single-byte triplets that are decoded. Space is in, so a
# card written `4111%201111%201111%201111` reads as the card it is; every control
# character is out.
_ASCII_LOW = 0x20
_ASCII_HIGH = 0x7E
# U+00A0, the first code point above the C1 controls. A multi-byte sequence that
# decodes below it is a C1 control and stays encoded, for the reason the C0
# controls do.
_FIRST_DECODED = 0xA0

# The two categories of combining mark, the ones `marks` drops.
_MARKS = ("Mn", "Me")

# U+2212 MINUS SIGN is Sm, not Pd, and reads as a hyphen in every font.
_MINUS = "\u2212"
# An ideographic full stop between two ASCII letters or digits reads as the dot
# of a host name: `corp。com`. Only there: in Japanese prose it ends a sentence,
# and `@tanaka。よろしく` must not become an address. U+FF61, the halfwidth form,
# reaches this as U+3002 through NFKC.
_IDEOGRAPHIC_STOP = re.compile("(?<=[A-Za-z0-9])\u3002(?=[A-Za-z0-9])")


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


@lru_cache(maxsize=16384)
def fold_character(character: str, marks: bool = False) -> str:
    """What one code point reads as.

    Nothing if it renders as nothing. Otherwise its compatibility form with
    every default-ignorable code point removed and every dash, Pd or U+2212,
    written `-`, because `sk‐ant‐` with U+2010 and `123–45–6789` with U+2013
    read as the hyphenated values they spell.

    With ``marks``, the form is NFKD and every combining mark, Mn or Me, is
    removed as well, so an accented letter reads as its base letter and a mark
    stacked on a character, U+0336 on every character of a token for instance,
    reads as nothing. `secrets` and `pii` ask for that: their patterns are ASCII,
    and a credential with a dot below one of its letters is the credential.
    `rules` does not, because a banned word and the same word with a diacritic
    can be two words (`año`, `ano`), and a caller's rule is about words.

    Without ``marks`` the form is NFKC, so a precomposed letter stays itself.

    Memoised because the answer depends on the code point and the flag alone.
    """
    ignorable = _ignorable()
    if character in ignorable:
        return ""
    folded: list[str] = []
    for produced in unicodedata.normalize("NFKD" if marks else "NFKC", character):
        if produced in ignorable:
            continue
        category = unicodedata.category(produced)
        if marks and category in _MARKS:
            continue
        folded.append("-" if category == "Pd" or produced == _MINUS else produced)
    return "".join(folded)


@dataclass(frozen=True, slots=True)
class Normalised:
    """The view, and the way from any span in it back to the source.

    The map holds only the edits that change LENGTH: a removed character, a
    character that expanded, a percent triplet. A character replaced by exactly
    one other, a fullwidth letter by its ASCII form for instance, moves no
    offset, so it needs no entry, and a megabyte of fullwidth text costs a
    megabyte of view and an empty map. Consecutive removals are one entry. The
    four columns are 32-bit arrays rather than objects, which is what keeps a
    megabyte with an invisible character between every two letters at a few
    megabytes of map instead of hundreds.
    """

    text: str
    source_length: int
    changed: bool
    source_starts: array[int]
    source_ends: array[int]
    view_starts: array[int]
    view_ends: array[int]

    @property
    def identity(self) -> bool:
        """Whether the view is the source, so a second scan would find nothing new."""
        return not self.changed

    def _source_of(self, index: int) -> tuple[int, int]:
        """The source run that produced the view character at ``index``.

        The LAST edit starting at or before ``index``. Several can start at one
        view offset, because a removal occupies no view characters: a removed
        zero-width space followed by an expanded ligature is a removal at view 5
        and an expansion at view 5, and the later one produced the character.
        """
        position = bisect_right(self.view_starts, index) - 1
        if position < 0:
            return index, index + 1
        if index < self.view_ends[position]:
            return self.source_starts[position], self.source_ends[position]
        source = self.source_ends[position] + (index - self.view_ends[position])
        return source, source + 1

    def offset(self, index: int) -> int:
        """The source offset a zero-width position in the view stands for."""
        if index >= len(self.text):
            return self.source_length
        return self._source_of(index)[0]

    def span(self, start: int, end: int) -> tuple[int, int]:
        """The source span covering the half-open view span ``[start, end)``.

        From the first source character of the first view character through the
        last source character of the last one. A removed character INSIDE the
        match is therefore inside the span, which is what a redaction needs, and
        one just outside the match is outside it. A view character produced by
        several source characters, a percent triplet for instance, maps to all of
        them, so a span never cuts `%5F` into `%5` and a stray `F`.
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


def _decoded_at(raw: bytes, index: int, marks: bool) -> tuple[int, str] | None:
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
    return width, fold_character(text, marks)


def _percent_edits(start: int, run: str, marks: bool) -> Iterable[tuple[int, int, str]]:
    """The triplets in one run that decode safely, each with what it reads as."""
    raw = bytes.fromhex(run.replace("%", ""))
    index = 0
    while index < len(raw):
        decoded = _decoded_at(raw, index, marks)
        if decoded is None:
            index += 1
            continue
        width, reading = decoded
        yield start + 3 * index, start + 3 * (index + width), reading
        index += width


# Runs at most this long are folded one character at a time.
_SHORT_RUN = 16


@lru_cache(maxsize=256)
def _class_of(characters: frozenset[str]) -> re.Pattern[str]:
    """One character class over a set of code points, compiled once per set."""
    return re.compile("[" + "".join(re.escape(c) for c in sorted(characters)) + "]")


@lru_cache(maxsize=4)
def normalise(content: str, marks: bool = False) -> Normalised:
    """The view of ``content`` the module docstring describes, with its map.

    Memoised on the content, so the checks in one chain run, which are all
    handed the same string, build its view once between them. Up to four views
    and the strings they were built from are kept alive by that cache.

    Raises:
        GuardrailUnavailableError: for content over ``MAX_CHARS``.
    """
    if len(content) > MAX_CHARS:
        raise GuardrailUnavailableError(
            f"content of {len(content)} characters is over the {MAX_CHARS}-character "
            "limit this package builds a normalised view for; a check that cannot "
            "read it refuses rather than reporting it clean. Split the content"
        )
    empty = array("i")
    if content.isascii() and "%" not in content:
        return Normalised(content, len(content), False, empty, empty, empty, empty)

    pieces: list[str] = []
    cursor = 0
    delta = 0
    source_starts, source_ends = array("i"), array("i")
    view_starts, view_ends = array("i"), array("i")

    def record(start: int, end: int, reading: str) -> None:
        nonlocal delta
        view_start = start + delta
        if (
            not reading
            and source_ends
            and source_ends[-1] == start
            and view_starts[-1] == view_ends[-1] == view_start
        ):
            source_ends[-1] = end
        else:
            source_starts.append(start)
            source_ends.append(end)
            view_starts.append(view_start)
            view_ends.append(view_start + len(reading))
        delta += len(reading) - (end - start)

    for match in _CHANGEABLE.finditer(content):
        run = match.group()
        if run[0] == "%":
            for start, end, reading in _percent_edits(match.start(), run, marks):
                pieces.append(content[cursor:start])
                pieces.append(reading)
                cursor = end
                record(start, end, reading)
            continue
        if len(run) == 1:
            reading = fold_character(run, marks)
            if reading == run:
                continue
            start = match.start()
            pieces.append(content[cursor:start])
            pieces.append(reading)
            cursor = start + 1
            if len(reading) != 1:
                record(start, cursor, reading)
            continue
        if len(run) <= _SHORT_RUN:
            # A short run, which is what a stray invisible character between two
            # letters is, folded character by character: building a translation
            # table and a pattern per run cost more than the run, and an
            # alternating `a` and U+200B is half a million one-character runs.
            readings = [fold_character(character, marks) for character in run]
            if all(reading == character for reading, character in zip(readings, run)):
                continue
            pieces.append(content[cursor : match.start()])
            pieces.append("".join(readings))
            cursor = match.end()
            for offset, reading in enumerate(readings, match.start()):
                if len(reading) != 1:
                    record(offset, offset + 1, reading)
            continue
        folds = {character: fold_character(character, marks) for character in set(run)}
        table = {ord(c): f for c, f in folds.items() if f != c}
        if not table:
            continue
        pieces.append(content[cursor : match.start()])
        pieces.append(run.translate(table))
        cursor = match.end()
        resized = frozenset(c for c, f in folds.items() if len(f) != 1)
        if resized:
            for found in _class_of(resized).finditer(run):
                start = match.start() + found.start()
                record(start, start + 1, folds[found.group()])
    if not pieces:
        text = content
    else:
        pieces.append(content[cursor:])
        text = "".join(pieces)
    if "\u3002" in text:
        text = _IDEOGRAPHIC_STOP.sub(".", text)
    if text == content:
        return Normalised(content, len(content), False, empty, empty, empty, empty)
    return Normalised(text, len(content), True, source_starts, source_ends, view_starts, view_ends)


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


def matched_both_ways(
    content: str, scan: Scan, marks: bool = False
) -> list[tuple[str, tuple[int, int]]]:
    """Every span ``scan`` claims over the content AND over its normalised view.

    ``scan`` maps a string to ``(type, span)`` pairs. It runs over the raw
    content, and, when normalising changed anything, over the view with each
    span mapped back to the content. A zero-width match, which only a caller's
    own pattern can produce, maps to the zero-width position it stands for
    rather than raising: it was reported as it stood before this module existed,
    and the chain is what refuses it.

    The raw findings are kept exactly as the scan reported them, so nothing a
    check said about content before this module existed changes. A view finding
    is added only when no raw finding of the same type already contains it, and
    a raw finding is then dropped only when an ADDED view finding of its type
    strictly contains it: a token the raw scan matched up to a zero-width space
    and the view matched whole is one finding over the whole token, not two.

    Sorted by span, which is the precondition `_spans._merge` states.
    """
    raw = list(scan(content))
    view = normalise(content, marks)
    if view.identity:
        return sorted(raw, key=lambda pair: pair[1])

    def mapped(span: tuple[int, int]) -> tuple[int, int]:
        start, end = span
        if start == end:
            position = view.offset(start)
            return position, position
        start, end = view.span(start, end)
        # A combining mark belongs to the character in front of it, so a match
        # that ends on a marked character takes its marks with it: the last
        # letter of a token struck through with U+0336 is redacted whole, and no
        # stray mark is left to sit on the placeholder.
        while marks and end < len(content) and unicodedata.category(content[end]) in _MARKS:
            end += 1
        return start, end

    found = dict.fromkeys((type_name, mapped(span)) for type_name, span in scan(view.text))

    raw_by_type: dict[str, list[tuple[int, int]]] = {}
    for type_name, span in raw:
        raw_by_type.setdefault(type_name, []).append(span)
    raw_contains = {name: _containment(spans) for name, spans in raw_by_type.items()}
    added = [
        (type_name, span)
        for type_name, span in found
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
