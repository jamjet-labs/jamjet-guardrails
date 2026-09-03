"""The three questions three checks ask of Unicode, answered from vendored tables.

`script_set` resolves a code point's scripts per UTS #39 section 5.1,
`skeleton` builds the confusable skeleton of a string per UTS #39 section 4,
and `identifier_allowed` answers whether a code point is in the UTS #39
Identifier Profile. `confusables`, `script-constraint` and the
`fold_confusables` option on `jamjet_guardrails.authoring.PatternGuardrail` are
the callers, and they share these functions rather than each deriving what they
need, because three derivations of "what script is this" would disagree the
first time one of them was tightened.

**Why `identifier_allowed` is here rather than inside the check that reads it.**
It is the third leg of one question, not a fourth question: a confusable
prototype is evidence of a spoof only when it names a string somebody could be
reading, and Identifier_Status is Unicode's own answer to which characters
strings are written in. Measured on the 16.0.0 tables, 140 of the 296 Cyrillic
letters have a prototype written wholly in Latin and only 104 of those have one
written wholly in the Identifier Profile. The gap is what separates Cyrillic
small a, whose prototype is Latin `a`, from Cyrillic small em, whose prototype
is U+028D LATIN SMALL LETTER TURNED W: both are Latin, and only the first is a
character a brand or a hostname is written in. Without the third table
`iPhoneом` in Russian prose is a mixed-script confusable and `почта.рф` is a
whole-script one, because Cyrillic er maps to `p` and Cyrillic ef to U+0278.
Denying the Russian ccTLD is not a check, it is a check that gets switched off.

**Why the tables are vendored and these questions are not asked of
`unicodedata`.** No interpreter from 3.10 to 3.14 exposes the Script property,
the confusables table or Identifier_Status at all, and the Unicode version
behind `unicodedata` runs from 13.0 to 16.0 across this project's CI matrix. The
tables under `jamjet_guardrails._unicode` are 16.0.0 on every interpreter, so a
corpus label means the same thing on every leg. Category, name and
Default_Ignorable questions stay with `unicodedata`, as `injection-structural`
already asks them.

**Normalisation is the exception, and it is deliberate.** `skeleton` calls
`unicodedata.normalize` and `unicodedata.combining` rather than carrying
decomposition data of its own. The Unicode Normalization Stability Policy is
what makes that safe: once a code point is assigned, its canonical
decomposition never changes, so the two NFD passes agree across 13.0 and 16.0
for every code point 13.0 knows about. What differs is a code point assigned
after the interpreter's version, which decomposes to itself there. That is a
smaller and better-behaved gap than carrying a fourth vendored table would be,
and it is disclosed rather than assumed.

**Cost, measured rather than estimated.** `scripts.py` is 45 KiB of source and
imports in 1.5 ms warm and 7.5 ms cold; `confusables.py` is 176 KiB and imports
in 2.5 ms warm and 21 ms cold; `identifiers.py` is 10 KiB and imports in 0.1 ms
warm and 1.4 ms cold, measured after `jamjet_guardrails._unicode` was already
imported so that figure is the table alone rather than `scripts.py` again.
Warm means the `__pycache__` byte code is
present, which is every run after the first and, because `pip install` compiles
what it installs, the ordinary case in a deployment too. Cold means it is
absent and the source is compiled. Medians of nine runs on Python 3.14.5, macOS
arm64, timed inside a child that had already imported `jamjet_guardrails`, so
each figure is the marginal cost of the table and not of the package:

    ./.venv/bin/python -c "import jamjet_guardrails, time; \\
        s = time.perf_counter(); import jamjet_guardrails._unicode.confusables; \\
        print((time.perf_counter() - s) * 1000)"

**A cheaper encoding of the confusables table exists and was not taken.** The
same 6,355 rows written as one row per line and split at import are 161 KiB,
9.0 ms cold and 1.1 ms warm; written as a single string literal split twice at
import they are 124 KiB, 3.0 ms cold and 1.1 ms warm. Both were built and
measured before this one was kept. What the dictionary literal buys for its
1.4 ms warm is that the table IS the table: a reviewer diffing it against
`unicode-data/16.0.0/confusables.txt` reads one row per line with no decoding
step in between, and a security library's vendored data is a thing people have
to be able to read. The blob costs that and saves a millisecond on a table
imported once per process, lazily, by one check.

`confusables.py` is imported inside `skeleton` and `identifiers.py` inside
`identifier_allowed`, not at the top of this module, so `script-constraint` pays
for `scripts.py` alone. The check modules in turn import THIS module inside the
methods that need it, so `import jamjet_guardrails` pays none of it, and
`tests/test_unicode.py::test_importing_the_package_loads_no_unicode_table`
holds that.

There is no CI timing gate on any of these numbers, per the performance posture
the phase 3 design fixes: byte-identical artifacts and wall-clock measurements
do not mix. The source SIZES are guarded, because they are a property of files
a test already reproduces byte for byte.

**Complexity.** `script_set` is two bisections over tables of 979 and 174
ranges and allocates nothing, so it is constant time in the content and safe to
call per character; `identifier_allowed` is one bisection over 391 ranges and is
the same. `skeleton` is linear in the length of the content, plus a sort of each
run of combining marks, which is bounded by how many marks sit on one base
character and is therefore near-linear overall.
"""

from __future__ import annotations

import unicodedata
from bisect import bisect_right
from functools import lru_cache

from jamjet_guardrails._fold import _Folded, compose, fold
from jamjet_guardrails._unicode.scripts import (
    EXTENSION_RANGES,
    EXTENSION_SETS,
    SCRIPT_RANGES,
    UNICODE_VERSION,
)

__all__ = ["UNICODE_VERSION", "UNKNOWN", "identifier_allowed", "script_set", "skeleton"]

# The Script value of every code point the pinned tables do not assign, which is
# `Scripts.txt`'s own @missing rule. Named rather than spelled at each use, so a
# caller comparing against it cannot drift from what this module returns.
UNKNOWN = "Unknown"

# Bisection keys, derived once. Kept beside the tables rather than generated
# into them: a second sorted tuple in the generated module would be a second
# thing a hand edit could put out of step with the first, and building it here
# costs one pass over 1,153 integers at import.
_SCRIPT_STARTS = tuple(start for start, _, _ in SCRIPT_RANGES)
_EXTENSION_STARTS = tuple(start for start, _, _ in EXTENSION_RANGES)

# One shared frozenset per script name and per extension set, built once.
#
# `script_set` is called per CHARACTER by two of its three callers, so a version
# that built `frozenset({script})` on each call would allocate once per
# character of every input. Interning also means a caller may compare the
# returned sets by identity, though nothing does and nothing should.
_SINGLETON: dict[str, frozenset[str]] = {
    script: frozenset({script}) for _, _, script in SCRIPT_RANGES
}
_SINGLETON[UNKNOWN] = frozenset({UNKNOWN})

_RESOLVED_SETS: tuple[frozenset[str], ...] = tuple(
    frozenset(entry.split()) for entry in EXTENSION_SETS
)


def script_set(character: str) -> frozenset[str]:
    """The resolved script set of one code point, per UTS #39 section 5.1.

    Script_Extensions where the code point has them, otherwise a single-element
    set holding its Script value, otherwise `Unknown`. Names are the long
    property values (`Latin`, `Cyrillic`, `Common`, `Inherited`) throughout,
    never the four-letter codes `ScriptExtensions.txt` is written in: an
    `allowed_scripts` option and a corpus label are both read by people, and two
    spellings in circulation is a constraint that matches nothing while
    reporting nothing.

    `Common` and `Inherited` come back AS THEMSELVES rather than as an empty set
    or as every script. They are wildcards to a caller and this function is not
    the place that decides what a wildcard does: `script-constraint` passes them
    under every constraint, and the `confusables` token scan drops them from the
    intersection. Returning an empty set here would make those two
    indistinguishable from a code point with no scripts at all, which is a
    different fact.

    Refuses a string that is not exactly one code point. The alternative is
    silently reading the first character of a longer one, and a caller that
    passed a two-character string meant something by it.
    """
    if len(character) != 1:
        raise ValueError(
            f"script_set takes one code point, got {len(character)} characters; "
            "a longer string has a script per character and no single answer"
        )
    code = ord(character)

    # Script_Extensions FIRST and Script only where there are none. That is the
    # order UTS #39 section 5.1 states, and reversing it is not a slower answer
    # but a different one: U+00B7 has Script=Common and extensions naming 16
    # scripts, so a Script-first lookup would report a middle dot as a wildcard
    # and a `{"Latin"}` constraint would pass Georgian and Glagolitic text
    # carrying one.
    position = bisect_right(_EXTENSION_STARTS, code) - 1
    if position >= 0:
        _, end, index = EXTENSION_RANGES[position]
        if code <= end:
            return _RESOLVED_SETS[index]

    position = bisect_right(_SCRIPT_STARTS, code) - 1
    if position >= 0:
        _, end, script = SCRIPT_RANGES[position]
        if code <= end:
            return _SINGLETON[script]

    return _SINGLETON[UNKNOWN]


@lru_cache(maxsize=1)
def _allowed_ranges() -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """The Identifier Profile table and its bisection keys, imported once.

    Cached rather than built at module import, for the reason `skeleton` imports
    `confusables.py` inside itself: `script-constraint` never asks this question
    and must not pay 10 KiB of source to compile for it. After the first call the
    import is a `sys.modules` lookup and this is a cache hit.
    """
    from jamjet_guardrails._unicode.identifiers import ALLOWED_RANGES

    return ALLOWED_RANGES, tuple(start for start, _ in ALLOWED_RANGES)


def identifier_allowed(character: str) -> bool:
    """Whether one code point is Identifier_Status=Allowed, per UTS #39 section 3.1.

    UNICODE'S OWN ANSWER TO WHICH CHARACTERS STRINGS ARE WRITTEN IN, and the
    `confusables` check reads it for exactly that. A confusable prototype is
    evidence of a spoof only when the string it produces is one a reader could
    mistake for something real, and a prototype written in phonetic, technical
    or obsolete letters names nothing anybody registers or reads. Cyrillic small
    a folds to Latin `a` and Cyrillic small em folds to U+028D LATIN SMALL LETTER
    TURNED W; both prototypes are Latin, and treating them alike denies every
    Russian word carrying a Latin brand name.

    The property is UTS #39's rather than this project's, which is the rule for
    every exemption in this package: the restriction levels in section 5.2 are
    themselves defined over this set, so a check citing section 5.2 already
    reaches it. `unicode-data/16.0.0/IdentifierStatus.txt` is the source and
    `corpora/NOTICE.md` carries its digest.

    Restricted is the DEFAULT, which is what the absence of a range means:
    `IdentifierStatus.txt` lists the Allowed code points and its own @missing
    line makes everything else Restricted. So an unassigned code point, and one
    assigned after this pin, are both Restricted here, and the direction that
    fails is toward reporting no spoof rather than toward inventing one.

    Refuses a string that is not exactly one code point, for the reason
    `script_set` does: reading the first character of a longer one silently
    answers a question the caller did not ask.
    """
    if len(character) != 1:
        raise ValueError(
            f"identifier_allowed takes one code point, got {len(character)} characters; "
            "a longer string has a status per character and no single answer"
        )
    ranges, starts = _allowed_ranges()
    position = bisect_right(starts, ord(character)) - 1
    if position < 0:
        return False
    start, end = ranges[position]
    return start <= ord(character) <= end


def _canonical_decomposition(character: str) -> str:
    """NFD of ONE character, which is that character's full decomposition."""
    return unicodedata.normalize("NFD", character)


def _reordered(view: _Folded) -> _Folded:
    """Canonical ordering, carrying the offset map through the permutation.

    This is the half of NFD that `_fold.fold` cannot express, and it is why
    `skeleton` below is not three calls to `fold`. Canonical decomposition is
    per character and folds perfectly; canonical ORDERING is defined over a
    combining sequence and sorts the marks in it by combining class, which is a
    rule about a run and not about a character. Building it here rather than
    widening `fold` keeps the per-character contract that module states, and
    keeps this one operation, which is the only one in the package that needs
    more, where a reader of `skeleton` will find it.

    A permutation carries its offsets with it: the marks move and each one
    takes its source index along, so the result is an exact map that is simply
    no longer non-decreasing. `_Folded.span` is written for that.

    The early return is a real fast path and not a micro-optimisation: the
    condition below is exactly when the ordering algorithm has anything to do
    (UAX #15's rule is to swap any adjacent pair whose classes are out of
    order), so ordinary text of any script returns the view it was handed
    without rebuilding the string or the map.
    """
    classes = [unicodedata.combining(character) for character in view.text]
    if not any(classes[index] > classes[index + 1] > 0 for index in range(len(classes) - 1)):
        return view

    characters: list[str] = []
    origin: list[int] = []
    run: list[tuple[int, str, int]] = []

    def flush() -> None:
        # `list.sort` is stable, and the stability is the specification rather
        # than an implementation detail: canonical ordering must not reorder two
        # marks of the SAME combining class, because those two do interact and
        # swapping them changes what the text says.
        run.sort(key=lambda item: item[0])
        for _, character, source in run:
            characters.append(character)
            origin.append(source)
        run.clear()

    for character, source, combining in zip(view.text, view.origin, classes):
        if combining == 0:
            flush()
            characters.append(character)
            origin.append(source)
        else:
            run.append((combining, character, source))
    flush()

    return _Folded(
        text="".join(characters),
        origin=tuple(origin),
        source_length=view.source_length,
    )


def _nfd(source: str) -> _Folded:
    """NFD of a string, with the map back to it.

    Asserted against the interpreter's own `unicodedata.normalize("NFD", ...)`
    over a sweep of the code space by `tests/test_unicode.py`, because a
    normalisation this package computes itself and that disagrees with the
    standard one is a skeleton that disagrees with every other implementation
    while looking right.
    """
    return _reordered(fold(source, _canonical_decomposition))


def skeleton(text: str) -> _Folded:
    """The UTS #39 section 4 skeleton of `text`, and the way back to `text`.

    NFD, map every code point through the confusables table, NFD again. The
    result is a view: a match found in `skeleton(content).text` is reported to
    the caller as `view.span(start, end)`, which is the run of ORIGINAL
    characters that produced it, because that is what the chain rewrites and
    what a corpus labels.

    **A multi-character prototype maps back to the one source character.**
    U+FDFA, the Arabic ligature for a whole phrase, has a prototype 18 code
    points long, and every one of those 18 view characters carries the index of
    the single source character. So a match that covers any part of the
    expansion comes back as a span covering that whole source character and
    nothing else: spans stay closed over the source run, and a redaction never
    cuts a character in half. That is what lets a banned word survive one
    substituted Cyrillic letter without the rules engine keeping a list of
    letters to watch for.

    **The second NFD is not a formality.** 32 rows of the 16.0.0 table have a
    source that survives the first NFD and a prototype that does not, so the
    third step is the only one that ever sees them. U+1E9A, a with right half
    ring, maps to U+1EA3, which is `a` plus a hook above; without the third
    step its skeleton is a precomposed character and the skeleton of the same
    letter written decomposed is two, so two strings UTS #39 says have one
    skeleton fold apart and a banned word written one way misses the same word
    written the other. U+2251 is the sharper case: its prototype is `=` plus
    dot-above plus dot-below, which is not in canonical ORDER, so the third
    step reorders as well as decomposes.

    **`confusables.py` is imported here rather than at the top of this module.**
    It is 176 KiB of source and 21 ms to compile, and `script-constraint` never
    asks for a skeleton. After the first call the import is a `sys.modules`
    lookup, and this function is called once per piece of content rather than
    once per character, so the cost is paid where it is used and nowhere else.
    """
    from jamjet_guardrails._unicode.confusables import PROTOTYPES

    decomposed = _nfd(text)
    mapped = fold(decomposed.text, lambda character: PROTOTYPES.get(character, character))
    return compose(compose(decomposed, mapped), _nfd(mapped.text))
