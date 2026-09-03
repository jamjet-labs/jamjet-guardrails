"""Does this run of characters decode to text, and does that text read as prose.

Two checks need that answer and they must not each write their own.
``url-exfiltration`` asks it of a URL component; ``encoded-content`` asks it of a
run anywhere in the content. Built twice they would be two answers that drift,
and the drift is invisible: each check has its own corpus, each corpus is green,
and the same base64 run gets two verdicts depending on which check saw it.

**Decodability, not entropy.** Hashes, UUIDs, git SHAs, signatures and random
API tokens are all high-entropy and none of them decodes to printable text.
Twelve random bytes are all printable with probability under one in a hundred
thousand. An entropy score cannot separate a payload from a signature; a decode
attempt does, and that difference is what the published precision rests on. The
corpus carries every one of those shapes as a labelled negative.

**One level.** Decoded text is NEVER handed back to ``candidates``. A doubly
encoded payload passes, and ``corpora/NOTICE.md`` discloses it with a worked
example. The rejected alternative is a decode loop, which turns one bounded pass
over the content into an unbounded one and makes every span a claim about a
string the caller never had.

Private by name and on purpose. Nothing here is re-exported from the package
root, and ``docs/conformance.md`` says a port may reach the same verdicts by any
other means.

Linear in the length of what it is given. Each alphabet is one ``finditer`` pass
and each candidate run is decoded at most once per alphabet, so a caller pays per
RUN rather than per document. It carries no published latency figure of its own,
because it is not a check and nothing calls it on a whole document: the figure
for its one caller today is in ``docs/performance.md`` under ``url-exfiltration``.

## The floors, and what each side of one costs

A run shorter than its alphabet's floor is not a candidate and does not decode.
Without a floor every two-character hex pair in a colour, every four-character
word in a slug and every ``%20`` in a path is a decode attempt whose result is
one or two characters of noise, and ``is_prose`` is the only thing standing
between that and a finding.

Each floor was swept from 1 to 259 in steps of one over
``corpora/url-exfiltration/in-repo.jsonl``, with every other value at its
shipped setting, scoring the corpus at each step. The shipped configuration
scores F1 0.9143 on that corpus.

| Alphabet | Floor | Plateau at F1 0.9143 | First floor that costs | Cost there |
|---|---:|---|---:|---|
| base64 | 16 | 1 to 32 | 33 | one true positive, F1 to 0.8986 |
| hex | 16 | 1 to 42 | 43 | one true positive, F1 to 0.8986 |
| percent | 6 | 1 to 106 | none below 260 | see below |
| rot13 | 24 | 1 to 109 | 110 | one true positive, F1 to 0.8986 |

**The sweep bounds these four from ABOVE and does not choose them, and saying
otherwise would be dressing noise as a measurement.** Every alphabet's curve is
flat from 1 up to its ceiling, so "the smallest floor reaching the best F1" is
1 for all four, and a base64 floor of 1 decodes every letter in every path
segment. The shipped value is instead the shortest run each alphabet can carry a
payload in: 12 bytes for base64, 8 bytes for hex, one triplet beside a word for
percent, three words for rot13. What the sweep contributes is the margin, which
is the part that could have been wrong: 16 is half the base64 ceiling and 38% of
the hex one, and 24 is 22% of the rot13 ceiling.

**The percent curve improves above its plateau, and the improvement is
rejected.** From 107 to 147 F1 rises to 0.9275, because one false positive stops
being read: case ``url-0076`` carries a 106-character percent run, a chart API's
``title`` parameter. That is not a better rule. It is a URL parser that declines
to read percent escapes above a length, and the length is one corpus case's
query string. The window shuts again at 148, one past the 147-character percent
run in ``url-0001``, which is a positive, and F1 falls to 0.8788, below what the
shipped floor scores. A window bounded at both ends by the lengths of two strings
in one corpus is the definition of a value fitted to that corpus, so 6 ships and
this paragraph is why.

``tests/test_decode.py::test_every_floor_sits_inside_the_flat_region_the_sweep_measures``
re-runs the sweep on every test run rather than trusting this table, so a corpus
edit that moves a ceiling under a shipped floor fails there instead of going
stale here.

## Rot13 is a measured risk, not a commitment

Every alphabetic run is a rot13 candidate, so rot13 has no alphabet to recognise
it by and its only evidence is a two-sided test: the ORIGINAL run must fail
``is_prose`` and the ROTATED run must pass it. One side alone is useless.
Testing only the rotated side fires on every second sentence of ordinary English
(rot13 is an involution, so "ordinary text" and "rot13 of ordinary text" are the
same population seen twice); testing only the original side is a prose detector
wearing an encoding's name.

**It ships, and the ablation is why.** Removing rot13 from the readings
``url-exfiltration`` takes of a URL component, with nothing else changed, moves
that check on its own corpus from precision 0.9143, recall 0.9143, F1 0.9143 and
6 wrong decisions to precision 0.9091, recall 0.8571, F1 0.8824 and 8 wrong
decisions. Two positives are lost, ``url-0071`` and ``url-0072``, and NO case
changes the other way: the false-positive count is 3 with rot13 and 3 without,
over a corpus whose 53 negatives are almost entirely ordinary English prose, all
of it a rot13 candidate. The two-sided test is what buys that, and the ablation
is re-run by
``tests/test_url_exfiltration.py::test_rot13_buys_two_positives_and_costs_no_precision``
so the paragraph cannot outlive the measurement.

## The function-word list

``is_prose`` asks how many of a text's words are among the forty commonest words
of a large sample of English. The spec this module was written from names the
external evaluation corpus in ``training/`` as the sample to derive that list
from. **That corpus is not in the repository.** ``training/evalset.py`` reads it
out of ``data/``, which ``.gitignore`` excludes by an anchored rule, so a clone
has the loader and not the rows. A list derived from a file a reader cannot open
is a list nobody can check.

So the list is derived from ``training/generated/rows.jsonl``, which IS tracked:
3,584 rows of generated English, the corpus this repository already publishes
provenance for. The rule is the whole rule, stated so it can be re-run: tokenise
each row's text case-folded by ``_WORD`` below, count, take the forty commonest,
tie-broken alphabetically. They cover 35.6% of the 112,473 tokens.
``tests/test_decode.py::test_the_function_words_are_the_forty_commonest_words_of_the_corpus_they_name``
re-derives the tuple from that file and requires it back character for
character.

**What the fallback costs, named rather than smoothed over.** Six of the forty
are content words of that generator's own subject matter: ``assistant``,
``instructions``, ``please``, ``previous``, ``security`` and ``system``. They are
kept. Removing them by hand would make the list "the forty commonest, minus the
ones I did not like", which is an enumeration and is the shape decision 9 of the
phase-3 design forbids everywhere else in this check. What they buy is a small
bias toward reading injection-flavoured English as prose, which moves recall on
the attack class and not precision on the negatives, because those six are also
ordinary English words that ordinary negatives contain.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping

# The four alphabets. A caller loops these rather than importing a regex table,
# and `decode` refuses anything outside them by name: an encoding argument that
# is a typo must not resolve to "this run does not decode", which is what a
# `return None` default would make it.
ENCODINGS: tuple[str, ...] = ("base64", "hex", "percent", "rot13")

FLOORS: Mapping[str, int] = {"base64": 16, "hex": 16, "percent": 6, "rot13": 24}

# Decoded bytes that are valid UTF-8 and still mostly unprintable are a binary
# body that happened to survive the decoder, not text. 0.9 rather than 1.0
# because real text carries the odd control character, and rather than 0.5
# because at that ratio half of a JPEG's decoded head reads as a payload.
_PRINTABLE_RATIO = 0.9

# `\t`, `\n` and `\r` are text; `str.isprintable` says otherwise.
_ALSO_PRINTABLE = frozenset("\t\n\r")

# Two alphabets, matched separately rather than as one class. A run mixing `+`
# with `-`, or `/` with `_`, is not base64 in either alphabet, and a union class
# would swallow it, decode it under one of them and report noise.
_B64_STANDARD = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_B64_URLSAFE = re.compile(r"[A-Za-z0-9_-]+={0,2}")
_B64_STANDARD_FULL = re.compile(r"\A[A-Za-z0-9+/]+={0,2}\Z")
_B64_URLSAFE_FULL = re.compile(r"\A[A-Za-z0-9_-]+={0,2}\Z")

_HEX = re.compile(r"[0-9a-fA-F]+")
_HEX_FULL = re.compile(r"\A[0-9a-fA-F]+\Z")

# A percent run is triplets and unreserved characters interleaved, and it must
# carry at least one triplet: without that requirement every bare word in a path
# is a percent candidate that decodes to itself.
_PERCENT = re.compile(r"[%0-9A-Za-z\-._~+]*%[0-9A-Fa-f]{2}[%0-9A-Za-z\-._~+]*")
_PERCENT_TRIPLET = re.compile(r"%[0-9A-Fa-f]{2}")

# Letters and the spaces between them. Punctuation ends a run, which splits
# ordinary prose into clauses and leaves each clause long enough to score.
_ROT13 = re.compile(r"[A-Za-z][A-Za-z ]*[A-Za-z]")
_ROT13_FULL = re.compile(r"\A[A-Za-z][A-Za-z ]*[A-Za-z]\Z")

# One apostrophe joins letters; a bare apostrophe is not a word. Matched to the
# tokeniser the function-word list is derived under, and the match is the point:
# derive under one tokeniser and score under another and the density means
# nothing. An earlier draft used `[a-z']+` in both, which made `'` itself the
# 25th commonest "word" of the corpus.
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

# The forty commonest words of `training/generated/rows.jsonl` under `_WORD`,
# tie-broken alphabetically. Derived, not chosen: see the module docstring for
# the rule, why this file rather than the one the spec named, and what the
# substitution costs.
_FUNCTION_WORDS = frozenset(
    {
        "a",
        "about",
        "all",
        "an",
        "and",
        "any",
        "are",
        "as",
        "assistant",
        "be",
        "can",
        "could",
        "do",
        "for",
        "from",
        "how",
        "if",
        "in",
        "instructions",
        "into",
        "is",
        "it",
        "me",
        "not",
        "now",
        "of",
        "on",
        "or",
        "our",
        "please",
        "previous",
        "security",
        "system",
        "that",
        "the",
        "this",
        "to",
        "with",
        "you",
        "your",
    }
)

# Three, because a density over one or two tokens is 0, 0.5 or 1 and is not a
# density. This is not a length floor on the text a caller cares about; callers
# set those, and this check exists so `is_prose("to the")` cannot answer True on
# the strength of two tokens.
_MIN_WORDS = 3

# Measured on the 3,584 rows of `training/generated/rows.jsonl` under the list
# above: the 1st percentile of per-row function-word density is 0.133, the 5th
# is 0.227 and the median is 0.360. 0.20 sits between the first two, so roughly
# 4% of genuine English rows fail this gate. That direction is deliberate: a
# rejected prose is a MISSED detection, which costs recall, and a rejected
# non-prose is a false positive that never happened.
_FUNCTION_WORD_DENSITY = 0.20

# A hex digest tokenises into short letter runs, several of which are in the
# list ("a", "be", "do"), and scored 0.150 density on the sample in
# `tests/test_decode.py`. Its LETTER ratio is 0.406, which is what actually
# rejects it, so this gate is load-bearing rather than belt and braces.
_LETTER_RATIO = 0.6

# The gate that separates a hyphenated slug from a sentence, and the one nothing
# else can do. `why-a-search-link-is-not-an-exfiltration-channel` scores 0.444
# density and 0.833 letters: it passes both gates above, because a slug is
# English words with the spaces taken out. Ordinary English runs about one space
# in six characters; 0.05 is the loosest bound that still refuses zero spaces
# and every one-space-in-twenty shape a path segment produces.
_SPACE_RATIO = 0.05

_ROT13_TABLE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
)


def is_prose(text: str) -> bool:
    """Whether ``text`` reads as natural language, by three ratios and no model.

    Letters, spaces and the density of the function-word list, all three
    required. Each one alone admits a shape the other two refuse: a hex digest
    clears the density gate and fails on letters; a hyphenated slug clears
    letters and density and fails on spaces; a run of punctuation clears spaces
    and fails on letters.

    Empty text and text with fewer than ``_MIN_WORDS`` words are not prose,
    which also keeps every ratio's denominator non-zero below.
    """
    if not text:
        return False
    letters = sum(1 for char in text if char.isalpha())
    if letters / len(text) < _LETTER_RATIO:
        return False
    if text.count(" ") / len(text) < _SPACE_RATIO:
        return False
    words = _WORD.findall(text.lower())
    if len(words) < _MIN_WORDS:
        return False
    hits = sum(1 for word in words if word in _FUNCTION_WORDS)
    return hits / len(words) >= _FUNCTION_WORD_DENSITY


def _printable(text: str) -> bool:
    """Whether decoded text is text rather than bytes that survived a decoder."""
    if not text:
        return False
    good = sum(1 for char in text if char.isprintable() or char in _ALSO_PRINTABLE)
    return good / len(text) >= _PRINTABLE_RATIO


def _decode_base64(run: str) -> str | None:
    """Strict in both alphabets, and strict about padding in each direction.

    A run carrying padding must be a multiple of four: `abcde=` is not base64
    that lost its padding, it is a run that is not base64, and accepting it
    turns every word ending in `=` inside a query string into a decode attempt.
    A run without padding may be padded here, except at a length of one more
    than a multiple of four, which no base64 encoding produces.
    """
    if _B64_STANDARD_FULL.match(run):
        body = run
    elif _B64_URLSAFE_FULL.match(run):
        body = run.replace("-", "+").replace("_", "/")
    else:
        return None
    stripped = body.rstrip("=")
    if len(body) != len(stripped) and len(body) % 4 != 0:
        return None
    if len(stripped) % 4 == 1:
        return None
    padded = stripped + "=" * (-len(stripped) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _decode_hex(run: str) -> str | None:
    """Half a byte is not a byte, and ``bytes.fromhex`` is what says so.

    An explicit ``len(run) % 2`` test stood here and was deleted rather than
    kept as belt and braces: it is unreachable. ``bytes.fromhex`` refuses an odd
    length on its own, and ``_HEX_FULL`` has already excluded the ASCII
    whitespace that method otherwise skips, so nothing could reach the second
    test that the first had not already refused. A guard no mutation can kill is
    not a guard; the ``except`` below is the one that turns the refusal into a
    ``None``.
    """
    if not _HEX_FULL.match(run):
        return None
    try:
        raw = bytes.fromhex(run)
    except ValueError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _decode_percent(run: str) -> str | None:
    """One level of percent decoding, refusing a stray ``%``.

    Not ``urllib.parse.unquote``, which leaves a malformed ``%zz`` standing and
    reports success. A run holding one is not percent-encoded, and reading it as
    if it were means reporting a decode of something that was never encoded.
    ``+`` is left alone: whether it means a space is a fact about the query
    syntax around the run, which this module does not see, so the caller that
    knows applies it before asking.
    """
    if not _PERCENT_TRIPLET.search(run):
        return None
    raw = bytearray()
    index = 0
    while index < len(run):
        char = run[index]
        if char == "%":
            triplet = run[index : index + 3]
            if len(triplet) != 3 or not _PERCENT_TRIPLET.fullmatch(triplet):
                return None
            raw.append(int(triplet[1:], 16))
            index += 3
            continue
        raw.extend(char.encode("utf-8"))
        index += 1
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _decode_rot13(run: str) -> str | None:
    """The two-sided test, and it lives here rather than in the caller.

    Both consumers of this module would otherwise have to remember the original
    side, and the one that forgot would fire on every English sentence long
    enough to be a candidate. Rot13 is an involution, so the encoded population
    and the plain population are the same set: the only thing that separates
    them is which direction reads as language.
    """
    if not _ROT13_FULL.match(run):
        return None
    rotated = run.translate(_ROT13_TABLE)
    if not is_prose(rotated) or is_prose(run):
        return None
    return rotated


_DECODERS = {
    "base64": _decode_base64,
    "hex": _decode_hex,
    "percent": _decode_percent,
    "rot13": _decode_rot13,
}


def decode(run: str, encoding: str) -> str | None:
    """The decoded text, or ``None`` where the run does not decode to text.

    ``None`` covers every way a run fails to be what it claims: an alphabet it
    does not fit, padding that does not add up, an odd hex length, a malformed
    triplet, bytes that are not UTF-8, decoded text that is mostly unprintable,
    and a run below its alphabet's floor.

    Raises:
        ValueError: for an encoding outside ``ENCODINGS``. A caller who
            misspells one must not be told the run does not decode; that reads
            as a measurement and is a check that stopped checking.
    """
    if encoding not in _DECODERS:
        raise ValueError(f"unknown encoding {encoding!r}; expected one of {list(ENCODINGS)}")
    if len(run) < FLOORS[encoding]:
        return None
    text = _DECODERS[encoding](run)
    if text is None:
        return None
    # Rot13 answers with prose or not at all, so the printable test would be
    # asking a question `is_prose` has already answered more strictly.
    if encoding != "rot13" and not _printable(text):
        return None
    return text


def candidates(content: str) -> list[tuple[int, int, str]]:
    """Every maximal run in each alphabet that clears that alphabet's floor.

    ``(start, end, encoding)``, half-open over code points of ``content``,
    sorted by span and then by the order ``ENCODINGS`` lists. Runs from
    different alphabets overlap freely and are all reported: a hex digest is
    also a base64 run, and which one decodes is what ``decode`` answers.

    Maximal, so a run is reported once at its full extent rather than once per
    substring. That is what keeps the span a caller reports the whole encoded
    region rather than the middle of it, and it is why the floor is applied to
    the maximal run and not to every window inside it.
    """
    found: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    order = {name: index for index, name in enumerate(ENCODINGS)}
    for encoding, pattern in (
        ("base64", _B64_STANDARD),
        ("base64", _B64_URLSAFE),
        ("hex", _HEX),
        ("percent", _PERCENT),
        ("rot13", _ROT13),
    ):
        floor = FLOORS[encoding]
        for match in pattern.finditer(content):
            start, end = match.span()
            if end - start < floor:
                continue
            key = (start, end, encoding)
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
    return sorted(found, key=lambda run: (run[0], run[1], order[run[2]]))
