"""Script-mixing and whole-script spoofs, per UTS #39.

A confusable is a character that RENDERS like another one. `pаypal` written
with a Cyrillic a is the same six glyphs as `paypal` and a different six code
points, so a substring check, a banned-word list and a human reviewer all read
it as the brand it is imitating. That is the attack this check exists for, and
it is also the attack that defeats the rules engine shipping beside it, which is
why `jamjet_guardrails.authoring.PatternGuardrail` grew `fold_confusables` in
the same release.

**Both halves of every rule here are load-bearing, and either half alone denies
a language.** A rule that fires on mixed script alone denies Russian, Ukrainian,
Serbian, Bulgarian and Greek prose carrying Latin brand names and code
identifiers. A rule that fires on whole-script confusables in prose denies every
Cyrillic word that happens to be spelled out of confusable letters, and there
are many: `сор`, `ухо`, `рот`. So a mixed token has to ALSO read as one script,
and a whole-script confusable has to ALSO be somewhere a spoof has a target,
which is a hostname or a handle and not a sentence.

**What a prototype has to be before it is evidence, which is where the design's
own example was wrong and the measurement corrected it.** The design said
`iPhoneом` in Russian prose passes because Cyrillic em has no Latin prototype.
Measured against confusables 16.0.0 it has one: U+028D LATIN SMALL LETTER TURNED
W. So does Cyrillic ef, U+0278 LATIN SMALL LETTER PHI, which would make every
`.рф` domain a whole-script confusable. 140 of the 296 Cyrillic letters have a
prototype written wholly in Latin, and a rule that asked only about script would
deny the Russian ccTLD and most Russian sentences carrying a Latin brand with a
case ending glued to it.

What separates the real spoof from those is not the script of the prototype but
whether the prototype names a string anybody could be reading. `а` folds to `a`;
`м` folds to a phonetic letter no brand, hostname or handle is written in. UTS
#39 already publishes that distinction as Identifier_Status, and its own
restriction levels in section 5.2 are defined over exactly that set, so a check
citing section 5.2 already reaches it. `jamjet_guardrails._unicode`
vendors it and `identifier_allowed` answers it. Under it, 104 of those 140
Cyrillic letters remain, `pаypal` fires, and `iPhoneом` and `почта.рф` do not.
`corpora/NOTICE.md` records the measurement.

**No exemption here is a list.** The permitted script combinations are UTS #39
section 5.2's own table, the identifier profile is UTS #39 section 3.1's own
table, and there is no host allowlist, no safe-domain list and no brand list.
Every exemption in this repository that approximated a set with a shape became
the channel it was written to deny.

Cost, and the shape of it. Linear in the content length. 186 ms median for one
megabyte of the seeded input recorded in `docs/performance.md`, on an Apple M3
Max under CPython 3.14.5, with the median rising 3.91x to 4.08x per 4x of input
across the whole range from 1 KB. Each character is categorised once to find
token boundaries; the script and confusable questions are asked only of tokens
that are not wholly ASCII, which on that input is none of them, so 186 ms is
this check's FLOOR and text in a non-Latin script costs more. That measurement
denies rather than redacts, and the seeded input carries no confusable, so it is
the scan alone with no finding built. `scripts/measure_throughput.py` reruns it,
and `docs/performance.md` states the machine, the input and the method.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from itertools import chain

from jamjet_guardrails._spans import _rewrite
from jamjet_guardrails.detectors.injection_structural import _DEFAULT_IGNORABLE
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import (
    Context,
    Decision,
    Direction,
    Finding,
    Kind,
    Provenance,
    Verdict,
)

CONFUSABLES_TYPES = frozenset(
    {
        "MIXED_SCRIPT_CONFUSABLE",
        "WHOLE_SCRIPT_CONFUSABLE",
    }
)

_VERSION = "0.1.0"

# UTS #39 section 5.1 returns Common and Inherited as themselves, and every
# caller decides for itself what a wildcard does. Here they contribute no script
# to a token: an ASCII digit inside `paypal2` must not make the token mixed, and
# a combining mark inside a Devanagari word must not make it single-script by
# itself. A set is a wildcard only if it holds nothing BUT these two, because
# U+00B7 has Script=Common and Script_Extensions naming sixteen real scripts,
# and that middle dot is a Georgian letter's business rather than a wildcard.
_WILDCARD = frozenset({"Common", "Inherited"})

# UTS #39 section 5.2, Highly Restrictive. Latin may be mixed with exactly one
# of three groups, and those three groups are Unicode's own table rather than
# this project's list. They are what lets Japanese, Chinese and Korean text
# carrying Latin brand names pass by RULE: `iPhone対応` is Latin plus Han and
# is covered by the first row, so nothing here has to know what a brand is.
#
# Restated from the standard rather than cited by number alone, because the
# check has to be portable from this module: a port that implemented "Latin plus
# any one other script" would allow Latin plus Cyrillic, which is the whole of
# the attack.
_HIGHLY_RESTRICTIVE: tuple[frozenset[str], ...] = (
    frozenset({"Latin", "Han", "Hiragana", "Katakana"}),
    frozenset({"Latin", "Han", "Bopomofo"}),
    frozenset({"Latin", "Han", "Hangul"}),
)

# The general categories a token is made of: letters, marks and numbers. Marks
# are in because a Devanagari or Arabic word is letters and marks and splitting
# it at every mark would make each piece single-script and hide nothing; numbers
# are in because `paypal2` is one word to a reader. Everything else ENDS a
# token, which is what makes the apostrophe and the hyphen token boundaries
# without either of them being named here: `iPhone-ом` and `iPhone'ом` are two
# single-script tokens and pass, and `iPhoneом` is one mixed token. Splitting is
# the conservative direction.
_TOKEN_CATEGORIES = frozenset({"L", "M", "N"})

# Default-ignorable code points END a token and belong to none, which is what
# keeps this check's character set disjoint from every `injection-structural`
# signal. The set is IMPORTED from that module rather than restated, so the two
# cannot drift: a code point added there is excluded here on the same commit,
# and `tests/test_confusables.py::test_this_check_claims_no_code_point_any_structural_signal_claims`
# holds the disjointness pairwise the way that module holds its own three.
#
# Ending a token rather than being skipped inside one is the only spelling that
# gives real disjointness, because a span is contiguous: a token that ran ACROSS
# a zero-width space would report a span containing it, and that span shares a
# code point with `ZERO_WIDTH_SMUGGLING`. The design document says these
# characters "belong to the token"; they cannot, and this is the half of that
# sentence the disjointness requirement in the same section overrides.
#
# What it costs is stated rather than hidden: `pаypal` with a zero-width space
# between the two halves is two single-script tokens here and is not reported by
# this check at all. It is reported by `injection-structural`, which owns that
# code point and denies by default, so a chain running both still denies. A
# chain running this check alone does not, and `corpora/NOTICE.md` says so.
_IGNORABLE: frozenset[str] = frozenset(
    chr(point) for low, high in _DEFAULT_IGNORABLE for point in range(low, high + 1)
)

# A scheme, then `://`, then the authority up to the first delimiter. Bare
# `example.com` is deliberately NOT a URL here: a dot between two words is a
# missing space after a full stop at least as often as it is a hostname, and
# widening this to catch one would put every such sentence in front of the
# whole-script rule. The cost is a spoofed host written without a scheme, which
# `corpora/NOTICE.md` discloses with the case that carries it.
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://([^\s/?#]*)")

# What an email local part and domain are made of. Deliberately not a validating
# address grammar: this is only ever asked to find the DOMAIN's labels, and an
# address shape that is wrong in some other way still has the labels a spoof
# would live in.
_ATOM_PUNCTUATION = frozenset("._%+-")
_HANDLE_PUNCTUATION = frozenset("_-")


def _is_token_character(character: str) -> bool:
    """Letter, mark or number, and not default-ignorable."""
    return unicodedata.category(character)[0] in _TOKEN_CATEGORIES and character not in _IGNORABLE


def _tokens(content: str) -> list[tuple[int, int]]:
    """Maximal runs of token characters, as half-open spans."""
    spans: list[tuple[int, int]] = []
    index = 0
    length = len(content)
    while index < length:
        if not _is_token_character(content[index]):
            index += 1
            continue
        start = index
        while index < length and _is_token_character(content[index]):
            index += 1
        spans.append((start, index))
    return spans


@lru_cache(maxsize=8192)
def _scripts_of(character: str) -> frozenset[str]:
    """`script_set` with a memo, because a token scan asks per character.

    The import is inside the function, not at the top of this module, so
    `import jamjet_guardrails` pays nothing for the vendored tables:
    `detectors/__init__.py` is imported from the package root and a top-level
    import here would drag 231 KiB of generated source in behind it.
    """
    from jamjet_guardrails._unicode import script_set

    return script_set(character)


def _is_wildcard(scripts: frozenset[str]) -> bool:
    return scripts <= _WILDCARD


@lru_cache(maxsize=8192)
def _confusable_into(character: str, script: str) -> bool:
    """Whether `character` folds to something written in `script`.

    THE TWO CONDITIONS ARE DIFFERENT QUESTIONS AND BOTH ARE NECESSARY.

    The first is the design's: every non-wildcard code point of the prototype
    lies in `script`. Without it a Cyrillic letter that folds to a Greek letter
    would count as a Latin spoof.

    The second is this module's, and it is the correction the measurement in the
    docstring forced: every code point of the prototype is in the UTS #39
    identifier profile. Without it Cyrillic em, which folds to a phonetic Latin
    letter, counts as a Latin spoof, and `iPhoneом` in Russian prose is denied
    along with `.рф` and most of the Russian sentences that name a Latin brand.
    A prototype outside the profile names no string anybody registers, types or
    reads, so it is not evidence that the token imitates one.

    A prototype of nothing but wildcards is refused too. A character folding to
    a bare digit or a bare space imitates no word in any script, and counting it
    would make the majority-script test turn on punctuation.

    Memoised because the answer depends on the character and the script alone,
    and a token scan over a megabyte asks it thousands of times about the same
    dozen letters.
    """
    from jamjet_guardrails._unicode import identifier_allowed, skeleton

    prototype = skeleton(character).text
    if not prototype:
        return False
    substantive = [point for point in prototype if not _is_wildcard(_scripts_of(point))]
    if not substantive:
        return False
    if not all(script in _scripts_of(point) for point in substantive):
        return False
    return all(identifier_allowed(point) for point in prototype)


def _resolved(characters: str) -> tuple[frozenset[str], frozenset[str], list[str]]:
    """The token's resolved script set, the union of its scripts, and its letters.

    Resolution is UTS #39 section 5.1: the INTERSECTION over the token's
    non-wildcard code points. Non-empty means every one of them can be read as
    one script, which is what single-script means for a character carrying
    Script_Extensions; empty means the token is mixed.

    The union comes back beside it because the Highly Restrictive test asks a
    different question of the same characters, and computing it twice from two
    walks is how two answers about one token start to disagree.
    """
    substantive = [
        character for character in characters if not _is_wildcard(_scripts_of(character))
    ]
    if not substantive:
        return frozenset(), frozenset(), substantive
    sets = [_scripts_of(character) for character in substantive]
    return frozenset.intersection(*sets), frozenset(chain.from_iterable(sets)), substantive


def _majority(substantive: list[str]) -> str:
    """The script most of the token's code points are in; the first one's on a tie.

    Counted per (code point, script) pair rather than per code point, because a
    character carrying Script_Extensions is in several scripts at once and
    picking one of them arbitrarily would make the majority depend on the order
    a set iterates in, which varies with PYTHONHASHSEED.

    The tie-break prefers a script the FIRST code point is in, which is the
    design's rule, and falls back to the alphabetically first candidate so that
    two runs over one token always agree.
    """
    counts: Counter[str] = Counter()
    for character in substantive:
        counts.update(_scripts_of(character))
    best = max(counts.values())
    candidates = {script for script, count in counts.items() if count == best}
    preferred = sorted(candidates & _scripts_of(substantive[0]))
    return preferred[0] if preferred else min(candidates)


def _mixed_script_spans(content: str) -> list[tuple[int, int]]:
    """Every token that fails Highly Restrictive and reads as its majority script."""
    spans: list[tuple[int, int]] = []
    for start, end in _tokens(content):
        characters = content[start:end]
        # ASCII cannot be mixed: every ASCII letter is Latin and every ASCII
        # digit is Common, so the intersection is `{Latin}` or the token is all
        # wildcards. A fast path rather than a special case, and it is what keeps
        # ordinary English text from asking the vendored tables anything at all.
        if characters.isascii():
            continue
        resolved, union, substantive = _resolved(characters)
        if resolved or not substantive:
            continue
        if any(union <= permitted for permitted in _HIGHLY_RESTRICTIVE):
            continue
        majority = _majority(substantive)
        outside = [character for character in substantive if majority not in _scripts_of(character)]
        if outside and all(_confusable_into(character, majority) for character in outside):
            spans.append((start, end))
    return spans


def _authority(content: str, start: int, end: int) -> tuple[int, int]:
    """The host part of a URL authority: userinfo and port removed.

    Returns an empty span for an IPv6 literal. `[::1]` has no labels a spoof
    could live in, and the bracket syntax would otherwise split into pieces this
    scan would read as labels.
    """
    at = content.rfind("@", start, end)
    if at != -1:
        start = at + 1
    if start < end and content[start] == "[":
        return (start, start)
    colon = content.rfind(":", start, end)
    if colon != -1 and content[colon + 1 : end].isdigit():
        end = colon
    return (start, end)


def _dot_labels(content: str, start: int, end: int) -> list[tuple[int, int]]:
    """The dot-separated pieces of `content[start:end]`, as spans, empties dropped."""
    labels: list[tuple[int, int]] = []
    cursor = start
    while cursor <= end:
        stop = content.find(".", cursor, end)
        if stop == -1:
            stop = end
        if stop > cursor:
            labels.append((cursor, stop))
        cursor = stop + 1
    return labels


def _is_atom(character: str) -> bool:
    return _is_token_character(character) or character in _ATOM_PUNCTUATION


def _is_handle_character(character: str) -> bool:
    return _is_token_character(character) or character in _HANDLE_PUNCTUATION


def _spoofable_labels(content: str) -> list[tuple[int, int]]:
    """Every host label, email domain label and handle in the content.

    THREE CONTEXTS AND NO MORE, because these are the three places a whole-script
    confusable has a target. The same word in a sentence is a word: `сор` is
    Russian for rubbish and `ухо` is Russian for an ear, and both are built
    entirely out of letters that fold to Latin. Reporting a whole-script
    confusable in prose denies the language, which is why the design forbids it
    and why this function exists instead of a content-wide scan.

    URLs are resolved first and their spans excluded from the address scan,
    because the `@` in `https://user@host/` is userinfo rather than an address
    and reading it as one would report the scheme as a local part.
    """
    labels: list[tuple[int, int]] = []
    urls: list[tuple[int, int]] = []
    for match in _URL.finditer(content):
        urls.append(match.span())
        host = _authority(content, match.start(1), match.end(1))
        labels.extend(_dot_labels(content, *host))

    for index, character in enumerate(content):
        if character != "@" or any(start <= index < end for start, end in urls):
            continue
        local = index
        while local > 0 and _is_atom(content[local - 1]) and content[local - 1] != "@":
            local -= 1
        domain = index + 1
        while domain < len(content) and _is_atom(content[domain]):
            domain += 1
        if local < index:
            # An address. The domain must carry a dot, or it is a mention with a
            # word in front of it rather than a mailbox.
            if content.find(".", index + 1, domain) != -1:
                labels.extend(_dot_labels(content, index + 1, domain))
            continue
        # A handle. Its own run rather than the atom run, because a handle
        # carries no dots and a trailing full stop is the sentence's.
        handle = index + 1
        while handle < len(content) and _is_handle_character(content[handle]):
            handle += 1
        if handle > index + 1:
            labels.append((index + 1, handle))
    return labels


def _whole_script_spans(content: str) -> list[tuple[int, int]]:
    """Single-script non-Latin labels that read, character for character, as Latin."""
    spans: set[tuple[int, int]] = set()
    for start, end in _spoofable_labels(content):
        characters = content[start:end]
        if characters.isascii() or any(character in _IGNORABLE for character in characters):
            continue
        resolved, _, substantive = _resolved(
            "".join(character for character in characters if _is_token_character(character))
        )
        if not substantive or not resolved or "Latin" in resolved:
            continue
        if all(_confusable_into(character, "Latin") for character in substantive):
            spans.add((start, end))
    return sorted(spans)


class ConfusablesGuardrail:
    """Detects a token written in one script and read as another."""

    # Annotated with the Literal types rather than left as bare assignments: a
    # bare `kind = "constraint"` infers `str`, and protocol attribute matching is
    # invariant, so it would not satisfy `kind: Kind`.
    name: str = "confusables"
    version: str = _VERSION
    kind: Kind = "constraint"
    # Both directions. A spoofed brand or hostname in model OUTPUT is the one
    # that reaches a person: it is rendered in a chat window, clicked, and
    # resolved. On input it is an instrument planted in retrieved content. The
    # threat is the same shape in both, so the default decision is too, and the
    # mapping form of `on_detect` is there for a caller who disagrees.
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def __init__(self, on_detect: Decision | Mapping[Direction, Decision] = "deny") -> None:
        """Refuses a decision this check cannot act on, and a mapping with a hole.

        `allow` is refused for the reason `PatternGuardrail` refuses it: a check
        configured to allow on a detection is a check that runs and cannot act.
        A mapping that omits a declared direction is refused because the
        alternative is a `KeyError` from inside `check`, which fails closed and
        names nothing.
        """
        if isinstance(on_detect, str):
            resolved: Mapping[Direction, Decision] = {
                direction: on_detect for direction in self.directions
            }
        else:
            missing = sorted(self.directions - set(on_detect))
            if missing:
                raise GuardrailUnavailableError(
                    f"{self.name!r} runs in {sorted(self.directions)} but on_detect names "
                    f"no decision for {missing}; the alternative is a KeyError from "
                    "inside check, which fails closed and names nothing"
                )
            extra = sorted(set(on_detect) - self.directions)
            if extra:
                raise GuardrailUnavailableError(
                    f"{self.name!r} runs in {sorted(self.directions)} but on_detect also "
                    f"names {extra}, which it would never be asked about; a policy for a "
                    "direction it does not declare would be silently dropped"
                )
            resolved = {direction: on_detect[direction] for direction in self.directions}
        for direction, decision in sorted(resolved.items()):
            if decision not in ("redact", "deny"):
                raise ValueError(
                    f"on_detect for {direction!r} must be 'redact' or 'deny', got "
                    f"{decision!r}. A check configured to allow on a detection is a check "
                    "that runs and cannot act"
                )
        self._on_detect = resolved

    def _matches(self, content: str) -> list[tuple[str, tuple[int, int]]]:
        """Every span both signals claim, sorted by span. Nothing is dropped.

        SORTED BY SPAN is a precondition of `_merge`, which tests each span
        against the running end of the region it is extending and looks no
        further back. Each signal walks left to right, so each list is ordered on
        its own and it is the concatenation that is not.

        The two signals can claim the SAME span, and unlike the three structural
        signals they are not disjoint in their character sets, so the sort has to
        be total on ties. A mixed token needs an empty resolved script set and a
        whole-script label needs a non-empty one, so no one token is both; a
        label carrying a hyphen splits into several tokens, and there one span
        can nest inside another. `sorted` is stable and a tie on the start puts
        the shorter first, which is the order `_merge` consumes correctly.
        """
        found = [("MIXED_SCRIPT_CONFUSABLE", span) for span in _mixed_script_spans(content)]
        found += [("WHOLE_SCRIPT_CONFUSABLE", span) for span in _whole_script_spans(content)]
        return sorted(found, key=lambda pair: pair[1])

    def check(self, content: str, context: Context) -> Verdict:
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        found = self._matches(content)
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        findings = [Finding(type=name, span=span) for name, span in found]
        if self._on_detect[context.direction] == "deny":
            return Verdict("deny", None, findings, provenance, saw(content))
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))
