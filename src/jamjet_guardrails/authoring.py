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
import unicodedata
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jamjet_guardrails._fold import _Folded, casefold_view, compose, fold
from jamjet_guardrails._spans import _rewrite, _scan
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import Context, Decision, Direction, Finding, Kind, Provenance, Verdict

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
    matrix. An opcode this walk does not recognise is ignored rather than
    descended into, which is the safe direction: it can only make the guard
    MISS a nesting that opcode's subpattern contained, never invent one that
    is not there, and every opcode the parser actually produces that carries
    a subpattern is handled by name above.
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


# A finding type is a label a corpus uses and a name a placeholder prints, so
# the domain is the one both can carry. Checked rather than documented: a type
# that cannot be labelled makes a corpus row that can never match, and a
# published per-type number for a type nothing can express.
_TYPE_NAME = re.compile(r"\A[A-Z][A-Z0-9_]*\Z")

_RUNNABLE: frozenset[Direction] = frozenset({"input", "output"})

# `chain._CALLER_STRING_LIMIT`, spelled here because importing `chain` from this
# module would close an import cycle: `detectors/__init__.py` imports `chain` and
# `authoring`. `tests/test_chain_identity.py` holds the two to each other.
_CALLER_STRING_LIMIT = 200


class PatternGuardrail:
    """A constraint over patterns, banned substrings and length.

    Every refusal in ``__init__`` is one shape of the same mistake: a check
    that is configured and would check less than the configuration says. That
    is the mistake ``build`` and ``build_chain`` already refuse five ways, and
    this constructor is a sixth door into the same room, reachable from a
    user's own configuration file.
    """

    kind: Kind = "constraint"

    def __init__(
        self,
        *,
        name: str,
        version: str,
        patterns: Mapping[str, str | re.Pattern[str]] | None = None,
        banned: Mapping[str, Sequence[str]] | None = None,
        limits: Limits | None = None,
        on_match: Decision | Mapping[Direction, Decision] = "deny",
        directions: frozenset[Direction] = frozenset({"input", "output"}),
        fold_case: bool = True,
        fold_confusables: bool = False,
    ) -> None:
        # The ceiling `GuardrailChain` refuses above, held HERE too, so a check
        # built through the documented path fails at the mistake rather than at
        # the chain. Without it, `PatternGuardrail(name="a" * 250, ...)`
        # constructed cleanly, satisfied the protocol, and then made
        # `build_chain` refuse the whole configuration: a factory that hands
        # back an object which only detonates later has moved the failure away
        # from the mistake, which is the argument `detectors.build`'s own
        # docstring makes against exactly this shape.
        for field, value in (("name", name), ("version", version)):
            if len(value) > _CALLER_STRING_LIMIT:
                raise ValueError(
                    f"{field} is {len(value)} characters, above the "
                    f"{_CALLER_STRING_LIMIT}-character ceiling; it is copied into the "
                    "provenance of every verdict this check produces, and a chain "
                    "refuses a guardrail that declares one this long"
                )
        self.name = name
        self.version = version
        self.directions = directions

        compiled: list[tuple[str, re.Pattern[str]]] = []
        for type_name, pattern in (patterns or {}).items():
            _require_type_name(type_name)
            if _nests_unbounded_repeats(pattern):
                raise ValueError(
                    f"the pattern for {type_name!r} NESTS unbounded repeats, which can "
                    "take exponential time on an input that fails to match. Rewrite it "
                    "so no unbounded repeat encloses another"
                )
            built = re.compile(pattern) if isinstance(pattern, str) else pattern
            # Catches a pattern that matches the empty string OUTRIGHT, such as
            # `x*` or `(a|)`. It does NOT catch a pattern gated by a lookbehind:
            # `built.search("")` probes only position zero of the empty string,
            # and `(?<=a)b*` returns None there, so it constructs cleanly and
            # then matches zero-width at (2, 2) against real content such as
            # "xa". No static probe over the pattern alone catches that in
            # general, so this check stays a narrower net than its message
            # might suggest. The real backstop is at match time, downstream in
            # the chain: it refuses to apply a redact whose finding span does
            # not satisfy `0 <= start < end <= len(content)`, and raises rather
            # than applying it, which fails closed per the caller-facing
            # contract in `GuardrailChain.run`'s docstring.
            if built.search("") is not None:
                raise ValueError(
                    f"the pattern for {type_name!r} matches the empty string, so it "
                    "would report a zero-width span the chain refuses to apply"
                )
            compiled.append((type_name, built))
        self._patterns = tuple(compiled)

        # Three fields: the type, the needle as `fold_confusables=False` would
        # match it, and its skeleton or None. Both needles are kept so the
        # option is a superset rather than a substitution; see `_banned_spans`.
        folded: list[tuple[str, str, str | None]] = []
        for type_name, substrings in (banned or {}).items():
            _require_type_name(type_name)
            if isinstance(substrings, (str, bytes)):
                # ValueError, not TypeError: every other refusal in this constructor is a
                # ValueError over a configuration mistake, and this one is the same mistake
                # (a value that iterates into something other than what was declared), not
                # a caller passing the wrong Python type. test_a_banned_entry_that_is_a_bare
                # _string_is_refused pins ValueError specifically.
                raise ValueError(  # noqa: TRY004
                    f"banned[{type_name!r}] must be a list of substrings, not the "
                    f"{type(substrings).__name__} {substrings!r}; a string is an "
                    "iterable of its own characters"
                )
            for substring in substrings:
                if not substring:
                    raise ValueError(
                        f"banned[{type_name!r}] carries an empty substring, which "
                        "matches everywhere"
                    )
                needle = substring.casefold() if fold_case else substring
                skeleton = None
                if fold_confusables:
                    skeleton = _skeleton(needle).text
                    if not skeleton:
                        # Unreachable against the 16.0.0 table, whose generator
                        # refuses a row mapping anything to nothing, and checked
                        # anyway: an empty needle matches at every offset and
                        # would report a zero-width span the chain refuses to
                        # apply, which is a detection that redacts nothing.
                        raise ValueError(
                            f"banned[{type_name!r}] carries {substring!r}, whose "
                            "confusable skeleton is empty, so it would match everywhere"
                        )
                # BOTH NEEDLES ARE KEPT, and the unfolded one is why
                # `fold_confusables` is a superset rather than a substitution.
                # See `_banned_spans`.
                folded.append((type_name, needle, skeleton))
        self._banned = tuple(folded)
        self._fold_case = fold_case
        self._fold_confusables = fold_confusables

        self._limits = limits

        if not (self._patterns or self._banned or self._limits):
            raise GuardrailUnavailableError(
                f"{name!r} was configured with no patterns, no banned substrings and no "
                "limits, so it would check nothing and allow any content at all"
            )

        # An option that selects nothing, refused for the reason every other
        # refusal in this constructor exists: a caller who wrote it meant
        # something by it. `fold_confusables` changes how BANNED SUBSTRINGS are
        # matched and nothing else, so with no banned substrings it is a setting
        # that reads as protection and buys none. Patterns are deliberately not
        # folded: a regex is matched against the view it was written for, and
        # running one over a skeleton would silently change what a character
        # class matches.
        if fold_confusables and not self._banned:
            raise ValueError(
                f"{name!r} sets fold_confusables and declares no banned substrings; the "
                "option only changes how banned substrings are matched, so here it would "
                "read as protection and buy none"
            )

        if not directions:
            raise GuardrailUnavailableError(
                f"{name!r} declares no direction it can run in, so every context would "
                f"skip it. Expected at least one of {sorted(_RUNNABLE)}"
            )

        # A member outside _RUNNABLE clears the check above (the set is not
        # empty) and clears on_match's resolution below (nothing there reads
        # against _RUNNABLE either), so a typo such as "inptu" built a
        # guardrail that runs, never matches a real Context, and is silent
        # about it. A Context only ever carries "input" or "output", so any
        # other member names a direction that can never arrive.
        unknown = sorted(directions - _RUNNABLE)
        if unknown:
            raise GuardrailUnavailableError(
                f"{name!r} declares direction(s) {unknown}, but a Context never "
                f"carries these (its direction is one of {sorted(_RUNNABLE)}), so "
                "the guardrail would never be checked for these directions"
            )

        if isinstance(on_match, str):
            resolved = {direction: on_match for direction in directions}
        else:
            missing = sorted(directions - set(on_match))
            if missing:
                raise GuardrailUnavailableError(
                    f"{name!r} declares directions {sorted(directions)} but on_match "
                    f"names no decision for {missing}; the alternative is a KeyError "
                    "from inside check, which fails closed and names nothing"
                )
            # The mirror mistake: a key on_match names that directions does not
            # declare. check now refuses that direction before it ever reads
            # on_match (see the guard at the top of check), so a policy for it
            # would never be consulted; resolving only the declared directions
            # here, rather than copying the mapping as given, is what makes that
            # true rather than merely likely. Refused rather than ignored,
            # because a mapping key a caller wrote on purpose (a real direction
            # they forgot to declare) or by typo (a direction that is not a
            # direction at all) both silently vanish otherwise, and the second
            # one is indistinguishable from the first without this check.
            extra = sorted(set(on_match) - directions)
            if extra:
                raise GuardrailUnavailableError(
                    f"{name!r} declares directions {sorted(directions)} but on_match "
                    f"also names {extra}, which this guardrail would never be asked "
                    "about; a policy for a direction it does not declare would be "
                    "silently dropped"
                )
            resolved = {direction: on_match[direction] for direction in directions}
        for direction, decision in sorted(resolved.items()):
            if decision not in ("redact", "deny"):
                raise ValueError(
                    f"on_match for {direction!r} must be 'redact' or 'deny', got "
                    f"{decision!r}. A check configured to allow on a match is a check "
                    "that runs and cannot act"
                )
        self._on_match: Mapping[Direction, Decision] = resolved

    def _matches(self, content: str) -> list[tuple[str, tuple[int, int]]]:
        """Every span every source claims, sorted by span. Nothing is dropped.

        SORTED BY SPAN is a precondition of what consumes this list, not
        tidiness. ``_merge`` tests each span against the running end of the
        region it is extending and looks no further back, so a list in any
        other order makes it emit regions that are wrong and ``_rewrite``
        writes those out. The three sources here scan the whole input
        independently and their results concatenate in source order, which is
        not span order.

        A tie on the start offset puts the shorter span first, and ``sorted``
        is stable, so equal spans keep the order the sources produced them in.
        """
        found = [
            (type_name, match.span())
            for type_name, pattern in self._patterns
            for match in _scan(pattern, content)
        ]
        found += self._banned_spans(content)
        if self._limits is not None:
            found += _limit_spans(content, self._limits)
        return sorted(found, key=lambda pair: pair[1])

    def _banned_spans(self, content: str) -> list[tuple[str, tuple[int, int]]]:
        """Banned substrings, matched over a folded view and reported in source.

        The view is where the match is found and the SOURCE is where the span
        points, because that is what the chain rewrites and what a corpus
        labels. The two are not the same length whenever folding changed a
        character's width, and the offset map is what keeps a match over a
        casefolded sharp s from spanning one character too many.

        Every occurrence, including overlapping ones, matching what ``_scan``
        does for patterns: the next search starts one past the previous match's
        start rather than at its end.

        **``fold_confusables`` SEARCHES TWO VIEWS AND UNIONS THEM, AND THAT IS
        THE FIX FOR A DENY IT TURNED INTO AN ALLOW.** Skeletonising the needle
        alone and the content whole is not distributive:
        ``skeleton(x + y) != skeleton(x) + skeleton(y)`` at a combining-mark
        boundary, because the skeleton ends in an NFD pass and NFD's canonical
        ordering is defined over a combining sequence rather than over a
        character. Measured on this file, with ``banned={"BAN": ["café"]}`` and
        the content ``"café"`` followed by U+0334 COMBINING TILDE OVERLAY:

            fold_confusables=False -> deny
            fold_confusables=True  -> allow

        because ``skeleton(needle)`` is ``cafe\u0301`` while the content folds
        to ``cafe\u0334\u0301``: the class-1 mark sorts in FRONT of the class-230
        one and lands inside the needle. 50 of the 112 marks in U+0300..U+036F
        do it, and so does every one of them after ``résumé``. The option a
        caller enables to stop a banned word being dodged by one substituted
        letter was making the rule evadable by one added mark.

        Two changes close it, and each answers a different half:

        - the UNFOLDED needle is searched too, so this option can only ever ADD
          matches to what ``fold_confusables=False`` finds. That is the property
          the docstring and `docs/conformance.md` both claim for it, and it now
          holds by construction rather than by argument.
        - the folded search TOLERATES canonical reordering in the needle's
          trailing combining marks, which is what `_find_folded` does. Without
          it a needle laundered BOTH ways at once -- one substituted Cyrillic
          letter and one added mark -- passes both views.

        THE TWO ARE REDUNDANT FOR THE SHAPE THAT WAS REPORTED AND NEITHER IS
        REDUNDANT IN GENERAL, which is a measurement rather than a hedge.
        Removing either one alone leaves the 112-mark sweep in
        ``tests/test_authoring.py`` passing; removing both fails it. Each has a
        shape the other cannot reach, and each has its own test.

        What is still open, and is disclosed rather than claimed away. A needle
        whose skeleton BEGINS with combining marks has the mirror of the problem
        the tolerance solves: a mark from the text BEFORE it can sort into the
        middle of its leading run, and nothing about what FOLLOWS the match can
        see that. The unfolded view catches it while the content carries the
        needle verbatim, which is the case
        ``test_a_needle_whose_own_marks_reorder_is_still_matched_verbatim``
        holds; it does not catch it once the needle is confusable-substituted as
        well, and then neither view does. Measured: needle U+0301 U+0334 e
        against ``x U+0316 U+0301 U+0334`` followed by CYRILLIC SMALL LETTER IE
        allows under both settings. No banned substring in this repository or
        its corpora begins with a combining mark, and a substring match over a
        normalised view cannot in general be made exact.
        """
        if not self._banned:
            return []
        view = casefold_view(content) if self._fold_case else fold(content, lambda ch: ch)
        folded = compose(view, _skeleton(view.text)) if self._fold_confusables else None
        found: list[tuple[str, tuple[int, int]]] = []
        seen: set[tuple[str, tuple[int, int]]] = set()
        for type_name, needle, skeleton in self._banned:
            for haystack, substring, tolerant in (
                (view, needle, False),
                (folded, skeleton, True),
            ):
                if haystack is None or substring is None:
                    continue
                start = 0
                while (
                    match := _find_folded(haystack.text, substring, start, tolerant)
                ) is not None:
                    claim = (type_name, haystack.span(*match))
                    if claim not in seen:
                        seen.add(claim)
                        found.append(claim)
                    start = match[0] + 1
        return found

    def check(self, content: str, context: Context) -> Verdict:
        """Refuses a direction this guardrail does not declare, before matching.

        ``GuardrailChain`` already filters on ``directions`` before it ever calls
        ``check``, but this class's own docstring contemplates "a caller holding
        one guardrail", so the chain is not the only caller and this method must
        hold the same line the chain holds for it.

        Without this guard, a context whose direction is not in ``directions``
        fell through to ``self._on_match[context.direction]`` below, but only
        past the early return on no match: clean content came back ``allow``,
        and matching content raised a bare ``KeyError`` naming neither the
        guardrail nor the direction. The ``allow`` is the worse of the two,
        because it reports that this content was checked and found clean, when
        this guardrail never declared itself able to check that direction at
        all. A guardrail asked about a direction it does not declare must not
        answer allow, deny or redact; it must refuse, the same way the
        constructor already refuses an ``on_match`` mapping that omits a
        declared direction, and for the same reason.
        """
        if context.direction not in self.directions:
            raise GuardrailUnavailableError(
                f"{self.name!r} was asked to check direction {context.direction!r} but "
                f"declares only {sorted(self.directions)}; answering would report that "
                "content was checked in a direction this guardrail never declared"
            )
        provenance = Provenance(kind="constraint", detector=self.name, version=self.version)
        found = self._matches(content)
        if not found:
            return Verdict("allow", None, [], provenance, saw(content))

        # One finding per match, each carrying its own original span, even
        # where several collapsed into one placeholder. The output loses that
        # detail by design; the audit record must not.
        findings = [Finding(type=type_name, span=span) for type_name, span in found]

        # The decision is read from the direction the CONTEXT carries, which
        # the chain has already checked is one this guardrail declares.
        if self._on_match[context.direction] == "deny":
            return Verdict("deny", None, findings, provenance, saw(content))

        # This guardrail's own view of the input, for a caller holding one
        # guardrail. A chain merges these spans with every other guardrail's
        # and rewrites once, through this same `_rewrite`.
        return Verdict("redact", _rewrite(content, found), findings, provenance, saw(content))


def _find_folded(haystack: str, needle: str, start: int, tolerant: bool) -> tuple[int, int] | None:
    """The next occurrence of ``needle`` in ``haystack`` at or after ``start``.

    With ``tolerant`` false this is ``str.find`` and nothing else.

    With it true, the needle's TRAILING combining marks are matched as a
    subsequence of the marks that follow the match rather than as a contiguous
    run. That is exactly the freedom NFD's canonical ordering takes: it is a
    stable sort by combining class over one combining sequence, so a mark from
    the text AFTER the needle can sort in front of one of the needle's own
    marks, and it can never cross a starter or reorder the needle's marks among
    themselves. Matching the tail as a subsequence accepts the interleaving and
    nothing else.

    The span it returns COVERS the interleaved characters, which is what a
    redaction needs: they sit inside the word the placeholder replaces.

    The needle is split at its last starter. With no starter at all -- a needle
    that is nothing but combining marks -- the tolerance would match almost
    anything, so the plain search is used instead.
    """
    if not tolerant:
        index = haystack.find(needle, start)
        return None if index == -1 else (index, index + len(needle))

    cut = 0
    for position, character in enumerate(needle):
        if unicodedata.combining(character) == 0:
            cut = position + 1
    if cut in (0, len(needle)):
        index = haystack.find(needle, start)
        return None if index == -1 else (index, index + len(needle))

    head, tail = needle[:cut], needle[cut:]
    index = haystack.find(head, start)
    while index != -1:
        cursor = index + len(head)
        remaining = 0
        while remaining < len(tail) and cursor < len(haystack):
            character = haystack[cursor]
            if unicodedata.combining(character) == 0:
                break
            cursor += 1
            if character == tail[remaining]:
                remaining += 1
        if remaining == len(tail):
            return (index, cursor)
        index = haystack.find(head, index + 1)
    return None


def _skeleton(text: str) -> _Folded:
    """The UTS #39 skeleton, imported where it is used and not at module import.

    ``jamjet_guardrails._unicode`` carries 231 KiB of generated tables and this
    module is reached from the package root, so a top-level import would make
    ``import jamjet_guardrails`` pay for them whether or not any caller ever
    sets ``fold_confusables``.
    ``tests/test_unicode.py::test_importing_the_package_loads_no_unicode_table``
    holds that, and it is the reason this one-line function exists rather than
    two copies of the import.
    """
    from jamjet_guardrails._unicode import skeleton

    return skeleton(text)


def _require_type_name(type_name: str) -> None:
    if not isinstance(type_name, str) or not _TYPE_NAME.match(type_name):
        raise ValueError(
            f"finding type {type_name!r} must match {_TYPE_NAME.pattern}; a type name "
            "is what a corpus labels and what a placeholder prints"
        )
