"""Labelled corpora: the evidence behind every published number.

Every precision and recall figure this library publishes is measured on one of
these files, so a corpus that loads wrong is a published number that is wrong,
and nothing downstream can tell the difference. The loader therefore has no
shape whose bad value resolves to something reasonable: a line it cannot parse,
a field it was not given, a field of the wrong type, a key it does not
recognise, an id it has already seen, a span that does not fit the text it
points into, a direction or decision the runtime cannot produce, a file that
mixes sources, and a file that yields no cases are each refused by name and by
line number.

The division of labour is deliberate. The loader owns everything that needs a
line number -- framing, JSON, key sets, duplicate ids, one-file-one-source,
emptiness. The value types own everything about a value's shape, so a Corpus
assembled in code (Task 10 does exactly that) is held to the same rules as one
read off disk rather than to none.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from jamjet_guardrails.types import Decision, Direction

_REQUIRED = ("id", "text", "direction", "expect", "source", "license")

# `expect` and each finding are checked for their EXACT key set, not merely for
# the presence of the ones we read. Both omissions move a published number in
# the flattering direction and neither is visible in the file:
#
#   - `findings` absent meaning "expects nothing" turns every real detection on
#     that case into a false positive, so one typo moves precision.
#   - `span` absent meaning "match on type alone" is a WEAKER bar than the exact
#     span rule the published number claims (Task 10 matches on type when the
#     span is None), so a forgotten key quietly buys both precision and recall.
#
# Writing `"span": null` asks for the weaker bar out loud. Rejecting unknown
# keys is the other half: with every field required, an unrecognised key is
# either a misspelling of a required one or something the schema does not
# understand, and a schema that ignores what it does not understand cannot tell
# those apart.
_EXPECT_KEYS = ("decision", "findings")
_FINDING_KEYS = ("type", "span")

# Listed literally and deliberately NOT derived from get_args(...) -- the reason
# types.py records for Kind and detectors/__init__.py for Direction. Both
# Literals are designed to grow, and deriving the accepted set would admit a new
# value into a corpus before evaluate() or the report knew how to score it.
# tests/test_corpus.py pins these two tuples against get_args, so adding to
# either Literal turns into a red test HERE rather than into a corpus carrying
# an expectation nothing can honour.
_DIRECTIONS: tuple[Direction, ...] = ("input", "output")
_DECISIONS: tuple[Decision, ...] = ("allow", "redact", "deny")


class CorpusError(Exception):
    """A corpus is malformed. Read from a file, it names the offending line.

    One error type covers both doors -- a file the loader read and a Corpus
    assembled in code -- rather than the ValueError/TypeError split a caller
    would then have to catch both halves of to be safe. Task 8 spent three fix
    rounds on exactly that shape: a malformed value escaping a seam as a bare
    TypeError nobody was catching. Task 13's CLI catches CorpusError, so this
    keeps every corpus problem inside the one net it already holds.
    """


def _require_text(value: object, field: str) -> None:
    """Refuse anything that is not a non-empty string.

    Empty counts as absent on purpose. An empty ``license`` is an unknown
    licence, and an empty ``text`` is a case that can neither match nor fail to
    match while still counting toward the "at least 20 cases" floor a published
    number rests on.

    The rejected value is NEVER echoed, only its type. ``text`` carries corpus
    content, which for these corpora is by design PII and credentials, and the
    error goes to a CI log. ``detectors/secrets.py`` keeps matched text out of
    the audit record for exactly this reason; a loader that prints the offending
    value undoes that on the first malformed line. The caller's prefix already
    names the file and the line, which is what locating the row actually needs.
    """
    if isinstance(value, str) and value:
        return
    shape = "an empty string" if isinstance(value, str) else type(value).__name__
    raise CorpusError(f"{field} must be a non-empty string, got {shape}")


@dataclass(frozen=True, slots=True)
class ExpectedFinding:
    """One labelled detection. ``span`` is None to match on type alone."""

    type: str
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _require_text(self.type, "finding type")
        span: object = self.span
        if span is None:
            return
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise CorpusError(f"span must be a two-element [start, end] or null, got {span!r}")
        start, end = span
        # bool BEFORE int, because bool IS an int: `[true, 26]` would otherwise
        # slice as [1, 26] and produce a false negative that reads exactly like
        # a detector bug.
        if isinstance(start, bool) or not isinstance(start, int):
            raise CorpusError(f"span start must be an integer, got {start!r}")
        if isinstance(end, bool) or not isinstance(end, int):
            raise CorpusError(f"span end must be an integer, got {end!r}")
        if start < 0:
            # text[-1:3] is legal Python and silently the wrong stretch of text.
            raise CorpusError(f"span start must not be negative, got {start}")
        if start >= end:
            # A zero-width span is a non-detection wearing a detection's clothes.
            raise CorpusError(f"span must cover at least one character, got ({start}, {end})")
        object.__setattr__(self, "span", (start, end))


@dataclass(frozen=True, slots=True)
class Case:
    """One labelled example: the text, and what a guardrail should say about it."""

    id: str
    text: str
    direction: Direction
    expect_decision: Decision
    expect_findings: tuple[ExpectedFinding, ...]
    source: str
    license: str

    def __post_init__(self) -> None:
        _require_text(self.id, "id")
        _require_text(self.text, "text")
        _require_text(self.source, "source")
        _require_text(self.license, "license")
        if self.direction not in _DIRECTIONS:
            raise CorpusError(
                f"direction must be one of {list(_DIRECTIONS)}, got {self.direction!r}"
            )
        if self.expect_decision not in _DECISIONS:
            raise CorpusError(
                f"expected decision must be one of {list(_DECISIONS)}, got {self.expect_decision!r}"
            )
        # A bare string is iterable, so tuple("EMAIL") is five one-character
        # findings rather than a TypeError.
        if not isinstance(self.expect_findings, (list, tuple)):
            raise CorpusError(
                f"expected findings must be a list, got {type(self.expect_findings).__name__}"
            )
        findings = tuple(self.expect_findings)
        for finding in findings:
            if not isinstance(finding, ExpectedFinding):
                raise CorpusError(
                    f"every expected finding must be an ExpectedFinding, "
                    f"got {type(finding).__name__}"
                )
            # Checked here rather than in ExpectedFinding because only the Case
            # knows the text. An end past it is the commonest corpus bug and it
            # presents as a detector regression, not as a data error.
            if finding.span is not None and finding.span[1] > len(self.text):
                raise CorpusError(
                    f"span {list(finding.span)} for {finding.type!r} runs past the end of "
                    f"a {len(self.text)}-character text"
                )
        # Copy first, validate the COPY, then publish it. `findings` above is
        # already a tuple of the caller's list, which is what makes the checks
        # meaningful: a caller who kept the list could otherwise empty it
        # afterwards and walk the checks above around themselves, and an
        # iterator would be exhausted by the first of them.
        #
        # This is the pattern types.Verdict and eval.metrics.Evaluation now
        # follow. Both once said "validate first, then freeze" and cited this
        # code for it, while this code did the opposite; both were storing an
        # exhausted iterator's worth of nothing until 2026-08-30.
        object.__setattr__(self, "expect_findings", findings)


@dataclass(frozen=True, slots=True)
class Corpus:
    """One labelled file's worth of cases, and the identity of what they are."""

    name: str
    source: str
    license: str
    cases: tuple[Case, ...]

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.source, "source")
        _require_text(self.license, "license")
        if not isinstance(self.cases, (list, tuple)):
            raise CorpusError(f"cases must be a list, got {type(self.cases).__name__}")
        cases = tuple(self.cases)
        if not cases:
            # Metrics(0, 0, 0) scores precision 1.0 and recall 1.0 by the
            # empty-set convention, so an empty corpus does not fail loudly
            # downstream -- it publishes a perfect number measured on nothing.
            raise CorpusError("a corpus must hold at least one case")
        for case in cases:
            if not isinstance(case, Case):
                raise CorpusError(f"every corpus entry must be a Case, got {type(case).__name__}")
        repeated = sorted(cid for cid, n in Counter(case.id for case in cases).items() if n > 1)
        if repeated:
            # The version sorts by id, so a repeated id is also a sort key that
            # cannot order two rows.
            raise CorpusError(f"duplicate case ids: {repeated}")
        sources = sorted({case.source for case in cases})
        if sources != [self.source]:
            raise CorpusError(
                f"corpus source {self.source!r} does not match its cases' {sources}; "
                "the source is reported next to the number and may not be decorative"
            )
        licences = sorted({case.license for case in cases})
        if ", ".join(licences) != self.license:
            raise CorpusError(
                f"corpus license {self.license!r} does not match its cases' {licences}"
            )
        object.__setattr__(self, "cases", cases)

    @property
    def version(self) -> str:
        """First 12 hex of SHA-256 over every case's content. Derived, never declared.

        The digest covers what is measured AND what the measurement is graded
        against: id, text, direction, expected decision, and every expected
        finding in order. Adding a case moves it, re-texting one moves it, and
        so does RELABELLING one.

        ``direction`` is in there because ``evaluate`` passes it straight into
        ``guardrail.check``, which makes it measured input in the same sense the
        text is. It is latent while both bundled detectors take a Context and
        never read it, and it stops being latent for the first direction-aware
        check -- by which time ``baselines.json`` exists and the input set can no
        longer be changed for free.

        That last one is the point. Changing an expectation to match whatever
        the detector currently does is the cheapest possible way to turn a
        failing check green, and under a digest of id and text alone it is
        invisible: the Task 12 baseline key does not move, so the new labels are
        graded against the old labels' recorded score. "Editing a corpus
        produces a new baseline key" has to hold for the edit most worth making
        dishonestly, not only for the ones that are obvious in a diff.

        Cases are sorted by id, so reordering the file changes nothing. Findings
        are NOT sorted: their order is content like any other, and treating a
        reorder as a change errs toward asking for a new baseline entry that was
        not strictly needed rather than reusing one that was.

        Fields are LENGTH-PREFIXED rather than delimited. A separator byte that
        can occur inside a field does not separate: with ``id + b"\\x00" +
        text``, the cases ``("a", "b\\x00c")`` and ``("a\\x00b", "c")`` produce
        identical digests, so two different corpora would share one version and
        therefore one baseline entry. Length prefixes are injective, so the
        question cannot arise for any content at all.

        The finding COUNT is emitted too, which makes each case's record
        self-delimiting. It is belt and braces today -- no two corpora can
        currently collide without it, because a case contributes an odd number
        of fields and a boundary shift would have to land a span rendering
        ("none" or "start:end") in a decision slot, which types.Decision has no
        value for. That argument is true and fragile: it stops holding the day
        Decision grows a value a span can also render as. The count means the
        encoding does not rest on it. No test can see this field, because with
        it there is no collision to write a test about.
        """
        digest = hashlib.sha256()
        for case in sorted(self.cases, key=lambda c: c.id):
            fields = [
                case.id,
                case.text,
                case.direction,
                case.expect_decision,
                str(len(case.expect_findings)),
            ]
            for finding in case.expect_findings:
                fields.append(finding.type)
                # None is RENDERED, not skipped, and the reason is injectivity,
                # not visibility. Skipping it would still move the version when a
                # span is dropped, because the count keeps each record
                # self-delimiting and the field list shortens. What skipping
                # loses is the ability to tell EMAIL(None) + "0:1"(0,1) from
                # EMAIL(0,1) + "0:1"(None): different expectations, scored
                # differently by Task 10, flattened to one field list. Pinned by
                # test_a_finding_type_that_renders_like_a_span_does_not_collide.
                fields.append(
                    "none" if finding.span is None else f"{finding.span[0]}:{finding.span[1]}"
                )
            for value in fields:
                encoded = value.encode("utf-8")
                digest.update(str(len(encoded)).encode("ascii"))
                digest.update(b":")
                digest.update(encoded)
        return digest.hexdigest()[:12]


def _check_keys(row: Mapping[str, object], allowed: tuple[str, ...], where: str, what: str) -> None:
    missing = [key for key in allowed if key not in row]
    if missing:
        raise CorpusError(f"{where} is missing {what} field(s) {missing}")
    unknown = sorted(str(key) for key in row if key not in allowed)
    if unknown:
        raise CorpusError(
            f"{where} has unrecognised {what} field(s) {unknown}; expected exactly "
            f"{list(allowed)}. A misspelled field is otherwise indistinguishable "
            "from a field the schema simply ignores."
        )


def _object_with_unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse a repeated JSON key rather than silently keeping the last one.

    json's default is last-wins, so ``{"license": "MIT", ..., "license":
    "Apache-2.0"}`` loads without complaint as Apache-2.0. That cannot fudge a
    number by itself -- whichever value wins is the one hashed into the version
    -- but it defeats review-by-diff, which is the control this module exists to
    provide: two lines that read differently and load identically are exactly
    what a reviewer cannot catch.

    Passed as ``object_pairs_hook``, so it runs on every object in the line, not
    only the outermost one.
    """
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise CorpusError(
                f"repeats the JSON key {key!r}; last-wins parsing would hide the "
                "difference between two lines that do not read the same"
            )
        seen.add(key)
    return dict(pairs)


def _finding_from(raw: object, where: str) -> ExpectedFinding:
    if not isinstance(raw, dict):
        raise CorpusError(f"{where} is not an object, got {type(raw).__name__}")
    _check_keys(raw, _FINDING_KEYS, where, "finding")
    try:
        return ExpectedFinding(type=raw["type"], span=raw["span"])
    except CorpusError as exc:
        raise CorpusError(f"{where}: {exc}") from None


def _case_from_row(raw: object, path: Path, lineno: int) -> Case:
    where = f"{path}: line {lineno}"
    if not isinstance(raw, dict):
        # A JSON array reaches `"license" not in row` as a membership test over
        # VALUES, so it passes a naive required-field check for any element that
        # happens to spell a field name.
        raise CorpusError(f"{where} is not a JSON object, got {type(raw).__name__}")
    _check_keys(raw, _REQUIRED, where, "case")

    expect = raw["expect"]
    if not isinstance(expect, dict):
        raise CorpusError(f"{where}: expect must be an object, got {type(expect).__name__}")
    _check_keys(expect, _EXPECT_KEYS, where, "expect")

    findings = expect["findings"]
    if not isinstance(findings, list):
        raise CorpusError(f"{where}: expect.findings must be a list, got {type(findings).__name__}")

    parsed = tuple(
        _finding_from(finding, f"{where}: expect.findings[{index}]")
        for index, finding in enumerate(findings)
    )
    try:
        return Case(
            id=raw["id"],
            text=raw["text"],
            direction=raw["direction"],
            expect_decision=expect["decision"],
            expect_findings=parsed,
            source=raw["source"],
            license=raw["license"],
        )
    except CorpusError as exc:
        raise CorpusError(f"{where}: {exc}") from None


def load_corpus(path: Path, name: str) -> Corpus:
    """Read one JSONL corpus file, or hand back no corpus at all.

    Raises:
        CorpusError: for anything wrong with the file's CONTENT -- bytes that
            are not UTF-8, an unparseable line, a repeated JSON key, a line that
            is not an object, a missing or unrecognised field, a value of the
            wrong type, a span that does not fit its text, a repeated id, more
            than one ``source``, or a file that yields no cases. Every message
            names the file, and the line wherever the fault belongs to one (a
            file with no cases, and a file that is not UTF-8, do not). No
            message echoes a field's value: see ``_require_text``.
        FileNotFoundError: if the file is absent. Deliberately not wrapped: an
            absent corpus and a malformed one are different faults, and Task 13
            catches CorpusError specifically to report the latter.
    """
    path = Path(path)
    numbered: list[tuple[int, Case]] = []
    first_line_of: dict[str, int] = {}

    # JSONL is newline-delimited, so split on "\n" rather than using
    # splitlines(), which also breaks on U+2028, U+2029 and U+0085. Those are
    # ordinary characters inside a JSON string and do occur in real text;
    # splitting on them would tear one valid case into two unparseable halves.
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # A stray byte is a malformed corpus, not a traceback for the caller to
        # interpret. Without this it escapes as UnicodeDecodeError, past a
        # docstring promising CorpusError and past Task 13's except clause. The
        # byte and its offset are named; the surrounding content is not.
        raise CorpusError(f"{path}: is not valid UTF-8: {exc.reason} at byte {exc.start}") from None

    for lineno, raw in enumerate(content.split("\n"), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw, object_pairs_hook=_object_with_unique_keys)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}: line {lineno} is not valid JSON: {exc}") from None
        except CorpusError as exc:
            # Raised by the pairs hook, which has no idea which line it is on.
            raise CorpusError(f"{path}: line {lineno} {exc}") from None

        case = _case_from_row(row, path, lineno)
        first = first_line_of.get(case.id)
        if first is not None:
            raise CorpusError(
                f"{path}: line {lineno} has duplicate id {case.id!r}, first seen on line {first}"
            )
        first_line_of[case.id] = lineno
        numbered.append((lineno, case))

    if not numbered:
        # Covers an empty file, a file of only blank lines, and a file of only
        # whitespace. Returning an empty corpus would publish a perfect score
        # measured on nothing.
        raise CorpusError(f"{path}: contains no cases")

    first_lineno, first_case = numbered[0]
    for lineno, case in numbered:
        if case.source != first_case.source:
            raise CorpusError(
                f"{path}: line {lineno} declares source {case.source!r} but line "
                f"{first_lineno} declares {first_case.source!r}; one file, one source, "
                "so in-repo and third-party numbers are never merged into one score"
            )

    cases = tuple(case for _, case in numbered)
    return Corpus(
        name=name,
        source=first_case.source,
        # Every distinct licence, not the first case's. Taking cases[0] would
        # mislabel the provenance of every other case in a file that mixes them.
        license=", ".join(sorted({case.license for case in cases})),
        cases=cases,
    )
