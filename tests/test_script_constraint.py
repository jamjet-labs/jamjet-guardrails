"""The script constraint: what it denies, and the much longer list it must not.

Every test here was watched failing under a named mutation before it was
trusted, and the mutation is recorded in the docstring. The mutations that
matter are the ones that make this check deny a language rather than a payload:
drop the wildcard rule and a comma is a finding; resolve Script instead of
Script_Extensions and an Arabic comma is not.

The refusals get a test each because there is no default here, so a
misconfiguration is the ordinary path into this check rather than an exotic
one, and each refusal is a different mistake a configuration file can make.
"""

from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import pytest

from jamjet_guardrails.detectors import build
from jamjet_guardrails.detectors.script_constraint import (
    SCRIPT_CONSTRAINT_TYPES,
    ScriptConstraintGuardrail,
)
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.eval.corpus import load_corpus
from jamjet_guardrails.eval.fixtures import options_for
from jamjet_guardrails.types import Context

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpora" / "script-constraint" / "in-repo.jsonl"
DOC = ROOT / "docs" / "conformance.md"

IN = Context(direction="input", origin="user")
OUT = Context(direction="output", origin="model")

# The fixture the published row is measured under, and the constraint most of
# these tests use, so a test that disagreed with the row would say so here.
FIXTURE = frozenset({"Latin", "Hiragana", "Katakana", "Han"})

# Named rather than written inline, because several tests reason about them and
# a literal in each is a literal that can drift from the others.
PROLONGED_SOUND_MARK = "ー"
IDEOGRAPHIC_COMMA = "、"
DEVANAGARI_DANDA = "।"
ARABIC_COMMA = "،"
PRIVATE_USE = ""


def _section() -> str:
    """The body of this check's H2 in `docs/conformance.md`, and nothing else.

    Scoped rather than file-wide for the reason `test_conformance_doc.py` scopes
    its own claim regions: the document names `Cyrillic`, `Greek`, `Arabic` and
    `Devanagari` in the prose about what this fixture LOCATES, so a whole-file
    or whole-section substring search would report a fixture naming any of them
    as correctly printed.
    """
    section = DOC.read_text(encoding="utf-8").split("## The script-constraint constraint", 1)[1]
    return section.split("\n## ", 1)[0]


def _guard(allowed: frozenset[str] = FIXTURE, on_match: str = "deny") -> ScriptConstraintGuardrail:
    return ScriptConstraintGuardrail(allowed_scripts=allowed, on_match=on_match)  # type: ignore[arg-type]


# ==========================================================================
# The refusals. There is no default, so every one of these is reachable from
# a configuration file rather than from a programming mistake.
# ==========================================================================


def test_building_with_no_allowed_scripts_is_refused() -> None:
    """The absence of a default IS the design, so it is tested as behaviour.

    Mutation: `if allowed_scripts is None:` to `if allowed_scripts is Ellipsis:`
    leaves `build("script-constraint")` handing back a guardrail whose allowed
    set is `None`, which then raises `TypeError` from inside `list()` at
    construction with no message about what was missing.
    """
    with pytest.raises(GuardrailUnavailableError) as caught:
        build("script-constraint")
    assert "allowed_scripts" in str(caught.value)
    assert "no default" in str(caught.value)


def test_a_bare_string_is_refused_rather_than_read_as_five_script_names() -> None:
    """`allowed_scripts: Latin` in a config file, not `[Latin]`.

    Mutation: dropping `str` from the isinstance tuple lets the value through
    to the unknown-name refusal, which then reports 'L', 'a', 'i', 'n' and 't'
    as five unknown scripts and never mentions the one mistake made.
    """
    with pytest.raises(GuardrailUnavailableError) as caught:
        build("script-constraint", allowed_scripts="Latin")
    assert "iterable of its own characters" in str(caught.value)


def test_something_that_cannot_be_iterated_is_refused_as_a_configuration_fault() -> None:
    """A TypeError here walks past every `except GuardrailUnavailableError` the
    caller wrapped their configuration seam in.

    Mutation: removing the try/except lets `list(7)` raise
    `TypeError: 'int' object is not iterable`, which is the error type this
    seam's contract does not carry.
    """
    with pytest.raises(GuardrailUnavailableError):
        build("script-constraint", allowed_scripts=7)


def test_an_empty_collection_is_refused() -> None:
    """`allowed_scripts: []` is a key that lost its value.

    Mutation: `if not requested:` to `if requested is None:` builds a check
    that reports every run of letters in every language, so its own
    deployment's traffic is denied on the first request.
    """
    with pytest.raises(GuardrailUnavailableError) as caught:
        build("script-constraint", allowed_scripts=frozenset())
    assert "denies all content" in str(caught.value)


def test_an_entry_that_is_not_a_string_is_refused_before_the_set_is_sorted() -> None:
    """Ordering, not decoration: `sorted(set({1, "Latin"}))` raises TypeError
    from inside the iteration guard above, which would report a list of names
    as something that cannot be iterated at all.

    Mutation: `not_names = []` reaches exactly that misreported TypeError.
    """
    with pytest.raises(ValueError, match="not strings"):
        build("script-constraint", allowed_scripts=["Latin", 7])


def test_a_script_the_tables_do_not_know_is_refused_and_the_valid_names_listed() -> None:
    """The four-letter code is the near miss that matters: `ScriptExtensions.txt`
    is written in them, so a caller reading the Unicode source writes `Latn`.

    Mutation: `unknown = []` builds a check whose allowed set holds a name no
    code point can ever resolve to, so the script it was meant to permit is
    denied everywhere and nothing says why.
    """
    with pytest.raises(ValueError) as caught:
        build("script-constraint", allowed_scripts={"Latn"})
    message = str(caught.value)
    assert "Latn" in message
    assert "'Latin'" in message and "'Cyrillic'" in message
    assert "long property values" in message


def test_naming_unknown_is_refused_with_its_own_reason() -> None:
    """`Unknown` is what `script_set` returns, so it is the one non-script a
    caller can reach by reading this package rather than by mistyping.

    Mutation: `if _UNKNOWN in requested:` to `if _UNKNOWN in known:` never
    fires, and the unknown-name refusal below catches it with a message saying
    the tables do not know a name they produce on every unassigned code point.
    """
    with pytest.raises(ValueError) as caught:
        build("script-constraint", allowed_scripts={"Latin", "Unknown"})
    assert "is not a script" in str(caught.value)


def test_an_on_match_that_cannot_act_is_refused() -> None:
    """Mutation: adding "allow" to the accepted tuple builds a check that runs
    over every request, finds what it was configured to find, and does
    nothing."""
    with pytest.raises(ValueError, match="redact"):
        build("script-constraint", allowed_scripts={"Latin"}, on_match="allow")


def test_a_refusal_never_echoes_a_caller_value_whole() -> None:
    """A value from a configuration file can BE the content, and every message
    here goes to whatever log wraps that seam.

    Mutation: making `_quoted` return `repr(value)` unconditionally turns this
    half-megabyte value into a half-megabyte error message, which is the defect
    `chain._refusal` measured and this ceiling exists to prevent.
    """
    with pytest.raises(GuardrailUnavailableError) as caught:
        build("script-constraint", allowed_scripts="Latin" * 100_000)
    assert len(str(caught.value)) < 500, "the refusal echoed the value it refused"


def test_a_constraint_of_wildcards_alone_is_permitted_and_not_refused() -> None:
    """Silly is not the same as refused, and the difference is where the value
    came from: an empty collection is a configuration key that lost its value,
    and this is a set somebody typed.

    What it then does is the same check the empty collection would have made,
    because the wildcards already pass unconditionally: every letter fires.
    That is stated here as behaviour rather than left to the reader.
    """
    guard = _guard(frozenset({"Common", "Inherited"}))
    assert guard.check("...", IN).decision == "allow"
    assert guard.check("plain latin words", IN).decision == "deny"


def test_the_allowed_scripts_property_does_not_report_wildcards_nobody_asked_for() -> None:
    """An audit reader who saw `Common` in a set they never typed would
    reasonably conclude the configuration had been rewritten under them."""
    assert _guard().allowed_scripts == FIXTURE


# ==========================================================================
# The signal.
# ==========================================================================


def test_a_run_of_a_disallowed_script_is_one_finding_over_the_whole_run() -> None:
    """Mutation: `elif start is None: start = index` to `else: start = index`
    reopens the run on every character, so the reported span covers the last
    character of the run and the five before it are left standing in a
    redaction."""
    verdict = _guard().check("The invoice was approved by Иванов.", IN)
    assert verdict.decision == "deny"
    assert [(f.type, f.span) for f in verdict.findings] == [("DISALLOWED_SCRIPT", (28, 34))]


def test_two_disallowed_scripts_side_by_side_are_one_run_and_not_two() -> None:
    """The run is maximal over disallowed-ness, never over one script. Greek
    followed immediately by Cyrillic is one stretch of content outside the
    constraint, and splitting it would report two facts where there is one and
    produce two placeholders where a caller expects one.

    Mutation: the same `else: start = index` mutation leaves (1, 2) here.
    """
    verdict = _guard().check("αб", IN)
    assert [f.span for f in verdict.findings] == [(0, 2)]


def test_an_allowed_code_point_between_two_runs_splits_them() -> None:
    """The space is `Common` and passes, so "Привет мир" is two findings.

    Mutation: dropping the `spans.append((start, index))` reset inside the loop
    merges them into one span that covers the space as well, and a redaction
    then eats a character the check has no objection to.
    """
    verdict = _guard().check("Привет мир", IN)
    assert [f.span for f in verdict.findings] == [(0, 6), (7, 10)]


def test_a_run_that_reaches_the_end_of_the_content_is_reported() -> None:
    """The half of the input a payload is appended to.

    Mutation: deleting the final `if start is not None` block reports nothing
    at all for this content.
    """
    verdict = _guard().check("The branch is Кириллица", IN)
    assert [f.span for f in verdict.findings] == [(14, 23)]


def test_a_disallowed_code_point_mid_token_is_found() -> None:
    """One Cyrillic letter inside a Latin word, which is what a homoglyph
    substitution looks like from this check's side."""
    verdict = _guard().check("Sign in at pаypal.example now.", IN)
    assert [f.span for f in verdict.findings] == [(12, 13)]


def test_an_unassigned_code_point_never_passes() -> None:
    """Fail-closed, and disclosed as such: a private-use code point and any
    code point assigned after the pinned tables resolve to `Unknown`.

    Mutation: adding "Unknown" to `_WILDCARDS` allows both, so a build that
    cannot name a code point passes it unexamined.
    """
    assert _guard().check(f"prompt {PRIVATE_USE} here", IN).decision == "deny"
    assert _guard().check("buffer ﷐ flushed", IN).decision == "deny"


# ==========================================================================
# The wildcards, which are the whole false-positive defence.
# ==========================================================================

WILDCARD_TEXTS = [
    "Total: $1,234.56; VAT 20%. See notes (page 7) [rev 3].",
    "Compute ∑ x ≤ ∞ for every x ∈ S.",
    "Copyright © 2026, all rights reserved ®.",
    "Family photo \U0001f468‍\U0001f469‍\U0001f467 attached.",
    "Reacted with ❤️ and \U0001f44d\U0001f3fd today.",
    "Decomposed: café and naïve.",
    "Bold math \U0001d400\U0001d401\U0001d402 in the formula.",
    "Quotes: “curly”, ‘single’ and «guillemets».",
    "A zero-width\u200bspace hides here.",
]


@pytest.mark.parametrize("text", WILDCARD_TEXTS, ids=range(len(WILDCARD_TEXTS)))
def test_common_and_inherited_pass_under_a_latin_only_constraint(text: str) -> None:
    """The narrowest constraint anybody writes, over punctuation, currency,
    mathematics, emoji, joiners, variation selectors and combining marks.

    Mutation: `self._allowed | _WILDCARDS` to `self._allowed` denies every one
    of these, which is the check being switched off in the first week.
    """
    assert _guard(frozenset({"Latin"})).check(text, IN).decision == "allow"


def test_the_zero_width_space_is_not_this_checks_business() -> None:
    """Deliberate, and the reason is in `injection-structural`: a zero-width
    run is a structural signal that check publishes a number for, and reporting
    it here as a script violation would double-count one payload and make this
    row's precision depend on that one's."""
    assert _guard().check("A zero-width\u200bspace hides here.", IN).decision == "allow"


# ==========================================================================
# Script_Extensions resolution, which is where a port goes wrong quietly.
# ==========================================================================


def test_the_prolonged_sound_mark_is_not_a_wildcard() -> None:
    """The case the fixture was built around. `U+30FC` has Script `Common` and
    Script_Extensions {Hiragana, Katakana}, so under a Latin-only constraint it
    is a finding, and only a port that resolves Script_Extensions can see that.

    Mutation: in `_unicode/__init__.py`, `if position >= 0:` to
    `if position < 0:` in the Script_Extensions block makes `script_set`
    resolve Script alone, and this mark then reads as a wildcard and passes.
    """
    assert _guard(frozenset({"Latin"})).check(PROLONGED_SOUND_MARK, IN).decision == "deny"
    assert _guard(frozenset({"Katakana"})).check(PROLONGED_SOUND_MARK, IN).decision == "allow"
    assert _guard(frozenset({"Hiragana"})).check(PROLONGED_SOUND_MARK, IN).decision == "allow"


def test_cjk_punctuation_passes_by_extension_and_not_by_being_common() -> None:
    """`U+3001` has no `Common` in its resolved set at all, so it passes under
    the fixture only because `Han` is one of its extensions.

    Mutation: the same Script-first mutation makes it a wildcard, which passes
    it for the wrong reason and under every constraint.
    """
    assert _guard().check(IDEOGRAPHIC_COMMA, IN).decision == "allow"
    assert _guard(frozenset({"Latin"})).check(IDEOGRAPHIC_COMMA, IN).decision == "deny"


@pytest.mark.parametrize(
    ("mark", "name"),
    [(DEVANAGARI_DANDA, "danda"), (ARABIC_COMMA, "arabic comma"), ("ـ", "tatweel")],
)
def test_punctuation_whose_script_is_common_still_fires_when_its_extensions_are_not(
    mark: str, name: str
) -> None:
    """The recall side of the same rule, and the one the fixture measures.

    Mutation: the Script-first mutation allows all three under the fixture,
    because each has Script `Common` and Script_Extensions naming only scripts
    the fixture does not.
    """
    assert _guard().check(f"clause {mark} ends", IN).decision == "deny", name


# ==========================================================================
# Decisions, directions and the rewrite.
# ==========================================================================


def test_the_default_is_deny() -> None:
    guard = build("script-constraint", allowed_scripts={"Latin"})
    assert guard.check("б", IN).decision == "deny"


def test_redact_replaces_each_run_with_one_placeholder() -> None:
    """Available because the spans are exact, and unusual because it is rarely
    what a caller wants: what is left of a Russian sentence with the Cyrillic
    removed is not what anybody wrote."""
    verdict = _guard(on_match="redact").check("Привет мир!", IN)
    assert verdict.decision == "redact"
    assert verdict.content == "[REDACTED:DISALLOWED_SCRIPT] [REDACTED:DISALLOWED_SCRIPT]!"


def test_it_runs_in_both_directions() -> None:
    """A model answering in a script the caller cannot render or moderate is
    the same fact as a retrieved page arriving in one, seen from the far end."""
    guard = _guard()
    assert guard.directions == frozenset({"input", "output"})
    assert guard.check("б", OUT).decision == "deny"


def test_empty_content_allows() -> None:
    assert _guard().check("", IN).decision == "allow"


def test_it_declares_the_one_type_it_reports() -> None:
    assert SCRIPT_CONSTRAINT_TYPES == frozenset({"DISALLOWED_SCRIPT"})
    verdict = _guard().check("б", IN)
    assert {f.type for f in verdict.findings} <= SCRIPT_CONSTRAINT_TYPES


def test_no_finding_carries_a_confidence() -> None:
    """It is a constraint: it matches or it does not."""
    assert [f.confidence for f in _guard().check("бв", IN).findings] == [None]


# ==========================================================================
# The two numbers the conformance document publishes about this check.
# Recomputed here, because a number in prose that counts a thing in code is a
# claim like any other.
# ==========================================================================


@lru_cache(maxsize=1)
def _range_starts() -> tuple[int, ...]:
    """The start of every script range, built once.

    `_script_only` is called for every code point `_extension_effects` walks, and
    it rebuilt this tuple from 979 ranges on each of them. Copilot found it on
    the pull request. The tables are pinned and byte-identity gated, so there is
    one answer.
    """
    from jamjet_guardrails._unicode.scripts import SCRIPT_RANGES

    return tuple(start for start, _, _ in SCRIPT_RANGES)


def _script_only(code: int) -> str:
    """What `script_set` would return if it resolved Script and not extensions.

    Written out rather than mutated in, so the cost of the rule this check
    depends on can be measured on every run instead of once by hand.
    """
    from jamjet_guardrails._unicode.scripts import SCRIPT_RANGES

    position = bisect_right(_range_starts(), code) - 1
    if position >= 0:
        _, end, script = SCRIPT_RANGES[position]
        if code <= end:
            return script
    return "Unknown"


def _extension_effects() -> tuple[int, int]:
    """(code points where extensions buy recall, where they buy precision).

    Over every code point any Script_Extensions range covers, under the
    published fixture: how many would a Script-only port allow that this one
    denies, and how many would it deny that this one allows.
    """
    from jamjet_guardrails._unicode import script_set
    from jamjet_guardrails._unicode.scripts import EXTENSION_RANGES

    effective = FIXTURE | frozenset({"Common", "Inherited"})
    recall = precision = 0
    for start, end, _ in EXTENSION_RANGES:
        for code in range(start, end + 1):
            script_passes = _script_only(code) in effective
            resolved_passes = bool(script_set(chr(code)) & effective)
            recall += script_passes and not resolved_passes
            precision += resolved_passes and not script_passes
    return recall, precision


def test_the_document_publishes_the_measured_cost_of_resolving_extensions() -> None:
    """Both halves, and the second is the interesting one: under this fixture
    Script_Extensions resolution buys recall and never precision, so a port
    that skips it does not fire on anything extra, it goes quiet."""
    recall, precision = _extension_effects()
    assert recall > 0 and precision == 0, (recall, precision)
    section = _section()
    assert f"{recall} code points" in section, f"the measured figure is {recall}"
    assert "never precision" in section


def test_the_document_publishes_the_number_of_allow_cases_the_wildcards_carry() -> None:
    """The same shape `injection-structural` publishes for its exemptions: a
    rule is only worth documenting if the cost of dropping it is measured."""
    from jamjet_guardrails._unicode import script_set

    corpus = load_corpus(CORPUS, name="script-constraint/in-repo")
    allows = [case for case in corpus.cases if case.expect_decision == "allow"]
    riding = [
        case.id
        for case in allows
        if any(not (script_set(character) & FIXTURE) for character in case.text)
    ]
    section = _section()
    assert f"{len(riding)} of its {len(allows)} `allow` cases" in section, (
        f"the measured figures are {len(riding)} of {len(allows)}"
    )


def test_the_conformance_document_prints_the_fixture_the_row_was_measured_under() -> None:
    """The fixture is the whole meaning of this row, so it is printed beside it
    and checked against the table the harness actually builds from.

    The whole rendered line, not the names one at a time. Checking membership
    per name passes for a fixture that swapped `Han` for `Cyrillic`, because
    the prose two paragraphs down names Cyrillic as a script this fixture
    LOCATES; the rendered line is exact in both directions, so a script added
    to the fixture and a script dropped from it both fail here.

    Mutation: swapping `Han` for `Cyrillic` in
    `jamjet_guardrails.eval.fixtures` was survived by the per-name form and is
    killed by this one.
    """
    fixture = options_for("script-constraint")
    allowed = fixture["allowed_scripts"]
    assert isinstance(allowed, frozenset)
    printed = ", ".join(f'"{script}"' for script in sorted(allowed))
    section = _section()
    assert f"allowed_scripts: [{printed}]" in section, (
        f"the document does not print the fixture, which renders as [{printed}]"
    )
    assert "on_match: deny" in section
