"""The authoring primitive: the one place a pattern-shaped check is built.

Everything here is about a check that would check LESS than it was configured
to. That is the failure this package exists to prevent, and a primitive handed
to contributors and to users writing runtime rules is where a new instance of
it would arrive.
"""

from __future__ import annotations

import re

import pytest

from jamjet_guardrails.authoring import (
    _CALLER_STRING_LIMIT,
    Limits,
    PatternGuardrail,
    _limit_spans,
    _nests_unbounded_repeats,
)
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import Guardrail


def test_content_at_the_limit_is_not_over_it() -> None:
    """The boundary in the direction that fails open. `>=` here would report a
    limit on content that respected it."""
    assert _limit_spans("x" * 40, Limits(max_chars=40)) == []


def test_content_one_character_past_the_limit_spans_the_excess() -> None:
    assert _limit_spans("x" * 41, Limits(max_chars=40)) == [("LENGTH_LIMIT", (40, 41))]


def test_the_span_runs_from_the_first_excess_character_to_the_end() -> None:
    assert _limit_spans("x" * 50, Limits(max_chars=40)) == [("LENGTH_LIMIT", (40, 50))]


def test_bytes_are_counted_as_utf8_not_as_characters() -> None:
    """Four characters, twelve bytes. A limit of 8 bytes is exceeded at the
    third character, and a character count would not notice at all."""
    content = "你好世界"
    assert _limit_spans(content, Limits(max_bytes=8)) == [("LENGTH_LIMIT", (2, 4))]
    assert _limit_spans(content, Limits(max_chars=8)) == []


def test_the_byte_span_starts_at_the_character_that_crossed_the_limit() -> None:
    assert _limit_spans("abécd", Limits(max_bytes=3)) == [("LENGTH_LIMIT", (2, 5))]


def test_byte_limit_boundary_exactly_at_the_limit() -> None:
    """A character whose cumulative byte total exactly equals the limit separates
    the two boundary forms. The test 'abécd' with max_bytes=3 jumps from 2 bytes
    to 4 bytes at the third character, so neither `>` nor `>=` sees the boundary.
    This case lands exactly on 4 bytes, where `>` and `>=` disagree: `>` fires at
    the next character (index 3), while `>=` fires at the character that reached
    the limit (index 2). The incorrect form truncates a character the limit did
    not object to."""
    assert _limit_spans("aébc", Limits(max_bytes=4)) == [("LENGTH_LIMIT", (3, 4))]


def test_lines_are_counted_and_the_span_starts_at_the_first_excess_line() -> None:
    assert _limit_spans("a\nb\nc", Limits(max_lines=2)) == [("LENGTH_LIMIT", (4, 5))]


def test_content_at_the_line_limit_is_not_over_it() -> None:
    assert _limit_spans("a\nb", Limits(max_lines=2)) == []


def test_a_trailing_newline_does_not_open_a_line_that_is_not_there() -> None:
    """ "a\\nb\\n" is two lines, not three. Counting newlines rather than line
    starts reports a third line with nothing in it, and the span would be
    empty, which the chain refuses."""
    assert _limit_spans("a\nb\n", Limits(max_lines=2)) == []


def test_several_limits_report_one_finding_from_the_earliest_breach() -> None:
    """One finding, not three. Three overlapping spans would merge into one
    placeholder anyway, and the audit record would claim three detections of
    one fact."""
    assert _limit_spans("x" * 50, Limits(max_chars=40, max_bytes=45)) == [
        ("LENGTH_LIMIT", (40, 50))
    ]


def test_limits_that_select_nothing_are_refused() -> None:
    with pytest.raises(ValueError, match="selects nothing"):
        Limits()


@pytest.mark.parametrize("field", ["max_chars", "max_bytes", "max_lines"])
def test_a_limit_below_one_is_refused(field: str) -> None:
    """A limit of zero denies every input including the empty one, which is a
    check that is on rather than a check that is configured."""
    with pytest.raises(ValueError, match="at least 1"):
        Limits(**{field: 0})


# Each row was run against the implementation on 3.10.20 and 3.14.5 and agreed.
# The `False` rows matter more than the `True` ones: a guard that refuses an
# ordinary pattern is a guard that gets worked around.
_NESTING = [
    (r"(a+)+b", True),
    (r"(a*)*", True),
    (r"([a-z]+)*", True),
    (r"(?:\w+\s?)*", True),
    (r"(?=(a+)+)", True),
    (r"^(a+)+$", True),
    (r"((a+))+", True),
    (r"(x|y+)*", True),
    (r"(a|aa)+$", False),
    (r"\bJIRA-\d{4,}\b", False),
    (r"[a-z0-9-]+\.corp\.example", False),
    (r"a{2,3}b+", False),
    (r"(a{1,3})+", False),
    (r"(?:ab)+c", False),
    (r"(a+){2}", False),
    (r"\d+", False),
    (r"(?:[a-f0-9]{2})+", False),
    (r"(a?)*", False),
]


@pytest.mark.parametrize(("pattern", "nests"), _NESTING, ids=[p for p, _ in _NESTING])
def test_the_nested_repeat_guard_agrees_with_the_recorded_table(pattern: str, nests: bool) -> None:
    assert _nests_unbounded_repeats(pattern) is nests


def test_an_alternation_of_bounded_repeats_is_not_nesting() -> None:
    """`(a|aa)+` is the textbook catastrophic pattern that this guard does NOT
    catch, and the docstring says so. Pinned here so the claim in the docstring
    is a measurement rather than a hedge: if the guard is ever widened to catch
    it, this test fails and the docstring is corrected in the same commit."""
    assert _nests_unbounded_repeats(r"(a|aa)+$") is False


def test_a_compiled_pattern_is_accepted_and_its_flags_are_used() -> None:
    assert _nests_unbounded_repeats(re.compile(r"(a+)+", re.IGNORECASE)) is True


def _guard(**options: object) -> PatternGuardrail:
    """A minimal valid guardrail, so each test below varies one thing."""
    base: dict[str, object] = {
        "name": "example",
        "version": "0.1.0",
        "patterns": {"TICKET_ID": r"\bJIRA-\d{4,}\b"},
    }
    base.update(options)
    return PatternGuardrail(**base)  # type: ignore[arg-type]


def test_a_built_guardrail_satisfies_the_protocol() -> None:
    assert isinstance(_guard(), Guardrail)


def test_it_declares_itself_a_constraint() -> None:
    """Not configurable. A classifier's findings must carry a confidence and
    nothing here produces one, so a caller who could declare this a classifier
    would build a guardrail whose every verdict `Verdict` rejects."""
    assert _guard().kind == "constraint"


def test_it_carries_the_name_and_version_it_was_given() -> None:
    guard = _guard(name="rules", version="2.3.4")
    assert (guard.name, guard.version) == ("rules", "2.3.4")


def test_a_configuration_that_selects_nothing_is_refused() -> None:
    """The same refusal `build_chain([])` makes, for the same reason: it would
    allow every input while its configuration says a check is running."""
    with pytest.raises(GuardrailUnavailableError, match="check nothing"):
        PatternGuardrail(name="example", version="0.1.0")


def test_an_empty_patterns_mapping_is_not_a_configuration() -> None:
    with pytest.raises(GuardrailUnavailableError, match="check nothing"):
        PatternGuardrail(name="example", version="0.1.0", patterns={}, banned={})


@pytest.mark.parametrize("bad", ["lower", "With Space", "1LEADING", "", "has-hyphen"])
def test_a_finding_type_outside_the_naming_rule_is_refused(bad: str) -> None:
    """Type names are what a corpus labels and what a placeholder prints, so a
    name that cannot appear in either is refused where it was written."""
    with pytest.raises(ValueError, match="finding type"):
        _guard(patterns={bad: r"x"})


def test_a_banned_type_is_held_to_the_same_naming_rule() -> None:
    with pytest.raises(ValueError, match="finding type"):
        _guard(patterns=None, banned={"lower": ("x",)})


def test_a_pattern_that_matches_the_empty_string_is_refused() -> None:
    """Catches a pattern that matches the empty string outright, such as `x*`.
    It does NOT catch a pattern gated by a lookbehind: `(?<=a)b*` matches
    nothing at position zero of "" and so passes this check, then matches
    zero-width at (2, 2) against real content such as "xa". The real backstop
    for that shape is downstream, at match time: the chain refuses to apply a
    redact whose finding span does not satisfy
    `0 <= start < end <= len(content)`, and raises rather than applying it,
    which fails closed."""
    with pytest.raises(ValueError, match="empty string"):
        _guard(patterns={"ANYTHING": r"x*"})


def test_a_pattern_nesting_unbounded_repeats_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="NESTS"):
        _guard(patterns={"SLOW": r"(a+)+b"})


def test_an_empty_banned_substring_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        _guard(patterns=None, banned={"CODENAME": ("",)})


def test_a_banned_entry_that_is_a_bare_string_is_refused() -> None:
    """A string is an iterable of its own characters, so `("bluebird")` without
    the comma would otherwise register eight one-character banned substrings
    and deny almost everything."""
    with pytest.raises(ValueError, match="list of substrings"):
        _guard(patterns=None, banned={"CODENAME": "bluebird"})


def test_an_on_match_mapping_missing_a_declared_direction_is_refused() -> None:
    """The alternative is a KeyError from inside `check`, which fails closed
    and names nothing. This names the direction."""
    with pytest.raises(GuardrailUnavailableError, match="output"):
        _guard(on_match={"input": "deny"})


def test_an_on_match_mapping_naming_an_undeclared_direction_is_refused() -> None:
    """The mirror of the missing-direction case above: a mapping that names a
    direction `directions` never declared states a policy that would never be
    consulted, since only the declared directions are ever checked. Silently
    dropping it would let a configuration read as though an output policy
    applied when it never could."""
    with pytest.raises(GuardrailUnavailableError, match="output"):
        _guard(directions=frozenset({"input"}), on_match={"input": "deny", "output": "redact"})


def test_an_on_match_mapping_with_a_typo_d_key_is_refused() -> None:
    """A key that is not a real direction at all, not merely one the guardrail
    declines to declare. It has to be caught by the same check as the case
    above: nothing about `set(on_match) - directions` distinguishes a typo from
    a real direction that was simply not declared."""
    with pytest.raises(GuardrailUnavailableError, match="outupt"):
        _guard(directions=frozenset({"input"}), on_match={"input": "deny", "outupt": "redact"})


@pytest.mark.parametrize("decision", ["allow", "warn", "REDACT", ""])
def test_a_decision_outside_redact_and_deny_is_refused(decision: str) -> None:
    """`allow` included, and that is the interesting one: a check configured to
    allow on a match is a check that runs and cannot act, which is the shape
    the other two bundled detectors refuse in their own constructors."""
    with pytest.raises(ValueError, match="'redact' or 'deny'"):
        _guard(on_match=decision)


def test_a_guardrail_declaring_no_direction_is_refused() -> None:
    with pytest.raises(GuardrailUnavailableError, match="no direction it can run in"):
        _guard(directions=frozenset())


def test_a_misspelled_direction_is_refused() -> None:
    """A `Context` only ever carries "input" or "output", so a typo such as
    "inptu" is not empty and clears the check above, then names a direction
    that can never arrive: the guardrail would construct, run, and never
    match anything, silently."""
    with pytest.raises(GuardrailUnavailableError, match="inptu"):
        _guard(directions=frozenset({"inptu"}))


def test_a_mix_of_one_good_and_one_bad_direction_is_refused() -> None:
    """The shape a real config is most likely to write: one direction that
    works and one that does not. A check for "at least one runnable
    direction" would wrongly let this through, since "input" is runnable;
    every declared direction has to be one a Context can carry."""
    with pytest.raises(GuardrailUnavailableError, match="inptu"):
        _guard(directions=frozenset({"input", "inptu"}))


def test_directions_default_to_both() -> None:
    assert _guard().directions == frozenset({"input", "output"})


def test_a_compiled_pattern_keeps_its_own_flags() -> None:
    guard = _guard(patterns={"SHOUT": re.compile(r"jira-\d{4,}", re.IGNORECASE)})
    assert guard._patterns[0][1].flags & re.IGNORECASE


from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.types import Context

IN = Context(direction="input", origin="user")
OUT = Context(direction="output", origin="model")


def test_clean_content_allows_with_no_findings() -> None:
    verdict = _guard().check("nothing to see", IN)
    assert verdict.decision == "allow"
    assert list(verdict.findings) == []
    assert verdict.content is None


def test_a_match_denies_by_default_and_carries_its_span() -> None:
    verdict = _guard().check("see JIRA-1234 please", IN)
    assert verdict.decision == "deny"
    assert [(f.type, f.span) for f in verdict.findings] == [("TICKET_ID", (4, 13))]
    assert verdict.content is None


def test_a_redacting_guardrail_rewrites_its_own_view() -> None:
    verdict = _guard(on_match="redact").check("see JIRA-1234 please", IN)
    assert verdict.decision == "redact"
    assert verdict.content == "see [REDACTED:TICKET_ID] please"


def test_every_finding_carries_no_confidence() -> None:
    """The constraint invariant. `Verdict` enforces it, so a finding built with
    one would raise rather than reach a caller; this pins the value."""
    verdict = _guard().check("JIRA-1234", IN)
    assert [f.confidence for f in verdict.findings] == [None]


def test_the_verdict_hashes_the_content_it_was_given() -> None:
    from jamjet_guardrails.protocol import saw

    content = "see JIRA-1234"
    assert _guard().check(content, IN).saw == saw(content)


def test_a_banned_substring_matches_regardless_of_case() -> None:
    guard = _guard(patterns=None, banned={"CODENAME": ("project bluebird",)})
    verdict = guard.check("about Project BlueBird today", IN)
    assert [(f.type, f.span) for f in verdict.findings] == [("CODENAME", (6, 22))]


def test_a_banned_substring_spans_the_source_when_folding_changed_its_length() -> None:
    """The sharp s casefolds to two characters, so a span read off the folded
    view would be one character too long and would eat the following space."""
    guard = _guard(patterns=None, banned={"STREET": ("strasse",)})
    verdict = guard.check("Straße 4", IN)
    assert [(f.type, f.span) for f in verdict.findings] == [("STREET", (0, 6))]
    assert verdict.decision == "deny"


def test_every_occurrence_of_a_banned_substring_is_reported() -> None:
    guard = _guard(patterns=None, banned={"CODENAME": ("bluebird",)})
    verdict = guard.check("bluebird and bluebird", IN)
    assert [f.span for f in verdict.findings] == [(0, 8), (13, 21)]


def test_overlapping_occurrences_are_both_reported() -> None:
    """`_scan` tries every start position, and the banned scan matches it, so
    two overlapping occurrences are two findings rather than one."""
    guard = _guard(patterns=None, banned={"REPEAT": ("aba",)})
    verdict = guard.check("ababa", IN)
    assert [f.span for f in verdict.findings] == [(0, 3), (2, 5)]


def test_findings_come_back_in_span_order_across_every_source() -> None:
    """`_merge` compares each span against the running end of the region it is
    extending and looks no further back, so an unsorted list makes it emit
    regions that are wrong rather than untidy. Patterns, banned substrings and
    limits each scan the whole input independently, so their concatenation is
    not in span order and the sort is load-bearing."""
    guard = _guard(
        patterns={"TICKET_ID": r"\bJIRA-\d{4,}\b"},
        banned={"CODENAME": ("bluebird",)},
        limits=Limits(max_chars=30),
        on_match="redact",
    )
    verdict = guard.check("bluebird then JIRA-1234 then a long tail here", IN)
    spans = [f.span for f in verdict.findings]
    # `Finding.span` is `tuple[int, int] | None` because a classifier finding
    # may carry none; every span a constraint's `check` produces is real, which
    # mypy cannot see from this test alone.
    assert spans == sorted(spans)  # type: ignore[type-var]
    assert [f.type for f in verdict.findings] == ["CODENAME", "TICKET_ID", "LENGTH_LIMIT"]


def test_two_types_claiming_one_region_collapse_into_one_placeholder() -> None:
    guard = _guard(
        patterns={"TICKET_ID": r"JIRA-\d{4,}"},
        banned={"CODENAME": ("jira-1234",)},
        on_match="redact",
    )
    assert guard.check("see JIRA-1234", IN).content == "see [REDACTED:CODENAME+TICKET_ID]"


def test_the_decision_follows_the_direction_the_context_carries() -> None:
    guard = _guard(on_match={"input": "redact", "output": "deny"})
    assert guard.check("JIRA-1234", IN).decision == "redact"
    assert guard.check("JIRA-1234", OUT).decision == "deny"


def test_clean_content_on_an_undeclared_direction_is_refused_not_allowed() -> None:
    """The worse of the two failures this guard exists to close: without it,
    clean content on a direction this guardrail never declared came back
    `allow`, reporting that the content was checked when it was not. A caller
    holding this guardrail directly, not through GuardrailChain, is who would
    otherwise be misled."""
    guard = _guard(directions=frozenset({"input"}))
    with pytest.raises(GuardrailUnavailableError, match="output"):
        guard.check("nothing to see", OUT)


def test_matching_content_on_an_undeclared_direction_is_refused_not_a_bare_keyerror() -> None:
    """Before this guard, matching content on an undeclared direction raised
    `KeyError('output')` from inside `_on_match`, which fails closed but names
    neither the guardrail nor why. This names both."""
    guard = _guard(directions=frozenset({"input"}))
    with pytest.raises(GuardrailUnavailableError, match="output"):
        guard.check("see JIRA-1234 please", OUT)


def test_it_composes_in_a_chain_with_a_bundled_detector() -> None:
    """The composition the chain was rebuilt for: both checks inspect the
    content the chain was given, and the two rewrites are merged into one pass."""
    guard = _guard(patterns={"TICKET_ID": r"JIRA-\d{4,}"}, on_match="redact")
    from jamjet_guardrails.detectors import build

    result = GuardrailChain([guard, build("secrets")]).run(
        "JIRA-1234 and sk-abcdefghijklmnopqrstuvwxyz012345", OUT
    )
    assert result.decision == "redact"
    assert result.content == "[REDACTED:TICKET_ID] and [REDACTED:OPENAI_KEY]"


def test_a_pattern_is_searched_rather_than_anchored() -> None:
    """`finditer` is search semantics, so `^` and `$` bind to the ends of the
    whole content unless the caller compiled with MULTILINE. An unanchored
    pattern therefore matches inside a longer token, which is the anchoring
    pitfall the documentation names."""
    guard = _guard(patterns={"MEDIA": r"packages/media/"})
    assert guard.check("foo/packages/media/bar", IN).decision == "deny"


def test_a_second_match_directly_behind_a_greedy_first_one_is_not_lost() -> None:
    """A user's own pattern reaches `_matches` unmodified, so the leak `_scan`
    exists to close is theirs to inherit if this primitive ever used
    `finditer` instead. Here the first token's greedy body runs past the
    second token's `tok_` prefix and stops at its own underscore, which is
    where `finditer` resumes: it finds one match covering characters 0 to 15
    and never tries starting inside the second token at all. `_scan` tries
    every start position, finds the second token too, and the merge collapses
    both into one region, so the redaction removes the second token's full
    eight-character body along with the first rather than leaving it standing
    behind a placeholder that claims the content was redacted."""
    guard = _guard(patterns={"TOK": r"tok_[A-Za-z0-9]+"}, on_match="redact")
    content = "tok_" + "A" * 8 + "tok_" + "B" * 8
    verdict = guard.check(content, IN)
    assert verdict.decision == "redact"
    assert verdict.content is not None
    assert "B" not in verdict.content


@pytest.mark.parametrize("field", ["name", "version"])
def test_a_name_or_version_above_the_chain_ceiling_is_refused_here(field: str) -> None:
    """The failure belongs at the mistake, not two seams later.

    `GuardrailChain` refuses a declared `name` or `version` above 200 characters,
    because both are copied into the `Provenance` of every verdict the guardrail
    produces. This constructor did not, so `PatternGuardrail(name="a" * 250)`
    built cleanly, satisfied the protocol, passed the registry, and then made
    `build_chain` refuse the whole configuration. A 250-character name is a
    plausible namespaced slug, so that is a real upgrade path for a deployment,
    failing at the seam furthest from the edit that caused it.

    `detectors.build`'s own docstring makes the argument this closes: "A factory
    that hands back an object which only detonates on first use has moved the
    failure away from the mistake."

    `ValueError`, not `GuardrailUnavailableError`, matching how this constructor
    already separates the two: an out-of-domain option value is a bad argument,
    and a configuration that selects nothing is an unavailable check. A name too
    long is the first of those.

    Mutation-checked: removing the length loop from `PatternGuardrail.__init__`
    makes this pass silently and the chain refuse instead, which is the state it
    replaces.
    """
    over = "x" * (_CALLER_STRING_LIMIT + 1)
    with pytest.raises(ValueError, match="above the"):
        _guard(**{field: over})


@pytest.mark.parametrize("field", ["name", "version"])
def test_a_name_or_version_at_the_ceiling_still_builds_and_runs(field: str) -> None:
    """The false-reject control. A guard that refused every name would also pass
    the test above, and this constructor is the one a contributor meets first."""
    at_limit = "x" * _CALLER_STRING_LIMIT
    guardrail = _guard(**{field: at_limit})
    assert getattr(guardrail, field) == at_limit
    verdict = guardrail.check("JIRA-1234", Context(direction="input", origin="user"))
    assert verdict.decision == "deny"
    recorded = verdict.provenance.detector if field == "name" else verdict.provenance.version
    assert recorded == at_limit


# ==========================================================================
# fold_confusables: the seam a rules engine has next to a confusables check.
#
# A banned substring is the exact thing one substituted Cyrillic letter
# defeats, so a rules engine shipping beside a check that detects those
# letters has to be able to close its own seam. Every test below was watched
# to fail against a mutation, with `__pycache__` cleared between runs:
#
# - the skeleton dropped from the needle at construction;
# - the skeleton dropped from the view in `_banned_spans`;
# - `compose` replaced by the skeleton view alone, so the span indexes the
#   case-folded view rather than the source;
# - the refusal of the option with no banned substrings removed.
# ==========================================================================

# Escapes, never literals: a Cyrillic letter inside a Latin word is invisible
# here, in the diff and in the review, which is the point of the option.
_CYRILLIC_I = "\u0456"  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
_CYRILLIC_O = "\u043e"  # CYRILLIC SMALL LETTER O
_SHARP_S = "\u00df"  # LATIN SMALL LETTER SHARP S


def _banned_guard(**options: object) -> PatternGuardrail:
    return PatternGuardrail(
        name="rules",
        version="0.1.0",
        banned={"PROJECT_CODENAME": ("project bluebird",)},
        on_match="redact",
        **options,  # type: ignore[arg-type]
    )


def test_a_banned_word_survives_one_substituted_cyrillic_letter() -> None:
    """The whole reason the option exists.

    Without it the same content allows, which is the assertion below it: the
    two together are the before and after, so a fold that silently stopped
    folding would fail here rather than pass both.
    """
    content = f"deck for project blueb{_CYRILLIC_I}rd, internal only"
    verdict = _banned_guard(fold_confusables=True).check(content, IN)
    assert verdict.decision == "redact"
    (finding,) = verdict.findings
    assert finding.span is not None
    assert content[finding.span[0] : finding.span[1]] == f"project blueb{_CYRILLIC_I}rd"
    assert verdict.content == "deck for [REDACTED:PROJECT_CODENAME], internal only"


def test_without_the_option_the_substituted_letter_dodges_the_word() -> None:
    """The false-reject control, and the measurement the option is sold on."""
    content = f"deck for project blueb{_CYRILLIC_I}rd, internal only"
    assert _banned_guard().check(content, IN).decision == "allow"


def test_the_fold_runs_over_both_sides_and_still_matches_the_plain_word() -> None:
    """Skeletonising the needle and not the content, or the other way round,
    breaks the ordinary case while leaving the spoofed one working."""
    guardrail = _banned_guard(fold_confusables=True)
    assert guardrail.check("about project bluebird today", IN).decision == "redact"
    assert guardrail.check("about PROJECT BLUEBIRD today", IN).decision == "redact"


def test_the_span_points_at_the_source_even_when_a_fold_changed_the_length() -> None:
    """Case folding expands the sharp s and the skeleton expands other rows, so
    the view and the source are different lengths and the offset map is the only
    thing that makes the span right.

    Mutation-checked: dropping `compose` and searching the skeleton view alone
    reports a span shifted by one character, and the redaction then leaves a
    letter of the banned word standing.
    """
    guardrail = PatternGuardrail(
        name="rules",
        version="0.1.0",
        banned={"CODENAME": ("strasse",)},
        on_match="redact",
        fold_confusables=True,
    )
    content = f"the Stra{_SHARP_S}e file"
    verdict = guardrail.check(content, IN)
    (finding,) = verdict.findings
    assert finding.span is not None
    assert content[finding.span[0] : finding.span[1]] == f"Stra{_SHARP_S}e"
    assert verdict.content == "the [REDACTED:CODENAME] file"


def test_a_needle_that_is_not_already_its_own_skeleton_is_folded_too() -> None:
    """The needle side of the fold, which an ASCII banned word cannot hold down.

    `project bluebird` is its own skeleton, so skeletonising the needle is a
    no-op for it and a test built on it says nothing about that half: measured,
    deleting the needle fold leaves every other test in this file green. A
    needle carrying a precomposed `e` with an acute accent does hold it: the
    skeleton decomposes it, so a view built from the content and a needle left
    alone no longer agree, and the banned word stops matching itself.

    Mutation-checked: dropping the needle fold makes both assertions below
    allow.
    """
    guardrail = PatternGuardrail(
        name="rules",
        version="0.1.0",
        banned={"CODENAME": ("caf\u00e9 noir",)},
        on_match="redact",
        fold_confusables=True,
    )
    assert guardrail.check("the caf\u00e9 noir project", IN).decision == "redact"
    # The same word written with a combining acute rather than a precomposed
    # letter. One skeleton, two spellings, and the corpus for `rules` carries
    # neither: this is where that equivalence is stated.
    assert guardrail.check("the cafe\u0301 noir project", IN).decision == "redact"


def test_a_different_word_of_the_same_shape_is_not_matched() -> None:
    """The fold is not a similarity measure. `bluebell` is not `bluebird`."""
    content = f"deck for pr{_CYRILLIC_O}ject bluebell"
    assert _banned_guard(fold_confusables=True).check(content, IN).decision == "allow"


def test_the_option_is_refused_where_it_would_change_nothing() -> None:
    """An option that selects nothing reads as protection and buys none.

    Mutation-checked: removing the refusal makes this construct cleanly and
    then allow every spoofed pattern match, because patterns are deliberately
    not folded.
    """
    with pytest.raises(ValueError, match="no banned substrings"):
        PatternGuardrail(
            name="rules",
            version="0.1.0",
            patterns={"TICKET_ID": r"\bJIRA-\d{4,}\b"},
            fold_confusables=True,
        )


def test_the_option_is_off_by_default() -> None:
    """A fold nobody asked for is a matching rule nobody wrote down."""
    content = f"deck for project blueb{_CYRILLIC_I}rd"
    assert _banned_guard().check(content, IN).decision == "allow"
    assert _banned_guard(fold_confusables=False).check(content, IN).decision == "allow"
