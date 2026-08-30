import itertools
import re
import time
from collections.abc import Iterator

import pytest

from jamjet_guardrails.detectors.pii import (
    _CONTIGUOUS_CARD,
    _PATTERNS,
    PII_TYPES,
    PiiGuardrail,
    _luhn,
)
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import Context

OUT = Context(direction="output", origin="model")
IN = Context(direction="input", origin="user")


def test_declares_the_four_types() -> None:
    assert PII_TYPES == frozenset({"EMAIL", "US_SSN", "CREDIT_CARD", "PHONE_NUMBER"})


def test_allows_clean_text_without_rewriting() -> None:
    verdict = PiiGuardrail().check("the weather is fine", OUT)
    assert verdict.decision == "allow"
    assert verdict.content is None
    assert list(verdict.findings) == []


def test_redacts_an_email() -> None:
    verdict = PiiGuardrail().check("write to alice@example.com now", OUT)
    assert verdict.decision == "redact"
    assert verdict.content == "write to [REDACTED:EMAIL] now"
    assert [f.type for f in verdict.findings] == ["EMAIL"]


def test_reports_the_span_of_the_original_text() -> None:
    text = "write to alice@example.com now"
    verdict = PiiGuardrail().check(text, OUT)
    # `Finding.span` is optional, so it needs narrowing before unpacking. The
    # narrowing doubles as an assertion worth making: a redaction that cannot say
    # where it redacted is not auditable.
    span = verdict.findings[0].span
    assert span is not None
    start, end = span
    assert text[start:end] == "alice@example.com"


def test_constraint_findings_carry_no_confidence() -> None:
    verdict = PiiGuardrail().check("alice@example.com", OUT)
    assert all(f.confidence is None for f in verdict.findings)
    assert verdict.provenance.kind == "constraint"


def test_redacts_ssn_card_and_phone() -> None:
    cases = {
        "ssn 123-45-6789": ("US_SSN", "ssn [REDACTED:US_SSN]"),
        "card 4111 1111 1111 1111": ("CREDIT_CARD", "card [REDACTED:CREDIT_CARD]"),
        "call 555-123-4567": ("PHONE_NUMBER", "call [REDACTED:PHONE_NUMBER]"),
    }
    for text, (expected_type, expected_out) in cases.items():
        verdict = PiiGuardrail().check(text, OUT)
        assert [f.type for f in verdict.findings] == [expected_type], text
        assert verdict.content == expected_out, text


def test_redacts_every_occurrence_not_just_the_first() -> None:
    verdict = PiiGuardrail().check("a@b.com and c@d.com", OUT)
    assert verdict.content == "[REDACTED:EMAIL] and [REDACTED:EMAIL]"
    assert len(verdict.findings) == 2


def test_selecting_a_subset_of_types_leaves_the_others_alone() -> None:
    verdict = PiiGuardrail(types=frozenset({"EMAIL"})).check("ssn 123-45-6789", OUT)
    assert verdict.decision == "allow"


def test_saw_hashes_the_input_not_the_output() -> None:
    from jamjet_guardrails.protocol import saw

    text = "alice@example.com"
    assert PiiGuardrail().check(text, OUT).saw == saw(text)


def test_negatives_that_look_like_pii_are_not_redacted() -> None:
    """Precision is the number we publish. These are the cases that move it."""
    for text in [
        "version 1.2.3-45-6789",  # not an SSN
        "the array is 4111,1111,1111",  # not a card
        "@example.com",  # no local part
        "sha 1111111111111111111111",  # not a card
    ]:
        assert PiiGuardrail().check(text, OUT).decision == "allow", text


# --------------------------------------------------------------------------------
# Beyond the brief: the guards that carry the precision number, and the assertions
# the brief's fixtures cannot distinguish from a hardcoded literal.
# --------------------------------------------------------------------------------


def test_the_class_declares_its_identity_and_both_directions() -> None:
    assert PiiGuardrail.name == "pii"
    assert PiiGuardrail.kind == "constraint"
    assert PiiGuardrail.directions == frozenset({"input", "output"})


def test_conforms_to_the_guardrail_protocol() -> None:
    assert isinstance(PiiGuardrail(), Guardrail)


def test_provenance_reports_the_instance_identity_not_a_module_literal() -> None:
    """Defeat the fixture coincidence rather than walk into it.

    `name` is "pii" and `version` is "0.1.0", so asserting either literal would
    pass just as well against a `Provenance(kind="constraint", detector="pii",
    version="0.1.0")` inlined in `check`. Overriding both on the instance makes
    the fixture disagree with every constant the code could hardcode, so only
    reading `self` satisfies it.

    The `model`/`threshold` assertions are the library's central claim in its
    audit record: a constraint has neither, and a verdict that named one would be
    describing a check that never happened.
    """
    guardrail = PiiGuardrail()
    guardrail.name = "pii-under-test"
    guardrail.version = "9.9.9"

    # Both branches of `check` build their own Provenance. Exercise each.
    for text in ("alice@example.com", "the weather is fine"):
        provenance = guardrail.check(text, OUT).provenance
        assert provenance.detector == "pii-under-test", text
        assert provenance.version == "9.9.9", text
        assert provenance.kind == "constraint", text
        assert provenance.model is None, text
        assert provenance.threshold is None, text


def test_the_allow_path_hashes_what_it_saw() -> None:
    """The brief pins `saw` on the redact path only; allow builds a second verdict."""
    for text in ("the weather is fine", ""):
        verdict = PiiGuardrail().check(text, OUT)
        assert verdict.decision == "allow", repr(text)
        assert verdict.saw == saw(text), repr(text)


def test_the_direction_does_not_change_the_verdict() -> None:
    """It declares both directions, and it means it: same bytes, same decision."""
    text = "write to alice@example.com now"
    assert PiiGuardrail().check(text, IN) == PiiGuardrail().check(text, OUT)


def test_one_instance_checks_many_inputs_without_carrying_state() -> None:
    """The per-call state really is per-call.

    `_merge` builds its regions from scratch on every call. Held on the instance
    instead, the first call's regions would still be open on the second, and every
    later match would merge into a region left over from an earlier input.
    """
    guardrail = PiiGuardrail()
    text = "call 555-123-4567 or mail a@b.com"

    first = guardrail.check(text, OUT)
    second = guardrail.check(text, OUT)

    assert first == second
    assert [f.type for f in second.findings] == ["PHONE_NUMBER", "EMAIL"]


@pytest.mark.parametrize(
    ("text", "expected", "types", "spans"),
    [
        (
            "mail 123-456-7890@example.com",  # PHONE_NUMBER sits inside the EMAIL match
            "mail [REDACTED:EMAIL+PHONE_NUMBER]",
            ["PHONE_NUMBER", "EMAIL"],
            [(5, 17), (5, 29)],
        ),
        (
            "mail 123-45-6789@example.com",  # US_SSN sits inside the EMAIL match
            "mail [REDACTED:EMAIL+US_SSN]",
            ["US_SSN", "EMAIL"],
            [(5, 16), (5, 28)],
        ),
        (
            "mail a123-456-7890@example.com",  # container sorts BEFORE what it contains
            "mail [REDACTED:EMAIL+PHONE_NUMBER]",
            ["EMAIL", "PHONE_NUMBER"],
            [(5, 30), (6, 18)],
        ),
    ],
)
def test_a_contested_span_names_every_type_that_claimed_it(
    text: str, expected: str, types: list[str], spans: list[tuple[int, int]]
) -> None:
    """Merging, from the side where two matches cover the same bytes.

    No pattern wins here and none is dropped. One placeholder covers the union,
    and it names both contributors sorted alphabetically so the output is a
    function of the input alone and not of the pattern table's order.

    Without merging of some kind, both matches reach the rebuild, the cursor
    walks backwards over a span it already consumed, and the local part leaks out
    of the placeholder as "[REDACTED:EMAIL][REDACTED:PHONE_NUMBER]@example.com".

    The findings keep one entry per match at its own original span, so the audit
    record still shows that two patterns fired even though the output shows one
    placeholder. They are listed by span, which puts the inner match first.
    """
    verdict = PiiGuardrail().check(text, OUT)

    assert verdict.content == expected
    assert [f.type for f in verdict.findings] == types
    # One finding per match at its OWN span, not the merged region's span. The
    # output collapses to a single placeholder; the audit record must not.
    assert [f.span for f in verdict.findings] == spans


def test_a_later_pattern_matching_earlier_in_the_text_is_kept_in_text_order() -> None:
    """Merging, from the side where it must NOT happen: two disjoint matches.

    US_SSN is listed after EMAIL in the pattern table but occurs before it in the
    text, so it opens its own region rather than joining the email's. A merge test
    that compared only one end of the interval, or matches left unsorted, would
    fold these two into a single placeholder spanning the words between them, and
    "and mail" would vanish from the output.
    """
    text = "ssn 123-45-6789 and mail a@b.com"
    verdict = PiiGuardrail().check(text, OUT)

    assert [f.type for f in verdict.findings] == ["US_SSN", "EMAIL"]
    assert verdict.content == "ssn [REDACTED:US_SSN] and mail [REDACTED:EMAIL]"


def test_spans_point_into_the_input_for_the_second_occurrence_too() -> None:
    """The first span is right by construction; the second is where cursor bugs live.

    The two emails are deliberately different lengths and neither is 16 characters,
    so a span measured against the rewritten string lands on different text.
    """
    text = "a@b.com and carol@example.org"
    verdict = PiiGuardrail().check(text, OUT)
    assert verdict.content == "[REDACTED:EMAIL] and [REDACTED:EMAIL]"

    quoted: list[str] = []
    for finding in verdict.findings:
        assert finding.span is not None
        start, end = finding.span
        quoted.append(text[start:end])
    assert quoted == ["a@b.com", "carol@example.org"]


# Each entry names the guard it holds in place. Loosen that guard and this input
# starts matching, which is exactly how a published precision number regresses.
_NEGATIVES = [
    ("login as admin@localhost", "EMAIL requires a dotted TLD"),
    ("npm i pkg@1.2.x", "EMAIL requires a TLD of two or more letters"),
    ("ticket 1234-56-7890", "US_SSN rejects a digit before it"),
    ("code 123-45-67890", "US_SSN rejects a digit after it"),
    ("case 123-45-6789-0000", "US_SSN rejects an adjoining dashed group"),
    ("batch 14111 1111 1111 1111", "CREDIT_CARD rejects a digit before it"),
    ("batch 4111 1111 1111 11111", "CREDIT_CARD rejects a digit after it"),
    ("array 4111,1111,1111,1111", "CREDIT_CARD separators are space and dash only"),
    ("batch 1234567890123456", "CREDIT_CARD requires separators"),
    ("trio 1111 1111 1111", "CREDIT_CARD requires four groups"),
    ("ref 1555-123-4567", "PHONE_NUMBER rejects a digit before it"),
    ("ext 555-123-45678", "PHONE_NUMBER rejects a digit after it"),
    ("ext 555-123-456", "PHONE_NUMBER requires four final digits"),
    ("ids 123 456 7890", "PHONE_NUMBER separators are dashes or dots, not spaces"),
    # The dash INSIDE each lookaround, which is a separate guard from the digit.
    # Without it a longer dashed run reads as PII: "2024-123-45-6789" becomes an
    # SSN, and a card or phone acquires neighbours it should have been fenced off
    # from.
    #
    # The guard is NARROWER than it was. It used to turn away any adjacent dash
    # and so hid the value behind its own label; now only a dash CONTINUING a
    # digit run fences one off. That leaves six sites, three fenced types by two
    # sides, and each needs a case of its own because `_BEFORE` and `_AFTER` are
    # shared text: a clause deleted from either is deleted from all three types
    # at once. The positive half of every pair, where a word hyphen must NOT
    # hide the value, is `_WORD_HYPHEN_SITES` below. Neither half proves the
    # rule alone.
    ("ref 2024-123-45-6789", "US_SSN is fenced by a digit and dash before it"),
    ("batch 12-4111 1111 1111 1111", "CREDIT_CARD is fenced by a digit and dash before it"),
    ("ext 2024-555-123-4567", "PHONE_NUMBER is fenced by a digit and dash before it"),
    ("code 123-45-6789-0", "US_SSN is fenced by a dash and digit after it"),
    ("batch 4111 1111 1111 1111-2", "CREDIT_CARD is fenced by a dash and digit after it"),
    ("ext 555-123-4567-8", "PHONE_NUMBER is fenced by a dash and digit after it"),
]


@pytest.mark.parametrize(("text", "guard"), _NEGATIVES)
def test_each_boundary_guard_has_a_negative(text: str, guard: str) -> None:
    """One negative per guard, so a regression says which guard slipped.

    The brief's own four negatives are all true negatives, but none of them holds
    a guard down on its own. Each stays clean even after the guard it looks like
    it was written for is deleted:

    - "version 1.2.3-45-6789" holds no three-digit run, so US_SSN misses it with
      both lookarounds removed.
    - "the array is 4111,1111,1111" carries three groups, not four, so it misses
      even once commas join the separator class.
    - "sha " plus twenty-two digits misses CREDIT_CARD even with the separator
      made optional, because no sixteen-digit window inside it has a non-digit on
      both sides.

    The fourth, "@example.com", was loose for the same reason and is not any more.
    It used to be turned away by the leading word-boundary anchor rather than by
    the local part's one-or-more, so it stayed clean either way. Both anchors have
    since been removed to close two redaction bypasses, and with them gone a
    zero-or-more local part matches at the `@` itself. The brief's negative now
    holds that bound down on its own.

    Every case below fails against exactly one loosening. See task-6-report.md.
    """
    verdict = PiiGuardrail().check(text, OUT)
    assert verdict.decision == "allow", f"{guard}: {text!r} -> {verdict.findings}"


# The six sites `_BEFORE` and `_AFTER` fence, as the POSITIVE half of each pair:
# a hyphen that does not continue a digit run is a word hyphen, and a word hyphen
# must not hide the value beside it. `(?<![\d-])` blocked every adjacent hyphen,
# so a label hid the value it labelled and the whole of "ssn-123-45-6789",
# "card-4111 1111 1111 1111" and "tel-555-123-4567" came back allow with the
# value intact and liftable.
#
# All three fenced types, both sides, derived by asking which patterns interpolate
# the shared lookarounds rather than from memory: US_SSN, CREDIT_CARD and
# PHONE_NUMBER each carry both, EMAIL carries neither. The clauses are shared
# text, so one deleted clause is deleted for three types at once, and one type's
# case would let the other two regress unseen.
_WORD_HYPHEN_SITES = [
    ("ssn-123-45-6789", "US_SSN", "ssn-[REDACTED:US_SSN]"),
    ("123-45-6789-ssn", "US_SSN", "[REDACTED:US_SSN]-ssn"),
    ("card-4111 1111 1111 1111", "CREDIT_CARD", "card-[REDACTED:CREDIT_CARD]"),
    ("4111 1111 1111 1111-card", "CREDIT_CARD", "[REDACTED:CREDIT_CARD]-card"),
    ("tel-555-123-4567", "PHONE_NUMBER", "tel-[REDACTED:PHONE_NUMBER]"),
    ("555-123-4567-tel", "PHONE_NUMBER", "[REDACTED:PHONE_NUMBER]-tel"),
]


@pytest.mark.parametrize(("text", "pii_type", "expected"), _WORD_HYPHEN_SITES)
def test_a_word_hyphen_does_not_hide_the_value_it_labels(
    text: str, pii_type: str, expected: str
) -> None:
    """The brief names one of these six. The same argument holds for all six."""
    verdict = PiiGuardrail().check(text, OUT)

    assert [f.type for f in verdict.findings] == [pii_type], text
    assert verdict.content == expected, text


def test_a_non_ascii_local_part_is_matched_in_both_spellings() -> None:
    """This WAS a documented miss. Task 18 closed it, in both spellings.

    The local-part class was ASCII, so a real address went unmatched entirely.
    The class is `\\w` now, which is Unicode for a str pattern, plus the combining
    marks NFD leaves behind.

    Both spellings, because they fail differently and a fixture in one of them
    proves nothing about the other. Composed, the character before the `@` is
    U+00E9, which `\\w` matches on its own. Decomposed it is U+0301, a nonspacing
    mark, which is NOT alphanumeric and so NOT `\\w`: without the mark range no
    start position reaches the `@` and the whole address is still a complete miss.
    Written as escapes so that an editor normalising this file cannot silently
    turn the decomposed case into the composed one.

    The local part's lower bound loses a witness here and keeps one. "@example.com"
    in the brief's negatives is now the only case holding `{1,64}` down: relax it
    to `{0,64}` and, with no leading anchor, the match begins at the `@` itself.
    """
    for spelling in ("caf\u00e9", "cafe\u0301"):
        verdict = PiiGuardrail().check(f"mail {spelling}@example.com", OUT)

        assert verdict.decision == "redact", spelling
        assert [f.type for f in verdict.findings] == ["EMAIL"], spelling
        assert "@example.com" not in (verdict.content or ""), spelling


def test_an_empty_type_set_is_refused_rather_than_selecting_nothing() -> None:
    """CHANGED: it asserted that an empty selection built a guardrail that allowed.

    It did, and the verdict it checked was `allow` with no findings over
    "alice@example.com". That is the same output `build_chain` refuses to build
    from an empty list of names, over content full of personal data, from a
    configuration a caller can write: `guardrails: {pii: {types: []}}`.

    `types=None` still means all four, and the distinction the old test was
    written to protect is still live: it is checked directly below, so a
    `selected = PII_TYPES if not types else types` that collapses the two would
    turn this refusal off and fail there instead.
    """
    with pytest.raises(GuardrailUnavailableError, match="empty set of types"):
        PiiGuardrail(types=frozenset())


def test_no_types_argument_still_selects_all_four() -> None:
    """The false-reject control for the refusal above, and the old test's real point."""
    assert PiiGuardrail().check("alice@example.com", OUT).decision == "redact"
    assert PiiGuardrail(types=None).check("alice@example.com", OUT).decision == "redact"


def test_rejects_an_unknown_type_at_construction() -> None:
    """Fail at construction, not by silently checking less than the caller asked for."""
    with pytest.raises(ValueError, match=r"unknown PII types: \['PASSPORT'\]"):
        PiiGuardrail(types=frozenset({"EMAIL", "PASSPORT"}))


def test_two_adjacent_matches_are_both_redacted() -> None:
    """The one input shape that tells a strict overlap test from a non-strict one.

    An address ends exactly where an SSN begins, with no match spanning the join.
    Touching spans do not overlap, so they must stay two regions. A `<=` in the
    merge folds them into one and the output claims a single finding where there
    were two, which an audit record cannot afford.

    THE FIXTURE CHANGED when the scan was fixed, and the reason is worth keeping.
    It used to be "a@b.com.a@b.com", two addresses joined by a dot. Now that
    every start position is tried, "b.com.a@b.com" is itself a match spanning the
    join, so those two addresses legitimately merge into one region. Nothing
    leaks, the coverage is character-for-character identical, and the output is
    one placeholder instead of two. It simply stopped being a touching-span case:
    the domain class is a subset of the local-part class, so ANY suffix of a
    first address is a valid local part and two adjacent addresses always bridge.
    Two different types are needed instead, which is what this is.
    """
    verdict = PiiGuardrail().check("a@b.com123-45-6789", OUT)

    assert verdict.content == "[REDACTED:EMAIL][REDACTED:US_SSN]"
    assert [f.span for f in verdict.findings] == [(0, 7), (7, 18)]


def test_two_bridged_addresses_merge_without_losing_a_character() -> None:
    """The fixture the test above used to use, kept for what it now proves.

    Every start position is tried, so the string that joins two addresses is
    itself an address and the three matches collapse to one region. This asserts
    the collapse costs nothing: one placeholder, and not one character of either
    address surviving. The findings still show both matches at their own spans,
    so the audit record says two things were seen.
    """
    verdict = PiiGuardrail().check("a@b.com.a@b.com", OUT)

    assert verdict.content == "[REDACTED:EMAIL]"
    assert [f.span for f in verdict.findings] == [(0, 7), (2, 15)]
    _assert_nothing_matched_survives("a@b.com.a@b.com")


def test_a_partial_overlap_redacts_the_union_not_just_the_winner() -> None:
    """Regression guard for a Critical: a straddling match used to leak the rest.

    CREDIT_CARD matches (0, 19) and EMAIL matches (15, 27), so neither contains
    the other. Dropping the loser whole left "4111 1111 1111 " standing in the
    output of a redactor, fifteen of the card's nineteen characters. Merging
    redacts the union, (0, 27), and names both types.

    Only a space-separated card can partially overlap an email this way: every
    other pattern sits wholly inside the local part when it meets one. That is
    what made this shape easy to miss and is why the invariant below is checked
    over a spread of fixtures rather than this one alone.
    """
    text = "4111 1111 1111 1111.a@b.com"
    verdict = PiiGuardrail().check(text, OUT)

    assert verdict.content == "[REDACTED:CREDIT_CARD+EMAIL]"
    assert [f.type for f in verdict.findings] == ["CREDIT_CARD", "EMAIL"]
    # Name the leak itself, not just the expected string: no run of card digits
    # may survive, however the placeholder is spelled.
    assert "1111" not in verdict.content
    assert "4111" not in verdict.content


# Positives spread across single matches, repeats, adjacency, containment and
# partial overlap. The invariant below has to hold for every one of them.
_NO_LEAK_FIXTURES = [
    "4111 1111 1111 1111.a@b.com",
    "mail 123-456-7890@example.com",
    "mail 123-45-6789@example.com",
    "mail a123-456-7890@example.com",
    "a@b.com.a@b.com",
    "a@b.com-carol@example.org",
    "ssn 123-45-6789 and mail a@b.com",
    "call 555-123-4567 or mail a@b.com",
    "card 4111 1111 1111 1111",
    "a@b.com and carol@example.org",
    "write to alice@example.com now",
    # The conditional match, in every relation it can have with the others: on
    # its own, touching an address, and inside a grouped card's own digits.
    "card 4111111111111111 and mail a@b.com",
    "4111111111111111a@b.com",
    "pay 378282246310005 by 415.555.2671",
]


_PLACEHOLDER = re.compile(r"\[REDACTED:[A-Z_+]+\]")


def _matched_offsets(text: str) -> set[int]:
    """Every character offset any pattern matches, computed WITHOUT the detector.

    Deliberately not built from `verdict.findings`. A finding is the detector's
    own account of what it saw, so an oracle derived from it shrinks by exactly
    the amount the detector missed. Drop a match from the findings and from the
    redaction together, which is precisely the shape of the suppression bug this
    guards against, and a findings-derived oracle calls the damage clean.

    Deliberately not built with `finditer` either, which is the harder lesson
    and cost this detector a sixth leak. `finditer` resumes at the END of each
    match, so a decoy address joined in front of a real one hides it, and while
    the oracle used the same call it inherited the detector's blind spot exactly
    and certified the leak as clean: 'a@b.comalice@example.com' came back as
    '[REDACTED:EMAIL]@example.com' with the sweep green. That is the circular
    oracle one level below the one this docstring already warned about: not
    "derived from the findings" but "derived from the same scan strategy". It
    now tries EVERY start position with an anchored `match`, a second
    implementation of the property rather than the same walk.

    KNOW WHAT THIS DOES NOT COVER. It imports `_PATTERNS` from the module under
    test, so it is independent of the scan, the merge, the rebuild, the cursor,
    the containment filter and the type selection, and of nothing above that.
    The patterns are shared, so a change to one moves the oracle with it and this
    check cannot see the difference.

    How much that costs, measured per pattern rather than assumed from one probe.
    Substituting a regex matching nothing:

        EMAIL         oracle tests RED    (census floors catch it)
        US_SSN        oracle tests GREEN  <- the real blind spot
        CREDIT_CARD   oracle tests RED    (census floors catch it)
        PHONE_NUMBER  oracle tests RED    (census floors catch it)

    Only US_SSN is invisible here, because it is the one type whose spans the
    shape floors below do not depend on. The floors were added for a different
    reason and closed three quarters of this hole as a side effect. The full
    suite catches all four through the fixture tests. Pattern changes still need
    their own adversarial cases: this oracle will not generate them.
    """
    offsets: set[int] = set()
    for _, pattern in _PATTERNS:
        for start in range(len(text)):
            match = pattern.match(text, start)
            if match is not None:
                offsets.update(range(*match.span()))
    offsets.update(_contiguous_card_offsets(text))
    return offsets


def _contiguous_card_offsets(text: str) -> set[int]:
    """The contiguous card's contribution to the oracle, rebuilt not reused.

    The one match this detector makes conditionally is not in `_PATTERNS`, so an
    oracle built from the table alone claims FEWER offsets than the detector
    covers, and `_assert_nothing_matched_survives` reads that as the detector
    eating text nobody matched. The invariant then fails on every correct
    redaction of a bare card number.

    Built from the primitives, `_CONTIGUOUS_CARD` and `_luhn`, and not from
    `_contiguous_cards`, for the reason the rest of this oracle exists: the
    detector's own account of what it saw cannot be the thing that checks it.
    Every start position, like the loop above, so the walk is a second
    implementation rather than the same one.
    """
    offsets: set[int] = set()
    for start in range(len(text)):
        match = _CONTIGUOUS_CARD.match(text, start)
        if match is not None and _luhn(match.group()):
            offsets.update(range(*match.span()))
    return offsets


def _assert_nothing_matched_survives(text: str) -> None:
    """Strip the placeholders; what is left must be exactly what no pattern matched.

    Checked per character, not per match. Asserting only that a whole matched
    string is absent would pass while a fragment survived, which is the failure
    being guarded: the original bug left fifteen of a card's characters standing.

    The equality is two guards in one. Extra text on the left is a leak. Missing
    text on the left is the opposite mistake, eating input that no pattern
    matched, which a redactor that simply blanked everything would commit.
    """
    verdict = PiiGuardrail().check(text, OUT)
    claimed = _matched_offsets(text)
    expected = "".join(ch for i, ch in enumerate(text) if i not in claimed)

    if not claimed:
        assert verdict.decision == "allow", repr(text)
        assert verdict.content is None, repr(text)
        return

    assert verdict.decision == "redact", repr(text)
    assert verdict.content is not None
    survived = _PLACEHOLDER.sub("", verdict.content)
    assert survived == expected, f"{text!r}: survived {survived!r}, wanted {expected!r}"


@pytest.mark.parametrize("text", _NO_LEAK_FIXTURES)
def test_not_one_matched_character_survives_into_the_output(text: str) -> None:
    """The invariant merging exists to hold: ambiguity resolves toward more redaction."""
    _assert_nothing_matched_survives(text)


# Emails, every numeric type, both card separators, an email whose match contains
# another type, and the joining characters that make spans touch, contain and
# partially overlap. Three tokens deep, so chains of all of those occur.
_SWEEP_TOKENS = [
    "a@b.com",
    "carol@example.org",
    "123-45-6789",
    "555-123-4567",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "123-456-7890@example.com",
    "a123-456-7890@example.com",
    # The conditional match. Without a token carrying a valid check digit the
    # corpus held ZERO 13-to-19 digit runs, so the whole contiguous-card path was
    # invisible to this sweep: measured, not assumed. With it, 445 of the 4913
    # strings carry a Luhn-valid run, and it contributes 49 partial overlaps and
    # 150 touching pairs the pattern table alone does not produce.
    "4111111111111111",
    "",
    " ",
    ".",
    "-",
    ",",
    "@",
    "x",
    "7",
]


def _sweep_corpus() -> Iterator[str]:
    """The single definition of the generated corpus.

    Shared by the sweep and the census on purpose. Two `itertools.product` calls
    would let the sweep's corpus drift from the one the coverage floors measure,
    and the floors would then vouch for a corpus that is not being tested.
    """
    for combo in itertools.product(_SWEEP_TOKENS, repeat=3):
        yield "".join(combo)


def _shape_census() -> dict[str, int]:
    """Count how many corpus strings exhibit each way two matches can relate.

    Computed from the patterns, like the oracle, and for the same reason: the
    question is what the CORPUS contains, which must not depend on what the
    detector did with it.

    The contiguous card counts here as well as in the oracle. Counting it in one
    and not the other would let the floors vouch for coverage the invariant is
    not checking, which is the drift `_sweep_corpus` exists to prevent one level
    up.
    """
    census = {"partial": 0, "contained": 0, "touching": 0}
    for text in _sweep_corpus():
        # Every start position, for the same reason the oracle uses them: what
        # the corpus contains must not depend on how the detector walks it.
        spans = sorted(
            [
                match.span()
                for _, pattern in _PATTERNS
                for start in range(len(text))
                if (match := pattern.match(text, start)) is not None
            ]
            + [
                match.span()
                for start in range(len(text))
                if (match := _CONTIGUOUS_CARD.match(text, start)) is not None
                and _luhn(match.group())
            ]
        )
        kinds: set[str] = set()
        for (a_start, a_end), (b_start, b_end) in itertools.combinations(spans, 2):
            if b_start < a_end and a_start < b_end:
                contained = (a_start <= b_start and b_end <= a_end) or (
                    b_start <= a_start and a_end <= b_end
                )
                kinds.add("contained" if contained else "partial")
            elif a_end == b_start or b_end == a_start:
                kinds.add("touching")
        for kind in kinds:
            census[kind] += 1
    return census


def test_nothing_matched_survives_across_a_generated_corpus() -> None:
    """Run the invariant over generated input, not only over cases I thought of.

    The leak that made this necessary was a shape nobody had written a fixture
    for. Fixtures prove the cases an author imagined; this covers the ones they
    did not. It lives in the suite rather than in a scratchpad so the evidence is
    reproducible by the next reader and runs in CI with everything else.
    """
    checked = redacted = 0
    for text in _sweep_corpus():
        checked += 1
        if _matched_offsets(text):
            redacted += 1
        _assert_nothing_matched_survives(text)

    # Guard the guard, on COVERAGE rather than volume. A ratio of redacting
    # inputs says nothing about which shapes are present, and its denominator
    # moves with the alphabet: deleting the space-separated card token takes
    # partial overlaps from 68 strings to ZERO while the ratio barely moves, and
    # the sweep would then run green against the very bug it exists to catch.
    # Partial overlap is that bug's shape, so it gets a floor of its own.
    assert checked == len(_SWEEP_TOKENS) ** 3
    shapes = _shape_census()
    # Floors sit just under the measured every-start counts (1763 / 2449 / 1198),
    # so they also catch the census regressing to a finditer walk, which would
    # report 98 / 1571 / 816. Partial overlap is the one that moves most, because
    # a bridging match between two adjacent addresses is exactly what finditer
    # cannot see.
    #
    # The counts moved with Task 18 and the old ones are kept here as the record
    # of by how much: 1510 / 2133 / 948 over sixteen tokens, against 95 / 1364 /
    # 680 for finditer. The rise is two things at once, a wider set of patterns
    # and a seventeenth token for the contiguous card, so the floors were
    # re-measured rather than scaled.
    assert shapes["partial"] >= 1700, f"corpus lost partial-overlap coverage: {shapes}"
    assert shapes["contained"] >= 2400, f"corpus lost containment coverage: {shapes}"
    assert shapes["touching"] >= 1150, f"corpus lost adjacency coverage: {shapes}"
    assert redacted > checked // 2, f"only {redacted} of {checked} inputs contained PII"


# One input per declared type, and the placeholder it should collapse to.
_SAMPLES = {
    "EMAIL": ("alice@example.com", "[REDACTED:EMAIL]"),
    "US_SSN": ("123-45-6789", "[REDACTED:US_SSN]"),
    "CREDIT_CARD": ("4111 1111 1111 1111", "[REDACTED:CREDIT_CARD]"),
    "PHONE_NUMBER": ("555-123-4567", "[REDACTED:PHONE_NUMBER]"),
}


def test_every_declared_type_is_reachable_on_its_own() -> None:
    """A name in PII_TYPES with no pattern behind it is a silent recall hole.

    `PII_TYPES` and the pattern table are two separate literals. A type added to
    the first and forgotten in the second is accepted at construction, matches
    nothing, and quietly lowers the recall this project publishes. Keying the
    samples by type and pinning the key set against PII_TYPES makes that a test
    failure rather than a number nobody can explain.

    Selecting one type at a time also exercises the subset filter for all four,
    where the brief exercises it for EMAIL alone.
    """
    assert set(_SAMPLES) == PII_TYPES

    for pii_type, (text, expected) in _SAMPLES.items():
        verdict = PiiGuardrail(types=frozenset({pii_type})).check(text, OUT)
        assert verdict.decision == "redact", pii_type
        assert verdict.content == expected, pii_type
        assert [f.type for f in verdict.findings] == [pii_type], pii_type


def test_the_email_pattern_does_not_degrade_quadratically() -> None:
    """A ReDoS guard on a component that reads attacker-influenced text.

    The local-part class contains `.`, so a word boundary holds after every dot.
    With unbounded repetition that gives O(n) start positions each scanning O(n)
    characters for an `@` that never comes, and the cost grew 4.0x per doubling:
    16 ms at 4 KB, 828 ms at 32 KB, 3.3 SECONDS at 64 KB. Bounded to RFC 5321's
    limits, and with no start anchor to close the bypass, it grows 2.0x per
    doubling and takes 8.9 ms at 64 KB.

    The budget below is roughly 220x the measured time and 0.6x the time the old
    pattern took, so it is loose enough not to flake on a slow or loaded runner
    and still fails outright if the bounds are ever taken off.
    """
    payload = "x " + "a." * 32000 + "@"
    assert len(payload) > 64000

    start = time.perf_counter()
    verdict = PiiGuardrail().check(payload, OUT)
    elapsed = time.perf_counter() - start

    assert verdict.decision == "allow"
    assert elapsed < 2.0, f"EMAIL pattern took {elapsed:.2f}s on {len(payload)} characters"


def test_padding_before_an_address_does_not_hide_it() -> None:
    """Regression guard: a bounded local part plus a start anchor was a bypass.

    The bound applies to the WHOLE local part, so with a leading word-boundary
    anchor sixty characters of padding put the `@` out of reach of every legal
    start position, and the address came back untouched with decision allow.
    The string is not an address, it CONTAINS one, and anyone reading the output
    could lift it straight out.

    Without the anchor the match simply begins later, inside the padding, so the
    address is covered whatever the padding length. Padding gets redacted along
    with it, which is the safe direction for a redactor to err in.

    An earlier version of this file asserted the miss as intended behaviour. That
    made it worse than an untested hole: Task 15 would have scored recall against
    a corpus that called this acceptable.
    """
    for padding in (0, 55, 60, 64, 200, 1000):
        text = "contact " + "u" * padding + "alice@example.com"
        verdict = PiiGuardrail().check(text, OUT)

        assert verdict.decision == "redact", padding
        assert verdict.content is not None
        assert "alice@example.com" not in verdict.content, padding
        # Not even the domain half may survive.
        assert "@example.com" not in verdict.content, padding


def test_an_over_long_dotless_domain_is_a_complete_miss() -> None:
    """The one place the RFC bound still costs recall, stated accurately.

    A domain run of more than 255 characters with no dot in it is not matched at
    all. An earlier report called this a "shorter match", which was true of a
    different payload and false of this one: nothing is redacted, the whole
    string comes back allow.

    Much lower severity than the padding bypass, and for a reason worth stating:
    padding a domain past 255 characters destroys the address rather than hiding
    a working one. The result is undeliverable, so there is no live address in
    the output to recover. Padding a LOCAL part hid a working address, which is
    why that one was a bug and this one is a documented limit.
    """
    assert PiiGuardrail().check("x@" + "d" * 255 + ".com", OUT).decision == "redact"

    verdict = PiiGuardrail().check("x@" + "d" * 256 + ".com", OUT)
    assert verdict.decision == "allow"
    assert verdict.content is None


def test_a_word_character_after_the_tld_does_not_hide_an_address() -> None:
    """Regression guard: the trailing anchor was the mirror of the leading one.

    One word character after the TLD defeated the trailing word boundary, so
    "a@b.com9" came back allow with the address whole and liftable. Over 150,000
    randomised trials, 45,516 delimited addresses survived entirely with that
    anchor in place and none survive without it.

    The anchor WAS doing something, and the earlier claim here that it "was never
    doing useful work" was false. It was buying precision on tokens whose dotted
    tail resembles a TLD and is followed by a digit, which is why removing it
    added the last three entries in the known-false-positive list below. That
    shape is close to absent from real text: 400 MB of source, logs, lockfiles
    and docs yielded zero new matches. A near-zero precision cost in exchange for
    an evasion that leaked whole addresses is a trade worth making, but it is a
    trade and not a free win.
    """
    for suffix in ("9", "7", "_", "0", "\u00e9"):
        text = "mail alice@example.com" + suffix
        verdict = PiiGuardrail().check(text, OUT)

        assert verdict.decision == "redact", suffix
        assert verdict.content is not None
        assert "alice@example.com" not in verdict.content, suffix
        assert "@example.com" not in verdict.content, suffix


# Every string known to match EMAIL that a reader would not call personal data.
# All twenty, not a sample: an earlier version of this file pinned six and the
# handoff claimed they were all of them, which would have had Task 14 build its
# corpus from a list that omitted the hard cases and publish a precision number
# measured on the easy ones.
#
# Split by cause, because the two groups should be scored differently. The first
# group is structurally a valid address and is only a false positive in context,
# so a corpus may reasonably label some of them true positives. The second group
# is not an address by any reading.
_ADDRESS_SHAPED_BUT_MAYBE_NOT_PII = [
    ("handle@twitter.com", "a social handle written as an address"),
    ("mod@github.com", "a repository reference"),
    ("user@host.com:8080", "URL userinfo, redacts the address half"),
    ("f@a.bc", "a minimal but well-formed address"),
]

_NOT_ADDRESSES_AT_ALL = [
    # Pre-existing under every variant, including the brief's original pattern.
    ("run@2x.png", "an asset filename"),
    ("img@sha256.abcdef", "an OCI-style digest"),
    ("app@v1.2.beta", "a prerelease version spec"),
    # Introduced deliberately when the trailing anchor was removed. Each is a
    # dotted tail that reads as a TLD followed by a digit, which is the exact
    # shape "a@b.com9" used to evade redaction with.
    ("myimage@sha256.abc123", "digest with a trailing digit"),
    ("build@node.js18", "a toolchain pin"),
    ("cache@v2.0.rc1", "a release candidate tag"),
    ("app@v1.2.beta3", "a prerelease with a build number"),
    ("dep@lodash.merge4", "a package subpath"),
    ("ref@main.sha1", "a git ref"),
    ("x@y.co2", "a short label with a digit"),
    ("svc@cluster.local1", "a Kubernetes service name"),
    ("key@aes.gcm256", "a cipher suite"),
    ("art@nexus.repo3", "an artifact repository"),
    ("job@ci.pipeline7", "a CI pipeline id"),
    ("id@ns.uuid4", "a namespaced identifier"),
    ("obj@config.yaml2", "a versioned config filename"),
]

_KNOWN_FALSE_POSITIVES = _ADDRESS_SHAPED_BUT_MAYBE_NOT_PII + _NOT_ADDRESSES_AT_ALL


@pytest.mark.parametrize(("text", "why"), _KNOWN_FALSE_POSITIVES)
def test_known_false_positives_are_recorded_not_forgotten(text: str, why: str) -> None:
    """Pin the precision cost so Task 14 scores it and Task 15 publishes it honestly.

    The thirteen digit-suffixed entries arrived with the trailing anchor removal.
    That trade was made deliberately and is defended in the pattern's own comment:
    the shape is essentially absent from real text, and the anchor was leaking
    whole addresses. It still belongs written down.

    These assert CURRENT behaviour, not desired behaviour. If a later change makes
    one of them allow, that is an improvement, and the failure here is the prompt
    to move it out of this list rather than to restore the old behaviour.
    """
    assert PiiGuardrail().check(text, OUT).decision == "redact", why


def test_the_false_positive_record_is_complete() -> None:
    """Guard the record itself, since the last one was silently partial."""
    assert len(_KNOWN_FALSE_POSITIVES) == 20
    assert len({text for text, _ in _KNOWN_FALSE_POSITIVES}) == 20


def test_an_over_long_dotted_domain_redacts_all_but_the_final_label() -> None:
    """The dotted sibling of the dotless miss, which had no test of its own.

    With interior dots the engine backtracks to a parse that fits the 255
    character bound, so this is a partial redaction rather than the complete miss
    the dotless case gives. The final label survives. That is a domain fragment
    and not a recoverable address, which is why it is recorded rather than
    treated as the bypass its dotless cousin resembles.
    """
    text = "a@" + "d" * 249 + ".example.com"
    verdict = PiiGuardrail().check(text, OUT)

    assert verdict.decision == "redact"
    assert verdict.content == "[REDACTED:EMAIL].com"


# U+0440 U+0444 is ".rf" in Cyrillic, the Russian ccTLD. Written as escapes for the
# same reason the non-ASCII local-part fixture is: a normalising editor must not be
# able to turn a miss record into something that quietly passes for another reason.
_RF = "\u0440\u0444"
_PRIMER = "\u043f\u0440\u0438\u043c\u0435\u0440"


def test_internationalised_domains_are_redacted() -> None:
    """This WAS the most consequential miss the detector had. Task 18 closed it.

    A whole deliverable address used to survive, because the domain and TLD
    classes were ASCII and an address written in native script matched nothing at
    all. Both halves are `\\w` now, and the TLD is `[^\\W\\d_]`, which is every
    Unicode LETTER and no digit.
    """
    for domain in (f"example.{_RF}", f"{_PRIMER}.com"):
        text = f"mail alice@{domain}"
        verdict = PiiGuardrail().check(text, OUT)

        assert verdict.decision == "redact", domain
        assert [f.type for f in verdict.findings] == ["EMAIL"], domain
        assert verdict.content == "mail [REDACTED:EMAIL]", domain


# U+092D U+093E U+0930 U+0924 is ".bharat" in Devanagari, India's IDN ccTLD.
_BHARAT = "\u092d\u093e\u0930\u0924"


def test_a_tld_carrying_an_indic_mark_is_still_a_miss() -> None:
    """The Unicode the widened classes still do NOT cover, recorded not guessed.

    U+093E is a Devanagari vowel sign: a spacing combining MARK, not a letter, so
    it is outside `\\w` and outside `[^\\W\\d_]`, and it is outside the U+0300 to
    U+036F range the local part adds for NFD tails. The TLD needs two letters in a
    row and gets one, so this address is a complete miss.

    Not closed here because closing it means every Unicode mark, and Python's `re`
    has no category class to say that: the honest fix enumerates marks per script
    or builds the class from `unicodedata` at import, which is a design decision
    rather than a patch. Task 14's corpus must carry this as a miss so the
    published recall number pays for it. The scripts WITHOUT combining marks in
    their labels, Cyrillic and Han among them, are covered and tested above.
    """
    verdict = PiiGuardrail().check(f"mail alice@example.{_BHARAT}", OUT)

    assert verdict.decision == "allow"
    assert verdict.content is None


def test_a_punycode_domain_is_redacted_but_loses_its_last_label() -> None:
    """The punycode spelling of the same address, which is ASCII and so does match.

    It matches only up to the final `xn--` label: the TLD class is letters only,
    so it stops at the first digit inside `p1ai` and the rest survives. A domain
    fragment rather than a recoverable address, in the same family as the dotted
    over-long domain above, but worth its own record because the input here is a
    perfectly ordinary address.
    """
    verdict = PiiGuardrail().check("mail alice@xn--p1ai.xn--p1ai", OUT)

    assert verdict.decision == "redact"
    assert verdict.content == "mail [REDACTED:EMAIL]--p1ai"
    assert "alice" not in (verdict.content or "")


# (label, decoy, real). Each decoy MATCHES; a failing decoy proves nothing here.
_SUCCEEDING_DECOYS = [
    ("domain eats the local part", "a@b.com", "alice@example.com"),
    ("short decoy, longer victim", "x@y.co", "bob@corp.example.com"),
    ("decoy carries a dotted tail", "u@v.example.org", "carol@example.net"),
]


@pytest.mark.parametrize(("label", "decoy", "real"), _SUCCEEDING_DECOYS)
def test_a_succeeding_decoy_does_not_hide_the_address_behind_it(
    label: str, decoy: str, real: str
) -> None:
    """Regression guard for a Critical: the scan skipped the start it needed.

    `finditer` resumes at the END of each match, so an address joined directly
    onto another one runs its greedy domain into the second address's local part,
    stops at the `@`, and leaves the scan resuming INSIDE the real address. Its
    start is then never tried:

        'a@b.comalice@example.com'  ->  '[REDACTED:EMAIL]@example.com'

    The domain half of a working address, standing in the output of a redactor,
    after five rounds of leak fixes on this file. The oracle could not see it
    because it used `finditer` too.

    This is not the padding bypass and not the partial-overlap drop; it is the
    scan itself, and it was live in `secrets.py` on four types at the same time.
    """
    text = f"mail {decoy}{real} now"
    verdict = PiiGuardrail().check(text, OUT)

    assert verdict.decision == "redact", label
    survived = verdict.content or ""
    assert real not in survived, f"{label}: whole address survived"
    # The domain half alone identifies a person's employer and is half the
    # address; a leak that keeps it is still a leak.
    assert real.split("@")[1] not in survived, f"{label}: domain survived"
    _assert_nothing_matched_survives(text)


# --------------------------------------------------------------------------------
# Task 18: the layouts an independent review measured coming back `allow`.
#
# Every case below was reproduced by hand before it was written down, and each
# addition to the detector carries its own negative directly underneath it. The
# negatives are the price list: an addition that cannot keep them allowed is not
# worth the recall it buys.
# --------------------------------------------------------------------------------

CONTIGUOUS_CARDS = [
    "4111111111111111",  # Visa, 16 digits, passes Luhn
    "378282246310005",  # Amex, 15 digits, passes Luhn
    "4000000000000000006",  # 19 digits, passes Luhn
]

GROUPED_CARDS = [
    "3782 822463 10005",  # Amex 4-6-5, the layout Amex actually prints
    "4111-1111-1111-1111",
]

PHONES = [
    "(415) 555-2671",
    "415.555.2671",
    "+1-415-555-2671",
    "+14155552671",
]


@pytest.mark.parametrize("value", CONTIGUOUS_CARDS + GROUPED_CARDS)
def test_a_card_layout_people_actually_paste_is_redacted(value: str) -> None:
    verdict = PiiGuardrail().check(value, OUT)
    assert verdict.decision == "redact"
    assert [f.type for f in verdict.findings] == ["CREDIT_CARD"]
    assert verdict.content == "[REDACTED:CREDIT_CARD]"


@pytest.mark.parametrize("value", PHONES)
def test_a_us_phone_layout_people_actually_paste_is_redacted(value: str) -> None:
    verdict = PiiGuardrail().check(value, OUT)
    assert verdict.decision == "redact"
    assert [f.type for f in verdict.findings] == ["PHONE_NUMBER"]


def test_a_non_ascii_local_part_is_still_an_email() -> None:
    verdict = PiiGuardrail().check("jose\u0301@example.com", OUT)
    assert [f.type for f in verdict.findings] == ["EMAIL"]


def test_a_non_ascii_domain_is_still_an_email() -> None:
    verdict = PiiGuardrail().check("alice@example.\u0440\u0444", OUT)
    assert [f.type for f in verdict.findings] == ["EMAIL"]


def test_a_word_hyphen_before_a_value_does_not_hide_it() -> None:
    r"""`(?<![\d-])` blocked any hyphen, so a LABEL hid the value it labelled.
    Only a hyphen continuing a digit run should block."""
    verdict = PiiGuardrail().check("ssn-123-45-6789", OUT)
    assert [f.type for f in verdict.findings] == ["US_SSN"]
    assert verdict.content == "ssn-[REDACTED:US_SSN]"


def test_a_digit_run_before_a_value_still_hides_it() -> None:
    """The negative half of the same rule, which must not regress."""
    assert PiiGuardrail().check("6789987-65-4321", OUT).decision == "allow"


# Negatives. Each is the precision cost of one addition above, and each must
# stay an allow or the addition is not worth its recall.
NOT_CARDS = [
    "4111111111111112",  # 16 digits, FAILS Luhn
    "1234567890123456",  # sequential, fails Luhn
    "20260830143012000000",  # a 20-digit timestamp id
    "123456789012",  # 12 digits, below the range
]


@pytest.mark.parametrize("value", NOT_CARDS)
def test_a_digit_run_that_is_not_a_card_stays_allowed(value: str) -> None:
    assert PiiGuardrail().check(value, OUT).decision == "allow"


NOT_PHONES = ["192.168.1.100", "2026.08.30", "(415) 55-2671"]


@pytest.mark.parametrize("value", NOT_PHONES)
def test_a_number_that_is_not_a_phone_stays_allowed(value: str) -> None:
    assert PiiGuardrail().check(value, OUT).decision == "allow"


def test_the_contiguous_card_respects_the_type_selection() -> None:
    """The second site the type filter has to reach, which is not in the table.

    `__init__` filters `_PATTERNS`, and the contiguous card is not in `_PATTERNS`.
    A filter that only walked the table would redact card numbers for a caller who
    asked for EMAIL alone.

    The `types=frozenset()` case that used to sit here is gone because that
    selection is now refused at construction; the refusal is asserted in
    `test_an_empty_type_set_is_refused_rather_than_selecting_nothing`.
    """
    card = "4111111111111111"

    assert PiiGuardrail(types=frozenset({"CREDIT_CARD"})).check(card, OUT).decision == "redact"
    assert PiiGuardrail(types=frozenset({"EMAIL"})).check(card, OUT).decision == "allow"


# Doubling as a table rather than as arithmetic: a second implementation of the
# only interesting half of Luhn, so the fixtures below are not built by the
# function they are used to test.
_DOUBLED = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 1, 6: 3, 7: 5, 8: 7, 9: 9}


def _with_check_digit(prefix: str) -> str:
    """Append the digit that makes `prefix` a valid Luhn string.

    The last digit of `prefix` ends up one place from the right in the result, so
    it is the first one doubled, which is why the parity here is the opposite of
    `_luhn`'s. Getting that backwards produces fixtures that are wrong in exactly
    the way that would still look convincing.
    """
    total = 0
    for index, char in enumerate(reversed(prefix)):
        value = int(char)
        total += _DOUBLED[value] if index % 2 == 0 else value
    return prefix + str(-total % 10)


def test_the_check_digit_agrees_with_a_second_implementation() -> None:
    """Cross-check `_luhn` against the table, on real cards and on near misses.

    `_luhn` is the one piece of arithmetic in this detector, and an arithmetic
    bug that made it MORE permissive would show up as a precision loss nobody
    could locate. Every string built by `_with_check_digit` must pass, and every
    string one digit away from it must fail: the check is worth nothing if it
    accepts a neighbour.
    """
    for real in ("4111111111111111", "378282246310005", "4000000000000000006"):
        assert _with_check_digit(real[:-1]) == real, real
        assert _luhn(real), real

    for prefix in ("41111111111", "4000000000000000", "37828224631000", "9"):
        valid = _with_check_digit(prefix)
        assert _luhn(valid), valid
        for wrong in range(10):
            candidate = prefix + str(wrong)
            assert _luhn(candidate) == (candidate == valid), candidate


@pytest.mark.parametrize(
    ("label", "prefix"),
    [
        ("twelve, below the range", "41111111111"),
        ("thirteen, at the floor", "411111111111"),
        ("nineteen, at the ceiling", "400000000000000000"),
        ("twenty, over the ceiling", "4000000000000000000"),
    ],
)
def test_the_contiguous_card_range_is_pinned_from_both_sides(label: str, prefix: str) -> None:
    """A bound nobody pins from both sides can be moved either way unnoticed.

    Thirteen and nineteen are real card lengths, so the range is not arbitrary:
    thirteen is a legacy Visa and nineteen is the ISO/IEC 7812 maximum. Every
    fixture here carries a VALID check digit, including the two that must not
    match, so it is the LENGTH turning those away and not the arithmetic. Built
    from a prefix rather than written out, because a hand-typed twenty-digit run
    that quietly failed Luhn would pass this test for the wrong reason.
    """
    digits = _with_check_digit(prefix)
    assert _luhn(digits), f"{label}: fixture must pass Luhn or it proves nothing"

    decision = PiiGuardrail().check(digits, OUT).decision
    expected = "redact" if 13 <= len(digits) <= 19 else "allow"
    assert decision == expected, f"{label}: {len(digits)} digits, {digits}"


def test_a_bare_ten_digit_run_needs_its_country_code() -> None:
    """The precision the `+1` branch buys, stated as the case it must not match.

    Ten unseparated digits is a Unix timestamp in seconds, which is in nearly
    every log line written. Dropping the `+1` requirement and matching bare tens
    would fire on all of them. The `+1` is what turns a digit run into a phone
    number here, the way Luhn does for a bare card run.
    """
    for allowed in ("4155552671", "ts 1756512000", "1756512000 1756512001"):
        assert PiiGuardrail().check(allowed, OUT).decision == "allow", allowed

    for redacted in ("+14155552671", "+1 4155552671", "+1-4155552671"):
        verdict = PiiGuardrail().check(redacted, OUT)
        assert verdict.decision == "redact", redacted
        assert [f.type for f in verdict.findings] == ["PHONE_NUMBER"], redacted


# Every valid IPv4 digit width, in every position: 0 and 255 for the ends, and a
# one, two and three digit value either side of each carry. 6561 addresses.
_IPV4_OCTETS = ["0", "1", "9", "10", "99", "100", "199", "200", "255"]


def test_no_valid_ipv4_address_reads_as_a_phone_number() -> None:
    """The dot separator's precision cost, bounded by measurement not by assertion.

    Accepting `.` as a phone separator is what makes "415.555.2671" redact, and it
    is also what puts a dotted digit run in reach of the pattern. The bound on the
    damage is structural: an octet is at most 255, so at most three digits, and
    the pattern's final group is FOUR consecutive digits. No valid address can
    contain that, so none of them can match, whatever the widths of the others.

    Swept rather than argued, because the argument is exactly the kind that is
    right about the case in hand and wrong one step over.
    """
    guardrail = PiiGuardrail()
    matched = [
        ip
        for quad in itertools.product(_IPV4_OCTETS, repeat=4)
        if guardrail.check((ip := ".".join(quad)), OUT).decision != "allow"
    ]

    assert matched == [], f"valid IPv4 addresses read as phone numbers: {matched[:5]}"


# The precision this task spent, priced. Each one redacts today and should not,
# and each is here so Task 14 scores it and Task 15 publishes it rather than
# discovering it. As with the list above, these assert CURRENT behaviour: if a
# later change makes one an allow, move it out rather than restoring anything.
_NEW_FALSE_POSITIVES = [
    (
        "host 192.168.100.4001",
        "PHONE_NUMBER",
        (
            "a dotted 3-3-4 digit run, here a malformed IPv4 whose last group is "
            "four digits. Every VALID address stays an allow, which is swept above."
        ),
    ),
    (
        "imei 490154203237518",
        "CREDIT_CARD",
        (
            "an IMEI. It carries a Luhn check digit of its own and begins 49, which "
            "is inside Visa's range, so neither guard separates it from a card. The "
            "redaction is the right direction, since an IMEI is personal data too; "
            "the TYPE in the audit record is what is wrong, and no leading digit "
            "can fix that."
        ),
    ),
    (
        "delta +12345678901",
        "PHONE_NUMBER",
        (
            "a signed integer of phone width. `+1` and ten digits is E.164 for a US "
            "number, so the characters do not distinguish the two and no guard here "
            "could without losing the E.164 spelling."
        ),
    ),
    (
        "order 1234 567890 12345",
        "CREDIT_CARD",
        (
            "a 4-6-5 grouped run that is not an Amex. The grouped pattern carries no "
            "Luhn check DELIBERATELY, because grouping is itself the evidence and a "
            "check behind it would turn cases that redact today into silent allows."
        ),
    ),
    (
        "sha256 4111111111111111abc",
        "CREDIT_CARD",
        (
            "a digest whose leading run happens to pass Luhn. Inherent to matching a "
            "bare run at all, and the reason the check digit is required for one."
        ),
    ),
]


@pytest.mark.parametrize(("text", "pii_type", "why"), _NEW_FALSE_POSITIVES)
def test_the_precision_this_task_spent_is_recorded(text: str, pii_type: str, why: str) -> None:
    verdict = PiiGuardrail().check(text, OUT)

    assert verdict.decision == "redact", why
    assert pii_type in [f.type for f in verdict.findings], why


def test_a_user_at_an_ip_address_is_not_an_email() -> None:
    """Why the TLD is LETTERS and not `\\w`, which the wider classes tempt.

    `\\.\\w{2,}` reads a trailing numeric label as a TLD, so "root@192.168.1.100"
    becomes an EMAIL: an SSH target, present in every server log there is, scored
    as a person. Letters cost nothing real in exchange, because no delegated TLD
    contains a digit except the `xn--` A-labels, and those stop at their own
    hyphen under either class, which the punycode test above pins.
    """
    for text in ("root@192.168.1.100", "admin@10.0.0.1", "deploy@172.16.0.42"):
        assert PiiGuardrail().check(text, OUT).decision == "allow", text


# --------------------------------------------------------------------------------
# The two guards the fix round added, each with the negatives that hold it down.
# Every fixture below was checked in BOTH directions before it was written here:
# it must be an allow now, and it must REDACT with its guard removed. A candidate
# UUID, "550e8400-e29b-41d4-a716-446655440000", was dropped for failing the second
# half, since a fixture that is clean either way holds nothing down.
# --------------------------------------------------------------------------------

_REQUIRED_SEPARATOR_NEGATIVES = [
    (
        "0ad830af-c60c-657d-c147-7008575ea35c",
        "a UUIDv4, whose '147-7008575' reads as a phone with the separator optional",
    ),
    ("part 123-4567890", "a part number in the NNN-NNNNNNN shape the `?` allowed"),
    ("ticket 800.5551212", "the same shape with the dot separator"),
]


@pytest.mark.parametrize(("text", "guard"), _REQUIRED_SEPARATOR_NEGATIVES)
def test_a_number_missing_its_second_separator_is_not_a_phone(text: str, guard: str) -> None:
    """One character of the pattern cost 371 UUIDs in 200,000 and bought nothing.

    With `\\d{3}[-.]\\d{3}[-.]?\\d{4}` the second separator was optional, which
    makes `NNN[-.]NNNNNNN` a phone number and takes the middle out of a UUID:

        0ad830af-c60c-657d-c147-7008575ea35c
            ->  0ad830af-c60c-657d-c[REDACTED:PHONE_NUMBER]ea35c

    A correlation id loses its middle AND the audit record asserts a phone number
    that nobody wrote, which is the worse half: a wrong redaction is noise, a
    wrong FINDING is a false statement about the input. Measured at 371 of
    200,000 random UUIDv4s with the character present and ZERO without it.

    It bought no recall at all. The whole suite passes with the character
    removed, so not one of the four target layouts ever needed it.
    """
    verdict = PiiGuardrail().check(text, OUT)
    assert verdict.decision == "allow", f"{guard}: {text!r} -> {verdict.findings}"


_MII_NEGATIVES = [
    ("1767205800009", "an epoch-millisecond timestamp that passes Luhn"),
    (
        '{"event":"click","ts":1767205800009,"user":"u42"}',
        "the same value in the JSON log line it actually arrives in",
    ),
    ("1767205800000008", "an epoch-microsecond timestamp, 16 digits, the commonest card length"),
    (
        "c81b352b270b2e498256b5e24c726a039cd665477a83d71d95633340782782d4",
        "a sha256 digest whose '95633340782782' run passes Luhn",
    ),
    ("0000000000000000", "a zero-padded fixed-width field, whose digit sum is 0"),
    ("9999999999999995", "a nines-padded sentinel with a real check digit"),
]


@pytest.mark.parametrize(("text", "guard"), _MII_NEGATIVES)
def test_a_luhn_valid_run_outside_the_card_issuing_range_is_not_a_card(
    text: str, guard: str
) -> None:
    """Luhn alone leaves one machine-written log line in ten reading as a card.

    Measured, not estimated: exactly 10.0000% of 20,000 consecutive epoch-ms
    values pass Luhn, and 10.0045% of 200,000 uniform 13-to-19 digit runs.
    Epoch-ms is thirteen digits and is in nearly every log line there is, so this
    was not a rare shape, it was the commonest one the detector would ever see.

    The first digit of a payment card is its Major Industry Identifier and cards
    are issued only under MII 2 to 6 (ISO/IEC 7812), so `[2-6]` is a constraint
    from a published standard rather than a threshold fitted to these fixtures.
    It halves what Luhn leaves: 4.9595% of uniform runs and 0.2155% of sha256
    digests, against 10.0045% and 0.4305%.

    THIS GUARD HAS AN EXPIRY DATE and the module says so beside the pattern.
    Epoch-ms first carries a leading 2 on 2033-05-18T03:33:20Z, and past that
    boundary timestamps are back inside the MII range: the same measurement gives
    10.0000% both with the guard and without it. These fixtures are all before
    that date, so they will keep passing; the shape they stand for will not.
    """
    verdict = PiiGuardrail().check(text, OUT)
    assert verdict.decision == "allow", f"{guard}: {text!r} -> {verdict.findings}"


def test_the_card_issuing_range_is_pinned_at_both_of_its_edges() -> None:
    """The range is 2 to 6, so 1 and 7 are where a wrong bound shows up.

    A fixture inside the range proves the pattern matches something; only the
    digits either side of each edge prove it is THIS range. Every value is
    sixteen digits and carries a valid check digit, built rather than typed, so
    the leading digit is the only thing that varies.
    """
    for lead in "0123456789":
        digits = _with_check_digit(lead + "11111111111111")
        assert _luhn(digits) and len(digits) == 16, digits

        decision = PiiGuardrail().check(digits, OUT).decision
        expected = "redact" if lead in "23456" else "allow"
        assert decision == expected, f"leading {lead}: {digits} -> {decision}"
