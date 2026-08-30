import base64
import hashlib
import itertools
import json
import re
import time
from collections.abc import Callable, Iterator
from typing import cast

import pytest

from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.detectors.secrets import (
    _PATTERNS,
    SECRET_TYPES,
    SecretsGuardrail,
    _private_key_spans,
)
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import Context, Decision

OUT = Context(direction="output", origin="model")
IN = Context(direction="input", origin="user")

# Wall-clock budget for the ReDoS probes below, in seconds, PER SECRET TYPE.
#
# A timing assertion cannot tell a slow machine from a slow algorithm: both
# raise the number. All a budget can do is sit between two costs, and it is
# only defensible next to both of them. Every figure below was measured on this
# machine, a 14-core arm64, on the same 256 KB payload each probe already uses.
#
#     type            linear   linear   quadratic   budget   flake   detect
#                       idle   loaded    (bounds             margin  margin
#                                        removed)
#     JWT              0.463    1.314       14.50      4.0     3.0x    3.6x
#     SLACK_TOKEN      0.117    0.360        9.16      1.5     4.2x    6.1x
#     PRIVATE_KEY      0.131    0.329           -      1.5     4.6x       -
#     OPENAI_KEY       0.076    0.198        4.17      1.5     7.6x    2.8x
#     ANTHROPIC_KEY    0.082    0.092        6.50      1.5    16.2x    4.3x
#     AWS_ACCESS_KEY   0.073    0.132           -      1.5    11.4x       -
#     GITHUB_TOKEN     0.002    0.002           -      1.5      777x       -
#
# "loaded" is 28 spin loops against 14 cores; each row is the worst of that
# type's probes, timed on `check` itself rather than read off pytest's
# durations, so it is the quantity the assertion compares. The whole suite was
# separately watched go from 3.9 s to 79 s across that same pair of conditions,
# a 20x wall-clock swing under which these probes themselves moved only 3x, and
# a shared CI runner is the loaded case by default AND slower per core, so the
# two factors multiply. AWS and GITHUB have no quadratic column because their
# bodies cannot swallow their own prefix, and PRIVATE_KEY is a walk rather than
# a regex; their probes pin that they stay cheap.
#
# The budget is PER TYPE and not one shared number, which is what the previous
# 2.0 s was. That single number was reasoned for JWT and then applied to the
# other 18 probes, whose linear costs span 0.002 s to 0.13 s and whose cheapest
# blowup is 3.5x under JWT's. One number over a 230x spread of linear cost is
# too high at one end and too low at the other:
#
#   - Too low for JWT. 1.31 s loaded is only 1.5x under 2.0 s, and these tests
#     have gone red on this machine when it was busy. A test that goes red when
#     the runner is busy destroys the meaning of a red build.
#   - Too high, once widened to cover JWT, for OPENAI_KEY. Its blowup is
#     4.17 s, so a 4.0 s budget clears it by 4%: a probe that still reads as a
#     guard while no longer being able to fail against the bug it exists for.
#     That is not a hypothetical. It is what a shared 4.0 s measured.
#
# Split by type, both ends have room. The tightest flake margin is JWT's 3.0x,
# so a runner has to be about 3x slower than this machine while fully saturated
# before anything goes red for the wrong reason; the tightest detection margin
# is OPENAI_KEY's 2.8x, so removing a bound still fails outright.
#
# Both halves were checked rather than argued. The loaded column is measured.
# The detection column is measured too, by taking the upper bound off each of
# the four patterns that HAS one (JWT, SLACK_TOKEN, ANTHROPIC_KEY, OPENAI_KEY)
# and watching all eight of their probes go red against these budgets. The
# other three types have no bound to remove, so their rows have no detection
# margin to claim and none is claimed: what their probes pin is that a future
# edit does not give them one of these costs.
#
# Marking them non-gating instead was rejected. Three of these patterns shipped
# quadratic and were caught here, so a probe that does not run in CI is the
# green check that means the review never ran.
#
# Keyed by type and pinned against SECRET_TYPES below, for the reason
# `_DENSE_PREFIXES` is: a new type with no entry must be a failure rather than
# an omission that inherits whatever default was nearest.
_REDOS_BUDGET_SECONDS: dict[str, float] = {
    "AWS_ACCESS_KEY": 1.5,
    "GITHUB_TOKEN": 1.5,
    "OPENAI_KEY": 1.5,
    "ANTHROPIC_KEY": 1.5,
    "SLACK_TOKEN": 1.5,
    "PRIVATE_KEY": 1.5,
    "JWT": 4.0,
}


def test_declares_its_types() -> None:
    assert SECRET_TYPES == frozenset(
        {
            "AWS_ACCESS_KEY",
            "GITHUB_TOKEN",
            "OPENAI_KEY",
            "ANTHROPIC_KEY",
            "SLACK_TOKEN",
            "PRIVATE_KEY",
            "JWT",
        }
    )


def test_redacts_an_aws_access_key() -> None:
    verdict = SecretsGuardrail().check("key AKIAIOSFODNN7EXAMPLE here", OUT)
    assert verdict.decision == "redact"
    assert verdict.content == "key [REDACTED:AWS_ACCESS_KEY] here"


def test_redacts_each_supported_shape() -> None:
    cases = {
        "AKIAIOSFODNN7EXAMPLE": "AWS_ACCESS_KEY",
        "ghp_" + "a" * 36: "GITHUB_TOKEN",
        "sk-ant-api03-" + "a" * 20: "ANTHROPIC_KEY",
        "sk-" + "a" * 32: "OPENAI_KEY",
        "sk-proj-" + "a" * 40: "OPENAI_KEY",
        "sk-proj-" + "a" * 15 + "_" + "b" * 15 + "-" + "c" * 10: "OPENAI_KEY",
        "sk-svcacct-" + "a" * 40: "OPENAI_KEY",
        "sk-admin-" + "a" * 40: "OPENAI_KEY",
        "xoxb-123456789012-abcdefghijkl": "SLACK_TOKEN",
        "-----BEGIN RSA PRIVATE KEY-----": "PRIVATE_KEY",
        "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln": "JWT",
    }
    for text, expected in cases.items():
        verdict = SecretsGuardrail().check(f"value {text} end", OUT)
        assert [f.type for f in verdict.findings] == [expected], text


def test_default_is_redact_and_deny_is_opt_in() -> None:
    assert SecretsGuardrail().check("AKIAIOSFODNN7EXAMPLE", OUT).decision == "redact"
    denying = SecretsGuardrail(on_match="deny")
    verdict = denying.check("AKIAIOSFODNN7EXAMPLE", OUT)
    assert verdict.decision == "deny"
    assert verdict.content is None  # a deny carries no rewritten content
    assert [f.type for f in verdict.findings] == ["AWS_ACCESS_KEY"]


def test_adversarial_negatives_are_not_flagged() -> None:
    """These are what an honest precision number is measured against."""
    for text in [
        "a" * 40,  # bare hex-ish blob
        "commit 3f2a1c9d8e7b6a5f4e3d2c1b0a9f8e7d6c5b4a39",  # git SHA
        "550e8400-e29b-41d4-a716-446655440000",  # UUID
        "c2VjcmV0IG1lc3NhZ2UgaGVyZQ==",  # plain base64
        "sk",  # prefix alone
        # Not an "sk-" lookalike: "sky" is s-k-y and the string holds no "sk-"
        # at all, so no loosening of a pattern that requires that literal can
        # reach it. The brief's Step 5 names this as the case that catches a
        # loose `sk-\S+`; the kebab identifier below is what actually does.
        "the sky-blue paint",
        "sk-this-is-a-very-long-kebab-case-identifier-name",  # see below
    ]:
        assert SecretsGuardrail().check(text, OUT).decision == "allow", text


def test_padding_around_a_credential_does_not_hide_it() -> None:
    """Task 6 shipped both of these in EMAIL and spent two rounds removing them.

    A leading word-boundary anchor means one character of padding hides a live
    credential; a trailing one means one character of suffix does. Each case
    below is a COMPLETE miss under the anchored form, with the whole key
    liftable from the output.
    """
    keys = {
        "AWS_ACCESS_KEY": "AKIAIOSFODNN7EXAMPLE",
        "GITHUB_TOKEN": "ghp_" + "a" * 36,
        "ANTHROPIC_KEY": "sk-ant-api03-" + "a" * 20,
        "OPENAI_KEY": "sk-proj-" + "a" * 40,
        "SLACK_TOKEN": "xoxb-123456789012-abcdefghijkl",
        "JWT": "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln",
    }
    for expected, key in keys.items():
        for payload in (f"x{key}", f"{key}9", f"{key}_", f"x{key}_"):
            verdict = SecretsGuardrail().check(f"tok {payload} end", OUT)
            assert verdict.decision == "redact", f"{expected}: {payload!r} not caught"
            assert key not in (verdict.content or ""), f"{expected}: {key!r} survived"


def test_a_long_kebab_identifier_is_not_an_openai_key() -> None:
    """Pins the two-branch OPENAI pattern against its obvious simplification.

    Collapsing both branches into `sk-[A-Za-z0-9_-]{32,}` covers every real key
    shape and looks like a tidy-up, but it also fires on any long kebab-case
    identifier starting `sk-`. This test is the only thing that catches it.
    """
    verdict = SecretsGuardrail().check("sk-this-is-a-very-long-kebab-case-identifier-name", OUT)
    assert verdict.decision == "allow"


def test_allows_clean_text() -> None:
    verdict = SecretsGuardrail().check("nothing to see", OUT)
    assert verdict.decision == "allow"
    assert list(verdict.findings) == []


def test_rejects_an_invalid_on_match() -> None:
    with pytest.raises(ValueError, match="on_match"):
        SecretsGuardrail(on_match="allow")


def test_findings_carry_no_confidence() -> None:
    verdict = SecretsGuardrail().check("AKIAIOSFODNN7EXAMPLE", OUT)
    assert all(f.confidence is None for f in verdict.findings)


# --------------------------------------------------------------------------------
# Beyond the brief: the guards that carry the published numbers, and the
# assertions the brief's fixtures cannot tell apart from a hardcoded literal.
# --------------------------------------------------------------------------------


def test_the_class_declares_its_identity_and_both_directions() -> None:
    assert SecretsGuardrail.name == "secrets"
    assert SecretsGuardrail.kind == "constraint"
    assert SecretsGuardrail.directions == frozenset({"input", "output"})


def test_conforms_to_the_guardrail_protocol() -> None:
    assert isinstance(SecretsGuardrail(), Guardrail)


def test_provenance_reports_the_instance_identity_not_a_module_literal() -> None:
    """Defeat the fixture coincidence rather than walk into it.

    `name` is "secrets" and `version` is "0.1.0", so asserting either literal
    would pass just as well against a `Provenance(kind="constraint",
    detector="secrets", version="0.1.0")` inlined in `check`. Overriding both on
    the instance makes the fixture disagree with every constant the code could
    hardcode, so only reading `self` satisfies it.

    All THREE return paths build a verdict from that provenance: allow, redact
    and deny. Exercise each, because one of them reading a literal while the
    others read `self` is exactly the shape a single-path test would miss.
    """
    modes: tuple[Decision, ...] = ("redact", "deny")
    for on_match in modes:
        guardrail = SecretsGuardrail(on_match=on_match)
        guardrail.name = "secrets-under-test"
        guardrail.version = "9.9.9"

        for text in ("AKIAIOSFODNN7EXAMPLE", "nothing to see"):
            provenance = guardrail.check(text, OUT).provenance
            assert provenance.detector == "secrets-under-test", (on_match, text)
            assert provenance.version == "9.9.9", (on_match, text)
            assert provenance.kind == "constraint", (on_match, text)
            # A constraint has no model and no threshold. A verdict that named
            # one would describe a check that never happened.
            assert provenance.model is None, (on_match, text)
            assert provenance.threshold is None, (on_match, text)


def test_every_return_path_hashes_what_it_saw() -> None:
    """`saw` is what a replay is reconstructed from, on all three paths."""
    for text in ("AKIAIOSFODNN7EXAMPLE", "nothing to see", ""):
        assert SecretsGuardrail().check(text, OUT).saw == saw(text), repr(text)
        assert SecretsGuardrail(on_match="deny").check(text, OUT).saw == saw(text), repr(text)


def test_the_direction_does_not_change_the_verdict() -> None:
    """It declares both directions, and it means it: same bytes, same decision."""
    text = "key AKIAIOSFODNN7EXAMPLE here"
    assert SecretsGuardrail().check(text, IN) == SecretsGuardrail().check(text, OUT)


def test_one_instance_checks_many_inputs_without_carrying_state() -> None:
    """The per-call state really is per-call.

    `_merge` builds its regions from scratch on every call. Held on the instance
    instead, the first call's regions would still be open on the second and every
    later match would merge into a region left over from an earlier input.
    """
    guardrail = SecretsGuardrail()
    text = "tok AKIAIOSFODNN7EXAMPLE and ghp_" + "a" * 36 + " end"

    first = guardrail.check(text, OUT)
    second = guardrail.check(text, OUT)

    assert first == second
    assert [f.type for f in second.findings] == ["AWS_ACCESS_KEY", "GITHUB_TOKEN"]


# --------------------------------------------------------------------------------
# Merging. The brief's implementation rebuilt the output one match at a time and
# leaked; every case below was measured against it.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "leaked", "types", "spans"),
    [
        (
            # AWS key inside a JWT payload. The brief's rebuild emitted
            # "[REDACTED:JWT][REDACTED:AWS_ACCESS_KEY]PAYLOADDATA.c2ln": the
            # cursor jumped BACK to the end of the inner match, so the rest of
            # the payload and the whole signature were re-emitted verbatim.
            "eyJhbGciOiJIUzI1NiJ9.AKIAIOSFODNN7EXAMPLEPAYLOADDATA.c2ln",
            "[REDACTED:AWS_ACCESS_KEY+JWT]",
            "PAYLOADDATA.c2ln",
            ["JWT", "AWS_ACCESS_KEY"],
            [(0, 57), (21, 41)],
        ),
        (
            # AWS key inside a Slack token: 13 of the token's 38 characters
            # survived as "-abcdefghijkl".
            "xoxb-AKIAIOSFODNN7EXAMPLE-abcdefghijkl",
            "[REDACTED:AWS_ACCESS_KEY+SLACK_TOKEN]",
            "-abcdefghijkl",
            ["SLACK_TOKEN", "AWS_ACCESS_KEY"],
            [(0, 38), (5, 25)],
        ),
        (
            # AWS key inside a GitHub token: 16 characters of the token body
            # survived.
            "ghp_AKIAIOSFODNN7EXAMPLE" + "a" * 16,
            "[REDACTED:AWS_ACCESS_KEY+GITHUB_TOKEN]",
            "a" * 16,
            ["GITHUB_TOKEN", "AWS_ACCESS_KEY"],
            [(0, 40), (4, 24)],
        ),
        (
            # The one shape that puts ANTHROPIC_KEY and OPENAI_KEY on one span,
            # and the reason the table's order is not what protects it. Both end
            # together, so nothing leaked even before merging; what changed is
            # that one placeholder now names both instead of claiming two keys.
            "sk-ant-api03-sk-" + "a" * 32,
            "[REDACTED:ANTHROPIC_KEY+OPENAI_KEY]",
            "",
            ["ANTHROPIC_KEY", "OPENAI_KEY"],
            [(0, 48), (13, 48)],
        ),
        (
            # Partial overlap, where neither match contains the other: the Slack
            # body stops at the JWT's first dot and the JWT runs past the Slack
            # match's end.
            "xoxb-eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln",
            "[REDACTED:JWT+SLACK_TOKEN]",
            "",
            ["SLACK_TOKEN", "JWT"],
            [(0, 25), (5, 41)],
        ),
    ],
)
def test_a_contested_span_redacts_the_union_and_names_every_claimant(
    text: str, expected: str, leaked: str, types: list[str], spans: list[tuple[int, int]]
) -> None:
    """No pattern wins a contested span, and none is walked past.

    One placeholder covers the union and names both contributors, sorted, so the
    output is a function of the input alone and not of the pattern table's order.
    The findings keep one entry per match at its OWN span: the output collapses
    to a single placeholder, the audit record must not.
    """
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.content == expected
    assert [f.type for f in verdict.findings] == types
    assert [f.span for f in verdict.findings] == spans
    # Name the leak itself, not just the expected string: whatever the
    # placeholder is spelled like, the fragment that used to survive must not.
    if leaked:
        assert leaked not in (verdict.content or "")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "AKIAIOSFODNN7EXAMPLE-----BEGIN RSA PRIVATE KEY-----",
            "[REDACTED:AWS_ACCESS_KEY][REDACTED:PRIVATE_KEY]",
        ),
        (
            "ghp_" + "a" * 36 + "-----BEGIN RSA PRIVATE KEY-----",
            "[REDACTED:GITHUB_TOKEN][REDACTED:PRIVATE_KEY]",
        ),
    ],
)
def test_two_touching_matches_stay_two_placeholders(text: str, expected: str) -> None:
    """The one input shape that tells a strict overlap test from a non-strict one.

    An AWS key is exactly twenty characters and a PEM header opens with a `-`,
    which is outside both bodies' character classes, so one match ends exactly
    where the next begins. Touching spans do not overlap and must stay two
    regions; a `<=` in `_merge` folds them into one and the output claims a
    single credential where there were two.

    Nothing else in this file distinguishes the two comparisons: every other
    fixture either overlaps properly or is separated by text.
    """
    assert SecretsGuardrail().check(text, OUT).content == expected


def test_two_disjoint_matches_are_redacted_in_text_order() -> None:
    """Merging, from the side where it must NOT happen.

    GITHUB_TOKEN is listed after AWS_ACCESS_KEY in the pattern table and also
    occurs later in the text. Matches left unsorted, or a merge that compared
    only one end of the interval, would fold these two into a single placeholder
    spanning the words between them, and "and" would vanish from the output.
    """
    text = "tok AKIAIOSFODNN7EXAMPLE and ghp_" + "a" * 36 + " end"
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.content == "tok [REDACTED:AWS_ACCESS_KEY] and [REDACTED:GITHUB_TOKEN] end"
    assert [f.type for f in verdict.findings] == ["AWS_ACCESS_KEY", "GITHUB_TOKEN"]


def test_a_later_pattern_matching_earlier_in_the_text_is_kept_in_text_order() -> None:
    """The pattern table's order must not reach the output. JWT is listed last."""
    text = "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln then AKIAIOSFODNN7EXAMPLE"
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.content == "[REDACTED:JWT] then [REDACTED:AWS_ACCESS_KEY]"
    assert [f.type for f in verdict.findings] == ["JWT", "AWS_ACCESS_KEY"]


def test_every_occurrence_is_redacted_not_just_the_first() -> None:
    text = "AKIAIOSFODNN7EXAMPLE and ASIAIOSFODNN7EXAMPLE"
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.content == "[REDACTED:AWS_ACCESS_KEY] and [REDACTED:AWS_ACCESS_KEY]"
    assert len(verdict.findings) == 2


# --------------------------------------------------------------------------------
# The leak invariant, checked per character against an oracle and over generated
# input rather than only over cases I thought of.
# --------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\[REDACTED:[A-Z_+]+\]")


def _matched_offsets(text: str) -> set[int]:
    """Every character offset any pattern matches from ANY start, without the detector.

    Deliberately not built from `verdict.findings`. A finding is the detector's
    own account of what it saw, so an oracle derived from it shrinks by exactly
    the amount the detector dropped, and the dropped-match bug this exists to
    catch would be called clean.

    Deliberately not built with `finditer` either, which is the harder lesson.
    The first version of this oracle used it, the detector used it too, and the
    oracle therefore inherited the detector's blind spot: `finditer` resumes at
    the END of each match, so a decoy of the same shape joined in front of a
    real credential hides it, and an oracle that skips the same start positions
    certifies the leak as clean. A circular oracle one level below the one Task 6
    fixed: not "derived from the findings" but "derived from the same scan
    strategy". This tries EVERY start position with an anchored `match`, which is
    a second implementation of the property rather than the same walk.

    WHAT IT STILL CANNOT SEE. It imports `_PATTERNS` from the module under test,
    so a pattern that matches nothing moves the oracle with it and this check
    reports clean. It is independent of the scan, the merge, the rebuild, the
    cursor, the containment filter and the `on_match` branch, and of nothing
    above that. Pattern changes need their own adversarial cases; the fixture
    tests and the coverage floors below are what catch them.
    """
    offsets: set[int] = set()
    for _, pattern in _PATTERNS:
        for start in range(len(text)):
            match = pattern.match(text, start)
            if match is not None:
                offsets.update(range(*match.span()))
    # PRIVATE_KEY is walked rather than matched, so there is no pattern to
    # re-run here and this part of the oracle IS circular: it calls the walk the
    # detector calls. Said plainly because the alternative would be a second
    # implementation of the walk in the test file, which would drift. The
    # instrument for PEM shapes is the fixture list below, which asserts on the
    # key's own bytes and does not route through this oracle at all.
    for span in _private_key_spans(text):
        offsets.update(range(*span))
    return offsets


def _assert_nothing_matched_survives(text: str) -> None:
    """Strip the placeholders; what is left must be exactly what no pattern matched.

    Checked per character, not per match. Asserting only that a whole credential
    is absent would pass while a fragment survived, and a fragment of a JWT
    payload or a Slack token is still somebody's credential material.

    The equality is two guards in one. Extra text on the left is a leak. Missing
    text on the left is the opposite mistake, eating input that no pattern
    matched, which a redactor that simply blanked everything would commit.
    """
    verdict = SecretsGuardrail().check(text, OUT)
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


# Every token is a real credential shape, plus the joining characters that make
# spans touch, contain and partially overlap. Three deep, so chains of all of
# those occur.
_SWEEP_TOKENS = [
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_" + "a" * 36,
    "sk-ant-api03-" + "a" * 20,
    "sk-proj-" + "a" * 40,
    "sk-" + "a" * 32,
    "xoxb-123456789012-abcdefghijkl",
    "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln",
    "-----BEGIN RSA PRIVATE KEY-----",
    "",
    " ",
    ".",
    "-",
    "_",
    "x",
    "9",
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

    `contained_inner_ends_first` is the leak shape specifically. Plain
    containment where both spans end together loses nothing even under a rebuild
    that walks past the inner match; it is the inner match ending EARLIER that
    sends the cursor backwards and re-emits the container's tail.
    """
    census = {"partial": 0, "contained": 0, "contained_inner_ends_first": 0, "touching": 0}
    for text in _sweep_corpus():
        # Every start position, for the same reason the oracle uses them: what the
        # corpus contains must not depend on how the detector walks it. This line
        # used `finditer` while the docstring above claimed otherwise, which is
        # the same circularity one layer out: the census would have under-counted
        # exactly the shapes a finditer-based scan cannot reach.
        spans = sorted(
            [
                match.span()
                for _, pattern in _PATTERNS
                for start in range(len(text))
                if (match := pattern.match(text, start)) is not None
            ]
            + _private_key_spans(text)
        )
        kinds: set[str] = set()
        for (a_start, a_end), (b_start, b_end) in itertools.combinations(spans, 2):
            if b_start < a_end and a_start < b_end:
                contained = (a_start <= b_start and b_end <= a_end) or (
                    b_start <= a_start and a_end <= b_end
                )
                if contained:
                    kinds.add("contained")
                    if a_end != b_end:
                        kinds.add("contained_inner_ends_first")
                else:
                    kinds.add("partial")
            elif a_end == b_start or b_end == a_start:
                kinds.add("touching")
        for kind in kinds:
            census[kind] += 1
    return census


@pytest.mark.parametrize(
    "text",
    [
        "eyJhbGciOiJIUzI1NiJ9.AKIAIOSFODNN7EXAMPLEPAYLOADDATA.c2ln",
        "xoxb-AKIAIOSFODNN7EXAMPLE-abcdefghijkl",
        "ghp_AKIAIOSFODNN7EXAMPLE" + "a" * 16,
        "sk-ant-api03-sk-" + "a" * 32,
        "xoxb-eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln",
        "AKIAIOSFODNN7EXAMPLE-----BEGIN RSA PRIVATE KEY-----",
        "tok AKIAIOSFODNN7EXAMPLE and ghp_" + "a" * 36 + " end",
        "key AKIAIOSFODNN7EXAMPLE here",
        "nothing to see",
    ],
)
def test_not_one_matched_character_survives_into_the_output(text: str) -> None:
    """The invariant merging exists to hold: ambiguity resolves toward more redaction."""
    _assert_nothing_matched_survives(text)


def test_nothing_matched_survives_across_a_generated_corpus() -> None:
    """Run the invariant over generated input, not only over cases I thought of.

    The leak that made this necessary was a containment shape the brief's own
    fixtures could not produce. Fixtures prove the cases an author imagined;
    this covers the ones they did not. It lives in the suite rather than in a
    scratchpad so the evidence is reproducible and runs in CI.
    """
    checked = redacted = 0
    for text in _sweep_corpus():
        checked += 1
        if _matched_offsets(text):
            redacted += 1
        _assert_nothing_matched_survives(text)

    # Guard the guard, on COVERAGE rather than volume: a ratio of redacting
    # inputs says nothing about which shapes are present. The counts themselves
    # live in exactly one place, the floors below, because this paragraph used to
    # repeat them and was wrong on all four numbers for three rounds running
    # while the correct figures sat four lines underneath it.
    assert checked == len(_SWEEP_TOKENS) ** 3
    shapes = _shape_census()
    # Floors sit just under the measured every-start counts (692 / 835 / 143 /
    # 474), not comfortably under them, so they do two jobs: they catch a token
    # list that thins a shape out, AND they catch the census regressing to a
    # finditer walk, which would report 546 / 713 / 130 / 463. All four separate
    # the two. They are deliberately brittle against detector changes: every
    # PRIVATE_KEY change has moved these counts, and this assertion is what
    # demands they be re-measured rather than assumed to still hold.
    assert shapes["partial"] >= 650, f"corpus lost partial-overlap coverage: {shapes}"
    assert shapes["contained"] >= 800, f"corpus lost containment coverage: {shapes}"
    assert shapes["contained_inner_ends_first"] >= 138, f"corpus lost the leak shape: {shapes}"
    assert shapes["touching"] >= 468, f"corpus lost adjacency coverage: {shapes}"
    assert redacted > checked // 2, f"only {redacted} of {checked} inputs held a credential"


# --------------------------------------------------------------------------------
# Precision. One negative per guard, each measured against the loosening it is
# supposed to defend.
# --------------------------------------------------------------------------------

# Which negative defends what was measured, not assumed: every negative in this
# file was run against twenty-three loosenings of the shipped table. The brief's
# seven hold five real guards between them. The blob, the git SHA and the base64
# string turn away a length-or-entropy rule and the loss of the AWS and GITHUB
# prefix anchors; the UUID turns away a long-token rule; the kebab identifier is
# the strongest, defending OPENAI's two branches, its named prefixes and
# ANTHROPIC's "ant-".
#
# Two of the brief's seven defend nothing that a plausible loosening reaches.
# "sk" and "the sky-blue paint" fire only if the body requirement disappears
# entirely (`sk\S*`), and six other cases catch that too, so neither is holding
# anything on its own. Neither one catches the `sk-\S+` loosening the brief's
# Step 5 attributes to the sky-blue case.
#
# Fifteen of the seventeen cases below catch a loosening that no brief negative
# catches, which is why they are here. The two that do not, "sk-proj-" with a
# 31-character body and "sk-test-" with a 40-character one, are redundant with
# the kebab identifier for every loosening tried; they stay because they name
# the guard they hold, so a regression says which one slipped instead of
# pointing at one shared case. See task-7-report.md for the full matrix.
_NEGATIVES = [
    ("ghp_abc123", "GITHUB_TOKEN requires a 36+ character body"),
    ("gh_" + "a" * 36, "GITHUB_TOKEN requires a type letter after gh"),
    ("ghz_" + "a" * 36, "GITHUB_TOKEN's type letter is one of p, o, u, s, r"),
    ("AKIA" + "B" * 15, "AWS_ACCESS_KEY requires 16 body characters"),
    ("akiaiosfodnn7example", "AWS_ACCESS_KEY's prefix is case sensitive"),
    ("AKIAiosfodnn7example", "AWS_ACCESS_KEY's body is uppercase and digits only"),
    ("sk-ant-short", "ANTHROPIC_KEY requires a 16+ character body"),
    ("sk-" + "a" * 31, "OPENAI_KEY's legacy branch requires 32+ characters"),
    ("sk-proj-" + "a" * 31, "OPENAI_KEY's prefixed branch requires a 32+ character body"),
    ("sk-test-" + "a" * 40, "OPENAI_KEY names its prefixes: proj, svcacct, admin"),
    ("xoxb-short1", "SLACK_TOKEN requires a 10+ character body"),
    ("xoxz-123456789012-abc", "SLACK_TOKEN's type letter is one of b, a, p, r, s"),
    ("-----BEGIN CERTIFICATE-----", "PRIVATE_KEY matches a PRIVATE KEY header, not any block"),
    ("-----BEGIN RSA PUBLIC KEY-----", "PRIVATE_KEY does not match a PUBLIC key header"),
    ("eyJhbGciOiJIUzI1NiJ9", "JWT requires three dot-separated segments"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ", "JWT requires a third segment"),
    ("eyJa.bc.de", "JWT segments are four or more characters each"),
]


@pytest.mark.parametrize(("text", "guard"), _NEGATIVES)
def test_each_prefix_guard_has_a_negative(text: str, guard: str) -> None:
    """One negative per guard, so a regression says which guard slipped."""
    verdict = SecretsGuardrail().check(text, OUT)
    assert verdict.decision == "allow", f"{guard}: {text!r} -> {verdict.findings}"


def test_prose_around_a_prefix_is_not_a_credential() -> None:
    """Precision on ordinary text, which is most of what this reads.

    Stated honestly: the first two hold a guard, the last two are documentation.
    "sk-learn" and the URL both fire under a loose `sk-\\S+`, which is the
    loosening the brief's Step 5 expects a negative to catch and which no case
    in the brief's own list reaches. The BEGIN and eyJ lines catch none of the
    twenty-three loosenings measured; they record what this detector does with
    prose that names a prefix, which is what a precision corpus is made of.
    """
    for text in [
        "the sk-learn docs and the ghp_ prefix",
        "see https://example.com/sk-proj-overview for the plan",
        "BEGIN and END are keywords, not a PRIVATE KEY",
        "eyJ is what a base64 JSON object starts with",
    ]:
        assert SecretsGuardrail().check(text, OUT).decision == "allow", text


# --------------------------------------------------------------------------------
# The deny path, which carries no content and therefore must not carry the
# credential anywhere else either.
# --------------------------------------------------------------------------------

_KEYS = {
    "AWS_ACCESS_KEY": "AKIAIOSFODNN7EXAMPLE",
    "GITHUB_TOKEN": "ghp_" + "a" * 36,
    "ANTHROPIC_KEY": "sk-ant-api03-" + "a" * 20,
    "OPENAI_KEY": "sk-proj-" + "a" * 40,
    "SLACK_TOKEN": "xoxb-123456789012-abcdefghijkl",
    "PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----",
    "JWT": "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln",
}


def test_every_declared_type_is_reachable_on_its_own() -> None:
    """A name in SECRET_TYPES with no pattern behind it is a silent recall hole.

    `SECRET_TYPES` and the pattern table are two separate literals. A type added
    to the first and forgotten in the second matches nothing and quietly lowers
    the recall this project publishes; a pattern whose type is not declared is
    the mirror hole, invisible to any caller reading SECRET_TYPES. Both
    directions of the equality are checked, and then every type is fired.
    """
    # PRIVATE_KEY is walked rather than matched, so it is the one declared type
    # with no entry in the pattern table. Named explicitly rather than excluded
    # by a filter, so adding a second walked type without a test fails here.
    assert {secret_type for secret_type, _ in _PATTERNS} | {"PRIVATE_KEY"} == SECRET_TYPES
    assert set(_KEYS) == SECRET_TYPES

    for secret_type, key in _KEYS.items():
        verdict = SecretsGuardrail().check(f"value {key} end", OUT)
        assert verdict.decision == "redact", secret_type
        assert verdict.content == f"value [REDACTED:{secret_type}] end", secret_type


@pytest.mark.parametrize(("secret_type", "key"), sorted(_KEYS.items()))
def test_a_deny_carries_the_finding_but_never_the_credential(secret_type: str, key: str) -> None:
    """A deny kills the call, so the credential must not ride along on anything.

    `content` is None, and the decision is what keeps the credential out of the
    chain's rewrite: only a redact contributes spans to it. The findings name a
    type and a span and nothing else, so the credential is not in the audit
    record either.
    That is checked against the finding's whole repr rather than its fields, so
    adding a field that carries matched text fails here.
    """
    verdict = SecretsGuardrail(on_match="deny").check(f"tok {key} end", OUT)

    assert verdict.decision == "deny"
    assert verdict.content is None
    assert [f.type for f in verdict.findings] == [secret_type]
    for finding in verdict.findings:
        assert key not in repr(finding)


@pytest.mark.parametrize(("secret_type", "key"), sorted(_KEYS.items()))
def test_deny_and_redact_find_exactly_the_same_things(secret_type: str, key: str) -> None:
    """The deny path must not detect less than the redact path.

    Both build their findings from the same list. If one ever filters or
    short-circuits, the audit records of the two modes stop agreeing and the
    published recall number would depend on which mode was measured.
    """
    text = f"tok {key} end"
    denied = SecretsGuardrail(on_match="deny").check(text, OUT)
    redacted = SecretsGuardrail().check(text, OUT)

    assert list(denied.findings) == list(redacted.findings)
    assert denied.saw == redacted.saw


def test_a_deny_stops_the_chain_from_rewriting_and_a_redact_does_not() -> None:
    """Through the chain, which is where a caller actually meets this.

    A deny contributes nothing to the chain's rewrite, so `ChainResult.content`
    still holds the credential. That is the documented contract: content is the
    audit record and is NOT safe to send on a deny, and the caller's branch on
    `decision` is the guard. Pinned
    here so the failure direction is visible rather than implied, and paired
    with the redact case, where the chain does rewrite and the credential is
    gone from the value a caller would forward.
    """
    text = "key AKIAIOSFODNN7EXAMPLE here"

    denied = GuardrailChain([SecretsGuardrail(on_match="deny")]).run(text, IN)
    assert denied.decision == "deny"
    assert denied.content == text
    assert denied.verdicts[0].content is None

    rewritten = GuardrailChain([SecretsGuardrail()]).run(text, IN)
    assert rewritten.decision == "redact"
    assert rewritten.content == "key [REDACTED:AWS_ACCESS_KEY] here"
    assert "AKIAIOSFODNN7EXAMPLE" not in rewritten.content


def test_rejects_every_on_match_that_is_not_redact_or_deny() -> None:
    """The brief pins "allow". The failure direction is what makes the others matter.

    An unvalidated `on_match` reaches the branch in `check`, which tests for
    "deny" and falls through to redact for anything else, so a typo'd or None
    setting silently becomes the weaker of the two behaviours a caller could
    have asked for. Fail at construction instead.

    The casts are the point: these are values the type system forbids, and the
    check they defeat is a runtime one for callers who are not type-checked.
    """
    for bad in ("allow", "DENY", "Redact", "", "block", None):
        with pytest.raises(ValueError, match="on_match"):
            SecretsGuardrail(on_match=cast(Decision, bad))


# --------------------------------------------------------------------------------
# JWT's bounded segments: the cost they remove and the recall they cost.
# --------------------------------------------------------------------------------


def test_the_jwt_pattern_does_not_degrade_quadratically() -> None:
    """A ReDoS guard on a component that reads attacker-influenced text.

    Every "eyJ" in the input is a candidate start, and unbounded, each one's
    first segment scans the rest of the input for a `.` that is not there. That
    is quadratic: 4.0x per doubling, 870 ms at 64 KB and 14.2 SECONDS at the
    256 KB used here. With the segments bounded the same input is linear: 112 ms
    at 64 KB, 459 ms at 256 KB, 1.92 s at 1 MB, 4.0x per 4x of input.

    The budget is `_REDOS_BUDGET_SECONDS["JWT"]`, the loosest entry in that map,
    because this is the dearest of the probes to run linearly. At 4.0 s it is
    8.6x this probe's idle time and 3.6x under its unbounded cost, so a loaded
    runner stays green and removing the bounds still fails outright: measured at
    14.50 s with the segment bounds taken off. The load measurement that moved
    it off a shared 2.0 s is recorded once at the map. It tightened when the
    header bound went from 1024 to 4096 to cover
    x5c headers: that is a 4x cost on this shape, paid for real tokens the old
    bound missed. The other patterns are linear on this shape too, each for its
    own reason, which the per-pattern probes below cover. An earlier version of
    this docstring explained it with "a run that reaches the end of the input
    MATCHES for them, so the scan is consumed rather than repeated". That model
    is FALSE, the decoy Critical disproved it, and it was deleted from the module
    two rounds ago while surviving here. Matches are computed per start position;
    finding one does not stop the next start being tried.
    """
    payload = "eyJa" * 65536
    assert len(payload) > 256000

    start = time.perf_counter()
    verdict = SecretsGuardrail().check(payload, OUT)
    elapsed = time.perf_counter() - start

    assert verdict.decision == "allow"
    assert elapsed < _REDOS_BUDGET_SECONDS["JWT"], (
        f"JWT pattern took {elapsed:.2f}s on {len(payload)} characters"
    )


def test_an_over_long_jwt_header_is_a_known_miss() -> None:
    """The recall the bounds cost, written down rather than left to be discovered.

    A header of more than 4096 characters after the "eyJ" prefix is not matched
    at all: the match fails at that start and no later start has an "eyJ" to
    begin from, so the whole token comes back allow.

    The bound was 1024 and that was under a whole class of real tokens. Measured
    header sizes, encoded: plain RS256 36 characters, `jwk` with an RSA-2048 key
    543, one x5c certificate 1670, a two-certificate chain 3292. x5c headers are ordinary
    in Open Banking, eIDAS and mTLS-bound JWS, so the old bound missed every one
    of them, which is a recall hole and not a curiosity. The test below fires all
    four shapes so the claim is checked rather than asserted in a comment.
    Task 14's corpus must label a chain longer than two a miss rather than
    counting it as clean text.

    This is NOT the padding bug in a different costume. The token that goes
    unmatched here is one that was already over-long when it was minted, and
    the test below pins the reason: padding cannot force this shape from
    outside.
    """
    body = ".eyJhIjoxfQ.c2ln"
    assert SecretsGuardrail().check("eyJ" + "a" * 4096 + body, OUT).decision == "redact"

    verdict = SecretsGuardrail().check("eyJ" + "a" * 4097 + body, OUT)
    assert verdict.decision == "allow"
    assert verdict.content is None


# Real headers, not byte counts, and the encoded size each one actually produces.
# The previous version passed a byte count into a synthetic x5c for every case,
# so "RS256 plain" was an x5c with a 24-character certificate and encoded to 83
# rather than the 36 a plain header takes, and "two certs" was one 2428-character
# entry encoding to 3288 rather than a real two-entry chain at 3292. Numbers that
# do not come from the thing they describe drift the moment either changes.
_X5C_HEADERS: list[tuple[str, dict[str, object], int]] = [
    ("RS256 plain", {"alg": "RS256", "typ": "JWT"}, 36),
    (
        "jwk RSA-2048",
        {"alg": "RS256", "typ": "JWT", "jwk": {"kty": "RSA", "n": "x" * 342, "e": "AQAB"}},
        543,
    ),
    ("x5c one cert", {"alg": "RS256", "typ": "JWT", "x5c": ["M" * 1214]}, 1670),
    ("x5c two certs", {"alg": "RS256", "typ": "JWT", "x5c": ["M" * 1214, "M" * 1214]}, 3292),
]


@pytest.mark.parametrize(("label", "header", "encoded"), _X5C_HEADERS)
def test_a_real_x5c_header_is_caught(label: str, header: dict[str, object], encoded: int) -> None:
    """The tokens the old 1024 bound missed, built to the sizes that were measured.

    A JWS header carrying a certificate chain is ordinary in Open Banking, eIDAS
    and mTLS-bound issuance. Encoded, one certificate takes the header to 1670
    characters and two to 3292, both over the bound this pattern used to carry,
    so every such token came back allow with the credential whole.
    """
    segment = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode())
    encoded_header = segment.decode().rstrip("=")
    # The size the case claims is the size it produces, asserted rather than
    # quoted, so the prose above cannot drift from the fixture below it.
    assert len(encoded_header) == encoded, label
    token = encoded_header + "." + "e" * 40 + "." + "s" * 43

    verdict = SecretsGuardrail().check(f"tok {token} end", OUT)

    assert verdict.decision == "redact", label
    assert [f.type for f in verdict.findings] == ["JWT"], label
    assert token not in (verdict.content or ""), label


@pytest.mark.parametrize(
    ("label", "prefix"),
    [
        # Decoys that FAIL to match, which is what this test is about.
        ("dotless run one over the bound", "eyJ" + "a" * 4097),
        ("the same, separated", "eyJ" + "a" * 4097 + " "),
        ("dotless run far over the bound", "eyJ" + "a" * 20000),
        ("no candidate start in it at all", "x" * 5000),
        # Decoys that MATCH, and so move the scan. Kept because the property has
        # to hold for both, but note they are covered properly by
        # test_a_succeeding_decoy_does_not_hide_the_credential_behind_it.
        ("a matching token first", "eyJ" + "b" * 1100 + "." + "c" * 10 + "." + "d" * 10 + " "),
        ("a matching dotless-then-dotted run", "eyJ" + "a" * 2000 + ".bbbb.cccc "),
    ],
)
def test_padding_cannot_force_the_bounded_jwt_miss(label: str, prefix: str) -> None:
    """The adversarial dual of the test above, and the reason the bound is safe.

    The bound is a length limit on a segment, which is the shape of guard that
    turned into a redaction bypass in Task 6: there, sixty characters of padding
    put the `@` out of reach of every legal start and a whole address came back
    allow. Naming the property is not enough, so here is the case that would
    violate it.

    It does not, and the mechanism was stated wrongly here before. The original
    explanation was "a failed match at one start does not end the scan", which is
    true and covered only part of the cases: two of the six labels below describe
    decoys that MATCH, and one of them, "eyJ" plus 2000 characters, stopped being
    over-long the moment the bound rose from 1024 to 4096: it matches at
    (0, 2013), and its label was corrected with it.

    The property holds for both kinds and for one reason: the scan tries every
    start. A failing decoy leaves the scan where it was, a matching one moves it,
    and neither can skip the real token's own "eyJ" because that start is visited
    on its own account. Lengthening a real token's header to escape the bound
    means editing inside the header, before the first dot, which changes it and
    destroys the signature.
    """
    real = "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln"
    text = prefix + real
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact", label
    assert verdict.content is not None
    assert real not in verdict.content, label
    # Not just the whole token: no character any pattern matched may survive.
    _assert_nothing_matched_survives(text)


# (label, decoy, real). Every decoy here MATCHES, which is the entire point:
# a FAILING decoy leaves the scan where it was, a SUCCEEDING one moves it past
# the credential behind it.
_SUCCEEDING_DECOYS = [
    ("GITHUB_TOKEN", "ghp_" + "a" * 40, "ghp_9xY2vK4mQ8pL1nR6tW3zA5bC7dE0fG2hJ4kM"),
    ("AWS_ACCESS_KEY", "AKIA" + "B" * 15, "AKIAIOSFODNN7EXAMPLE"),
    ("OPENAI_KEY legacy victim", "sk-" + "a" * 40, "sk-" + "b" * 32),
    ("OPENAI_KEY prefixed victim", "sk-" + "a" * 40, "sk-proj-" + "b" * 40),
    ("JWT", "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln", "eyJhbGciOiJIUzI1NiJ9.eyJwIjoyfQ.c3Vy"),
]


@pytest.mark.parametrize(("label", "decoy", "real"), _SUCCEEDING_DECOYS)
def test_a_succeeding_decoy_does_not_hide_the_credential_behind_it(
    label: str, decoy: str, real: str
) -> None:
    """Regression guard for a Critical: the scan skipped the start it needed.

    `finditer` resumes at the END of each match. A decoy of the same shape joined
    directly in front of a real credential runs its greedy body past the real
    one's prefix, stops on the first character outside the body class, and the
    scan then resumes INSIDE the real credential, so its start is never tried.
    Measured before the fix:

        ghp_<40 decoy>ghp_9xY2v...  ->  [REDACTED:GITHUB_TOKEN]_9xY2v...

    a whole 36-character body standing behind a publicly known 4-character
    prefix. AWS lost 19 of its 20 characters the same way, both OPENAI branches
    their whole bodies, and a JWT its payload and signature.

    This is the case `test_padding_cannot_force_the_bounded_jwt_miss` was aimed
    at and missed: all six of its decoys FAIL to match, so they leave the scan
    where it was. A failing decoy proves nothing about a succeeding one.

    Note the mechanism is per pattern, since each pattern gets its own scan. A
    decoy only hides a credential its own pattern would have matched, which is
    why every pair here is one type against itself.
    """
    text = f"tok {decoy}{real} end"
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact", label
    survived = verdict.content or ""
    assert real not in survived, f"{label}: whole credential survived"
    # Not just the whole string: no eight-character window of it may survive,
    # because a body standing behind a known prefix is still the credential.
    for i in range(len(real) - 8):
        assert real[i : i + 8] not in survived, f"{label}: {real[i : i + 8]!r} survived"
    _assert_nothing_matched_survives(text)


_PEM_BODY = (
    "MIIEowIBAAKCAQEAvB2yPZ8Kq9nR3mL7xT4wF6jH1sD5gA0cV8bN2pQ9eY3uI7oX\n"
    "kJ5rM4tW1zC6vB8nH0dF2gS7aP3qL9xY4eR6uT1iO5wK8mN3bV7cX2zA9fG4hJ6k"
)
_PEM = f"-----BEGIN RSA PRIVATE KEY-----\n{_PEM_BODY}\n-----END RSA PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("terminated block", _PEM),
        ("unterminated block", _PEM.rsplit("\n", 1)[0]),
        ("one line", _PEM.replace("\n", " ")),
        ("crlf line endings", _PEM.replace("\n", "\r\n")),
        (
            "openssh",
            (
                "-----BEGIN OPENSSH PRIVATE KEY-----\n"
                "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAt\n"
                "-----END OPENSSH PRIVATE KEY-----"
            ),
        ),
        ("encrypted", _PEM.replace("RSA PRIVATE", "ENCRYPTED PRIVATE")),
        ("in a sentence", f"here is the key: {_PEM} do not commit it"),
    ],
)
def test_a_private_key_block_loses_its_body_not_just_its_header(label: str, text: str) -> None:
    """Regression guard for the worst leak by volume, and the one that scored a HIT.

    The pattern used to match the header LINE, so a whole PEM block came back
    with its key material standing under a placeholder:

        '[REDACTED:PRIVATE_KEY]\\nMIIEowIBAAKCAQEAvB2yPZ8...\\n-----END RSA...'

    and the finding said PRIVATE_KEY, so Task 14 would have scored the leak a
    recall hit. Every test in this file used the bare header string, which is
    exactly what hid it: a fixture that cannot contain the bug cannot fail on it.

    The body is now walked rather than matched, so a truncated block, a one-line
    PEM, CRLF endings and OPENSSH keys are all the same code path: find the
    header, then classify each following segment.
    """
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact", label
    assert [f.type for f in verdict.findings] == ["PRIVATE_KEY"], label
    survived = verdict.content or ""
    for line in _PEM_BODY.splitlines():
        assert line not in survived, f"{label}: key material survived"
    assert "MIIEowIBAAKCAQEA" not in survived, label
    assert "b3BlbnNzaC1rZXktdjEA" not in survived, label
    _assert_nothing_matched_survives(text)


def test_a_private_key_header_still_detects_without_swallowing_prose() -> None:
    """The fallback, and the precision guard that stops it eating the sentence.

    A header with no key behind it must still detect, because a truncated leak is
    still a leak. It must not consume the words after it either, which is what
    the classifier buys: a sentence is not half base64 and does not end in its
    hash, so it is not key material and the header stands alone.
    """
    bare = SecretsGuardrail().check("-----BEGIN RSA PRIVATE KEY-----", OUT)
    assert bare.decision == "redact"
    assert bare.content == "[REDACTED:PRIVATE_KEY]"

    prose = SecretsGuardrail().check(
        "-----BEGIN RSA PRIVATE KEY----- is the standard header for a PEM file", OUT
    )
    assert prose.content == "[REDACTED:PRIVATE_KEY] is the standard header for a PEM file"

    embedded = SecretsGuardrail().check("value -----BEGIN RSA PRIVATE KEY----- end", OUT)
    assert embedded.content == "value [REDACTED:PRIVATE_KEY] end"


def test_the_private_key_block_does_not_degrade_on_many_start_positions() -> None:
    """ReDoS measured on the shape that matters, not on one long run.

    A single long body says nothing: the cost that bites is many candidate starts
    each doing work. Both shapes here are linear. With the header repeated every
    start fails on the first body character; with header-plus-body repeated each
    start reads its own body and the bodies do not overlap.
    """
    for payload in (
        "-----BEGIN RSA PRIVATE KEY-----" * 8000,
        ("-----BEGIN RSA PRIVATE KEY-----\n" + "A" * 64 + "\n") * 8000,
    ):
        assert len(payload) > 200000
        start = time.perf_counter()
        SecretsGuardrail().check(payload, OUT)
        elapsed = time.perf_counter() - start
        assert elapsed < _REDOS_BUDGET_SECONDS["PRIVATE_KEY"], (
            f"PEM pattern took {elapsed:.2f}s on {len(payload)} characters"
        )


@pytest.mark.parametrize(
    ("label", "token"),
    [
        ("github_pat_ fine-grained PAT", "github_pat_11ABCDEFG0abcdefghijklm_" + "A" * 59),
        ("xapp- Slack app-level token", "xapp-1-A01B2C3D4E5-1234567890123-" + "b" * 64),
    ],
)
def test_a_known_missing_shape_is_recorded_not_silently_absent(label: str, token: str) -> None:
    """Two shapes this detector does NOT catch, pinned so nobody has to discover them.

    `github_pat_` is GitHub's fine-grained personal access token, which is the
    modern default and the one a user is most likely to be issued today.
    `xapp-` is a Slack app-level token. Neither is in the brief's list of types
    and neither is close enough to any pattern here to be caught by accident, so
    both are COMPLETE misses of live credentials.

    Recorded rather than fixed, because the type list is the brief's to change.
    Task 15 must not publish a recall number that leaves them invisible: the JWT
    limit got a written record and these did not, which is the difference between
    a known limitation and a number nobody can explain. If either is ever added
    to SECRET_TYPES, this test fails and says so.
    """
    verdict = SecretsGuardrail().check(f"tok {token} end", OUT)
    assert verdict.decision == "allow", f"{label} is now caught: update SECRET_TYPES and this test"


def test_two_adjacent_keys_report_one_finding_over_the_run() -> None:
    """Pins the containment filter, which nothing else in this file could see.

    Two Anthropic keys joined directly are one match of the pattern, because the
    body class spans the join, and the second key is ALSO a match starting later
    and ending at the same offset. The later one is contained, so it is dropped
    from the audit record and the run is reported once.

    Removing the filter left this suite green until this test existed: it fires
    on 158 of the 5,449 matches the generated corpus produces here, against
    34,816 of 40,756 in `pii`, where every suffix of a local part is its own
    address. Load-bearing there, nearly idle here, and idle is not the same as
    unreachable.

    KNOWN COST, stated rather than discovered later: two adjacent credentials of
    one type are reported as ONE finding spanning both. Coverage is unaffected,
    the whole run is redacted, but a corpus that scores recall by counting
    findings rather than by checking coverage would call this a miss. Task 14
    should score coverage.
    """
    text = "sk-ant-api03-" + "a" * 20 + "sk-ant-api03-" + "b" * 20
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.content == "[REDACTED:ANTHROPIC_KEY]"
    assert [f.type for f in verdict.findings] == ["ANTHROPIC_KEY"]
    assert [f.span for f in verdict.findings] == [(0, len(text))]
    _assert_nothing_matched_survives(text)


# One dense-prefix payload per declared type, keyed so a new type cannot be added
# without one. The prefix is the pattern's OWN, repeated: that is the condition
# that makes every offset a candidate start, and it is the only shape that
# reaches the cost. Prose cannot, `eyJa` cannot (it matches nothing), and `ghp_`
# cannot (its body class excludes `_`, so a match cannot span two prefixes).
_DENSE_PREFIXES: dict[str, tuple[str, ...]] = {
    "AWS_ACCESS_KEY": ("AKIA", "ASIA"),
    "GITHUB_TOKEN": ("ghp_", "ghs_"),
    "ANTHROPIC_KEY": ("sk-ant-",),
    "OPENAI_KEY": ("sk-proj-", "sk-svcacct-", "sk-"),
    "SLACK_TOKEN": ("xoxb-", "xoxp-"),
    "PRIVATE_KEY": (
        "-----BEGIN RSA PRIVATE KEY-----",
        # The bare header alone never reaches the body walk, which needs segments
        # carrying base64 runs. These three do. The middle one is the shape that
        # cost 2.5 SECONDS on 256 KB back when the body was still a regex.
        "-----BEGIN RSA PRIVATE KEY-----\nAAAAAAAAAAAAAAAA",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAvB2yPZ8Kq9nR3mL7xT4wF6jH1sD5gA0cV8bN2pQ9eY3uI7oX\n",
        "-----BEGIN RSA PRIVATE KEY----- MIIEowIBAAKCAQEAvB2yPZ8Kq9nR3mL7 ",
        # The JSON-escaped form, where a line break is a literal backslash-n and
        # a whole key is one physical line. The walk has a dedicated line-break
        # alternative for it and a mutation defending that alternative, and the
        # probe list still had nothing that used it.
        "-----BEGIN RSA PRIVATE KEY-----\\nMIIEowIBAAKCAQEAvB2yPZ8Kq9nR3mL7xT4wF6jH1sD5gA0cV8bN2pQ9eY3uI7oX\\n",
    ),
    "JWT": ("eyJh", "eyJa"),
}


def test_every_declared_type_has_a_dense_prefix_probe() -> None:
    """A type with no ReDoS probe is how the last one got through.

    JWT was the only pattern with a cost test, which is exactly why three others
    shipped quadratic. Keying the probes by type and pinning the key set against
    SECRET_TYPES makes a missing probe a failure rather than an omission.

    PRESENCE IS NOT VALIDITY, which is the second half and was added after the
    first half proved insufficient: a probe of "x" repeated satisfies a key-set
    check while creating no candidate start at all, leaving a quadratic pattern
    green. So each type must also carry at least one probe that is a genuine
    prefix OF a known positive for that type, taken from `_KEYS` rather than
    written out again here, and each probe must actually be dense.

    The budget map is pinned here too, and for the same reason one step over. A
    probe that runs with no budget of its own would have to fall back on some
    default, and every default is either too loose for the cheapest blowup or
    too tight for the dearest linear cost, so an absent entry is a failure
    rather than an omission that quietly inherits the wrong number.
    """
    assert set(_DENSE_PREFIXES) == SECRET_TYPES
    assert set(_REDOS_BUDGET_SECONDS) == SECRET_TYPES

    for secret_type, prefixes in _DENSE_PREFIXES.items():
        known = _KEYS[secret_type]
        assert any(known.startswith(prefix) for prefix in prefixes), (
            f"{secret_type}: no probe is a prefix of the known positive {known[:40]!r}, "
            "so none of them is guaranteed to create a candidate start"
        )
        for prefix in prefixes:
            payload = (prefix * (262144 // len(prefix) + 1))[:262144]
            assert payload.count(prefix[:8]) > 1000, f"{secret_type}: {prefix!r} is not dense"


@pytest.mark.parametrize(
    ("secret_type", "prefix"),
    [(t, p) for t, prefixes in sorted(_DENSE_PREFIXES.items()) for p in prefixes],
)
def test_no_pattern_degrades_on_a_dense_run_of_its_own_prefix(
    secret_type: str, prefix: str
) -> None:
    """Every pattern, on the one shape that can make its scan quadratic.

    Trying every start position is what closed the decoy bypass, and it is also
    what turns an UNBOUNDED greedy body into a quadratic scan: if the body class
    contains the prefix's own characters, every prefix occurrence starts a match
    that runs to the end of the input, and there are O(n) of them. Measured on
    256 KB before the bounds went in:

        xoxb- repeated     8.97 SECONDS   4.0x per doubling
        sk-ant- repeated   6.22 SECONDS   4.0x
        sk-proj- repeated  5.54 SECONDS   4.0x

    The other patterns are linear on their own dense prefix and it is worth
    saying why, because the reason is the rule: AWS is fixed-length, GITHUB's
    body class excludes the `_` its prefix ends with, OPENAI's legacy branch
    excludes the `-`, PRIVATE_KEY's excludes it too, and JWT is bounded. The
    safety property is not "fixed length", it is whether the body class can
    swallow the next prefix.

    My earlier ReDoS evidence used prose, `eyJa` and `ghp_`, none of which can
    create the condition. A probe that cannot produce the cost proves nothing.
    """
    payload = (prefix * (262144 // len(prefix) + 1))[:262144]

    start = time.perf_counter()
    SecretsGuardrail().check(payload, OUT)
    elapsed = time.perf_counter() - start

    assert elapsed < _REDOS_BUDGET_SECONDS[secret_type], (
        f"{secret_type} took {elapsed:.2f}s on {len(payload)} characters"
    )


@pytest.mark.parametrize(
    ("secret_type", "prefix", "longest_real"),
    [
        ("SLACK_TOKEN", "xoxb-", "xoxp-123456789012-123456789012-1234567890123-" + "a" * 32),
        ("ANTHROPIC_KEY", "sk-ant-", "sk-ant-api03-" + "a" * 95),
        ("OPENAI_KEY", "sk-proj-", "sk-svcacct-" + "a" * 160),
    ],
)
def test_each_body_bound_is_pinned_from_both_sides(
    secret_type: str, prefix: str, longest_real: str
) -> None:
    """A bound nobody pins can be moved either way without a test noticing.

    Three edges, because the bound is a cost decision with a recall price and
    both need to be visible:

    1. The longest real token in the family is fully covered. Measured lengths:
       xoxb bot token 56 characters, xoxp user token 77, sk-ant-api03 108,
       sk-proj 164, sk-svcacct 171. Observed length is the weaker number though:
       Slack's changelog tells integrators to plan for tokens up to 255, so the
       honest headroom is 4x against vendor guidance rather than 6x against
       what exists today. Tightening to 256 would land on Slack's ceiling.
    2. A body exactly at the bound is caught whole.
    3. A body one character over the bound is NOT a miss, which is where these
       differ from JWT. JWT requires a `.` after its header, so an over-long
       header fails the whole match; these have no required suffix, so the match
       simply stops at the bound and the remainder stands. What survives is the
       TAIL: the prefix and the first 1024 body characters are redacted, so the
       remainder is not a liftable credential, and no real token in any of these
       families comes within a factor of five of the bound.

    Why 1024 and not JWT's 4096: there the data demanded it, because a real
    two-certificate x5c header is 3292 characters. Here nothing real comes near
    1024, and 4096 costs 3.5x more on the dense-prefix shape, 285 ms against
    80 ms for SLACK at 256 KB, to buy no recall at all.
    """
    covered = SecretsGuardrail().check(f"tok {longest_real} end", OUT)
    assert covered.content == f"tok [REDACTED:{secret_type}] end", secret_type

    at_bound = prefix + "a" * 1024
    caught = SecretsGuardrail().check(f"tok {at_bound} end", OUT)
    assert caught.content == f"tok [REDACTED:{secret_type}] end", secret_type

    over = prefix + "a" * 1030
    truncated = SecretsGuardrail().check(f"tok {over} end", OUT)
    assert [f.type for f in truncated.findings] == [secret_type], secret_type
    # Six characters over the bound, six characters of tail left standing, and
    # the prefix gone with the rest.
    assert truncated.content == f"tok [REDACTED:{secret_type}]aaaaaa end", secret_type
    assert prefix not in (truncated.content or ""), secret_type


_B64_LINE = "MIIEowIBAAKCAQEAvB2yPZ8Kq9nR3mL7xT4wF6jH1sD5gA0cV8bN2pQ9eY3uI7oX"
_ENCRYPTED_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "Proc-Type: 4,ENCRYPTED\n"
    "DEK-Info: AES-128-CBC,9F2A1C7B3E5D8A04\n"
    "\n" + _B64_LINE + "\n" + _B64_LINE + "\n"
    "-----END RSA PRIVATE KEY-----"
)
_JSON_PEM = (
    '{"type":"service_account","private_key":"-----BEGIN PRIVATE KEY-----\\n'
    + _B64_LINE
    + "\\n"
    + _B64_LINE
    + '\\n-----END PRIVATE KEY-----\\n"}'
)
_WRAPPED_16 = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    + "\n".join(_B64_LINE[i : i + 16] for i in range(0, 64, 16))
    + "\n-----END RSA PRIVATE KEY-----"
)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("RFC 1421 encrypted", _ENCRYPTED_PEM),
        ("RFC 1421 encrypted, truncated", _ENCRYPTED_PEM.rsplit("\n", 1)[0]),
        ("GCP service account JSON", _JSON_PEM),
        ("GCP JSON, truncated", _JSON_PEM.split("\\n-----END")[0]),
        ("wrapped at 16 characters", _WRAPPED_16),
        ("wrapped at 16, truncated", _WRAPPED_16.rsplit("\n", 1)[0]),
        (
            "JSON array of quoted lines",
            '{"key":["-----BEGIN PRIVATE KEY-----","'
            + _B64_LINE
            + '","'
            + _B64_LINE
            + '","-----END PRIVATE KEY-----"]}',
        ),
        (
            "YAML block interrupted by a comment",
            "key: |\n  -----BEGIN PRIVATE KEY-----\n  "
            + _B64_LINE
            + "\n  # note - keep this\n  "
            + _B64_LINE
            + "\n  -----END PRIVATE KEY-----\n",
        ),
        (
            "8.5 KB block, longer than any bound a regex version carried",
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + "\n".join([_B64_LINE] * 130)
            + "\n-----END RSA PRIVATE KEY-----",
        ),
    ],
)
def test_a_private_key_survives_none_of_the_field_shapes(label: str, text: str) -> None:
    """The shapes a key actually arrives in, which the fixture shapes are not.

    These shapes were leaked by three successive versions of a regex, and each
    fix moved the failure rather than removing it:

      v1  required 32 unbroken base64 characters right after the header, which
          RFC 1421 encrypted PEMs, GCP JSON and 16-column wraps all lack
      v2  bounded a terminated block at 4096, which missed a key logged line by
          line, because the END sits past the bound and the run branch cannot
          cross a timestamp
      v3  added a line branch that needed an UNBROKEN run of good lines, which
          one Proc-Type line or one interleaved log entry ended

    Each reported a PRIVATE_KEY finding with the key in the output, so all three
    would have scored recall hits. They are all one code path now: the header is
    found by pattern and the body is walked in Python, which needs no bound.
    """
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact", label
    assert [f.type for f in verdict.findings] == ["PRIVATE_KEY"], label
    survived = verdict.content or ""
    for i in range(0, len(_B64_LINE), 16):
        assert _B64_LINE[i : i + 16] not in survived, f"{label}: key material survived"
    _assert_nothing_matched_survives(text)


def _logged(
    lines: int, prefix: str = "2026-08-29T14:02:11Z app[web.1]: ", terminated: bool = True
) -> str:
    """A PEM key as it appears in a log: one line per line, each with a prefix."""
    rows = ["-----BEGIN RSA PRIVATE KEY-----"] + [_B64_LINE] * lines
    if terminated:
        rows.append("-----END RSA PRIVATE KEY-----")
    return "\n".join(prefix + row for row in rows)


def _yaml_with_comments(pairs: int) -> str:
    """A PEM in a YAML block scalar with a comment every second line."""
    body = []
    for i in range(pairs):
        body += ["  " + _B64_LINE, "  " + _B64_LINE, f"  # rotated segment {i}"]
    return (
        "key: |\n  -----BEGIN PRIVATE KEY-----\n"
        + "\n".join(body)
        + "\n  -----END PRIVATE KEY-----\n"
    )


@pytest.mark.parametrize(
    ("label", "text"),
    [
        # The blocker of round three: a body interrupted on every line, and past
        # the 4096-character bound the terminated-block branch carried then.
        # 5027 characters, so its END line sat beyond the bound.
        ("RSA-4096 logged line by line, terminated", _logged(50)),
        ("the same, unterminated", _logged(50, terminated=False)),
        ("logged with a suffix as well", _logged(50, terminated=False) + " | rotated"),
        ("logged at 128 lines", _logged(128, terminated=False)),
        # Bodies interrupted by quotes and hashes, several kilobytes long. Under
        # the regex only the bounded block branch could take these; the walk
        # tolerates the interruptions directly.
        ("YAML with comments, 6.3 KB", _yaml_with_comments(40)),
        (
            "JSON array of 100 quoted lines, 6.8 KB",
            '{"k":["-----BEGIN PRIVATE KEY-----",'
            + ",".join('"' + _B64_LINE + '"' for _ in range(100))
            + ',"-----END PRIVATE KEY-----"]}',
        ),
        # Longer than the 128-line ceiling the regex's line branch carried. The
        # walk has no line ceiling, so this is now an ordinary case.
        (
            "300-line plain block",
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + "\n".join([_B64_LINE] * 300)
            + "\n-----END RSA PRIVATE KEY-----",
        ),
        ("one-line PEM, unterminated", "-----BEGIN RSA PRIVATE KEY----- " + _B64_LINE),
        # Exactly two body segments, which is the walk's minimum for keeping a
        # body that has no END marker. Nothing else here distinguishes two from
        # three.
        ("logged over exactly two lines", _logged(2, terminated=False)),
        # One JSON-escaped segment then truncated: below the two-segment minimum,
        # so it is carried by the single-segment rule, and only if a literal
        # backslash-n is recognised as a line break.
        (
            "JSON escaped, one line, truncated",
            '{"private_key":"-----BEGIN PRIVATE KEY-----\\n' + _B64_LINE,
        ),
    ],
)
def test_an_interrupted_body_is_taken_whatever_its_length(label: str, text: str) -> None:
    """An interrupted body is taken whatever its length.

    This began as a regression guard for a Critical in the regex era: a length
    bound on terminated blocks was a cliff for exactly the bodies that branch
    existed to cover, and an RSA-4096 key logged line by line with a timestamp
    left 50 of 50 base64 lines in the output under a PRIVATE_KEY finding.

    The pin that should have caught it could not: its "over the bound" fixture
    was a PLAIN block, the one shape with no cliff. A fixture has to be over the
    bound AND interrupted.

    None of these have a length bound to be over any more, since the body is
    walked rather than matched. They are kept because they are real shapes and
    because every one of them was, at some point, a leak.
    """
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact", label
    assert [f.type for f in verdict.findings] == ["PRIVATE_KEY"], label
    survived = verdict.content or ""
    for i in range(0, len(_B64_LINE), 16):
        assert _B64_LINE[i : i + 16] not in survived, f"{label}: key material survived"
    _assert_nothing_matched_survives(text)


_LOGBACK = '{{"@timestamp":"2026-08-29T14:02:11.123Z","level":"INFO","message":"{}"}}'


def _wrapped(width: int, lines: int, envelope: str = "{}", terminated: bool = True) -> str:
    """A key wrapped at `width` columns inside `envelope`."""
    body = (_B64_LINE + "abcdefgh") * 3
    rows = ["-----BEGIN RSA PRIVATE KEY-----"] + [body[:width]] * lines
    if terminated:
        rows.append("-----END RSA PRIVATE KEY-----")
    return "\n".join(envelope.format(row) for row in rows)


@pytest.mark.parametrize(
    ("label", "text", "needle"),
    [
        (
            "more than eight consecutive non-key lines",
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + "\n".join(["deploy bot heartbeat, nothing to see"] * 9)
            + "\n"
            + _B64_LINE
            + "\n"
            + _B64_LINE,
            _B64_LINE,
        ),
        (
            "the final short line of a 64-column key in a long JSON envelope",
            "\n".join(
                _LOGBACK.format(row)
                for row in ["-----BEGIN RSA PRIVATE KEY-----"]
                + [_B64_LINE] * 5
                + [((_B64_LINE + "abcdefgh") * 3)[:39]]
            ),
            ((_B64_LINE + "abcdefgh") * 3)[:39],
        ),
        (
            "wrapped at 32 columns in a long JSON envelope",
            _wrapped(32, 40, _LOGBACK),
            ((_B64_LINE + "abcdefgh") * 3)[:32],
        ),
        (
            "wrapped below 16 columns, in any envelope",
            _wrapped(12, 40),
            ((_B64_LINE + "abcdefgh") * 3)[:12],
        ),
    ],
)
def test_a_private_key_shape_this_cannot_take_is_recorded(
    label: str, text: str, needle: str
) -> None:
    """The shapes that still leak key material, written down rather than found later.

    Each returns `decision == "redact"` with a PRIVATE_KEY finding while part of
    the body survives, which is the combination that makes them dangerous:
    **a corpus that scores finding presence scores all of them as hits, so
    Task 14 must score COVERAGE.** That is true of every residual limit this
    detector has, not just these four.

    What each one is, and which rule owns it:

      - more than eight consecutive non-key lines ends a block. That is k, the
        miss tolerance, and it is the only one of the four that is a deliberate
        tolerance rather than a side effect.
      - the FINAL short line of a key survives when the envelope is long. Up to
        39 characters, measured on standard openssl 64-column output inside
        standard logback JSON. Owned by the density fallback, since a 39-column
        line inside a 74-character wrapper is under the threshold.
      - a body wrapped below 40 columns is owned by the density fallback too, and
        below 40 the envelope decides: covered in short envelopes, missed in long
        ones like logback JSON.
      - a body wrapped below 16 columns has no run long enough to be key material
        at all, so it is missed in EVERY envelope, including plain.

    Three shapes left this list when the density threshold moved to 0.45 and the
    single-segment rule started asking the run rule: a key behind a kilobyte of
    prose on one line, key material embedded in a sentence, and a 16-column wrap
    in a both-sides envelope. All three are covered now.

    This test asserts the leak, so it fails if any of these is ever fixed. That
    is deliberate: they come back with a decision and a finding, so nothing else
    in the suite would notice the change.
    """
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact", label
    assert [f.type for f in verdict.findings] == ["PRIVATE_KEY"], label
    assert needle in (verdict.content or ""), (
        f"{label}: this shape is now fully covered, which is good news: "
        "remove it from the known-limit list and tell Task 14"
    )


_DOCS_SENTENCES = [
    (
        "The file must start with -----BEGIN RSA PRIVATE KEY----- and was added in "
        "commit 3f2a1c9d8e7b6a5f4e3d2c1b0a9f8e7d6c5b4a39 by the deploy bot"
    ),
    (
        "Look for -----BEGIN RSA PRIVATE KEY----- in image sha256:"
        "9f2a1c7b3e5d8a04b6c9e2f1a8d7c4b3e6f9a2d5c8b1e4f7a0d3c6b9e2f5a8d1 before shipping"
    ),
    "Rotate anything matching -----BEGIN OPENSSH PRIVATE KEY----- immediately, see RB0042",
    "Grep for -----BEGIN EC PRIVATE KEY----- across the repo\nand check the results manually",
    "Our scanner flags -----BEGIN DSA PRIVATE KEY----- in any file under config/ or secrets/",
    (
        "Rotate the key in vault path secret/data/prod/tls after "
        "-----BEGIN RSA PRIVATE KEY----- appears in any log line"
    ),
    # TWO lines of prose after the mention, each carrying a hash. One such line
    # is turned away by the two-segment rule whatever the thresholds are, so a
    # pair is needed to reach them at all. Hashes are hex, so this pair is held
    # by the hex rule.
    (
        "A key file starts with -----BEGIN RSA PRIVATE KEY-----\n"
        "added in commit 3f2a1c9d8e7b6a5f4e3d2c1b0a9f8e7d6c5b4a39 by the deploy bot\n"
        "reverted in commit 9c8b7a6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a09 the next morning"
    ),
    # The same shape with NON-hex identifiers, which the hex rule cannot help
    # with. These two lines are a quarter base64 and do not end in their token,
    # so it is the density threshold alone that turns them away: at 0.05 the
    # paragraph is swallowed.
    (
        "The release notes mention -----BEGIN RSA PRIVATE KEY----- and\n"
        "the artifact id QmXk9vB2yPZ8Kq9nR and a few other identifiers besides\n"
        "the second artifact QmZt4wF6jH1sD5gA0 and some more prose after it"
    ),
    # The same again at 0.4211 and 0.4237, just under the threshold, with tokens
    # of 24 and 25 characters so the 40-run rule cannot reach them either. This
    # holds the threshold's VALUE down rather than merely its existence: at 0.42
    # both lines are accepted and this paragraph is swallowed.
    #
    # BOTH lines have to sit above the mutated threshold for the leak to appear,
    # because one accepted segment is turned away by the two-segment rule. The
    # second line used to be 0.4138, so a mutation to 0.42 accepted one line and
    # not the other and the suite stayed green while the interval was described
    # as pinned at 0.4211. The pin is only ever as tight as the LOWER of the two.
    (
        "A key file starts with -----BEGIN RSA PRIVATE KEY----- here\n"
        "the build artifact QmXk9vB2yPZ8Kq9nR3mL7xT4 was signed ok\n"
        "the second artifact QmZt4wF6jH1sD5gA0cV8bN2pQ was signed ok"
    ),
]


@pytest.mark.parametrize("sentence", _DOCS_SENTENCES)
def test_prose_that_mentions_a_pem_header_loses_only_the_header(sentence: str) -> None:
    """Over-redaction on a true positive, which is what Task 15 measures precision on.

    The header itself is a true positive and gets redacted; the sentence around it
    must not. Before the run was required to follow a newline, any 16-character
    alphanumeric run within 256 characters of a header pulled the rest of the
    sentence in, and a git SHA or a container digest supplies one:

        "The file must start with -----BEGIN RSA PRIVATE KEY----- and was added
         in commit 3f2a1c9d... by the deploy bot"
      -> "The file must start with [REDACTED:PRIVATE_KEY]"

    Five of eight sentences like this lost 93 to 114 characters. What separates
    them from key material now is that a SHA is hex and a base64 body line is
    not, and that a wrapped PEM line carries an unbroken run of 40 or more.
    """
    verdict = SecretsGuardrail().check(sentence, OUT)

    assert verdict.decision == "redact"
    assert verdict.content is not None
    # Everything except the header survives: strip the placeholder and what is
    # left must be the sentence minus its header.
    survived = _PLACEHOLDER.sub("", verdict.content)
    header = re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", sentence)
    assert header is not None
    assert survived == sentence[: header.start()] + sentence[header.end() :]


_BODY_STYLES = {
    "plain": [_B64_LINE, _B64_LINE],
    "encrypted": [
        "Proc-Type: 4,ENCRYPTED",
        "DEK-Info: AES-128-CBC,9F2A1C7B3E5D8A04",
        "",
        _B64_LINE,
        _B64_LINE,
    ],
    "wrapped at 16": [_B64_LINE[i : i + 16] for i in range(0, 64, 16)],
}
_LINE_BREAKS = {"lf": "\n", "crlf": "\r\n", "json escape": "\\n"}
_PREFIXES = {"none": "", "log": "2026-08-29T14:02:11Z app[web.1]: "}


def _pem(body: str, terminated: bool, prefix: str, newline: str) -> str:
    lines = ["-----BEGIN RSA PRIVATE KEY-----"] + _BODY_STYLES[body]
    if terminated:
        lines.append("-----END RSA PRIVATE KEY-----")
    return newline.join(_PREFIXES[prefix] + line for line in lines)


@pytest.mark.parametrize("body", sorted(_BODY_STYLES))
@pytest.mark.parametrize("terminated", [True, False])
@pytest.mark.parametrize("prefix", sorted(_PREFIXES))
@pytest.mark.parametrize("newline", sorted(_LINE_BREAKS))
def test_no_combination_of_pem_shapes_leaks(
    body: str, terminated: bool, prefix: str, newline: str
) -> None:
    """Cross the axes instead of adding one shape per round.

    Every leak in this file so far was a shape the fixtures had not been taught,
    and twice it was an INTERSECTION of two shapes that each had a fixture: round
    two fixed encrypted PEMs, round three fixed keys logged line by line, and
    encrypted-AND-logged leaked 50 of 50 body lines because nothing crossed them.
    A suite built one shape per round tests the shapes it was taught, not the
    space they span.

    So the axes are crossed here: three body styles (plain, RFC 1421 encrypted,
    wrapped at 16 columns), terminated or not, with and without a log prefix on
    every line, over three line breaks (LF, CRLF, and the literal backslash-n a
    key has inside JSON). Thirty-six combinations, each asserting that not one
    sixteen-character piece of the key survives.

    The assertion is on the key's own bytes and does not route through the
    oracle, which for PRIVATE_KEY calls the same walk the detector calls. That is
    what makes this list the instrument rather than the generated sweep.
    """
    text = _pem(body, terminated, prefix, _LINE_BREAKS[newline])
    verdict = SecretsGuardrail().check(text, OUT)

    label = f"{body}/{'term' if terminated else 'trunc'}/{prefix}/{newline}"
    assert verdict.decision == "redact", label
    assert [f.type for f in verdict.findings] == ["PRIVATE_KEY"], label
    survived = verdict.content or ""
    for i in range(0, len(_B64_LINE), 16):
        assert _B64_LINE[i : i + 16] not in survived, f"{label}: key material survived"


def test_an_end_marker_across_a_window_boundary_is_still_seen() -> None:
    """The classification window bounds work, so markers must be read past it.

    Segments are handed to the walk in pieces of at most 1024 characters, which
    is what keeps input with no newlines linear. A marker lying across one of
    those seams would be invisible to both halves, so marker searches reach a
    little beyond the segment. Here the END marker starts at offset 1000 of a
    single long line and runs past the seam.

    Without the overlap the block does not close where it should, and the words
    after the key are pulled in with it.
    """
    text = (
        "-----BEGIN RSA PRIVATE KEY----- "
        + "A" * 1000
        + "-----END RSA PRIVATE KEY----- trailing words here"
    )
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.content == "[REDACTED:PRIVATE_KEY] trailing words here"


def test_eight_interleaved_lines_do_not_end_a_block() -> None:
    """The lower edge of k, which nothing else pins.

    Nine consecutive non-key lines end a block and that is recorded as a limit.
    Eight must NOT, and the difference is easy to lose by one: the empty segment
    the newline after a header produces used to count as a miss, so a block with
    eight interleaved log lines ended on the eighth and leaked its whole body.
    Empty segments are not evidence of leaving a block; a blank line inside an
    RFC 1421 block is the same shape.
    """
    # Eight noise lines AND a blank line, which is what separates the two
    # exemptions: without the empty-segment one the blank is a ninth miss and the
    # body is lost, and without the header-remainder one the JSON case below is.
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        + "\n".join(["deploy bot heartbeat"] * 8)
        + "\n\n"
        + _B64_LINE
        + "\n"
        + _B64_LINE
    )
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact"
    survived = verdict.content or ""
    for i in range(0, len(_B64_LINE), 16):
        assert _B64_LINE[i : i + 16] not in survived


# The ENVELOPE axis: what a logging stack wraps around each line. Every earlier
# fixture put text on ONE side only, and that is why two rounds missed this:
# text on both sides is what drives the base64 share of a line down far enough
# to matter. Measured shares for a 64-character body line: plain 1.00, syslog
# 0.60, pino 0.47, logback 0.34, GCP 0.3459. Eleven formats, which is the number
# to quote: an earlier summary said thirteen, from a probe script that had two
# more than the fixture.
_ENVELOPES: dict[str, Callable[[str], str]] = {
    "plain": lambda line: line,
    "prefix only": lambda line: f"2026-08-29T14:02:11Z app[web.1]: {line}",
    "suffix only": lambda line: f"{line} <- key material, rotate now",
    "both sides": lambda line: f"[app] {line} (captured)",
    "syslog rfc5424": lambda line: f"<134>1 2026-08-29T14:02:11Z host app 1 - - {line}",
    "pino json": lambda line: (
        '{"level":30,"time":1724932931123,"pid":4242,"hostname":"web-1","msg":"' + line + '"}'
    ),
    "logback json": lambda line: (
        '{"@timestamp":"2026-08-29T14:02:11.123Z","level":"INFO",'
        '"logger_name":"com.acme.KeyDumper","thread_name":"main","message":"' + line + '"}'
    ),
    "gcp logging": lambda line: (
        '{"insertId":"1a2b3c","jsonPayload":{"message":"' + line + '"},'
        '"resource":{"type":"k8s_container"},"timestamp":"2026-08-29T14:02:11Z"}'
    ),
    "logfmt": lambda line: f'ts=2026-08-29T14:02:11Z level=info msg="{line}"',
    "docker": lambda line: f"2026-08-29T14:02:11.123456789Z stdout F {line}",
    "klog": lambda line: f"I0829 14:02:11.123456       1 dumper.go:42] {line}",
}


@pytest.mark.parametrize("envelope", sorted(_ENVELOPES))
@pytest.mark.parametrize("terminated", [True, False])
def test_no_logging_envelope_hides_a_key(envelope: str, terminated: bool) -> None:
    """The axis two rounds failed on, and the one the other cross does not span.

    A key reaches a guardrail through a log pipeline, and the pipeline wraps
    every line. The first version of the walk decided key material by what share
    of the line was base64, which sounds like a property of the key and is
    really a property of the VICTIM'S LOGGING CONFIGURATION: at a threshold of
    0.5, plain text and syslog passed while pino, logback and GCP JSON leaked
    every line of every key, terminated and unterminated alike, each under a
    single PRIVATE_KEY finding.

    That is why the primary rule is now a 40-character unbroken base64 run
    instead. PEM wraps at 64, 70 or 76 columns, so a body line clears 40 in every
    envelope here, because all eleven of them wrap the line rather than rewrite
    it. That is NOT the same as "nothing can shorten a run", which this docstring
    used to claim: PHP's json_encode escapes a forward slash by default and drops
    25.7% of 64-column lines under a 40-run, and base64url output drops 50.9%,
    both measured over 200,000 random 64-column lines. Neither
    loses a whole key today, since a block needs two accepted segments, but the
    over-claim would justify raising the bound toward 64, where it would bite.

    Both-sides envelopes are the discriminating ones. Text on one side alone
    keeps the share above 0.286, which is why prefix-only fixtures never caught
    this.

    These use the 64-column wrap that openssl and ssh-keygen emit. Wrap width is
    a second axis and it decides WHICH rule carries a line: at 40 columns and
    above the run rule does, in every envelope here. Below that the density
    fallback decides and the envelope starts to matter again, which is pinned
    from above by the test below and recorded as a limit for the long envelopes.
    """
    wrap = _ENVELOPES[envelope]
    body_line = _B64_LINE
    rows = ["-----BEGIN RSA PRIVATE KEY-----"] + [body_line] * 40
    if terminated:
        rows.append("-----END RSA PRIVATE KEY-----")
    text = "\n".join(wrap(row) for row in rows)

    verdict = SecretsGuardrail().check(text, OUT)
    label = f"{envelope}/{'term' if terminated else 'unterm'}"

    assert verdict.decision == "redact", label
    assert [f.type for f in verdict.findings] == ["PRIVATE_KEY"], label
    survived = verdict.content or ""
    for i in range(0, len(body_line), 16):
        assert body_line[i : i + 16] not in survived, f"{label}: key material survived"


def test_the_header_line_remainder_does_not_spend_a_tolerated_miss() -> None:
    """The k off-by-one, in the envelope where it actually bites.

    The header's own line remainder is segment zero of the walk. In JSON, or
    behind any wrapper, it is non-empty and not key material, so it charged one
    of the eight tolerated misses and the recorded tolerance of eight was really
    seven for exactly the formats that need it most.

    Eight interleaved non-key lines after a header that carries trailing text
    must still leave the body covered.
    """
    text = "\n".join(
        ['{"msg":"-----BEGIN RSA PRIVATE KEY-----","seq":1}']
        + [f'{{"msg":"deploy bot heartbeat","seq":{i}}}' for i in range(2, 10)]
        + [f'{{"msg":"{_B64_LINE}","seq":{i}}}' for i in range(10, 13)]
    )
    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact"
    survived = verdict.content or ""
    for i in range(0, len(_B64_LINE), 16):
        assert _B64_LINE[i : i + 16] not in survived


def test_a_short_wrapped_line_is_held_by_the_density_fallback() -> None:
    """The one shape where density is the only rule that applies, pinned from above.

    A 32-column body line inside a both-sides envelope has a 32-character run, so
    the 40-character rule does not reach it, and the segment does not end in its
    run, so that rule does not either. Only the density fallback keeps it, at
    0.6531. Raise the threshold past that and this key is lost, which is what
    makes this the case that holds the value down from above; the prose pair in
    the precision tests, at 0.4211, holds it down from below.

    So the separating interval is (0.4211, 0.6531] and the threshold sits at
    0.45, inside it. An earlier version of this docstring said no value could
    separate a 16-column line, at 0.4848, from the prose pair. That was
    arithmetic run backwards: 0.4211 < 0.4848, so every value in (0.4211, 0.4848]
    splits them, and 0.45 does. The claim was wrong, not merely pessimistic, and
    it would have told Task 14 to treat a fixable case as inherent.
    """
    body_line = _B64_LINE[:32]
    rows = ["-----BEGIN RSA PRIVATE KEY-----"] + [body_line] * 20
    text = "\n".join(f"[app] {row} (captured)" for row in rows)

    verdict = SecretsGuardrail().check(text, OUT)

    assert verdict.decision == "redact"
    survived = verdict.content or ""
    for i in range(0, len(body_line), 16):
        assert body_line[i : i + 16] not in survived


# --------------------------------------------------------------------------------
# Task 18: the finding that did NOT reproduce, recorded so nobody chases it twice.
# --------------------------------------------------------------------------------


def _key_body(der_bytes: int) -> str:
    """base64 of high-entropy bytes, at the size a real DER key of that class is.

    Deterministic without being repetitive: a sha256 chain, so the fixture is the
    same on every run and every machine while the body still looks like what
    base64 of a modulus and two primes looks like. That distinction is the whole
    point of this file's newest lesson.
    """
    blob = b""
    seed = b"jamjet-guardrails-task-18"
    while len(blob) < der_bytes:
        seed = hashlib.sha256(seed).digest()
        blob += seed
    return base64.b64encode(blob[:der_bytes]).decode()


def _pem_block(body: str) -> str:
    """A PEM block wrapped at 64 columns, which is what openssl emits."""
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return (
        "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join(lines) + "\n-----END RSA PRIVATE KEY-----"
    )


@pytest.mark.parametrize(
    ("label", "der_bytes"),
    [("RSA-2048", 1190), ("RSA-4096", 2349)],
)
def test_a_realistically_sized_key_leaves_nothing_standing(label: str, der_bytes: int) -> None:
    """A review reported a key over a size bound coming back a complete allow.

    IT DOES NOT REPRODUCE, and this test is the measurement rather than the
    claim. A PKCS#1 RSA-2048 key is about 1190 DER bytes and an RSA-4096 about
    2349, which is 1588 and 3132 base64 characters, and both redact to exactly
    the placeholder with not one character of body surviving. There is no size
    bound on this path at all: the header is found by pattern and the body is
    walked, so length is not a number anything here compares against.

    The report was reproduced from a fixture whose body was one character
    repeated, and the fixture is what carried the result. See the test below for
    what that shape actually exercises, which is not size.
    """
    verdict = SecretsGuardrail().check(_pem_block(_key_body(der_bytes)), OUT)

    assert verdict.decision == "redact", label
    assert [f.type for f in verdict.findings] == ["PRIVATE_KEY"], label
    assert verdict.content == "[REDACTED:PRIVATE_KEY]", label


@pytest.mark.parametrize(
    ("label", "filler", "leaks"),
    [
        ("all-hex body, the reported shape", "A", True),
        ("all-hex body, lowercase", "f", True),
        ("all-hex body, digits", "0", True),
        ("base64 body outside the hex alphabet", "Z", False),
        ("base64 body, a symbol", "+", False),
    ],
)
def test_a_repeated_character_body_is_a_hex_rule_case_not_a_size_case(
    label: str, filler: str, leaks: bool
) -> None:
    """What the phantom actually was, measured across the alphabet rather than at one point.

    A body of repeated `A` DOES leave its body standing, at 1588 characters and
    at 20000 alike, so it is not a size effect. The cause is the hex-only rule:
    `A` is a hex digit, so a line of nothing but `A` is a hex run, and hex runs
    are dropped on purpose because git SHAs and image digests are hex and prose
    is full of them. Every line then classifies as prose, no segment is accepted,
    and the body is not covered.

    The same body built from `Z` or `+`, which are base64 and NOT hex, redacts
    whole at the same length. That is the control that separates the two
    explanations, and it is why the rule is not loosened here: a real 64-column
    key line is all-hex with probability (22/64)**64, and the shape that trips it
    is not a key.

    A FIXTURE CAN FAIL BY NOT RESEMBLING THE THING IT STANDS FOR, and it fails in
    the direction of whoever built it. That is the lesson worth keeping.
    """
    text = _pem_block(filler * 1588)
    verdict = SecretsGuardrail().check(text, OUT)
    survived = verdict.content or ""

    assert verdict.decision == "redact", label
    assert (filler * 64 in survived) is leaks, label
    if not leaks:
        assert survived == "[REDACTED:PRIVATE_KEY]", label
