import dataclasses
from collections.abc import Callable
from typing import cast, get_args

import pytest

from jamjet_guardrails.types import (
    ChainResult,
    Context,
    Decision,
    Direction,
    Finding,
    Kind,
    Origin,
    Provenance,
    Verdict,
    combine,
)


def _prov(kind: Kind = "constraint") -> Provenance:
    return Provenance(kind=kind, detector="pii", version="0.1.0")


def test_combine_is_restrictive() -> None:
    assert combine("allow", "allow") == "allow"
    assert combine("allow", "redact") == "redact"
    assert combine("redact", "allow") == "redact"
    assert combine("redact", "deny") == "deny"
    assert combine("deny", "allow") == "deny"
    assert combine("allow", "deny") == "deny"


def test_combine_never_weakens_for_any_pair() -> None:
    order: list[Decision] = ["allow", "redact", "deny"]
    for a in order:
        for b in order:
            assert order.index(combine(a, b)) >= max(order.index(a), order.index(b))


def test_constraint_finding_must_not_carry_confidence() -> None:
    with pytest.raises(ValueError, match="constraint"):
        Verdict(
            decision="redact",
            content="x",
            findings=[Finding(type="EMAIL", span=(0, 1), confidence=0.9)],
            provenance=_prov("constraint"),
            saw="a" * 64,
        )


def test_classifier_finding_must_carry_confidence() -> None:
    with pytest.raises(ValueError, match="classifier"):
        Verdict(
            decision="deny",
            content=None,
            findings=[Finding(type="INJECTION", span=None, confidence=None)],
            provenance=_prov("classifier"),
            saw="a" * 64,
        )


def test_redact_verdict_must_supply_content() -> None:
    with pytest.raises(ValueError, match="redact"):
        Verdict("redact", None, [], _prov(), "a" * 64)


def test_saw_must_be_a_sha256_hex_digest() -> None:
    with pytest.raises(ValueError, match="saw"):
        Verdict("allow", None, [], _prov(), "not-a-hash")
    with pytest.raises(ValueError, match="saw"):
        Verdict("allow", None, [], _prov(), "A" * 64)  # uppercase is not our format


def test_findings_passed_as_an_iterator_are_not_silently_dropped() -> None:
    """``__post_init__`` read `findings` twice before storing it: once for the
    error-verdict check and once for the confidence loop. Validating before
    freezing meant an iterator was exhausted by the time `tuple()` ran, so a
    verdict stored NO findings while passing every check above it.

    A redact verdict with an empty findings tuple is an audit record with the
    decision and none of the evidence, which is the one thing this type exists
    to carry. ``corpus.Case`` already copies before it validates.
    """
    findings = [Finding(type="EMAIL", span=(0, 1)), Finding(type="PHONE", span=(2, 3))]
    # Through a Callable alias, not a type: ignore. The annotation says
    # Sequence and mypy is right to refuse a generator; the point of the test is
    # what happens at RUNTIME in a caller mypy never sees.
    construct = cast(Callable[..., Verdict], Verdict)
    verdict = construct(
        decision="redact",
        content="x",
        findings=(finding for finding in findings),
        provenance=_prov("constraint"),
        saw="a" * 64,
    )
    assert len(verdict.findings) == 2
    assert [f.type for f in verdict.findings] == ["EMAIL", "PHONE"]


def test_verdicts_are_frozen() -> None:
    v = Verdict("allow", None, [], _prov(), "a" * 64)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.decision = "deny"  # type: ignore[misc]


def test_error_verdict_must_deny() -> None:
    with pytest.raises(ValueError, match="must deny"):
        Verdict("allow", None, [], _prov(), "a" * 64, error="boom")


def test_error_verdict_denies_without_findings() -> None:
    v = Verdict("deny", None, [], _prov("classifier"), "a" * 64, error="boom")
    assert v.error == "boom"
    assert list(v.findings) == []


def test_context_and_chain_result_construct() -> None:
    assert Context(direction="output", origin="model").direction == "output"
    assert list(ChainResult(decision="allow", content="hi", verdicts=[]).verdicts) == []


def test_findings_cannot_be_mutated_through_the_caller_s_list() -> None:
    """An invariant enforced only at construction is a suggestion, not an invariant.

    A caller who keeps the list it passed could otherwise append a confidence-bearing
    finding to a CONSTRAINT verdict after __post_init__ has run, walking the whole
    constraint-versus-classifier guard around itself.
    """
    findings = [Finding(type="EMAIL", span=(0, 1))]
    v = Verdict("redact", "x", findings, _prov("constraint"), "a" * 64)

    findings.append(Finding(type="SMUGGLED", confidence=0.9))

    assert len(v.findings) == 1
    assert [f.type for f in v.findings] == ["EMAIL"]
    assert isinstance(v.findings, tuple)


def test_chain_result_verdicts_cannot_be_mutated_through_the_caller_s_list() -> None:
    """The audit record of a run must not change after the run reported it."""
    v = Verdict("allow", None, [], _prov(), "a" * 64)
    verdicts = [v]
    result = ChainResult(decision="allow", content="hi", verdicts=verdicts)

    verdicts.append(v)

    assert len(result.verdicts) == 1
    assert isinstance(result.verdicts, tuple)


def test_saw_must_be_exactly_sixty_four_characters() -> None:
    """Isolate the length rule.

    The other saw test's cases are rejected for other reasons too ("not-a-hash"
    contains a hyphen, "A"*64 is uppercase), so widening the {64} quantifier leaves
    both of them passing. Length is what distinguishes a sha256 from arbitrary
    lowercase hex, and these are the only assertions that pin it.
    """
    for bad in ("a" * 63, "a" * 65, ""):
        with pytest.raises(ValueError, match="saw"):
            Verdict("allow", None, [], _prov(), bad)


def test_unknown_provenance_kind_is_rejected() -> None:
    """A kind we have no confidence rule for must fail closed, not silently skip both.

    `Kind` is designed to grow. When a third kind is added to the Literal, this test
    fails until someone writes that kind's confidence rule -- which is the point of it.

    The casts are the point too: these values are what the type system forbids, and
    the runtime guard exists for the callers static typing never reaches -- config,
    JSON, a plugin built against an older Kind. Casting states that deliberately;
    leaving `_prov` untyped would state it about every call site by accident.
    """
    for bogus in ("heuristic", "Constraint", ""):
        with pytest.raises(ValueError, match="unknown provenance kind"):
            Verdict(
                decision="allow",
                content=None,
                findings=[Finding(type="X")],
                provenance=_prov(cast(Kind, bogus)),
                saw="a" * 64,
            )


def test_unknown_provenance_kind_is_rejected_even_with_no_findings() -> None:
    """The kind check must not hide inside the per-finding loop.

    A bogus kind with zero findings would otherwise never be looked at.
    """
    with pytest.raises(ValueError, match="unknown provenance kind"):
        Verdict("allow", None, [], _prov(cast(Kind, "heuristic")), "a" * 64)


def test_classifier_verdict_with_a_confident_finding_constructs() -> None:
    """Guard the false-reject direction, which Phase 1 exercises nowhere else.

    Every other classifier verdict in this suite either raises or has no findings, so
    a classifier branch that rejected everything would leave the suite green.
    """
    v = Verdict(
        decision="deny",
        content=None,
        findings=[Finding(type="INJECTION", confidence=0.9)],
        provenance=_prov("classifier"),
        saw="a" * 64,
    )
    assert [f.type for f in v.findings] == ["INJECTION"]
    assert v.findings[0].confidence == 0.9


def test_error_verdict_must_not_carry_findings() -> None:
    """The docstring promises this; make it a guarantee rather than a convention.

    Task 5 synthesises these verdicts and will act on the promise.
    """
    with pytest.raises(ValueError, match="must not carry findings"):
        Verdict("deny", None, [Finding(type="EMAIL")], _prov(), "a" * 64, error="boom")


# ==========================================================================
# The domains a caller supplies, refused where nothing else refuses them.
# ==========================================================================
#
# The casts throughout are deliberate, for the reason
# test_unknown_provenance_kind_is_rejected gives: these are the values static
# typing forbids, and the runtime guard exists for the callers it never reaches.
# mypy catches `Context(direction="Output", ...)` under --strict and does NOT
# catch `Context(direction=json.loads(blob)["direction"], ...)`, which is where
# a direction actually comes from.


@pytest.mark.parametrize("bad", ["out", "in", "Output", "OUTPUT", "", "outputs", "both"])
def test_an_unknown_direction_is_refused_rather_than_silencing_the_chain(bad: str) -> None:
    """A direction outside the domain used to make the whole chain a no-op.

    `chain.run` runs a guardrail only when the context's direction is in that
    guardrail's `directions`, so an unrecognised value matched nothing: allow,
    zero verdicts, content untouched, over content full of credentials. It is
    the mirror of the case `detectors.build` already refuses, where a guardrail
    declares no direction it can run in.
    """
    with pytest.raises(ValueError, match="unknown direction"):
        Context(direction=cast(Direction, bad), origin="model")


@pytest.mark.parametrize("bad", ["retreived", "User", "", "assistant", "system"])
def test_an_unknown_origin_is_refused(bad: str) -> None:
    """Nothing branches on `origin` today, which is why a typo would never surface.

    It travels into the audit record, so a caller filtering their own log for
    origin="retrieved" would silently never see the rows written as "retreived".
    """
    with pytest.raises(ValueError, match="unknown origin"):
        Context(direction="output", origin=cast(Origin, bad))


def test_every_direction_and_origin_in_the_domain_still_constructs() -> None:
    """The false-reject control. A guard that refused everything would also pass above."""
    for direction in get_args(Direction):
        for origin in get_args(Origin):
            assert Context(direction=direction, origin=origin).direction == direction


@pytest.mark.parametrize("bad", ["allowed", "Allow", "", "block", "deny "])
def test_an_unknown_decision_is_refused(bad: str) -> None:
    """The kind check's own argument, applied to the field the kind check sits beside.

    Every other rule in `Verdict.__post_init__` is a positive match on a known
    decision, so an unrecognised one walked past all of them and constructed
    cleanly. A caller branching `if deny ... elif redact` then reads it as an
    allow, and inside a chain it reached `combine` and raised a bare
    `ValueError: tuple.index(x): x not in tuple` naming neither the detector
    nor the value.
    """
    with pytest.raises(ValueError, match="unknown decision"):
        Verdict(cast(Decision, bad), None, [], _prov(), "a" * 64)


def test_every_decision_in_the_domain_still_constructs() -> None:
    """The false-reject control for the check above."""
    for decision in get_args(Decision):
        content = "x" if decision == "redact" else None
        assert Verdict(decision, content, [], _prov(), "a" * 64).decision == decision
