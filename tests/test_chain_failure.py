"""The chain's failure posture: a guardrail whose check raises becomes a deny.

A detector that dies must never be able to produce an allow, and must never
propagate out of `run` as an exception the caller might catch and shrug off.
"""

from typing import cast

import pytest

from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import Context, Direction, Finding, Kind, Provenance, Verdict

OUT = Context(direction="output", origin="model")


# The doubles annotate all five protocol members at class level: a bare
# `kind = "constraint"` infers `str`, which is not `Kind`.
class Exploder:
    """A constraint whose check is broken."""

    name: str = "exploder"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:
        raise RuntimeError("detector is broken")


class ClassifierExploder:
    """A classifier whose check is broken.

    Its kind, name, version AND exception type all differ from Exploder's, so
    that the tests using the two between them pin every field to the dead
    guardrail's own value. A fixture must differ from every plausible hardcoded
    constant: with both doubles agreeing on a field, code that invents that
    field instead of reading it satisfies both tests and escapes. It raises
    TypeError rather than RuntimeError for exactly that reason -- see
    test_every_exception_type_fails_closed for the same point made directly.
    """

    name: str = "injection"
    version: str = "2.5.0"
    kind: Kind = "classifier"
    directions: frozenset[Direction] = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:
        raise TypeError("weights missing")


class Allower:
    """Always allows: the weakest verdict, for the never-weakens test."""

    name: str = "allower"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "allow",
            None,
            [],
            Provenance(kind="constraint", detector="allower", version="0.1.0"),
            saw(content),
        )


class Rewriter:
    """Redacts "one", so a run holding it has a rewrite in it to reason about."""

    name: str = "rewriter"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "redact",
            content.replace("one", "two"),
            [Finding(type="TOK", span=(0, 3))],
            Provenance(kind="constraint", detector="rewriter", version="0.1.0"),
            saw(content),
        )


def test_a_raising_guardrail_denies_rather_than_propagating() -> None:
    result = GuardrailChain([Exploder()]).run("hello", OUT)
    assert result.decision == "deny"


def test_the_synthesised_verdict_records_the_error_and_no_findings() -> None:
    """CHANGED: it asserted the detector's own message was COPIED into the verdict.

    `error` was `f"{type(exc).__name__}: {exc}"`, so whatever the detector chose
    to say landed in the audit record. The shape a detector naturally writes is
    the one that leaks. `raise ValueError(f"could not parse: {content!r}")` is
    that shape, and `test_the_error_does_not_quote_the_content` below runs it. The
    message is now dropped whole and the type kept, so this asserts the type is
    there and the detector's own words are not.
    """
    result = GuardrailChain([Exploder()]).run("hello", OUT)
    verdict = result.verdicts[0]
    assert verdict.decision == "deny"
    assert "RuntimeError" in (verdict.error or "")
    assert "detector is broken" not in (verdict.error or "")
    assert list(verdict.findings) == []
    assert verdict.saw == saw("hello")


def test_the_error_names_the_exception_type() -> None:
    """The message alone does not say what broke; RuntimeError vs TypeError does."""
    result = GuardrailChain([Exploder()]).run("hello", OUT)
    assert "RuntimeError" in (result.verdicts[0].error or "")


class DetectorSpecificError(Exception):
    """A detector's own exception class. Nothing in this library has heard of it."""


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("broken"),
        TypeError("bad signature"),
        KeyError("model"),
        AttributeError("no such attribute"),
        DetectorSpecificError("custom"),
    ],
    ids=["RuntimeError", "TypeError", "KeyError", "AttributeError", "custom"],
)
def test_every_exception_type_fails_closed(exc: Exception) -> None:
    """Pin the LOWER bound on the catch: it must be broad, not just non-empty.

    test_keyboard_interrupt_is_not_swallowed pins the upper bound. Nothing else
    pins this one, and every other double in this file raises RuntimeError, so
    narrowing `except Exception` to `except RuntimeError` -- an easy reading of
    "fail closed on any detector bug" as "on the bugs we have seen" -- would
    otherwise ship silently, and a detector raising anything else would
    propagate straight out of run(). The last case matters most: a real detector
    raises its own exception class, which no narrowed tuple can enumerate.
    """

    class Raiser:
        name: str = "raiser"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            raise exc

    result = GuardrailChain([Raiser()]).run("hello", OUT)
    assert result.decision == "deny"
    assert type(exc).__name__ in (result.verdicts[0].error or "")


def test_the_synthesised_verdict_carries_no_content() -> None:
    """Nothing inspected this string, so the verdict must not appear to vouch for it.

    The chain ignores content on a non-redact, so handing `current` back here
    would not change the run's outcome. It would change the audit record: a
    reader of the verdict alone would see content sitting on a deny that no
    guardrail ever approved.
    """
    result = GuardrailChain([Exploder()]).run("hello", OUT)
    assert result.verdicts[0].content is None


def test_the_synthesised_verdict_keeps_the_guardrails_own_provenance() -> None:
    """A classifier that dies must not be recorded as a constraint.

    The error assertion is here rather than beside the other error tests because
    this is the double that raises something other than RuntimeError: code that
    hardcoded the exception name would still satisfy every RuntimeError fixture
    in this file.
    """
    result = GuardrailChain([ClassifierExploder()]).run("hello", OUT)
    verdict = result.verdicts[0]
    assert verdict.provenance.kind == "classifier"
    assert verdict.provenance.detector == "injection"
    assert verdict.provenance.version == "2.5.0"
    assert "TypeError" in (verdict.error or "")


def test_a_dying_constraint_is_not_recorded_as_a_classifier() -> None:
    """The other half of the truth table for the test above.

    On its own, the classifier test is satisfied by stamping a hardcoded
    kind="classifier". This one is satisfied by a hardcoded "constraint". Only
    reading `guardrail.kind` satisfies both, and the same holds for detector and
    version because the two doubles disagree on every field.
    """
    result = GuardrailChain([Exploder()]).run("hello", OUT)
    prov = result.verdicts[0].provenance
    assert prov.kind == "constraint"
    assert prov.detector == "exploder"
    assert prov.version == "0.1.0"


def test_the_synthesised_verdict_hashes_what_the_dead_guardrail_was_handed() -> None:
    """CHANGED: it asserted `saw("two")`, the previous guardrail's OUTPUT.

    That was right while the chain threaded each rewrite into the next
    guardrail, and threading is what let a redaction split a credential and
    leave its tail standing. Every guardrail now inspects the content the chain
    was given, so what the dead guardrail was handed IS the input, and the
    exploder's `saw` is the input's digest.

    A redact still runs in front of it, and the assertion is now that the hash
    did NOT move: with sequential rewriting restored this reads `saw("two")` and
    fails, so the test still separates the two designs rather than passing under
    either.
    """
    result = GuardrailChain([Rewriter(), ClassifierExploder()]).run("one", OUT)
    assert result.verdicts[1].saw == saw("one")
    assert result.verdicts[1].saw != saw("two")
    assert result.decision == "deny"
    assert result.content == "[REDACTED:TOK]"


def test_a_later_guardrail_cannot_undo_an_earlier_error_deny() -> None:
    result = GuardrailChain([Exploder(), Allower()]).run("hello", OUT)
    assert result.decision == "deny"
    assert len(result.verdicts) == 2  # the allower still ran and is on the record


def test_content_is_unchanged_by_a_failed_guardrail() -> None:
    result = GuardrailChain([Exploder()]).run("hello", OUT)
    assert result.content == "hello"


def test_keyboard_interrupt_is_not_swallowed() -> None:
    """Fail-closed covers detector bugs, not the operator pressing ctrl-c."""

    class Interrupter:
        name: str = "int"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        GuardrailChain([Interrupter()]).run("hello", OUT)


def test_a_guardrail_lying_about_its_kind_raises_rather_than_being_recorded_wrongly() -> None:
    """Pin the one exception `run` still lets out, and why it is the right one.

    Synthesising provenance means trusting `guardrail.kind`. A guardrail that
    declares a kind the library does not know cannot be given one: any fallback
    would write a kind into the audit record that the detector never claimed.
    Verdict rejects it, that ValueError propagates, and nothing is allowed --
    loud and closed rather than quiet and wrong.
    """

    class LyingExploder:
        name: str = "liar"
        version: str = "0.1.0"
        kind: Kind = cast(Kind, "bogus")
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            raise RuntimeError("detector is broken")

    with pytest.raises(ValueError, match="unknown provenance kind"):
        GuardrailChain([LyingExploder()]).run("hello", OUT)


# ==========================================================================
# The error string is the only detector-derived string in a Verdict.
# ==========================================================================

SECRETS = "AKIAIOSFODNN7EXAMPLE and sk-abcdefghijklmnopqrstuvwxyz012345"


class Quoter:
    """A detector that names the content it choked on, as a detector naturally would.

    This is not a hostile double. Quoting the offending value is the standard
    way to write a parse error, which is exactly why the chain must not copy
    the message: `secrets.check` guarantees "the credential is not in the audit
    record either" and `corpus._reject` guarantees a rejected value is never
    echoed, and `error` was the one path that broke both.
    """

    name: str = "quoter"
    version: str = "0.1.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:
        raise ValueError(f"could not parse: {content!r}")


def test_the_error_does_not_quote_the_content() -> None:
    result = GuardrailChain([Quoter()]).run(SECRETS, OUT)
    error = result.verdicts[0].error or ""
    assert "AKIAIOSFODNN7EXAMPLE" not in error
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in error
    assert "could not parse" not in error
    assert "ValueError" in error


def test_the_error_is_bounded_however_long_the_detectors_message_is() -> None:
    """Length, not just content. `error` is persisted by whatever logs a ChainResult.

    The type name is bounded too, because a class name is caller-supplied as
    well and a detector can define one of any length.
    """

    class Verbose:
        name: str = "verbose"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            raise RuntimeError("x" * 100_000)

    long_name = type("E" * 10_000, (Exception,), {})

    class LongNamed:
        name: str = "long-named"
        version: str = "0.1.0"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"output"})

        def check(self, content: str, context: Context) -> Verdict:
            raise long_name("boom")

    for guardrail in (Verbose(), LongNamed()):
        error = GuardrailChain([guardrail]).run("hello", OUT).verdicts[0].error or ""
        assert len(error) < 400, len(error)
