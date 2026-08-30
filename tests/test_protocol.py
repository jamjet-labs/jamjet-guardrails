import hashlib

import pytest

from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import Context, Provenance, Verdict


def test_saw_is_sha256_of_utf8() -> None:
    assert saw("hello") == hashlib.sha256(b"hello").hexdigest()
    assert len(saw("")) == 64
    # An ASCII fixture cannot tell utf-8 apart from ascii, latin-1 or cp1252, and the
    # cafe pair below only asserts a difference, which U+00E9 preserves under all of
    # them. U+65E5 is outside latin-1 entirely, so these bytes pin utf-8 specifically.
    # Task 16 fixes "SHA-256 over the exact inspected string, encoded UTF-8" for a
    # second implementation: a TypeScript twin and a latin-1 Python would disagree on
    # every non-ASCII input with both suites green. Worse than a wrong digest, a
    # latin-1 encode RAISES, taking the chain down on the first CJK, Cyrillic or emoji.
    assert saw("日") == hashlib.sha256(b"\xe6\x97\xa5").hexdigest()


def test_saw_distinguishes_similar_looking_strings() -> None:
    """If these collide, replay is a lie."""
    assert saw("cafe") != saw("café")


def test_saw_hashes_the_exact_bytes_not_a_normalised_form() -> None:
    """The three cases above survive case-folding and stripping; these do not.

    "hello", "" and the cafe pair are all unchanged by .lower() or .strip(), so a
    saw() that normalised before hashing would leave them green while quietly
    breaking replay: two different pieces of content would claim the same hash.
    """
    assert saw("Secret") == hashlib.sha256(b"Secret").hexdigest()
    assert saw("Secret") != saw("secret")
    assert saw(" x ") != saw("x")


class _Stub:
    name = "stub"
    version = "0.1.0"
    kind = "constraint"
    directions = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "allow",
            None,
            [],
            Provenance(kind="constraint", detector="stub", version="0.1.0"),
            saw(content),
        )


def test_structural_conformance_is_runtime_checkable() -> None:
    assert isinstance(_Stub(), Guardrail)


def test_an_object_without_check_does_not_conform() -> None:
    class NoCheck:
        name = "x"
        version = "0.1.0"
        kind = "constraint"
        directions = frozenset({"output"})

    assert not isinstance(NoCheck(), Guardrail)


def test_a_conforming_guardrail_returns_a_verdict_carrying_the_hash_it_saw() -> None:
    """Exercise the stub rather than only its shape.

    This is the cross-module contract: whatever saw() returns must satisfy the
    sha256 pattern Verdict enforces, or every real guardrail raises on construction.
    """
    verdict = _Stub().check("hello", Context(direction="output", origin="model"))

    assert verdict.saw == saw("hello")
    assert verdict.decision == "allow"


# A runtime_checkable protocol checks only that the members are PRESENT: never their
# types, never their signatures. An object whose `check` is the integer 3 passes
# isinstance. So the cases below remove a member outright, which is the one thing
# isinstance can actually see. Static typing is what catches a wrong signature.
_CONFORMING_MEMBERS = {
    "name": "stub",
    "version": "0.1.0",
    "kind": "constraint",
    "directions": frozenset({"output"}),
    "check": lambda self, content, context: None,
}


@pytest.mark.parametrize("missing", sorted(_CONFORMING_MEMBERS))
def test_every_declared_member_is_required(missing: str) -> None:
    """Pin the whole member list, not just `check`.

    A Guardrail protocol that declared `check` alone would still pass the test
    above. `kind` in particular is load-bearing: the chain stamps provenance onto
    the deny verdict it synthesises when a guardrail raises, and it must not have
    to guess whether the dead detector was a constraint or a classifier.
    """
    attrs = {name: value for name, value in _CONFORMING_MEMBERS.items() if name != missing}

    assert not isinstance(type("Partial", (), attrs)(), Guardrail)


def test_the_full_member_set_conforms() -> None:
    """Keep the parametrisation above honest.

    If _CONFORMING_MEMBERS were missing something the protocol requires, every
    case above would pass for the wrong reason and pin nothing.
    """
    assert isinstance(type("Whole", (), dict(_CONFORMING_MEMBERS))(), Guardrail)
