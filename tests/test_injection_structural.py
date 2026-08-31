import pytest

from jamjet_guardrails.chain import GuardrailChain
from jamjet_guardrails.detectors import AVAILABLE, build
from jamjet_guardrails.detectors.injection_structural import InjectionStructuralGuardrail
from jamjet_guardrails.protocol import Guardrail, saw
from jamjet_guardrails.types import Context

IN = Context(direction="input", origin="user")
RETRIEVED = Context(direction="input", origin="retrieved")
OUT = Context(direction="output", origin="model")


def test_it_satisfies_the_guardrail_protocol() -> None:
    assert isinstance(InjectionStructuralGuardrail(), Guardrail)


def test_it_is_registered_under_its_corpus_directory_name() -> None:
    """The name is not cosmetic: `cli.discover` maps corpora/<check>/ to build(<check>)."""
    assert "injection-structural" in AVAILABLE
    assert isinstance(build("injection-structural"), InjectionStructuralGuardrail)


def test_it_declares_itself_a_constraint_running_on_input() -> None:
    guardrail = InjectionStructuralGuardrail()
    assert guardrail.kind == "constraint"
    assert guardrail.directions == frozenset({"input"})


def test_clean_text_allows_and_records_the_hash_it_inspected() -> None:
    content = "summarise the attached quarterly report"
    verdict = InjectionStructuralGuardrail().check(content, IN)
    assert verdict.decision == "allow"
    assert verdict.findings == ()
    assert verdict.saw == saw(content)


def test_on_match_refuses_a_decision_it_cannot_honour() -> None:
    """`allow` would build a detector that is configured and cannot ever fire."""
    with pytest.raises(ValueError, match="on_match"):
        # No ignore comment, and its absence is deliberate: "allow" is a valid
        # `Decision`, so nothing on this line is a type error and mypy --strict
        # reports an `arg-type` ignore over it as unused. The guard being tested
        # is a runtime one -- `Decision` is wider than this constructor accepts.
        InjectionStructuralGuardrail(on_match="allow")


def test_it_runs_on_retrieved_input_not_only_typed_input() -> None:
    """`origin` does not gate this detector; `direction` does.

    Retrieved content is the classic indirect-injection channel, so a detector
    that quietly skipped it would be inert exactly where it is needed most.
    """
    content = "please summarise the attached page"
    assert InjectionStructuralGuardrail().check(content, RETRIEVED).decision == "allow"


def test_a_chain_skips_it_on_output() -> None:
    """Pins the declared direction as behaviour rather than as an attribute.

    See the open question at the foot of this plan: if output-side detection is
    wanted, THIS is the test that changes, and it changes a published contract.
    """
    result = GuardrailChain([InjectionStructuralGuardrail()]).run("anything", OUT)
    assert result.verdicts == ()
    assert result.decision == "allow"
