"""The audit record: what it carries, and the one thing it must never carry."""

from __future__ import annotations

import json
from typing import Any

from jamjet_guardrails import Context, Direction, Kind, build_chain, saw

from jamjet_guardrails_nemo import parse_audit_record
from jamjet_guardrails_nemo._audit import record_for, record_for_failure

SECRET = "sk-abcdefghijklmnopqrstuvwxyz012345"
TEXT = f"mail alice@example.com and use {SECRET}"
OUT = Context(direction="output", origin="model")


def _record() -> dict[str, Any]:
    result = build_chain(["pii", "secrets"]).run(TEXT, OUT)
    return parse_audit_record(record_for("output", result))


def test_the_record_carries_provenance_decision_types_and_spans() -> None:
    record = _record()
    assert record["decision"] == "redact"
    assert record["direction"] == "output"
    assert record["saw"] == saw(TEXT)
    verdicts = record["verdicts"]
    assert isinstance(verdicts, list)
    by_detector = {v["detector"]: v for v in verdicts}
    assert set(by_detector) == {"pii", "secrets"}
    assert by_detector["pii"]["types"] == ["EMAIL"]
    assert by_detector["pii"]["spans"] == [[5, 22]]
    assert by_detector["secrets"]["decision"] == "redact"
    assert by_detector["secrets"]["kind"] == "constraint"
    assert isinstance(by_detector["secrets"]["version"], str)


def test_the_record_never_carries_the_content() -> None:
    """The whole point, asserted over the serialised bytes rather than a field.

    Checking field by field would only cover the fields somebody remembered. The
    record is one string, so the question "is the secret in it" has one answer,
    and it stays answerable when a field is added.
    """
    result = build_chain(["pii", "secrets"]).run(TEXT, OUT)
    record = record_for("output", result)
    assert SECRET not in record
    assert "alice@example.com" not in record
    # And not the rewritten content either. A placeholder is harmless, but the
    # chain's merged string is the whole message with holes in it, which is
    # still the message.
    assert result.content not in record


def test_every_verdict_saw_the_same_digest_the_record_reports() -> None:
    """What makes the single top-level `saw` honest.

    `GuardrailChain` hashes the content once and stamps that digest onto every
    verdict it builds, so flattening the per-verdict values into one field loses
    nothing. That is a property of the chain, not of this module, so it is
    asserted against a real run: if it ever stopped holding, this record would be
    quietly reporting one detector's view as if it were every detector's.
    """
    result = build_chain(["pii", "secrets"]).run(TEXT, OUT)
    record = parse_audit_record(record_for("output", result))
    assert len({verdict.saw for verdict in result.verdicts}) == 1
    assert record["saw"] == result.verdicts[0].saw


def test_the_failure_record_carries_the_exception_type_and_not_its_message() -> None:
    """An exception's message routinely quotes what it choked on.

    What it choked on is the content. `jamjet_guardrails.chain` makes the same
    choice one layer down and for the same reason.
    """
    record = parse_audit_record(
        record_for_failure("input", saw(TEXT), ValueError(f"bad value {SECRET}"))
    )
    assert record["error"] == "ValueError"
    assert SECRET not in json.dumps(record)
    assert record["saw"] == saw(TEXT)
    assert record["decision"] == "deny"
    assert record["verdicts"] == []


def test_a_failure_with_no_content_reports_no_digest_rather_than_a_wrong_one() -> None:
    """A hash of the empty string would claim something was inspected."""
    record = parse_audit_record(record_for_failure("input", None, ValueError("x")))
    assert record["saw"] is None
    assert record["saw"] != saw("")


def test_types_and_spans_line_up_by_index() -> None:
    """They are read together, so a shorter spans list mislabels every span after
    the gap.

    Exercised with a finding that HAS no span, which is what makes the assertion
    mean something: over the bundled constraints every finding is located, so a
    record that silently dropped unlocated findings from `spans` would keep the
    two lists the same length and this test would pass over the bug. A classifier
    reports `span=None` by design, so the gap is a real case rather than a
    contrived one.
    """
    from jamjet_guardrails import Finding, GuardrailChain, Provenance, Verdict

    class Classifier:
        name: str = "toy-classifier"
        version: str = "0.1.0"
        kind: Kind = "classifier"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "deny",
                None,
                [
                    Finding(type="LOCATED", span=(0, 4), confidence=0.9),
                    Finding(type="UNLOCATED", span=None, confidence=0.8),
                ],
                Provenance(kind="classifier", detector="toy-classifier", version="0.1.0"),
                saw(content),
            )

    result = GuardrailChain([Classifier()]).run(TEXT, OUT)
    record = parse_audit_record(record_for("output", result))
    verdict = record["verdicts"][0]
    assert verdict["types"] == ["LOCATED", "UNLOCATED"]
    assert verdict["spans"] == [[0, 4], None]

    for real in _record()["verdicts"]:
        assert len(real["types"]) == len(real["spans"])
