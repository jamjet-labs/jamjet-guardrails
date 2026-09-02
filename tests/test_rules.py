"""The `rules` check: the primitive under a registry name.

The primitive's own behaviour is `tests/test_authoring.py`. What is here is
what registration adds: the name on the verdict, the default decision, the
refusal to build without options, and the corpus the published row is measured
on.
"""

from __future__ import annotations

import pytest

from jamjet_guardrails.detectors import AVAILABLE, TYPES, build
from jamjet_guardrails.detectors.rules import RULES_TYPES
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.eval.fixtures import options_for
from jamjet_guardrails.types import Context

IN = Context(direction="input", origin="user")


def test_rules_is_registered() -> None:
    assert "rules" in AVAILABLE


def test_building_rules_with_no_options_is_refused() -> None:
    """The whole reason this check needs a fixture. A rules engine with no
    rules allows everything while its configuration says a check is running."""
    with pytest.raises(GuardrailUnavailableError, match="check nothing"):
        build("rules")


def test_the_verdict_names_the_registered_check_and_its_version() -> None:
    guard = build("rules", **options_for("rules"))
    verdict = guard.check("see JIRA-1234", IN)
    assert verdict.provenance.detector == "rules"
    assert verdict.provenance.version == guard.version


def test_rules_defaults_to_deny_when_the_caller_does_not_choose() -> None:
    """The FIXTURE redacts, so the default is not visible in the published row
    and needs its own test. A caller who has not thought about it gets the safe
    answer."""
    guard = build("rules", patterns={"TICKET_ID": r"JIRA-\d{4,}"})
    assert guard.check("JIRA-1234", IN).decision == "deny"


def test_rules_runs_in_both_directions() -> None:
    assert build("rules", **options_for("rules")).directions == frozenset({"input", "output"})


def test_the_declared_types_are_the_fixture_types_plus_the_length_limit() -> None:
    """The two cannot drift: RULES_TYPES is what the README row lists and what
    the corpus may label, and the fixture is what the row is measured under."""
    fixture = options_for("rules")
    declared: set[str] = {"LENGTH_LIMIT"}
    # `fixture.get(...)` is typed `object` (FIXTURES is Mapping[str, object]),
    # so `set()` sees an argument outside its overloads: call-overload, not
    # arg-type, is the code mypy actually reports here.
    declared |= set(fixture.get("patterns", {}))  # type: ignore[call-overload]
    declared |= set(fixture.get("banned", {}))  # type: ignore[call-overload]
    assert RULES_TYPES == declared
    assert TYPES["rules"] == RULES_TYPES
