"""Evaluation metrics: where a guardrail's behaviour becomes a number.

The brief's ten tests come first, verbatim apart from the annotations mypy
strict needs on the two helpers. Everything after them exists because a defect
here is not a leak but a lie. A leak is discoverable by anyone who tries the
input; a wrong precision figure is believed until somebody rebuilds the
harness.

Each added test was written against a mutation of the implementation and
watched fail. The mutations and their RED output are recorded in
task-10-report.md.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import pytest

from jamjet_guardrails.detectors.pii import PiiGuardrail
from jamjet_guardrails.eval.corpus import Case, Corpus, ExpectedFinding
from jamjet_guardrails.eval.metrics import (
    Evaluation,
    EvaluationError,
    Failure,
    Metrics,
    evaluate,
)
from jamjet_guardrails.protocol import saw
from jamjet_guardrails.types import Context, Decision, Direction, Kind, Provenance, Verdict


def _case(cid: str, text: str, decision: Decision, findings: list[ExpectedFinding]) -> Case:
    return Case(
        id=cid,
        text=text,
        direction="output",
        expect_decision=decision,
        expect_findings=tuple(findings),
        source="in-repo",
        license="Apache-2.0",
    )


def _corpus(cases: list[Case]) -> Corpus:
    return Corpus(name="pii/test", source="in-repo", license="Apache-2.0", cases=tuple(cases))


def test_metrics_arithmetic() -> None:
    m = Metrics(true_positives=8, false_positives=2, false_negatives=2)
    assert m.precision == 0.8
    assert m.recall == 0.8
    assert round(m.f1, 4) == 0.8


def test_empty_prediction_and_expectation_is_perfect() -> None:
    m = Metrics(0, 0, 0)
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)


def test_predictions_with_no_expectations_is_zero_precision() -> None:
    assert Metrics(0, 3, 0).precision == 0.0


def test_a_perfect_run_scores_one() -> None:
    corpus = _corpus(
        [
            _case("a", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (5, 22))]),
            _case("b", "nothing here", "allow", []),
        ]
    )
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.overall.precision == 1.0
    assert ev.overall.recall == 1.0
    # DEVIATION from the brief, recorded rather than silent: the brief asserts
    # `== []`. `Evaluation` freezes its containers in __post_init__, following
    # types.py:112 and :125, so a published result cannot be edited in place.
    # The empty tuple is the same assertion against the frozen form.
    assert ev.failures == ()


def test_a_missed_detection_is_a_false_negative() -> None:
    corpus = _corpus([_case("a", "no pii at all", "redact", [ExpectedFinding("EMAIL", (0, 2))])])
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.overall.false_negatives == 1
    assert ev.overall.recall == 0.0
    assert [f.kind for f in ev.failures] == ["decision_mismatch", "false_negative"]


def test_an_unexpected_detection_is_a_false_positive() -> None:
    ev = evaluate(PiiGuardrail(), _corpus([_case("a", "mail alice@example.com", "allow", [])]))
    assert ev.overall.false_positives == 1
    assert ev.overall.precision == 0.0


def test_wrong_span_is_not_a_true_positive() -> None:
    """Exact spans. Fuzzy matching makes the published number unfalsifiable."""
    corpus = _corpus(
        [_case("a", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (0, 4))])]
    )
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.overall.true_positives == 0
    assert ev.overall.false_positives == 1
    assert ev.overall.false_negatives == 1


def test_expected_finding_without_span_matches_on_type_alone() -> None:
    corpus = _corpus(
        [_case("a", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", None)])]
    )
    assert evaluate(PiiGuardrail(), corpus).overall.true_positives == 1


def test_per_type_breakdown_is_reported() -> None:
    corpus = _corpus(
        [
            _case("a", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (5, 22))]),
            _case("b", "ssn 123-45-6789", "redact", [ExpectedFinding("US_SSN", (4, 15))]),
        ]
    )
    ev = evaluate(PiiGuardrail(), corpus)
    assert set(ev.per_type) == {"EMAIL", "US_SSN"}
    assert ev.per_type["EMAIL"].true_positives == 1


def test_evaluation_records_corpus_and_detector_identity() -> None:
    """Identity must be READ from the guardrail and corpus, not invented.

    The fixture deliberately disagrees with every value `evaluate` could
    plausibly hardcode: a detector named "probe" at version "9.9.9" over a
    corpus whose source is "unit-fixture". Asserting `ev.detector == "pii"`
    against a real PiiGuardrail would pass even if `evaluate` hardcoded the
    string, because the fixture would coincide with the constant. That exact
    mistake shipped three times earlier in this plan before being caught.
    """

    class Probe:
        name: str = "probe"
        version: str = "9.9.9"
        kind: Kind = "constraint"
        directions: frozenset[Direction] = frozenset({"input", "output"})

        def check(self, content: str, context: Context) -> Verdict:
            return Verdict(
                "allow",
                None,
                [],
                Provenance(kind="constraint", detector="probe", version="9.9.9"),
                saw(content),
            )

    cases = (
        Case(
            id="a",
            text="clean",
            direction="output",
            expect_decision="allow",
            expect_findings=(),
            source="unit-fixture",
            license="Apache-2.0",
        ),
    )
    corpus = Corpus(
        name="probe/unit-fixture",
        source="unit-fixture",
        license="Apache-2.0",
        cases=cases,
    )

    ev = evaluate(Probe(), corpus)
    assert ev.corpus_version == corpus.version
    assert ev.corpus_source == "unit-fixture"
    assert ev.detector == "probe"
    assert ev.detector_version == "9.9.9"


# Everything below is additional. The brief's ten tests leave four of this
# module's guards unpinned: both degenerate branches of precision and recall,
# the type half of the matching rule, and the consumption of a matched
# expectation. Each mutation below was run and its RED output recorded.


class _Fixed:
    """A guardrail whose decision never depends on the content.

    The two shapes that matter are the broken ones: allow-everything and
    deny-everything. A metric that lets either score well is not measuring what
    its name says.
    """

    name: str = "fixed"
    version: str = "0.0.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def __init__(self, decision: Decision) -> None:
        self._decision = decision

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            self._decision,
            content if self._decision == "redact" else None,
            [],
            Provenance(kind="constraint", detector=self.name, version=self.version),
            saw(content),
        )


class _Recorder:
    """Records the Context it was handed, so what evaluate passes is visible."""

    name: str = "recorder"
    version: str = "0.0.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def __init__(self) -> None:
        self.seen: list[Context] = []

    def check(self, content: str, context: Context) -> Verdict:
        self.seen.append(context)
        return Verdict(
            "allow",
            None,
            [],
            Provenance(kind="constraint", detector=self.name, version=self.version),
            saw(content),
        )


class _OutputOnly:
    """Declares one direction, the way a real direction-aware check would."""

    name: str = "output-only"
    version: str = "0.0.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"output"})

    def check(self, content: str, context: Context) -> Verdict:
        return Verdict(
            "allow",
            None,
            [],
            Provenance(kind="constraint", detector=self.name, version=self.version),
            saw(content),
        )


def _case_in(cid: str, text: str, direction: Direction) -> Case:
    return Case(
        id=cid,
        text=text,
        direction=direction,
        expect_decision="allow",
        expect_findings=(),
        source="in-repo",
        license="Apache-2.0",
    )


def test_nothing_predicted_while_something_was_expected_is_zero_precision() -> None:
    """The branch that decides what a detector which finds nothing scores.

    Precision is 0/0 here, so the convention picks the answer, and the two
    candidate answers are not equally safe. Returning 1.0 hands perfect
    precision to a detector that produced no output at all on a corpus with
    labels in it, which is the single most likely way for this module to
    publish a flattering lie. The brief's ten tests never assert precision when
    nothing was predicted and something was missed, so `return 1.0` in that
    branch survives all of them.
    """
    assert Metrics(0, 0, 3).precision == 0.0
    assert Metrics(0, 0, 3).recall == 0.0
    assert Metrics(0, 0, 3).f1 == 0.0


def test_spurious_predictions_score_zero_recall() -> None:
    """Recall's mirror of the same rule, and unpinned by the brief as well.

    Nothing was expected, so nothing could be missed, and vacuous truth would
    say 1.0. This module says 0.0 whenever a false positive is present, which
    keeps the two degenerate branches consistent: 1.0 is reserved for a run
    with no findings on either side, never handed to a run that got something
    wrong.
    """
    assert Metrics(0, 3, 0).recall == 0.0
    assert Metrics(0, 3, 0).f1 == 0.0


def test_f1_is_zero_rather_than_a_zero_division() -> None:
    assert Metrics(0, 0, 1).f1 == 0.0
    assert Metrics(0, 1, 0).f1 == 0.0


def test_metrics_refuses_a_negative_count() -> None:
    """BEYOND THE BRIEF, and argued for in the report.

    A negative count is wrong in the flattering direction rather than merely
    wrong, in both of the shapes it comes in. Metrics(-1, 1, 0) sums to nothing
    predicted, takes the degenerate branch and reports precision 1.0.
    Metrics(5, -2, 0) sums to three predicted and reports precision 1.667, a
    ratio ABOVE 1.0 leaving a public value type, which nothing downstream that
    formats or thresholds a precision expects to see.

    Each message names its own field, so matching on the field name confirms
    the right comparison fired rather than some other one carrying the same
    word.
    """
    with pytest.raises(ValueError, match="true_positives must not be negative"):
        Metrics(-1, 1, 0)
    with pytest.raises(ValueError, match="false_positives must not be negative"):
        Metrics(5, -2, 0)
    with pytest.raises(ValueError, match="false_negatives must not be negative"):
        Metrics(0, 0, -1)


def test_a_detector_that_allows_everything_scores_zero() -> None:
    """The headline question: what does a broken detector get?

    Every finding in the corpus goes unfound, so precision, recall and F1 are
    all 0.0 and every case produces two failures. This is the number that stops
    an empty implementation passing Task 12's gate.
    """
    corpus = _corpus(
        [
            _case("a", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (5, 22))]),
            _case("b", "ssn 123-45-6789", "redact", [ExpectedFinding("US_SSN", (4, 15))]),
        ]
    )
    ev = evaluate(_Fixed("allow"), corpus)
    assert ev.overall == Metrics(0, 0, 2)
    assert (ev.overall.precision, ev.overall.recall, ev.overall.f1) == (0.0, 0.0, 0.0)
    assert [f.kind for f in ev.failures] == [
        "decision_mismatch",
        "false_negative",
        "decision_mismatch",
        "false_negative",
    ]


def test_a_deny_everything_detector_is_caught_by_the_decision_mismatch_count() -> None:
    """What the ratios cannot see, and the number that can.

    Precision and recall count FINDINGS. A detector that denies every case
    while reporting no findings at all therefore scores 1.0 and 1.0 on a corpus
    that expects no findings, because on that corpus it is true that it missed
    nothing and invented nothing. The two ratios are simply blind to the thing
    that is wrong with it.

    `decision_mismatches` is the number that is not blind to it: two cases,
    two mismatches, so a caller thresholding it rejects this detector on the
    evidence rather than on the absence of evidence. It is derived from
    `failures` rather than stored, so it cannot disagree with the record it
    summarises.

    On a corpus anyone would actually publish, one with labelled findings in
    it, the same detector also scores 0.0 precision through the degenerate
    branch above. Both controls have to be checked, because a corpus of
    negatives alone leaves only the second one.
    """
    corpus = _corpus(
        [_case("a", "nothing here", "allow", []), _case("b", "still clean", "allow", [])]
    )
    ev = evaluate(_Fixed("deny"), corpus)
    assert ev.overall == Metrics(0, 0, 0)
    assert (ev.overall.precision, ev.overall.recall) == (1.0, 1.0)
    assert ev.decision_mismatches == len(corpus.cases) == 2
    assert [f.kind for f in ev.failures] == ["decision_mismatch", "decision_mismatch"]

    labelled = _corpus(
        [_case("a", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (5, 22))])]
    )
    assert evaluate(_Fixed("deny"), labelled).overall.precision == 0.0


def test_decision_mismatches_counts_only_the_decision_failures() -> None:
    """The fixture carries all three failure kinds, so counting all of them fails.

    Four failures, two of which are decision mismatches. A count over the whole
    failure list would say four, and a count that never matched would say zero.
    """
    corpus = _corpus(
        [
            _case("a", "mail alice@example.com", "allow", []),
            _case("b", "no pii at all", "redact", [ExpectedFinding("EMAIL", None)]),
        ]
    )
    ev = evaluate(PiiGuardrail(), corpus)
    assert [f.kind for f in ev.failures] == [
        "decision_mismatch",
        "false_positive",
        "decision_mismatch",
        "false_negative",
    ]
    assert ev.decision_mismatches == 2


def test_a_perfect_run_reports_no_decision_mismatches() -> None:
    corpus = _corpus(
        [
            _case("a", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (5, 22))]),
            _case("b", "nothing here", "allow", []),
        ]
    )
    assert evaluate(PiiGuardrail(), corpus).decision_mismatches == 0


def test_a_published_evaluation_cannot_be_edited_in_place() -> None:
    """A frozen value type whose containers are not frozen is not frozen.

    types.py:112 and :125 freeze theirs for the same reason: an audit record
    that a caller can edit after it was reported is not a record. It matters
    more here than there, because `decision_mismatches` reads `failures` on
    every access, so a live list would let the summary drift from the run.

    The casts are deliberate. They ask the type checker to look away so the
    test can attempt exactly what a careless caller would attempt, which is the
    only way to prove the containers refuse it at runtime.
    """
    corpus = _corpus([_case("a", "mail alice@example.com", "allow", [])])
    ev = evaluate(PiiGuardrail(), corpus)
    assert isinstance(ev.failures, tuple)
    assert isinstance(ev.per_type, MappingProxyType)
    with pytest.raises(AttributeError):
        cast(list[Failure], ev.failures).append(Failure("x", "false_positive", "a", "b"))
    with pytest.raises(TypeError):
        cast(dict[str, Metrics], ev.per_type)["EMAIL"] = Metrics()


def test_the_case_count_comes_from_the_corpus_not_from_the_findings() -> None:
    """The denominator Task 11 publishes, and the only field that separates a
    corpus of clean negatives handled perfectly from a corpus nobody scored.

    Two of these three cases expect nothing and find nothing, so they are
    invisible in tp, fp and fn and in the failure list. The count is 3 where
    every other number in the evaluation is 1 or 0, which is what makes a
    count taken from the findings instead fail here.
    """
    corpus = _corpus(
        [
            _case("a", "nothing to see here", "allow", []),
            _case("b", "still nothing at all", "allow", []),
            _case("c", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (5, 22))]),
        ]
    )
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.cases == 3
    assert ev.overall == Metrics(1, 0, 0)
    assert ev.failures == ()


def test_an_evaluation_cannot_omit_its_case_count() -> None:
    """No default, on purpose. A default of 0 would mean "nothing measured" for
    every caller who forgot it, which is both the flattering direction and the
    invisible one.
    """
    # Called through a Callable alias rather than under a type: ignore. The
    # point of the test is the RUNTIME refusal, and mypy rejecting the same
    # call is a second, independent guard that stays switched on for every
    # other line in this file.
    construct = cast(Callable[..., Evaluation], Evaluation)
    with pytest.raises(TypeError, match="required positional argument: 'cases'"):
        construct(
            corpus_name="n",
            corpus_version="v",
            corpus_source="s",
            detector="d",
            detector_version="dv",
            overall=Metrics(1, 0, 0),
            per_type={},
        )


def test_a_negative_case_count_is_refused() -> None:
    """Metrics rejects a negative count, and this one is a published
    denominator: it would appear in the README beside the scores it qualifies.
    """
    with pytest.raises(ValueError, match="cases must not be negative"):
        Evaluation(
            corpus_name="n",
            corpus_version="v",
            corpus_source="s",
            detector="d",
            detector_version="dv",
            cases=-1,
            overall=Metrics(1, 0, 0),
            per_type={},
        )


def test_zero_cases_beside_a_finding_is_refused() -> None:
    """A negative denominator was already refused because it would print beside
    the scores it qualifies. Zero cases with a finding prints the same way and
    is the same impossibility: no case means no text was checked, so there was
    nothing for a finding to be found in.
    """
    with pytest.raises(ValueError, match="cases is 0"):
        Evaluation(
            corpus_name="n",
            corpus_version="v",
            corpus_source="s",
            detector="d",
            detector_version="dv",
            cases=0,
            overall=Metrics(41, 3, 2),
            per_type={},
        )


def test_zero_cases_beside_a_failure_is_refused() -> None:
    """The other half. A failure cannot come from a case that does not exist,
    and checking only the finding counts would let this one through: a decision
    mismatch moves no count in `overall`.
    """
    with pytest.raises(ValueError, match="cases is 0"):
        Evaluation(
            corpus_name="n",
            corpus_version="v",
            corpus_source="s",
            detector="d",
            detector_version="dv",
            cases=0,
            overall=Metrics(0, 0, 0),
            per_type={},
            failures=[Failure("a", "decision_mismatch", "redact", "allow")],
        )


def test_zero_cases_beside_a_per_type_entry_is_refused() -> None:
    """The third place a count is published, and the one the guard first missed.

    `overall` and `failures` were checked and `per_type` was not, so this
    published a headline row of `| 0 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |`
    directly above a per-type table reading `| EMAIL | 0.556 | 0.625 | 5 | 4 |
    3 |` for the same corpus, and `"cases": 0` beside `"true_positives": 5` in
    one JSON object.
    """
    with pytest.raises(ValueError, match="cases is 0"):
        Evaluation(
            corpus_name="n",
            corpus_version="v",
            corpus_source="s",
            detector="d",
            detector_version="dv",
            cases=0,
            overall=Metrics(0, 0, 0),
            per_type={"EMAIL": Metrics(5, 4, 3)},
        )


def test_zero_cases_beside_an_all_zero_per_type_entry_is_refused() -> None:
    """Emptiness, not nonzero counts. A per-type entry of Metrics(0, 0, 0) has
    no counts to contradict the denominator and still publishes a row reading
    `| EMAIL | 1.000 | 1.000 | 0 | 0 | 0 |` for a type on a corpus with no
    cases, which is the same false perfect score one table lower down.
    """
    with pytest.raises(ValueError, match="cases is 0"):
        Evaluation(
            corpus_name="n",
            corpus_version="v",
            corpus_source="s",
            detector="d",
            detector_version="dv",
            cases=0,
            overall=Metrics(0, 0, 0),
            per_type={"EMAIL": Metrics(0, 0, 0)},
        )


def test_zero_cases_on_its_own_stays_legal() -> None:
    """The unscored row Task 11 publishes. Refusing this would push a caller
    back toward omitting the count, which is the state the field exists to make
    visible.
    """
    ev = Evaluation(
        corpus_name="n",
        corpus_version="v",
        corpus_source="s",
        detector="d",
        detector_version="dv",
        cases=0,
        overall=Metrics(0, 0, 0),
        per_type={},
    )
    assert ev.cases == 0


def test_more_wrong_decisions_than_cases_is_refused() -> None:
    """``evaluate`` compares one decision per case, so it appends at most one
    ``decision_mismatch`` per case and the count cannot exceed the denominator.

    Task 11 prints the two in adjacent columns of the headline row and Task 12
    gates on the count, so a pair that cannot both be true is a published
    contradiction rather than an internal one.
    """
    with pytest.raises(ValueError, match="decision_mismatches is 2"):
        Evaluation(
            corpus_name="n",
            corpus_version="v",
            corpus_source="s",
            detector="d",
            detector_version="dv",
            cases=1,
            overall=Metrics(0, 0, 0),
            per_type={},
            failures=[
                Failure("a", "decision_mismatch", "redact", "allow"),
                Failure("b", "decision_mismatch", "deny", "allow"),
            ],
        )


def test_one_wrong_decision_per_case_is_legal() -> None:
    """The boundary, and it has to be inclusive. Every case getting its decision
    wrong is a terrible result and an entirely possible one, and a guard on
    ``>=`` would refuse the worst honest run a detector can have.
    """
    ev = Evaluation(
        corpus_name="n",
        corpus_version="v",
        corpus_source="s",
        detector="d",
        detector_version="dv",
        cases=2,
        overall=Metrics(0, 0, 0),
        per_type={},
        failures=[
            Failure("a", "decision_mismatch", "redact", "allow"),
            Failure("b", "decision_mismatch", "deny", "allow"),
        ],
    )
    assert ev.decision_mismatches == 2


def test_several_failures_from_one_case_is_legal() -> None:
    """The bound is on decision mismatches, not on failures. One case can carry
    several findings and so several failures, so a guard that counted failures
    instead would refuse the ordinary case while looking like the same check.
    """
    ev = Evaluation(
        corpus_name="n",
        corpus_version="v",
        corpus_source="s",
        detector="d",
        detector_version="dv",
        cases=1,
        overall=Metrics(0, 0, 3),
        per_type={},
        failures=[
            Failure("a", "false_negative", "EMAIL@(0, 7)", "nothing"),
            Failure("a", "false_negative", "PHONE@(8, 20)", "nothing"),
            Failure("a", "false_negative", "IBAN@(21, 43)", "nothing"),
        ],
    )
    assert ev.cases == 1
    assert len(ev.failures) == 3
    assert ev.decision_mismatches == 0


def test_failures_passed_as_an_iterator_are_not_silently_dropped() -> None:
    """``__post_init__`` reads `failures` three times: for truthiness, through
    `decision_mismatches`, and to store it. Validating before freezing meant an
    iterator was truthy on the first read and exhausted by the third, so the
    evaluation stored NO failures and published `decision_mismatches: 0`, which
    is the number Task 12 gates on. A run that got every decision wrong would
    have reported a clean sheet and passed.
    """
    failures = [
        Failure("a", "decision_mismatch", "redact", "allow"),
        Failure("b", "false_negative", "EMAIL@(0, 7)", "nothing"),
    ]
    # Through a Callable alias, not a type: ignore. mypy is right that a
    # generator is not a Sequence; the defect is what happened at runtime when
    # a caller it does not check passed one anyway.
    construct = cast(Callable[..., Evaluation], Evaluation)
    ev = construct(
        corpus_name="n",
        corpus_version="v",
        corpus_source="s",
        detector="d",
        detector_version="dv",
        cases=20,
        overall=Metrics(0, 0, 1),
        per_type={},
        failures=(failure for failure in failures),
    )
    assert len(ev.failures) == 2
    assert ev.decision_mismatches == 1


def test_an_evaluation_copies_the_containers_it_was_handed() -> None:
    """MappingProxyType is a VIEW, so freezing without copying freezes nothing.

    The caller keeps both containers it passed and edits them afterwards. This
    is corpus.Case's "copy first, then validate the copy" argument in its other
    form: a caller who keeps the reference can otherwise walk the guard around
    itself. types.Verdict is cited nowhere here any more, because it used to
    name the opposite order and was changed to this one.
    """
    per_type = {"EMAIL": Metrics(1, 0, 0)}
    failures = [Failure("a", "false_positive", "nothing", "EMAIL@(0, 7)")]
    ev = Evaluation(
        corpus_name="n",
        corpus_version="v",
        corpus_source="s",
        detector="d",
        detector_version="dv",
        cases=1,
        overall=Metrics(1, 0, 0),
        per_type=per_type,
        failures=failures,
    )
    per_type["US_SSN"] = Metrics(0, 9, 0)
    failures.clear()

    assert dict(ev.per_type) == {"EMAIL": Metrics(1, 0, 0)}
    assert len(ev.failures) == 1


def test_a_finding_of_the_wrong_type_is_not_a_true_positive() -> None:
    """The other half of the matching rule. The span alone must not match.

    The expectation and the prediction cover exactly the same characters and
    disagree only about what those characters are. Scoring that as a hit would
    publish recall for a detector that found the right text for the wrong
    reason, and Task 11 prints the type next to the number.
    """
    corpus = _corpus([_case("a", "ssn 123-45-6789", "redact", [ExpectedFinding("EMAIL", (4, 15))])])
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.overall == Metrics(0, 1, 1)
    assert set(ev.per_type) == {"EMAIL", "US_SSN"}


def test_one_expectation_cannot_absorb_two_predictions() -> None:
    """A matched expectation is consumed, so counts stay conservation-safe.

    Two addresses, one label. Without consumption both predictions match the
    single expectation and the case scores 2 true positives out of 1 label,
    which is precision above what the corpus can support.
    """
    corpus = _corpus([_case("a", "a@b.com c@d.com", "redact", [ExpectedFinding("EMAIL", None)])])
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.overall == Metrics(1, 1, 0)


def test_greedy_matching_can_understate_and_never_overstates() -> None:
    """The approximation this module makes, recorded rather than left implicit.

    Each prediction takes the first compatible expectation, which is not the
    best possible pairing. Here the span-free label absorbs the first address,
    leaving the exactly-labelled one to be scored as a false positive plus a
    false negative, where an optimal assignment would have found two hits.

    Greedy always produces a VALID pairing, and optimal is the maximum over
    valid pairings, so this can only ever report fewer true positives than the
    truth, never more. The error direction is the one a published number is
    allowed to have.
    """
    corpus = _corpus(
        [
            _case(
                "a",
                "a@b.com c@d.com",
                "redact",
                [ExpectedFinding("EMAIL", None), ExpectedFinding("EMAIL", (0, 7))],
            )
        ]
    )
    assert evaluate(PiiGuardrail(), corpus).overall == Metrics(1, 1, 1)


def test_per_type_counts_false_positives_and_false_negatives_separately() -> None:
    corpus = _corpus(
        [
            _case("a", "no pii at all", "redact", [ExpectedFinding("EMAIL", None)]),
            _case("b", "ssn 123-45-6789", "redact", []),
        ]
    )
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.per_type["EMAIL"] == Metrics(0, 0, 1)
    assert ev.per_type["US_SSN"] == Metrics(0, 1, 0)


def test_per_type_is_ordered_by_type_not_by_first_appearance() -> None:
    """The fixture meets the two orderings in opposite orders on purpose.

    US_SSN is seen first and sorts second, so a per_type dict left in insertion
    order and one sorted by type cannot both satisfy this. Task 11 prints these
    rows, and a report whose row order depends on corpus order is a diff that
    moves for no reason.
    """
    corpus = _corpus(
        [
            _case("a", "ssn 123-45-6789", "redact", [ExpectedFinding("US_SSN", (4, 15))]),
            _case("b", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (5, 22))]),
        ]
    )
    assert list(evaluate(PiiGuardrail(), corpus).per_type) == ["EMAIL", "US_SSN"]


def test_a_failure_names_the_case_it_came_from() -> None:
    """BOTH cases fail, so "names its own case" is the only thing that passes.

    An earlier version of this test had only the second case fail, and
    `["second", "second"]` is equally satisfied by naming the last case, by
    naming the current case, and by a hardcoded string. Two failing cases
    separate all three.
    """
    corpus = _corpus(
        [
            _case("first", "ssn 123-45-6789", "allow", []),
            _case("second", "mail alice@example.com", "allow", []),
        ]
    )
    ev = evaluate(PiiGuardrail(), corpus)
    assert [f.case_id for f in ev.failures] == ["first", "first", "second", "second"]


def test_a_decision_mismatch_records_expected_then_predicted() -> None:
    """Order matters: Task 11 prints these two columns under those headings."""
    ev = evaluate(PiiGuardrail(), _corpus([_case("a", "mail alice@example.com", "allow", [])]))
    assert ev.failures[0] == Failure("a", "decision_mismatch", "allow", "redact")


def test_a_false_positive_failure_names_what_was_predicted() -> None:
    ev = evaluate(PiiGuardrail(), _corpus([_case("a", "mail alice@example.com", "allow", [])]))
    assert ev.failures[1] == Failure("a", "false_positive", "nothing", "EMAIL@(5, 22)")


def test_a_false_negative_failure_names_what_was_expected() -> None:
    corpus = _corpus([_case("a", "no pii at all", "redact", [ExpectedFinding("EMAIL", (0, 2))])])
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.failures[1] == Failure("a", "false_negative", "EMAIL@(0, 2)", "nothing")


def test_the_case_direction_and_a_model_origin_reach_the_guardrail() -> None:
    """direction is measured input, which is why the corpus version hashes it.

    The two cases carry different directions in a fixed order, so hardcoding
    either value fails. origin is not a corpus field and has one value by
    construction; it is asserted so a change to it has to be deliberate.
    """
    recorder = _Recorder()
    evaluate(recorder, _corpus([_case_in("a", "one", "input"), _case_in("b", "two", "output")]))
    assert [c.direction for c in recorder.seen] == ["input", "output"]
    assert {c.origin for c in recorder.seen} == {"model"}


def test_a_guardrail_is_not_scored_on_a_direction_it_does_not_declare() -> None:
    """BEYOND THE BRIEF, and argued for in the report.

    GuardrailChain skips a guardrail whose directions exclude the context's
    direction, so scoring this pairing would publish a number for a run that
    never happens in deployment. Silently scoring it is the failure this
    module cannot afford; refusing the whole corpus rather than the offending
    cases keeps the denominator equal to the corpus the version key names.

    The error is typed, and the type is the point. A bare ValueError out of
    `evaluate` passes straight through Task 13's CLI, which nets CorpusError
    and GuardrailUnavailableError, and lands as a traceback. EvaluationError
    is one more net of the same shape, following eval/corpus.py's rule of one
    error type per door.
    """
    corpus = _corpus([_case_in("a", "one", "input")])
    with pytest.raises(EvaluationError, match=r"does not run on direction 'input'"):
        evaluate(_OutputOnly(), corpus)


def test_evaluation_records_the_corpus_name() -> None:
    """The one identity field the brief's identity test leaves unasserted."""
    corpus = Corpus(
        name="probe/named-fixture",
        source="unit-fixture",
        license="Apache-2.0",
        cases=(
            Case(
                id="a",
                text="clean",
                direction="output",
                expect_decision="allow",
                expect_findings=(),
                source="unit-fixture",
                license="Apache-2.0",
            ),
        ),
    )
    assert evaluate(_Fixed("allow"), corpus).corpus_name == "probe/named-fixture"


def test_counts_accumulate_across_cases() -> None:
    corpus = _corpus(
        [
            _case("a", "mail alice@example.com", "redact", [ExpectedFinding("EMAIL", (5, 22))]),
            _case("b", "no pii at all", "redact", [ExpectedFinding("EMAIL", None)]),
        ]
    )
    ev = evaluate(PiiGuardrail(), corpus)
    assert ev.overall == Metrics(1, 0, 1)
    assert ev.overall.precision == 1.0
    assert ev.overall.recall == 0.5
    assert round(ev.overall.f1, 4) == 0.6667
