"""Precision, recall, and the failing cases behind them.

This is where a guardrail's behaviour becomes a number. Task 11 formats that
number, Task 12 refuses to let it regress, and the README publishes it. Every
defect upstream of here leaks; a defect here lies, and a lie is worse. Anyone
can try an input and discover a leak, while a wrong precision figure is
believed until somebody rebuilds the harness.

Three rules do the work, and each of them is a choice about which way an
ambiguous case falls.

**Matching is exact, never fuzzy.** A prediction is a hit when an expectation
has the same type and, when that expectation names a span, the same span.
Overlap matching would let a detector claim credit for a region it only partly
found, and a published number nobody can falsify is not evidence.

**A degenerate ratio resolves toward 0.0, not toward 1.0.** Precision and
recall are both 0/0 in the cases that matter most, and 1.0 is reserved for the
single honest one: nothing predicted and nothing expected. Any other 0/0 has a
finding on one side of it and scores zero. The reason is what the flattering
answer would buy: a detector that returns allow for every input predicts
nothing, so under a vacuous 1.0 it would publish perfect precision on a fully
labelled corpus.

**Identity is read, never invented.** Which detector, which version, which
corpus, which corpus version: all four come off the guardrail and the corpus
that were passed in. A hardcoded string here mislabels the evidence behind
every number this project publishes, and it is invisible in the output because
the value it prints is a plausible one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from jamjet_guardrails.eval.corpus import Corpus
from jamjet_guardrails.protocol import Guardrail
from jamjet_guardrails.types import Context

FailureKind = Literal["false_positive", "false_negative", "decision_mismatch"]


class EvaluationError(Exception):
    """A guardrail and a corpus that cannot honestly be scored against each other.

    Typed, and local to this module, for the reason ``CorpusError`` gives one
    module over: one error type per door, rather than a builtin a caller has to
    know to catch. A bare ``ValueError`` out of ``evaluate`` would pass straight
    through Task 13's CLI, which nets ``CorpusError`` and
    ``GuardrailUnavailableError``, and surface as a traceback.

    It lives here rather than in ``errors.py`` because ``errors.py`` is the
    runtime chain's contract. This is the harness's, and nothing that runs in
    production can raise it.
    """


@dataclass(frozen=True, slots=True)
class Metrics:
    """Three counts, and the three ratios everything downstream quotes."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def __post_init__(self) -> None:
        # A negative count is not merely wrong, it is wrong in the flattering
        # direction, in BOTH of the two shapes it comes in.
        #
        #   Metrics(-1, 1, 0) has predicted == 0, so it takes the degenerate
        #   branch below and reports precision 1.0.
        #   Metrics(5, -2, 0) has predicted == 3 and reports precision 1.667,
        #   a ratio ABOVE 1.0 leaving a public value type, which nothing
        #   downstream that formats or thresholds a precision expects to see.
        #
        # Nothing in this package produces either today, and this type is
        # public, so the guard costs one comparison and closes both shapes.
        for name in ("true_positives", "false_positives", "false_negatives"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")

    @property
    def precision(self) -> float:
        """Of what was predicted, how much was right.

        With nothing predicted the ratio is 0/0 and the convention decides. It
        decides 1.0 only when nothing was expected either, and 0.0 when
        something was expected and missed. That second branch is the one that
        makes an empty detector score zero rather than perfect.
        """
        predicted = self.true_positives + self.false_positives
        if predicted == 0:
            return 1.0 if self.false_negatives == 0 else 0.0
        return self.true_positives / predicted

    @property
    def recall(self) -> float:
        """Of what was expected, how much was found.

        The mirror of precision, including its degenerate branch: nothing
        expected and nothing predicted is 1.0, nothing expected while
        something was predicted is 0.0. Vacuous truth would say 1.0 for the
        second, which would hand perfect recall to a detector that fired on
        every clean case.
        """
        actual = self.true_positives + self.false_negatives
        if actual == 0:
            return 1.0 if self.false_positives == 0 else 0.0
        return self.true_positives / actual

    @property
    def f1(self) -> float:
        """Harmonic mean, zero when both inputs are zero rather than a crash."""
        p, r = self.precision, self.recall
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)


@dataclass(frozen=True, slots=True)
class Failure:
    """One case that did not do what the corpus says it should.

    ``expected`` and ``predicted`` are rendered strings rather than values,
    because the three kinds compare different things: two decisions, or a
    finding against nothing. Task 11 prints them as two columns under those
    two headings, so their order is part of the contract.
    """

    case_id: str
    kind: FailureKind
    expected: str
    predicted: str


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One detector measured against one corpus, and the identity of both.

    The five identity fields are not decoration. A precision figure without
    the corpus version beside it cannot be reproduced or regression-checked,
    which is exactly what Task 12 needs from it.

    ``cases`` is the denominator, and it is the field that makes the rest
    falsifiable. The three counts in ``overall`` say what was FOUND, not what
    was measured: a corpus of clean negatives scored perfectly produces
    ``Metrics(0, 0, 0)``, an empty ``per_type`` and no failures, which is
    byte-identical to a corpus that was never scored at all. "Precision 0.953"
    and "precision 0.953 over 44 cases" are not the same claim, and only the
    second one can be checked.

    It has NO default, deliberately. Every construction site has to state it,
    and a default of 0 would quietly mean "nothing measured" for any caller who
    forgot, which is the flattering direction and the invisible one.
    """

    corpus_name: str
    corpus_version: str
    corpus_source: str
    detector: str
    detector_version: str
    cases: int
    overall: Metrics
    per_type: Mapping[str, Metrics]
    failures: Sequence[Failure] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Copied and frozen BEFORE anything reads them. The copy is what makes
        # the freeze real: a caller who kept the dict or the list it passed could
        # otherwise edit a result after it was reported, and `decision_mismatches`
        # reads `failures` on every access, so a live list would let the count
        # drift from the run it describes. MappingProxyType is a VIEW, so
        # wrapping the caller's dict without copying it first would freeze
        # nothing at all.
        #
        # Before, and not after, because the checks below read
        # `failures` three times: once for truthiness, once through
        # `decision_mismatches`, and once to store it. An iterator is truthy
        # while unread, empty by the third read, so a caller passing a generator
        # would publish `decision_mismatches: 0` for an evaluation that had them
        # all. That is the gated number, silently zeroed.
        object.__setattr__(self, "per_type", MappingProxyType(dict(self.per_type)))
        object.__setattr__(self, "failures", tuple(self.failures))
        # Metrics rejects a negative count for the reason that applies here
        # twice over: this one is a published denominator, so a negative value
        # would appear in the README beside the scores it is supposed to
        # qualify.
        if self.cases < 0:
            raise ValueError(f"cases must not be negative, got {self.cases}")
        # Zero cases with something found is the same defect as a negative
        # count, and it prints the same way: a denominator standing beside
        # numbers it cannot have produced. No case means no text was checked,
        # so there was nothing for a finding, a failure or a TYPE to come from.
        #
        # per_type is checked for EMPTINESS, not for nonzero counts. A per-type
        # entry of Metrics(0, 0, 0) still publishes a row reading 1.000 | 1.000
        # for a type on a corpus with no cases, which is the same false perfect
        # score one table lower down.
        #
        # cases=0 ON ITS OWN stays legal. That is the honest unscored row, and
        # refusing it would push a caller back toward omitting the count.
        found = (
            self.overall.true_positives
            + self.overall.false_positives
            + self.overall.false_negatives
        )
        if self.cases == 0 and (found or self.failures or self.per_type):
            raise ValueError(
                f"cases is 0 but the evaluation reports {found} finding(s), "
                f"{len(self.failures)} failure(s) and {len(self.per_type)} per-type "
                "entries; no case means nothing was checked"
            )
        # The same argument one step over. ``evaluate`` compares one decision
        # per case, so it appends at most one ``decision_mismatch`` per case and
        # the count can never exceed the denominator. Task 11 prints the two in
        # adjacent columns of the headline row and Task 12 gates on the count,
        # so a pair that cannot both be true is a published contradiction rather
        # than an internal one.
        #
        # The bound is on decision mismatches, NOT on failures: one case can
        # carry several findings and so several failures, which is ordinary.
        #
        # The next member of this family, "distinct case ids among the failures
        # must not exceed cases", is true of anything evaluate produces and is
        # deliberately NOT enforced. The line is not that those ids are
        # unpublished, because to_json prints every one of them in the same
        # object as `cases`. It is that they are not published AS A NUMBER: the
        # guards here protect figures a reader sees side by side and can compare
        # at a glance, and a reader who would have to count a JSON list to find
        # the contradiction is doing arithmetic the artifact never asked them to
        # do. That line is a judgement about the artifact, not about the data.
        if self.decision_mismatches > self.cases:
            raise ValueError(
                f"decision_mismatches is {self.decision_mismatches} but only {self.cases} "
                "case(s) were scored; a case can disagree about its decision at most once"
            )

    @property
    def decision_mismatches(self) -> int:
        """Cases where the verdict disagreed with the label.

        The number precision and recall cannot see. Both count findings, so a
        guardrail that locates every span and then returns the wrong decision
        scores 1.0 on both while being wrong about every case a user cares
        about. Task 11 publishes this beside the ratios and Task 12 gates on
        it.

        A property rather than a field: derived from ``failures``, so it cannot
        drift from them, and a hand-built Evaluation cannot misreport it.
        """
        return sum(1 for f in self.failures if f.kind == "decision_mismatch")


def evaluate(guardrail: Guardrail, corpus: Corpus) -> Evaluation:
    """Score one guardrail over one corpus, exactly, or refuse to score it.

    Findings are matched greedily: each prediction takes the FIRST expectation
    compatible with it, and that expectation is then consumed so a second
    prediction cannot claim it too. Greedy is not the best possible pairing.
    An expectation with no span can absorb a prediction that an
    exactly-labelled expectation would have matched, leaving the latter to be
    counted as a false positive plus a false negative where an optimal
    assignment would have found two hits.

    That approximation is kept deliberately, because of its direction. Greedy
    produces a valid pairing and optimal is the maximum over valid pairings,
    so this can only ever report FEWER true positives than the truth, never
    more. A published number is allowed to understate a detector. It is not
    allowed to overstate one, which is what any error in the other direction
    would do.

    Note what the two ratios do and do not cover. They count findings.
    ``expect_decision`` is checked too, but a mismatch lands in ``failures``
    and moves neither number, so a detector that reports the right findings
    and the wrong decision every time still scores 1.0 and 1.0. That is what
    ``Evaluation.decision_mismatches`` reports, and a caller gating on this
    output gates on it as well as on the two ratios.

    Raises:
        EvaluationError: if any case carries a direction the guardrail does
            not declare. ``GuardrailChain`` SKIPS a guardrail outside its
            directions, so scoring that pairing would publish a number for a
            run that never happens in deployment. Every case is checked before
            any is scored: dropping the offending cases instead would measure
            a smaller corpus than the one whose version key is printed beside
            the result, which is the same lie one step quieter.
    """
    for case in corpus.cases:
        if case.direction not in guardrail.directions:
            raise EvaluationError(
                f"guardrail {guardrail.name!r} does not run on direction "
                f"{case.direction!r}, which case {case.id!r} of corpus "
                f"{corpus.name!r} carries; the chain skips a guardrail outside its "
                "directions, so a score for this pairing would describe a run that "
                "never happens"
            )

    tp = fp = fn = 0
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    failures: list[Failure] = []

    for case in corpus.cases:
        # direction comes off the case because it is measured input: the
        # corpus version hashes it for the same reason. origin has no corpus
        # field and is fixed here; the bundled constraints do not read it.
        verdict = guardrail.check(case.text, Context(direction=case.direction, origin="model"))

        if verdict.decision != case.expect_decision:
            failures.append(
                Failure(case.id, "decision_mismatch", case.expect_decision, verdict.decision)
            )

        # A per-case working copy, so consuming a match cannot reach the
        # corpus, and so an expectation can only be claimed by a prediction
        # from its own case.
        expected = list(case.expect_findings)
        for predicted in verdict.findings:
            match = next(
                (
                    e
                    for e in expected
                    if e.type == predicted.type and (e.span is None or e.span == predicted.span)
                ),
                None,
            )
            if match is not None:
                expected.remove(match)
                tp += 1
                per[predicted.type][0] += 1
            else:
                fp += 1
                per[predicted.type][1] += 1
                failures.append(
                    Failure(
                        case.id,
                        "false_positive",
                        "nothing",
                        f"{predicted.type}@{predicted.span}",
                    )
                )

        for leftover in expected:
            fn += 1
            per[leftover.type][2] += 1
            failures.append(
                Failure(case.id, "false_negative", f"{leftover.type}@{leftover.span}", "nothing")
            )

    return Evaluation(
        corpus_name=corpus.name,
        corpus_version=corpus.version,
        corpus_source=corpus.source,
        detector=guardrail.name,
        detector_version=guardrail.version,
        # The denominator, read off the corpus that was actually scored rather
        # than counted from the findings. Every case above went through
        # guardrail.check, including the ones that expected nothing and got
        # nothing, and those are invisible in tp, fp and fn.
        cases=len(corpus.cases),
        overall=Metrics(tp, fp, fn),
        # Sorted by type, so the report's row order is a property of the
        # detector rather than of which case happened to come first in the
        # file.
        per_type={t: Metrics(*counts) for t, counts in sorted(per.items())},
        failures=failures,
    )
