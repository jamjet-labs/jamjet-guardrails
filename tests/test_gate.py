"""The regression gate: the module that refuses, and the only one that decides.

The brief's twenty tests come first, verbatim apart from the annotations mypy
strict needs on the two helpers and on every test, and the explicit
``dict[str, dict[str, object]]`` on the baseline fixtures (a bare literal infers
``dict[str, float]``, which will not hold the string, bool and float entries the
later tests put in it).

Everything after them was written against a mutation of the brief's own
implementation and watched fail. Those mutations and their RED output are
recorded in task-12-report.md.
"""

from collections.abc import Sequence
from typing import cast

import pytest

from jamjet_guardrails.eval.gate import (
    MAX_EPSILON,
    RegressionError,
    baseline_key,
    check_regression,
)
from jamjet_guardrails.eval.metrics import Evaluation, Failure, Metrics

BASE: dict[str, dict[str, object]] = {
    "pii/in-repo/abc123def456": {
        "precision": 0.9,
        "recall": 0.9,
        "decision_mismatches": 0,
    }
}


def _ev(
    tp: int = 9,
    fp: int = 1,
    fn: int = 1,
    version: str = "abc123def456",
    failures: Sequence[Failure] = (),
    cases: int = 20,
) -> Evaluation:
    # `cases` is 20 and shares no value with tp/fp/fn, so a field that read the
    # wrong count cannot render the same number.
    return Evaluation(
        corpus_name="pii/in-repo",
        corpus_version=version,
        corpus_source="in-repo",
        detector="pii",
        detector_version="0.1.0",
        cases=cases,
        overall=Metrics(tp, fp, fn),
        per_type={},
        failures=list(failures),
    )


def _wrong(n: int) -> list[Failure]:
    """n cases whose verdict disagreed with the label."""
    return [Failure(f"c{i}", "decision_mismatch", "redact", "allow") for i in range(n)]


def test_baseline_key_shape() -> None:
    assert baseline_key(_ev()) == "pii/in-repo/abc123def456"


def test_passes_when_scores_match_baseline() -> None:
    check_regression([_ev()], BASE)


def test_passes_when_scores_improve() -> None:
    check_regression([_ev(tp=10, fp=0, fn=0)], BASE)


def test_fails_when_precision_drops() -> None:
    with pytest.raises(RegressionError, match="precision"):
        check_regression([_ev(tp=5, fp=5, fn=1)], BASE)


def test_fails_when_recall_drops() -> None:
    with pytest.raises(RegressionError, match="recall"):
        check_regression(
            [_ev(tp=5, fp=1, fn=5)],
            {
                "pii/in-repo/abc123def456": {
                    "precision": 0.5,
                    "recall": 0.9,
                    "decision_mismatches": 0,
                }
            },
        )


def test_a_tiny_drop_inside_epsilon_is_tolerated() -> None:
    check_regression(
        [_ev()],
        {
            "pii/in-repo/abc123def456": {
                "precision": 0.902,
                "recall": 0.902,
                "decision_mismatches": 0,
            }
        },
    )


def test_a_gate_that_checked_nothing_does_not_pass() -> None:
    """The CodeRabbit failure mode: a green check meaning the gate never ran."""
    with pytest.raises(RegressionError, match="checked nothing"):
        check_regression([], BASE)


def test_an_orphaned_baseline_fails_loudly() -> None:
    """Deleting a corpus must not silently retire its guard."""
    baselines = dict(BASE)
    baselines["secrets/in-repo/cafebabe1234"] = {
        "precision": 0.9,
        "recall": 0.9,
        "decision_mismatches": 0,
    }
    with pytest.raises(RegressionError, match="no corresponding corpus"):
        check_regression([_ev()], baselines)


def test_a_missing_baseline_fails_loudly() -> None:
    """A changed corpus produces a new key. An unknown key must never pass silently."""
    with pytest.raises(RegressionError, match="no baseline"):
        check_regression([_ev(version="deadbeef0000")], BASE)


def test_the_error_names_every_failing_check_not_just_the_first() -> None:
    with pytest.raises(RegressionError) as exc:
        check_regression([_ev(tp=1, fp=9, fn=1), _ev(version="ffffffffffff")], BASE)
    assert "abc123def456" in str(exc.value)
    assert "ffffffffffff" in str(exc.value)


def test_a_nan_epsilon_is_refused() -> None:
    """Every `<` against NaN is False, so a NaN tolerance passes every regression
    while looking like a number in a CI yaml nobody re-reads. Without the guard
    this call returns cleanly on a detector scoring 0.000."""
    with pytest.raises(RegressionError, match="epsilon"):
        check_regression([_ev(tp=0, fp=9, fn=9)], BASE, epsilon=float("nan"))


def test_an_epsilon_wide_enough_to_disable_the_gate_is_refused() -> None:
    """Without the guard, epsilon=1.0 passes a detector that found nothing."""
    with pytest.raises(RegressionError, match="epsilon"):
        check_regression([_ev(tp=0, fp=9, fn=9)], BASE, epsilon=1.0)


def test_a_negative_epsilon_is_refused() -> None:
    """This evaluation beats its baseline, so the guard is the only thing that
    can raise here."""
    with pytest.raises(RegressionError, match="epsilon"):
        check_regression([_ev(tp=10, fp=0, fn=0)], BASE, epsilon=-0.1)


def test_a_baseline_missing_a_metric_fails_loudly() -> None:
    with pytest.raises(RegressionError, match="records no recall"):
        check_regression(
            [_ev()],
            {"pii/in-repo/abc123def456": {"precision": 0.9, "decision_mismatches": 0}},
        )


def test_a_baseline_missing_the_decision_count_fails_loudly() -> None:
    """The count a hand-edited baselines file is most likely to omit, because it
    is the newest of the three. Absent must not read as unlimited."""
    with pytest.raises(RegressionError, match="records no decision_mismatches"):
        check_regression(
            [_ev(failures=_wrong(3))],
            {"pii/in-repo/abc123def456": {"precision": 0.9, "recall": 0.9}},
        )


def test_fails_when_more_decisions_go_wrong() -> None:
    """Both ratios are UNCHANGED here. Only the decision count moves, which is
    the whole reason it is gated separately."""
    with pytest.raises(RegressionError, match="decision_mismatches 1 above baseline 0"):
        check_regression([_ev(failures=_wrong(1))], BASE)


def test_passes_when_fewer_decisions_go_wrong() -> None:
    baselines: dict[str, dict[str, object]] = {
        "pii/in-repo/abc123def456": {
            "precision": 0.9,
            "recall": 0.9,
            "decision_mismatches": 5,
        }
    }
    check_regression([_ev(failures=_wrong(2))], baselines)


def test_a_non_integer_decision_baseline_fails_loudly() -> None:
    with pytest.raises(RegressionError, match="decision_mismatches is not an integer"):
        check_regression(
            [_ev()],
            {
                "pii/in-repo/abc123def456": {
                    "precision": 0.9,
                    "recall": 0.9,
                    "decision_mismatches": 1.5,
                }
            },
        )


def test_a_non_numeric_baseline_fails_loudly() -> None:
    with pytest.raises(RegressionError, match="not a finite number"):
        check_regression(
            [_ev()],
            {
                "pii/in-repo/abc123def456": {
                    "precision": "0.9",
                    "recall": 0.9,
                    "decision_mismatches": 0,
                }
            },
        )


def test_a_boolean_baseline_is_not_a_number() -> None:
    """True == 1, so an unchecked bool reads as a perfect baseline that nothing
    can ever beat, and every later run reports a regression against it."""
    with pytest.raises(RegressionError, match="not a finite number"):
        check_regression(
            [_ev()],
            {
                "pii/in-repo/abc123def456": {
                    "precision": True,
                    "recall": 0.9,
                    "decision_mismatches": 0,
                }
            },
        )


# --- Beyond the brief -------------------------------------------------------
#
# Each of these was watched failing against the brief's own implementation.


def test_a_one_shot_iterable_of_evaluations_is_still_gated() -> None:
    """The gate's own failure mode, one step over from an empty corpus list.

    The brief's body iterates its argument TWICE: once to collect the keys the
    orphan check needs, once to compare. Hand it a generator, which is what
    `check_regression(evaluate(g, c) for c in corpora)` produces, and the second
    pass sees an exhausted iterator: every comparison is skipped, `problems`
    stays empty and the gate returns cleanly. A green check meaning the gate
    never ran, produced by the most natural way to call it.
    """
    regressed = [_ev(tp=0, fp=9, fn=9)]
    with pytest.raises(RegressionError, match="precision"):
        check_regression(iter(regressed), BASE)


def test_an_empty_iterator_is_a_gate_that_checked_nothing() -> None:
    """Emptiness has to be judged after the argument is drawn, not before.

    An exhausted iterator is truthy, so `if not evaluations` is False for the
    one shape that most needs it to be True.
    """
    nothing: list[Evaluation] = []
    with pytest.raises(RegressionError, match="checked nothing"):
        check_regression(iter(nothing), BASE)


def test_a_baseline_entry_that_is_not_an_object_fails_loudly() -> None:
    """baselines.json is hand-edited, and double-encoded JSON is a real shape.

    A string entry that happens to contain the word "precision" passes the
    `metric not in baseline` test by substring, then raises TypeError on
    subscript: a bare builtin escaping this seam, past the one net Task 13's
    CLI holds.
    """
    entries = {"pii/in-repo/abc123def456": '{"precision": 0.9, "recall": 0.9}'}
    with pytest.raises(RegressionError, match="is not an object"):
        check_regression([_ev()], cast(dict[str, dict[str, object]], entries))


def test_a_nan_baseline_is_named_for_its_shape_not_its_position() -> None:
    """NaN is refused twice over, and the order of the two guards decides the
    message. Every comparison against NaN is False, so the range check below
    would also refuse it, reporting a value "outside [0, 1]" for a value that is
    not on the line at all. The finite check runs first and says so."""
    with pytest.raises(RegressionError, match="not a finite number"):
        check_regression(
            [_ev()],
            {
                "pii/in-repo/abc123def456": {
                    "precision": float("nan"),
                    "recall": 0.9,
                    "decision_mismatches": 0,
                }
            },
        )


def test_a_null_baseline_entry_is_named_for_what_it_is() -> None:
    """A `null` entry is a present key holding no measurement, and the two
    readings of "absent" come apart on it. Read as `.get(key) is None` it reports
    that the corpus changed, a cause the gate has not established, while the
    orphan check two branches up has already counted the key as exercised."""
    entries: dict[str, dict[str, object]] = {
        "pii/in-repo/abc123def456": cast(dict[str, object], None)
    }
    with pytest.raises(RegressionError, match="is not an object"):
        check_regression([_ev()], entries)


def test_a_negative_baseline_score_is_refused() -> None:
    """A baseline below 0 is not a measurement, it is a disabled guard.

    Nothing else in this call can raise: the detector found nothing at all and
    still clears both recorded numbers.
    """
    with pytest.raises(RegressionError, match="not a possible score"):
        check_regression(
            [_ev(tp=0, fp=9, fn=9)],
            {
                "pii/in-repo/abc123def456": {
                    "precision": -1.0,
                    "recall": 0.0,
                    "decision_mismatches": 0,
                }
            },
        )


def test_a_baseline_score_above_one_is_refused() -> None:
    """The other end of the same argument. Unguarded it fails, but it blames the
    detector for a number no detector can reach, which sends the reader to the
    wrong file."""
    with pytest.raises(RegressionError, match="not a possible score"):
        check_regression(
            [_ev()],
            {
                "pii/in-repo/abc123def456": {
                    "precision": 1.5,
                    "recall": 0.9,
                    "decision_mismatches": 0,
                }
            },
        )


def test_a_negative_decision_baseline_is_refused() -> None:
    """Same argument, third field. A negative count is not reachable, and left
    unnamed it reports as a regression in the detector that no change to the
    detector can clear."""
    with pytest.raises(RegressionError, match="cannot be below zero"):
        check_regression(
            [_ev()],
            {
                "pii/in-repo/abc123def456": {
                    "precision": 0.9,
                    "recall": 0.9,
                    "decision_mismatches": -1,
                }
            },
        )


def test_an_epsilon_that_is_not_a_number_is_refused() -> None:
    """`math.isfinite("0.005")` raises TypeError, so the string form of a CI
    yaml value escapes as a builtin rather than as this module's error."""
    with pytest.raises(RegressionError, match="epsilon"):
        check_regression([_ev()], BASE, epsilon=cast(float, "0.005"))


def test_a_boolean_epsilon_is_refused() -> None:
    """The bool trap the brief closes on baseline values, one step over.

    YAML reads a bare `no` as False, which is 0, which is a valid tolerance, so
    the gate would run with a config value it never agreed to.
    """
    with pytest.raises(RegressionError, match="epsilon"):
        check_regression([_ev(tp=10, fp=0, fn=0)], BASE, epsilon=cast(float, False))


def test_a_nan_epsilon_is_named_for_its_shape_not_its_width() -> None:
    """The same reason as the NaN baseline above, and the reason the epsilon
    guard is three statements rather than one condition. The bounded range
    refuses NaN by accident, because `0.0 <= nan` is False, so a single `or`
    chain leaves the finite clause invisible to every test that could be
    written. Split, this pins it, and it names the trap in the message a CI log
    will carry."""
    with pytest.raises(RegressionError, match="every comparison against NaN is"):
        check_regression([_ev()], BASE, epsilon=float("nan"))


def test_a_boolean_decision_baseline_is_not_a_count() -> None:
    """The third bool trap, and the only one the brief's tests leave open.

    `test_a_non_integer_decision_baseline_fails_loudly` uses 1.5, which kills
    only the `not isinstance(allowed, int)` half of the clause. JSON `true` is an
    int by inheritance and reads as "one wrong decision allowed", so the
    evaluation below, which got exactly one wrong, passes against a baseline that
    records no number at all.
    """
    with pytest.raises(RegressionError, match="decision_mismatches is not an integer"):
        check_regression(
            [_ev(failures=_wrong(1))],
            {
                "pii/in-repo/abc123def456": {
                    "precision": 0.9,
                    "recall": 0.9,
                    "decision_mismatches": True,
                }
            },
        )


def test_a_decision_baseline_above_the_case_count_is_refused() -> None:
    """The upper half of the bound the ratios already carry on both sides.

    `Evaluation` refuses a measured count above `cases`, so a baseline above it
    is not a number this harness produced. It disables the one quantity the
    brief calls the one a guardrail user cares about most.

    21 against 20 cases, which is AT the boundary. An earlier version of this
    test probed 200, and every loosened bound passed it: `> cases * 5` and
    `> cases + 100` both refuse 200 with the same message while admitting 21,
    and a baseline of 21 retires the gate outright, since the measured count can
    never exceed 20. That is the argument
    `test_a_drop_just_outside_epsilon_is_refused` makes about epsilon, and it
    holds here for the same reason.
    """
    with pytest.raises(RegressionError, match="exceeds the 20 case"):
        check_regression(
            [_ev(failures=_wrong(1))],
            {
                "pii/in-repo/abc123def456": {
                    "precision": 0.9,
                    "recall": 0.9,
                    "decision_mismatches": 21,
                }
            },
        )


def test_a_decision_baseline_of_exactly_the_case_count_is_a_measurement() -> None:
    """The other end of that bound, and the fourth-boundary argument the ratios
    already carry. A corpus whose every case got the wrong decision is a real,
    terrible measurement, so `cases` is the largest baseline the gate accepts
    rather than the first it refuses. Nothing else here can raise: the ratios sit
    on their baseline and the measured count is within the recorded one."""
    check_regression(
        [_ev(failures=_wrong(20), cases=20)],
        {
            "pii/in-repo/abc123def456": {
                "precision": 0.9,
                "recall": 0.9,
                "decision_mismatches": 20,
            }
        },
    )


def test_the_widest_accepted_tolerance_is_five_hundredths() -> None:
    """MAX_EPSILON's VALUE, pinned with the literal.

    `test_an_epsilon_of_exactly_the_maximum_is_accepted` reads the constant
    symbolically, so it passes at any value the constant takes, and the only
    other width probe uses 1.0. Set to 0.4 the constant survives this whole
    file while admitting a tolerance that passes a drop from 0.900 to 0.550.
    A test that reads a constant symbolically can never pin that constant.
    """
    assert MAX_EPSILON == 0.05
    with pytest.raises(RegressionError, match="epsilon must be within"):
        check_regression([_ev()], BASE, epsilon=0.051)


def test_an_over_long_integer_epsilon_leaves_as_this_modules_error() -> None:
    """`math.isfinite(10**400)` raises OverflowError rather than answering, and
    an over-long literal is what `json.loads` builds from a corrupted config. It
    is loud either way, so this is about the promise in the module docstring:
    nothing leaves here as a bare builtin. The value is an ordinary integer for
    the bound that follows, which refuses it by name."""
    with pytest.raises(RegressionError, match="epsilon must be within"):
        check_regression([_ev()], BASE, epsilon=10**400)


def test_an_over_long_integer_baseline_leaves_as_this_modules_error() -> None:
    """The same call one field over. `json.loads` produces a Python int of any
    length, so a corrupted baselines file reaches the finite check as a value it
    cannot convert. The range check names it accurately: it is not NaN, it is
    outside [0, 1]."""
    with pytest.raises(RegressionError, match="not a possible score"):
        check_regression(
            [_ev()],
            {
                "pii/in-repo/abc123def456": {
                    "precision": 10**400,
                    "recall": 0.9,
                    "decision_mismatches": 0,
                }
            },
        )


def test_an_epsilon_of_zero_is_the_strictest_setting_not_a_refused_one() -> None:
    """0 is a tolerance a user chooses deliberately: no slack at all. The bound
    is inclusive at both ends and nothing else pins that."""
    check_regression([_ev()], BASE, epsilon=0.0)


def test_an_epsilon_of_exactly_the_maximum_is_accepted() -> None:
    """The other end of the same bound. MAX_EPSILON is the widest tolerance the
    gate accepts, not the first one it refuses."""
    check_regression([_ev()], BASE, epsilon=MAX_EPSILON)


def test_a_baseline_of_exactly_one_is_a_measurement() -> None:
    """1.000 is the score every published table hopes for, and the range check
    has to admit it. Nothing else in this call can raise."""
    check_regression(
        [_ev(tp=10, fp=0, fn=0)],
        {
            "pii/in-repo/abc123def456": {
                "precision": 1.0,
                "recall": 1.0,
                "decision_mismatches": 0,
            }
        },
    )


def test_a_baseline_of_exactly_zero_is_a_measurement() -> None:
    """And so is the score a detector that finds nothing earns. A recorded 0.0
    is a poor baseline, not an impossible one."""
    check_regression(
        [_ev(tp=0, fp=9, fn=9)],
        {
            "pii/in-repo/abc123def456": {
                "precision": 0.0,
                "recall": 0.0,
                "decision_mismatches": 0,
            }
        },
    )


def test_a_drop_just_outside_epsilon_is_refused() -> None:
    """Pins the width of the tolerance, not merely its existence.

    precision is 0.8900 against a baseline of 0.9000, a drop of 0.01, twice
    epsilon. Every other failing test in this file drops by 0.4 or more, so an
    epsilon inflated by any constant below 0.4 passes all of them.
    """
    with pytest.raises(RegressionError, match="precision 0.8900 below baseline 0.9000"):
        check_regression([_ev(tp=89, fp=11, fn=10, cases=200)], BASE)


def test_each_ratio_is_compared_against_its_own_baseline() -> None:
    """precision and recall differ here, and both baselines are 0.9, so an
    implementation that read one ratio twice, or crossed the two, still reports
    a regression naming both metrics. Only the rendered numbers tell them
    apart."""
    with pytest.raises(RegressionError) as exc:
        check_regression([_ev(tp=5, fp=1, fn=5)], BASE)
    assert "precision 0.8333 below baseline 0.9000" in str(exc.value)
    assert "recall 0.5000 below baseline 0.9000" in str(exc.value)
