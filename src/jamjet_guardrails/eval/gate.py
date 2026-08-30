"""Refuse to merge a change that makes a published number worse.

Everything upstream of here measures and formats. This module decides, and it
is the only one that does. The README's claim is not "we publish precision and
recall", it is "CI refuses a change that makes them worse", so a gate that
passes when it should fail does not weaken the claim, it converts the whole
harness into decoration. Silently, which is the failure mode this project has a
phrase for: a green check that means the review never ran.

That shapes every branch here. A gate has one dangerous direction and it is
always the same one, because absent, empty, unknown, unmatched and ambiguous
all resolve to "nothing to report" unless something makes them resolve
otherwise. So the question asked of each branch below is not "is this correct"
but "what reaches the end of this function without raising, and is passing the
right answer for every one of those".

Four rules do the work.

**Three quantities are gated, not two.** Precision and recall count FINDINGS. A
guardrail that locates every span and then returns the wrong decision scores
1.000 on both, and a corpus of clean negatives scores 1.000 for a detector that
denies everything. ``Evaluation.decision_mismatches`` is the independent signal
and the one a guardrail user actually cares about: did it block the bad thing.
It is an integer, so it is gated without an epsilon. One more wrong decision
than last time is a regression.

**Silence is never a pass.** Not an empty evaluation list, not an evaluation
whose key has no baseline, not a baseline whose corpus has gone, not a baseline
field that is missing. Each of those is the state a broken discovery glob, a
renamed corpus or a hand-edited JSON file produces, and each of them makes
every comparison below run zero times.

**A tolerance is bounded on both sides, and so is a baseline.** Every unchecked
value of either points the same way: a wider tolerance and a lower baseline
both mean greener CI, and neither looks wrong in a config file. The values that
are not measurements at all (NaN, a bool, a string, a score outside [0, 1], a
negative count) are refused by name rather than compared.

**Nothing that comes out of a JSON file is trusted to be the shape it should
be.** ``baselines`` is hand-edited, and the values in it reach comparisons and
format strings. A malformed entry leaves here as ``RegressionError``, the one
net Task 13's CLI holds, and never as a bare builtin.

``cases`` is deliberately not gated. The baseline key carries the corpus
version, which is a digest of every case's content, so the case count cannot
change without producing a different key and failing on the missing-baseline
branch first. A required ``cases`` baseline field would guard a state that
cannot be reached.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from jamjet_guardrails.eval.metrics import Evaluation

MAX_EPSILON = 0.05
"""Widest tolerance the gate will accept. Past this it stops being slack for
floating-point noise and becomes permission to regress."""


class RegressionError(Exception):
    """A published score regressed, or a corpus has no recorded baseline.

    Also every malformed input this module refuses. One error type covers the
    lot for the reason ``CorpusError`` gives two modules over: a caller nets one
    name, not a builtin it has to know to expect. A ``TypeError`` out of a
    subscript here would surface through Task 13's CLI as a traceback.
    """


def _is_nan_or_infinite(value: float) -> bool:
    """Whether a number is NaN or an infinity, with the OverflowError closed.

    ``math.isfinite`` raises rather than answering for an int too large to
    convert to a float, and ``json.loads`` builds exactly that from an over-long
    literal in a config or a baselines file. Both callers below read JSON, so
    both can reach it, and a bare ``OverflowError`` would leave past the one net
    this module promises.

    The promise is scoped to values a JSON document can carry, which is what
    both callers read. An int of more than 4300 digits also raises ValueError
    from ``repr`` under CPython's integer string conversion limit, and this
    helper does not close that: ``json.loads`` refuses to parse such a literal
    in the first place, so no baselines file can deliver one.

    It answers False for such a value, which is the accurate answer: an
    enormous integer is not NaN and not an infinity, it is an ordinary number
    that no bound here admits. Both callers then refuse it by the bound, whose
    comparison is exact for ints and needs no conversion, and the message names
    what is actually wrong with it rather than calling it non-finite.
    """
    try:
        return not math.isfinite(value)
    except OverflowError:
        return False


def baseline_key(ev: Evaluation) -> str:
    return f"{ev.detector}/{ev.corpus_source}/{ev.corpus_version}"


def check_regression(
    evaluations: Iterable[Evaluation],
    baselines: Mapping[str, Mapping[str, object]],
    epsilon: float = 0.005,
) -> None:
    """Raise unless every published number is at least as good as its baseline.

    Args:
        evaluations: what this run measured. Every entry needs a baseline and
            every baseline needs an entry.
        baselines: recorded scores, keyed by ``baseline_key``. Read as
            untrusted: it is JSON a person edits.
        epsilon: tolerance on the two ratios, for floating-point noise only.
            Bounded by ``MAX_EPSILON``.

    Raises:
        RegressionError: on a score below its baseline, on more wrong decisions
            than the baseline records, on a missing or orphaned baseline, on a
            baseline value that is not a possible measurement, on an
            unusable epsilon, and on an empty set of evaluations. Every problem
            found is named in one message rather than only the first, because a
            gate that reports one failure per run costs one CI cycle per
            failure.
    """
    problems: list[str] = []

    # Every unchecked value of epsilon points the same way: wider tolerance,
    # greener CI. Three shapes, three refusals, kept apart because they are
    # three different mistakes and because one condition spanning all three
    # cannot be pinned clause by clause: the bounded range below refuses NaN on
    # its own, since `0.0 <= nan` is False like every other NaN comparison, so
    # written as one `or` chain the finite check is a clause no test can tell
    # from its absence.
    if isinstance(epsilon, bool) or not isinstance(epsilon, int | float):
        # bool first, because True == 1 and False == 0: a YAML `epsilon: no`
        # reads as a valid tolerance of zero, so the gate would run on a value
        # nobody wrote. A string would reach the finite check below and leave
        # as a TypeError, loud but untyped, past the one net this module
        # promises.
        raise RegressionError(f"epsilon must be a real number, got {epsilon!r}")
    if _is_nan_or_infinite(epsilon):
        # The sharpest of the three, because it does not look wrong. Every `<`
        # against NaN is False, so a NaN tolerance passes every regression while
        # reading as a number in a CI config nobody re-reads.
        raise RegressionError(
            f"epsilon must be finite, got {epsilon!r}: every comparison against NaN is "
            "False, so a NaN tolerance passes every regression it is asked about"
        )
    if not 0.0 <= epsilon <= MAX_EPSILON:
        raise RegressionError(
            f"epsilon must be within [0, {MAX_EPSILON}], got {epsilon!r}. A wider "
            "tolerance does not make the gate more forgiving, it makes it decorative."
        )

    # Drawn once, before anything asks whether it is empty. The two passes below
    # (keys for the orphan check, then comparisons) both consume this argument,
    # and a one-shot iterable is the natural way to call this:
    # `check_regression(evaluate(g, c) for c in corpora)`. Left unmaterialised,
    # the first pass exhausts it, the second compares nothing, and the gate
    # returns cleanly. An exhausted iterator is also truthy, so the emptiness
    # check below would agree that there was work to do.
    items = list(evaluations)

    if not items:
        # A gate that checked nothing must never report success. If corpora are
        # renamed, moved, or the discovery glob drifts, every loop below runs
        # zero times and this function would return cleanly: a green CI check
        # meaning "the gate never ran", which is worse than no gate at all.
        raise RegressionError(
            "no evaluations to gate: discovery found no corpora. A gate that "
            "checked nothing does not pass."
        )

    # Every recorded baseline must be exercised. Without this, deleting or
    # renaming a corpus silently retires its guard while the gate stays green:
    # the orphaned baseline is simply never looked up.
    evaluated = {baseline_key(ev) for ev in items}
    orphaned = sorted(set(baselines) - evaluated)
    if orphaned:
        problems.append(
            f"baselines with no corresponding corpus: {orphaned}. The corpus was "
            "deleted, renamed, or its content changed. Remove the baseline "
            "deliberately or restore the corpus."
        )

    for ev in items:
        key = baseline_key(ev)
        if key not in baselines:
            # Membership, not `.get(key) is None`, because those differ on a
            # hand-edited `null` entry: the key IS present, the orphan check
            # above has already counted it as exercised, and this message would
            # name a cause it has not established.
            problems.append(
                f"{key}: no baseline recorded. The corpus version is derived from its "
                "content, so this means the corpus changed. Review the new numbers and "
                "add the baseline in the same commit."
            )
            continue
        baseline = baselines[key]
        if not isinstance(baseline, Mapping):
            # Double-encoded JSON is the shape that makes this worth a branch: a
            # string entry answers `in` by SUBSTRING, so it passes the
            # missing-metric check below and then raises on subscript.
            problems.append(
                f"{key}: baseline entry is not an object: {baseline!r}. Every entry is "
                "a mapping of metric name to recorded score."
            )
            continue
        for metric, actual in (
            ("precision", ev.overall.precision),
            ("recall", ev.overall.recall),
        ):
            # The baseline file is JSON someone hand-edited. A metric it does not
            # record, or records as a string, must not reach the comparison: a
            # bool is the worst case, because True == 1 compares as a number and
            # reads as a perfect baseline.
            if metric not in baseline:
                problems.append(f"{key}: baseline records no {metric}")
                continue
            expected = baseline[metric]
            if (
                isinstance(expected, bool)
                or not isinstance(expected, int | float)
                or _is_nan_or_infinite(expected)
            ):
                problems.append(f"{key}: baseline {metric} is not a finite number: {expected!r}")
                continue
            if not 0.0 <= expected <= 1.0:
                # The same argument as the epsilon bound, one field over. A
                # ratio outside [0, 1] is not a score this harness can ever have
                # produced, so it is a typo or a deliberate disabling, and below
                # zero it passes every detector including one that finds
                # nothing. Above one it does the opposite and fails forever,
                # naming the detector for a number no detector can reach, which
                # sends the reader to the wrong file.
                problems.append(
                    f"{key}: baseline {metric} is not a possible score: {expected!r} is "
                    "outside [0, 1]"
                )
                continue
            if actual < expected - epsilon:
                problems.append(f"{key}: {metric} {actual:.4f} below baseline {expected:.4f}")

        # Both ratios count findings, so neither can see a guardrail that
        # located every span and then returned the wrong decision. An integer,
        # so no epsilon: one more wrong decision than last time is a regression.
        # Comparable across runs because the baseline key pins the corpus
        # version, which is content-derived, so the case count is fixed.
        if "decision_mismatches" not in baseline:
            problems.append(f"{key}: baseline records no decision_mismatches")
        else:
            allowed = baseline["decision_mismatches"]
            if isinstance(allowed, bool) or not isinstance(allowed, int):
                problems.append(
                    f"{key}: baseline decision_mismatches is not an integer: {allowed!r}"
                )
            elif allowed < 0:
                # Third field, same argument as the two ratios above. A negative
                # count is not reachable: `decision_mismatches` counts failures
                # and starts at zero. Left unnamed it does raise, since any real
                # count exceeds it, but it raises against the detector for the
                # life of the file and no change to the detector can clear it.
                problems.append(
                    f"{key}: baseline decision_mismatches cannot be below zero: {allowed!r}"
                )
            elif allowed > ev.cases:
                # The upper half of the bound the two ratios carry on both
                # sides. Evaluation refuses a MEASURED count above `cases`,
                # since a case can disagree about its decision at most once, so
                # a baseline above it is not a number this harness produced. It
                # points the disabling way, and a typo'd digit reaches it: 200
                # against a 20-case corpus reads as an ordinary recorded value
                # and retires the gate on the quantity that matters most.
                problems.append(
                    f"{key}: baseline decision_mismatches {allowed} exceeds the "
                    f"{ev.cases} case(s) scored; a case can disagree about its decision "
                    "at most once, so this is not a count this corpus can produce"
                )
            elif ev.decision_mismatches > allowed:
                problems.append(
                    f"{key}: decision_mismatches {ev.decision_mismatches} above baseline {allowed}"
                )

    if problems:
        raise RegressionError("; ".join(problems))
