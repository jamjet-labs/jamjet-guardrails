"""What a set of predictions scored, and where a threshold puts the line.

Separated from `training/train.py` for one reason, and it is not tidiness.
That module imports torch, which is deliberately absent from the package's
`.venv` and from CI, so nothing in it can be tested by the suite that tests
everything else. These four functions decide every number the stage publishes
about our own model -- the confusion counts every rate is re-derived from, and
the threshold a later task inherits -- and a rule nothing exercises is a rule
nobody has checked.

Nothing here imports anything but the standard library. `train.py` cross-checks
`scored` against scikit-learn on every evaluation, so the arithmetic has a
second opinion where scikit-learn is installed and a test where it is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: The label a positive is. One place, because a metric computed against the
#: other one is a metric that reads plausibly and measures the complement.
POSITIVE = 1

#: The other label. Derived rather than written as 0, so `POSITIVE` really is
#: the only place the polarity is decided.
NEGATIVE = 1 - POSITIVE


class ScoringError(ValueError):
    """The predictions cannot be scored against these labels."""


@dataclass(frozen=True, slots=True)
class Scored:
    """One evaluation over one set, with the counts the rates came from kept.

    The counts are carried rather than the rates alone. A rate restated
    elsewhere drifts and both copies go on looking right; a rate that can be
    re-derived from four integers cannot, and `tests/test_training_data.py`
    re-derives every one of them from the committed run record.
    """

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    def as_record(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


def counts(actual: Sequence[int], predicted: Sequence[int]) -> tuple[int, int, int, int]:
    """The confusion counts, as (tp, fp, fn, tn) against `POSITIVE`.

    Raises on a length mismatch rather than stopping at the shorter one. `zip`
    truncates, and a metric computed over the first half of a set is a metric
    that looks like a metric.
    """
    if len(actual) != len(predicted):
        raise ScoringError(f"{len(actual)} labels against {len(predicted)} predictions")
    tp = sum(1 for a, p in zip(actual, predicted) if a == POSITIVE and p == POSITIVE)
    fp = sum(1 for a, p in zip(actual, predicted) if a != POSITIVE and p == POSITIVE)
    fn = sum(1 for a, p in zip(actual, predicted) if a == POSITIVE and p != POSITIVE)
    tn = sum(1 for a, p in zip(actual, predicted) if a != POSITIVE and p != POSITIVE)
    return tp, fp, fn, tn


def scored(actual: Sequence[int], predicted: Sequence[int]) -> Scored:
    """Precision, recall, F1 and accuracy from the counts.

    Zero rather than an error or a NaN when a denominator is empty. A NaN
    compares False against every threshold, so it reads as "did not clear" for
    a model that flagged nothing at all and for a model that was never scored,
    which is one answer for two situations. `training/ship_bar.py` made the
    same choice for the same reason.
    """
    tp, fp, fn, tn = counts(actual, predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    total = tp + fp + fn + tn
    return Scored(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        accuracy=(tp + tn) / total if total else 0.0,
    )


def at(chance: Sequence[float], threshold: float) -> list[int]:
    """The decisions a threshold makes.

    `>=`, and written down because the boundary is where a rule fails quietly:
    a row the model put exactly on the threshold is decided by this comparison
    and by nothing else, and `>` and `>=` differ on exactly that row. The sweep
    below walks the observed probabilities themselves, so the threshold it
    returns always has a row sitting on it.
    """
    return [POSITIVE if value >= threshold else NEGATIVE for value in chance]


def sweep(chance: Sequence[float], actual: Sequence[int]) -> tuple[float, Scored]:
    """The threshold with the best F1 on this set, and what it scores there.

    Swept over the probabilities the model actually produced rather than over a
    round-number grid, so the chosen threshold sits on a real boundary between
    two rows instead of in whatever gap a grid happened to leave.

    Ties go to the LOWER threshold, which is the recall-favouring side, and
    that is the side an injection detector should lose to: a missed injection
    is the failure the check exists to prevent and a false positive is an
    inconvenience. 0.5 is the starting candidate because it is the argmax the
    epoch metrics use, so a sweep that finds nothing better returns the rule
    that was already being reported rather than an arbitrary neighbour of it.

    Raises on an empty input. Returning some default would answer "0.5 is best"
    for a set nothing was measured on, which is the answer that means "fine".
    """
    if not chance:
        raise ScoringError("no probabilities to sweep, so every threshold scores the same")
    best_threshold, best = 0.5, scored(actual, at(chance, 0.5))
    for threshold in sorted(set(chance)):
        here = scored(actual, at(chance, threshold))
        if here.f1 > best.f1 or (here.f1 == best.f1 and threshold < best_threshold):
            best_threshold, best = threshold, here
    return best_threshold, best
