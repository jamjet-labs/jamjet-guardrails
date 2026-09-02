"""Train and dev, divided by TWIN and never by row.

The corpus is generated in matched twins: one call produces a hard negative and
an attack that share an opening, a length and a register, and the two rows land
next to each other in `rows.jsonl`. That pairing is what stopped the classes
being separable by style, and it creates a hazard of its own one step later.

**A row-wise split leaks.** The two members of a twin are alike on purpose, and
in one pair they were measured alike to a median word-trigram Jaccard of 0.538.
Split the rows at random and one member lands in train and the other in dev:
the model has already seen most of the held-out text, under the opposite label.
What it learns from that is the local difference between two near-copies, which
generalises nowhere and scores well on the split that created it. The ship bar
compares our fine-tuned encoder against two reference models that were never
fitted on any of this, so a leak of that shape lifts our number alone, which is
the one failure the bar exists to rule out.

So the unit of the split is the twin. `split` assigns whole twins, and
`separated_twins` is the check that says whether any partition, however it was
built, broke one apart. Stage 2b-2 builds the split; this module is what it has
to build it through, and `tests/test_training_data.py` fails if a twin is
separated.

Nothing here is a substitute for an independent evaluation set, and nothing
here is one. Every row on both sides of this split came out of one generator
under one of eight prompts, so a number measured on the held-out half is
optimistic by an unknown margin no matter how the halves were chosen. Splitting
by twin removes one leak; it does not make a corpus its own benchmark. The
evaluation set is an external public corpus, read by `training/evalset.py`, and
the two sides here are TRAIN and DEV.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: How many rows one generated twin holds. One hard negative and one attack,
#: adjacent, sharing a prompt and a seed.
TWIN = 2

#: The share of twins held out by default. Twins, not rows: because a twin
#: contributes one row to each class, holding out a share of the twins holds out
#: the same share of each class, and the held-out half is balanced by
#: construction rather than by stratifying afterwards.
#:
#: Named for what it holds out and not for what the held-out rows are for. They
#: are the DEV set: the EVALUATION set is external, and `Split.held_out` says
#: why that distinction is carried in the names rather than in a comment.
HELD_OUT_SHARE = 0.2


class SplitError(ValueError):
    """The rows are not the twin-structured corpus this module can divide."""


def shuffled(count: int, seed: int) -> list[int]:
    """`range(count)` in a seeded order that does not depend on the interpreter.

    A seeded linear congruential shuffle rather than `random.shuffle`, for the
    reason `split` records: a split has to be reproducible from the seed alone,
    and `random` is documented to produce the same stream for a seed but its
    `shuffle` has changed algorithm before now. A committed split that a later
    Python re-derives differently is a split nobody can check.

    Public, and used by `training/cluster.py` as well as by `split` below.
    Written once for the reason two copies of a rule are always written once
    here: the cluster-wise split and the twin-wise split have to be the same
    shuffle, and two implementations of a Fisher-Yates loop drift by one
    character while both sides go on looking right.
    """
    order = list(range(count))
    state = seed
    for i in range(len(order) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (1 << 31)
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


@dataclass(frozen=True, slots=True)
class Split:
    """Row indices on each side of the line, by twin.

    The second side is `held_out`, and the name is the point. It was called
    `evaluation` while the plan was to score the classifier on a held-out slice
    of the synthetic corpus, and that plan was abandoned in this same stage:
    the corpus is separable by register, so it cannot grade itself, and the
    EVALUATION set is now an external public corpus read by
    `training/evalset.py`. A field still called `evaluation` would invite the
    one reading this module exists to prevent: somebody measuring the ship bar
    on these rows and publishing the number.

    These rows are the DEV set. They choose a checkpoint and a threshold, and
    nothing measured on them gets published.
    """

    train: tuple[int, ...]
    held_out: tuple[int, ...]


def twins(labels: Sequence[int], keys: Sequence[object]) -> list[tuple[int, int]]:
    """Adjacent (negative, attack) index pairs, checked rather than assumed.

    `keys` is whatever identifies the call a row came from -- in practice
    `(prompt_id, seed)`. Both members of a twin carry the same one, so a corpus
    that has been re-ordered, filtered or appended to since it was generated
    fails here instead of being split into halves that are not twins at all.

    Raising rather than repairing. A corpus this function cannot read is a
    corpus somebody changed outside the generator, and guessing which rows were
    meant to be together would be the wrong kind of helpful.
    """
    if len(labels) != len(keys):
        raise SplitError(f"{len(labels)} labels against {len(keys)} keys")
    if len(labels) % TWIN:
        raise SplitError(f"{len(labels)} rows is not a whole number of twins")
    out: list[tuple[int, int]] = []
    for start in range(0, len(labels), TWIN):
        left, right = start, start + 1
        if (labels[left], labels[right]) != (0, 1):
            raise SplitError(
                f"rows {left} and {right} are labelled {labels[left]} and {labels[right]}, "
                "so they are not a (negative, attack) twin"
            )
        if keys[left] != keys[right]:
            raise SplitError(
                f"rows {left} and {right} came from {keys[left]!r} and {keys[right]!r}, "
                "so they are two halves of two different calls"
            )
        out.append((left, right))
    return out


def split(
    labels: Sequence[int],
    keys: Sequence[object],
    held_out_share: float = HELD_OUT_SHARE,
    seed: int = 42,
) -> Split:
    """Hold out `held_out_share` of the TWINS, both members of each, together.

    The shuffle is the same seeded linear congruential one
    `training/separability.py` folds with, so a split is reproducible without
    depending on what `random` does in a later interpreter.
    """
    if not 0.0 < held_out_share < 1.0:
        raise SplitError(f"held_out_share {held_out_share} is not a share between 0 and 1")
    found = twins(labels, keys)
    order = shuffled(len(found), seed)
    held = max(1, round(len(found) * held_out_share))
    kept = sorted(index for position in order[:held] for index in found[position])
    train = sorted(index for position in order[held:] for index in found[position])
    return Split(tuple(train), tuple(kept))


def separated_twins(labels: Sequence[int], keys: Sequence[object], made: Split) -> list[int]:
    """Twins with one member on each side of the line. The enforced rule.

    Written to score ANY partition, not only one this module made, because that
    is what it is for: a later task builds the split, and this is the check that
    fails when it builds it by row. It also catches a row that landed on both
    sides or on neither, which is the same defect wearing different clothes.

    Returns the index of the first row of each broken twin, so a failure names
    rows somebody can go and read.
    """
    on_left = set(made.train)
    on_right = set(made.held_out)
    # A side naming one row twice is not a partition, and a set would swallow
    # the second naming silently. `split` cannot produce this; the caller this
    # function exists to check can, and did in review.
    if len(on_left) != len(made.train) or len(on_right) != len(made.held_out):
        raise SplitError("a side of the split names the same row more than once")
    broken: list[int] = []
    for left, right in twins(labels, keys):
        sides = [(index in on_left, index in on_right) for index in (left, right)]
        placed_once = [side for side in sides if side in ((True, False), (False, True))]
        if len(placed_once) != TWIN or sides[0] != sides[1]:
            broken.append(left)
    return broken
