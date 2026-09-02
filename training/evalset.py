"""The external evaluation corpus, and the check that it is external.

**The synthetic corpus TRAINS. A public corpus EVALUATES.** This module is the
second half of that sentence, and the reasoning behind it is the reason the
manifest carries a `role: eval` on a corpus a reference model was trained on,
which on its face is the exact mistake every screen in this tree exists to
prevent.

**Why the held-out synthetic rows are not an evaluation set.** The generated
corpus is separable by register. A classifier that reads no content word at all
scores 0.848 within a pair against 0.907 for a full bag of words, measured by
`training/separability.py` over the committed rows. That is structural and not a
wording bug: one generator writing both labels produces register that correlates
with the label. Making the twins more alike creates near-duplicate leakage
across any split; making them less alike raises the register signal. There is no
wording that escapes both. So a fine-tuned encoder scored on a held-out slice of
that corpus could take its number from an artifact the reference models have
never seen, and the ship bar would be measuring artifact exploitation rather
than detection. That is the one failure the bar exists to rule out, so the
evaluation set has to come from outside.

**Why a contaminated corpus is the right one to come from outside.**
`jackhhao/jailbreak-classification` is named on ProtectAI's v2 card as its own
training data. Everywhere else in this tree that is a disqualification, and here
it is not, because of which way the bias runs. Contamination in the EVALUATION
set biases towards the REFERENCE model: DeBERTa may have memorised these rows,
and our encoder has never seen one of them. So the gate can fail us unfairly and
cannot pass us unfairly, and that asymmetry is the correct one for a ship bar. A
win measured here is meaningful; a loss is inconclusive, and has to be reported
as inconclusive rather than as a verdict.

Two caveats, stated here rather than left in a report nobody re-reads:

- This corpus is JAILBREAK classification. Jailbreak is adjacent to prompt
  injection and is not the same task: a jailbreak talks the model out of its
  own policy, an injection talks it out of its caller's instructions, and the
  two overlap without coinciding. The corpus also sits inside the reference
  model's own training distribution, which makes it maximally favourable to the
  reference on both counts at once.
- The external evaluation rests on ONE corpus. `training/sources.yaml` records
  the further candidates that were screened and what refused each. A single
  source is a real limit on what any number measured here supports, and the
  README says so in the same words.

The contamination check below is the other half of the argument. It is only
honest to call this corpus external if our model has not seen it, so `compare`
reads CONTENT and not identifiers: the eval rows against the synthetic corpus
and against every source the manifest admits for training, exact and near
duplicate. An identifier check alone would pass a corpus that had been
republished under another name, which is most of how public corpora travel.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: The manifest entry this module reads. Named rather than inferred, so the
#: identifier screen over this tree accounts for it and so that changing the
#: evaluation corpus is an edit somebody makes on purpose.
EVAL_SOURCE = "jackhhao/jailbreak-classification"

#: The corpus's own columns, and the label vocabulary it uses.
#:
#: A CLOSED map, and `load_eval` raises on a value outside it. A permissive
#: reader that treated an unknown label as benign, or skipped the row, would
#: change the class balance of the evaluation set without saying so, and every
#: precision figure measured afterwards would be about a corpus nobody
#: described.
TEXT_COLUMN = "prompt"
LABEL_COLUMN = "type"
LABELS: dict[str, int] = {"benign": 0, "jailbreak": 1}

#: Word-trigram Jaccard at which two rows count as the same row.
#:
#: The value `training.generate.NearDuplicateIndex` deduplicated the synthetic
#: corpus at, and it is the same question asked across two corpora rather than
#: within one, so it is the same number.
#: `test_the_contamination_threshold_is_the_one_the_generator_deduplicated_at`
#: holds the two equal, because a constant restated in a second module is a
#: constant that drifts while both sides go on looking right.
NEAR_DUPLICATE = 0.6

#: Rows this reader will accept from one CSV cell. `csv` caps a field at 128 KiB
#: by default and raises past it, and this corpus's longest prompt is 26 KiB, so
#: the cap is raised once, here, rather than reached in a later run over a
#: longer corpus and read as a corrupt download. Not `sys.maxsize`: the
#: setter takes a C long and the largest one is a build-dependent
#: OverflowError waiting to happen, where a bound above any corpus this tree
#: reads is all the rule needs.
_FIELD_LIMIT = 4 * 1024 * 1024

_WORDS = re.compile(r"[^a-z0-9 ]+")


class EvalError(RuntimeError):
    """The evaluation corpus is not the shape this module can read."""


@dataclass(frozen=True, slots=True)
class EvalRow:
    """One row of the external evaluation corpus."""

    text: str
    label: int


def load_eval(path: Path) -> list[EvalRow]:
    """The pinned corpus, as rows, refusing anything it cannot account for.

    Strict in three places, and each one is a way a quiet reader would change
    what the evaluation set is: a missing column, a label outside `LABELS`, and
    an empty text. A row dropped for any of those without a word said moves the
    class balance, and the balance is what a precision number is relative to.
    """
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, _FIELD_LIMIT))
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            missing = [name for name in (TEXT_COLUMN, LABEL_COLUMN) if name not in columns]
            if missing:
                raise EvalError(f"{path.name} carries columns {columns}, missing {missing}")
            rows: list[EvalRow] = []
            for number, record in enumerate(reader, start=2):
                raw = (record.get(LABEL_COLUMN) or "").strip()
                if raw not in LABELS:
                    raise EvalError(
                        f"{path.name} line {number} is labelled {raw!r}, which is not one of "
                        f"{sorted(LABELS)}; a reader that guessed would change the class "
                        "balance every precision figure here is relative to"
                    )
                text = (record.get(TEXT_COLUMN) or "").strip()
                if not text:
                    raise EvalError(f"{path.name} line {number} carries no {TEXT_COLUMN}")
                rows.append(EvalRow(text=text, label=LABELS[raw]))
    finally:
        csv.field_size_limit(previous)
    if not rows:
        raise EvalError(f"{path.name} holds no rows")
    return rows


def balance(rows: Iterable[EvalRow]) -> dict[int, int]:
    """How many rows of each label, which is not something to assume.

    This corpus is 2:1 benign to jailbreak, not 1:1 like the generated one. A
    harness that assumed balance would report an accuracy figure whose baseline
    is 0.667 as though its baseline were 0.5.
    """
    counted: dict[int, int] = {label: 0 for label in sorted(set(LABELS.values()))}
    for row in rows:
        counted[row.label] = counted.get(row.label, 0) + 1
    return counted


def normalised(text: str) -> str:
    """A row as an exact-duplicate check compares it: case and punctuation gone.

    Two corpora that quote the same attack rarely quote it byte for byte. One
    capitalises the first word, one keeps a trailing newline, one had its
    smart quotes flattened in transit. A byte comparison calls all three
    different rows, which is the answer that means "not contaminated".
    """
    return " ".join(_WORDS.sub(" ", text.casefold()).split())


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    """Word trigrams of a row, punctuation and case removed.

    The same shingles `training.generate.NearDuplicateIndex` compares rows with
    and `training.separability` measures twins with, written out again here for
    the reason that module records: this is a screen over a corpus this tree did
    not generate, and importing the generator would make the screen depend on
    the thing it is screening against.
    """
    words = normalised(text).split()
    if len(words) < 3:
        return frozenset({(word,) for word in words})
    return frozenset(tuple(words[i : i + 3]) for i in range(len(words) - 2))


@dataclass(frozen=True, slots=True)
class Contamination:
    """What an evaluation corpus shares with the data a model was fitted on.

    Indices and counts, never the rows themselves. A record that quoted the
    overlapping text would put the training corpus into a committed artifact,
    which is a licence question this repository has already had to answer once.
    """

    #: Eval rows whose normalised text is a training row's, exactly.
    exact: tuple[int, ...]
    #: Eval rows within `threshold` word-trigram Jaccard of a training row.
    near: tuple[int, ...]
    #: The highest similarity any eval row reached against any training row,
    #: recorded whether or not it crossed. A run reporting 0 overlaps at a max
    #: of 0.59 is a different fact from one reporting 0 at a max of 0.04, and a
    #: record that keeps only the count cannot tell them apart.
    max_similarity: float
    threshold: float
    eval_rows: int
    train_rows: int

    @property
    def clean(self) -> bool:
        return not self.exact and not self.near


def compare(
    eval_texts: Sequence[str],
    train_texts: Sequence[str],
    threshold: float = NEAR_DUPLICATE,
) -> Contamination:
    """Every eval row that is also a training row, by content.

    Inverted on trigrams rather than compared against everything: only rows
    sharing a trigram can possibly be close, and the naive form is tens of
    millions of set intersections.

    The exact arm is not the near arm at a threshold of 1.0. Two rows can be
    normalised-identical and score below 1.0 here when one of them is under
    three words, because `shingles` falls back to single words for a short text
    and the two shingle sets then differ in kind. A short attack string
    reappearing verbatim is exactly the contamination worth catching, so it is
    caught by its own comparison rather than by a threshold that happens to be
    high.
    """
    if not 0.0 < threshold <= 1.0:
        raise EvalError(f"threshold {threshold} is not a Jaccard threshold")
    train_normal = {normalised(text) for text in train_texts}
    train_shingles = [shingles(text) for text in train_texts]
    index: dict[tuple[str, ...], list[int]] = {}
    for position, gram_set in enumerate(train_shingles):
        for gram in gram_set:
            index.setdefault(gram, []).append(position)

    exact: list[int] = []
    near: list[int] = []
    highest = 0.0
    for position, text in enumerate(eval_texts):
        if normalised(text) in train_normal:
            exact.append(position)
        grams = shingles(text)
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(index.get(gram, ()))
        best = 0.0
        for other in candidates:
            union = len(grams | train_shingles[other])
            if not union:
                continue
            score = len(grams & train_shingles[other]) / union
            best = max(best, score)
        highest = max(highest, best)
        if best >= threshold:
            near.append(position)
    return Contamination(
        exact=tuple(exact),
        near=tuple(near),
        max_similarity=highest,
        threshold=threshold,
        eval_rows=len(eval_texts),
        train_rows=len(train_texts),
    )
