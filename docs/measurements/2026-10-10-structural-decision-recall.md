# Decision-level recall of injection-structural after the selector signal

Measured on **2026-10-10** with `injection-structural` 0.3.0 over
`corpora/injection-structural/in-repo.jsonl` at corpus version `635fcdcd46c2`
(178 cases).

## Why this is here and not in the ship bar

`training/ship_bar.json` records the structural layer's DECISION-level recall as
the floor a trained classifier had to hold. That floor, 0.891, was measured on
corpus `b704703f431d` (154 cases), and the verdict in
`training/artifacts/ship_check.json` was judged against it. The verdict ran
once. The bar stays as it was, so this number is recorded beside it rather than
in it. It does not judge anything; it is the same measurement taken again on the
corpus that ships now. A future classifier gets a new bar, recorded before that
model is trained.

## Result

| Corpus version | Cases | TP | FP | FN | TN | Decision-level recall |
|---|---|---|---|---|---|---|
| `b704703f431d` (the bar, measured with 0.2.0) | 154 | 49 | 2 | 6 | 97 | 0.891 |
| `e2014aeba636` (first version of the selector signal) | 168 | 56 | 2 | 5 | 105 | 0.918 |
| `635fcdcd46c2` (this measurement, 0.3.0 as shipped) | 178 | 62 | 3 | 5 | 108 | 0.925 |

The middle row is the selector signal as first proposed: it allowed one selector
after each whitespace character and denied doubled VS16. The last row adds
those five whitespace-cover payloads and one presentation-selector bitstream as
positives, and four negatives, one of which (`inj-0177`) it denies.

Two things moved between the first row and the last, and they are not
separable from this table alone: the check gained the variation-selector
signal, and the corpus gained 24 cases and relabelled `inj-0098`. Today's check
over the old 154 cases scores 50 / 2 / 5 / 97: the one decision it changes there
is `inj-0098`, which was a miss and is now caught.

`BENCHMARKS.md` publishes the FINDING-level numbers for the same corpus, where a
case with four expected spans counts four. The two recalls are different
quantities on the same file.

## Reproduce

    PYTHONPATH=src:. ./.venv/bin/python -c \
      "from training.ship_bar import structural_floor, harness; print(structural_floor(harness()))"
