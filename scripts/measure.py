"""Every separability number the generated corpus is described by, in one run.

Written as a script and not only as tests because the tests assert and this
prints. A threshold answers "did it move"; a report has to say what it moved
to, and a person deciding whether a corpus is fit to be an evaluation set needs
the whole panel at once rather than whichever line failed first.

    ./.venv/bin/python scripts/measure.py [rows.jsonl]

The default is the committed corpus. Pass a path to measure another one, which
is how a regeneration is compared against what it replaced:

    git show <rev>:training/generated/rows.jsonl > /tmp/before.jsonl
    ./.venv/bin/python scripts/measure.py /tmp/before.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.generate import GENERATED, PAIR_OF, load_generated
from training.separability import (
    bag_of_words_accuracy,
    conditional_accuracy,
    cross_validated_accuracy,
    first_token_purity,
    function_word_features,
    leave_one_group_out_accuracy,
    majority_baseline,
    single_token_separability,
    style_features,
    surface_features,
    twin_similarity,
)

OPENER_PURITY = 0.95


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else GENERATED
    rows = load_generated(path)
    texts = [row.text for row in rows]
    labels = [row.label for row in rows]
    groups = [PAIR_OF[row.kind] for row in rows]

    print(f"{path}: {len(rows)} rows, {labels.count(0)} negatives, {labels.count(1)} attacks")
    print(f"majority baseline                    {majority_baseline(labels):.3f}")
    print(
        "marginal   style only                 "
        f"{cross_validated_accuracy([style_features(t) for t in texts], labels):.3f}"
    )
    print(
        "marginal   function words only        "
        f"{cross_validated_accuracy([function_word_features(t) for t in texts], labels):.3f}"
    )
    surface = [surface_features(t) for t in texts]
    print(f"marginal   no content word          {cross_validated_accuracy(surface, labels):.3f}")
    print(f"marginal   bag of words             {bag_of_words_accuracy(texts, labels):.3f}")
    print(
        "CONDITIONAL style only                "
        f"{conditional_accuracy([style_features(t) for t in texts], labels, groups):.3f}"
    )
    print(
        "CONDITIONAL function words only       "
        f"{conditional_accuracy([function_word_features(t) for t in texts], labels, groups):.3f}"
    )
    print(
        f"CONDITIONAL no content word          {conditional_accuracy(surface, labels, groups):.3f}"
    )
    print(
        f"CONDITIONAL bag of words             {bag_of_words_accuracy(texts, labels, groups):.3f}"
    )

    purity = first_token_purity(texts, labels)
    behind = sum(count for count, share in purity.values() if share >= OPENER_PURITY)
    print(
        f"opener share >= {OPENER_PURITY:.0%} pure          {behind / len(rows):.3f} ({behind} rows)"
    )

    single = single_token_separability(texts, labels, groups)
    worst = max(single.items(), key=lambda item: item[1][0])
    print(f"worst per-pair single token          {worst[1][0]:.3f}  {worst[1][1]!r} in {worst[0]}")
    for group, (score, word) in sorted(single.items(), key=lambda item: -item[1][0]):
        print(f"    {score:.3f}  {word:<14} {group}")

    transfer = leave_one_group_out_accuracy(surface, labels, groups)
    pooled = sum(score * groups.count(group) for group, score in transfer.items()) / len(rows)
    print(f"leave-one-pair-out, no content word  {pooled:.3f} pooled")
    for group, score in sorted(transfer.items(), key=lambda item: -item[1]):
        print(f"    {score:.3f}  {group}")

    similarity = sorted(twin_similarity(texts))
    middle = similarity[len(similarity) // 2]
    print(f"twin similarity, median              {middle:.3f}")
    by_pair: dict[str, list[float]] = {}
    for start in range(0, len(rows) - 1, 2):
        by_pair.setdefault(groups[start], []).append(twin_similarity(texts[start : start + 2])[0])
    for group, values in sorted(
        by_pair.items(), key=lambda item: -sorted(item[1])[len(item[1]) // 2]
    ):
        ordered = sorted(values)
        print(
            f"    median {ordered[len(ordered) // 2]:.3f}  mean {sum(values) / len(values):.3f}  {group}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
