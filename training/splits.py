"""Builds the split the stage 2b classifier is fitted and selected on.

Three sets, and only two of them come from the same place.

- **train** and **dev**: the synthetic corpus, divided by cluster-of-twins.
  `dev` is for choosing a checkpoint and a threshold, and for nothing else.
- **eval**: `training/evalset.py`, an external public corpus, pinned by digest
  in `training/sources.yaml`. That module carries the reasoning for why the
  evaluation set is not a held-out slice of the synthetic rows, and why the
  corpus it names is the right one despite being contaminated. Read it before
  reading a number measured through it.

The held-out synthetic half is called `dev` here and `Split.evaluation` in
`training/split.py`, and the two names are the same rows. `training/split.py`
was written when the held-out half was still going to be the evaluation set;
its own docstring says in the same breath that nothing in it is a substitute for
an independent one. It was right, that is what changed, and the field keeps its
name rather than being renamed under a module three other files import.

**What this writes.** `training/generated/splits.json`, committed, holding the
indices on each side, the cluster each row was assigned, the digests of the
corpus and the embedding model the assignment came from, and the contamination
finding. Committed because CI has neither a model server nor a network and has
to be able to check the split anyway: with the clusters recorded, the assignment
is re-derivable from the seed by arithmetic alone, and `tests/test_training_data.py`
re-derives it rather than trusting the file.

Run it with a local Ollama and the pinned corpora downloaded:

    ./.venv-training/bin/python -m training.splits
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.cluster import EMBED_MODEL, THRESHOLD, cluster_ids, coarsen, embed, split_by_cluster
from training.evalset import EVAL_SOURCE, EvalRow, balance, compare, load_eval, normalised
from training.fetch import ROOT, Source, fetch, load_sources, sha256_of
from training.generate import GENERATED, Row, load_generated, model_digest
from training.split import EVAL_SHARE, Split, separated_twins, twins

#: Where the built split lands, beside the corpus it divides.
SPLITS = ROOT / "training" / "generated" / "splits.json"

MANIFEST = ROOT / "training" / "sources.yaml"

#: The seed the split is derived from. Recorded in the artifact as well, because
#: a seed that lives only in a default argument is a seed a later edit changes
#: without the committed file saying anything.
SEED = 20260831

#: The fewest words a training text needs before it is compared for
#: contamination.
#:
#: A cell shorter than a trigram falls back to single words in
#: `training.evalset.shingles`, so `TRUE` in a corpus's `for_devs` column would
#: match every eval row containing the word. Those matches are not
#: contamination and they would be most of what a similarity figure reported.
#: Applied to the TRAINING side only; every eval row is compared whatever its
#: length.
MIN_WORDS = 3

#: The pool stage 2b actually fits on, named once so the artifact, the guard and
#: the prose all mean the same rows by it. The contamination finding for this
#: pool is a gate; the finding for any other pool is a record.
FITTED_ON = "training/generated/rows.jsonl"


def corpus_keys(rows: Sequence[Row]) -> tuple[list[int], list[tuple[str, int]]]:
    """The labels and per-call keys `training.split.twins` reads."""
    return [row.label for row in rows], [(row.prompt_id, row.seed) for row in rows]


def twin_ids(rows: Sequence[Row]) -> list[int]:
    """A per-row group id naming the call each row came out of.

    Derived through `twins`, which CHECKS the twin structure rather than
    assuming it, so a corpus that has been re-ordered or filtered outside the
    generator fails here instead of being clustered into groups that are not
    twins.
    """
    labels, keys = corpus_keys(rows)
    ids = [-1] * len(rows)
    for number, (left, right) in enumerate(twins(labels, keys)):
        ids[left] = ids[right] = number
    return ids


def training_pools(sources: Sequence[Source]) -> dict[str, list[str]]:
    """Every text the classifier could be fitted on, kept in separate pools.

    SEPARATE, not joined, and the separation is the finding. Stage 2b fits on
    `FITTED_ON` and on nothing else; the corpora the manifest admits under
    `role: train` are what a later stage may add. Pooled together, one figure
    covers both and cannot say which of them an overlap came from, which is the
    difference between "the evaluation set leaks into what we trained" and "a
    corpus we have not used yet would leak if we did".

    That is not hypothetical. `fka/awesome-chatgpt-prompts` carries the DAN
    prompt, and so does the evaluation corpus.

    Every CELL of a fetched corpus is read, not a column somebody nominated. A
    reader that picked the prompt column would have to be right about each
    corpus's schema, and being wrong reports a clean result over a column it
    never looked at.
    """
    pools: dict[str, list[str]] = {FITTED_ON: [row.text for row in load_generated(GENERATED)]}
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, 4 * 1024 * 1024))
    try:
        for source in sources:
            if source.role != "train":
                continue
            texts: list[str] = []
            with fetch(source).open(newline="", encoding="utf-8") as handle:
                for record in csv.DictReader(handle):
                    for value in record.values():
                        if value and len(normalised(value).split()) >= MIN_WORDS:
                            texts.append(value)
            pools[source.name] = texts
    finally:
        csv.field_size_limit(previous)
    return pools


def build(
    seed: int = SEED, dev_share: float = EVAL_SHARE, threshold: float = THRESHOLD
) -> dict[str, Any]:
    """The whole split, as the record that gets committed.

    Needs a model server for the embeddings and the network for the corpora.
    Nothing in `tests/` calls it; the tests read what it wrote.
    """
    rows = load_generated(GENERATED)
    sources = load_sources(MANIFEST)

    vectors = embed([row.text for row in rows])
    semantic = cluster_ids(vectors, threshold=threshold)
    groups = coarsen(semantic, twin_ids(rows))
    train, dev = split_by_cluster(rows, groups, eval_fraction=dev_share, seed=seed)

    labels, keys = corpus_keys(rows)
    broken = separated_twins(labels, keys, Split(tuple(train), tuple(dev)))
    if broken:
        raise RuntimeError(f"{len(broken)} twins straddle the split, first at row {broken[0]}")

    evaluation, eval_rows = _evaluation(sources)
    eval_texts = [row.text for row in eval_rows]
    pools = training_pools(sources)
    evaluation["contamination"] = {
        "fitted_on": FITTED_ON,
        "pools": [
            {"pool": name, **asdict(compare(eval_texts, texts))}
            for name, texts in sorted(pools.items())
        ],
    }

    sizes = _cluster_sizes(groups)
    return {
        "built_on": datetime.now(tz=timezone.utc).date().isoformat(),
        "seed": seed,
        "dev_share": dev_share,
        "rows": len(rows),
        "rows_sha256": sha256_of(GENERATED),
        "embedding": {
            "model": EMBED_MODEL,
            "model_digest": model_digest(EMBED_MODEL),
            "threshold": threshold,
            "clusters": len(sizes),
            "largest_cluster": max(sizes),
            "clustered_rows": sum(size for size in sizes if size > 1),
        },
        "cluster_of_row": groups,
        "train": train,
        "dev": dev,
        "balance": {
            "train": _labels(labels, train),
            "dev": _labels(labels, dev),
        },
        "eval": evaluation,
    }


def _evaluation(sources: Sequence[Source]) -> tuple[dict[str, Any], list[EvalRow]]:
    """The external corpus, fetched against its recorded digest and counted."""
    by_name = {source.name: source for source in sources}
    source = by_name.get(EVAL_SOURCE)
    if source is None:
        raise RuntimeError(f"{EVAL_SOURCE} is not in {MANIFEST}")
    if source.role != "eval":
        raise RuntimeError(f"{EVAL_SOURCE} carries role {source.role!r}, not 'eval'")
    # `fetch` verifies the download against the manifest digest, and verifies a
    # file already on disk too, so this is the pin being checked rather than
    # recorded a second time.
    rows = load_eval(fetch(source))
    return {
        "source": source.name,
        "sha256": source.sha256,
        "rows": len(rows),
        "labels": {str(label): count for label, count in sorted(balance(rows).items())},
    }, rows


def _labels(labels: Sequence[int], side: Sequence[int]) -> dict[str, int]:
    counted: dict[str, int] = {}
    for index in side:
        key = str(labels[index])
        counted[key] = counted.get(key, 0) + 1
    return dict(sorted(counted.items()))


def _cluster_sizes(ids: Sequence[int]) -> list[int]:
    sizes: dict[int, int] = {}
    for group in ids:
        sizes[group] = sizes.get(group, 0) + 1
    return sorted(sizes.values(), reverse=True)


def report(record: dict[str, Any]) -> str:
    """The statistics the brief asks for, printed rather than left in a file."""
    rows, embedding = record["rows"], record["embedding"]
    largest = embedding["largest_cluster"]
    lines = [
        f"rows            {rows}",
        f"clusters        {embedding['clusters']} at cosine {embedding['threshold']}",
        f"largest cluster {largest} rows ({largest / rows:.1%} of the corpus)",
        (
            f"train / dev     {len(record['train'])} / {len(record['dev'])} "
            f"(held out {len(record['dev']) / rows:.3f}, asked for {record['dev_share']})"
        ),
        f"train balance   {record['balance']['train']}",
        f"dev balance     {record['balance']['dev']}",
        (
            f"eval            {record['eval']['rows']} rows from {record['eval']['source']}, "
            f"labels {record['eval']['labels']}"
        ),
    ]
    found = record["eval"]["contamination"]
    for pool in found["pools"]:
        mark = "  <-- FITTED ON" if pool["pool"] == found["fitted_on"] else ""
        lines.append(
            f"contamination   {pool['pool']}: {len(pool['exact'])} exact, {len(pool['near'])} "
            f"near at {pool['threshold']}, highest {pool['max_similarity']:.3f}, over "
            f"{pool['train_rows']} texts{mark}"
        )
    if largest > rows * 0.10:
        lines.append(
            f"WARNING: the largest cluster holds {largest / rows:.1%} of the corpus, so it is "
            "dominated by one paraphrase family and the split cannot balance around it. That "
            "changes what a number measured on either side means."
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dev-share", type=float, default=EVAL_SHARE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--out", type=Path, default=SPLITS)
    args = parser.parse_args(argv)

    record = build(seed=args.seed, dev_share=args.dev_share, threshold=args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report(record))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
