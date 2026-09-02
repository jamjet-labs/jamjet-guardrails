"""Builds the split the stage 2b classifier is fitted and selected on.

Three sets, and only two of them come from the same place.

- **train** and **dev**: the synthetic corpus, divided by cluster-of-twins.
  `dev` is for choosing a checkpoint and a threshold, and for nothing else.
- **eval**: `training/evalset.py`, an external public corpus, pinned by digest
  in `training/sources.yaml`. That module carries the reasoning for why the
  evaluation set is not a held-out slice of the synthetic rows, and why the
  corpus it names is the right one despite being contaminated. Read it before
  reading a number measured through it.

The held-out synthetic half is called `dev` here and `Split.held_out` in
`training/split.py`, and the two names are the same rows. That field was called
`evaluation` while the held-out half was still going to be the evaluation set.
It is not, so it was renamed: a field called `evaluation` holding the dev rows
is an invitation to publish a number measured on the corpus the model was fitted
through, which is the one mistake this whole stage is arranged to prevent.

**What this refuses to write.** `leaks` is the gate, and `build` raises through
it rather than reporting. A split that separates a twin, and a training pool
that overlaps the evaluation set, are one failure with two doors: either way the
model has read the rows it is about to be graded on. Both are checked in that
one function so neither can be relaxed while the other goes on looking defended.

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
from training.fetch import ROOT, Source, base_id, fetch, load_sources, sha256_of
from training.generate import GENERATED, Row, load_generated, model_digest
from training.split import HELD_OUT_SHARE, Split, separated_twins, twins

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
#: the prose all mean the same rows by it.
FITTED_ON = "training/generated/rows.jsonl"

#: Corpora compared against the evaluation set although the manifest does not
#: admit them for training.
#:
#: Screening is not admission and cannot be turned into it. `admitted` below
#: reads the manifest `role` and reads nothing here, so a name added to this
#: tuple widens what gets compared and can never quiet a comparison that failed.
#: The direction is the whole safety property: this list can only ever cause
#: more work, never less.
#:
#: `fka/awesome-chatgpt-prompts` is here because it carries the DAN prompt and
#: so does the evaluation corpus, which is why it is `role: excluded` rather
#: than `role: train`. Dropping it from the comparison would delete the measured
#: overlap that exclusion rests on, and an exclusion whose evidence has been
#: tidied away is an exclusion the next reader undoes.
SCREENED: tuple[str, ...] = ("fka/awesome-chatgpt-prompts",)


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


def admitted(sources: Sequence[Source]) -> dict[str, str]:
    """Every pool stage 2b may be fitted on: pool key to the name it goes by.

    Keyed on `base_id`, so a corpus re-pinned to another revision or spelled
    with a different case is the same pool. That is not tidiness. The overlap
    recorded against `fka/awesome-chatgpt-prompts` would not be found again for
    `fka/awesome-chatgpt-prompts@fdf3857` under a string comparison, and a
    guard that reads a re-pinned name as an unrecognised one reports nothing,
    which is the answer that means "safe".

    Reads `role` and only `role`. `SCREENED` widens what gets COMPARED and is
    deliberately invisible here, so nothing a later edit adds to that tuple can
    make a corpus admitted or stop one being admitted.
    """
    pools = {FITTED_ON: FITTED_ON}
    for source in sources:
        if source.role == "train":
            pools[base_id(source.name)] = source.name
    return pools


def pool_key(pool: str) -> str:
    """The identity two spellings of one pool are compared under.

    `FITTED_ON` is a path in this repository and not a dataset id, so it is its
    own key; everything else is a manifest name and goes through `base_id`,
    which raises on a name it cannot parse rather than passing it through.
    """
    return pool if pool == FITTED_ON else base_id(pool)


def training_pools(sources: Sequence[Source]) -> dict[str, list[str]]:
    """Every text a classifier here could be fitted on, kept in separate pools.

    SEPARATE, not joined, and the separation is the finding. Stage 2b fits on
    `FITTED_ON` and on nothing else; the corpora the manifest admits under
    `role: train` are what a later stage may add. Pooled together, one figure
    covers both and cannot say which of them an overlap came from, which is the
    difference between "the evaluation set leaks into what we trained" and "a
    corpus we have not used yet would leak if we did".

    Also compares every corpus in `SCREENED`, which the manifest does NOT admit
    for training. A corpus excluded because it overlaps the evaluation set has
    to keep showing that overlap, or the record stops saying why it was
    excluded. `leaks` gates on `admitted` and never on this set, so screening a
    corpus here cannot admit it.

    Every CELL of a fetched corpus is read, not a column somebody nominated. A
    reader that picked the prompt column would have to be right about each
    corpus's schema, and being wrong reports a clean result over a column it
    never looked at.
    """
    pools: dict[str, list[str]] = {FITTED_ON: [row.text for row in load_generated(GENERATED)]}
    screened = {base_id(name) for name in SCREENED}
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, 4 * 1024 * 1024))
    try:
        for source in sources:
            if source.role != "train" and base_id(source.name) not in screened:
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


def leaks(record: dict[str, Any], sources: Sequence[Source], rows: Sequence[Row]) -> list[str]:
    """Every way the number measured on the evaluation set could be memorisation.

    ONE function, because there are two ways in and they are the same failure.
    A twin separated across the line between train and dev puts most of a held-out row into
    training under the opposite label. A training pool overlapping the external
    evaluation set puts whole evaluation rows into training. Checked in two
    places, the two rules drift: one gets relaxed for a reason that is only good
    for the other, and the split still looks defended because a test somewhere
    else is still green.

    Comparison is by CONTENT. The overlap counts come from
    `training.evalset.compare`, which is handed texts and no names at all, so a
    corpus republished under another id -- which is most of how public corpora
    travel -- is caught by what is in it. Identity is by `base_id` only where
    two names have to be recognised as one pool.

    An admitted pool that was never compared is a leak too, and it is reported
    as one. The alternative is to skip it, which returns the same empty list as
    a clean result: adding a corpus to `role: train` would then widen what may
    be trained on and narrow what is checked, in one edit, silently.

    Returns a line per finding, so a failure says which pool and how much.
    """
    found: list[str] = []

    labels, keys = corpus_keys(rows)
    broken = separated_twins(labels, keys, Split(tuple(record["train"]), tuple(record["dev"])))
    if broken:
        found.append(
            f"{len(broken)} twins straddle the line between train and dev, first at row {broken[0]}; both "
            "members of a twin came out of one call asked to make them alike"
        )

    contamination = record["eval"]["contamination"]
    if contamination["fitted_on"] != FITTED_ON:
        found.append(
            f"the record names {contamination['fitted_on']} as the pool that was fitted on, "
            f"and this module fits on {FITTED_ON}"
        )
    compared = {pool_key(pool["pool"]): pool for pool in contamination["pools"]}
    for key, name in sorted(admitted(sources).items()):
        pool = compared.get(key)
        if pool is None:
            found.append(
                f"{name} is admitted for training and was never compared against the "
                "evaluation set, so its overlap is unknown rather than zero"
            )
            continue
        if pool["exact"] or pool["near"]:
            found.append(
                f"{name} is admitted for training and shares {len(pool['exact'])} exact and "
                f"{len(pool['near'])} near-duplicate rows with the evaluation set, so those "
                "evaluation rows would stop being held out"
            )
        elif pool["max_similarity"] >= pool["threshold"]:
            found.append(
                f"{name} reaches {pool['max_similarity']} against the evaluation set at a "
                f"threshold of {pool['threshold']} while recording no overlap, so the record "
                "disagrees with itself"
            )
    return found


def build(
    seed: int = SEED, dev_share: float = HELD_OUT_SHARE, threshold: float = THRESHOLD
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
    train, dev = split_by_cluster(rows, groups, held_out_fraction=dev_share, seed=seed)

    labels, _ = corpus_keys(rows)
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
    record = {
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
    # The gate, not a report. `main` writes what this returns, so an artifact
    # whose split separates a twin, or whose training pools overlap the
    # evaluation set, cannot be written at all. That is the difference between
    # a rule and a note asking somebody not to.
    found = leaks(record, sources, rows)
    if found:
        raise RuntimeError("; ".join(found))
    return record


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
    fitted_on = {source.name for source in load_sources(MANIFEST) if source.role == "train"}
    for pool in found["pools"]:
        if pool["pool"] == found["fitted_on"]:
            mark = "  <-- FITTED ON"
        elif pool["pool"] in fitted_on:
            mark = "  <-- admitted for training"
        else:
            mark = "  <-- screened, not admitted"
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
    parser.add_argument("--dev-share", type=float, default=HELD_OUT_SHARE)
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
