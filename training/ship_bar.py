"""Records the ship bar: the number that decides whether a trained model ships.

**No model exists when this runs, and that is the point.** A threshold written
after seeing the score it judges is not a threshold, it is a rationalisation
with a decimal point on it. So the rule is encoded here as arithmetic over
numbers this module measures, `required_minimum` is applied without a human in
the loop, and `training/ship_bar.json` is committed before any training run.

**It reuses `benchmarks/run.py` rather than reimplementing it.** The reference
models are loaded through `run.classifier`, which verifies every file against
`benchmarks/pins.json` by SHA-256, and scored through `run.score`, which is one
boolean per input counted four ways. Those are the same two functions that will
score our model. "Same harness" is a property of the import, not a claim in a
report: there is no second copy of the loading or the scoring here to drift
from the first.

**The revision comes from the pin, never from the model name.** This branch
once pinned v1 while the plan said v2, and nothing caught it, because a name is
not a hash. Both revisions are read off `run.CLASSIFIER_PINS`, the artifact
records which pin each score came from, and `tests/test_ship_bar.py` holds the
recorded revision to the pin the harness actually loads.

**What is measured, and why on a corpus the reference was trained on.**
`training/evalset.py` carries the full argument; the short form is that the
contamination runs towards the REFERENCE. `jackhhao/jailbreak-classification`
is named on ProtectAI's v2 card as its own training data, and our model has
never seen a row of it. So the gate can fail us unfairly and cannot pass us
unfairly. A win is meaningful, a loss is inconclusive, and the artifact says so
in its own text because on its face this looks like the mistake every screen in
this tree exists to prevent.

Run it from the benchmark virtualenv, with both pinned models downloaded as
`benchmarks/README.md` describes:

    PYTHONPATH=src:. /tmp/guardrails-bench/bin/python -m training.ship_bar \\
      --model-dir /tmp/deberta-prompt-injection \\
      --model-dir-v2 /tmp/deberta-prompt-injection-v2
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from training.evalset import EVAL_SOURCE, EvalRow, balance, load_eval
from training.fetch import ROOT, fetch, load_sources

#: Where the bar lands, beside the tree that will be judged by it.
SHIP_BAR = ROOT / "training" / "ship_bar.json"

#: The split artifact Task 5 committed. Read for the corpus digest and the class
#: balance, so this module and that one cannot disagree about which rows were
#: scored: a corpus identified twice is a corpus that drifts while both records
#: go on looking right on their own.
SPLITS = ROOT / "training" / "generated" / "splits.json"

MANIFEST = ROOT / "training" / "sources.yaml"

BENCHMARKS = ROOT / "benchmarks"

#: The metric, defined here rather than left to the reader of a number.
#:
#: Decision level: ONE boolean per input. That is the only level a classifier
#: can be compared at, because a classifier emits no spans, and it is the level
#: `benchmarks/RESULTS.md` already reports both detectors at.
METRIC = "f1"
METRIC_DEFINITION = (
    "Decision-level F1 of the positive class, one boolean per input. A positive "
    "prediction is the model's INJECTION label; a positive row is this corpus's "
    "`jailbreak` label. precision = TP/(TP+FP), recall = TP/(TP+FN), "
    "F1 = 2*P*R/(P+R), and F1 is 0.0 when TP is 0 rather than undefined. Counted "
    "by benchmarks/run.py:score, the same function that will count our model."
)

#: How the recorded floor was arrived at, kept beside the number so a later
#: reader cannot mistake it for the published finding-level one.
STRUCTURAL_LEVEL = "decision"
STRUCTURAL_CORPUS = "corpora/injection-structural/in-repo.jsonl"

#: The published FINDING-level recall on the same corpus, recorded so the two
#: numbers are visibly different things rather than one number that moved.
#: BENCHMARKS.md counts findings, so a case with four expected spans contributes
#: four; this file counts decisions. Scoring a classifier against 0.873 would
#: compare quantities that are not the same kind of thing.
#:
#: Moved from 0.870 to 0.873 by Task 11 of the phase 3 foundation session,
#: which widened `injection-structural` to run on output as well as input and
#: added 8 cases to its corpus (146 -> 154), 5 of them `allow`. That raised
#: BENCHMARKS.md's finding-level recall for this check from 0.870 to 0.873.
PUBLISHED_FINDING_LEVEL_RECALL = 0.873

#: Places the recorded floor is rounded to, matching how BENCHMARKS.md and
#: benchmarks/RESULTS.md print every rate in this repository.
FLOOR_PLACES = 3


class ShipBarError(RuntimeError):
    """The bar cannot be recorded from what is on disk."""


def harness() -> ModuleType:
    """`benchmarks/run.py`, imported as a module rather than copied.

    `run.py` imports its sibling `render` by bare name, so the directory has to
    be on the path before the import rather than after it. Inserted at the
    front and left there: this module is a one-shot script, and a path entry
    removed in a `finally` would be removed before the caller has used a single
    thing the module returned.
    """
    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))
    return importlib.import_module("run")


def f1(counts: dict[str, int]) -> dict[str, float]:
    """Precision, recall and F1 from one set of decision counts.

    Zero is handled by testing for the case that HAS an answer rather than by
    excluding the cases that do not. `TP == 0` is the whole of it: precision,
    recall and F1 are all 0.0 there, including when the denominators are zero,
    and a division is only reached once there is a true positive to divide.
    Written this way because the negative form -- guarding `TP + FP == 0` and
    falling through otherwise -- leaves a second denominator unguarded and
    returns a NaN that compares False against every threshold it is given.
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    if tp <= 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def required_minimum(scores: Sequence[float]) -> float:
    """The bar, as arithmetic over the reference scores. Decided before measuring.

    **The maximum, not the current model's score.** Two revisions are measured,
    and if the superseded one is the stronger on this corpus then clearing only
    the current one would be clearing the weaker of two bars while calling it
    the reference. Which revision a vendor happens to be advertising is not a
    fact about how hard this corpus is.

    **Strictly greater, with no tolerance.** The plan this task came from
    drafted the bar as the reference less 0.03 absolute. That slack was
    considered and rejected on the ruling the corpus rests on: contamination
    runs towards the reference, so a LOSS here is inconclusive and only a WIN is
    evidence. A tolerance band is exactly the width of the inconclusive region,
    and a bar that can be cleared from inside it authorises shipping on a result
    that establishes nothing. The rejection is recorded in the artifact rather
    than left in this docstring, because the next reader will be holding the
    JSON.

    This is the honest consequence: the bar may well be unreachable, and the
    outcome it then produces -- ship the structural layer alone -- is a real
    outcome the plan names, not a failure of the gate.
    """
    if not scores:
        raise ShipBarError("no reference scores; there would be nothing to set a bar from")
    return max(scores)


COMPARISON = ">"


def eval_rows() -> tuple[list[EvalRow], str]:
    """The external evaluation corpus, verified against BOTH records of it.

    `training/sources.yaml` pins the digest and `training/generated/splits.json`
    records the digest the split was built over. They are checked against each
    other and then against the bytes on disk. Either alone would pass a run over
    a corpus the other record does not describe.
    """
    sources = [source for source in load_sources(MANIFEST) if source.name == EVAL_SOURCE]
    if len(sources) != 1:
        raise ShipBarError(f"{MANIFEST} carries {len(sources)} entries for {EVAL_SOURCE}, want 1")
    source = sources[0]
    if source.role != "eval":
        raise ShipBarError(f"{EVAL_SOURCE} carries role {source.role!r}, not 'eval'")

    split = json.loads(SPLITS.read_text(encoding="utf-8"))["eval"]
    if split["source"] != EVAL_SOURCE:
        raise ShipBarError(f"{SPLITS} was built over {split['source']}, not {EVAL_SOURCE}")
    if split["sha256"] != source.sha256:
        raise ShipBarError(
            f"{SPLITS} records sha256 {split['sha256']} for the evaluation corpus and "
            f"{MANIFEST} pins {source.sha256}; two records of one corpus disagree"
        )

    # `fetch` verifies the bytes against the pin whether it downloaded them or
    # found them already there, which is the case a digest exists for.
    rows = load_eval(fetch(source))
    counted = balance(rows)
    if len(rows) != int(split["rows"]) or counted != {
        int(k): v for k, v in split["labels"].items()
    }:
        raise ShipBarError(
            f"the corpus reads {len(rows)} rows balanced {counted}; {SPLITS} records "
            f"{split['rows']} rows balanced {split['labels']}"
        )
    return rows, source.sha256


def distinct_models(run: ModuleType, dirs: dict[str, Path]) -> None:
    """Refuse to score one model twice and report it as two.

    Two revisions of the same fine-tune can legitimately land on identical
    counts, so identical numbers are not themselves evidence of a mistake. What
    would BE the mistake is one directory reaching both flags, or two pins that
    are the same pin, and neither of those is visible in a results table: it
    prints two rows either way. So it is refused here, structurally, before any
    inference, rather than inferred afterwards from numbers that look alike.
    """
    pins = list(run.CLASSIFIER_PINS)
    if len(pins) < 2:
        raise ShipBarError(
            f"benchmarks/pins.json carries {len(pins)} classifiers; the bar compares the "
            "current revision against every other one and needs at least two"
        )
    for field, seen in (
        ("revision", [str(pin["revision"]) for pin in pins]),
        (
            "set of pinned file digests",
            # Every file the pin records, not the weights alone. Derived from
            # the pin's own `files` map rather than by naming a key: a check
            # written against one filename goes quiet the day a pin spells that
            # file differently, and it would go quiet by passing.
            [
                ",".join(sorted(str(entry["sha256"]) for entry in pin["files"].values()))
                for pin in pins
            ],
        ),
        ("directory", [str(Path(dirs[str(pin["id"])]).resolve()) for pin in pins]),
    ):
        if len(set(seen)) != len(seen):
            raise ShipBarError(
                f"two reference models share a {field} ({seen}); one model scored twice "
                "prints as two rows and nothing in a results table says otherwise"
            )


def cases(run: ModuleType, rows: Sequence[EvalRow]) -> list[Any]:
    """The evaluation rows as `run.Case`, in the one shape every model is scored in.

    Factored out of `measure_reference` so `training/decide.py` scores our model
    over the same construction rather than over a second one that looks like it.
    The id, the category and the label are exactly what `run.score` counts by, so
    a copy that spelled any of the three differently would print a table that
    reads like the reference table and is not comparable with it.
    """
    return [
        run.Case(
            f"eval-{index:04d}", row.text, "jailbreak" if row.label else "benign", bool(row.label)
        )
        for index, row in enumerate(rows)
    ]


def measure_reference(
    run: ModuleType, rows: Sequence[EvalRow], dirs: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every pinned reference model over the evaluation corpus, through run.py.

    In `run.CLASSIFIER_PINS` order, which `run.py` sorts current-first from each
    pin's own `status`. Each model passes `run.controls` before its numbers are
    kept: a harness wired up wrong -- a label index off by one, a predicate
    stuck on False -- produces a plausible table and no error.

    The second return value is how far apart the two references decided, counted
    from the ids `run.score` flagged. Recorded because the measured counts came
    back with the same TP and the same FN for both revisions, which reads like
    one model scored twice; the per-input disagreement is what shows it was not.
    """
    inputs = cases(run, rows)
    positives = {case.id for case in inputs if case.label}
    measured: list[dict[str, Any]] = []
    flagged: dict[str, set[str]] = {}
    for pin in run.CLASSIFIER_PINS:
        model_dir = dirs[str(pin["id"])]
        predict, _tokenizer, max_length = run.classifier(model_dir, pin)
        run.controls(predict, "classifier", str(pin["model"]))
        scored = run.score(inputs, predict)
        counts = dict(scored["overall"])
        flagged[str(pin["id"])] = {str(case_id) for case_id in scored["flagged"]}
        measured.append(
            {
                "id": str(pin["id"]),
                "model": str(pin["model"]),
                # From the pin the harness just loaded and verified by digest,
                # not from the model name and not from a plan.
                "revision": str(pin["revision"]),
                "status": str(pin["status"]),
                "truncation": int(max_length),
                "counts": {key: int(counts[key]) for key in ("tp", "fp", "fn", "tn")},
                **f1({key: int(counts[key]) for key in ("tp", "fp", "fn")}),
            }
        )
    first, second = (entry["id"] for entry in measured[:2])
    differ = flagged[first] ^ flagged[second]
    disagreement = {
        "between": [first, second],
        "inputs_decided_differently": len(differ),
        "on_positives": len(differ & positives),
        "on_negatives": len(differ - positives),
        "note": (
            "How many inputs the two references decided differently. Recorded because "
            "equal TP and equal FN across two rows of a results table is also what one "
            "model scored twice looks like, and this count is what tells them apart. "
            "`distinct_models` refuses the mistake itself, before any inference."
        ),
    }
    return measured, disagreement


def why_the_structural_side_exists() -> dict[str, Any]:
    """What each reference classifier scored on the STRUCTURAL corpus, from the record.

    This is the argument for the second side of the bar, and it is a
    measurement rather than a worry. `benchmarks/RESULTS.md` reports that a
    tokenizer of this family maps a contiguous run of invisible tag characters
    to a single `[UNK]` at any length, so the payload never reaches the model:
    a semantic classifier can be blind to that whole class by construction, not
    by being undertrained. A classifier added to this package therefore has to
    be a layer beside the structural check and cannot be a layer over it.

    Read from `benchmarks/results/measurements.json` rather than typed in.
    Numbers copied between files are numbers that disagree later while each
    file goes on looking right, and this stage has been bitten by that once
    already.
    """
    path = BENCHMARKS / "results" / "measurements.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "id": str(entry["detector"]["id"]),
            "counts": {key: int(entry["overall"][key]) for key in ("tp", "fp", "fn", "tn")},
            **f1({key: int(entry["overall"][key]) for key in ("tp", "fp", "fn")}),
        }
        for entry in record["runs"]
        if entry["corpus"]["id"] == "in-repo" and str(entry["detector"]["id"]).startswith("classi")
    ]
    if not rows:
        raise ShipBarError(f"{path} records no classifier run over the structural corpus")
    return {
        "source": "benchmarks/results/measurements.json",
        "measured": str(record["measured"]),
        "corpus": STRUCTURAL_CORPUS,
        "reference_classifiers": rows,
        "finding": (
            "Both pinned reference classifiers score far below the structural check on "
            "the structural corpus. The tokenizer study in benchmarks/RESULTS.md gives "
            "the mechanism: a contiguous run of invisible tag characters becomes a "
            "single [UNK] token at any length, so the payload never reaches the model. "
            "A semantic classifier can be blind to that class by construction. That is "
            "why clearing the semantic side alone does not clear this bar."
        ),
    }


def structural_floor(run: ModuleType) -> dict[str, Any]:
    """The structural layer's DECISION-level recall, re-derived from the corpus.

    Not copied from BENCHMARKS.md, which publishes the finding-level number, and
    not copied from the plan either. The same `run.score` counts the same
    `run.structural_evaluator` over the shipped corpus, so the floor moves only
    when the corpus or the check moves.
    """
    cases, _spans = run.load_structural_cases()
    structural = run.structural_evaluator()
    run.controls(structural, "constraint", "injection-structural")
    counts = {key: int(value) for key, value in run.score(cases, structural)["overall"].items()}
    rates = f1(counts)
    return {
        "corpus": STRUCTURAL_CORPUS,
        "cases": sum(counts.values()),
        "counts": counts,
        "recall": rates["recall"],
        "floor": round(rates["recall"], FLOOR_PLACES),
    }


def record(run: ModuleType, dirs: dict[str, Path], now: str) -> dict[str, Any]:
    """The whole artifact, assembled from measurements and the rule above."""
    rows, digest = eval_rows()
    distinct_models(run, dirs)
    references, disagreement = measure_reference(run, rows, dirs)
    structural = structural_floor(run)

    current = [entry for entry in references if entry["status"] == "current"]
    if len(current) != 1:
        raise ShipBarError(
            f"{len(current)} pinned classifiers claim status 'current'; the artifact names "
            "one reference model and cannot name none or two"
        )
    reference = current[0]
    minimum = required_minimum([float(entry["f1"]) for entry in references])

    return {
        "metric": METRIC,
        "metric_definition": METRIC_DEFINITION,
        "reference_model": reference["model"],
        "reference_revision": reference["revision"],
        "reference_score": reference["f1"],
        "reference_measured_by": "this-repo",
        "reference_models": references,
        "reference_disagreement": disagreement,
        "held_out_corpus": {
            "name": EVAL_SOURCE,
            "sha256": digest,
            "rows": len(rows),
            "labels": {
                name: balance(rows)[value] for name, value in (("benign", 0), ("jailbreak", 1))
            },
            "recorded_in": "training/generated/splits.json",
            "external": True,
            "note": (
                "Not a held-out slice of the synthetic corpus. training/evalset.py carries "
                "why a held-out slice could not serve: the generated rows are separable by "
                "register, so a score taken from them would measure artifact exploitation "
                "rather than detection."
            ),
        },
        "harness": {
            "entry_point": "benchmarks/run.py",
            "load": "benchmarks/run.py:classifier",
            "score": "benchmarks/run.py:score",
            "driver": "training/ship_bar.py",
            "execution_provider": str(run.EXECUTION_PROVIDER),
            "identical_for_both_models": True,
            "note": (
                "Our model will be scored by the same two functions. The models are pinned "
                "per file by sha256 in benchmarks/pins.json and run.classifier verifies "
                "every one of them on load."
            ),
        },
        "our_minimum": minimum,
        "our_minimum_comparison": COMPARISON,
        "our_minimum_basis": (
            "max(f1) over every pinned reference revision, measured here. The maximum "
            "rather than the current revision's score, so the bar cannot be lowered by "
            "which revision a vendor is advertising. Strictly greater, with no tolerance: "
            "the 0.03 absolute slack this task was drafted with was rejected because a "
            "tolerance band is exactly the width of the inconclusive region described "
            "under `contamination` below."
        ),
        "structural_floor": structural["floor"],
        "structural_floor_level": STRUCTURAL_LEVEL,
        "structural_floor_detail": structural,
        "why_the_structural_side_exists": why_the_structural_side_exists(),
        "structural_floor_note": (
            f"DECISION level, one boolean per input, which is the only level a classifier "
            f"can be compared at because a classifier emits no spans. BENCHMARKS.md "
            f"publishes {PUBLISHED_FINDING_LEVEL_RECALL:.3f} for the same corpus at FINDING "
            f"level, where a case with four expected spans contributes four. The two are "
            f"different quantities, not one quantity that moved; scoring a classifier "
            f"against the finding-level number would compare things that are not the same "
            f"kind of thing."
        ),
        "published_finding_level_recall": PUBLISHED_FINDING_LEVEL_RECALL,
        "clears_the_bar": {
            "semantic": (
                f"our decision-level F1 on {EVAL_SOURCE}, measured through this harness, "
                f"{COMPARISON} our_minimum"
            ),
            "structural": (
                f"the shipped chain's decision-level recall on {STRUCTURAL_CORPUS} "
                f">= structural_floor"
            ),
            "both_required": True,
            "otherwise": (
                "ship the structural layer alone. Failing either side is a real outcome "
                "this stage plans for, not a reason to move the bar."
            ),
        },
        "contamination": {
            "eval_is_in_the_reference_training_data": True,
            "deliberate": True,
            "direction": "towards the reference",
            "argument": (
                "jackhhao/jailbreak-classification is named on ProtectAI's v2 model card as "
                "its own training data. Everywhere else in this tree that is a "
                "disqualification. Here it is not, because of which way the bias runs: "
                "DeBERTa may have memorised these rows and our encoder has never seen one "
                "of them, so this gate can fail us unfairly and cannot pass us unfairly. A "
                "WIN measured here is meaningful. A LOSS is INCONCLUSIVE and must be "
                "reported as inconclusive rather than explained away or treated as a "
                "verdict on our model."
            ),
            "task_adjacency": (
                "This corpus is jailbreak classification. A jailbreak talks a model out of "
                "its own policy; an injection talks it out of its caller's instructions. "
                "The two overlap without coinciding, which is a second way the corpus "
                "favours the reference and a limit on what any number here supports."
            ),
            "single_source": (
                "The external evaluation rests on ONE corpus. training/sources.yaml records "
                "every further candidate screened and what refused each."
            ),
        },
        "authorises": {
            "ship": {
                "allowed_when": "both sides of clears_the_bar are met",
                "meaning": (
                    "the fine-tuned classifier may be shipped in the package as a detector."
                ),
            },
            "publish_headline_precision_and_recall": {
                "allowed": False,
                "reason": (
                    "There is no independent evaluation source for semantic injection of "
                    "the same standing as the corpora behind pii, secrets and "
                    "injection-structural. This bar rests on ONE external corpus, for an "
                    "adjacent task, inside the reference model's own training "
                    "distribution. That is enough to authorise SHIPPING and it is not "
                    "enough to put a headline precision and recall row for this check beside "
                    "the other three, where every number is gated in CI against a corpus "
                    "chosen for that purpose. These are two decisions and this file keeps "
                    "them apart deliberately."
                ),
            },
        },
        "recorded_utc": now,
        "recorded_before_any_model_existed": True,
        "rationale": (
            "Two-sided, and recorded before any training run so that neither side can be "
            "moved to fit a result. SEMANTIC: our decision-level F1 on an EXTERNAL corpus "
            "must beat every pinned reference revision, measured by us through the same "
            "harness that will score us, with no tolerance because a loss on this corpus "
            "is inconclusive and only a win is evidence. STRUCTURAL: adding a classifier "
            "must not cost the structural layer its decision-level recall on "
            f"{STRUCTURAL_CORPUS}, whose floor is re-derived here from the shipped corpus "
            "rather than copied. Failing either side means shipping the structural layer "
            "alone. Clearing both authorises shipping and does not authorise publishing a "
            "headline precision and recall row; see `authorises`. Not to be moved afterwards."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    run = harness()
    parser = argparse.ArgumentParser(description=__doc__)
    for pin in run.CLASSIFIER_PINS:
        parser.add_argument(
            pin["flag"],
            type=Path,
            required=True,
            dest=str(pin["id"]).replace("-", "_"),
            help=f"directory holding model.onnx, config.json and tokenizer.json for "
            f"{pin['model']} at revision {pin['revision']}",
        )
    parser.add_argument(
        "--out", type=Path, default=SHIP_BAR, help="where to write the recorded bar"
    )
    args = parser.parse_args(argv)

    dirs = {
        str(pin["id"]): getattr(args, str(pin["id"]).replace("-", "_"))
        for pin in run.CLASSIFIER_PINS
    }
    now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds")
    bar = record(run, dirs, now)
    out = Path(args.out)
    out.write_text(json.dumps(bar, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    for entry in bar["reference_models"]:
        print(
            f"{entry['model']} @ {entry['revision'][:12]} ({entry['status']}): "
            f"P {entry['precision']:.4f} R {entry['recall']:.4f} F1 {entry['f1']:.4f} "
            f"{entry['counts']}"
        )
    print(f"structural floor  {bar['structural_floor']} ({bar['structural_floor_level']} level)")
    print(f"our_minimum       {bar['our_minimum']!r} {bar['our_minimum_comparison']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
