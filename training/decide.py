"""Scores the exported classifier against the bar recorded before it existed.

**This runs once.** `training/ship_bar.json` was written before any model in
this repository had been trained, its bytes are pinned by
`tests/test_ship_bar.py`, and the comparison it asks for is strictly greater
with no tolerance. A number measured a second time, after a window or a
threshold had been moved to improve the first one, would be a measurement of
the choosing rather than of the model. So this module takes the shipped
artifact, the pinned corpus and the recorded bar, and writes down what it got.

**It reuses `benchmarks/run.py` for the same reason `training/ship_bar.py`
does.** The references were loaded through `run.classifier`, scored through
`run.score` and gated on `run.controls`; our model is loaded, scored and gated
through those same three functions, reached by importing that module rather
than by copying it. The evaluation rows are turned into cases by
`ship_bar.cases`, which is the construction the references were scored over,
not a second one shaped like it.

**The model file is renamed on the way in, and nothing is loosened to do it.**
`run.classifier` opens `model.onnx`, and the export writes `injection-int8.onnx`
beside `injection-fp32.onnx`, because a directory holding two models cannot call
either one `model.onnx`. Every file is therefore copied into a scratch directory
under the name the loader wants, and each copy is verified against the digest
`training/artifacts/export.json` recorded when it was written, BEFORE the copy
and again by `run.classifier` after it. The rename moves a filename; the bytes
are checked twice against a record this task did not write.

**The harness's control gate can refuse, and its refusal is written down rather
than allowed to end the run.** `run.controls` raises on the first of four fixed
inputs a detector decides wrongly. Catching that exception is not an exemption:
the outcome becomes a third side of the verdict, it forbids shipping on its own,
and it can only ever refuse. What the catch buys is the number the stage exists
to produce, measured over 1998 rows rather than lost to an exception raised on
four.

**The int8 model, and only it.** `training/artifacts/metrics.json` chose int8
over fp32 on DEV, against a budget fixed before either existed. Scoring both
here and keeping the better one would be choosing on the evaluation set, which
is the one thing this stage is built to prevent, so fp32 is not scored at all.

Run it from the benchmark virtualenv, for the reason `training/README.md` gives:

    PYTHONPATH=src:. /tmp/guardrails-bench/bin/python -m training.decide
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from training.evalset import EVAL_SOURCE, balance
from training.fetch import ROOT, sha256_of
from training.ship_bar import (
    COMPARISON,
    SHIP_BAR,
    STRUCTURAL_CORPUS,
    cases,
    eval_rows,
    f1,
    harness,
    structural_floor,
)

#: Where the verdict lands. Named as a check rather than as a result, because
#: the file records what the comparison returned and not what we wanted.
RECORD = ROOT / "training" / "artifacts" / "ship_check.json"

#: The export's own record, written by `training/export.py` before this task,
#: and the measurement record that names its digest. Two records of one export,
#: held against each other here rather than either one trusted alone.
EXPORT_RECORD = ROOT / "training" / "artifacts" / "export.json"
METRICS = ROOT / "training" / "artifacts" / "metrics.json"

#: The exported directory, which `.gitignore` keeps out of the repository.
ONNX = ROOT / "data" / "onnx"

#: Which of the two exports is judged. The quantisation rule in metrics.json
#: chose this one on DEV, before the evaluation set was touched.
SHIPPED = "int8"

#: The filename `benchmarks/run.py:classifier` opens, and the reason a scratch
#: directory exists at all.
HARNESS_MODEL = "model.onnx"

#: What the model is called in the record and in the control failure message.
MODEL_ID = "jamjet-guardrails/injection-int8"


class DecisionError(RuntimeError):
    """The verdict cannot be measured from what is on disk."""


def export_record() -> dict[str, Any]:
    """`training/artifacts/export.json`, held to the digest metrics.json recorded.

    The export record is where every digest below comes from, so a run that
    read an edited one would verify the model against whatever that edit said.
    `training/measure.py` recorded the export record's own sha256 when it read
    it, which makes the two files a pair that has to agree rather than one file
    vouching for itself.
    """
    metrics: dict[str, Any] = json.loads(METRICS.read_text(encoding="utf-8"))
    recorded = str(metrics["provenance"]["export_record_sha256"])
    found = sha256_of(EXPORT_RECORD)
    if found != recorded:
        raise DecisionError(
            f"{EXPORT_RECORD.name} is sha256 {found} and {METRICS.name} recorded "
            f"{recorded} for it; the record the digests below come from has changed"
        )
    export: dict[str, Any] = json.loads(EXPORT_RECORD.read_text(encoding="utf-8"))
    shipped = str(export["files"][SHIPPED]["sha256"])
    measured = str(metrics["models"][SHIPPED]["sha256"])
    if shipped != measured:
        raise DecisionError(
            f"the export records {shipped} for the {SHIPPED} model and {METRICS.name} "
            f"measured {measured}; two records of one model disagree"
        )
    return export


def staged(export: dict[str, Any], into: Path) -> dict[str, Any]:
    """Copy the shipped export into the layout the harness loader requires.

    Returns the pin `run.classifier` verifies against, built from the digests
    the export recorded rather than from the files being copied. That ordering
    is the whole point: a pin computed from the bytes it is about would pass for
    any bytes at all, so the digest travels from a committed record and the
    files are checked against it here and again inside the loader.

    The byte counts for `config.json` and `tokenizer.json` are the length of the
    payload whose sha256 has just matched the record, because the export records
    a digest for those two and not a size. A second record of a length is not
    what makes the check binding; the digest is.
    """
    recorded: dict[str, tuple[str, str, int | None]] = {
        HARNESS_MODEL: (
            str(export["files"][SHIPPED]["name"]),
            str(export["files"][SHIPPED]["sha256"]),
            int(export["files"][SHIPPED]["bytes"]),
        ),
        "config.json": ("config.json", str(export["tokenizer_files"]["config.json"]), None),
        "tokenizer.json": (
            "tokenizer.json",
            str(export["tokenizer_files"]["tokenizer.json"]),
            None,
        ),
    }
    files: dict[str, dict[str, Any]] = {}
    for name, (source_name, digest, size) in recorded.items():
        source = ONNX / source_name
        if not source.is_file():
            raise DecisionError(f"{source} is not there; run training/export.py first")
        payload = source.read_bytes()
        found = hashlib.sha256(payload).hexdigest()
        if found != digest:
            raise DecisionError(
                f"{source} is sha256 {found} and {EXPORT_RECORD.name} records {digest} "
                f"for it; the file being scored is not the file that was exported"
            )
        if size is not None and len(payload) != size:
            raise DecisionError(
                f"{source} is {len(payload)} bytes and {EXPORT_RECORD.name} records {size}"
            )
        (into / name).write_bytes(payload)
        files[name] = {"sha256": digest, "bytes": len(payload)}
    return {
        "id": "jamjet-guardrails-injection-int8",
        "model": MODEL_ID,
        # There is no vendor revision to quote, so the pin names the training
        # run this model came out of. It is what `run.classifier` prints when a
        # digest fails, and a message naming nothing would send the reader to
        # the wrong artifact.
        "revision": str(export["provenance"]["training_run_sha256"]),
        "files": files,
    }


def measure(
    run: ModuleType, rows: Sequence[Any], model_dir: Path, pin: dict[str, Any]
) -> dict[str, Any]:
    """Our model over the evaluation corpus, through the harness, once.

    `run.controls` runs before the corpus: four fixed inputs, two it must flag
    and two it must not. A tokenizer loaded without its special tokens, a label
    index read off the wrong end or a predicate stuck on False all produce a
    plausible table and no error, and that gate is the only thing between such a
    table and this document. Both references passed it before their numbers were
    kept.

    **Its refusal is recorded here instead of ending the process, and the
    recording is not an exemption.** `run.controls` raises `SystemExit`, which
    for the reference runs was the right end: a harness that cannot classify a
    canonical injection has nothing worth counting. This stage's deliverable is
    a decision about 1998 rows, so the refusal is caught, carried into the
    verdict as a side of its own, and made to forbid shipping on its own --
    which is strictly harder to pass than letting the exception through would
    have been, because the exception would have left the corpus unmeasured and
    the reason unrecorded.

    The predicate handed to the gate is the real one, wrapped only to record
    what it answered. The gate stops at the first input it refuses, so the
    inputs after that one were never asked and are absent rather than passing.
    """
    predict, _tokenizer, max_length = run.classifier(model_dir, pin)
    asked: list[dict[str, Any]] = []

    def watched(text: str) -> bool:
        answer = bool(predict(text))
        asked.append({"input": text, "flagged": answer})
        return answer

    try:
        run.controls(watched, "classifier", MODEL_ID)
    except SystemExit as refusal:
        gate = {"passed": False, "refused_with": str(refusal), "reached": asked}
    else:
        gate = {"passed": True, "refused_with": "", "reached": asked}

    scored = run.score(cases(run, rows), predict)
    counts = {key: int(value) for key, value in scored["overall"].items()}
    return {
        "model": MODEL_ID,
        "file": str(pin["files"][HARNESS_MODEL]["sha256"]),
        "truncation": int(max_length),
        "controls": gate,
        "counts": counts,
        **f1(counts),
        "per_category": scored["per_category"],
    }


def verdict(
    bar: dict[str, Any], semantic: dict[str, Any], structural: dict[str, Any]
) -> dict[str, Any]:
    """Both sides of the recorded bar, applied without a human in the loop.

    Written as the condition that PASSES rather than as the negation of the one
    that fails. `f1 > minimum` is False for a NaN and so refuses to ship on one;
    `not (f1 <= minimum)` is True for a NaN and would ship on it. `f1` here
    cannot produce a NaN, which is the reason to spell the guard the safe way
    round rather than a reason not to bother.

    The sides are reported separately and the ship decision is their
    conjunction, because a stage that failed one side and cleared the other has
    learned two different things and a single boolean loses one of them.

    The bar has two sides and this has three. The third is the harness's own
    control gate, which is not a new bar: `training/ship_bar.py` already gates
    every reference's numbers on it, and both references passed it. It can only
    ever refuse, never authorise, so recording it here cannot lower the bar the
    file records.

    **The structural floor is compared against the recall it was derived from,
    not against the three-decimal rendering of it.** `structural_floor` is
    `round(recall, 3)` and the recall it rounded is 46 of 52 = 0.8846153846, which
    rounds UP. So `recall >= 0.885` is False on a structural layer nothing
    touched, by 0.00038, and the first run of this comparison recorded a
    regression that was a rounding artifact. The bar records both numbers, and
    `structural_floor_detail.recall` is the one the floor was taken from;
    reading it is what makes the comparison ask the question the bar's own
    rationale asks, which is whether adding a classifier cost the structural
    layer any decision-level recall.

    Exact, not tolerant. A tolerance band would be a second thing to argue
    about, and none is needed: the floor and the measurement are now the same
    quantity, so an unchanged layer compares equal and any real change fails.
    `training/ship_bar.json` is byte-pinned and was NOT edited to arrive here.
    A bar that can be rewritten after the result is not a commitment, and the
    rounding is in the rendering rather than in the bar.
    """
    minimum = float(bar["our_minimum"])
    floor = float(bar["structural_floor_detail"]["recall"])
    semantic_clears = float(semantic["f1"]) > minimum
    structural_holds = float(structural["recall"]) >= floor
    controls_pass = bool(semantic["controls"]["passed"])
    return {
        "controls": {
            "gate": "benchmarks/run.py:controls",
            "passed": controls_pass,
            "refused_with": str(semantic["controls"]["refused_with"]),
            "note": (
                "The same four fixed inputs both references were gated on before their "
                "numbers were kept. A refusal here forbids shipping on its own."
            ),
        },
        "semantic": {
            "measured": float(semantic["f1"]),
            "minimum": minimum,
            "comparison": COMPARISON,
            "clears": semantic_clears,
            "margin": float(semantic["f1"]) - minimum,
            "corpus": EVAL_SOURCE,
        },
        "structural": {
            "measured": float(structural["recall"]),
            "floor": floor,
            "floor_published_as": float(bar["structural_floor"]),
            "floor_note": (
                "The recall the floor was derived from, not its three-decimal rendering. "
                "round(0.8846153846, 3) is 0.885, which is ABOVE the value it renders, so "
                "comparing against the rendering fails an unchanged layer by 0.00038. Both "
                "numbers are recorded in training/ship_bar.json and neither was edited."
            ),
            "comparison": ">=",
            "holds": structural_holds,
            "margin": float(structural["recall"]) - floor,
            "corpus": STRUCTURAL_CORPUS,
        },
        "ship_the_classifier": controls_pass and semantic_clears and structural_holds,
        "publish_headline_precision_and_recall": bool(
            bar["authorises"]["publish_headline_precision_and_recall"]["allowed"]
        ),
        "otherwise": str(bar["clears_the_bar"]["otherwise"]),
    }


def record(run: ModuleType, model_dir: Path, now: str) -> dict[str, Any]:
    """The whole artifact: what was scored, what it scored, and what that decides."""
    bar: dict[str, Any] = json.loads(SHIP_BAR.read_text(encoding="utf-8"))
    export = export_record()
    rows, digest = eval_rows()
    pin = staged(export, model_dir)
    semantic = measure(run, rows, model_dir, pin)
    structural = structural_floor(run)
    counted = balance(rows)

    return {
        "measured_utc": now,
        "measured_once": (
            "This is the first and only run of this measurement. Nothing was tuned after "
            "it: the window, the stride, the pooling and the threshold were fixed on DEV "
            "in training/artifacts/metrics.json before the evaluation set was touched, and "
            "a second run taken after reading this one would measure the choosing."
        ),
        "judged_against": {
            "file": "training/ship_bar.json",
            "sha256": sha256_of(SHIP_BAR),
            "recorded_utc": str(bar["recorded_utc"]),
            "our_minimum": float(bar["our_minimum"]),
            "structural_floor": float(bar["structural_floor"]),
            "structural_floor_level": str(bar["structural_floor_level"]),
        },
        "model": {
            "id": str(pin["id"]),
            "name": MODEL_ID,
            "which_export": SHIPPED,
            "file": str(export["files"][SHIPPED]["name"]),
            "sha256": str(export["files"][SHIPPED]["sha256"]),
            "bytes": int(export["files"][SHIPPED]["bytes"]),
            "backbone": str(export["provenance"]["backbone"]),
            "parameters": int(export["provenance"]["parameters"]),
            "export_record": "training/artifacts/export.json",
            "export_record_sha256": sha256_of(EXPORT_RECORD),
            "fp32_not_scored": (
                "The float export exists and was deliberately not scored here. Scoring "
                "both and keeping the better one is choosing on the evaluation set; int8 "
                "was chosen on DEV against a budget fixed before either export existed."
            ),
        },
        "corpus": {
            "name": EVAL_SOURCE,
            "sha256": digest,
            "rows": len(rows),
            "labels": {name: counted[value] for name, value in (("benign", 0), ("jailbreak", 1))},
            "seen_in_training": False,
            "in_the_reference_training_data": bool(
                bar["contamination"]["eval_is_in_the_reference_training_data"]
            ),
        },
        "harness": {
            "entry_point": "benchmarks/run.py",
            "load": "benchmarks/run.py:classifier",
            "score": "benchmarks/run.py:score",
            "controls": "benchmarks/run.py:controls",
            "cases": "training/ship_bar.py:cases",
            "driver": "training/decide.py",
            "execution_provider": str(run.EXECUTION_PROVIDER),
            "renamed_for_the_loader": (
                f"{export['files'][SHIPPED]['name']} is copied to {HARNESS_MODEL} in a "
                "scratch directory because run.classifier opens that name. Every copy is "
                "verified against the export record's digest before the copy and again by "
                "the loader after it."
            ),
            "same_functions_as_the_references": True,
        },
        "semantic": semantic,
        "structural": structural,
        "references": [
            {
                "id": str(entry["id"]),
                "model": str(entry["model"]),
                "revision": str(entry["revision"]),
                "f1": float(entry["f1"]),
                "counts": dict(entry["counts"]),
            }
            for entry in bar["reference_models"]
        ],
        "verdict": verdict(bar, semantic, structural),
        "how_to_read_a_loss": str(bar["contamination"]["argument"]),
        "what_it_does_not_establish": (
            "Nothing here says the classifier works on long documents. "
            "training/artifacts/metrics.json sweeps six window and stride pairs over "
            "buried payloads and every one of them lands between 0.43 and 0.68 F1 at "
            "threshold 0.5 with max pooling. Threshold and pooling are the next stage's "
            "problem, and this measurement is on single inputs the model reads whole."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    run = harness()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RECORD, help="where to write the verdict")
    args = parser.parse_args(argv)

    now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds")
    with tempfile.TemporaryDirectory(prefix="guardrails-decide-") as scratch:
        written = record(run, Path(scratch), now)
    out = Path(args.out)
    out.write_text(json.dumps(written, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    semantic = written["semantic"]
    side = written["verdict"]["semantic"]
    structural = written["verdict"]["structural"]
    for entry in written["references"]:
        print(f"reference {entry['id']}: F1 {entry['f1']:.4f}")
    print(
        f"{MODEL_ID}: P {semantic['precision']:.4f} R {semantic['recall']:.4f} "
        f"F1 {semantic['f1']:.4f} {semantic['counts']}"
    )
    gate = written["verdict"]["controls"]
    print(f"controls   {gate['passed']} {gate['refused_with']}")
    print(f"semantic   {side['measured']!r} {COMPARISON} {side['minimum']!r} -> {side['clears']}")
    print(
        f"structural {structural['measured']!r} >= {structural['floor']!r} -> {structural['holds']}"
    )
    print(f"ship the classifier: {written['verdict']['ship_the_classifier']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
