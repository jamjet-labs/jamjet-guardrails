"""Export the selected checkpoint to ONNX, quantise it, and record what shipped.

The digest is not decoration. Stage 2b-2 pins the exported file in code and
checks it at load, so a weights swap fails loudly instead of silently moving a
published number. Everything here exists to make that pin worth having.

**It refuses to export weights this repository has not recorded.** The
checkpoint under `data/` is not committed and never will be, so the only thing
that says which bytes it is meant to be is `training/artifacts/training_run.json`.
`checkpoint` holds the directory to that record file by file, and reports an
extra file as loudly as a changed one: a second `model.safetensors` beside the
first is how the wrong checkpoint gets exported under the right name.

**It traces with the sequence axis dynamic, not just the batch axis.** The
brief's sketch made only the batch axis dynamic, which would have pinned the
export at 256 tokens and made the window sweep in `training/measure.py`
impossible to run at 128 or 512 -- the sweep would have had to be deleted or
faked. MiniLM's positional embeddings run to 512, which is the real ceiling and
is asserted here rather than assumed.

**It writes a label map.** The checkpoint's own `config.json` carries
`LABEL_0`/`LABEL_1`, and an index is exactly the thing that inverts in silence.
The exported config names `INJECTION` on the side `training.scoring.POSITIVE`
puts it, so a consumer resolves the class by name. `benchmarks/run.py:classifier`
already reads `id2label` that way, which is the harness the ship bar is measured
through.

**It says whether it reproduces, by writing a record another run can be
compared against.** Run it twice, into two directories with two records, and
`tests/test_training_data.py` compares the digests rather than taking a
sentence about determinism on trust. That is the same shape as
`training_run.json` and `training_run_repeat.json`.

Run it, from the training virtualenv, after `training.train` has selected a
checkpoint:

    ./.venv-training/bin/python -m training.export

The ONNX files land under `data/`, which `.gitignore` excludes. 88 MB and 22 MB
of parameters are not something this repository commits; the record names where
they are and what they hash to, which is what makes a copy on another machine
checkable.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import torch
import transformers
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.fetch import ROOT, sha256_of
from training.scoring import NEGATIVE, POSITIVE
from training.train import MAX_LEN
from training.train import RECORD as TRAINING_RECORD

#: The checkpoint `training.train` selected. Under `data/`, gitignored.
SELECTED = ROOT / "data" / "model" / "selected"

#: Where the export lands. Under `data/` for the same reason the weights are.
EXPORT = ROOT / "data" / "onnx"

#: Where the record of the export lands. Committed, and small.
RECORD = ROOT / "training" / "artifacts" / "export.json"

#: The two files the export produces, named once so the record, the measurement
#: and the prose all mean the same bytes by each name.
FP32 = "injection-fp32.onnx"
INT8 = "injection-int8.onnx"

#: ONNX opset. 17 is what the plan named and what `onnx==1.19.0` and
#: `onnxruntime==1.23.0` both carry; the benchmark tree's `onnxruntime==1.29.0`
#: reads it too, which matters because the ship bar is scored there and not here.
OPSET = 17

#: Dynamic quantisation to signed 8-bit weights. The alternative, QUInt8, is
#: the one whose accuracy depends on the platform's dot-product kernel; QInt8 is
#: what onnxruntime's own guidance names for x86-64 with AVX-512 VNNI and for
#: arm64, which is every machine this is meant to run on.
WEIGHT_TYPE = QuantType.QInt8

#: The label map the export writes, keyed by the polarity `training.scoring`
#: already decides. Derived from `POSITIVE` rather than written as `{0: ..., 1: ...}`
#: so there is one place the polarity lives and no second place to disagree.
LABELS = {POSITIVE: "INJECTION", NEGATIVE: "SAFE"}

#: The tokenizer files copied beside the export, so the directory is loadable on
#: its own. `vocab.txt` is in the list because `tokenizer.json` alone is enough
#: for the `tokenizers` library and not for every loader that might read this.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


class ExportError(RuntimeError):
    """The export cannot honestly start, or cannot honestly be recorded."""


def checkpoint(directory: Path, record: dict[str, Any]) -> None:
    """Hold a checkpoint directory to the digests the run record wrote for it.

    Every fault at once, not the first. A caller who has to re-run to find the
    second problem fixes the first one and re-runs into the same wall, and the
    two together are usually one cause.

    An EXTRA file is a fault too. The weights are outside git, so the only thing
    that says what is in that directory is this record; a file the record does
    not name is a file nothing screened, and a second checkpoint beside the
    first is how the wrong one gets exported under the right name.
    """
    expected: dict[str, str] = record["weights"]["files"]
    faults: list[str] = []
    if not directory.is_dir():
        raise ExportError(f"{directory} does not exist; run `python -m training.train` first")
    present = {path.name for path in sorted(directory.iterdir()) if path.is_file()}
    for name in sorted(expected):
        path = directory / name
        if not path.is_file():
            faults.append(f"{name} is recorded and absent")
            continue
        digest = sha256_of(path)
        if digest != expected[name]:
            faults.append(f"{name} hashes to {digest} and the record says {expected[name]}")
    for name in sorted(present - set(expected)):
        faults.append(f"{name} is present and the record does not name it")
    if faults:
        raise ExportError(
            f"{directory} is not the checkpoint {TRAINING_RECORD.name} recorded: "
            + "; ".join(faults)
        )


def positions(directory: Path) -> int:
    """The longest sequence the exported graph may be given, from the config.

    Read rather than assumed. The window sweep runs at 512, and 512 is only a
    legal window because this model's positional embeddings reach it; a backbone
    with 256 positions would make the widest row of that sweep a crash at run
    time instead of a number.
    """
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    limit = int(config["max_position_embeddings"])
    if limit < MAX_LEN:
        raise ExportError(
            f"the checkpoint carries {limit} positions and was fitted at {MAX_LEN} tokens"
        )
    return limit


def to_onnx(model: Any, tokenizer: Any, path: Path) -> None:
    """Trace the classifier to ONNX with batch and sequence both dynamic.

    Traced at `MAX_LEN`, which is the length the model was fitted at, so the
    single shape the exporter definitely gets right is the one that matters
    most. Both axes are named, so the graph accepts any batch and any sequence
    the positional embeddings allow.

    `token_type_ids` is deliberately not an input. BERT defaults it to zeros and
    `training.train.Rows` never set it, so an export that took it would offer a
    third input nothing upstream fills, which is a way for a caller to feed the
    model something the training loop never did.
    """
    sample = tokenizer(
        "placeholder",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )


def quantise(source: Path, destination: Path) -> None:
    """Dynamic int8 quantisation of the weights, activations left in float.

    Dynamic rather than static, and the difference is a calibration set. Static
    quantisation needs representative inputs to fix the activation ranges, and
    the only inputs with any claim to be representative of what this detector
    meets are the ones in the evaluation corpus. Choosing a calibration set from
    there would be tuning the shipped model on the set it is measured against
    once. Dynamic quantisation needs no such set, which is why it is the scheme
    here rather than the scheme that scored best.
    """
    quantize_dynamic(source, destination, weight_type=WEIGHT_TYPE)


def described(path: Path) -> dict[str, Any]:
    """One artifact, as the record names it: name, size, digest and graph opset.

    The NAME is in the record and not only in the filesystem, because
    `training/measure.py` reads it from here rather than declaring it a second
    time. A filename spelled in two modules is a filename that can disagree
    with itself, and the disagreement reads as a missing file.
    """
    graph = onnx.load(str(path), load_external_data=False)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "opset": [
            {"domain": entry.domain or "ai.onnx", "version": entry.version}
            for entry in graph.opset_import
        ],
        "ir_version": graph.ir_version,
    }


def versions() -> dict[str, str]:
    """What produced this export, in the form a re-run has to match."""
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": np.__version__,
    }


def _config(source: Path, destination: Path) -> dict[str, Any]:
    """The checkpoint's config with a label map that names the positive class."""
    config: dict[str, Any] = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["id2label"] = {str(index): name for index, name in sorted(LABELS.items())}
    config["label2id"] = {name: index for index, name in sorted(LABELS.items())}
    destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def _provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Where this export came from, in terms another machine can resolve.

    Every field is copied off the run record rather than re-derived, so the two
    files cannot disagree about which model was exported. The run record's own
    digest is here too: the fields below are a summary of that file, and a
    summary is only checkable if you can tell whether the file it summarises
    moved.
    """
    return {
        "training_run": str(TRAINING_RECORD.relative_to(ROOT)),
        "training_run_sha256": sha256_of(TRAINING_RECORD),
        "trained_utc": record["trained_utc"],
        "seed": record["hyperparameters"]["seed"],
        "max_length": record["hyperparameters"]["max_length"],
        "selected_epoch": record["selected"]["epoch"],
        "dev_f1_in_torch": record["selected"]["dev"]["f1"],
        "backbone": record["backbone"]["model_id"],
        "backbone_revision": record["backbone"]["revision"],
        "parameters": record["parameters"],
        "fitted_on": record["data"]["fitted_on"],
        "rows_sha256": record["data"]["rows_sha256"],
        "split": record["data"]["split"],
        "split_sha256": record["data"]["split_sha256"],
        "split_seed": record["data"]["split_seed"],
        "weights": record["weights"],
    }


def build(model_dir: Path, out: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Verify, export, quantise, and describe what landed."""
    checkpoint(model_dir, record)
    limit = positions(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    out.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out / FP32, out / INT8
    to_onnx(model, tokenizer, fp32)
    quantise(fp32, int8)

    config = _config(model_dir, out / "config.json")
    for name in TOKENIZER_FILES:
        shutil.copyfile(model_dir / name, out / name)

    described_files = {"fp32": described(fp32), "int8": described(int8)}
    smaller = described_files["int8"]["bytes"] / described_files["fp32"]["bytes"]
    return {
        "exported_on": datetime.now(tz=timezone.utc).date().isoformat(),
        "exported_utc": datetime.now(tz=timezone.utc).isoformat(),
        "directory": str(out.resolve().relative_to(ROOT)),
        "opset_requested": OPSET,
        "traced_at_length": MAX_LEN,
        "max_position_embeddings": limit,
        "dynamic_axes": ["batch", "sequence"],
        "inputs": ["input_ids", "attention_mask"],
        "outputs": ["logits"],
        "quantisation": {
            "scheme": "dynamic",
            "weight_type": WEIGHT_TYPE.name,
            "calibration_set": None,
            "why_dynamic": (
                "static quantisation needs a calibration set, and the only inputs with a "
                "claim to represent what this detector meets are in the corpus it is "
                "measured against once. Dynamic quantisation needs no such set."
            ),
        },
        "id2label": config["id2label"],
        "files": described_files,
        "tokenizer_files": {
            name: sha256_of(out / name) for name in sorted(("config.json", *TOKENIZER_FILES))
        },
        "size_ratio_int8_over_fp32": smaller,
        "provenance": _provenance(record),
        "versions": versions(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=SELECTED)
    parser.add_argument("--out", type=Path, default=EXPORT)
    parser.add_argument("--run", type=Path, default=TRAINING_RECORD)
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args(argv)
    args.model = args.model if args.model.is_absolute() else ROOT / args.model
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.run = args.run if args.run.is_absolute() else ROOT / args.run
    args.record = args.record if args.record.is_absolute() else ROOT / args.record

    run = json.loads(args.run.read_text(encoding="utf-8"))
    report = build(args.model, args.out, run)

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, entry in report["files"].items():
        print(f"{name:5s} {entry['bytes']:>10,} bytes  sha256 {entry['sha256']}")
    print(f"int8 is {report['size_ratio_int8_over_fp32']:.3f} of fp32")
    print(f"wrote {args.record} and the models to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
