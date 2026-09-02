"""Fine-tune a MiniLM backbone with a binary injection head.

Small enough to train on a laptop, which is the point: anyone can reproduce the
run without an account. The seed is threaded through `random`, `numpy` and
`torch` because a training run nobody can repeat is a number nobody can defend.

A plain loop rather than a `Trainer` subclass, so the whole thing reads at
once. What it adds to the loop, and why each addition is not decoration:

**It fits on TRAIN and selects on DEV, and it cannot reach the evaluation set.**
The three sets are described in `training/splits.py`. `train` and `dev` are the
two halves of the synthetic corpus, divided by cluster-of-twins; the evaluation
set is an external public corpus read by `training/evalset.py`, measured once,
by a later task, against a bar written down before this model existed. Nothing
in this module imports that module, opens that corpus or takes a number from
it, and `tests/test_training_data.py` reads this file's source and fails if it
ever does. A checkpoint chosen by peeking at the evaluation set is a checkpoint
whose published number measures the peeking.

**It re-checks the split before it fits on it.** `training/generated/splits.json`
is committed, so the row indices this loads were computed somewhere else, at
some other time. Two things are therefore checked here rather than assumed: the
corpus on disk still hashes to what the split recorded, and no twin straddles
the line. Both members of a twin came out of one call asked to make them alike,
so a separated twin puts most of a held-out row into training under the
opposite label, and every number downstream of that is inflated.

**It refuses to start on an unscreened or unpinned backbone.** The fine-tuned
weights are the released weights with their parameters moved, so the backbone's
licence follows this model into the wheel. `training/backbone.py` holds the
pin, the licence finding and the digests; `verified` raises rather than warns.

**It writes down what it did.** Seed, hyperparameters, library versions, the
digests of the corpus, the split and the backbone, per-epoch DEV metrics with
the confusion counts they were derived from, and the digests of the weights it
saved. A model with no provenance cannot be shipped by this project, and a
record written by hand after the fact is not provenance.

Run it, from the training virtualenv, with the pinned corpus and split
committed in this tree:

    ./.venv-training/bin/python -m training.train

Weights land under `data/`, which `.gitignore` excludes. They are not committed
anywhere: the record says where they are and what they hash to, which is what
makes a copy on another machine checkable.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import transformers
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.backbone import BACKBONE, Backbone, described, download, verified
from training.fetch import ROOT, sha256_of
from training.generate import GENERATED, Row, load_generated
from training.scoring import POSITIVE, Scored, at, scored, sweep
from training.split import Split, separated_twins
from training.splits import FITTED_ON, SPLITS, corpus_keys

#: The longest text the encoder reads, in tokens.
#:
#: MiniLM's positional embeddings run to 512 and this takes half of them. The
#: corpus was generated to a length bound well inside that, and a longer window
#: costs quadratic attention for padding: `_truncated` records how many rows
#: were actually cut, so the choice is a measurement rather than a habit.
MAX_LEN = 256

#: Where the run record lands. Committed, and named beside the export the later
#: task writes into the same directory.
RECORD = ROOT / "training" / "artifacts" / "training_run.json"

#: Where the weights land. Under `data/`, which is gitignored: 90 MB of
#: parameters is not something this repository commits.
WEIGHTS = ROOT / "data" / "model"


class TrainingError(RuntimeError):
    """The run cannot honestly start, or cannot honestly be recorded."""


class Rows:
    """The texts and labels of one side of the split, tokenised once.

    Not a `torch.utils.data.Dataset` subclass. `DataLoader` asks a map-style
    dataset for `__len__` and `__getitem__` and for nothing else, and torch
    ships no type information, so subclassing it here would buy an unchecked
    base class in exchange for nothing the loader uses.

    Tokenised in the constructor rather than per item, so the same rows are
    encoded once instead of once per epoch and a run's tokenisation cannot
    depend on the order the loader happened to visit them in.
    """

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: Any) -> None:
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor([int(label) for label in labels])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def seed_everything(seed: int) -> None:
    """Every generator this run draws from, from one number.

    `torch.manual_seed` seeds the MPS generator as well as the CPU one, and the
    loader is handed its own generator in `main` rather than relying on the
    global stream, so a shuffle cannot silently depend on how many other draws
    happened first.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def cross_checked(actual: Sequence[int], predicted: Sequence[int]) -> Scored:
    """`training.scoring.scored`, held equal to scikit-learn where it is installed.

    Two implementations of one quantity, and not belt and braces. The counts
    are what the record carries and what the suite re-derives every published
    rate from, so a counting error would be re-derived faithfully by every test
    downstream and agree with itself all the way to a number somebody quotes.
    `training/scoring.py` imports nothing, so CI tests the arithmetic without
    scikit-learn; this is the second opinion on the machine that has it.
    """
    mine = scored(actual, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", pos_label=POSITIVE, zero_division=0
    )
    for name, ours, theirs in (
        ("precision", mine.precision, float(precision)),
        ("recall", mine.recall, float(recall)),
        ("f1", mine.f1, float(f1)),
    ):
        if abs(ours - theirs) > 1e-9:
            raise TrainingError(f"the counts give {name} {ours} and scikit-learn gives {theirs}")
    return mine


def probabilities(model: Any, loader: Any, device: Any) -> tuple[list[float], list[int]]:
    """The model's probability of `POSITIVE` for every row, and the true labels.

    Probabilities rather than decisions, because the decision rule is a
    separate choice. `sweep` picks a threshold over these on DEV; the epoch
    metrics use the argmax, which is this at 0.5.
    """
    model.eval()
    chance: list[float] = []
    actual: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            chance += torch.softmax(logits.float(), dim=-1)[:, POSITIVE].cpu().tolist()
            actual += [int(label) for label in batch["labels"].tolist()]
    return chance, actual


def corpus(rows: Sequence[Row], split: dict[str, Any]) -> tuple[Split, list[str], list[int]]:
    """The committed split, re-checked against the corpus it was built from.

    Three checks, and none of them is ceremony. The recorded digest says the
    rows have not changed under the indices; the twin rule says the two halves
    of a call did not land on opposite sides; the coverage check says every row
    is on exactly one side, which is the same leak wearing different clothes.
    """
    found: list[str] = []
    digest = sha256_of(GENERATED)
    if split["rows_sha256"] != digest:
        found.append(
            f"{FITTED_ON} hashes to {digest} and the split recorded "
            f"{split['rows_sha256']}; the corpus moved under the indices"
        )
    if split["rows"] != len(rows):
        found.append(f"the split divides {split['rows']} rows and the corpus holds {len(rows)}")
    made = Split(tuple(split["train"]), tuple(split["dev"]))
    labels, keys = corpus_keys(rows)
    if not found:
        broken = separated_twins(labels, keys, made)
        if broken:
            found.append(
                f"{len(broken)} twins straddle the line between train and dev, first at row "
                f"{broken[0]}; both members of a twin came out of one call asked to make "
                "them alike, so a separated twin puts the held-out text into training"
            )
        placed = sorted(made.train + made.held_out)
        if placed != list(range(len(rows))):
            found.append(
                f"the split places {len(placed)} of {len(rows)} rows, and a row on both "
                "sides or on neither is the same leak as a separated twin"
            )
    if found:
        raise TrainingError("; ".join(found))
    return made, [row.text for row in rows], labels


def _truncated(texts: Sequence[str], tokenizer: Any) -> int:
    """How many texts the window actually cut. Recorded, not assumed to be zero."""
    lengths = tokenizer(list(texts), truncation=False)["input_ids"]
    return sum(1 for tokens in lengths if len(tokens) > MAX_LEN)


def _side(
    texts: Sequence[str], labels: Sequence[int], indices: Sequence[int]
) -> tuple[list[str], list[int]]:
    return [texts[index] for index in indices], [labels[index] for index in indices]


def _balance(labels: Sequence[int]) -> dict[str, int]:
    counted: dict[str, int] = {}
    for label in labels:
        counted[str(label)] = counted.get(str(label), 0) + 1
    return dict(sorted(counted.items()))


def _rooted(path: Path) -> Path:
    """A command-line path, read as relative to the repository root."""
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    """A path as the record names it: relative to the repository where it can be.

    A record read on another machine has to say WHERE something is in terms that
    machine can resolve, and an absolute path from this laptop is not one. A
    destination outside the repository is recorded as given rather than
    rejected, because refusing it would be this function deciding a policy that
    belongs to the caller.
    """
    resolved = path.resolve()
    root = ROOT.resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def _digests(directory: Path) -> dict[str, str]:
    return {path.name: sha256_of(path) for path in sorted(directory.iterdir()) if path.is_file()}


def _considered(directory: Path | None) -> list[dict[str, Any]]:
    """Every other configuration that was tried, from the records it wrote.

    Read off the artifacts rather than retyped. A selection described in prose
    is a claim about runs nobody can check; this one names each run's record,
    its hyperparameters and the DEV F1 it reached, and every one of those files
    is a full record of its own.
    """
    if directory is None or not directory.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        other = json.loads(path.read_text(encoding="utf-8"))
        out.append(
            {
                "record": path.name,
                "epochs": other["hyperparameters"]["epochs"],
                "learning_rate": other["hyperparameters"]["learning_rate"],
                "batch_size": other["hyperparameters"]["batch_size"],
                "seed": other["hyperparameters"]["seed"],
                "selected_epoch": other["selected"]["epoch"],
                "dev_f1": other["selected"]["dev"]["f1"],
            }
        )
    return out


def versions() -> dict[str, str]:
    """What produced this run, in the form a re-run has to match.

    Recorded because a number is a number under a library version. torch's MPS
    kernels are not bit-identical across releases, and a record that omitted
    the version would make a re-run that differs look like an unseeded loop.
    """
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def train(
    backbone: Backbone,
    directory: Path,
    texts: Sequence[str],
    labels: Sequence[int],
    made: Split,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    patience: int,
    out: Path,
) -> dict[str, Any]:
    """The loop. Fits on TRAIN, scores DEV after every epoch, keeps the best.

    "Best" is the highest DEV F1, ties going to the earlier epoch: two
    checkpoints that score the same are not equally good, and the one that took
    fewer passes over the training rows has had fewer chances to memorise them.
    """
    seed_everything(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory, num_labels=2).to(device)
    parameters = sum(int(p.numel()) for p in model.parameters())
    if parameters != backbone.parameters:
        raise TrainingError(
            f"{backbone.model_id} loaded {parameters} parameters and the registry records "
            f"{backbone.parameters}; the pin describes a different model"
        )

    train_texts, train_labels = _side(texts, labels, made.train)
    dev_texts, dev_labels = _side(texts, labels, made.held_out)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader: Any = DataLoader(
        Rows(train_texts, train_labels, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    dev_loader: Any = DataLoader(Rows(dev_texts, dev_labels, tokenizer), batch_size=batch_size)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, Any]] = []
    best_epoch, best_f1 = -1, -1.0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        total = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            ).loss
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        chance, actual = probabilities(model, dev_loader, device)
        here = cross_checked(actual, at(chance, 0.5))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(train_loader),
                "seconds": round(time.monotonic() - started, 1),
                "dev": here.as_record(),
            }
        )
        print(
            f"epoch {epoch} loss {total / len(train_loader):.4f} "
            f"P {here.precision:.4f} R {here.recall:.4f} F1 {here.f1:.4f}"
        )
        if here.f1 > best_f1:
            best_epoch, best_f1 = epoch, here.f1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        elif patience and epoch - best_epoch >= patience:
            print(f"stopping: {patience} epochs without a better dev F1 than epoch {best_epoch}")
            break

    if best_epoch < 0:
        raise TrainingError("no epoch ran, so there is no checkpoint to select")
    model.load_state_dict(best_state)
    model.to(device)
    chance, actual = probabilities(model, dev_loader, device)
    threshold, tuned = sweep(chance, actual)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return {
        "device": str(device),
        "parameters": parameters,
        "truncated_rows": _truncated(texts, tokenizer),
        "epochs_run": len(history),
        "history": history,
        "selected": {
            "epoch": best_epoch,
            "why": (
                "the highest dev F1 of every epoch that ran, ties to the earlier epoch. "
                "Selected on DEV and on nothing else: the external evaluation set is "
                "measured once, by a later task, against a bar recorded before this model "
                "existed"
            ),
            "dev": history[best_epoch]["dev"],
            "dev_at_tuned_threshold": {"threshold": threshold, **tuned.as_record()},
        },
        "weights": {"path": _relative(out), "files": _digests(out)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="stop after this many epochs with no better dev F1; 0 never stops early",
    )
    parser.add_argument("--split", type=Path, default=SPLITS)
    parser.add_argument("--out", type=Path, default=WEIGHTS)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument(
        "--considered",
        type=Path,
        default=None,
        help="a directory of run records this one was selected over, summarised into the record",
    )
    args = parser.parse_args(argv)
    # Resolved against the repository rather than the working directory. Every
    # command in `training/README.md` is written to be pasted from the root,
    # and a run that wrote its weights beside wherever the shell happened to be
    # is a run whose record points at nothing.
    args.split = _rooted(args.split)
    args.out = _rooted(args.out)
    args.record = _rooted(args.record)
    args.considered = None if args.considered is None else _rooted(args.considered)

    # Before anything is loaded. A backbone whose licence this repository has
    # not screened, or whose bytes are not the ones that were screened, is not
    # a thing to find out about after an hour of fitting.
    directory = verified(BACKBONE, download(BACKBONE))

    rows = load_generated(GENERATED)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    made, texts, labels = corpus(rows, split)

    record: dict[str, Any] = {
        # Both, and not one. The date is what a reader wants; the timestamp is
        # what an ORDERING needs. `training/ship_bar.json` records a UTC instant
        # and has to predate every model it judges, and two values on the same
        # day cannot be put in order by their dates.
        "trained_on": datetime.now(tz=timezone.utc).date().isoformat(),
        "trained_utc": datetime.now(tz=timezone.utc).isoformat(),
        "backbone": described(BACKBONE),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "patience": args.patience,
            "max_length": MAX_LEN,
            "optimiser": "AdamW",
        },
        "data": {
            "fitted_on": FITTED_ON,
            "rows_sha256": sha256_of(GENERATED),
            "split": _relative(args.split),
            "split_sha256": sha256_of(args.split),
            "split_seed": split["seed"],
            "train": {
                "rows": len(made.train),
                "labels": _balance(_side(texts, labels, made.train)[1]),
            },
            "dev": {
                "rows": len(made.held_out),
                "labels": _balance(_side(texts, labels, made.held_out)[1]),
            },
        },
        "versions": versions(),
    }
    record.update(
        train(
            BACKBONE,
            directory,
            texts,
            labels,
            made,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            patience=args.patience,
            out=args.out,
        )
    )
    record["considered"] = _considered(args.considered)

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"\nselected epoch {record['selected']['epoch']}, dev F1 {record['selected']['dev']['f1']:.4f}"
    )
    print(f"wrote {args.record} and weights to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
