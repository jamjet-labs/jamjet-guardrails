"""Build `corpora/pii/third-party.jsonl` from `nvidia/Nemotron-PII`.

    python scripts/sample_nemotron.py test-00000-of-00001.parquet corpora/pii/third-party.jsonl

The parquet is 151 MB and is NOT committed. This script is, so the committed
slice is reproducible from the pinned revision:

    https://huggingface.co/datasets/nvidia/Nemotron-PII
    revision b70ffaf5ff39e079776134c5bf4381f00a9fd1ed
    data/test-00000-of-00001.parquet
    sha256 1a4b0512ecb5370f0992d29d0f9c07351e6de13f0d7ea33bb18cecb984780247

Needs `pyarrow`, which is deliberately not a dependency of this package: it is a
dev tool, it is not shipped in the wheel, and nothing under src imports it.

Selection is deterministic and independent of row order: candidates are sorted
by the SHA-256 of their `uid` and the first N of each stratum are taken. A seeded
RNG would also be deterministic, but only against one implementation of one
version of Python.

`uid` IS NOT UNIQUE in this file. Every one of the 50,000 uids appears twice,
once with `locale == "us"` and once with `locale == "intl"`, so the locale filter
below is load-bearing rather than a leftover, and a case id carries the locale as
well as the uid. Dropping the filter produces two rows per id, which the loader
refuses; reading a row back by uid alone silently reads the wrong one, which it
cannot refuse. Measured while checking this corpus, where it doubled a count.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REVISION = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
SOURCE = "nvidia/Nemotron-PII@b70ffaf"
LICENSE = "CC-BY-4.0"

# Nemotron labels this corpus keeps, and what each becomes. Everything else is
# dropped, which means a value our patterns match under a dropped label counts as
# a false positive: an SSN-shaped `tax_id` or a phone-shaped `account_number` is
# a wrong claim about what the value IS, and a published precision number should
# pay for it.
#
# `fax_number` is the one mapping that is not in the research report's table, and
# it is a deliberate deviation. Over the first 20,000 `us` rows the PHONE_NUMBER
# pattern makes 5,937 predictions, 4,612 of them landing exactly on a
# `phone_number` span and 1,173 on a `fax_number` span, so this single decision
# moves per-type phone precision from 0.777 to 0.974 and is worth stating rather
# than burying.
#
# Re-derived 2026-09-02 against the pinned parquet. The pair published here read
# 0.773 and 0.969 and no longer did: the two span counts were exact, and the
# prediction total behind the ratios was not, so the ratios were the half nobody
# re-measured after the pattern moved. Nothing in CI can catch that, because the
# parquet is 151 MB and is not committed, which is why the prediction total is
# now published beside the counts and both ratios share it as a denominator.
#
# It is mapped because the corpus records what SHOULD happen. A fax number is a
# telephone number, PHONE_NUMBER is the only telephone type this library has, and
# redacting a fax number out of a document is a redactor doing its job. Calling
# that a false positive would publish a defect that does not exist, and would
# teach whoever reads the number next that the fix is to stop redacting fax
# numbers, which in a PII redactor is a real regression.
#
# The unmapped identifier labels are NOT the same case. A `tax_id` is not an SSN
# and a `medical_record_number` is not one either, so US_SSN over one of them is
# a wrong type on a value the pattern cannot tell apart. Those stay false
# positives, and they are the honest cost of a regex with no issuer check.
LABELS = {
    "email": "EMAIL",
    "ssn": "US_SSN",
    "credit_debit_card": "CREDIT_CARD",
    "phone_number": "PHONE_NUMBER",
    "fax_number": "PHONE_NUMBER",
}

# Fake Name Generator's ten house domains. Their identities are dual-licensed
# GPLv3 / CC-BY-SA-3.0-US, and two otherwise-ideal corpora advertise MIT while
# carrying them, so a licence tag is not what tells you. Checked here as well as
# in tests/test_corpora.py because this script is the door a future corpus comes
# through, and the test is the lock on the room it ends up in.
FNG_DOMAINS = (
    "dayrep.com",
    "armyspy.com",
    "rhyta.com",
    "cuvox.de",
    "einrot.com",
    "fleckens.hu",
    "gustr.com",
    "jourrapide.com",
    "superrito.com",
    "teleworm.us",
)


def _findings(text: str, raw_spans: str) -> tuple[list[dict[str, Any]], int]:
    """The target findings in one row, and how many of its spans did not fit.

    A span whose slice does not match the value the dataset recorded beside it is
    a MISALIGNED LABEL, and a misaligned label is worse than no label: it scores
    as a false negative that reads exactly like a detector bug. The comparison is
    case-insensitive because 0.24% of rows differ only in case, and none of the
    five labels above has a real misalignment in the whole `us` half (measured:
    0 of roughly 47,000 target spans).
    """
    findings: list[dict[str, Any]] = []
    mismatched = 0
    for span in ast.literal_eval(raw_spans):
        label = LABELS.get(span["label"])
        if label is None:
            continue
        start, end = span["start"], span["end"]
        # str(): a handful of rows record a numeric value (an age, a CVV) as a
        # number rather than a string. None of them is a target label, so this
        # only has to not crash.
        if text[start:end].lower() != str(span["text"]).lower():
            mismatched += 1
            continue
        findings.append({"type": label, "span": [start, end]})
    return findings, mismatched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--positives", type=int, default=200)
    parser.add_argument("--negatives", type=int, default=100)
    args = parser.parse_args()

    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []
    rows = dropped = 0
    columns = ["uid", "locale", "text", "spans", "document_format"]

    for batch in pq.ParquetFile(args.parquet).iter_batches(batch_size=2000, columns=columns):
        for row in batch.to_pylist():
            if row["locale"] != "us":
                continue
            rows += 1
            text = row["text"]
            findings, mismatched = _findings(text, row["spans"])
            if mismatched:
                # The whole ROW goes, not the offending span. A row that keeps
                # its good spans and quietly loses a bad one is a row whose
                # unlabelled PII scores as a false positive.
                dropped += 1
                continue
            lowered = text.lower()
            if any(domain in lowered for domain in FNG_DOMAINS):
                raise SystemExit(
                    f"{row['uid']}: carries a Fake Name Generator house domain, which is "
                    "GPLv3 / CC-BY-SA-3.0-US and cannot be redistributed from an "
                    "Apache-2.0 repository"
                )
            line = json.dumps(
                {
                    "id": f"nemotron-us-{row['uid']}",
                    "text": text,
                    # Ours, not the dataset's: it records a document format and
                    # not a direction. An unstructured document reads as model
                    # output, a filled structured form as something handed in.
                    "direction": "output" if row["document_format"] == "unstructured" else "input",
                    "expect": {
                        "decision": "redact" if findings else "allow",
                        "findings": findings,
                    },
                    "source": SOURCE,
                    "license": LICENSE,
                }
            )
            key = hashlib.sha256(row["uid"].encode("ascii")).hexdigest()
            (positives if findings else negatives).append((key, line))

    chosen = sorted(positives)[: args.positives] + sorted(negatives)[: args.negatives]
    if len(chosen) < args.positives + args.negatives:
        raise SystemExit(f"only {len(chosen)} rows survived filtering; wanted more")
    # Written in sample order (positives then negatives, each by digest) rather
    # than shuffled, so a regenerated file diffs against the committed one line
    # by line.
    args.out.write_text("".join(line + "\n" for _, line in chosen), encoding="utf-8")

    print(
        f"us rows {rows}, dropped for a misaligned span {dropped}, "
        f"positives available {len(positives)}, negatives available {len(negatives)}, "
        f"written {len(chosen)} to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
