"""Emit one corpus line with spans computed from substrings.

    python scripts/make_case.py pii-0001 "mail alice@example.com" \
        --find EMAIL "alice@example.com" --source in-repo --license Apache-2.0

A dev tool. It is not shipped in the wheel, which packages src/jamjet_guardrails
only, and nothing under src imports it.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("case_id")
    p.add_argument("text")
    p.add_argument("--find", nargs=2, action="append", metavar=("TYPE", "SUBSTRING"), default=[])
    p.add_argument(
        "--find-type",
        action="append",
        metavar="TYPE",
        default=[],
        help=(
            "expect this type with a null span, for cases where the exact "
            "boundary is genuinely not well defined (a PEM block)"
        ),
    )
    p.add_argument("--direction", default="output")
    p.add_argument("--decision", choices=["allow", "redact", "deny"])
    p.add_argument("--source", required=True)
    p.add_argument("--license", required=True)
    args = p.parse_args()

    findings = []
    for type_name, substring in args.find:
        # index() finds the FIRST occurrence, so a substring appearing twice
        # would be labelled at the wrong place in half the cases. Refuse rather
        # than pick one: a silently wrong span is indistinguishable from a
        # detector bug and costs hours to run down.
        occurrences = args.text.count(substring)
        if occurrences == 0:
            raise SystemExit(f"{args.case_id}: {substring!r} does not occur in the text")
        if occurrences > 1:
            raise SystemExit(
                f"{args.case_id}: {substring!r} occurs {occurrences} times; extend it "
                "with surrounding characters until it is unique"
            )
        start = args.text.index(substring)
        findings.append({"type": type_name, "span": [start, start + len(substring)]})

    findings += [{"type": type_name, "span": None} for type_name in args.find_type]

    decision = args.decision or ("redact" if findings else "allow")
    if bool(findings) != (decision != "allow"):
        raise SystemExit(f"{args.case_id}: decision {decision!r} disagrees with its findings")

    print(
        json.dumps(
            {
                "id": args.case_id,
                "text": args.text,
                "direction": args.direction,
                "expect": {"decision": decision, "findings": findings},
                "source": args.source,
                "license": args.license,
            }
        )
    )


if __name__ == "__main__":
    main()
