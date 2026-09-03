"""What every registered check costs per call, over inputs from 1 KB to 1 MB.

    ./.venv/bin/python scripts/measure_throughput.py

Prints a table, then the same measurement as JSON. `--json PATH` sends the JSON
to a file instead and leaves the table alone on stdout.

WHY THIS IS A SCRIPT AND NOT A TEST. `ci.yml` regenerates `benchmarks.json` and
`BENCHMARKS.md` on every matrix leg and diffs them byte for byte, which works
only because those artifacts are deterministic. A wall clock is not. A timing
assertion in that suite would be either loose enough to catch nothing or tight
enough to fail on a busy runner, and either way it would be red for a reason
that has nothing to do with a detector. So this measures and prints, and
`docs/performance.md` records what it printed, on a machine it names.

It lives under `scripts/` and therefore ships in neither distribution: the
wheel target packages `src/jamjet_guardrails` alone, and `release.yml` opens the
built wheel and fails if it carries anything outside that package.

WHAT IS MEASURED is one `check` call on one guardrail, built the way its
PUBLISHED ROW is built: through `jamjet_guardrails.detectors.build` with
`jamjet_guardrails.eval.fixtures.options_for`, so a check that cannot exist
without options is timed under the same fixture its precision and recall were
measured under and the two numbers describe the same object. The chain is not
in the loop. A chain is these calls plus one rewrite, and attributing its cost
to a check would hide which check is expensive.

WHAT THE NUMBERS INCLUDE. Garbage collection stays enabled. A p99 is a tail,
collection pauses are part of the tail a caller actually sees, and disabling the
collector would produce a p99 lower than anything a caller could reproduce.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jamjet_guardrails import Context
from jamjet_guardrails.detectors import AVAILABLE, build
from jamjet_guardrails.eval.fixtures import options_for
from jamjet_guardrails.types import Direction

# Powers of four from 1 KB, so linearity is visible by dividing adjacent rows: a
# linear check costs 4x per step and a quadratic one costs 16x. Two adjacent
# sizes cannot show that; this range can, and its top is the one megabyte the
# design document names.
SIZES: tuple[int, ...] = (1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576)

# Enough samples that the 99th percentile is the fourth slowest call rather than
# the slowest one. At 200 the nearest-rank index for p99 is 197, so three calls
# sit above it; at 100 the p99 IS the maximum, which is one scheduling hiccup
# away from reporting noise as a percentile.
REPETITIONS = 200

# Discarded, never averaged in. The first calls through a check pay for a cold
# branch predictor and for the allocator growing to the size of the input, and
# at 1 KB that first call is several times the median.
WARMUP = 5

PERCENTILES: tuple[float, ...] = (0.50, 0.95, 0.99)

# Five ZERO WIDTH SPACE, written as escapes. Spelled literally, this constant is
# an empty-looking string that no reviewer can see and no diff can show, and the
# whole point of the seeded input is that a reader knows what is in it.
_ZERO_WIDTH = "\u200b" * 5

# One block of the generated input: ordinary prose with seeded values spaced
# through it, so each check sees roughly one of its own findings per kilobyte and
# the finding-construction and rewriting paths are timed rather than skipped. The
# `findings` column of the output is what that density actually came to, measured
# rather than asserted here.
#
# Every value is a literal. There is no random number generator anywhere in this
# file, so the input at a given size is the same string on every machine and
# every interpreter, and a number printed here can be reproduced elsewhere.
#
# The seeded values are the shapes the corpora use. The AWS key is the one AWS
# publishes as an example in its own documentation, the address is at
# `example.com`, and the ticket id and internal host are the patterns in
# `jamjet_guardrails.eval.fixtures`, so the `rules` row is measured on content
# its own fixture is written to match.
_BLOCK: tuple[str, ...] = (
    "The quarterly reconciliation ran overnight and cleared without exception. ",
    "Settlement instructions were received from the correspondent at 04:12 UTC. ",
    "alice.mercer@example.com raised the discrepancy against the ledger. ",
    "No manual intervention was required for the second batch. ",
    "Operations confirmed the balance against the nostro statement. ",
    "The reference data feed was stale for eleven minutes and then recovered. ",
    "Deployment notes are attached to JIRA-48213 for the release manager. ",
    "Throughput held steady across both regions during the window. ",
    "A retry was scheduled for the three messages held in the queue. ",
    "See ledger-primary.corp.example for the current replication lag. ",
    "The auditor asked for the full trail rather than the summary. ",
    "Nothing in this paragraph is confidential or personally identifying. ",
    "Access was granted using AKIAIOSFODNN7EXAMPLE before the rotation. ",
    "The runbook step was skipped because the primary never failed over. ",
    f"An invisible run{_ZERO_WIDTH}sits in this sentence and renders as nothing. ",
    "Sign-off is pending from the second reviewer on the change record. ",
)


def content_of(size: int) -> str:
    """Exactly `size` characters of the deterministic input.

    Built by cycling `_BLOCK` and slicing, so the last sentence is usually cut
    mid-word. That is deliberate and stated rather than padded away: padding to
    a sentence boundary would make the real length depend on the block, and a
    published size has to be the size a reader asked for.
    """
    if size < 1:
        raise ValueError(f"size must be at least one character, not {size}")
    block = "".join(_BLOCK)
    return (block * math.ceil(size / len(block)))[:size]


def percentile(ordered: Sequence[float], fraction: float) -> float:
    """Nearest rank, over an already sorted sample.

    Nearest rank rather than an interpolating definition, because every value it
    returns is a call that really happened. An interpolated p99 is an average of
    two calls, and re-running either of them will not produce it.
    """
    if not ordered:
        raise ValueError("no samples to take a percentile of")
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def cpu_name() -> str:
    """The processor, best effort, because a timing number without one is noise.

    Best effort and NOT fatal. `platform.processor()` returns an empty string on
    most Linux builds and the bare architecture on macOS, so the platform read is
    tried first and falls back rather than raising: a machine this cannot name
    still produces valid measurements, and refusing to measure there would be
    the wrong trade.
    """
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def measure(
    name: str,
    size: int,
    repetitions: int,
    warmup: int,
    direction: Direction,
) -> dict[str, Any]:
    """One check, one size: `repetitions` timed calls after `warmup` discarded.

    The guardrail and the content are built ONCE, outside the loop. Building
    either inside it would time regex compilation and string construction and
    report the total as the cost of a check.
    """
    guardrail = build(name, **options_for(name))
    context = Context(direction=direction, origin="user" if direction == "input" else "model")
    content = content_of(size)

    for _ in range(warmup):
        guardrail.check(content, context)

    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        guardrail.check(content, context)
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    samples.sort()

    # Recorded beside the timings, because a row whose decision is `allow` and
    # whose finding count is zero is a row that measured the scan and nothing
    # else. A reader has to be able to tell those apart from a row that also
    # paid for building findings and rewriting the content.
    verdict = guardrail.check(content, context)
    return {
        "check": name,
        "chars": size,
        "utf8_bytes": len(content.encode("utf-8")),
        "repetitions": repetitions,
        "decision": verdict.decision,
        "findings": len(verdict.findings),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def environment(
    sizes: Sequence[int],
    repetitions: int,
    warmup: int,
    direction: str,
) -> dict[str, Any]:
    """Everything a reader needs to say whether their run should agree with this one."""
    return {
        "cpu": cpu_name(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "sizes": list(sizes),
        "repetitions": repetitions,
        "warmup": warmup,
        "direction": direction,
        "percentiles": list(PERCENTILES),
        "percentile_method": "nearest rank over the sorted sample",
        "garbage_collection": "enabled",
    }


_COLUMNS = ("check", "chars", "findings", "p50 ms", "p95 ms", "p99 ms", "MB/s")


def table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{_COLUMNS[0]:<22}{_COLUMNS[1]:>10}{_COLUMNS[2]:>10}"
        f"{_COLUMNS[3]:>10}{_COLUMNS[4]:>10}{_COLUMNS[5]:>10}{_COLUMNS[6]:>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        rate = (row["utf8_bytes"] / 1_048_576) / (row["p50_ms"] / 1000)
        lines.append(
            f"{row['check']:<22}{row['chars']:>10}{row['findings']:>10}"
            f"{row['p50_ms']:>10.3f}{row['p95_ms']:>10.3f}{row['p99_ms']:>10.3f}"
            f"{rate:>9.1f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time every registered check over deterministic inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--direction", choices=("input", "output"), default="input")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SIZES),
        help="input sizes in characters",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the JSON here instead of printing it after the table",
    )
    args = parser.parse_args(argv)

    direction: Direction = "output" if args.direction == "output" else "input"
    sizes = sorted(args.sizes)
    # Checks in name order and sizes ascending, so two runs of this script emit
    # rows in the same order and can be diffed against each other.
    rows = [
        measure(name, size, args.repetitions, args.warmup, direction)
        for name in sorted(AVAILABLE)
        for size in sizes
    ]
    document = {
        "environment": environment(sizes, args.repetitions, args.warmup, args.direction),
        "results": rows,
    }

    print(table(rows))
    serialised = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.json is None:
        print()
        print(serialised, end="")
    else:
        args.json.write_text(serialised, encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
