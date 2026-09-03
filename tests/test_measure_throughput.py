"""The throughput script's arithmetic, on the machine that cannot see its own work.

`scripts/measure_throughput.py` exists so a reader can reproduce the published
latency page on their own machine, which makes the machine least like this one
the machine it most has to work on. A coarse `perf_counter` is the shape that
breaks it: `secrets` over a kilobyte takes 8 microseconds here, and on a
platform whose clock ticks every 15.6 milliseconds that lands as a p50 of
exactly zero. Dividing by it raised `ZeroDivisionError` and took the script down
before it printed anything, which is the worst possible failure for a script
whose whole purpose is to hand somebody a number.

Imported through `importlib` rather than run as a subprocess, because the input
that triggers this is a measurement the subprocess cannot be made to produce on
a machine with a good clock. `training/ship_bar.py` imports `benchmarks/run.py`
the same way and for the same reason: a dev-tree module that is not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "measure_throughput.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_throughput", SCRIPT)
    assert spec is not None and spec.loader is not None, f"{SCRIPT} is not importable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("p50", [0.0, -0.0, -1.0])
def test_a_p50_the_clock_could_not_see_reports_no_throughput(p50: float) -> None:
    """A p50 of zero is an absent measurement, not a fast one.

    Zero and negative zero are different floats that compare equal, and both
    arrive from a timer that ticked once across the whole call. A negative value
    cannot arrive from `perf_counter`, which is monotonic, and is refused with
    them because the alternative is a negative throughput printed as fact.

    Mutation-checked: restoring the division makes this raise
    `ZeroDivisionError` for the first two, and print a negative rate for the
    third.
    """
    module = _module()
    assert module.throughput(1024, p50) is None


def test_a_measurable_p50_reports_the_rate_the_arithmetic_gives() -> None:
    """The false-reject control. A guard returning None for everything would
    also pass the test above, and the table would then print a dash for every
    row on every machine."""
    module = _module()
    # One mebibyte in one second is one mebibyte per second, which is the only
    # value in this function that can be checked without repeating its own
    # arithmetic back at it.
    assert module.throughput(1_048_576, 1000.0) == pytest.approx(1.0)


def test_the_table_renders_an_unseeable_row_without_crashing() -> None:
    """The end to end shape, because the crash was in the caller and not in the
    arithmetic. A rate of None has to reach the page as a dash beside the real
    rows rather than as a traceback instead of the page."""
    module = _module()
    rendered = module.table(
        [
            {
                "check": "secrets",
                "chars": 1024,
                "findings": 0,
                "utf8_bytes": 1024,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            },
            {
                "check": "pii",
                "chars": 1024,
                "findings": 3,
                "utf8_bytes": 1024,
                "p50_ms": 0.204,
                "p95_ms": 0.215,
                "p99_ms": 0.250,
            },
        ]
    )
    lines = rendered.splitlines()
    assert lines[2].split()[-1] == "-", lines[2]
    assert lines[3].split()[-1] == "4.8", lines[3]
    # And the reader is told what the dash means, in the same output, rather
    # than being left to guess that a check ran infinitely fast.
    assert "perf_counter resolution" in rendered
