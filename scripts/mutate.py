"""Apply a mutation, watch the named test FAIL, revert, watch it PASS.

A test that has never been watched failing is a test nobody has evidence for.
This runs that loop mechanically over a list of mutations, and reports any that
stayed green, which is the interesting case: a test passing for a reason other
than the one its name claims.

## The bytecode trap, which is why this is committed rather than described

Python invalidates a cached `.pyc` on `(mtime, size)`. A mutation that changes a
byte without changing the length -- `2,` to `5,`, `0.60` to `0.99`, `[A-Za-z0-9]`
to `[A-Za-z0-8]` -- can be written inside the same mtime tick as the restore
before it, and the interpreter then imports the STALE bytecode. The test passes
against code that is not on disk, and the mutation is recorded as green.

That is a false pass produced by the tool used to look for false passes, and it
happened here once in twenty-nine. An independent reviewer reproduced it on five
of seven same-length mutations across three files.

`_clear_bytecode` is the load-bearing fix. `-B` is NOT: it sets
`sys.dont_write_bytecode`, which stops Python WRITING a `.pyc` and does nothing
about reading one that already exists. It is passed as well, to stop the stale
file being recreated for the next run, but clearing is what makes the difference.

## Usage

    ./.venv/bin/python scripts/mutate.py mutations.json

The file is yours to write; none is committed. Each entry is either a `text`
mutation, which replaces the first occurrence of `old` with `new` in `path`, or
a `rows` mutation, which transforms `training/generated/rows.jsonl` through the
lambda in `fn`. Both are reverted afterwards and the test is re-run to prove the
revert was clean.

    [
      {
        "name": "widen the pattern",
        "test": "tests/test_my_check.py::test_ordinary_text_is_allowed",
        "path": "src/jamjet_guardrails/detectors/my_check.py",
        "old": "r\"MY-\\d+\"",
        "new": "r\".+\""
      }
    ]

`test` is a full node id. A bare name still means `tests/test_training_data.py`,
which is where the entries written before this were.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
ROWS = ROOT / "training" / "generated" / "rows.jsonl"


def _clear_bytecode() -> None:
    """The load-bearing half. See the module docstring."""
    for cache in ROOT.rglob("__pycache__"):
        for compiled in cache.glob("*.pyc"):
            compiled.unlink()


def node_id(test: str) -> str:
    """The pytest node id for one mutation entry.

    A bare name means `tests/test_training_data.py`, which is where every entry
    written before this function lived and is why the path was hardcoded there.
    That made the tool unusable for the audience CONTRIBUTING points at it: a
    contributor adding a check writes `tests/test_<name>.py`, and this script
    could not address that file at all, so the mutation loop it exists to
    mechanise had to be run by hand for every test outside one module.

    THE TEST IS THE FILESYSTEM, not the presence of `::`. The first version
    asked only whether the string contained `::`, so a whole FILE selector,
    which is a perfectly good node id, was treated as a bare test name and
    rewritten into `tests/test_training_data.py::tests/test_my_check.py`. A
    Copilot review on the pull request found it.

    That is not a cosmetic mistake in a tool of this kind. pytest exits
    non-zero on a selector it cannot collect, and this harness reads a non-zero
    exit as the mutation having been CAUGHT, so every whole-file mutation entry
    a contributor wrote would have reported a kill without running anything.
    That is the exact failure mode `run` below now refuses outright, and it was
    live in the tool that exists to prevent it.
    """
    path = test.split("::", 1)[0]
    if "::" in test or (ROOT / path).is_file():
        return test
    return f"tests/test_training_data.py::{test}"


#: pytest's own exit codes for "this run measured nothing". 4 is a usage error,
#: which a malformed selector produces; 5 is a selector that collected no tests.
#: Both are non-zero, and a mutation harness that reads non-zero as a kill counts
#: them as evidence that a guard works.
_COLLECTED_NOTHING = (4, 5)


def run(test: str) -> tuple[bool, str]:
    """Run one test by node id. Returns (passed, last line of output).

    REFUSES a selector that collected nothing, rather than reporting it as a
    failure. Every caller here reads `passed=False` as "the mutation was
    caught", so a selector naming a test that does not exist, or a file that
    does not exist, would be recorded as a guard doing its job while nothing
    ran at all. That is the single most common way a mutation check lies, it
    was hit three times by hand during phase 3, and it is worth an exception
    rather than a boolean the caller has to remember to interpret.
    """
    _clear_bytecode()
    result = subprocess.run(
        [
            str(PYTHON),
            "-B",
            "-m",
            "pytest",
            node_id(test),
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            "NO_COLOR": "1",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
    )
    lines = result.stdout.strip().splitlines()
    if result.returncode in _COLLECTED_NOTHING:
        raise SystemExit(
            f"the selector {node_id(test)!r} collected no test (pytest exit "
            f"{result.returncode}). A mutation harness reads a non-zero exit as the "
            f"mutation being caught, so this would have been recorded as a guard "
            f"working while nothing ran. Fix the selector.\n{lines[-1] if lines else ''}"
        )
    return result.returncode == 0, lines[-1] if lines else ""


def apply_text(path: str, old: str, new: str) -> str:
    target = ROOT / path
    original = target.read_text(encoding="utf-8")
    if old not in original:
        raise SystemExit(f"anchor not found in {path}: {old[:60]!r}")
    target.write_text(original.replace(old, new, 1), encoding="utf-8")
    return original


def apply_rows(source: str) -> str:
    original = ROWS.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in original.splitlines() if line]
    # A local development harness reading a file the developer wrote, never a
    # runtime path and never reachable from the package.
    transform = eval(source)
    ROWS.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in transform(rows)),
        encoding="utf-8",
    )
    return original


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: mutate.py <mutations.json>")
    mutations: list[dict[str, Any]] = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    problems: list[str] = []
    for mutation in mutations:
        name, test = mutation["name"], mutation["test"]
        if mutation.get("kind") == "rows":
            target, original = ROWS, apply_rows(mutation["fn"])
        else:
            target = ROOT / mutation["path"]
            original = apply_text(mutation["path"], mutation["old"], mutation["new"])
        green, mutated_line = run(test)
        target.write_text(original, encoding="utf-8")
        restored, restored_line = run(test)
        verdict = "FAIL" if green else "PASS"
        print(f"{verdict}  {name}\n      mutated  -> {mutated_line}", flush=True)
        print(f"      reverted -> {'pass' if restored else 'STILL BROKEN: ' + restored_line}")
        if green:
            problems.append(f"{name}: GREEN ON MUTATION")
        if not restored:
            problems.append(f"{name}: revert did not restore green")
    print(f"\n{len(mutations)} mutations, {len(problems)} problems")
    for problem in problems:
        print("  ", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
