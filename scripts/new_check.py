"""Scaffold the files a new check needs.

    python scripts/new_check.py url-exfiltration

Writes the detector, a starter corpus and a test module, then prints the edits
it deliberately does NOT make. A scaffold that wrote them would write artifacts
nobody had thought about, which is worse than failing tests naming what is
missing.

The printed list is ORDERED and it is the whole list. An earlier version printed
four edits and said tests/test_completeness.py fails until all four are done. A
recorded walkthrough that followed it literally, in a fresh working copy with
only CONTRIBUTING.md as a guide, found three things wrong with that:

- Registering the check without importing the two names raises NameError while
  importing the package, so pytest dies at COLLECTION with 24 errors and zero
  test results. The one failure the list could not describe was the one it
  produced, so the import is now step 2 and is spelled out.
- test_completeness.py is parametrised over AVAILABLE, so before registration it
  sees nothing at all. What goes red first is test_corpora.py and the scaffolded
  tests themselves.
- Four of the twelve edits the suite demands were in files the list never named:
  tests/test_registry.py, tests/test_chain.py and two places in README.md. The
  headline row is ORDERED after the baseline run because it is copied out of the
  regenerated BENCHMARKS.md, which an unordered list of four hid.

A dev tool. It is not shipped in the wheel, which packages src/jamjet_guardrails
only, and nothing under src imports it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from jamjet_guardrails.detectors import AVAILABLE

_NAME = re.compile(r"\A[a-z][a-z0-9-]*\Z")

_DETECTOR = '''"""The {name} check.

Replace this docstring with what the check detects and why its precision is
defensible. Every signal wants one sentence saying what no legitimate document
produces by accident, because that sentence is what a published number rests on.
"""

from __future__ import annotations

from jamjet_guardrails.authoring import PatternGuardrail
from jamjet_guardrails.protocol import Guardrail

_VERSION = "0.1.0"

{const} = frozenset({{"{type_name}"}})

_PATTERNS = {{"{type_name}": r"REPLACE-ME-\\d+"}}


# `on_match` is a named parameter and NOT a literal beside `**options`. Written
# the second way, `build("{name}", on_match="redact")` raises
# `TypeError: PatternGuardrail() got multiple values for keyword argument
# 'on_match'` -- a TypeError out of `build`, which is the error class
# `detectors.build` exists to keep out of that seam. Every check scaffolded
# before this was fixed shipped unconfigurable.
def build_{module}(on_match: object = "deny", **options: object) -> Guardrail:
    """Construct the {name} check."""
    return PatternGuardrail(
        name="{name}",
        version=_VERSION,
        patterns=_PATTERNS,
        on_match=on_match,  # type: ignore[arg-type]
        **options,  # type: ignore[arg-type]
    )
'''

_TEST = '''"""The {name} check."""

from __future__ import annotations

from jamjet_guardrails.detectors import build
from jamjet_guardrails.types import Context

IN = Context(direction="input", origin="user")


# The comments below say what to RUN, not that it was run. They were "Mutation
# checked: ..." and that was a claim about work the reader had not done: true of
# the template's own `REPLACE-ME-\\d+`, false the moment the pattern is
# replaced, which is the first edit anyone makes. Watch each one fail, then
# rewrite the comment to say what you watched.


def test_{module}_is_registered() -> None:
    # Mutate: delete the registry line. Expect GuardrailUnavailableError naming
    # what is installed.
    assert build("{name}").name == "{name}"


def test_a_positive_is_denied() -> None:
    # Mutate: replace the pattern with one that cannot match. Expect allow.
    assert build("{name}").check("REPLACE-ME-1", IN).decision == "deny"


def test_ordinary_text_is_allowed() -> None:
    # Mutate: widen the pattern to `.+`. Expect deny.
    assert build("{name}").check("an ordinary sentence", IN).decision == "allow"
'''

_CASES = [
    (
        '{{"id": "{module}-0001", "text": "REPLACE-ME-1", "direction": "input", '
        '"expect": {{"decision": "deny", "findings": [{{"type": "{type_name}", "span": [0, 12]}}]}}, '
        '"source": "in-repo", "license": "Apache-2.0"}}'
    ),
    (
        '{{"id": "{module}-0002", "text": "an ordinary sentence", "direction": "input", '
        '"expect": {{"decision": "allow", "findings": []}}, '
        '"source": "in-repo", "license": "Apache-2.0"}}'
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(prog="new_check.py")
    parser.add_argument("name", help="the registry name, lowercase with hyphens")
    parser.add_argument("--into", type=Path, default=Path("."), help="repository root")
    args = parser.parse_args()

    name: str = args.name
    if not _NAME.match(name):
        print(f"{name!r} must match {_NAME.pattern}", file=sys.stderr)
        return 2
    if name in AVAILABLE:
        print(f"{name!r} is already registered", file=sys.stderr)
        return 2

    module = name.replace("-", "_")
    type_name = module.upper().replace("-", "_") + "_MATCH"
    const = module.upper() + "_TYPES"
    root: Path = args.into

    targets = {
        root / "src" / "jamjet_guardrails" / "detectors" / f"{module}.py": _DETECTOR.format(
            name=name, module=module, const=const, type_name=type_name
        ),
        root / "tests" / f"test_{module}.py": _TEST.format(name=name, module=module),
        root / "corpora" / name / "in-repo.jsonl": "\n".join(
            case.format(module=module, type_name=type_name) for case in _CASES
        )
        + "\n",
    }
    existing = [path for path in targets if path.exists()]
    if existing:
        print(f"refusing to overwrite {[str(p) for p in existing]}", file=sys.stderr)
        return 2
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}")

    print(
        f"\nTwelve edits this script deliberately did not make, IN ORDER.\n"
        f"Steps 1 and 2 are one edit to one file and neither works alone.\n"
        f"\n"
        f"  1. import it, in src/jamjet_guardrails/detectors/__init__.py:\n"
        f"       from jamjet_guardrails.detectors.{module} import (\n"
        f"           {const},\n"
        f"           build_{module},\n"
        f"       )\n"
        f"  2. register it, in the same file:\n"
        f"       AVAILABLE[{name!r}] = build_{module}\n"
        f"       TYPES[{name!r}] = {const}\n"
        f"  3. tests/test_registry.py: add {name!r} to the literal set in\n"
        f"     test_every_bundled_detector_is_registered\n"
        f"  4. tests/test_chain.py: add {name!r} to _DECISIONS_PRODUCED, _ON_MATCH\n"
        f"     and _DECISION_KEYWORD, and add one input this check detects to\n"
        f"     _SAMPLES, or the control cannot reach the decisions you declared\n"
        f"  5. record the baseline: jamjet-guardrails --corpora-dir corpora\n"
        f"     --json benchmarks.json --md BENCHMARKS.md"
        f" --write-baselines corpora/baselines.json\n"
        f"  6. README.md 'Measured, not asserted': copy your row out of the\n"
        f"     regenerated BENCHMARKS.md, in the position that file sorts it into.\n"
        f"     This one cannot be done before step 5\n"
        f"  7. README.md 'The checks': a row with kind, directions and every type\n"
        f"  8. README.md: add {name!r} to the sentence naming every self-graded\n"
        f"     check, unless you shipped a third-party corpus for it\n"
        f"  9. README.md 'What it catches': a row, in plain words\n"
        f" 10. docs/conformance.md: a '## The {name} constraint' section\n"
        f" 11. corpora/NOTICE.md: BOTH the summary-table row near the top AND a\n"
        f"     '### corpora/{name}/in-repo.jsonl' section\n"
        f" 12. mutation-check every test you wrote, and rewrite each comment to\n"
        f"     say what you watched fail\n"
        f"\nSteps 5 to 11 are what tests/test_completeness.py and tests/test_readme.py\n"
        f"name for you. Steps 1 to 4 are not: 1 fails at IMPORT, so pytest reports\n"
        f"collection errors and no test results at all, and 2 to 4 fail in files\n"
        f"whose messages do not mention this scaffold."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
