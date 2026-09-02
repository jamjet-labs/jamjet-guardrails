"""Scaffold the files a new check needs.

    python scripts/new_check.py url-exfiltration

Writes the detector, a starter corpus and a test module, then prints the four
edits it deliberately does NOT make: the registry line, the recorded baseline,
the conformance section and the NOTICE entry. Those four are what
tests/test_completeness.py demands, and a scaffold that wrote them would write
four artifacts nobody had thought about, which is worse than four failing tests
naming exactly what is missing.

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


def build_{module}(**options: object) -> Guardrail:
    """Construct the {name} check."""
    return PatternGuardrail(
        name="{name}",
        version=_VERSION,
        patterns=_PATTERNS,
        on_match="deny",
        **options,  # type: ignore[arg-type]
    )
'''

_TEST = '''"""The {name} check."""

from __future__ import annotations

from jamjet_guardrails.detectors import build
from jamjet_guardrails.types import Context

IN = Context(direction="input", origin="user")


def test_{module}_is_registered() -> None:
    # Mutation checked: removing the registry line makes this fail with
    # GuardrailUnavailableError naming what is installed.
    assert build("{name}").name == "{name}"


def test_a_positive_is_denied() -> None:
    # Mutation checked: replacing the pattern with one that cannot match makes
    # this report allow.
    assert build("{name}").check("REPLACE-ME-1", IN).decision == "deny"


def test_ordinary_text_is_allowed() -> None:
    # Mutation checked: widening the pattern to `.+` makes this deny.
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
        "\nFour edits this script deliberately did not make:\n"
        f"  1. register it: AVAILABLE[{name!r}] = build_{module} and "
        f"TYPES[{name!r}] = {const}, in src/jamjet_guardrails/detectors/__init__.py\n"
        "  2. record the baseline: jamjet-guardrails --corpora-dir corpora "
        "--json benchmarks.json --md BENCHMARKS.md --write-baselines corpora/baselines.json\n"
        f"  3. add a '## The {name} constraint' section to docs/conformance.md\n"
        f"  4. add a 'corpora/{name}/in-repo.jsonl' entry to corpora/NOTICE.md\n"
        "\ntests/test_completeness.py fails until all four are done, naming each one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
