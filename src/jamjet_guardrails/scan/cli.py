"""``jamjet-guardrails scan``: find hidden instructions in files on disk.

Exit codes follow the eval command's meaning. ``0``: nothing found, or
``--no-fail``. ``1``: at least one finding, which includes an agent file or a
named file that could not be read cleanly. ``2``: the invocation is wrong (an
unknown check, a check that cannot run without options, a path that does not
exist, a report path that cannot be written). Never a traceback.

A scan that reads no file at all exits ``0`` and says ``scanned 0 files``. That
differs from the eval gate's rule on purpose: a repository with no agent
instruction files is clean, not misconfigured. A path named explicitly that
does not exist is still a ``2``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from jamjet_guardrails.errors import GuardrailChainError
from jamjet_guardrails.scan.formats import to_json, to_sarif, to_text
from jamjet_guardrails.scan.positions import safe_line
from jamjet_guardrails.scan.scanner import scan_paths
from jamjet_guardrails.scan.targets import DEFAULT_CHECKS, TargetError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jamjet-guardrails scan",
        description=(
            "Scan agent instruction files (or, with --all-text, every text file) "
            "for instructions hidden in the encoding."
        ),
        epilog="Put -- before paths that start with a dash.",
    )
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories (default: .)")
    parser.add_argument(
        "--checks",
        default=",".join(DEFAULT_CHECKS),
        help="comma-separated check names (default: %(default)s)",
    )
    parser.add_argument(
        "--all-text",
        action="store_true",
        help="scan every text file, not only agent instruction files",
    )
    parser.add_argument("--format", choices=("text", "json", "sarif"), default="text")
    parser.add_argument(
        "--sarif",
        type=Path,
        help="also write a SARIF 2.1.0 report to this path, in the same run",
    )
    parser.add_argument("--no-fail", action="store_true", help="exit 0 even with findings")
    return parser


def main(argv: Sequence[str]) -> int:
    args = _parser().parse_args(list(argv))
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    if not checks:
        print("scan: --checks names no check", file=sys.stderr)
        return 2
    paths = list(args.paths) or [Path(".")]
    try:
        report = scan_paths(paths, checks=checks, all_text=args.all_text, root=Path.cwd())
    except GuardrailChainError as exc:
        print(f"scan: {exc}", file=sys.stderr)
        return 2
    except TargetError as exc:
        print(f"scan: {safe_line(str(exc))}", file=sys.stderr)
        return 2

    if args.sarif is not None:
        try:
            args.sarif.write_text(to_sarif(report), encoding="utf-8")
        except OSError as exc:
            print(f"scan: cannot write {args.sarif}: {exc.strerror}", file=sys.stderr)
            return 2

    # A console whose encoding cannot carry a character in a path or an excerpt
    # must not turn a finding into a traceback.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")
    if args.format == "text":
        sys.stdout.write(to_text(report))
    else:
        sys.stdout.write(to_json(report) if args.format == "json" else to_sarif(report))
        print(to_text(report).splitlines()[-1], file=sys.stderr)

    if report.hits and not args.no_fail:
        return 1
    return 0
