"""Run checks over files and report where each finding sits.

Each check runs on its own rather than in a chain. A scan reports every
finding, and a chain stops being informative once an earlier check rewrites.
A check that raises is reported as a ``CHECK_ERROR`` hit with decision
``deny``, the same direction the chain fails in, and its message is withheld
because a detector's message may quote the content it failed on.

A ``.json`` file is scanned twice over, once as written and once with every
string literal that contains an escape decoded. ``json.dumps`` writes an astral
character as a surrogate-pair escape by default, so a tag-character payload in
an MCP configuration is plain ASCII on disk and invisible to a scan of the raw
bytes. A hit found in a decoded literal is placed at the literal's opening
quote, because the decoded string has no columns of its own in the file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jamjet_guardrails.detectors import build
from jamjet_guardrails.protocol import Guardrail
from jamjet_guardrails.scan.positions import decode_hidden, excerpt, line_col, utf16_col
from jamjet_guardrails.scan.targets import DEFAULT_CHECKS, discover_counted, load
from jamjet_guardrails.types import Context, Decision

_CONTEXT = Context(direction="input", origin="retrieved")

# A JSON string literal. Outside a string JSON has no quote characters, so a
# left-to-right scan finds every literal of a well-formed file. JSON with
# comments can misalign this; the raw scan still covers every literal byte.
_JSON_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')

# Findings about the file rather than about its content. Named by the
# property, and each one is a deny: a file the scan could not read cleanly is
# a file the scan cannot vouch for.
SCAN_PROBLEMS = {
    "NOT_UTF8": "not valid UTF-8; decoded with replacement characters and scanned",
    "CONTAINS_NUL": "contains a NUL byte; scanned past it",
    "OVER_SIZE_CAP": "larger than the size cap; only the first part was scanned",
    "UNREADABLE": "could not be read at all",
}


@dataclass(frozen=True, slots=True)
class ScanHit:
    path: str
    line: int
    column: int
    end_line: int
    end_column: int
    utf16_column: int
    utf16_end_column: int
    check: str
    finding_type: str
    decision: Decision
    excerpt: str
    hidden_text: str | None


@dataclass(frozen=True, slots=True)
class ScanReport:
    files_scanned: int
    files_skipped: int
    hits: tuple[ScanHit, ...]


def _sort_key(h: ScanHit) -> tuple[str, int, int, str, str]:
    return (h.path, h.line, h.column, h.check, h.finding_type)


def _run(
    path: str, text: str, guards: Sequence[tuple[str, Guardrail]], anchor: int | None, file: str
) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for name, guard in guards:
        try:
            verdict = guard.check(text, _CONTEXT)
        except Exception:  # noqa: BLE001 - any raise is a deny, as in the chain
            line, col = line_col(file, anchor or 0)
            u16 = utf16_col(file, anchor or 0)
            hits.append(
                ScanHit(path, line, col, line, col, u16, u16, name, "CHECK_ERROR", "deny", "", None)
            )
            continue
        if verdict.decision == "allow":
            continue
        for finding in verdict.findings:
            if anchor is not None:
                start = end = anchor
                matched = text if finding.span is None else text[finding.span[0] : finding.span[1]]
            elif finding.span is None:
                start = end = 0
                matched = ""
            else:
                start, end = finding.span
                matched = text[start:end]
            line, col = line_col(file, start)
            end_line, end_col = line_col(file, end)
            hits.append(
                ScanHit(
                    path=path,
                    line=line,
                    column=col,
                    end_line=end_line,
                    end_column=end_col,
                    utf16_column=utf16_col(file, start),
                    utf16_end_column=utf16_col(file, end),
                    check=name,
                    finding_type=finding.type,
                    decision=verdict.decision,
                    excerpt=excerpt(matched),
                    hidden_text=decode_hidden(matched),
                )
            )
    return hits


def _build(checks: Sequence[str]) -> list[tuple[str, Guardrail]]:
    return [(name, build(name)) for name in checks]


def _scan_built(path: str, text: str, guards: Sequence[tuple[str, Guardrail]]) -> list[ScanHit]:
    hits = _run(path, text, guards, None, text)
    if path.lower().endswith(".json"):
        seen = {(h.check, h.finding_type, h.hidden_text) for h in hits}
        for m in _JSON_STRING.finditer(text):
            literal = m.group()
            if "\\" not in literal:
                continue
            try:
                decoded = json.loads(literal)
            except ValueError:
                continue
            for hit in _run(path, decoded, guards, m.start(), text):
                key = (hit.check, hit.finding_type, hit.hidden_text)
                if key not in seen:
                    hits.append(hit)
    return sorted(hits, key=_sort_key)


def scan_text(path: str, text: str, checks: Sequence[str]) -> list[ScanHit]:
    """Every non-allow finding of ``checks`` over ``text``, located."""
    return _scan_built(path, text, _build(checks))


def scan_paths(
    paths: Sequence[Path],
    *,
    checks: Sequence[str] = DEFAULT_CHECKS,
    all_text: bool = False,
    root: Path,
) -> ScanReport:
    """Discover, read and scan. Builds each check once, before reading anything."""
    guards = _build(checks)
    targets, skipped = discover_counted(paths, all_text=all_text, root=root)
    hits: list[ScanHit] = []
    for target in targets:
        try:
            loaded = load(target)
        except OSError:
            hits.append(
                ScanHit(target.rel, 1, 1, 1, 1, 1, 1, "scan", "UNREADABLE", "deny", "", None)
            )
            continue
        for problem in loaded.problems:
            hits.append(ScanHit(target.rel, 1, 1, 1, 1, 1, 1, "scan", problem, "deny", "", None))
        hits.extend(_scan_built(target.rel, loaded.text, guards))
    return ScanReport(len(targets), skipped, tuple(sorted(hits, key=_sort_key)))
