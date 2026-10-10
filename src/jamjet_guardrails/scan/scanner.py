"""Run checks over files and report where each finding sits.

Each check runs on its own rather than in a chain. A scan reports every
finding, and a chain stops being informative once an earlier check rewrites.
A check that raises is reported as a ``CHECK_ERROR`` hit with decision
``deny``, the same direction the chain fails in, and its message is withheld
because a detector's message may quote the content it failed on.

Every file is scanned twice over when it holds escapes: once as written, and
once with every ``\\uXXXX`` (surrogate pairs joined) and ``\\UXXXXXXXX``
escape decoded, wherever it stands. JSON, TOML, YAML and Python all write an
astral character that way, so a tag-character payload in an MCP config, a
Gemini command or a skill's frontmatter can be plain ASCII on disk. Decoding by
escape rather than by parsing string literals is deliberate: a quote inside a
JSONC comment misaligned the literal parser this replaced, and the payload was
never decoded. A hit in the decoded text is placed at the escapes it came from.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jamjet_guardrails.detectors import build
from jamjet_guardrails.protocol import Guardrail
from jamjet_guardrails.scan.positions import Locator, decode_hidden, excerpt
from jamjet_guardrails.scan.targets import DEFAULT_CHECKS, discover_counted, load
from jamjet_guardrails.types import Context, Decision

_CONTEXT = Context(direction="input", origin="retrieved")

_ESCAPE = re.compile(
    r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})"
    r"|\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})"
)


def unescape(text: str) -> tuple[str, list[int]] | None:
    """``text`` with escapes decoded, and each decoded character's source offset.

    ``None`` when no escape decodes to anything but ASCII, so ordinary files
    pay nothing. A lone surrogate escape is left as written: it is not a
    character, and decoding it would hand the checks something no reader sees.
    The offset list has one extra entry, ``len(text)``, for span ends.
    """
    out: list[str] = []
    where: list[int] = []
    pos = 0
    changed = False
    for m in _ESCAPE.finditer(text):
        hi, lo, bmp, long = m.groups()
        if hi is not None:
            cp = 0x10000 + ((int(hi, 16) - 0xD800) << 10) + (int(lo, 16) - 0xDC00)
        else:
            cp = int(bmp or long, 16)
            if 0xD800 <= cp <= 0xDFFF or cp > 0x10FFFF:
                continue
        out.append(text[pos : m.start()])
        where.extend(range(pos, m.start()))
        out.append(chr(cp))
        where.append(m.start())
        pos = m.end()
        changed = changed or cp > 0x7F
    if not changed:
        return None
    out.append(text[pos:])
    where.extend(range(pos, len(text)))
    where.append(len(text))
    return "".join(out), where


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
    path: str,
    text: str,
    guards: Sequence[tuple[str, Guardrail]],
    where: list[int] | None,
    locate: Locator,
) -> list[ScanHit]:
    """Run each check over ``text``; ``where`` maps its offsets into the file."""

    def at(offset: int) -> int:
        return offset if where is None else where[offset]

    hits: list[ScanHit] = []
    for name, guard in guards:
        try:
            verdict = guard.check(text, _CONTEXT)
        except Exception:  # noqa: BLE001 - any raise is a deny, as in the chain
            line, col = locate.line_col(0)
            u16 = locate.utf16_col(0)
            hits.append(
                ScanHit(path, line, col, line, col, u16, u16, name, "CHECK_ERROR", "deny", "", None)
            )
            continue
        if verdict.decision == "allow":
            continue
        for finding in verdict.findings:
            if finding.span is None:
                start = end = 0
                matched = ""
            else:
                start, end = at(finding.span[0]), at(finding.span[1])
                matched = text[finding.span[0] : finding.span[1]]
            line, col = locate.line_col(start)
            end_line, end_col = locate.line_col(end)
            hits.append(
                ScanHit(
                    path=path,
                    line=line,
                    column=col,
                    end_line=end_line,
                    end_column=end_col,
                    utf16_column=locate.utf16_col(start),
                    utf16_end_column=locate.utf16_col(end),
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
    locate = Locator(text)
    hits = _run(path, text, guards, None, locate)
    decoded = unescape(text)
    if decoded is not None:
        seen = {(h.check, h.finding_type, h.hidden_text) for h in hits}
        for hit in _run(path, decoded[0], guards, decoded[1], locate):
            if (hit.check, hit.finding_type, hit.hidden_text) not in seen:
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
