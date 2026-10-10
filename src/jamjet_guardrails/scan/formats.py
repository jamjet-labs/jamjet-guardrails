"""Scan reports as text, JSON and SARIF 2.1.0.

Nothing a report prints may start a log line with untrusted text. Paths,
excerpts and decoded hidden text go through ``safe_line`` in the text format;
JSON and SARIF are written with ``ensure_ascii`` so every control, bidi and
invisible character is an escape and a value cannot break its own line.
"""

from __future__ import annotations

import json
from typing import Any

from jamjet_guardrails import __version__
from jamjet_guardrails.scan.positions import safe_line
from jamjet_guardrails.scan.scanner import SCAN_PROBLEMS, ScanHit, ScanReport

INFORMATION_URI = "https://jamjet.dev/guardrails/"
HELP_URI = "https://github.com/jamjet-labs/jamjet-guardrails/blob/main/docs/scanning.md"

# GitHub reads `security-severity` on a rule to rank an alert; without it the
# alert is not classified as a security alert at all. A content finding that
# denies is high, one that redacts is medium, and a file the scan could not
# read cleanly is medium: it is a reason to look, not evidence of a payload.
_SEVERITY = {"deny": "8.0", "redact": "5.0", "allow": "0.0"}
_PROBLEM_SEVERITY = "5.0"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _summary(report: ScanReport) -> str:
    parts = [
        f"scanned {_plural(report.files_scanned, 'file')}",
        _plural(len(report.hits), "finding"),
    ]
    if report.files_skipped:
        parts.append(f"{report.files_skipped} skipped")
    return ", ".join(parts)


def _detail(hit: ScanHit) -> str:
    if hit.check == "scan":
        return SCAN_PROBLEMS.get(hit.finding_type, "")
    return hit.excerpt


def to_text(report: ScanReport) -> str:
    lines: list[str] = []
    for hit in report.hits:
        # Escaped as a whole, not path alone: a path of ":" followed by the
        # ":line" separator would otherwise start the line with "::".
        head = safe_line(
            f"{hit.path}:{hit.line}:{hit.column}: {hit.decision} {hit.check}/{hit.finding_type}"
        )
        detail = safe_line(_detail(hit))
        lines.append(f"{head} {detail}" if detail else head)
        if hit.hidden_text is not None:
            lines.append(f'  hidden text: "{safe_line(hit.hidden_text)}"')
    lines.append(_summary(report))
    return "\n".join(lines) + "\n"


def to_json(report: ScanReport) -> str:
    data = {
        "version": __version__,
        "files_scanned": report.files_scanned,
        "files_skipped": report.files_skipped,
        "hits": [
            {name: getattr(hit, name) for name in ScanHit.__dataclass_fields__}
            for hit in report.hits
        ],
    }
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


# RFC 3986 unreserved characters plus the path separator. Written out rather
# than taken from urllib.parse, because the README's no-network claim is
# checked by refusing any import of urllib from shipped code, and a guard that
# grew an exemption for one submodule is a guard with a hole in it.
_URI_SAFE = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~/")


def quote(path: str) -> str:
    """Percent-encode a path for a URI: UTF-8 bytes, every reserved byte."""
    return "".join(
        chr(b) if b in _URI_SAFE else f"%{b:02X}" for b in path.encode("utf-8", "surrogateescape")
    )


def _rule_id(hit: ScanHit) -> str:
    return f"{hit.check}/{hit.finding_type}"


def _rule(hit: ScanHit) -> dict[str, Any]:
    if hit.check == "scan":
        text = f"File {SCAN_PROBLEMS.get(hit.finding_type, hit.finding_type)}"
        severity = _PROBLEM_SEVERITY
    else:
        text = f"{hit.check} reported {hit.finding_type}"
        severity = _SEVERITY[hit.decision]
    return {
        "id": _rule_id(hit),
        "name": _rule_id(hit).replace("/", "_").replace("-", "_"),
        "shortDescription": {"text": text},
        "helpUri": HELP_URI,
        "properties": {"tags": ["security"], "security-severity": severity},
    }


def _uri(path: str) -> dict[str, str]:
    if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        # Outside the scan root and not expressible relative to it.
        return {"uri": "file://" + quote(path if path.startswith("/") else "/" + path)}
    return {"uri": quote(path), "uriBaseId": "%SRCROOT%"}


def _message(hit: ScanHit) -> str:
    # A code-scanning alert is rendered and often copied into a log, and a
    # variation-selector payload decodes to arbitrary text, so the message is
    # escaped exactly as a text-format line is.
    text = f"{hit.check} reported {hit.finding_type}"
    detail = safe_line(_detail(hit))
    if detail:
        text += f": {detail}"
    if hit.hidden_text is not None:
        text += f'. Hidden text: "{safe_line(hit.hidden_text)}"'
    return text


def to_sarif(report: ScanReport) -> str:
    rules: dict[str, dict[str, Any]] = {}
    for hit in report.hits:
        rules.setdefault(_rule_id(hit), _rule(hit))
    results = [
        {
            "ruleId": _rule_id(hit),
            "level": "error" if hit.decision == "deny" else "warning",
            "message": {"text": _message(hit)},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": _uri(hit.path),
                        "region": {
                            "startLine": hit.line,
                            "startColumn": hit.utf16_column,
                            "endLine": hit.end_line,
                            "endColumn": hit.utf16_end_column,
                        },
                    }
                }
            ],
        }
        for hit in report.hits
    ]
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "jamjet-guardrails",
                        "version": __version__,
                        "informationUri": INFORMATION_URI,
                        "rules": [rules[k] for k in sorted(rules)],
                    }
                },
                "columnKind": "utf16CodeUnits",
                "results": results,
            }
        ],
    }
    return json.dumps(sarif, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
