"""Scan files on disk with the library's checks.

``jamjet-guardrails scan`` is the command line over this package. By default
it reads agent instruction files only and runs ``injection-structural`` only.
"""

from __future__ import annotations

from jamjet_guardrails.scan.scanner import ScanHit, ScanReport, scan_paths, scan_text

__all__ = ["ScanHit", "ScanReport", "scan_paths", "scan_text"]
