"""Which files a repository scan reads, and how it reads them.

The scan is adversarial, so every way a file can avoid being read is a bypass.
Two kinds of file are therefore never skipped: an agent instruction file (a
path matching ``AGENT_FILE_PATTERNS``), and any file named on the command line.
Each is decoded leniently and scanned, and whatever stopped a clean read is
returned as a problem the scanner reports as a finding:

- ``NOT_UTF8``: not valid UTF-8. A UTF-16 or UTF-32 byte order mark selects
  that codec; anything else is decoded as UTF-8 with replacement characters.
- ``CONTAINS_NUL``: a NUL byte, which an agent reading the file does not stop
  at, and which a binary sniff would otherwise use as a reason to look away.
- ``OVER_SIZE_CAP``: larger than ``MAX_BYTES``; the first ``MAX_BYTES`` are
  scanned and the rest is not, which is why it is a finding.

Every other file, which only an ``--all-text`` walk reaches, is skipped and
counted when it has any of those three properties.

Directory symlinks are followed, because ``.cursor/rules -> ../shared`` is an
ordinary layout, and a directory already walked (by its resolved path) is not
walked again, so a link loop terminates.
"""

from __future__ import annotations

import codecs
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_CHECKS: tuple[str, ...] = ("injection-structural",)

AGENT_FILE_PATTERNS: tuple[str, ...] = (
    "**/AGENTS.md",
    "**/AGENT.md",
    "**/CLAUDE.md",
    "**/CLAUDE.local.md",
    "**/GEMINI.md",
    "CONVENTIONS.md",
    ".cursorrules",
    ".windsurfrules",
    ".clinerules",
    ".clinerules/**",
    ".cursor/rules/**",
    ".windsurf/rules/**",
    ".roo/**",
    ".kiro/steering/**",
    ".continue/**",
    ".amazonq/**",
    ".junie/**",
    ".gemini/**",
    ".github/copilot-instructions.md",
    ".github/instructions/**",
    ".github/prompts/**",
    ".github/agents/**",
    ".github/chatmodes/**",
    "**/*.prompt.md",
    "**/*.prompt",
    "**/*.mdc",
    "**/SKILL.md",
    ".claude/skills/**",
    ".claude/agents/**",
    ".claude/commands/**",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".mcp.json",
    "mcp.json",
    ".vscode/mcp.json",
    ".cursor/mcp.json",
)

MAX_BYTES = 2_000_000

SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)

_SNIFF = 8192

_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


class TargetError(Exception):
    """An explicit path that does not exist: the invocation is wrong."""


@dataclass(frozen=True, slots=True)
class Target:
    path: Path
    rel: str
    forced: bool


@dataclass(frozen=True, slots=True)
class Loaded:
    text: str
    problems: tuple[str, ...]


def _translate(pattern: str) -> re.Pattern[str]:
    # `**/` is "any number of leading directories, including none", and `/**`
    # is "anything below". fnmatch's `*` already crosses `/`, so a bare `*`
    # inside a segment is narrowed to one segment here. Path.match is not used:
    # its `**` semantics changed between 3.10 and 3.13.
    out = ["(?s:"]
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("/.+")
            i += 3
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    out.append(r")\Z")
    return re.compile("".join(out))


@lru_cache(maxsize=1)
def _compiled() -> tuple[re.Pattern[str], ...]:
    return tuple(_translate(p) for p in AGENT_FILE_PATTERNS)


def is_agent_file(rel_posix: str) -> bool:
    """Whether a root-relative POSIX path is an agent instruction file."""
    return any(rx.match(rel_posix) for rx in _compiled())


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(path, root)).as_posix()
        except ValueError:
            return path.as_posix()


def _readable_text(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_BYTES:
            return False
        data = path.read_bytes()
    except OSError:
        return False
    if b"\x00" in data[:_SNIFF]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _walk(top: Path, root: Path, all_text: bool, seen: set[str]) -> tuple[list[Target], int]:
    found: list[Target] = []
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(top, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in filenames:
            path = Path(dirpath) / name
            if not path.is_file():
                continue
            rel = _rel(path, root)
            if is_agent_file(rel):
                found.append(Target(path, rel, forced=True))
            elif all_text:
                if _readable_text(path):
                    found.append(Target(path, rel, forced=False))
                else:
                    skipped += 1
    return found, skipped


def discover_counted(
    paths: Sequence[Path], *, all_text: bool, root: Path
) -> tuple[list[Target], int]:
    """``discover`` plus the number of ``--all-text`` files skipped."""
    found: dict[str, Target] = {}
    skipped = 0
    seen: set[str] = set()
    for raw in paths:
        if not raw.exists():
            raise TargetError(f"no such file or directory: {raw}")
        if raw.is_dir():
            walked, n = _walk(raw, root, all_text, seen)
            skipped += n
            for t in walked:
                found.setdefault(t.rel, t)
        else:
            rel = _rel(raw, root)
            found[rel] = Target(raw, rel, forced=True)
    return [found[k] for k in sorted(found)], skipped


def discover(paths: Sequence[Path], *, all_text: bool, root: Path) -> list[Target]:
    """The files a scan reads, sorted by root-relative path, de-duplicated."""
    return discover_counted(paths, all_text=all_text, root=root)[0]


def load(target: Target) -> Loaded:
    """Read a target leniently and say what stopped a clean read, if anything."""
    with target.path.open("rb") as fh:
        data = fh.read(MAX_BYTES + 1)
    problems: list[str] = []
    for bom, codec in _BOMS:
        if data.startswith(bom):
            # A UTF-16 or UTF-32 BOM is not a NUL problem: those NULs are the
            # encoding. The encoding itself is the finding.
            text = data[len(bom) : MAX_BYTES].decode(codec, errors="replace")
            problems.append("NOT_UTF8")
            if len(data) > MAX_BYTES:
                problems.append("OVER_SIZE_CAP")
            return Loaded(text, tuple(problems))
    if len(data) > MAX_BYTES:
        data = data[:MAX_BYTES]
        oversize = True
    else:
        oversize = False
    data = data.removeprefix(codecs.BOM_UTF8)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A cut at the size cap can split one character; that is not an
        # encoding problem, it is the cap.
        if oversize and exc.start >= len(data) - 3:
            text = data.decode("utf-8", errors="ignore")
        else:
            text = data.decode("utf-8", errors="replace")
            problems.append("NOT_UTF8")
    if "\x00" in text:
        problems.append("CONTAINS_NUL")
    if oversize:
        problems.append("OVER_SIZE_CAP")
    return Loaded(text, tuple(problems))


__all__ = [
    "AGENT_FILE_PATTERNS",
    "DEFAULT_CHECKS",
    "MAX_BYTES",
    "SKIP_DIRS",
    "Loaded",
    "Target",
    "TargetError",
    "discover",
    "discover_counted",
    "is_agent_file",
    "load",
]
