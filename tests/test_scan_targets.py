"""Which files a repository scan reads, and what it refuses to skip.

The scanner's job is adversarial, so every way a file can avoid being read is a
candidate bypass. An agent instruction file, and any file named on the command
line, is read whatever its encoding, size or bytes, and anything about it that
stopped a clean read is reported rather than skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from jamjet_guardrails.scan.targets import (
    AGENT_FILE_PATTERNS,
    DEFAULT_CHECKS,
    MAX_BYTES,
    TargetError,
    discover,
    is_agent_file,
    load,
)


def _w(root: Path, rel: str, data: str | bytes = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def _rels(root: Path, *, all_text: bool = False, paths: list[Path] | None = None) -> list[str]:
    return [t.rel for t in discover(paths or [root], all_text=all_text, root=root)]


def test_the_default_check_set_is_the_structural_check_alone() -> None:
    assert DEFAULT_CHECKS == ("injection-structural",)


def test_default_walk_picks_agent_files_only(tmp_path: Path) -> None:
    for rel in [
        "AGENTS.md",
        "pkg/AGENTS.md",
        ".cursor/rules/a.mdc",
        ".github/copilot-instructions.md",
        ".mcp.json",
        "README.md",
        "src/app.py",
    ]:
        _w(tmp_path, rel)
    assert _rels(tmp_path) == [
        ".cursor/rules/a.mdc",
        ".github/copilot-instructions.md",
        ".mcp.json",
        "AGENTS.md",
        "pkg/AGENTS.md",
    ]


@pytest.mark.parametrize(
    "rel",
    [
        ".github/agents/reviewer.agent.md",
        ".github/chatmodes/plan.chatmode.md",
        ".windsurf/rules/style.md",
        ".clinerules/01-style.md",
        ".clinerules",
        ".roo/rules/a.md",
        ".kiro/steering/product.md",
        ".continue/rules/a.md",
        ".amazonq/rules/a.md",
        ".junie/guidelines.md",
        "CONVENTIONS.md",
        "AGENT.md",
        "sub/AGENT.md",
        ".claude/skills/pdf/SKILL.md",
        ".claude/skills/pdf/scripts/fill.py",
        ".claude/skills/pdf/reference.md",
        ".claude/settings.json",
        ".gemini/settings.json",
        "GEMINI.md",
        "CLAUDE.md",
        "deep/dir/CLAUDE.md",
        ".claude/agents/a.md",
        ".claude/commands/b.md",
        ".github/prompts/x.prompt.md",
        "docs/y.prompt.md",
        ".vscode/mcp.json",
    ],
)
def test_each_agent_location_is_an_agent_file(rel: str) -> None:
    assert is_agent_file(rel), rel


@pytest.mark.parametrize("rel", ["README.md", "src/app.py", "docs/guide.md", "AGENTS.md.bak"])
def test_ordinary_files_are_not_agent_files(rel: str) -> None:
    assert not is_agent_file(rel), rel


def test_every_pattern_is_reachable_by_some_path() -> None:
    # A pattern no path can match is a coverage claim that covers nothing.
    for pattern in AGENT_FILE_PATTERNS:
        probe = pattern.replace("**/", "a/").replace("/**", "/b/c.md").replace("*", "x")
        assert is_agent_file(probe), (pattern, probe)


def test_all_text_includes_every_text_file(tmp_path: Path) -> None:
    _w(tmp_path, "README.md")
    _w(tmp_path, "img.bin", b"\x00\x01")
    _w(tmp_path, "latin1.txt", b"caf\xe9")
    targets = discover([tmp_path], all_text=True, root=tmp_path)
    assert [t.rel for t in targets] == ["README.md"]


def test_explicit_file_is_always_scanned(tmp_path: Path) -> None:
    f = _w(tmp_path, "notes.txt")
    assert _rels(tmp_path, paths=[f]) == ["notes.txt"]


def test_explicit_missing_path_is_an_invocation_error(tmp_path: Path) -> None:
    with pytest.raises(TargetError):
        discover([tmp_path / "nope.md"], all_text=False, root=tmp_path)


def test_skips_vendored_directories(tmp_path: Path) -> None:
    _w(tmp_path, "node_modules/x/AGENTS.md")
    _w(tmp_path, ".git/AGENTS.md")
    _w(tmp_path, ".venv/lib/CLAUDE.md")
    assert _rels(tmp_path) == []


def test_an_agent_file_that_is_not_utf8_is_read_and_reported(tmp_path: Path) -> None:
    _w(tmp_path, "AGENTS.md", b"rules \xff\xfe here")
    (target,) = discover([tmp_path], all_text=False, root=tmp_path)
    loaded = load(target)
    assert "rules" in loaded.text and "here" in loaded.text
    assert loaded.problems == ("NOT_UTF8",)


def test_a_utf16_agent_file_is_decoded_by_its_bom_and_reported(tmp_path: Path) -> None:
    hidden = "".join(chr(0xE0000 + ord(c)) for c in "hi")
    _w(tmp_path, "AGENTS.md", "\ufeffBe kind.".encode("utf-16-le") + hidden.encode("utf-16-le"))
    (target,) = discover([tmp_path], all_text=False, root=tmp_path)
    loaded = load(target)
    assert loaded.text == "Be kind." + hidden
    assert loaded.problems == ("NOT_UTF8",)


def test_an_agent_file_with_a_nul_is_read_and_reported(tmp_path: Path) -> None:
    _w(tmp_path, "CLAUDE.md", b"before\x00after")
    (target,) = discover([tmp_path], all_text=False, root=tmp_path)
    loaded = load(target)
    assert "after" in loaded.text
    assert loaded.problems == ("CONTAINS_NUL",)


def test_an_oversized_agent_file_is_read_up_to_the_cap_and_reported(tmp_path: Path) -> None:
    _w(tmp_path, "CLAUDE.md", "a" * (MAX_BYTES + 10))
    (target,) = discover([tmp_path], all_text=False, root=tmp_path)
    loaded = load(target)
    assert len(loaded.text) == MAX_BYTES
    assert loaded.problems == ("OVER_SIZE_CAP",)


def test_a_clean_agent_file_has_no_problems_and_loses_its_utf8_bom(tmp_path: Path) -> None:
    _w(tmp_path, "AGENTS.md", b"\xef\xbb\xbfUse tabs.\n")
    (target,) = discover([tmp_path], all_text=False, root=tmp_path)
    loaded = load(target)
    assert loaded.text == "Use tabs.\n" and loaded.problems == ()


def test_an_explicit_binary_file_is_never_skipped(tmp_path: Path) -> None:
    f = _w(tmp_path, "blob.dat", b"\x00\x01\x02")
    (target,) = discover([f], all_text=False, root=tmp_path)
    assert load(target).problems == ("CONTAINS_NUL",)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlinked_rules_directory_is_followed(tmp_path: Path) -> None:
    _w(tmp_path, "shared/style.md")
    (tmp_path / ".cursor").mkdir()
    os.symlink(tmp_path / "shared", tmp_path / ".cursor" / "rules")
    assert _rels(tmp_path) == [".cursor/rules/style.md"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlink_loop_terminates(tmp_path: Path) -> None:
    _w(tmp_path, "a/AGENTS.md")
    os.symlink(tmp_path, tmp_path / "a" / "loop")
    assert _rels(tmp_path) == ["a/AGENTS.md"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlinked_agent_file_is_read(tmp_path: Path) -> None:
    _w(tmp_path, "real.txt", "x")
    os.symlink(tmp_path / "real.txt", tmp_path / "AGENTS.md")
    assert _rels(tmp_path) == ["AGENTS.md"]


# Regressions from the adversarial review of this branch. Each was a payload
# in a file an agent reads that the scan reported as clean.


def test_a_directory_other_than_the_cwd_matches_agent_files_relative_to_itself(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _w(repo, ".cursorrules")
    _w(repo, ".github/copilot-instructions.md")
    _w(repo, "README.md")
    got = [t.rel for t in discover([repo], all_text=False, root=tmp_path)]
    assert got == ["repo/.cursorrules", "repo/.github/copilot-instructions.md"]


def test_where_the_checkout_lives_does_not_make_its_files_agent_files(tmp_path: Path) -> None:
    # A checkout under a directory called `.continue` is not a rules directory:
    # its README is matched as `README.md`, relative to the walked directory.
    repo = tmp_path / ".continue" / "repo"
    _w(repo, "README.md")
    _w(repo, "AGENTS.md")
    got = [t.rel for t in discover([repo], all_text=False, root=tmp_path)]
    assert got == [".continue/repo/AGENTS.md"]


def test_nested_rule_directories_are_agent_files() -> None:
    for rel in ["pkg/.cursor/rules/r.md", "a/b/.clinerules/x.md", "svc/.claude/skills/s/x.py"]:
        assert is_agent_file(rel), rel


def test_agent_file_names_match_without_regard_to_case() -> None:
    # macOS and Windows resolve AGENTS.md to a file named agents.md.
    for rel in ["agents.md", "Claude.md", ".CursorRules", ".Claude/settings.json"]:
        assert is_agent_file(rel), rel


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlinked_rules_directory_is_read_although_its_target_was_walked_first(
    tmp_path: Path,
) -> None:
    _w(tmp_path, ".a/rules.md")
    os.symlink(tmp_path / ".a", tmp_path / ".roo")
    assert _rels(tmp_path) == [".roo/rules.md"]


def test_an_explicit_path_that_cannot_be_statted_is_an_invocation_error(tmp_path: Path) -> None:
    with pytest.raises(TargetError):
        discover([tmp_path / ("a" * 300)], all_text=False, root=tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs here")
def test_an_explicit_fifo_is_refused_rather_than_read(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "f")
    with pytest.raises(TargetError):
        discover([tmp_path / "f"], all_text=False, root=tmp_path)
