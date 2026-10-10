"""The hidden-Unicode scanner benchmark under benchmarks/unicode-scanners/.

CI does not install the four third-party scanners, so their rows are measured
at the pinned versions and recorded. Everything that CAN be recomputed here is:
the corpus is rebuilt from its builder, RESULTS.md is re-rendered from the
committed measurements, and the `jamjet-guardrails` and naive rows are
re-measured from the code on every run, so a detector change that moves our
row fails here until the benchmark is rerun and committed.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from jamjet_guardrails.scan import scan_paths

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "unicode-scanners"


def _module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", BENCH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus() -> list[dict[str, Any]]:
    lines = (BENCH / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def _measurements() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (BENCH / "results" / "measurements.json").read_text(encoding="utf-8")
    )
    return data


def _tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = json.loads((BENCH / "tools.json").read_text(encoding="utf-8"))[
        "tools"
    ]
    return tools


def test_the_corpus_file_is_what_the_builder_builds() -> None:
    built = _module("build_corpus").render()
    assert (BENCH / "corpus.jsonl").read_text(encoding="utf-8") == built


def test_the_corpus_is_ascii_on_disk() -> None:
    # Every invisible character is a JSON escape, so no tool between here and
    # a reader can drop one and GitHub shows no hidden-Unicode banner on it.
    assert (BENCH / "corpus.jsonl").read_bytes().isascii()


def test_the_corpus_has_both_labels_and_unique_ids() -> None:
    corpus = _corpus()
    ids = [c["id"] for c in corpus]
    assert len(ids) == len(set(ids))
    assert {c["expect"] for c in corpus} == {"flag", "pass"}
    assert all(c["source"] == "in-repo" and c["license"] == "Apache-2.0" for c in corpus)


def test_results_md_is_the_rendering_of_the_committed_json() -> None:
    render = _module("render")
    assert (BENCH / "RESULTS.md").read_text(encoding="utf-8") == render.render(*render._load())


def test_every_tool_was_measured_on_every_case() -> None:
    data = _measurements()
    ids = {c["id"] for c in _corpus()}
    assert set(data["decisions"]) == {t["id"] for t in _tools()}
    for tool, decisions in data["decisions"].items():
        assert set(decisions) == ids, f"{tool} was measured on a different corpus"
        assert set(decisions.values()) <= {"flag", "pass", "error"}


def _case_dir(base: Path, case: dict[str, Any]) -> Path:
    target: Path = base / case["id"] / case["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(case["text"].encode("utf-8"))
    return target


def test_the_jamjet_row_is_what_the_scan_decides_now(tmp_path: Path) -> None:
    """`jamjet-guardrails scan -- FILE` exits 1 exactly when the report has a hit."""
    recorded = _measurements()["decisions"]["jamjet-guardrails"]
    fresh = {}
    for case in _corpus():
        target = _case_dir(tmp_path, case)
        report = scan_paths([target], root=tmp_path / case["id"])
        assert report.files_scanned == 1
        fresh[case["id"]] = "flag" if report.hits else "pass"
    assert fresh == recorded, "rerun benchmarks/unicode-scanners/run.py and render.py"


def test_the_naive_row_is_what_the_rule_decides_now() -> None:
    run = _module("run")
    recorded = _measurements()["decisions"]["naive-cf-cc"]
    fresh = {c["id"]: "flag" if run.naive_flags(c["text"]) else "pass" for c in _corpus()}
    assert fresh == recorded


def test_the_versions_that_ran_are_the_versions_pinned() -> None:
    data = _measurements()
    for tool in _tools():
        if "commit" in tool:
            assert data["versions"][tool["id"]] == tool["version"], tool["id"]
    pkg = json.loads((BENCH / "tools" / "package.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in _tools()}
    assert pkg["dependencies"] == {
        "@antonsoloviev/ghostchars": by_id["ghostchars"]["version"],
        "anti-trojan-source": by_id["anti-trojan-source"]["version"],
    }
    lock = json.loads((BENCH / "tools" / "package-lock.json").read_text(encoding="utf-8"))
    assert (
        lock["packages"]["node_modules/@antonsoloviev/ghostchars"]["version"]
        == by_id["ghostchars"]["version"]
    )
    reqs = (BENCH / "tools" / "requirements.txt").read_text(encoding="utf-8")
    usg = by_id["unicode-smuggling-guard"]
    assert f"unicode-smuggling-guard=={usg['version']}" in reqs
    assert f"--hash=sha256:{usg['wheel_sha256']}" in reqs


def test_the_disclosed_single_selector_miss_is_published_as_ours() -> None:
    """The one-selector-per-character encoding is a miss corpora/NOTICE.md names.

    A benchmark this project publishes has to show it, in our own section.
    """
    cases = {c["id"]: c for c in _corpus()}
    assert cases["vs-07"]["expect"] == "flag"
    assert _measurements()["decisions"]["jamjet-guardrails"]["vs-07"] == "pass"
    results = (BENCH / "RESULTS.md").read_text(encoding="utf-8")
    ours = results.split("### jamjet-guardrails scan", 1)[1].split("\n### ", 1)[0]
    assert "Missed: `vs-07`" in ours


def test_every_case_annotated_as_a_known_jamjet_error_is_one() -> None:
    """An annotation that says "disclosed" about a case we get right is stale."""
    decisions = _measurements()["decisions"]["jamjet-guardrails"]
    for case in _corpus():
        if case["ours"]:
            wrong = (decisions[case["id"]] == "flag") != (case["expect"] == "flag")
            assert wrong, f"{case['id']} is annotated as a known error and is not one"


def test_our_section_comes_first() -> None:
    results = (BENCH / "RESULTS.md").read_text(encoding="utf-8")
    sections = re.findall(r"^### (.+)$", results, re.MULTILINE)
    assert sections[0] == "jamjet-guardrails scan"


@pytest.mark.parametrize("tool", ["jamjet-guardrails", "naive-cf-cc"])
def test_the_rows_ci_recomputes_are_marked_as_this_checkout(tool: str) -> None:
    assert _measurements()["versions"][tool].split(" at ")[0] in {"run.py", _version()}


def _version() -> str:
    from jamjet_guardrails import __version__

    return __version__


def test_the_changelog_states_the_corpus_sizes() -> None:
    corpus = _corpus()
    flag = sum(c["expect"] == "flag" for c in corpus)
    changelog = " ".join((ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split())
    assert (
        f"on {len(corpus)} files: {flag} hidden payloads and {len(corpus) - flag} hard negatives"
        in changelog
    )


def test_every_relative_link_in_results_md_resolves_beside_it() -> None:
    results = (BENCH / "RESULTS.md").read_text(encoding="utf-8")
    links = [t for t in re.findall(r"\]\(([^)]+)\)", results) if "://" not in t]
    assert links
    missing = [t for t in links if not (BENCH / t).exists()]
    assert missing == []
