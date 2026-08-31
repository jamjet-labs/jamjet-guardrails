"""The README is checked against the repository, not against itself.

This is the only page here whose reader has not already decided to trust it,
and the library's whole pitch is published, falsifiable numbers. A claim on
this page that nothing checks is the most expensive kind this project can
ship: it undoes the thing being sold.

So almost nothing below asserts that a sentence is PRESENT and calls that a
test. Each check recomputes the fact from the code, from the built packaging
metadata or from the corpora, and then requires the README to say that. Both
directions, because a claim can go stale from either end: a sentence that
drifts fails, and so does a fact that moves while the sentence stands.

Some claims genuinely cannot be recomputed: that this is a library rather than
a service, that the checks here are constraints rather than classifiers. This
file does not pretend otherwise. The quickstart's output block is executed
rather than read, so the spans it prints are the spans the library produces.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import hashlib
import io
import json
import re
from importlib.metadata import metadata, requires
from pathlib import Path
from typing import Any, get_args

import pytest

from jamjet_guardrails import (
    Context,
    Decision,
    Direction,
    GuardrailChain,
    GuardrailUnavailableError,
    Kind,
    Provenance,
    Verdict,
    build,
    build_chain,
    combine,
    saw,
)
from jamjet_guardrails.detectors import AVAILABLE
from jamjet_guardrails.detectors.injection_structural import INJECTION_TYPES
from jamjet_guardrails.detectors.pii import PII_TYPES
from jamjet_guardrails.detectors.secrets import SECRET_TYPES

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CORPORA = ROOT / "corpora"

# The distribution name is written once. `metadata()` raises
# PackageNotFoundError if the repository ever builds something else, which is
# the rename this constant would otherwise hide.
DIST = "jamjet-guardrails"

# The detector each check reports its findings under. Keyed by registry name so
# that registering a new check makes the parametrised tests below demand an
# entry here rather than quietly leaving the new row unchecked.
TYPES: dict[str, frozenset[str]] = {
    "injection-structural": INJECTION_TYPES,
    "pii": PII_TYPES,
    "secrets": SECRET_TYPES,
}

BANNED = [
    "production-ready",
    "production-grade",
    "enterprise-grade",
    "battle-tested",
    "blazing",
    "seamless",
    "robust",
    "coming soon",
    "roadmap",
]

# Top-level stdlib and third-party names that open a socket. "No network calls"
# is a claim about the shipped source, so it is checked over the shipped source.
NETWORK_MODULES = frozenset(
    {
        "ftplib",
        "http",
        "httpx",
        "imaplib",
        "poplib",
        "requests",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "telnetlib",
        "urllib",
        "urllib3",
        "webbrowser",
        "xmlrpc",
    }
)

# Credential prefixes the README names as shapes this library does not match.
# A published miss is a claim like any other: it has to still be a miss.
KNOWN_MISSES = [
    ("github_pat_", "token github_pat_11ABCDEFG0abcdefghijklm_" + "A" * 59),
    ("xapp-", "token xapp-1-A01B2C3D4E5-1234567890123-" + "b" * 64),
]


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """The body of one H2, up to the next one.

    Asserts the heading is present rather than letting `split(...)[1]` raise an
    IndexError, because a missing section and a section whose content is wrong
    are different failures and only one of them is this helper's fault.
    """
    text = _text()
    assert heading in text, f"the README has no {heading!r} section"
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def _flat(text: str) -> str:
    """One string with every run of whitespace collapsed to a single space.

    Every prose claim below is matched against this rather than against the raw
    file. A line rewrap changes nothing a reader sees, so it must not fail a
    check, and a sentence broken across two lines must not slip past one. The
    checks-table parse deliberately does NOT use this: it needs line structure.
    """
    return " ".join(text.split())


def _blocks(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", _text(), re.DOTALL)


def _corpus_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _third_party_corpora() -> dict[Path, list[dict[str, Any]]]:
    """Every corpus file whose cases did not come from this repository.

    Derived from the `source` field rather than from the filename, because the
    filename is a convention and the field is what the loader and the scores
    key on.
    """
    found = {}
    for path in sorted(CORPORA.rglob("*.jsonl")):
        cases = _corpus_cases(path)
        if cases and cases[0]["source"] != "in-repo":
            found[path] = cases
    return found


def _results() -> list[dict[str, Any]]:
    rows = json.loads((ROOT / "benchmarks.json").read_text(encoding="utf-8"))["results"]
    assert rows, "benchmarks.json has no results; the checks over it would prove nothing"
    return list(rows)


# ==========================================================================
# The quickstart: executed, not read.
# ==========================================================================


def test_the_quickstart_runs_and_prints_what_the_readme_says_it_prints() -> None:
    """The most-read code in the repository and the only piece nothing compiles.

    If this fails, the README is what is wrong. Paste the real output.
    """
    code = _blocks("python")
    assert len(code) == 1, f"expected exactly one python block, found {len(code)}"
    expected = _blocks("text")
    assert len(expected) == 1, f"expected exactly one output block, found {len(expected)}"

    namespace: dict[str, Any] = {"__name__": "__readme__"}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        # S102 is suppressed below because executing the snippet IS the check. A
        # quickstart that is read rather than run is the one piece of code in this
        # repository nothing compiles, which is what this test exists to stop.
        exec(compile(code[0], "README.md", "exec"), namespace)  # noqa: S102

    assert buffer.getvalue() == expected[0]


# ==========================================================================
# Copy that cannot be checked, and links that can.
# ==========================================================================


def test_the_readme_makes_no_claim_it_cannot_check() -> None:
    text = _flat(_text()).lower()
    found = [phrase for phrase in BANNED if phrase in text]
    assert found == [], f"unfalsifiable or forward-looking copy in README: {found}"


def test_the_readme_has_no_em_dash() -> None:
    text = _text()
    assert "—" not in text
    assert " -- " not in text


# Markdown link targets, plus repository paths cited in backticks. Both are
# citations and both rot the same way; only the first form is a link.
_CITED_PATH = re.compile(r"`((?:corpora|tests|scripts|src/jamjet_guardrails|docs)/[^`]*)`")


def test_every_repository_path_the_readme_cites_exists() -> None:
    """Presence is not truth.

    An earlier task in this project shipped a test asserting a published
    paragraph EXISTED while the paragraph said something false, so every
    citation here is resolved against the filesystem. A trailing slash means a
    directory.
    """
    text = _text()
    links = [
        target
        for target in re.findall(r"\]\(([^)]+)\)", text)
        if "://" not in target and not target.startswith(("#", "mailto:"))
    ]
    cited = sorted(set(links) | set(_CITED_PATH.findall(text)))
    assert cited, "the README cites nothing in the repository; this check would prove nothing"
    missing = [
        target
        for target in cited
        if not (
            (README.parent / target).is_dir()
            if target.endswith("/")
            else (README.parent / target).is_file()
        )
    ]
    assert missing == [], f"README cites paths that do not exist: {missing}"


# ==========================================================================
# The published numbers.
# ==========================================================================


def test_the_readme_quotes_the_current_headline_table() -> None:
    """CI diffs this block against BENCHMARKS.md, so a README edited by hand
    breaks the build. Checking it here means the failure arrives at the desk of
    whoever edited it rather than on a runner ten minutes later."""

    def table(text: str) -> list[str]:
        rows: list[str] = []
        for line in text.splitlines():
            if line.startswith("| Check |") or (rows and line.startswith("|")):
                rows.append(line)
            elif rows:
                break
        return rows

    generated = table((ROOT / "BENCHMARKS.md").read_text(encoding="utf-8"))
    assert generated, "BENCHMARKS.md has no headline table"
    assert table(_text()) == generated


def test_the_readme_points_at_the_published_numbers() -> None:
    text = _flat(_text())
    assert "BENCHMARKS.md" in text
    assert "docs/conformance.md" in text


def test_the_misses_the_readme_says_are_published_really_are() -> None:
    """ "The misses are published beside the scores" is a claim about another
    file, and it is the one this project stakes the most on. Derived from the
    stored failures rather than asserted, so a report that stopped emitting the
    section, or a run with nothing left to publish, both land here.
    """
    assert any(row["failures"] for row in _results()), (
        "no misses are recorded at all, so the README claims to publish nothing"
    )
    assert "## Worst misses" in (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8")


def test_the_readme_names_and_attributes_every_third_party_corpus() -> None:
    """CC-BY-4.0 attribution is a licence condition, not a courtesy, and the
    obvious way to lose it is an editing pass that tightens the copy. Any
    corpus whose licence or source is not this repository's own must be named
    where the numbers are.
    """
    sources = {case["source"] for path in CORPORA.rglob("*.jsonl") for case in _corpus_cases(path)}
    licences = {
        case["license"] for path in CORPORA.rglob("*.jsonl") for case in _corpus_cases(path)
    }
    assert sources and licences, "no corpus cases found; this check would prove nothing"
    text = _flat(_text())
    for licence in sorted(licences - {"Apache-2.0"}):
        assert licence in text, f"{licence} corpus is used but not attributed in the README"
    for source in sorted(sources - {"in-repo"}):
        assert source in text, f"{source} is measured but not named in the README"


def test_the_stress_set_claim_is_true_of_the_published_numbers() -> None:
    """The README explains a low precision figure by saying the corpus behind
    it is a deliberate stress set and scores lower than the third-party one. If
    that stops being so, the paragraph is no longer an explanation."""
    precision = {(row["detector"], row["corpus_source"]): row["precision"] for row in _results()}
    in_repo = precision[("pii", "in-repo")]
    third_party = [
        p
        for (detector, source), p in precision.items()
        if detector == "pii" and source != "in-repo"
    ]
    assert third_party, "no third-party pii row; the README's comparison has nothing to compare"
    assert in_repo < min(third_party), (
        f"the README says the in-repo corpus scores lower precision, but {in_repo} is not "
        f"below {min(third_party)}"
    )


def test_the_row_count_the_readme_states_is_the_size_of_the_third_party_corpus() -> None:
    corpora = _third_party_corpora()
    assert len(corpora) == 1, f"expected one third-party corpus, found {sorted(corpora)}"
    ((_, cases),) = corpora.items()
    assert f"{len(cases)} rows" in _flat(_text()), (
        f"the README does not state the third-party corpus size, which is {len(cases)} rows"
    )


def test_the_readme_says_the_secrets_numbers_are_self_graded_because_they_are() -> None:
    """The absent row is a claim too, and it is the one a reader will not
    notice going stale in the direction that matters."""
    third_party_checks = {path.parent.name for path in _third_party_corpora()}
    assert "secrets" not in third_party_checks, (
        "a third-party secrets corpus now exists; the README says there is none"
    )
    assert "There is no third-party secrets corpus." in _flat(_text())


# ==========================================================================
# Packaging claims: read from the built metadata, not from pyproject.toml.
# tomllib is 3.11+ and this package's floor is 3.10, which CI runs.
# ==========================================================================


def test_the_install_command_names_the_distribution_this_repo_builds() -> None:
    assert f"pip install {metadata(DIST)['Name']}" in _flat(_text())


def test_the_python_floor_the_readme_states_is_the_one_the_package_declares() -> None:
    declared = metadata(DIST)["Requires-Python"]
    floor = re.fullmatch(r">=(\d+\.\d+)", declared)
    assert floor is not None, f"unexpected Requires-Python {declared!r}; the README claim is unread"
    assert f"Python {floor.group(1)} and above" in _flat(_text())


def test_the_no_dependencies_claim_is_what_the_installed_metadata_says() -> None:
    """Both directions. A dependency added later must break the sentence."""
    runtime = [r for r in (requires(DIST) or []) if "extra ==" not in r]
    assert ("No dependencies." in _flat(_text())) == (runtime == []), (
        f"the README's dependency claim disagrees with the built metadata: {runtime}"
    )


def test_the_licence_the_readme_states_is_the_one_the_package_declares() -> None:
    declared = metadata(DIST)["License-Expression"]
    assert declared, "the built metadata declares no licence expression"
    assert declared in _flat(_section("## Licence"))


def test_the_no_network_claim_holds_over_every_module_that_ships() -> None:
    offenders = []
    scanned = 0
    for path in (ROOT / "src").rglob("*.py"):
        scanned += 1
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [
                f"{path.name}: {name}" for name in names if name.split(".")[0] in NETWORK_MODULES
            ]
    assert scanned > 0, f"scanned no files under {ROOT / 'src'}; this guard proves nothing"
    assert offenders == [], f"the README claims no network calls, but these import one: {offenders}"
    assert "No network calls." in _flat(_text())


# ==========================================================================
# The checks table: every column recomputed from the check it describes.
# ==========================================================================

# The name class carries `-` as well as `_`. `injection-structural` is the first
# registry key that is not a bare Python identifier, and the failure it caused is
# the quiet kind: `[a-z_]+` did not reject that row, it matched nothing at all, so
# the table test reported the check undocumented against a README that documented
# it correctly. Widening the class is what keeps the parse and the eye agreeing.
_ROW = re.compile(r"^\|\s*`([a-z_-]+)`\s*\|([^|]*)\|([^|]*)\|([^|]*)\|\s*$", re.MULTILINE)


def _checks_table() -> dict[str, tuple[str, set[str], set[str]]]:
    rows = {
        name: (
            kind.strip(),
            {direction.strip() for direction in runs_on.split(",")},
            set(re.findall(r"`([A-Z_]+)`", types)),
        )
        for name, kind, runs_on, types in _ROW.findall(_section("## The checks"))
    }
    assert rows, "no rows parsed out of the README's checks table"
    return rows


def test_the_checks_table_lists_every_check_the_registry_can_build() -> None:
    """Both directions: a check registered and undocumented is one a reader
    never learns exists, and a row for a check `build` refuses is a row that
    sends them to a name that raises."""
    assert set(_checks_table()) == set(AVAILABLE)


@pytest.mark.parametrize("name", sorted(AVAILABLE))
def test_each_row_states_the_kind_and_directions_that_check_declares(name: str) -> None:
    kind, runs_on, _ = _checks_table()[name]
    guardrail = build(name)
    assert kind == guardrail.kind
    assert runs_on == set(guardrail.directions)


@pytest.mark.parametrize("name", sorted(AVAILABLE))
def test_each_row_lists_exactly_the_types_that_check_can_report(name: str) -> None:
    assert name in TYPES, f"no type set is known for {name!r}, so its README row is unchecked"
    _, _, listed = _checks_table()[name]
    assert listed == set(TYPES[name])


@pytest.mark.parametrize(("prefix", "sample"), KNOWN_MISSES, ids=[p for p, _ in KNOWN_MISSES])
def test_the_prefixes_the_readme_names_as_misses_are_still_misses(prefix: str, sample: str) -> None:
    """A published false negative is a claim, and the expensive direction is
    the one where it quietly stops being true: the README would then be telling
    a reader to add a second tool they no longer need."""
    verdict = build("secrets").check(sample, Context(direction="output", origin="model"))
    assert verdict.decision == "allow", (
        f"{prefix} is now matched as {[f.type for f in verdict.findings]}; the README calls it a miss"
    )
    assert f"`{prefix}`" in _flat(_section("## The checks"))


# ==========================================================================
# What the library does with a verdict, and which way it fails.
# ==========================================================================


def test_the_readme_states_the_combination_order_combine_implements() -> None:
    """The order is recovered from behaviour, never copied from the page.

    Rank each decision by how many decisions it survives combination with,
    which recovers the severity order without reading `_SEVERITY`.
    """
    values: tuple[Decision, ...] = get_args(Decision)
    rank = {value: sum(1 for other in values if combine(value, other) == value) for value in values}
    assert len(set(rank.values())) == len(values), f"combination order is not total: {rank}"
    order = " > ".join(f"`{value}`" for value in sorted(values, key=lambda value: -rank[value]))
    assert order in _flat(_text()), (
        f"the README does not state the order combine implements: {order}"
    )


def test_the_fields_the_readme_names_are_fields_these_types_have() -> None:
    verdict = {field.name for field in dataclasses.fields(Verdict)}
    assert {"decision", "findings", "provenance", "saw"} <= verdict, verdict
    provenance = {field.name for field in dataclasses.fields(Provenance)}
    assert {"detector", "version"} <= provenance, provenance


def test_the_hash_the_readme_names_is_the_one_saw_computes() -> None:
    content = "alice@example.com"
    assert saw(content) == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert "SHA-256" in _flat(_text())


class _Exploding:
    """A detector that dies, so the direction it fails in can be measured."""

    name: str = "exploding"
    version: str = "0.0.0"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input", "output"})

    def check(self, content: str, context: Context) -> Verdict:
        raise RuntimeError("this detector is broken")


def test_the_direction_a_broken_check_fails_in_is_the_one_the_readme_publishes() -> None:
    """Derived, then required of the page. Flip the chain to fail open and this
    looks for the sentence that would then be true, does not find it, and
    fails."""
    result = GuardrailChain([_Exploding()]).run(
        "anything at all", Context(direction="output", origin="model")
    )
    assert f"`{result.decision}`, never" in _flat(_section("## How it fails"))
    # The rest of the same sentence: the error is recorded on that check's
    # verdict, and the chain carries on rather than abandoning the run.
    assert [verdict.error is not None for verdict in result.verdicts] == [True]


def test_the_error_a_missing_check_raises_is_the_one_the_readme_names() -> None:
    missing = "no-such-check"
    assert missing not in AVAILABLE, f"{missing!r} is registered; this check would prove nothing"
    with pytest.raises(GuardrailUnavailableError) as caught:
        build_chain(["pii", missing])
    assert type(caught.value).__name__ in _flat(_section("## How it fails"))


def test_an_empty_list_of_checks_is_refused_as_the_readme_says() -> None:
    """The sentence after the bullet. An empty chain allows everything, which
    is the same fault as a missing check wearing different clothes."""
    with pytest.raises(GuardrailUnavailableError):
        build_chain([])
    assert "An empty list of checks is refused" in _flat(_section("## How it fails"))
