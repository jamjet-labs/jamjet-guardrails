"""Every action this repository runs is pinned to a commit, in every workflow.

A tag is a movable pointer. `actions/checkout@v7` says "whatever the owner of
that tag points it at today", and what it points at runs inside the job that
builds, tests and publishes this package: it reads the repository, it holds a
job token, and in `release.yml` it stands next to an OIDC identity that PyPI
trusts. A security library that publishes a measured number for every check it
ships and then floats its own supply chain has made the argument and not applied
it to itself, which is the same fault `pyproject.toml` records against a bare
`Apache-2.0` licence field. The count of checks is deliberately not written here:
a number in prose that counts a thing in code is a claim, and this one would go
stale on the next check that lands.

DERIVED FROM WHAT GIT TRACKS, never from a list of filenames, and that is the
whole shape of this file. The defect this repository produces more than any
other is a guard written for whichever file its author had open:
`tests/test_published_docs.py` exists because two path guards were written that
way and the other two published documents got neither. A pinning guard naming
`ci.yml` and `release.yml` would have passed on the day `codeql.yml` was added
floating a tag, and it would have kept passing.

WHY BOTH A PARSE AND A LINE SCAN. The pin and the comment naming it live on one
line, and YAML discards comments, so the parser alone cannot see half of what
is being checked. The line scan alone is worse: a `uses:` it fails to match is
a step it silently exempts, and an exemption nobody wrote down is the one that
becomes the channel. So the parser produces the authoritative inventory of
action references, the line scan produces what is written on the page, and
`test_the_line_scan_sees_exactly_the_references_the_parser_finds` requires the
two to agree before either is trusted.

What is NOT checked here: that a pinned SHA is really the commit its comment
names. That needs the network, and this suite has none. It is a manual step,
and the header comment at the top of each workflow says how to do it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

# A GitHub action reference: `owner/repo@ref`, optionally with a subdirectory,
# as `github/codeql-action/init@ref`. The ref is captured whole rather than
# matched as a SHA here, so a floating tag is a FAILING pin rather than a line
# this pattern skips.
_REFERENCE = re.compile(r"^(?P<action>[^@\s]+)@(?P<ref>[^@\s]+)$")

# A full commit SHA and nothing shorter. An abbreviated SHA is not a pin: it is
# a prefix, and a prefix can be extended.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# The trailing comment, which must open with something shaped like a release.
# `# v7.0.1` and `# 1.14.2` pass; `# checkout` does not. The requirement is not
# decoration: the SHA is what the runner reads and the comment is the only
# thing a human reads, so a pin whose comment does not name a version is a pin
# nobody can review and nobody will bump.
_VERSION_COMMENT = re.compile(r"^v?\d[\w.+-]*(?:\s.*)?$")

# The line scan. Anchored at the start of the line after optional indentation
# and an optional list dash, so a `uses:` inside a comment or a string does not
# match. Justified rather than assumed: the test named in this module's
# docstring requires what this finds to be exactly what the YAML parser finds,
# in both directions, so a line this misses and a line it invents both fail.
_USES_LINE = re.compile(
    r"^[ \t]*(?:-[ \t]+)?uses:[ \t]*(?P<ref>\S+)[ \t]*(?:#[ \t]*(?P<tag>.*?))?$"
)


def _tracked_workflows() -> list[Path]:
    """Every workflow file git tracks, read from git rather than from a glob.

    An untracked workflow does not run, and a workflow added in the same change
    as this guard is covered by it without anyone remembering to add a name.

    Restricted to `.yml` and `.yaml`, and that restriction is an exemption, so
    it is made by a PROPERTY and its reason is written down: GitHub reads only
    those two suffixes out of this directory, so a file with any other suffix
    cannot run an action and cannot float a tag. Without the filter, an ordinary
    README placed beside the workflows would fail
    `test_every_workflow_is_valid_yaml`, which is a red suite over a file that
    has nothing to do with the supply chain and exactly the sort of failure
    somebody fixes by deleting a guard. The earlier version of this docstring
    claimed to return workflow files and returned every tracked path here.
    """
    tracked = subprocess.run(
        ["git", "ls-files", ".github/workflows"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [ROOT / name for name in tracked if name.endswith((".yml", ".yaml"))]


def _parsed_references(document: Any) -> list[str]:
    """Every value under a `uses` key, anywhere in the document.

    A recursive walk rather than `jobs -> steps -> uses`, because `uses` is
    legal in three places: a step, a job calling a reusable workflow, and a
    composite action's own steps. Walking the whole tree covers all three and
    covers a fourth nobody has thought of yet.
    """
    found: list[str] = []
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "uses" and isinstance(value, str):
                found.append(value)
            else:
                found += _parsed_references(value)
    elif isinstance(document, list):
        for item in document:
            found += _parsed_references(item)
    return found


def _scanned_references(text: str) -> list[tuple[str, str | None]]:
    """Every `uses:` line, as the reference it names and the comment beside it."""
    found: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        match = _USES_LINE.match(line)
        if match is not None:
            found.append((match.group("ref"), match.group("tag")))
    return found


def _workflow_ids() -> list[str]:
    return [str(path.relative_to(ROOT)) for path in _tracked_workflows()]


def test_there_are_workflows_to_check() -> None:
    """A derived list that came back empty would make every test below vacuous.

    Both halves are asserted: that something was found, and that the two
    workflows this repository cannot operate without are among them. `ci.yml`
    is the benchmark gate and `release.yml` is the only path to PyPI, so a
    `git ls-files` that returned nothing, or that returned the issue templates
    instead, has to fail here rather than pass everything silently.
    """
    found = _tracked_workflows()
    assert found, "git tracks no workflow files; every check in this module would be vacuous"
    names = {path.name for path in found}
    assert {"ci.yml", "release.yml"} <= names, names


@pytest.mark.parametrize("workflow", _tracked_workflows(), ids=_workflow_ids())
def test_every_workflow_is_valid_yaml(workflow: Path) -> None:
    """A workflow that does not parse does not run, and GitHub says so nowhere
    a local check can see. `gh workflow view` cannot read an unpushed branch, so
    this is the only thing between a malformed file and a push."""
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{workflow.name} is not a YAML mapping: {type(document)}"
    assert "jobs" in document, f"{workflow.name} declares no jobs"


@pytest.mark.parametrize("workflow", _tracked_workflows(), ids=_workflow_ids())
def test_the_line_scan_sees_exactly_the_references_the_parser_finds(workflow: Path) -> None:
    """The guard on the guard, and the reason the line scan is allowed at all.

    Sorted lists rather than sets, so a reference used twice has to be seen
    twice. Both directions matter and they fail differently: a reference the
    scan misses is a step exempted from pinning in silence, and a reference the
    scan invents is a line the pinning checks below would judge that the runner
    never reads.
    """
    text = workflow.read_text(encoding="utf-8")
    parsed = sorted(_parsed_references(yaml.safe_load(text)))
    scanned = sorted(reference for reference, _ in _scanned_references(text))
    assert parsed == scanned, (
        f"{workflow.name}: the YAML parser finds {parsed} and the line scan finds {scanned}; "
        "the pinning checks read the line scan, so they would cover neither reliably"
    )


@pytest.mark.parametrize("workflow", _tracked_workflows(), ids=_workflow_ids())
def test_every_action_is_pinned_to_a_full_commit_sha(workflow: Path) -> None:
    """`actions/checkout@v7` is a request for whatever that tag points at today.

    A reference beginning with `./` names a directory in this repository, which
    the commit under test already pins, so it is required to be tracked instead
    of required to carry a SHA. That is the only reference shape exempt from the
    SHA rule, it is exempt by a PROPERTY rather than by name, and the property
    is checked rather than assumed.
    """
    offenders: list[str] = []
    checked = 0
    for reference, _ in _scanned_references(workflow.read_text(encoding="utf-8")):
        checked += 1
        if reference.startswith("./"):
            local = ROOT / reference[2:]
            if not local.exists():
                offenders.append(f"{reference} names a path this repository does not have")
            continue
        match = _REFERENCE.match(reference)
        if match is None:
            offenders.append(f"{reference} is not owner/repo@ref")
        elif not _COMMIT_SHA.match(match.group("ref")):
            offenders.append(f"{reference} is pinned to {match.group('ref')!r}, not a commit SHA")
    assert checked > 0, f"{workflow.name} has no `uses:` lines; this check would prove nothing"
    assert offenders == [], f"{workflow.name} runs unpinned actions: {offenders}"


@pytest.mark.parametrize("workflow", _tracked_workflows(), ids=_workflow_ids())
def test_every_pin_names_the_version_it_resolves_to(workflow: Path) -> None:
    """The other half, and the half that decides whether the first one survives.

    A pin nobody can read is a pin nobody bumps, and an action left on a two
    year old commit because its SHA said nothing is a supply chain problem in
    the other direction: the pin holds and the fixed vulnerability never lands.
    """
    offenders: list[str] = []
    checked = 0
    for reference, comment in _scanned_references(workflow.read_text(encoding="utf-8")):
        if reference.startswith("./"):
            continue
        checked += 1
        if comment is None:
            offenders.append(f"{reference} carries no version comment")
        elif not _VERSION_COMMENT.match(comment):
            offenders.append(f"{reference} carries {comment!r}, which does not name a version")
    assert checked > 0, f"{workflow.name} pins no third-party action; this check proves nothing"
    assert offenders == [], f"{workflow.name} has pins nobody can read: {offenders}"


def test_the_pinning_checks_reject_what_they_are_written_to_reject() -> None:
    """The rules, over text, so they are properties of the patterns rather than
    of whichever workflows happen to exist today.

    `_COMMIT_SHA` was written with `re.match` and no `$`, which accepts a
    40-hex prefix of anything longer and accepts uppercase never. Both halves
    are pinned here rather than left to the files above, which currently carry
    no example of either.
    """
    assert _COMMIT_SHA.match("3d3c42e5aac5ba805825da76410c181273ba90b1")
    assert not _COMMIT_SHA.match("3d3c42e"), "an abbreviated SHA is a prefix, not a pin"
    assert not _COMMIT_SHA.match("3d3c42e5aac5ba805825da76410c181273ba90b1x")
    assert not _COMMIT_SHA.match("v7.0.1")

    assert _VERSION_COMMENT.match("v7.0.1")
    assert _VERSION_COMMENT.match("1.14.2")
    assert _VERSION_COMMENT.match("v4.37.9 security-and-quality suite")
    assert not _VERSION_COMMENT.match("checkout")
    assert not _VERSION_COMMENT.match("")

    scanned = _scanned_references(
        "jobs:\n"
        "  a:\n"
        "    steps:\n"
        "      - uses: owner/repo@0123456789abcdef0123456789abcdef01234567 # v1.2.3\n"
        "      - uses: owner/other@v9\n"
        "      # - uses: owner/commented@v9\n"
        '      - run: echo "uses: owner/quoted@v9"\n'
    )
    assert scanned == [
        ("owner/repo@0123456789abcdef0123456789abcdef01234567", "v1.2.3"),
        ("owner/other@v9", None),
    ], scanned


# The tag shapes `release.yml` fires on, and the ONE job each must reach. Written
# out rather than derived from the workflow, because deriving both sides from the
# same file is a test that agrees with whatever is there.
_RELEASE_TAGS = {
    "v0.4.0": "publish",
    "v1.0.0-rc.1": "publish",
    "nemo-v0.1.0": "publish-nemo",
    "validators-v0.1.0": "publish-validators",
}


def _fires(condition: str, ref: str) -> bool:
    """Evaluate the subset of the GitHub expression language these jobs use.

    Three operators, and nothing else is accepted: `startsWith`, `contains` and
    a leading `!`, joined by `&&`. A condition using anything outside that
    raises rather than being guessed at, because a guard that silently mis-reads
    a condition it does not understand reports a routing nobody has.
    """
    for clause in condition.split("&&"):
        clause = clause.strip()
        negated = clause.startswith("!")
        clause = clause.lstrip("!").strip()
        match = re.fullmatch(r"(startsWith|contains)\(github\.ref_name, '([^']*)'\)", clause)
        if match is None:
            raise AssertionError(f"this guard cannot evaluate the clause {clause!r}")
        operator, needle = match.group(1), match.group(2)
        value = ref.startswith(needle) if operator == "startsWith" else needle in ref
        if value == negated:
            return False
    return True


def test_the_condition_evaluator_agrees_with_python_on_the_operators_it_claims() -> None:
    """The guard on the guard. An evaluator that answered True to everything
    would make the routing test below pass over any condition at all."""
    assert _fires("startsWith(github.ref_name, 'v')", "validators-v0.1.0") is True
    assert _fires("startsWith(github.ref_name, 'nemo-v')", "v0.4.0") is False
    assert _fires("!contains(github.ref_name, '-v')", "v0.4.0") is True
    assert _fires("!contains(github.ref_name, '-v')", "nemo-v0.1.0") is False
    assert _fires("startsWith(github.ref_name, 'v') && !contains(github.ref_name, '-v')", "v0.4.0")
    with pytest.raises(AssertionError):
        _fires("github.event_name == 'push'", "v0.4.0")


def test_each_release_tag_shape_fires_exactly_one_publish_job() -> None:
    """A tag must reach one publisher, and `startsWith` alone does not do that.

    `startsWith(github.ref_name, 'v')` is TRUE for `validators-v0.1.0`: the
    string starts with the letter v. So the core publish job fired on a
    validators release as well as its own. It failed rather than publishing
    anything wrong, because the version gate strips one leading `v` and compares
    `alidators-v0.1.0` against the pyproject version, but a release that always
    carries one red job is a release nobody reads the jobs of.

    Both directions are asserted: the expected job fires, AND no other one does.
    Only the second half catches this, and only because a prerelease shape is in
    the table too, so the fix cannot be "core tags contain no hyphen".

    Mutation-checked: dropping the `!contains` clause from the core job makes
    the validators row fail, and swapping any two expected jobs fails.
    """
    document = yaml.safe_load((WORKFLOWS / "release.yml").read_text(encoding="utf-8"))
    jobs = document["jobs"]
    publishers = {name: job["if"] for name, job in jobs.items() if "if" in job}
    assert len(publishers) >= 3, f"expected a publish job per distribution, found {publishers}"

    for ref, expected in _RELEASE_TAGS.items():
        firing = sorted(name for name, cond in publishers.items() if _fires(cond, ref))
        assert firing == [expected], (
            f"the tag {ref!r} fires {firing} and should fire exactly ['{expected}']"
        )
