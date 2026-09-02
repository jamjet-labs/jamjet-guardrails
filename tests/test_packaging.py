import json
import re
from importlib.metadata import metadata, requires, version
from importlib.resources import files
from pathlib import Path

import jamjet_guardrails

ROOT = Path(__file__).resolve().parent.parent


def _installed_classifiers() -> set[str]:
    """The classifiers in the BUILT metadata, not the ones pyproject.toml asks for.

    Read through `importlib.metadata` rather than by parsing TOML, and the
    reason is not style. `tomllib` arrived in 3.11 and this package's floor is
    3.10, which CI runs, so a test that reached for it would fail on the floor
    leg for a reason that has nothing to do with packaging. Reading the built
    metadata needs no parser, works on every supported version, and checks the
    artefact that is actually published.
    """
    return set(metadata("jamjet-guardrails").get_all("Classifier") or [])


def test_package_exposes_a_version() -> None:
    assert isinstance(jamjet_guardrails.__version__, str)
    assert jamjet_guardrails.__version__.count(".") == 2


def test_dunder_version_matches_the_distribution_version() -> None:
    """`__version__` and the packaged version must not drift.

    This library stamps provenance onto every decision it returns, so a version
    that disagrees with the installed distribution is a silent falsehood in the
    artefact the product sells.
    """
    assert jamjet_guardrails.__version__ == version("jamjet-guardrails")


def test_the_installed_distribution_declares_no_runtime_dependencies() -> None:
    """The core must install into a Lambda. Guard the promise, do not just state it.

    Asserts the BUILT metadata, not what pyproject.toml claims. `extra ==` markers
    are the dev extra and are expected; anything else is a runtime dependency and
    this library ships none.
    """
    declared = requires("jamjet-guardrails") or []
    runtime = [r for r in declared if "extra ==" not in r]
    assert runtime == [], f"unexpected runtime dependencies: {runtime}"


def test_every_python_classifier_is_a_version_ci_runs() -> None:
    """A classifier is a support claim. An untested version is a claim we
    cannot back, and PyPI shows it to everyone who opens the page.
    """
    claimed = {
        c.rsplit(" :: ", 1)[1]
        for c in _installed_classifiers()
        if c.startswith("Programming Language :: Python :: 3.")
    }
    assert claimed, "no Python version classifiers found; this check would prove nothing"

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # Anchored on the matrix itself rather than on every quoted "3.x" in the
    # file. A loose scan would count a version named in a comment, or the one
    # the release workflow builds on, as a leg that runs the test suite.
    matrix = re.search(r"python-version:\s*\[([^\]]*)\]", workflow)
    assert matrix is not None, "could not find the CI python-version matrix"
    tested = set(re.findall(r'"(3\.\d+)"', matrix.group(1)))
    # Also catches the YAML trap of writing the matrix unquoted, where 3.10
    # parses as the float 3.1 and the leg silently runs the wrong interpreter.
    assert tested, f"no quoted versions in the CI matrix {matrix.group(1)!r}"

    assert claimed <= tested, f"claimed but untested: {sorted(claimed - tested)}"


def test_the_installed_metadata_carries_every_classifier_pyproject_declares() -> None:
    """Without this, the check above can pass against a stale editable install.

    `pip install -e` writes the metadata once, so editing pyproject.toml and
    not reinstalling leaves the built classifiers behind while the file reads
    correctly. CI installs fresh and would catch it; this makes the local run
    mean the same thing.
    """
    declared = re.findall(
        r'"(Programming Language :: Python :: 3\.\d+)"',
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    )
    assert declared, "pyproject.toml declares no Python version classifiers"
    missing = [c for c in declared if c not in _installed_classifiers()]
    assert missing == [], f"installed metadata is stale; reinstall. Missing: {missing}"


def test_the_installed_distribution_ships_a_py_typed_marker() -> None:
    """PEP 561. Without it a consumer's type checker ignores this package.

    Note honestly what this sees. Under the editable install CI and local
    development both use, `files()` resolves to the source tree, so this
    catches the marker being deleted and NOT the wheel being built without it.
    The wheel itself is asserted in the release workflow, before publication.
    """
    assert (files("jamjet_guardrails") / "py.typed").is_file()


def test_the_typing_classifier_and_the_py_typed_marker_agree() -> None:
    """Two ways of saying the same thing, so they must not say different things.

    `Typing :: Typed` on PyPI tells a consumer their type checker will read
    this package. The marker is what makes that true. Either without the other
    is a claim with nothing behind it or a capability nobody is told about.
    """
    claimed = "Typing :: Typed" in _installed_classifiers()
    shipped = (files("jamjet_guardrails") / "py.typed").is_file()
    assert claimed == shipped, (
        f"the Typing :: Typed classifier is {claimed} and the py.typed marker is {shipped}"
    )


def test_source_never_imports_jamjet() -> None:
    """Zero JamJet dependency is a hard constraint. Prove it over the source tree."""
    offenders = []
    scanned = 0
    for path in (ROOT / "src").rglob("*.py"):
        scanned += 1
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import jamjet", "from jamjet")) and (
                "jamjet_guardrails" not in stripped
            ):
                offenders.append(f"{path.name}: {stripped}")
    assert scanned > 0, f"scanned no files under {ROOT / 'src'}; this guard proves nothing"
    assert offenders == [], f"these import JamJet: {offenders}"


def test_no_description_a_reader_receives_promises_a_kind_we_do_not_ship() -> None:
    """WIDENED: it checked the PyPI Summary only, and the same claim shipped next door.

    The Summary said "constraints and classifiers" while every registered
    detector declares `kind="constraint"`. A classifier is a real thing in this
    library's type system, so a reader takes that as a shipped feature rather
    than a design intention. The commit that fixed the Summary left the
    identical sentence as the package docstring one file over, where
    `help(jamjet_guardrails)` prints it: the argument applied to the case in
    hand and not to the case one step over, in the last commit on the branch.

    So the domain is both strings a reader is handed, not the one that happened
    to be noticed. Both are read from the built or imported artifact rather
    than from a source file: `tomllib` is 3.11+ and the floor is 3.10, and the
    installed metadata is what PyPI actually renders.

    Checked against the registry rather than a word list, so it stays true the
    day a classifier is added.
    """
    from importlib.metadata import metadata

    import jamjet_guardrails
    from jamjet_guardrails.detectors import AVAILABLE, build
    from jamjet_guardrails.eval.fixtures import options_for

    described = {
        "the PyPI Summary": metadata("jamjet-guardrails")["Summary"],
        "the package docstring": jamjet_guardrails.__doc__ or "",
    }
    assert all(described.values()), f"nothing to check in {described}"
    # A registry value is a factory whose options are the check's own, so
    # constructing it bare (`cls()`) is a call this package refuses on
    # purpose: `rules` raises GuardrailUnavailableError with no arguments.
    kinds = {build(name, **options_for(name)).kind for name in AVAILABLE}
    if "classifier" not in kinds:
        offenders = [where for where, text in described.items() if "classifier" in text.lower()]
        assert offenders == [], f"{offenders} promise a classifier; none is registered"


def test_the_declared_licence_covers_every_licence_the_corpora_carry() -> None:
    """The licence field is derived from what actually ships, not restated.

    The distribution declared `Apache-2.0` while the sdist redistributed 300
    CC-BY-4.0 rows. `corpora/NOTICE.md` travelled beside them, so the legal
    chain held; the licence FIELD did not, and it is the one line most
    consumers read. `docs/conformance.md` rejects two corpora with the argument
    that "an Apache-2.0 tag downstream does not cure a share-alike upstream,
    and no licence field anywhere in that chain reveals it", which is this
    project's own argument turned on it.

    The domain is READ from the corpora rather than listed here, so adding a
    corpus under a third licence fails until the expression names it. Every
    case in a corpus carries its own `license`, and `load_corpus` already
    refuses a file whose rows disagree.
    """
    from importlib.metadata import metadata

    declared = metadata("jamjet-guardrails")["License-Expression"]
    shipped = sorted(
        {
            json.loads(line)["license"]
            for path in sorted((ROOT / "corpora").rglob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    assert shipped, "no corpus licences found; this check would prove nothing"
    missing = [licence for licence in shipped if licence not in declared]
    assert missing == [], (
        f"the distribution declares {declared!r} but ships corpora licensed {missing}"
    )


def test_the_ship_bar_was_recorded_before_any_model_existed() -> None:
    """The bar is a commitment, and a commitment made after the result is none.

    Enforced by ordering rather than by trust: `training/ship_bar.json` is
    committed in the first task of the stage, before any training run, and
    `training/artifacts/` is where a trained model would land. A bar sitting
    beside a model artifact is a bar that could have been written after seeing
    the score, which is the one thing it cannot be and still mean anything.

    This is the shape check. `tests/test_ship_bar.py` holds the contents to the
    harness the numbers came from.

    `tomllib` is deliberately absent: the brief wrote this file's neighbours
    over it, and it arrived in 3.11 while this package's floor is 3.10, which
    CI runs. The two properties that draft asserted are already asserted here
    and in tests/test_benchmarks.py against the BUILT metadata and the wheel
    target, which is a stronger reading of the same claim.
    """
    bar = json.loads((ROOT / "training" / "ship_bar.json").read_text(encoding="utf-8"))
    for key in (
        "metric",
        "reference_model",
        "reference_revision",
        "reference_score",
        "reference_measured_by",
        "held_out_corpus",
        "harness",
        "our_minimum",
        "structural_floor",
        "structural_floor_level",
        "recorded_utc",
        "rationale",
    ):
        assert key in bar, f"ship_bar.json is missing {key!r}"
    assert isinstance(bar["our_minimum"], (int, float))
    assert 0.0 < bar["our_minimum"] <= 1.0
    # The reference score must be OURS, measured on the held-out corpus with the
    # same harness. A figure quoted from a vendor's own evaluation compares two
    # datasets and is the flaw this bar was rewritten to remove.
    assert bar["reference_measured_by"] == "this-repo", (
        "the reference score must be measured by us on the held-out corpus, "
        "not quoted from a published leaderboard"
    )
    # The ordering, asserted rather than described. This used to be "no model
    # exists yet", which witnessed it exactly once: the day a model was trained
    # the assertion became a failure whose obvious fix was to delete the line.
    # What survives a model existing is that the bar is not re-recorded beside
    # one. `training/artifacts/` holds the run records, and nothing in it may
    # be a second copy of the bar for a later reader to prefer.
    artifacts = ROOT / "training" / "artifacts"
    for path in sorted(artifacts.glob("*.json")) if artifacts.is_dir() else []:
        beside = json.loads(path.read_text(encoding="utf-8"))
        assert "our_minimum" not in beside, (
            f"{path.name} carries our_minimum, so a ship bar has been written beside the "
            "model it judges; the one in training/ship_bar.json is the only one"
        )
    # `tests/test_ship_bar.py` holds the bar's own bytes to the digest they had
    # before any model existed, and holds every run record's trained_utc after
    # the bar's recorded_utc. That is the ordering itself; this is the rule
    # that stops a second bar appearing where nothing would compare it.


# ==========================================================================
# No credential-shaped literal anywhere reads as a live one.
# ==========================================================================

# The four families `corpora/NOTICE.md` names. AWS, JWT and PEM are excluded
# because the notice makes a different promise about each of them and each is
# checked where that promise lives: the AWS value is Amazon's own published
# example, the JWTs are signed over random bytes, and the PEM bodies are base64
# of random bytes rather than DER.
_MARKED_FAMILIES = frozenset({"ANTHROPIC_KEY", "GITHUB_TOKEN", "OPENAI_KEY", "SLACK_TOKEN"})

_MARKERS = ("EXAMPLEONLY", "EXAMPLE_ONLY", "notarealtoken", "notarealkey")

# The bodies that predate the rule and carry no marker, each one inspected. They
# are listed exactly rather than admitted by a shape test, because a shape test
# here is a guess at "what looks synthetic" and the first body that slips past
# such a guess is the one worth catching. Anything not on this list and not
# marked fails, including a benign addition: the friction is the point, since
# the alternative is a reviewer deciding case by case whether a random-looking
# body is real.
_INSPECTED_UNMARKED = frozenset(
    {
        # A sequential alphabet with a digit run behind it. The README quickstart
        # and the two chain tests that mirror it.
        "sk-abcdefghijklmnopqrstuvwxyz012345",
        # The 16-character prefix a split token leaves behind, quoted in the
        # docstring of the composition-leak test as the stump `secrets` matched.
        "xoxb-0000000000-",
        # `tests/test_secrets.py` fixtures: a counted digit run, Amazon's own
        # published example key nested inside a Slack shape, and the standard
        # HS256 header. Each is chosen to be recognisable to a reader, which is
        # the property that also makes it unmistakable for a live token.
        "xoxb-123456789012-abcdefghijkl",
        "xoxb-AKIAIOSFODNN7EXAMPLE-abcdefghijkl",
        "xoxb-eyJhbGciOiJIUzI1NiJ9",
        "xoxp-123456789012-123456789012-1234567890123-",
    }
)


def _tracked_text() -> list[tuple[str, str]]:
    """Every tracked file that decodes as UTF-8, with its text.

    Derived from git rather than from a list of directories, for the reason
    `tests/test_published_docs.py` gives: the guard that reads a list covers the
    files its author had open, and this repository has produced that defect more
    than any other. A file added later is covered without anyone remembering.
    """
    import subprocess

    names = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    out = []
    for name in names:
        try:
            out.append((name, (ROOT / name).read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return out


def test_the_credential_scan_has_something_to_scan() -> None:
    """A derived list that came back empty makes the guard below vacuous."""
    files = _tracked_text()
    assert len(files) >= 50, f"expected the tracked tree, found {len(files)} files"
    assert any(name == "README.md" for name, _ in files)


def test_no_credential_shaped_literal_in_the_repository_reads_as_a_live_one() -> None:
    """`corpora/NOTICE.md` promises this, and it used to be true of three files.

    The notice said the GitHub, OpenAI, Anthropic and Slack tokens carry
    `EXAMPLEONLY` or `notarealtoken` inside their own bodies. That sentence sat
    under a heading about the three corpus files and was true there. Two bodies
    outside it were not covered by anything: a canonical Slack bot token with a
    random 24-character secret in the docstring of `GuardrailChain`, and a
    36-character GitHub token body in the docstring of `_scan`. Both are in the
    SHIPPED package, so both went out in every wheel, and a scanner or a reader
    meeting either one has no way to tell it from a live credential.

    The detector this package ships is what finds them, which is the cheapest
    honest oracle available: the thing that decides what a credential looks like
    for callers decides it here too. Where it fires, the body must say what it
    is.
    """
    from jamjet_guardrails import Context
    from jamjet_guardrails.detectors.secrets import SecretsGuardrail

    guardrail = SecretsGuardrail()
    context = Context(direction="output", origin="model")
    unmarked: list[tuple[str, str, str]] = []
    for name, text in _tracked_text():
        for finding in guardrail.check(text, context).findings:
            if finding.type not in _MARKED_FAMILIES:
                continue
            span = finding.span
            # Optional in the type, never absent for a credential finding. Asserted
            # rather than skipped: `continue` here would let a finding whose span
            # went missing pass the guard silently, which is the fail-open shape
            # this library exists to refuse.
            assert span is not None, f"{finding.type} finding in {name} carries no span"
            body = text[span[0] : span[1]]
            if any(marker in body for marker in _MARKERS):
                continue
            if body in _INSPECTED_UNMARKED:
                continue
            unmarked.append((name, finding.type, body))
    assert unmarked == [], (
        "credential-shaped literals with no marker in their bodies and no entry in "
        f"_INSPECTED_UNMARKED: {unmarked}"
    )


def test_every_inspected_unmarked_body_is_still_in_the_tree() -> None:
    """An exemption nothing uses is an exemption nobody re-reads.

    The list above is the part of this guard a future change routes around, so
    it has to shrink when the tree does. A body deleted from the repository and
    left on the list widens the hole for whatever is written next.

    Its own file is excluded from the corpus, and that exclusion is the check
    rather than a detail. The list above quotes every body it exempts, so a
    staleness test that read the tracked tree including this file found each
    body in the list itself and stayed green on a body deleted everywhere else.
    That is this repository's own recorded failure, a guard reproducing what it
    excludes, arriving a second time; mutating the tree and watching this test
    pass anyway is what found it.
    """
    corpus = "\n".join(text for name, text in _tracked_text() if name != "tests/test_packaging.py")
    stale = sorted(body for body in _INSPECTED_UNMARKED if body not in corpus)
    assert stale == [], f"_INSPECTED_UNMARKED names bodies no longer in the tree: {stale}"
