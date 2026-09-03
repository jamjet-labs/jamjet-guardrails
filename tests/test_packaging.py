import json
import re
import subprocess
import sys
import tarfile
import zipfile
from importlib.metadata import metadata, requires, version
from importlib.resources import files
from pathlib import Path

import pytest

import jamjet_guardrails

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DATA = ROOT / "template-data"


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


def test_the_changelogs_newest_heading_matches_the_package_version() -> None:
    """The guard CHANGELOG.md has never had.

    `CHANGELOG.md`'s own preamble says a release that moves a published number
    says so in its entry, which only means something if the entry is labelled
    with the version that shipped it. Nothing before this test checked that an
    `Unreleased` heading gets renamed when the version it describes is cut, so a
    bumped `pyproject.toml` and an unrenamed heading could ship together and
    nothing would notice. `[Unreleased]` is skipped by the pattern itself, not
    by name: it never matches `\\d+\\.\\d+\\.\\d+`, so the first version-shaped
    heading found is the newest RELEASED one regardless of whether an
    `Unreleased` section still sits above it.
    """
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.MULTILINE)
    assert headings, "no version-shaped heading (## [X.Y.Z]) found in CHANGELOG.md"
    assert headings[0] == jamjet_guardrails.__version__


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


def _built_sdist_names() -> list[str]:
    """Build the sdist with the declared backend and list what is inside it.

    Through `hatchling.build.build_sdist`, the PEP 517 hook `pyproject.toml`
    already names as this project's backend, rather than through the `build`
    frontend. Same artifact, no isolated environment to create and nothing to
    download, and it takes under half a second, which is what makes it
    affordable on every CI leg.

    A subprocess rather than an in-process call, because the hook reads
    `pyproject.toml` from the working directory and the test suite's is not
    guaranteed to be the repository root.
    """
    import subprocess
    import sys
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory() as out:
        built = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, hatchling.build as b; print(b.build_sdist(sys.argv[1]))",
                out,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        name = built.stdout.strip().splitlines()[-1]
        with tarfile.open(Path(out) / name) as archive:
            return archive.getnames()


def test_the_sdist_carries_nothing_from_the_adapter_packages() -> None:
    """The exclusion in pyproject.toml, held against the artifact it describes.

    MEASURED before it was written: hatchling 1.32.0 includes `packages/` in the
    sdist by DEFAULT, so this is not a precaution. Without
    `[tool.hatch.build.targets.sdist] exclude`, the core source distribution
    carried 25 files belonging to the two adapter distributions, including their
    own pyproject.toml files, each declaring runtime dependencies on a framework.

    The sdist is the evidence for the zero-dependency claim: it is what a
    distribution reviewer, a licence scanner or a rebuild reads to see what this
    package IS. And the adapters are separately released distributions with their
    own tags and their own PyPI names, so nothing about them belongs in this one.
    """
    names = _built_sdist_names()
    assert names, "the sdist is empty"
    assert any(name.endswith("src/jamjet_guardrails/__init__.py") for name in names), (
        f"this does not look like this project's sdist: {names[:5]}"
    )
    # The other half, and not decoration: if `packages/` ever stops existing, the
    # assertion below passes over an empty question and the exclusion could be
    # deleted without anything noticing.
    adapters = sorted((ROOT / "packages").glob("*/pyproject.toml"))
    assert adapters, f"no adapter packages under {ROOT / 'packages'}; this guard is vacuous"
    stray = [name for name in names if "/packages/" in name]
    assert stray == [], f"the core sdist carries adapter files: {stray}"


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


def test_the_declared_licence_covers_every_licence_template_data_carries() -> None:
    """The third half, and the one that was missing while the material shipped.

    `template-data/` redistributes tokenizer configuration read out of eight
    model repositories, four licence families between them, and the licence
    field named none of them for as long as the files were in the sdist. The
    guard above could not see it, for the same reason it could not see the
    Unicode data: this is not a corpus, it carries no `license` row and no
    loader refuses it. So the suite was green over exactly the gap this file
    exists to close, which is the defect `tests/test_published_docs.py` was
    written about, arriving in a third place.

    DERIVED from the marker table's own `Source` records rather than from a
    list here. A source added to `scripts/generate_template_markers.py` under a
    fifth licence fails this until the expression names it, and no one has to
    remember.

    Two of the four families have no SPDX short identifier, because they are
    vendor community licences with use restrictions rather than OSI-approved
    terms. SPDX's own mechanism for that is a `LicenseRef-` identifier, and the
    mapping from the source's own licence string to that identifier is the one
    thing this test cannot derive: it is a judgement about which published
    licence a name refers to. It is written out here, and a licence string this
    mapping does not know fails rather than being skipped, because a source
    whose licence nobody classified is exactly the one worth stopping for.
    """
    from importlib.metadata import metadata

    from jamjet_guardrails.detectors._template_markers import HTML_ELEMENT_SOURCE, SOURCES

    spdx = {
        "MIT": "MIT",
        "Apache-2.0": "Apache-2.0",
        "LLAMA 2 Community License": "LicenseRef-Llama-2-Community",
        "Meta Llama 3 Community License": "LicenseRef-Meta-Llama-3-Community",
        "Gemma Terms of Use": "LicenseRef-Gemma-Terms",
    }
    declared = metadata("jamjet-guardrails")["License-Expression"]
    sources = [*SOURCES, HTML_ELEMENT_SOURCE]
    assert sources, "no template sources found; this guard would prove nothing"

    unclassified = sorted({s.licence for s in sources if s.licence not in spdx})
    assert unclassified == [], (
        f"template-data carries licences this test cannot map to an SPDX identifier: "
        f"{unclassified}. Classify them here rather than letting them ship unnamed"
    )
    missing = sorted({spdx[s.licence] for s in sources if spdx[s.licence] not in declared})
    assert missing == [], (
        f"the distribution declares {declared!r} and ships template-data licensed {missing}"
    )


# ==========================================================================
# What is INSIDE the built archives, which nothing above this line can see.
# ==========================================================================
#
# Every other packaging guard in this file reads the installed METADATA, and
# `tests/test_benchmarks.py` says plainly that it reads `pyproject.toml` and
# that "nothing here can see inside an artifact". That was true and it was a
# hole: the wheel target's `packages` list is a configuration, and what ships
# is an archive. The two agree until a `.gitignore` rule, a `force-include` or
# a hatchling default moves under them, and none of those edits touches the
# line a configuration test reads.
#
# So these two build the real thing. `hatchling` is in the dev extra for this
# and is already the build backend, so nothing new is being trusted; what is
# new is that the test suite opens the archive.
#
# All three were watched to FAIL, `__pycache__` cleared between runs, against
# `exclude = ["unicode-data"]` on the sdist target, a `force-include` of the
# raw files into the wheel, and `Unicode-3.0` removed from the licence
# expression.


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The sdist and the wheel, built once for the session.

    A subprocess rather than hatchling's Python API, because the API builds
    relative to the process's working directory and a test that chdir'd into
    the repository would leak that into every test after it.
    """
    out = tmp_path_factory.mktemp("dist")
    done = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-d", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"hatchling build failed:\n{done.stdout}\n{done.stderr}"
    sdists = list(out.glob("*.tar.gz"))
    wheels = list(out.glob("*.whl"))
    assert len(sdists) == 1 and len(wheels) == 1, f"built {sdists} and {wheels}"
    return sdists[0], wheels[0]


def test_the_built_sdist_carries_the_pinned_unicode_data(built: tuple[Path, Path]) -> None:
    """The sdist is the evidence, and evidence that ships nowhere is none.

    The two generated modules under `_unicode/` are a derivation, and a
    derivation nobody can rerun is a table a reader has to take on trust. The
    four published files are what
    `tests/test_unicode.py::test_the_generated_modules_are_byte_identical_to_a_regeneration`
    reruns it from, so they travel with the source distribution or that test
    can only ever be run from a git checkout.

    Also asserts the tests and the corpora are still there, because the failure
    this would otherwise miss is an `include` list added to the sdist target
    that names `unicode-data/` and drops everything else: the raw files would
    be present and the sdist would have lost its other reason to exist.
    """
    sdist, _ = built
    with tarfile.open(sdist) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    missing = [
        f"unicode-data/16.0.0/{name}"
        for name in (
            "PropertyValueAliases.txt",
            "ScriptExtensions.txt",
            "Scripts.txt",
            "confusables.txt",
        )
        if f"unicode-data/16.0.0/{name}" not in names
    ]
    assert missing == [], f"the sdist does not carry {missing}"
    for expected in (
        "scripts/generate_unicode_tables.py",
        "tests/test_unicode.py",
        "corpora/NOTICE.md",
    ):
        assert expected in names, f"the sdist no longer carries {expected}"


def test_the_built_wheel_carries_no_raw_unicode_data(built: tuple[Path, Path]) -> None:
    """1.4 MB of text no installed code reads must not reach an installer.

    The generator runs from a checkout and the tables it produced are already
    inside the package, so nothing under `unicode-data/` is ever opened at
    runtime. The wheel target's `packages` list is what keeps it out; this is
    what checks that it did.

    The positive half is not decoration. "No raw Unicode data in the wheel" is
    also true of a wheel that lost the generated tables, and that wheel
    installs cleanly and fails at the first skeleton.
    """
    _, wheel = built
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    # Two rules, and the second is the wider one. Nothing anywhere may name
    # `unicode-data`, and inside the package itself the only things allowed are
    # modules and the PEP 561 marker, so any DATA file added to the package
    # directory fails here and not only the four this task is about. The
    # `.dist-info/` tree is left out because it is the installer's own
    # bookkeeping (`entry_points.txt` lives there) and is not this package's to
    # constrain.
    stowaways = sorted(
        name
        for name in names
        if "unicode-data" in name
        or (
            name.startswith("jamjet_guardrails/")
            and not name.endswith(".py")
            and name != "jamjet_guardrails/py.typed"
        )
    )
    assert stowaways == [], f"the wheel carries {stowaways}"
    for expected in (
        "jamjet_guardrails/_unicode/__init__.py",
        "jamjet_guardrails/_unicode/scripts.py",
        "jamjet_guardrails/_unicode/confusables.py",
    ):
        assert expected in names, f"the wheel does not carry {expected}"


def test_the_declared_licence_names_unicode_while_unicode_data_ships(
    built: tuple[Path, Path],
) -> None:
    """The other half of the licence guard, derived from the artifacts.

    `test_the_declared_licence_covers_every_licence_the_corpora_carry` reads
    the corpora, and the Unicode data is not a corpus: it carries no `license`
    field and no loader refuses it, so that guard cannot see it at all. What
    makes this one derived rather than a restatement is that it reads the BUILT
    archives for the material and the BUILT metadata for the claim. Drop
    `unicode-data/` from the sdist and delete the generated tables from the
    wheel and it stops demanding `Unicode-3.0`; ship either and it does.

    The generated modules are in the domain, not only the raw files. They are
    the same property values re-encoded, which is a derivative work of the data
    files, and they are the half that reaches an installer.

    The claim is read out of the WHEEL's own METADATA rather than through
    `importlib.metadata`, and that is not the house habit for a reason. An
    editable install writes its metadata once, so every other licence and
    classifier guard in this file is checked against whatever was installed
    last and needs the reinstall test above to stay honest. Here both sides
    come from the same build, so the test says the same thing on a machine that
    has not reinstalled since the licence changed.
    """
    sdist, wheel = built
    with tarfile.open(sdist) as archive:
        in_sdist = any("unicode-data/" in name for name in archive.getnames())
    with zipfile.ZipFile(wheel) as zipped:
        in_wheel = any("_unicode/" in name for name in zipped.namelist())
        wheel_metadata = next(
            zipped.read(name).decode("utf-8")
            for name in zipped.namelist()
            if name.endswith(".dist-info/METADATA")
        )
    assert in_sdist or in_wheel, (
        "neither artifact carries Unicode data or anything derived from it; "
        "this guard would prove nothing"
    )
    found = re.search(r"^License-Expression: (.+)$", wheel_metadata, re.MULTILINE)
    assert found is not None, "the built wheel declares no License-Expression"
    declared = found.group(1)
    assert "Unicode-3.0" in declared, (
        f"the distribution declares {declared!r} while shipping data published by "
        "Unicode, Inc. under the Unicode License v3 (raw in the sdist: "
        f"{in_sdist}, derived in the wheel: {in_wheel})"
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
# What the two BUILT artifacts carry, opened rather than inferred.
# ==========================================================================
#
# Every other test in this file reads the installed metadata or the source
# tree. Under the editable install CI and local development both use, those
# resolve to the working directory, so nothing above can tell the difference
# between a file being present and a file being SHIPPED. The release workflow
# opens the wheel, and until now that was the only place anything did; a
# release workflow is the wrong place for the first look, because it runs after
# the pull request that broke the packaging was merged.
#
# `template-data/` is what made it worth building here. It is the evidence
# behind a generated table: the raw tokenizer configuration the markers were
# read out of, which somebody auditing the table has to be able to get hold of.
# It belongs in the sdist for exactly that reason, and it must not be in the
# wheel, where it would be dead weight in every consumer's environment and
# would put third-party licensed files inside an installed package.
#
# The archives are built ONCE for the session by `built` above, and the member
# names come from that same build rather than from a second one. Two fixtures
# named `built` arrived here from two branches, one returning the paths and one
# returning the member names, and ruff caught the redefinition; a second build
# would also have been a second answer to one question, which is how two tests
# end up disagreeing about what shipped.


@pytest.fixture(scope="session")
def built_names(built: tuple[Path, Path]) -> tuple[list[str], list[str]]:
    """The member names inside the sdist and the wheel `built` produced."""
    sdist, wheel = built
    with tarfile.open(sdist) as archive:
        # The sdist nests everything under `<name>-<version>/`. Strip it, so the
        # names below read as repository paths and a version bump does not
        # change what this file asserts.
        sdist_names = [name.partition("/")[2] for name in archive.getnames()]
    return sdist_names, zipfile.ZipFile(wheel).namelist()


def test_the_built_sdist_carries_every_committed_template_data_file(
    built_names: tuple[list[str], list[str]],
) -> None:
    """The sdist is the evidence, so the evidence has to be in it.

    Derived from what is on disk rather than from a list, for the reason
    `tests/test_published_docs.py` gives about guards written over whichever
    files their author had open: a source added to
    `scripts/generate_template_markers.py` is covered here without anyone
    remembering.
    """
    sdist_names, _ = built_names
    expected = sorted(
        str(path.relative_to(ROOT)) for path in TEMPLATE_DATA.rglob("*") if path.is_file()
    )
    assert expected, "nothing under template-data/; this guard would prove nothing"
    missing = [name for name in expected if name not in sdist_names]
    assert missing == [], f"the built sdist ships no evidence for: {missing}"


def test_the_built_wheel_carries_the_table_and_none_of_its_raw_sources(
    built_names: tuple[list[str], list[str]],
) -> None:
    """Both halves, because either alone is satisfied by the wrong artifact.

    A wheel with neither the table nor the raw files passes a test that only
    forbids the raw files, and a wheel carrying both passes a test that only
    requires the table. The table is generated code and belongs in the package;
    the tokenizer configuration it was generated from is third-party material
    that has no business inside a consumer's site-packages.
    """
    _, wheel_names = built_names
    assert "jamjet_guardrails/detectors/_template_markers.py" in wheel_names, (
        "the wheel ships no marker table"
    )
    stray = sorted(name for name in wheel_names if "template-data" in name)
    assert stray == [], f"the wheel ships raw tokenizer configuration: {stray}"


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
