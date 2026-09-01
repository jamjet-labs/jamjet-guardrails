"""The licence and contamination screens over `training/sources.yaml`.

Run in CI over committed data. `training/fetch.py` reads the manifest with
`yaml.safe_load`, and PyYAML is in the `dev` extra beside pytest, ruff and
mypy, so `pip install -e ".[dev]"` -- which is what every CI leg runs -- brings
it in and these screens run everywhere. The package itself still declares
`dependencies = []`; `tests/test_packaging.py` reads the built metadata and
holds that.

What the screens can and cannot see is stated at `PROTECTAI_NAMES_AS_TRAINING_DATA`
below, and it matters more than any assertion here: the contamination denylist
is known to be partial, and a source it does not match is a source nobody has
checked rather than a source it cleared.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

from training.fetch import (
    DATA,
    HEX64,
    NO_DIGEST,
    ROLES,
    Source,
    SourceError,
    fetch,
    load_sources,
    sha256_of,
    verify,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "training" / "sources.yaml"
TRAINING_README = ROOT / "training" / "README.md"
NOTICE = ROOT / "corpora" / "NOTICE.md"

# The datasets the `datasets:` metadata on ProtectAI's model card for
# `deberta-v3-base-prompt-injection-v2` names, read from the live card on
# 2026-09-01.
#
# THIS LIST IS KNOWN TO BE PARTIAL, and treating it as anything else is the
# mistake it exists to prevent. The same card's licence summary accounts for 22
# source datasets -- 1 CC-BY-3.0, 8 MIT, 1 CC0-1.0, 6 with no licence, 5
# Apache-2.0, 1 CC-BY-4.0 -- so 15 of them are counted and never named
# anywhere. A screen over a third-party model's training data cannot be
# exhaustive when the third party did not publish that data.
#
# So this catches a case the card happens to name, and nothing more. A source
# absent from it is NOT cleared for evaluation; it is a source whose provenance
# nobody has established. Anything selecting an evaluation corpus has to
# establish that separately rather than reading a pass here as a guarantee.
PROTECTAI_NAMES_AS_TRAINING_DATA = frozenset(
    {
        "natolambert/xstest-v2-copy",
        "VMware/open-instruct",
        "alespalla/chatbot_instruction_prompts",
        "HuggingFaceH4/grok-conversation-harmless",
        "Harelix/Prompt-Injection-Mixed-Techniques-2024",
        "OpenSafetyLab/Salad-Data",
        "jackhhao/jailbreak-classification",
    }
)

#: How many source datasets the same card's licence summary accounts for, as
#: against the seven its metadata names. Recorded as a number beside the list
#: so the gap between them is a fact this file states rather than a caveat
#: somebody has to remember.
PROTECTAI_LICENCE_SUMMARY_TOTAL = 22

#: The two the plan requires `training/sources.yaml` to record by name. A
#: strict subset of the list above, which is the whole point: recording these
#: two is a floor, not a demonstration that the denylist is complete.
RECORDED_EXCLUSIONS = frozenset(
    {
        "jackhhao/jailbreak-classification",
        "Harelix/Prompt-Injection-Mixed-Techniques-2024",
    }
)

#: Sources on the denylist whose licence requires attribution, per ProtectAI's
#: own card. Neither is in the manifest today. If either is ever recorded as
#: something this repository uses, it needs an entry in `corpora/NOTICE.md` the
#: way `corpora/pii/third-party.jsonl` already does -- attribution is a
#: condition of the licence, not a courtesy, and the file that carries it is
#: the one that ships.
ATTRIBUTION_REQUIRED = {
    "VMware/open-instruct": "CC-BY-3.0",
    "natolambert/xstest-v2-copy": "CC-BY-4.0",
}


def _contaminated(sources: Iterable[Source]) -> list[str]:
    """Evaluation sources the reference model is named as trained on. The rule.

    Factored out of the test that applies it to the shipped manifest so the
    same rule can be pointed at a manifest that breaks it. Today's manifest
    declares no `eval` source at all, so applying the rule to it alone would
    pass over an empty loop and prove nothing.
    """
    return sorted(
        source.name
        for source in sources
        if source.role == "eval" and source.name in PROTECTAI_NAMES_AS_TRAINING_DATA
    )


def _unattributed(sources: Iterable[Source], notice: str) -> list[str]:
    """Sources in use under an attribution licence that the NOTICE does not name.

    Same shape as `_contaminated`, and for the same reason: the manifest
    records no such source today, so the rule has to be exercised against one
    that does or it is an empty loop wearing a test's name.
    """
    return sorted(
        source.name
        for source in sources
        if source.name in ATTRIBUTION_REQUIRED
        and source.role != "excluded"
        and source.name not in notice
    )


def _manifest(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_every_source_declares_a_licence_and_a_hash() -> None:
    """WIDENED from the plan's version, which required 64 hex digits everywhere.

    One of the two sources this file is required to record cannot carry a
    digest: `Harelix/Prompt-Injection-Mixed-Techniques-2024` answered HTTP 401
    on every Hugging Face endpoint on 2026-09-01, so there is no content to
    hash. A fabricated 64-character string would have satisfied the original
    assertion exactly, which is the failure mode worth avoiding here.

    Written as one assertion per source rather than as a branch on `role`. The
    branch would have been dead: `load_sources` refuses an unpinned source in
    any role but `excluded`, so no `train` or `eval` source without a digest
    can reach this loop, and an arm that cannot execute proves nothing about
    the manifest it appears to be checking. What is left is the invariant
    itself, live on every row.
    """
    sources = load_sources(SOURCES)
    assert sources, "sources.yaml lists nothing"
    for source in sources:
        assert source.license, f"{source.name} declares no licence"
        assert source.role in ROLES, f"{source.name} has role {source.role!r}"
        assert HEX64.match(source.sha256) or (
            source.sha256 == NO_DIGEST and source.role == "excluded"
        ), f"{source.name} carries {source.sha256!r} in role {source.role!r}"
    assert any(HEX64.match(source.sha256) for source in sources), (
        "not one source in the manifest carries a real digest, so nothing above "
        "checked a digest against anything"
    )


def test_no_evaluation_source_is_one_protectai_names_as_training_data() -> None:
    """The contamination trap, enforced at the level of source selection.

    Scoring against a corpus the reference model memorised would publish a
    number that flatters us for the wrong reason, and no downstream test could
    detect it -- the rows would look like ordinary held-out data.

    Passing this is necessary and nowhere near sufficient: the denylist names
    what ProtectAI's card names, which is 7 of the 22 datasets the same card
    counts. See `PROTECTAI_NAMES_AS_TRAINING_DATA`.
    """
    assert _contaminated(load_sources(SOURCES)) == []


def test_the_contamination_rule_catches_a_manifest_that_breaks_it(tmp_path: Path) -> None:
    """The mutation check for the test above, which today loops over nothing.

    `training/sources.yaml` declares no `eval` source yet -- a later task adds
    them once each licence is verified -- so the assertion above passes
    vacuously and would keep passing if the rule were inverted. This runs the
    same rule against a manifest that names a ProtectAI training corpus as
    evaluation data, and requires it to say so.

    The entry used is one of the five the denylist gained after review, so this
    also fails if the list is narrowed back to the two the manifest records.
    """
    contaminated = _manifest(
        tmp_path,
        "- name: OpenSafetyLab/Salad-Data\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"a" * 64}"\n'
        "  role: eval\n",
    )
    assert _contaminated(load_sources(contaminated)) == ["OpenSafetyLab/Salad-Data"]


def test_the_denylist_is_recorded_as_partial_and_not_as_a_clearance() -> None:
    """The relationship between the three constants, which is the finding itself.

    A denylist of 2 was read as the complete set of what the reference model
    was trained on, and the fix is not a denylist of 7 read the same way. What
    has to hold is that the file says the list is smaller than the thing it
    screens for: the card's metadata names 7, its licence summary counts 22,
    and the two recorded exclusions are a strict subset of the 7.
    """
    assert RECORDED_EXCLUSIONS < PROTECTAI_NAMES_AS_TRAINING_DATA, (
        "the recorded exclusions are no longer a strict subset of the denylist, so the "
        "manifest's two entries are being treated as the whole of it"
    )
    assert len(PROTECTAI_NAMES_AS_TRAINING_DATA) < PROTECTAI_LICENCE_SUMMARY_TOTAL, (
        "the denylist now claims to cover every dataset the card accounts for, which is "
        "the completeness claim this test exists to refuse"
    )


def test_the_readme_states_the_same_partiality_the_denylist_records() -> None:
    """Two tables, and they have to agree.

    `training/README.md` states these counts in prose and this module holds
    them as data. A number in prose that counts a thing in code is a claim, and
    the last one drifted from 7 to "two" with nothing to catch it.
    """
    named = len(PROTECTAI_NAMES_AS_TRAINING_DATA)
    unnamed = PROTECTAI_LICENCE_SUMMARY_TOTAL - named
    readme = TRAINING_README.read_text(encoding="utf-8")
    assert f"names {named} datasets" in readme
    assert f"accounts for {PROTECTAI_LICENCE_SUMMARY_TOTAL} source datasets" in readme
    assert f"{unnamed} are counted and never named" in readme


def test_the_manifest_records_the_exclusions_the_plan_requires() -> None:
    """The exclusions are the file's reason for existing, so their absence fails.

    Without this, deleting either entry leaves every other test in this module
    passing: the contamination rule has nothing to reject, and the licence loop
    has one fewer row to walk.

    Scoped to `RECORDED_EXCLUSIONS` rather than to the whole denylist on
    purpose. The denylist names datasets this repository has never fetched and
    has no reason to record, and requiring every name on it to appear here
    would turn a partial list into a demand for entries nobody has verified.
    """
    by_name = {source.name: source for source in load_sources(SOURCES)}
    for name in sorted(RECORDED_EXCLUSIONS):
        assert name in by_name, f"{name} is not recorded in sources.yaml"
        assert by_name[name].role == "excluded", f"{name} is recorded as usable"
        assert by_name[name].note.strip(), f"{name} is excluded with no reason recorded"


def test_a_source_under_an_attribution_licence_is_named_in_the_notice() -> None:
    """Vacuous today, and pinned so it stops being vacuous the moment it matters.

    Neither `VMware/open-instruct` (CC-BY-3.0) nor `natolambert/xstest-v2-copy`
    (CC-BY-4.0) is in the manifest. Both are on the denylist, both are
    plausible sources for a later task, and both carry a licence whose
    attribution term is a condition of use. `corpora/NOTICE.md` is where this
    repository discharges that, and it is checked here rather than remembered.
    """
    assert _unattributed(load_sources(SOURCES), NOTICE.read_text(encoding="utf-8")) == []


def test_the_attribution_rule_catches_a_manifest_that_breaks_it(tmp_path: Path) -> None:
    """The same rule against a manifest that uses one without attributing it."""
    used = _manifest(
        tmp_path,
        "- name: VMware/open-instruct\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        '  license: "CC-BY-3.0"\n'
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    assert _unattributed(load_sources(used), "a NOTICE that names nothing") == [
        "VMware/open-instruct"
    ]


def test_the_loader_refuses_an_unpinned_source_that_gets_measured_on(tmp_path: Path) -> None:
    """The exemption above, closed. An unpinned source may only be an excluded one.

    Flipping `role` on an entry that records no digest is the one way the
    licence test's exemption could become a way to ship an unpinned evaluation
    corpus, so the loader rejects the pair rather than the test.
    """
    unpinned = _manifest(
        tmp_path,
        "- name: someone/unfetchable\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{NO_DIGEST}"\n'
        "  role: eval\n"
        "  note: gated behind a login\n",
    )
    with pytest.raises(SourceError, match="Only an excluded source may go unpinned"):
        load_sources(unpinned)


def test_the_loader_refuses_an_unpinned_source_with_no_reason(tmp_path: Path) -> None:
    """An absence somebody has to justify is one somebody can re-check."""
    unexplained = _manifest(
        tmp_path,
        "- name: someone/unfetchable\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{NO_DIGEST}"\n'
        "  role: excluded\n",
    )
    with pytest.raises(SourceError, match="no note saying why"):
        load_sources(unexplained)


def test_the_loader_refuses_a_role_outside_the_three(tmp_path: Path) -> None:
    """`Role` is a Literal, so an unknown role would otherwise reach a typed field."""
    wrong_role = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"b" * 64}"\n'
        "  role: maybe\n",
    )
    with pytest.raises(SourceError, match="role is 'maybe'"):
        load_sources(wrong_role)


def test_the_loader_refuses_a_digest_that_is_not_one(tmp_path: Path) -> None:
    """A truncated or upper-case digest never matches, so it must not be recorded."""
    short = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        '  sha256: "abc123"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="neither 64 lower-case hex"):
        load_sources(short)


def test_the_loader_refuses_a_manifest_yaml_cannot_read(tmp_path: Path) -> None:
    """A parse failure has to reach the caller as this module's own exception.

    Three spaces of indentation rather than two is the kind of edit a
    hand-maintained file actually receives. PyYAML raises its own error type
    for it; a caller that had to catch `yaml.YAMLError` as well as
    `SourceError` is a caller that will catch one of them and not the other.
    """
    misindented = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '   url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"c" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="not readable as YAML"):
        load_sources(misindented)


def test_the_loader_refuses_a_document_that_is_not_a_list_of_sources(tmp_path: Path) -> None:
    """One source written without its `- ` is a mapping, and reads as valid YAML.

    Nothing about the parse fails: the file is a perfectly good document of the
    wrong shape, and every key on it would otherwise be walked as though it
    were a source.
    """
    single = _manifest(
        tmp_path,
        "name: someone/corpus\n"
        'url: "https://example.invalid/corpus.csv"\n'
        "license: Apache-2.0\n"
        f'sha256: "{"c" * 64}"\n'
        "role: train\n",
    )
    with pytest.raises(SourceError, match="a manifest is a list of sources"):
        load_sources(single)


@pytest.mark.parametrize("text", ["# nothing but a comment\n", "[]\n"])
def test_the_loader_refuses_an_empty_manifest(tmp_path: Path, text: str) -> None:
    """Two spellings of nothing, and they take two different paths through the loader.

    A file with no document in it parses to `None`; a file holding `[]` parses
    to an empty list. Only the second reaches the emptiness check in
    `load_sources`, so a test that used the first alone would leave that check
    unexercised and passing.
    """
    with pytest.raises(SourceError, match="lists no sources"):
        load_sources(_manifest(tmp_path, text))


def test_the_loader_refuses_a_field_yaml_read_as_something_other_than_text(
    tmp_path: Path,
) -> None:
    """YAML types values, and the type it picks is not always the one meant.

    `role: yes` is a bool under the 1.1 resolver PyYAML implements, so the
    membership test against the three roles would compare `True` to strings and
    the entry would be refused for the wrong reason -- or, on a field with no
    membership test, accepted with a value no screen can read. Refused at the
    field rather than coerced.
    """
    retyped = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"c" * 64}"\n'
        "  role: yes\n",
    )
    with pytest.raises(SourceError, match="must carry one non-empty piece of text"):
        load_sources(retyped)


def test_the_loader_refuses_a_nested_value_on_a_known_field(tmp_path: Path) -> None:
    """The manifest is flat mappings of text. A source that needs more must not half-parse.

    A known key whose value is a nested list parses cleanly and would leave the
    field holding a list, which the licence loop and the note check would both
    walk straight past.
    """
    nested = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"c" * 64}"\n'
        "  role: train\n"
        "  note:\n"
        "    - one\n"
        "    - two\n",
    )
    with pytest.raises(SourceError, match="must carry one non-empty piece of text"):
        load_sources(nested)


def test_the_loader_refuses_an_unknown_field(tmp_path: Path) -> None:
    """A field nothing reads is a fact nobody is enforcing. Say so at load."""
    extra = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"d" * 64}"\n'
        "  role: train\n"
        '  screened: "by hand"\n',
    )
    with pytest.raises(SourceError, match=r"unknown keys \['screened'\]"):
        load_sources(extra)


def test_verify_rejects_a_file_whose_hash_moved(tmp_path: Path) -> None:
    target = tmp_path / "corpus.jsonl"
    target.write_text("hello", encoding="utf-8")
    with pytest.raises(SourceError, match="sha256"):
        verify(target, "0" * 64)


def test_verify_accepts_the_digest_it_recorded(tmp_path: Path) -> None:
    """The dual of the test above. A checker that rejects everything also passes it."""
    target = tmp_path / "corpus.jsonl"
    target.write_text("hello", encoding="utf-8")
    verify(target, sha256_of(target))


def test_sha256_of_reads_a_file_larger_than_one_block(tmp_path: Path) -> None:
    """`sha256_of` reads in 64 KiB blocks, so the loop is the part worth checking.

    A block-wise digest that dropped or repeated a block would agree with a
    one-shot digest on every short file, which is every other file in this
    module.
    """
    target = tmp_path / "big.bin"
    payload = bytes(range(256)) * 1024
    assert len(payload) > 65536, "the payload no longer exercises more than one read"
    target.write_bytes(payload)
    assert sha256_of(target) == hashlib.sha256(payload).hexdigest()


def test_fetch_refuses_a_source_with_no_recorded_digest(tmp_path: Path) -> None:
    """There is nothing to check the download against, so there is no download."""
    source = Source(
        name="someone/unfetchable",
        url="https://example.invalid/corpus.csv",
        license="Apache-2.0",
        sha256=NO_DIGEST,
        role="excluded",
        note="gated",
    )
    with pytest.raises(SourceError, match="records no digest"):
        fetch(source, tmp_path / "data")
    assert not (tmp_path / "data").exists(), "fetch created its target directory before refusing"


def test_fetch_verifies_what_it_downloaded(tmp_path: Path) -> None:
    """Hermetic: the URL is a `file://` one, so nothing here reaches the network.

    Also pins the local name. The URL's extension is kept rather than a fixed
    `.jsonl`, because the sources pinned so far are CSV and a CSV under a
    `.jsonl` name tells the next stage to parse it the wrong way.
    """
    origin = tmp_path / "origin.csv"
    origin.write_text("prompt,type\nhello,benign\n", encoding="utf-8")
    source = Source(
        name="someone/corpus",
        url=origin.as_uri(),
        license="Apache-2.0",
        sha256=sha256_of(origin),
        role="train",
    )
    target = fetch(source, tmp_path / "data")
    assert target.name == "someone__corpus.csv"
    assert target.read_text(encoding="utf-8") == origin.read_text(encoding="utf-8")


def test_fetch_rejects_a_download_that_is_not_what_was_recorded(tmp_path: Path) -> None:
    origin = tmp_path / "origin.csv"
    origin.write_text("prompt,type\nhello,benign\n", encoding="utf-8")
    source = Source(
        name="someone/corpus",
        url=origin.as_uri(),
        license="Apache-2.0",
        sha256="e" * 64,
        role="train",
    )
    with pytest.raises(SourceError, match="the corpus moved"):
        fetch(source, tmp_path / "data")


def test_fetch_still_verifies_a_file_it_did_not_download(tmp_path: Path) -> None:
    """The stale-cache case, which is the one a hash is really there for.

    `fetch` skips the download when the target exists. A file left behind by an
    interrupted run, or by an earlier revision of the manifest, would otherwise
    be used unchecked, and the run would be measured on content nobody pinned.
    """
    into = tmp_path / "data"
    into.mkdir()
    (into / "someone__corpus.csv").write_text("stale\n", encoding="utf-8")
    origin = tmp_path / "origin.csv"
    origin.write_text("prompt,type\nhello,benign\n", encoding="utf-8")
    source = Source(
        name="someone/corpus",
        url=origin.as_uri(),
        license="Apache-2.0",
        sha256=sha256_of(origin),
        role="train",
    )
    with pytest.raises(SourceError, match="the corpus moved"):
        fetch(source, into)


def test_fetch_refuses_a_scheme_it_does_not_serve(tmp_path: Path) -> None:
    """`urlretrieve` opens more than https, so what may be opened is stated.

    `file://` is allowed because the tests above need a local origin to stay
    hermetic. `ftp://` is the one `urlretrieve` also honours by default, and it
    is the case that would otherwise reach the network from a manifest edit
    nobody screened.
    """
    source = Source(
        name="someone/corpus",
        url="ftp://example.invalid/corpus.csv",
        license="Apache-2.0",
        sha256="f" * 64,
        role="train",
    )
    with pytest.raises(SourceError, match="served over 'ftp'"):
        fetch(source, tmp_path / "data")


def test_fetch_refuses_a_destination_inside_the_repository(tmp_path: Path) -> None:
    """`training/data/` is not ignored, so a download written there is committed.

    The natural destination for a later script to type, because everything else
    this tree owns lives under `training/`. `.gitignore` anchors `/data/` to
    the repository root and reaches nothing nested, so the mistake is silent:
    the corpus appears in `git status` looking like a file somebody meant to
    add.
    """
    source = Source(
        name="someone/corpus",
        url=(tmp_path / "origin.csv").as_uri(),
        license="Apache-2.0",
        sha256="f" * 64,
        role="train",
    )
    inside = ROOT / "training" / "data"
    with pytest.raises(SourceError, match="would be committed"):
        fetch(source, inside)
    assert not inside.exists(), "fetch created a committed directory before refusing"


def test_the_default_destination_is_the_directory_gitignore_keeps_out() -> None:
    """The guard's exemption and the ignore rule have to name the same directory.

    Two statements of one fact -- `DATA` in `training/fetch.py` and `/data/` in
    `.gitignore` -- and they are what makes the default safe. If either moves
    alone, downloads land in a directory git will happily commit and every
    other test here still passes.
    """
    assert DATA == ROOT / "data"
    assert inspect.signature(fetch).parameters["into"].default == DATA
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/data/" in rules, ".gitignore no longer ignores the directory fetch writes to"


def test_fetch_refuses_a_name_that_is_a_path_rather_than_a_filename(tmp_path: Path) -> None:
    """`name.replace('/', '__')` is a POSIX answer to a question with two spellings.

    A backslash in a source's name is a separator on Windows, so the local file
    would be written one directory up from where every other one goes. Checked
    with the Windows path rules on every platform, so the refusal is the same
    everywhere rather than only where it would do damage.
    """
    source = Source(
        name="someone\\..\\corpus",
        url="https://example.invalid/corpus.csv",
        license="Apache-2.0",
        sha256="f" * 64,
        role="train",
    )
    with pytest.raises(SourceError, match="which is a path"):
        fetch(source, tmp_path / "data")


def test_every_url_in_the_manifest_is_pinned_to_a_revision_or_carries_no_digest() -> None:
    """A floating URL under a recorded hash is a hash that will start failing.

    Hugging Face serves `/resolve/<ref>/...`, and `<ref>` may be a branch. A
    branch moves, so the next fetch either downloads different bytes and trips
    `verify` -- which is at least loud -- or, worse, the recorded digest gets
    updated to match and the corpus has changed under every number measured on
    it. Anything pinned to a 40-character commit sha cannot do that.
    """
    unpinned: list[str] = []
    for source in load_sources(SOURCES):
        if source.sha256 == NO_DIGEST:
            continue
        if re.search(r"/resolve/[0-9a-f]{40}/", source.url) is None:
            unpinned.append(f"{source.name} -> {source.url}")
    assert unpinned == [], f"these carry a digest but not a pinned revision: {unpinned}"


def test_every_url_in_the_manifest_is_https() -> None:
    """The manifest side of the scheme allowlist.

    `fetch` permits `file://` so the tests above can run without a network. A
    manifest entry that used it would read a path on whoever ran the fetch,
    which is not a source anyone else can obtain, and the recorded digest would
    describe their filesystem.
    """
    wrong = [s.name for s in load_sources(SOURCES) if not s.url.startswith("https://")]
    assert wrong == [], f"these are not served over https: {wrong}"
