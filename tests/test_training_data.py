"""The licence and contamination screens over `training/sources.yaml`.

Run in CI over committed data, which is the reason `training/fetch.py` reads
the manifest with the standard library. CI installs `.[dev]` and nothing else,
so a PyYAML import in the module under test would turn every assertion here
into a skip on every leg, and a screen that never runs is not a screen.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

from training.fetch import (
    HEX64,
    NO_DIGEST,
    Source,
    SourceError,
    fetch,
    load_sources,
    sha256_of,
    verify,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "training" / "sources.yaml"

# ProtectAI's model card names these among its training data. A detector scored
# on its own training corpus reports memorisation under the word "recall", so
# these are usable for training and never for evaluation.
PROTECTAI_TRAINED_ON = frozenset(
    {
        "jackhhao/jailbreak-classification",
        "Harelix/Prompt-Injection-Mixed-Techniques-2024",
    }
)


def _contaminated(sources: Iterable[Source]) -> list[str]:
    """Evaluation sources the reference model was trained on. The rule itself.

    Factored out of the test that applies it to the shipped manifest so the
    same rule can be pointed at a manifest that breaks it. Today's manifest
    declares no `eval` source at all, so applying the rule to it alone would
    pass over an empty loop and prove nothing.
    """
    return sorted(
        source.name
        for source in sources
        if source.role == "eval" and source.name in PROTECTAI_TRAINED_ON
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

    So the digest is required where a digest is possible -- every source that
    is trained from or measured on -- and the recorded absence is confined to
    `excluded`, the one role neither of those happens to. `load_sources`
    enforces that confinement at load time, and
    `test_the_loader_refuses_an_unpinned_source_that_gets_measured_on` is what
    stops the exemption becoming a way to skip the hash.
    """
    sources = load_sources(SOURCES)
    assert sources, "sources.yaml lists nothing"
    for source in sources:
        assert source.license, f"{source.name} declares no licence"
        assert source.role in ("train", "eval", "excluded")
        if source.role != "excluded":
            assert HEX64.match(source.sha256), f"{source.name} has no usable sha256"
        else:
            assert HEX64.match(source.sha256) or source.sha256 == NO_DIGEST, (
                f"{source.name} records neither a digest nor the absence of one"
            )
    assert any(HEX64.match(source.sha256) for source in sources), (
        "not one source in the manifest carries a real digest, so nothing above "
        "checked a digest against anything"
    )


def test_no_evaluation_source_is_one_protectai_trained_on() -> None:
    """The contamination trap, enforced at the level of source selection.

    Scoring against a corpus the reference model memorised would publish a
    number that flatters us for the wrong reason, and no downstream test could
    detect it -- the rows would look like ordinary held-out data.
    """
    assert _contaminated(load_sources(SOURCES)) == []


def test_the_contamination_rule_catches_a_manifest_that_breaks_it(tmp_path: Path) -> None:
    """The mutation check for the test above, which today loops over nothing.

    `training/sources.yaml` declares no `eval` source yet -- Task 3 adds them
    once each licence is verified -- so the assertion above passes vacuously
    and would keep passing if the rule were inverted. This runs the same rule
    against a manifest that names a ProtectAI training corpus as evaluation
    data, and requires it to say so.
    """
    contaminated = _manifest(
        tmp_path,
        "- name: jackhhao/jailbreak-classification\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"a" * 64}"\n'
        "  role: eval\n",
    )
    assert _contaminated(load_sources(contaminated)) == ["jackhhao/jailbreak-classification"]


def test_the_manifest_records_both_sources_protectai_trained_on() -> None:
    """The exclusions are the file's reason for existing, so their absence fails.

    Without this, deleting either entry leaves every other test in this module
    passing: the contamination rule has nothing to reject, and the licence loop
    has one fewer row to walk.
    """
    by_name = {source.name: source for source in load_sources(SOURCES)}
    for name in sorted(PROTECTAI_TRAINED_ON):
        assert name in by_name, f"{name} is not recorded in sources.yaml"
        assert by_name[name].role == "excluded", f"{name} is recorded as usable"
        assert by_name[name].note.strip(), f"{name} is excluded with no reason recorded"


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


def test_the_reader_refuses_a_line_it_does_not_understand(tmp_path: Path) -> None:
    """A hand-written subset parser has to fail loudly on what it cannot read.

    The alternative is the failure this whole module exists to prevent: a
    manifest line that is silently dropped, leaving a source unscreened while
    every assertion above still walks the rows it did parse. Three spaces of
    indentation rather than two is the cheapest way to produce one, and the
    kind of edit a hand-maintained file actually receives.
    """
    misindented = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '   url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"c" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="is not a "):
        load_sources(misindented)


def test_the_reader_refuses_a_nested_structure(tmp_path: Path) -> None:
    """The subset is flat mappings. A source that needs more must not half-parse.

    A key whose value is a nested list reads as a key with an empty value, and
    an empty value is refused rather than defaulted: a source with a silently
    blank field is one the licence loop above would walk straight past.
    """
    nested = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        "  files:\n"
        "    - one.csv\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"c" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="every field a source declares must carry one"):
        load_sources(nested)


def test_the_reader_refuses_an_unknown_field(tmp_path: Path) -> None:
    """A field nothing reads is a fact nobody is enforcing. Say so at load."""
    extra = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"d" * 64}"\n'
        "  role: train\n"
        "  screened: yes\n",
    )
    with pytest.raises(SourceError, match=r"unknown keys \['screened'\]"):
        load_sources(extra)


def test_the_manifest_reader_agrees_with_the_yaml_library() -> None:
    """The subset reader against the real thing, wherever the real thing is present.

    Skipped in CI by design: PyYAML is pinned in `training/requirements.txt`
    and is deliberately absent from the package's `.venv`, so this runs in the
    training virtualenv and on any machine that has it. Compared on 2026-09-01
    under PyYAML 6.0.3, the two agreed on every field of both sources.

    Notes are compared after stripping. A folded (`>`) block clips to a single
    trailing newline in YAML and this reader emits none, which is the one
    difference between them and the only one this is allowed to tolerate.
    """
    yaml = pytest.importorskip("yaml")
    reference = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))
    ours = load_sources(SOURCES)
    assert isinstance(reference, list)
    assert len(reference) == len(ours), "the two readers disagree on how many sources there are"
    for expected, source in zip(reference, ours):
        assert set(expected) <= {"name", "url", "license", "sha256", "role", "note"}
        for field in ("name", "url", "license", "sha256", "role"):
            assert expected[field] == getattr(source, field), f"{source.name}: {field}"
        assert expected.get("note", "").strip() == source.note


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
