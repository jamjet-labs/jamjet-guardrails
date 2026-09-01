"""The licence and contamination screens over `training/sources.yaml`.

Run in CI over committed data. `training/fetch.py` reads the manifest with
PyYAML, through its `ManifestLoader`, and PyYAML is in the `dev` extra beside
pytest, ruff and mypy, so `pip install -e ".[dev]"` -- which is what every CI
leg runs -- brings it in and these screens run everywhere. The package itself
still declares `dependencies = []`; `tests/test_packaging.py` reads the built
metadata and holds that.

What the screens can and cannot see is stated at `NAMED_TRAINING_DATA`
below, and it matters more than any assertion here: the contamination denylist
is known to be partial, and a source it does not match is a source nobody has
checked rather than a source it cleared.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from training.fetch import (
    DATA,
    HEX64,
    NO_DIGEST,
    QUALIFIED_ID,
    ROLES,
    ManifestLoader,
    Source,
    SourceError,
    base_id,
    fetch,
    load_sources,
    sha256_of,
    verify,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "training" / "sources.yaml"
TRAINING_README = ROOT / "training" / "README.md"
NOTICE = ROOT / "corpora" / "NOTICE.md"

#: Every file under `training/` that could name a reference model. Globbed
#: rather than listed, so a file added to the tree tomorrow is scanned without
#: anyone remembering to add it here.
TRAINING_FILES = tuple(
    sorted(
        path
        for path in (ROOT / "training").rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".txt", ".yaml"}
    )
)

#: How a ProtectAI prompt-injection model is spelled in prose when its org
#: prefix is left off. The optional version suffix is part of the match, and
#: `re` prefers the longer alternative, so `-v2` is never read as the
#: unversioned v1 name.
#:
#: This pattern is family-scoped and cannot be the whole screen: a comparator
#: from another family matches nothing here, and two of them were added to the
#: README in review without a single test noticing.
#: `test_every_external_identifier_in_this_tree_is_accounted_for` is the screen
#: that does not depend on what a model is called; this one only adds the
#: org-less spelling of the two models already registered.
MODEL_REFERENCE = re.compile(r"deberta-v3-base-prompt-injection(?:-v\d+)?")

#: Identifiers named under `training/` that are neither a reference model nor a
#: dataset this repository records anywhere. Empty, and an entry is a decision
#: someone wrote down with a reason rather than a name a screen skipped.
UNSCREENED_IDS: dict[str, str] = {}


@dataclass(frozen=True)
class ReferenceModel:
    """A model this stage is measured against, and what its card discloses.

    One entry per model rather than one flat denylist, because the gap that
    review found was structural: the list held v2's datasets while the
    benchmark scored v1 as well, and nothing about a flat list could have shown
    that. `NAMED_TRAINING_DATA` is derived from these entries, so registering a
    model is what extends the screen, and
    `test_every_reference_model_named_in_this_tree_has_its_datasets_registered`
    is what stops a model being referenced without being registered.
    """

    #: The Hugging Face id the benchmark loads.
    model_id: str
    #: What the card's `datasets:` metadata names. Partial by construction; see
    #: `licence_summary_total`.
    named_datasets: frozenset[str]
    #: How many source datasets the card's own licence summary accounts for, or
    #: `None` where the card gives no total at all. `None` is not zero unnamed
    #: datasets: it is a card that does not say, which is the weaker position of
    #: the two and has to read as such.
    licence_summary_total: int | None
    #: When the card was read. A card can be edited; a number taken from one
    #: without a date is a number nobody can re-check.
    card_read_on: str

    @property
    def slug(self) -> str:
        """The id without its org, which is how the cards are cited in prose."""
        return self.model_id.split("/", 1)[1]


#: `deberta-v3-base-prompt-injection`, the v1 model. Its `datasets:` metadata
#: names twelve, read from the live card on 2026-09-01.
#:
#: The card gives NO total anywhere. Its training-data section says only "The
#: model was trained on a custom dataset from multiple open-source ones", and
#: its licence notice says it "was trained on one or more datasets that may be
#: subject to more restrictive licensing terms, including non-commercial use
#: provisions" without naming which. So unlike v2 there is not even a count to
#: measure the naming against: the twelve are what the card discloses, and how
#: much it withholds is not a number anyone has.
PROTECTAI_V1 = ReferenceModel(
    model_id="protectai/deberta-v3-base-prompt-injection",
    named_datasets=frozenset(
        {
            "Lakera/gandalf_ignore_instructions",
            "rubend18/ChatGPT-Jailbreak-Prompts",
            "imoxto/prompt_injection_cleaned_dataset-v2",
            "hackaprompt/hackaprompt-dataset",
            "fka/awesome-chatgpt-prompts",
            "teven/prompted_examples",
            "Dahoas/synthetic-hh-rlhf-prompts",
            "Dahoas/hh_prompt_format",
            "MohamedRashad/ChatGPT-prompts",
            "HuggingFaceH4/instruction-dataset",
            "HuggingFaceH4/no_robots",
            "HuggingFaceH4/ultrachat_200k",
        }
    ),
    licence_summary_total=None,
    card_read_on="2026-09-01",
)

#: `deberta-v3-base-prompt-injection-v2`. Its `datasets:` metadata names seven,
#: read from the live card on 2026-09-01, and its licence summary accounts for
#: 22 source datasets -- 1 CC-BY-3.0, 8 MIT, 1 CC0-1.0, 6 with no licence, 5
#: Apache-2.0, 1 CC-BY-4.0. The seven the summary names by name are the same
#: seven the metadata names, so 15 are counted and named nowhere.
PROTECTAI_V2 = ReferenceModel(
    model_id="protectai/deberta-v3-base-prompt-injection-v2",
    named_datasets=frozenset(
        {
            "natolambert/xstest-v2-copy",
            "VMware/open-instruct",
            "alespalla/chatbot_instruction_prompts",
            "HuggingFaceH4/grok-conversation-harmless",
            "Harelix/Prompt-Injection-Mixed-Techniques-2024",
            "OpenSafetyLab/Salad-Data",
            "jackhhao/jailbreak-classification",
        }
    ),
    licence_summary_total=22,
    card_read_on="2026-09-01",
)

#: Every model this stage is measured against. The benchmark harness runs both,
#: so contamination is screened against both: a published v1 comparison is as
#: published as a v2 one, and a corpus either model memorised flatters us the
#: same way.
REFERENCE_MODELS: tuple[ReferenceModel, ...] = (PROTECTAI_V1, PROTECTAI_V2)

#: The denylist: the union of what every reference model's card names.
#:
#: DERIVED, never written out by hand, so a model added to `REFERENCE_MODELS`
#: extends the screen by construction and cannot be registered while leaving
#: the screen behind.
#:
#: KNOWN TO BE PARTIAL, and treating it as anything else is the mistake it
#: exists to prevent. v2's card counts 22 datasets and names 7 of them; v1's
#: card names 12 and counts nothing at all. A screen over a third party's
#: training data cannot be exhaustive when the third party did not publish that
#: data.
#:
#: So this catches a case a card happens to name, and nothing more. A source
#: absent from it is NOT cleared for evaluation; it is a source whose
#: provenance nobody has established. Anything selecting an evaluation corpus
#: has to establish that separately rather than reading a pass here as a
#: guarantee.
NAMED_TRAINING_DATA: frozenset[str] = frozenset[str]().union(
    *(model.named_datasets for model in REFERENCE_MODELS)
)

#: The two the plan requires `training/sources.yaml` to record by name. A
#: strict subset of the denylist, which is the whole point: recording these two
#: is a floor, not a demonstration that the denylist is complete.
RECORDED_EXCLUSIONS = frozenset(
    {
        "jackhhao/jailbreak-classification",
        "Harelix/Prompt-Injection-Mixed-Techniques-2024",
    }
)

#: Datasets on the denylist whose licence requires attribution. Neither is in
#: the manifest today. If either is ever recorded as something this repository
#: uses, it needs an entry in `corpora/NOTICE.md` the way
#: `corpora/pii/third-party.jsonl` already does -- attribution is a condition of
#: the licence, not a courtesy, and the file that carries it is the one that
#: ships.
#:
#: Both values come from v2's licence summary, which is the only per-dataset
#: licence statement either card makes. v1's card names no licence for any of
#: its twelve and warns that some "may be subject to more restrictive licensing
#: terms, including non-commercial use provisions", so nothing from v1's list
#: belongs in this map until its own dataset card has been read.
ATTRIBUTION_REQUIRED = {
    "VMware/open-instruct": "CC-BY-3.0",
    "natolambert/xstest-v2-copy": "CC-BY-4.0",
}


#: The denylist as the screens actually compare it: base ids, pins stripped,
#: case folded. Derived from `NAMED_TRAINING_DATA` through the manifest's own
#: name grammar, never written out.
DENYLIST_BASES = frozenset(base_id(name) for name in NAMED_TRAINING_DATA)

#: The same, for the attribution map.
ATTRIBUTION_BASES = frozenset(base_id(name) for name in ATTRIBUTION_REQUIRED)


def _base_ids_in(text: str) -> frozenset[str]:
    """Every `org/name` identifier a piece of prose names, as base ids.

    URLs are removed first. A pinned Hugging Face download URL contains the
    dataset id as path segments, and a screen that read those would call a
    corpus attributed because a link to it appears somewhere on the page.

    Matched with `QUALIFIED_ID`, which is built from the same atoms as
    `SOURCE_NAME`, and compared as a set of base ids. `name in text` would have
    been shorter and would have said yes to `vmware/open-instruct` because the
    text mentioned `vmware/open-instruct-2`, which is the defect this whole
    round is about.
    """
    without_urls = re.sub(r"https?://\S+", " ", text)
    return frozenset(base_id(match.group(0)) for match in QUALIFIED_ID.finditer(without_urls))


def _contaminated(sources: Iterable[Source]) -> list[str]:
    """Evaluation sources a reference model is named as trained on. The rule.

    Compared on `base_id`, not on the recorded string. Every source in this
    tree is required to be pinned, and this repository's idiom for a pinned
    dataset is `name@revision` -- `corpora/NOTICE.md` records
    `nvidia/Nemotron-PII@b70ffaf` -- so the spelling a real manifest entry
    carries is exactly the spelling an exact-match screen returns "clean" for.
    ProtectAI's own licence summary writes one of these datasets
    `natolambert/xstest-v2-copy:1_full_compliance`, so copying a name off the
    card was enough to walk past this rule before it normalised.

    Factored out of the test that applies it to the shipped manifest so the
    same rule can be pointed at a manifest that breaks it. Today's manifest
    declares no `eval` source at all, so applying the rule to it alone would
    pass over an empty loop and prove nothing.
    """
    return sorted(
        source.name
        for source in sources
        if source.role == "eval" and base_id(source.name) in DENYLIST_BASES
    )


def _unattributed(sources: Iterable[Source], notice: str) -> list[str]:
    """Sources in use under an attribution licence that the NOTICE does not name.

    Same shape as `_contaminated`, same normalisation, and for the same reason:
    `VMware/open-instruct@1234abc` is `VMware/open-instruct` under a licence
    whose attribution term is a condition of use, and a screen that read the
    two as different corpora would let a CC-BY corpus in unattributed.

    The NOTICE is matched by identifier too, not by substring, for the reason
    `_base_ids_in` records.
    """
    attributed = _base_ids_in(notice)
    return sorted(
        source.name
        for source in sources
        if base_id(source.name) in ATTRIBUTION_BASES
        and source.role != "excluded"
        and base_id(source.name) not in attributed
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

    Screened against every model in `REFERENCE_MODELS`, not just the newest.
    The benchmark runs v1 and v2 through one harness, and a v1 number published
    beside ours is as published as a v2 one.

    Passing this is necessary and nowhere near sufficient: the denylist is the
    union of what two cards happen to name. See `NAMED_TRAINING_DATA`.
    """
    assert _contaminated(load_sources(SOURCES)) == []


@pytest.mark.parametrize(
    "name",
    [
        # Bare, one from each reference model's card.
        "hackaprompt/hackaprompt-dataset",
        "OpenSafetyLab/Salad-Data",
        # Pinned, which is the spelling a real entry carries: every source in
        # this tree must be pinned, and `name@revision` is this repository's own
        # idiom for that (`corpora/NOTICE.md` records
        # `nvidia/Nemotron-PII@b70ffaf`). Before the screen normalised, all four
        # of these returned "not contaminated".
        "hackaprompt/hackaprompt-dataset@abc1234",
        "Lakera/gandalf_ignore_instructions@deadbee",
        # The config spelling, which is how ProtectAI's own licence summary
        # writes this dataset. Copying the name off the card was enough.
        "natolambert/xstest-v2-copy:1_full_compliance",
        "hackaprompt/hackaprompt-dataset:default",
        # Case, which the Hugging Face hub does not distinguish.
        "HACKAPROMPT/Hackaprompt-Dataset@abc1234",
    ],
)
def test_the_contamination_rule_catches_a_manifest_that_breaks_it(
    tmp_path: Path, name: str
) -> None:
    """The mutation check for the test above, which today loops over nothing.

    `training/sources.yaml` declares no `eval` source yet -- a later task adds
    them once each licence is verified -- so the assertion above passes
    vacuously and would keep passing if the rule were inverted. This runs the
    same rule against a manifest that names a ProtectAI training corpus as
    evaluation data, and requires it to say so.

    Parametrised over one name from each reference model, because a rule
    screening the union has two ways to be wrong and a single fixture would
    exercise one of them. `hackaprompt/hackaprompt-dataset` is v1's and is one
    of the most obvious public injection evaluation corpora there is; it was
    outside the denylist entirely until v1 was registered.
    """
    contaminated = _manifest(
        tmp_path,
        f"- name: {name}\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"a" * 64}"\n'
        "  role: eval\n",
    )
    assert _contaminated(load_sources(contaminated)) == [name]


def test_the_denylist_is_recorded_as_partial_and_not_as_a_clearance() -> None:
    """The relationship between the constants, which is the finding itself.

    A denylist of 2 was read as the complete set of what the reference model
    was trained on, and the fix is not a denylist of 19 read the same way. What
    has to hold is that the file says the list is smaller than the thing it
    screens for, model by model, and that the two recorded exclusions are a
    strict subset rather than the whole of it.
    """
    assert RECORDED_EXCLUSIONS < NAMED_TRAINING_DATA, (
        "the recorded exclusions are no longer a strict subset of the denylist, so the "
        "manifest's two entries are being treated as the whole of it"
    )
    assert REFERENCE_MODELS, "no reference model is registered, so the denylist is empty"
    for model in REFERENCE_MODELS:
        assert model.named_datasets <= NAMED_TRAINING_DATA, (
            f"{model.model_id} is registered but its datasets are not in the denylist, so "
            "the union is not being derived from the registry"
        )
        if model.licence_summary_total is None:
            # The weaker card: it names datasets and counts nothing, so there
            # is no total to be smaller than. What must not happen is a total
            # appearing here without the prose that explains it, so the README
            # agreement test carries this case instead.
            continue
        assert len(model.named_datasets) < model.licence_summary_total, (
            f"{model.model_id}'s denylist now claims to cover every dataset its card "
            "accounts for, which is the completeness claim this test exists to refuse"
        )


def test_no_dataset_is_named_by_more_than_one_reference_model() -> None:
    """Not a rule, a fact worth pinning: the two cards name disjoint sets.

    It is what makes the union's size the sum of its parts, which the README
    states as a number. If a future card overlaps, the union stops being
    19 and the README has to say something else; this is what says so.
    """
    counted = sum(len(model.named_datasets) for model in REFERENCE_MODELS)
    assert counted == len(NAMED_TRAINING_DATA), (
        "two reference models name the same dataset, so the union is smaller than the "
        "sum of the lists and the README's arithmetic no longer holds"
    )


def test_every_external_identifier_in_this_tree_is_accounted_for() -> None:
    """The screen that does not depend on what a model happens to be called.

    The registry scan below matches one model family by name, so naming
    `meta-llama/Llama-Prompt-Guard-2-86M` as a further comparator left the
    whole module green: two models declared as scored, their training data
    screened by nothing, and no test with anything to say. A pattern is only as
    wide as the names somebody thought of.

    So this reads every `org/name` identifier in the tree and requires each one
    to be something this repository already accounts for: a registered
    reference model, a source in the manifest, a dataset on the denylist or in
    the attribution map, a corpus recorded in `corpora/NOTICE.md`, or an entry
    in `UNSCREENED_IDS` with a reason. Anything else fails, whatever it is
    called.

    Repository paths are told apart from identifiers by asking the filesystem
    rather than by a list of prefixes: `src/jamjet_guardrails` is the same shape
    as an org/model id, and what distinguishes it is that it exists. URLs go
    first, because a pinned download URL carries a dataset id as path segments.

    What this cannot see, stated rather than left to be discovered: a
    comparator written with no org prefix and outside the deberta family. That
    is why the test below requires every registered model's full `model_id` to
    appear in the README -- citing the full id is the convention this makes
    checkable.
    """
    prefixes = frozenset(entry.name.lstrip(".") for entry in ROOT.iterdir())
    accounted = set(DENYLIST_BASES) | set(ATTRIBUTION_BASES)
    accounted |= {base_id(model.model_id) for model in REFERENCE_MODELS}
    accounted |= {base_id(source.name) for source in load_sources(SOURCES)}
    accounted |= _base_ids_in(NOTICE.read_text(encoding="utf-8"))
    accounted |= {base_id(name) for name in UNSCREENED_IDS}

    unaccounted: dict[str, str] = {}
    examined: set[str] = set()
    for path in TRAINING_FILES:
        text = re.sub(r"https?://\S+", " ", path.read_text(encoding="utf-8"))
        for match in QUALIFIED_ID.finditer(text):
            token = match.group(0)
            before = text[match.start() - 1] if match.start() else " "
            if before in "./!":
                # The tail of a longer path (`./.venv-training/bin/python`) or
                # of a YAML tag (`!!python/object/apply`). An identifier is
                # cited after whitespace or a backtick, never mid-path.
                continue
            if (ROOT / token).exists() or token.split("/")[0] in prefixes:
                # A path in this repository, including one not created yet:
                # `training/artifacts/` is made by the task that exports a
                # model. The repository's own top-level names are read from the
                # filesystem rather than listed, so adding a directory does not
                # mean editing this test.
                continue
            examined.add(base_id(token))
            if base_id(token) not in accounted:
                unaccounted[token] = path.name
    # Not vacuous. Every rule above is a `continue`, and a discriminator that
    # skipped everything would leave `unaccounted` empty and this test green
    # over a tree it never looked at. The two models the stage is measured
    # against are cited by full id in the README, so they have to come out of
    # the scan.
    for model in REFERENCE_MODELS:
        assert base_id(model.model_id) in examined, (
            f"the scan did not reach {model.model_id}, so it is not reading the tree"
        )
    assert unaccounted == {}, (
        "these identifiers are named under training/ and accounted for by nothing: a "
        f"comparator whose training data nothing screens, or a name to record: {unaccounted}"
    )


def test_every_reference_model_named_in_this_tree_has_its_datasets_registered() -> None:
    """The gap N-1 found, closed by construction rather than by remembering.

    The denylist held v2's datasets while the benchmark scored v1 as well, and
    nothing could have shown that: a flat list has no place to record which
    model it came from. Now every file under `training/` is scanned for a
    ProtectAI prompt-injection model name, and a name that is not in
    `REFERENCE_MODELS` fails here. Since `NAMED_TRAINING_DATA` is derived from
    the registry, registering the model is what extends the screen.

    Equality in both directions on purpose. A model referenced and not
    registered is the gap itself; a model registered and never referenced is a
    screen against something this tree does not document measuring against,
    which is the same drift pointing the other way.
    """
    assert TRAINING_FILES, "the scan found no files, so it is checking nothing"
    found: set[str] = set()
    for path in TRAINING_FILES:
        found.update(MODEL_REFERENCE.findall(path.read_text(encoding="utf-8")))
    assert found, "no reference model is named anywhere under training/, so this is vacuous"
    registered = {model.slug for model in REFERENCE_MODELS}
    assert sorted(found - registered) == [], (
        "these models are named under training/ but carry no entry in REFERENCE_MODELS, so "
        "nothing screens their training data: " + str(sorted(found - registered))
    )
    assert sorted(registered - found) == [], (
        "these models are registered but named nowhere under training/, so the tree does "
        "not document what it is measured against: " + str(sorted(registered - found))
    )
    # And the full `org/model` id, which is the form the identifier scan above
    # can see without knowing what family a model belongs to. Citing a
    # comparator by its bare name is what let two of them into the README with
    # nothing to say about it.
    cited = _base_ids_in(TRAINING_README.read_text(encoding="utf-8"))
    for model in REFERENCE_MODELS:
        assert base_id(model.model_id) in cited, (
            f"{model.model_id} is registered but the README never cites its full id, so "
            "the convention that makes the identifier scan work is not being followed"
        )


def test_the_readme_states_the_same_partiality_the_denylist_records() -> None:
    """Two tables, and they have to agree.

    `training/README.md` states these counts in prose and this module holds
    them as data. A number in prose that counts a thing in code is a claim, and
    the last one drifted from 7 to "two" with nothing to catch it.

    Whitespace is normalised first so a sentence that wraps differently after
    an edit is still the same sentence. The claim is the words, not the fill.
    """
    readme = " ".join(TRAINING_README.read_text(encoding="utf-8").split())
    v1, v2 = PROTECTAI_V1, PROTECTAI_V2
    assert v1.licence_summary_total is None
    assert v2.licence_summary_total is not None
    unnamed = v2.licence_summary_total - len(v2.named_datasets)
    assert f"names {len(v1.named_datasets)} datasets and the card gives no total" in readme
    assert f"names {len(v2.named_datasets)} datasets and its licence summary" in readme
    assert f"accounts for {v2.licence_summary_total} source datasets" in readme
    assert f"so {unnamed} are counted and never named" in readme
    assert f"union of the two, {len(NAMED_TRAINING_DATA)} names" in readme
    # Matched with the same pattern the scan uses rather than by `in`. A bare
    # `"deberta-v3-base-prompt-injection" in readme` is satisfied by any
    # occurrence of the `-v2` name, so the v1 half of this would have been
    # true whether or not the README mentioned v1 at all.
    named = set(MODEL_REFERENCE.findall(readme))
    for model in REFERENCE_MODELS:
        assert model.slug in named, f"{model.model_id} is registered but the README omits it"


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


@pytest.mark.parametrize(
    "name",
    [
        "VMware/open-instruct",
        # The pinned and config spellings, which are what a real entry carries
        # and what ProtectAI's licence summary writes. Both returned "attributed"
        # before the screen normalised, so a CC-BY corpus could enter with no
        # NOTICE entry at all.
        "VMware/open-instruct@1234abc",
        "natolambert/xstest-v2-copy:1_full_compliance",
        "vmware/open-instruct",
    ],
)
def test_the_attribution_rule_catches_a_manifest_that_breaks_it(tmp_path: Path, name: str) -> None:
    """The same rule against a manifest that uses one without attributing it."""
    used = _manifest(
        tmp_path,
        f"- name: {name}\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        '  license: "CC-BY-3.0"\n'
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    assert _unattributed(load_sources(used), "a NOTICE that names nothing") == [name]


def test_the_attribution_rule_reads_the_notice_by_identifier_not_by_substring(
    tmp_path: Path,
) -> None:
    """The dual, and the trap inside it.

    A NOTICE that names the corpus, in any spelling, satisfies the rule; a
    NOTICE that merely contains the name as part of a longer identifier does
    not. `name in notice` said yes to the second, which is how a corpus gets
    counted as attributed by a NOTICE that attributes a different one.
    """
    used = _manifest(
        tmp_path,
        "- name: VMware/open-instruct@1234abc\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        '  license: "CC-BY-3.0"\n'
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    sources = load_sources(used)
    assert _unattributed(sources, "derived from `VMware/open-instruct`, CC-BY-3.0") == []
    assert _unattributed(sources, "derived from `VMware/open-instruct-2`, CC-BY-3.0") == [
        "VMware/open-instruct@1234abc"
    ]


def test_a_url_in_the_notice_does_not_attribute_a_corpus() -> None:
    """URLs are stripped before the NOTICE is read for identifiers.

    A pinned Hugging Face download URL carries the dataset id as path segments,
    so a NOTICE that merely links to a corpus would otherwise read as one that
    credits it. Attribution is a licence term; a hyperlink is not it.
    """
    linked = "see https://huggingface.co/datasets/VMware/open-instruct/resolve/main/x.csv"
    assert base_id("VMware/open-instruct") not in _base_ids_in(linked)


def test_two_distinct_dataset_ids_do_not_normalise_together() -> None:
    """The opposite error, which a normaliser is the natural way to introduce.

    Stripping too much collides two real corpora, and a collision here reads as
    "contaminated" for a corpus nobody was trained on, or as "attributed" for
    one nobody credited. Checked over the whole denylist rather than a sample:
    19 names have to stay 19 base ids.

    The explicit pairs are the ones a looser normaliser would join. `_` and `-`
    are different characters in a hub id, and a suffix is only a suffix after
    `@` or `:`.
    """
    assert len(DENYLIST_BASES) == len(NAMED_TRAINING_DATA), (
        "two denylisted datasets normalise to the same base id, so the screen can no "
        "longer tell them apart"
    )
    distinct = [
        ("HuggingFaceH4/no_robots", "HuggingFaceH4/no-robots"),
        ("Dahoas/hh_prompt_format", "Dahoas/synthetic-hh-rlhf-prompts"),
        ("natolambert/xstest-v2-copy", "natolambert/xstest-v2-copy-extra"),
        ("hackaprompt/hackaprompt-dataset", "hackaprompt/hackaprompt-dataset-v2"),
        ("a/b", "a/b-c"),
    ]
    for left, right in distinct:
        assert base_id(left) != base_id(right), f"{left} and {right} normalise together"


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


@pytest.mark.parametrize(
    "name", ["someone corpus", "someone//corpus", "/someone/corpus", "someone/corpus@"]
)
def test_the_loader_refuses_a_name_the_screens_cannot_read(tmp_path: Path, name: str) -> None:
    """A name outside the grammar is a source nothing can screen.

    The contamination and attribution rules compare `base_id`, which is the
    grammar's own base group. A name it cannot parse has no base id, so a
    screen would have to either guess or skip it, and skipping is the answer
    that reads as "clean". Refused at load instead, which leaves no third
    spelling that both loads and evades.
    """
    unreadable = _manifest(
        tmp_path,
        f'- name: "{name}"\n'
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="which is not"):
        load_sources(unreadable)


def test_the_loader_refuses_a_duplicate_key_inside_one_source(tmp_path: Path) -> None:
    """The one refusal the parser swap gave up, and the worst one to give up.

    `yaml.safe_load` keeps the last of two identical keys and says nothing, so
    a second `sha256:` line -- the kind a rebase or a copy-paste produces --
    pins the corpus to whichever was written second. Both values are valid on
    their own, so every screen downstream passes over the survivor and the
    recorded content hash, which is this file's entire reason to exist, quietly
    describes a different corpus.
    """
    two_digests = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"a" * 64}"\n'
        f'  sha256: "{"b" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="duplicate key 'sha256'"):
        load_sources(two_digests)


def test_the_loader_refuses_an_anchor(tmp_path: Path) -> None:
    """One field taking another's text is a manifest that does not read as it says.

    `license: *n` against `name: &n someone/corpus` gives a source whose
    licence is its own name. Every field still passes the non-empty-text check,
    so nothing downstream objects; what is wrong is that no reader of the file
    would expect it. The anchor is where it is caught, because the anchor is
    where it starts.
    """
    aliased = _manifest(
        tmp_path,
        "- name: &n someone/corpus\n"
        "  license: *n\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="only useful to an alias"):
        load_sources(aliased)


def test_the_loader_refuses_an_alias_with_no_anchor(tmp_path: Path) -> None:
    """The other arm of the same rule, and it is not the same input.

    A defined anchor is refused at its definition, so the alias arm never sees
    an alias to one. What reaches it is an alias to an anchor that does not
    exist -- and an `AliasEvent` carries that name in `.anchor`, so the anchor
    arm would refuse this input too, one message later.

    That makes the wording the only thing separating the arms, so the wording
    is what this asserts: a refusal that calls an alias an anchor sends the
    reader looking for an `&` that is not there. Matching on "an alias" alone
    did not do it, because the anchor message ends "only useful to an alias"
    and passed for exactly that reason.
    """
    dangling = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        "  license: *n\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="takes another field's text"):
        load_sources(dangling)


def test_the_loader_refuses_a_merge_key(tmp_path: Path) -> None:
    """A merge key needs no anchor when the mapping is inline, so it is refused too.

    Refusing anchors alone would leave `<<: {…}` open, and a source that
    inherits a digest is a source pinned to a corpus nobody recorded for it.
    """
    merged = _manifest(
        tmp_path,
        "- <<: {license: Apache-2.0}\n"
        "  name: someone/corpus\n"
        '  url: "https://example.invalid/corpus.csv"\n'
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="merge key"):
        load_sources(merged)


def test_the_loader_refuses_a_comment_that_would_truncate_a_plain_value(
    tmp_path: Path,
) -> None:
    """` #` inside a plain scalar is a comment, and the rest of the URL is gone.

    Spec-correct YAML and a silently different source: the old reader refused
    it outright and this restores that. The failure is invisible in the loaded
    value, which is what makes it worth a guard rather than a convention.
    """
    truncated = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        "  url: https://example.invalid/corpus.csv #real\n"
        "  license: Apache-2.0\n"
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="which YAML reads as a comment"):
        load_sources(truncated)


@pytest.mark.parametrize(
    ("url_line", "expected"),
    [
        (
            '  url: "https://example.invalid/corpus.csv#rows"\n',
            "https://example.invalid/corpus.csv#rows",
        ),
        (
            '  url: "https://example.invalid/corpus.csv" # the pinned file\n',
            "https://example.invalid/corpus.csv",
        ),
    ],
)
def test_a_quoted_value_keeps_its_hash_and_may_be_commented(
    tmp_path: Path, url_line: str, expected: str
) -> None:
    """The dual of the test above, and quoting is the answer its message gives.

    Two cases, because the refusal has two ways to be too broad and one fixture
    catches neither on its own. A `#` inside quotes is content, and a URL
    fragment is a legitimate thing for a pinned source to carry. A comment
    *after* a quoted value is a comment, because the quotes have already said
    where the value ends -- which is exactly what the error message tells the
    reader to do, so refusing it would make that advice wrong.

    Written after a mutation that dropped the `style is None` test stayed green:
    the fragment case passes either way, because the `#` sits before the
    closing quote and there is nothing after it to look at.
    """
    fragment = _manifest(
        tmp_path,
        "- name: someone/corpus\n"
        + url_line
        + "  license: Apache-2.0\n"
        + f'  sha256: "{"a" * 64}"\n'
        + "  role: train\n",
    )
    assert load_sources(fragment)[0].url == expected


def test_the_loader_refuses_a_python_tag(tmp_path: Path) -> None:
    """The dangerous half of YAML, which `SafeLoader` closes and this must keep closed.

    `ManifestLoader` adds strictness to `yaml.SafeLoader`; it would be easy to
    add it to `yaml.Loader` instead and lose the thing that matters most. A
    manifest is a file in a repository, and a repository gets pull requests.
    """
    tagged = _manifest(
        tmp_path,
        '- name: !!python/object/apply:os.system ["echo owned"]\n'
        '  url: "https://example.invalid/corpus.csv"\n'
        "  license: Apache-2.0\n"
        f'  sha256: "{"a" * 64}"\n'
        "  role: train\n",
    )
    with pytest.raises(SourceError, match="could not determine a constructor"):
        load_sources(tagged)


def test_the_loader_refuses_a_stream_it_cannot_re_read() -> None:
    """The comment check reads the source text back, so it fails closed without it.

    PyYAML fills a mark's buffer only when the document was handed to it as a
    string. Given a file object it does not, and the ` #` check would have
    nothing to look at. It raises rather than passing: a guard that cannot see
    is not a guard that agrees, and this is the one caller mistake that would
    have turned the refusal off silently.
    """
    with (
        SOURCES.open(encoding="utf-8") as stream,
        pytest.raises(SourceError, match="not read as text"),
    ):
        yaml.load(stream, ManifestLoader)


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
