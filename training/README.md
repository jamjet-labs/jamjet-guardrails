# training/

Build tooling for the stage 2b injection classifier. Nothing in this directory
ships in any artifact this repository publishes.

## Why it is a separate tree

`jamjet-guardrails` declares `dependencies = []`, and that is the property that
lets the core install into a Lambda without anyone opting into a machine
learning stack. The tooling that produces the classifier needs torch,
transformers and onnxruntime; the package that consumes the classifier needs
none of them.

Two things keep those apart, and both are mechanical rather than a convention
somebody remembers:

- The wheel is built from `packages = ["src/jamjet_guardrails"]`, so this
  directory is outside it. `tests/test_packaging.py` asserts the built
  distribution declares no runtime dependency.
- The pins live in `training/requirements.txt`, which is referenced by no
  `[project]` table in `pyproject.toml` and is never installed into `.venv`.
  One name on that list, `pyyaml`, does also appear in the `dev` extra, because
  `training/fetch.py` reads the manifest with it and the screens over that
  manifest run in CI. A dev extra is not a runtime dependency and cannot become
  one by accident: `tests/test_packaging.py` reads the built metadata and
  filters out `extra ==` markers. The machine learning pins are in no
  `[project]` table at all.

`.venv` at the repository root is the *package's* environment: pytest, ruff,
mypy and an editable install. Installing training dependencies into it would
make the isolation invisible rather than absent, which is worse, because the
suite would keep passing while an import that only works locally crept in.

## The training virtualenv

From the repository root:

    python3.13 -m venv .venv-training
    ./.venv-training/bin/python -m pip install -r training/requirements.txt

`.gitignore` carries `/.venv-training/` as its own rule. The `.venv/` rule
above it does not reach this directory: a gitignore pattern ending in `/`
matches a directory of exactly that name, and `.venv-training` is a different
name. Written out because the first draft of this file claimed the opposite,
and a venv holding a gigabyte of torch is not something to find out about from
`git status`.

### From fetch to export

There is no command sequence to write down yet. `training/fetch.py` is the only
entry point this tree has, and nothing calls it. The scripts that build the
splits, fine-tune the encoder, export to ONNX and score the ship bar arrive
with the tasks that write them, and the sequence lands here when the commands
it names exist. A sequence documented ahead of its scripts is a list of
commands that do not run, which is the class of claim this repository spends
most of its effort not making.

### Why 3.13 and not 3.14

Wheel availability, read from the PyPI JSON API on 2026-09-01 for the exact
pins in `training/requirements.txt`:

| Pin | CPython wheels published | sdist |
|---|---|---|
| `torch==2.8.0` | cp310-cp313, macOS arm64 and manylinux x86\_64 | none |
| `onnxruntime==1.23.0` | cp310-cp313, macOS arm64 and manylinux x86\_64 | none |
| `onnx==1.19.0` | cp310-cp313 macOS universal2 and manylinux; cp314 manylinux aarch64 only | yes |
| `scikit-learn==1.7.2` | cp310-cp314, macOS arm64 and manylinux x86\_64 | yes |
| `transformers==4.57.1` | pure Python, one wheel for every interpreter | yes |
| `pyyaml==6.0.3` | cp310-cp314, macOS arm64 and manylinux x86\_64 | yes |

Neither torch 2.8.0 nor onnxruntime 1.23.0 publishes an sdist, so on CPython
3.14 there is nothing to fall back to: the install fails outright rather than
building from source. 3.13 is the highest interpreter every pin covers.

The set was resolved together on macOS arm64 under CPython 3.13.2 with
`pip install --dry-run -r training/requirements.txt`, which reported 37
distributions to install and no conflict.

This is a different question from the one
`docs/measurements/2026-09-01-phase2b-onnxruntime-support.md` answers, and the
two floors have separate causes. That document records why a *distribution*
that depends on a bare `onnxruntime` cannot claim `>=3.10`: onnxruntime stopped
publishing cp310 wheels after 1.23.2 and publishes no sdist at all, so a
`>=3.10` claim installs cleanly on 3.10 today and stops being true the moment
the resolver picks a newer release. The pin above is older than that cut, which
is why this tree's floor and the shipped distribution's floor do not have to
agree.

## What is committed and what is not

`.gitignore` carries `/data/` and nothing else from this tree. A file written
under `training/` is therefore committed by default and a file written under
`data/` is not, so the split has to be made by choosing where a script writes:

- `data/` holds raw downloads and intermediate splits. Nothing there is a
  published input. Every download is pinned by a recorded sha256, so a
  re-fetch is either byte-identical or loud.
- `training/` holds anything a published number is measured on or measured by.
  A number that describes an artifact nobody else can obtain is not a
  measurement.
- `training/generated/` holds the synthetic corpus and the record of what
  produced it. Committed for the same reason: a classifier is fitted on it, so
  a number measured on the resulting model is a number measured through this
  file. It is described under [Generated data](#generated-data) below.
- `training/artifacts/` is therefore committed, and it is named here rather
  than left to the general rule because it is the directory that makes the
  published numbers reproducible: the exported model, its tokenizer and the
  ship-bar record. It does not exist yet. The task that exports a model creates
  it, and nothing in `.gitignore` reaches it.

The `/data/` rule is anchored with a leading slash. An unanchored `data/`
matches a directory of that name at any depth, so it would also swallow one
added later under `src/` or `corpora/`, without a word. No directory named
`data` exists anywhere in the tree today -- `git ls-files` finds none -- which
is the reason to write the rule for the one that does not exist yet rather
than after it does.

`fetch` will not write outside that directory. Its `into` argument defaults to
the repository-root `data/`, and a destination inside the repository but
outside it is refused rather than created: `training/data/` is the natural
thing for a later script to type, and `.gitignore` does not reach it.

## What ships and what does not

Measured on 2026-09-01 by building both distributions with `python -m build`
and listing what came out:

- The wheel carries no file from this tree, which is what
  `packages = ["src/jamjet_guardrails"]` buys.
- The sdist carries all of it.

The second is not an oversight. The sdist ships `tests/`, and
`tests/test_training_data.py` imports `training.fetch`, so an sdist without
this tree would be an sdist whose own test suite fails to collect. The
consequence is worth stating rather than discovering: nothing in
`pyproject.toml` excludes anything under `training/` from the sdist, so a large
file added here is a large file added to every source distribution.

## Sources

`training/sources.yaml` is the manifest -- every corpus this tree may touch,
with where it came from, the licence it carries, the digest it hashed to, and
what it may be used for. `training/fetch.py` reads it and downloads against
the recorded hash; `tests/test_training_data.py` screens it in CI.

`role` decides what a source may be used for, and the three values are not
interchangeable. As of 2026-09-01 the manifest holds 10 sources:

| `role` | sources | meaning |
|---|---|---|
| `train` | 2 | may be fitted on |
| `eval` | 0 | may be scored on |
| `excluded` | 8 | neither, with the reason recorded in the entry itself |

The counts are in the table because a number in prose that counts rows in a
file is a claim like any other, and this one went stale the moment the manifest
grew past the two entries the plan required by name.
`test_the_readme_states_the_roles_the_manifest_records` recomputes all three
from `training/sources.yaml` and fails when they disagree.

**No public corpus carries `role: eval`, and that is a decision.** The
contamination denylist below is the union of what two model cards happen to
name, and neither card names all of it: v2's licence summary counts more source
datasets than it names, and v1's card gives no total at all. A public injection
corpus published before those models is a plausible member of what they count
and never name, so absence from the denylist is not provenance. Evaluation
stays where the classifier design puts it: our own held-out rows, and a
third-party benchmark we do not control. The counts themselves are stated once,
below, where a test recomputes them.

Four rules hold this manifest to more than a reviewer's attention, and they
are enforced in different places, which is worth knowing before relying on any
of them:

- **A source that is trained from or measured on carries a 64-character
  digest.** Enforced by `load_sources`, so it fails at load wherever the
  manifest is read. The one recorded absence, `unavailable`, is confined to
  `excluded`,
  because that is the only role nothing is measured on. Flipping such an entry
  to `train` or `eval` raises at load rather than shipping an unpinned corpus.
- **Nothing either reference model is named as having trained on may be an
  evaluation source, and this screen cannot see all of it.** Enforced by
  `tests/test_training_data.py` rather than by `load_sources`, because the rule
  needs a list of names that is not part of the manifest.

  This stage is measured against two ProtectAI models,
  `deberta-v3-base-prompt-injection` and its successor
  `deberta-v3-base-prompt-injection-v2` -- in full,
  `protectai/deberta-v3-base-prompt-injection` and
  `protectai/deberta-v3-base-prompt-injection-v2`, and the full id is the form
  a comparator has to be cited in, because that is the form a screen can
  recognise without knowing what the model is called. Contamination is screened
  against both. Scoring a detector on a corpus either model memorised publishes
  memorisation as recall, and a v1 number published beside ours is as published
  as a v2 one.

  `REFERENCE_MODELS` holds one entry per model with the datasets its card
  names, and the denylist `NAMED_TRAINING_DATA` is their union. The v1 card's
  `datasets:` metadata names 12 datasets and the card gives no total anywhere:
  its training-data section says only that the model was trained on "a custom
  dataset from multiple open-source ones", so how much it withholds is not a
  number anyone has. The v2 card's metadata names 7 datasets and its licence
  summary accounts for 22 source datasets -- 1 CC-BY-3.0, 8 MIT, 1 CC0-1.0, 6
  with no licence, 5 Apache-2.0, 1 CC-BY-4.0 -- so 15 are counted and never
  named anywhere. The union of the two, 19 names, is everything the two cards
  disclose between them, and no dataset appears on both lists.

  What that makes the list is a known-partial denylist. A source it does not
  match is not a source it cleared; it is a source whose provenance nobody has
  established. A screen over a third party's training data cannot be
  exhaustive when the third party did not publish that data, so choosing an
  evaluation corpus means establishing its provenance separately and reading a
  pass here as a floor rather than a guarantee. The two entries in
  `training/sources.yaml` are the two the plan requires by name, not the whole
  of what either reference model saw.

  A third reference model cannot leave a silent gap the way the second one did.
  `test_every_reference_model_named_in_this_tree_has_its_datasets_registered`
  scans every file under `training/` for a model of this family and fails on a
  name that carries no `REFERENCE_MODELS` entry, and the denylist is derived
  from that registry rather than written out, so registering a model is what
  extends the screen.

  Two names on the list carry attribution licences, per v2's card:
  `VMware/open-instruct` (CC-BY-3.0) and `natolambert/xstest-v2-copy`
  (CC-BY-4.0). Neither is in the manifest. If either is ever recorded as a
  source this repository uses, it needs an entry in `corpora/NOTICE.md`
  alongside the one for `corpora/pii/third-party.jsonl`, because attribution
  is a condition of those licences rather than a courtesy.
  `test_a_source_under_an_attribution_licence_is_named_in_the_notice` fails if
  it does not get one. v1's card names no licence for any of its twelve and
  warns that some may carry non-commercial terms, so nothing from that list can
  be recorded until its own dataset card has been read.

  That rule reads two things, not one. The list of names above covers corpora
  this manifest has never recorded a licence field for; the licence each source
  declares covers everything else, and it is the arm that does the work here,
  because not one source in this manifest is on the list. It caught
  `yanismiraoui/prompt_injections` on the day it was written: Apache-2.0, in
  role `train`, and section 4(d) of that licence asks for the upstream NOTICE
  to travel with the work.

- **A source that is trained from or measured on carries a licence this
  repository can ship under, and the rule is an allowlist.** Enforced by
  `test_no_source_this_repository_uses_carries_a_licence_it_cannot_ship` over
  `licence_refusal` in `training/screen.py`. This library is Apache-2.0 and the
  artifact its data produces gets installed by people who will use it
  commercially, so a non-commercial, share-alike, research-only or undeclared
  corpus cannot be fitted into it, and none of that is curable downstream.

  Written as an allowlist rather than as a list of forbidden terms, because a
  denylist fails open: the restriction nobody thought of reads as clean. 4 of
  the 8 excluded entries are refused here, and for four different reasons --
  one non-commercial, one share-alike, one declaring no licence at all, and one
  declaring two different licences in the same card. The last two are recorded
  in the manifest as `none-declared` and `conflicting`, which are not SPDX
  identifiers and are not pretending to be; they are what those cards did
  instead of granting a licence, and writing them down is how the file avoids
  guessing a grant into existence.

- **A source is screened on its VALUES, not only on its metadata.** Enforced by
  `fingerprint_hits` in `training/screen.py`, run over a corpus when it is
  fetched, with the outcome and the date recorded in that entry's note.

  Phase 1 is why. `beki/privy` advertises MIT, and 1,068 of the 3,101,988 lines
  of the smallest member of its pinned archive carry a Fake Name Generator
  house domain, which makes those values dual GPLv3 / CC-BY-SA-3.0-US whatever
  the tag says. Its entry is the worked example this file keeps deliberately: a
  corpus whose licence field passes the rule above and whose rows do not.

  The screen runs where a corpus is fetched rather than in CI, which has
  neither network nor corpora. What CI holds is the screen itself -- that every
  house domain matches, in mixed case as well as lower; that the excerpt it
  reports back is bounded, because a screen that prints what it found has
  published it; and that the house domains it covers are the ones
  `tests/test_corpora.py` rejects in the committed corpora, in the number
  `docs/conformance.md` publishes, since three copies of one list drift and
  every side looks right alone.

Every URL that carries a digest is pinned to a commit rather than to a branch,
and `test_every_url_in_the_manifest_is_pinned_to_a_revision_or_carries_no_digest`
is what holds it there. A branch under a recorded hash either starts failing
verification, which is at least loud, or gets its hash updated to match, which
changes the corpus under every number measured on it.

6 entries carry no digest, and the absence means three different things, which
each entry's note says outright. One cannot be fetched at all: every Hugging
Face endpoint for it answered HTTP 401. Four were refused on their licence and
never downloaded, because a corpus this repository may not use is not one it
should be keeping a copy of to prove a point, and one of those four is gated as
well, so it could not have been hashed either way. The last was refused on its
own authors' recommendation rather than on its licence. `load_sources` confines
an absent digest to `excluded` and requires a note beside it in every case.

The count in that paragraph was wrong when it was written -- it said five --
and `test_the_readme_states_how_many_entries_carry_no_digest` is why it is not
wrong now.

The manifest is read with PyYAML, through `ManifestLoader` in
`training/fetch.py`. PyYAML lives in the `dev` extra beside pytest, ruff and
mypy, so `pip install -e ".[dev]"` -- what every CI leg runs -- brings it in and
the screens run everywhere.

Values are validated at the boundary rather than trusted: YAML types what it
reads, and `role: yes` is a bool, so every field is required to be non-empty
text before anything looks at what it says.

`ManifestLoader` is `yaml.SafeLoader` with four refusals added back, because
plain YAML admits things the hand-rolled reader refused and each of them
changes what a source says without reporting it:

| refused | what it would otherwise do |
|---|---|
| a repeated key inside one entry | YAML keeps the last, so two `sha256:` lines pin the corpus to whichever was written second |
| an anchor or an alias | one field takes another's text, and the file stops reading as it says |
| a `<<:` merge key | a source inherits a digest recorded for a different corpus |
| a ` #` after a plain value | YAML reads it as a comment, so an unquoted URL loses its fragment |

A `#` inside quotes is content and stays, which is why every URL in the
manifest is quoted. `SafeLoader` itself closes the half that matters most:
`!!python/object/apply` reaches the caller as an unconstructible tag rather
than a call, and `test_the_loader_refuses_a_python_tag` keeps it that way.

An earlier revision of this tree hand-rolled a YAML subset parser here, on the
belief that a PyYAML entry in any `[project]` table would break the package's
zero-dependency promise. It would not. The promise is
`[project].dependencies = []`, and
`tests/test_packaging.py::test_the_installed_distribution_declares_no_runtime_dependencies`
reads the built metadata and filters out `extra ==` markers, which is exactly
how pytest, ruff and mypy already live in the `dev` extra. The parser cost 101
lines at `b06b881` -- `_read_manifest`, `_scalar` and the three line-shape
regexes, countable from `git show b06b881:training/fetch.py` -- and its only
check against real YAML was a test that skipped on every CI leg.

## Generated data

`training/generated/rows.jsonl` holds 759 generated rows: 375 hard
negatives and 384 attacks, across 8 hard-negative kinds and
8 attack kinds. The thinnest kind holds 41 rows, and
`tests/test_training_data.py` requires at least 41 rows in every one of
them, so a later run that fell over part way cannot land looking complete.

The hard negatives are why any of this is generated. Task 3's licence screen
left two usable public corpora and no public `eval` corpus, and neither of the
two contains the shape a deployed injection classifier actually fails on: text
that talks ABOUT instructions without being one. Public benign sets are ordinary
prose, which a classifier separates from an injection without learning anything
worth having.

| kind | `label` | rows | prompt |
|---|---|---|---|
| `user_correcting_themselves` | 0 | 47 | v1 |
| `documentation_quoting_an_attack` | 0 | 41 | v2 |
| `security_report_with_payload` | 0 | 48 | v2 |
| `prompt_engineering_tutorial` | 0 | 47 | v1 |
| `roleplay_request` | 0 | 48 | v2 |
| `config_or_code_with_instructions` | 0 | 48 | v2 |
| `translation_request` | 0 | 48 | v1 |
| `meta_question_about_the_system` | 0 | 48 | v1 |
| `direct_override` | 1 | 49 | v1 |
| `indirect_via_retrieved_content` | 1 | 49 | v4 |
| `role_reassignment` | 1 | 47 | v1 |
| `delimiter_confusion` | 1 | 47 | v2 |
| `encoded_payload` | 1 | 48 | v2 |
| `multi_turn_setup` | 1 | 48 | v2 |
| `tool_misuse_request` | 1 | 48 | v2 |
| `exfiltration_request` | 1 | 48 | v1 |

### What produced them, and what that is checkable against

`training/generated/provenance.json` records the model, the sha256 of its
weights blob, the sampling options, the seed range per kind, the ollama version,
and the full text of every prompt with its digest. A row carries the tag, the
weights digest and the prompt id; the rest belongs to the run rather than to the
row.

The generator is `qwen2.5:14b`, which is Qwen/Qwen2.5-14B-Instruct, and its licence
was screened the way a corpus licence is. It is `apache-2.0`, which
`training/screen.py` accepts, and the finding is pinned by the sha256 of the
licence text the artifact itself ships rather than by the name alone. That
distinction earns its keep here: the 3B size of the same generation carries
`qwen-research`, a research-only licence the same screen refuses, so the grant
is a property of the size and not of the family. Apache-2.0 says nothing about
model output, so it places no term on this data and none on a classifier fitted
to it.

Reproducibility was measured rather than assumed, on ollama 0.24.0: the same
seed returns a byte-identical reply, and it stays byte-identical when the
request is one of six issued concurrently. What that does not establish is
reproduction on other hardware or a later ollama build, and nothing here claims
it.

Two tests re-derive the recorded digests from the live model. Both are gated on
`JAMJET_GUARDRAILS_OLLAMA=1` and skip by default, here as well as in CI, because
a suite that fails on a laptop with no model server is a suite people learn to
ignore. Everything else about the corpus is checked from the committed files
alone.

### The prompts were rewritten after their output was read

9 of the 16 prompts are past v1, because reading a sample of what
they produced showed they were not producing it. Two examples, both of which
every other test in this module would have passed:

- `documentation_quoting_an_attack` and `security_report_with_payload` read
  "injection" as SQL injection and wrote `admin' OR '1'='1` and `DROP TABLE
  users`. Fluent, on topic for a different topic, and no use to a classifier
  that has to tell prose about a model from an instruction to one.
- `multi_turn_setup` wrote ordinary workplace messages: "Legal's reviewed our
  proposal", "The boss gave us the green light". Under `label = 1` those rows
  teach a classifier that a status update from a colleague is an attack, which
  spends precision on the traffic a deployed detector sees most of.

`prompt_id` carries the revision, so a row generated under one wording and a row
generated under another are not recorded as the same thing.

One filter exists for the same reason. Asked for a document with an instruction
planted inside it, the model addressed the planted sentence to "Qwen" in three
replies out of four, because that is the name it answers to. Left in, "says
Qwen" would have been the cheapest rule separating the two classes, and the
classifier would have learned which model wrote its training data rather than
what an injection is. The prompt now asks for a generic address and the
generator drops what still gets through.

### Where it lands

Nothing under `training/` is in the wheel, so none of this reaches an installed
package. It is all in the sdist, for the reason the section above gives, so
these 279 KB travel with every source distribution.
