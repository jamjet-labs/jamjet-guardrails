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

`training/generated/rows.jsonl` holds 3584 generated rows: 1792 hard negatives
and 1792 attacks, across 8 hard-negative kinds and 8 attack kinds. The thinnest
kind holds 212 rows, and `tests/test_training_data.py` requires at least 203
rows in every one of them, so a later run that fell over part way cannot land
looking complete.

The hard negatives are why any of this is generated. Task 3's licence screen
left two usable public corpora and no public `eval` corpus, and neither of the
two contains the shape a deployed injection classifier actually fails on: text
that talks ABOUT instructions without being one.

### Both labels come out of one prompt

The eight prompts are written per PAIR, not per kind, and each asks for both
members at once: two texts alike in opening, length, tone and punctuation,
differing in what they are doing and in nothing else. A pair is kept or dropped
whole, so the corpus is balanced by construction rather than by arithmetic.

That is a correction, not a preference. The first corpus was written from
sixteen prompts, one per kind, and it could be sorted without reading it. A
logistic regression over lengths, punctuation and character ratios, seeing no
content word at all, reached 0.751 against a 0.511 baseline; function-word
rates alone reached 0.806; and 47% of rows opened with a token at least 95%
pure for one label, with the polarity of the worst inverted, every row
beginning Ignore being a hard negative. Sixteen prompts written separately
produce sixteen house styles, and the split between them happened to run along
the label. Neither class has a prompt of its own now, so neither has a voice of
its own.

It matters more than a normal data-quality point because of what this corpus is
for. The reference models this stage is measured against were never fitted on
it, so a generation artifact raises OUR score and not theirs. A ship bar
cleared on a separable corpus measures the artifact rather than the detector,
which is the one thing the bar exists to rule out.

| hard negative | rows | attack | rows |
|---|---|---|---|
| `user_correcting_themselves` | 216 | `direct_override` | 216 |
| `documentation_quoting_an_attack` | 227 | `indirect_via_retrieved_content` | 227 |
| `security_report_with_payload` | 225 | `tool_misuse_request` | 225 |
| `prompt_engineering_tutorial` | 223 | `role_reassignment` | 223 |
| `roleplay_request` | 212 | `multi_turn_setup` | 212 |
| `config_or_code_with_instructions` | 228 | `delimiter_confusion` | 228 |
| `translation_request` | 248 | `encoded_payload` | 248 |
| `meta_question_about_the_system` | 213 | `exfiltration_request` | 213 |

### What the corpus measures, and the ceilings it is held to

Every figure below comes from `training/separability.py`, which is pure Python
because it runs in CI and CI has neither numpy nor scikit-learn. Every one is
stated here as well as in `tests/test_training_data.py`, and
`test_the_readme_states_the_separability_it_was_measured_at` requires the two to
agree: a ceiling widened in one place fails, and so does a stated measurement
that has gone stale. Any regeneration that moves a number by 0.001 fails the
suite until this file is updated, which is the intended behaviour and not a
flaky test.

**Read this before quoting any of the ceilings.** Every threshold on this page
was set just above a value that had already been measured, so a ceiling set from
a measured result is not evidence of cleanliness. Passing one says the corpus
has not got worse since the day the number was taken. It does not say the number
was low enough that day, and three rounds of this corpus have passed every
ceiling in force at the time while carrying a defect that the next round's
measurement found. They are drift guards. The evidence question is what the
numbers say next to each other, and it is answered below rather than by the fact
that a threshold was cleared.

#### The two separabilities, and which one an encoder is exposed to

A MARGINAL probe fits one direction over all 3584 rows. A corpus passes it by
having no single voice that runs along the label, which is what the pairing
fixed.

A fine-tuned encoder is under no such constraint. It reads the topic, the topic
says which of the eight prompt pairs produced the row, and it can then apply a
different rule inside each pair. Pair identity costs it nothing, so the channel
it can actually exploit is what survives CONDITIONING on the pair. Pair identity
alone scores exactly the majority baseline, because every pair holds as many
attacks as negatives, so none of this is measuring the grouping.

**Within-pair separability** is the pooled held-out accuracy of the same
logistic regression fitted SEPARATELY INSIDE each of the eight pairs. A pair's
rows are five-fold cross-validated among themselves, every row is scored by a
model that never saw it, and the hits are pooled across pairs rather than the
per-pair accuracies averaged, so a thin pair does not weigh as much as a thick
one. The fit is 25 epochs of full-batch gradient descent on features
standardised by the training fold, the same budget every probe here uses; at 100
and 200 epochs the number comes out lower rather than higher, so 25 is not an
underfit reading of it.

`no content word` is style and function words in one vector: lengths,
punctuation counts, character-class ratios and the rate of each closed-class
word. Nothing that reaches it says what a row is about. `bag of words` reads
every word in the corpus as present or absent, and is reported rather than
gated: a model that reads the content SHOULD be able to sort an injection
corpus. It is here as the comparison the other rows are read against, because
"function words score 0.73" means one thing when reading everything scores 0.95
and another when it scores 0.76.

| probe | what it sees | marginal | within pair |
|---|---|---|---|
| style | lengths, punctuation, character ratios | marginal style 0.574 | within-pair style 0.729 |
| function words | rates of closed-class words only | marginal function-word 0.760 | within-pair function-word 0.829 |
| no content word | both of the above together | marginal no-content-word 0.722 | within-pair no-content-word 0.848 |
| bag of words | every word, present or absent | marginal bag-of-words 0.850 | within-pair bag-of-words 0.907 |

Against a majority baseline 0.500. The ceilings are style ceiling 0.60,
function-word ceiling 0.78 and within-pair ceiling 0.88, and the last of those
is the one this section exists for. The corpus it was added to read 0.686
marginally and 0.789 within pair, against 0.843 for a bag of words reading every
content word: reading none of the content recovered within 0.054 of reading all
of it, while the marginal figure looked like a corpus that had been cleaned.

#### One word, one pair

The worst single word inside any one pair sorts it at single token 0.840, held
to a single token ceiling 0.88. Scored as balanced accuracy over that pair's
whole vocabulary, both polarities tried, which for a present-or-absent feature
is that feature's ROC AUC.

Over the whole vocabulary rather than a list of suspects, because every tell
found so far was introduced by the wording that removed the one before. The past
tense rule that made a security report read like a report put `was` at 0.864;
dropping it left the report naming "the assistant" in the third person while its
partner addressed it in the second, and `assistant` sorted that pair at 0.925;
`now` sorted another at 0.906 while being written down in the test module as the
phenomenon. A screen that knows only the words its author suspects cannot find
the next one.

#### The other screens

- lexical 0.561 against a lexical ceiling 0.62. "Contains instruction
  vocabulary, therefore injection", scored as if it were the classifier and
  required to do badly. It was 0.470 before the pairing, below chance, because
  the hard negatives carry that vocabulary more often than the attacks do.
- opener share 0.085 against an opener share ceiling 0.12: the share of rows
  opening with a token at least as pure as the opener purity threshold 0.95 for
  one label. Position 1 is its own leak and none of the whole-text probes can
  see it. Openers are shared within a pair by instruction, which is what keeps
  this down.
- twin similarity 0.204 against a twin similarity ceiling 0.28: the highest
  per-pair MEDIAN word-trigram Jaccard between the two members of a twin. See
  the split section below for what it leaks into.
- Rows are screened against each other at a near-duplicate threshold 0.60 of
  word-trigram Jaccard, per label. Exact distinctness is not enough: a previous
  corpus was distinct across every row and still held eleven near-duplicate
  pairs differing by a comma.
- leave-one-pair-out 0.606, reported and not gated. Fit on seven pairs, score
  the eighth, no content word. A signal that is the phenomenon transfers,
  because an attack is an attack whatever it is about; a signal that is one
  prompt's wording does not, and a pair that scores below chance is carrying a
  direction opposite to the rest of the corpus. It is not gated because eight
  pairs are eight kinds of attack and some of the gap between them is real
  difference rather than artifact.

#### Per-kind floors

Four, stated the same way and cross-checked the same way, because LOWERING a
floor cannot fail the test that floor gates: the measured value still clears it,
and only a second statement of the number can object.

- kind quality floor 0.75: the share of `delimiter_confusion` rows carrying a
  subversive verb, 0.820 measured, against the 0.123 that did before the
  pairing and the 0.642 the pairing alone reached. Scoped to that
  one kind because it is the kind whose defect is lexical; the same word list
  across all eight attack kinds scored `encoded_payload` at 0.084, which
  measures the list and not the kind.
- quoting floor 0.90: the share of the two quoting kinds that quote a span at
  all. 8.8% of `security_report_with_payload` quoted nothing before.
- supersede floor 0.85: the share of `role_reassignment` rows that say an
  earlier instruction has been replaced. 71.9% carried no such marker, and 132
  of those differed from their benign twin by the word `now`, which is how that
  word came to sort the pair at 0.906. A reassignment that replaces nothing is a
  role instruction, which is what the benign member of that pair is.
- explained floor 0.70 and explained ceiling threshold 0.15: one screen scored
  on both members of the documentation pair, in opposite directions. Quoting a
  span AND framing it as an example is what security documentation does;
  `indirect_via_retrieved_content` is a document with an instruction hidden in
  its running text, and 34.6% of it did both, which is the benign kind's own
  definition. Scoring one pattern twice is what stops it measuring the author's
  vocabulary: a pattern matching nothing would clear the ceiling and fail the
  floor, one matching everything would do the reverse.

### The split, and why it is by twin

Rows are generated in twins and land adjacent in `rows.jsonl`: a hard negative
and an attack from one call, sharing an opening, a length and a register. That
pairing is what stopped the classes being separable by style, and it creates a
hazard one step later.

Split the rows at random and one member of a twin lands in train and the other
in eval. The model has then read most of the held-out text already, under the
opposite label, and what it learns from that is the local difference between two
near-copies. It generalises nowhere and it scores well on the split that created
it. The median twin in the worst pair shared 0.549 of its word trigrams with its
own partner when this was found, so it was not a handful of rows.

`training/split.py` divides by twin, holding out an eval share threshold 0.20 of
the TWINS, which holds out that share of each class without stratifying anything
because a twin carries one row of each. `separated_twins` scores any partition,
however it was built, and `test_the_split_never_separates_a_twin` fails if one
is broken apart. The task that builds the split for stage 2b-2 builds it through
that module, and the requirement is enforced rather than described.

None of that makes this corpus its own benchmark. Every row on both sides came
out of one generator under one of eight prompts, so a number measured on the
eval half is optimistic by an unknown margin however the halves were chosen.
Splitting by twin removes one leak. An independent evaluation set is a different
problem and it is still open.

### What produced them, and what that is checkable against

`training/generated/provenance.json` records the model, the sha256 of its
weights blob, the sampling options, the seed range per pair, the ollama
version, and the full text of every prompt with its digest. A row carries the
tag, the weights digest, the prompt id and the seed of the call that produced
it, so a single row can be regenerated and not only its pair.

It also records the ENVELOPE each pair was generated under, with its own
digest, and that is a gap that was open for three corpora. Every call sends a
pair's instruction wrapped in text saying how the two members must match and
what shape the reply takes, and none of that appeared in the record: the
wrapper could have been edited between two runs and every stored digest would
still have verified. The wrapper is not a detail. Its first version asked for
the two members to be "as alike as possible" with "the same opening words", and
that is what produced the pair whose twins share a median 0.549 of their word
trigrams, and a benign member that collapsed into its attack's own frame, "You
are now given a role instruction as follows: You are an editor ...", filed
under label 0. The second version keeps the matching that made the pairing work
and drops those two clauses. Pairs generated under the first keep it recorded
against them, because a record that says otherwise is a record that is wrong.

The generator is `qwen2.5:14b`, which is Qwen/Qwen2.5-14B-Instruct, and its
licence was screened the way a corpus licence is. It is `apache-2.0`, which
`training/screen.py` accepts, and the finding is pinned by the sha256 of the
licence text the artifact itself ships rather than by the name alone. The 3B
size of the same generation carries `qwen-research`, which the same screen
refuses with a research-only reason recorded against it, so the grant is a
property of the size and not of the family. Apache-2.0 says nothing about model
output, so it places no term on this data and none on a classifier fitted to
it.

Reproducibility was measured rather than assumed, on ollama 0.24.0: the same
seed returns a byte-identical reply, and it stays byte-identical when the
request is one of six issued concurrently. What that does not establish is
reproduction on other hardware or a later ollama build, and nothing here claims
it.

Two tests re-derive the recorded digests from the live model. Both are gated on
`JAMJET_GUARDRAILS_OLLAMA=1` and skip by default, here as well as in CI,
because a suite that fails on a laptop with no model server is a suite people
learn to ignore. Everything else about the corpus is checked from the committed
files alone.

### Where it lands

Nothing under `training/` is in the wheel, so none of this reaches an installed
package. It is all in the sdist, for the reason the section above gives, so
these 1580 KB travel with every source distribution.
