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
interchangeable:

| `role` | meaning |
|---|---|
| `train` | may be fitted on |
| `eval` | may be scored on |
| `excluded` | neither, with the reason recorded in the entry itself |

Two rules hold this manifest to more than a reviewer's attention, and they
are enforced in different places, which is worth knowing before relying on
either:

- **A source that is trained from or measured on carries a 64-character
  digest.** Enforced by `load_sources`, so it fails at load wherever the
  manifest is read. The one recorded absence, `unavailable`, is confined to
  `excluded`,
  because that is the only role nothing is measured on. Flipping such an entry
  to `train` or `eval` raises at load rather than shipping an unpinned corpus.
- **Nothing ProtectAI is named as having trained on may be an evaluation
  source, and this screen cannot see all of it.** Enforced by
  `tests/test_training_data.py` rather than by `load_sources`, because the rule
  needs a list of names that is not part of the manifest. This stage measures
  against
  `deberta-v3-base-prompt-injection-v2`, and scoring a detector on a corpus
  that model memorised publishes memorisation as recall. The `datasets:`
  metadata on their model card names 7 datasets, and
  `PROTECTAI_NAMES_AS_TRAINING_DATA` in `tests/test_training_data.py` carries
  all 7. The same card's licence summary accounts for 22 source datasets --
  1 CC-BY-3.0, 8 MIT, 1 CC0-1.0, 6 with no licence, 5 Apache-2.0, 1 CC-BY-4.0
  -- so 15 are counted and never named anywhere.

  What that makes the list is a known-partial denylist. A source it does not
  match is not a source it cleared; it is a source whose provenance nobody has
  established. A screen over a third party's training data cannot be
  exhaustive when the third party did not publish that data, so choosing an
  evaluation corpus means establishing its provenance separately and reading a
  pass here as a floor rather than a guarantee. The two entries in
  `training/sources.yaml` are the two the plan requires by name, not the whole
  of what the reference model saw.

  Two names on the list carry attribution licences, per the same card:
  `VMware/open-instruct` (CC-BY-3.0) and `natolambert/xstest-v2-copy`
  (CC-BY-4.0). Neither is in the manifest. If either is ever recorded as a
  source this repository uses, it needs an entry in `corpora/NOTICE.md`
  alongside the one for `corpora/pii/third-party.jsonl`, because attribution
  is a condition of those licences rather than a courtesy.
  `test_a_source_under_an_attribution_licence_is_named_in_the_notice` fails if
  it does not get one.

Every URL that carries a digest is pinned to a commit rather than to a branch,
and `test_every_url_in_the_manifest_is_pinned_to_a_revision_or_carries_no_digest`
is what holds it there. A branch under a recorded hash either starts failing
verification, which is at least loud, or gets its hash updated to match, which
changes the corpus under every number measured on it. The one entry without a
digest is the one nothing can be downloaded from at all.

The manifest is read with `yaml.safe_load`. PyYAML lives in the `dev` extra
beside pytest, ruff and mypy, so `pip install -e ".[dev]"` -- what every CI leg
runs -- brings it in and the screens run everywhere. Values are still validated
at the boundary rather than trusted: YAML types what it reads, and `role: yes`
is a bool, so every field is required to be non-empty text before anything
looks at what it says.

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
