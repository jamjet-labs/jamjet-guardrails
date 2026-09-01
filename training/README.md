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
- The pins live in `training/requirements.txt`, which is not referenced by any
  `[project]` table in `pyproject.toml` and is never installed into `.venv`.

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

### Why 3.13 and not 3.14

Wheel availability, read from the PyPI JSON API on 2026-09-01 for the exact
pins in `training/requirements.txt`:

| Pin | CPython wheels published | sdist |
|---|---|---|
| `torch==2.8.0` | cp310-cp313, macOS arm64 and manylinux x86\_64 | none |
| `onnxruntime==1.23.0` | cp310-cp313, macOS arm64 and manylinux x86\_64 | none |
| `onnx==1.19.0` | cp310-cp313 macOS universal2; cp310-cp314 manylinux | yes |
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

The `/data/` rule is anchored with a leading slash. An unanchored `data/`
matches a directory of that name at any depth, and this repository already
carries directories a rule like that could swallow without a word.

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

Two rules are enforced by `load_sources` rather than by review:

- **A source that is trained from or measured on carries a 64-character
  digest.** The one recorded absence, `unavailable`, is confined to `excluded`,
  because that is the only role nothing is measured on. Flipping such an entry
  to `train` or `eval` raises at load rather than shipping an unpinned corpus.
- **Nothing ProtectAI trained on may be an evaluation source.** Their model
  card for `deberta-v3-base-prompt-injection-v2` names two datasets, and this
  stage measures against that model. Scoring a detector on a corpus the
  reference model memorised publishes memorisation as recall.

URLs are pinned to a commit rather than to a branch. A branch under a recorded
hash either starts failing verification, which is at least loud, or gets its
hash updated to match, which changes the corpus under every number measured on
it.

The manifest reader is the standard library, not PyYAML, because
`tests/test_training_data.py` imports it and CI installs `.[dev]` and nothing
else. It implements the subset the manifest uses and raises on anything else.
`test_the_manifest_reader_agrees_with_the_yaml_library` compares it against
PyYAML wherever PyYAML is installed, which includes the training virtualenv;
it skips in the package's `.venv`, which has no PyYAML by design.
