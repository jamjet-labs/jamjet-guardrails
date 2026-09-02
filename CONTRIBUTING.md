# Contributing

Thanks for looking at this. Before you write code, read this file. Most of it
is about one thing that makes this repository different from a library that
merely has tests: **every number this project publishes is measured in CI
against a corpus committed beside it, and a change that moves a number fails
the build until a human moves the baseline in the same pull request.**

## Running everything

Python 3.10 or newer. The package itself has no dependencies; the tools do.

    python3 -m venv .venv
    ./.venv/bin/pip install -e ".[dev]"

The five checks CI runs, in the order it runs them:

    ./.venv/bin/ruff check .
    ./.venv/bin/ruff format --check .
    ./.venv/bin/mypy
    ./.venv/bin/pytest -q
    ./.venv/bin/jamjet-guardrails \
      --corpora-dir corpora \
      --json benchmarks.json \
      --md BENCHMARKS.md \
      --gate corpora/baselines.json

Run `mypy` with no path argument. A path on the command line overrides
`[tool.mypy] files` and silently drops `tests/` and `training/` from the check.

The last command rewrites `benchmarks.json` and `BENCHMARKS.md` from a fresh
measurement. CI then runs `git diff --exit-code` on both, so if your change
moved a score, the working tree now differs from what is committed and the
build is red until you commit the new artifacts. That is the mechanism, not a
formality: the published numbers cannot drift away from the code without
somebody noticing.

The suite runs in about 40 seconds and needs no network and no model.

## Moving a number

`corpora/baselines.json` records, per corpus, the precision, the recall and
the count of wrong decisions. The gate refuses a run that scores below a
baseline by more than `--epsilon`, or that gets one more decision wrong than
the baseline records. The epsilon is for floating-point noise and is bounded
at 0.05 by the gate itself, which refuses a wider one, a negative one and a
NaN.

So a change that improves a detector is easy and a change that costs one is
deliberate. If your change lowers a score on purpose, say so in the pull
request, then:

    ./.venv/bin/jamjet-guardrails \
      --corpora-dir corpora \
      --json benchmarks.json \
      --md BENCHMARKS.md \
      --write-baselines corpora/baselines.json

`--write-baselines` is refused together with `--gate`, so you cannot record a
baseline and pass a gate against it in one run. It rounds down to three
decimals, so the epsilon absorbs noise rather than masking a real drop.

Do not edit `benchmarks.json`, `BENCHMARKS.md`, `benchmarks/RESULTS.md` or
`corpora/baselines.json` by hand. All four are generated, and a test re-renders
each and fails on any difference.

## Adding or changing a corpus case

The schema is specified in [docs/conformance.md](docs/conformance.md), which is
the porting spec and is the authority. Three rules are easy to get wrong:

- **A case is labelled with what SHOULD happen, never with what the detector
  does.** A known false positive is labelled `allow` and costs precision. A
  known false negative is labelled `deny` or `redact` and costs recall. Labelling
  a case with current behaviour makes the corpus a description of the code and
  the score meaningless.
- **One file, one source.** The loader refuses a file that mixes `source`
  values, which is what keeps numbers we measured on our own corpus from ever
  being merged with numbers measured on somebody else's.
- **Provenance is a condition, not a courtesy.** Any third party whose data or
  model a published figure uses needs an entry in
  [corpora/NOTICE.md](corpora/NOTICE.md). That file states the rule about
  itself: a published figure is a use. `tests/test_corpora.py` and
  `tests/test_training_data.py` enforce parts of it.

No real credential and no real person's data goes into any corpus or any
docstring. Credential-shaped strings carry `EXAMPLEONLY` or `notarealtoken`
inside their own bodies, and `tests/test_packaging.py` runs the shipped secrets
detector over every tracked file to hold that repository-wide.

## Mutation-check your tests

**A test that has never been watched failing is a test nobody has evidence
for.** Before you open a pull request, for each test you added or changed:
break the property it claims, run it, watch it FAIL, then revert and watch it
pass. Say in the pull request what you broke and what the failure said.

This is not a style preference here. Guards in this repository have passed for
reasons other than the one their name claimed, more than once, including a
guard that reproduced the very string it existed to keep out and a staleness
check that searched its own exemption list and so could never go stale.

`scripts/mutate.py` runs the loop mechanically over a list of mutations and
reports any that stayed green:

    ./.venv/bin/python scripts/mutate.py mutations.json

Use it rather than editing by hand where you can. It clears the bytecode cache
between steps, and that is load-bearing: a same-length edit written inside one
mtime tick makes Python import the stale `.pyc`, the test passes against code
that is not on disk, and the mutation is recorded as green. That has happened
here.

## Prose is checked too

Numbers in documentation are claims and are treated as such. `tests/` holds
guards that template published sentences from the records behind them, resolve
every repository path a document cites, and check that a case id quoted beside
"labelled `allow`" really carries that label. If you write a number into a
document, make it derivable from something in the repository and, where you
can, add the guard that derives it. Where a figure genuinely cannot be
recomputed in CI, say so beside it and publish the counts so a reader can check
the ratio.

Two mechanical rules: no em dashes, and no unfalsifiable or forward-looking
copy in `README.md`. Both are tested.

## Pull requests

- One change per pull request.
- Keep the suite green. A red leg on one Python version is a real failure; the
  matrix is 3.10 through 3.14 and every leg regenerates and diffs the published
  artifacts.
- Explain what moved and why in the description. If a published number changed,
  the description is where the reader learns it was on purpose.
- New behaviour needs a test, and the test needs a mutation check.

## Scope

This library ships deterministic constraints: patterns and structural rules
with published false-positive and false-negative rates. It does not classify
intent, score toxicity, or call a model, and a contribution that adds a
dependency on one will not be merged. `dependencies = []` is checked against
the built metadata, not against `pyproject.toml`.

## Licence

Contributions are accepted under Apache-2.0, the licence of the code in this
repository. See [LICENSE](LICENSE).
