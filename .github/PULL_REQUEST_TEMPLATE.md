## What this changes

<!-- One or two sentences. What moved and why. -->

## Published numbers

<!-- Delete the line that does not apply. -->

- [ ] No published precision, recall or wrong-decision count changed.
- [ ] A published number changed. Old and new values, and why the change is
      intended:

<!--
If a number moved, `corpora/baselines.json`, `benchmarks.json` and
`BENCHMARKS.md` must be regenerated and committed in this pull request. CI
regenerates them and diffs, so a stale artifact is a red build.
-->

## Mutation checks

<!--
For each test added or changed: what you broke, and what the failure said. A
test nobody has watched fail is a test nobody has evidence for. Delete this
section only if the change adds no test, and say why it needs none.
-->

| Test | What was mutated | Failure observed |
|---|---|---|
|  |  |  |

## Checks

- [ ] `ruff check .` and `ruff format --check .`
- [ ] `mypy` (no path argument)
- [ ] `pytest -q`
- [ ] `jamjet-guardrails --corpora-dir corpora --json benchmarks.json --md BENCHMARKS.md --gate corpora/baselines.json`
- [ ] Corpus cases, if any, are labelled with what SHOULD happen and not with
      what the detector currently does
- [ ] Any third party whose data or model a published figure uses is named in
      `corpora/NOTICE.md`
