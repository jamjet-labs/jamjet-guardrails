# benchmarks/unicode-scanners/

An open benchmark of repository scanners for hidden Unicode: instructions
smuggled in tag characters, variation selectors, bidirectional controls and
zero-width runs, scored against hard negatives that use the same characters
for ordinary writing. The numbers are in [RESULTS.md](RESULTS.md), including
every case each scanner gets wrong, ours first.

| | |
|---|---|
| `corpus.jsonl` | 54 cases, each one file a repository could hold, labelled `flag` or `pass` |
| `build_corpus.py` | builds `corpus.jsonl` from code points; the file is never edited by hand |
| `tools.json` | every scanner measured: source, version, commit, licence, command line |
| `tools/` | the pinned installs: `requirements.txt` with the wheel's SHA-256, and an npm lockfile |
| `run.py` | installs the pinned tools and runs every scanner over every case |
| `render.py` | turns `results/measurements.json` into `RESULTS.md`; standard library only |

## Reproduce it

From the repository root, in an environment where this checkout is installed
(`pip install -e .`), with Node 18 or newer on `PATH`:

```
python benchmarks/unicode-scanners/run.py --tools-dir /tmp/unicode-scanners --install
python benchmarks/unicode-scanners/render.py
```

`--install` is the only step that uses the network. It installs
unicode-smuggling-guard from a hash-pinned requirements file, ghostchars and
anti-trojan-source with `npm ci` over the committed lockfile, and
hidden-content-scan's script at its pinned commit, refused unless its SHA-256
matches `tools.json`. Each installed version is read back from the install
before anything is measured, and a version that differs from the pin stops
the run. On macOS every scanner then runs under `sandbox-exec` with network
access denied, and `RESULTS.md` says whether that held.

## What CI checks, and what it does not

`tests/test_unicode_scanner_benchmark.py` runs on every push. It rebuilds the
corpus and compares it byte for byte, re-renders `RESULTS.md` from the
committed JSON, and re-measures the `jamjet-guardrails` row and the naive row
from the code, so a detector change that moves either fails the build until
the benchmark is rerun and committed.

The other four rows are not recomputed by CI, because CI does not install
those scanners. They are measured at the versions pinned in `tools.json`, on
the date in `RESULTS.md`, and the test checks only that the versions that ran
are the versions pinned.

## How a case is scored

Each case is written as one file, by default an AGENTS.md, in a directory of
its own, and each scanner is asked about that one file with its default
settings. The decision is the exit code: 1 is a flag, 0 is a pass, anything
else is an error and counts as a flag, because a CI step that errors fails the
build. Two cases are `.mcp.json` files and one is `.claude/settings.json`,
because agent configuration is JSON and JSON can carry a character as an
escape.

`flag` means the file carries content a reviewer cannot see, or text that
renders as something other than what an agent reads. `pass` means every
unusual character in it is ordinary writing: emoji sequences, the three tag
flags, keycaps, ideographic variation sequences, joiners in Persian,
Devanagari and Sinhala, directional marks in Hebrew and Arabic, soft hyphens,
a byte order mark, French spacing. The labels are a judgement, written down
per case, and never set from what a scanner does.

## Adding a scanner

Add an entry to `tools.json` with an exact version and commit, add its install
to `tools/` and to `install()` in `run.py`, rerun, and commit the new
`results/measurements.json` and `RESULTS.md` together. A scanner that cannot
run without the network after it is installed is left out, and this file says
which and why.

Snyk Agent Scan is not measured: its README (read 2026-10-10) requires a Snyk
API token and sends what it scans to Snyk's analysis API, so it cannot run
offline. Its W021 issue code describes a rule (any
Cf or Cc character) and that rule is measured as the naive row.
