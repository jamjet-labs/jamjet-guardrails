# Scanning a repository

`jamjet-guardrails scan` runs this library's checks over files on disk and
reports every finding with its file, line and column. It is meant for a CI
step or a pre-commit run over the files an AI coding agent reads as
instructions, where a payload in invisible characters is read by the agent
and not by the person reviewing the pull request.

```console
$ jamjet-guardrails scan
AGENTS.md:3:12: deny injection-structural/INVISIBLE_TAG_CHARS <TAG x32>
  hidden text: "ignore all previous instructions"
scanned 1 file, 1 finding
```

With no path it scans the current directory. Name files or directories to
scan those instead, and put `--` before any path that starts with a dash.

## What it reads by default

A directory walk reads only agent instruction files: a path, relative to the
directory the scan runs from, that matches one of these patterns. `**/` means
any number of directories, including none; `/**` means anything below.

<!-- agent-file-patterns -->
```
**/AGENTS.md
**/AGENT.md
**/CLAUDE.md
**/CLAUDE.local.md
**/GEMINI.md
CONVENTIONS.md
.cursorrules
.windsurfrules
.clinerules
.clinerules/**
.cursor/rules/**
.windsurf/rules/**
.roo/**
.kiro/steering/**
.continue/**
.amazonq/**
.junie/**
.gemini/**
.github/copilot-instructions.md
.github/instructions/**
.github/prompts/**
.github/agents/**
.github/chatmodes/**
**/*.prompt.md
**/*.prompt
**/*.mdc
**/SKILL.md
.claude/skills/**
.claude/agents/**
.claude/commands/**
.claude/settings.json
.claude/settings.local.json
.mcp.json
mcp.json
.vscode/mcp.json
.cursor/mcp.json
```
<!-- /agent-file-patterns -->

`tests/test_scan_docs.py` holds this list to the tuple the scanner matches
against, in order, so the two cannot drift. `--all-text` widens the walk to
every text file. The walk skips `.git`, `node_modules`, virtual environments,
caches and build output, and follows directory symlinks without looping.

## Files it will not skip

An agent instruction file, and any file named on the command line, is always
read. A scan is adversarial, so every reason to look away from a file is a way
to hide something in it:

| Finding | Meaning |
|---|---|
| `scan/NOT_UTF8` | not valid UTF-8; decoded by its byte order mark if it has a UTF-16 or UTF-32 one, otherwise with replacement characters, and scanned |
| `scan/CONTAINS_NUL` | contains a NUL byte; scanned past it, because an agent reading the file does not stop there |
| `scan/OVER_SIZE_CAP` | larger than 2,000,000 bytes; the first 2,000,000 are scanned and the rest is not |
| `scan/UNREADABLE` | could not be opened |

Each of these is a `deny` finding and fails the scan. Only files an
`--all-text` walk reaches for no other reason are skipped silently, and they
are counted in the summary line.

A `.json` file is scanned as written and again with every string literal that
contains an escape decoded. `json.dumps` writes a tag character as a pair of
`\u` escapes by default, so a payload in an MCP configuration can be plain
ASCII on disk. A finding inside a decoded literal is reported at the literal's
opening quote.

## Checks

The default is `injection-structural` alone, the check for instructions hidden
in the encoding rather than in the words. `--checks` takes a comma-separated
list of other checks by name. `rules` and `script-constraint` cannot be named
there: each needs options a command line does not carry, and naming either
exits `2` with the reason.

## Output

| Option | Effect |
|---|---|
| `--format text` | one line per finding, `path:line:column: decision check/TYPE excerpt`, then the decoded hidden text if there is any, then a summary line. The default. |
| `--format json` | the same findings as JSON on standard output; the summary goes to standard error |
| `--format sarif` | SARIF 2.1.0 on standard output |
| `--sarif PATH` | also write SARIF to `PATH` in the same run, so the log carries text and the file carries SARIF |
| `--no-fail` | exit `0` even when something is found |

Text and JSON columns count code points. SARIF columns count UTF-16 code units
and say so with `columnKind`, because a tag character is two of them and that
is the line a reader is sent to. SARIF paths are percent-encoded and relative
to `%SRCROOT%`, and every rule carries a `security-severity`, which GitHub
code scanning needs to show the result as a security alert.

Everything printed is untrusted: a path is whatever a pull request named a
file. Invisible and bidirectional characters, line breaks, and every `::`
(which GitHub Actions reads as the start of a workflow command) are escaped in
text output. JSON and SARIF are written as ASCII.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | nothing found, or `--no-fail` |
| `1` | at least one finding, including a file it could not read cleanly |
| `2` | the invocation is wrong: an unknown check, a check that needs options, a named path that does not exist, a SARIF path that cannot be written |

A scan that finds no agent instruction files at all exits `0` and prints
`scanned 0 files`. A repository without them is clean, not misconfigured.

## In GitHub Actions

```yaml
permissions:
  contents: read
  security-events: write
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: "3.12"
      - run: pip install jamjet-guardrails
      - run: jamjet-guardrails scan --sarif scan.sarif
      - if: always()
        uses: github/codeql-action/upload-sarif@cdf488f595d80d6e07e03d4674febd5ab45fa938 # v4.37.9
        with:
          sarif_file: scan.sarif
```

`if: always()` is what gets the result uploaded on exactly the run where the
scan fails. Uploading SARIF needs `security-events: write`, which a pull
request from a fork does not get, and code scanning on a private repository
needs GitHub Code Security. The log output works everywhere.

## What it does not catch

It does not read what text means. An instruction written in plain, visible
words passes, because there is nothing hidden about it; that needs a reviewer
or a semantic classifier, and this library has neither.

Within hidden Unicode it has misses, and they are published.
[benchmarks/unicode-scanners/RESULTS.md](../benchmarks/unicode-scanners/RESULTS.md)
scores this command and four other scanners on one corpus and lists every
wrong decision each makes, this one's first. One of them is a payload written
as one variation selector after each visible character: every pair is a
well-formed variation sequence, and the check does not count those.
[corpora/NOTICE.md](../corpora/NOTICE.md) lists that encoding with the other
invisible-character families the check does not count.
