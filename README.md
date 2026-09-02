# jamjet-guardrails

Content guardrails for LLM applications. Every decision carries provenance:
which check made it, over exactly what text, and why.

No dependencies. No network calls. No model downloads. Python 3.10 and above.

## The problem

An LLM application that handles real data has three ways to leak it, and all
three are quiet:

- The model repeats back a customer email or card number it was given.
- A key pasted into a prompt is echoed into a log, a trace, or a reply.
- A check is configured, silently fails to load, and the application runs
  unguarded while its configuration says otherwise.

This library addresses the first two by inspecting content and the third by
refusing to start.

## Install

```
pip install jamjet-guardrails
```

## Quickstart

```python
from jamjet_guardrails import Context, build_chain

chain = build_chain(["pii", "secrets"])
result = chain.run(
    "mail alice@example.com and use sk-abcdefghijklmnopqrstuvwxyz012345",
    Context(direction="output", origin="model"),
)

print(result.decision)
print(result.content)
for verdict in result.verdicts:
    for finding in verdict.findings:
        print(finding.type, finding.span, verdict.provenance.detector)
```

```text
redact
mail [REDACTED:EMAIL] and use [REDACTED:OPENAI_KEY]
EMAIL (5, 22) pii
OPENAI_KEY (31, 66) secrets
```

## What you get back

Every check returns a `Verdict` carrying its decision, its findings, and a
provenance record naming the detector and its version. Each verdict also
carries `saw`, the SHA-256 of the exact string that check inspected, so a
decision can be tied afterwards to the text it was made about.

Decisions combine restrictively: `deny` > `redact` > `allow`. No code path can
weaken a decision another check has already made.

Every check in a chain inspects the content you passed in. No check sees a
string another check has already rewritten, so every span above indexes into
your input and every verdict hashes the same text. Redactions from all the
checks are merged and applied in one pass, and a region two checks both claim
comes back as one placeholder naming both.

That rule is a leak fix, not a tidiness one. Rewriting one check at a time let a
personal-data redaction cut a credential in half, so the next check matched only
the stump and the rest of the credential survived into content the chain
reported as redacted.

On a `deny` the returned content is the audit record, not something to send.
Branch on the decision first.

## The checks

| Name | Kind | Runs on | Types |
|---|---|---|---|
| `injection-structural` | constraint | input, output | `BIDI_OVERRIDE`, `INVISIBLE_TAG_CHARS`, `ZERO_WIDTH_SMUGGLING` |
| `pii` | constraint | input, output | `CREDIT_CARD`, `EMAIL`, `PHONE_NUMBER`, `US_SSN` |
| `rules` | constraint | input, output | `INTERNAL_HOST`, `LENGTH_LIMIT`, `PROJECT_CODENAME`, `TICKET_ID` |
| `secrets` | constraint | input, output | `ANTHROPIC_KEY`, `AWS_ACCESS_KEY`, `GITHUB_TOKEN`, `JWT`, `OPENAI_KEY`, `PRIVATE_KEY`, `SLACK_TOKEN` |

A type name says what a check labels a match as, not that every value of that
kind matches. The secrets patterns are anchored on a prefix, and two common
shapes are named here rather than left for you to find: `github_pat_`
fine-grained GitHub tokens and `xapp-` Slack app-level tokens are not among
the prefixes matched, so both pass through untouched. Per-type precision and
recall are in [BENCHMARKS.md](BENCHMARKS.md).

**`rules` is the one check whose types you choose.** It takes your own regular
expressions, banned substrings and size limits, so the types in its row are the
ones the published measurement was taken with, not a fixed set. That row
measures the engine: whether a span is right, whether two rules claiming one
stretch of text collapse into one placeholder, whether a limit fires one
character past its bound. It is not a measurement of any rule you write, and
your rules carry their own rates. The exact configuration behind the row is in
[docs/conformance.md](docs/conformance.md).

Size limits are characters, bytes and lines. There is no token limit, because
counting tokens needs a tokenizer this library does not carry and will not
guess at.

**This row is not comparable to the other rows in the table below.** The other
checks are heuristics over open-ended text, and their precision and recall
describe how often the heuristic is right on text nobody controlled. `rules`
here is a deterministic engine running against a fixed set of rules we wrote
for the published measurement, so a high score means the engine computes
spans, merges overlapping regions and applies limits correctly against that
configuration. It is not a claim that any rule is well chosen, and a perfect
score on this row carries none of the weight a perfect score on `pii` or
`secrets` would.

The published measurement also only exercises one of the three limit kinds.
The fixture sets a character limit and no byte or line limit, so the row never
reaches the byte-boundary or line-boundary code paths at all. A limit the row
cannot reach is a limit the row does not cover, not a limit proven correct.

## Measured, not asserted

Every check ships with a labelled corpus and published precision and recall.
CI refuses a change that lowers either beyond a small tolerance, or that gets
one more decision wrong than the committed baseline. The misses are published
beside the scores.

| Check | Corpus | Source | Version | Cases | Precision | Recall | F1 | TP | FP | FN | Wrong decisions |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| injection-structural | injection-structural/in-repo | in-repo | `b704703f431d` | 154 | 0.972 | 0.873 | 0.920 | 103 | 3 | 15 | 8 |
| pii | pii/in-repo | in-repo | `06fb3b601aba` | 81 | 0.631 | 0.872 | 0.732 | 41 | 24 | 6 | 24 |
| pii | pii/third-party | nvidia/Nemotron-PII@b70ffaf | `c25ef538d677` | 300 | 0.960 | 0.997 | 0.978 | 340 | 14 | 1 | 6 |
| rules | rules/in-repo | in-repo | `f1b809114b13` | 40 | 1.000 | 1.000 | 1.000 | 28 | 0 | 0 | 0 |
| secrets | secrets/in-repo | in-repo | `e9e0ed70dc37` | 39 | 0.957 | 0.880 | 0.917 | 22 | 1 | 3 | 4 |

**A *balanced* override still reorders, and `injection-structural` allows it.**
The signal is imbalance, not presence. `transfer <RLO>001<PDF> USD` renders as
`transfer 100 USD`, measured with GNU FriBidi 1.0.16, and this check reports
nothing, so Trojan Source written with a closed pair passes it. That is named
here rather than left for you to find. The reason it is allowed is that flagging
balanced controls would deny ordinary Arabic and Hebrew, which are written with
these controls; what imbalance buys is a divergence the author cannot bound,
because an unclosed control runs to the end of the paragraph.

**It also does not read every invisible character, and it publishes no minimum
cost for getting past it.** Variation selectors and the directional marks are
not counted, because counting them denies five keycaps, a Japanese document
naming five people whose names take variant glyphs, and a bilingual invoice; the
control families and several others are not counted either. So a payload encoded
in any of them goes through: 256 variation selectors is a byte per character,
and measured, 32 of them carry a 32-character instruction with nothing on the
page and nothing reported. That figure is the cost of that one encoding and not
a bound. No minimum is published, because a minimum is a claim about every
possible encoding and a measurement only ever exhibits one.
[corpora/NOTICE.md](corpora/NOTICE.md) lists the uncounted families with one
measured encoder each, which is the claim this check can actually support.

See [BENCHMARKS.md](BENCHMARKS.md) for the per-type scores and the worst misses
behind these numbers, and [corpora/NOTICE.md](corpora/NOTICE.md) for what each
corpus is and where it came from.

**The in-repo PII corpus is a stress set, not a sample of ordinary traffic.** It
is written to hold the shapes this detector is worst at, so its precision is
lower than you would see on real text and is meant to be.
[corpora/NOTICE.md](corpora/NOTICE.md) breaks that figure down, names the one
shape it over-represents on purpose, and scores the same corpus without it. The
third-party corpus is the one to read for ordinary text: 300 rows we did not
write, named in the Source column beside its own numbers.

**Every corpus here labels a case with what should happen, never with what the
detector does.** A known false positive is labelled `allow` and costs precision;
a known false negative is labelled `deny` and costs recall. That is why these
numbers are lower than the checks behave on ordinary text, and it is the only
way two rows in one table can be compared. Fifteen `injection-structural`
cases carry such a label and eight of them fail on purpose: two deny text
somebody wrote on purpose, and six allow a payload that really is in there. All
fifteen are named by case id in [corpora/NOTICE.md](corpora/NOTICE.md).

**Four scattered invisible characters go through, and that is a deliberate
trade.** This check reports an unexplained zero-width character when two are
ADJACENT or when five appear anywhere in the input. Four, no two of them
touching, is allowed, because four is what ordinary text reaches: Thai marked up
for line breaking, Persian written with ASCII digits, a 2,503-character page
with four incidental zero-width characters, mathematical markup extracted to
plain text, and four UTF-8 files concatenated with each keeping its own
byte-order mark are all four occurrences and all pass. The corpus carries three
payloads of exactly four characters that this lets through, labelled `deny` so
they cost recall. The residual is bounded rather than a channel: a fifth
character denies whatever else the message contains.

Numbers measured on a corpus we wrote are reported separately from numbers
measured on a corpus we did not, and the two are never merged. There is no
third-party corpus for `injection-structural`, `rules` or `secrets`. No
compatibly licensed one was found for any of them, so all three are measured on
our own corpora only and are self-graded.

The third-party PII corpus is derived from
[nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII),
used under CC-BY-4.0. Changes were made, and they are listed in
[corpora/NOTICE.md](corpora/NOTICE.md).

## How it fails

Two failure modes, chosen deliberately.

- **A check that raises becomes `deny`, never `allow`.** The chain records the
  error on that check's verdict and carries on. A crashing detector blocks
  content rather than passing it through unexamined.
- **A check that would be configured and silent raises `GuardrailUnavailableError`.
  This is raised at construction when a guardrail is not installed or cannot
  be built, and also when ``PatternGuardrail.check`` is called with a direction
  it does not declare.** Configuration that silently means "this check is not
  running" is the failure this library exists to prevent, so it is refused
  before any content is processed. An empty list of checks is refused for the
  same reason.

Treat any exception out of `run` as a deny. The cases that raise abandon the
run, so there is no result and no audit record, which is acceptable only
because nothing was allowed through.

## What this is not

It does not classify intent, score toxicity, or call a model. The checks here
are constraints: patterns with published false-positive and false-negative
rates. That is why the numbers exist and why they are worth reading.

It is a library, not a service. No configuration file, no daemon, no account.

## Porting it

[docs/conformance.md](docs/conformance.md) specifies the verdict fields, the
combination order, the single-pass rewriting rule, the `saw` hash and the corpus
schema, and states what is deliberately unspecified. An implementation in
another language conforms if it produces the same verdicts on the same corpora,
whatever machinery it uses to get there.

## Licence

The code is Apache-2.0. See [LICENSE](LICENSE).

The published distribution declares `Apache-2.0 AND CC-BY-4.0`, because the
source distribution also carries `corpora/pii/third-party.jsonl`, derived from
[nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) under
CC-BY-4.0. Attribution and the list of changes are in
[corpora/NOTICE.md](corpora/NOTICE.md). The installed wheel contains code only.
