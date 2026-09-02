# Corpus provenance and attribution

Every published precision and recall figure the package reports about itself is
measured on one of the files under `corpora/`. This file records where each one
came from and under what licence, because for one of them that is a condition of
use and not a courtesy.

That sentence used to say "every published precision and recall figure in this
repository", and `benchmarks/RESULTS.md` made it false: it publishes precision
and recall measured on a dataset that is Lakera's, against models that are
ProtectAI's, and neither was named here. The rule this file states about itself
is that a published figure is a use, so those two are attributed at the end of
this document under [Third-party material behind published
measurements](#third-party-material-behind-published-measurements).

Each corpus is one file, one source: the loader refuses a file that mixes them,
so in-repo and third-party numbers can never be merged into one score.

| Corpus | `source` field | Licence |
|---|---|---|
| `corpora/injection-structural/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/pii/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/secrets/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/pii/third-party.jsonl` | `nvidia/Nemotron-PII@b70ffaf` | `CC-BY-4.0` |

## First-party corpora

The three `in-repo` files were written for this repository and are covered by
its own Apache-2.0 licence. Every value in them is invented or is a published test
value: `AKIAIOSFODNN7EXAMPLE` is AWS's own documentation example, `4111 1111
1111 1111` is the universally published test PAN, and `example.com` and
`example.org` are reserved for documentation by RFC 2606. The two
internationalised addresses use that same `example` label under a Cyrillic and a
Devanagari TLD.

No real credential and no real person's data is in any of the three. The GitHub,
OpenAI, Anthropic and Slack tokens carry `EXAMPLEONLY` or `notarealtoken` inside
their own bodies; the JWTs are a standard HS256 header with an invented payload
and a signature of random bytes, so they verify against nothing; and the PEM
bodies are base64 of random bytes rather than DER, so no tool can load one as a
key.

That sentence used to stop at these three files, and two credential-shaped
strings outside them were covered by nothing: a canonical Slack bot token with a
random 24-character secret, and a 36-character GitHub token body, each written
into a docstring in `src/` to show a defect that string had caused. Both shipped
in every wheel. Neither carried a marker, so nothing about either one told a
scanner or a reader that it was not live. Both now carry the same markers the
corpus values do, and the rule is repository-wide rather than file-scoped: the
detector this package ships is run over every tracked file, and where it reports
a GitHub, OpenAI, Anthropic or Slack token the body has to say what it is.
`test_no_credential_shaped_literal_in_the_repository_reads_as_a_live_one` in
`tests/test_packaging.py` is that check. Six older bodies carry no marker and are
listed there one by one instead: a sequential alphabet, a counted digit run,
Amazon's published example key and the standard HS256 header. They are named
rather than admitted by a rule about what looks synthetic, because the first
body a rule like that lets through is the one worth catching.

### `corpora/injection-structural/in-repo.jsonl`

Every value in this file is invented too, and none of it is a credential or a
person: the payloads say "ignore all previous instructions" and "do evil",
written in Unicode tag characters or as a zero-width bitstream, and the one
domain in them is `evil.example`, reserved for documentation by RFC 2606.

Its negatives are not invented, and could not be. The three subdivision flags,
the emoji ZWJ sequences, the Brahmic conjuncts and the Persian, Urdu, Arabic,
Hindi, Malayalam and Thai text are the sequences Unicode itself publishes or
the orthography of a living script, because the point of a negative here is
that it is text somebody really writes. They are spelled out of code points
rather than pasted, and the whole file is written as `\uXXXX` escapes, so what
a reviewer reads in the diff is what the loader decodes.

**There is no third-party corpus for this check.** No compatibly-licensed
labelled corpus of structural injection was found, so these numbers are
measured on our own file only and are self-graded in the same way the secrets
numbers are.

Numbers measured on them are **self-graded**: we wrote both the detector and the
labels. That is why the third-party corpus below exists.

## How to read the numbers these corpora produce

**`corpora/pii/in-repo.jsonl` is a stress set, not a sample of real text, and
its precision figure should be read that way.** It carries all twenty-five false
positives `tests/test_pii.py` records, and sixteen of those are one shape: a
dotted tail that reads as a TLD (`img@sha256.abcdef`, `build@node.js18`). That
shape was measured at zero occurrences across 400 MB of source, logs, lockfiles
and docs, so it is close to absent from ordinary text and is heavily
over-represented here on purpose. The same corpus without those sixteen cases
scores 0.837 precision against the 0.631 it publishes. Neither figure is wrong;
they answer different questions, and only the third-party corpus answers "what
happens on realistic documents".

Carrying all of them is a checked fact rather than a promise:
`tests/test_corpora.py::test_every_recorded_false_positive_is_in_the_corpus`
reads both lists out of `tests/test_pii.py`, so deleting a case to raise the
number fails, and so does recording a new false positive without adding one.

**Some of those false positives are the right redaction under the wrong type.**
An IMEI carries a Luhn check digit and begins 49, inside Visa's range, so
`imei 490154203237518` redacts as a `CREDIT_CARD`; a `tax_id` written
`123-45-6789` redacts as a `US_SSN`. Redacting is the right direction, because
both are personal data. The type in the audit record is what is wrong, so both
are labelled `allow` here and score as false positives. The opposite call was
made for `fax_number`, which really is a telephone number, and the line between
the two is whether the value is the same KIND of thing as the type claiming it.

**One guard behind these numbers expires on a date.** The bare-card scan
requires a leading digit of 2 to 6, the Major Industry Identifier range payment
cards are issued under, and most of what that buys is the exclusion of
epoch-millisecond timestamps, which begin with a 1 today and appear in nearly
every machine-written log line. Epoch-ms first carries a leading 2 on
**2033-05-18T03:33:20Z**, and epoch-microseconds crosses on the same date. After
that boundary timestamps sit back inside the range: measured at 0 of 5,000
values redacting before it and 500 of 5,000 after. A published precision figure
measured on these corpora is a statement about the detector as it behaves before
that date.

**`corpora/injection-structural/in-repo.jsonl` labels its own failures as
failures, and its published numbers describe a check that trades recall for
precision at a bound that was raised deliberately.** 0.971 precision, 0.870
recall, 8 wrong decisions over 146 cases. Eight cases fail, every one of them on
purpose, and there are no other failures. Fifteen cases are listed below: the
eight that fail, plus the three-case balanced-override set and the four
imbalanced-control cases -- a stray PDF, a stray PDI, and the two halves of a
document split across a balanced pair, of which `inj-0139` keeps the unclosed
INITIATOR and only `inj-0140` is a stray terminator -- which pass and are here
because a reader can reasonably expect the opposite of each.

**The list was twenty-four, and what changed it was one constant.** `_MIN_TOTAL`
counts unexplained zero-width characters anywhere in the input and was 4; it is
now 5. Twelve cases of ordinary text that used to be reported now pass: the Thai
line-break hints; three Persian and Urdu numeral compounds written with ASCII
digits (`inj-0092`, `inj-0093`, `inj-0095`) and one Persian plural suffix on
Latin acronyms (`inj-0094`), which is a different shape and was named as one
before this list was rewritten; a retrieved page carrying four incidental
U+200B; MathML extracted to plain text; and five more -- Korean prose about
jamo, a Khmer dictionary entry, U+034F blocking a collation contraction, U+034F
fixing point order in Biblical Hebrew, and four UTF-8 files concatenated with
each keeping its own BOM. **They stay in the corpus.** They are the evidence
that the check allows that text, and deleting them would leave the raise
unjustified. They are no longer disclosed because nothing about them is
surprising any more.

Three cases went the other way, and they are the price. Every one is four
zero-width characters with no two adjacent, which is one under the volume bound
and carries no adjacent pair for the run bound to see.

Two things did NOT change with the bound, and both are worth reading as a pair.
`_MIN_RUN` is still 2: two adjacent counted characters are what a
bit-per-character encoder emits and what ordinary prose does not, so it is a
signal about shape rather than volume, and the sweep that offered 0.980
precision for raising it too was refused because 0.009 precision is not worth
it. And `inj-0129` is still a false positive for exactly that reason -- musical
notation where an END BEAM abuts the next BEGIN BEAM is four controls containing
an adjacent pair, so the run bound reports it and the volume bound never did.

- **Two deny text somebody wrote on purpose, and are labelled `allow`.** They
  score as the corpus's three false positives. `inj-0091` wraps a two-line value
  in `FSI ... PDI`, the idiom Unicode recommends and the one `<bdi>` implements;
  a control's scope ends at its paragraph, so the PDI on the second line closes
  nothing and both controls are reported, while the text renders byte-identically
  to the same string with the wrapper deleted -- measured with GNU FriBidi
  1.0.16. `inj-0129` is musical notation extracted to plain text, denied on the
  RUN bound as described above.
- **Six allow a payload that is really there, and are labelled `deny`.** They
  score as the corpus's fifteen false negatives. Three are the cost of raising
  `_MIN_TOTAL`: `inj-0051` is a ZWNJ between emoji four times over, `inj-0052`
  puts a ZWJ with one pictographic neighbour four times over, and `inj-0053`
  begins with a ZWNJ and carries three ZWSPs through Arabic text. Each is four
  characters carrying four bits, each decodes back out, and none of them can be
  extended: a fifth character denies whatever else the message contains, so this
  is a bounded residual rather than a channel.

  The other three pre-date the raise. `inj-0097`, `inj-0098` and `inj-0099` each
  carry the string `exfiltrate` past the check, decoded back out to check that:
  presence-and-absence spacing of a joiner behind a Devanagari cover, the same
  encoding between variation selectors at 119 characters with nothing on the
  page at all, and a bitstream deperiodised with one spare cover character every
  three bits.
- **Three are labelled `allow` and pass, and they are one construct rather
  than three cases.** `inj-0030`, `inj-0038` and `inj-0096` are each a BALANCED
  override whose scope contains characters whose order it reverses. Measured
  with GNU FriBidi 1.0.16: `the label reads <RLO>9876<PDF> on the box` renders
  `the label reads 6789 on the box`, `<RLO>abc def<PDF>` renders `fed cba`, and
  `transfer <RLO>001<PDF> USD` renders `transfer 100 USD`. Nothing distinguishes
  them, so **they move as a set**: relabelling one means relabelling all three.

  They are labelled `allow` because this is the BOUNDARY of the signal rather
  than a miss inside it. The rule is imbalance, not presence, and denying a
  balanced pair would deny ordinary Arabic and Hebrew. That last clause used to
  say "which use these controls for exactly this" and nothing in the file
  supported it: every right-to-left negative here used an embedding or an
  isolate, and every balanced OVERRIDE was Latin or digits. `inj-0141` and
  `inj-0142` are the negatives that make it true, a balanced override around
  Hebrew and around Arabic, both of which allow and both of which render
  unchanged because forcing right-to-left onto text that is already
  right-to-left reorders nothing. That is the same category as the `secrets` corpus's
  `github_pat_` and `xapp-` cases, which are tested as `allow` because they are
  shapes outside the pattern table rather than shapes it fails on -- and it is a
  different category from `inj-0097`, `inj-0098` and `inj-0099` above, which are
  failures inside the boundary and are labelled `deny` accordingly.

  The reader-facing consequence is that Trojan Source written with a closed pair
  passes this check, and because a reader can reasonably expect otherwise it is
  stated in `README.md` beside the published numbers rather than only here.
  `inj-0029` is the fourth balanced override in the file and is NOT part of this
  set: it is `the code is <LRO>A-1234<PDF> exactly`, which renders unchanged,
  because forcing left-to-right onto text that is already left-to-right reorders
  nothing.

- **Two are labelled `deny` and pass, on a weaker ground than the rest of the
  bidi signal.** `inj-0019` and `inj-0020` are a stray PDF and a stray PDI. The
  signal's stated rationale is that imbalance makes the rendered order diverge,
  and that rationale does not reach these: measured with GNU FriBidi 1.0.16,
  `harmless<PDF> text` and `harmless<PDI> text` render byte-identically to
  `harmless text`. **A stray terminator reorders nothing at all.** They are
  denied as a malformed control sequence -- a terminator that closes nothing is
  a document that was cut, or a probe -- which is defensible and is a different
  claim from the one the signal makes about initiators. The rationale in
  `_bidi_spans` now states both grounds separately.

  The realistic population is not an attacker. A pipeline that splits a document
  across a balanced `LRE ... PDF` puts an unclosed initiator in one chunk and a
  stray terminator in the next, which is exactly these two shapes: `inj-0139`
  and `inj-0140` are those chunks, and they are labelled `deny` because they ARE
  `inj-0019` and `inj-0020`. A corpus cannot deny one and allow the other.

**The convention, stated so that nobody improves the number by flipping a
label.** A case is labelled with what SHOULD happen, never with what the
detector does. A known false positive is therefore labelled `allow` and costs
precision; a known false negative is labelled `deny` and costs recall. The
alternative -- labelling each at the detector's own behaviour and explaining it
in prose -- was tried first here and produced 1.000 on both ratios with no wrong
decisions, sitting in the same table as the PII corpus's 0.631, which does it
the honest way. Two numbers that differ only in how their authors chose to score
their own mistakes cannot be read side by side.
`tests/test_corpora.py::test_a_disclosed_injection_shape_is_in_the_corpus_and_in_the_notice`
holds all fifteen ids against this section, in both directions.

**The corpus moves when the detector does, and that was measured rather than
assumed.** One hundred and twenty-two copies of `injection_structural.py` were
made, each with a single constant, range-table row or guard condition broken on
its own, and this corpus was scored against every one. **Seventy-two of the 122
break at least one case beyond the eight that already fail.** The other fifty
are invisible to the corpus and are caught by the test suite instead, which is
the honest shape of the answer: a corpus is one of two gates and not the only
one.

Two mutants survive the whole suite, and neither is a gap. Loosening `_chains`
from "exactly `step` apart" to "at most" cannot change any result, for the
reason its docstring argues and a sweep confirms; and narrowing the walk in
`_mark_base` to `Mn`, `Me` and `Cf` leaves every verdict where it is, which is
what the comment beside it already claims. One further mutant does not
terminate: the two range tests in `_tag_spans` are one constant used twice, and
making them disagree leaves the scanner unable to advance past a CANCEL TAG.

No superlative is offered for which mutant does the most damage, and the reason
is instructive. Deleting the virama branch used to turn thirteen cases of
ordinary Brahmic and Malayalam text into denials; measured after `_MIN_TOTAL`
was raised to 5 it moves **no case at all**, because the denials it caused were
four-occurrence ones. That rule is now held by the test suite alone. Dropping
the soft-hyphen exception, at the other end, still moves exactly one case,
`inj-0108`. A ranking of mutants is a property of the corpus and the bounds
together, and it does not survive either of them changing.

Cases that exist only because a rule survived a sweep with nothing to show for
it: `inj-0115` for the CANCEL TAG condition, `inj-0116` for the periodicity
bound from underneath, `inj-0117` for the virama's own script, and `inj-0118`
and `inj-0119` for WORD JOINER and the BOM, which until then appeared only in
samples that allow either way.

**Raising `_MIN_TOTAL` disarmed three tests that nothing else was holding, and
that is a hazard worth naming.** Each of the three pinned its rule with an input
carrying exactly four unexplained characters, which stopped denying the moment
the bound became five: the ASCII-digit Persian and Urdu samples, which alone
pinned decimal digits as excusing neighbours, and the Kaithi cluster, which
alone pinned format characters as transparent in the walk to a virama's base.
All three inputs carry five occurrences now. A volume bound that moves silently
disarms every test that reached it exactly.

## Third-party corpus

Portions of this evaluation corpus are derived from **Nemotron-PII** by Amy
Steier, Andre Manoel, Alexa Haushalter and Maarten Van Segbroeck (NVIDIA
Corporation), licensed under CC BY 4.0.

- Dataset: <https://huggingface.co/datasets/nvidia/Nemotron-PII>
- Revision: `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed`
- File: `data/test-00000-of-00001.parquet`, sha256
  `1a4b0512ecb5370f0992d29d0f9c07351e6de13f0d7ea33bb18cecb984780247`
- SPDX identifier: `CC-BY-4.0`
- Licence text: <https://creativecommons.org/licenses/by/4.0/>

**Changes were made.** Rows were filtered to `locale == "us"`, 300 of them were
sampled deterministically by the SHA-256 of each row's `uid`, the four labels
this library detects were renamed to its own type names, `fax_number` was mapped
onto `PHONE_NUMBER`, and every other label was dropped. `direction` is ours and
not the dataset's: it records a document format, and an unstructured document is
read here as model output and a structured one as an input. The conversion is
`scripts/sample_nemotron.py`, which reproduces the committed file from the
revision above.

CC BY 4.0 asks for attribution wherever the material is used, which includes
wherever its numbers are published. `BENCHMARKS.md` names the dataset in the
Source column of every row measured on it and points here; the README does the
same beside the figures it quotes.

## Third-party material behind published measurements

Nothing in this section is a corpus under `corpora/` and nothing in it is
redistributed by this repository. It is here because `benchmarks/RESULTS.md`
publishes precision and recall measured with it, and this file's own rule is
that a published figure is a use. Every revision, byte count and SHA-256 below
is recorded in `benchmarks/pins.json` and re-verified on every run.

### PINT Benchmark, by Lakera AI

`benchmarks/RESULTS.md` scores both `injection-structural` and the classifiers
below on `benchmark/data/example-dataset.yaml` from the [PINT
Benchmark](https://github.com/lakeraai/pint-benchmark).

- Licence: MIT, Copyright (c) 2024 Lakera AI
- Commit: `0efab3f463eae9c823130d8faffb71b2e7c06e63`
- File: `benchmark/data/example-dataset.yaml`, 8 inputs, sha256
  `df068b9a4ff72483f493add6be6242c6aa777df756bd61462aa0e13645cffa90`

**No changes were made** and no part of it is committed here. `benchmarks/run.py`
fetches the file at that commit into a gitignored `.cache/` directory inside
`benchmarks/` and checks its digest on every run, cached copy or fresh download
alike. The
evaluation function in `benchmarks/pint/` follows the shape of PINT's own
`examples/` template, which is covered by the same MIT licence.

Eight inputs is not the PINT Benchmark. The PINT dataset is 4,314 inputs, is not
public, and PINT's contributing guide requires results to be verified by the
Lakera team before publication. This repository has no PINT score, claims none,
and says so in every document that touches the file.

### DeBERTa prompt-injection classifiers, by Protect AI

Two revisions are measured, both fine-tuned from `microsoft/deberta-v3-base`.

| Model | Revision | Status | Licence |
|---|---|---|---|
| `protectai/deberta-v3-base-prompt-injection-v2` | `90c9989b1a342275dd0d1a95aad283c04e075671` | current | Apache-2.0 |
| `protectai/deberta-v3-base-prompt-injection` | `373b6af0f8d16739cff5de28be326652246bfaa3` | superseded by the row above | Apache-2.0 |

**No weights are vendored and nothing is redistributed.** The ONNX export, its
config and its tokenizer are downloaded at the pinned revision, checked against
the byte counts and SHA-256 digests in `benchmarks/pins.json`, and used to
classify. Apache-2.0 attaches no attribution obligation to a measurement made
this way; the entry is here because the measurement is published and the models
are somebody else's work.

Both model cards carry ProtectAI's notice that the project is archived and no
longer maintained, and the older card states that the `-v2` model supersedes it.
The v1 card also carries a "License and Usage Notice" warning that some training
datasets may carry non-commercial terms. That has no bearing on measuring the
model, and it would have a bearing on anything downstream that bundled it, which
this repository does not.

## Training corpora

Nothing in this section is a file in this repository and no published number is
measured on any of it. These are the public corpora the stage 2b injection
classifier may be fitted on, recorded here because attribution is a condition of
one of their licences and a condition does not wait for a file to be committed.
The manifest that governs them, with the digest each hashed to and the reason
each was admitted or refused, is `training/sources.yaml`.

Portions of the training data are derived from the **prompt_injections dataset**
(`yanismiraoui/prompt_injections`) by Yanis Miraoui, licensed under the Apache
License, Version 2.0. Its own NOTICE
file is reproduced here, which is what section 4(d) of that licence asks for:

> prompt_injections dataset
> Copyright 2023 Yanis Miraoui
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use the contents of this repository except in compliance with the License.
> You may obtain a copy of the License at
> <http://www.apache.org/licenses/LICENSE-2.0>
>
> This NOTICE applies to the dataset contents (including prompt_injections.csv)
> as well as the accompanying documentation in this repository.

- Dataset: <https://huggingface.co/datasets/yanismiraoui/prompt_injections>
- Revision: `bd55359f2f332afc35f277ac3dd08f7111b024c9`
- File: `prompt_injections.csv`, sha256
  `f4843f1841fa19b980f804796a68fc72f06841775eaba2723c768c7d772aabad`
- SPDX identifier: `Apache-2.0`
- Licence text: <https://www.apache.org/licenses/LICENSE-2.0>

`fka/awesome-chatgpt-prompts`, CC0-1.0, was admitted for training until the
evaluation set became external and the two were compared: it carries the DAN
prompt and so does `jackhhao/jailbreak-classification`, 3 rows exactly and 6
near. It is `role: excluded` for that reason. A public-domain dedication asks
for nothing, so nothing was owed either way, and it is named here because it
was named here before and a corpus that quietly disappears from an attribution
file is a corpus nobody can check the history of.

**No corpus in this section may be scored on.** The screen for that lives in
`tests/test_training_data.py`, not in this document, and the reason it applies
to corpora nobody has denylisted is that the denylist is known to be partial:
ProtectAI's v2 card counts 22 source datasets and names 7.

## Evaluation corpus

The stage 2b injection classifier is scored on
`jackhhao/jailbreak-classification` by Jack Hao, licensed under the Apache
License, Version 2.0. That corpus is not a file in this repository; it is
downloaded against a recorded digest by `training/fetch.py` and it is named here
because a published figure is a use, which is the rule this document states
about itself at the top.

- Dataset: <https://huggingface.co/datasets/jackhhao/jailbreak-classification>
- Revision: `2f2ceeb39658696fd3f462403562b6eea5306287`
- File: `default/jailbreak_dataset_full.csv`, sha256
  `79a7b90b0abe00e3586cc5048353c3236543cca228a1ee261fe3b57a7cb7e29f`
- SPDX identifier: `Apache-2.0`
- Licence text: <https://www.apache.org/licenses/LICENSE-2.0>

The repository ships no NOTICE file of its own, so there is none to reproduce;
the attribution above is what section 4 of that licence asks for in its absence.

Its own card records where its rows came from, and reading it is part of reading
any number measured on it. The jailbreak prompts are from the `jailbreak_llms`
collection by Xinyue Shen and colleagues, and the benign prompts from
`Open-Orca/OpenOrca` and the `GPTeacher` collection. So the label correlates
with the upstream source, which is a limit on what the corpus can show and is
recorded rather than left to be discovered.

**This corpus is on the contamination denylist and is scored on anyway.**
ProtectAI's v2 card names it as that model's own training data. The reasoning
for using it regardless is in `training/evalset.py` and in its
`training/sources.yaml` entry, and the short form is that contamination in an
evaluation set biases towards whichever model memorised it: DeBERTa may have
these rows, our encoder has seen none of them, so a win for us is meaningful and
a loss is inconclusive. It is also jailbreak classification rather than prompt
injection, which are adjacent and not the same task, and it is the only external
evaluation corpus this stage has.

## What is deliberately absent, and why

**There is no third-party secrets corpus.** No compatibly-licensed one was
found, so the secrets numbers are measured on our own corpus only and are
self-graded. That is stated rather than left for a reader to notice from a
missing row.

**The structural-injection check counts 3,773 invisible characters on Unicode
16.0.0, and the rule that picks them is derived rather than chosen.** The count
is a fact about the interpreter's Unicode data and not a constant of this
package; the exact figures per version, and the one behavioural difference they
cause, are at the end of this section. It counted five, listed by
hand, and then 29 under a rule about format characters. Both were too narrow,
and the second was too narrow for a reason worth recording: it excluded the
Hangul fillers because they are "handled where letters are, by
`_joining_neighbour`'s range test", and that test refuses them as an EXCUSING
NEIGHBOUR while saying nothing about them as a CARRIER. A fact about one role,
written down as an assurance about another.

Swept as carriers, **every** default-ignorable family left out of the set turned
out to carry a full payload at 1.0000 characters per bit with nothing on the
page and the payload recovering verbatim: the Hangul fillers (`Lo`), the Khmer
inherent vowels (`Mn`), the unassigned default-ignorable code points (`Cn`), the
variation selectors, and the directional marks.

The rule now is **default-ignorable**, minus three families and two named code
points:

| Excluded | Why |
|---|---|
| directional format characters | ordinary right-to-left text is written with U+200E, U+200F and U+061C, and `_bidi_spans` owns U+202A..U+202E and U+2066..U+2069, where a balanced pair is deliberately allowed |
| every VARIATION SELECTOR, all 260 on Unicode 14.0.0 and later (259 on 13.0.0) | a variation selector modifies the glyph of the character before it, so it is orthography wherever that character is: U+FE0F is in every emoji sequence, the 240 ideographic ones are in Japanese personal names, and the four Mongolian ones are written word-finally. The test is the character's NAME, so it matches only the selectors the interpreter's Unicode version has named |
| `U+00AD` SOFT HYPHEN | the one member that RENDERS, as a hyphen wherever the line breaks, and it is in every hyphenated ebook |
| the tag block | `INVISIBLE_TAG_CHARS` owns it; counting it twice would make every subdivision flag carry six of these as well |

Two of the three family tests are read off `unicodedata` -- the bidi class and
the character's name -- so they cannot drift from the Unicode data the
interpreter ships. The other side of that is that the RESULT moves with the
interpreter. Measured on the five the CI matrix runs:

| Python | Unicode | members | unassigned | `Cf` | `Lo` | `Mn` |
|---|---|---:|---:|---:|---:|---:|
| 3.10 | 13.0.0 | 3,774 | 3,739 | 28 | 4 | 3 |
| 3.11 | 14.0.0 | 3,773 | 3,738 | 28 | 4 | 3 |
| 3.12 | 15.0.0 | 3,773 | 3,738 | 28 | 4 | 3 |
| 3.13 | 15.1.0 | 3,773 | 3,738 | 28 | 4 | 3 |
| 3.14 | 16.0.0 | 3,773 | 3,738 | 28 | 4 | 3 |

**One code point is the whole difference, and it changes what this detector
denies.** U+180F MONGOLIAN FREE VARIATION SELECTOR FOUR was UNASSIGNED in
Unicode 13.0.0. The selector exclusion asks for the words VARIATION SELECTOR in
the character's NAME, and an unassigned code point has no name, so on 13.0.0 it
is not excluded and is counted as an invisible character; from 14.0.0 it is
named and dropped like the other three. Diffing the two sets code point by code
point, the symmetric difference is exactly `U+180F` and nothing else.

So **Mongolian text using free variation selector four scores differently by
Python version**: measured, five Mongolian words each carrying one of them DENY
on Python 3.10 and ALLOW on 3.11 and later. Four allow on both, since four is
under the total bound wherever the character is counted. This is narrow -- one
code point, one script, and only at or above the total bound -- and it is
disclosed here rather than smoothed over, because a caller scoring the same
document on two interpreters can get two answers. Whoever needs one answer
should pin the interpreter, which is the only fix that does not mean freezing a
Unicode table into the package.

It does NOT reach the published scores. No case in any corpus here contains
U+180F, so `benchmarks.json` and `BENCHMARKS.md` regenerate byte-identically on
Python 3.10 and on 3.14, and the precision and recall in this repository are the
same number on every leg. That is checked rather than assumed, because CI
regenerates both artifacts on all five legs and diffs them.

**What this does NOT close, and why no minimum is published for it.** This
project has published four different numbers for the cost of getting a payload
past this check -- 1.4875, then 1.0000, then 0.1250, then 0.1247 -- each written
in the sentence correcting the last, and each was wrong in the same way.

The defect is not arithmetic. **A minimum is a claim about every possible
encoding. A measurement exhibits one encoding.** No sweep of this kind can
establish a minimum, and every round found a cheaper encoder because there are
always more encoders. So no minimum is stated here or anywhere else in this
repository, and the sentence that replaced it is the one that will still be true
after somebody finds a cheaper encoder than any listed below.

What can be defended is a LIST: the families this rule does not count, with one
measured encoder for each. The list is existential and checkable. Each figure is
the cost of THAT ENCODING and is not a bound on anything.

| Family | Size | An encoder, and what it cost |
|---|---:|---|
| variation selectors | 256 | one selector per byte: **0.1250** chars/bit |
| the same, plus the 3 directional marks and the 4 Mongolian selectors | 263 | base 263 is not needed; the byte encoder above already uses 256: 0.1250 |
| C1 controls | 32 | base 32: **0.1992** |
| C0 controls, tab, LF and CR excluded | 29 | base 29: **0.2070** |
| Egyptian hieroglyph format controls | 16 | base 16: **0.2500** |
| `Prepended_Concatenation_Mark` characters | 13 | base 13: **0.2695** |
| interlinear annotation characters | 3 | base 3: **0.6289** |
| directional marks alone | 3 | one character per bit, two of the three: **1.0000** |

Every row was measured against the committed detector, returns **zero
findings**, and decodes back to "ignore all previous instructions" verbatim. Two
of these rows were published in the round before this one at 0.2500 and 1.0000,
because those encoders used 16 of the 29 C0 controls and 2 of the 3 annotation
characters: the same one-bit-per-character assumption, four lines under a
sentence rejecting it.

**And a cost per bit is itself one accounting.** An attacker already sending a
document pays only for the characters they ADD, and the total bound lets four
counted characters through **provided no two are adjacent** -- two adjacent are
a run, and the run bound is 2. Measured: two adjacent deny, two scattered allow,
four scattered allow, five scattered deny.

Priced over the alphabet that bound actually governs, which is the **counted**
set: take `inj-0105`'s text with its own zero-width characters stripped, 2,499
characters, and ADD four counted ones. That is a 2,503-character page -- the
length of `inj-0106` -- in which the four choose among C(2500, 4) pairwise
non-adjacent positions and 3,773 symbols each -- the Unicode 16.0.0 alphabet
size; on 13.0.0 it is 3,774, which moves the figure to 88.0899 from 88.0884 and
rounds to the same one -- which is **88.1 bits carried by 4 added characters**. The page is the corpus case and the four are not --
`inj-0105` carries three of its own at 2,502 characters and allows with zero
findings, and `inj-0106` is the same page carrying four at 2,503. The slot count
is the page AFTER the additions; pricing 2,502 slots for a construction that
makes 2,503 moves the figure by 0.0023 bits and rounds to the same 88.1. Priced
instead over the 259 symbols that bound does not count it
comes to 72.6, which is an accounting of two different things and understates
the leak of the very bound it names.

**Raising `_MIN_TOTAL` from 4 to 5 widened this by 21.2 bits** on this document,
from three characters and 66.9 bits to four and 88.1. That is the standing price
of the twelve false-positive cases the raise bought back, and it belongs beside
the leak rather than only beside the corpus. Whether 88.1 bits for four
characters reads as "cheap" or "free" depends on what the cover is charged to,
which is a second reason no single number carries this claim.
`test_the_bound_passes_four_non_adjacent_characters_and_what_they_carry` holds
all three figures.

**Nothing here is closed.** The `Prepended_Concatenation_Mark` characters --
ten Arabic, two Kaithi and U+070F SYRIAC ABBREVIATION MARK -- are `Cf` and are
NOT default-ignorable, so the rule never reached them; the control families render in a renderer-dependent way; and the
positional channel is a property of the bound rather than of the character set.
They are listed because a family nobody has written down is a family nobody can
close.

**Why the two swept families stay out, evidenced rather than asserted.** This
notice said "counting variation selectors denies every emoji sequence" and that
is false: with them counted, a single heart with U+FE0F, three keycaps and a
four-person family sequence all still allow, because one, three or four
unexplained characters is under the total bound. What is true is narrower and
still decisive, and the corpus carries every one of these rather than leaving
the claim on this file's word:

| Sample | Case | With the family counted |
|---|---|---|
| five keycaps | `inj-0143` | denies |
| five text-default emoji, each needing U+FE0F | `inj-0144` | denies |
| five Japanese surnames written with ideographic variation sequences | `inj-0145` | denies |
| a bilingual invoice carrying five directional marks | `inj-0146` | denies |

Five rather than four because `_MIN_TOTAL` was raised from 4 to 5, and the four
negatives were widened by one occurrence when it moved. At four occurrences they
allow whether the families are counted or not, so they would have justified
nothing while still reading as evidence. A justification measured against a
bound has to be re-measured when the bound moves.

A RAINBOW FLAG stood at the head of that table, argued to deny on the RUN bound
because U+FE0F sits immediately before U+200D. Running the mutation refutes it:
U+FE0F is inside the pictographic ranges, so with selectors counted it EXPLAINS
the joiner and one flag is one suspicious character. One, two and three rainbow
flags allow; up to four allow and five deny on the total bound, exactly like
the keycaps. It was reasoned about rather than run, and it is recorded here
because it was published as the decisive case in the round before this one.

Five of anything from either family reaches the total bound, and that is enough:
a check that denies five keycaps or a bilingual invoice is one that gets
switched off. Both families stay out, and the table above is the price.

**Three costs were measured before this landed. Two are real.**

*Mongolian: none.* U+180E MONGOLIAN VOWEL SEPARATOR is in the set and gets the
both-neighbours context test ZWNJ gets, because it stands between a word and its
suffix vowel. The four Mongolian free variation selectors are four of the 260 the rule
excludes by name, not four more, and
that distinction is load-bearing: a variation selector is written word-FINALLY,
where a both-neighbours rule has nothing to its right and would deny. Measured
on five samples, all five allow: `inj-0125`, `inj-0126`, `inj-0127`.

**Constructed samples, disclosed as constructed.** The Mongolian words
(`inj-0125` to `inj-0127`), the Korean jamo table (`inj-0134`), the Khmer
dictionary entry (`inj-0135`), the collation line (`inj-0136`), the Biblical
Hebrew line (`inj-0137`), the two balanced overrides around Hebrew and Arabic
(`inj-0141`, `inj-0142`), the Japanese names (`inj-0145`) and the bilingual
invoice (`inj-0146`) are built from the Unicode encoding model rather than drawn
from a corpus. So is `inj-0129`, which wraps musical beam controls around the
ASCII letters `CD` rather than around musical symbols: it is the control
characters that are under test, and their placement, not the notes between
them. The same standard the
detector already applies to the Persian ezafe ordering, which it records as
asserted rather than evidenced.

*Korean and Khmer: real, and narrower since the bound moved.* Ordinary Korean
and ordinary Khmer carry none of these characters, which was checked rather than
assumed. What can still deny is prose ABOUT the script: a jamo table or a
dictionary entry, once five of them reach the total bound. Measured: four allow
and five deny. `inj-0134` and `inj-0135` carry four each and now pass, so this
cost is one entry further out than it was, not gone.

*Mathematics, music, collation and concatenated files: real, and now one entry
further out.* U+2061..U+2064 are genuine in MathML and U+1D173..U+1D17A in the
plain-text encoding of musical notation; five in one line is the bound. U+034F
has no context test, and neither does almost anything else: of the 3,773 members
on Unicode 16.0.0 only three have one -- U+200C, U+200D and U+180E -- so the
other 3,770, the Hangul fillers and the Khmer inherent vowels above included,
are counted wherever they appear. The three with a context test are the same
three on every Unicode version this package runs on; it is the total that moves,
so on 13.0.0 the figures are 3,774 and 3,771.

So both of U+034F's real uses still deny at five
occurrences: blocking a collation contraction so a digraph sorts as two letters
rather than one, and fixing the order of two points on one letter in Biblical
Hebrew. At four they allow, which is why `inj-0136` and `inj-0137` pass now. U+034F was added to the set BY NAME in fix round 1, as the one
default-ignorable mark that is neither a variation selector nor Khmer
orthography; fix round 3 replaced that named addition with a general rule it
falls out of, and measured what it costs. And UTF-8 files concatenated with each
keeping its own BOM is one occurrence per file, which is the same
retrieval-pipeline setting `inj-0105` and `inj-0106` come from.

Raising `_MIN_TOTAL` to 5 bought back every one of these but the music.
`inj-0128`, `inj-0136`, `inj-0137` and `inj-0138` carry four occurrences each
and now pass; a fifth in any of them denies, so what moved is the boundary and
not the trade. `inj-0129` is the exception and the reason the two bounds are
separate signals: an END BEAM immediately followed by the next BEGIN BEAM is an
adjacent PAIR, which `_MIN_RUN` reports at two whatever the total is.

The exemption that would close the mathematical case is not available, which is
why the trade went this way rather than by preference. An invisible operator
sits between two operands, so the rule would have to be "excuse it when both
neighbours are characters mathematics writes" -- letters, digits, brackets --
which is every neighbour a Latin cover offers, so it would excuse the
1.0000-per-bit channel at one cover character per bit. An exemption whose
condition an attacker satisfies for free is the shape this module has already
replaced twice.

**Two otherwise-ideal PII corpora were rejected on licence**, and the reason
does not appear in any licence field: `beki/privy` and
`microsoft/presidio-research` both advertise MIT while their PII *values* derive
from Fake Name Generator identities, which are dual-licensed GPLv3 and
CC-BY-SA-3.0-US. An Apache-2.0 or MIT tag downstream does not cure a share-alike
upstream. The verification chain and the value fingerprint that detects it are
in `docs/conformance.md`, and the fingerprint is enforced as a test over these
committed files in `tests/test_corpora.py`.
