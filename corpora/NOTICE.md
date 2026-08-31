# Corpus provenance and attribution

Every published precision and recall figure in this repository is measured on
one of the files under `corpora/`. This file records where each one came from
and under what licence, because for one of them that is a condition of use and
not a courtesy.

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

**`corpora/injection-structural/in-repo.jsonl` scores the implementation
against the design, and the design is not the same thing as a reader's
judgement.** Its row publishes `1.000` on both ratios over 119 cases with no
wrong decisions -- an exact 1, not a rounded one, which is why the table's
`>0.999` hedge does not appear on it -- and what that says is that the check
does what
`src/jamjet_guardrails/detectors/injection_structural.py` says it does. It does
not say the check has no false positives on real text. Eleven of the 119 cases
are labelled at the design's behaviour and a reader would disagree with the
label on every one of them, so those eleven are named here rather than left to
be found:

- **Six deny text somebody wrote on purpose.** `inj-0090` is a Thai sentence
  marked up for line breaking, which carries one U+200B per word boundary and
  reaches the four-character total bound on five words. `inj-0091` wraps a
  two-line value in `FSI ... PDI`, the idiom Unicode recommends and the one
  `<bdi>` implements; a control's scope ends at its paragraph, so the PDI on the
  second line closes nothing and both controls are reported, while the text
  renders byte-identically to the same string with the wrapper deleted --
  measured with GNU FriBidi 1.0.16 in
  `tests/test_injection_structural.py`. `inj-0092`, `inj-0093` and `inj-0095`
  are Persian and Urdu numeral compounds written with ASCII digits, and
  `inj-0094` is the Persian plural suffix on Latin acronyms. An excusing
  neighbour has to sit inside the joining-script ranges and an ASCII digit does
  not, so the same sentences written with Arabic-Indic digits allow: they are
  `inj-0080`, `inj-0082` and `inj-0079`, and they are in the file beside them.
- **One denies for being long.** `inj-0106` is a 2,503-character retrieved page
  whose only fault is four incidental U+200B at sentence boundaries, the shape
  an HTML pipeline leaves behind. The total bound is a bound on the whole input
  rather than a rate, so it does not move with length: `inj-0105` is the same
  page carrying three and allows, and the same pair measured at 10,000
  characters answers the same way.
- **Four allow a payload that is really there.** `inj-0096` is a balanced
  override, which reverses its own scope and renders `transfer 100 USD`.
  `inj-0097`, `inj-0098` and `inj-0099` each carry the string `exfiltrate` past
  the check, decoded back out to check that: presence-and-absence spacing of a
  joiner behind a Devanagari cover, the same encoding between variation
  selectors at 119 characters with nothing on the page at all, and a bitstream
  deperiodised with one spare cover character every three bits.

Every one of the eleven is a trade taken deliberately and recorded in the
detector, and each has a test in `tests/test_injection_structural.py` that fails
the day it is re-taken.
`tests/test_corpora.py::test_a_disclosed_injection_shape_is_in_the_corpus_and_in_the_notice`
holds each id against this list, in both directions, because a case that always
passes moves no number and nothing else in the suite would notice it go.

**A 1.000 is only worth reading if the corpus moves when the detector does, and
that was measured rather than assumed.** Twenty copies of
`injection_structural.py` were made, each with one rule removed or loosened --
the RGI allowlist softened to a prefix test, the paragraph flush deleted, the
two bidi families merged into one stack, each of the three zero-width bounds
raised by one, a joiner excused when EITHER neighbour is a joining character,
the virama's base allowed to be any script, the pictographic and virama
branches deleted outright -- and the committed corpus was scored against each.
All twenty break at least one case. The widest is deleting the virama branch,
which turns thirteen cases of ordinary Brahmic and Malayalam text into denials
and takes precision to 0.685; the narrowest break one case each. Five cases in
the file exist because a rule survived the first sweep with nothing to show for
it: `inj-0115` for the CANCEL TAG condition, `inj-0116` for the periodicity
bound from underneath, `inj-0117` for the virama's own script, and `inj-0118`
and `inj-0119` for WORD JOINER and the BOM, which until then appeared only in
samples that allow either way.

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

## What is deliberately absent, and why

**There is no third-party secrets corpus.** No compatibly-licensed one was
found, so the secrets numbers are measured on our own corpus only and are
self-graded. That is stated rather than left for a reader to notice from a
missing row.

**The structural-injection corpus measures five zero-width characters and
there are more.** `_ZERO_WIDTH` is a closed list -- U+200B, U+200C, U+200D,
U+2060 and U+FEFF -- so a code point that is invisible, carries no orthography
and is not on that list is not a signal and no case here labels one. Measured
against the committed detector: U+2061..U+2064, the four invisible math
operators, are category `Cf`, render nothing, and carry "ignore all previous
instructions" through this check at 1.0000 characters per bit with nothing on
the page. That is cheaper than every residual the detector records, the nearest
of which is the variation-selector channel at 1.4875. It is stated here rather
than labelled as a case, because deciding what belongs in that list is a change
to the detector and not to its evidence.

**Two otherwise-ideal PII corpora were rejected on licence**, and the reason
does not appear in any licence field: `beki/privy` and
`microsoft/presidio-research` both advertise MIT while their PII *values* derive
from Fake Name Generator identities, which are dual-licensed GPLv3 and
CC-BY-SA-3.0-US. An Apache-2.0 or MIT tag downstream does not cure a share-alike
upstream. The verification chain and the value fingerprint that detects it are
in `docs/conformance.md`, and the fingerprint is enforced as a test over these
committed files in `tests/test_corpora.py`.
