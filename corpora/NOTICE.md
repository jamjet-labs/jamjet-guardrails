# Corpus provenance and attribution

Every published precision and recall figure in this repository is measured on
one of the files under `corpora/`. This file records where each one came from
and under what licence, because for one of them that is a condition of use and
not a courtesy.

Each corpus is one file, one source: the loader refuses a file that mixes them,
so in-repo and third-party numbers can never be merged into one score.

| Corpus | `source` field | Licence |
|---|---|---|
| `corpora/pii/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/secrets/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/pii/third-party.jsonl` | `nvidia/Nemotron-PII@b70ffaf` | `CC-BY-4.0` |

## First-party corpora

The two `in-repo` files were written for this repository and are covered by its
own Apache-2.0 licence. Every value in them is invented or is a published test
value: `AKIAIOSFODNN7EXAMPLE` is AWS's own documentation example, `4111 1111
1111 1111` is the universally published test PAN, and `example.com` and
`example.org` are reserved for documentation by RFC 2606. The two
internationalised addresses use that same `example` label under a Cyrillic and a
Devanagari TLD.

No real credential and no real person's data is in either file. The GitHub,
OpenAI, Anthropic and Slack tokens carry `EXAMPLEONLY` or `notarealtoken` inside
their own bodies; the JWTs are a standard HS256 header with an invented payload
and a signature of random bytes, so they verify against nothing; and the PEM
bodies are base64 of random bytes rather than DER, so no tool can load one as a
key.

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

**Two otherwise-ideal PII corpora were rejected on licence**, and the reason
does not appear in any licence field: `beki/privy` and
`microsoft/presidio-research` both advertise MIT while their PII *values* derive
from Fake Name Generator identities, which are dual-licensed GPLv3 and
CC-BY-SA-3.0-US. An Apache-2.0 or MIT tag downstream does not cure a share-alike
upstream. The verification chain and the value fingerprint that detects it are
in `docs/conformance.md`, and the fingerprint is enforced as a test over these
committed files in `tests/test_corpora.py`.
