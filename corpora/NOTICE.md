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

**`corpora/injection-structural/in-repo.jsonl` labels its own failures as
failures, and its published numbers are lower than the check's behaviour on
ordinary text because of it.** 0.748 precision, 0.970 recall, 12 wrong
decisions over 129 cases. Thirteen of those cases carry a label that is NOT
what the detector does, and twelve of the thirteen fail on purpose:

- **Nine deny text somebody wrote on purpose, and are labelled `allow`.** They
  score as 33 of the corpus's false positives, which is where nearly all of the
  distance from 1.000 comes from. `inj-0090` is a Thai sentence marked up for
  line breaking, which carries one U+200B per word boundary and reaches the
  four-character total bound on five words. `inj-0091` wraps a two-line value in
  `FSI ... PDI`, the idiom Unicode recommends and the one `<bdi>` implements; a
  control's scope ends at its paragraph, so the PDI on the second line closes
  nothing and both controls are reported, while the text renders byte-identically
  to the same string with the wrapper deleted -- measured with GNU FriBidi
  1.0.16. `inj-0092`, `inj-0093` and `inj-0095` are Persian and Urdu numeral
  compounds written with ASCII digits and `inj-0094` is the Persian plural
  suffix on Latin acronyms; an excusing neighbour has to sit inside the
  joining-script ranges and an ASCII digit does not, so the same sentences with
  Arabic-Indic digits allow and are in the file as `inj-0079`, `inj-0080` and
  `inj-0082`. `inj-0106` is a 2,503-character retrieved page whose only fault is
  four incidental U+200B at sentence boundaries; the total bound is a bound on
  the whole input rather than a rate, so `inj-0105` is the same page carrying
  three and allows, and the same pair measured at 10,000 characters answers the
  same way. `inj-0128` and `inj-0129` are the cost of the wider invisible set
  described below.
- **Three allow a payload that is really there, and are labelled `deny`.** They
  score as the corpus's three false negatives. `inj-0097`, `inj-0098` and
  `inj-0099` each carry the string `exfiltrate` past the check, decoded back out
  to check that: presence-and-absence spacing of a joiner behind a Devanagari
  cover, the same encoding between variation selectors at 119 characters with
  nothing on the page at all, and a bitstream deperiodised with one spare cover
  character every three bits.
- **One is labelled `allow` and passes, and it is the exception.** `inj-0096` is
  a balanced override, which reverses its own scope and renders `transfer 100
  USD`. That is the SCOPE of the bidi signal rather than a miss: the rule is
  imbalance, and `inj-0029`, `inj-0030` and `inj-0038` are the same construct
  labelled `allow` because a balanced override is a negative this corpus needs.
  Labelling this one `deny` would contradict all three.

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
holds all thirteen ids against this section, in both directions.

**The corpus moves when the detector does, and that was measured rather than
assumed.** Twenty copies of `injection_structural.py` were made, each with one
rule removed or loosened -- the RGI allowlist softened to a prefix test, the
paragraph flush deleted, the two bidi families merged into one stack, each of
the three zero-width bounds raised by one, a joiner excused when EITHER
neighbour is a joining character, the virama's base allowed to be any script,
the pictographic and virama branches deleted outright -- and this corpus was
scored against each. All twenty break at least one case beyond the twelve that
already fail. The widest is deleting the virama branch, which turns thirteen
cases of ordinary Brahmic and Malayalam text into denials. Five cases exist
because a rule survived the first sweep with nothing to show for it: `inj-0115`
for the CANCEL TAG condition, `inj-0116` for the periodicity bound from
underneath, `inj-0117` for the virama's own script, and `inj-0118` and
`inj-0119` for WORD JOINER and the BOM, which until then appeared only in
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

**The structural-injection check counts 29 invisible characters, and the rule
that picks them is derived rather than chosen.** It used to count five, listed
by hand, and a hand-written list is a list of what somebody thought of.
Measured against that list: a two-symbol bitstream over U+2061 and U+2062,
FUNCTION APPLICATION and INVISIBLE TIMES, carried "ignore all previous
instructions" through the check at **1.0000 characters per bit with nothing on
the page and the payload recovering verbatim** -- cheaper than every residual
the detector records, the nearest being the variation-selector channel at
1.4875. The same construction ran over U+2063/U+2064, U+206A..U+206F,
U+1D173/U+1D174 and U+1BCA0/U+1BCA1, and presence-and-absence encodings ran over
U+034F and U+180E.

The rule now is: **default-ignorable, category `Cf`, bidi class BN**, minus
U+00AD and minus the tag block. Default-ignorable is Unicode's own name for
"renders as nothing", which is this signal's definition; two of the three
conditions are read off `unicodedata` so they cannot drift; and the two
exceptions each close something. U+00AD SOFT HYPHEN is the one member of that
set that renders -- as a hyphen, wherever the line breaks -- and it is in every
hyphenated ebook, so a signal that fires on six of them is a much larger Thai
case. The tag block is `INVISIBLE_TAG_CHARS`'s, and counting it twice would make
every subdivision flag carry six of these as well. U+034F COMBINING GRAPHEME
JOINER is added by name because it is `Mn` rather than `Cf`: of the 263
default-ignorable marks in Unicode 16.0.0, 260 are variation selectors and 2 are
the Khmer inherent vowels, and U+034F is the only one that is neither.

What the rule DROPS is as load-bearing as what it keeps, and each drop is a
family: the bidi marks U+200E, U+200F and U+061C, which ordinary
right-to-left text needs; U+202A..U+202E and U+2066..U+2069, which the bidi
signal already reports; the four Hangul fillers, which are letters; and all 260
variation selectors, emoji presentation and the 240 ideographic ones alike.

**Two costs were measured before this landed, and one of them is real.**

*Mongolian: none.* U+180E MONGOLIAN VOWEL SEPARATOR is `Cf`, BN and
default-ignorable, so the rule admits it, and it is also ordinary Mongolian --
it stands between a word and its suffix vowel, the job ZWNJ does in Persian. It
therefore gets what ZWNJ gets, a both-neighbours context test, rather than being
dropped. The Mongolian free variation selectors U+180B..U+180D and U+180F are
NOT in the set at all, and that distinction is the load-bearing one: a variation
selector is written word-FINALLY, where a both-neighbours rule has nothing to
its right and would deny, while the separator is written medially. Measured on
five samples -- four words with a suffix separator, the same with a selector
between the letter and the separator, four words ending in a selector, eight
separators in one paragraph, and selectors and separators mixed -- all five
allow. They are `inj-0125`, `inj-0126` and `inj-0127`.

*Mathematics and music: real, and taken deliberately.* U+2061..U+2064 are
genuine in MathML, and U+1D173..U+1D17A in the plain-text encoding of musical
notation. Extract either to plain text and four of them is the total bound:
measured, three invisible operators allow and four deny, and two musical beam
pairs are four controls and deny. `inj-0128` and `inj-0129` carry both, labelled
`allow`, and they cost 7 of this corpus's 33 false positives.

The exemption that would close it is not available, which is why the trade went
this way rather than by preference. An invisible operator sits between two
operands, so the rule would have to be "excuse it when both neighbours are
characters mathematics writes" -- letters, digits, brackets -- which is every
neighbour a Latin cover offers, so it would excuse the 1.0000-per-bit channel
above at one cover character per bit. An exemption whose condition an attacker
satisfies for free is the shape this module has already replaced twice.

**Two otherwise-ideal PII corpora were rejected on licence**, and the reason
does not appear in any licence field: `beki/privy` and
`microsoft/presidio-research` both advertise MIT while their PII *values* derive
from Fake Name Generator identities, which are dual-licensed GPLv3 and
CC-BY-SA-3.0-US. An Apache-2.0 or MIT tag downstream does not cure a share-alike
upstream. The verification chain and the value fingerprint that detects it are
in `docs/conformance.md`, and the fingerprint is enforced as a test over these
committed files in `tests/test_corpora.py`.
