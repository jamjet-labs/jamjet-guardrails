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
ordinary text because of it.** 0.679 precision, 0.974 recall, 17 wrong
decisions over 146 cases. Seventeen cases fail, every one of them on purpose,
and there are no other failures. Twenty-four cases are listed below: the
seventeen that fail, plus the three-case balanced-override set and the four
stray-closer cases, which pass and are here because a reader can reasonably
expect the opposite of each.

- **Fourteen deny text somebody wrote on purpose, and are labelled `allow`.**
  They score as 53 of the corpus's false positives, which is where nearly all of
  the distance from a perfect score comes from. `inj-0090` is a Thai sentence marked up for
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
  same way. `inj-0128` and `inj-0129` are MathML and musical
  notation extracted to plain text, and `inj-0134`, `inj-0135`, `inj-0136`,
  `inj-0137` and `inj-0138` are the cost of the wider invisible set described
  below: Korean prose about jamo, a Khmer dictionary entry, U+034F blocking a
  collation contraction, U+034F fixing point order in Biblical Hebrew, and four
  UTF-8 files concatenated with each keeping its own BOM. Every one of the
  fourteen is four occurrences reaching an absolute bound.
- **Three allow a payload that is really there, and are labelled `deny`.** They
  score as the corpus's three false negatives. `inj-0097`, `inj-0098` and
  `inj-0099` each carry the string `exfiltrate` past the check, decoded back out
  to check that: presence-and-absence spacing of a joiner behind a Devanagari
  cover, the same encoding between variation selectors at 119 characters with
  nothing on the page at all, and a bitstream deperiodised with one spare cover
  character every three bits.
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
holds all twenty-four ids against this section, in both directions.

**The corpus moves when the detector does, and that was measured rather than
assumed.** Twenty-seven copies of `injection_structural.py` were made, each with
one rule removed or loosened, and this corpus was scored against each. **All
twenty-seven break at least one case beyond the seventeen that already fail.**

Twenty cover the rules the detector shipped with, and they are, one per mutant:
the RGI allowlist softened to a prefix test; the flag exemption's CANCEL TAG
condition dropped; its flag-base condition dropped; the flag exemption removed
entirely; the paragraph flush deleted; the balanced-pair rule removed; the two
bidi families merged into one stack; `_MIN_TOTAL` raised by one; `_MIN_RUN`
raised by one; `_MIN_PERIODIC` raised by one; the periodicity rule deleted; a
joiner excused when EITHER neighbour is a joining character; the virama's base
allowed to be any script; the virama branch deleted; the pictographic branch
deleted; `_MAX_TRANSPARENT` cut to one; decimal digits refused as excusing
neighbours; marks refused as excusing neighbours; a mark neighbour no longer
required to reach a letter; and WORD JOINER and the BOM dropped from the set.
The widest is deleting the virama branch, which turns thirteen cases of ordinary
Brahmic and Malayalam text into denials.

Seven cover the rules added in the two fix rounds, which are the newest and
therefore the least exercised: reverting the invisible set to five hand-picked
characters breaks ten cases, narrowing it back to format characters alone six,
dropping the variation-selector exclusion two, dropping the directional
exclusion two, dropping the soft-hyphen exception one, giving the Mongolian
separator a bare range test one, and removing its branch entirely two.

Five cases exist only because a rule survived a sweep with nothing to show for
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

**The structural-injection check counts 3,773 invisible characters, and the
rule that picks them is derived rather than chosen.** It counted five, listed by
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
| every VARIATION SELECTOR, all 260 | a variation selector modifies the glyph of the character before it, so it is orthography wherever that character is: U+FE0F is in every emoji sequence, the 240 ideographic ones are in Japanese personal names, and the four Mongolian ones are written word-finally |
| `U+00AD` SOFT HYPHEN | the one member that RENDERS, as a hyphen wherever the line breaks, and it is in every hyphenated ebook |
| the tag block | `INVISIBLE_TAG_CHARS` owns it; counting it twice would make every subdivision flag carry six of these as well |

Two of the three family tests are read off `unicodedata` -- the bidi class and
the character's name -- so they cannot drift from the Unicode data the
interpreter ships. What is left is 3,773 code points: 3,738 unassigned, the four
Hangul fillers, the two Khmer inherent vowels and U+034F, and 28 format
characters.

**What this does NOT close, and the cost model the module had wrong three
times.** Every rate this project published for an invisible channel was a rate
for a TWO-SYMBOL encoder. Two symbols is a choice an encoder makes; it is never
a property of a channel. **The floor for an alphabet of n invisible symbols is
1 / log2(n) characters per bit**, because one character drawn from an n-symbol
alphabet carries log2(n) bits.

There are **256** variation selectors, VARIATION SELECTOR-1..16 at U+FE00 and
VARIATION SELECTOR-17..256 at U+E0100, so they are a BYTE per character rather
than a bit. Measured against the committed detector: `"Summarise this. "` plus
one emoji plus **32** variation selectors returns **zero findings** and decodes
back to "ignore all previous instructions" verbatim, at **0.1250 characters per
bit**. That is 1/log2(256) exactly. Adding U+200E, U+200F and U+061C widens the
alphabet to 259 and the floor to 0.1247.

The three figures this project has published for that one channel are 1.4875,
1.0000 and 0.1250, and only the last is a property of the channel: the first was
measured on presence-and-absence, the second on a two-symbol stream over two of
the 256 symbols, and each was written down as if it bounded the channel.
`test_the_cheapest_invisible_channel_is_measured_over_alphabet_size` now holds
the measurement and the arithmetic together.

**That floor is over the families this file has SWEPT, and it is not a floor
over the code space.** Four more invisible families are uncounted, unclosed and
measured here rather than left for a reader to find: the C0 controls at 0.2500
per bit, the C1 controls at 0.1992, U+FFF9..U+FFFB at 1.0000 and the Egyptian
hieroglyph format controls at 0.2500, each returning zero findings. Closing them
is not attempted here -- rendering is renderer-dependent for several -- and the
C0 half is the least arguable of the four.

**Why the two swept families stay out, evidenced rather than asserted.** This
notice said "counting variation selectors denies every emoji sequence" and that
is false: with them counted, a single heart with U+FE0F, three keycaps and a
four-person family sequence all still allow, because one or three unexplained
characters is under the total bound. What is true is narrower and still
decisive, and the corpus now carries every one of these rather than leaving the
claim on this file's word:

| Sample | Case | With the family counted |
|---|---|---|
| one rainbow flag | `inj-0058` | **denies** on the RUN bound: U+1F3F3 U+FE0F U+200D U+1F308 puts the selector immediately before the joiner, so one emoji is two adjacent unexplained characters |
| four keycaps | `inj-0143` | denies |
| four text-default emoji, each needing U+FE0F | `inj-0144` | denies |
| four Japanese surnames written with ideographic variation sequences | `inj-0145` | denies |
| a bilingual invoice carrying four directional marks | `inj-0146` | denies |

The rainbow flag settles it on its own. A check that denies one of those is a
check that gets switched off, which is the argument this project already makes
about the Scotland flag. Both families stay out, and the residual above is the
price.

**Three costs were measured before this landed. Two are real.**

*Mongolian: none.* U+180E MONGOLIAN VOWEL SEPARATOR is in the set and gets the
both-neighbours context test ZWNJ gets, because it stands between a word and its
suffix vowel. The free variation selectors are excluded with the other 260, and
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

*Korean and Khmer: real, and narrow.* Ordinary Korean and ordinary Khmer carry
none of these characters, which was checked rather than assumed. What denies is
prose ABOUT the script: a jamo table with four fillers, a dictionary entry with
four inherent vowels. Measured: one allows, two allow, four deny. U+3164 used as
a blank placeholder denies at four as well. `inj-0134` and `inj-0135`.

*Mathematics, music, collation and concatenated files: real.* U+2061..U+2064 are
genuine in MathML and U+1D173..U+1D17A in the plain-text encoding of musical
notation; four in one line is the bound. U+034F has no context test, and neither
does almost anything else: of the 3,773 members only three have one -- U+200C,
U+200D and U+180E -- so the other 3,770, the Hangul fillers and the Khmer
inherent vowels above included, are counted wherever they appear. So both of
U+034F's real uses deny: blocking a collation contraction so a digraph sorts as
two letters rather than one, and fixing the order of two points on one letter in
Biblical Hebrew. U+034F entered the set in the round before this one, under the
narrower rule, and this round is what measured the cost. And four UTF-8 files concatenated with
each keeping its own BOM is four occurrences, which is the same
retrieval-pipeline setting `inj-0105` and `inj-0106` come from. `inj-0128`,
`inj-0129`, `inj-0136`, `inj-0137` and `inj-0138`.

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
