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

The same widening happened a second time for a different reason. [Vendored
Unicode data](#vendored-unicode-data) is not a corpus and no number is measured
on it, but it is the one third-party thing this repository REDISTRIBUTES, and
redistribution is the strongest form of use there is. A file that recorded
provenance for what we measure on and not for what we ship would have the rule
exactly backwards.

Each corpus is one file, one source: the loader refuses a file that mixes them,
so in-repo and third-party numbers can never be merged into one score.

| Corpus | `source` field | Licence |
|---|---|---|
| `corpora/confusables/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/encoded-content/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/injection-structural/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/pii/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/rules/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/script-constraint/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/secrets/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/url-exfiltration/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/template-integrity/in-repo.jsonl` | `in-repo` | `Apache-2.0` |
| `corpora/pii/third-party.jsonl` | `nvidia/Nemotron-PII@b70ffaf` | `CC-BY-4.0` |

## First-party corpora

The `in-repo` files were written for this repository and are covered by
its own Apache-2.0 licence. Every value in them is invented or is a published test
value: `AKIAIOSFODNN7EXAMPLE` is AWS's own documentation example, `4111 1111
1111 1111` is the universally published test PAN, and `example.com` and
`example.org` are reserved for documentation by RFC 2606. The two
internationalised addresses use that same `example` label under a Cyrillic and a
Devanagari TLD.

No real credential and no real person's data is in any of them. The GitHub,
OpenAI, Anthropic and Slack tokens carry `EXAMPLEONLY` or `notarealtoken` inside
their own bodies; the JWTs are a standard HS256 header with an invented payload
and a signature of random bytes, so they verify against nothing; and the PEM
bodies are base64 of random bytes rather than DER, so no tool can load one as a
key.

That sentence used to stop at these corpus files, and two credential-shaped
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

### `corpora/rules/in-repo.jsonl`

Written for this repository and covered by its Apache-2.0 licence. It is not a
corpus of rules; it is a corpus of ENGINE behaviour, so its cases are built
around one fixed set of rules recorded in `docs/conformance.md` and are chosen
for the mechanics they separate: a span that must not run one character too far
when case folding changed a character's width, two rules claiming one stretch of
text, a limit at its bound and one past it, and an unanchored pattern matching
inside a longer path.

Every value in it is invented. The ticket ids are not any tracker's, the host
names use the `example` label reserved by RFC 2606, and the codename is a
colour and a bird.

**The disclosed misses.** There are none. The fixture scores 1.000 precision
and 1.000 recall over all 40 cases, with zero wrong decisions. That is a claim
about the engine under one fixed configuration and not about any rule a user
writes, so it carries none of the weight a perfect score on a heuristic check
like `pii` or `secrets` would: a corpus a detector passes completely is a
corpus that has stopped measuring. The two cases nearest a boundary are
`rules-0018`, 2,000 characters exactly at `max_chars` and labelled `allow`, and
`rules-0019`, the same text with one character appended, 2,001 characters,
labelled `redact` with a `LENGTH_LIMIT` finding at `[2000, 2001]`.

**There is no third-party corpus for this check**, and there could not be one:
the numbers depend on the rules, so a corpus somebody else wrote would measure
their rules through our engine. The row is self-graded in the same way the
secrets row is.

### `corpora/secrets/in-repo.jsonl`

Written for this repository and covered by its Apache-2.0 licence. Every
credential in it is invented or is a published example value, and the rule the
file follows is that a reader must be able to tell that from the value itself.
The GitHub, OpenAI, Anthropic and Slack bodies spell `EXAMPLEONLY`,
`EXAMPLE_ONLY`, `notarealtoken` or `notarealkey` inside themselves, which is the
repository-wide rule
`tests/test_packaging.py::test_no_credential_shaped_literal_in_the_repository_reads_as_a_live_one`
enforces. Every AWS access key id written into the file is either Amazon's own
published `AKIAIOSFODNN7EXAMPLE` and `ASIAIOSFODNN7EXAMPLE` or an `AKIA` or
`ASIA` prefix over a body spelling `EXAMPLEONLY`; where this check draws a span
across the join between one of those and its neighbour, as it does in
`sec-0098`, that span is a fragment of them and not some third value. The JWTs
carry a standard HS256 header and an invented payload over a signature of random
bytes, so they verify against nothing, and the PEM bodies are base64 of random
bytes rather than DER, so no tool can load one as a key.

**It grew from 39 cases to 160 on 2026-09-03, and the published figures moved
with it.** The 39 were 23 `redact` and 16 `allow`, all of them on the `output`
direction; the 160 are 95 `redact` and 65 `allow`, 112 `output` and 48 `input`.
Precision went from 0.957 to 0.881 and recall from 0.880 to 0.873, over 4 wrong
decisions before and 8 now. **Nothing about the detector changed.** The old
figures were measured on 39 examples, which is too few to carry a published
number, and the classes the new ones are measured over are the ones a
credential detector is actually wrong about:

- two credentials of one type joined directly, with nothing between them, for
  all six of the prefix families;
- a decoy of the same shape immediately in front of a real credential, both
  where the decoy is itself well-formed and where it is not;
- a credential butted against a PEM envelope on either side, and a credential
  adjacent to an email address or to a card number, which is where a chain's
  merged placeholder gets exercised;
- every declared type at several body lengths, from each pattern's own minimum
  upward, and in eight surroundings: bare, in prose, in a shell export, as a
  YAML value, inside a JSON string, inside a URL, inside a code fence, and at
  each end of the content;
- the hard negatives precision is made of, which the old file carried 16 of and
  this one carries 65: SHA-1, SHA-256 and MD5 digests, a git short SHA,
  UUIDs in both cases, npm and Go integrity hashes, base64 blobs and a data
  URI, `CERTIFICATE`, `PUBLIC KEY` and `CERTIFICATE REQUEST` envelopes, an
  `ssh-rsa` public key, high-entropy tokens carrying no issuer prefix, keys
  masked with asterisks and elided with an ellipsis, environment variable names
  with no value, a JSON schema naming key fields, issuer prefixes quoted in prose, and
  credentials truncated under their pattern's bound;
- both directions, where the old file measured only one.

A number that improved because the corpus got easier would be worse than one
that dropped because the corpus got honest, so the drop is the point and the
old figure is recorded beside it in `CHANGELOG.md`.

**What the 39 could not see, measured rather than argued.** `_scan` in
`src/jamjet_guardrails/_spans.py` tries every start position instead of resuming
at the end of each match, and its docstring records why: with `finditer`
semantics a decoy joined in front of a real credential leaves a whole 36-character
token body standing behind a publicly known prefix. That resume rule was put
back to `finditer` and both corpora were scored against the result. **The 39
cases scored 0.957 precision, 0.880 recall and 4 wrong decisions, which is
exactly what they score against the correct implementation**: the shape the
Critical is about was in none of them. The 160 score 0.884 and 0.827, five
findings lost. A corpus that cannot tell a fixed detector from the bug it was
fixed for is not measuring the fix.

**The disclosed misses: seventeen cases, and every one of them is labelled with
what SHOULD happen.** `docs/conformance.md` groups them into eight classes and
gives one worked input per class; they are named here by id so that a case
cannot quietly leave the file, and
`tests/test_corpora.py::test_a_case_that_records_a_miss_is_not_labelled_with_what_the_detector_does`
is what stops one being relabelled onto the detector's own output.

- **Span arithmetic, five classes, nine cases.** A greedy body running on into
  the next credential's prefix: `sec-0090`, `sec-0092`, `sec-0095`, `sec-0107`.
  Two credentials of one type reported as one finding over the run: `sec-0091`,
  `sec-0093`. Two JWTs joined producing three findings across the join:
  `sec-0094`. A decoy sharing the fixed-length AWS prefix producing a phantom
  second key: `sec-0098`. An Anthropic key whose own body contains `sk-` also
  reported as an OpenAI key: `sec-0099`. **None of these leaks.** In every one
  the credential is redacted in full and the audit record is what is wrong,
  which is why they are scored rather than described: a corpus that counted
  coverage instead of findings would report all nine as successes.
- **Complete misses, five cases.** `github_pat_` fine-grained tokens
  (`sec-0020`, `sec-0109`), `xapp-` Slack app-level tokens (`sec-0021`) and
  `xoxe-` Slack refresh tokens (`sec-0110`) are shapes the pattern table has
  none of, so each allows. They are the same category as the
  `injection-structural` corpus's balanced overrides: outside the boundary
  rather than failures inside it. `sec-0022` is a JWT whose header is 4137
  characters, past the 4096 bound, and a token past any of the three bounds
  matches nothing at all rather than matching short.
- **False positives, three cases.** `sec-0034`, `sec-0158` and `sec-0159` each
  mention `-----BEGIN PRIVATE KEY-----` in a sentence with no key behind it, and
  the private-key walk claims the header on its own. They are labelled `allow`
  and cost precision. The behaviour is kept rather than fixed because the
  alternative fails open, and that argument is in `docs/conformance.md`.

**There is no third-party corpus for this check.** The screen and what it found
are under [What is deliberately absent, and
why](#what-is-deliberately-absent-and-why).

### `corpora/url-exfiltration/in-repo.jsonl`

Written for this repository and covered by its Apache-2.0 licence. 94 cases, 39
positives and 55 negatives, in both directions.

Every attacker value in it is invented and every attacker host uses a label RFC
2606 reserves for documentation: `attacker.example`, `evil.example`,
`exfil.example`, `collector.example`, `redir.example`. The payloads are three
invented sentences about an invented conversation. No real person, no real
account and no real credential is in the file.

**The negatives could not be invented, and are not.** A precision figure for this
check is a statement about ordinary web content, so the negatives are the URL
shapes ordinary web content is made of: this repository's own README and
conformance links, image CDN parameters (`w`, `q`, `fm`, `fit`, width, height,
format, quality, signature, `hmac`, `exp`), Cloudinary transformation path
segments, two shields.io badges, search URLs on three engines, two OAuth
authorization links whose `redirect_uri` is a percent-encoded absolute URL,
YouTube watch and share links, UTM tracking parameters, a real 1x1 base64 PNG, a
real base64 JPEG, a real base64 GIF, two benign SVG data URIs, `mailto:` and
`tel:`, a signed CDN asset path, a UUID, a git SHA, an MD5 avatar hash and a
Mapbox-shaped access token. The hosts of those are the real ones, because a
negative that is not the shape people really write is a negative that proves
nothing.

**Six cases were added after a whole-branch review found two fail-opens that
the first 88 could not see.** Four are positives and two are negatives, and the
negatives are the half that binds a port in both directions:

- `url-0089`, `url-0090` and `url-0091` are the query that no path precedes.
  `https://host?d=...` is a URL RFC 3986 permits and every browser resolves to
  `https://host/?d=...`, and this implementation used to read its query only
  after a `/`, so the whole query string went into the authority and was
  discarded. Three of the five signals read nothing but the query, so deleting
  one character from an attacker's URL turned a deny into an allow. Not one of
  the original 88 cases put a `?` before the first `/`. `url-0091` is the
  negative: a changelog link with three UTM parameters and no path, which is
  the shape half the tracking links on the web have.
- `url-0092`, `url-0093` and `url-0094` are the laundered URL. This check
  resolves HTML entity references before it reads a scheme, because the
  consumer's parser does, and it used to do that for the scheme ALONE: the
  data-URI test and the query parse were handed the raw string one line later.
  So `dat&#97;:text/html,<script>` was allowed where `data:text/html,<script>`
  was denied, and `p.png&#63;d=<payload>` had no query here and a query in
  every renderer. The corpus had exactly one entity-laundered case, `url-0019`,
  and it pinned the normalisation for `SCRIPT_SCHEME` and for nothing else.
  `url-0094` is the negative, and it is the commonest HTML there is: a link
  whose query separators are written `&amp;`, which must stay allowed and must
  not have its payload split into pieces by the resolution either.

**The disclosed misses.** 0.923 precision, 0.923 recall, 6 wrong decisions over
94 cases. All six are below, and there are no other failures. They are the same
six the 88-case corpus disclosed: the six new cases are all answered correctly,
which is why the row moved from 0.914 / 0.914 and the wrong-decision count did
not move at all.

Three cost precision:

- `url-0083` and `url-0084` are a share intent and a prefilled GitHub issue,
  carrying 206 and 263 characters of ordinary prose in a query string. Both are
  labelled `allow` and both are denied. `LINK_QUERY_PAYLOAD` cannot separate
  them from a payload, because there is nothing to separate: the two populations
  overlap in length and in content.
- `url-0076` is a charting API called with a `title` parameter, 78 characters of
  prose in an image URL, labelled `allow` and denied. The argument for the image
  signal is that an image request does not need to say anything. A charting
  endpoint is where that argument stops being true.

Three cost recall, and each is a residual named in `docs/conformance.md` as
well:

- `url-0078` carries the payload as a hex DNS label and `url-0079` as a
  hyphenated sentence in a subdomain. **DNS-label exfiltration is not detected**,
  and neither is prose in a hostname. Hostname labels that decode to text are too
  close to ordinary hostnames to defend a number.
- `url-0080` is base64 of percent-encoded prose. **A doubly encoded payload
  passes**, because decoded text is never fed back to the decoder.

**Two more residuals carry no failing case, because they are allowed by design
and the corpus says so.** A `redirect_uri` whose inner absolute URL is plainly
percent-encoded does not fire `NESTED_REDIRECT`: `url-0046`, `url-0047` and
`url-0081` are labelled `allow` and are allowed, and closing that would deny
every OAuth link there is. And a search link longer than the 136-character floor
would fire: `url-0088` is a 135-character search query that passes by one
character, which is what a floor fitted to a corpus looks like from the inside.

**The floors were swept, and one sweep result was rejected.** Each of the four
decode floors and both prose floors was swept from 1 to 259 in steps of one over
this corpus, one at a time with everything else at its shipped setting. The two
prose floors are the smallest value reaching the best F1: 30 for an image query
(29 costs precision, 65 costs recall) and 136 for a link query (135 costs
precision, 177 costs recall). Both plateaus are where they were on the 88-case
corpus. The four decode floors are NOT the sweep's argmax and the module says
why: their curves are flat from 1 up to a ceiling, so the sweep bounds them and
does not choose them.

The rejected result is the percent floor, and the six new cases settled it.
On the 88-case corpus F1 rose from 0.9143 to 0.9275 anywhere between 107 and
147, by refusing to read a 106-character `title` parameter in one case and
shutting again one past a 147-character run in another. A window bounded at both
ends by two strings in one corpus is a value fitted to that corpus, so 6 shipped
instead. Two of the new cases carry percent runs of 104 and 119 characters and
both are positives, so on this corpus the window peaks at 0.9211 against the
shipped floor's 0.9231 and no longer wins at all. The rejected number was the
overfit, and the only reason that can be said is that the refusal was written
down instead of argued.

**Rot13 ships, and here is the measurement it shipped on.** Every alphabetic run
is a rot13 candidate, so the test is two-sided: the original run must fail the
prose test and the rotated run must pass it. Removing rot13 with nothing else
changed moves this corpus from 0.9231 / 0.9231 / 6 wrong to 0.9189 precision,
0.8718 recall and 8 wrong. Two positives are lost, `url-0071` and `url-0072`, and
the false-positive count does not move: 3 either way, over 55 negatives that are
almost entirely ordinary English prose and therefore rot13 candidates every one.

**The function-word list behind the prose test is a substitute, and it is named
as one.** It was to be derived from the external evaluation corpus in
`training/`, which loads out of `data/` and is gitignored, so a clone does not
have it. It is derived instead from `training/generated/rows.jsonl`, which is
tracked: the forty commonest words of its 112,473 tokens. Six of the forty are
content words of that corpus's own subject matter, and they are kept rather than
removed by hand, because "the forty commonest, minus the ones I did not like" is
an enumeration.

**There is no third-party corpus for this check.** No compatibly-licensed
labelled corpus of URL exfiltration was found, so these numbers are measured on
our own file only and are self-graded in the same way the injection-structural,
rules and secrets numbers are.
### `corpora/script-constraint/in-repo.jsonl`

Written for this repository and covered by its Apache-2.0 licence. It holds no
credential and no person: the values are ordinary sentences, and the one thing
in it that could be mistaken for a target is `pаypal.example`, which uses the
`example` label reserved by RFC 2606 and carries a Cyrillic letter where a
reader expects a Latin one.

Like the injection corpus, its negatives could not be invented. The Japanese
sentences, the Devanagari, Arabic, Hebrew, Thai, Armenian, Georgian, Ethiopic,
Tibetan, Khmer, Bengali, Tamil, Coptic and Mongolian fragments, and the emoji
sequences are text somebody really writes, because the point of a negative here
is exactly that. What IS invented is where they sit: each foreign-script run was
placed inside a Latin or Japanese sentence written for this file.

Every span in the file was computed from a substring of its own case text rather
than counted by hand, and the whole file is written as `\uXXXX` escapes, so a
reviewer reading the diff sees the same code points the loader decodes. That
matters more here than anywhere else in `corpora/`: half of these cases turn on
a single character that renders identically to a Latin one, or on a combining
mark that renders as part of the letter before it.

**The disclosed misses.** There are none, and the reason is worth stating rather
than celebrating. This check is a mechanical property of a table: a code point
either resolves to an allowed script or it does not, and there is no threshold,
no heuristic and nothing to tune. A perfect score therefore says the corpus
agrees with the table, not that the check is good, and the numbers that carry
information about it are the two measured in `docs/conformance.md`: 33 of its 42
`allow` cases stop allowing if `Common` and `Inherited` stop passing, and 186
code points stop being denied if Script is resolved in place of
Script_Extensions.

**What this file cannot measure at all** is any other value of
`allowed_scripts`. The row is the check under
`{"Latin", "Hiragana", "Katakana", "Han"}` and nothing else, and a deployment
that names a different set is running a check whose false-positive population
this corpus has never seen. `docs/conformance.md` prints the fixture beside the
section for that reason.

**There is no third-party corpus for this check**, and there could not be a
useful one: the labels depend on which scripts are allowed, so a corpus somebody
else wrote would be labelled against their constraint. The row is self-graded in
the same way the secrets row is.
### `corpora/confusables/in-repo.jsonl`

Written for this repository and covered by its Apache-2.0 licence. 115 cases,
54 labelled `deny` and 61 labelled `allow`, in both directions.

**The negatives are the point of this corpus and they outnumber the positives.**
A confusables check that keeps one half of either of its two rules passes almost
every input anybody would think to type and denies a language, so the corpus is
built to make that visible: Russian, Ukrainian, Serbian, Bulgarian and Greek
prose carrying Latin brand names and case endings; Japanese, Chinese and Korean
text carrying Latin; genuine Cyrillic domains including `почта.рф`,
`президент.рф` and `мвд.рф`; mathematics mixing Greek and Latin; two
transliteration tables, which are the shape that puts both alphabets on one
line on purpose; and ordinary code, SQL and shell. `cnf-0108` is there for a
reason nothing else in the file covers: `3カ月間` is Han beside Katakana, its
majority is Han, and the one Katakana character folds to a Han character inside
the identifier profile, so the UTS #39 Highly Restrictive table is the only
thing that allows it. That was found by mutating the check, not by reading it.

Every value in it is invented or public. The spoofed hostnames use the `example`
labels reserved by RFC 2606 or the brand's own name misspelled in another
script, and no credential, address or personal name appears in any case.

**Every non-ASCII character is written as a `\uXXXX` escape in the file.** That
is the same rule the injection corpus follows and it is stronger here: the whole
subject of this corpus is characters that are invisible as differences, so a raw
diff of a case would show a reviewer two identical-looking lines.

**Six cases carry a laundered spoof**, and they are here because a claim this
file made about the other seven was false. `cnf-0110` puts a soft hyphen on both
sides of the substituted letter, `cnf-0111` a variation selector, `cnf-0112` a
left-to-right mark inside a URL host label and `cnf-0113` a zero-width space
inside an email domain. `cnf-0114` and `cnf-0115` are the negatives that keep
the rule honest: hyphenated prose carrying soft hyphens, and three keycap
emoji, neither of which may become a finding. See the correction under the
misses below.

**The disclosed misses, by name.** Six cases labelled `deny` are allowed here.

- `cnf-0045` and `cnf-0046`: a whole-script spoof outside a URL, an email domain
  or a handle. `cnf-0045` is a spoofed hostname written with no scheme;
  `cnf-0046` is the same word in a sentence. A bare dotted string is not read as
  a host here, because a dot between two words is a missing space after a full
  stop at least as often as it is a hostname, and reading it as one would put
  every such sentence in front of the whole-script rule.
- `cnf-0047` and `cnf-0048`: a spoof whose substituted letter is outside the
  identifier profile. Cyrillic en folds to U+029C and Cyrillic te to U+1D1B, so
  the token does not read as Latin and the second condition fails. This is the
  standing cost of the condition that keeps `iPhoneом` and `.рф` out of the
  false positives.
- `cnf-0044` and `cnf-0049`: a token with no majority. Two Cyrillic letters and
  two Latin ones tie, the tie-break takes the first code point's script, and the
  Latin half becomes the minority.
`cnf-0050` WAS A SEVENTH AND IS NOT ANY MORE, AND THE ENTRY THAT STOOD HERE FOR
IT WAS WRONG IN THE WAY THAT MATTERS MOST. It read: "a spoof laundered with a
zero-width space, which ends the token and leaves two single-script tokens.
`injection-structural` reports that code point and denies by default, so a chain
running both still denies. This check alone does not, and the disjointness that
makes the two checks partition the code space is what costs it."

A reader was being pointed at a compensating control that does not exist, which
is worse than the miss it was excusing. Measured on this repository, in two
independent ways:

- The two sets are DISJOINT, not a partition. This check treated 4,174
  default-ignorable code points as token boundaries; `injection-structural`
  reports 3,773 of them. 401 -- all 260 variation selectors, U+00AD, U+061C,
  U+200E, U+200F, the bidi embeddings and isolates, and the 128 tag characters --
  split a spoofed token here and were reported by nothing.
- Even inside the 3,773, that check reports a code point only once five of them
  are present or two are adjacent. One zero-width space is under both bounds.

So `https://p<U+00AD>а<U+00AD>ypal.com/login` was ALLOWED by a chain running
both checks, and it is the same pixels as the string that chain denies. The
token rule is corrected, `cnf-0050` is an ordinary positive, and `cnf-0110`
through `cnf-0113` carry the other laundering forms. The disjointness is gone
with it: a span from this check may now cover a code point
`injection-structural` also claims, and the chain's span merge collapses the two
into one region naming both types.

**The disclosed false positives, by name.** Three cases labelled `allow` are
denied here.

- `cnf-0058` and `cnf-0061`: a Latin brand with a Cyrillic case ending glued
  straight on, where every letter of the ending is in the identifier profile.
  `Androidа` and `Windowsа` deny; `iPhoneом` and `Photoshopом` do not, because
  Cyrillic em is outside the profile. The two ordinary spellings that separate
  the word, `iPhone-ом` and `iPhone'ом`, both pass, which is what makes this a
  narrow residual rather than a rule that denies Russian.
- `cnf-0075`: `hν`, the physics notation for photon energy. Greek nu folds to
  Latin `v`, which is in the profile, so a Latin `h` beside a Greek `ν` is one
  mixed token that reads as `hv`.

**There is no third-party corpus for this check.** The row is self-graded in the
same way the injection-structural and secrets rows are.

### `corpora/encoded-content/in-repo.jsonl`

Written for this repository and covered by its Apache-2.0 licence. 81 cases, 39
positives and 42 negatives, in both directions.

Every payload in it is invented and every encoded credential decodes to a body
carrying `EXAMPLEONLY`, `notarealtoken` or `notarealkey`, except the AWS one,
which decodes to Amazon's own published example key. The repository-wide
credential rule cannot see any of them, because a base64 blob is not a credential
to a scanner reading the file, and that is the entire premise of this check; so
the same rule is applied to what the blobs DECODE to, by
`test_every_encoded_credential_in_the_corpus_decodes_to_a_marked_body` in
`tests/test_encoded_content.py`. The credential positives cover every type the
`secrets` check names, because `ENCODED_CREDENTIAL` inherits that check's
patterns and an inheritance measured on a subset of them is a claim about the
subset; `test_the_corpus_carries_an_encoded_form_of_every_type_secrets_names`
derives the list from `SECRET_TYPES` rather than repeating it here.

**The negatives are the whole argument, and they could not be invented.** An
entropy score flags every high-entropy string in ordinary text, and the claim
this check makes is that decodability separates a payload from a signature where
entropy cannot. That claim is only worth what the negatives are: SHA-1, SHA-256
and MD5 digests, short git SHAs, UUIDs, JWTs, real base64 PNG, JPEG and GIF
bodies, a PEM certificate (base64 of DER, not text), a base64 woff2 font,
base64-encoded JSON, a base64 MIME part carrying an ordinary email body, an SSH
public key, a Mapbox-shaped access token, a long random API token, a bcrypt hash,
hex colours, a MAC address, an IPv6 address, ULIDs, release tags, a git binary
patch, an MD5 column in a CSV, a percent-encoded search query, an OAuth callback
with a percent-encoded inner URL, a Kubernetes Secret with base64 values, a Basic
auth header, paragraphs of ordinary English long enough to be rot13 candidates,
and encoded status notes that are prose and are not instructions.

Two of those paragraphs carry an imperative sentence IN THE CLEAR: `enc-0065`
says `Ignore the earlier draft that went out on Monday` and `enc-0066` says `Stop
the batch if the reconciliation fails twice in a row`, both labelled `allow`.
This check is about what is HIDDEN, and a plain English instruction is not
hidden; it is also every alphabetic run's worth of rot13 candidate, which is what
those two cases cost the rot13 reading if the two-sided test were ever dropped.

**The disclosed misses.** 1.000 precision, 0.875 recall, 5 wrong decisions over
81 cases. There are no false positives at all. All five wrong decisions are
misses, all five are below, and there are no other failures.

- `enc-0035` opens `Forget every instruction` and `enc-0039` opens `Scratch the
  earlier plan`, both labelled `deny` and both allowed. **Two imperative verbs
  are outside the derived lexicon**, and `forget` is the canonical injection
  verb. It clears the position floor at 40 sentence-initial occurrences in the
  attack rows and the morphology step removes it, because no present participle
  of it occurs anywhere in that corpus. Adding it by hand would make the list
  "the derived one, plus the ones I wanted", which is the enumeration this check
  refuses everywhere else, so the cost is paid in the published recall instead.
- `enc-0036` is base64 of base64 of an imperative sentence, labelled `deny` and
  allowed. **A doubly encoded payload passes**, because decoded text is never
  handed back to the candidate scan. One level is the rule.
- `enc-0037` is base32, labelled `deny` and allowed. **An encoding outside the
  four alphabets is not read.** Base32's alphabet is a subset of base64's, so
  every base32 run is already a base64 candidate; what stops it is that the bytes
  it decodes to under base64 are not UTF-8.
- `enc-0038` is base64 of fourteen Unicode tag characters and nothing else,
  labelled `deny` and allowed. **A run that decodes to NOTHING BUT structural
  characters does not decode at all**, because decoded text under 90% printable
  is refused as bytes that survived a decoder, and every character
  `ENCODED_MARKUP` reports is non-printable by `str.isprintable`. So the signal
  reaches a payload whose controls are at most a tenth of the text they hide in.
  `enc-0028` and `enc-0034` are that shape and both fire.

**Two further residuals carry no failing case, because they are allowed by design
and the corpus says so.** Decoded prose that is not imperative does not fire:
`enc-0055` is a base64 MIME part carrying an ordinary email body, and `enc-0080`
and `enc-0081` are encoded status notes about a migration and a ledger, all three
labelled `allow` and all three allowed, and closing that would deny every base64
MIME part there is. And the **marker half of `ENCODED_MARKUP` is not
here**: the type reads the three structural signals of `injection-structural` and
not a chat-template marker one encoding layer down, no corpus case labels one,
and the published per-type recall for `ENCODED_MARKUP` measures the structural
half only.

**The position floor was swept and it does choose its value.** The lexicon is
re-derived at every floor from 1 to 40 and the corpus scored at each, with
everything else at its shipped setting. 5 is the smallest value reaching the best
F1 and both sides of it cost a case: at 4 the lexicon takes in `work`, `access`,
`note`, `report`, `address` and `check`, which are nouns everywhere outside those
rows, and `enc-0080` stops being an ordinary status note; at 1 `enc-0081` goes
with it; at 7 `redirect` falls out at 6 occurrences and `enc-0006` is missed.
That is unlike the four decode floors, whose curves are flat and which the sweep
bounds rather than chooses.

| Floor | Lexicon | F1 | What moves |
|---:|---:|---:|---|
| 1 | 121 words | 0.9091 | `enc-0080` and `enc-0081` both deny |
| 2 to 4 | 72 to 45 | 0.9211 | `enc-0080` denies |
| 5 to 6 | 29 to 26 | 0.9333 | shipped |
| 7 | 21 words | 0.9189 | `enc-0006` is missed |

**The imperative lexicon is derived, and here is what the derivation costs.** It
is not a word list somebody wrote down. Take every sentence-initial word of every
attack row of `training/generated/rows.jsonl`, which is 1,792 rows, 3,526 tokens
and 363 distinct words; keep the 92 that occur at least 5 times; keep the 29 of
those for which the same corpus contains a present participle, which is a verb
test with the corpus as its own evidence rather than a dictionary this package
does not carry. The second step is what makes this a check on hidden
INSTRUCTIONS rather than on hidden PROSE: position alone ranks fourteen words
ahead of every verb it keeps, `please`, `the`, `can`, `could`, `you`, `to`,
`your`, `now`, `for`, `i`, `in`, `before`, `we` and `this`, because those are
simply how sentences start. It costs the five verbs the morphology test cannot
confirm, two of which are the misses above, and it keeps four whose use in these
rows is domestic rather than adversarial, `mix`, `concentrate`, `wear` and
`wait`, because "the derived list minus the ones off topic" is the same
enumeration wearing the other sign.

**Rot13 ships here too, and the ablation is stronger than the one next door.**
Removing rot13 from what this check reads, with nothing else changed, moves this
corpus from 1.000 / 0.875 / 5 wrong to 1.000 precision, 0.800 recall and 8 wrong.
Three positives are lost, `enc-0015`, `enc-0016` and `enc-0017`, and the
false-positive count does not move: 0 either way, over 42 negatives of which six
are paragraphs of ordinary English and therefore rot13 candidates every one. The
two-sided test in the decode helper is what buys that.

**There is no third-party corpus for this check.** No compatibly-licensed
labelled corpus of encoded payloads was found, so these numbers are measured on
our own file only and are self-graded in the same way the injection-structural,
rules, secrets and url-exfiltration numbers are.
### `corpora/template-integrity/in-repo.jsonl`

Written for this repository and covered by its Apache-2.0 licence. 152 cases,
102 positives and 50 negatives, in both directions.

**The whole file is written as `\uXXXX` escapes**, for the reason the
injection-structural corpus gives: half the positives here are laundered with
characters that render as nothing, and pasted literally a diff would show a
reviewer a marker that looks clean beside one that looks identical. What a
reviewer reads is what the loader decodes.
`tests/test_template_integrity.py::test_the_corpus_is_written_as_escapes_so_a_reviewer_reads_what_it_decodes`
holds that.

Every attacker value in it is invented and none of it is a credential or a
person. The markers are not invented and could not be: each of the 59 entries in
`src/jamjet_guardrails/detectors/_template_markers.py` carries one labelled case,
so the published recall is a guard on the table and a regeneration that dropped
half of it would move the row rather than pass quietly.
`tests/test_template_integrity.py::test_every_marker_in_the_table_is_labelled_in_the_corpus`
is that guard.

**The negatives are the shapes ordinary documents are made of**, and they decide
the precision: HTML with a strikethrough tag, C++ and Java generics, a .NET
`Web.config`, a Maven `pom.xml` fragment, a Spring bean definition, an XACML
policy, a Kubernetes manifest, a Docker Compose service with a `user:` key, a
psql connection dump, a bug report quoting a transcript as a blockquote, a
markdown heading reading `## System:`, prose with role words followed by colons
mid-line, a stack trace, a SQL statement, a CSV, German prose and Japanese prose.

**This corpus is a stress set for the classes this check is worst at, and its
precision should be read that way.** Fifteen of its fifty negatives are drawn
from the three populations that fire, and no sample of ordinary web traffic
looks like that. What the figure measures is how the check behaves where it is
weakest, which is the only place a published number is worth having.

**The disclosed misses.** 0.820 precision, 0.965 recall, 19 wrong decisions over
152 cases. 15 negatives cost precision and 4 positives cost recall, all nineteen
are below, and there are no others. `docs/conformance.md` groups the same
nineteen by class with a worked input each.

The fifteen that cost precision:

- `tpl-0138`, `tpl-0139`, `tpl-0140`, `tpl-0141` and `tpl-0142` are
  documentation quoting a marker, two of them inside a code fence and two in
  inline code. All five are labelled `allow` and all five are denied.
  **Documentation quoting a marker is the hard negative class**, and under the
  default it fires. Nothing in the content separates a document that quotes a
  delimiter from content that uses one.
- `tpl-0143`, `tpl-0144` and `tpl-0145` are ordinary developer prose carrying
  `<function-name>` or `<args-json-object>`, the two Qwen 2.5 placeholders the
  marker table already names as its weakest entries.
- `tpl-0147`, `tpl-0148`, `tpl-0149`, `tpl-0150` and `tpl-0151` are real
  elements of real formats whose names CONTAIN a role word: `<system.web>`,
  `<systemPropertyVariables>`, `<policyholder>`, `<Policy>` and
  `<systemd_unit>`. Containment is what catches `<systemPrompt>` and
  `<assistant_instructions>`, and these five are its price.
- `tpl-0146` is prose about air handling whose paragraph opens `System:` after a
  blank line, and `tpl-0152` is a bug report that pastes one transcript turn the
  same way. Both are labelled `allow` and both are denied.

The four that cost recall, each a residual named in `docs/conformance.md` as
well:

- `tpl-0099`, `tpl-0100` and `tpl-0101` carry an OpenChat, a Mistral v7 and a Yi
  delimiter. **A marker from a model not in the table is not detected**, and
  eight repositories is not the field.
- `tpl-0102` writes `System:` on the line directly after a sentence. **A
  role-prefix line that does not follow a blank line is not detected**, and an
  attacker who reads this sentence will delete a blank line. The restriction is
  what keeps the signal off every specification and every configuration block,
  and removing it costs more than it buys.

**The code-fence exemption was measured, not argued.** Turning
`exempt_code_fences` on moves this corpus from 0.820 precision and 0.965 recall
to 0.866 precision and 0.912 recall.
It buys 8 fewer false positives, all of them documentation, and it costs
6 true positives across `tpl-0096`, `tpl-0097` and `tpl-0098`, which are
injections wrapped in a fence by an attacker who read this paragraph. F1 moves
from 0.886 to 0.888, which is to say the option is a redistribution rather than
an improvement, and a deployment that turns it on is choosing which of the two
failures it would rather have.

**The two weak marker slots were measured too.** Removing `<function-name>` and
`<args-json-object>` from the matching table, with nothing else changed, moves
this corpus from 0.820 precision and 0.965 recall to 0.843 precision and 0.947
recall. They are kept, and the reason
is not the number: removing two strings from a generated table by name is a hand
list, which is the one thing this repository does not allow an exemption to be,
and the sign of the difference is set by how many developer-prose cases this
corpus carries against how many labelled markers the table has. The measurement
is published so the next person to argue about them argues with a number.
`tests/test_template_integrity.py::test_the_two_weak_marker_slots_cost_what_the_notice_publishes`
re-takes it.

**There is no third-party corpus for this check.** No compatibly-licensed
labelled corpus of chat-template injection was found, so these numbers are
measured on our own file only and are self-graded in the same way the
injection-structural, rules, secrets and url-exfiltration numbers are.

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
precision at a bound that was raised deliberately.** 0.972 precision, 0.873
recall, 8 wrong decisions over 154 cases. Eight cases fail, every one of them on
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

## Vendored Unicode data

This is the one third-party thing in this repository that is REDISTRIBUTED
rather than fetched, measured against or merely named, so it is the one with a
condition attached to the act of committing it. The five files under
`unicode-data/16.0.0/` are published by Unicode, Inc., are committed verbatim,
and travel inside the source distribution.

| File | Bytes | SHA-256 |
|---|---:|---|
| [`Scripts.txt`](https://www.unicode.org/Public/16.0.0/ucd/Scripts.txt) | 189,588 | `9e88f0a677df47311106340be8ede2ecdacd9c1c931831218d2be6d5508e0039` |
| [`ScriptExtensions.txt`](https://www.unicode.org/Public/16.0.0/ucd/ScriptExtensions.txt) | 20,576 | `049117ce26b9769fe2749b06eef51a50a89faef4a97764dd2d81daa715980700` |
| [`PropertyValueAliases.txt`](https://www.unicode.org/Public/16.0.0/ucd/PropertyValueAliases.txt) | 80,773 | `440fd3e5460b9bfe31da67b6f923992e1989d31fe2ed91e091c4b8f8e2620bf9` |
| [`confusables.txt`](https://www.unicode.org/Public/security/16.0.0/confusables.txt) | 722,509 | `95bd0aad6dced5ebc63436f459c06ab21a8d107cd842fb57f5c3a1e91bca8611` |
| [`IdentifierStatus.txt`](https://www.unicode.org/Public/security/16.0.0/IdentifierStatus.txt) | 48,622 | `c6108ca140e054b55a5b0378e7ebed8b1ef0e846251f6195361bc9af8ffc61b1` |

**No changes were made to any of them.** `scripts/generate_unicode_tables.py`
reads them and writes `src/jamjet_guardrails/_unicode/scripts.py`,
`src/jamjet_guardrails/_unicode/confusables.py` and
`src/jamjet_guardrails/_unicode/identifiers.py`, which are DERIVED from them:
the same property values re-encoded as Python literals, with ranges that carry
one value joined where the published file splits them by general category.
Those three modules ship in the wheel, so the wheel carries derived Unicode data
even though it carries none of the files above. That is why the distribution's
licence expression names `Unicode-3.0` and not only the two licences the
corpora carry.

Both directions are checked rather than asserted.
`tests/test_unicode.py::test_the_generated_modules_are_byte_identical_to_a_regeneration`
rebuilds all three modules from the committed files and requires the bytes to
match, and
`tests/test_unicode.py::test_each_generated_module_records_the_digest_of_every_file_it_read`
holds the digests above against the files on disk. A third test, skipped unless
`JAMJET_GUARDRAILS_NETWORK=1` is set and never set in CI, re-downloads all five
and compares them with what unicode.org publishes today.

**Why `IdentifierStatus.txt` is one of them.** It is the UTS #39 Identifier
Profile, and it is what decides whether a confusable prototype is evidence of a
spoof. Measured on these files, 140 of the 296 Cyrillic letters fold to a string
written wholly in Latin and only 104 of those fold to one written wholly in the
profile. Cyrillic small em folds to U+028D LATIN SMALL LETTER TURNED W and
Cyrillic small ef to U+0278 LATIN SMALL LETTER PHI: both Latin, and neither a
character any brand or hostname is written in. Without this file the
`confusables` check denies `iPhoneом` in Russian prose and every `.рф` domain
there is, which is a check that gets switched off rather than a check.

**Why they are vendored at all.** `unicodedata` exposes no Script property,
no confusables table and no Identifier_Status on any interpreter from 3.10 to
3.14, and the Unicode
version behind it runs from 13.0 to 16.0 across this project's CI matrix. A
check deriving script from `unicodedata.name()` prefixes would reach different
verdicts on different legs of one test suite, and a corpus label written on one
leg would be wrong on another. The pinned tables answer the same on every leg.
`docs/conformance.md` says that the table format is this implementation's means
and not the contract: a port matching verdicts on the corpora conforms with any
Unicode data source at any version.

### Unicode License v3

The licence requires this notice to appear with copies of the Data Files or in
associated documentation. It is reproduced here verbatim, from
<https://www.unicode.org/license.txt>. The data files themselves carry the
notice line "© 2024 Unicode®, Inc." and point at
<https://www.unicode.org/terms_of_use.html>.

> UNICODE LICENSE V3
>
> COPYRIGHT AND PERMISSION NOTICE
>
> Copyright © 1991-2026 Unicode, Inc.
>
> NOTICE TO USER: Carefully read the following legal agreement. BY
> DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
> SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
> TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
> DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.
>
> Permission is hereby granted, free of charge, to any person obtaining a
> copy of data files and any associated documentation (the "Data Files") or
> software and any associated documentation (the "Software") to deal in the
> Data Files or Software without restriction, including without limitation
> the rights to use, copy, modify, merge, publish, distribute, and/or sell
> copies of the Data Files or Software, and to permit persons to whom the
> Data Files or Software are furnished to do so, provided that either (a)
> this copyright and permission notice appear with all copies of the Data
> Files or Software, or (b) this copyright and permission notice appear in
> associated Documentation.
>
> THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
> KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
> THIRD PARTY RIGHTS.
>
> IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
> BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
> OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
> WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
> ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
> FILES OR SOFTWARE.
>
> Except as contained in this notice, the name of a copyright holder shall
> not be used in advertising or otherwise to promote the sale, use or other
> dealings in these Data Files or Software without prior written
> authorization of the copyright holder.

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

## Chat-template marker sources

`src/jamjet_guardrails/detectors/_template_markers.py` is a generated table of
the delimiter strings chat templates use to mark a turn or a role:
`<|im_start|>`, `[INST]`, `<<SYS>>`, `<start_of_turn>` and their kin. It is
built by `scripts/generate_template_markers.py` from the tokenizer
configuration of the repositories below, each pinned to a commit, and those raw
files are committed under `template-data/` so the table can be regenerated and
diffed with no network. They are in the sdist and not in the wheel, because the
sdist is the evidence.

The `template-integrity` check reads this table, and its published row is
measured on `corpora/template-integrity/in-repo.jsonl`, which carries one
labelled case per marker. This section existed before that row did, because
attribution is a condition of two of the licences below and a condition does not
wait for a number.

**What is redistributed, in plain terms.** The files under `template-data/` are
tokenizer configuration: JSON settings, the special-token names, and the Jinja
chat template, a few kilobytes each. No model weights, no model outputs, no
training data, and nothing a model produced. What the generated table extracts
from them is shorter still: 59 markers, each a bracketed delimiter of at most
sixty-four characters, recorded beside the repository, revision and SHA-256
they were read from.

| Source | Repository read | Revision | Licence |
|---|---|---|---|
| Llama 2 chat | `unsloth/llama-2-7b-chat` | `a6d63d7c9ac31fd7e6d31e66ee0d1c784a489fcf` | LLAMA 2 Community License |
| Llama 3 instruct | `NousResearch/Meta-Llama-3-8B-Instruct` | `53346005fb0ef11d3b6a83b12c895cca40156b6c` | Meta Llama 3 Community License |
| Qwen 2.5 instruct | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | Apache-2.0 |
| Mistral instruct | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | Apache-2.0 |
| Gemma 2 instruct | `unsloth/gemma-2-9b-it` | `fc7d4737cda11c3a19af2b722319e846670b4d89` | Gemma Terms of Use |
| Phi-3 instruct | `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` | MIT |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3` | `e815299b0bcbac849fa540c768ef21845365c9eb` | MIT (LICENSE-CODE) |
| GPT-2 | `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | MIT |
| HTML element index | `w3c/webref` | `f3b81966c45f34f62df20e7f8d6f66d5b5ba9279` | MIT |

### Three sources are gated, and the mirror is named

An anonymous request for any file in these three repositories answers HTTP 401
until a licence is accepted in a browser, so nothing in this repository can
fetch them and no CI job could reproduce a table built from them. Each is
recorded with the revision it was pinned at, and the markers come from a named
non-gated mirror of that model rather than from an unattributed copy.

| Gated repository | Revision | Read instead from |
|---|---|---|
| `meta-llama/Llama-2-7b-chat-hf` | `f5db02db724555f92da89c216ac04704f23d4590` | `unsloth/llama-2-7b-chat` |
| `meta-llama/Meta-Llama-3-8B-Instruct` | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` | `NousResearch/Meta-Llama-3-8B-Instruct` |
| `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | `unsloth/gemma-2-9b-it` |

The Llama 2 mirror declares `apache-2.0` on its Hub page. That declaration is
not accepted here: a mirror cannot relicense Meta's material, so the stricter
upstream licence is the one recorded above and the one whose notice is carried
below.

### The notices those licences ask for

Llama 2 is licensed under the LLAMA 2 Community License, Copyright (c) Meta
Platforms, Inc. All Rights Reserved. The agreement is at
<https://ai.meta.com/llama/license/> and the acceptable use policy at
<https://ai.meta.com/llama/use-policy/>.

Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright
(c) Meta Platforms, Inc. All Rights Reserved. The agreement is at
<https://llama.meta.com/llama3/license/> and the acceptable use policy at
<https://llama.meta.com/llama3/use-policy/>. Built with Meta Llama 3.

Gemma is provided under and subject to the Gemma Terms of Use found at
<https://ai.google.dev/gemma/terms>. The use restrictions those terms carry are
the Gemma Prohibited Use Policy at
<https://ai.google.dev/gemma/prohibited_use_policy>, and they travel with this
distribution to anyone who receives it.

Qwen 2.5 and Mistral are Apache-2.0. Phi-3, GPT-2 and the element index are
MIT. DeepSeek V3 ships two licences: `LICENSE-CODE` is MIT and covers the
repository's code and configuration, which is all that is read here, and
`LICENSE-MODEL` covers weights this repository never touches.

### What the table leaves out, and what that costs

Two populations are removed, each by a property rather than by a hand list,
because an exemption spelled as a list of strings is the exemption that becomes
the channel.

**Markers that are also HTML element names are excluded**, and today that is 2
HTML element names, `<s>` and `</s>`. Each is a sentence-boundary token rather
than a claim to a role, and each is also the strikethrough tag, so a check that
denied them would deny ordinary HTML in any document that uses one. The rule is
membership in the element index of the HTML Standard, pinned above, and not the
two strings themselves: the day a model adopts `<p>` or `<code>` as a
delimiter, the same rule removes that one too, and a hand-written pair would
not have. The excluded markers stay in the module under `EXCLUDED_AS_HTML` so
the cost is visible rather than deleted.

**Reserved vocabulary slots are dropped**, and today that is 1018 reserved
slots: `<|reserved_special_token_0|>`, `[control_8]`, `<unused12>` and the rest
of their blocks. A tokenizer allocates these to be named later, and Llama 3.1
did exactly that when it turned `<|reserved_special_token_2|>` into
`<|python_tag|>`. No chat template emits one. The rule is that the name ends in
a number, so a slot that is later given a real name stops matching it and
arrives in the table on the next regeneration.

**Two entries are weak and are named rather than quietly kept.**
`<function-name>` and `<args-json-object>` are written by the Qwen 2.5
tool-calling template into the system prompt it builds, as placeholders inside
a JSON example. They were read out of a real template and are kept for that
reason, and they are also the two entries most likely to occur in ordinary
developer prose. What they cost is now measured rather than suspected, and the
measurement is under
[`corpora/template-integrity/in-repo.jsonl`](#corporatemplate-integrityin-repojsonl)
above.

**A marker from a model not in the table is not in the table.** Nine
repositories is not the field. The check that consumes this will say so, and
the table grows by adding a source and a pinned revision, never by typing a
string into the module.

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

The screen was run again on 2026-09-03, against the standard `training/screen.py`
states: an allowlist of SPDX identifiers, everything else refused, and an
Apache tag downstream does not cure a share-alike or unverified upstream. Every
candidate failed, on one of four grounds: a licence outside the allowlist, an
access gate, real credentials in the rows, or not being a labelled corpus at
all. What was checked, and which ground each fell on:

| Candidate | Refused because |
|---|---|
| `mazen160/secrets-patterns-db` | CC-BY-SA-4.0 on the repository itself, which is share-alike and outside the allowlist. It is also a file of regular expressions rather than labelled text, so it could not be scored on either way |
| SecretBench (Basak et al., MSR 2023) and its FPSecretBench companion | the data deposit is tagged "Other (Open)", which is not an allowlisted identifier; access needs a signed data-protection agreement; and the rows are real secrets harvested from live public repositories, which this repository will not carry under any licence |
| PassFinder (ICSE 2022) | the annotated set was never published, by the authors' own statement, for the same live-credential reason |
| `bigcode/bigcode-pii-dataset` | no licence declared, gated, and its terms of use forbid sharing the dataset or any modified version. `bigcode/pii-annotated-toloka` is gated and its licence could not be read at all |
| `Podric/prowl-secrets-corpus` | CC-BY-NC-4.0, non-commercial |
| CommonLeak / TrustedFalseSecrets | CC-BY-NC-ND-4.0, non-commercial and no derivatives, and only its negative rows are published |
| `CyCraftAI/TraceSafe` | Apache-2.0 and synthetic, which clears both halves of the screen, and gated behind accepting access terms. A corpus a reader cannot fetch cannot reproduce a published number |
| `Samsung/CredData` | the repository is tagged Apache-2.0 and its own README says each file keeps the licence of the project it came from, across 297 upstream repositories, none of them verified against the allowlist. This is the shape `training/screen.py` was written about |
| `trufflesecurity/trufflehog` | AGPL-3.0, and it publishes no benchmark corpus |
| `Yelp/detect-secrets` and `gitleaks/gitleaks` test data | Apache-2.0 and MIT respectively, which pass, and both are a handful of regression fixtures rather than a corpus. Nothing here could carry a published precision figure |
| GitGuardian's public sample repository | no licence file at all, which the allowlist refuses by construction |
| NIST SARD and the Juliet suite | a US government work and so effectively unrestricted, and its hardcoded-credential cases are generic passwords rather than the issuer-prefixed shapes this check matches |

Two of those are worth separating, because they failed for opposite reasons and
between them describe the gap. `CyCraftAI/TraceSafe` is the right shape under
the right licence and cannot be fetched without accepting terms. `Yelp/detect-secrets`
can be fetched under a licence that passes and is not a corpus. Nothing found so
far is both.

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
