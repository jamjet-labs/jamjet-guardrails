# Conformance

What a second implementation of `jamjet-guardrails` has to match, and what the
numbers it publishes are measured on.

The bar is verdicts on the corpora, not code. Everything above
[Third-party corpora](#third-party-corpora) is contract: a port that disagrees
with any of it is a different library wearing the same name. Everything in
[What is deliberately unspecified](#what-is-deliberately-unspecified) is free,
including all of the machinery this implementation happens to use.

The corpora under `corpora/` are the fixture that decides. `Corpus.version`
identifies which corpus a published figure was measured on, and
`corpora/NOTICE.md` records where each one came from.

## Verdict fields

One guardrail is one function: `check(content: str, context: Context) -> Verdict`.

### `Context`

What is being checked and where it came from. Guardrails never mutate it.

| Field | Type | Domain |
|---|---|---|
| `direction` | `Direction` | `input` or `output` |
| `origin` | `Origin` | `user`, `retrieved`, `model` or `tool` |
| `metadata` | mapping of string to anything | free; empty by default |

### `Verdict`

One guardrail's decision about one piece of content.

| Field | Type | Domain |
|---|---|---|
| `decision` | `Decision` | `allow`, `redact` or `deny` |
| `content` | string or absent | the rewritten content; required on `redact`, ignored otherwise |
| `findings` | sequence of `Finding` | copied to an immutable sequence on construction; may be empty |
| `provenance` | `Provenance` | which kind of check decided |
| `saw` | string | the digest defined in [The saw hash](#the-saw-hash) |
| `error` | string or absent | set only on a verdict the chain synthesised because `check` raised |

### `Provenance`

Which kind of check decided, and what is needed to reproduce it.

| Field | Type | Domain |
|---|---|---|
| `kind` | `Kind` | `constraint` or `classifier` |
| `detector` | string | the detector's own name |
| `version` | string | the detector's own version |
| `model` | string or absent | absent by default |
| `revision` | string or absent | absent by default |
| `threshold` | number or absent | absent by default |

### `Finding`

One detection.

| Field | Type | Domain |
|---|---|---|
| `type` | string | the detector's own type name, such as `EMAIL` |
| `span` | pair of offsets or absent | half-open `[start, end)` into the string this verdict's `saw` hashes |
| `confidence` | number or absent | see the invariant below |

### `ChainResult`

The audit record of one chain run.

| Field | Type | Domain |
|---|---|---|
| `decision` | `Decision` | the combination of every verdict's decision |
| `content` | string | every guardrail's redactions merged into the input; always a real string, and **not safe to send on a `deny`** |
| `verdicts` | sequence of `Verdict` | copied to an immutable sequence on construction, in the order the guardrails ran |

### The invariants `Context` and `Verdict` enforce

`Context` and `Verdict` validate themselves on construction and refuse these
states. `Provenance` and `Finding` reject nothing, so a port may enforce the
span and offset conventions wherever it finds convenient.

A `Context` refuses:

- **A `direction` outside its domain**, and **an `origin` outside its domain**.
  A chain runs only the guardrails whose declared `directions` contain the
  context's, so an unrecognised `direction` matches none of them and the chain
  returns `allow` with no verdicts over any content at all. This is the mirror of
  the refusal a registry makes when a guardrail declares no direction it can run
  in: both are a check that is configured and silent, reached from the two ends.
  A static type catches a literal typo and not a value read from configuration,
  which is where a direction comes from.

A `Verdict` refuses:

- **An unrecognised `decision`.** Every other rule here is a positive match on a
  known decision, so an unrecognised one satisfies all of them, and a caller
  branching "deny, else redact, else allow" reads it as an allow.
- **A constraint finding carries no confidence, and a classifier finding always
  carries one.** A constraint matches or it does not, so a number attached to
  one describes nothing. A classifier without one has thrown away the only thing
  that makes its decision reviewable. This is checked per finding.
- **An unrecognised `kind` is refused**, whether or not the verdict carries any
  findings. Both rules above are positive matches on a known kind, so without
  this check a third kind would switch the confidence invariant off in silence.
- **`saw` must be 64 lowercase hex characters.**
- **A `redact` must supply `content`.** A rewrite that rewrote nothing tells the
  caller to forward an un-redacted string.
- **A verdict carrying an `error` must be a `deny`, and must carry no findings.**
  An error is not a detection, and a classifier finding would need a confidence
  that does not exist. Deny plus a reason.
- **`findings` is copied before it is validated**, not after. The checks above
  read the sequence more than once, so an iterator would be truthy on the first
  read, exhausted by the second, and stored empty: a verdict that had silently
  dropped every finding would pass every check.

## Combination order

`deny` > `redact` > `allow`.

Combining two decisions returns the more severe of the two. A chain starts at
`allow` and folds each verdict's decision into a running decision, so the result
is the most severe verdict any guardrail returned.

**No code path may weaken a decision.** Combination is the only operation the
running decision passes through, so a later guardrail cannot talk an earlier
`deny` back down to `allow`, and a guardrail that runs after a `deny` still has
its verdict recorded. A broken detector is recorded as a synthesised `deny`,
which is the most severe value there is, so failing closed and combining
restrictively are the same rule applied twice.

Two consequences worth stating, because both are the direction a port fails in
if it gets them wrong:

- **A chain holding no guardrails returns `allow` over anything.** That is why
  building one from a list of names refuses an empty list: a configuration that
  names no checks is a mistake, not a request to disable the library.
- **The corpora do not exercise this rule.** Scoring runs one guardrail against
  one case and compares that single verdict, so combination is specified here
  and measured nowhere. A port matching the corpora has not thereby checked it.

## Single-pass rewriting

A chain runs its guardrails in the order it was given them, skipping any whose
declared `directions` do not include the context's `direction`.

**Every guardrail inspects the content the chain was given.** No guardrail is
ever handed a string another guardrail has already rewritten. Every `Verdict` in
one run therefore carries the same `saw`, and every span in the run indexes into
that same string, which is the string the caller passed to `run`.

**Redactions are collected and applied once.** Each `redact` verdict contributes
the spans of its findings; the chain merges them all, exactly as overlapping
spans from two patterns inside one guardrail are merged, and rewrites the
original content in a single pass. A region claimed by more than one guardrail
becomes one replacement naming every type that claimed any part of it.

**A guardrail's own `Verdict.content` is not what the chain returns.** It is that
one guardrail's rewrite of the same input, correct on its own and what a caller
holding a single guardrail uses. `ChainResult.content` is the composed rewrite,
which only the chain can compute because only the chain sees every span.

Why it is specified this way, since the alternative is the obvious one and this
implementation shipped it first: rewriting one guardrail at a time lets a
redaction cut a match a later guardrail was about to make. A personal-data check
redacting a Luhn-valid digit run inside a Slack bot token splits the token, the
credential check then matches only the prefix, and the rest of the credential
survives into content the chain returns as `redact` with a finding naming the
credential. Measured at 4.87% of canonical-format Slack bot tokens with the
guardrails in one order and 0% in the other. **A rule whose safety depends on
the order two checks are configured in is the defect, not the ordering.** A port
that rewrites sequentially reproduces it.

Only a `redact` contributes to the rewrite. Content returned alongside an
`allow` or a `deny` is ignored.

**The corpora do not exercise this rule either.** Scoring calls one guardrail on
one case and compares that single verdict, so nothing measured here runs two
guardrails over the same string. Composition is specified in this section and
measured nowhere, exactly as combination is. That is how the defect above lived:
each guardrail was right about the text it was handed, and no score in this
repository was computed over a chain.

When `check` raises, the chain records a synthesised verdict and keeps going:

- `deny`, carrying an `error` and no findings.
- Provenance taken from the guardrail's own declared `kind`, `name` and
  `version`, so a `classifier` that dies is never recorded as a `constraint`.
- `saw` over the same content every other guardrail in the run inspected.

The `error` string **must not quote the content**. A detector's own exception
message is the natural place for the value it choked on, so this implementation
records the exception type and a fixed sentence and drops the message. The
wording is not fixed by this document; not echoing the content is.

**A conforming chain rebuilds the verdict rather than trusting the one it was
returned.** Returning without raising is not evidence that the return value is
honest, only that the call finished, and a `Verdict` a detector builds for
itself is a claim about what happened, not a record of it. The chain is the only
party positioned to grade that claim: it is the one that computed `saw` and
passed `content` to `check`, so it alone can tell an audit record that describes
this run from one that describes something else. A guardrail attesting to its
own provenance is marking its own homework, and the failure is silent everywhere
else -- a false record and a true one look identical in shape, and only the
party holding the original content and the original digest can tell them apart.

Grading is necessary and it is not sufficient. Checking a returned object and
then continuing to use that same object is a hole rather than a check, because
the object is the detector's and it stays the detector's: a `str` subclass whose
`__eq__` returns True passes any comparison and then reports something else
downstream, an object whose `__class__` attribute names `Verdict` passes a type
test and can answer a property differently on the read that validates it and the
read that records it, and an `int` subclass passes `0 <= start < end <=
len(content)` and then indexes as a negative number. A conforming implementation
therefore reads each field ONCE, checks the read strictly, and constructs a new
verdict from those reads. Nothing a detector returned reaches the audit record,
the combined decision or the rewrite AS THE OBJECT it arrived in -- every field
is copied into a value this implementation built.

One field is copied through ON PURPOSE rather than replaced: a finding's
`type`. A redaction placeholder has to name what claimed the region it
replaces, so the type a detector reports is the one detector-chosen string
this document requires in `ChainResult.content` at all -- `EMAIL` in
`[REDACTED:EMAIL]` is a finding's `type`, read straight out of the finding
that redacted it. What is fixed is its LENGTH, not its content: a `type`
longer than **200 characters** is a failed check, refused like any other
over-long caller string (`_ERROR_TYPE_LIMIT` in this implementation, shared
with the bound its own error messages already used), because unbounded it
stops being a label: a `redact` whose finding's type IS the content it
redacted reproduces that content, verbatim, inside a string this document
calls safe to forward, and a type of a few million characters inflates every
audit record downstream by the same amount for one finding. The bound does
not, and cannot, stop a SHORT type from equalling a short secret -- a real
type name is a short constant such as `INVISIBLE_TAG_CHARS`, and 200
characters is generous next to one.

Read once, then check:

- **The returned value is exactly a `Verdict`.** Not a subclass, not something
  that claims to be one. A `check` is free to return anything.
- **Its `decision` is exactly one of `allow`, `redact`, `deny`**, and a `redact`
  carries content.
- **`verdict.saw` equals the digest the chain itself computed** over the
  content it gave `check`.
- **`verdict.provenance.kind`, `.detector` and `.version` equal the
  guardrail's own declared `kind`, `name` and `version`** -- the identity the
  chain already knows the guardrail by, not anything the verdict claims about
  itself.
- **Every finding's span, on every decision and not only `redact`, is a pair of
  plain integers satisfying `0 <= start < end <= len(content)`.** A `None` span
  stays legal: a classifier finding carries no span at all, and that is not the
  same failure as one that fails the bound. A span that is not a pair, or not a
  pair of integers, is a failed check and not an exception: `(1, 2, 3)`,
  `("a", "b")` and `5` are shapes a chain must refuse without abandoning the
  run.
- **Every finding's `type` is no longer than 200 characters.** Not a claim
  about content, only about length -- see above.
- **`provenance.threshold` and a finding's `confidence`, where either is
  present, are finite.** Both are numbers a caller may threshold or sort
  against, and NaN and infinity are both a `float`, so a type check alone
  admits them. A value that walks through a comparison-based guard is this
  library's own hole once already: every comparison against NaN is false, so a
  bound spelled as one more comparison treats NaN as though it had passed.
- **`error` is not set.** It is the chain's field, and it says the chain
  synthesised this verdict.

Then rebuild: a fresh verdict whose `saw` is the chain's digest, whose
`provenance` carries the identity the chain holds for that guardrail, and whose
findings are fresh objects built from the checked reads. Where a value the chain
can verify and the detector's claim about it agree, the chain records its own.

A verdict that fails any check is replaced exactly as a raised exception is: a
synthesised `deny` carrying the guardrail's own DECLARED provenance -- never the
false one the verdict returned -- and an `error` naming which check failed. Not
raised, for the same reason a raising `check` does not abandon the run either:
raising would lose the audit record entirely, and a synthesised deny keeps both
the fail-closed decision and the evidence that a check misbehaved.

**An error message must not repeat a value the detector chose**: not a claimed
provenance, not a finding's type, not the class name of whatever was returned,
not a span. This is the rule about `error` never quoting the content, arriving
from the return side rather than from an exception's message. The values a
detector returns are chosen after it has seen the content, so a detector whose
class name or finding type IS the content writes the content into the audit
record through the message, and bounding the value only shortens the leak. Name
the clause that failed. The guardrail's own declared name may be named, bounded:
it is what tells a reader which check misbehaved, and interpolating an unbounded
one several times per message is how a 2,000,000-character name became a
4,000,073-character `error`.

Continuing is safe precisely because the decision cannot weaken. A later
guardrail may deny too, and its verdict belongs in the audit record.

**The identity is read once, when the chain is built.** A conforming chain reads
each guardrail's `name`, `version`, `kind` and `directions` a single time at
construction, checks them, and uses its own copies for every verdict afterwards.
Two failures close together here. A `name` read once to validate a claim and
again to stamp the record can differ between the two reads, and both reads look
right where they stand. And a `kind` this library does not know cannot be
stamped onto anything, so discovering it per verdict puts the failure inside the
chain's own fail-closed path: the synthesised deny is unbuildable and the whole
run is lost at the moment a detector misbehaved. A guardrail that cannot declare
a usable identity is refused when the chain is built, before any content has
been checked, with the same error a registry raises for a check that would not
check.

**No guardrail behaviour abandons a run.** A `redact` the chain cannot locate,
meaning no findings or a finding without a span, is the last shape that did. A
chain that rewrites from spans and is given none would report `redact` over a
string nothing rewrote, so it must not be allowed; a conforming chain refuses it
the way it refuses every other false account a guardrail gives of itself, with a
synthesised `deny` that keeps the run's audit record. An out-of-range or
malformed span is caught by the checks above, on every decision; a `redact`
carrying no content is a verdict the chain will not rebuild. **A caller must
still treat any exception out of a chain run as a deny.** There is no audit
record when a run is abandoned, which is acceptable only because nothing was
allowed through.

**A guardrail that cannot run is refused when the chain is built.** A guardrail
whose declared `directions` contain none the runtime can carry is inert: it is
skipped in every context, so it is configured, silent, and indistinguishable
from a working check in every artifact the chain produces. Alone it yields the
empty chain's output, and beside a live check it makes the chain quieter than
the configuration says. A conforming implementation refuses it at construction
rather than skipping it at run time, and refuses a declared `name` or `version`
long enough to be a payload rather than an identity, because both are copied
into the provenance of every verdict that guardrail produces. This
implementation's ceiling is 200 characters; the ceiling itself is not part of
the contract, and having one is.

## The saw hash

SHA-256 over the exact inspected string encoded UTF-8, rendered as
64 lowercase hex characters.

**Exact means exact**: no case folding, no stripping, no Unicode normalisation,
no trimming of trailing newlines. A chain replays from these hashes, so two
different pieces of content must never produce the same one.

Worked vectors, which a port can check itself against directly:

| Input | `saw` |
|---|---|
| the empty string | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `alice@example.com` | `ff8d9819fc0e12bf0d24892e45987e249a28dce836a85cad60e28eaaa8c6d976` |

## Corpus schema

A corpus is one JSONL file, one JSON object per line, blank lines ignored. Every
field is required and no other field is accepted: with all of them mandatory an
unrecognised key is either a misspelling of a required one or something the
schema does not understand, and a schema that ignores what it does not
understand cannot tell those apart.

| Field | Type | Meaning |
|---|---|---|
| `id` | non-empty string | unique within the file, and the sort key of the version digest |
| `text` | non-empty string | the content handed to `check` |
| `direction` | `input` or `output` | the `Direction` the case is scored under |
| `expect.decision` | `allow`, `redact` or `deny` | the decision the verdict must carry |
| `expect.findings` | list of objects | each with exactly `type` and `span` |
| `expect.findings[].type` | non-empty string | the finding type expected |
| `expect.findings[].span` | `[start, end]` or `null` | `null` matches on type alone, which is a weaker bar and has to be asked for out loud |
| `source` | non-empty string | one value per file; a file may not mix sources |
| `license` | non-empty string | the licence the case is redistributed under; an SPDX identifier by convention |

A span must be a two-element pair of non-negative integers covering at least one
character, and must not run past the end of its own `text`. A zero-width span is
a non-detection wearing a detection's clothes.

### The version rule

`Corpus.version` is the first 12 hex characters of a SHA-256 digest over every
case, sorted by `id`. Per case, in this order:

    id, text, direction, expect.decision, count(expect.findings),
    then for each finding in order: type, then span

Each value is encoded UTF-8 and emitted **length-prefixed** as
`<byte length>:<bytes>`. A span is rendered `start:end`, or the literal `none`
when it is `null`. The finding count is emitted so that each case's record is
self-delimiting.

Every one of those fields is an input to what gets measured, which is why they
are all in the digest. A field that changes the numbers while leaving the
version fixed makes a published baseline a lie, and the edit most worth making
dishonestly is the quietest one: changing an expectation to match whatever the
detector currently does is the cheapest way to turn a failing check green, and
under a digest of `id` and `text` alone it is invisible.

Three details a port has to match rather than infer:

- **Length prefixes, not separators.** A delimiter byte that can occur inside a
  field does not separate. With `id + NUL + text`, the cases `("a", "b\0c")` and
  `("a\0b", "c")` produce identical digests, so two different corpora would
  share one version and therefore one baseline entry.
- **`null` spans are rendered, not skipped.** Skipping would still move the
  version when a span is dropped, because the count keeps each record
  self-delimiting. What it loses is the ability to tell an `EMAIL` with no span
  followed by a finding of type `0:1` from an `EMAIL` at `(0, 1)` followed by a
  finding of type `0:1` with no span. Those are different expectations, scored
  differently.
- **Cases are sorted by `id`; findings are not sorted.** Reordering the file
  changes nothing. Reordering a case's findings is content like any other and
  moves the version.

`source` and `license` are **outside** the digest. The version identifies what
was measured, not where it came from, so re-attributing a corpus does not
invalidate a baseline whose numbers did not move.

### A worked example

This single-line corpus:

```json
{"id": "a1", "text": "mail alice@example.com", "direction": "output", "expect": {"decision": "redact", "findings": [{"type": "EMAIL", "span": [5, 22]}]}, "source": "in-repo", "license": "Apache-2.0"}
```

has version `97db329cfa16`.

### How a corpus is scored

Findings are matched greedily: each prediction takes the first compatible
expectation, and that expectation is then consumed. Greedy is not the optimal
pairing, and it is kept because of its direction. It can only ever report fewer
true positives than the truth, never more. A published number is allowed to
understate a detector; it is not allowed to overstate one.

Precision and recall count **findings only**. `expect.decision` is checked and a
mismatch is reported separately, but it moves neither ratio, so a detector that
reports the right findings and the wrong decision every time still scores 1.0
and 1.0. Anything gating on these numbers gates on the decision mismatch count
as well.

Scoring calls `check` directly with the case's `direction` and an `origin` of
`model`. A case carrying a direction the guardrail does not declare is refused
rather than dropped: a chain skips a guardrail outside its directions, so
scoring that pairing would publish a number for a run that never happens.

## The injection-structural constraint

`pii` and `secrets` are patterns over what a document says, and the sections above are enough to port them:
their type names are the labels their corpora use, and their bar is the score on those corpora. The
section on `secrets` below adds no requirement to that; it discloses which cases sit behind that check's
published score. This check
constrains how a document is ENCODED rather than what it says, and reproducing its numbers is not the same
as reproducing it. 33 of its 99 `allow` cases stop allowing when one of the exemptions or exclusions below
is switched off, and neither an exemption nor an exclusion is visible in a precision figure.

Its `kind` is `constraint`, so the invariant above applies unchanged: no finding
it produces carries a `confidence`. Its corpus is
`corpora/injection-structural/in-repo.jsonl`, and a corpus directory name is the
name the guardrail is built under, by the rule at the end of this document.

**Where the corpus carries a label the label decides; where it is silent, this
document does.** The two halves are not symmetric and the difference runs
through everything below. A case labelled `allow` binds a port in BOTH
directions: denying it is a false positive and a wrong decision, so it is
contract. A case labelled `deny` that this implementation allows binds a port in
ONE direction only -- a port that denies it matches the label better than this
implementation does, and conforms by scoring higher. And a property no case can
express is contract because it is written here, not because anything scores it;
the direction this check declares is the one such property, and it is stated
below rather than measured.

### Three finding types

`INVISIBLE_TAG_CHARS`, `BIDI_OVERRIDE` and `ZERO_WIDTH_SMUGGLING`. Each is a
label the corpus uses, so each is what a prediction is matched against by name.

### Spans count code points

Every span is half-open over CODE POINTS of the string `saw` hashes, as every
span in this document is. It is called out here because this is the check where
the three plausible units disagree in ordinary use: the Unicode tag block
begins at U+E0000, so every tag character is four UTF-8 bytes and two UTF-16
code units.

`inj-0001` is the worked vector. Its text is `Summarise this.` followed by the
six tag characters spelling `ignore`, and its expected span is `[15, 21]`. The
same run is `[15, 39]` counted in UTF-8 bytes and `[15, 27]` counted in UTF-16
code units. A port that indexes in either -- the natural choice in Java,
JavaScript, Go or Rust -- reports a span this corpus scores as a miss, and
redacts the wrong bytes.

### It runs on input and on output

`directions` holds both, so a chain runs this check in either direction. That is
a statement about coverage rather than about wiring: a model that emits tag
characters into its own output is smuggling to whatever reads that output next,
which in an agent chain is another model, and a check that looked only at input
could not see it.

It shipped input-only at 0.1.0 and widened at 0.2.0. **The corpora could not
have told a port either version.** Scoring calls `check` with each case's own
direction, so while every case carried an `input` direction a port declaring
output as well scored identically. The corpus now carries cases in both
directions, which measures that a port runs in both; what it still cannot
measure is a port that declares MORE directions than these two, and there are no
more to declare.

### The exemptions

Three, and each one is behaviour rather than implementation detail: a group of
`allow` cases turns on it, so a port that does not make it denies text somebody
wrote on purpose. Each list below was measured by disabling that one exemption
and scoring the corpus again, and is re-measured on every run by
`tests/test_conformance_doc.py::test_every_case_list_the_exemptions_publish_is_the_list_the_measurement_gives`
rather than left to go stale.

- **Balanced bidi controls are allowed.** The signal is IMBALANCE -- an
  initiator nothing closes, or a terminator that closes nothing -- and never the
  presence of a control. Reporting every control instead denies these, all
  labelled `allow`: `inj-0027`, `inj-0028`, `inj-0029`, `inj-0030`, `inj-0031`,
  `inj-0032`, `inj-0035`, `inj-0036`, `inj-0037`, `inj-0038`, `inj-0096`,
  `inj-0141`, `inj-0142` and `inj-0153`. They are balanced embeddings, overrides
  and isolates around Latin, digits, Hebrew and Arabic. Right-to-left text is
  written with these controls, so a check that reported them would report a
  language.

- **The three RGI subdivision flag sequences are allowed.** Unicode tag
  characters mirror ASCII invisibly and have exactly one legitimate use: the
  flags of England, Scotland and Wales, each written as U+1F3F4, the tag
  spelling of its subdivision code, and U+E007F CANCEL TAG. Unicode defines
  those three and no others, so the set is closed. Dropping the exemption denies
  these, all labelled `allow`: `inj-0002`, `inj-0015`, `inj-0016`, `inj-0017`,
  `inj-0018` and `inj-0150`, which carry all three flags singly and in a row. A
  check that denies the Scotland flag is a check that gets switched off.

- **The joiner exemption is contextual, by script.** ZWJ and ZWNJ are
  orthography in the scripts that write them and structure inside an emoji
  sequence, and nothing anywhere else, so what excuses one is its NEIGHBOURS and
  never its identity. Dropping the exemption denies these, all labelled `allow`:
  `inj-0055`, `inj-0089` and `inj-0151`, which are family emoji; `inj-0061` and
  `inj-0152`, which are Devanagari conjuncts; and `inj-0063`, `inj-0079`,
  `inj-0087` and `inj-0095`, which are Persian and Arabic, both written in the
  Arabic script. A port that exempts a joiner wherever it appears has exempted
  the attack along with the orthography; one that exempts it nowhere denies
  conjunct Devanagari, ZWNJ in the Arabic script, and every emoji ZWJ sequence.
  Those are what these nine carry, and they are narrower than the eight ranges
  this implementation declares.

  Every other joiner case in the corpus still allows with the exemption
  disabled, because it carries too few joiners to reach either bound. Those
  nine are what hold the rule.

### Where this implementation falls short of its own corpus

The recall figure in `BENCHMARKS.md` is below 1.0 on purpose, and a port reading
only the number cannot see which cases are behind it. Two shapes a reader would
otherwise assume closed, pulled out because they are the ones that surprise.

- **A balanced override still reorders, and is allowed.** Trojan Source written
  with a closed pair passes this check. `inj-0030`, `inj-0038` and `inj-0096`
  are labelled `allow` and they move as a set: nothing distinguishes them, so
  relabelling one means relabelling all three. A port that denies them fails the
  corpus, which makes this the one item here that binds a port in the ordinary
  direction.

- **Two invisible channels stay open.** A payload can be carried by the PRESENCE
  OR ABSENCE of a joiner rather than by a choice between two, which defeats a
  rule that asks for the symbol to change; and a bitstream can be spaced out to
  defeat a rule about periodicity. `inj-0097`, `inj-0098` and `inj-0099` are
  labelled `deny` and this implementation allows all three, so they cost it
  recall rather than hiding in prose. The first two are the presence-and-absence
  encoding, behind a Devanagari cover and between variation selectors; the third
  is the spaced-out one. `inj-0098` is 119 characters of variation selectors and
  joiners, with nothing on the page at all.

  These are the corpus's bar, not a licence: by the rule at the top of this
  section a port that denies them matches the label where this implementation
  does not. What a port is held to is the `allow` side, and the two exclusions
  those channels ride on are load-bearing there. Measured:

  - counting variation selectors denies `inj-0143`, `inj-0144` and `inj-0145`,
    all labelled `allow`: five keycaps, five emoji carrying U+FE0F, and five
    Japanese names taking variant glyphs.
  - dropping the exclusion for the directional format characters denies
    `inj-0037` and `inj-0146`, both labelled `allow`. `inj-0146` is a bilingual
    invoice carrying five directional MARKS, which is the case that exclusion
    reads as being for. `inj-0037` is twenty balanced isolate pairs and twenty
    balanced embedding pairs, and it is there because the bidi CONTROLS are
    default-ignorable too: this one exclusion is also what keeps the
    balanced-control exemption above from being undone by the zero-width signal,
    which is why that case is cited in both places. Naming only the invoice here
    describes the narrower mutation -- counting U+200E, U+200F and U+061C alone
    -- and this bullet does not make that one.

  `corpora/NOTICE.md` lists the families this check does not count, with one
  measured encoder for each. **No minimum cost for getting a payload past this
  check is published**, and the absence is deliberate: a minimum is a claim
  about every possible encoding, and a measurement only ever exhibits one.

## The secrets constraint

`secrets` matches credentials on their ISSUER PREFIX and scores no entropy at
all. That is the whole of its reach, and it is the first thing a port needs:
this row is reproduced by reproducing the pattern table, not by finding keys. A
credential family the table has no shape for is invisible to this check however
random it looks, and three such families are in the corpus below costing recall.

Its `kind` is `constraint`, so the invariant above applies unchanged: no finding
it produces carries a `confidence`. Its corpus is
`corpora/secrets/in-repo.jsonl`, and a corpus directory name is the name the
guardrail is built under, by the rule at the end of this document.

**The asymmetry the section above states applies here too.** A case labelled
`allow` binds a port in both directions, because denying it is a false positive
and a wrong decision, so it is contract. A case labelled `redact` that this
implementation allows binds a port in one direction only: a port that redacts it
matches the label better than this implementation does, and conforms by scoring
higher.

### Seven finding types

`ANTHROPIC_KEY`, `AWS_ACCESS_KEY`, `GITHUB_TOKEN`, `JWT`, `OPENAI_KEY`,
`PRIVATE_KEY` and `SLACK_TOKEN`. Each is a label the corpus uses, so each is
what a prediction is matched against by name.

### A span covers the credential, not the run it sits in

Every span is half-open over code points, as every span in this document is.
Two credentials with nothing between them are TWO findings over two spans, not
one finding over the run, and a credential butted against a PEM envelope is two
placeholders rather than one, because `_merge` collapses spans that overlap and
leaves spans that merely touch alone. `sec-0100` is the worked pair: an AWS
access key id joined directly to a `-----BEGIN PRIVATE KEY-----` block, labelled
`AWS_ACCESS_KEY` over the first twenty characters and `PRIVATE_KEY` over the
block, and this implementation produces exactly that.

That is what the corpus asks for everywhere. It is not what this implementation
always delivers, and the next subsection is the list.

### Where this implementation falls short of its own corpus

Neither published figure is 1.0 and the gap is deliberate. Every case behind it
sits in the corpus labelled with what SHOULD happen, so each one costs precision
or recall rather than scoring as a success. Eight classes, one input each. The
first five are span arithmetic and none of them leaks: the credential is
redacted in full and the AUDIT RECORD is what is wrong. The last three change
what reaches the caller.

- **A greedy body runs on into the next credential's prefix.** `GITHUB_TOKEN`
  and `OPENAI_KEY` bodies stop at the first character outside their character
  class, and where a second credential of the same family is joined directly on
  the end that character is a few positions INSIDE the second one's prefix. In
  `sec-0090`, `ghp_` and a 36-character body twice over with nothing between
  them, the corpus labels `[0, 40)` and `[40, 80)` and this implementation
  reports `[0, 43)` and `[40, 80)`: the first span carries three characters of
  the second token's prefix. `sec-0092` is the same defect on `OPENAI_KEY` at
  two characters, `sec-0095` on a same-shape decoy at three, and `sec-0107` is
  the widest of them, a `ghs_` token whose body swallows an AWS access key id
  joined to its end and reports `[0, 60)` where the corpus labels `[0, 40)`.

- **Two credentials of one type joined directly are reported as one finding.**
  Where the body class can contain the whole of the second credential, which is
  true of `SLACK_TOKEN`'s `[A-Za-z0-9-]` and of `ANTHROPIC_KEY`'s
  `[A-Za-z0-9\-_]`, the first match covers both and `_scan` drops the second as
  contained in one already kept. `sec-0091` is two `xoxb-` tokens with nothing
  between them, labelled `[0, 45)` and `[45, 90)` and reported as a single
  `SLACK_TOKEN` over `[0, 90)`; `sec-0093` is the same on `sk-ant-api03-`. Both
  redact correctly. What is lost is the COUNT: an audit record that says one
  credential leaked where two did.

- **Two JWTs joined directly produce three findings across the join.** A JWT's
  own payload segment begins `eyJ`, so a run of two tokens offers start
  positions the bounded three-segment pattern can satisfy in more ways than one.
  `sec-0094` labels `[0, 183)` and `[183, 366)` and this implementation reports
  `[0, 219)`, `[37, 322)` and `[183, 366)`.

- **A decoy sharing a fixed-length prefix produces a phantom key.**
  `AWS_ACCESS_KEY` is `AKIA` or `ASIA` plus exactly sixteen characters and has
  no word boundary on either side, deliberately, because a boundary anchor is
  what lets one character of padding hide a whole key. The cost is the other
  direction: in `sec-0098`, `AKIASHORTAKIAIOSFODNN7EXAMPLE`, the decoy's prefix
  plus the first eleven characters of the real key is itself a well-formed
  match, so this implementation reports two keys, `[0, 20)` and `[9, 29)`, where
  the corpus labels the one real key at `[9, 29)`.

- **An Anthropic key whose own body contains `sk-` is also reported as an
  OpenAI key.** `sec-0099`, `sk-ant-api03-sk-` plus 32 alphanumerics, is one
  credential and gets two findings, `ANTHROPIC_KEY` over `[4, 52)` and
  `OPENAI_KEY` over `[17, 52)`. They share the right edge, so they merge into
  one placeholder naming both types and nothing is under-redacted; the second
  finding is a false positive in the record and not in the output.

- **A credential family the table has no shape for is a complete miss.** Three
  are in the corpus, labelled `redact` so that they cost recall rather than
  passing as clean text: `github_pat_` fine-grained personal access tokens,
  which are GitHub's modern default (`sec-0020`, `sec-0109`), `xapp-` Slack
  app-level tokens (`sec-0021`), and `xoxe-` Slack refresh tokens (`sec-0110`).
  None of them is close enough to a pattern here to be caught by accident. A
  port that matches them scores higher than this implementation and conforms.

- **A JWT over the header bound is a complete miss, not a shorter match.** The
  three segments are bounded at 4096, 65536 and 8192 characters, which is what
  keeps the scan linear on adversarial input. A token past any of the three
  matches nothing at all. `sec-0022` carries a header of 4137 characters, past
  the 4096 bound and longer than the two-certificate `x5c` chain that bound was
  measured against, and it allows.

- **A PEM header in prose is redacted on its own.** The private-key walk anchors
  on the header and always claims it, so a sentence that merely names
  `-----BEGIN PRIVATE KEY-----` produces a `PRIVATE_KEY` finding over those 27
  characters and a `redact` decision. `sec-0034`, `sec-0158` and `sec-0159` are
  labelled `allow` and score as three of this corpus's false positives. Nothing
  is disclosed by the redaction, since the redacted text is the public marker
  itself, and the behaviour is kept rather than fixed because the alternative
  fails open: a header whose body this walk does not accept would then produce
  no finding and no signal at all.

`corpora/NOTICE.md` names every one of these cases by id and states what grew
when the corpus did.

## The rules constraint

`rules` is the only check here whose finding types are chosen by the caller, and
that changes what its published row means. The row is measured under one fixed
configuration, recorded below, and it measures the ENGINE: whether a span is
right, whether two rules claiming one stretch of text collapse into one
placeholder, whether a limit fires one character past its bound and not at it.
It is not a measurement of any rule a user writes, and a port reproducing it has
reproduced the engine and nothing about rule content.

Its `kind` is `constraint`, so no finding it produces carries a `confidence`.

**A user's rule may name a finding type a built-in check also reports.**
Finding types are the caller's; nothing here refuses `rules` a type that
collides with one PII, secrets or injection-structural already reports, an
`EMAIL` from `rules` beside the `EMAIL` from `pii`, for instance. The two
verdicts stay separable by detector regardless, because each guardrail's own
`Verdict` keeps its own findings and its own span; only where a redaction
merges their spans does the collision show, in one placeholder that names the
colliding type once rather than twice. This is not refused, because refusing
it needs knowledge this primitive does not have: it cannot see what else is
running beside it in a chain.

### The configuration the row was measured under

    patterns:
      TICKET_ID:      \bJIRA-\d{4,}\b
      INTERNAL_HOST:  \b[a-z0-9][a-z0-9-]*\.corp\.example\b
    banned:
      PROJECT_CODENAME: ["project bluebird"]
    limits:
      max_chars: 2000
    on_match: redact

A conforming implementation scores `corpora/rules/in-repo.jsonl` with these
options. `on_match` is `redact` rather than the registered default `deny`
because a deny never reaches the rewrite, so the spans would be published
without ever being applied and the corpus would grade nothing that matters
here.

**The published row exercises only one of the three limit kinds.** The
fixture sets `max_chars` and neither `max_bytes` nor `max_lines`, so the
byte-boundary and line-boundary branches of `_limit_spans` are never reached
by anything this row measures. A limit the row cannot exercise is a limit the
row does not cover, not a limit proven correct. `README.md` carries the same
disclosure beside the published figures.

### What is fixed and what is not

Fixed, because the corpus measures it:

- **Matching is search, not anchoring.** A pattern matches anywhere in the
  content, so `packages/media/` matches inside `foo/packages/media/bar`. A port
  that anchors implicitly fails cases labelled for the unanchored behaviour.
- **Every occurrence is a finding, except a match wholly contained within one
  already reported, which is dropped.** `aba` in `ababa` is two findings at
  (0, 3) and (2, 5), not one: overlapping matches that are not contained are
  both kept. `X.{0,4}` over `XabcX` is ONE finding spanning (0, 5), not two,
  because the second match, `X` alone at (4, 5), is wholly inside the first.
  Dropping a contained match cannot uncover a character: the container already
  covers every offset the contained match covered.
- **Banned substrings match case-insensitively**, over a case-folded view, and
  the span reported is the SOURCE span. Where folding changes a character's
  width the two differ, and the corpus carries a case that separates them.
- **Size limits count characters, bytes and lines**, and a limit fires one past
  its bound and not at it. The finding is `LENGTH_LIMIT` and its span runs from
  the first excess character to the end of the content. Several limits breached
  produce ONE finding, from the earliest breach.

Not fixed:

- **The type names.** They are the caller's, and here they are the fixture's.
- **Whether an implementation refuses a pattern that nests unbounded repeats.**
  This implementation does, at construction, and says in its own documentation
  that the guard catches the textbook shape and is not a proof. A port may
  refuse differently, or not at all, and still reach every verdict in the
  corpus.
- **What a size limit does on a `redact`.** This implementation truncates at the
  limit and writes the placeholder. Nothing in the corpus compares rewritten
  content, so a port may spell the truncation differently.

### There is no token limit

Counting tokens needs a tokenizer, this library carries none, and a limit that
approximately counts tokens is a limit nobody can reason about. A deployment
that needs a token ceiling derives a character ceiling from a ratio it measured
on its own traffic, which is a number it can check.

## The url-exfiltration constraint

`url-exfiltration` reads what a URL CARRIES rather than where it points, and that makes it the first
check here whose signals are not visible in the characters on the page: a query value that decodes to
a sentence and a query value that decodes to noise are the same shape until something decodes them.
Its corpus is `corpora/url-exfiltration/in-repo.jsonl`, and a corpus directory name is the name the
guardrail is built under, by the rule at the end of this document.

Its `kind` is `constraint`, so no finding it produces carries a `confidence`.

**Where the corpus carries a label the label decides; where it is silent, this document does.** The
asymmetry the `injection-structural` section states applies here unchanged: a case labelled `allow`
binds a port in both directions, a case labelled `deny` that this implementation allows binds it in
one, and a property no case can express is contract because it is written here.

### Five finding types

`DATA_URI_PAYLOAD`, `LINK_QUERY_PAYLOAD`, `MARKDOWN_IMAGE_EXFIL`, `NESTED_REDIRECT` and
`SCRIPT_SCHEME`. Each is a label the corpus uses, so each is what a prediction is matched against by
name.

### A span covers the construct where there is one, and the URL where there is not

This is the one place this check's spans differ from every other check in the package, and the reason
is the rewrite rather than the report. A markdown image or link is reported over the WHOLE construct,
from `![` or `[` through the closing parenthesis, because a span over the URL alone leaves `![alt]()`
standing in redacted content for a renderer to draw a broken image from. `url-0016` is the worked
vector: its text is `Click to continue: ` followed by a markdown link carrying a `javascript:`
scheme, and its expected span is `[19, 91]`, which runs from the opening bracket through the closing
parenthesis and does not stop at either end of the URL inside it.

Everywhere else the span is the URL. An HTML attribute is reported over the URL inside the quotes,
not over the tag, because a tag's span swallows attributes that have nothing to do with the finding:
`url-0006` spans `[31, 210]`, the value of one `src` inside a tag that continues with `width` and
`height`. A reference-style image is reported over the URL in its DEFINITION, which is the only place
the URL exists: `url-0007` spans `[49, 242]`, on the `[pixel]:` line and not on the `![status][pixel]`
that uses it. A bare URL is reported over the URL, trimmed of sentence punctuation.

Spans are half-open over code points, as every span in this document is.

### It runs on input and on output, and one type runs on output only

`directions` holds both. The default decision differs between them, which no other bundled check does:
`deny` on output and `redact` on input. The exfiltration happens when output carrying the URL reaches a
client that renders it, so output is the enforcement point; on input the same URL is an instrument in a
retrieved page or a question a user is asking about it, and redacting keeps the page and the question.
A port is free to choose differently, because the corpus labels the decision per case and the cases
carry both directions.

`LINK_QUERY_PAYLOAD` fires on OUTPUT ONLY, and that is contract rather than configuration. A long
prose query on a non-image link is an exfiltrated conversation when a model wrote it and is a search,
a share or a prefilled issue when a user or a page did. The corpus cannot express this on its own: it
would need the same text labelled two ways in two directions, and no case does that, so it is written
here.

### There are no exemptions, and this is what stands in their place

There is no host allowlist, no safe-domain list and no scheme allowlist. The only thing that excuses
a URL is the absence of the structural property, and `javascript:` and `vbscript:` are named as
SIGNALS rather than as the complement of a trusted set. So the list every other check's section fills
with exemptions is filled here with the two limits on what is looked at in the first place, because
those bound the check in the same way and would otherwise go unstated.

- **Bare URLs are found by scheme.** Any `scheme://` form is found, plus `data:`, `javascript:`,
  `vbscript:`, `mailto:` and `tel:`. A bare opaque scheme outside those five is not found in running
  text. The cost of closing that is a check that reports `Note:` and `Warning:` in prose, and a
  markdown or HTML construct is discovered whatever its scheme, so what this misses is text that no
  renderer links either.
- **The authority is not a component.** Only path segments and query keys and values are decoded and
  tested. Hostname labels are not, and two corpus cases pay for it; see below.

### Where this implementation falls short of its own corpus

The published row is 0.914 precision and 0.914 recall over 88 cases with 6 wrong decisions, and every
one of the six is named here. Three are false positives and three are false negatives, and they are
the same trade seen from both ends.

- **A benign link whose query really is a long piece of prose fires.** `url-0083` is a share intent
  carrying 206 characters of ordinary writing and `url-0084` is a prefilled issue body carrying 263,
  both labelled `allow` and both denied. The floor that separates a search query from a conversation
  is 136 characters, and it does not separate these: no floor does, because the two populations
  overlap. `url-0088`, a 135-character search query, passes by one character, which is what a floor
  fitted to a corpus looks like. `corpora/NOTICE.md` carries the sweep.
- **A legitimate caption in an image query fires.** `url-0076` is a chart API's `title` parameter,
  labelled `allow` and denied. An image request does not need to say anything, which is the whole
  argument for the signal, and a charting endpoint is the counterexample the argument does not survive.
- **Data in a hostname is not detected.** `url-0078` carries the payload as a hex DNS label and
  `url-0079` as a hyphenated sentence in a subdomain; both are labelled `deny` and both are allowed.
  Hostname labels that decode to text are too close to ordinary hostnames to defend a number, so this
  is disclosed rather than half-built.
- **A doubly encoded payload is not detected.** `url-0080` is base64 of percent-encoded prose,
  labelled `deny` and allowed. Decoded text is never fed back to the decoder: one level is the rule,
  and a decode loop would make every span a claim about a string the caller never had.

These six are the corpus's bar and not a licence. A port that denies `url-0078`, `url-0079` or
`url-0080` matches the label where this implementation does not; what a port is held to is the `allow`
side.
## The script-constraint constraint

`script-constraint` is the second check here whose behaviour is chosen by the
caller, and the only one with no default at all. It carries one option,
`allowed_scripts`, a collection of long Unicode script names, and it reports the
stretches of content that no allowed script covers. `build("script-constraint")`
with no options is refused: the two candidate defaults are a check that permits
every script and reports nothing, and a check that decides for a deployment
which languages are ordinary.

Its `kind` is `constraint`, so no finding it produces carries a `confidence`.
Its corpus is `corpora/script-constraint/in-repo.jsonl`.

### The configuration the row was measured under

    allowed_scripts: ["Han", "Hiragana", "Katakana", "Latin"]
    on_match: deny

A conforming implementation scores `corpora/script-constraint/in-repo.jsonl`
with these options. **The row measures the check under this configuration and
promises nothing about any other**: a different `allowed_scripts` is a different
check with a different false-positive population, and no number here transfers
to it. The set is Japanese plus Latin because Japanese is the smallest ordinary
text that needs three scripts in one sentence and carries punctuation whose
script resolution is not obvious, and because Cyrillic, Greek, Arabic and
Devanagari then sit outside it and can be located inside that text.

### One finding type

`DISALLOWED_SCRIPT`. A finding covers a MAXIMAL RUN of code points that no
allowed script covers, half-open over code points of the string `saw` hashes.

Maximal over disallowed-ness, never over one script: Greek followed immediately
by Cyrillic is ONE finding, because the fact reported is that a stretch of
content lies outside the constraint and not which script it is in. An allowed
code point between two disallowed runs separates them, so `Привет мир` is two
findings and the space between them belongs to neither.

### How a code point is resolved

Per UTS #39 section 5.1: Script_Extensions where the code point has them,
otherwise its Script value. A code point PASSES when any member of that resolved
set is allowed. Containment is the wrong test and would deny ordinary text: the
middle dot `U+00B7` resolves to sixteen scripts, and requiring all of them to be
allowed denies it under every constraint anybody would write.

Two rules sit on top of that, and both are contract:

- **`Common` and `Inherited` always pass.** Punctuation, digits, currency,
  mathematical symbols, whitespace, emoji, combining marks, ZWJ, ZWNJ and
  variation selectors resolve to one of the two, so they pass under every
  constraint. This is the whole false-positive defence: a `{"Latin"}` constraint
  that denied a comma would be switched off in its first week. Measured on this
  corpus: dropping the rule denies 33 of its 42 `allow` cases, held by
  `tests/test_script_constraint.py::test_the_document_publishes_the_number_of_allow_cases_the_wildcards_carry`.
- **`Unknown` never passes.** A code point the implementation's tables do not
  assign is a finding. That is fail-closed and it is disclosed below.

### Script_Extensions, and what resolving Script instead would cost

A port that resolved Script alone would look correct on Japanese text and be
wrong in one direction only. Measured over every code point any
Script_Extensions range covers, under the fixture above: 186 code points would
then pass that this implementation denies, and none would be denied that this
implementation passes. So the rule buys recall and never precision here, and a
port that skips it does not become noisier, it goes quiet. The figure is
recomputed on every run by
`tests/test_script_constraint.py::test_the_document_publishes_the_measured_cost_of_resolving_extensions`.

The two shapes behind it are in the corpus. The Katakana-Hiragana prolonged
sound mark `U+30FC` has Script `Common` and Script_Extensions
{Hiragana, Katakana}, so it passes under the fixture and FIRES under a
`{"Latin"}` constraint, which is the opposite of what a Script-only port
reports. The Devanagari danda `U+0964`, the Arabic comma `U+060C` and the Arabic
tatweel `U+0640` are the same fact seen from the other side: Script `Common`,
extensions naming no script the fixture allows, so they fire here and pass
there. `sc-0008`, `sc-0081` and `sc-0083` carry the second shape.

### It runs on input and on output

`directions` holds both. On input the check locates a language the deployment
does not handle, which is where a retrieved page carries a payload nobody on the
team can read; on output it locates a model answering in a script the caller
cannot render or moderate. The corpus carries cases in both directions.

### The exemptions

None by enumeration. There is no list of characters this check forgives and no
allowlist of anything. The two rules above derive from a Unicode property, the
resolution rule is UTS #39's own, and `allowed_scripts` is the caller's
statement rather than this project's.

### Where this implementation falls short of its own corpus

- **A code point assigned after the pinned tables fires.** The tables are
  16.0.0, so a code point assigned later resolves to `Unknown` and is denied
  even where it belongs to a script the caller allowed. Fail-closed and
  deliberate, and a pin bump is the fix rather than an exemption. A private-use
  code point behaves the same way, which is what `sc-0021` records.
- **A constraint that names one script of a language that needs two denies that
  language.** `{"Hiragana", "Katakana"}` without `Han` denies the kanji in every
  ordinary Japanese sentence, and `{"Latin"}` denies the Japanese punctuation in
  a bilingual page. Nothing here detects that, because the check cannot know
  which language the caller meant; naming the scripts a language is written in
  is the caller's job and the corpus does not do it for them.
- **A combining mark inside a disallowed run can split it.** The combining acute
  `U+0301` has Latin among its extensions, so under a constraint naming Latin it
  passes and `Приве́т` is reported as two findings rather than one. That is the
  maximal-run rule applied consistently, and its consequence on a `redact` is
  that the mark is left standing between two placeholders. `sc-0020` labels it.
- **A run is not a word, and this check does not read.** A single Cyrillic
  letter substituted into a Latin word is a one-character finding, and so is a
  Cyrillic letter that a Russian speaker typed on purpose. Telling those apart
  is `confusables`' question, not this one's.

## Third-party corpora

Precision and recall measured only on a corpus we wrote are self-graded: the
same hands chose the detector's behaviour and the labels it is graded against.
One corpus here comes from somebody else, and one deliberately does not.

### What shipped

| | |
|---|---|
| Dataset | **Nemotron-PII**, by Amy Steier, Andre Manoel, Alexa Haushalter and Maarten Van Segbroeck (NVIDIA Corporation) |
| URL | <https://huggingface.co/datasets/nvidia/Nemotron-PII> |
| SPDX | `CC-BY-4.0` |
| Revision | `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed` |
| Slice | `locale == "us"`, 300 rows sampled by the SHA-256 of each row's `uid` |
| Converter | `scripts/sample_nemotron.py` |

Attribution is a condition of that licence, not a courtesy. The table above is a
pointer. The full CC BY notice is maintained in one place only,
`corpora/NOTICE.md`, which carries the source file's own digest and the list of
changes made to it. `BENCHMARKS.md` names the dataset in the Source column of
every row measured on it and points at the same file.

It was chosen over better-annotated candidates for one reason: it is the only
one examined whose values are in enough different formats to produce an
informative number on all four PII types. A corpus whose values happen to sit in
exactly the formats a detector accepts reports close to 100% and proves nothing.
That failure mode has a shape worth naming, because it survives every check a
licence review makes: **a corpus shaped like the detector is the self-grading
problem wearing a third-party badge.** One rejected candidate scored 100%
precision and 100% recall on two types and 0% on a third, purely because its
cards were written without separators.

**There is no third-party secrets corpus.** No compatibly-licensed one was
found, so the secrets numbers are measured on our own corpus only. That is why
`BENCHMARKS.md` shows a third-party row for `pii` and not for `secrets`.

### The share-alike finding, and why a licence field does not catch it

Two otherwise-ideal PII corpora, `beki/privy` and `microsoft/presidio-research`,
**advertise MIT and cannot be redistributed from an Apache-2.0 repository.**
Their PII *values* derive from Fake Name Generator identities, and those are
copyleft. The chain, verified from primary sources:

1. <https://www.fakenamegenerator.com/license.php> states that Fake Name
   Generator identities are dual-licensed under the GPLv3 and Creative Commons
   Attribution-Share Alike 3.0 United States licences, and that either may be
   chosen but must be complied with fully.
2. `pixie-io/pixie`'s generator source
   (`src/datagen/pii/privy/privy/providers/english_us.py`) loads
   `FakeNameGenerator.com_1000_American.csv` from `presidio_evaluator`.
3. That feeds `beki/privy`, which declares MIT and whose card's Licensing
   Information section reads `[More Information Needed]`.
4. Which in turn feeds `gravitee-io/pii-detection-dataset`, which declares
   Apache-2.0.

**An Apache-2.0 tag downstream does not cure a share-alike upstream**, and no
licence field anywhere in that chain reveals it.

What does reveal it is a value fingerprint. Fake Name Generator issues email
addresses on 10 house domains and Faker does not, so their presence is
diagnostic: in the tainted slice 98.8% of emails carry one, and in everything
shipped here, zero.

    dayrep.com   armyspy.com   rhyta.com      cuvox.de       einrot.com
    fleckens.hu  gustr.com     jourrapide.com superrito.com  teleworm.us

**The fingerprint covers email domains and nothing else.** It is diagnostic for
the two datasets above, whose values are Fake Name Generator identities complete
with their addresses, and the research measured 98.8% of that slice's emails
carrying one. A corpus derived from the same identities with the emails stripped
or rewritten, keeping only names, street addresses and phone numbers, would pass
this screen while carrying the same share-alike values. The screen is evidence,
not proof, and a new candidate still needs its provenance read.

The failure direction is silence. Someone regenerating or extending a corpus
without the filter admits share-alike data into an Apache-2.0 repository, and
the only symptom is a licence violation nobody can see. So the fingerprint is
enforced as a test over the committed files
(`tests/test_corpora.py::test_no_corpus_carries_share_alike_values`) rather than
by remembering, and the converter refuses such a row as well.

### Labels, and the one that was remapped

The dataset's labels map onto this library's four types: `email` to `EMAIL`,
`ssn` to `US_SSN`, `credit_debit_card` to `CREDIT_CARD`, and both `phone_number`
and `fax_number` to `PHONE_NUMBER`. Everything else is dropped, so a value this
library matches under a dropped label counts as a false positive.

`fax_number` is mapped because a fax number is a telephone number and
`PHONE_NUMBER` is the only telephone type here, so redacting one is a redactor
doing its job rather than a defect. The decision is worth stating because it
moves a published figure: over the first 20,000 `us` rows the phone pattern makes
5,937 `PHONE_NUMBER` predictions, of which 4,612 land exactly on a
`phone_number` span and 1,173 on a `fax_number` span. That is per-type phone
precision of 0.974 with the mapping and 0.777 without it.

The counts are published beside the ratios because this one is the only figure
in this document that CI cannot re-derive: the parquet it is measured on is 151
MB and is not committed, so nothing in the suite recomputes it. Reproducing it
takes the pinned revision named above, `scripts/sample_nemotron.py`'s label map,
and the same `evaluate` the published tables use. Both ratios share the 5,937
denominator, so either can be checked against the other.

The identifier labels are **not** treated the same way. A `tax_id`, a
`medical_record_number` and an `account_number` are not Social Security numbers
and not payment cards, so a `US_SSN` or `CREDIT_CARD` finding on one of them
stays a false positive. That is the honest cost of a pattern with no issuer
check, and it is most of the gap between this corpus's per-type precision
figures and 1.000.

### Why a published precision figure has a shelf life

**One guard behind these numbers stops working on a date.** This
implementation's scan for a bare run of card digits requires a leading digit of
2 to 6, the Major Industry Identifier range payment cards are issued under, and
most of what that buys is the exclusion of epoch-millisecond timestamps, which
begin with a 1 today and appear in nearly every machine-written log line.

Epoch-milliseconds first carry a leading 2 at **2033-05-18T03:33:20Z**, the
instant the epoch second reaches 2,000,000,000. Epoch-microseconds cross on the
same date. After that boundary timestamps sit back inside the range, and roughly
one in ten of them survives the check digit; `corpora/NOTICE.md` records the
measurement.

A precision figure published from these corpora is therefore a statement about
this detector as it behaves **before that date**, and it is stated here rather
than left in a source comment because the number outlives the code it describes.

The guard is this implementation's choice and is not part of the contract. What
a port has to match is the verdicts on the corpora, and those are what change on
that date.

## What is deliberately unspecified

This section is the point of the document, not a disclaimer. **A conforming
implementation matches verdicts on the corpora. It does not match code.**

Nothing below is specified, and a port is free to make an entirely different
choice for every one of them:

- **How a detector decides.** Regular expressions, a parser, a table, a trie, a
  model, or anything else. This implementation's four PII types and its
  credential prefixes are one way of hitting the numbers, not the definition of
  them.
- **The pattern set.** Which patterns exist, how many there are, what they
  match, how they are anchored, whether validation such as a check digit is
  applied and where.
- **Model choice, weights, revision and thresholds** for anything of `kind`
  `classifier`. `Provenance` has fields to record them precisely so that they
  can vary; a port that records different values there is behaving correctly.
- **Confidence values.** The invariant is that a `classifier` finding has one
  and a `constraint` finding does not. What number a classifier reports is its
  own business.
- **The redaction placeholder.** Scoring compares decisions and findings and
  never compares the rewritten `content`, so the text a redaction substitutes is
  not fixed by this document or measured by the corpora.
- **Whether overlapping matches are merged before rewriting**, and how a merged
  region is labelled. What is fixed is the findings: one per detection, each
  with its own span.
- **Detector names, versions and type names**, with two exceptions that scoring
  forces. Corpora are discovered as `<check>/<source>.jsonl` and the `<check>`
  directory name is the name the guardrail is built under, so the two have to
  agree. And a finding `type` has to match the label the corpus uses, since that
  is what a prediction is matched against.
- **Error text, its wording only.** A synthesised verdict must deny, must carry
  no findings, and must not quote the content; those three are fixed above. What
  sentence a port writes around the exception type is nobody's contract, and
  this implementation does not require it to be non-empty.
- **`injection-structural`'s thresholds, its derived character set, and how far
  back its context walks look.** These are the first bullet applied to one
  check, and they are named separately because they read like a specification
  in a way a regular expression does not: two integers and a set of code points
  look like something to copy. They are a point on a sweep and a derivation
  from a Unicode property, each recorded in the source with what it costs on
  both sides. Any machinery reaching the same verdicts on the corpus conforms,
  and so does any evidence for it -- the mutation battery `corpora/NOTICE.md`
  describes is this implementation's argument that its corpus moves when its
  detector does, not a requirement on a port.
- **Where Unicode property data comes from, and at what version.** This
  implementation vendors Script, Script_Extensions and confusables tables
  generated from files published by Unicode, Inc. at 16.0.0, because
  `unicodedata` exposes neither property on any supported interpreter and its
  own Unicode version varies from 13.0 to 16.0 across the CI matrix. The table
  format, the generator and the pin are this implementation's means. A port
  reaching the same verdicts on the corpora conforms with any Unicode data
  source at any version, and each corpus records the version its labels were
  written against.
- **The fold machinery.** How a check that matches over a transformed view of
  the content maps a match back to a span in the original is not specified.
  What is specified is the span itself: it indexes the content the chain was
  given, which is the string `saw` hashes, whatever view the match was found
  in.
- **How a run of characters is decided to decode to text, and the floors that
  decide it.** `url-exfiltration` reads a URL component through four alphabets
  and a function-word list, at floors recorded in the source with the sweep that
  bounded each. Those are this implementation's means: the function-word list is
  derived from a file in this repository and is not a specification of English,
  and a port reaching the same verdicts with a different list, different ratios
  or a different set of alphabets conforms. What is NOT free is the one-level
  rule, which is behaviour: `url-0080` is labelled `deny`, this implementation
  allows it, and a port that decodes twice matches the label where this one does
  not, which the section above says is conforming in the direction that binds
  nobody.
- **Everything about performance**, threading, and how a port lays its modules
  out.

What is not free is everything above [Third-party corpora](#third-party-corpora):
the fields and their domains, the `Verdict` invariants, the combination order,
the single-pass rewriting rule, the `saw` digest, the corpus schema and its
version rule, the `injection-structural` types, spans, direction and exemptions,
the `url-exfiltration` types, spans, directions and its output-only type, and the
scores on the corpora.
