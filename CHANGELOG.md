# Changelog

Notable changes to this package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published precision or recall figure is part of the interface here. A release
that moves one says so in its entry, with the old value and the new one, because
a number that changes quietly is a number nobody can rely on.

## [Unreleased]

### Added

- Both adapter READMEs now say that the framework they install reaches the
  network, and name the switch that stops it. The core's front page promises
  "No dependencies. No network calls. No model downloads", which is true of the
  core and stops being true of a reader's environment the moment they install an
  adapter. `nemoguardrails` 0.24.0 posts usage statistics to an NVIDIA endpoint
  by default, off with `NEMO_GUARDRAILS_NO_USAGE_STATS=1` or `DO_NOT_TRACK=1`;
  `guardrails-ai` 0.11.0 posts usage metrics and builds an OpenTelemetry
  exporter at import, off with `OTEL_SDK_DISABLED` before the import and
  `settings.rc.enable_metrics = False` after it. Both facts were already known
  and written down in a `conftest.py`, which is where the test suite turns them
  off and which no user reads. A test requires each adapter README to name its
  own opt-out, because a disclosure a reader cannot act on is a paragraph.



- New check `confusables`, a constraint in both directions denying by default
  and configurable per direction through `on_detect`. Two finding types:
  `MIXED_SCRIPT_CONFUSABLE`, a token that fails the UTS #39 Highly Restrictive
  test and whose every code point outside its majority script folds into that
  script, and `WHOLE_SCRIPT_CONFUSABLE`, a single-script non-Latin URL host
  label, email domain label or `@handle` whose every code point folds into
  Latin. Published row: **0.936 precision and 0.863 recall over 109 cases** on
  `corpora/confusables/in-repo.jsonl`, 50 of them positives and 59 negatives.
  Seven disclosed misses and three disclosed false positives, ten wrong
  decisions in all, are named by case id in `corpora/NOTICE.md`.
- `fold_confusables` on `jamjet_guardrails.authoring.PatternGuardrail`, off by
  default. It matches banned substrings over the UTS #39 skeleton of both the
  content and the substring, so one substituted Cyrillic letter no longer dodges
  a banned word, and the spans still point at the source run through the offset
  map. It is refused where a guardrail declares no banned substrings, because
  there it would read as protection and buy none. Patterns are deliberately not
  folded: a regex is matched against the view it was written for.
- A fifth pinned Unicode data file, `IdentifierStatus.txt`, and a third
  generated module, `src/jamjet_guardrails/_unicode/identifiers.py`, reached
  through `identifier_allowed`. It is the UTS #39 Identifier Profile, which the
  restriction levels in section 5.2 are themselves defined over, and it is what
  decides whether a confusable prototype is evidence. Measured on 16.0.0, 140 of
  the 296 Cyrillic letters fold to a string written wholly in Latin and only 104
  of those fold to one written wholly in the profile: without it the new check
  denies `iPhoneом` in Russian prose and every `.рф` domain there is. The file's
  digest is in `corpora/NOTICE.md` and the module is held byte for byte to a
  regeneration, like the other two.

- The `script-constraint` check, detector version 0.1.0, constraint kind, both
  directions, `on_match` defaulting to `deny`. One required option,
  `allowed_scripts`, a collection of long Unicode script names, and one finding
  type, `DISALLOWED_SCRIPT`, whose span covers a maximal run of code points that
  no allowed script covers. `Common` and `Inherited` pass under every
  constraint, so punctuation, digits, currency, mathematics, whitespace, emoji,
  combining marks, ZWJ, ZWNJ and variation selectors are never findings; a code
  point the pinned 16.0.0 tables do not assign resolves to `Unknown` and always
  is one. Published row: precision 1.000, recall 1.000 over 85 cases of
  `corpora/script-constraint/in-repo.jsonl`, measured under the fixture
  `{"Latin", "Hiragana", "Katakana", "Han"}` recorded in
  `jamjet_guardrails.eval.fixtures` and printed beside the row in
  `docs/conformance.md`. That row measures the check under those options and
  promises nothing about any other set.
- **There is no default `allowed_scripts` and `build("script-constraint")`
  refuses.** So do a bare string, which is an iterable of its own characters; a
  value that cannot be iterated; an empty collection, which would deny every run
  of letters in every language; an entry that is not a string; a name the pinned
  tables do not know, which is refused with the valid names listed; and
  `Unknown`, which is not a script but what the tables return for a code point
  they cannot name. A collection naming only `Common` or `Inherited` is
  permitted and does what the empty one would have done.
- `docs/performance.md` publishes p50, p95 and p99 per check over deterministic
  inputs from 1 KB to 1 MB, with the machine, the interpreter, the input shape,
  the repetition count and the command that reproduces them. Each check's own
  figure and whether it is linear or near-linear in the content length now sit
  in that check's module docstring. There is no CI timing gate and there is not
  going to be one: `ci.yml` diffs the generated benchmark artifacts byte for
  byte, which works because they are deterministic, and a wall clock is not.
- `scripts/measure_throughput.py` produces those numbers. It builds every check
  through `jamjet_guardrails.detectors.build` with
  `jamjet_guardrails.eval.fixtures.options_for`, so a check that needs options is
  timed under the same fixture its published precision and recall row was
  measured under. Local only: it is not in the wheel and not in the sdist.
- A CodeQL workflow, on push and pull request to `main` and weekly, running the
  `security-and-quality` suite. It builds nothing and installs nothing, because
  `ci.yml` already lints, typechecks, tests and gates the benchmarks on five
  interpreters and a scan that repeats that work is a scan nobody reads.
- Vendored Unicode tables, pinned at 16.0.0. The five published files
  `Scripts.txt`, `ScriptExtensions.txt`, `PropertyValueAliases.txt`,
  `confusables.txt` and `IdentifierStatus.txt` are committed under
  `unicode-data/16.0.0/`, and `scripts/generate_unicode_tables.py` writes
  `src/jamjet_guardrails/_unicode/scripts.py`,
  `src/jamjet_guardrails/_unicode/confusables.py` and
  `src/jamjet_guardrails/_unicode/identifiers.py` from them. `confusables` and
  the `fold_confusables` option on the rules engine read them; `script-constraint`
  will. `unicodedata` carries no Script property, no confusables table and no
  Identifier_Status on any supported interpreter, and its Unicode version runs
  from 13.0 to 16.0 across the CI matrix, so a check deriving any of them from
  it would reach different verdicts on different legs of one test suite.
- Three private functions over those tables. `script_set` resolves one code
  point's scripts per UTS #39 section 5.1, returning long property names, with
  Script_Extensions taking precedence over Script, `Common` and `Inherited`
  returned as themselves, and an unassigned code point resolving to `Unknown`.
  `skeleton` builds the UTS #39 section 4 confusable skeleton together with the
  offset map back to the source, so a match found over a skeleton is reported
  as the span of the original characters that produced it, including where a
  prototype expanded one character into eighteen. `identifier_allowed` answers
  whether one code point is in the UTS #39 Identifier Profile.
- `_fold.compose`, which joins a view of the source and a view of that view
  into one map, and `_Folded.span` now takes the minimum and the maximum over
  the matched range. The skeleton's canonical ordering step permutes combining
  marks, which is the first offset map in this package that is not
  non-decreasing; reading its first and last entries returned a span one
  character short of the run the match covered, and a redaction over that span
  would have left a reordered mark standing inside content reported as
  rewritten. Every map built by `fold` is non-decreasing, so no existing span
  moved.
- Three guards on the vendored data: all three generated modules are rebuilt
  from the committed files and compared byte for byte with no network; each
  module's recorded SHA-256 digests are checked against those files; and a test
  skipped unless `JAMJET_GUARDRAILS_NETWORK=1`, which is never set in CI,
  re-downloads all five and compares them with what unicode.org publishes.
- `tests/test_packaging.py` now BUILDS the sdist and the wheel and reads their
  member lists, rather than reading the configuration that is supposed to
  produce them. `unicode-data/` is in the sdist and out of the wheel, and both
  halves are asserted against the archives. `hatchling`, already the build
  backend, joins the dev extra so the suite can do that offline.
- Property tests, in `tests/test_properties.py`, over the invariants
  `docs/conformance.md` already states: span merging covers exactly the union of
  its inputs and no character any span claimed survives a rewrite; a folded
  view's offset map has one entry per view character and a view span maps back to
  a source run closed over what produced the match; the structural injection
  check never raises and reports spans that index into the content it was given;
  a chain's decision is the restrictive combination of its verdicts, every
  verdict hashes the string the chain was given, no guardrail is handed another's
  rewrite, and no byte inside a reported span survives the composed redaction.
  Every Critical this package has recorded was a class of input nobody wrote
  down rather than a wrong answer on a case somebody thought of, and each was a
  short counterexample to one of those sentences.
- `hypothesis` in the `dev` extra. It cannot reach a consumer's environment: the
  zero-dependency promise is checked against the BUILT metadata with `extra ==`
  markers filtered out, by
  `tests/test_packaging.py::test_the_installed_distribution_declares_no_runtime_dependencies`.
- A recorded known miss, `tests/test_properties.py::test_a_word_boundary_gated_pattern_reports_a_zero_width_span`.
  A pattern gated by `\b` or by a lookbehind constructs, because it does not
  match at position zero of the empty string, and then reports a zero-width span
  against real content. It is left failing, marked `xfail`, rather than repaired
  by dropping zero-width matches inside the scan: that repair would turn a broken
  rule into a check that runs and never matches, which is the configured-and-
  silent failure the constructor already refuses five other ways. The chain
  refuses the verdict, so the composition fails closed, and the test asserts that
  containment beside the miss.
- `docs/migrating-from-llm-guard.md`, linked from `README.md`. One row per
  llm-guard scanner class at its final release 0.3.16, all 37 of them, each
  marked mapped, partial or gap with the check it maps to and one sentence on
  what differs. Counts: 6 mapped, 9 partial, 22 gap, printed at the top and held
  against the table's own rows by `tests/test_migration_guide.py`, which also
  refuses a replacement naming a check `build` cannot construct and refuses a
  percentage anywhere on the page. The page counts exported classes at the final
  release and says why that is 37 where the modules are 36, the archived branch
  is 38 and the documentation navigation is 35. It states what is lost as well
  as what is gained: 23 of the 37 classes are a model making a judgment and
  nothing here replaces them.
- A guard on the README's first table:
  `tests/test_readme.py::test_the_lead_table_names_every_check_the_registry_can_build`.
  A recorded contributor walkthrough added a check end to end, followed every
  message the suite produced, and reached five green gates with "What it
  catches" never mentioning it.
- A guard on the scaffold's own output:
  `tests/test_completeness.py::test_the_scaffold_writes_a_detector_a_caller_can_still_configure`.

### Changed

- The distribution's SPDX expression names the licences `template-data/` adds:
  `MIT`, and `LicenseRef-Llama-2-Community`, `LicenseRef-Meta-Llama-3-Community`
  and `LicenseRef-Gemma-Terms` for the three vendor community licences that have
  no SPDX short identifier. That material has been in the source distribution
  since the marker table landed and the licence field named none of it. The two
  guards that already existed could not see it, for the same reason: one reads
  the corpora, where every row carries its own `license`, and the other reads
  the built archives for Unicode data. A third guard now derives the domain from
  the marker table's own `Source` records, so a source added under a fifth
  licence fails until the expression names it, and a licence string the mapping
  does not recognise fails rather than being skipped.


- A `redact` the chain cannot locate, meaning no findings or a finding carrying
  no span, is now a synthesised `deny` instead of a `GuardrailChainError` out of
  `run`. It was the last shape in which one misbehaving detector cost the whole
  run its audit record, including the verdicts of every guardrail that had
  already run. Both shapes are reachable from an ordinary `Guardrail`
  implementation, because a `Verdict` may carry a finding with no span and
  nothing required a `redact` to carry findings at all, which makes them
  detector contract violations rather than assertions about this library. The
  content is not forwarded either way. `GuardrailChain._spans_of` keeps the same
  refusals for a direct caller, where they are now genuinely unreachable through
  `run`.
- `training/ship_bar.json` records the structural corpus's own version digest
  beside its path. The recorded floor is defined as decision-level recall
  measured on the corpus at that path, so unrelated work that legitimately grows
  that corpus moves the floor; with only a path recorded, nothing could tell a
  re-derivation from a silent edit. A test now re-derives the digest and fails
  until a move is disclosed in `structural_floor_rederived`.
- The bar's digest is split in two. The semantic registration, which is what the
  bar actually is, is digested on its own and has never moved; the whole file's
  digest moves with a disclosed re-derivation of the structural side.
  `clears_the_bar`, the file's prose statement of the same pass rule its values
  state, was on neither side of that split until an adversarial review found it,
  so the rule could be relaxed from `>` to `>=` in the sentence describing it
  with every digest still green. A test now refuses any key on neither side, so
  a field added later is a decision about which side it belongs to rather than a
  field nobody digests.


- `corpora/NOTICE.md` gains a "Chat-template marker sources" section: every
  source repository with its revision and licence, the three gated repositories
  named with the mirror each was read from instead, the notices the Llama 2,
  Meta Llama 3 and Gemma terms ask for, and what the table leaves out. Two
  exclusions are recorded with their measured cost: 2 HTML element names, `<s>`
  and `</s>`, removed by membership in the pinned element index rather than by
  naming the two strings, and 1018 reserved slots removed by the rule that
  their name ends in a number.
- `tests/test_published_docs.py` resolves cited `template-data/` paths too, so a
  published document that names a raw source file is held to it existing.
- The core sdist no longer carries `packages/`. Measured: hatchling 1.32.0
  includes it by default, so before this the source distribution shipped 25
  files belonging to the two adapters, including their pyproject.toml files and
  the framework dependencies those declare. The sdist is the evidence for the
  zero-dependency claim, and
  `tests/test_packaging.py::test_the_sdist_carries_nothing_from_the_adapter_packages`
  now builds it and holds that. The wheel was already limited to `src/` and is
  unchanged.
- `tests/test_published_docs.py` recognises `packages/` as a repository path, so
  a citation from either adapter's README is checked like any other.
- **The `rules` published row moves, deliberately, because its fixture gained a
  rule.** `jamjet_guardrails.eval.fixtures` now sets `fold_confusables=True` for
  `rules`, and the corpus gained the confusable-dodged codename case that option
  exists for, plus the near-miss negative beside it. The `rules` corpus version
  moves from `f1b809114b13` to `8fe119ddb734` and the case count from 40 to 42.
  Precision and recall are unchanged at 1.000 and 1.000, with zero wrong
  decisions.
  `corpora/baselines.json` carries the new key and no longer carries the old
  one.
- **The published `rules` latency moves with it, from 143 ms to 731 ms per
  megabyte**, for the same reason: the fixture's `fold_confusables=True` builds
  the UTS #39 skeleton of the whole content on every call. `docs/performance.md`
  carries the new rows and says which rows that run re-measured and which it
  left alone. The option is off by default and a caller who leaves it off pays
  none of it.
- The distribution declares `Apache-2.0 AND CC-BY-4.0 AND Unicode-3.0`. The
  generated tables are derived from data files published by Unicode, Inc. under
  the Unicode License v3 and ship in the wheel, and the raw files ship in the
  sdist; the licence requires its notice to travel with copies or in associated
  documentation. `corpora/NOTICE.md` carries the notice, the five digests and
  the reason the data is vendored, and the README's licence section says so in
  one sentence.
- `docs/conformance.md` adds two entries to "What is deliberately unspecified":
  where Unicode property data comes from and at what version, and the fold
  machinery. A port reaching the same verdicts on the corpora conforms with any
  Unicode data source at any version. What stays specified is the span itself,
  which indexes the content the chain was given whatever view the match was
  found in.

### Fixed

- A `validators-v*` tag no longer fires the core PyPI publish job.
  `startsWith(github.ref_name, 'v')` is true for `validators-v0.1.0`, because
  that string starts with the letter v, so the core job ran on a validators
  release as well as its own. It failed rather than publishing anything wrong,
  since the version gate strips one leading `v` and compares the remainder
  against the packaged version, but a release that always carries one red job
  is a release nobody reads the jobs of. The condition now also requires the
  tag not to contain `-v`, which is a property of the naming scheme rather than
  a list of adapter names: an adapter tag is the distribution name followed by
  `-v` and a version, and a core prerelease such as `v0.4.0-rc.1` is not. A
  test evaluates all three job conditions against all four tag shapes and
  requires each to fire exactly one publisher.
- The "Adding a check" section of `CONTRIBUTING.md`, and the checklist
  `scripts/new_check.py` prints, rewritten from a recorded walkthrough that
  followed the old four-item version literally in a fresh clone. Twelve edits
  now, in order, because four of them were in files the list never named and two
  of them cannot be done in either order. The import is a step of its own:
  registering a check without importing it raises `NameError` while the package
  is imported, so pytest reports collection errors and no test results at all,
  and nothing in that output names the missing line. The claim that
  `tests/test_completeness.py` fails until each edit is done is gone, because
  every test in that module is parametrised over `AVAILABLE` and cannot see an
  unregistered check.
- `scripts/new_check.py` writes a detector that takes `on_match`. It passed
  `on_match="deny"` and `**options` into the same call, so
  `build("<name>", on_match="redact")` raised `TypeError: PatternGuardrail() got
  multiple values for keyword argument 'on_match'`, and every check scaffolded
  before this shipped unconfigurable. Its test template now says which mutation
  to run rather than claiming one was run.
- `scripts/mutate.py` takes a full pytest node id. It built one from
  `tests/test_training_data.py` and a bare test name, with the path hardcoded,
  so it could not address the test file a new check writes, which is the file
  `CONTRIBUTING.md` points its readers at. A bare name still means that module, so existing lists keep
  working. The module docstring now carries an example entry, and the file
  itself is one the contributor writes: none is committed.

### Security

- Every GitHub Action in every workflow is pinned to a full commit SHA with a
  comment naming the release it resolves to, in place of the floating tags
  (`actions/checkout@v7` and the rest) the workflows used before. A tag is a
  movable pointer, and these jobs read the repository, hold a job token and,
  in `release.yml`, stand beside the OIDC identity PyPI trusts.
- `tests/test_workflows.py` keeps them pinned. It reads every workflow file git
  tracks rather than a list of names, parses each with PyYAML, and fails on any
  `uses:` that is not a 40-hex commit SHA carrying a version comment.
- `GuardrailChain` refuses, at construction, a guardrail whose declared
  `directions` hold none the runtime can carry. `build` and `build_chain`
  already refused this on the registry door; `GuardrailChain` did not, and it is
  a supported door, because the chain's own documentation tells a caller who
  wants no checks to construct `GuardrailChain([])` directly. Such a guardrail
  is inert: it is skipped in every context, so alone it produced the empty
  chain's output, which is `allow` with the content untouched and no verdicts,
  and beside a live detector it made the chain quieter than the configuration
  said while raising nothing to report it. The test is intersection with the
  runnable set rather than emptiness, so `{"inptu"}` and `{"stream"}` are
  refused with `frozenset()`. **Potentially breaking**: a configuration holding
  an inert guardrail stops building instead of running quietly without it.
- `GuardrailChain` refuses a declared `name` or `version` longer than 200
  characters, the ceiling every other caller-supplied string in an audit record
  was already held to. Both are copied into the `Provenance` of every verdict
  that guardrail produces, and they were bounded only where they reached an
  error message, so a two-million-character name cost one truncated error string
  and an unbounded `provenance.detector` in every verdict of every run.
  Truncating the stored copy was rejected: a truncated name is a record that
  does not say what the guardrail declared, and the chain grades a returned
  `provenance.detector` against that copy, so the honest guardrail returning its
  own full name would be the one recorded as lying. **Potentially breaking**: a
  deployment whose detector name is longer than that stops building its chain on
  upgrade. `authoring.PatternGuardrail` holds the same ceiling, so a check built
  through the documented path now fails at its own constructor rather than two
  seams later at the chain.
- Neither refusal quotes the values it refuses. `directions` is data the
  guardrail declares, read from caller code that runs after the caller has
  content in hand, so a declared direction can be the content; both refusals now
  report how many directions were declared and name the guardrail by position.
  The first draft of the chain's refusal interpolated the set whole, which an
  adversarial review found, and it is the defect the module's own `_bounded_str`
  exists to close, reached from a third side. `detectors.build` carried the same
  unbounded message and now bounds it too.
- The five copies of the runnable-direction set are held to `Direction` and to
  each other. `types._DIRECTIONS` and `authoring._RUNNABLE` were outside the
  first version of that guard, and `types._DIRECTIONS` is the copy whose drift
  fails OPEN: growing it alone lets a `Context` carry a direction no guardrail
  declares, so every guardrail is skipped and the run reports `allow` over
  content nothing checked. That mutation passed the entire suite.
- `_scan` no longer loops forever on a pattern that matches ZERO-WIDTH at the end
  of the content. `Pattern.search` clamps a `pos` past the end of the string back
  to `len(content)`, so such a match was found again at the same offset on every
  pass, the containment filter dropped the repeat as no longer than the one
  already kept, and `pos` was set back to the same value. Nothing advanced.
  `PatternGuardrail` refuses a pattern that matches the empty string outright and
  cannot refuse one gated by a lookbehind or a word boundary, so a user rule as
  ordinary as `\ba*` over the content `"0"`, or `(?<=a)` over `"a"`, reached the
  scan through the caller-configured `rules` check and hung the process. A
  guardrail that never returns is the one failure a fail-closed library cannot
  report: there is no verdict, no synthesised deny and no audit record, and the
  caller's own timeout is the only thing that ends the request. The scan now
  stops once its resume position passes the end of the content, which is what
  makes the backstop `authoring.py` already described reachable: the zero-width
  match is reported once and the chain refuses the verdict, so the composition
  fails closed. **No published number moves.** Every pattern in every bundled
  check and in the `rules` fixture is unable to match the empty string at any
  position, so none of them can produce a zero-width match at all.
- The distribution declares `Apache-2.0 AND CC-BY-4.0 AND Unicode-3.0`. The
  generated tables are derived from data files published by Unicode, Inc. under
  the Unicode License v3 and ship in the wheel, and the raw files ship in the
  sdist; the licence requires its notice to travel with copies or in associated
  documentation. `corpora/NOTICE.md` carries the notice, the four digests and
  the reason the data is vendored, and the README's licence section says so in
  one sentence.
- `docs/conformance.md` adds two entries to "What is deliberately unspecified":
  where Unicode property data comes from and at what version, and the fold
  machinery. A port reaching the same verdicts on the corpora conforms with any
  Unicode data source at any version. What stays specified is the span itself,
  which indexes the content the chain was given whatever view the match was
  found in.


- A `redact` the chain cannot locate, meaning no findings or a finding carrying
  no span, is now a synthesised `deny` instead of a `GuardrailChainError` out of
  `run`. It was the last shape in which one misbehaving detector cost the whole
  run its audit record, including the verdicts of every guardrail that had
  already run. Both shapes are reachable from an ordinary `Guardrail`
  implementation, because a `Verdict` may carry a finding with no span and
  nothing required a `redact` to carry findings at all, which makes them
  detector contract violations rather than assertions about this library. The
  content is not forwarded either way. `GuardrailChain._spans_of` keeps the same
  refusals for a direct caller, where they are now genuinely unreachable through
  `run`.
- `training/ship_bar.json` records the structural corpus's own version digest
  beside its path. The recorded floor is defined as decision-level recall
  measured on the corpus at that path, so unrelated work that legitimately grows
  that corpus moves the floor; with only a path recorded, nothing could tell a
  re-derivation from a silent edit. A test now re-derives the digest and fails
  until a move is disclosed in `structural_floor_rederived`.
- The bar's digest is split in two. The semantic registration, which is what the
  bar actually is, is digested on its own and has never moved; the whole file's
  digest moves with a disclosed re-derivation of the structural side.
  `clears_the_bar`, the file's prose statement of the same pass rule its values
  state, was on neither side of that split until an adversarial review found it,
  so the rule could be relaxed from `>` to `>=` in the sentence describing it
  with every digest still green. A test now refuses any key on neither side, so
  a field added later is a decision about which side it belongs to rather than a
  field nobody digests.


- Every GitHub Action in every workflow is pinned to a full commit SHA with a
  comment naming the release it resolves to, in place of the floating tags
  (`actions/checkout@v7` and the rest) the workflows used before. A tag is a
  movable pointer, and these jobs read the repository, hold a job token and,
  in `release.yml`, stand beside the OIDC identity PyPI trusts.
- `tests/test_workflows.py` keeps them pinned. It reads every workflow file git
  tracks rather than a list of names, parses each with PyYAML, and fails on any
  `uses:` that is not a 40-hex commit SHA carrying a version comment.

- `docs/performance.md` gains `url-exfiltration`: 112 ms median for one megabyte
  of the seeded input, 9.4 megabytes per second, ratios of 3.90 to 4.04 per 4x of
  input, which is the fastest of the four scanning checks. Its rows come from
  their own run of `scripts/measure_throughput.py` on the machine that page names
  rather than from a regeneration of the whole table, and the page says so: two
  runs on one laptop differ by a few percent in the p50 and more in the p99, so
  rewriting every other check's numbers would have read as a regression in
  checks nobody had touched.

### Disclosed

- **DNS-label exfiltration is not detected**, and neither is prose in a hostname.
  Only path segments and query components are decoded and tested. `url-0078` and
  `url-0079` are labelled `deny`, are allowed, and cost recall rather than
  sitting in prose.
- **A doubly encoded payload passes.** One level of decoding only, and decoded
  text is never fed back to the decoder. `url-0080` is the measured example.
- **A nested redirect whose inner URL is plainly percent-encoded does not fire.**
  That is by design: percent-encoding is what the query syntax already provides,
  and firing on it denies every OAuth authorization link there is.
- **A link query longer than 136 characters fires whether or not it is a
  payload.** The floor was set by sweep and it does not separate the two
  populations, because they overlap: a share intent and a prefilled issue body in
  the corpus carry 206 and 263 characters of ordinary prose and are denied.
- **Rot13 ships**, on a two-sided test and a measured ablation: removing it costs
  two true positives and returns no precision.
- A generated chat-template marker table,
  `src/jamjet_guardrails/detectors/_template_markers.py`, holding 59 delimiter
  strings read out of the tokenizer configuration of nine pinned repositories:
  Llama 2 chat, Llama 3 instruct, Qwen 2.5 instruct, Mistral instruct, Gemma 2
  instruct, Phi-3 instruct, DeepSeek V3, GPT-2 and, for one exclusion, the HTML
  Standard's element index. Every marker records the repository, the commit and
  the SHA-256 of the file it came from. `scripts/generate_template_markers.py`
  builds it and the raw files it read are committed under `template-data/`, in
  the sdist and out of the wheel, so the table can be regenerated and diffed
  offline. Nothing imports it yet: the `template-integrity` check lands
  separately with its corpus and its published row, and until then the table is
  private and unregistered.
- Three guards on that table in `tests/test_template_markers.py`: regeneration
  from the committed raw files must reproduce the module byte for byte, the
  recorded digests must be those files' digests in both directions, and a
  network-gated run under `JAMJET_GUARDRAILS_NETWORK=1`, never set in CI,
  re-fetches every pinned revision for the day a pin is bumped.
- `tests/test_packaging.py` now builds the sdist and the wheel and opens both,
  which nothing outside the release workflow did before. It asserts that every
  file under `template-data/` is in the sdist and that none of it is in the
  wheel, and that the generated table is. `build` and `hatchling` join the dev
  extra to make that possible with no network.
- Two distribution adapters, each its own package under `packages/`, its own
  PyPI distribution, its own CI job on Python 3.12 and its own release tag.
  Neither changes anything about the core package, which still declares no
  runtime dependencies.
- `jamjet-guardrails-nemo`, released under tags `nemo-v<semver>`. Two NeMo
  Guardrails system actions and two Colang 1.0 flows shipped as `.co` files.
  Both chains are built when the rails load, so a check that is named and not
  installed fails the load rather than the first message, and a direction whose
  flow and check list disagree is refused there too. Every call writes a JSON
  audit record to the rail context under `jamjet_guardrails` carrying the
  decision, each verdict's detector, version, kind, decision, finding types and
  spans, the SHA-256 of what was inspected, and an error where a check failed.
  Never the content. Configuration lives under `custom_data` in the rails
  `config.yml`: measured on nemoguardrails 0.24.0, `RailsConfig` has the pydantic
  default `extra="ignore"`, so a top-level custom key is parsed and discarded
  with no error anywhere.
- `jamjet-guardrails-validators`, released under tags `validators-v<semver>`.
  One Guardrails AI validator per registered core check, generated from
  `jamjet_guardrails.detectors.AVAILABLE` at import, plus the `JamJetChain`
  composite, which is the one to use. Stacking two per-check validators is a
  measured leak on guardrails-ai 0.11.0 in both of its validator services: the
  sequential service reproduces the sequential-rewrite leak `GuardrailChain` was
  rebuilt to close, character for character, and the default async service loses
  a deny to a redact depending on the order the validators were listed in.
  `JamJetChain` runs one chain and returns one merged fix, so neither happens.

## [0.3.0]

### Security

- `GuardrailChain` no longer trusts anything a guardrail returns. It reads each
  field of a returned `Verdict` once, checks it against what the chain itself
  knows, and then BUILDS a new verdict from those reads: `saw` is the chain's own
  digest over the content it passed to `check`, `provenance` carries the identity
  the chain holds for that guardrail, and the findings are fresh objects with
  their own strings and plain integer spans. Nothing a detector returned reaches
  `ChainResult.verdicts`, the combined decision or the rewrite AS THE OBJECT it
  arrived in. Previously a `check` that returned successfully could report any
  detector name, version and kind, hash text it was never given, and carry a
  finding's span past the end of the content, and all of it was recorded
  unexamined. Verifying that verdict and then keeping it was not enough either: a
  `str` subclass whose `__eq__` returns True passes every comparison, an object
  whose `__class__` says `Verdict` passes `isinstance` and can answer a property
  honestly while it is being checked and falsely while it is being recorded, and
  an `int` subclass passes `0 <= start < end <= len(content)` and then slices as
  a negative number, which emitted a prefix of the ORIGINAL content in front of a
  redaction placeholder under a `redact` decision. Every strictness check reads
  `type(x) is T`, never `isinstance`, because a subclass with a lying `__eq__` or
  `__str__` walks straight through the comparison or the coercion `isinstance`
  would still allow.
- A finding's `type` is bounded to the same length every other caller-chosen
  string in an error message already was. It is still, by design, the one
  detector-chosen string that reaches `ChainResult.content`: a redaction
  placeholder has to name what claimed the region, so `EMAIL` in
  `[REDACTED:EMAIL]` is a finding's `type`. Unbounded, a `redact` whose
  finding's type IS the content it redacted reproduced that content, verbatim,
  inside a string this library calls safe to forward, and a type running to
  millions of characters inflated the audit record by the same amount for one
  finding. The bound stops the second failure, not the first: a short type
  equalling a short secret is not new, and is not what the bound closes.
- `provenance.threshold` and a finding's `confidence` are checked for
  finiteness, not only type. Both are `float | None`, and NaN and infinity are
  both a `float`, so the type check alone let either one reach the audit
  record. A non-finite value is now a contract violation like any other.
- A malformed finding span no longer abandons the run. A span of `(1, 2, 3)`,
  `("a", "b")` or `5` raised out of `run` from validation that sat outside the
  try, losing the whole run's audit record and every guardrail after the one
  that misbehaved. It is now a synthesised `deny` like any other failed check.
- An error message never repeats a value a detector chose: not a claimed
  provenance, not a finding's type, not the class name of whatever was returned,
  not a span. A detector's returned values are picked after it has seen the
  content, so a detector whose class name or finding type IS the content used to
  write the content into the audit record through the message. Messages name the
  clause that failed, and name the guardrail from the chain's own bounded copy
  of the name it declared; a 2,000,000-character name produced a
  4,000,073-character `error` before that bound.
- `GuardrailChain.__init__` reads each guardrail's `name`, `version`, `kind` and
  `directions` once, checks them, and uses those copies for every verdict
  afterwards, refusing a guardrail that cannot declare a usable identity with
  `GuardrailUnavailableError` -- the error `build` and `build_chain` already
  raise for a configuration that would check less than it claims. A `name` that
  answered one way while it was being validated and another way while it was
  being recorded is no longer read twice. A guardrail declaring a `kind` this
  library does not know is refused at construction rather than raising
  `ValueError` from inside `run`: because provenance is stamped from the
  declared kind, that guardrail used to turn the chain's own fail-closed path
  into an exception, so an honest verdict or a raising `check` took the entire
  run down with it.

### Upgrading from 0.2.0

Every refusal below is new, and each one turns a shape that used to work into a
`deny` or a refusal to build. None of them affects the four bundled checks. If
you have written or installed a custom check, read this list against it.

- **A `Verdict` SUBCLASS is refused.** The chain tests `type(verdict) is Verdict`,
  because a subclass is how an object lies about itself: `isinstance` consults
  `__class__`, which a caller sets. If you subclassed `Verdict` to carry extra
  fields, return a plain `Verdict` and keep your own data beside it.
- **A guardrail declaring a `kind` outside `constraint` and `classifier` is
  refused when the chain is built**, not when it runs. It used to work if its
  verdicts were honest.
- **The verdict you return is not the verdict that is recorded.** The chain
  rebuilds it, so `ChainResult.verdicts[i] is your_verdict` is now False and any
  field the chain does not rebuild is not carried. Compare by value.
- **A finding `type` longer than 200 characters is refused**, and a
  non-finite `threshold` or `confidence` is refused.
- **A `Verdict` you return with `error` set is refused.** That field belongs to
  the chain.

A check that returns a plain `Verdict` whose `saw` is the digest of the content
it was given, whose provenance names itself, and whose spans index into that
content, is unaffected. That is what every bundled check already did.

### Changed

- A `redact` carrying no content, and a `redact` whose span does not index into
  the content, no longer raise `GuardrailChainError` out of `run`. Both are
  verdicts the chain will not rebuild, so both become a synthesised `deny` and
  the run keeps its audit record. One shape still abandons a run: a `redact` the
  chain cannot locate, meaning no findings or a finding without a span.
- A `Verdict` returned with `error` set is refused. `error` is the chain's own
  field, set on a verdict the chain synthesised, and a detector writing its own
  prose there put an unbounded, detector-chosen string into the audit record.

## [0.2.0]

### Added

- `PatternGuardrail` and `Limits`, exported from the package root. A check is
  now typed regular expressions, banned substrings and size limits, and the
  spans, the merging and the refusals come with it. `docs/conformance.md`
  specifies what a published row measured through it does and does not promise.
- `rules`, a check that takes your own patterns, banned substrings and size
  limits. Size limits are characters, bytes and lines; there is no token limit,
  because counting tokens needs a tokenizer this library does not carry.
- `TYPES`, beside `AVAILABLE`, naming the finding types each check can report.
- `scripts/new_check.py`, which writes the files a new check needs and names
  the four it leaves to you.

### Changed

- `injection-structural` runs on output as well as input, and its detector
  version moves to 0.2.0. A chain that previously skipped this check on model
  output now runs it there and can deny. A model emitting tag characters into
  its own output is smuggling to whatever reads that output next.
- Its published precision moved from 0.971 to 0.972 and its recall from 0.870
  to 0.873, because the corpus gained eight output cases and grew from 146 to
  154. The detector's behaviour on every case it already scored is unchanged.
- `rules` is new and is published at 1.000 precision and 1.000 recall on 40
  cases. That row measures the engine under one fixed configuration, printed
  in `docs/conformance.md`, and says nothing about a rule you write yourself.

### Security

- `GuardrailUnavailableError` is now also raised from `PatternGuardrail.check`,
  not only at construction, when a caller passes a context whose direction the
  guardrail does not declare. This is the same error as the constructor raises
  when a check is misconfigured, because both mistakes mean a check was asked
  to evaluate something it was not set up for, and answering would falsely
  report that content had been examined. This replaces the previous silent allow
  on content with no matches and a bare `KeyError` on content with matches. A
  chain never triggers this, since it only calls a guardrail whose declared
  directions contain the context's direction; the error is for a caller holding
  a single guardrail directly.

### Notes

- Still no runtime dependencies. The authoring primitive is standard library
  only, and `dependencies = []` is still checked against the built metadata.

## [0.1.0]

First release.

### Added

- `build_chain` and `GuardrailChain`, returning a decision, the findings behind
  it, and a provenance record naming the detector and its version for every
  check that ran. Each verdict carries `saw`, the SHA-256 of the exact string
  that check inspected.
- Three deterministic checks. `injection-structural` (input) reports
  `BIDI_OVERRIDE`, `INVISIBLE_TAG_CHARS` and `ZERO_WIDTH_SMUGGLING`. `pii`
  (input and output) reports `CREDIT_CARD`, `EMAIL`, `PHONE_NUMBER` and
  `US_SSN`. `secrets` (input and output) reports `ANTHROPIC_KEY`,
  `AWS_ACCESS_KEY`, `GITHUB_TOKEN`, `JWT`, `OPENAI_KEY`, `PRIVATE_KEY` and
  `SLACK_TOKEN`.
- Published precision and recall per check and per type, measured on corpora
  committed in `corpora/` and gated in CI against `corpora/baselines.json`. The
  scores at this release are in [BENCHMARKS.md](BENCHMARKS.md).
- `jamjet-guardrails`, a console script that scores the corpora, writes
  `benchmarks.json` and `BENCHMARKS.md`, and gates a run against recorded
  baselines.
- [docs/conformance.md](docs/conformance.md), specifying the verdict fields, the
  combination order, the single-pass rewriting rule, the `saw` hash and the
  corpus schema, so the checks can be reimplemented in another language and
  scored on the same corpora.
- PEP 561 typing marker. The wheel ships `py.typed` and the release workflow
  asserts it is inside the built artifact.

### Security

- Decisions combine restrictively, `deny` > `redact` > `allow`, and no path can
  weaken a decision another check has already made.
- Every check inspects the content the chain was given, and redactions from all
  checks are merged and applied in one pass. Rewriting sequentially let a
  personal-data redaction cut a credential in half so the next check matched
  only the stump, and the rest of the credential survived into content the chain
  reported as redacted.
- A check that raises becomes `deny`, never `allow`, and the error message is
  withheld from the verdict because a detector's message may quote the content.
- A check named in configuration that is not installed raises
  `GuardrailUnavailableError` at construction rather than running unguarded. An
  empty list of checks is refused for the same reason.

### Notes

- No runtime dependencies, no network calls, no model downloads. The installed
  distribution declaring none is checked against the built metadata.
- The distribution declares `Apache-2.0 AND CC-BY-4.0`. The code is Apache-2.0;
  the source distribution also carries `corpora/pii/third-party.jsonl`, derived
  from `nvidia/Nemotron-PII` under CC-BY-4.0. Attribution and the list of
  changes are in [corpora/NOTICE.md](corpora/NOTICE.md). The wheel contains code
  only.

[Unreleased]: https://github.com/jamjet-labs/jamjet-guardrails/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jamjet-labs/jamjet-guardrails/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jamjet-labs/jamjet-guardrails/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jamjet-labs/jamjet-guardrails/releases/tag/v0.1.0
