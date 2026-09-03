# Performance

What each check costs per call. These are wall-clock measurements from one
machine, named below, and they are not a service level: they are here so a
reader can decide whether an in-process check fits their latency budget, and so
they can run the same script and see whether their machine agrees.

There is no CI timing gate, deliberately. `.github/workflows/ci.yml` regenerates
`benchmarks.json` and `BENCHMARKS.md` on every matrix leg and diffs them byte
for byte, which works only because those artifacts are deterministic. A clock is
not. A timing assertion in that suite would be either loose enough to catch
nothing or tight enough to fail on a busy runner.

## How to reproduce

```
./.venv/bin/python scripts/measure_throughput.py
```

That prints the table below and then the same measurement as JSON. `--json PATH`
writes the JSON to a file instead. `--repetitions`, `--warmup`, `--sizes` and
`--direction` all take overrides, and this page was produced with none of them:
every setting recorded under Method is the script's own default. The JSON
carries the same settings in its `environment` block, so a run whose parameters
differ from these can be told apart from one that does not.

**A check added after this page was first written brings its own rows from its
own run**, on the machine and interpreter named below and with the same
defaults. `url-exfiltration` and `encoded-content` are two. Rows from two runs
of the same script on one machine are a few percent apart in the p50 and further
apart in the p99, which is the variance this page publishes a p99 to show rather
than to hide; the alternative was rewriting every other check's numbers from
whichever run happened to be last, and a busier laptop would then read as a
regression in a check nobody had touched.
defaults. `url-exfiltration` and `template-integrity` are two. Rows from two
runs of the same script on
one machine are a few percent apart in the p50 and further apart in the p99,
which is the variance this page publishes a p99 to show rather than to hide; the
alternative was rewriting every other check's numbers from whichever run
happened to be last, and a busier laptop would then read as a regression in a
check nobody had touched.

## The machine and the interpreter

| | |
|---|---|
| Machine | Apple MacBook Pro, `Mac15,11`, Apple M3 Max, 14 cores, 36 GB |
| Operating system | macOS 26.6.2, build 25G83, `arm64` |
| Interpreter | CPython 3.14.5, built with Clang 21.0.0 |
| Build | the repository's own `.venv`, package installed editable |
| Load | interactive laptop, on mains power, otherwise idle |

Nothing was pinned to a core and the machine was not otherwise quiesced, so the
tail includes whatever else the operating system was doing. That is the point of
publishing a p99 rather than a mean.

On a machine whose clock is coarser than the call, the script prints a dash in
the MB/s column and names the resolution of its own timer underneath the table.
That is not a check running infinitely fast, it is a row that measured the clock
instead: `secrets` costs 8 microseconds over a kilobyte here, which is inside one
tick of a 15.6 millisecond timer by three orders of magnitude. Raise
`--sizes` until the numbers move.

Rows are ADDED to this table when a check ships, and the rows already in it are
not re-measured at the same time. Rerunning the whole script for every release
would move numbers for checks nobody touched, and a reader diffing this page
would read that as a regression somebody should explain. Every row here was
taken on the machine and the interpreter named above; `script-constraint` was
measured on 2026-09-03 and the other four when this page was first written.

## The input

One deterministic string per size, built by
`scripts/measure_throughput.py::content_of`: a fixed block of business prose
repeated and then sliced to exactly the requested character count. There is no
random number generator in the script, so the input at a given size is the same
string on every machine and every interpreter.

The block is ordinary prose with matchable values spaced through it, at a
density of roughly one finding per check per kilobyte: an email address at
`example.com`, an AWS access key id, a JIRA ticket id, an internal `corp.example`
hostname and a run of `U+200B ZERO WIDTH SPACE`. The `findings` column below is
what that density actually came to at each size, so a reader can see that each
row paid for building findings and not only for scanning.

The content is ASCII apart from the zero-width run, so a size in characters and
the same size in UTF-8 bytes differ by well under a percent. Sizes are powers of
four from 1 KB, so linearity is visible by dividing adjacent rows: a linear check
costs 4x per step and a quadratic one costs 16x.

Each check is built through `jamjet_guardrails.detectors.build` with
`jamjet_guardrails.eval.fixtures.options_for`, which is how the evaluation
harness builds it. So `rules`, which cannot exist without options, is timed under
the same fixture its published precision and recall row was measured under, and
the two numbers describe the same object.

## Method

200 timed calls per check per size, after 5 warmup calls that are discarded.
Each call is one `guardrail.check(content, context)`, timed with
`time.perf_counter_ns`; the guardrail and the content are built once, outside
the loop. The context carries `direction="input"`, `origin="user"`.

Percentiles are nearest rank over the sorted sample, so every figure printed is
a call that really happened rather than an interpolation between two of them.
Garbage collection stays enabled: a p99 is a tail, collection pauses are part of
the tail a caller sees, and switching the collector off would produce a number
nobody could reproduce.

## The numbers

Milliseconds per call. `MB/s` is derived from the p50.

| Check | Chars | Findings | p50 ms | p95 ms | p99 ms | MB/s |
|---|---:|---:|---:|---:|---:|---:|
| confusables | 1 024 | 0 | 0.182 | 0.190 | 0.197 | 5.4 |
| confusables | 4 096 | 0 | 0.720 | 0.781 | 0.807 | 5.4 |
| confusables | 16 384 | 0 | 2.814 | 2.934 | 2.984 | 5.6 |
| confusables | 65 536 | 0 | 11.296 | 11.739 | 12.291 | 5.5 |
| confusables | 262 144 | 0 | 45.491 | 47.770 | 48.426 | 5.5 |
| confusables | 1 048 576 | 0 | 185.595 | 216.458 | 256.182 | 5.4 |
| encoded-content | 1 024 | 0 | 0.166 | 0.171 | 0.179 | 5.9 |
| encoded-content | 4 096 | 0 | 0.648 | 0.691 | 0.776 | 6.1 |
| encoded-content | 16 384 | 0 | 2.550 | 2.656 | 2.753 | 6.2 |
| encoded-content | 65 536 | 0 | 10.208 | 10.533 | 10.983 | 6.2 |
| encoded-content | 262 144 | 0 | 42.053 | 43.697 | 44.273 | 6.0 |
| encoded-content | 1 048 576 | 0 | 168.925 | 174.867 | 177.190 | 6.0 |
| injection-structural | 1 024 | 1 | 0.123 | 0.139 | 0.149 | 8.0 |
| injection-structural | 4 096 | 3 | 0.486 | 0.535 | 0.558 | 8.1 |
| injection-structural | 16 384 | 15 | 1.987 | 2.089 | 2.127 | 7.9 |
| injection-structural | 65 536 | 61 | 7.891 | 8.220 | 8.376 | 8.0 |
| injection-structural | 262 144 | 244 | 31.487 | 32.428 | 33.316 | 8.0 |
| injection-structural | 1 048 576 | 979 | 124.167 | 126.180 | 126.833 | 8.1 |
| pii | 1 024 | 1 | 0.204 | 0.215 | 0.250 | 4.8 |
| pii | 4 096 | 4 | 0.820 | 0.841 | 0.879 | 4.8 |
| pii | 16 384 | 16 | 3.289 | 3.701 | 3.770 | 4.8 |
| pii | 65 536 | 62 | 12.996 | 13.551 | 13.769 | 4.9 |
| pii | 262 144 | 245 | 51.916 | 53.372 | 54.060 | 4.9 |
| pii | 1 048 576 | 979 | 206.674 | 211.254 | 215.892 | 4.9 |
| rules | 1 024 | 2 | 0.135 | 0.150 | 0.176 | 7.3 |
| rules | 4 096 | 9 | 0.518 | 0.552 | 0.573 | 7.6 |
| rules | 16 384 | 31 | 2.117 | 2.196 | 2.331 | 7.4 |
| rules | 65 536 | 123 | 8.313 | 8.767 | 8.901 | 7.6 |
| rules | 262 144 | 491 | 34.193 | 36.405 | 37.047 | 7.4 |
| rules | 1 048 576 | 1 959 | 142.608 | 150.364 | 152.402 | 7.1 |
| script-constraint | 1 024 | 0 | 0.248 | 0.274 | 0.320 | 4.0 |
| script-constraint | 4 096 | 0 | 0.998 | 1.094 | 1.118 | 3.9 |
| script-constraint | 16 384 | 0 | 4.049 | 4.252 | 4.312 | 3.9 |
| script-constraint | 65 536 | 0 | 15.961 | 16.535 | 17.112 | 4.0 |
| script-constraint | 262 144 | 0 | 63.889 | 66.362 | 68.548 | 3.9 |
| script-constraint | 1 048 576 | 0 | 255.746 | 263.393 | 267.481 | 3.9 |
| rules | 1 024 | 2 | 0.699 | 0.763 | 0.838 | 1.4 |
| rules | 4 096 | 9 | 2.769 | 2.925 | 2.999 | 1.4 |
| rules | 16 384 | 31 | 10.844 | 11.312 | 11.536 | 1.4 |
| rules | 65 536 | 123 | 43.738 | 45.006 | 45.828 | 1.4 |
| rules | 262 144 | 491 | 175.289 | 178.582 | 182.292 | 1.4 |
| rules | 1 048 576 | 1 959 | 730.951 | 752.094 | 766.248 | 1.4 |
| secrets | 1 024 | 1 | 0.008 | 0.010 | 0.010 | 125.9 |
| secrets | 4 096 | 4 | 0.021 | 0.026 | 0.027 | 184.1 |
| secrets | 16 384 | 15 | 0.073 | 0.078 | 0.087 | 216.9 |
| secrets | 65 536 | 61 | 0.290 | 0.326 | 0.335 | 217.2 |
| secrets | 262 144 | 244 | 1.141 | 1.232 | 1.276 | 221.2 |
| secrets | 1 048 576 | 979 | 4.697 | 4.990 | 5.199 | 214.9 |
| template-integrity | 1 024 | 0 | 0.033 | 0.034 | 0.034 | 29.7 |
| template-integrity | 4 096 | 0 | 0.121 | 0.129 | 0.135 | 32.6 |
| template-integrity | 16 384 | 0 | 0.465 | 0.480 | 0.518 | 33.9 |
| template-integrity | 65 536 | 0 | 1.866 | 1.949 | 2.037 | 33.8 |
| template-integrity | 262 144 | 0 | 7.493 | 8.280 | 8.544 | 33.7 |
| template-integrity | 1 048 576 | 0 | 29.724 | 31.835 | 32.471 | 34.0 |
| url-exfiltration | 1 024 | 0 | 0.112 | 0.117 | 0.156 | 9.2 |
| url-exfiltration | 4 096 | 0 | 0.439 | 0.470 | 0.486 | 9.4 |
| url-exfiltration | 16 384 | 0 | 1.733 | 1.811 | 1.839 | 9.5 |
| url-exfiltration | 65 536 | 0 | 6.961 | 7.354 | 7.471 | 9.5 |
| url-exfiltration | 262 144 | 0 | 28.090 | 29.141 | 34.361 | 9.4 |
| url-exfiltration | 1 048 576 | 0 | 112.116 | 115.217 | 118.193 | 9.4 |

## What each check does with the length

The ratio quoted for each check is p50 at one size divided by p50 at a quarter
of it, so 4.0 is exactly linear.

- **`confusables`** is linear. Ratios across the range run 3.91 to 4.08 and the
  rate holds at 5.4 to 5.6 megabytes per second. Its `findings` column is 0 at
  every size, and that is a real limitation of this row rather than a rounding:
  the seeded input carries no confusable, so what is timed is the token scan and
  the label scan with no finding built and nothing rewritten. The input is
  shared with every other row on this page and was deliberately not changed to
  add one. Almost every token in it is ASCII, which the check skips before
  asking the vendored tables anything, so this row is the FLOOR for this check
  and text in a non-Latin script costs more.
- **`injection-structural`** is linear. Ratios across the range run 3.94 to 4.09
  and the rate never leaves 7.9 to 8.1 megabytes per second. It denies on this
  input rather than redacting, so the number is a scan and the construction of
  the findings, with no rewrite.
- **`pii`** is linear. Ratios run 3.95 to 4.02 and the rate holds at 4.8 to 4.9
  megabytes per second across three orders of magnitude. It is the slowest check
  here, and the Luhn validation of every candidate digit run is why: it redacts,
  so the number includes rewriting the content.
- **`rules`** is near-linear, drifting slightly above 4.0 at the top of the
  range: 3.93 at 65 KB, 4.11 at 256 KB, 4.17 at 1 MB. The drift is the finding
  count, which grows faster than for the other checks because this fixture has
  more patterns matching this input, and merging spans is what those extra
  findings cost. The fixture also sets `max_chars=2000`, so above that size every
  call reports a `LENGTH_LIMIT` finding and the redacted output is truncated. The
  patterns are still scanned over the whole content, which is what these rows
  measure.
- **`encoded-content`** is linear. Ratios run 3.90 to 4.12 and the rate holds at
  5.9 to 6.2 megabytes per second across the range. **The findings column is zero
  at every size, and that is what these rows measure**: nothing in the seeded
  input decodes, so the number is the candidate scan plus one decode attempt per
  candidate, and none of the three signals. What makes it the second slowest
  check here is the alphabet with no alphabet: every run of letters and spaces is
  a rot13 candidate, so ordinary prose is nothing but candidates, and each one is
  rotated and scored for prose twice, once in each direction. Content full of
  base64 costs less per byte than the English around it.
- **`url-exfiltration`** is linear. Ratios run 3.90 to 4.04 and the rate holds
  at 9.2 to 9.5 megabytes per second across the range, which makes it the
  fastest of the five scanning checks. **The findings column is zero at every
  size, and that is what these rows measure**: the seeded input carries no URL,
  so the number is the discovery pass over the whole content and none of the
  decoding. Content full of URLs costs more, because each one is taken apart and
  its components are decoded at up to four alphabets each; that work is bounded
  by the number of URLs and by their length, not by the length of the document
  around them, so it is linear in a different variable rather than a worse power
  of this one.
- **`script-constraint`** is linear. Ratios run 3.94 to 4.06 and the rate holds
  at 3.9 to 4.0 megabytes per second across the whole range. It is the slowest
  check here, and the reason is structural rather than fixable: every other
  check is a regular expression that rejects most start positions on their first
  character, and this one has to resolve every code point it is handed, through
  two bisections over the vendored tables. Its `findings` column is 0 at every
  size, because the seeded input is Latin and Common throughout and the fixture
  allows both, so these rows are the PASS path with no findings built and
  nothing rewritten. Content that fires costs more, and how much more depends on
  how many separate runs it carries rather than on how much of it is disallowed.
  range: 4.01 at 256 KB and 4.17 at 1 MB. The drift is the finding count, which
  grows faster than for the other checks because this fixture has more patterns
  matching this input, and merging spans is what those extra findings cost. The
  fixture also sets `max_chars=2000`, so above that size every call reports a
  `LENGTH_LIMIT` finding and the redacted output is truncated. The patterns are
  still scanned over the whole content, which is what these rows measure.

  **These figures are 5.1x the ones this page carried before, and the cause is a
  configuration change rather than a regression.** The fixture now sets
  `fold_confusables=True`, so every call builds the UTS #39 skeleton of the whole
  content: two normalisation passes, a fold through the confusables table, and an
  offset map composed through all three. That is five passes over the content and
  one integer per character per pass, and it is what the option costs. It is off
  by default, and a caller who does not set it pays none of it. There is no ASCII
  shortcut to be had: the confusables table maps eight ASCII code points,
  including `m` to `rn` and `1`, `I` and `|` to `l`, so an all-ASCII string is
  not its own skeleton.
- **`template-integrity`** is linear on both of its paths, and the rows above
  measure only the one most content takes. Ratios run 3.84 to 4.02 from 4 KB
  upward and the rate holds at 33 to 34 megabytes per second, which makes it the
  fastest of the scanning checks after `secrets`; the first step reads 3.67
  because at 33 microseconds a call the fixed cost still dominates. **The
  findings column is zero at every size, and that is what these rows measure**:
  the seeded input carries no marker, so the number is the folding pass and the
  three scans over it and none of the offset map. Content that FIRES pays for the
  map as well, which is four Python-level passes building one integer per view
  character, and it costs 31x. The same one-megabyte input with a single
  `<|im_start|>` appended to it, timed the same way, comes to 923 ms at p50
  against 29.6 ms without it. That path is linear too: at 30 timed calls per size
  rather than 200, because a run of 200 at that cost takes three minutes, the
  ratios across the top three sizes are 3.97, 4.02 and 4.18. The map is built
  once per call and only after a signal has already fired, so content that
  matches nothing never pays for it, and a large document an attacker has seeded
  with one marker is exactly the input that does. Reproduce both with:

  ```
  ./.venv/bin/python - <<'EOF'
  import sys, time
  sys.path.insert(0, "scripts")
  from measure_throughput import content_of, percentile
  from jamjet_guardrails import Context
  from jamjet_guardrails.detectors import build

  guardrail = build("template-integrity")
  context = Context(direction="input", origin="user")
  for label, content in (
      ("no marker", content_of(1_048_576)),
      ("one marker", content_of(1_048_576 - 12) + "<|im_start|>"),
  ):
      for _ in range(5):
          guardrail.check(content, context)
      samples = []
      for _ in range(200):
          start = time.perf_counter_ns()
          guardrail.check(content, context)
          samples.append((time.perf_counter_ns() - start) / 1e6)
      samples.sort()
      print(label, round(percentile(samples, 0.50), 3))
  EOF
  ```
- **`secrets`** is linear from 16 KB upward, at ratios of 3.97, 3.93 and 4.12.
  Below that it is dominated by fixed overhead rather than by the content: the
  first step reads as 2.62x and the second as 3.48x, climbing towards 4.0 as the
  content starts to matter. At 8 microseconds a call the measurement is close to
  the cost of taking it, and those two ratios are that cost rather than a
  property of the check. It is the fastest check here by more than an order of
  magnitude because every pattern is anchored on an issuer prefix, so almost
  every start position fails on the first character. The reasoning behind the one
  pattern that had to be bounded to stay linear, and the quadratic behaviour that
  bound removed, is recorded in
  `src/jamjet_guardrails/detectors/secrets.py`.

## Which rows were re-measured, and when

The table is not one run. Rows are added and replaced as the checks that produce
them change, and a row nobody touched is left where it is: rewriting every figure
on this page whenever one check moves would report a change in checks nobody
edited, which is exactly the false alarm this page exists not to raise.

The `confusables` rows and the `rules` rows were measured together in one later
run on the same machine and interpreter. That run also re-measured the three
untouched checks, and they came out 5% to 13% above the figures recorded here
(`injection-structural` at 1 MB read 139.909 rather than 124.167, `pii` 209.751
rather than 206.674, `secrets` 4.729 rather than 4.697). The machine was busier,
not slower. Their rows were left alone and this paragraph is the reason: read
across checks with that spread in mind, and read within one check freely.

## What this does not say

These are per-check figures on one input shape. A chain of several checks costs
roughly the sum of its checks plus one rewrite, and the shape of your content
matters: text with no matchable values at all is faster than this, and text that
is nothing but matchable values is slower. Nothing here is a guarantee, a budget
or a claim about any other machine. Run the script and read your own numbers.
