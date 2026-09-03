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
| secrets | 1 024 | 1 | 0.008 | 0.010 | 0.010 | 125.9 |
| secrets | 4 096 | 4 | 0.021 | 0.026 | 0.027 | 184.1 |
| secrets | 16 384 | 15 | 0.073 | 0.078 | 0.087 | 216.9 |
| secrets | 65 536 | 61 | 0.290 | 0.326 | 0.335 | 217.2 |
| secrets | 262 144 | 244 | 1.141 | 1.232 | 1.276 | 221.2 |
| secrets | 1 048 576 | 979 | 4.697 | 4.990 | 5.199 | 214.9 |

## What each check does with the length

The ratio quoted for each check is p50 at one size divided by p50 at a quarter
of it, so 4.0 is exactly linear.

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

## What this does not say

These are per-check figures on one input shape. A chain of several checks costs
roughly the sum of its checks plus one rewrite, and the shape of your content
matters: text with no matchable values at all is faster than this, and text that
is nothing but matchable values is slower. Nothing here is a guarantee, a budget
or a claim about any other machine. Run the script and read your own numbers.
