# onnxruntime wheel availability, cp310 through cp314

Read from PyPI on **2026-09-01** (Asia/Kolkata), which is `2026-08-31T21:24:20Z`.
This is a moving fact. Anything below is only as good as that date, and the
section [Repeating this](#repeating-this) is how to re-run it.

## Why this was measured

`pyproject.toml` carries a `Programming Language :: Python` classifier for each
of 3.10 through 3.14, and `tests/test_packaging.py` holds that list to the CI
matrix, because a version claimed and not tested is a claim we cannot back and
PyPI shows it to everyone who opens the page.

The core package is pure Python and has no runtime dependencies, so nothing here
touches it. The next stage adds a separate distribution that depends on
`onnxruntime`, which ships as compiled wheels. If onnxruntime has no wheel for an
interpreter the core supports, that distribution's floor is not the core's floor,
and the two classifier lists have to differ.

## Result

**There is a gap, and it is at the floor rather than the ceiling.**

onnxruntime stopped publishing CPython 3.10 wheels after **1.23.2**
(2025-10-22). Every release since carries no cp310 wheel on any platform.
3.11 through 3.14 are all served by the newest release.

| Interpreter | manylinux x86_64 | macOS arm64 | Newest onnxruntime overall? |
|---|---|---|---|
| cp310 | 1.23.2 | 1.23.2 | no, 10 releases behind |
| cp311 | 1.29.0 | 1.29.0 | yes |
| cp312 | 1.29.0 | 1.29.0 | yes |
| cp313 | 1.29.0 | 1.29.0 | yes |
| cp314 | 1.29.0 | 1.29.0 | yes |

Newest onnxruntime on PyPI at the time of reading: **1.29.0**, uploaded
2026-08-17. `last_serial` on the JSON response was `40120392`.

The cut is by interpreter, not by platform. The same table for
`manylinux aarch64` and `win_amd64` is identical, cell for cell: 1.23.2 on
cp310 and 1.29.0 on the other four. No platform loses an interpreter that
another platform keeps.

3.14 was the version the earlier brief expected to be missing. It is not.
cp314 wheels first appear in **1.24.1** (2026-02-05) and have shipped in every
release since, on both platforms. A free-threaded `cp314t` build ships alongside
them, but on **manylinux only** -- x86_64 and aarch64, and nothing else. "Both
platforms" is true of cp314 and false of cp314t, and this sentence used to hand
the reader one scope for both.

### Where cp310 stops, exactly

| Release | Uploaded | `Requires-Python` in metadata | cp310 wheel files |
|---|---|---|---|
| 1.23.0 | 2025-09-25 | `>=3.10` | 5 |
| 1.23.1 | 2025-10-08 | `>=3.10` | 5 |
| 1.23.2 | 2025-10-22 | `>=3.10` | 5 |
| 1.24.1 | 2026-02-05 | `>=3.10` | 0 |
| 1.24.2 | 2026-02-19 | `>=3.10` | 0 |
| 1.24.3 | 2026-03-05 | `>=3.10` | 0 |
| 1.24.4 | 2026-03-17 | `>=3.11` | 0 |
| 1.25.0 to 1.29.0 | 2026-04-22 to 2026-08-17 | `>=3.11` | 0 |

Worth reading closely. Three releases, 1.24.1 through 1.24.3, still declared
`Requires-Python: >=3.10` while shipping no cp310 wheel at all. The wheels are
the leading signal; the metadata caught up at 1.24.4 on 2026-03-17, 146 days
after the last cp310 wheel shipped. A check that
read only `Requires-Python` would have called 3.10 supported for every release
from 1.24.1 to 1.24.3, that is from 2026-02-05 until 1.24.4 landed on
2026-03-17, and been wrong for the whole of that window.

There is no 1.24.0 on PyPI. The series starts at 1.24.1.

### The failure mode on 3.10 is silent

onnxruntime publishes **no sdist**. All 1170 files in the index are wheels.

So on 3.10, pip does not error. It finds no compatible wheel for 1.24.1 and
above, has no source fallback to try, backtracks, and resolves to 1.23.2 without
comment. Measured against pip's real resolver rather than inferred from
filenames:

```
$ pip download --no-deps --only-binary=:all: \
    --python-version 310 --implementation cp --abi cp310 \
    --platform manylinux_2_28_x86_64 onnxruntime
Saved onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

A package that depended on a bare `onnxruntime` and claimed 3.10 would therefore
pass its own install step on the floor leg, and quietly run a native runtime ten
months and six minor versions behind the one every other leg runs. That is a
worse outcome than a hard failure, because nothing surfaces it.

### Permanent, or not yet published?

**Permanent.** Four things say so, and none of them is the absence of a wheel by
itself:

1. Ten consecutive releases ship zero cp310 wheels, spanning 2026-02-05 to
   2026-08-17 (193 days). The last release that shipped one, 1.23.2 uploaded
   `2025-10-22T03:46:21Z`, is 313 days old measured to the UTC read instant in
   the header, `2026-08-31T21:24:20Z`. Measured instead from the LOCAL date this
   document is filed under, 2026-09-01, the same subtraction gives 314; the
   figure is anchored to the UTC instant, which is the only one the data has.
2. From 1.24.4 the drop is stated in metadata, not merely implied by the absence
   of a file. A build-infrastructure lag does not get written into
   `Requires-Python`.
3. It moved in one direction. There is no release after 1.23.2 that brings cp310
   back.
4. CPython 3.10 left its bugfix phase on 2023-04-05 and reaches end-of-life on
   2026-10-31 (`https://endoflife.date/api/python.json`, read the same day), so
   upstream dropping it in late 2025 is on the normal curve rather than early.

The contrast case is instructive, and it is in the same index. cp314 is absent
from 1.23.2 (2025-10-22, two weeks after CPython 3.14's own release on
2025-10-07) and present in the very next release, 1.24.1. That is what "not yet
published" looks like here: one release, then it appears.

### One secondary finding: the macOS platform floor

Not asked for, but it falls out of the same data and it constrains a CI runner
choice rather than a classifier, so it is recorded here rather than lost.

macOS arm64 wheels raised their platform floor from `macosx_13_0` to
`macosx_14_0` at 1.24.1, and every release since is `macosx_14_0_arm64`. A
macOS 13 runner cannot install any onnxruntime from 1.24.1 onward, on any
interpreter. The same check turns up something broader: from 1.24.1 the only
macOS platform tag published at all is `macosx_14_0_arm64`, so Intel macOS is
gone too, for every interpreter.

```
$ pip download --no-deps --only-binary=:all: \
    --python-version 314 --implementation cp --abi cp314 \
    --platform macosx_13_0_arm64 onnxruntime
ERROR: Could not find a version that satisfies the requirement onnxruntime (from versions: none)
```

If the next stage runs anything on macOS in CI, the runner image has to be
macOS 14 or newer.

## What this forces

Recorded, not applied. Nothing in this repository changes on account of it; the
next stage's packaging is not this branch's work.

1. **The core package is unaffected.** It is pure Python with no runtime
   dependencies. `requires-python = ">=3.10"` and the 3.10 through 3.14
   classifiers stay exactly as they are, and the CI matrix stays at five legs.
2. **The next stage's onnxruntime-dependent distribution floors at 3.11.**
   `requires-python = ">=3.11"`, and classifiers for 3.11 through 3.14 only.
   Claiming 3.10 there would mean either pinning that distribution to a runtime
   upstream no longer supports, or claiming a version whose install silently
   resolves to a different runtime than the one the artifact was validated
   against. Neither is a claim this repository is willing to publish.
3. **The two floors have to be asserted separately.**
   `test_every_python_classifier_is_a_version_ci_runs` currently reads one
   distribution's built metadata and compares it to one CI matrix. With two
   distributions at two floors, the shared matrix stops being the right
   denominator: the 3.10 leg will run the core's tests and must not be expected
   to install the injection distribution at all. The assertion needs a
   per-distribution floor, not a single global one, and the 3.10 leg needs to
   skip the injection package explicitly rather than by accident.
4. **Pin an onnxruntime floor, not just a name.** Since `Requires-Python` lagged
   the wheel drop by three releases, a dependency of bare `onnxruntime` inherits
   whatever upstream decides next without telling anyone. A lower bound in the
   dependency specifier makes the supported range a stated claim.

## Repeating this

Everything above comes from public PyPI endpoints, and nothing was installed.
The per-interpreter table was derived independently from each of the two index
endpoints below and the two derivations agree cell for cell. The `pip download`
commands are a third path: they exercise the real resolver rather than a
filename parser, and they write a wheel to disk without installing it.

```bash
# 1. The JSON API: every release with all its files.
curl -sS https://pypi.org/pypi/onnxruntime/json -o onnxruntime.json

# 2. The PEP 691 simple index, as an independent cross-check.
curl -sS -H 'Accept: application/vnd.pypi.simple.v1+json' \
     https://pypi.org/simple/onnxruntime/ -o onnxruntime-simple.json
```

Both returned 1170 files across 61 releases, none yanked, none an sdist. For each
file, split the wheel filename on `-`, take the last three fields as the
compressed python, abi and platform tag sets, expand each on `.`, and keep rows
where the python tag and the abi tag are both the interpreter in question.
`manylinux x86_64` is a platform tag starting `manylinux` and ending `_x86_64`;
macOS arm64 is one starting `macosx_` and ending `_arm64` or `_universal2`.
Sort the surviving versions with `packaging.version.Version`, not as strings,
or 1.9.0 sorts above 1.29.0.

Two things that make the result trustworthy rather than merely produced:

- onnxruntime publishes **no** `abi3` wheels. Every wheel is tagged for one
  exact interpreter, so "has a wheel for cp310" needs no forward-compatibility
  reasoning. Checked, not assumed.
- The query was run against negative controls before its output was believed:
  `cp399` on manylinux and cp310 on a fabricated platform tag both return
  nothing, and `cp312` on `win_amd64` returns 1.29.0. A filter that silently
  matched nothing would have produced the same empty cells as a real absence.
  A fourth control, cp314 on `musllinux`, is recorded here as NOT carrying any
  weight: onnxruntime publishes zero musllinux wheels for any interpreter, so
  that query returns nothing whether the filter works or not and cannot tell the
  two apart. It is degenerate as a control. The three above are what the claim
  rests on.

Free-threaded builds were counted separately and excluded from the table above.
`cp313t` and `cp314t` wheels do exist and track the same 1.29.0, but on
**manylinux only** -- x86_64 and aarch64, with no macOS or Windows free-threaded
wheel at any version -- and they are a distinct ABI, not what the default
interpreter of a CI leg resolves.

## The other three measurements, and why they are moot

This document was scoped from a brief that asked for five measurements. Only the
one above still gates anything.

The other three (feasibility of a TurboQuant-style quantizer over the model
weights, Hadamard blocking versus padding for a hidden dimension that is not a
power of two, and the ONNX Runtime custom-operator ABI range) all existed to
serve one plan: compressing a large published model far enough to fit under
PyPI's per-file size limit. That plan was replaced. The next stage produces its
own, much smaller classifier artifact, which fits under the limit with headroom
and needs no compression at all.

With no compression there is no quantizer to validate, no transform to choose,
and no custom runtime operator to build, so no compiled wheel matrix and no ABI
range to pin. The three questions are not deferred or unresolved. They stopped
being questions.

The remaining brief item, choosing a third-party injection corpus disjoint from
the model's training data, is unaffected by the redesign and is not addressed
here. One gating, three moot, one untouched: five.
