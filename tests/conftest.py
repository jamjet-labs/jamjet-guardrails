"""Hypothesis settings, registered explicitly because the defaults are wrong here.

CI runs five interpreters on every push and, on every one of them, regenerates
`benchmarks.json` and `BENCHMARKS.md` and diffs them byte for byte against what
is committed. A suite that explores different inputs on each leg turns a red
build into a coin flip and a green one into nothing: the leg that found the
counterexample is the leg somebody re-runs until it passes, and nobody can say
which inputs the green run actually covered.

So the default profile trades exploration for reproducibility, deliberately and
in both directions:

- ``derandomize=True`` derives the input sequence from the test's own source
  rather than from a clock, so all five legs try the SAME inputs and a failure
  reproduces on a laptop from the name of the test alone.
- ``database=None`` because the example database is the second source of drift.
  A stored failing example replays on the machine that stored it and nowhere
  else, so a run that is green in CI and red locally, or the reverse, is a
  property of a directory rather than of the code.
- ``max_examples`` is bounded: this suite runs in about half a minute and that
  is what makes people run it before pushing. See the note on the number below.
- The deadline is generous rather than absent, for the reason in its own note.

The cost is that the default profile explores a fixed slice. The way to explore
MORE is opt in, not a wider default: ``HYPOTHESIS_PROFILE=explore`` picks a
profile that randomises, runs ten times the examples and keeps the database, so
a developer hunting for a counterexample gets a different slice on every run and
Hypothesis remembers what it found. `CONTRIBUTING.md` documents it. A
counterexample found under `explore` belongs in the suite as an
``@example``-seeded case, so that the fixed profile keeps checking it forever.
"""

from __future__ import annotations

import os
from datetime import timedelta

from hypothesis import HealthCheck, settings

# 200, and the number is a measurement rather than a round figure. The property
# module runs in about three seconds at this setting, against a suite that took
# 27.7 seconds before it existed, which keeps the total inside the budget
# CONTRIBUTING states. Raising it does not buy proportionally more: Hypothesis
# spends its early examples on the shapes these properties actually break at,
# and two spans that touch, a fold that deletes every character and an empty
# input are all reached well inside 200.
#
# What it does NOT buy is a boundary one input wide. Widening one content
# alphabet moved the derandomised draw sequence far enough to stop generating a
# string of exactly the length a size limit fires at, and a mutation that had
# been caught went green. A boundary a property depends on is seeded with
# `@example` rather than left to this number.
_CI_EXAMPLES = 200

# 2 seconds per example, which is not a performance target and is not meant to
# be reached. The slowest property here costs well under a millisecond per
# example locally, so this is roughly a 2,000x margin, and that margin is the
# point: a shared CI runner under load stalls a process for entire seconds, and
# a deadline tight enough to be informative is a deadline that fires on a
# neighbour's build rather than on this code. It is not `None` because a
# property that becomes genuinely pathological -- a strategy that starts
# generating megabyte strings, a detector that goes quadratic -- should still
# fail rather than quietly turn the suite into a long one.
_DEADLINE = timedelta(seconds=2)

settings.register_profile(
    "ci",
    derandomize=True,
    database=None,
    max_examples=_CI_EXAMPLES,
    deadline=_DEADLINE,
    # `too_slow` is the one health check suppressed, and only because of what it
    # measures: how long Hypothesis spent BUILDING inputs, which on a loaded
    # shared runner is a fact about the runner. Everything else stays on, and
    # two of them are load-bearing rather than tolerated. `filter_too_much`
    # fails a strategy whose `assume` rejects most of what it draws, which is
    # exactly the failure mode of a text strategy that is not reaching the code
    # points it was written for; `data_too_large` fails one that has quietly
    # started generating inputs too big to shrink. Both would otherwise make a
    # property that explores almost nothing look identical to one that explores
    # a lot.
    suppress_health_check=[HealthCheck.too_slow],
    # The reproduction blob is printed with a failure, so a counterexample found
    # on a CI leg can be replayed locally without guessing at the seed.
    print_blob=True,
)

settings.register_profile(
    "explore",
    derandomize=False,
    max_examples=_CI_EXAMPLES * 10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)

# Read once, here, so a typo names a profile Hypothesis has not registered and
# raises at collection rather than silently running the default. An unrecognised
# value that fell back to `ci` would be the same silence this package refuses
# everywhere else: a knob that is set and does nothing.
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
