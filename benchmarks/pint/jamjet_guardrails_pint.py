"""A PINT evaluation function for the `injection-structural` check.

PINT drives a detector through one callable: a string in, a bool out, True
meaning "this input is a prompt injection". That is the whole interface this
file implements.

What it wraps is a constraint, not a model. `injection-structural` reads the
encoding rather than the words: invisible tag characters, unbalanced
bidirectional overrides, and unexplained zero-width runs. It has no opinion
about an instruction written in plain view, so on a corpus of semantic
injections it is expected to report almost nothing. It is a layer under a
semantic detector, not a replacement for one.

There is no PINT score for this package. The PINT dataset is not public and
results are verified by the Lakera team, so nothing self-run against the
example dataset is a PINT score and none is claimed anywhere in this
repository.
"""

from jamjet_guardrails import Context, build

# Built once. The check is pure Python with no I/O, no model and no network,
# so construction is cheap and the object is stateless across calls.
#
# `build` rather than the class: a name that resolves through the registry is a
# name the registry can refuse, and it refuses at construction rather than
# returning something that goes quiet at first use.
INJECTION_STRUCTURAL = build("injection-structural")


def evaluate_jamjet_guardrails(prompt: str) -> bool:
    """True when the input carries structural instruction smuggling.

    `direction="input"` because that is the only direction this check declares;
    a chain skips a guardrail outside its directions, so any other value would
    measure a run that never happens. `origin="user"` is what a prompt arriving
    from a person is; this check does not read it.

    Tested against `!= "allow"` rather than `== "deny"`. Deny is the default,
    but a caller may construct the check with `on_match="redact"`, and under
    `== "deny"` that configuration would return False for every input it caught.
    """
    verdict = INJECTION_STRUCTURAL.check(prompt, Context(direction="input", origin="user"))
    return verdict.decision != "allow"
