"""The `rules` check: the authoring primitive, registered.

There is no detector here, and that is the point. `rules` is
`jamjet_guardrails.authoring.PatternGuardrail` under a registry name, so a user
configuring rules at runtime and a contributor adding a check to this
repository exercise one engine with one set of published numbers behind it.

Its version is this module's, not the primitive's, because what a caller pins
when they pin `rules` is the behaviour of this registration: the name on the
verdict, the default decision, and which directions it declares.

Cost, and the shape of it. Near-linear in the content length. 143 ms median for
one megabyte of the seeded input recorded in `docs/performance.md`, on an Apple
M3 Max under CPython 3.14.5, with the median rising 4.17x over the last 4x step
of input rather than 4.0x. The drift is span merging over a finding count that
grows faster than the input does, not the scan. Measured under the fixture in
`jamjet_guardrails.eval.fixtures`, which is the configuration the published row
uses; patterns a user writes carry their own cost and no claim here covers them.
`scripts/measure_throughput.py` reruns it, and `docs/performance.md` states the
machine, the input and the method.
"""

from __future__ import annotations

from jamjet_guardrails.authoring import PatternGuardrail
from jamjet_guardrails.protocol import Guardrail

_VERSION = "0.1.0"

# LENGTH_LIMIT is the one type this check reports that a caller does not name,
# so it is the one type that is always in its domain. Everything else is the
# caller's, and for the PUBLISHED ROW it is the fixture's; see
# jamjet_guardrails.eval.fixtures for what that row does and does not promise.
RULES_TYPES = frozenset({"LENGTH_LIMIT", "TICKET_ID", "INTERNAL_HOST", "PROJECT_CODENAME"})


def build_rules(**options: object) -> Guardrail:
    """Construct the `rules` check.

    A factory rather than the class itself in `AVAILABLE`, because the registry
    calls its value with the caller's options alone and the name and version
    are this module's to supply. Registering the class would make every caller
    pass them and let two callers disagree about what `rules` is called.
    """
    return PatternGuardrail(name="rules", version=_VERSION, **options)  # type: ignore[arg-type]
