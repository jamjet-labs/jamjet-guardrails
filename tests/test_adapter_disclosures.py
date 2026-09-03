"""Each adapter README says that its framework makes network calls.

The core README's front page says "No dependencies. No network calls. No model
downloads." That is true of the core and it is the sentence a reader carries
into the adapters, where it stops being true of their environment:
`nemoguardrails` 0.24.0 posts usage statistics to an NVIDIA endpoint by default,
and `guardrails-ai` 0.11.0 posts usage metrics and builds an OpenTelemetry
exporter at import.

Both facts were known when the adapters landed. Both were written down in
`packages/jamjet-guardrails-validators/tests/conftest.py`, which is where the
test suite turns them off, and **a user never reads a conftest**. So the
knowledge was in the repository and the disclosure was not, which is the gap
this file closes.

The check is deliberately not "the README mentions telemetry". It requires the
OPT-OUT to be named, because a disclosure a reader cannot act on is a paragraph
rather than a disclosure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ROOT / "packages"

#: Skipped where the adapters are not present, which is the SOURCE DISTRIBUTION.
#: `pyproject.toml` excludes `packages/` from the sdist deliberately, and states
#: why: the sdist is the evidence for the zero-dependency claim, and an sdist
#: carrying two adapters' metadata, each declaring a framework dependency,
#: invites exactly one wrong reading.
#:
#: So this module cannot run there and must SAY so rather than fail. The
#: distinction matters because the same `pyproject.toml` claims three times that
#: the sdist ships the tests as its evidence, and a suite that fails inside its
#: own evidence is a claim that does not hold. Skipped on the property that the
#: directory is absent, never on a filename or an environment variable.
_NO_ADAPTERS = pytest.mark.skipif(
    not PACKAGES.is_dir(),
    reason="packages/ is excluded from the source distribution by design",
)

#: What each adapter's README must name, and it is the mechanism rather than the
#: prose. Written out here because the mapping from a framework to the switch
#: that silences it is a fact about that framework, which this repository cannot
#: derive: it is read out of the installed source once and recorded.
#:
#: Verified against `nemoguardrails` 0.24.0 and `guardrails-ai` 0.11.0 on
#: 2026-09-03 by reading `nemoguardrails/telemetry.py` and by watching a
#: `guardrails` import attempt an OTLP export with no network route.
_OPT_OUTS = {
    "jamjet-guardrails-nemo": ("NEMO_GUARDRAILS_NO_USAGE_STATS", "DO_NOT_TRACK"),
    "jamjet-guardrails-validators": ("OTEL_SDK_DISABLED", "enable_metrics"),
}


def _adapters() -> list[str]:
    """Every package under `packages/`, read from disk rather than listed.

    A third adapter is covered without anyone remembering, which is the whole
    reason this repository derives its guards: the defect it produces more than
    any other is a guard written for whichever file its author had open.
    """
    if not PACKAGES.is_dir():
        return []
    return sorted(p.name for p in PACKAGES.iterdir() if (p / "README.md").is_file())


@_NO_ADAPTERS
def test_there_are_adapters_to_check() -> None:
    """The vacuity guard. An empty `packages/` would make the tests below pass
    over nothing at all."""
    found = _adapters()
    assert found, "no adapter packages found; every check below would be vacuous"
    assert set(found) == set(_OPT_OUTS), (
        f"packages/ holds {found} and this guard knows the opt-out for "
        f"{sorted(_OPT_OUTS)}; an adapter whose framework nobody checked for "
        "telemetry is exactly the one worth stopping for"
    )


@_NO_ADAPTERS
@pytest.mark.parametrize("adapter", _adapters())
def test_each_adapter_readme_discloses_that_its_framework_reaches_the_network(
    adapter: str,
) -> None:
    """The core promises no network calls. An adapter that quietly breaks that
    promise for the reader's environment has to say so where the reader is.

    Mutation-checked: deleting the section from either README fails this, and so
    does keeping the prose while removing the environment variable that turns
    the telemetry off.
    """
    readme = (PACKAGES / adapter / "README.md").read_text(encoding="utf-8")
    assert "network calls" in readme, (
        f"{adapter}/README.md does not tell the reader that installing it brings a "
        "framework which reaches the network, while the core README's front page "
        "promises none"
    )
    missing = [switch for switch in _OPT_OUTS[adapter] if switch not in readme]
    assert missing == [], (
        f"{adapter}/README.md discloses the network calls and does not name {missing}, "
        "so a reader cannot act on it"
    )
