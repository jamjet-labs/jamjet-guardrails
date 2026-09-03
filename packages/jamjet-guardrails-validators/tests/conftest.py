"""Keep the test suite offline and deterministic.

guardrails-ai posts usage metrics to an AWS endpoint unless
`settings.rc.enable_metrics` is false, and it installs an OpenTelemetry batch
span exporter at import time whose flush happens at interpreter exit. In an
environment with no route to that host each run ends in several seconds of OTLP
retries; in an environment with one, a test run phones home. Neither belongs in
CI.

`OTEL_SDK_DISABLED` is set BEFORE `guardrails` is imported, and the ordering is
the whole point: the exporter is built during import, so a fixture that ran
later would be turning off something already wired up. `enable_metrics` is a
settings flag rather than an environment variable, so it is set as well.
"""

from __future__ import annotations

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from guardrails import settings

settings.rc.enable_metrics = False
settings.disable_tracing = True
