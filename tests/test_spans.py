"""Unit tests for the span arithmetic every regex-backed detector shares.

These moved here verbatim from `test_pii.py` and `test_secrets.py` when the
duplicated `_Region`, `_scan` and `_merge` were extracted to `_spans.py`. They
were the same test written twice against two copies of the same code. The type
names they use are still the detectors' own, because the cases are real arrival
orders those detectors produce and a synthetic "A", "B", "C" would stop being
evidence about anything.

The scan and merge behaviours are exercised through the detectors, in
`test_pii.py` and `test_secrets.py`, against a per-character oracle over a
generated corpus. That is where they belong: what those guards are worth depends
on the patterns feeding them, and each detector's corpus is built from its own.
"""

from jamjet_guardrails._spans import _Region


def test_the_placeholder_sorts_type_names_whatever_order_they_arrive_in() -> None:
    """Pin the sort deterministically, which a set could not do.

    While `_Region.types` was a set, dropping the `sorted` still produced sorted
    output under most hash seeds, so no assertion could hold the sort down: it was
    pinned in about 92% of runs, which is not pinned. `types` is now a list in
    first-seen order, so arrival order is fixed by the input rather than by
    PYTHONHASHSEED, and a case whose arrival order differs from its sorted order
    fails every time the sort is removed.

    "mail 123-456-7890@example.com" is that case end to end: PHONE_NUMBER's span
    sorts first so it arrives first, while EMAIL sorts first alphabetically.
    """
    region = _Region(0, 1, [])
    for pii_type in ("PHONE_NUMBER", "EMAIL", "CREDIT_CARD"):
        region.claim(pii_type)

    assert region.types == ["PHONE_NUMBER", "EMAIL", "CREDIT_CARD"], "arrival order kept"
    assert region.placeholder == "[REDACTED:CREDIT_CARD+EMAIL+PHONE_NUMBER]"


def test_a_region_records_each_type_once() -> None:
    """Two matches of one type over a single region name it once, not twice."""
    region = _Region(0, 1, [])
    for pii_type in ("EMAIL", "EMAIL", "US_SSN", "EMAIL"):
        region.claim(pii_type)

    assert region.types == ["EMAIL", "US_SSN"]
    assert region.placeholder == "[REDACTED:EMAIL+US_SSN]"


def test_a_region_records_each_type_once_and_sorts_the_names() -> None:
    """Pin the sort deterministically, which a set could not do.

    `types` is a list in first-seen order, so arrival order is fixed by the input
    rather than by PYTHONHASHSEED, and an arrival order that differs from the
    sorted order fails every time the sort is removed. "xoxb-" plus a nested AWS
    key is exactly that case: SLACK_TOKEN arrives first, AWS_ACCESS_KEY sorts
    first.
    """
    region = _Region(0, 1, [])
    for secret_type in ("SLACK_TOKEN", "AWS_ACCESS_KEY", "SLACK_TOKEN", "JWT"):
        region.claim(secret_type)

    assert region.types == ["SLACK_TOKEN", "AWS_ACCESS_KEY", "JWT"], "arrival order kept"
    assert region.placeholder == "[REDACTED:AWS_ACCESS_KEY+JWT+SLACK_TOKEN]"
