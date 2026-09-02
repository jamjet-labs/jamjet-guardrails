"""The licence screen: what a corpus may be used for, decided twice.

A corpus reaches this repository past two different questions, and Phase 1
learned the hard way that only one of them is answered by a licence field.

**What the metadata says.** `licence_refusal` is an allowlist, not a denylist.
An identifier it does not recognise is refused, so a corpus tagged `other`, a
corpus tagged nothing at all, and a corpus whose card declares two different
licences all come out the same way: no established grant, no use. A denylist of
forbidden terms is the shape that fails open, because the spelling nobody
thought of reads as clean.

**What the values say.** Two corpora advertising MIT both derived from Fake
Name Generator identities under dual GPLv3 / CC-BY-SA-3.0-US, verified through
pixie-io's generator back to
<https://www.fakenamegenerator.com/license.php>. Every licence field in
that chain was clean. An Apache-2.0 tag downstream does not cure a share-alike
upstream, and nothing but the data itself says so.

The two halves are independent, and a corpus has to pass both. `beki/privy` is
in `training/sources.yaml` as the worked example of why: its licence field
passes `licence_refusal` and its rows do not pass `fingerprint_hits`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_EXCERPT = 60

# The ten Fake Name Generator house domains. A corpus carrying these is derived
# from FNG identities whatever its metadata claims.
#
# The same ten are enforced over the committed corpora by
# `tests/test_corpora.py`, and published in `docs/conformance.md`. Three
# copies of one list is two too many, so the copies are held equal by
# `test_the_house_domains_here_are_the_ones_the_corpus_screen_rejects`, and
# `docs/conformance.md` is tied to the corpus screen by
# `tests/test_conformance_doc.py`.
_FNG_DOMAINS = (
    "cuvox.de",
    "dayrep.com",
    "einrot.com",
    "fleckens.hu",
    "gustr.com",
    "jourrapide.com",
    "rhyta.com",
    "superrito.com",
    "teleworm.us",
    "armyspy.com",
)

FINGERPRINTS: dict[str, tuple[re.Pattern[str], str]] = {
    "fake_name_generator": (
        # Case-insensitive because a domain is. The generator issues its
        # addresses in lower case, but a corpus that title-cased a column, or
        # quoted one inside a sentence, carries the same share-alike values
        # under a spelling a case-sensitive pattern reads as clean. A screen
        # that only sees the tidy spelling of its own example is inert against
        # the file it was written for.
        re.compile("|".join(re.escape(domain) for domain in _FNG_DOMAINS), re.IGNORECASE),
        (
            "Fake Name Generator house domains. FNG identities are dual GPLv3 / "
            "CC-BY-SA-3.0-US, and two corpora advertising MIT were found to derive "
            "from them in Phase 1. Share-alike upstream is not cured downstream."
        ),
    ),
}


def fingerprint_hits(rows: Iterable[str]) -> dict[str, list[str]]:
    """Which fingerprints a corpus trips, with bounded excerpts.

    Excerpts are bounded and the whole row is never returned: a screen that
    prints what it found has published the thing it was written to keep out.
    """
    hits: dict[str, list[str]] = {}
    for row in rows:
        for name, (pattern, _reason) in FINGERPRINTS.items():
            match = pattern.search(row)
            if match is None:
                continue
            start = max(0, match.start() - _EXCERPT // 2)
            hits.setdefault(name, []).append(row[start : start + _EXCERPT])
    return hits


#: What a licence still asks of us once it has been accepted.
NO_CONDITION = "none"
ATTRIBUTION = "attribution"

#: The two conditions a usable licence may carry. Every entry in
#: `USABLE_LICENCES` records one of them, so adding a licence is a decision
#: about attribution rather than a name appended to a set.
CONDITIONS = (NO_CONDITION, ATTRIBUTION)

#: Every licence a corpus may carry and still be trained from or measured on
#: here, keyed by its normalised SPDX identifier.
#:
#: An ALLOWLIST, and that is the whole design. `jamjet-guardrails` is
#: Apache-2.0 and ships a model to people who will use it commercially, so a
#: non-commercial, share-alike, research-only or undeclared corpus cannot be
#: fitted into that artifact. A screen written as a list of forbidden terms
#: would have to anticipate every spelling of every restriction; this one has
#: to anticipate nothing, because anything it does not recognise is refused.
#:
#: The condition is recorded per entry because it is what
#: `corpora/NOTICE.md` has to discharge. MIT and Apache-2.0 are permissive and
#: still require their notices to travel; only a public-domain dedication asks
#: for nothing.
USABLE_LICENCES: dict[str, str] = {
    "apache-2.0": ATTRIBUTION,
    "mit": ATTRIBUTION,
    "bsd-2-clause": ATTRIBUTION,
    "bsd-3-clause": ATTRIBUTION,
    "cc-by-3.0": ATTRIBUTION,
    "cc-by-4.0": ATTRIBUTION,
    "odc-by-1.0": ATTRIBUTION,
    "cc0-1.0": NO_CONDITION,
    "unlicense": NO_CONDITION,
}

#: Why a licence this repository has actually met is refused. The allowlist
#: above is what decides; this only names the term, so that a refusal reads as
#: a reason rather than as an absence. A spelling absent from both is refused
#: too, by `_UNRECOGNISED`.
#:
#: `none-declared` and `conflicting` are not SPDX identifiers and are not
#: pretending to be. They are the two things a real dataset card did instead of
#: declaring a licence, and `training/sources.yaml` records them verbatim so
#: that the manifest never has to guess a grant into existence.
REFUSED_LICENCES: dict[str, str] = {
    "cc-by-nc-4.0": (
        "non-commercial: the licence forbids commercial use, and a model shipped inside an "
        "Apache-2.0 library is used commercially by whoever installs it"
    ),
    "cc-by-nc-sa-4.0": "non-commercial and share-alike, either of which alone is disqualifying",
    "cc-by-sa-3.0": (
        "share-alike: a work derived from it has to carry the same licence, which an "
        "Apache-2.0 distribution cannot do"
    ),
    "cc-by-sa-3.0-us": "share-alike, the licence Fake Name Generator identities carry",
    "cc-by-sa-4.0": (
        "share-alike, as above; the version number moves the compatibility question and "
        "not the answer"
    ),
    "gpl-3.0": "copyleft: the same objection as share-alike, in stronger terms",
    "agpl-3.0": (
        "copyleft, and reaching further than GPLv3: it follows the work across a network "
        "boundary, so a hosted service built on it is a distribution too"
    ),
    "qwen-research": (
        "research-only, and named here rather than left to the allowlist default. Refusing it "
        "as an unrecognised spelling reaches the same answer for the wrong reason, and the "
        "difference matters: a reader told only that it was 'not screened' may screen it and "
        "add it, where a reader told it restricts use to research will not. The 3B size of the "
        "Qwen2.5 generation carries it while the 14B size this repository generates with is "
        "Apache-2.0, so the two sit one tag apart"
    ),
    "research-only": (
        "research-only: whatever the exact wording, a term restricting use to research "
        "excludes the people this library is published for"
    ),
    "other": (
        "the hub's catch-all tag, which names no terms at all; a licence nobody can read is "
        "not a grant anybody can rely on"
    ),
    "none-declared": (
        "no licence at all. Silence is not permission: copyright is the default, and a corpus "
        "with no grant is one nobody has been given the right to use"
    ),
    "conflicting": (
        "the source declares more than one licence and they are not the same licence, so which "
        "grant applies is not established. Picking the convenient one is not reading a licence"
    ),
}

_UNRECOGNISED = (
    "not a licence identifier this repository has screened. The rule is an allowlist, so an "
    "unrecognised spelling is refused rather than assumed permissive; record the SPDX "
    "identifier, or `none-declared` or `conflicting` where the source gives no single one"
)


def normalise_licence(license: str) -> str:
    """A licence identifier as the screen compares it.

    Case only. SPDX identifiers are matched case-insensitively by the
    specification and the Hugging Face hub writes its tags in lower case, so
    `Apache-2.0` and `apache-2.0` are the same grant and both are spellings a
    real manifest carries.

    Nothing else is folded, and that is deliberate rather than unfinished. A
    normaliser that also mapped spaces or dropped words would be inventing
    equivalences it cannot check -- and it does not need to, because an
    unrecognised spelling is refused by the allowlist rather than admitted.
    """
    return license.strip().casefold()


def licence_refusal(license: str) -> str:
    """Why this licence cannot be trained from or measured on, or `""` if it can.

    Returns a reason rather than a bool so that a refusal can be recorded where
    it happens. A screen that answers only yes or no leaves whoever hits it
    guessing which of several terms was the problem.
    """
    key = normalise_licence(license)
    if key in USABLE_LICENCES:
        return ""
    return REFUSED_LICENCES.get(key, _UNRECOGNISED)


def requires_attribution(license: str) -> bool:
    """Whether using this corpus obliges us to name it in `corpora/NOTICE.md`.

    False for a licence that is not usable at all. There is nothing to
    attribute in a corpus this repository may not use, and reporting one as
    needing a NOTICE entry would invite somebody to add the entry and consider
    the question settled.
    """
    return USABLE_LICENCES.get(normalise_licence(license)) == ATTRIBUTION
