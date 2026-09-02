"""The encoder stage 2b is fitted on, pinned by revision and by bytes.

A generator's licence reaches the corpus. A BACKBONE's licence reaches the
shipped artifact, because the fine-tuned weights are the released weights with
their parameters moved: whatever terms came with them travel into the wheel.
So this is the one model in the tree whose grant is not a question about
process. It is a question about what `jamjet-guardrails` is allowed to be.

Kept out of `training/train.py` on purpose. That module imports torch, which is
deliberately absent from the package's `.venv` and from CI, so a registry
living there could not be screened by the suite that screens everything else.
Nothing here imports anything heavier than the standard library.

**Three separate questions, answered separately.**

- *What does the card say.* `licence` is screened by
  `training.screen.licence_refusal`, the same allowlist every corpus in
  `training/sources.yaml` goes through. An unrecognised spelling is refused
  rather than assumed permissive.
- *What does the card say it came FROM.* `upstream_licence` is screened by the
  same allowlist. A permissive tag on a redistribution does not cure a
  restrictive one upstream, which is the finding
  `training/screen.py` was written around, and this repository has already met
  the model-shaped version of it: the 3B size of the generator family ships
  under `qwen-research` while the 14B size is Apache-2.0. Reading a licence off
  a sibling is how research weights reach a shipped artifact.
- *Are these the bytes that were screened.* `files` pins a sha256 per file, and
  `digest_mismatches` compares them against what is actually on disk. A
  revision id says which commit was read; only the digests say which bytes were
  loaded. `train.py` refuses to start when they disagree.

The declaration itself is pinned too. This model publishes no `LICENSE` file:
its grant is the `license:` field in the front matter of its card, so
`licence_sha256` is the sha256 of that card at `revision`. Pinning the name and
not the bytes would leave the whole finding resting on a file anybody with push
access can edit.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from training.fetch import HEX64, ROOT, sha256_of
from training.screen import licence_refusal

#: A commit id on the Hugging Face hub, which is a git sha1.
#:
#: Required rather than accepted alongside a tag. `main` is a moving target and
#: so is a branch name; a run pinned to one is a run nobody can repeat once
#: somebody pushes.
REVISION = re.compile(r"\A[0-9a-f]{40}\Z")

#: Where a pinned backbone is materialised. Under `data/`, which `.gitignore`
#: excludes: 90 MB of weights is not something this repository commits, and the
#: digests below are what make the copy on any one machine checkable.
LOCAL = ROOT / "data" / "backbone"

_HUB = "https://huggingface.co"


class BackboneError(RuntimeError):
    """The backbone is not the screened, pinned model this tree may fit on."""


@dataclass(frozen=True, slots=True)
class Backbone:
    """A pretrained encoder this repository fine-tunes and therefore ships.

    Registered rather than named in passing, for the reason `GENERATORS` in
    `training/generate.py` is a registry: the identifier scan over this tree
    accounts for a model by finding its entry, so a backbone swapped in without
    one fails rather than passing quietly.
    """

    #: The hub id, in the form the identifier scan can see.
    model_id: str
    #: The commit the weights were read at.
    revision: str
    #: The SPDX identifier, screened by `training/screen.py`.
    licence: str
    #: sha256 of the bytes that DECLARE that licence, at `revision`.
    licence_sha256: str
    #: Which file those bytes are, so the digest can be re-taken.
    licence_declared_in: str
    #: The model this one declares itself derived from, and its licence. Both
    #: screened: see the module docstring.
    upstream_id: str
    upstream_licence: str
    #: Trainable parameters, as counted from the loaded weights rather than
    #: read off a card. `training/train.py` records what it actually counted
    #: and `tests/test_training_data.py` holds the two equal.
    parameters: int
    #: When the card was read. A model card is editable; a licence read without
    #: a date is one nobody can re-check.
    read_on: str
    #: Every file the training run loads, by sha256. Not every file in the
    #: repository: pinning digests for weights nothing reads would be a longer
    #: list saying less.
    files: Mapping[str, str] = field(default_factory=dict)
    #: What the finding rests on, and what it does not reach.
    note: str = ""


MINILM = Backbone(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    licence="apache-2.0",
    licence_sha256="dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0",
    licence_declared_in="README.md",
    upstream_id="nreimers/MiniLM-L6-H384-uncased",
    upstream_licence="mit",
    parameters=22713986,
    read_on="2026-09-02",
    files={
        "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        "special_tokens_map.json": (
            "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"
        ),
        "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        "tokenizer_config.json": (
            "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b"
        ),
        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    },
    note=(
        "Apache-2.0, read on 2026-09-02 from the card at the pinned revision and hashed "
        "above. The card declares one licence and only one, `license: apache-2.0` in its "
        "front matter, and the repository publishes no LICENSE file at that revision, so "
        "the front matter IS the grant and its bytes are what is pinned. The card text "
        "carries no acceptable-use policy, no research-only term and no clause reaching "
        "model outputs or derivative weights, which is the question that had to be asked: "
        "a fine-tuned checkpoint is the released weights with their parameters moved, so a "
        "term of that kind would follow this model into an Apache-2.0 wheel rather than "
        "stopping at the training tree. Screened a second time one step up. The card's "
        "`base_model` and the weights' own `_name_or_path` both name "
        "`nreimers/MiniLM-L6-H384-uncased`, whose card declares MIT, and MIT is on the "
        "same allowlist; the hub API was read for both on 2026-09-02. What this does NOT "
        "establish is anything about the corpora behind either model. The card lists the "
        "training datasets by name and several of them carry their own terms; the grant "
        "covers the weights that were released, and a claim about what went into them is "
        "not available to anyone outside the people who trained it. 22.7M parameters, "
        "6 layers, hidden size 384, vocabulary 30522, counted from the loaded weights and "
        "read from the pinned config."
    ),
)

#: Every pretrained encoder this repository fine-tunes.
#:
#: One entry, and the tuple shape is not premature. `tests/test_training_data.py`
#: walks it to account for identifiers and to screen licences, and a registry
#: that had to be reshaped to hold a second model is a registry somebody adds
#: the second model beside instead.
BACKBONES: tuple[Backbone, ...] = (MINILM,)

#: The one stage 2b fits on.
BACKBONE = MINILM


def local_dir(backbone: Backbone = BACKBONE, into: Path = LOCAL) -> Path:
    """Where this backbone's pinned files live on a machine that has them."""
    return into / backbone.model_id.split("/")[-1]


def licence_refusals(backbone: Backbone = BACKBONE) -> list[str]:
    """Why this backbone may not be fitted on, or `[]` if it may.

    Both licences, in one pass, because they are one question. A backbone
    cleared on its own tag and refused one step up is still a backbone this
    repository cannot ship, and checking the two in separate places is how one
    of them comes to be checked and the other described.
    """
    found: list[str] = []
    for label, identifier, licence in (
        ("", backbone.model_id, backbone.licence),
        (" upstream ", backbone.upstream_id, backbone.upstream_licence),
    ):
        refusal = licence_refusal(licence)
        if refusal:
            found.append(f"the{label or ' '}model {identifier} is {licence}: {refusal}")
    return found


def pin_faults(backbone: Backbone = BACKBONE) -> list[str]:
    """Every way this entry fails to pin anything, or `[]` if it pins.

    Shape only, and separate from `digest_mismatches` for that reason: this
    says whether the record COULD identify a byte, and that says whether the
    bytes on disk are those ones. An entry with an empty `files` map passes
    every digest comparison there is, by having none to make.
    """
    found: list[str] = []
    if not REVISION.match(backbone.revision):
        found.append(
            f"{backbone.model_id} is pinned to {backbone.revision!r}, which is not a commit "
            "id; a branch or a tag can be repointed under the numbers measured through it"
        )
    if not HEX64.match(backbone.licence_sha256):
        found.append(
            f"{backbone.model_id} records {backbone.licence_sha256!r} as the digest of its "
            "licence declaration, which is not a sha256"
        )
    if not backbone.files:
        found.append(
            f"{backbone.model_id} pins no file digests, so every byte comparison against it "
            "passes by having nothing to compare"
        )
    for name, digest in sorted(backbone.files.items()):
        if not HEX64.match(digest):
            found.append(f"{backbone.model_id} records {digest!r} for {name}, not a sha256")
    if backbone.parameters <= 0:
        found.append(f"{backbone.model_id} records {backbone.parameters} parameters")
    return found


def digest_mismatches(backbone: Backbone, directory: Path) -> list[str]:
    """Every pinned file that is missing from `directory` or is different there.

    Missing counts as a mismatch and is reported as one. Skipping an absent
    file returns the same empty list a clean directory does, so a download that
    fetched half of what it was asked for would read as verified.
    """
    found: list[str] = []
    for name, expected in sorted(backbone.files.items()):
        path = directory / name
        if not path.is_file():
            found.append(
                f"{name} is not in {directory}, so its digest is unknown rather than {expected}"
            )
            continue
        actual = sha256_of(path)
        if actual != expected:
            found.append(f"{name} hashes to {actual}, and {backbone.model_id} pins {expected}")
    return found


def download(backbone: Backbone = BACKBONE, into: Path | None = None) -> Path:
    """Fetch the pinned files, by revision, for any that are not already here.

    Written rather than left to the hub client so the run has one obtainable
    description: a URL carrying a commit id fetches the same bytes forever,
    where a cache populated by an id and a tag depends on when the cache was
    filled. Verification is `verified` below and happens either way, so a file
    left behind by an interrupted download is caught rather than reused.
    """
    directory = local_dir(backbone) if into is None else into
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(backbone.files):
        target = directory / name
        if target.is_file():
            continue
        url = f"{_HUB}/{backbone.model_id}/resolve/{backbone.revision}/{name}"
        with urllib.request.urlopen(url) as response, target.open("wb") as handle:
            while block := response.read(1 << 16):
                handle.write(block)
    return directory


def verified(backbone: Backbone = BACKBONE, directory: Path | None = None) -> Path:
    """The directory to load from, or `BackboneError` saying why not.

    All three checks, raised together. A caller told only the first reason
    fixes it, re-runs, and is told the second, which is how a screen comes to
    feel like an obstacle instead of a finding.
    """
    at = local_dir(backbone) if directory is None else directory
    found = licence_refusals(backbone) + pin_faults(backbone) + digest_mismatches(backbone, at)
    if found:
        raise BackboneError("; ".join(found))
    return at


def described(backbone: Backbone = BACKBONE) -> dict[str, object]:
    """The backbone, in the form a run record carries it.

    Every field a later reader needs to obtain the same weights, and nothing
    derived: the run record repeats the pin rather than summarising it, so
    `tests/test_training_data.py` can hold the two equal.
    """
    return {
        "model_id": backbone.model_id,
        "revision": backbone.revision,
        "licence": backbone.licence,
        "licence_sha256": backbone.licence_sha256,
        "licence_declared_in": backbone.licence_declared_in,
        "upstream_id": backbone.upstream_id,
        "upstream_licence": backbone.upstream_licence,
        "parameters": backbone.parameters,
        "read_on": backbone.read_on,
        "files": dict(sorted(backbone.files.items())),
    }


def identifiers(backbones: Sequence[Backbone] = BACKBONES) -> list[str]:
    """Every hub id these entries account for, screened one and upstream both."""
    return sorted({name for entry in backbones for name in (entry.model_id, entry.upstream_id)})
