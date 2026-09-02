"""Near-duplicate clustering, so a paraphrase family lands wholly on one side.

A random row split puts one paraphrase in training and its sibling in the
held-out half. The figure that comes back then measures recall of memorised
text, and nothing downstream can tell: the held-out rows look like ordinary
unseen data.

`training/split.py` already closes one shape of that leak, the twin: the two
rows one generation call produced, adjacent in `rows.jsonl`, alike on purpose.
This module closes the other shape, which is two rows from DIFFERENT calls that
came out saying the same thing. The generator's own `NearDuplicateIndex` drops a
row whose word trigrams overlap an accepted one, so what is left is the pair
that is close in meaning and far in wording, which no surface rule sees. An
embedding does see it.

The two are composed rather than chosen between. `coarsen` merges the twin
partition into the cluster partition, so the unit that gets assigned is a
paraphrase family with whole twins inside it, and a split over that unit
separates neither. `training/split.py::separated_twins` is what says so
afterwards, over the committed split rather than over this module's own output.

**No numpy, and that is a requirement rather than a preference.** The brief for
this task wrote these functions over `numpy.ndarray`. CI installs `.[dev]`,
which is pytest, ruff, mypy and PyYAML, and nothing else: there is no numpy on
any leg. `training/separability.py` reached the same wall first and records the
same answer. A module the test suite imports has to import in the environment
the test suite runs in, so the vectors here are lists of floats and the dot
product is a `sum`. The corpus is 3,584 rows and the greedy pass below is a few
million dot products, which is minutes rather than the seconds numpy would take,
and it runs once per corpus on a laptop.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import TypeVar

from training.split import shuffled

T = TypeVar("T")

Vector = Sequence[float]

#: Where a local Ollama answers. The same host `training/generate.py` posts to,
#: spelled the same way: `127.0.0.1` rather than `localhost`, because a machine
#: whose `localhost` resolves to `::1` first meets a connection refused that
#: reads as a missing model server.
ENDPOINT = "http://127.0.0.1:11434/api/embeddings"

#: The embedding model, as a reader would type it to obtain the same weights.
#: Registered in `training.generate.GENERATORS` with its licence and the sha256
#: of the blob the tag resolves to, because a tag is mutable and a clustering
#: derived from moved weights is a clustering nobody can reproduce.
EMBED_MODEL = "nomic-embed-text"

#: Cosine above which two rows are one cluster.
#:
#: The value the brief set, and it is deliberately far above the 0.6 word-
#: trigram threshold `training.generate.NEAR_DUPLICATE` deduplicates at. The two
#: numbers are not on the same scale and neither is a translation of the other:
#: 0.6 of the trigrams shared is a strong claim about wording, where cosine 0.92
#: between two normalised `nomic-embed-text` vectors is a moderate claim about
#: meaning. Set lower, this merges rows that are merely on the same topic, and
#: since a merged cluster is assigned whole, that costs the split its balance
#: for nothing.
THRESHOLD = 0.92


class ClusterError(RuntimeError):
    """The embeddings or the partitions handed here do not describe one corpus."""


def embed(
    texts: Sequence[str],
    model: str = EMBED_MODEL,
    endpoint: str = ENDPOINT,
    timeout: float = 120.0,
) -> list[list[float]]:
    """L2-normalised embeddings, so a dot product IS cosine similarity.

    One request per text rather than a batch. The endpoint accepts one prompt,
    the corpus is thousands of rows and not millions, and a loop that fails on
    row 900 fails naming row 900.

    Raises rather than returning a short list when the server answers something
    other than an embedding. A caller that got fewer vectors than texts and did
    not notice would align every row after the gap with the wrong text, and the
    clustering would look entirely normal.
    """
    vectors: list[list[float]] = []
    for index, text in enumerate(texts):
        payload = json.dumps({"model": model, "prompt": text}).encode()
        request = urllib.request.Request(
            endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ClusterError(f"{model} did not answer for text {index}: {error}") from error
        raw = body.get("embedding")
        if not isinstance(raw, list) or not raw:
            raise ClusterError(
                f"{model} returned no embedding for text {index}; the reply carried {sorted(body)}"
            )
        vectors.append(normalise([float(value) for value in raw]))
    return vectors


def normalise(vector: Vector) -> list[float]:
    """A unit vector, or the zero vector unchanged.

    A zero vector would divide by zero. Dividing anyway gives NaN, which
    compares false against every threshold, so the row would silently become a
    cluster of its own and the split would look fine. Returned as it came
    instead, which scores 0.0 against everything and reaches the same
    partition by an arithmetic a reader can follow.
    """
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def cosine(left: Vector, right: Vector) -> float:
    """The dot product of two vectors `normalise` has already been over."""
    if len(left) != len(right):
        raise ClusterError(f"a {len(left)}-dimensional vector against a {len(right)}-dimensional")
    return sum(a * b for a, b in zip(left, right))


def cluster_ids(vectors: Sequence[Vector], threshold: float = THRESHOLD) -> list[int]:
    """Greedy single-pass clustering against representatives.

    Greedy rather than agglomerative: the corpus is thousands of rows, the
    clusters that matter are tight paraphrase families, and a rule a reviewer
    can hold in their head is worth more here than a marginally better partition
    out of a library nobody checks.

    What greedy costs, stated rather than left to be discovered. The partition
    depends on the order the rows arrive in, and it is not transitive: A can
    join B's cluster and C can be close to A without joining it. That makes this
    an under-merger at the margin, never an over-merger, and under-merging is
    the safe direction for a split: it leaves a near-duplicate pair separable,
    which `coarsen` and the twin check can still catch, rather than collapsing
    the corpus into one cluster that cannot be divided at all.
    """
    ids: list[int] = []
    representatives: list[Vector] = []
    for vector in vectors:
        assigned = -1
        for index, representative in enumerate(representatives):
            if cosine(vector, representative) >= threshold:
                assigned = index
                break
        if assigned < 0:
            representatives.append(vector)
            assigned = len(representatives) - 1
        ids.append(assigned)
    return ids


def coarsen(*partitions: Sequence[int]) -> list[int]:
    """One partition no finer than any of its inputs, by union-find.

    The composition this task needs. A cluster id says which rows say the same
    thing; a twin id says which rows came out of one call. Splitting on either
    alone leaves the other leak open, and splitting on both separately is not a
    thing a split can do. Merging them first is: two rows share an id here if
    they share one in ANY input, so the unit that gets assigned is a paraphrase
    family with its twins whole inside it.

    Ids are renumbered from 0 in order of first appearance, so the result is
    comparable between runs and does not carry the input's numbering.
    """
    if not partitions:
        raise ClusterError("no partition to coarsen")
    sizes = {len(partition) for partition in partitions}
    if len(sizes) != 1:
        raise ClusterError(f"partitions of different lengths {sorted(sizes)} are not of one corpus")
    count = sizes.pop()

    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for partition in partitions:
        first: dict[int, int] = {}
        for position, group in enumerate(partition):
            anchor = first.setdefault(group, position)
            left, right = find(anchor), find(position)
            if left != right:
                parent[right] = left

    renumbered: dict[int, int] = {}
    out: list[int] = []
    for position in range(count):
        root = find(position)
        out.append(renumbered.setdefault(root, len(renumbered)))
    return out


def split_by_cluster(
    rows: Sequence[T],
    ids: Sequence[int],
    held_out_fraction: float = 0.2,
    seed: int = 20260831,
) -> tuple[list[int], list[int]]:
    """Assign whole clusters, never rows, and never split one across the line.

    Returns `(train_indices, held_out_indices)`, both sorted, together covering
    every row exactly once.

    The shuffle comes from `training.split.shuffled` rather than from `random`,
    so this split and the twin-wise one are reproducible from a seed by the same
    arithmetic and cannot drift apart under a later interpreter.

    Clusters are taken in shuffled order until the held-out side reaches its
    target, which overshoots by at most the size of the cluster that crossed the
    line. Trimming to the target exactly would mean splitting that cluster,
    which is the one thing this function exists not to do.
    """
    if len(rows) != len(ids):
        raise ClusterError(f"{len(rows)} rows against {len(ids)} cluster ids")
    if not 0.0 < held_out_fraction < 1.0:
        raise ClusterError(f"held_out_fraction {held_out_fraction} is not a share between 0 and 1")
    members: dict[int, list[int]] = {}
    for position, cluster in enumerate(ids):
        members.setdefault(cluster, []).append(position)

    order = sorted(members)
    shuffle = shuffled(len(order), seed)
    target = len(rows) * held_out_fraction
    held: list[int] = []
    for position in shuffle:
        if len(held) >= target:
            break
        held.extend(members[order[position]])
    chosen = set(held)
    train = [position for position in range(len(rows)) if position not in chosen]
    return train, sorted(held)


def separated_clusters(
    ids: Sequence[int], train: Sequence[int], held_out: Sequence[int]
) -> list[int]:
    """Cluster ids with rows on both sides, or with a row on neither. The check.

    Written to score ANY partition, not only one `split_by_cluster` made, for
    the reason its twin-wise counterpart `training.split.separated_twins`
    records: a later stage builds the split it actually trains on, and this is
    what fails when it builds it by row. A row on neither side, or on both, is
    the same defect wearing different clothes and is reported the same way.
    """
    on_left, on_right = set(train), set(held_out)
    if len(on_left) != len(train) or len(on_right) != len(held_out):
        raise ClusterError("a side names the same row twice, so the split is not a partition")
    broken: list[int] = []
    sides: dict[int, set[bool]] = {}
    for position, cluster in enumerate(ids):
        left, right = position in on_left, position in on_right
        if left == right:
            # On both sides, or on neither. Either way this row is not placed.
            broken.append(cluster)
            continue
        sides.setdefault(cluster, set()).add(left)
    broken.extend(cluster for cluster, seen in sides.items() if len(seen) > 1)
    return sorted(set(broken))
