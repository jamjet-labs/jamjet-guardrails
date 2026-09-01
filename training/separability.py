"""Can the corpus be sorted without reading it?

The lexical screen in `tests/test_training_data.py` asks whether one word
decides the label. It answered 0.470, below chance, and the corpus it answered
about was separable at 0.712 by style alone and 0.793 by function words alone.
Both statements were true. The screen was not wrong, it was narrow: it measured
one axis and was read as though it measured the corpus.

This module measures the two axes that screen cannot see.

**Style.** Lengths, punctuation counts, character-class ratios, a stopword
ratio. No content word reaches the model at all, so a score above chance means
the two classes are written differently rather than saying different things.

**Function words.** Counts of `the`, `a`, `you`, `to`, and the rest of the
closed class. These carry almost no topic and a great deal of register, which is
what makes them the standard authorship-attribution feature and the right probe
for "did one generator write these two classes in two voices".

Why it matters more than it looks. The reference models this stage is measured
against were never fitted on our corpus, so a generation artifact lifts OUR
score and not theirs. A ship bar cleared on a corpus like that measures our
model's ability to exploit the artifact, which is the one thing the bar exists
to rule out.

Pure Python and no dependency, by the same rule as the rest of the tree: this
runs in CI, and CI has neither numpy nor scikit-learn. Logistic regression by
full-batch gradient descent on standardised features, k-fold cross-validated.
The absolute numbers are not comparable to a tuned scikit-learn fit and are not
meant to be; what they have to do is move when the corpus becomes separable,
and stay near the majority baseline when it is not.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

#: The closed class. Function words carry register rather than subject, which is
#: why authorship attribution has leant on them since Mosteller and Wallace: two
#: texts about the same thing in two voices differ here, and two texts about
#: different things in one voice do not.
FUNCTION_WORDS: tuple[str, ...] = (
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "more",
    "most",
    "my",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
)

_WORD = re.compile(r"[A-Za-z']+")
_FUNCTION_INDEX = {word: index for index, word in enumerate(FUNCTION_WORDS)}


def style_features(text: str) -> list[float]:
    """Shape only. Nothing here can see which words a text used.

    Every feature is a length, a count of a punctuation mark, a ratio of
    character classes, or the proportion of tokens that are function words. A
    model fitted on these and scoring above the majority baseline has learned
    how the two classes are WRITTEN, because it was never shown what they say.
    """
    words = _WORD.findall(text)
    length = max(len(text), 1)
    count = max(len(words), 1)
    stop = sum(1 for word in words if word.casefold() in _FUNCTION_INDEX)
    return [
        len(text),
        len(words),
        sum(len(word) for word in words) / count,
        text.count("'") / length,
        text.count('"') / length,
        text.count(",") / length,
        text.count(".") / length,
        text.count(":") / length,
        text.count(";") / length,
        text.count("?") / length,
        text.count("!") / length,
        text.count("\n") / length,
        text.count("-") / length,
        sum(character.isdigit() for character in text) / length,
        sum(character.isupper() for character in text) / length,
        sum(not character.isalnum() and not character.isspace() for character in text) / length,
        stop / count,
        float(text[:1].isupper()),
        float(text.endswith((".", "!", "?"))),
    ]


def function_word_features(text: str) -> list[float]:
    """Rates of each closed-class word. No content word survives this."""
    words = _WORD.findall(text)
    total = max(len(words), 1)
    counts = [0.0] * len(FUNCTION_WORDS)
    for word in words:
        index = _FUNCTION_INDEX.get(word.casefold())
        if index is not None:
            counts[index] += 1.0
    return [count / total for count in counts]


def majority_baseline(labels: Sequence[int]) -> float:
    """What always guessing the commoner class scores. The number to beat."""
    if not labels:
        return 0.0
    ones = sum(labels)
    return max(ones, len(labels) - ones) / len(labels)


def _standardise(
    train: list[list[float]], test: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    """Centre and scale by the TRAINING fold's statistics only.

    Using the whole corpus to standardise would leak the test fold into the fit.
    The leak is small for these features and it is exactly the kind of thing
    this module exists to complain about, so it is not done here either.
    """
    width = len(train[0])
    means = [sum(row[i] for row in train) / len(train) for i in range(width)]
    deviations = []
    for i in range(width):
        variance = sum((row[i] - means[i]) ** 2 for row in train) / len(train)
        deviations.append(math.sqrt(variance) or 1.0)

    def scale(rows: list[list[float]]) -> list[list[float]]:
        return [[(row[i] - means[i]) / deviations[i] for i in range(width)] for row in rows]

    return scale(train), scale(test)


def _fit(
    features: list[list[float]], labels: list[int], epochs: int, rate: float
) -> tuple[list[float], float]:
    """Logistic regression, full-batch gradient descent, no regularisation."""
    width = len(features[0])
    weights = [0.0] * width
    bias = 0.0
    scale = rate / len(features)
    for _ in range(epochs):
        gradient = [0.0] * width
        bias_gradient = 0.0
        for row, label in zip(features, labels):
            total = bias
            for i in range(width):
                total += weights[i] * row[i]
            # Guard the exponent rather than trusting the optimiser to stay in
            # range: one overflow makes the whole score meaningless and silent.
            error = (1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))) - label
            bias_gradient += error
            for i in range(width):
                gradient[i] += error * row[i]
        for i in range(width):
            weights[i] -= scale * gradient[i]
        bias -= scale * bias_gradient
    return weights, bias


def cross_validated_accuracy(
    features: list[list[float]],
    labels: list[int],
    folds: int = 5,
    seed: int = 42,
    epochs: int = 25,
    rate: float = 4.0,
) -> float:
    """k-fold accuracy of a logistic regression over these features.

    Folds are assigned by a seeded shuffle so the number is reproducible, and
    stratification is not attempted: the corpus is balanced by construction now
    that pairs are generated together, and a fold imbalance would show up in the
    majority baseline this is compared against.
    """
    correct, total = _cross_validated(features, labels, folds, seed, epochs, rate)
    return correct / total if total else 0.0


def _cross_validated(
    features: list[list[float]],
    labels: list[int],
    folds: int,
    seed: int,
    epochs: int,
    rate: float,
) -> tuple[int, int]:
    """Held-out hits and rows, so a caller can pool several fits honestly.

    Pooling accuracies by averaging them weights a 60-row group like a 600-row
    one. Pooling the counts is the same number a single fit over the union would
    have produced, which is what `conditional_accuracy` has to report.
    """
    if not features or len(set(labels)) < 2:
        return 0, 0
    order = list(range(len(features)))
    state = seed
    # A small deterministic shuffle, so the fold assignment does not depend on
    # the platform's random module changing under us.
    for i in range(len(order) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (1 << 31)
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]

    correct = 0
    for fold in range(folds):
        test_ids = [order[i] for i in range(fold, len(order), folds)]
        test_set = set(test_ids)
        train_ids = [i for i in order if i not in test_set]
        train_x, test_x = _standardise(
            [features[i] for i in train_ids], [features[i] for i in test_ids]
        )
        train_y = [labels[i] for i in train_ids]
        weights, bias = _fit(train_x, train_y, epochs, rate)
        for row, index in zip(test_x, test_ids):
            total = bias + sum(weights[i] * row[i] for i in range(len(weights)))
            correct += int((total >= 0.0) == bool(labels[index]))
    return correct, len(features)


def first_token_purity(
    texts: Sequence[str], labels: Sequence[int], minimum: int = 5
) -> dict[str, tuple[int, float]]:
    """Openers that decide the label, and how completely.

    Position 1 is its own leak, separate from anything the whole-text screens
    see. In the corpus this replaced, "ignore" appeared in 137 negatives against
    96 attacks, which is healthy, while every one of the 38 rows that BEGAN with
    "Ignore" was a negative. A classifier reading position 1 learned that a
    message opening "Ignore ..." is safe, which is the exact inverse of the
    truth.

    Returns opener -> (rows, purity), for openers appearing at least `minimum`
    times.
    """
    seen: dict[str, list[int]] = {}
    for text, label in zip(texts, labels):
        match = _WORD.search(text)
        token = match.group(0).casefold() if match else ""
        seen.setdefault(token, []).append(label)
    out: dict[str, tuple[int, float]] = {}
    for token, found in seen.items():
        if len(found) < minimum:
            continue
        share = sum(found) / len(found)
        out[token] = (len(found), max(share, 1.0 - share))
    return out


def surface_features(text: str) -> list[float]:
    """Everything a model can see WITHOUT reading a content word.

    Style and function words in one vector rather than two separate probes, and
    that is the point of it. Fitted apart, the two answer "is the shape a tell"
    and "is the register a tell"; fitted together they answer the question the
    corpus is actually held to, which is whether the label survives being
    written down with every content word removed.
    """
    return style_features(text) + function_word_features(text)


def conditional_accuracy(
    features: list[list[float]],
    labels: list[int],
    groups: Sequence[str],
    folds: int = 5,
    seed: int = 42,
    epochs: int = 25,
    rate: float = 4.0,
) -> float:
    """The same probe, fitted SEPARATELY INSIDE each group and pooled.

    This is the number the marginal probes cannot see, and it is the one that
    predicts what a fine-tuned encoder can take for free.

    A marginal probe has to find one direction that sorts the whole corpus. An
    encoder does not: it reads the topic first, which tells it which pair a row
    came from, and it can then apply a different rule per pair. Every prompt
    pair here is a topic, so pair identity is free. Conditioning on it is
    therefore not a pessimistic reading of the corpus, it is the realistic one.

    Measured on the corpus this replaced, the gap was the whole finding: style
    and function words together scored 0.805 conditioned on the pair against a
    marginal 0.734, and a bag of words reading every content word scored 0.816.
    A model that never learns what an injection is could score 0.805 on that
    evaluation, and the marginal number reported 0.734 and looked like progress.

    Pair identity on its own scores 0.500, because every pair is balanced, so
    nothing here is measuring the grouping itself.
    """
    hits = 0
    rows = 0
    for group in sorted(set(groups)):
        chosen = [i for i, name in enumerate(groups) if name == group]
        correct, total = _cross_validated(
            [features[i] for i in chosen], [labels[i] for i in chosen], folds, seed, epochs, rate
        )
        hits += correct
        rows += total
    return hits / rows if rows else 0.0


def leave_one_group_out_accuracy(
    features: list[list[float]],
    labels: list[int],
    groups: Sequence[str],
    epochs: int = 25,
    rate: float = 4.0,
) -> dict[str, float]:
    """Fit on every group but one, score the one. Per group.

    The test that separates a phenomenon from an artifact, and the one probe
    here whose answer is allowed to be low. An attack is an attack whatever it
    is about, so a signal that is the phenomenon transfers to a kind of attack
    the fit never saw. A signal that is an artifact of one prompt's wording is
    local to that prompt and transfers nowhere; a group that scores BELOW
    chance is carrying a direction opposite to the rest of the corpus, which is
    a wording, not a phenomenon.

    Not gated on, and deliberately so. A high transfer number would be good news
    and a low one is ambiguous: eight prompt pairs are eight kinds of attack, and
    some of the gap between them is real difference between kinds rather than
    artifact. It is reported so the split between the two can be argued about
    with a number in hand.
    """
    out: dict[str, float] = {}
    for group in sorted(set(groups)):
        train = [i for i, name in enumerate(groups) if name != group]
        test = [i for i, name in enumerate(groups) if name == group]
        if not train or not test or len({labels[i] for i in train}) < 2:
            continue
        train_x, test_x = _standardise([features[i] for i in train], [features[i] for i in test])
        weights, bias = _fit(train_x, [labels[i] for i in train], epochs, rate)
        correct = 0
        for row, index in zip(test_x, test):
            total = bias + sum(weights[i] * row[i] for i in range(len(weights)))
            correct += int((total >= 0.0) == bool(labels[index]))
        out[group] = correct / len(test)
    return out


def bag_of_words(texts: Sequence[str]) -> tuple[list[list[int]], int]:
    """Every word in the corpus, as presence indices per row.

    Presence rather than a count, and the whole vocabulary rather than a capped
    one, because this probe exists to be the CEILING the other probes are read
    against: what does reading every word buy over reading none of the content?
    A capped vocabulary would make that ceiling depend on the cap.

    Sparse indices rather than a dense vector because the vocabulary runs to
    thousands of words and this has to run in CI with no numpy under it.
    """
    vocabulary: dict[str, int] = {}
    rows: list[list[int]] = []
    for text in texts:
        present = set()
        for word in _WORD.findall(text):
            folded = word.casefold()
            index = vocabulary.get(folded)
            if index is None:
                index = len(vocabulary)
                vocabulary[folded] = index
            present.add(index)
        rows.append(sorted(present))
    return rows, len(vocabulary)


def _fit_sparse(
    rows: list[list[int]], labels: list[int], width: int, epochs: int, rate: float
) -> tuple[list[float], float]:
    """Logistic regression over presence indices. Same optimiser, no dense row.

    No standardisation: centring a presence vector destroys the sparsity that
    makes this affordable, and the features are already on one scale because
    every one of them is 0 or 1.
    """
    weights = [0.0] * width
    bias = 0.0
    scale = rate / max(len(rows), 1)
    for _ in range(epochs):
        gradient: dict[int, float] = {}
        bias_gradient = 0.0
        for row, label in zip(rows, labels):
            total = bias
            for index in row:
                total += weights[index]
            error = (1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))) - label
            bias_gradient += error
            for index in row:
                gradient[index] = gradient.get(index, 0.0) + error
        for index, value in gradient.items():
            weights[index] -= scale * value
        bias -= scale * bias_gradient
    return weights, bias


def _sparse_cross_validated(
    rows: list[list[int]],
    labels: list[int],
    width: int,
    folds: int,
    seed: int,
    epochs: int,
    rate: float,
) -> tuple[int, int]:
    """k-fold over presence indices. Folds assigned exactly as the dense probe."""
    if not rows or len(set(labels)) < 2:
        return 0, 0
    order = list(range(len(rows)))
    state = seed
    for i in range(len(order) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (1 << 31)
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    correct = 0
    for fold in range(folds):
        test_ids = [order[i] for i in range(fold, len(order), folds)]
        test_set = set(test_ids)
        train_ids = [i for i in order if i not in test_set]
        weights, bias = _fit_sparse(
            [rows[i] for i in train_ids], [labels[i] for i in train_ids], width, epochs, rate
        )
        for index in test_ids:
            total = bias + sum(weights[i] for i in rows[index])
            correct += int((total >= 0.0) == bool(labels[index]))
    return correct, len(rows)


def bag_of_words_accuracy(
    texts: Sequence[str],
    labels: list[int],
    groups: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
    epochs: int = 120,
    rate: float = 8.0,
) -> float:
    """What reading every word buys, marginally or conditioned on the group.

    Reported rather than gated. It is not a defect for a model that reads the
    content to sort an injection corpus: that is the corpus working. The number
    matters as the comparison the content-blind probes are read against, because
    "function words score 0.73" means one thing when everything scores 0.95 and
    another when everything scores 0.76.
    """
    rows, width = bag_of_words(texts)
    if groups is None:
        correct, total = _sparse_cross_validated(rows, labels, width, folds, seed, epochs, rate)
        return correct / total if total else 0.0
    hits = 0
    counted = 0
    for group in sorted(set(groups)):
        chosen = [i for i, name in enumerate(groups) if name == group]
        correct, total = _sparse_cross_validated(
            [rows[i] for i in chosen],
            [labels[i] for i in chosen],
            width,
            folds,
            seed,
            epochs,
            rate,
        )
        hits += correct
        counted += total
    return hits / counted if counted else 0.0


def single_token_separability(
    texts: Sequence[str], labels: Sequence[int], groups: Sequence[str]
) -> dict[str, tuple[float, str]]:
    """The best one-word rule INSIDE each group, and the word that does it.

    Scored as balanced accuracy, which for a rule reading one present-or-absent
    feature is exactly the ROC AUC of that feature: (share of attacks holding
    the word + share of negatives lacking it) / 2. Both polarities are tried and
    the better one reported, because "holds the word, therefore benign" sorts a
    pair as thoroughly as its opposite and the previous corpus's worst tell was
    that way round.

    Balanced rather than raw accuracy so that the number does not move with a
    group's label mix, and over the WHOLE vocabulary rather than a list of
    suspects. The list is what the previous round measured: it named `was` and
    `decode`, fixed both, and the wording that fixed one of them put `assistant`
    at 0.925 in the same pair. A screen that can only see the words its author
    thought of measures its author.

    A rare word cannot score highly here and needs no minimum-frequency cutoff:
    a word in one attack row of 213 and in no negative scores 0.502.
    """
    out: dict[str, tuple[float, str]] = {}
    for group in sorted(set(groups)):
        chosen = [i for i, name in enumerate(groups) if name == group]
        attacks = sum(labels[i] for i in chosen)
        negatives = len(chosen) - attacks
        if not attacks or not negatives:
            continue
        counts: dict[str, list[int]] = {}
        for i in chosen:
            for word in {token.casefold() for token in _WORD.findall(texts[i])}:
                seen = counts.setdefault(word, [0, 0])
                seen[labels[i]] += 1
        best = (0.0, "")
        for word, (in_negatives, in_attacks) in counts.items():
            score = (in_attacks / attacks + 1.0 - in_negatives / negatives) / 2.0
            score = max(score, 1.0 - score)
            if score > best[0]:
                best = (score, word)
        out[group] = best
    return out


def twin_similarity(texts: Sequence[str]) -> list[float]:
    """Word-trigram Jaccard between the two members of each adjacent twin.

    The hazard the pairing itself creates. Both members of a twin come out of
    one call asking for them to be alike, and a split made row by row then
    puts one member on each side: the model has seen most of the held-out text
    already, under the opposite label. What it learns from that is the local
    difference between two near-copies, which generalises nowhere and scores
    well on a split built the obvious way.

    Two mitigations, and this measures the first. Cap the similarity, so the two
    members share an opening and a register rather than a paragraph; and split
    by twin rather than by row, which `training/split.py` does.
    """
    out: list[float] = []
    for start in range(0, len(texts) - 1, 2):
        left, right = _shingles(texts[start]), _shingles(texts[start + 1])
        union = len(left | right)
        out.append(len(left & right) / union if union else 0.0)
    return out


def _shingles(text: str) -> frozenset[tuple[str, ...]]:
    """Word trigrams, punctuation and case removed.

    The same shingles `training.generate.NearDuplicateIndex` compares rows with,
    reimplemented here rather than imported because this module is the one that
    runs against a corpus it did not generate: importing the generator would
    make a screen over the rows depend on the thing that wrote them.
    """
    words = re.sub(r"[^a-z0-9 ]+", " ", text.casefold()).split()
    if len(words) < 3:
        return frozenset({(word,) for word in words})
    return frozenset(tuple(words[i : i + 3]) for i in range(len(words) - 2))
