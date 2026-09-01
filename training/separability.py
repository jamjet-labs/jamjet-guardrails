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
    if not features or len(set(labels)) < 2:
        return 0.0
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
    return correct / len(features)


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
