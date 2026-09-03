"""Guards on `training/artifacts/ship_check.json`, the verdict the stage ends on.

The bar is guarded next door, in `tests/test_ship_bar.py`, and for the opposite
reason: that file has to still say what it said before any model existed. This
one has to still say what the harness returned. Between them sits the only
interesting failure mode left, which is a verdict that drifts away from the
numbers it was computed from while both files go on looking right.

So nothing here re-states a rate. Every rate is re-derived: the F1 from the
counts through `training/ship_bar.py:f1`, the comparison from the bar's own
`our_minimum` and `our_minimum_comparison`, the structural side by rerunning
`structural_floor` over the shipped corpus for real. And the ship decision is
checked as a truth table over all three sides rather than as the one row this
run happened to land on, because the row we landed on is the one row where a
conjunction written as `or` would still have produced the right answer.

Nothing here needs a network, a model or `onnxruntime`. The semantic side is
read from the artifact, which is why it is held to the counts beside it; the
structural side is recomputed, which needs only the package and a committed
corpus.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from test_ship_bar import (
    SHIP_BAR_CORE_SHA256,
    SHIP_BAR_SHA256_AT_VERDICT,
    _core_digest,
)
from training.decide import EXPORT_RECORD, METRICS, SHIPPED, verdict
from training.evalset import EVAL_SOURCE
from training.fetch import load_sources, sha256_of
from training.ship_bar import (
    COMPARISON,
    FLOOR_PLACES,
    SHIP_BAR,
    STRUCTURAL_CORPUS_VERSION,
    f1,
    harness,
    structural_floor,
)

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "training" / "artifacts" / "ship_check.json"
SPLITS = ROOT / "training" / "generated" / "splits.json"
MANIFEST = ROOT / "training" / "sources.yaml"
RESULTS = ROOT / "docs" / "measurements" / "2026-08-31-phase2b1-results.md"


def check() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(RECORD.read_text(encoding="utf-8"))
    return loaded


def bar() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(SHIP_BAR.read_text(encoding="utf-8"))
    return loaded


def test_the_verdict_exists_and_names_every_field_the_decision_rests_on() -> None:
    """Vacuity guard. Every test below reads this file and would pass on a stub."""
    recorded = check()
    for key in (
        "measured_utc",
        "judged_against",
        "model",
        "corpus",
        "harness",
        "semantic",
        "structural",
        "references",
        "verdict",
    ):
        assert key in recorded, f"ship_check.json is missing {key!r}"
    for key in ("controls", "semantic", "structural", "ship_the_classifier"):
        assert key in recorded["verdict"], f"the verdict is missing {key!r}"


def test_the_rates_are_the_counts_and_not_a_second_record_of_them() -> None:
    """Precision, recall and F1 re-derived through the function that wrote them.

    A rate written beside the counts it came from is a number that survives an
    edit to either one. Re-derived rather than compared to a literal, so this
    test says the artifact is internally consistent and does not smuggle in a
    second copy of the result.
    """
    semantic = check()["semantic"]
    counts = {key: int(semantic["counts"][key]) for key in ("tp", "fp", "fn")}
    recomputed = f1(counts)
    for rate in ("precision", "recall", "f1"):
        assert semantic[rate] == recomputed[rate], (
            f"the artifact records {rate} {semantic[rate]} and the counts beside it give "
            f"{recomputed[rate]}"
        )


def test_the_semantic_side_is_the_comparison_the_bar_asks_for() -> None:
    """Strictly greater, against the bar's own number, with no tolerance.

    The bar is read for all three of the value, the operator and the outcome. A
    verdict that recorded its own minimum, or applied `>=` where the bar says
    `>`, would be a bar this task set for itself.
    """
    recorded = check()["verdict"]["semantic"]
    recorded_bar = bar()
    minimum = float(recorded_bar["our_minimum"])
    assert recorded["minimum"] == minimum, (
        f"the verdict was taken against {recorded['minimum']!r} and the bar records {minimum!r}"
    )
    assert recorded["comparison"] == recorded_bar["our_minimum_comparison"] == COMPARISON
    measured = float(check()["semantic"]["f1"])
    assert recorded["measured"] == measured
    assert recorded["clears"] is (measured > minimum), (
        f"the verdict says clears={recorded['clears']} for {measured!r} {COMPARISON} {minimum!r}"
    )
    assert recorded["margin"] == measured - minimum
    assert recorded["corpus"] == EVAL_SOURCE


def test_the_structural_side_is_re_derived_from_the_shipped_corpus() -> None:
    """Recomputed for real, not read back out of the artifact.

    The structural half needs no model, so there is no reason to trust a
    recorded copy of it. `structural_floor` runs the shipped check over the
    shipped corpus through the same `run.score` the semantic half used, and the
    counts have to be what the artifact says they were.
    """
    recomputed = structural_floor(harness())
    recorded_structural = check()["structural"]
    # Compared on the keys the ARTIFACT carries, and the delta is asserted
    # rather than ignored. `structural_floor` grew `corpus_version` on
    # 2026-09-03, after this verdict was recorded; back-filling it into
    # `ship_check.json` would edit a record of what was measured, which is the
    # one thing a recorded verdict may never do. Naming the allowed delta is
    # what keeps this a total comparison: a key added later that nobody thought
    # about fails here rather than disappearing into a subset match.
    assert set(recomputed) - set(recorded_structural) == {"corpus_version"}, (
        "structural_floor produces keys this recorded verdict does not carry; a verdict "
        "is a record of what was measured and is not back-filled"
    )
    assert {key: recomputed[key] for key in recorded_structural} == recorded_structural, (
        "the structural side of the verdict is not what the shipped corpus scores today"
    )
    # And the corpus is the same one, which is the claim `corpus_version` exists
    # to make and the only reason the comparison above is still about one thing.
    assert recomputed["corpus_version"] == STRUCTURAL_CORPUS_VERSION
    recorded = check()["verdict"]["structural"]
    # The floor compared against is the recall it was DERIVED from, not the
    # three-decimal rendering the bar also carries. `round(0.8909090909, 3)` is
    # 0.891, which is ABOVE the value it renders, so the rendering fails an
    # untouched structural layer by 0.00009. Both numbers are in the bar,
    # whose semantic half is byte-pinned by `SHIP_BAR_CORE_SHA256` and whose
    # structural half is re-derived above; the verdict records which one it used.
    floor = float(bar()["structural_floor_detail"]["recall"])
    published = float(bar()["structural_floor"])
    assert recorded["floor"] == floor
    assert recorded["floor_published_as"] == published
    assert floor <= published, (
        "the rendering is below the value it renders, so this comment describes "
        "a rounding that no longer happens"
    )
    assert recorded["measured"] == float(recomputed["recall"])
    assert recorded["holds"] is (float(recomputed["recall"]) >= floor)


def test_the_floor_a_verdict_compares_against_is_never_above_its_own_measurement() -> None:
    """The property, not the instance: a floor derived from a value must admit it.

    This is the defect in one line. `structural_floor` is `round(recall, 3)`, and
    round-to-nearest can land ABOVE its argument, so the artifact recorded a
    regression on a detector nothing had touched. Asserted through `verdict`
    rather than by re-reading the artifact, because the artifact is one row and
    the rule is what has to hold for the next one.

    Mutation: point `verdict` back at `bar["structural_floor"]` and this fails on
    the recorded counts, which is the run that produced the artifact.
    """
    recorded_bar = bar()
    exact = float(recorded_bar["structural_floor_detail"]["recall"])
    decided = verdict(
        recorded_bar,
        {"f1": 0.0, "controls": {"passed": True, "refused_with": ""}},
        {"recall": exact},
    )
    assert decided["structural"]["holds"] is True, (
        "the structural layer that DEFINED the floor does not clear it, so the floor "
        "is above the measurement it was derived from"
    )
    assert decided["structural"]["margin"] == 0.0


def test_rounding_the_floor_to_three_places_could_never_have_hidden_a_real_change() -> None:
    """Why the rendering is safe to publish even though it is unsafe to compare.

    The corpus has one granularity: decision-level recall moves in steps of
    1/(TP+FN). If that step were smaller than the rounding, the published 0.891
    would be hiding real movement rather than merely mis-comparing. It is not,
    by a factor of about 36, and the factor is asserted rather than stated so it
    fails if the corpus is ever cut down.
    """
    counts = structural_floor(harness())["counts"]
    positives = counts["tp"] + counts["fn"]
    step = 1 / positives
    rounding = 0.5 * 10**-FLOOR_PLACES
    assert step > 2 * rounding, (
        f"one decision is {step:.6f} of recall and rounding to {FLOOR_PLACES} places moves "
        f"a rate by up to {rounding:.6f}; the published floor can now hide a real change"
    )


def test_the_structural_layer_scored_exactly_what_the_bar_recorded_for_it() -> None:
    """No regression, stated as an identity rather than as an inequality.

    The bar recorded the structural counts before any model existed and nothing
    in this stage touched the structural check, so the four counts have to be
    the same four. This is the claim the inequality was there to protect, and it
    is the stronger one: a chain that had lost a true positive and gained
    another somewhere else would keep the recall and fail here.
    """
    assert check()["structural"]["counts"] == bar()["structural_floor_detail"]["counts"], (
        "the structural counts moved; adding a classifier to the package has cost the "
        "structural layer decisions it used to get right"
    )


def test_the_ship_decision_is_the_conjunction_of_all_three_sides() -> None:
    """The truth table, not the row this run landed on.

    Every side of this run failed, and on that row a conjunction spelled `or`,
    `and`, or `any` returns False alike. So the rule is exercised over all eight
    combinations through the function that produced the artifact, with the real
    bar and synthetic measurements, and the positive row is checked as
    positively as the negative ones: all three sides clearing has to return
    True, or the guard would pass over a function that never ships anything.
    """
    recorded_bar = bar()
    minimum = float(recorded_bar["our_minimum"])
    floor = float(recorded_bar["structural_floor"])
    for controls in (True, False):
        for semantic_clears in (True, False):
            for structural_holds in (True, False):
                semantic = {
                    "f1": minimum + 0.01 if semantic_clears else minimum,
                    "controls": {"passed": controls, "refused_with": ""},
                }
                structural = {"recall": floor if structural_holds else floor - 0.01}
                decided = verdict(recorded_bar, semantic, structural)
                assert decided["ship_the_classifier"] is (
                    controls and semantic_clears and structural_holds
                ), (
                    f"controls={controls} semantic={semantic_clears} "
                    f"structural={structural_holds} decided "
                    f"{decided['ship_the_classifier']}"
                )
                assert decided["semantic"]["clears"] is semantic_clears
                assert decided["structural"]["holds"] is structural_holds
                assert decided["controls"]["passed"] is controls


def test_the_semantic_side_is_not_cleared_by_equalling_the_bar() -> None:
    """`>` and not `>=`, which is the whole of the no-tolerance rule.

    Written as its own test because the truth table above uses a value that
    clears by 0.01, and equality is the one input where the two operators
    disagree. The bar rejected a 0.03 tolerance band on the argument that the
    band is exactly the width of the inconclusive region; a `>=` here would give
    that band back at zero width.
    """
    recorded_bar = bar()
    minimum = float(recorded_bar["our_minimum"])
    semantic = {"f1": minimum, "controls": {"passed": True, "refused_with": ""}}
    structural = {"recall": float(recorded_bar["structural_floor"])}
    decided = verdict(recorded_bar, semantic, structural)
    assert decided["semantic"]["clears"] is False, (
        "a score equal to the bar cleared it; the comparison is not strict"
    )
    assert decided["ship_the_classifier"] is False


def test_the_control_gate_the_harness_applied_is_recorded_with_what_it_refused() -> None:
    """Two records of one gate, held equal, with the input it stopped on.

    The verdict's copy of the control outcome is what a reader meets first, and
    the measurement's copy is what the harness actually returned. A summary that
    said `passed` over a measurement that did not is the failure this pair
    exists to make impossible.
    """
    measured = check()["semantic"]["controls"]
    summarised = check()["verdict"]["controls"]
    assert summarised["passed"] is measured["passed"]
    assert summarised["refused_with"] == measured["refused_with"]
    assert summarised["gate"] == "benchmarks/run.py:controls"
    reached = measured["reached"]
    assert reached, "the gate recorded no inputs, so nothing was actually asked of the model"
    if measured["passed"]:
        assert measured["refused_with"] == ""
    else:
        assert measured["refused_with"], "the gate refused and did not say what refused it"
        # The gate stops at the first input it refuses, so the last input it
        # reached is the one named in the refusal and the model's answer to it
        # is the wrong one.
        assert reached[-1]["input"][:60] in measured["refused_with"]


def test_the_model_scored_is_the_export_two_other_records_describe() -> None:
    """Three records of one model, and they have to agree.

    The verdict names a digest, the export record names it, and the DEV
    measurement named it independently when it scored the same file. A verdict
    over some other file would leave all three looking right on their own.
    """
    model = check()["model"]
    export = json.loads(EXPORT_RECORD.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert (
        model["sha256"]
        == export["files"][SHIPPED]["sha256"]
        == metrics["models"][SHIPPED]["sha256"]
    ), "the model that was judged is not the model that was exported and measured"
    assert (
        model["bytes"] == export["files"][SHIPPED]["bytes"] == metrics["models"][SHIPPED]["bytes"]
    )
    assert model["file"] == export["files"][SHIPPED]["name"] == metrics["models"][SHIPPED]["file"]
    assert (
        model["export_record_sha256"]
        == sha256_of(EXPORT_RECORD)
        == str(metrics["provenance"]["export_record_sha256"])
    ), "the export record has changed since the model was measured against it"
    assert model["which_export"] == SHIPPED == "int8"


def test_the_verdict_was_taken_against_the_bar_as_it_was_recorded() -> None:
    """The bar's own bytes, named in the verdict and pinned to the commit that wrote it.

    `tests/test_ship_bar.py` already refuses an edited bar. This adds the half
    that file cannot say: that THIS verdict was computed against those bytes and
    not against a copy that was different at the time.
    """
    judged = check()["judged_against"]
    # The bar's bytes moved once after this verdict was taken, to record the
    # structural corpus's version digest beside its path; the file discloses
    # that in `structural_floor_rederived.version_pin_added`. So the verdict's
    # recorded digest is held to what the bar was AT THE VERDICT, and the claim
    # that the verdict still stands is carried by the half of the bar that has
    # never moved: `SHIP_BAR_CORE_SHA256` over the semantic registration, which
    # is the side this verdict was decided on. The classifier failed the
    # semantic side by 0.367 against a required 0.900, so nothing about the
    # structural pin could reach the outcome, and `structural_floor_rederived.
    # verdict_unaffected` in the bar makes that argument in full.
    assert judged["sha256"] == SHIP_BAR_SHA256_AT_VERDICT
    assert _core_digest(bar()) == SHIP_BAR_CORE_SHA256, (
        "the semantic registration this verdict was judged against has changed; the "
        "verdict no longer describes a comparison anyone can reproduce"
    )
    assert judged["recorded_utc"] == bar()["recorded_utc"]
    assert judged["our_minimum"] == float(bar()["our_minimum"])
    assert judged["structural_floor"] == float(bar()["structural_floor"])
    assert judged["structural_floor_level"] == "decision"
    measured = datetime.fromisoformat(check()["measured_utc"])
    written = datetime.fromisoformat(str(bar()["recorded_utc"]))
    assert measured > written, "the verdict predates the bar it claims to have been judged by"


def test_the_corpus_scored_is_the_pinned_evaluation_set_in_full() -> None:
    """Every row, counted, against both records of what the corpus is.

    A run that scored a subset would produce a smaller confusion table wearing
    the full one's name, and the four counts are the only place that shows.
    """
    corpus = check()["corpus"]
    source = next(entry for entry in load_sources(MANIFEST) if entry.name == EVAL_SOURCE)
    split = json.loads(SPLITS.read_text(encoding="utf-8"))["eval"]
    assert corpus["sha256"] == source.sha256 == split["sha256"]
    assert corpus["rows"] == int(split["rows"])
    assert corpus["labels"] == {"benign": 1332, "jailbreak": 666}
    assert corpus["labels"] == {
        name: int(split["labels"][str(value)]) for name, value in (("benign", 0), ("jailbreak", 1))
    }
    counts = check()["semantic"]["counts"]
    assert sum(int(counts[key]) for key in ("tp", "fp", "fn", "tn")) == corpus["rows"]
    assert int(counts["tp"]) + int(counts["fn"]) == corpus["labels"]["jailbreak"]
    assert int(counts["fp"]) + int(counts["tn"]) == corpus["labels"]["benign"]


def test_the_verdict_does_not_authorise_publishing_a_headline_row() -> None:
    """Shipping and publishing are two decisions, and the bar keeps them apart.

    Read from the bar rather than asserted here, so the verdict cannot grant an
    authorisation the bar withholds.
    """
    assert check()["verdict"]["publish_headline_precision_and_recall"] is False
    assert bar()["authorises"]["publish_headline_precision_and_recall"]["allowed"] is False


def test_the_results_note_states_the_numbers_the_verdict_holds() -> None:
    """Two tables, and they have to agree, across a document a reader will quote.

    `docs/measurements/2026-08-31-phase2b1-results.md` is the stage's finding,
    and every figure in it is a claim about data that lives in
    `training/artifacts/`. Templated from those records here rather than read
    back as prose, so a sentence that stopped being true fails instead of
    persuading. The reference rows are rebuilt from their counts through the
    same `f1`, which is what makes the published table a view of the record and
    not a second copy of it.

    The three headline rows the note says this measurement does NOT join are
    read out of `BENCHMARKS.md`, for the same reason: quoting a published
    precision that has since moved would misdescribe the very comparison the
    note exists to refuse.
    """
    note = " ".join(RESULTS.read_text(encoding="utf-8").split())
    recorded = check()
    semantic = recorded["semantic"]
    counts = semantic["counts"]
    ours = (
        f"| `{recorded['model']['file']}` | {counts['tp']} | {counts['fp']} | "
        f"{counts['fn']} | {counts['tn']} | {semantic['precision']:.4f} | "
        f"{semantic['recall']:.4f} | **{semantic['f1']:.4f}** |"
    )
    assert ours in note, f"the note does not state {ours!r}"
    for entry in recorded["references"]:
        rates = f1({key: int(entry["counts"][key]) for key in ("tp", "fp", "fn")})
        row = (
            f"| `{entry['model']}` | {entry['counts']['tp']} | {entry['counts']['fp']} | "
            f"{entry['counts']['fn']} | {entry['counts']['tn']} | {rates['precision']:.4f} | "
            f"{rates['recall']:.4f} | {rates['f1']:.4f} |"
        )
        assert row in note, f"the note does not state {row!r}"

    verdicts = recorded["verdict"]
    structural = recorded["structural"]
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    dev = float(metrics["at_the_chosen_configuration"][SHIPPED]["dev"]["f1"])
    buried = [
        float(row["f1"]) for which in ("fp32", "int8") for row in metrics["sweep"][which]["buried"]
    ]
    total = int(recorded["corpus"]["rows"])
    positives = int(recorded["corpus"]["labels"]["jailbreak"])
    negatives = int(recorded["corpus"]["labels"]["benign"])
    everything = 2 * (positives / total) / ((positives / total) + 1)
    claims = [
        f"The bar is `> {verdicts['semantic']['minimum']}`",
        f"Measured {semantic['f1']:.4f}",
        f"**{abs(float(verdicts['semantic']['margin'])):.4f} absolute** below",
        (
            f"{structural['counts']['tp']} TP, {structural['counts']['fp']} FP, "
            f"{structural['counts']['fn']} FN, {structural['counts']['tn']} TN over "
            f"{structural['cases']} cases, decision-level recall {structural['recall']:.10f}"
        ),
        f"= {structural['floor']}, and the check is `recall >= floor`",
        f"the recall the floor was taken from, {verdicts['structural']['floor']!r}",
        f"holds with a margin of {abs(float(verdicts['structural']['margin'])):.5f}",
        f"scores F1 {dev:.4f} on DEV",
        f"is {dev - float(semantic['f1']):.4f} absolute",
        f"would score F1 {everything:.4f} on this corpus",
        f"Accuracy was {(counts['tp'] + counts['tn']) / total:.4f} against {negatives / total:.4f}",
        f"between {min(buried):.2f} and {max(buried):.2f} F1",
        (
            f"the {total} external rows of `{recorded['corpus']['name']}` "
            f"({negatives} benign, {positives} jailbreak)"
        ),
    ]
    for claim in claims:
        assert claim in note, f"the note does not state {claim!r}"

    published = {
        row.split("|")[1].strip(): (row.split("|")[6].strip(), row.split("|")[7].strip())
        for row in (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8").splitlines()
        if row.startswith("| ") and row.count("|") > 8 and "in-repo" in row
    }
    assert {"pii", "secrets", "injection-structural"} <= set(published), (
        f"BENCHMARKS.md no longer publishes the three headline rows: {sorted(published)}"
    )
    for check_name in ("pii", "secrets", "injection-structural"):
        precision, recall = published[check_name]
        assert f"`{check_name}` ({precision} / {recall})" in note, (
            f"the note quotes a published row for {check_name} that BENCHMARKS.md does not"
        )
