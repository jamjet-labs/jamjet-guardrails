"""Guards on `training/ship_bar.json`, the number that decides whether we ship.

The bar is the one artifact in this stage that has to be trustworthy *later*,
when a trained model exists and there is an incentive to read it generously. So
what is checked here is not that the file parses. It is that every figure in it
is still the figure that was measured, and that the things it claims about
itself are re-derivable from the repository rather than asserted in prose.

Four properties, and each one is a way the bar could be quietly moved:

- the recorded revision is the pin `benchmarks/run.py` actually loads, read from
  `run.CLASSIFIER_PINS` and not from `pins.json` again and not from the model
  name. A name is not a hash, and this branch has already pinned v1 while a plan
  said v2 with nothing catching it;
- `reference_measured_by` still says `this-repo`, because a leaderboard figure
  swapped in here would compare two different datasets and that is the exact
  flaw the bar was rewritten to remove;
- the structural floor is `0.885` at DECISION level, re-derived from the shipped
  corpus, and is not the `0.870` FINDING-level number `BENCHMARKS.md` publishes
  for the same file;
- `our_minimum` is what `required_minimum` returns for the recorded reference
  scores, so the bar cannot drift away from the rule that produced it.

Nothing here needs a network or a model server. The reference scores are read
from the artifact and re-derived from the counts recorded beside them; the
structural floor is recomputed for real, because that half needs only the
package and a corpus that ships with it.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from training.evalset import EVAL_SOURCE
from training.fetch import load_sources, sha256_of
from training.ship_bar import (
    FLOOR_PLACES,
    PUBLISHED_FINDING_LEVEL_RECALL,
    STRUCTURAL_LEVEL,
    ShipBarError,
    distinct_models,
    f1,
    harness,
    required_minimum,
    structural_floor,
    why_the_structural_side_exists,
)

ROOT = Path(__file__).resolve().parent.parent
SHIP_BAR = ROOT / "training" / "ship_bar.json"
SPLITS = ROOT / "training" / "generated" / "splits.json"
MANIFEST = ROOT / "training" / "sources.yaml"
ARTIFACTS = ROOT / "training" / "artifacts"

#: sha256 of `training/ship_bar.json` as it was committed, in
#: `feat: record the ship bar before any model exists`, before any model in
#: this repository had been trained.
#:
#: This replaced a weaker check and the replacement is the point. The ordering
#: used to be witnessed by `training/artifacts/` not existing, which is a
#: property that can only ever be true once: the day a model was trained the
#: assertion stopped being a guard and became a failure, and the tempting fix
#: was to delete the line. A digest keeps the same claim afterwards. The bar is
#: a commitment made before the result, and a commitment that can be edited
#: after the result is none, so any edit to that file from here on fails here
#: whatever else in the suite still passes.
SHIP_BAR_SHA256 = "92d6c734ada7e28c081af635e24fea6d53b4fd250056f6fbb3bbc1f0b259cfbc"


def bar() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(SHIP_BAR.read_text(encoding="utf-8"))
    return loaded


def test_the_bar_exists_and_names_every_field_the_stage_promised() -> None:
    """Vacuity guard. Every test below reads this file and would pass on a stub."""
    recorded = bar()
    for key in (
        "metric",
        "metric_definition",
        "reference_model",
        "reference_revision",
        "reference_score",
        "reference_measured_by",
        "reference_models",
        "held_out_corpus",
        "harness",
        "our_minimum",
        "structural_floor",
        "structural_floor_level",
        "recorded_utc",
        "rationale",
        "authorises",
        "contamination",
    ):
        assert key in recorded, f"ship_bar.json is missing {key!r}"


def test_the_recorded_revision_is_the_pin_the_harness_actually_loaded() -> None:
    """A name is not a hash, and this branch has already been bitten by that.

    `benchmarks/run.py` sorts its pins current-first from each pin's own
    `status`, so `CLASSIFIER_PINS[0]` is the model a reader should quote and is
    the model the artifact calls `reference_model`. Read through the harness
    module rather than by re-parsing `pins.json`: the property being guarded is
    that the artifact agrees with what the scoring code loads, and a second
    parse of the same file is not that.
    """
    run = harness()
    current = [pin for pin in run.CLASSIFIER_PINS if pin["status"] == "current"]
    assert len(current) == 1, f"{len(current)} pins claim status 'current'"
    recorded = bar()
    assert recorded["reference_revision"] == current[0]["revision"]
    assert recorded["reference_model"] == current[0]["model"]
    assert recorded["reference_revision"] == run.CLASSIFIER_PINS[0]["revision"], (
        "the artifact names a revision that is not the one run.py loads first"
    )


def test_every_measured_reference_matches_its_pin_field_for_field() -> None:
    """Both revisions, not just the current one, and by id rather than by order.

    The bar is `max` over these entries, so a row whose revision has drifted
    from its pin is a row that could raise or lower the bar while the file goes
    on looking correct.
    """
    run = harness()
    pins = {str(pin["id"]): pin for pin in run.CLASSIFIER_PINS}
    measured = {str(entry["id"]): entry for entry in bar()["reference_models"]}
    assert set(measured) == set(pins), (
        f"the artifact measured {sorted(measured)}; the harness pins {sorted(pins)}"
    )
    for pin_id, entry in measured.items():
        for field in ("model", "revision", "status"):
            assert entry[field] == pins[pin_id][field], (
                f"{pin_id} records {field}={entry[field]!r}; the pin says {pins[pin_id][field]!r}"
            )


def test_both_reference_revisions_are_measured_and_labelled() -> None:
    """v1 and v2, current and superseded, so neither can be quoted as the other.

    The v1 model card says a newer version supersedes it. A file carrying one
    unlabelled score lets a reader take the superseded model for ProtectAI's
    current one, which is the same reason `benchmarks/RESULTS.md` runs both.
    """
    measured = bar()["reference_models"]
    assert len(measured) >= 2, "the bar rests on one reference model"
    statuses = [entry["status"] for entry in measured]
    assert statuses.count("current") == 1, f"statuses are {statuses}"
    assert "superseded" in statuses
    assert statuses[0] == "current", "the current model must be the first row a reader meets"


def test_the_reference_score_was_measured_by_us_and_not_quoted() -> None:
    """A leaderboard figure compares two datasets and is the flaw this bar removed.

    There is no PINT score to quote for this repository and the plan says why:
    the 4,314-input corpus is Lakera's, `benchmark/data/` holds an 8-input
    example, and results are verified by Lakera before publication. So the only
    honest reference number is one measured here, on this corpus, through this
    harness.
    """
    recorded = bar()
    assert recorded["reference_measured_by"] == "this-repo", (
        "the reference score must be measured by us on the evaluation corpus, not "
        "quoted from a published leaderboard"
    )
    assert recorded["harness"]["entry_point"] == "benchmarks/run.py"
    assert recorded["harness"]["identical_for_both_models"] is True
    assert recorded["harness"]["execution_provider"] == harness().EXECUTION_PROVIDER


def test_every_recorded_rate_is_the_rate_its_own_counts_give() -> None:
    """Precision, recall and F1 re-derived from the TP/FP/FN beside them.

    A score edited without its counts, or counts edited without their score, is
    the cheapest way to move this bar: both halves go on looking plausible
    alone. `math.isclose` rather than `==` because these are floats that made a
    round trip through JSON, at a tolerance far tighter than one input of 1998
    could shift any of them.
    """
    for entry in bar()["reference_models"]:
        expected = f1({key: int(entry["counts"][key]) for key in ("tp", "fp", "fn")})
        for name, value in expected.items():
            assert math.isclose(entry[name], value, rel_tol=1e-12), (
                f"{entry['id']} records {name}={entry[name]} and its counts give {value}"
            )


def test_the_recorded_counts_add_up_to_the_corpus_they_were_measured_on() -> None:
    """TP+FP+FN+TN is the corpus, and TP+FN is its positive class.

    Scores from a subset of the corpus are the shape a truncated or filtered run
    leaves behind, and nothing about a rate says how many rows it came from.
    """
    corpus = bar()["held_out_corpus"]
    for entry in bar()["reference_models"]:
        counts = entry["counts"]
        assert sum(counts.values()) == corpus["rows"], (
            f"{entry['id']} scored {sum(counts.values())} inputs; the corpus holds {corpus['rows']}"
        )
        assert counts["tp"] + counts["fn"] == corpus["labels"]["jailbreak"]
        assert counts["fp"] + counts["tn"] == corpus["labels"]["benign"]


def test_our_minimum_is_still_what_the_rule_returns_for_these_scores() -> None:
    """The bar re-derived from the rule, not read as a number somebody typed.

    `required_minimum` is the max over every measured reference, and the
    comparison is strict. Both are checked, because a bar loosened from `>` to
    `>=` moves nothing visible in the file and changes what clears it.
    """
    recorded = bar()
    scores = [float(entry["f1"]) for entry in recorded["reference_models"]]
    assert math.isclose(recorded["our_minimum"], required_minimum(scores), rel_tol=1e-12)
    assert recorded["our_minimum"] >= max(scores), (
        "the bar sits below a reference model we measured, so a model can clear it "
        "while losing to a published one"
    )
    assert recorded["our_minimum_comparison"] == ">", (
        "the semantic side is a strict win; a `>=` would let a tie clear a bar whose "
        "own contamination note says only a win is evidence"
    )


def test_the_bar_carries_no_tolerance_band() -> None:
    """The drafted 0.03 slack was rejected, and rejecting it has to stay visible.

    A tolerance is exactly the width of the region the artifact calls
    inconclusive, so a bar with one authorises shipping on a result that
    establishes nothing. This holds `our_minimum` to the reference itself rather
    than to any number below it.
    """
    recorded = bar()
    assert recorded["our_minimum"] == max(float(e["f1"]) for e in recorded["reference_models"])
    assert recorded["our_minimum"] >= recorded["reference_score"]


def test_the_structural_floor_is_the_decision_level_recall_of_the_shipped_corpus() -> None:
    """Re-derived, not read. This is the half of the bar CI can recompute.

    `run.score` over `run.load_structural_cases` is the same counting the
    artifact used, and it needs only the package and a corpus that ships with
    it. If the corpus or the check moves, the recorded floor has to move with
    it or this fails.
    """
    recomputed = structural_floor(harness())
    recorded = bar()
    assert recorded["structural_floor"] == recomputed["floor"]
    assert recorded["structural_floor_detail"]["counts"] == recomputed["counts"]
    assert recorded["structural_floor_level"] == STRUCTURAL_LEVEL == "decision"
    # The unrounded rate too, and against the counts rather than against itself.
    # Rounding to three places hides WHICH rate was recorded: precision here is
    # 0.958 and recall 0.885, and a mutation swapping one for the other inside
    # `structural_floor` survived until this line existed, because the rounded
    # `floor` was computed from a value the test never looked at.
    counts = recomputed["counts"]
    exact = counts["tp"] / (counts["tp"] + counts["fn"])
    assert math.isclose(recomputed["recall"], exact, rel_tol=1e-12), (
        f"structural_floor reports {recomputed['recall']} where TP/(TP+FN) is {exact}"
    )
    assert math.isclose(recorded["structural_floor_detail"]["recall"], exact, rel_tol=1e-12)
    assert recorded["structural_floor"] == round(exact, FLOOR_PLACES)


def test_the_structural_floor_is_not_the_published_finding_level_number() -> None:
    """0.885 and 0.870 are two quantities, not one quantity that moved.

    `BENCHMARKS.md` publishes the finding-level recall, where a case with four
    expected spans contributes four. A classifier emits no spans, so it can only
    be compared at the decision level. Recording the finding-level number
    against a classifier would compare things that are not the same kind of
    thing, and it would do it silently: both are recalls on the same file.
    """
    recorded = bar()
    assert recorded["structural_floor"] == 0.885
    assert recorded["structural_floor"] != PUBLISHED_FINDING_LEVEL_RECALL
    assert recorded["published_finding_level_recall"] == PUBLISHED_FINDING_LEVEL_RECALL == 0.870
    assert "0.870" in recorded["structural_floor_note"], (
        "the note must show the finding-level number it is telling the reader apart from"
    )
    published = (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8")
    assert "0.870" in published, "BENCHMARKS.md no longer publishes the finding-level recall"


def test_the_evaluation_corpus_is_the_external_one_both_records_agree_on() -> None:
    """The digest, held to `sources.yaml` and to `splits.json` at once.

    Two records of one corpus can drift while each looks right on its own, and
    the number in this file is only about the rows those digests describe.
    """
    corpus = bar()["held_out_corpus"]
    assert corpus["name"] == EVAL_SOURCE
    assert corpus["external"] is True
    sources = [source for source in load_sources(MANIFEST) if source.name == EVAL_SOURCE]
    assert len(sources) == 1, f"{MANIFEST} carries {len(sources)} entries for {EVAL_SOURCE}"
    assert sources[0].role == "eval"
    assert corpus["sha256"] == sources[0].sha256
    split = json.loads(SPLITS.read_text(encoding="utf-8"))["eval"]
    assert corpus["sha256"] == split["sha256"]
    assert corpus["rows"] == split["rows"]
    assert corpus["labels"] == {"benign": split["labels"]["0"], "jailbreak": split["labels"]["1"]}


def test_the_evaluation_corpus_is_not_a_slice_of_the_corpus_we_train_on() -> None:
    """The whole reason the bar is measured externally.

    The synthetic rows are separable by register, so a score taken from a
    held-out slice of them would measure artifact exploitation. The guard is
    that the recorded corpus is not the generated one under any of its names.
    """
    corpus = bar()["held_out_corpus"]
    generated = json.loads(SPLITS.read_text(encoding="utf-8"))
    assert corpus["sha256"] != generated["rows_sha256"], (
        "the bar was measured on the corpus stage 2b fits on"
    )
    assert "training/generated" not in corpus["name"]


def test_the_contamination_argument_is_recorded_with_its_direction() -> None:
    """Ruling 20. On its face this eval looks like a mistake; the file must argue it.

    Measuring against a corpus the reference was trained on is a disqualification
    everywhere else in this tree. What makes it right here is which way the bias
    runs, and a reader who is not handed that argument will read the number as
    an error. So the direction and the asymmetry are held as content, not left
    to a report nobody re-reads.
    """
    contamination = bar()["contamination"]
    assert contamination["eval_is_in_the_reference_training_data"] is True
    assert contamination["deliberate"] is True
    assert contamination["direction"] == "towards the reference"
    argument = contamination["argument"].lower()
    assert "inconclusive" in argument, "a loss must be recorded as inconclusive"
    assert "cannot pass us unfairly" in argument
    assert contamination["task_adjacency"], "the jailbreak/injection gap must be recorded"
    assert contamination["single_source"], "resting on one corpus must be recorded"


def test_shipping_and_publishing_are_recorded_as_two_separate_decisions() -> None:
    """Ruling 15. Clearing the bar authorises shipping and not publishing.

    There is no independent evaluation source for semantic injection of the
    standing the `pii`, `secrets` and `injection-structural` corpora have. One
    field carrying both decisions is how a headline precision/recall row ends up
    beside the other three on the strength of an adjacent-task corpus the
    reference was trained on.
    """
    authorises = bar()["authorises"]
    assert set(authorises) == {"ship", "publish_headline_precision_and_recall"}
    assert authorises["ship"]["allowed_when"]
    publish = authorises["publish_headline_precision_and_recall"]
    assert publish["allowed"] is False
    assert publish["reason"], "a refusal with no reason is one a later task will overturn"


def test_the_bar_is_two_sided_and_says_both_sides_are_required() -> None:
    """Semantic and structural. Either alone is a bar that can be cleared halfway."""
    clears = bar()["clears_the_bar"]
    assert clears["both_required"] is True
    assert "our_minimum" in clears["semantic"]
    assert "structural_floor" in clears["structural"]
    assert clears["otherwise"], "what happens on a failure has to be written down"
    rationale = bar()["rationale"]
    assert "SEMANTIC" in rationale and "STRUCTURAL" in rationale


def test_the_metric_is_defined_at_the_decision_level_in_the_file() -> None:
    """A metric named and not defined is one the next reader defines for themselves."""
    recorded = bar()
    assert recorded["metric"] == "f1"
    definition = recorded["metric_definition"]
    assert "decision-level" in definition.lower()
    assert "TP/(TP+FP)" in definition and "TP/(TP+FN)" in definition


def test_the_bar_was_recorded_before_any_model_existed() -> None:
    """The ordering, which is the whole reason the bar means anything.

    A model now exists, so the old form of this check is gone: it asserted that
    `training/artifacts/` did not exist, which witnessed the ordering exactly
    once and then turned into a failure whose obvious fix was to delete it. The
    claim has not changed and neither has its strength. It is now carried by two
    things that go on holding after a model is trained:

    - **the bar cannot have been edited.** Its bytes are pinned by
      `SHIP_BAR_SHA256`, taken at the commit that recorded it, before any
      training run in this repository.
    - **the bar predates every model.** Each run record carries `trained_utc`,
      and the bar carries `recorded_utc`. Both are instants rather than dates,
      because the first model was fitted on the same day the bar was written
      and two dates cannot be put in order.

    Timestamps alone would not be enough, which is why the digest is first: a
    `recorded_utc` that can be edited is a timestamp that says whatever the last
    editor needed it to say.
    """
    recorded = bar()
    assert recorded["recorded_before_any_model_existed"] is True
    assert recorded["recorded_utc"].endswith("+00:00"), (
        f"recorded_utc is {recorded['recorded_utc']!r}, which is not UTC"
    )
    assert sha256_of(SHIP_BAR) == SHIP_BAR_SHA256, (
        "training/ship_bar.json has changed since it was recorded; a bar edited after the "
        "model it judges is not a commitment made before the result"
    )

    written = datetime.fromisoformat(recorded["recorded_utc"])
    records = sorted(ARTIFACTS.glob("*.json")) if ARTIFACTS.is_dir() else []
    assert records, (
        "no model record exists yet, so the ordering below is vacuous; delete this line "
        "only when one does"
    )
    for path in records:
        run = json.loads(path.read_text(encoding="utf-8"))
        trained = datetime.fromisoformat(run["trained_utc"])
        assert trained > written, (
            f"{path.name} was trained at {trained.isoformat()} and the bar was recorded at "
            f"{written.isoformat()}; the bar has to come first"
        )


def test_the_two_references_did_not_decide_identically() -> None:
    """Equal TP and equal FN in two rows is also what one model scored twice looks like.

    They are two distinct models, verified per file by SHA-256 on load, and they
    disagreed on inputs. Recorded so that a reader meeting two rows with the
    same recall is not left to wonder.
    """
    disagreement = bar()["reference_disagreement"]
    assert disagreement["inputs_decided_differently"] > 0, (
        "the two pinned references decided every input identically, which is what one "
        "model scored under both flags would also produce"
    )
    assert (
        disagreement["on_positives"] + disagreement["on_negatives"]
        == disagreement["inputs_decided_differently"]
    )


# --- the rule itself, away from the artifact ---------------------------------


def test_f1_is_zero_and_not_nan_when_nothing_was_flagged() -> None:
    """A NaN compares False against every bar and would read as "did not clear".

    It would also read as "did not clear" for a model that flagged nothing at
    all, which is the same answer for two different situations. Zero is the
    answer that survives being sorted, compared and rendered.
    """
    for counts in (
        {"tp": 0, "fp": 0, "fn": 0},
        {"tp": 0, "fp": 10, "fn": 0},
        {"tp": 0, "fp": 0, "fn": 10},
        {"tp": 0, "fp": 10, "fn": 10},
    ):
        rates = f1(counts)
        assert rates == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        assert not any(math.isnan(value) for value in rates.values())


def test_f1_is_the_harmonic_mean_of_the_two_rates() -> None:
    rates = f1({"tp": 551, "fp": 8, "fn": 115})
    assert math.isclose(rates["precision"], 551 / 559)
    assert math.isclose(rates["recall"], 551 / 666)
    assert math.isclose(rates["f1"], 2 * 551 / (2 * 551 + 8 + 115))


def test_the_rule_takes_the_strongest_reference_and_not_the_current_one() -> None:
    """Otherwise the bar is set by which revision a vendor is advertising.

    Ordered with the weaker score first and then last, so a rule that returned
    `scores[0]` or `scores[-1]` fails one of the two.
    """
    assert required_minimum([0.80, 0.90]) == 0.90
    assert required_minimum([0.90, 0.80]) == 0.90
    assert required_minimum([0.90]) == 0.90


def test_the_rule_refuses_to_set_a_bar_from_nothing() -> None:
    with pytest.raises(ShipBarError):
        required_minimum([])


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("revision", lambda pins: pins[1].__setitem__("revision", pins[0]["revision"])),
        (
            "set of pinned file digests",
            lambda pins: pins[1].__setitem__("files", pins[0]["files"]),
        ),
    ],
)
def test_two_pins_that_are_one_model_are_refused(
    field: str, mutate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One model measured twice prints as two rows and nothing says otherwise."""
    run = harness()
    pins = json.loads(json.dumps(list(run.CLASSIFIER_PINS)))
    mutate(pins)
    monkeypatch.setattr(run, "CLASSIFIER_PINS", pins)
    dirs = {str(pin["id"]): Path(pin["local_dir"]) for pin in pins}
    with pytest.raises(ShipBarError, match=field.replace(".", r"\.")):
        distinct_models(run, dirs)


def test_one_directory_reaching_both_flags_is_refused() -> None:
    """The flags are separate and a caller can still pass the same path to both."""
    run = harness()
    dirs = {str(pin["id"]): Path("/tmp/one-model") for pin in run.CLASSIFIER_PINS}
    with pytest.raises(ShipBarError, match="directory"):
        distinct_models(run, dirs)


def test_a_single_pinned_classifier_cannot_set_this_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """`max` over one score is that score, and the comparison the bar makes is gone."""
    run = harness()
    pins = json.loads(json.dumps(list(run.CLASSIFIER_PINS)))[:1]
    dirs = {str(pin["id"]): Path(pin["local_dir"]) for pin in pins}
    monkeypatch.setattr(run, "CLASSIFIER_PINS", pins)
    with pytest.raises(ShipBarError, match="at least two"):
        distinct_models(run, dirs)


def test_every_runnable_training_module_is_a_documented_command() -> None:
    """The README's command block, held to the modules that actually run.

    Both directions. A module gaining a `__main__` and not appearing here is a
    step nobody can reproduce; a command listed for a module with no `__main__`
    is a line that fails the moment somebody pastes it. The first draft of this
    file said "Two commands exist so far" while three modules had one, which is
    a count in prose about code and so a claim, not a description.
    """
    readme = (ROOT / "training" / "README.md").read_text(encoding="utf-8")
    runnable = {
        path.stem
        for path in sorted((ROOT / "training").glob("*.py"))
        if "__main__" in path.read_text(encoding="utf-8")
    }
    assert runnable, "no runnable training modules found; this guard would prove nothing"
    documented = set(re.findall(r"python -m training\.(\w+)", readme))
    assert runnable == documented, (
        f"modules with a __main__: {sorted(runnable)}; commands in the README: {sorted(documented)}"
    )
    # The count is also written in the README's prose, in words, and a count in
    # prose about code is a claim. Mapped rather than string-replaced, so the
    # guard still means something at four.
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    stated = f"{words[len(runnable)]} commands exist so far"
    assert stated in readme, f"the README does not say {stated!r}"
    for wrong in (
        phrase
        for count, word in words.items()
        if count != len(runnable)
        for phrase in (f"{word} commands exist so far",)
    ):
        assert wrong not in readme, f"the README also says {wrong!r}"


def test_the_structural_side_records_what_a_semantic_classifier_scored_there() -> None:
    """The reason the bar has a second side, held to the measurement it came from.

    Both pinned classifiers were run over the structural corpus in
    `benchmarks/results/measurements.json`, and this re-reads that record rather
    than trusting the copy. A number restated in a second file is a number that
    drifts while both files go on looking right, which is the failure this
    stage has already had once.
    """
    recorded = bar()["why_the_structural_side_exists"]
    # Through the function that wrote it, not by re-implementing the read. A
    # test that reads the same file the same way passes over a reader that went
    # to the wrong corpus, which is a mutation that survived until this line
    # called the function instead of imitating it.
    assert recorded == why_the_structural_side_exists()

    measurements = json.loads(
        (ROOT / "benchmarks" / "results" / "measurements.json").read_text(encoding="utf-8")
    )
    source = {
        str(run["detector"]["id"]): {
            key: int(run["overall"][key]) for key in ("tp", "fp", "fn", "tn")
        }
        for run in measurements["runs"]
        if run["corpus"]["id"] == "in-repo" and str(run["detector"]["id"]).startswith("classi")
    }
    assert source, "measurements.json records no classifier over the structural corpus"
    assert {entry["id"]: entry["counts"] for entry in recorded["reference_classifiers"]} == source
    assert recorded["measured"] == measurements["measured"]
    assert recorded["corpus"] == bar()["structural_floor_detail"]["corpus"], (
        "the classifiers were scored on a corpus other than the one the floor came from"
    )

    # The claim the finding rests on: every reference classifier is well below
    # the structural check on this corpus. Asserted rather than described, so a
    # future measurement that reversed it would fail here instead of leaving a
    # sentence behind that is no longer true.
    floor = bar()["structural_floor_detail"]["recall"]
    for entry in recorded["reference_classifiers"]:
        assert entry["recall"] < floor, (
            f"{entry['id']} now reaches recall {entry['recall']} on the structural corpus "
            f"against the structural check's {floor}; the finding no longer holds"
        )
