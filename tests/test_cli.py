"""The `eval` command: the seam where everything above becomes one exit code.

The brief's fourteen tests come first, verbatim apart from the annotations mypy
strict needs on the helper and on every test.

Everything after them was written against a mutation of the brief's own
implementation and watched fail. Those mutations and their RED output are
recorded in task-13-report.md.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from jamjet_guardrails.detectors import AVAILABLE, build
from jamjet_guardrails.eval.cli import discover, main
from jamjet_guardrails.eval.gate import MAX_EPSILON
from jamjet_guardrails.types import Context, Direction, Kind, Verdict

# A non-ASCII corpus name, spelled from a code point rather than typed, so this
# file stays pure ASCII and an editor cannot normalise the fixture away. It is
# the smallest thing that tells UTF-8 from a cp1252 default.
NON_ASCII = "caf" + chr(0xE9)

CASE = {
    "id": "pii-0001",
    "text": "mail alice@example.com",
    "direction": "output",
    "expect": {"decision": "redact", "findings": [{"type": "EMAIL", "span": [5, 22]}]},
    "source": "in-repo",
    "license": "Apache-2.0",
}


def _corpora(tmp_path: Path) -> Path:
    d = tmp_path / "corpora" / "pii"
    d.mkdir(parents=True)
    (d / "in-repo.jsonl").write_text(json.dumps(CASE) + "\n")
    return tmp_path / "corpora"


def test_discover_finds_check_and_source_from_the_path(tmp_path: Path) -> None:
    found = discover(_corpora(tmp_path))
    assert [(c, s) for c, s, _ in found] == [("pii", "in-repo")]


def test_discover_is_sorted_for_stable_report_ordering(tmp_path: Path) -> None:
    root = _corpora(tmp_path)
    (root / "pii" / "third-party.jsonl").write_text(
        json.dumps(dict(CASE, source="third-party")) + "\n"
    )
    assert [s for _, s, _ in discover(root)] == ["in-repo", "third-party"]


def test_main_writes_both_artifacts(tmp_path: Path) -> None:
    root = _corpora(tmp_path)
    out_json, out_md = tmp_path / "b.json", tmp_path / "B.md"
    code = main(["--corpora-dir", str(root), "--json", str(out_json), "--md", str(out_md)])
    assert code == 0
    payload = json.loads(out_json.read_text())
    assert payload["results"][0]["detector"] == "pii"
    assert payload["results"][0]["precision"] == 1.0
    assert "# Benchmarks" in out_md.read_text()


def test_main_gates_and_fails_on_regression(tmp_path: Path) -> None:
    root = _corpora(tmp_path)
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps({"pii/in-repo/does-not-match": {"precision": 1.0, "recall": 1.0}})
    )
    assert main(["--corpora-dir", str(root), "--gate", str(baselines)]) == 1


def test_main_gate_passes_when_the_baseline_matches(tmp_path: Path) -> None:
    """The baseline key is built from the corpus on disk and from the documented
    key shape, never read back out of the JSON the tool just wrote. A key scheme
    and a report that drifted together would sail through a test whose expected
    value came from the output under test."""
    from jamjet_guardrails.eval.corpus import load_corpus

    root = _corpora(tmp_path)
    version = load_corpus(root / "pii" / "in-repo.jsonl", name="pii/in-repo").version
    assert len(version) == 12
    assert set(version) <= set("0123456789abcdef")

    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps(
            {
                f"pii/in-repo/{version}": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "decision_mismatches": 0,
                }
            }
        )
    )
    assert main(["--corpora-dir", str(root), "--gate", str(baselines)]) == 0


def test_the_published_version_is_the_corpus_version(tmp_path: Path) -> None:
    """Ties the artifact to the corpus independently of the gate."""
    from jamjet_guardrails.eval.corpus import load_corpus

    root = _corpora(tmp_path)
    out = tmp_path / "b.json"
    assert main(["--corpora-dir", str(root), "--json", str(out)]) == 0
    published = json.loads(out.read_text())["results"][0]["corpus_version"]
    assert published == load_corpus(root / "pii" / "in-repo.jsonl", name="pii/in-repo").version


def test_main_returns_two_for_an_unreadable_baseline_file(tmp_path: Path) -> None:
    root = _corpora(tmp_path)
    bad = tmp_path / "baselines.json"
    bad.write_text("{not json")
    assert main(["--corpora-dir", str(root), "--gate", str(bad)]) == 2


def test_main_returns_two_when_the_baseline_file_is_not_an_object(tmp_path: Path) -> None:
    root = _corpora(tmp_path)
    bad = tmp_path / "baselines.json"
    bad.write_text("[]")
    assert main(["--corpora-dir", str(root), "--gate", str(bad)]) == 2


def test_a_gate_disabling_epsilon_does_not_exit_zero(tmp_path: Path) -> None:
    """--epsilon is a knob on a CI gate, and every value of it points one way:
    wider tolerance, greener CI.

    The baseline here MATCHES and the run MEETS it, so this invocation exits 0
    on any accepted epsilon. That makes the refusal the only thing that can
    produce a 1. A mismatched key would have exited 1 regardless and proved
    nothing about the epsilon bound.
    """
    from jamjet_guardrails.eval.corpus import load_corpus

    root = _corpora(tmp_path)
    version = load_corpus(root / "pii" / "in-repo.jsonl", name="pii/in-repo").version
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps(
            {
                f"pii/in-repo/{version}": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "decision_mismatches": 0,
                }
            }
        )
    )
    argv = ["--corpora-dir", str(root), "--gate", str(baselines)]

    assert main(argv) == 0
    assert main([*argv, "--epsilon", "0.005"]) == 0
    assert main([*argv, "--epsilon", "1.0"]) == 1
    assert main([*argv, "--epsilon", "nan"]) == 1
    assert main([*argv, "--epsilon", "-0.1"]) == 1


def test_write_baselines_records_every_evaluated_corpus(tmp_path: Path) -> None:
    from jamjet_guardrails.eval.corpus import load_corpus

    root = _corpora(tmp_path)
    out = tmp_path / "baselines.json"
    assert main(["--corpora-dir", str(root), "--write-baselines", str(out)]) == 0
    version = load_corpus(root / "pii" / "in-repo.jsonl", name="pii/in-repo").version
    assert json.loads(out.read_text()) == {
        f"pii/in-repo/{version}": {
            "precision": 1.0,
            "recall": 1.0,
            "decision_mismatches": 0,
        }
    }


def test_written_baselines_pass_their_own_gate(tmp_path: Path) -> None:
    """Pins writer/gate agreement on the key format, not key correctness.
    test_main_gate_passes_when_the_baseline_matches pins the shape independently.
    """
    root = _corpora(tmp_path)
    out = tmp_path / "baselines.json"
    assert main(["--corpora-dir", str(root), "--write-baselines", str(out)]) == 0
    assert main(["--corpora-dir", str(root), "--gate", str(out)]) == 0


def test_writing_and_gating_in_one_run_is_refused(tmp_path: Path) -> None:
    """A run that records its own baseline and then gates on it always passes."""
    root = _corpora(tmp_path)
    assert (
        main(
            [
                "--corpora-dir",
                str(root),
                "--write-baselines",
                str(tmp_path / "a.json"),
                "--gate",
                str(tmp_path / "b.json"),
            ]
        )
        == 2
    )


def test_main_returns_two_for_a_missing_corpora_dir(tmp_path: Path) -> None:
    assert main(["--corpora-dir", str(tmp_path / "nope")]) == 2


def test_main_refuses_an_empty_corpora_dir(tmp_path: Path) -> None:
    """A directory that exists but matches nothing must not exit 0."""
    empty = tmp_path / "corpora"
    (empty / "pii").mkdir(parents=True)
    assert main(["--corpora-dir", str(empty)]) == 2


# Everything below closes a mutation that survived the brief's own list. Each is
# recorded in task-13-report.md with the mutation it was written against and the
# RED it was watched to produce.


# Six findings expected and three predicted: two matched, one false alarm, four
# missed. Precision 2/3, recall 1/3, two wrong decisions over four cases.
#
# Every one of those numbers is chosen to make a guard visible, and the first
# version of this fixture failed that test. The brief's own fixture scores a
# perfect 1.0 on both ratios with zero wrong decisions, where rounding UP and
# rounding DOWN are the same number, precision and recall are the same number,
# and the mismatch count is the same number as a hardcoded zero. The first
# rewrite fixed precision (2/3, which floors to 0.666 and ceils to 0.667) and
# left recall at 1.0, so `math.floor` mutated to `math.ceil` on the RECALL line
# still survived: half a guard, pinned on the half that cannot violate it. Both
# ratios are now non-terminating, they differ from each other so a swap is
# visible, and the mismatch count is neither 0 nor the case count.
_IMPERFECT = (
    {
        "id": "pii-a",
        "text": "mail alice@example.com",
        "direction": "output",
        "expect": {"decision": "redact", "findings": [{"type": "EMAIL", "span": [5, 22]}]},
        "source": "in-repo",
        "license": "Apache-2.0",
    },
    {
        "id": "pii-b",
        "text": "mail bob@example.com",
        "direction": "output",
        "expect": {"decision": "redact", "findings": [{"type": "EMAIL", "span": [5, 20]}]},
        "source": "in-repo",
        "license": "Apache-2.0",
    },
    # A clean case the detector redacts anyway: one false positive, one wrong
    # decision.
    {
        "id": "pii-c",
        "text": "ping carol@example.com",
        "direction": "output",
        "expect": {"decision": "allow", "findings": []},
        "source": "in-repo",
        "license": "Apache-2.0",
    },
    # Four labelled findings of a type the bundled detector does not model, in
    # text it finds nothing in: four false negatives and a second wrong
    # decision.
    {
        "id": "pii-d",
        "text": "no personal data here",
        "direction": "output",
        "expect": {
            "decision": "redact",
            "findings": [
                {"type": "PERSON_NAME", "span": [0, 2]},
                {"type": "PERSON_NAME", "span": [3, 11]},
                {"type": "PERSON_NAME", "span": [12, 16]},
                {"type": "PERSON_NAME", "span": [17, 21]},
            ],
        },
        "source": "in-repo",
        "license": "Apache-2.0",
    },
)


def _imperfect_corpora(tmp_path: Path) -> Path:
    d = tmp_path / "corpora" / "pii"
    d.mkdir(parents=True)
    (d / "in-repo.jsonl").write_text("".join(json.dumps(row) + "\n" for row in _IMPERFECT))
    return tmp_path / "corpora"


class _InputOnlyGuardrail:
    """A guardrail that runs on input only, so an output corpus cannot be scored.

    Both bundled detectors declare both directions, so ``EvaluationError`` is
    unreachable from the command line without one of these. A net pinned only on
    the two errors that ARE reachable is a net with an untested hole in exactly
    the place the brief warns about.
    """

    name: str = "inputonly"
    version: str = "0.0.1"
    kind: Kind = "constraint"
    directions: frozenset[Direction] = frozenset({"input"})

    def check(self, content: str, context: Context) -> Verdict:
        raise AssertionError("evaluate must refuse this pairing before calling check")


def test_a_corpus_that_cannot_be_loaded_exits_one_and_publishes_nothing(tmp_path: Path) -> None:
    """The path the brief left unpinned: a run that scored nothing, exiting 0.

    Written against `return 1` mutated to `return 0` in the load/evaluate net,
    and against `CorpusError` dropped from that net. Both left the brief's
    fourteen tests green.
    """
    root = tmp_path / "corpora"
    (root / "pii").mkdir(parents=True)
    (root / "pii" / "in-repo.jsonl").write_text(json.dumps({"id": "pii-0001"}) + "\n")
    out_json, out_md = tmp_path / "b.json", tmp_path / "B.md"
    assert main(["--corpora-dir", str(root), "--json", str(out_json), "--md", str(out_md)]) == 1
    # A run that could not score every corpus publishes no numbers at all. A
    # partial artifact is indistinguishable from a complete one.
    assert not out_json.exists()
    assert not out_md.exists()


def test_a_check_directory_naming_no_installed_detector_exits_one(tmp_path: Path) -> None:
    """Written against `GuardrailUnavailableError` dropped from the except net."""
    root = tmp_path / "corpora"
    (root / "nosuchcheck").mkdir(parents=True)
    (root / "nosuchcheck" / "in-repo.jsonl").write_text(json.dumps(CASE) + "\n")
    assert main(["--corpora-dir", str(root)]) == 1


def test_a_corpus_the_detector_cannot_run_on_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written against `EvaluationError` dropped from the except net.

    Without it the mismatched pairing leaves as a traceback rather than a clean
    exit 1, and no test in the brief's list can see the difference.
    """
    monkeypatch.setitem(AVAILABLE, "inputonly", _InputOnlyGuardrail)
    # Asserted, because an exit 1 from an UNREGISTERED name looks identical from
    # the outside: without this the test would pass just as well on a
    # monkeypatch that never took, and prove nothing about EvaluationError.
    assert build("inputonly").name == "inputonly"
    root = tmp_path / "corpora"
    (root / "inputonly").mkdir(parents=True)
    # direction "output"; the guardrail declares "input" only.
    (root / "inputonly" / "in-repo.jsonl").write_text(json.dumps(CASE) + "\n")
    assert main(["--corpora-dir", str(root)]) == 1


def test_written_baselines_round_down_and_carry_the_measured_mismatch_count(
    tmp_path: Path,
) -> None:
    """Written against four mutations at once, all invisible on a perfect corpus:
    `math.floor` to `math.ceil` on the precision line, the same on the recall
    line, precision and recall swapped, and the mismatch count replaced by a
    literal 0.

    Rounding DOWN is the direction that matters. A baseline recorded above what
    the run measured fails forever against the detector that produced it; a
    baseline recorded below it is what the gate's epsilon is for.
    """
    from jamjet_guardrails.eval.corpus import load_corpus

    root = _imperfect_corpora(tmp_path)
    out = tmp_path / "baselines.json"
    assert main(["--corpora-dir", str(root), "--write-baselines", str(out)]) == 0
    version = load_corpus(root / "pii" / "in-repo.jsonl", name="pii/in-repo").version
    assert json.loads(out.read_text()) == {
        f"pii/in-repo/{version}": {
            # 2/3 is 0.6666..., which is 0.666 down and 0.667 up.
            "precision": 0.666,
            # 1/3 is 0.3333..., which is 0.333 down and 0.334 up.
            "recall": 0.333,
            "decision_mismatches": 2,
        }
    }


def test_a_written_imperfect_baseline_passes_its_own_gate(tmp_path: Path) -> None:
    """Rounding down has to leave the recorded value gate-passable, not only lower."""
    root = _imperfect_corpora(tmp_path)
    out = tmp_path / "baselines.json"
    assert main(["--corpora-dir", str(root), "--write-baselines", str(out)]) == 0
    assert main(["--corpora-dir", str(root), "--gate", str(out)]) == 0


def test_discover_orders_by_check_before_source(tmp_path: Path) -> None:
    """Written against the sort key with its two halves swapped.

    The brief's ordering test uses two sources under ONE check, where sorting by
    (source, check) and by (check, source) give the same answer. These two order
    oppositely under the two keys.
    """
    root = tmp_path / "corpora"
    (root / "pii").mkdir(parents=True)
    (root / "secrets").mkdir()
    (root / "pii" / "zeta.jsonl").write_text("")
    (root / "secrets" / "alpha.jsonl").write_text("")
    assert [(c, s) for c, s, _ in discover(root)] == [("pii", "zeta"), ("secrets", "alpha")]


def test_the_published_corpus_name_carries_both_halves_of_the_path(tmp_path: Path) -> None:
    """Written against `name=f"{check}/{source}"` reduced to `name=check`.

    The name is what the markdown headline and the JSON record call the corpus.
    Reduced to the check it repeats the detector column and identifies nothing.
    """
    root = _corpora(tmp_path)
    out = tmp_path / "b.json"
    assert main(["--corpora-dir", str(root), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["results"][0]["corpus_name"] == "pii/in-repo"


def test_every_discovered_corpus_is_evaluated(tmp_path: Path) -> None:
    """Written against `discover(...)` truncated to its first entry.

    The brief measures one corpus everywhere, so a loop that scores only the
    first of them publishes the same artifact.
    """
    root = _corpora(tmp_path)
    (root / "pii" / "third-party.jsonl").write_text(
        json.dumps(dict(CASE, source="third-party")) + "\n"
    )
    out = tmp_path / "b.json"
    assert main(["--corpora-dir", str(root), "--json", str(out)]) == 0
    results = json.loads(out.read_text())["results"]
    assert [r["corpus_source"] for r in results] == ["in-repo", "third-party"]


def test_an_empty_baselines_object_does_not_pass_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against a `if not baselines: return 0` short-circuit before the gate.

    `{}` is a well-formed JSON object, so the two exit-2 tests above cannot see
    it, and it is the state a truncated or newly created baselines file is in.
    Every corpus is then unrecorded, which is a regression, not a pass.
    """
    root = _corpora(tmp_path)
    empty = tmp_path / "baselines.json"
    empty.write_text("{}")
    assert main(["--corpora-dir", str(root), "--gate", str(empty)]) == 1
    err = capsys.readouterr().err
    # An empty file records nothing for a corpus that was scored, which is the
    # file being incomplete rather than a detector getting worse.
    assert "stale or malformed" in err
    assert "REGRESSION" not in err


def test_a_file_passed_as_the_corpora_dir_exits_two(tmp_path: Path) -> None:
    """The other half of the corpora-dir guard: present, but not a directory."""
    not_a_dir = tmp_path / "corpora"
    not_a_dir.write_text("")
    assert main(["--corpora-dir", str(not_a_dir)]) == 2


def test_a_corpus_nested_below_the_layout_is_not_discovered(tmp_path: Path) -> None:
    """Written against the glob widened from `*/*.jsonl` to `**/*.jsonl`.

    The layout is exactly two levels: `<check>/<source>.jsonl`. A widened glob
    reaches `corpora/archive/pii/retired.jsonl`, reads `pii` off the parent
    directory, and scores a corpus that was deliberately moved out of the run.
    Every other nesting the widened glob picks up fails loudly at `build`, so
    this shape is the one that is silent, and silently ADDING a corpus moves a
    published number.
    """
    root = _corpora(tmp_path)
    retired = root / "archive" / "pii"
    retired.mkdir(parents=True)
    (retired / "retired.jsonl").write_text(json.dumps(dict(CASE, id="pii-9999")) + "\n")
    assert [(c, s) for c, s, _ in discover(root)] == [("pii", "in-repo")]


def test_the_summary_line_reports_precision_and_recall_under_their_own_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the P and R values swapped in the progress line.

    The line CI operators read is not a published artifact, so nothing else
    looks at it, and the two values sit next to each other in one f-string.
    Asserted by value on a corpus where the two ratios differ; the surrounding
    layout is deliberately not pinned.
    """
    assert main(["--corpora-dir", str(_imperfect_corpora(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "P=0.667" in out
    assert "R=0.333" in out


# Three fixes ruled on by the coordinator after the first mutation rounds: an
# unreadable corpus and an unwritable artifact escaped as tracebacks, and an
# unusable --epsilon announced itself as a regression.


def test_a_corpus_that_cannot_be_read_exits_one(tmp_path: Path) -> None:
    """Discovery found it and the loader cannot open it.

    A DIRECTORY named `<source>.jsonl` rather than a chmod, deliberately: the
    glob matches it, `read_text` raises `IsADirectoryError`, and the result does
    not depend on who the test runs as. A 0o000 file is readable by root, so in
    a container that variant would pass for the wrong reason.

    Exit 1, not 2: the fault is the data. `load_corpus` leaves OSError unwrapped
    on purpose, so the CLI has to name it rather than let it escape.
    """
    root = tmp_path / "corpora"
    (root / "pii" / "in-repo.jsonl").mkdir(parents=True)
    assert [(c, s) for c, s, _ in discover(root)] == [("pii", "in-repo")], (
        "the fixture proves nothing unless discovery finds this path"
    )
    assert main(["--corpora-dir", str(root)]) == 1


def test_an_unwritable_json_destination_exits_two(tmp_path: Path) -> None:
    """A destination under a directory that does not exist. The environment is
    wrong rather than the data, so 2 rather than 1."""
    root = _corpora(tmp_path)
    assert main(["--corpora-dir", str(root), "--json", str(tmp_path / "nope" / "b.json")]) == 2


def test_an_unwritable_md_destination_exits_two(tmp_path: Path) -> None:
    """The same guard one flag over. Pinned separately because the two calls
    pass different paths and different content, and a swap between them is the
    defect this project sees most."""
    root = _corpora(tmp_path)
    assert main(["--corpora-dir", str(root), "--md", str(tmp_path / "nope" / "B.md")]) == 2


def test_an_unwritable_baselines_destination_exits_two(tmp_path: Path) -> None:
    """--write-baselines is the third write, and the one whose failure would
    otherwise return 0: it is the last statement before its own `return 0`."""
    root = _corpora(tmp_path)
    out = tmp_path / "nope" / "baselines.json"
    assert main(["--corpora-dir", str(root), "--write-baselines", str(out)]) == 2


def test_each_artifact_is_written_to_its_own_destination(tmp_path: Path) -> None:
    """Written against the two writes swapped: markdown to --json, JSON to --md.

    Both files exist and both are non-empty under the swap, so a test that only
    checked they had been written could not see it.
    """
    root = _corpora(tmp_path)
    out_json, out_md = tmp_path / "b.json", tmp_path / "B.md"
    assert main(["--corpora-dir", str(root), "--json", str(out_json), "--md", str(out_md)]) == 0
    assert json.loads(out_json.read_text())["results"][0]["detector"] == "pii"
    assert out_md.read_text().startswith("# Benchmarks")


def test_an_unusable_epsilon_is_not_reported_as_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the classification collapsed to a constant either way.

    The exit code is 1 under both branches, which is correct and is exactly why
    it cannot tell them apart: the whole fix is which sentence is printed. Both
    halves are asserted, the label that must appear and the label that must not,
    because the false sentence is what is being removed.
    """
    from jamjet_guardrails.eval.corpus import load_corpus

    root = _corpora(tmp_path)
    version = load_corpus(root / "pii" / "in-repo.jsonl", name="pii/in-repo").version
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps(
            {
                f"pii/in-repo/{version}": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "decision_mismatches": 0,
                }
            }
        )
    )
    # The baseline MATCHES and the run MEETS it, so nothing here regressed and
    # the only possible fault is the flag.
    assert main(["--corpora-dir", str(root), "--gate", str(baselines), "--epsilon", "1.0"]) == 1
    err = capsys.readouterr().err
    assert "unusable --epsilon" in err
    assert "REGRESSION" not in err


def _matching_baselines(tmp_path: Path, root: Path, **fields: float) -> Path:
    """A well-formed baselines file under the key this run will actually produce.

    Written for the regression tests, which need the file to be BEYOND reproach
    so that the only thing left to fail on is the comparison. An earlier version
    of these tests used a deliberately non-matching key, which is a stale file
    rather than a score that moved, and the three-way label caught it: the test
    was asserting the same false sentence the label exists to remove.
    """
    from jamjet_guardrails.eval.corpus import load_corpus

    version = load_corpus(root / "pii" / "in-repo.jsonl", name="pii/in-repo").version
    out = tmp_path / "baselines.json"
    out.write_text(
        json.dumps(
            {
                f"pii/in-repo/{version}": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "decision_mismatches": 0,
                    **fields,
                }
            }
        )
    )
    return out


def test_a_real_regression_is_still_reported_as_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dual of the test above, and the half that makes it mean something.

    A classification stuck at "unusable epsilon" would pass that test alone
    while relabelling every genuine regression.

    The corpus scores 2/3 and 1/3 against a baseline recording 1.0 and 1.0
    under the RIGHT key with a well-formed entry, so the file is beyond
    reproach and the scores are the only thing wrong.
    """
    root = _imperfect_corpora(tmp_path)
    baselines = _matching_baselines(tmp_path, root, decision_mismatches=2)
    assert main(["--corpora-dir", str(root), "--gate", str(baselines)]) == 1
    err = capsys.readouterr().err
    assert "REGRESSION" in err
    assert "unusable --epsilon" not in err
    assert "stale or malformed" not in err


# The fix round: one Important and six Minors from review. Each test below names
# the mutation it was written against; the RED is recorded in task-13-report.md.


def test_a_missing_baselines_file_exits_two(tmp_path: Path) -> None:
    """Written against `except (OSError, json.JSONDecodeError)` reduced to
    `except json.JSONDecodeError`.

    Every other test reaching that clause supplies a file that EXISTS with bad
    content, so the OSError arm was pinned by nothing while the JSONDecodeError
    arm was pinned twice. A missing `--gate baselines.json` is the likeliest
    operator error against this command: first run, wrong path, or the file not
    committed yet.
    """
    root = _corpora(tmp_path)
    absent = tmp_path / "not-created-yet.json"
    assert not absent.exists()
    assert main(["--corpora-dir", str(root), "--gate", str(absent)]) == 2


def test_a_stale_baselines_file_is_not_reported_as_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the two-way label, which called this a regression.

    A key that does not correspond means the corpus was edited, renamed or
    added. Nothing regressed, and sending the reader to a detector diff that
    does not exist is the misdirection the epsilon fix exists to remove, one
    step over.
    """
    root = _corpora(tmp_path)
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps(
            {
                "pii/in-repo/does-not-match": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "decision_mismatches": 0,
                }
            }
        )
    )
    assert main(["--corpora-dir", str(root), "--gate", str(baselines)]) == 1
    err = capsys.readouterr().err
    assert "stale or malformed" in err
    assert "REGRESSION" not in err
    # The gate's own message still follows the label, so no detail is lost to
    # the classification.
    assert "no baseline recorded" in err


def _drop_precision(entry: dict[str, object]) -> object:
    del entry["precision"]
    return entry


# Every way an entry can fail to be the shape --write-baselines writes, one per
# clause of `_is_a_recorded_baseline`. Parametrised rather than merged, because
# a single entry damaged five ways would be refused by whichever clause runs
# first and would say nothing about the other four.
_DAMAGE: tuple[tuple[str, object], ...] = (
    ("a field is missing", _drop_precision),
    # Double-encoded JSON: the entry is a string, which answers `in` by
    # SUBSTRING and then raises on subscript.
    ("the entry is not an object", lambda entry: "precision=1.0"),
    ("a ratio is not a number", lambda entry: {**entry, "precision": "high"}),
    ("a ratio is outside [0, 1]", lambda entry: {**entry, "precision": 2.0}),
    # True == 1, so a bool reads as a perfect recorded score.
    ("a ratio is a bool", lambda entry: {**entry, "recall": True}),
    ("the count is negative", lambda entry: {**entry, "decision_mismatches": -1}),
    ("the count is not an integer", lambda entry: {**entry, "decision_mismatches": 0.5}),
    # A case can disagree about its decision at most once, and _corpora holds
    # one case, so 2 is a count this tool cannot have written.
    ("the count exceeds the cases scored", lambda entry: {**entry, "decision_mismatches": 2}),
    # null and a bare number are the shapes that need the isinstance(Mapping)
    # clause specifically: without it `set(entry)` raises TypeError and the
    # traceback escapes main. A STRING entry does not reach that far, because
    # the field-set clause below refuses it first, so the string case pins the
    # clause after this one and not this one.
    ("the entry is null", lambda entry: None),
    ("the entry is a bare number", lambda entry: 3),
)


@pytest.mark.parametrize(("label", "damage"), _DAMAGE, ids=[d[0] for d in _DAMAGE])
def test_a_malformed_baseline_entry_is_not_reported_as_a_regression(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    label: str,
    damage: Callable[[dict[str, object]], object],
) -> None:
    """The other half of Minor 3: the KEY corresponds and the entry does not.

    Pinned separately from the stale-key case because the classifier tests two
    different things, and pinned once per damaged shape because each is a
    separate clause of `_is_a_recorded_baseline`. Every one of these is a file
    someone hand-edited or a file this tool never wrote, and none of them is a
    score that moved.
    """
    root = _corpora(tmp_path)
    baselines = _matching_baselines(tmp_path, root)
    loaded = json.loads(baselines.read_text())
    key = next(iter(loaded))
    loaded[key] = damage(loaded[key])
    baselines.write_text(json.dumps(loaded))
    assert main(["--corpora-dir", str(root), "--gate", str(baselines)]) == 1
    err = capsys.readouterr().err
    assert "stale or malformed" in err, label
    assert "REGRESSION" not in err, label


def test_a_regression_at_the_strictest_tolerance_is_still_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the epsilon classifier's lower bound made exclusive.

    0.0 is the strictest tolerance a user can deliberately choose and the gate
    accepts it. The existing epsilon tests probe the upper bound at 1.0 and the
    lower only at -0.1, which is OUTSIDE it, so `<=` mutated to `<` on the
    lower side survived and would relabel every real regression found under
    `--epsilon 0` as a bad flag.
    """
    root = _imperfect_corpora(tmp_path)
    baselines = _matching_baselines(tmp_path, root, decision_mismatches=2)
    assert main(["--corpora-dir", str(root), "--gate", str(baselines), "--epsilon", "0"]) == 1
    err = capsys.readouterr().err
    assert "REGRESSION" in err
    assert "unusable --epsilon" not in err


def test_two_destinations_naming_one_path_are_refused(tmp_path: Path) -> None:
    """Written against the distinctness check removed.

    `--json X --md X` writes one artifact over the other and exits 0: a success
    that destroyed one of its own outputs, which is the exit-0 this task exists
    to prevent. The --write-baselines/--gate refusal cannot see it, because
    that one is about flag PRESENCE and these flags are legitimately present
    together.
    """
    root = _corpora(tmp_path)
    shared = tmp_path / "same.out"
    assert main(["--corpora-dir", str(root), "--json", str(shared), "--md", str(shared)]) == 2
    # Refused before anything was written, so neither artifact was produced at
    # all rather than one being produced and then overwritten.
    assert not shared.exists()


def test_one_path_spelled_two_ways_is_still_one_path(tmp_path: Path) -> None:
    """Two spellings of one file, so the check resolves before it compares.
    Written against `.resolve()` dropped.

    `sub/..` and not `./`: the first version of this test used `./b.json`, and
    `Path` collapses a `.` segment at CONSTRUCTION, so the two arguments were
    already the identical Path and the test passed with `.resolve()` removed.
    A `..` segment is preserved, because collapsing one lexically is wrong in
    the presence of a symlink, which is the whole reason `resolve` exists.
    """
    root = _corpora(tmp_path)
    out = tmp_path / "b.json"
    detour = tmp_path / "sub" / ".." / "b.json"
    (tmp_path / "sub").mkdir()
    assert detour != out, "the fixture proves nothing unless the two spellings differ"
    assert main(["--corpora-dir", str(root), "--json", str(out), "--md", str(detour)]) == 2
    assert not out.exists()


def test_a_recorded_baseline_may_not_overwrite_another_destination(tmp_path: Path) -> None:
    """The third destination, pinned on its own. --write-baselines and --gate
    are already mutually exclusive, so this is the only collision it can have,
    and without it that entry could be dropped from the list unnoticed."""
    root = _corpora(tmp_path)
    shared = tmp_path / "same.out"
    assert (
        main(
            [
                "--corpora-dir",
                str(root),
                "--json",
                str(shared),
                "--write-baselines",
                str(shared),
            ]
        )
        == 2
    )
    assert not shared.exists()


def test_a_gate_source_may_not_double_as_a_destination(tmp_path: Path) -> None:
    """`--json X --gate X` overwrites the baselines file and then gates against
    the report that replaced it. The worst member of the family, because it
    would not fail: it would pass, having destroyed the thing it was checking."""
    root = _corpora(tmp_path)
    baselines = _matching_baselines(tmp_path, root)
    before = baselines.read_text()
    assert (
        main(["--corpora-dir", str(root), "--json", str(baselines), "--gate", str(baselines)]) == 2
    )
    assert baselines.read_text() == before


def test_every_unscorable_corpus_is_reported_not_just_the_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the loop returning on the first failure.

    `gate.py` collects every problem into one message because "a gate that
    reports one failure per run costs one CI cycle per failure". The load loop
    carries the same argument and did not carry the behaviour. The two corpora
    here fail through DIFFERENT except arms, so this also pins that collection
    spans both.
    """
    root = tmp_path / "corpora"
    (root / "pii").mkdir(parents=True)
    # alpha is malformed content, which is a CorpusError; beta is a DIRECTORY
    # named like a corpus, which the loader opens and gets IsADirectoryError
    # from. Those land in the two DIFFERENT except arms.
    #
    # The first version of this fixture paired a malformed corpus with an
    # uninstalled detector. CorpusError and GuardrailUnavailableError are both
    # plain Exception subclasses caught by the SAME arm, so it spanned one arm
    # while its name and its docstring both said two, and reverting only the
    # OSError arm to a fail-fast left the suite green.
    (root / "pii" / "alpha.jsonl").write_text(json.dumps({"id": "pii-0001"}) + "\n")
    (root / "pii" / "beta.jsonl").mkdir()
    assert main(["--corpora-dir", str(root)]) == 1
    err = capsys.readouterr().err
    assert "alpha.jsonl" in err
    assert "beta.jsonl" in err


def test_the_measurement_survives_a_failed_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the summary printed after the writes.

    The scoring succeeded; only the filesystem refused. Discarding the numbers
    because the destination was wrong loses the one record of a run that did
    the expensive part correctly.
    """
    root = _corpora(tmp_path)
    assert main(["--corpora-dir", str(root), "--json", str(tmp_path / "nope" / "b.json")]) == 2
    captured = capsys.readouterr()
    assert "P=1.000" in captured.out
    assert "cannot write the --json artifact" in captured.err


def test_the_usage_line_names_the_command_that_exists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The console script takes no positional argument, so a usage line reading
    `jamjet-guardrails eval ...` instructs the reader to type something argparse
    then refuses. False in every state rather than in a reachable one."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage: jamjet-guardrails [" in out
    assert "jamjet-guardrails eval" not in out


def test_a_looped_symlink_destination_exits_two(tmp_path: Path) -> None:
    """The path comparison must not raise on a symlink loop.

    `Path.resolve()` is non-strict and still raises `RuntimeError: Symlink loop`
    on 3.10, 3.11 and 3.12; only 3.13 and later return the path. This package's
    floor is 3.10, so an unguarded resolve turns this input into a traceback and
    an exit 1, the code reserved for the DATA being wrong. It is an invocation
    fault, and it was a clean 2 before the distinctness check existed.

    Note honestly what this test can and cannot see: on 3.13+ it passes under
    BOTH `resolve` and `realpath`, so on this interpreter it pins the outcome
    rather than the bug. The interpreter sweep in task-13-report.md is the
    evidence for the versions that differ.
    """
    root = _corpora(tmp_path)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    assert main(["--corpora-dir", str(root), "--json", str(loop)]) == 2


def test_a_destination_inside_the_corpora_tree_is_refused(tmp_path: Path) -> None:
    """Written against the containment check removed, and against it written as
    equality.

    `--json corpora/pii/in-repo.jsonl` printed the summary, overwrote the
    corpus and exited 0: a success that destroyed one of its own INPUTS. Worse
    than overwriting an artifact, because `corpus_version` is derived from
    corpus content, so the damage propagates into every baseline key the next
    run computes. A file inside a directory never equals it, so equality cannot
    see this.
    """
    root = _corpora(tmp_path)
    corpus = root / "pii" / "in-repo.jsonl"
    before = corpus.read_text()
    assert main(["--corpora-dir", str(root), "--json", str(corpus)]) == 2
    assert corpus.read_text() == before


def test_the_baselines_path_the_plan_prescribes_is_accepted(tmp_path: Path) -> None:
    """`--corpora-dir corpora --write-baselines corpora/baselines.json` must run.

    That is the line Task 15 Step 1 executes. The first version of this guard
    refused anything under the corpora root, which refused this while ACCEPTING
    `--gate` on the same path: a guard coarser than the harm, since `discover`
    reads only `<check>/<source>.jsonl` two levels down, so a `.json` at the
    root is not evidence and destroying it is not possible.
    """
    root = _corpora(tmp_path)
    out = root / "baselines.json"
    assert main(["--corpora-dir", str(root), "--write-baselines", str(out)]) == 0
    assert out.exists()
    # And writing it there did not turn it into a corpus.
    assert [(c, s) for c, s, _ in discover(root)] == [("pii", "in-repo")]


def test_a_regression_at_the_widest_accepted_tolerance_is_still_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The upper boundary of the accepted interval, which nothing probed.

    Every epsilon the other tests use agrees under `<=` and `<`, so the mutant
    survived and would relabel a genuine regression at exactly MAX_EPSILON as an
    unusable flag: the same false sentence Minor 2 closed at the other end of
    the same interval. MAX_EPSILON is imported rather than written out, so this
    tracks the gate if the gate moves it.
    """
    root = _imperfect_corpora(tmp_path)
    baselines = _matching_baselines(tmp_path, root, decision_mismatches=2)
    argv = ["--corpora-dir", str(root), "--gate", str(baselines), "--epsilon", str(MAX_EPSILON)]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "REGRESSION" in err
    assert "unusable --epsilon" not in err


def test_a_regression_behind_a_stale_key_is_named_as_well(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the stale label suppressing the regression.

    "A detector regressed AND a new corpus has no baseline yet" is an ordinary
    state of a pull request. Reporting only the file problem sends the reader to
    add a baseline and stop looking, which is how the regression ships.
    """
    root = _imperfect_corpora(tmp_path)
    baselines = _matching_baselines(tmp_path, root, decision_mismatches=2)
    loaded = json.loads(baselines.read_text())
    loaded["pii/gone/000000000000"] = {
        "precision": 1.0,
        "recall": 1.0,
        "decision_mismatches": 0,
    }
    baselines.write_text(json.dumps(loaded))
    assert main(["--corpora-dir", str(root), "--gate", str(baselines)]) == 1
    err = capsys.readouterr().err
    assert "stale or malformed" in err
    assert "REGRESSION" in err


def test_a_stale_key_without_a_regression_says_only_that(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dual of the test above. A classifier that always appended the
    regression clause would pass that one while relabelling every clean stale
    file as a detector that got worse."""
    root = _corpora(tmp_path)
    baselines = _matching_baselines(tmp_path, root)
    loaded = json.loads(baselines.read_text())
    loaded["pii/gone/000000000000"] = {
        "precision": 1.0,
        "recall": 1.0,
        "decision_mismatches": 0,
    }
    baselines.write_text(json.dumps(loaded))
    assert main(["--corpora-dir", str(root), "--gate", str(baselines)]) == 1
    err = capsys.readouterr().err
    assert "stale or malformed" in err
    assert "REGRESSION" not in err


def test_a_corpus_with_no_baseline_yet_is_not_reported_as_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the MISSING-KEY half of the classifier removed.

    A file that is entirely well formed for the corpora it does cover, and
    silent about one that was scored, is the state of every pull request that
    adds a corpus. The other stale-file tests all put a key in the file that the
    run does not produce, which the first half of the classifier catches; this
    is the opposite direction and nothing reached it, so the second half was
    unpinned and every added corpus would have been announced as a detector that
    got worse.
    """
    root = _corpora(tmp_path)
    (root / "pii" / "third-party.jsonl").write_text(
        json.dumps(dict(CASE, source="third-party")) + "\n"
    )
    # A baseline for in-repo only, correct in every respect, and nothing for the
    # corpus just added.
    baselines = _matching_baselines(tmp_path, root)
    # Asserted BY VALUE, not by counting. The kill depends on
    # `set(usable) == set(baselines)` holding, which needs this key to be the
    # in-repo run's OWN key; "exactly one key" is true of a stale file too, so
    # the earlier form would have degenerated into a duplicate of the stale-file
    # test the moment `_matching_baselines` drifted.
    from jamjet_guardrails.eval.corpus import load_corpus

    version = load_corpus(root / "pii" / "in-repo.jsonl", name="pii/in-repo").version
    assert set(json.loads(baselines.read_text())) == {f"pii/in-repo/{version}"}
    assert main(["--corpora-dir", str(root), "--gate", str(baselines)]) == 1
    err = capsys.readouterr().err
    assert "stale or malformed" in err
    assert "REGRESSION" not in err


def test_a_symlinked_check_directory_still_protects_its_corpora(tmp_path: Path) -> None:
    """Written against comparing a destination's REAL path to the tree root.

    A check directory that is a symlink out of the tree puts the destination's
    real path outside the root, so the root comparison passed while the write
    still landed on a corpus: same command line, same exit 0, same corpus
    destroyed. Comparing against the discovered PATHS closes it, because the
    corpus and the destination resolve to the same file whichever way round the
    symlink is named.
    """
    shared = tmp_path / "shared" / "pii"
    shared.mkdir(parents=True)
    (shared / "in-repo.jsonl").write_text(json.dumps(CASE) + "\n")
    root = tmp_path / "corpora"
    root.mkdir()
    (root / "pii").symlink_to(shared, target_is_directory=True)

    corpus = root / "pii" / "in-repo.jsonl"
    before = corpus.read_text()
    assert main(["--corpora-dir", str(root), "--json", str(corpus)]) == 2
    assert corpus.read_text() == before


def test_a_symlinked_corpus_file_still_protects_its_target(tmp_path: Path) -> None:
    """The other aliasing shape, and the ordinary layout for a vendored corpus:
    the FILE is a symlink to somewhere outside the tree."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "data.jsonl").write_text(json.dumps(CASE) + "\n")
    root = tmp_path / "corpora"
    (root / "pii").mkdir(parents=True)
    corpus = root / "pii" / "in-repo.jsonl"
    corpus.symlink_to(vendor / "data.jsonl")

    before = (vendor / "data.jsonl").read_text()
    assert main(["--corpora-dir", str(root), "--json", str(corpus)]) == 2
    assert (vendor / "data.jsonl").read_text() == before


def test_a_corpus_named_by_its_real_path_is_protected_too(tmp_path: Path) -> None:
    """The reverse direction: the corpora tree is reached through a symlink and
    the DESTINATION is spelled as the real file. Neither side may be trusted to
    be the canonical spelling, which is why both sets are compared."""
    shared = tmp_path / "shared" / "pii"
    shared.mkdir(parents=True)
    (shared / "in-repo.jsonl").write_text(json.dumps(CASE) + "\n")
    root = tmp_path / "corpora"
    root.mkdir()
    (root / "pii").symlink_to(shared, target_is_directory=True)

    real = shared / "in-repo.jsonl"
    before = real.read_text()
    assert main(["--corpora-dir", str(root), "--json", str(real)]) == 2
    assert real.read_text() == before


def test_a_missing_corpora_tree_is_reported_as_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Written against the collision check running before the directory test.

    `--corpora-dir nope --json nope/out.json` announced a corpora tree that does
    not exist, which hides the real fault. Exit 2 either way, so this is a
    message guard: the same rule the corpora-path message itself carries, that a
    message must be true in every state that reaches it.
    """
    absent = tmp_path / "nope"
    assert main(["--corpora-dir", str(absent), "--json", str(absent / "out.json")]) == 2
    err = capsys.readouterr().err
    assert "is missing or is not a directory" in err
    assert "corpus this run reads" not in err


def test_a_destination_that_is_a_symlink_to_a_corpus_is_refused(tmp_path: Path) -> None:
    """Written against the destination being compared by its LEXICAL path only.

    The corpora tree here is entirely ordinary; the DESTINATION is the symlink.
    `write_text` follows it, so the write lands on the corpus while the path
    typed on the command line resembles nothing under the tree. Lexical
    comparison alone cannot see this, which is why the destination is compared
    by both spellings and not just the one that reads naturally.
    """
    root = _corpora(tmp_path)
    corpus = root / "pii" / "in-repo.jsonl"
    decoy = tmp_path / "report.json"
    decoy.symlink_to(corpus)

    before = corpus.read_text()
    assert main(["--corpora-dir", str(root), "--json", str(decoy)]) == 2
    assert corpus.read_text() == before


# ==========================================================================
# The two ways this tool can be run, and the encoding the artifacts carry.
# ==========================================================================


def test_the_module_form_actually_runs(tmp_path: Path) -> None:
    """`python -m jamjet_guardrails.eval.cli` used to exit 0 having measured nothing.

    No `if __name__ == "__main__"` guard: the module defined `main`, called
    nothing and returned success. This is the module whose own docstring says
    "a run that measured nothing must never exit 0" and warns about "a green
    check meaning the benchmark never ran", and the `-m` form is the first
    thing a reader tries when the console script is not on PATH.

    A subprocess, because that is the only way to exercise the guard: importing
    the module in-process sets `__name__` to its dotted path by construction.
    """
    root = tmp_path / "corpora"
    (root / "pii").mkdir(parents=True)
    (root / "pii" / "in-repo.jsonl").write_text(json.dumps(CASE) + "\n", encoding="utf-8")
    out = tmp_path / "b.json"

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "jamjet_guardrails.eval.cli",
            "--corpora-dir",
            str(root),
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["results"], run.stdout


def test_the_module_form_forwards_the_exit_code(tmp_path: Path) -> None:
    """The half that matters: the guard has to return main's code, not just call it.

    A bare `main()` under the guard runs the benchmark and still exits 0, which
    is the same green-check-over-nothing one step over.
    """
    run = subprocess.run(
        [sys.executable, "-m", "jamjet_guardrails.eval.cli", "--corpora-dir", str(tmp_path / "no")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0, run.stdout


def test_the_md_artifact_is_written_as_utf8_whatever_the_platform_default(
    tmp_path: Path,
) -> None:
    """`--md` renders a corpus name verbatim, so non-ASCII reaches `write_text`.

    `eval/corpus.py` pins UTF-8 on the read side and
    `scripts/sample_nemotron.py` pins its write; `cli.py` used the platform
    default for both of its artifacts. On a cp1252 default this either raises,
    and is reported as "cannot write", which names the wrong fault, or writes
    mojibake into a PUBLISHED artifact.

    Asserted by decoding the BYTES as UTF-8 rather than by reading with the
    default, which would be the same assumption the code is being held to.
    `--json` is not the fixture here because `json.dumps` escapes non-ASCII on
    its own, so it could not tell the two encodings apart.
    """
    root = tmp_path / "corpora"
    (root / "pii").mkdir(parents=True)
    (root / "pii" / f"{NON_ASCII}.jsonl").write_text(
        json.dumps(dict(CASE, source=NON_ASCII)) + "\n", encoding="utf-8"
    )
    out_md = tmp_path / "b.md"

    assert main(["--corpora-dir", str(root), "--md", str(out_md)]) == 0
    assert NON_ASCII in out_md.read_bytes().decode("utf-8")


def test_no_shipped_code_reads_or_writes_with_the_platform_default(tmp_path: Path) -> None:
    """The class, not the two sites. `-X warn_default_encoding` finds them all.

    Enumerating the offending lines by grep is the mistake this whole review is
    about: the answer would be right today and silent about the next one. CPython
    already knows how to answer it, so the domain is derived rather than listed.
    Every artifact path this tool has is exercised in one run, with
    EncodingWarning promoted to an error, so a `read_text` or `write_text` added
    anywhere under it without an explicit encoding fails here.
    """
    root = tmp_path / "corpora"
    (root / "pii").mkdir(parents=True)
    (root / "pii" / "in-repo.jsonl").write_text(json.dumps(CASE) + "\n", encoding="utf-8")
    baselines = tmp_path / "baselines.json"
    strict = [
        sys.executable,
        "-X",
        "warn_default_encoding",
        "-W",
        "error::EncodingWarning",
        "-m",
        "jamjet_guardrails.eval.cli",
        "--corpora-dir",
        str(root),
    ]

    write = subprocess.run(
        [
            *strict,
            "--json",
            str(tmp_path / "b.json"),
            "--md",
            str(tmp_path / "b.md"),
            "--write-baselines",
            str(baselines),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert write.returncode == 0, write.stderr

    # The read path is a separate invocation because --write-baselines returns
    # before the gate runs, so one run cannot cover both.
    gate = subprocess.run(
        [*strict, "--gate", str(baselines)], capture_output=True, text=True, check=False
    )
    assert gate.returncode == 0, gate.stderr
