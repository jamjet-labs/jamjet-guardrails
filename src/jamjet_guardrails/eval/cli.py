"""`jamjet-guardrails`: run every corpus, publish the numbers, gate on them.

This is the seam where every module above becomes one exit code, and CI runs
exactly this command. The README's headline claim is true only if this file
behaves, so the question asked of every branch here is the one
``eval.gate`` asks of its own: what reaches a ``return 0``, and is success the
right answer in every one of those states.

Three exit codes carry the meaning. ``0`` is success, ``1`` is a regression or a
corpus that could not be scored, ``2`` is a bad argument. ``2`` covers the two
shapes of "there was nothing to measure" as well, because a run that measured
nothing must never exit 0: a missing corpora directory and a directory that
exists and matches nothing are both a misconfigured invocation, and both would
otherwise produce a green check meaning the benchmark never ran. A destination
this process cannot write is a ``2`` for the same reason: the data is fine and
the environment is wrong. So is one path named twice, because a run that
overwrote one of its own outputs and reported success is that same green check
with the evidence destroyed.

The line between ``1`` and ``2`` is where the fault is. A corpus that will not
load or will not read is the DATA, so it is a ``1`` next to the regression it
sits beside; a path that cannot be written and a flag that cannot be honoured
are the INVOCATION, so they are a ``2``. Neither ever escapes as a traceback: a
traceback out of a CLI is a bad message rather than a bad outcome, but a bad
message from a gate is how a real fault gets read as a broken tool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from jamjet_guardrails.detectors import build
from jamjet_guardrails.errors import GuardrailUnavailableError
from jamjet_guardrails.eval.corpus import CorpusError, load_corpus
from jamjet_guardrails.eval.fixtures import options_for
from jamjet_guardrails.eval.gate import (
    MAX_EPSILON,
    RegressionError,
    baseline_key,
    check_regression,
)
from jamjet_guardrails.eval.metrics import Evaluation, EvaluationError, evaluate
from jamjet_guardrails.eval.report import to_json, to_markdown


def discover(corpora_dir: Path) -> list[tuple[str, str, Path]]:
    """Find corpora/<check>/<source>.jsonl, sorted for stable report ordering.

    The directory name is the detector name handed to ``build``; the file stem
    is the source label reported beside the number. Sorted by check and then by
    source, explicitly and not by path, so the two committed artifacts diff
    against a fresh run rather than against whatever order the filesystem
    happened to hand back.
    """
    found = [
        (path.parent.name, path.stem, path) for path in sorted(Path(corpora_dir).glob("*/*.jsonl"))
    ]
    return sorted(found, key=lambda triple: (triple[0], triple[1]))


def _real(path: Path) -> Path:
    """An absolute, symlink-followed path for comparison, on every version we support.

    ``os.path.realpath`` and NOT ``Path.resolve``. A non-strict ``resolve``
    raises ``RuntimeError: Symlink loop`` on 3.10, 3.11 and 3.12, measured here
    on 3.10.20, 3.11.7 and 3.12.13; only 3.13 and later return the path. This
    package's floor is 3.10 and CI runs that leg, so a looped-symlink
    destination would escape ``main`` as a traceback and exit 1, the code this
    module reserves for the DATA being wrong, for what is an invocation fault
    worth a 2.

    ``realpath`` detects the loop and returns the path unresolved instead, so
    the write attempt goes on to reach ``_write_artifact``, fail with
    ``OSError`` (ELOOP), and be reported as the clean 2 it was before this
    comparison existed at all.
    """
    return Path(os.path.realpath(path))


def _lexical(path: Path) -> Path:
    """An absolute path normalised WITHOUT touching the filesystem.

    The companion to ``_real``, and kept beside it because the two fail in
    different directions. ``_real`` follows symlinks, which is what catches two
    names for one file; it also returns its input unchanged when it cannot
    resolve, which a symlink loop makes it do. This one cannot consult the
    filesystem at all, so it always answers, and it guarantees that the literal
    path a caller typed is never the literal path of a file this run reads.
    """
    return Path(os.path.abspath(path))


_BASELINE_FIELDS = ("precision", "recall", "decision_mismatches")


def _recorded(ev: Evaluation) -> dict[str, float]:
    """One baseline record: what this run measured, rounded DOWN to three decimals.

    Down, because the gate's epsilon is there to absorb float noise rather than
    to mask a real drop, and a value recorded above what the run measured fails
    forever against the detector that produced it.

    Factored out because two things need to agree about this shape: the writer
    below, and ``_is_a_recorded_baseline``, which asks whether a file on disk
    looks like something this tool wrote.
    """
    return {
        "precision": math.floor(ev.overall.precision * 1000) / 1000,
        "recall": math.floor(ev.overall.recall * 1000) / 1000,
        "decision_mismatches": ev.decision_mismatches,
    }


def _is_a_recorded_baseline(entry: object, cases: int) -> bool:
    """Whether one entry is the shape ``--write-baselines`` emits.

    The CLI owns the WRITE format; ``eval.gate`` owns the COMPARISON. Asking
    whether a file is in the format this tool writes is the writer checking its
    own output, not a second copy of the gate's rules, and it decides nothing:
    the gate has already refused by the time this is asked, and all it chooses
    is which sentence says so.

    The range tests do the work of a finiteness test as well, and deliberately:
    NaN fails every comparison, both infinities fall outside, and an integer too
    large to convert to a float is refused by the bound without a conversion
    that would raise ``OverflowError``. ``json.loads`` can produce all three.

    ``cases`` is the count this run scored, so the upper bound on
    ``decision_mismatches`` is checked here too. It is a property of the RUN
    rather than of the file, which is why it arrives as an argument; leaving it
    to the gate alone was the one message in this family that still pointed at
    the wrong thing.
    """
    if not isinstance(entry, Mapping):
        return False
    if not set(entry) >= set(_BASELINE_FIELDS):
        return False
    for ratio in ("precision", "recall"):
        value = entry[ratio]
        # bool before int, because True == 1 reads as a perfect recorded score.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not 0.0 <= value <= 1.0:
            return False
    count = entry["decision_mismatches"]
    if isinstance(count, bool) or not isinstance(count, int):
        return False
    # A case can disagree about its decision at most once, so a count above
    # the case count is not a number this tool can ever have written.
    return 0 <= count <= cases


def _usable_baselines(
    baselines: Mapping[str, object], evaluations: Sequence[Evaluation]
) -> dict[str, Evaluation]:
    """The evaluations whose baseline entry this tool could have written itself.

    One dict answering both questions a refusal raises: whether the FILE is
    stale or damaged (its keys and the run's do not correspond, or an entry is
    not the shape ``--write-baselines`` emits), and whether a real regression is
    hiding behind that damage.
    """
    by_key = {baseline_key(ev): ev for ev in evaluations}
    return {
        key: by_key[key]
        for key, entry in baselines.items()
        if key in by_key and _is_a_recorded_baseline(entry, by_key[key].cases)
    }


def _hides_a_regression(
    usable: Mapping[str, Evaluation], baselines: Mapping[str, object], epsilon: float
) -> bool:
    """Whether a score also moved, among the entries that are usable at all.

    Asked of the GATE rather than recomputed, by re-gating the corresponding
    subset on its own. "A detector regressed AND a new corpus has no baseline
    yet" is an ordinary state of a pull request, and reporting only the second
    of those sends the reader to add a baseline and stop looking.
    """
    if not usable:
        return False
    # Rebuilt with an isinstance test rather than a subscript, because the
    # entries are `object` until something narrows them. `_is_a_recorded_baseline`
    # has already established that every usable one IS a mapping, so this
    # re-establishes it for the type checker and drops nothing.
    subset = {
        key: entry
        for key, entry in baselines.items()
        if key in usable and isinstance(entry, Mapping)
    }
    try:
        check_regression(list(usable.values()), subset, epsilon=epsilon)
    except RegressionError:
        return True
    return False


def _write_artifact(path: Path, text: str, flag: str) -> bool:
    """Write one artifact, or say which destination refused it.

    An unwritable path is the environment being wrong rather than the data, so
    the caller turns a False into a ``2``. Reported here rather than left to
    escape, because ``write_text`` raises ``FileNotFoundError`` for a missing
    parent directory and ``PermissionError`` for a read-only one, and a
    traceback naming neither the flag nor what the run was doing is a worse
    message for the same outcome.
    """
    try:
        # UTF-8 explicitly, never the platform default. `eval/corpus.py` pins the
        # read side for the same reason, and `scripts/sample_nemotron.py` pins
        # its write. Latent today, since nothing in either artifact is non-ASCII,
        # and not latent by design: a corpus case id or a finding type carrying
        # one non-ASCII character reaches here, and on a cp1252 default this
        # either raises, and gets reported as "cannot write", which names the
        # wrong fault, or writes mojibake into a PUBLISHED artifact.
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"cannot write the {flag} artifact {path}: {exc}", file=sys.stderr)
        return False
    return True


def _epsilon_is_unusable(epsilon: float) -> bool:
    """Whether the gate's own bound refuses this tolerance.

    This decides no outcome. ``check_regression`` has already decided, and it
    raises the same ``RegressionError`` for an unusable tolerance as for a real
    regression; all this chooses is which sentence describes that decision, so
    that a configuration error does not announce itself as a score moving.

    The bound is READ from ``MAX_EPSILON`` rather than restated, so it still
    lives in the one module that owns it and there is nothing here to drift out
    of step with the gate. The range test alone covers NaN and both infinities,
    since every comparison against NaN is False and neither infinity is inside
    the interval; the gate keeps those apart because it must name each in its
    own message, and this only has to sort them into one of two sentences. The
    gate's other refusals (a bool, a string) cannot arrive here at all, because
    argparse's ``type=float`` has already rejected them with its own exit 2.
    """
    # Both bounds inclusive. 0.0 is the strictest tolerance a user can
    # deliberately choose and the gate accepts it, so an exclusive lower bound
    # here would relabel every real regression found under `--epsilon 0` as a
    # bad flag.
    return not 0.0 <= epsilon <= MAX_EPSILON


def main(argv: Sequence[str] | None = None) -> int:
    # `jamjet-guardrails`, with no subcommand: the console script takes no
    # positional argument, so a prog of "jamjet-guardrails eval" printed a usage
    # line instructing the reader to type something argparse then refuses. False
    # in every state rather than in a reachable one.
    parser = argparse.ArgumentParser(prog="jamjet-guardrails")
    parser.add_argument("--corpora-dir", default="corpora", type=Path)
    parser.add_argument("--json", dest="json_out", type=Path)
    parser.add_argument("--md", dest="md_out", type=Path)
    parser.add_argument("--gate", dest="baselines", type=Path)
    parser.add_argument(
        "--write-baselines",
        type=Path,
        help=(
            "record this run's scores as the baseline file, rounded DOWN to three "
            "decimals so the gate's epsilon absorbs float noise rather than masking "
            "a real drop. Refused together with --gate."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.005,
        help=(
            "tolerance for a score below baseline. Bounded by the gate itself: "
            "a value outside [0, 0.05], or NaN, is refused rather than honoured."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.write_baselines and args.baselines:
        print(
            "--write-baselines and --gate are mutually exclusive: a run that "
            "records its own baseline and then gates on it always passes",
            file=sys.stderr,
        )
        return 2

    # No two of the four paths this invocation names may be the same file. `--json X --md X` writes one
    # artifact over the other and exits 0, a success that destroyed one of its
    # own outputs; `--json X --gate X` overwrites the baselines file and then
    # gates against the report that replaced it. `--gate` is in this set because
    # it can be CLOBBERED, even though it is only ever read.
    #
    # The same argument as the --write-baselines/--gate refusal above, one level
    # down. That pair is about flag PRESENCE and cannot see this, because these
    # flags are all legitimately present together: it is their VALUES that
    # collide.
    destinations: tuple[tuple[str, Path | None], ...] = (
        ("--json", args.json_out),
        ("--md", args.md_out),
        ("--write-baselines", args.write_baselines),
    )
    seen: dict[Path, str] = {}
    for flag, named in (*destinations, ("--gate", args.baselines)):
        if named is None:
            continue
        real = _real(named)
        if real in seen:
            print(
                f"{seen[real]} and {flag} name the same path {named}; "
                "each would overwrite or re-read the other",
                file=sys.stderr,
            )
            return 2
        seen[real] = flag

    if not args.corpora_dir.is_dir():
        # Both states named, because the check covers both and only one of them
        # is "no such directory". A path that exists as a FILE reaches here too,
        # and telling that reader their directory is missing sends them looking
        # for something that is not the problem.
        print(
            f"corpora path {args.corpora_dir} is missing or is not a directory",
            file=sys.stderr,
        )
        return 2

    corpora = discover(args.corpora_dir)
    if not corpora:
        # Distinct from the check above: this IS a directory, and the glob
        # matched nothing inside it. Silently evaluating zero corpora and
        # exiting 0 is how a published-numbers gate becomes decorative.
        print(
            f"no corpora found under {args.corpora_dir} (expected <check>/<source>.jsonl)",
            file=sys.stderr,
        )
        return 2

    # No destination may be a file this run READS. Checked here, below
    # discovery, because it is the discovered paths it compares against, and
    # because running it earlier made it announce a corpora tree before anything
    # had established that the tree exists.
    #
    # Against the discovered paths and NOT against the corpora root, for two
    # independent reasons.
    #
    # The root rule was too coarse. `discover` globs `<check>/<source>.jsonl`
    # exactly two levels down, so a `.json` at the tree root is not evidence and
    # cannot be destroyed; the root rule refused
    # `--write-baselines corpora/baselines.json`, which is where the plan keeps
    # the baselines, while accepting `--gate` on that same path.
    #
    # The root rule was also unsound in the direction it existed to cover. It
    # resolved both sides, so a check directory or a corpus file that is a
    # SYMLINK out of the tree put the destination's real path outside the root
    # and the guard passed while the write still landed on a corpus. That is the
    # ordinary shape of a vendored third-party corpus, which is exactly what
    # this layout models.
    #
    # Both spellings of every path, unioned, and a destination matching any of
    # them is the same file. What each half actually earns, stated exactly,
    # because an earlier version of this comment credited the lexical half with
    # a job it does not do:
    #
    #   - The RESOLVED corpus path is load-bearing in both directions: it is
    #     what catches a check directory or a corpus file that is a symlink out
    #     of the tree.
    #   - The RESOLVED destination is load-bearing on its own: a destination
    #     that is itself a symlink to a corpus resembles nothing under the tree
    #     lexically, and `write_text` follows it.
    #   - The two LEXICAL halves are defensive and no reachable input
    #     distinguishes them. If a destination's lexical path equals a corpus's
    #     lexical path they are one string and resolve alike; and a resolved
    #     path cannot equal an unresolved one that differs from it, because
    #     `realpath` output has no symlink components left to differ by. They
    #     are kept as cheap insurance against a platform where `realpath`
    #     answers differently, not because a test can tell them from their own
    #     absence.
    #
    # Scope line, deliberate: this refuses destroying a corpus, not CREATING
    # one. `--json corpora/pii/new.jsonl` writes a file the next run would
    # discover and fail to load, which is loud, one run later, and not evidence
    # being lost.
    reads = {_real(path) for _, _, path in corpora} | {_lexical(path) for _, _, path in corpora}
    for flag, named in destinations:
        if named is not None and {_real(named), _lexical(named)} & reads:
            print(
                f"{flag} {named} is a corpus this run reads; writing it would destroy "
                "the evidence the published numbers are measured on",
                file=sys.stderr,
            )
            return 2

    evaluations: list[Evaluation] = []
    unscored: list[str] = []
    for check, source, path in corpora:
        try:
            corpus = load_corpus(path, name=f"{check}/{source}")
            evaluations.append(evaluate(build(check, **options_for(check)), corpus))
        except (CorpusError, EvaluationError, GuardrailUnavailableError) as exc:
            # EvaluationError covers a corpus paired with a guardrail that does
            # not run on its direction. Task 10 made it a typed error rather
            # than a bare ValueError precisely so this net can hold it.
            unscored.append(str(exc))
        except OSError as exc:
            # A corpus that discovery FOUND and the loader cannot read: a
            # permission bit, a path that is a directory, a file deleted between
            # the glob and the open. `load_corpus` deliberately leaves these
            # unwrapped, because an absent corpus and a malformed one are
            # different faults, so the net that catches malformed has to name
            # this one separately. It is the data rather than the invocation, so
            # it exits 1 beside the corpus errors above and not 2.
            unscored.append(f"cannot read corpus {path}: {exc}")

    if unscored:
        # Every corpus is tried before any is reported, for the reason gate.py
        # gives for collecting its own problems: a run that reports one failure
        # per invocation costs one CI cycle per failure. Nothing is published,
        # because a set of numbers missing the corpora that would not score is
        # a smaller benchmark wearing the full one's name.
        for problem in unscored:
            print(problem, file=sys.stderr)
        return 1

    # Printed BEFORE the writes. A destination that cannot be written must not
    # also discard the numbers the run just measured: the measurement succeeded
    # and belongs in the CI log whatever the filesystem then does with it.
    for ev in evaluations:
        print(
            f"{ev.detector:10} {ev.corpus_source:12} "
            f"P={ev.overall.precision:.3f} R={ev.overall.recall:.3f} "
            f"({ev.corpus_version})"
        )

    # Both attempted, then reported together, the same argument the corpus loop
    # above carries. Two bad destinations named one per run is two CI cycles.
    #
    # A run that writes --json and then fails on --md leaves one artifact on
    # disk beside an exit 2. That is ACCEPTED rather than overlooked: it is an
    # I/O failure rather than a logic error, and Task 15 diffs the committed
    # artifacts against a fresh run, so a partial or stale file fails CI there.
    # Temp-and-rename machinery would buy atomicity nothing here needs.
    json_written = args.json_out is None or _write_artifact(
        args.json_out, json.dumps(to_json(evaluations), indent=2) + "\n", "--json"
    )
    md_written = args.md_out is None or _write_artifact(
        args.md_out, to_markdown(evaluations), "--md"
    )
    if not (json_written and md_written):
        return 2

    if args.write_baselines:
        recorded = json.dumps(
            {baseline_key(ev): _recorded(ev) for ev in evaluations},
            indent=2,
            sort_keys=True,
        )
        if not _write_artifact(args.write_baselines, recorded + "\n", "--write-baselines"):
            return 2
        return 0

    if args.baselines:
        # A baselines file that cannot be read is a bad argument, not a passing
        # gate. Letting json.JSONDecodeError escape would work by accident today
        # and stop working the moment anything wraps main().
        try:
            # UTF-8, matching the write above and `corpus.load_corpus`. A
            # baselines file this tool wrote on one machine has to read back on
            # another, and the platform default is the one thing in that round
            # trip that differs between them.
            baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read baselines {args.baselines}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(baselines, dict):
            print(f"baselines {args.baselines} is not a JSON object", file=sys.stderr)
            return 2
        try:
            check_regression(evaluations, baselines, epsilon=args.epsilon)
        except RegressionError as exc:
            # One exception type, three very different things to tell the
            # reader. All three exit 1, and that is right: non-zero fails CI,
            # which is the direction that matters, and refusing either of the
            # first two here instead would put the gate's rules in two places.
            # What must not happen is any of them announcing itself as a score
            # that moved. "REGRESSION: epsilon must be within [0, 0.05]" and
            # "REGRESSION: baseline records no precision" both send a reader to
            # read a detector diff that does not exist.
            #
            # The gate's own message follows the label and is printed verbatim
            # in every case, so a misclassification here costs a wrong headline
            # and never a lost detail.
            usable = _usable_baselines(baselines, evaluations)
            if _epsilon_is_unusable(args.epsilon):
                label = f"unusable --epsilon {args.epsilon!r}"
            elif set(usable) != set(baselines) or set(usable) != {
                baseline_key(ev) for ev in evaluations
            }:
                # An entry this tool could not have written, or a key on one side
                # and not the other: a stale or hand-damaged FILE, not a detector
                # that got worse. Compared with the gate's own `baseline_key`, so
                # there is no second idea of what a key is.
                label = f"stale or malformed baselines {args.baselines}"
                if _hides_a_regression(usable, baselines, args.epsilon):
                    # Both, when both. A damaged file does not suspend the
                    # question the gate exists to answer.
                    label += ", and a REGRESSION among the entries that are usable"
            else:
                label = "REGRESSION"
            print(f"{label}: {exc}", file=sys.stderr)
            return 1

    return 0


# Without this, `python -m jamjet_guardrails.eval.cli --gate ...` imports the
# module, defines `main`, calls nothing and exits 0. This is the module whose
# own docstring says "a run that measured nothing must never exit 0" and warns
# about "a green check meaning the benchmark never ran", and the `-m` form is
# the first thing a reader tries when the console script is not on PATH.
#
# `sys.exit(main())`, not `main()`: the exit code is the whole product of this
# tool, and a `-m` invocation that measured a regression and exited 0 would be
# the same failure by a different route.
if __name__ == "__main__":
    sys.exit(main())
