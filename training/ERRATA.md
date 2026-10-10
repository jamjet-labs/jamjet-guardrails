# Errata for the ship bar and its verdict

`training/ship_bar.json` and `training/artifacts/ship_check.json` are not edited
to correct them. The bar is a pre-registration and the verdict is a record of a
measurement, so a correction goes here and both files stay as they are.

## 1. The verdict was measured three times, not once, and the bar says otherwise

`ship_bar.json`, `structural_floor_rederived.verdict_unaffected`, says the
recorded verdict "was reached against the EARLIER floor, 0.885 over the
146-case corpus". That is false. `ship_check.json`, `measured_once`, says "This
is the first and only run of this measurement". That is false too.

What happened on 2026-09-02, read from `git log -- training/artifacts/ship_check.json`:

| Commit | What it did to `ship_check.json` | `measured_utc` | Structural corpus | Floor judged against |
|---|---|---|---|---|
| `7b9e332` | First run of `training.decide` | 10:46:25Z | 146 cases, 46 / 2 / 6 / 92 | 0.885 |
| `12dc493` | Recomputed the verdict block from the recorded measurements, not re-measured (its message says so) | 10:46:25Z | unchanged | 0.8846 (the unrounded recall) |
| `2c116c2` | Re-ran `training.decide` after the corpus grew for output-direction cases | 16:42:43Z | 154 cases, 49 / 2 / 6 / 97 | 0.891 |
| `21376d8` | Re-ran `training.decide` again, so the verdict's digest of the bar matched the bar after `structural_floor_rederived` was added | 17:00:16Z | 154 cases, 49 / 2 / 6 / 97 | 0.891, bar sha256 `74fb86...` |

`21376d8` is also the commit that wrote the `verdict_unaffected` sentence, and
its own message says it "re-runs training.decide". So the verdict that is on
record was reached against 0.891 over 154 cases, judged against the bar at sha256
`74fb86231e6ef98813d09433041d032989ce7ed1512fb6c03d00d7e672eb2ae0`, at 17:00:16Z.
`SHIP_BAR_SHA256_AT_VERDICT` in `tests/test_ship_bar.py` and
`STRUCTURAL_CORPUS_VERSION` in `training/ship_bar.py` (`b704703f431d`, 154
cases) already say that; the sentence in the bar does not.

What did not change across the three runs: the semantic side. The `semantic`
block is identical in all four versions of the file, F1 0.36722606120434353
against a required strictly-greater 0.8995918367346939, and
`ship_the_classifier` is false in every one. The outcome stands. What is wrong
is the claim that the measurement ran once, which is the claim the
pre-registration exists to make checkable.

## 2. `version_pin_added` describes a guard that no longer works that way

`ship_bar.json`, `structural_floor_rederived.version_pin_added`, says that from
2026-09-03 "a corpus that moves fails tests/test_ship_bar.py until the move is
disclosed in this key". From 2026-10-10 it does not. The bar is frozen at the
corpus the verdict was judged on, `b704703f431d`, and the test asserts the bar
names that corpus. A corpus that moves is gated by its own key in
`corpora/baselines.json`, which the same test checks exists for the shipped
corpus. Later decision-level measurements of the structural layer are in
`docs/measurements/`, not in the bar.
