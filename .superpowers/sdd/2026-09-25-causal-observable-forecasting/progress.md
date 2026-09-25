# SDD ledger — plan: docs/superpowers/plans/2026-09-25-causal-observable-forecasting.md

BASE: 43076d891163cf4cad0b911f5a834970a0ef18b9

## Preflight rulings

Ruling: Work in the current `rmuc-llm-staged` checkout — the project has extensive uncommitted and untracked baseline files, including `eval/benchmark.py`, that later plan tasks consume; a clean worktree would omit that baseline and copying/stashing it risks losing the user's working state. Cost if wrong: unrelated local edits remain shared with this implementation until final review, so file-scoped edits and review packages must exclude them.

Ruling: Parallel implementers will not run `git add` or commit while sharing this index — independent file changes can proceed in parallel, then task-scoped review and commits will be serialized by the controller. Cost if wrong: commits occur later than the plan's per-task step, while no code is lost and reviews remain scoped.

Ruling: Correct Task 2's blue-perspective test to preserve source team and derive self/opponent role — with `self_team='蓝'`, the blue `enemy-1` sample is self, while red `own-1` is opponent. Cost if wrong: a reversed side mapping would invert side-conditioned inputs and targets across the evaluation.

Ruling: Apply freshness per observation source before selecting an entity snapshot, then preserve a newer stale status transition when it conflicts with older live/absent tracks — source lifetimes differ, and status transitions must not revive old coordinates. Cost if wrong: cross-sensor age differences can hide valid measurements or reverse confirmed presence changes.

Ruling: Require explicit match end time or duration for chronological splits instead of assuming start-only rows have zero duration — the database provides `时长秒`, and boundary overlap must account for full play windows. Cost if wrong: refusing incomplete metadata prevents a split when a consumer only has start times, while accepting it could leak match outcomes across train/validation/test.

Ruling: Drop all measured fields from non-current observable entities when constructing side examples — stale or unknown snapshots must not expose retained coordinates even if a caller passes a malformed raw mapping. Cost if wrong: useful stale-track history is omitted from this target-builder output until a separate belief-state feature contract is added.

Ruling: Determine forecast censoring per exact target key, not by requiring red and blue to share `entity_or_region` — the two perspectives naturally predict different robot IDs and one valid side remains scoreable if the opposite target is censored. Cost if wrong: strict paired censoring loses valid observations; per-key censoring means red/blue completeness must be reported separately from individual score availability.

Ruling: Keep the legacy decision JSONL contract explicit as `game_id, side, index, decision` and keep probabilistic forecasts on the immutable forecast ledger interface. Cost if wrong: consumers sending only `global_mode` must adapt to the full structured decision schema; a single unambiguous interface avoids silently mixing categorical decisions with distribution forecasts.

Ruling: Treat the current global replay database as `oracle_replay` and block deployment readiness until a named, validated sensor/radar visibility provider exists. Cost if wrong: this can delay a deployability claim, while allowing global truth into features would invalidate the real-time evaluation.

Correction after user pointed to existing project data: the official 2026 communication protocol allows radar to receive opponent robot XY, HP, projectile allowance, macro status and buffs; the replay contains only a subset. `data/radar_visibility_replay.py` now exposes the replay's rule-allowed opponent type/XY/HP/team-coins/highlight fields, treats `(0,0)` as unsent, strips unsupported fine details, and retains complete self-side state. This supersedes the blanket “no visibility provider” conclusion; raw packets, latency/loss, opponent remaining ammo and full buffs remain absent.

## Preflight scan

| Pair / Task | Shared file or interface | Finding | Ruling |
|---|---|---|---|
| Task 1 / Task 3 | `data/features.py` causal timestamps consumed by target builder | Task 3 must consume only emitted timestamps and preserve `observed_at <= issued_at`; compatible | Run Task 1 contract in parallel; integrate Task 3 after it |
| Task 2 / Task 3 | `build_observable_snapshot` consumed by `build_side_examples` | Task 3 must consume observable snapshots; never read global replay truth into features | Task 2 contract first, Task 3 integration follows |
| Task 3 / Task 5 | targets and chronological split consumed by evaluator | Same side/horizon keys must be used by scores | Task 3 precedes benchmark integration |
| Task 4 / Task 5 | `validate_forecast`, ledger records consumed by benchmark | Benchmark must preserve invalid and immature records with counts | Task 4 first, Task 5 integration follows |
| Task 5 / Task 6 | benchmark report and aggregates consumed by readiness report | Both use side macro, pooled micro, and game-group IDs | Task 5 precedes Task 6 |
| Task 1 | grid anchor vs no-prior-row behavior | Plan tests require first-observation grid and no future fallback; consistent | Keep spec's causal cutoff binding |
| Task 2 | stale, unknown, absent, destroyed entity distinctions | Plan only specifies stale explicitly; implementation must not collapse absent/unknown/destroyed | Preserve these as separate enum values |
| Task 3 | exact target time and game-tail censoring | Test uses exact timestamp and `None` when absent; interface says no nearest future row | Use exact target lookup and explicit censor reason |
| Task 4 | probability labels and sum validation | Forecast fixture includes labels and probability map; consistent | Reject rather than repair invalid distributions |
| Task 5 | side macro vs pooled micro | Hand calculation expects equal side weighting and sample-weighted micro; consistent | Keep red/blue outputs plus both aggregates |
| Task 6 | deployment readiness and match identity | Missing visibility provider blocks readiness; match grouping may be synthesized from region/scheduled match | Report provider missing as not deployable |

## Progress

Task 1: complete (commit 43076d8..da73038, tests: `python -m pytest tests/test_features_causal.py tests/test_features.py tests/test_worldstate.py -q` → 11 passed; independent review approved).
Task 2: complete (commit da73038..8fdfd58, tests: `python -m pytest tests/test_observable.py -q` → 14 passed; independent review approved).
Task 4: complete (commit 8fdfd58..9a36b0a, tests: `python -m pytest tests/test_forecast_schema.py tests/test_prediction_ledger.py -q` → 41 passed; independent review approved).
Task 3: complete (commit 9a36b0a..d95ba1f, tests: `python -m pytest tests/test_forecast_targets.py tests/test_forecast_splits.py -q` → 16 passed; independent review approved).
Task 5: complete (commit a33064e; probability metrics and two-sided benchmark integration; focused tests 25 passed at independent re-review; review APPROVED; final full suite 344 passed).
Task 6: complete (commit 96bdc21 plus radar/protocol-field follow-up; manifest/readiness gates, full extraction and handoff updates; final full suite 353 passed; generated 171,292 dual-view examples; deployment report identifies the one-second protocol-field replay and missing raw packets/latency/ammo/buff data).

