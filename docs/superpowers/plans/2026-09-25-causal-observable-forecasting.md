# Causal Observable Multi-Horizon Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe, sensor-observable, probabilistic multi-horizon evaluation pipeline that runs every game from both red and blue perspectives and reports side-balanced performance.

**Architecture:** Keep global replay truth separate from deployment-observable inputs. Build causal timestamped snapshots first, create future targets at fixed horizons, then evaluate archived probability forecasts only when their target timestamps mature. Each game is evaluated from red and blue self/opponent views; report each side, their macro average, and pooled micro metrics, with uncertainty resampled by whole match/game groups.

**Tech Stack:** Python 3, SQLite, NumPy, pytest, existing `data.features`, `worldstate`, `schema.layered`, and `eval` modules. The Jev adapter consumes typed probability JSON; no Jev network call or credentials are required for these tasks.

**Spec:** `docs/superpowers/specs/2026-09-25-causal-observable-forecasting-design.md`

## Global Constraints

- Deployment features must come only from vehicle sensors/radar and have `observed_at <= issued_at`.
- Global replay truth may be used for future labels and scoring, never as a deployment feature.
- Unknown, stale, absent, and confirmed-destroyed entities remain distinct states.
- Fixed first-release horizons are 10, 30, 60, and 120 seconds; incomplete end-of-game targets are censored.
- Every game must run in red and blue self/opponent views; report both views, their side-level macro average, and pooled micro metrics.
- Red/blue views from one game are one statistical cluster. Bootstrap by complete match/game group.
- Never claim deployment-valid results from the existing omniscient replay table until a sensor log, visibility mask, or validated visibility replay provider supplies observable inputs.
- Never treat the existing lookahead heuristic as a human gold label or causal action-value label.

## Review Focus

- A robot has no observation before its first timestamp: must be unknown/absent, never filled from its future first row. Test the no-prior-row boundary in Task 1.
- A sensor report arrives late or expires: must be excluded after its usable time or staleness limit. Test out-of-order and stale observations in Task 2.
- The game ends before a requested horizon: target is censored and excluded only from that horizon's score. Test each boundary in Task 3.
- A probability omits a class, is negative, or sums incorrectly: reject it and count the prediction invalid; never silently substitute a neutral action. Test malformed distributions in Task 4.
- One side has more valid targets than the other: side macro average gives each side equal weight while preserving pooled micro; bootstrap resamples whole game/match clusters. Test in Task 5.

---

### Task 1: Make replay state alignment strictly causal

**Files:**
- Modify: `data/features.py`
- Test: `tests/test_features_causal.py`

**Interfaces:**
- `build_decision_points(rows, game_id, step=DF_STEP, trend=TREND, start_time=None) -> list[dict]` anchors the default grid at the earliest observed game timestamp, unless `start_time` is explicitly given.
- `_closest(series, t, forward=False) -> row | None` returns only a row with `row['时刻秒'] <= t`; it returns `None` if no such row exists.
- Each emitted robot state carries `observed_at` and `age_seconds`; missing prior state is omitted from `robots` and must not use future values.

- [ ] **Step 1: Write the failing tests**

```python
def row(game_id, robot_id, t, x):
    return {
        'game_id': game_id, 'robot_id': robot_id, '时刻秒': float(t),
        '机器人类型': '英雄', '阵营': '红', '学校名': 'A',
        '当前血量': 1000.0, '最大血量': 1000.0, 'x': float(x), 'y': 0.0, 'z': 0.0,
        '枪口朝向': 0.0, '底盘功率': 0.0, '小热量': 0.0, '大热量': 0.0,
        '累计17mm发弹': 0.0, '累计42mm发弹': 0.0, '是否易伤': 0,
        '队伍总金币': 0.0, '队伍剩余金币': 0.0,
    }


def test_closest_returns_none_before_first_observation():
    rows = [{'时刻秒': 1.0, 'x': 10}, {'时刻秒': 2.0, 'x': 20}]
    assert features._closest(rows, 0.0) is None


def test_decision_grid_starts_at_first_observation_and_never_reads_future():
    rows = [row(game_id=7, robot_id=1, t=1, x=10),
            row(game_id=7, robot_id=1, t=2, x=20),
            row(game_id=7, robot_id=1, t=4, x=40)]
    dps = features.build_decision_points(rows, 7, step=3, trend=2)
    assert [dp['t'] for dp in dps] == [1.0, 4.0]
    assert dps[0]['robots'][1]['x'] == 10
    assert dps[0]['robots'][1]['observed_at'] == 1.0
    assert dps[1]['robots'][1]['x'] == 40


def test_no_future_state_is_inserted_for_late_entity():
    rows = [row(game_id=7, robot_id=1, t=1, x=10),
            row(game_id=7, robot_id=2, t=4, x=40)]
    dps = features.build_decision_points(rows, 7, step=3, trend=2)
    assert 2 not in dps[0]['robots']
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_features_causal.py -q`
Expected: FAIL because `_closest` currently falls back to the first future row and the grid starts at zero.

- [ ] **Step 3: Implement causal as-of alignment**

Remove the future fallback from `_closest`. Anchor the grid to the earliest observed timestamp, and attach observation time and age to each selected state. If no current or trend-base row exists at or before a requested time, retain the entity as unknown/omit the trend rather than reading a later row.

- [ ] **Step 4: Run targeted and related tests**

Run: `python -m pytest tests/test_features_causal.py tests/test_worldstate.py -q`
Expected: PASS; existing state construction remains valid and no emitted state has `observed_at > dp['t']`.

- [ ] **Step 5: Commit**

```bash
git add data/features.py tests/test_features_causal.py
git commit -m "fix: prevent future state in replay alignment"
```

### Task 2: Define deployment-observable input and side mapping

**Files:**
- Create: `data/observable.py`
- Create: `tests/test_observable.py`
- Modify: `data/__init__.py` only if exports are needed

**Interfaces:**
- `build_observable_snapshot(observations, issued_at, self_team, max_age_by_source) -> dict` accepts sensor/radar observations with `entity_id`, `team`, `entity_type`, `observed_at`, `received_at`, `source`, `confidence`, and measured fields.
- It returns `{issued_at, self_team, opponent_team, entities, visibility_summary}` with per-entity `presence_state`, `observability_state`, `observed_at`, and `age_seconds`.
- `observations` whose `received_at > issued_at` or `observed_at > issued_at` are rejected. Stale observations are marked `stale` and excluded from current coordinates. Enemy positions appear only if a sensor/radar observation supplies them.
- No adapter may silently convert global replay truth into observable input. If a caller requests deployment mode without a visibility provider, it raises `MissingVisibilityProvider`.

- [ ] **Step 1: Write tests for side swap, late arrival, and staleness**

```python
def observation(entity_id='enemy-1', team='蓝', observed_at=10, received_at=10,
                source='radar', confidence=.9, position=(2.0, 3.0)):
    return {'entity_id': entity_id, 'team': team, 'entity_type': '英雄',
            'observed_at': observed_at, 'received_at': received_at, 'source': source,
            'confidence': confidence, 'position': position}


obs_pair = [observation(entity_id='own-1', team='红'), observation()]


def test_snapshot_maps_self_opponent_without_changing_global_team():
    snap = build_observable_snapshot(obs_pair, 12, self_team='蓝', max_age_by_source={'radar': 2})
    assert snap['self_team'] == '蓝'
    assert snap['opponent_team'] == '红'
    assert snap['entities']['enemy-1']['team'] == '蓝'
    assert snap['entities']['enemy-1']['role'] == 'self'
    assert snap['entities']['own-1']['team'] == '红'
    assert snap['entities']['own-1']['role'] == 'opponent'


def test_future_received_or_observed_measurement_is_rejected():
    with pytest.raises(ValueError):
        build_observable_snapshot([observation(observed_at=13, received_at=13)], 12,
                                  self_team='红', max_age_by_source={'radar': 2})


def test_stale_enemy_track_has_no_current_position():
    snap = build_observable_snapshot([observation(team='蓝', observed_at=2, received_at=2)], 12,
                                     self_team='红', max_age_by_source={'radar': 2})
    enemy = snap['entities']['enemy-1']
    assert enemy['observability_state'] == 'stale'
    assert 'position' not in enemy
```

- [ ] **Step 2: Run tests to verify the missing interface fails**

Run: `python -m pytest tests/test_observable.py -q`
Expected: FAIL because `data.observable` and its typed visibility errors do not exist.

- [ ] **Step 3: Implement the explicit observable snapshot contract**

Create the module and exception, validate usable times/confidence/source, map `self/opponent`, and preserve unknown/stale distinctions. Require the caller to supply observations; do not add a provider that reads global replay tables.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_observable.py -q`
Expected: PASS, including no enemy coordinates in absent/stale tracks.

- [ ] **Step 5: Commit**

```bash
git add data/observable.py tests/test_observable.py
git commit -m "feat: define sensor-observable state snapshots"
```

### Task 3: Build exact-horizon targets and chronological match splits

**Files:**
- Create: `data/forecast_targets.py`
- Create: `data/forecast_splits.py`
- Create: `tests/test_forecast_targets.py`
- Create: `tests/test_forecast_splits.py`

**Interfaces:**
- `build_targets(global_truth_by_time, issued_at, horizons=(10, 30, 60, 120), side) -> dict[int, dict | None]` returns `{horizon: {'target_at': int, 'truth': dict, 'censored': False}}`; `None` means censored or unavailable, never nearest-future substitution.
- `build_side_examples(game_id, observable_history, global_truth, side, horizons=(10, 30, 60, 120)) -> list[dict]` generates a separate row for `side='红'` and `side='蓝'`, with self/opponent feature mapping and global-truth targets.
- `chronological_group_split(match_rows, train_fraction, validation_fraction) -> dict` sorts by competition order and assigns each complete `match_id` exactly once to train, validation, or test.
- Split rows must provide an explicit `end_time` or game duration (`时长秒`/`duration_seconds`) so match-window overlaps can be checked; missing duration fails closed.
- The manifest records all match_id/game_id, time ranges, horizon definitions, feature cutoff, target cutoff, and sensor-provider version. A missing observable history raises a visibility-provider error instead of falling back to global truth.

- [ ] **Step 1: Write exact-horizon, censoring, and two-side tests**

```python
def truth(t, red_x, blue_x):
    return {'target_at': t, 'entities': {'红-hero': {'x': red_x}, '蓝-hero': {'x': blue_x}}}


observations = [{'issued_at': 10, 'entities': {
    '红-hero': {'position': [1, 0]}, '蓝-hero': {'position': [9, 0]}}}]
truths = {10: truth(10, 1, 9), 20: truth(20, 2, 8)}


def test_target_uses_exact_requested_future_time():
    result = build_targets({10: truth(10, 0, 0), 19: truth(19, 1, 1),
                            20: truth(20, 2, 2)},
                           issued_at=10, horizons=(10,), side='红')
    assert result[10]['target_at'] == 20
    assert result[10]['truth'] == truth(20, 2, 2)


def test_target_is_censored_when_game_ends_before_horizon():
    result = build_targets({10: truth(10, 0, 0), 19: truth(19, 1, 1)},
                           issued_at=10, horizons=(10,), side='红')
    assert result[10] is None


def test_both_side_examples_swap_self_opponent_and_keep_same_future_time():
    red = build_side_examples(1, observations, truths, side='红', horizons=(10,))
    blue = build_side_examples(1, observations, truths, side='蓝', horizons=(10,))
    assert red[0]['side'] == '红' and blue[0]['side'] == '蓝'
    assert red[0]['observable']['self_entities'] == blue[0]['observable']['opponent_entities']
    assert red[0]['observable']['opponent_entities'] == blue[0]['observable']['self_entities']
    assert red[0]['issued_at'] == blue[0]['issued_at']
    assert red[0]['targets'][10]['target_at'] == blue[0]['targets'][10]['target_at']
```

- [ ] **Step 2: Write match-order isolation tests**

```python
def test_chronological_split_keeps_each_match_whole_and_ordered():
    split = chronological_group_split(rows, train_fraction=.6, validation_fraction=.2)
    assert not (set(split['train']) & set(split['validation']))
    assert not (set(split['train']) & set(split['test']))
    assert max(split['train_order']) < min(split['validation_order'])
    assert max(split['validation_order']) < min(split['test_order'])
```

- [ ] **Step 3: Run target and split tests to verify they fail**

Run: `python -m pytest tests/test_forecast_targets.py tests/test_forecast_splits.py -q`
Expected: FAIL because the exact-horizon builder and match-group splitter do not exist.

- [ ] **Step 4: Implement target creation and grouped rolling-origin split**

Use exact target timestamps only. Resolve match identity from the existing match metadata columns; if a stable match_id is absent, construct the group from region plus scheduled match number and validate it is unique. Preserve all rows from one group in one split.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_forecast_targets.py tests/test_forecast_splits.py -q`
Expected: PASS, including censored targets and no match overlap.

```bash
git add data/forecast_targets.py data/forecast_splits.py tests/test_forecast_targets.py tests/test_forecast_splits.py
git commit -m "feat: build causal forecast targets and splits"
```

### Task 4: Validate probability forecasts and archive them

**Files:**
- Create: `eval/forecast_schema.py`
- Create: `eval/prediction_ledger.py`
- Create: `tests/test_forecast_schema.py`
- Create: `tests/test_prediction_ledger.py`

**Interfaces:**
- `validate_forecast(row) -> tuple[bool, list[str]]` requires `prediction_id`, `game_id`, `side`, `issued_at`, `valid_at`, `horizon`, `task`, `target_schema_version`, `input_snapshot_hash`, `visibility_summary`, `model_version`, and a finite probability distribution or quantiles.
- For each categorical forecast, probabilities are finite, lie in `[0,1]`, include every declared label, and sum to 1 within `1e-6`. For continuous forecasts, quantiles are ordered and bounds finite.
- `append_prediction(path, row) -> None` writes one immutable JSONL record after validation. `load_predictions(path) -> list[dict]` reads archived rows. `mature_predictions(rows, now) -> list` returns only rows with `valid_at <= now`; it never edits their original probabilities or calibration version.
- Missing/malformed predictions remain explicit invalid records in benchmark summaries.

- [ ] **Step 1: Write tests for valid, malformed, immutable, and immature forecasts**

```python
def forecast(issued_at=10, valid_at=20, probabilities=None):
    return {
        'prediction_id': 'p1', 'game_id': 1, 'side': '红',
        'issued_at': issued_at, 'valid_at': valid_at, 'horizon': valid_at - issued_at,
        'task': 'mode', 'target_schema_version': 'v1',
        'input_snapshot_hash': 'abc123', 'visibility_summary': {'radar_tracks': 1},
        'model_version': 'fixture', 'calibration_version': 'none',
        'labels': ['攻', '守'], 'probabilities': probabilities or {'攻': .7, '守': .3},
    }


def test_rejects_probability_vector_with_wrong_sum():
    ok, errors = validate_forecast(forecast(probabilities={'攻': .6, '守': .6}))
    assert not ok and any('sum' in e for e in errors)


def test_future_prediction_is_not_mature():
    row = forecast(issued_at=10, valid_at=20)
    assert mature_predictions([row], now=19) == []
    assert mature_predictions([row], now=20) == [row]


def test_append_prediction_preserves_archived_distribution(tmp_path):
    row = forecast(probabilities={'攻': .7, '守': .3})
    append_prediction(tmp_path / 'predictions.jsonl', row)
    assert load_predictions(tmp_path / 'predictions.jsonl')[0]['probabilities'] == {'攻': .7, '守': .3}
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_forecast_schema.py tests/test_prediction_ledger.py -q`
Expected: FAIL because the schema and ledger modules do not exist.

- [ ] **Step 3: Implement strict schema validation and append-only ledger**

Use JSONL and existing standard-library JSON tools; do not add a database dependency. Preserve invalid prediction diagnostics rather than substituting a neutral state.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_forecast_schema.py tests/test_prediction_ledger.py -q`
Expected: PASS.

```bash
git add eval/forecast_schema.py eval/prediction_ledger.py tests/test_forecast_schema.py tests/test_prediction_ledger.py
git commit -m "feat: add immutable probabilistic forecast ledger"
```

### Task 5: Add proper scores and red/blue balanced benchmark

**Files:**
- Create: `eval/probabilistic_metrics.py`
- Modify: `eval/benchmark.py`
- Create: `tests/test_probabilistic_metrics.py`
- Create: `tests/test_benchmark_sides.py`

**Interfaces:**
- `multiclass_brier(y_true, probabilities, labels) -> float` uses the sum of squared class-probability errors per example and averages over examples.
- `multiclass_log_loss(y_true, probabilities, labels, epsilon=1e-15) -> float` clips only for logarithm stability and rejects unknown labels.
- `crps_ensemble(observation, samples) -> float` computes the empirical ensemble CRPS; `energy_score(observation_vector, sample_vectors) -> float` computes the multivariate ensemble score.
- `aggregate_sides(per_side) -> dict` returns `red_view`, `blue_view`, `side_macro_average`, and `pooled_micro`; side macro average gives red and blue equal weight.
- `bootstrap_game_clusters(rows_by_game, n_bootstrap=2000, seed=0) -> list[list[list[dict]]]` returns bootstrap draws, each containing sampled whole-game row groups with both side views intact.
- `cluster_bootstrap_side_delta(rows_by_game, metric, n_bootstrap=2000, seed=0) -> dict` samples game/match groups and carries both side views together on every draw.
- `aggregate_game_scores(per_game_side) -> dict` aggregates rows shaped as `{game_id: {'红': {'score': float, 'n': int}, '蓝': {'score': float, 'n': int}}}`.
- `eval.benchmark` runs each selected model on both sides for every game. Predictions must be keyed by `(game_id, side, index)`; missing predictions count invalid and are excluded only from conditional score denominators with missing rate separately reported.

- [ ] **Step 1: Write hand-computable scoring tests**

```python
def test_brier_score_for_two_class_distribution():
    assert multiclass_brier('攻', {'攻': .8, '守': .2}, ['攻', '守']) == pytest.approx(.08)


def test_log_loss_penalizes_low_probability_on_observed_class():
    good = multiclass_log_loss('攻', {'攻': .9, '守': .1}, ['攻', '守'])
    bad = multiclass_log_loss('攻', {'攻': .1, '守': .9}, ['攻', '守'])
    assert good < bad


def test_side_macro_is_equal_weight_even_when_sample_counts_differ():
    scores = aggregate_sides({'红': {'score': .2, 'n': 100}, '蓝': {'score': .8, 'n': 10}})
    assert scores['side_macro_average']['score'] == pytest.approx(.5)
    assert scores['pooled_micro']['score'] == pytest.approx(28.0 / 110.0)
```

- [ ] **Step 2: Write benchmark symmetry and clustered-bootstrap tests**

```python
def test_aggregate_keeps_both_views_and_reports_macro_and_micro():
    report = aggregate_game_scores({
        1: {'红': {'score': .2, 'n': 100}, '蓝': {'score': .8, 'n': 10}},
        2: {'红': {'score': .4, 'n': 20}, '蓝': {'score': .6, 'n': 20}},
    })
    assert set(report['sides']) == {'红', '蓝'}
    assert report['side_macro_average']['n_sides'] == 2
    assert report['pooled_micro']['n_points'] == sum(report['sides'][s]['n'] for s in ('红', '蓝'))


def test_bootstrap_resamples_game_clusters_with_both_sides_intact():
    rows_by_game = {
        'g1': [{'side': '红', 'score': .1}, {'side': '蓝', 'score': .2}],
        'g2': [{'side': '红', 'score': .3}, {'side': '蓝', 'score': .4}],
    }
    draws = bootstrap_game_clusters(rows_by_game, n_bootstrap=20, seed=0)
    assert all(all({row['side'] for row in cluster} == {'红', '蓝'}
                   for cluster in draw) for draw in draws)
```

- [ ] **Step 3: Run tests and verify they fail**

Run: `python -m pytest tests/test_probabilistic_metrics.py tests/test_benchmark_sides.py -q`
Expected: FAIL because proper-score functions and side aggregation do not exist.

- [ ] **Step 4: Implement metrics and two-sided reporting**

Add Brier, log loss, empirical CRPS, energy score, calibration bins with counts, side macro/micro aggregation, and game-cluster bootstrap. Keep accuracy and macro-F1 as secondary metrics. Remove the current behavior that fills missing predictions with a neutral decision.

- [ ] **Step 5: Verify on synthetic fixtures**

Run: `python -m pytest tests/test_probabilistic_metrics.py tests/test_benchmark_sides.py -q`
Expected: PASS; report contains separate red/blue scores, an equal-side macro average, pooled micro scores, and game-cluster confidence intervals.

- [ ] **Step 6: Commit**

```bash
git add eval/probabilistic_metrics.py eval/benchmark.py tests/test_probabilistic_metrics.py tests/test_benchmark_sides.py
git commit -m "feat: score forecasts with red blue balanced metrics"
```

### Task 6: Produce manifests and deployment-readiness report

**Files:**
- Modify: `eval/benchmark.py`
- Create: `eval/forecast_report.py`
- Create: `tests/test_forecast_report.py`
- Modify: `AGENT_PROJECT_OVERVIEW.md` and `CODEX_HANDOFF.md` after measured results exist

**Interfaces:**
- `build_forecast_manifest(matches, split, horizons, visibility_provider, label_version) -> dict` stores every match/game ID, exact split, horizons, region-map version, visibility provider, target builder version, both sides, and cutoff policy.
- `summarize_readiness(manifest, feature_audit, prediction_audit, scores) -> dict` reports `causal_input_pass`, `observable_input_pass`, `split_pass`, `probability_schema_pass`, `red_blue_complete`, `side_macro_scores`, `pooled_micro_scores`, and `deployment_ready`.
- `deployment_ready` is false when visibility source is missing, any future feature is detected, any group overlaps splits, either side is absent, or the probability output contract fails.

- [ ] **Step 1: Write tests for complete manifests and fail-closed readiness**

```python
matches = [
    {'game_id': 1, 'match_id': 'm1', 'competition_order': 1},
    {'game_id': 2, 'match_id': 'm2', 'competition_order': 2},
]
split = {'train': ['m1'], 'validation': [], 'test': ['m2']}


def test_manifest_contains_every_game_and_both_sides():
    manifest = build_forecast_manifest(matches, split, (10, 30, 60, 120), 'fixture_sensor', 'v1')
    assert set(manifest['sides']) == {'红', '蓝'}
    assert sorted(manifest['all_game_ids']) == sorted(g['game_id'] for g in matches)


def test_missing_visibility_provider_blocks_deployment_readiness():
    empty_manifest = build_forecast_manifest(matches, split, (10, 30, 60, 120), None, 'v1')
    clean_audit = {'future_feature_count': 0, 'split_overlap_count': 0}
    clean_predictions = {'invalid_count': 0, 'red_blue_complete': True}
    scores = {'side_macro_average': {}, 'pooled_micro': {}}
    report = summarize_readiness(empty_manifest, clean_audit, clean_predictions, scores)
    assert report['observable_input_pass'] is False
    assert report['deployment_ready'] is False
```

- [ ] **Step 2: Implement manifest and readiness checks**

Reuse the match-group split and prediction validators from earlier tasks. Include exact row counts and hash of ordered IDs. Keep historical global-truth-only reports explicitly labeled `oracle_replay`.

- [ ] **Step 3: Run the focused suite and full repository suite**

Run: `python -m pytest tests/test_forecast_report.py tests/test_forecast_splits.py tests/test_benchmark_sides.py -q`
Expected: PASS.

Run: `python -m pytest tests -q`
Expected: PASS; report any pre-existing unrelated failures separately.

- [ ] **Step 4: Update handoff docs with measured state**

Document current readiness gates and generated artifact paths. State that deployment validation is blocked until a real sensor log or validated visibility replay provider exists; do not report oracle replay scores as deployment performance.

- [ ] **Step 5: Commit**

```bash
git add eval/benchmark.py eval/forecast_report.py tests/test_forecast_report.py AGENT_PROJECT_OVERVIEW.md CODEX_HANDOFF.md
git commit -m "docs: report forecast evaluation readiness gates"
```

## Plan Caveats

- The current SQLite replay does not include a validated onboard visibility stream. Tasks 2 and 6 establish and enforce this boundary; deployable training and deployment claims remain blocked until such a provider is supplied.
- This plan creates a generic probabilistic adapter and ledger. A live Jev client is a separate follow-up once its SDK/API contract, credentials and runtime latency budget are available.
- Action recommendation quality remains separate from forecast score until historical action records or a credible forward simulator supports counterfactual evaluation.
