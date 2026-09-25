# Task 6 manifest and readiness report

Implemented `eval/forecast_report.py` and `tests/test_forecast_report.py`.

- Deterministic manifest records all game/match IDs, exact match-group split assignment, competition order, fixed horizons, both red/blue views, provider and target/label versions, causal/target cutoff policies, oracle-vs-sensor source mode, and SHA-256 identifiers.
- Manifest construction rejects duplicate games, duplicated or missing match groups, invalid horizons, and absent match identity.
- Readiness reports causal feature, observable provider, split integrity, probability schema, and red/blue coverage gates; it includes side macro and pooled micro score payloads and explicit blocking reasons.
- Deployment readiness fails closed when visibility provider is missing, source mode is oracle replay, future features or split overlaps exist, forecast schema fails, or either side lacks complete scored coverage.
- Focused suite: `python -m pytest tests/test_forecast_report.py tests/test_forecast_splits.py tests/test_benchmark_sides.py tests/test_probabilistic_metrics.py -q` -> 37 passed.
- Full suite: `python -m pytest tests -q` -> 344 passed.

Current data limitation: SQLite `timeseries` contains global replay information, not a validated vehicle sensor/radar visibility stream. Thus deployment readiness remains false and any replay-only model score must be labeled `oracle_replay`.
