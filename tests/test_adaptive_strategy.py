from runtime.adaptive_strategy import AdaptiveStrategyRouter


def _prediction():
    return {
        "issued_at": 10.0,
        "input_snapshot_hash": "fixture",
        "horizons": {
            "10": {
                "global_mode": {
                    "top": "进攻", "confidence": 0.9,
                    "probabilities": {"进攻": 0.9, "均衡": 0.1},
                }
            }
        },
    }


def _baseline(label="防守"):
    return {"global_mode": {"10": {label: 1.0, "进攻": 0.0, "均衡": 0.0}}}


def test_router_keeps_model_during_warmup_and_registers_feedback_id():
    router = AdaptiveStrategyRouter(min_samples=3, patience=2)
    result = router.apply(_prediction(), baselines=_baseline())
    assert result["horizons"]["10"]["global_mode"]["adaptive_strategy"] == "model"
    assert result["adaptive_strategy"]["feedback_required"] is True
    forecast_id = result["adaptive_strategy"]["forecast_id"]
    for _ in range(2):
        router.observe_outcome(forecast_id, {"global_mode": {"10": "防守"}})
        result = router.apply(_prediction(), baselines=_baseline())
        forecast_id = result["adaptive_strategy"]["forecast_id"]
    assert router.status()["global_mode@10s"]["strategy"] == "model"


def test_router_switches_to_persistence_after_sustained_bad_model():
    router = AdaptiveStrategyRouter(min_samples=3, patience=2, margin=0.01)
    for _ in range(5):
        result = router.apply(_prediction(), baselines=_baseline())
        router.observe_outcome(result["adaptive_strategy"]["forecast_id"],
                               {"global_mode": {"10": "防守"}})
    assert router.status()["global_mode@10s"]["strategy"] == "persistence"
    result = router.apply(_prediction(), baselines=_baseline())
    head = result["horizons"]["10"]["global_mode"]
    assert head["adaptive_strategy"] == "persistence"
    assert head["top"] == "防守"


def test_router_recovers_only_after_sustained_model_advantage():
    router = AdaptiveStrategyRouter(window=4, min_samples=2, patience=1,
                                    recovery_patience=2, margin=0.01)
    for _ in range(3):
        result = router.apply(_prediction(), baselines=_baseline())
        router.observe_outcome(result["adaptive_strategy"]["forecast_id"],
                               {"global_mode": {"10": "防守"}})
    assert router.status()["global_mode@10s"]["strategy"] == "persistence"
    # The model is confidently correct while this supplied persistence
    # distribution is deliberately stale and diffuse.
    good_baseline = {"global_mode": {"10": {"进攻": 0.4, "防守": 0.3, "均衡": 0.3}}}
    for _ in range(6):
        result = router.apply(_prediction(), baselines=good_baseline)
        router.observe_outcome(result["adaptive_strategy"]["forecast_id"],
                               {"global_mode": {"10": "进攻"}})
    assert router.status()["global_mode@10s"]["strategy"] == "model"


def test_outcome_must_arrive_for_known_forecast_id():
    router = AdaptiveStrategyRouter()
    assert router.observe_outcome("missing", {"global_mode": {"10": "进攻"}})["updated"] == 0


def test_horizon_feedback_is_scored_incrementally_without_consuming_future_targets():
    router = AdaptiveStrategyRouter()
    prediction = _prediction()
    prediction["horizons"]["30"] = {
        "global_mode": {"top": "进攻", "confidence": 0.9,
                        "probabilities": {"进攻": 0.9, "均衡": 0.1}}
    }
    baseline = _baseline()
    baseline["global_mode"]["30"] = dict(baseline["global_mode"]["10"])
    result = router.apply(prediction, baselines=baseline)
    forecast_id = result["adaptive_strategy"]["forecast_id"]

    early = router.observe_outcome(
        forecast_id, {"global_mode": {"10": "防守"}}, matured_horizons=[10])
    assert early["updated"] == 1
    assert early["complete"] is False
    assert early["pending_horizons"] == ["30"]
    assert router.status()["global_mode@10s"]["samples"] == 1
    assert router.status()["global_mode@30s"]["samples"] == 0

    late = router.observe_outcome(
        forecast_id, {"global_mode": {"30": "进攻"}}, matured_horizons=[30])
    assert late["updated"] == 1
    assert late["complete"] is True
    assert late["pending_horizons"] == []
    assert router.status()["global_mode@30s"]["samples"] == 1


def test_router_scores_raw_model_when_static_blend_is_present():
    prediction = _prediction()
    head = prediction["horizons"]["10"]["global_mode"]
    # The emitted probability is a static causal blend leaning toward the
    # current state, while model_probabilities preserve the neural expert.
    head["probabilities"] = {"进攻": 0.55, "均衡": 0.45}
    head["model_probabilities"] = {"进攻": 0.9, "均衡": 0.1}
    router = AdaptiveStrategyRouter(min_samples=1, patience=1, margin=0.01)
    result = router.apply(prediction, baselines=_baseline())
    forecast_id = result["adaptive_strategy"]["forecast_id"]
    router.observe_outcome(forecast_id, {"global_mode": {"10": "进攻"}})
    state = router.status()["global_mode@10s"]
    # The model wins against the persistence prior; this would be reversed if
    # the already-blended 0.55 distribution were scored as the model expert.
    assert state["rolling_gap"] < 0
