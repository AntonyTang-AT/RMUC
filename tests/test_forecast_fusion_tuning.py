import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.tune_forecast_fusion import _score
from runtime.temporal_forecast import TemporalForecastRuntime, _persistence_blend_from_checkpoint


def _tiny_data():
    return {
        "mode_y": np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]),
        "trend_y": np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]),
        "region_y": np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]),
        "base_mode": np.asarray([0, 0, 0]),
        "base_region": np.asarray([0, 0, 0]),
        "sides": np.asarray(["红", "蓝", "蓝"]),
    }


def _flat_probs(n=3):
    return {
        task: {h: np.tile(np.asarray([0.8, 0.2] + [0.0] * (n - 2)), (3, 1))
               for h in (10, 30, 60, 120)}
        for task, n in (("global_mode", 3), ("opponent_trend", 3), ("enemy_region", 12))
    }


def test_score_reports_pooled_and_each_side_metrics():
    score = _score(_tiny_data(), _flat_probs())
    bucket = score["global_mode@10s"]
    assert set(bucket["by_side"]) == {"红", "蓝"}
    assert bucket["by_side"]["红"]["n"] == 1
    assert bucket["by_side"]["蓝"]["n"] == 2
    assert bucket["accuracy"] == 1 / 3
    assert bucket["balanced_side_score"] == (
        bucket["by_side"]["红"]["score"] + bucket["by_side"]["蓝"]["score"]
    ) / 2


def test_tuned_checkpoint_is_separate_and_runtime_applies_embedded_alphas(tmp_path):
    from eval.tune_forecast_fusion import _write_deployment_checkpoint

    source = tmp_path / "trained.pt"
    candidate = tmp_path / "deployment.pt"
    original = {
        "model": {"weight": torch.tensor([3.0])},
        "model_type": "toy",
        "custom_metadata": {"epoch": 7},
        "persistence_fusion": {"model_weight": 0.70},
    }
    torch.save(original, source)
    alphas = {f"{task}@{h}s": 1.0
              for task in ("global_mode", "opponent_trend", "enemy_region")
              for h in (10, 30, 60, 120)}
    alphas["global_mode@10s"] = 0.25

    _write_deployment_checkpoint(source, candidate, alphas)

    assert source.exists() and torch.load(source, weights_only=False)["model_type"] == "toy"
    deployed = torch.load(candidate, weights_only=False)
    assert deployed["model_type"] == "toy"
    assert deployed["custom_metadata"] == original["custom_metadata"]
    assert torch.equal(deployed["model"]["weight"], original["model"]["weight"])
    assert _persistence_blend_from_checkpoint(deployed)["alphas"] == alphas

    labels = ["进攻", "均衡", "防守"]
    probs = TemporalForecastRuntime._classification(
        {10: torch.log(torch.tensor([[0.2, 0.7, 0.1]]))}, labels,
        lambda probabilities: {10: probabilities[10][0]},
        baseline_labels={"global_mode": "进攻"}, task="global_mode",
        blend_config=_persistence_blend_from_checkpoint(deployed),
    )
    assert probs["10"]["probabilities"]["进攻"] == 0.785
    assert probs["10"]["probabilities"]["均衡"] == 0.1825


def test_legacy_scalar_fusion_metadata_is_not_treated_as_tuned_alpha():
    legacy = {
        "persistence_fusion": {"enabled": True, "model_weight": 0.2},
    }
    assert _persistence_blend_from_checkpoint(legacy) is None
    probs = TemporalForecastRuntime._classification(
        {10: torch.log(torch.tensor([0.2, 0.7, 0.1]))}, ["进攻", "均衡", "防守"],
        lambda probabilities: probabilities,
        baseline_labels={"global_mode": "进攻"}, task="global_mode",
        blend_config=_persistence_blend_from_checkpoint(legacy),
    )
    assert probs["10"]["probabilities"]["进攻"] == 0.2
    assert probs["10"]["probabilities"]["均衡"] == 0.7
