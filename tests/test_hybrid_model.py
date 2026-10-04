import torch

from forecast.entity import ENTITY_SLOTS, ENTITY_TARGET_DIM, ENTITY_ZONE_DIM, ENTITY_CANONICAL_ZONE_DIM
from forecast.hybrid_model import HybridTemporalForecastNet
from forecast.temporal import FEATURE_DIM, HORIZONS


def test_hybrid_model_emits_all_heads_and_uses_causal_mask():
    model = HybridTemporalForecastNet(
        input_dim=FEATURE_DIM,
        hidden=64,
        layers=2,
        entity_slots=len(ENTITY_SLOTS),
        zone_classes=ENTITY_ZONE_DIM,
        canonical_zone_classes=ENTITY_CANONICAL_ZONE_DIM,
        nhead=4,
        max_seq_len=8,
    )
    outputs = model(torch.zeros(2, 12, FEATURE_DIM))
    assert set(outputs) == set(HORIZONS)
    assert outputs[10]["global_mode"].shape == (2, 3)
    assert outputs[120]["entity_state"].shape == (2, len(ENTITY_SLOTS), ENTITY_TARGET_DIM)
    assert outputs[60]["entity_zone_state"].shape[-1] == ENTITY_ZONE_DIM
    assert outputs[30]["entity_canonical_zone_state"].shape[-1] == ENTITY_CANONICAL_ZONE_DIM
    mask = model.causal_mask(5, torch.device("cpu"))
    assert mask.shape == (5, 5)
    assert not mask[0, 0] and mask[0, 1] and not mask[4, 0]


def test_hybrid_model_serializable_config():
    model = HybridTemporalForecastNet(input_dim=FEATURE_DIM, hidden=64, layers=2, nhead=4)
    config = model.architecture_config()
    assert config["architecture"] == "tcn_transformer_hybrid_v1"
    assert config["hidden"] == 64
    assert config["entity_stream_enabled"] is False
