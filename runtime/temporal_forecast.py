"""Causal runtime adapter for the V3 per-entity temporal forecaster.

The adapter accepts an already visibility-masked observable snapshot.  It does
not build a snapshot from the replay database, and therefore cannot leak
future or hidden opponent fields into deployment inference.
"""

from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path


DEFAULT_CHECKPOINT = "out_m2/temporal_forecast_v3/temporal_gru_v1.pt"


class TemporalForecastRuntime:
    """Maintain a causal feature window and emit structured forecasts."""

    def __init__(self, checkpoint_path, *, device="auto", context=12,
                 team="红", fail_closed=True):
        self.checkpoint_path = str(checkpoint_path)
        self.device_name = device
        self.context = int(context)
        self.team = team
        self.fail_closed = bool(fail_closed)
        self.history = deque(maxlen=self.context)
        self._loaded = False
        self._load_error = None
        self._torch = None
        self._model = None
        self._device = None
        self._feature_vector = None
        self._horizons = None
        self._labels = None
        self._calibrate = None
        self._decode_entities = None

    def reset(self):
        self.history.clear()

    def available(self):
        self._ensure_loaded()
        return self._loaded

    def _ensure_loaded(self):
        if self._loaded or self._load_error is not None:
            return self._loaded
        if not Path(self.checkpoint_path).exists():
            self._load_error = f"checkpoint not found: {self.checkpoint_path}"
            return False
        try:
            import numpy as np
            import torch
            from forecast.entity import decode_entity_forecast
            from forecast.temporal import HORIZONS, MODES, REGION_LABELS, TRENDS, feature_vector
            from forecast.train import load_model
            from forecast.calibration import enforce_monotonic_horizon_confidence

            if self.device_name in (None, "auto", ""):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_name
            self._torch = torch
            self._np = np
            self._device = torch.device(device)
            self._model, self._checkpoint = load_model(
                self.checkpoint_path, device=self._device)
            if not getattr(self._model, "entity_slots", 0):
                raise ValueError("checkpoint has no per-entity head; use temporal_forecast_v3")
            self._feature_vector = feature_vector
            self._horizons = HORIZONS
            self._labels = {
                "global_mode": MODES,
                "opponent_trend": TRENDS,
                "enemy_region": REGION_LABELS,
            }
            self._calibrate = enforce_monotonic_horizon_confidence
            self._decode_entities = decode_entity_forecast
            self._loaded = True
        except Exception as exc:  # deployment must fail closed, not guess
            self._load_error = f"temporal forecast unavailable: {exc}"
            if not self.fail_closed:
                raise
        return self._loaded

    @staticmethod
    def _classification(logits_by_horizon, labels, calibrate):
        raw = {
            horizon: logits_by_horizon[horizon].detach().cpu().numpy()
            for horizon in logits_by_horizon
        }
        calibrated = calibrate(raw)
        output = {}
        for horizon, values in calibrated.items():
            probs = {label: round(float(value), 6)
                     for label, value in zip(labels, values)}
            top = max(probs, key=probs.get)
            output[str(horizon)] = {
                "top": top,
                "confidence": probs[top],
                "probabilities": probs,
            }
        return output

    def predict(self, observable_snapshot):
        """Predict from one current observable snapshot and past snapshots."""
        if not isinstance(observable_snapshot, dict):
            raise ValueError("observable_snapshot must be a mapping")
        if observable_snapshot.get("self_team", self.team) != self.team:
            raise ValueError("observable snapshot side does not match forecaster team")
        if not self._ensure_loaded():
            return None
        issued = float(observable_snapshot.get("issued_at", 0.0))
        grouped = {"self_entities": {}, "opponent_entities": {}}
        for entity_id, entity in (observable_snapshot.get("entities") or {}).items():
            role = entity.get("role")
            if role not in ("self", "opponent"):
                role = "self" if entity.get("team") == self.team else "opponent"
            grouped[f"{role}_entities"][str(entity_id)] = dict(entity)
        example = {
            "side": self.team,
            "issued_at": issued,
            "observable": grouped,
        }
        vector = self._feature_vector(example)
        self.history.append(vector)
        window = self._np.zeros((self.context, vector.shape[0]), dtype=self._np.float32)
        values = list(self.history)
        window[-len(values):] = self._np.asarray(values, dtype=self._np.float32)
        tensor = self._torch.from_numpy(window[None]).to(self._device)
        with self._torch.no_grad():
            outputs = self._model(tensor)
        global_mode = self._classification(
            {h: outputs[h]["global_mode"][0] for h in self._horizons},
            self._labels["global_mode"], self._calibrate)
        opponent_trend = self._classification(
            {h: outputs[h]["opponent_trend"][0] for h in self._horizons},
            self._labels["opponent_trend"], self._calibrate)
        enemy_region = self._classification(
            {h: outputs[h]["enemy_region"][0] for h in self._horizons},
            self._labels["enemy_region"], self._calibrate)
        horizons = {}
        for horizon in self._horizons:
            entities = self._decode_entities(
                outputs[horizon]["entity_state"][0].detach().cpu().numpy(),
                observing_side=self.team)
            horizons[str(horizon)] = {
                "global_mode": global_mode[str(horizon)],
                "opponent_trend": opponent_trend[str(horizon)],
                "enemy_region": enemy_region[str(horizon)],
                "entities": entities,
            }
        observable_hash = hashlib.sha256(
            repr(observable_snapshot).encode("utf-8")).hexdigest()
        return {
            "schema_version": "temporal-entity-forecast-v1",
            "model_version": "temporal_gru_entity_v1",
            "calibration_version": "monotonic_horizon_v1",
            "team": self.team,
            "issued_at": issued,
            "input_snapshot_hash": observable_hash,
            "causal_history_length": len(self.history),
            "horizons": horizons,
        }


def resolve_temporal_forecaster(checkpoint="auto", *, device="auto", team="红",
                                context=12):
    """Create the V3 runtime adapter, or return ``None`` when disabled."""
    if checkpoint in (None, "", "none", "null", "-"):
        return None
    path = DEFAULT_CHECKPOINT if checkpoint == "auto" else str(checkpoint)
    return TemporalForecastRuntime(path, device=device, team=team, context=context)
