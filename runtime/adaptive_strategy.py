"""Causal, feedback-driven strategy routing for temporal forecasts.

The router is deliberately separate from the forecaster.  A prediction is
registered at issue time, and it can only be scored after the corresponding
future truth is supplied by the caller.  While a model is healthy the output
is unchanged.  If a rolling comparison shows that a persistence baseline is
consistently better, the router switches that task/horizon to persistence;
after sustained recovery it switches back to the learned model.

This is a runtime guardrail, not a second training loop: no future replay
fields are read while a forecast is being issued.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import math


TASKS = ("global_mode", "opponent_trend", "enemy_region")
ADAPTIVE_STRATEGY_VERSION = "rolling_brier_router_v1"


class MatchLocalOpponentResidualCalibrator:
    """Small, resettable residual corrector for 10-second enemy trend.

    It estimates class-wise bias from *matured issued forecasts* within the
    current observable motion regime. The default gate applies only in the
    hold/mixed regime; other contexts retain the base model distribution.
    It never persists team identity or updates model parameters.
    """

    VERSION = "match_local_opponent_residual_v1"

    def __init__(self, *, window=20, prior_strength=10.0,
                 apply_context="hold_or_mixed"):
        self.window = max(1, int(window))
        self.prior_strength = max(0.0, float(prior_strength))
        self.apply_context = str(apply_context)
        self.reset()

    def reset(self):
        self._residuals = defaultdict(lambda: deque(maxlen=self.window))

    def sample_count(self, context="hold_or_mixed"):
        return len(self._residuals[str(context)])

    def observe(self, probabilities, context, truth):
        """Update with the forecast actually issued and its matured label."""
        probabilities = _as_probabilities(probabilities)
        if probabilities is None or str(truth) not in probabilities:
            return {"updated": False, "samples": self.sample_count(context)}
        residual = {
            label: (1.0 if label == str(truth) else 0.0) - probabilities[label]
            for label in probabilities
        }
        self._residuals[str(context)].append(residual)
        return {"updated": True, "samples": self.sample_count(context)}

    def apply(self, probabilities, context):
        """Return calibrated probabilities plus an audit record."""
        probabilities = _as_probabilities(probabilities)
        if probabilities is None:
            return None, {"version": self.VERSION, "applied": False,
                          "reason": "invalid_probabilities"}
        context = str(context or "unknown")
        history = list(self._residuals[context])
        applied = context == self.apply_context and bool(history)
        if not applied:
            return probabilities, {
                "version": self.VERSION, "applied": False,
                "context": context, "samples": len(history),
                "reason": ("context_gate" if context != self.apply_context
                           else "warmup"),
            }
        labels = list(probabilities)
        mean_residual = {
            label: sum(row.get(label, 0.0) for row in history) / len(history)
            for label in labels
        }
        shrink = (len(history) / (len(history) + self.prior_strength)
                  if self.prior_strength else 1.0)
        logits = {label: math.log(max(probabilities[label], 1e-9))
                  + shrink * mean_residual[label] for label in labels}
        peak = max(logits.values())
        exp_values = {label: math.exp(value - peak) for label, value in logits.items()}
        total = sum(exp_values.values()) or 1.0
        calibrated = {label: value / total for label, value in exp_values.items()}
        return calibrated, {
            "version": self.VERSION, "applied": True,
            "context": context, "samples": len(history),
            "shrink": round(float(shrink), 6),
        }


def _as_probabilities(value):
    """Normalise a probability mapping and reject malformed input."""
    if not isinstance(value, dict) or not value:
        return None
    result = {}
    for label, probability in value.items():
        try:
            p = float(probability)
        except (TypeError, ValueError):
            continue
        if math.isfinite(p):
            result[str(label)] = max(0.0, p)
    total = sum(result.values())
    if not result or total <= 0.0:
        return None
    return {label: probability / total for label, probability in result.items()}


def _one_hot(label, labels):
    """Build a persistence distribution from the currently observed label."""
    if label is None:
        return None
    labels = [str(item) for item in (labels or [])]
    label = str(label)
    if labels and label not in labels:
        return None
    return {item: 1.0 if item == label else 0.0 for item in (labels or [label])}


def _brier(probabilities, truth):
    probabilities = probabilities or {}
    labels = set(probabilities) | {str(truth)}
    return sum((float(probabilities.get(label, 0.0)) -
                (1.0 if label == str(truth) else 0.0)) ** 2
               for label in labels)


def _baseline_for_task(task, horizon, baseline_source, prediction):
    """Resolve a caller-provided persistence distribution for one head."""
    source = baseline_source or {}
    task_source = source.get(task, {}) if isinstance(source, dict) else {}
    if isinstance(task_source, dict):
        value = task_source.get(str(horizon), task_source.get(int(horizon)))
        if isinstance(value, dict) and "probabilities" in value:
            value = value.get("probabilities")
        normalised = _as_probabilities(value)
        if normalised is not None:
            return normalised

    # The prediction may carry a precomputed baseline.  This keeps the router
    # usable outside TemporalForecastRuntime as well.
    embedded = prediction.get("adaptive_baseline") if isinstance(prediction, dict) else None
    if isinstance(embedded, dict):
        return _baseline_for_task(task, horizon, embedded, {})
    return None


class AdaptiveStrategyRouter:
    """Switch between a learned head and persistence using matured outcomes.

    ``min_samples`` prevents reacting to a few unlucky frames.  ``patience``
    and ``recovery_patience`` count consecutive post-warmup observations whose
    rolling Brier gap exceeds ``margin``.  State is maintained per task and
    horizon, so a weak 120-second region head cannot disable a good 10-second
    global-mode head.
    """

    def __init__(self, *, window=40, min_samples=20, patience=3,
                 recovery_patience=4, margin=0.03, max_pending=4096):
        self.window = max(2, int(window))
        self.min_samples = max(1, int(min_samples))
        self.patience = max(1, int(patience))
        self.recovery_patience = max(1, int(recovery_patience))
        self.margin = max(0.0, float(margin))
        self.max_pending = max(1, int(max_pending))
        self._states = defaultdict(self._new_state)
        self._pending = {}
        self._pending_order = deque()

    def reset(self):
        """Clear match-local losses, pending forecasts, and route state."""
        self._states = defaultdict(self._new_state)
        self._pending = {}
        self._pending_order = deque()

    def _new_state(self):
        return {
            "strategy": "model",
            "samples": 0,
            "model_losses": deque(maxlen=self.window),
            "baseline_losses": deque(maxlen=self.window),
            "bad_streak": 0,
            "good_streak": 0,
            "switch_count": 0,
            "last_gap": None,
            "last_reason": "warmup",
        }

    @staticmethod
    def _key(task, horizon):
        return str(task), str(int(horizon))

    @staticmethod
    def _forecast_id(prediction):
        issued = prediction.get("issued_at", 0.0) if isinstance(prediction, dict) else 0.0
        source_hash = prediction.get("input_snapshot_hash", "") if isinstance(prediction, dict) else ""
        material = f"{issued}|{source_hash}|{id(prediction)}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]

    def _state_meta(self, task, horizon):
        state = self._states[self._key(task, horizon)]
        return {
            "strategy": state["strategy"],
            "samples": state["samples"],
            "rolling_gap": (round(float(state["last_gap"]), 6)
                             if state["last_gap"] is not None else None),
            "bad_streak": state["bad_streak"],
            "good_streak": state["good_streak"],
            "reason": state["last_reason"],
        }

    def _record_pending(self, forecast_id, record):
        self._pending[forecast_id] = record
        self._pending_order.append(forecast_id)
        while len(self._pending_order) > self.max_pending:
            old = self._pending_order.popleft()
            self._pending.pop(old, None)

    def apply(self, prediction, *, baselines=None, forecast_id=None):
        """Register and adapt a forecast, returning a detached copy.

        ``baselines`` should be a mapping ``task -> horizon -> probabilities``
        generated from the latest observable state.  If omitted, the router
        looks for ``prediction['adaptive_baseline']``.  Missing baselines leave
        a head on the learned model and are reported as ``unavailable``.
        """
        if not isinstance(prediction, dict):
            raise ValueError("prediction must be a mapping")
        output = deepcopy(prediction)
        forecast_id = str(forecast_id or self._forecast_id(prediction))
        pending = {}
        metadata = {
            "version": ADAPTIVE_STRATEGY_VERSION,
            "forecast_id": forecast_id,
            "feedback_required": True,
            "by_task_horizon": {},
        }
        horizons = output.get("horizons") or {}
        for horizon_key, horizon_output in horizons.items():
            try:
                horizon = int(horizon_key)
            except (TypeError, ValueError):
                continue
            for task in TASKS:
                head = horizon_output.get(task) or {}
                # ``TemporalForecastRuntime`` may already apply the static,
                # validation-tuned persistence blend before this online
                # router runs.  Compare the raw calibrated expert to the
                # current-state baseline, otherwise a persistence-selected
                # head is mistakenly recorded as the neural model and the
                # feedback loop can never learn which side was better.
                model_probs = _as_probabilities(
                    head.get("model_probabilities") or head.get("probabilities"))
                if model_probs is None:
                    continue
                baseline = _baseline_for_task(task, horizon, baselines, prediction)
                key = self._key(task, horizon)
                state = self._states[key]
                selected = state["strategy"]
                if baseline is None:
                    selected = "model"
                elif selected == "persistence":
                    labels = set(model_probs) | set(baseline)
                    baseline = {label: float(baseline.get(label, 0.0)) for label in labels}
                    total = sum(baseline.values()) or 1.0
                    baseline = {label: value / total for label, value in baseline.items()}
                    head["probabilities"] = {label: round(float(value), 6)
                                              for label, value in baseline.items()}
                    head["top"] = max(baseline, key=baseline.get)
                    head["confidence"] = round(float(baseline[head["top"]]), 6)
                    head["adaptive_strategy"] = "persistence"
                    head["adaptive_reason"] = state["last_reason"]
                else:
                    head["adaptive_strategy"] = "model"
                    # Keep the static causal fusion in the emitted forecast
                    # during warmup.  The online router only decides whether
                    # to fall back all the way to the persistence expert; it
                    # should not silently remove a validated fixed blend.
                metadata["by_task_horizon"][f"{task}@{horizon}s"] = self._state_meta(task, horizon)
                pending[key] = {"model": model_probs, "baseline": baseline}
        self._record_pending(forecast_id, {"tasks": pending,
                                           "issued_at": output.get("issued_at")})
        output["adaptive_strategy"] = metadata
        return output

    # Descriptive alias for callers that do not use the word "apply".
    register_forecast = apply

    def observe_outcome(self, forecast_id, truth, *, matured_horizons=None):
        """Score matured forecast horizons and update their strategy state.

        ``truth`` accepts ``task -> horizon -> label`` or
        ``task -> label`` (the latter is applied to every matured horizon).
        A caller may report horizons incrementally; un-matured horizons stay
        pending and cannot affect their router state. Calling this before an
        actual target timestamp would destroy causality, so the runtime does
        not do it.
        """
        forecast_id = str(forecast_id)
        record = self._pending.get(forecast_id)
        if record is None:
            return {"forecast_id": forecast_id, "updated": 0, "reason": "unknown_forecast"}
        truth = truth or {}
        task_records = record.get("tasks") or {}
        if matured_horizons is None:
            # Infer which horizons the supplied labels cover. A scalar label
            # or {label/top: ...} keeps the historical all-horizons behavior.
            matured = set()
            for task, task_truth in (truth.items() if isinstance(truth, dict) else []):
                task_horizons = {horizon for candidate_task, horizon in task_records
                                 if candidate_task == task}
                if isinstance(task_truth, dict):
                    numeric = set()
                    for key in task_truth:
                        try:
                            numeric.add(str(int(key)))
                        except (TypeError, ValueError):
                            continue
                    matured.update(task_horizons & numeric if numeric else task_horizons)
                else:
                    matured.update(task_horizons)
        else:
            matured = {str(int(horizon)) for horizon in matured_horizons}
        updated = 0
        remaining = {}
        processed = set()
        for key, item in task_records.items():
            task, horizon = key
            if horizon not in matured:
                remaining[key] = item
                continue
            processed.add(key)
            task_truth = truth.get(task) if isinstance(truth, dict) else None
            if isinstance(task_truth, dict):
                label = task_truth.get(horizon, task_truth.get(int(horizon),
                                                               task_truth.get("label")))
            else:
                label = task_truth
            if isinstance(label, dict):
                label = label.get("top", label.get("label"))
            if label is None or item.get("baseline") is None:
                continue
            state = self._states[key]
            model_loss = _brier(item["model"], label)
            baseline_loss = _brier(item["baseline"], label)
            state["model_losses"].append(model_loss)
            state["baseline_losses"].append(baseline_loss)
            state["samples"] += 1
            updated += 1
            model_mean = sum(state["model_losses"]) / len(state["model_losses"])
            baseline_mean = sum(state["baseline_losses"]) / len(state["baseline_losses"])
            gap = model_mean - baseline_mean
            state["last_gap"] = gap
            if state["samples"] < self.min_samples:
                state["last_reason"] = "warmup"
                continue
            if gap > self.margin:
                state["bad_streak"] += 1
                state["good_streak"] = 0
            elif gap < -self.margin:
                state["good_streak"] += 1
                state["bad_streak"] = 0
            else:
                state["bad_streak"] = 0
                state["good_streak"] = 0
            if state["strategy"] == "model" and state["bad_streak"] >= self.patience:
                state["strategy"] = "persistence"
                state["switch_count"] += 1
                state["last_reason"] = "model_underperforms_persistence"
                state["good_streak"] = 0
            elif (state["strategy"] == "persistence" and
                  state["good_streak"] >= self.recovery_patience):
                state["strategy"] = "model"
                state["switch_count"] += 1
                state["last_reason"] = "model_recovered"
                state["bad_streak"] = 0
            elif state["strategy"] == "model":
                state["last_reason"] = "model_kept"
            else:
                state["last_reason"] = "persistence_kept"
        if remaining:
            record["tasks"] = remaining
        else:
            self._pending.pop(forecast_id, None)
        return {"forecast_id": forecast_id, "updated": updated,
                "complete": not remaining,
                "pending_horizons": sorted({horizon for _, horizon in remaining}),
                "processed": [f"{task}@{horizon}s" for task, horizon in sorted(processed)],
                "states": {f"{task}@{horizon}s": self._state_meta(task, horizon)
                            for task, horizon in sorted(processed)}}

    def status(self):
        """Return an immutable diagnostic snapshot suitable for telemetry."""
        return {f"{task}@{horizon}s": self._state_meta(task, horizon)
                for task, horizon in sorted(self._states)}
