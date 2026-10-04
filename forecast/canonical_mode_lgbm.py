# -*- coding: utf-8 -*-
"""Runtime adapter for the locked-test canonical-side global-mode expert."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class CanonicalModeLGBMEnsemble:
    """Load frozen per-horizon LightGBM experts and decode causal features."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = str(manifest_path)
        self._models: dict[int, Any] = {}
        self._configs: dict[int, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        manifest = json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))
        if manifest.get("schema") != "canonical-side-causal-lightgbm-production-v1":
            raise ValueError("unsupported canonical mode LightGBM manifest")
        from lightgbm import Booster
        root = Path(self.manifest_path).resolve().parent
        for raw_horizon, row in (manifest.get("horizons") or {}).items():
            horizon = int(raw_horizon)
            path = Path(row["model_path"])
            if not path.is_absolute():
                path = root / path
            # LightGBM's Windows C API is more reliable with a normalized
            # ASCII-relative path than with a Unicode absolute path.
            model_file = str(path)
            try:
                model_file = os.path.relpath(str(path), os.getcwd())
            except (ValueError, OSError):
                pass
            self._models[horizon] = Booster(model_file=model_file)
            self._configs[horizon] = dict(row)
        if set(self._models) != {10, 30, 60, 120}:
            raise ValueError("canonical mode LightGBM manifest must contain four horizons")

    @staticmethod
    def _features(history_rows: list[np.ndarray], issued_at: float,
                  baseline_mode: int) -> np.ndarray:
        if not history_rows:
            raise ValueError("empty causal history")
        rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in history_rows]
        cur = rows[-1]
        prev = rows[-2] if len(rows) >= 2 else cur
        mean3 = np.mean(np.asarray(rows[-3:]), axis=0)
        mean6 = np.mean(np.asarray(rows[-6:]), axis=0)
        d1 = np.clip(cur - prev, -2.0, 2.0)
        d3 = np.clip(cur - (rows[-4] if len(rows) >= 4 else prev), -2.0, 2.0)
        one_hot = np.zeros(3, dtype=np.float32)
        if 0 <= int(baseline_mode) < 3:
            one_hot[int(baseline_mode)] = 1.0
        time = np.asarray([max(0.0, min(1.0, float(issued_at) / 420.0))], dtype=np.float32)
        return np.concatenate([cur, mean3, mean6, d1, d3, one_hot, time], dtype=np.float32)[None, :]

    @staticmethod
    def _probabilities(raw: np.ndarray, alpha: float, bias: list[float], baseline: int) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64).reshape(1, 3)
        prior = np.full((1, 3), 0.01, dtype=np.float64)
        if 0 <= int(baseline) < 3:
            prior[0, int(baseline)] = 0.98
        prior /= prior.sum(axis=1, keepdims=True)
        mixed = float(alpha) * raw + (1.0 - float(alpha)) * prior
        logits = np.log(np.clip(mixed, 1e-8, 1.0)) + np.asarray(bias, dtype=np.float64).reshape(1, 3)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, history_rows: list[np.ndarray], issued_at: float,
                baseline_mode: int) -> dict[str, dict[str, Any]]:
        x = self._features(history_rows, issued_at, baseline_mode)
        labels = ("进攻", "均衡", "防守")
        result = {}
        for horizon in (10, 30, 60, 120):
            cfg = self._configs[horizon]
            raw = self._models[horizon].predict(x)[0]
            probs = self._probabilities(raw, float(cfg["alpha"]),
                                        list(cfg.get("class_logit_bias") or [0.0, 0.0, 0.0]),
                                        int(baseline_mode))[0]
            pmap = {label: round(float(value), 6) for label, value in zip(labels, probs)}
            top = max(pmap, key=pmap.get)
            result[str(horizon)] = {
                "top": top,
                "confidence": pmap[top],
                "probabilities": pmap,
                "source": "canonical_mode_lightgbm_locked_test_pass",
                "model_vs_baseline": {
                    "baseline_top": labels[int(baseline_mode)] if 0 <= int(baseline_mode) < 3 else None,
                    "selected_agrees_with_baseline": top == labels[int(baseline_mode)] if 0 <= int(baseline_mode) < 3 else None,
                    "baseline_confidence": 0.98 if 0 <= int(baseline_mode) < 3 else None,
                    "alpha": float(cfg["alpha"]),
                    "class_logit_bias": list(cfg.get("class_logit_bias") or [0.0, 0.0, 0.0]),
                },
            }
        return result
