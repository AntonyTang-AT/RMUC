"""Runtime adapter for locked-test trend/region LightGBM experts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from forecast.temporal import REGION_LABELS, TRENDS


class GlobalTaskLGBMEnsemble:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = str(manifest_path)
        self._models: dict[str, dict[int, Any]] = {"opponent_trend": {}, "enemy_region": {}}
        self._configs: dict[str, dict[int, dict[str, Any]]] = {"opponent_trend": {}, "enemy_region": {}}
        self._load()

    @staticmethod
    def _booster(path: Path):
        from lightgbm import Booster
        try:
            relative = os.path.relpath(str(path), os.getcwd())
        except (ValueError, OSError):
            relative = str(path)
        try:
            return Booster(model_file=relative)
        except Exception:
            return Booster(model_file=str(path))

    def _load(self):
        manifest_path = Path(self.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "global-task-lightgbm-manifest-v1":
            raise ValueError("unsupported global task LightGBM manifest")
        root = manifest_path.resolve().parent / "models"
        for task, horizons in (manifest.get("tasks") or {}).items():
            for raw_horizon, row in (horizons or {}).items():
                h = int(raw_horizon)
                self._configs.setdefault(task, {})[h] = dict(row)
                if row.get("enabled"):
                    self._models.setdefault(task, {})[h] = self._booster(root / row["model_path"])
        if not self._models["opponent_trend"]:
            raise ValueError("no promoted opponent trend expert")

    @staticmethod
    def _features(history_rows: list[np.ndarray], issued_at: float,
                  baseline_mode: int) -> np.ndarray:
        if not history_rows:
            raise ValueError("empty causal history")
        rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in history_rows]
        cur, prev = rows[-1], rows[-2] if len(rows) >= 2 else rows[-1]
        mean3 = np.mean(np.asarray(rows[-3:]), axis=0)
        mean6 = np.mean(np.asarray(rows[-6:]), axis=0)
        d1 = np.clip(cur - prev, -2.0, 2.0)
        d3 = np.clip(cur - (rows[-4] if len(rows) >= 4 else prev), -2.0, 2.0)
        slot_delta = d1[:-8].reshape(-1, 14)
        motion = np.concatenate([
            slot_delta[:, :2].mean(axis=0),
            np.abs(slot_delta[:, :2]).mean(axis=0),
            np.asarray([(slot_delta[:, 9] > .05).mean()], dtype=np.float32),
        ])
        base = np.zeros(3, dtype=np.float32)
        if 0 <= int(baseline_mode) < 3:
            base[int(baseline_mode)] = 1.0
        time = np.asarray([max(0.0, min(1.0, float(issued_at) / 420.0))], dtype=np.float32)
        return np.concatenate([cur, mean3, mean6, d1, d3, motion, base, time], dtype=np.float32)[None, :]

    @staticmethod
    def _probabilities(raw, alpha, baseline, classes):
        raw = np.asarray(raw, dtype=np.float64).reshape(1, -1)
        prior = np.full_like(raw, .01 / max(1, classes - 1))
        if 0 <= baseline < classes:
            prior[0, baseline] = .98
        mixed = float(alpha) * raw + (1.0 - float(alpha)) * prior
        mixed /= mixed.sum(axis=1, keepdims=True)
        return mixed[0]

    def predict(self, history_rows: list[np.ndarray], issued_at: float,
                baseline_mode: int, baseline_region: int) -> dict[str, dict[str, Any]]:
        x = self._features(history_rows, issued_at, baseline_mode)
        result = {}
        for horizon in (10, 30, 60, 120):
            row = {}
            for task, baseline, labels in (
                ("opponent_trend", TRENDS.index("稳定"), TRENDS),
                ("enemy_region", int(baseline_region), REGION_LABELS),
            ):
                model = self._models.get(task, {}).get(horizon)
                config = self._configs.get(task, {}).get(horizon, {})
                if model is None:
                    continue
                probs = self._probabilities(model.predict(x)[0],
                                            float(config.get("alpha", 1.0)),
                                            baseline, len(labels))
                pmap = {label: round(float(value), 6) for label, value in zip(labels, probs)}
                top = max(pmap, key=pmap.get)
                row[task] = {
                    "top": top,
                    "confidence": pmap[top],
                    "probabilities": pmap,
                    "source": "global_task_lightgbm_locked_test_pass",
                    "model_vs_baseline": {
                        "baseline_top": labels[baseline] if 0 <= baseline < len(labels) else None,
                        "baseline_confidence": .98 if 0 <= baseline < len(labels) else None,
                        "alpha": float(config.get("alpha", 1.0)),
                    },
                }
            result[str(horizon)] = row
        return result


__all__ = ["GlobalTaskLGBMEnsemble"]
