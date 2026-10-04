"""Causal enemy-region transition-table sidecar.

The table is fitted from train rows only.  Runtime lookup uses only the
current visible region, side, issue-time bucket, and current mode, then blends
the learned destination distribution with a smoothed persistence prior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forecast.temporal import REGION_LABELS


class EnemyRegionTransitionEnsemble:
    """Load and apply a frozen train-to-validation transition table."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = str(manifest_path)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("schema") != "enemy-region-transition-manifest-v1":
            raise ValueError("unsupported enemy region transition manifest")
        self.horizon_seconds = int(manifest.get("horizon_seconds", 30))
        self.alpha = float(manifest["alpha_transition"])
        self.smoothing = float(manifest["laplace_smoothing"])
        self.time_bucket_seconds = int(manifest["time_bucket_seconds"])
        if not 0.0 <= self.alpha <= 1.0 or self.smoothing <= 0.0:
            raise ValueError("invalid enemy region transition parameters")
        self._table = manifest.get("table") or {}

    @staticmethod
    def _side_key(side: str) -> str:
        return str(side)

    def _key(self, base_region: int, side: str, issued_at: float,
             base_mode: int) -> str:
        return json.dumps([
            int(base_region), self._side_key(side),
            int(max(0.0, float(issued_at)) // self.time_bucket_seconds),
            int(base_mode),
        ], ensure_ascii=False, separators=(",", ":"))

    def predict(self, *, base_region: int, side: str, issued_at: float,
                base_mode: int) -> dict:
        classes = len(REGION_LABELS)
        if not 0 <= int(base_region) < classes:
            raise ValueError("invalid baseline enemy region")
        row = self._table.get(self._key(base_region, side, issued_at, base_mode))
        if row is None:
            transition = np.full(classes, 1.0 / classes, dtype=np.float64)
            matched = False
        else:
            transition = np.asarray(row, dtype=np.float64)
            transition = transition / max(float(transition.sum()), 1e-12)
            matched = True
        persistence = np.full(classes, 0.02 / (classes - 1), dtype=np.float64)
        persistence[int(base_region)] = 0.98
        probabilities = self.alpha * transition + (1.0 - self.alpha) * persistence
        probabilities /= max(float(probabilities.sum()), 1e-12)
        pmap = {label: round(float(value), 6)
                for label, value in zip(REGION_LABELS, probabilities)}
        top = max(pmap, key=pmap.get)
        return {
            "top": top,
            "confidence": pmap[top],
            "probabilities": pmap,
            "source": f"enemy_region_transition_{self.horizon_seconds}s_locked_test_pass",
            "model_vs_baseline": {
                "baseline_top": REGION_LABELS[int(base_region)],
                "baseline_confidence": 0.98,
                "alpha_transition": self.alpha,
                "transition_key_matched": matched,
                "laplace_smoothing": self.smoothing,
            },
        }


__all__ = ["EnemyRegionTransitionEnsemble"]
