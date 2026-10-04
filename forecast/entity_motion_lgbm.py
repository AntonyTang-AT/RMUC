"""Runtime adapter for the locked-test per-robot motion/zone experts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from forecast.entity import ENTITY_SLOTS
from forecast.temporal import MAP_X, MAP_Y
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS, locate_zone
from worldstate.functional_zone import FUNCTIONAL_ZONE_LABELS, assign_functional_zone


class EntityMotionLGBMEnsemble:
    """Load frozen horizon-specific position and zone experts.

    The adapter is deliberately sidecar-only: if a model or manifest cannot
    be loaded, the neural trajectory remains the caller's fallback.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = str(manifest_path)
        self._models: dict[int, dict[str, Any]] = {}
        self._configs: dict[int, dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _booster(path: Path):
        from lightgbm import Booster
        model_file = str(path)
        try:
            model_file = os.path.relpath(str(path), os.getcwd())
        except (ValueError, OSError):
            pass
        try:
            return Booster(model_file=model_file)
        except Exception:
            return Booster(model_file=str(path))

    def _load(self):
        manifest_path = Path(self.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "entity-motion-zone-lightgbm-manifest-v1":
            raise ValueError("unsupported entity motion LightGBM manifest")
        root = manifest_path.resolve().parent / "models"
        for raw_horizon, config in (manifest.get("horizons") or {}).items():
            horizon = int(raw_horizon)
            position = [self._booster(root / name) for name in config["position"]]
            classifiers = {}
            class_maps = {}
            for name, row in (config.get("classifiers") or {}).items():
                classifiers[name] = self._booster(root / row["model"])
                class_maps[name] = np.asarray(row.get("classes") or [], dtype=np.int64)
            self._models[horizon] = {
                "position": position, "classifiers": classifiers,
                "classes": class_maps,
            }
            self._configs[horizon] = dict(config)
        if set(self._models) != {10, 30, 60, 120}:
            raise ValueError("entity motion manifest must contain four horizons")

    @staticmethod
    def _features(history_rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not history_rows:
            raise ValueError("empty causal history")
        cur = np.asarray(history_rows[-1], dtype=np.float32).reshape(-1)
        prev = (np.asarray(history_rows[-2], dtype=np.float32).reshape(-1)
                if len(history_rows) >= 2 else cur)
        if cur.size != 232:
            raise ValueError(f"expected canonical 232 features, got {cur.size}")
        slots = cur[:-8].reshape(16, 14)
        prev_slots = prev[:-8].reshape(16, 14)
        delta = slots - prev_slots
        positions = slots[:, :2].reshape(-1)
        presence = slots[:, 9]
        globals_ = cur[-8:]
        rows = []
        for slot in range(16):
            one_hot = np.zeros(16, dtype=np.float32)
            one_hot[slot] = 1.0
            rows.append(np.concatenate([
                slots[slot], delta[slot], positions, presence, globals_, one_hot,
            ]))
        return np.asarray(rows, dtype=np.float32), slots[:, :2].copy(), (
            (slots[:, 3] > .5) & (slots[:, 9] > .5))

    @staticmethod
    def _zone_from_positions(points: np.ndarray, canonical: bool = False) -> list[str | None]:
        output = []
        for point in np.asarray(points):
            x, y = float(point[0] * MAP_X), float(point[1] * MAP_Y)
            zone = locate_zone(x, y) if canonical else assign_functional_zone(x, y)
            labels = CANONICAL_ZONE_LABELS if canonical else FUNCTIONAL_ZONE_LABELS
            output.append(zone if zone in labels else None)
        return output

    def predict(self, history_rows: list[np.ndarray]) -> dict[str, dict[str, Any]]:
        features, current, current_present = self._features(history_rows)
        result = {}
        for horizon in (10, 30, 60, 120):
            models = self._models[horizon]
            config = self._configs[horizon]
            pos = np.column_stack([model.predict(features) for model in models["position"]])
            pos = np.clip(pos, 0.0, 1.0)
            alpha = (float(config.get("blend_alpha", 1.0))
                     if bool(config.get("position_enabled", True)) else 0.0)
            blended = np.clip((1.0 - alpha) * current + alpha * pos, 0.0, 1.0)
            # Do not hallucinate movement for absent or currently untracked
            # slots. Presence/death remains owned by the neural lifecycle head.
            blended[~current_present] = current[~current_present]
            functional = self._predict_zone(
                models, "functional", features, current, current_present,
                float(config.get("functional_confidence_threshold", 1.01)),
                FUNCTIONAL_ZONE_LABELS)
            canonical = self._predict_zone(
                models, "canonical", features, current, current_present,
                float(config.get("canonical_confidence_threshold", 1.01)),
                CANONICAL_ZONE_LABELS)
            result[str(horizon)] = {
                "positions": blended,
                "raw_positions": pos,
                "functional_zones": functional,
                "canonical_zones": canonical,
                "blend_alpha": alpha,
                "source": "entity_motion_lightgbm_locked_test_candidate",
            }
        return result

    @staticmethod
    def _predict_zone(models, name, features, current, current_present,
                      threshold, labels):
        model = models["classifiers"].get(name)
        classes = models["classes"].get(name, np.asarray([], dtype=np.int64))
        current_zones = EntityMotionLGBMEnsemble._zone_from_positions(
            current, canonical=(name == "canonical"))
        result = [zone if bool(current_present[index]) else None
                  for index, zone in enumerate(current_zones)]
        if model is None or classes.size == 0:
            return result
        raw = model.predict(features)
        nclass = len(classes)
        raw = np.asarray(raw).reshape(len(features), nclass)
        index = np.argmax(raw, axis=1)
        confidence = np.max(raw, axis=1)
        for slot in range(len(result)):
            if not current_present[slot] or float(confidence[slot]) < threshold:
                continue
            class_idx = int(classes[int(index[slot])])
            if 0 <= class_idx < len(labels):
                result[slot] = labels[class_idx]
        return result


__all__ = ["EntityMotionLGBMEnsemble"]
