"""Memory-safe, streaming evaluator for the fixed forecast contract.

The source NPZ is extracted member-by-member to .npy files and opened with
memory mapping.  Predictions are reduced to counters per batch; no per-row
forecast ledger is retained in RAM.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.contract import contract_metadata
from forecast.temporal import HORIZONS, MODES, TRENDS, REGION_LABELS
from forecast.train import load_model


def materialize_memmap(npz_path: Path, cache_dir: Path) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as archive:
        for info in archive.infolist():
            target = cache_dir / Path(info.filename).name
            if not target.exists() or target.stat().st_size != info.file_size:
                with archive.open(info) as src, target.open("wb") as dst:
                    while True:
                        block = src.read(8 * 1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
    arrays = {}
    for path in cache_dir.glob("*.npy"):
        arrays[path.stem] = np.load(path, mmap_mode="r", allow_pickle=False)
    return arrays


def sample_indices(game_ids, sides, max_examples: int) -> np.ndarray:
    n = len(game_ids)
    if max_examples <= 0 or max_examples >= n:
        return np.arange(n, dtype=np.int64)
    # Equal quota per game and side, with deterministic evenly spaced rows.
    groups = []
    unique_games = np.unique(game_ids)
    quota = max(1, max_examples // max(1, len(unique_games) * 2))
    for gid in unique_games:
        for side in ("红", "蓝"):
            rows = np.flatnonzero((game_ids == gid) & (sides == side))
            if not len(rows):
                continue
            k = min(len(rows), quota)
            groups.append(rows[np.linspace(0, len(rows) - 1, k, dtype=np.int64)])
    idx = np.unique(np.concatenate(groups)) if groups else np.arange(0, n, max(1, n // max_examples))
    if len(idx) > max_examples:
        idx = idx[np.linspace(0, len(idx) - 1, max_examples, dtype=np.int64)]
    return np.asarray(idx, dtype=np.int64)


def confusion(n_classes):
    return np.zeros((n_classes, n_classes), dtype=np.int64)


def cat_score(cm):
    n = int(cm.sum())
    if not n:
        return {"n": 0, "accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    recalls, f1s = [], []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c]); fp = float(cm[:, c].sum() - tp); fn = float(cm[c, :].sum() - tp)
        support = tp + fn
        recalls.append(tp / support if support else 0.0)
        den = 2.0 * tp + fp + fn
        f1s.append(2.0 * tp / den if den else 0.0)
    return {"n": n, "accuracy": float(np.trace(cm) / n),
            "macro_f1": float(np.mean(f1s)), "balanced_accuracy": float(np.mean(recalls))}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test-npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--max-examples", type=int, default=0,
                   help="0=all rows; use a positive number for a safe stratified smoke run")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--position-alpha", type=float, default=1.0,
                   help="blend model position with persistence: alpha=1 model, alpha=0 persistence")
    a = p.parse_args(argv)
    npz_path = Path(a.test_npz)
    cache = Path(a.cache_dir) if a.cache_dir else npz_path.with_suffix(".npy_cache")
    data = materialize_memmap(npz_path, cache)
    indices = sample_indices(data["game_ids"], data["sides"].astype(str), a.max_examples)
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                       else ("cpu" if a.device == "auto" else a.device))
    model, checkpoint = load_model(a.model, device=dev)
    model.eval()
    widths = 20 if int(data["feature_dim"][0]) == 329 else 14
    slots = int(data["entity_y"].shape[2])
    tasks = [("global_mode", "mode_y", len(MODES), "base_mode"),
             ("opponent_trend", "trend_y", len(TRENDS), None),
             ("enemy_region", "region_y", len(REGION_LABELS), "base_region")]
    metrics = {}
    for h in HORIZONS:
        metrics[h] = {"global": {}, "entity": {
            "presence_model": [0, 0], "presence_base": [0, 0],
            "position_model": [0.0, 0], "position_base": [0.0, 0],
            "hp_model": [0.0, 0], "hp_base": [0.0, 0],
            "zone_model": [0, 0],
        }}
        for task, _key, classes, _base in tasks:
            metrics[h]["global"][task] = {"model": confusion(classes), "baseline": confusion(classes)}

    with torch.no_grad():
        for start in range(0, len(indices), int(a.batch_size)):
            rows = indices[start:start + int(a.batch_size)]
            xb = torch.from_numpy(np.asarray(data["x"][rows], dtype=np.float32)).to(dev, non_blocking=True)
            outputs = model(xb)
            last = xb[:, -1, :].detach().cpu().numpy()
            for hi, h in enumerate(HORIZONS):
                out = outputs[h]
                for task, key, classes, base_key in tasks:
                    y = np.asarray(data[key][rows, hi], dtype=np.int64)
                    valid = y >= 0
                    if not np.any(valid):
                        continue
                    pred = out[task].argmax(-1).detach().cpu().numpy()
                    cm = metrics[h]["global"][task]
                    np.add.at(cm["model"], (y[valid], pred[valid]), 1)
                    if base_key:
                        base = np.asarray(data[base_key][rows], dtype=np.int64)
                    else:
                        base = np.full(len(rows), 1, dtype=np.int64)
                    np.add.at(cm["baseline"], (y[valid], base[valid]), 1)
                target = np.asarray(data["entity_y"][rows, hi], dtype=np.float32)
                state = out["entity_state"].detach().cpu()
                presence = torch.sigmoid(state[..., 0]).numpy() >= 0.5
                valid = target[..., 0] >= 0
                truth_present = target[..., 0] > 0.5
                base_present = np.stack([last[:, i * widths + 3] > 0.5 for i in range(slots)], axis=1)
                em = metrics[h]["entity"]
                em["presence_model"][0] += int(np.sum(presence[valid] == truth_present[valid]))
                em["presence_model"][1] += int(np.sum(valid))
                em["presence_base"][0] += int(np.sum(base_present[valid] == truth_present[valid]))
                em["presence_base"][1] += int(np.sum(valid))
                present = valid & truth_present
                if np.any(present):
                    pos = torch.sigmoid(state[..., 1:3]).numpy()
                    alpha = float(np.clip(a.position_alpha, 0.0, 1.0))
                    hp = torch.sigmoid(state[..., 3]).numpy()
                    current_pos = np.stack([last[:, i * widths:i * widths + 2] for i in range(slots)], axis=1)
                    pos = current_pos + alpha * (pos - current_pos)
                    current_hp = np.stack([last[:, i * widths + 2] for i in range(slots)], axis=1)
                    em["position_model"][0] += float(np.abs(pos[present] - target[..., 1:3][present]).sum())
                    em["position_model"][1] += int(np.sum(present) * 2)
                    em["position_base"][0] += float(np.abs(current_pos[present] - target[..., 1:3][present]).sum())
                    em["position_base"][1] += int(np.sum(present) * 2)
                    em["hp_model"][0] += float(np.abs(hp[present] - target[..., 3][present]).sum())
                    em["hp_model"][1] += int(np.sum(present))
                    em["hp_base"][0] += float(np.abs(current_hp[present] - target[..., 3][present]).sum())
                    em["hp_base"][1] += int(np.sum(present))
                zone_y = np.asarray(data["entity_zone_y"][rows, hi], dtype=np.int64)
                zv = zone_y >= 0
                if np.any(zv):
                    zp = out["entity_zone_state"].argmax(-1).detach().cpu().numpy()
                    em["zone_model"][0] += int(np.sum(zp[zv] == zone_y[zv]))
                    em["zone_model"][1] += int(np.sum(zv))

    result = {"contract": contract_metadata(), "device": str(dev),
              "model": checkpoint.get("model_type"), "loss_profile": checkpoint.get("loss_profile"),
              "rows_total": int(len(data["x"])), "rows_scored": int(len(indices)),
              "sampled": bool(a.max_examples > 0), "batch_size": int(a.batch_size),
              "memmap_cache": str(cache), "horizons": {}}
    for h in HORIZONS:
        g = {}
        for task, _, _, _ in tasks:
            g[task] = {name: cat_score(cm) for name, cm in metrics[h]["global"][task].items()}
        e = metrics[h]["entity"]
        result["horizons"][str(h)] = {
            "global": g,
            "entity": {
                "presence": {"model_accuracy": e["presence_model"][0] / max(1, e["presence_model"][1]),
                              "baseline_accuracy": e["presence_base"][0] / max(1, e["presence_base"][1]),
                              "n": e["presence_model"][1]},
                "position": {"model_mae_normalized": e["position_model"][0] / max(1, e["position_model"][1]),
                              "baseline_mae_normalized": e["position_base"][0] / max(1, e["position_base"][1]),
                              "n": e["position_model"][1]},
                "hp": {"model_mae": e["hp_model"][0] / max(1, e["hp_model"][1]),
                       "baseline_mae": e["hp_base"][0] / max(1, e["hp_base"][1]),
                       "n": e["hp_model"][1]},
                "functional_zone": {"model_accuracy": e["zone_model"][0] / max(1, e["zone_model"][1]),
                                     "coverage": e["zone_model"][1] / max(1, len(indices) * slots)},
            },
        }
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "rows_scored": len(indices), "device": str(dev)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
