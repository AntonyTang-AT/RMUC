"""Pre-registered 50:50 LightGBM/GRU OOF blend on death-aware train only.

Five whole-game GroupKFold splits; no validation or test split is read. GRU
training duration and model configuration are fixed before scoring. The
50:50 mixture is never tuned.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.model import TemporalForecastNet
from forecast.temporal import HORIZONS, MODES, load_compact
from eval.experiment_death_aware_mode_lightgbm import features as tree_features

CLASS_IDS = np.arange(len(MODES))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_metrics(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = p.argmax(1)
    conf = np.zeros((3, 3), dtype=np.int64)
    np.add.at(conf, (y, pred), 1)
    f1s, recalls, classes = [], [], {}
    for c in CLASS_IDS:
        tp = int(conf[c, c])
        fp = int(conf[:, c].sum() - tp)
        fn = int(conf[c, :].sum() - tp)
        support = int(conf[c].sum())
        predicted = int(conf[:, c].sum())
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        if support:
            recalls.append(recall)
        f1s.append(f1)
        classes[MODES[c]] = {"support": support, "predicted": predicted,
                             "precision": float(precision), "recall": float(recall),
                             "f1": float(f1)}
    onehot = np.eye(3, dtype=np.float64)[y]
    pp = np.clip(p, 1e-8, 1.)
    pp = pp / pp.sum(1, keepdims=True)
    return {"n": int(len(y)), "accuracy": float(np.mean(pred == y)),
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.,
            "macro_f1": float(np.mean(f1s)),
            "brier": float(np.mean(np.sum((pp - onehot) ** 2, axis=1))),
            "log_loss": float(-np.log(pp[np.arange(len(y)), y]).mean()),
            "per_class": classes, "confusion_true_rows_pred_cols": conf.tolist()}


def persistence_probs(base):
    base = np.asarray(base, dtype=np.int64)
    p = np.full((len(base), 3), .01, dtype=np.float32)
    valid = (base >= 0) & (base < 3)
    p[np.arange(len(base))[valid], base[valid]] = .98
    return p


def per_game_side_metrics(y, p, gids, sides):
    scores = {}
    for gid in np.unique(gids):
        side_rows = []
        for side in ("红", "蓝"):
            m = (gids == gid) & (sides == side)
            if m.any():
                side_rows.append(model_metrics(y[m], p[m]))
        if len(side_rows) == 2:
            scores[int(gid)] = {metric: float(np.mean([s[metric] for s in side_rows]))
                                for metric in ("accuracy", "balanced_accuracy", "macro_f1",
                                               "brier", "log_loss")}
    return scores


def paired_game_bootstrap(candidate, comparator, seed, reps=2000):
    games = sorted(set(candidate) & set(comparator))
    if not games:
        raise ValueError("No complete red/blue games for paired bootstrap")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(games), size=(reps, len(games)))
    out = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "brier", "log_loss"):
        delta = np.asarray([candidate[g][metric] - comparator[g][metric] for g in games])
        boot = delta[idx].mean(1)
        out[metric] = {"mean_delta_blend_minus_comparator": float(delta.mean()),
                       "ci95": [float(np.quantile(boot, .025)),
                                float(np.quantile(boot, .975))]}
    out["paired_games"] = len(games)
    out["bootstrap_replicates"] = reps
    return out


def fold_loss_config(y_train, device):
    # Canonical business-weighted global-mode loss. Weights use the training
    # fold only; no held-out game labels enter any training parameter.
    counts = np.stack([np.bincount(y_train[y_train[:, j] >= 0, j], minlength=3)
                       for j in range(4)]).astype(np.float64)
    weights, smoothing = [], [0.03, 0.06, 0.10, 0.15]
    horizon_weight = [1.45, 1.15, .85, .60]
    for j in range(4):
        safe = np.maximum(counts[j], 1.)
        w = np.sqrt(counts[j].sum() / (3. * safe))
        w = np.clip(w, 1 / 3., 3.)
        w /= w.mean()
        weights.append(torch.tensor(w, dtype=torch.float32, device=device))
    return weights, smoothing, horizon_weight


def train_gru_fold(x, y, train_rows, held_rows, *, device, epochs, batch_size, seed):
    seed_everything(seed)
    model = TemporalForecastNet(input_dim=int(x.shape[-1]), hidden=128, layers=2).to(device)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y.astype(np.int64, copy=False))
    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(Subset(dataset, train_rows.tolist()), batch_size=batch_size,
                        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
                        generator=generator)
    weights, smoothing, horizon_weights = fold_loss_config(y[train_rows], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        sum_loss = 0.
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            losses = []
            for j, h in enumerate(HORIZONS):
                target = yb[:, j]
                valid = target >= 0
                if valid.any():
                    losses.append(horizon_weights[j] * F.cross_entropy(
                        out[h]["global_mode"][valid], target[valid], weight=weights[j],
                        label_smoothing=smoothing[j]))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            sum_loss += float(loss.detach()) * len(xb)
            n_seen += len(xb)
        epoch_losses.append(sum_loss / max(1, n_seen))

    model.eval()
    out_probs = {h: [] for h in HORIZONS}
    with torch.no_grad():
        for start in range(0, len(held_rows), batch_size * 2):
            rows = held_rows[start:start + batch_size * 2]
            xb = torch.from_numpy(x[rows]).to(device)
            out = model(xb)
            for h in HORIZONS:
                out_probs[h].append(torch.softmax(out[h]["global_mode"], dim=-1).cpu().numpy())
    return {h: np.concatenate(out_probs[h], axis=0).astype(np.float32)
            for h in HORIZONS}, epoch_losses


def run(train_path, output_dir, *, folds=5, epochs=2, batch_size=512,
        threads=8, device_name="auto", seed=1927):
    started = time.perf_counter()
    data = load_compact(train_path)
    contract_values = np.asarray(data["global_mode_label_version"]).astype(str)
    if len(np.unique(contract_values)) != 1 or "death_aware" not in contract_values[0]:
        raise ValueError(f"Expected one death-aware label contract, got {contract_values}")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else ("cpu" if device_name == "auto" else device_name))
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["mode_y"], dtype=np.int64)
    base = np.asarray(data["base_mode"], dtype=np.int64)
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"]).astype(str)
    tree_x = tree_features(data)
    n = len(y)
    probs = {name: np.full((n, 4, 3), np.nan, dtype=np.float32)
             for name in ("gru", "lightgbm", "blend", "persistence")}
    valid_all = np.zeros((n, 4), dtype=bool)
    fold_reports = []
    cv = GroupKFold(n_splits=folds)
    for fold, (tr, te) in enumerate(cv.split(x, groups=gids)):
        fold_start = time.perf_counter()
        gru_by_h, losses = train_gru_fold(
            x, y, tr, te, device=device, epochs=epochs, batch_size=batch_size,
            seed=seed + fold)
        fold_lgbm_n = {}
        for j, h in enumerate(HORIZONS):
            tr_valid = (y[tr, j] >= 0) & (base[tr] >= 0) & (base[tr] < 3)
            te_valid = (y[te, j] >= 0) & (base[te] >= 0) & (base[te] < 3)
            train_rows = tr[tr_valid]
            held_rows = te[te_valid]
            if len(np.unique(y[train_rows, j])) != 3:
                raise ValueError(f"fold {fold} horizon {h}: train fold lacks a class")
            lgbm = LGBMClassifier(objective="multiclass", num_class=3,
                n_estimators=260, learning_rate=.045, num_leaves=15,
                max_depth=-1, min_child_samples=90, reg_lambda=4.,
                colsample_bytree=.8, subsample=.85, subsample_freq=1,
                class_weight=None, verbosity=-1, n_jobs=threads,
                random_state=seed + fold)
            lgbm.fit(tree_x[train_rows], y[train_rows, j])
            lgbm_p = np.zeros((len(held_rows), 3), dtype=np.float32)
            for col, cls in enumerate(lgbm.classes_):
                lgbm_p[:, int(cls)] = lgbm.predict_proba(tree_x[held_rows])[:, col]
            gru_p = gru_by_h[h][te_valid]
            mixture = .5 * gru_p + .5 * lgbm_p
            mixture /= mixture.sum(1, keepdims=True)
            persist = persistence_probs(base[held_rows])
            probs["gru"][held_rows, j] = gru_p
            probs["lightgbm"][held_rows, j] = lgbm_p
            probs["blend"][held_rows, j] = mixture
            probs["persistence"][held_rows, j] = persist
            valid_all[held_rows, j] = True
            fold_lgbm_n[f"{h}s"] = {"train_rows": int(len(train_rows)),
                                     "heldout_rows": int(len(held_rows))}
        fold_reports.append({"fold": fold, "train_games": int(len(np.unique(gids[tr]))),
                             "heldout_games": int(len(np.unique(gids[te]))),
                             "gru_epochs": epochs, "gru_global_mode_train_loss": losses,
                             "lightgbm_rows": fold_lgbm_n,
                             "runtime_seconds": time.perf_counter() - fold_start})
        print(json.dumps({"fold": fold, "seconds": fold_reports[-1]["runtime_seconds"],
                          "gru_loss": losses}, ensure_ascii=False), flush=True)

    report = {
        "schema": "death-aware-gru-lightgbm-fixed-50-50-group-oof-v1",
        "test_access": False, "validation_access": False,
        "train_npz": str(Path(train_path).resolve()),
        "label_contract": contract_values[0], "train_rows": int(n),
        "train_games": int(len(np.unique(gids))), "fold_count": int(folds),
        "grouping": "GroupKFold by whole game_id; both red/blue views stay in the same fold",
        "prediction_policy": "un-calibrated softmax GRU and LightGBM probabilities; fixed arithmetic blend 0.5*GRU+0.5*LightGBM; no tuning or validation selection",
        "gru": {"architecture": "TemporalForecastNet GRU", "hidden": 128,
                "layers": 2, "epochs_per_fold": int(epochs),
                "batch_size": int(batch_size), "optimizer": "AdamW lr=0.002 weight_decay=0.0001",
                "loss": "global_mode-only business-weighted CE; horizon weights 1.45/1.15/0.85/0.60; class weights estimated separately on each training fold; smoothing 0.03/0.06/0.10/0.15"},
        "lightgbm": {"features": "current snapshot, trailing 3/6 means, causal 1/3 deltas, current-mode one-hot",
                     "n_estimators": 260, "learning_rate": .045, "num_leaves": 15,
                     "min_child_samples": 90, "reg_lambda": 4., "class_weight": None},
        "device": str(device), "folds": fold_reports, "horizons": {},
        "runtime_seconds": None,
        "limitations": ["OOF only; no validation or test scored", "GRU trained for a fixed two epochs per fold without early stopping; candidate is preliminary",
                         "probabilities are raw and uncalibrated; 50:50 weight was pre-fixed and not tuned",
                         "train group OOF estimates within-corpus generalization, not an untouched competition estimate"],
    }
    for j, h in enumerate(HORIZONS):
        mask = valid_all[:, j]
        yy, ss, gg = y[mask, j], sides[mask], gids[mask]
        per_method = {}
        side_metrics = {}
        group_metrics = {}
        for name in probs:
            p = probs[name][mask, j]
            per_method[name] = model_metrics(yy, p)
            side_metrics[name] = {
                side: model_metrics(yy[ss == side], p[ss == side])
                for side in ("红", "蓝")}
            group_metrics[name] = per_game_side_metrics(yy, p, gg, ss)
        paired = {}
        for i, comparator in enumerate(("lightgbm", "gru", "persistence")):
            paired[comparator] = paired_game_bootstrap(
                group_metrics["blend"], group_metrics[comparator],
                seed=seed + 500 + j * 10 + i)
        report["horizons"][f"{h}s"] = {
            "scored_rows": int(mask.sum()),
            "games_red_blue_paired": int(len(set(gg))),
            "pooled_all_rows": per_method,
            "by_side_pooled": side_metrics,
            "game_equal_red_blue_equal": {
                name: {metric: float(np.mean([v[metric] for v in gm.values()]))
                       for metric in ("accuracy", "balanced_accuracy", "macro_f1", "brier", "log_loss")}
                for name, gm in group_metrics.items()},
            "paired_game_bootstrap_blend_minus": paired,
            "invalid_or_uncorable_rows": int(n - mask.sum()),
        }
    report["runtime_seconds"] = time.perf_counter() - started
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "oof_probabilities.npz",
                        game_ids=gids, sides=sides, issued_at=data["issued_at"],
                        mode_y=y, base_mode=base, valid=valid_all,
                        **{name: p for name, p in probs.items()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "out_m2/temporal_forecast_death_aware_candidate/train.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "out_m2/death_aware_global_mode_experiments/gru_lgbm_50_50_oof"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1927)
    args = parser.parse_args()
    report = run(args.train, args.output_dir, folds=args.folds,
                 epochs=args.epochs, batch_size=args.batch_size,
                 threads=args.threads, device_name=args.device, seed=args.seed)
    print(json.dumps({"output_dir": args.output_dir,
                      "runtime_seconds": report["runtime_seconds"],
                      "summary": {h: row["game_equal_red_blue_equal"]
                                  for h, row in report["horizons"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
