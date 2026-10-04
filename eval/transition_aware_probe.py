"""Independent transition-aware global-mode probe on frozen compact splits.

This is an experiment only.  It does not alter the production feature or model
pipeline.  Inputs are the current-frame compact feature vector plus a one-hot
encoding of the current causal mode.  For each horizon, logistic regression
predicts whether the mode changes and, conditional on a change, its destination.
The validation split selects only the transition probability threshold.  The
test split is evaluated once after that choice is frozen.

Example::

    python eval/transition_aware_probe.py \
      --work-dir out_m2/temporal_hybrid_motion_death_aware \
      --out out_m2/transition_aware_probe/transition_probe_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.hybrid_model import HybridTemporalForecastNet
from forecast.temporal import HORIZONS, MODES


def load_split(path: Path):
    """Load only the input feature key and the global-mode scoring columns."""
    with np.load(path, allow_pickle=False) as archive:
        x_sequence = archive["x"]
        current_features = np.array(x_sequence[:, -1, :], dtype=np.float32, copy=True)
        del x_sequence
        y = np.array(archive["mode_y"], dtype=np.int64, copy=True)
        current = np.array(archive["base_mode"], dtype=np.int64, copy=True)
        sides = np.array(archive["sides"], copy=True)
        game_ids = np.array(archive["game_ids"], dtype=np.int64, copy=True)
    valid_current = (current >= 0) & (current < len(MODES))
    one_hot = np.eye(len(MODES), dtype=np.float32)[np.clip(current, 0, len(MODES) - 1)]
    features = np.concatenate((current_features, one_hot), axis=1)
    del current_features, one_hot
    gc.collect()
    return features, y, current, sides, game_ids, valid_current


def transition_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=250,
            tol=1e-3, random_state=0,
        ),
    )


def destination_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=250,
            tol=1e-3, random_state=0,
        ),
    )


def predictions_from_parts(base, transition_probability, destination, threshold):
    predict_change = transition_probability >= threshold
    result = np.asarray(base, dtype=np.int64).copy()
    if np.any(predict_change):
        result[predict_change] = destination[predict_change]
    return result


def metrics(y, pred, base):
    true_flip = y != base
    pred_flip = pred != base
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(len(MODES)),
                                    average="macro", zero_division=0)),
        "flip_recall": (float(recall_score(true_flip, pred_flip, zero_division=0))
                        if np.any(true_flip) else None),
        "flip_precision": (float(precision_score(true_flip, pred_flip, zero_division=0))
                           if np.any(pred_flip) else 0.0),
        "flip_support": int(np.sum(true_flip)),
        "predicted_flip_count": int(np.sum(pred_flip)),
        "n": int(len(y)),
    }


def precision_score(y_true, y_pred, zero_division=0):
    tp = int(np.sum(np.asarray(y_true, dtype=bool) & np.asarray(y_pred, dtype=bool)))
    fp = int(np.sum(~np.asarray(y_true, dtype=bool) & np.asarray(y_pred, dtype=bool)))
    if tp + fp == 0:
        return float(zero_division)
    return tp / (tp + fp)


def side_metrics(y, pred, base, sides):
    result = {}
    for side in ("红", "蓝"):
        mask = sides == side
        result[side] = metrics(y[mask], pred[mask], base[mask])
    numeric = ("accuracy", "macro_f1", "flip_recall", "flip_precision")
    result["red_blue_equal_weight"] = {
        key: (float(np.mean([result[s][key] for s in ("红", "蓝")]))
              if all(result[s][key] is not None for s in ("红", "蓝")) else None)
        for key in numeric
    }
    return result


def choose_threshold(y, base, p_change, destination):
    """Pick validation macro-F1, limiting accuracy regression to 2 points."""
    persistence_accuracy = float(accuracy_score(y, base))
    candidates = []
    for threshold in [i / 100 for i in range(0, 101, 2)]:
        pred = predictions_from_parts(base, p_change, destination, threshold)
        score = metrics(y, pred, base)
        if score["accuracy"] >= persistence_accuracy - 0.02:
            candidates.append((score["macro_f1"], score["accuracy"],
                               score["flip_recall"], threshold, score))
    constraint_met = bool(candidates)
    if not candidates:
        for threshold in [i / 100 for i in range(0, 101, 2)]:
            pred = predictions_from_parts(base, p_change, destination, threshold)
            score = metrics(y, pred, base)
            candidates.append((score["macro_f1"], score["accuracy"],
                               score["flip_recall"], threshold, score))
    # Conservative threshold wins a complete tie.
    selected = max(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    return float(selected[3]), constraint_met, selected[4]


def hybrid_test_predictions(test_npz: Path, checkpoint_path: Path, device_name="auto"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else ("cpu" if device_name == "auto" else device_name))
    model = HybridTemporalForecastNet(
        input_dim=int(checkpoint["input_dim"]), hidden=int(checkpoint["hidden"]),
        layers=int(checkpoint["layers"]), nhead=int(checkpoint["nhead"]),
        dropout=float(checkpoint["dropout"]),
        max_seq_len=int(checkpoint.get("max_seq_len", 256)),
        entity_slots=int(checkpoint.get("entity_slots", 0)),
        zone_classes=int(checkpoint.get("zone_classes", 0)),
        canonical_zone_classes=int(checkpoint.get("canonical_zone_classes", 0)),
        horizon_fusion=bool(checkpoint.get("horizon_fusion", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with np.load(test_npz, allow_pickle=False) as archive:
        x = archive["x"]
    predictions = {h: [] for h in HORIZONS}
    batch_size = 256
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.asarray(x[start:start + batch_size])).to(device)
            output = model(xb)
            for h in HORIZONS:
                predictions[h].extend(output[h]["global_mode"].argmax(-1).cpu().numpy().tolist())
            del xb, output
    del x, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {h: np.asarray(values, dtype=np.int64) for h, values in predictions.items()}, device.type


def run(args):
    work_dir = Path(args.work_dir)
    train_x, train_y, train_base, train_side, train_game, train_valid = load_split(work_dir / "train.npz")
    val_x, val_y, val_base, val_side, _val_game, val_valid = load_split(work_dir / "validation.npz")
    test_x, test_y, test_base, test_side, test_game, test_valid = load_split(work_dir / "test.npz")

    report = {
        "schema_version": "transition-aware-current-frame-v1",
        "input_contract": "x[:, -1, :] plus current base_mode one-hot; no future targets in X",
        "method": "per-horizon balanced logistic transition classifier plus balanced destination classifier",
        "validation_policy": "threshold maximizes macro-F1 subject to accuracy >= persistence - 0.02; threshold grid 0.00..1.00 step 0.02",
        "split_sizes": {"train": int(len(train_y)), "validation": int(len(val_y)), "test": int(len(test_y))},
        "modes": list(MODES), "by_horizon": {},
    }
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Read the pre-existing held-out temporal hybrid exactly once for flip diagnostics.
    hybrid_pred, hybrid_device = hybrid_test_predictions(
        work_dir / "test.npz", work_dir / "temporal_hybrid_v1.pt", args.device)
    report["hybrid_inference_device"] = hybrid_device

    for h_idx, horizon in enumerate(HORIZONS):
        tr_mask = train_valid & (train_y[:, h_idx] >= 0)
        va_mask = val_valid & (val_y[:, h_idx] >= 0)
        te_mask = test_valid & (test_y[:, h_idx] >= 0)
        Xt = train_x[tr_mask]
        yt = train_y[tr_mask, h_idx]
        bt = train_base[tr_mask]
        Xv = val_x[va_mask]
        yv = val_y[va_mask, h_idx]
        bv = val_base[va_mask]
        Xs = test_x[te_mask]
        ys = test_y[te_mask, h_idx]
        bs = test_base[te_mask]
        sides = test_side[te_mask]
        changed_train = yt != bt
        changed_val = yv != bv

        stay = transition_model()
        stay.fit(Xt, changed_train.astype(np.int8))
        dest = destination_model()
        dest.fit(Xt[changed_train], yt[changed_train])
        p_val = stay.predict_proba(Xv)[:, list(stay.classes_).index(1)]
        dest_val = dest.predict(Xv)
        threshold, accuracy_guard_met, validation_selected = choose_threshold(
            yv, bv, p_val, dest_val)

        # No training, threshold tuning, or model selection uses test targets.
        p_test = stay.predict_proba(Xs)[:, list(stay.classes_).index(1)]
        dest_test = dest.predict(Xs)
        selected_test = predictions_from_parts(bs, p_test, dest_test, threshold)
        persistence_test = bs.copy()
        hybrid_test = hybrid_pred[horizon][te_mask]

        report["by_horizon"][f"{horizon}s"] = {
            "threshold": threshold,
            "validation_accuracy_guard_met": bool(accuracy_guard_met),
            "validation_selected": validation_selected,
            "validation_n": int(len(yv)),
            "test": {
                "transition_aware": side_metrics(ys, selected_test, bs, sides),
                "persistence": side_metrics(ys, persistence_test, bs, sides),
                "temporal_hybrid": side_metrics(ys, hybrid_test, bs, sides),
                "test_target_distribution": {
                    MODES[i]: int(np.sum(ys == i)) for i in range(len(MODES))
                },
                "test_current_to_target_transition_rate": float(np.mean(ys != bs)),
            },
        }
        del stay, dest, Xt, yt, bt, Xv, yv, bv, Xs, ys, bs
        gc.collect()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "device": hybrid_device,
        "by_horizon": {
                k: {
                    name: {
                    "accuracy": round(v["red_blue_equal_weight"]["accuracy"], 4),
                    "macro_f1": round(v["red_blue_equal_weight"]["macro_f1"], 4),
                    "flip_recall": round(v["red_blue_equal_weight"]["flip_recall"], 4),
                }
                for name in ("transition_aware", "persistence", "temporal_hybrid")
                for v in (value["test"][name],)
            }
            for k, value in report["by_horizon"].items()
        },
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="out_m2/temporal_hybrid_motion_death_aware")
    parser.add_argument("--out", default="out_m2/transition_aware_probe/transition_probe_report.json")
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
