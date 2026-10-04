"""Exploratory causal early-style cluster prior screen; no model/default writes."""

from __future__ import annotations

import argparse
import collections
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "out_m2" / "temporal_forecast_canonical_side"
HORIZONS = (10, 30, 60, 120)
N_CLASSES = 3


def load_early_profiles(split: str) -> dict:
    path = DATA / f"{split}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        game_ids = arrays["game_ids"]
        sides = arrays["sides"]
        issued_at = arrays["issued_at"]
        base_mode = arrays["base_mode"]
        mode_y = arrays["mode_y"]
        trend_y = arrays["trend_y"]

        rows_by_game_side: dict[tuple[int, str], list[int]] = collections.defaultdict(list)
        for i, (game_id, side, t) in enumerate(zip(game_ids, sides, issued_at)):
            if t <= 30:
                side_key = "R" if ord(str(side)) == 0x7EA2 else "B"
                rows_by_game_side[(int(game_id), side_key)].append(i)
        selected = {
            key: max(indices, key=lambda i: issued_at[i])
            for key, indices in rows_by_game_side.items()
        }
        labels = {
            key: (int(base_mode[i]), mode_y[i].copy(), trend_y[i].copy())
            for key, i in selected.items()
        }

    row_bytes = 12 * 232 * np.dtype(np.float32).itemsize
    inputs: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(path) as archive, archive.open("x.npy") as data:
        version = np.lib.format.read_magic(data)
        shape, _fortran_order, _dtype = np.lib.format._read_array_header(data, version)
        if tuple(shape)[1:] != (12, 232):
            raise ValueError(f"unexpected x.npy shape: {shape}")
        data_offset = data.tell()
        for row_id in sorted(selected.values()):
            data.seek(data_offset + row_id * row_bytes)
            inputs[row_id] = np.frombuffer(data.read(row_bytes), dtype=np.float32).reshape(12, 232)

    profiles: dict[tuple[int, str], np.ndarray] = {}
    for game_side, row_id in selected.items():
        side = game_side[1]
        now, previous = inputs[row_id][-1], inputs[row_id][-4]
        slots = []
        for slot in range(8, 14):
            offset = slot * 14
            x, y, hp, has_position = now[offset : offset + 4]
            visible = float(has_position > 0.5)
            previous_slot = previous[offset : offset + 4]
            vx = (x - previous_slot[0]) / 3 if visible and previous_slot[3] > 0.5 else 0.0
            vy = (y - previous_slot[1]) / 3 if visible and previous_slot[3] > 0.5 else 0.0
            # Rotate blue coordinates by 180 degrees to align the two viewpoints.
            slots.append(
                [
                    visible,
                    float(x if side == "R" else 1 - x),
                    float(y if side == "R" else 1 - y),
                    float(hp),
                    float(vx if side == "R" else -vx),
                    float(vy if side == "R" else -vy),
                    float(now[offset + 6]),
                ]
            )
        slot_array = np.asarray(slots, dtype=np.float64)
        visible_slots = slot_array[:, 0] > 0.5
        visible = slot_array[visible_slots]
        summary = [
            visible_slots.mean(),
            visible[:, 1].mean() if len(visible) else 0.0,
            visible[:, 1].std() if len(visible) else 0.0,
            visible[:, 2].mean() if len(visible) else 0.0,
            visible[:, 3].mean() if len(visible) else 0.0,
        ]
        profiles[game_side] = np.r_[slot_array.ravel(), summary]

    return {"profiles": profiles, "labels": labels}


def classification_scores(y: np.ndarray, probabilities: np.ndarray) -> dict:
    valid = (y >= 0) & (y < probabilities.shape[1])
    y = y[valid]
    probabilities = probabilities[valid]
    prediction = probabilities.argmax(axis=1)
    confusion = np.zeros((probabilities.shape[1], probabilities.shape[1]), dtype=np.int64)
    for truth, guess in zip(y, prediction):
        confusion[truth, guess] += 1
    f1_values = []
    for c in range(probabilities.shape[1]):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        f1_values.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    one_hot = np.eye(probabilities.shape[1])[y]
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(prediction == y)),
        "macro_f1": float(np.mean(f1_values)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "logloss": float(-np.mean(np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-9)))),
    }


def run() -> dict:
    started = time.time()
    train = load_early_profiles("train")
    validation = load_early_profiles("validation")
    train_keys = list(train["profiles"])
    validation_keys = list(validation["profiles"])
    train_x = np.vstack([train["profiles"][key] for key in train_keys])
    validation_x = np.vstack([validation["profiles"][key] for key in validation_keys])
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-6] = 1.0
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale

    result = {}
    for task, target_index in (("global_mode", 1), ("opponent_trend", 2)):
        train_y = np.vstack([train["labels"][key][target_index] for key in train_keys])
        validation_y = np.vstack([validation["labels"][key][target_index] for key in validation_keys])
        current_mode = np.asarray([train["labels"][key][0] for key in train_keys])
        validation_current_mode = np.asarray([validation["labels"][key][0] for key in validation_keys])
        for horizon_index, horizon in enumerate(HORIZONS):
            target_train = train_y[:, horizon_index]
            target_validation = validation_y[:, horizon_index]
            valid_train = (target_train >= 0) & (target_train < N_CLASSES)
            prior_counts = np.bincount(target_train[valid_train], minlength=N_CLASSES) + 1
            global_probabilities = np.tile(prior_counts / prior_counts.sum(), (len(target_validation), 1))
            horizon_result = {"train_global_prior": classification_scores(target_validation, global_probabilities)}

            if task == "global_mode":
                persistence = np.full((len(target_validation), N_CLASSES), 0.01)
                valid_current = (validation_current_mode >= 0) & (validation_current_mode < N_CLASSES)
                persistence[np.arange(len(target_validation))[valid_current], validation_current_mode[valid_current]] = 0.98
                horizon_result["persistence"] = classification_scores(target_validation, persistence)

            for n_clusters in (2, 3):
                clusters = KMeans(n_clusters=n_clusters, n_init=20, random_state=29)
                train_cluster = clusters.fit_predict(train_x)
                validation_cluster = clusters.predict(validation_x)
                probabilities = np.empty((len(target_validation), N_CLASSES), dtype=np.float64)
                for cluster_id in range(n_clusters):
                    mask = (train_cluster == cluster_id) & valid_train
                    counts = np.bincount(target_train[mask], minlength=N_CLASSES) + 1
                    probabilities[validation_cluster == cluster_id] = counts / counts.sum()
                horizon_result[f"cluster_prior_K{n_clusters}"] = classification_scores(
                    target_validation, probabilities
                )
            result[f"{task}@{horizon}s"] = horizon_result

    return {
        "train_profile_count": len(train_keys),
        "validation_profile_count": len(validation_keys),
        "test_access": False,
        "horizon_metrics": result,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for JSON output")
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
