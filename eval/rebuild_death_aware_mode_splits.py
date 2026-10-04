"""Rebuild only global-mode labels over frozen train/validation split rows.

Features and non-global targets are copied verbatim from canonical-side NPZs.
No test path is accepted or opened.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from forecast.temporal import HORIZONS, MODES, baseline_labels, target_labels


CONTRACT = "relative_margin_v2_death_aware_hp_le_zero"


def _row_key(row):
    return int(row["game_id"]), str(row["side"]), round(float(row["issued_at"]), 5)


def rebuild(split, source_jsonl, source_npz, output_npz, output_jsonl):
    old = {k: v.copy() for k, v in np.load(source_npz, allow_pickle=False).items()}
    n = len(old["game_ids"])
    new_mode = np.full_like(old["mode_y"], -1)
    new_base = np.full_like(old["base_mode"], -1)
    cursor = 0
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source_jsonl, "rt", encoding="utf-8") as src, \
            gzip.open(output_jsonl, "wt", encoding="utf-8", compresslevel=5) as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if cursor >= n:
                raise ValueError(f"{split}: source has more rows than frozen compact NPZ")
            expected = (int(old["game_ids"][cursor]), str(old["sides"][cursor]),
                        round(float(old["issued_at"][cursor]), 5))
            if _row_key(row) != expected:
                raise ValueError(f"{split}: row key mismatch at {cursor}: {_row_key(row)} != {expected}")
            base = baseline_labels(row)["global_mode"]
            new_base[cursor] = MODES.index(base)
            horizon_labels = {}
            for j, h in enumerate(HORIZONS):
                labels = target_labels(row, h)
                if labels is not None:
                    mode = labels["global_mode"]
                    new_mode[cursor, j] = MODES.index(mode)
                    horizon_labels[str(h)] = mode
                else:
                    horizon_labels[str(h)] = None
            row["death_aware_global_mode_labels"] = {
                "contract": CONTRACT, "base_mode": base,
                "target_mode_by_horizon_s": horizon_labels,
            }
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            cursor += 1
    if cursor != n:
        raise ValueError(f"{split}: source rows {cursor} != frozen compact rows {n}")
    old_mode = old["mode_y"]
    old_base = old["base_mode"]
    arrays = old
    arrays["mode_y"] = new_mode
    arrays["base_mode"] = new_base
    arrays["global_mode_label_version"] = np.asarray([CONTRACT])
    np.savez_compressed(output_npz, **arrays)
    valid = (new_mode >= 0) & (old_mode >= 0)
    return {
        "split": split, "rows": n,
        "game_ids": sorted(int(v) for v in np.unique(old["game_ids"])),
        "row_key_parity": True, "features_and_non_global_labels": "copied verbatim",
        "valid_future_rows_by_horizon": {str(h): int((new_mode[:, j] >= 0).sum())
                                         for j, h in enumerate(HORIZONS)},
        "target_label_changes_vs_frozen_contract": {str(h): int(np.sum(valid[:, j] & (new_mode[:, j] != old_mode[:, j])))
                                                     for j, h in enumerate(HORIZONS)},
        "current_base_mode_changes_vs_frozen_contract": int(np.sum(new_base != old_base)),
        "npz": str(output_npz), "jsonl": str(output_jsonl),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="out_m2/forecast_dataset")
    p.add_argument("--frozen-dir", default="out_m2/temporal_forecast_canonical_side")
    p.add_argument("--output-dir", default="out_m2/temporal_forecast_death_aware_candidate")
    a = p.parse_args()
    dataset, frozen, out = map(Path, (a.dataset_dir, a.frozen_dir, a.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    results = [rebuild(s, dataset / f"{s}.jsonl.gz", frozen / f"{s}.npz",
                       out / f"{s}.npz", out / f"{s}.death_aware.jsonl.gz")
               for s in ("train", "validation")]
    report = {"schema": "death-aware-mode-only-rebuild-v1", "contract": CONTRACT,
              "train_and_validation_only": True, "test_access": False,
              "source_split_preserved": True, "splits": results}
    (out / "data_rebuild_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
