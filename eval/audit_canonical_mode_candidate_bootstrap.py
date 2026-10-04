"""Score a frozen canonical-mode candidate with paired whole-game intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import Booster

from eval.evaluate_production_ensemble import _paired_game_bootstrap, _persistence_probs
from eval.train_canonical_mode_lightgbm import bias_prob, features, fused
from forecast.temporal import MODES


def audit(report_path: Path, *, root: Path, experiment_dir: Path,
          bootstrap_replicates: int = 2000) -> dict:
    training_report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    with np.load(Path(root) / "test.npz", allow_pickle=False) as data_file:
        data = {key: np.asarray(data_file[key]) for key in data_file.files}
    X = features(data)
    horizons = {}
    for index, horizon in enumerate((10, 30, 60, 120)):
        row = training_report["reports"][str(horizon)]
        chosen = row["validation"]["chosen"]
        model_path = Path(experiment_dir) / "models" / f"global_mode_{horizon}s.txt"
        model = Booster(model_file=str(model_path))
        probability = model.predict(X)
        base = np.asarray(data["base_mode"], dtype=np.int64)
        probability = bias_prob(
            fused(probability, base, chosen["alpha"]),
            tuple(chosen.get("class_logit_bias") or (0.0, 0.0, 0.0)))
        target = np.asarray(data["mode_y"][:, index], dtype=np.int64)
        baseline = _persistence_probs(base, len(MODES))
        horizons[str(horizon)] = _paired_game_bootstrap(
            target, probability, baseline, data["sides"], data["game_ids"], MODES,
            replicates=bootstrap_replicates, seed=20260930 + horizon)
    return {
        "schema_version": "canonical-mode-candidate-paired-bootstrap-v1",
        "source_training_report": str(Path(report_path)),
        "test_split": str(Path(root) / "test.npz"),
        "locked_test_games": int(len(np.unique(data["game_ids"]))),
        "selection": "five-fold whole-game OOF, Macro-F1 objective; frozen before test scoring",
        "comparison": "candidate versus persistence; production-minus-persistence deltas",
        "horizons": horizons,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--root", default="out_m2/temporal_forecast_canonical_side")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    result = audit(Path(args.report), root=Path(args.root),
                   experiment_dir=Path(args.experiment_dir),
                   bootstrap_replicates=args.bootstrap_replicates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        horizon: {
            metric: values["metrics"][metric]["ci95"]
            for metric in ("accuracy", "macro_f1", "brier")
        }
        for horizon, values in result["horizons"].items()
    }
    print(json.dumps({"output": str(output.resolve()), "ci95": compact}, ensure_ascii=False))


if __name__ == "__main__":
    main()
