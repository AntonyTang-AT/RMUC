"""Train and evaluate the causal TCN + Transformer forecast candidate.

Example::

    python eval/train_hybrid_forecast.py \
      --dataset-dir out_m2/forecast_dataset \
      --work-dir out_m2/temporal_hybrid_context40 \
      --context 40 --epochs 10

The script creates context-specific compact NPZ files, trains with early
stopping, and writes a ledger report containing the persistence baseline and
the hybrid model under the same task/horizon contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.temporal import prepare_compact_split
from forecast.hybrid_train import train_hybrid_model
from eval.run_temporal_forecast import _evaluate
from forecast.contract import contract_metadata


def _compatible(path: Path, context: int, feature_schema: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_dim = 329 if feature_schema == "motion" else 14
            return ("x" in data.files and int(data["x"].shape[1]) == int(context)
                    and int(data["x"].shape[2]) == expected_dim
                    and "entity_canonical_zone_y" in data.files)
    except (OSError, ValueError, KeyError, IndexError):
        return False


def run(args):
    dataset_dir = Path(args.dataset_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    compact = {}
    for split in ("train", "validation", "test"):
        target = work / f"{split}.npz"
        if not _compatible(target, args.context, args.feature_schema):
            source = dataset_dir / f"{split}.jsonl.gz"
            if not source.exists():
                raise FileNotFoundError(source)
            print(prepare_compact_split(source, target, args.context,
                                        feature_schema=args.feature_schema), flush=True)
        compact[split] = target
    model_path = work / "temporal_hybrid_v1.pt"
    train_result = train_hybrid_model(
        compact["train"], compact["validation"], model_path,
        epochs=args.epochs, batch_size=args.batch_size,
        hidden=args.hidden, layers=args.layers, heads=args.heads,
        dropout=args.dropout, max_context=max(args.max_context, args.context),
        lr=args.lr, weight_decay=args.weight_decay, device=args.device,
        seed=args.seed, loss_profile=args.loss_profile, patience=args.patience,
        resume_from=args.resume_from, blend_alpha=args.blend_alpha,
        horizon_fusion=args.horizon_fusion)
    report_path = work / "temporal_hybrid_report.json"
    report = _evaluate(model_path, compact["test"], report_path,
                       device=args.device, batch_size=args.batch_size,
                       model_name="temporal_hybrid_v1")
    checkpoint = __import__("torch").load(model_path, map_location="cpu", weights_only=False)
    report["persistence_fusion"] = checkpoint.get("persistence_fusion", {
        "enabled": True, "method": "probability_linear_blend",
        "model_weight": args.blend_alpha,
        "persistence_weight": 1.0 - args.blend_alpha,
    })
    report["training"] = train_result
    report["forecast_contract"] = contract_metadata()
    report["data_contract"] = {"context": args.context, "train_npz": str(compact["train"]),
                               "validation_npz": str(compact["validation"]),
                               "test_npz": str(compact["test"])}
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    summary = {"model": str(model_path), "report": str(report_path),
               "models": list(report.get("models", {})),
               "weighted_acceptance": report.get("weighted_acceptance", {}),
               "best_val_loss": train_result.get("best_val_loss")}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="out_m2/forecast_dataset")
    parser.add_argument("--work-dir", default="out_m2/temporal_hybrid_context40")
    parser.add_argument("--context", type=int, default=40)
    parser.add_argument("--feature-schema", default="legacy",
                        choices=("legacy", "motion"),
                        help="feature contract used to build compact train/validation/test arrays")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-profile", default="evidence_adaptive_v1",
                        choices=("baseline", "business_weighted", "evidence_adaptive_v1"))
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--blend-alpha", type=float, default=0.70)
    parser.add_argument("--horizon-fusion", action="store_true",
                        help="learn horizon-specific recent-vs-smoothed context fusion")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
