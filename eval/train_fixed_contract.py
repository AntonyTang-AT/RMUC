"""Train the fixed-contract hybrid forecaster on existing compact NPZ files.

This entry point intentionally does not rebuild labels or touch the test split
during training.  Validation selects the checkpoint; the test archive is read
once afterwards for the comparable report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.contract import contract_metadata
from forecast.hybrid_train import train_hybrid_model
from eval.run_temporal_forecast import _evaluate


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", required=True)
    p.add_argument("--validation-npz", required=True)
    p.add_argument("--test-npz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--context", type=int, default=40)
    p.add_argument("--max-context", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--evaluate-test", action="store_true",
                   help="opt in to a separate test evaluation; training never reads test by default")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "temporal_hybrid_contract_v1.pt"
    train = train_hybrid_model(
        args.train_npz, args.validation_npz, model_path,
        epochs=args.epochs, batch_size=args.batch_size,
        hidden=args.hidden, layers=args.layers, heads=args.heads,
        dropout=args.dropout, max_context=args.max_context,
        lr=args.lr, weight_decay=args.weight_decay, device=args.device,
        seed=args.seed, loss_profile="evidence_adaptive_v1",
        patience=args.patience, blend_alpha=1.0, horizon_fusion=True,
        architecture="hierarchical_expert_v1",
    )
    report_path = out / "temporal_hybrid_contract_report.json"
    report = {"training": train, "forecast_contract": contract_metadata(),
              "evaluation_protocol": {"test_read_after_training": False}}
    if args.evaluate_test:
        report.update(_evaluate(model_path, args.test_npz, report_path,
                                device=args.device, batch_size=args.batch_size,
                                model_name="temporal_hybrid_contract_v1"))
    report["forecast_contract"] = contract_metadata()
    report["training"] = train
    report["evaluation_protocol"] = {
        "checkpoint_selected_on": "validation_loss",
        "test_read_after_training": bool(args.evaluate_test),
        "baselines": ["persistence_v1", "constant_velocity", "legal_zone_transition"],
        "aggregation": "red_blue_equal_within_match_then_match_equal",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "model": str(model_path), "report": str(report_path),
        "best_val_loss": train.get("best_val_loss"),
        "device": train.get("device"),
        "history": train.get("history", []),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
