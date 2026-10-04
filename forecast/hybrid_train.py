"""Training entry point for the causal TCN + Transformer forecaster.

The trainer consumes the compact NPZ contract produced by
``forecast.temporal.prepare_compact_split``.  It intentionally reuses the
existing business weighted loss and ledger evaluator, which makes the hybrid
model directly comparable with the GRU and persistence baselines.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .hybrid_model import HybridTemporalForecastNet
from .temporal import HORIZONS, FEATURE_DIM, FEATURE_SCHEMA_VERSION, load_compact
from .train import (TASKS, build_loss_config, loss_for_outputs, set_seed,
                    serializable_loss_config)
from .contract import contract_metadata
from worldstate.canonical_zone import CANONICAL_ZONE_DIM
from worldstate.functional_zone import FUNCTIONAL_ZONE_LABELS


def _keys(data):
    keys = ["x"] + [key for key, _ in TASKS.values()]
    for key in ("entity_y", "entity_zone_y", "entity_canonical_zone_y", "label_quality"):
        if key in data:
            keys.append(key)
    return keys


def _loader(data, batch_size, shuffle, device):
    tensors = [torch.from_numpy(data[key]) for key in _keys(data)]
    return DataLoader(TensorDataset(*tensors), batch_size=int(batch_size), shuffle=shuffle,
                      pin_memory=(device.type == "cuda"), num_workers=0)


def _targets(batch, data, device):
    targets = {key: batch[idx + 1].to(device, non_blocking=True)
               for idx, (key, _) in enumerate(TASKS.values())}
    offset = 1 + len(TASKS)
    for key in ("entity_y", "entity_zone_y", "entity_canonical_zone_y", "label_quality"):
        if key in data:
            targets[key] = batch[offset].to(device, non_blocking=True)
            offset += 1
    return targets


def _model_from_data(data, *, hidden=192, layers=3, heads=4,
                     dropout=0.15, max_context=256, horizon_fusion=False,
                     architecture="tcn_transformer_hybrid_v1"):
    entity_slots = int(data["entity_y"].shape[2]) if "entity_y" in data else 0
    zone_classes = len(FUNCTIONAL_ZONE_LABELS) if "entity_zone_y" in data else 0
    canonical_zone_classes = CANONICAL_ZONE_DIM if "entity_canonical_zone_y" in data else 0
    model_cls = HybridTemporalForecastNet
    if architecture in ("hierarchical_expert_v1", "hierarchical_expert_v2_kinematic",
                        "hierarchical_expert_v3_kinematic", "hierarchical"):
        from .hierarchical_model import HierarchicalExpertForecastNet
        model_cls = HierarchicalExpertForecastNet
    return model_cls(
        input_dim=int(data["feature_dim"][0]), hidden=int(hidden), layers=int(layers),
        nhead=int(heads), dropout=float(dropout), max_seq_len=int(max_context),
        entity_slots=entity_slots, zone_classes=zone_classes,
        canonical_zone_classes=canonical_zone_classes,
        horizon_fusion=bool(horizon_fusion))


def train_hybrid_model(train_npz, val_npz, out_path, *, epochs=10, batch_size=256,
                       hidden=192, layers=3, heads=4, dropout=0.15,
                       max_context=256, lr=1e-3, weight_decay=1e-4,
                       device="auto", seed=0, loss_profile="business_weighted",
                       patience=3, resume_from=None, blend_alpha=0.70,
                       horizon_fusion=False, architecture="tcn_transformer_hybrid_v1"):
    """Train with early stopping and save a self-describing checkpoint."""
    set_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    train = load_compact(train_npz)
    val = load_compact(val_npz)
    if int(train["feature_dim"][0]) != int(val["feature_dim"][0]):
        raise ValueError("train/validation feature dimensions differ")
    model = _model_from_data(train, hidden=hidden, layers=layers, heads=heads,
                             dropout=dropout, max_context=max_context,
                             horizon_fusion=horizon_fusion,
                             architecture=architecture).to(dev)
    prior_history = []
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=dev, weights_only=False)
        for key, value in (("input_dim", model.input_dim), ("hidden", model.hidden),
                           ("layers", model.layers), ("nhead", model.nhead),
                           ("horizon_fusion", model.horizon_fusion),
                           ("entity_slots", model.entity_slots),
                           ("zone_classes", model.zone_classes),
                           ("canonical_zone_classes", model.canonical_zone_classes)):
            if int(checkpoint.get(key, -1)) != int(value):
                raise ValueError(f"resume checkpoint {key}={checkpoint.get(key)} does not match {value}")
        model.load_state_dict(checkpoint["model"])
        prior_history = list(checkpoint.get("history", []))
    cfg = build_loss_config(train, loss_profile, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=1, min_lr=1e-5)
    train_loader = _loader(train, batch_size, True, dev)
    val_loader = _loader(val, batch_size, False, dev)
    best = min((float(item.get("val_loss", float("inf"))) for item in prior_history), default=float("inf"))
    history = prior_history[:]
    start_epoch = len(history) + 1
    stale = 0
    out_path = Path(out_path)
    for epoch in range(start_epoch, start_epoch + int(epochs)):
        model.train()
        train_total = 0.0
        train_n = 0
        for batch in train_loader:
            x = batch[0].to(dev, non_blocking=True)
            loss = loss_for_outputs(model(x), _targets(batch, train, dev), cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_total += float(loss.item()) * len(x)
            train_n += len(x)
        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(dev, non_blocking=True)
                loss = loss_for_outputs(model(x), _targets(batch, val, dev), cfg)
                val_total += float(loss.item()) * len(x)
                val_n += len(x)
        record = {"epoch": epoch, "train_loss": train_total / max(1, train_n),
                  "val_loss": val_total / max(1, val_n),
                  "lr": float(opt.param_groups[0]["lr"])}
        history.append(record)
        scheduler.step(record["val_loss"])
        improved = record["val_loss"] < best - 1e-6
        if improved:
            best = record["val_loss"]
            stale = 0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_checkpoint(model, cfg, history, seed, blend_alpha), out_path)
        else:
            stale += 1
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if stale >= int(patience):
            break
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_checkpoint(model, cfg, history, seed, blend_alpha), out_path)
    else:
        saved = torch.load(out_path, map_location="cpu", weights_only=False)
        saved["history"] = history
        saved["completed_epochs"] = int(history[-1]["epoch"]) if history else 0
        saved["early_stopping"] = {"patience": int(patience), "best_val_loss": float(best)}
        torch.save(saved, out_path)
    return {"path": str(out_path), "device": str(dev), "best_val_loss": float(best),
            "history": history, "loss_profile": loss_profile,
            "train_examples": len(train["x"]), "validation_examples": len(val["x"]),
        "model_type": model.architecture_name}


def _checkpoint(model, cfg, history, seed, blend_alpha):
    return {
        "model": model.state_dict(), "model_type": model.architecture_name,
        "input_dim": model.input_dim, "hidden": model.hidden, "layers": model.layers,
        "heads": model.nhead, "nhead": model.nhead, "dropout": model.dropout_rate,
        "max_context": model.max_seq_len, "max_seq_len": model.max_seq_len,
        "tcn_kernel": model.tcn_kernel, "transformer_layers": model.transformer_layers,
        "horizon_fusion": model.horizon_fusion,
        "entity_slots": model.entity_slots, "zone_classes": model.zone_classes,
        "canonical_zone_classes": model.canonical_zone_classes,
        "loss_profile": cfg["profile"], "loss_config": serializable_loss_config(cfg),
        "forecast_contract": contract_metadata(),
        "history": history, "seed": seed,
        "persistence_fusion": {
            "enabled": True, "method": "probability_linear_blend",
            "model_weight": float(blend_alpha), "persistence_weight": float(1.0 - blend_alpha),
            "fallback": "persistence_when_confidence_or_feedback_is_low",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--out", required=True)
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
    parser.add_argument("--architecture", default="tcn_transformer_hybrid_v1",
                        choices=("tcn_transformer_hybrid_v1", "hierarchical_expert_v1",
                                 "hierarchical_expert_v2_kinematic",
                                 "hierarchical_expert_v3_kinematic"))
    args = parser.parse_args(argv)
    options = vars(args).copy()
    options["out_path"] = options.pop("out")
    result = train_hybrid_model(**options)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
