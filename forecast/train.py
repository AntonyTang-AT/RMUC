"""Training and evaluation helpers for the local temporal forecaster."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import TemporalForecastNet
from .temporal import HORIZONS, MODES, REGION_LABELS, TRENDS, load_compact


TASKS = {
    "global_mode": ("mode_y", len(MODES)),
    "opponent_trend": ("trend_y", len(TRENDS)),
    "enemy_region": ("region_y", len(REGION_LABELS)),
}


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_ce(logits, target):
    valid = target >= 0
    if not torch.any(valid):
        return logits.sum() * 0.0
    return torch.nn.functional.cross_entropy(logits[valid], target[valid])


def loss_for_outputs(outputs, batch_targets):
    losses = []
    for h_idx, horizon in enumerate(HORIZONS):
        for task, (key, _) in TASKS.items():
            losses.append(_masked_ce(outputs[horizon][task], batch_targets[key][:, h_idx]))
    return sum(losses) / max(1, len(losses))


def train_model(train_npz, val_npz, out_path, *, epochs=5, batch_size=256,
                hidden=128, layers=2, lr=2e-3, weight_decay=1e-4,
                device="auto", seed=0):
    set_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    train = load_compact(train_npz)
    val = load_compact(val_npz)
    def loader(data, shuffle):
        tensors = [torch.from_numpy(data["x"])] + [torch.from_numpy(data[key]) for key, _ in TASKS.values()]
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                          pin_memory=(dev.type == "cuda"), num_workers=0)
    model = TemporalForecastNet(input_dim=int(train["feature_dim"][0]), hidden=hidden, layers=layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = loader(train, True)
    val_loader = loader(val, False)
    best = float("inf")
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x = batch[0].to(dev, non_blocking=True)
            targets = {key: batch[idx + 1].to(dev, non_blocking=True)
                       for idx, (key, _) in enumerate(TASKS.values())}
            loss = loss_for_outputs(model(x), targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * len(x)
            n_train += len(x)
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(dev, non_blocking=True)
                targets = {key: batch[idx + 1].to(dev, non_blocking=True)
                           for idx, (key, _) in enumerate(TASKS.values())}
                value = loss_for_outputs(model(x), targets)
                val_loss += float(value.item()) * len(x)
                n_val += len(x)
        record = {"epoch": epoch, "train_loss": train_loss / max(1, n_train),
                  "val_loss": val_loss / max(1, n_val)}
        history.append(record)
        if record["val_loss"] < best:
            best = record["val_loss"]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "input_dim": model.input_dim,
                        "hidden": model.hidden, "layers": model.layers,
                        "history": history, "seed": seed}, out_path)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return {"path": str(out_path), "device": str(dev), "best_val_loss": best,
            "history": history, "train_examples": len(train["x"]),
            "validation_examples": len(val["x"])}


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TemporalForecastNet(input_dim=checkpoint["input_dim"],
                                hidden=checkpoint["hidden"], layers=checkpoint["layers"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint

