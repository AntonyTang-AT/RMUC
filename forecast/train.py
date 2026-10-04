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
from .contract import contract_metadata
from worldstate.canonical_zone import CANONICAL_ZONE_DIM
from worldstate.functional_zone import FUNCTIONAL_ZONE_LABELS


TASKS = {
    "global_mode": ("mode_y", len(MODES)),
    "opponent_trend": ("trend_y", len(TRENDS)),
    "enemy_region": ("region_y", len(REGION_LABELS)),
}

# The forecast is a multi-objective product.  A raw mean treats a noisy
# 120-second region guess and a safety-critical 10-second presence/position
# estimate as equally valuable.  These weights make that business priority
# explicit while keeping the old loss available as ``baseline``.
LOSS_PROFILES = {
    'baseline': {
        'task': {'global_mode': 1.0, 'opponent_trend': 1.0, 'enemy_region': 1.0},
        'horizon': {10: 1.0, 30: 1.0, 60: 1.0, 120: 1.0},
        'horizon_smoothing': {10: 0.0, 30: 0.0, 60: 0.0, 120: 0.0},
        'task_smoothing': {'global_mode': 0.0, 'opponent_trend': 0.0, 'enemy_region': 0.0},
        'entity': {'presence': 1.0, 'position': 1.0, 'hp': 1.0,
                   'count': 1.0, 'zone': 1.0, 'canonical_zone': 0.15},
    },
    'business_weighted': {
        # Global mode and short-horizon trend drive the fast decision route.
        'task': {'global_mode': 1.8, 'opponent_trend': 1.35, 'enemy_region': 0.75},
        # Near forecasts are both more actionable and more reliably labelled.
        'horizon': {10: 1.45, 30: 1.15, 60: 0.85, 120: 0.60},
        'horizon_smoothing': {10: 0.03, 30: 0.06, 60: 0.10, 120: 0.15},
        'task_smoothing': {'global_mode': 1.0, 'opponent_trend': 1.15, 'enemy_region': 0.45},
        # Presence and position are safety/action fields.  HP is less reliable
        # for radar-only opponents; count and categorical zone are secondary.
        'entity': {'presence': 2.0, 'position': 1.35, 'hp': 0.55,
                   'count': 0.30, 'zone': 0.35, 'canonical_zone': 0.15},
    },
    # Contract-v1 profile derived from the field/horizon audit.  It gives
    # short-horizon entity position/presence and the 30s trend head more
    # capacity, while treating 120s region detail as a low-weight summary.
    'evidence_adaptive_v1': {
        'task': {'global_mode': 1.65, 'opponent_trend': 1.55, 'enemy_region': 0.55},
        'horizon': {10: 1.40, 30: 1.25, 60: 0.85, 120: 0.45},
        'horizon_smoothing': {10: 0.02, 30: 0.04, 60: 0.08, 120: 0.12},
        'task_smoothing': {'global_mode': 0.80, 'opponent_trend': 1.00, 'enemy_region': 0.50},
        'entity': {'presence': 2.20, 'position': 1.60, 'hp': 0.45,
                   'count': 0.25, 'zone': 0.50, 'canonical_zone': 0.10},
        'class_weight_cap': 4.0,
    },
}


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_ce(logits, target, class_weight=None, sample_weight=None,
               label_smoothing=0.0):
    valid = target >= 0
    if not torch.any(valid):
        return logits.sum() * 0.0
    loss = torch.nn.functional.cross_entropy(
        logits[valid], target[valid], weight=class_weight,
        reduction='none', label_smoothing=float(label_smoothing))
    if sample_weight is None:
        return loss.mean()
    weights = torch.clamp(sample_weight[valid].to(dtype=loss.dtype), min=0.05)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def _balanced_class_weight(values, class_count, cap=3.0):
    """Inverse-square-root class weights, clipped to avoid rare-class noise."""
    values = np.asarray(values).reshape(-1)
    counts = np.bincount(values[values >= 0].astype(np.int64),
                         minlength=int(class_count)).astype(np.float64)
    if not np.any(counts):
        return np.ones(int(class_count), dtype=np.float32)
    safe = np.maximum(counts, 1.0)
    weights = np.sqrt(float(np.sum(counts)) / (len(counts) * safe))
    weights = np.clip(weights, 1.0 / cap, cap)
    weights /= max(float(np.mean(weights)), 1e-8)
    return weights.astype(np.float32)


def build_loss_config(train, profile='business_weighted', device='cpu'):
    """Build reproducible importance and imbalance weights from train only."""
    if profile not in LOSS_PROFILES:
        raise ValueError(f'unknown loss profile: {profile}')
    base = LOSS_PROFILES[profile]
    cfg = {
        'profile': profile,
        'task': dict(base['task']), 'horizon': dict(base['horizon']),
        'horizon_smoothing': dict(base.get('horizon_smoothing', {})),
        'task_smoothing': dict(base.get('task_smoothing', {})),
        'entity': dict(base['entity']), 'class_weights': {},
    }
    dev = torch.device(device)
    class_weight_cap = float(base.get('class_weight_cap', 3.0))
    for task, (key, class_count) in TASKS.items():
        values = train[key]
        for h_idx, horizon in enumerate(HORIZONS):
            cfg['class_weights'][(task, int(horizon))] = torch.as_tensor(
                _balanced_class_weight(values[:, h_idx], class_count, cap=class_weight_cap),
                dtype=torch.float32, device=dev)
    if 'entity_zone_y' in train:
        zone_values = train['entity_zone_y']
        zone_count = len(FUNCTIONAL_ZONE_LABELS)
        for h_idx, horizon in enumerate(HORIZONS):
            cfg['class_weights'][('entity_zone', int(horizon))] = torch.as_tensor(
                _balanced_class_weight(zone_values[:, h_idx], zone_count, cap=class_weight_cap),
                dtype=torch.float32, device=dev)
    if 'entity_canonical_zone_y' in train:
        zone_values = train['entity_canonical_zone_y']
        for h_idx, horizon in enumerate(HORIZONS):
            cfg['class_weights'][('entity_canonical_zone', int(horizon))] = torch.as_tensor(
                _balanced_class_weight(zone_values[:, h_idx], CANONICAL_ZONE_DIM, cap=class_weight_cap),
                dtype=torch.float32, device=dev)
    # Checkpoint JSON cannot serialize tuple keys/tensors; this is filled by
    # ``serializable_loss_config`` when saving metadata.
    return cfg


def serializable_loss_config(config):
    return {
        'profile': config['profile'], 'task': config['task'],
        'horizon': {str(k): v for k, v in config['horizon'].items()},
        'horizon_smoothing': {str(k): v for k, v in config.get('horizon_smoothing', {}).items()},
        'task_smoothing': config.get('task_smoothing', {}),
        'entity': config['entity'],
        'contract': contract_metadata(),
        'class_weights': {
            f'{task}@{horizon}s': [float(v) for v in weight.detach().cpu().tolist()]
            for (task, horizon), weight in config.get('class_weights', {}).items()
        },
    }


def loss_for_outputs(outputs, batch_targets, loss_config=None):
    config = loss_config or LOSS_PROFILES['baseline']
    class_weights = config.get('class_weights', {})
    task_weights = config.get('task', {})
    horizon_weights = config.get('horizon', {})
    horizon_smoothing = config.get('horizon_smoothing', {})
    task_smoothing = config.get('task_smoothing', {})
    entity_weights = config.get('entity', {})
    quality = batch_targets.get('label_quality')
    losses = []
    for h_idx, horizon in enumerate(HORIZONS):
        horizon_weight = float(horizon_weights.get(horizon, 1.0))
        for task, (key, _) in TASKS.items():
            task_index = {'global_mode': 0, 'opponent_trend': 1,
                          'enemy_region': 2}[task]
            sample_quality = (quality[:, h_idx, task_index]
                              if quality is not None else None)
            smoothing = min(0.25, float(horizon_smoothing.get(horizon, 0.0)) *
                            float(task_smoothing.get(task, 0.0)))
            losses.append(horizon_weight * float(task_weights.get(task, 1.0)) *
                          _masked_ce(outputs[horizon][task], batch_targets[key][:, h_idx],
                                     class_weights.get((task, int(horizon))),
                                     sample_weight=sample_quality,
                                     label_smoothing=smoothing))
        if "entity_y" in batch_targets and "entity_state" in outputs[horizon]:
            target = batch_targets["entity_y"][:, h_idx]
            pred = outputs[horizon]["entity_state"]
            valid = target[..., 0] >= 0
            if torch.any(valid):
                losses.append(horizon_weight * float(entity_weights.get('presence', 1.0)) *
                              torch.nn.functional.binary_cross_entropy_with_logits(
                                  pred[..., 0][valid], target[..., 0][valid]))
                present = valid & (target[..., 0] > 0.5)
                if torch.any(present):
                    # Position is actionable; HP and count are less reliable
                    # or less important and therefore receive separate weights.
                    losses.append(horizon_weight * float(entity_weights.get('position', 1.0)) *
                                  torch.nn.functional.smooth_l1_loss(
                                      torch.sigmoid(pred[..., 1:3][present]), target[..., 1:3][present]))
                    losses.append(horizon_weight * float(entity_weights.get('hp', 1.0)) *
                                  torch.nn.functional.smooth_l1_loss(
                                      torch.sigmoid(pred[..., 3][present]), target[..., 3][present]))
                    losses.append(horizon_weight * float(entity_weights.get('count', 1.0)) *
                                  torch.nn.functional.smooth_l1_loss(
                                      torch.sigmoid(pred[..., 4][present]), target[..., 4][present]))
                # The hierarchical model exposes an explicit death-risk head;
                # train it on the same censored-safe presence target without
                # changing the legacy entity_state shape.
                if "entity_death_risk" in outputs[horizon]:
                    risk = outputs[horizon]["entity_death_risk"]
                    losses.append(horizon_weight * float(entity_weights.get('presence', 1.0)) * 0.35 *
                                  torch.nn.functional.binary_cross_entropy_with_logits(
                                      risk[valid], 1.0 - target[..., 0][valid]))
                if "entity_uncertainty" in outputs[horizon] and torch.any(present):
                    logvar = outputs[horizon]["entity_uncertainty"]
                    err = (torch.sigmoid(pred[..., 1:3][present]) - target[..., 1:3][present]) ** 2
                    lv = torch.clamp(logvar[present], -4.0, 3.0)
                    # Small auxiliary Gaussian NLL encourages the uncertainty
                    # output to grow near ambiguous long-horizon positions.
                    losses.append(horizon_weight * float(entity_weights.get('position', 1.0)) * 0.08 *
                                  (0.5 * (torch.exp(-lv) * err + lv)).mean())
        if "entity_zone_y" in batch_targets and "entity_zone_state" in outputs[horizon]:
            zone_target = batch_targets["entity_zone_y"][:, h_idx]
            zone_pred = outputs[horizon]["entity_zone_state"]
            valid_zone = zone_target >= 0
            if torch.any(valid_zone):
                losses.append(horizon_weight * float(entity_weights.get('zone', 1.0)) *
                              torch.nn.functional.cross_entropy(
                                  zone_pred[valid_zone], zone_target[valid_zone],
                                  weight=class_weights.get(('entity_zone', int(horizon)))))
        if "entity_canonical_zone_y" in batch_targets and "entity_canonical_zone_state" in outputs[horizon]:
            zone_target = batch_targets["entity_canonical_zone_y"][:, h_idx]
            zone_pred = outputs[horizon]["entity_canonical_zone_state"]
            valid_zone = zone_target >= 0
            if torch.any(valid_zone):
                losses.append(horizon_weight * float(entity_weights.get('canonical_zone', 0.15)) *
                              torch.nn.functional.cross_entropy(
                                  zone_pred[valid_zone], zone_target[valid_zone],
                                  weight=class_weights.get(('entity_canonical_zone', int(horizon)))))
    return sum(losses) / max(1, len(losses))


def train_model(train_npz, val_npz, out_path, *, epochs=5, batch_size=256,
                hidden=128, layers=2, lr=2e-3, weight_decay=1e-4,
                device="auto", seed=0, loss_profile='business_weighted',
                resume_from=None, architecture="gru", nhead=4,
                dropout=0.15, max_seq_len=256):
    set_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    train = load_compact(train_npz)
    val = load_compact(val_npz)
    def loader(data, shuffle):
        keys = ["x"] + [key for key, _ in TASKS.values()]
        if "entity_y" in data:
            keys.append("entity_y")
        if "entity_zone_y" in data:
            keys.append("entity_zone_y")
        if "entity_canonical_zone_y" in data:
            keys.append("entity_canonical_zone_y")
        if "label_quality" in data:
            keys.append("label_quality")
        tensors = [torch.from_numpy(data[key]) for key in keys]
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                          pin_memory=(dev.type == "cuda"), num_workers=0)
    entity_slots = int(train["entity_y"].shape[2]) if "entity_y" in train else 0
    zone_classes = len(FUNCTIONAL_ZONE_LABELS) if "entity_zone_y" in train else 0
    canonical_zone_classes = CANONICAL_ZONE_DIM if "entity_canonical_zone_y" in train else 0
    model_kwargs = dict(input_dim=int(train["feature_dim"][0]), hidden=hidden,
                        layers=layers, entity_slots=entity_slots,
                        zone_classes=zone_classes,
                        canonical_zone_classes=canonical_zone_classes)
    if architecture in ("hybrid", "tcn_transformer_hybrid_v1"):
        from .hybrid_model import HybridTemporalForecastNet
        model = HybridTemporalForecastNet(**model_kwargs, nhead=nhead,
                                         dropout=dropout,
                                         max_seq_len=max_seq_len).to(dev)
        architecture_name = model.architecture_name
    elif architecture in ("gru", "temporal_gru_v1"):
        model = TemporalForecastNet(**model_kwargs).to(dev)
        architecture_name = "temporal_gru_v1"
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    prior_history = []
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=dev, weights_only=False)
        expected = {
            "input_dim": model.input_dim, "hidden": model.hidden,
            "layers": model.layers, "entity_slots": model.entity_slots,
            "zone_classes": model.zone_classes,
            "canonical_zone_classes": model.canonical_zone_classes,
        }
        if checkpoint.get("model_type", checkpoint.get("architecture", "temporal_gru_v1")) != architecture_name:
            raise ValueError("resume checkpoint architecture does not match requested architecture")
        for key, value in expected.items():
            if int(checkpoint.get(key, -1)) != int(value):
                raise ValueError(f"resume checkpoint {key}={checkpoint.get(key)} does not match {value}")
        model.load_state_dict(checkpoint["model"])
        prior_history = list(checkpoint.get("history", []))
    loss_config = build_loss_config(train, loss_profile, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = loader(train, True)
    val_loader = loader(val, False)
    best = min((float(item.get("val_loss", float("inf"))) for item in prior_history),
               default=float("inf"))
    history = prior_history[:]
    start_epoch = len(history) + 1
    saved = False
    for epoch in range(start_epoch, start_epoch + int(epochs)):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x = batch[0].to(dev, non_blocking=True)
            targets = {key: batch[idx + 1].to(dev, non_blocking=True)
                       for idx, (key, _) in enumerate(TASKS.values())}
            offset = 1 + len(TASKS)
            if "entity_y" in train:
                targets["entity_y"] = batch[offset].to(dev, non_blocking=True)
                offset += 1
            if "entity_zone_y" in train:
                targets["entity_zone_y"] = batch[offset].to(dev, non_blocking=True)
                offset += 1
            if "entity_canonical_zone_y" in train:
                targets["entity_canonical_zone_y"] = batch[offset].to(dev, non_blocking=True)
                offset += 1
            if "label_quality" in train:
                targets["label_quality"] = batch[offset].to(dev, non_blocking=True)
            loss = loss_for_outputs(model(x), targets, loss_config)
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
                offset = 1 + len(TASKS)
                if "entity_y" in val:
                    targets["entity_y"] = batch[offset].to(dev, non_blocking=True)
                    offset += 1
                if "entity_zone_y" in val:
                    targets["entity_zone_y"] = batch[offset].to(dev, non_blocking=True)
                    offset += 1
                if "entity_canonical_zone_y" in val:
                    targets["entity_canonical_zone_y"] = batch[offset].to(dev, non_blocking=True)
                    offset += 1
                if "label_quality" in val:
                    targets["label_quality"] = batch[offset].to(dev, non_blocking=True)
                value = loss_for_outputs(model(x), targets, loss_config)
                val_loss += float(value.item()) * len(x)
                n_val += len(x)
        record = {"epoch": epoch, "train_loss": train_loss / max(1, n_train),
                  "val_loss": val_loss / max(1, n_val)}
        history.append(record)
        if record["val_loss"] < best:
            best = record["val_loss"]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "model_type": architecture_name,
                        "architecture": architecture_name, "input_dim": model.input_dim,
                        "hidden": model.hidden, "layers": model.layers,
                        "entity_slots": model.entity_slots,
                        "zone_classes": model.zone_classes,
                        "canonical_zone_classes": model.canonical_zone_classes,
                        "nhead": int(getattr(model, "nhead", nhead)),
                        "dropout": float(getattr(model, "dropout_rate", dropout)),
                        "max_seq_len": int(getattr(model, "max_seq_len", max_seq_len)),
                        "loss_profile": loss_profile,
                        "loss_config": serializable_loss_config(loss_config),
                        "history": history, "seed": seed}, out_path)
            saved = True
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if not saved:
        # Preserve a usable artifact even when the resumed run does not beat
        # the previous validation minimum.
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "model_type": architecture_name,
                    "architecture": architecture_name, "input_dim": model.input_dim,
                    "hidden": model.hidden, "layers": model.layers,
                    "entity_slots": model.entity_slots,
                    "zone_classes": model.zone_classes,
                    "canonical_zone_classes": model.canonical_zone_classes,
                    "nhead": int(getattr(model, "nhead", nhead)),
                    "dropout": float(getattr(model, "dropout_rate", dropout)),
                    "max_seq_len": int(getattr(model, "max_seq_len", max_seq_len)),
                    "loss_profile": loss_profile,
                    "loss_config": serializable_loss_config(loss_config),
                    "history": history, "seed": seed}, out_path)
    else:
        # The best-weight file may have been written before the final epoch.
        # Refresh metadata without replacing those best weights so downstream
        # reports retain the complete continuation history.
        checkpoint = torch.load(out_path, map_location="cpu", weights_only=False)
        checkpoint["history"] = history
        checkpoint["completed_epochs"] = int(history[-1]["epoch"]) if history else 0
        checkpoint["resumed_from"] = str(resume_from) if resume_from else None
        torch.save(checkpoint, out_path)
    return {"path": str(out_path), "device": str(dev), "best_val_loss": best,
            "history": history, "loss_profile": loss_profile,
            "architecture": architecture_name,
            "resumed_from": str(resume_from) if resume_from else None,
            "train_examples": len(train["x"]),
            "validation_examples": len(val["x"])}


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") in ("hierarchical_expert_v1", "hierarchical_expert_v2_kinematic",
                                         "hierarchical_expert_v3_kinematic", "hierarchical"):
        from .hierarchical_model import HierarchicalExpertForecastNet
        model = HierarchicalExpertForecastNet(
            input_dim=checkpoint["input_dim"], hidden=checkpoint["hidden"],
            layers=checkpoint.get("layers", 3), nhead=checkpoint.get("nhead", checkpoint.get("heads", 4)),
            dropout=checkpoint.get("dropout", 0.15),
            max_seq_len=checkpoint.get("max_seq_len", checkpoint.get("max_context", 256)),
            horizon_fusion=checkpoint.get("horizon_fusion", False),
            entity_slots=checkpoint.get("entity_slots", 0),
            zone_classes=checkpoint.get("zone_classes", 0),
            canonical_zone_classes=checkpoint.get("canonical_zone_classes", 0))
    elif checkpoint.get("model_type") in ("hybrid_tcn_transformer_v1", "tcn_transformer_hybrid_v1"):
        from .hybrid_model import HybridTemporalForecastNet
        model = HybridTemporalForecastNet(
            input_dim=checkpoint["input_dim"], hidden=checkpoint["hidden"],
            layers=checkpoint.get("layers", 3), nhead=checkpoint.get("nhead", checkpoint.get("heads", 4)),
            tcn_kernel=checkpoint.get("tcn_kernel", 3),
            transformer_layers=checkpoint.get("transformer_layers", checkpoint.get("layers", 3)),
            dropout=checkpoint.get("dropout", 0.15),
            max_seq_len=checkpoint.get("max_seq_len", checkpoint.get("max_context", 256)),
            horizon_fusion=checkpoint.get("horizon_fusion", False),
            entity_slots=checkpoint.get("entity_slots", 0),
            zone_classes=checkpoint.get("zone_classes", 0),
            canonical_zone_classes=checkpoint.get("canonical_zone_classes", 0))
    else:
        model = TemporalForecastNet(input_dim=checkpoint["input_dim"],
                                    hidden=checkpoint["hidden"], layers=checkpoint["layers"],
                                    entity_slots=checkpoint.get("entity_slots", 0),
                                    zone_classes=checkpoint.get("zone_classes", 0),
                                    canonical_zone_classes=checkpoint.get("canonical_zone_classes", 0))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
