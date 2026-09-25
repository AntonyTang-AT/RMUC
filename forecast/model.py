"""Small causal multi-horizon temporal forecaster."""

from __future__ import annotations

import torch
from torch import nn

from .temporal import FEATURE_DIM, HORIZONS, MODES, REGION_LABELS, TRENDS


# Near forecasts preserve high-frequency snapshots; distant forecasts use
# progressively coarser temporal bins.  The compact dataset keeps one causal
# window, and this profile changes only the view consumed by each head.
HORIZON_POOL = {10: 1, 30: 2, 60: 3, 120: 6}


class TemporalForecastNet(nn.Module):
    """Shared GRU encoder with typed heads for every forecast horizon."""

    def __init__(self, input_dim=FEATURE_DIM, hidden=128, layers=2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden), nn.LayerNorm(self.hidden), nn.GELU()
        )
        self.encoder = nn.GRU(self.hidden, self.hidden, num_layers=self.layers,
                              batch_first=True, dropout=0.1 if self.layers > 1 else 0.0)
        self.norm = nn.LayerNorm(self.hidden)
        self.heads = nn.ModuleDict({
            str(h): nn.ModuleDict({
                "global_mode": nn.Linear(self.hidden, len(MODES)),
                "opponent_trend": nn.Linear(self.hidden, len(TRENDS)),
                "enemy_region": nn.Linear(self.hidden, len(REGION_LABELS)),
            }) for h in HORIZONS
        })

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected [batch,time,{self.input_dim}], got {tuple(x.shape)}")
        projected = self.input_proj(x)
        outputs = {}
        for horizon in HORIZONS:
            factor = HORIZON_POOL[horizon]
            if factor > 1:
                usable = (projected.shape[1] // factor) * factor
                if usable == 0:
                    view = projected[:, -1:]
                else:
                    view = projected[:, -usable:]
                    view = view.reshape(view.shape[0], usable // factor, factor, view.shape[-1]).mean(dim=2)
            else:
                view = projected
            encoded, _ = self.encoder(view)
            pooled = self.norm(encoded[:, -1])
            outputs[horizon] = {
                task: head(pooled) for task, head in self.heads[str(horizon)].items()
            }
        return outputs
