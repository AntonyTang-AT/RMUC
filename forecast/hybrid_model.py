"""Causal TCN + Transformer forecaster.

This module is the larger architecture candidate for the temporal forecast
benchmark.  It deliberately keeps the output contract of
``TemporalForecastNet`` so that the existing loss, decoder, and evaluation
code can be reused by selecting the model class in the training runner.

The two encoders have complementary jobs:

* the dilated causal TCN captures local motion changes and short transitions;
* the causal Transformer attends over the complete observed window for
  longer range formations and macro state changes.

No target or future frame is used.  The Transformer receives an upper
triangular causal mask and the convolution is left padded only.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .temporal import (FEATURE_DIM, FEATURES_PER_SLOT, HORIZONS, MODES, N_GLOBAL,
                       MOTION_FEATURES_PER_SLOT, MOTION_FEATURE_DIM, MOTION_N_GLOBAL,
                       REGION_LABELS, TRENDS)
from .entity import ENTITY_SLOTS, ENTITY_TARGET_DIM
from worldstate.functional_zone import FUNCTIONAL_ZONE_LABELS
from worldstate.canonical_zone import CANONICAL_ZONE_DIM


# Keep this in sync with forecast.model.  Pooling the most recent encoded
# bins makes the long horizon heads progressively less sensitive to one
# individual snapshot while preserving all observations causally.
HORIZON_POOL = {10: 1, 30: 2, 60: 3, 120: 6}


class _CausalConvBlock(nn.Module):
    """Residual left-padded dilated convolution over [B, T, C]."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def _conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        # F.pad pads the left side of the time dimension.  Trim defensively
        # because this keeps the block correct for kernel/dilation changes.
        y = conv(F.pad(x, (self.left_padding, 0)))
        return y[..., :x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self._conv(self.conv1, x.transpose(1, 2)).transpose(1, 2)
        y = self.norm1(y)
        y = self.dropout(F.gelu(y))
        y = self._conv(self.conv2, y.transpose(1, 2)).transpose(1, 2)
        y = self.norm2(y)
        y = self.dropout(F.gelu(y))
        return residual + y


class _SinusoidalPosition(nn.Module):
    """Non-learned positions, extended on demand for arbitrary replay windows."""

    def __init__(self, dim: int, initial_length: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("encoding", self._make(initial_length), persistent=False)

    def _make(self, length: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / self.dim))
        encoding = torch.zeros(length, self.dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div)
        # For odd d_model, the cosine stream has one fewer channel.
        encoding[:, 1::2] = torch.cos(positions * div[:encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            self.encoding = self._make(x.shape[1]).to(device=x.device)
        return x + self.encoding[:, :x.shape[1]].to(dtype=x.dtype, device=x.device)


class HybridTemporalForecastNet(nn.Module):
    """TCN + causal Transformer multi-horizon forecaster.

    Parameters intentionally mirror :class:`forecast.model.TemporalForecastNet`.
    ``entity_slots`` enables a small shared entity stream: each 14-value slot
    is projected with the same weights, receives a learned role embedding, and
    is mean pooled before fusion with the eight global features.  This gives
    the encoder a stable entity/global decomposition while retaining the flat
    ``[B, T, F]`` dataset interface.
    """

    architecture_name = "tcn_transformer_hybrid_v1"

    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 192,
                 layers: int = 3, entity_slots: int = 0, zone_classes: int = 0,
                 canonical_zone_classes: int = 0, nhead: int = 4,
                 dropout: float = 0.15, tcn_kernel: int = 3,
                 transformer_layers: Optional[int] = None,
                 max_seq_len: int = 256,
                 horizon_fusion: bool = False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.entity_slots = int(entity_slots)
        self.zone_classes = int(zone_classes)
        self.canonical_zone_classes = int(canonical_zone_classes)
        self.nhead = int(nhead)
        self.dropout_rate = float(dropout)
        self.tcn_kernel = int(tcn_kernel)
        self.transformer_layers = int(transformer_layers or layers)
        self.max_seq_len = int(max_seq_len)
        # Optional horizon-conditioned fusion.  The original v1 path is kept
        # byte-for-byte compatible for existing checkpoints; the optimized
        # path learns how much to trust the most recent state versus a
        # horizon-specific smoothed context and adds a small horizon code.
        self.horizon_fusion = bool(horizon_fusion)
        if self.hidden <= 0 or self.layers <= 0:
            raise ValueError("hidden and layers must be positive")
        if self.hidden % self.nhead:
            raise ValueError(f"hidden={self.hidden} must be divisible by nhead={self.nhead}")
        if self.entity_slots and self.entity_slots != len(ENTITY_SLOTS):
            raise ValueError(f"entity_slots must be {len(ENTITY_SLOTS)}")
        if self.zone_classes and self.zone_classes != len(FUNCTIONAL_ZONE_LABELS):
            raise ValueError(f"zone_classes must be {len(FUNCTIONAL_ZONE_LABELS)}")
        if self.canonical_zone_classes and self.canonical_zone_classes != CANONICAL_ZONE_DIM:
            raise ValueError(f"canonical_zone_classes must be {CANONICAL_ZONE_DIM}")

        # The current feature contract is 14 values per entity slot followed
        # by N_GLOBAL values.  Keep a flat projection fallback for old or
        # synthetic inputs so this module remains useful in unit tests.
        self.entity_slot_width = FEATURES_PER_SLOT
        self.global_dim = self.input_dim
        if self.input_dim == self.entity_slots * MOTION_FEATURES_PER_SLOT + MOTION_N_GLOBAL:
            self.entity_slot_width = MOTION_FEATURES_PER_SLOT
            self.global_dim = MOTION_N_GLOBAL
        elif self.input_dim >= self.entity_slots * FEATURES_PER_SLOT + N_GLOBAL:
            self.global_dim = self.input_dim - self.entity_slots * FEATURES_PER_SLOT
        self.entity_stream_enabled = bool(
            self.entity_slots and self.input_dim >= self.entity_slots * self.entity_slot_width + self.global_dim
        )
        if self.entity_stream_enabled:
            self.entity_slot_proj = nn.Sequential(
                nn.Linear(self.entity_slot_width, self.hidden),
                nn.LayerNorm(self.hidden), nn.GELU(),
            )
            self.entity_role = nn.Parameter(torch.zeros(self.entity_slots, self.hidden))
            nn.init.normal_(self.entity_role, std=0.02)
            self.global_proj = nn.Sequential(
                nn.Linear(self.global_dim, self.hidden),
                nn.LayerNorm(self.hidden), nn.GELU(),
            )
            self.input_proj = nn.Sequential(
                nn.Linear(self.hidden * 2, self.hidden),
                nn.LayerNorm(self.hidden), nn.GELU(),
            )
        else:
            self.global_dim = self.input_dim
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden),
                nn.LayerNorm(self.hidden), nn.GELU(),
            )

        dilations = [2 ** i for i in range(self.layers)]
        self.tcn = nn.ModuleList([
            _CausalConvBlock(self.hidden, self.tcn_kernel, dilation, self.dropout_rate)
            for dilation in dilations
        ])
        self.tcn_norm = nn.LayerNorm(self.hidden)

        self.position = _SinusoidalPosition(self.hidden, initial_length=self.max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=self.nhead,
            dim_feedforward=self.hidden * 4,
            dropout=self.dropout_rate, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.transformer_layers)
        self.transformer_norm = nn.LayerNorm(self.hidden)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden),
            nn.LayerNorm(self.hidden), nn.GELU(), nn.Dropout(self.dropout_rate),
        )
        if self.horizon_fusion:
            self.horizon_embedding = nn.Embedding(len(HORIZONS), self.hidden)
            self.horizon_gate = nn.Sequential(
                nn.Linear(self.hidden * 2, self.hidden),
                nn.LayerNorm(self.hidden), nn.GELU(),
                nn.Linear(self.hidden, self.hidden), nn.Sigmoid(),
            )
            self.horizon_norm = nn.LayerNorm(self.hidden)

        self.heads = nn.ModuleDict({
            str(h): nn.ModuleDict({
                "global_mode": nn.Linear(self.hidden, len(MODES)),
                "opponent_trend": nn.Linear(self.hidden, len(TRENDS)),
                "enemy_region": nn.Linear(self.hidden, len(REGION_LABELS)),
            }) for h in HORIZONS
        })
        if self.entity_slots:
            self.entity_heads = nn.ModuleDict({
                str(h): nn.Linear(self.hidden, self.entity_slots * ENTITY_TARGET_DIM)
                for h in HORIZONS
            })
        else:
            self.entity_heads = nn.ModuleDict()
        if self.zone_classes:
            self.entity_zone_heads = nn.ModuleDict({
                str(h): nn.Linear(self.hidden, self.entity_slots * self.zone_classes)
                for h in HORIZONS
            })
        else:
            self.entity_zone_heads = nn.ModuleDict()
        if self.canonical_zone_classes:
            self.entity_canonical_zone_heads = nn.ModuleDict({
                str(h): nn.Linear(self.hidden, self.entity_slots * self.canonical_zone_classes)
                for h in HORIZONS
            })
        else:
            self.entity_canonical_zone_heads = nn.ModuleDict()

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        """Return the bool upper-triangular mask used by the Transformer."""
        if length <= 0:
            raise ValueError("sequence length must be positive")
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _embed_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.entity_stream_enabled:
            return self.input_proj(x)
        entity_end = self.entity_slots * self.entity_slot_width
        slots = x[..., :entity_end].reshape(
            x.shape[0], x.shape[1], self.entity_slots, self.entity_slot_width
        )
        slot_hidden = self.entity_slot_proj(slots)
        slot_hidden = slot_hidden + self.entity_role.view(1, 1, self.entity_slots, self.hidden)
        slot_hidden = slot_hidden.mean(dim=2)
        global_hidden = self.global_proj(x[..., entity_end:])
        return self.input_proj(torch.cat([slot_hidden, global_hidden], dim=-1))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        projected = self._embed_input(x)
        tcn = projected
        for block in self.tcn:
            tcn = block(tcn)
        tcn = self.tcn_norm(tcn)
        transformer_input = self.position(projected)
        transformer = self.transformer(
            transformer_input,
            mask=self.causal_mask(projected.shape[1], projected.device),
        )
        transformer = self.transformer_norm(transformer)
        return self.fusion(torch.cat([tcn, transformer], dim=-1))

    @staticmethod
    def _horizon_view(encoded: torch.Tensor, horizon: int) -> torch.Tensor:
        factor = HORIZON_POOL[horizon]
        if factor <= 1:
            return encoded[:, -1]
        usable = (encoded.shape[1] // factor) * factor
        if usable == 0:
            return encoded[:, -1]
        view = encoded[:, -usable:]
        return view.reshape(view.shape[0], usable // factor, factor, view.shape[-1]).mean(dim=2)[:, -1]

    def _horizon_representation(self, encoded: torch.Tensor, horizon: int) -> torch.Tensor:
        pooled = self._horizon_view(encoded, horizon)
        if not self.horizon_fusion:
            return pooled
        last = encoded[:, -1]
        horizon_idx = HORIZONS.index(horizon)
        code = self.horizon_embedding.weight[horizon_idx].view(1, -1)
        gate = self.horizon_gate(torch.cat([pooled, last], dim=-1))
        # Gate is per channel, which lets short horizons preserve fast motion
        # components while long horizons prefer the smoothed context.
        fused = pooled + gate * (last - pooled) + code
        return self.horizon_norm(fused)

    def forward(self, x: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected [batch,time,{self.input_dim}], got {tuple(x.shape)}")
        if x.shape[1] <= 0:
            raise ValueError("sequence length must be positive")
        encoded = self._encode(x)
        outputs: Dict[int, Dict[str, torch.Tensor]] = {}
        for horizon in HORIZONS:
            pooled = self._horizon_representation(encoded, horizon)
            outputs[horizon] = {
                task: head(pooled) for task, head in self.heads[str(horizon)].items()
            }
            if self.entity_slots:
                outputs[horizon]["entity_state"] = self.entity_heads[str(horizon)](pooled).reshape(
                    x.shape[0], self.entity_slots, ENTITY_TARGET_DIM
                )
                if self.zone_classes:
                    outputs[horizon]["entity_zone_state"] = self.entity_zone_heads[str(horizon)](pooled).reshape(
                        x.shape[0], self.entity_slots, self.zone_classes
                    )
                if self.canonical_zone_classes:
                    outputs[horizon]["entity_canonical_zone_state"] = self.entity_canonical_zone_heads[str(horizon)](pooled).reshape(
                        x.shape[0], self.entity_slots, self.canonical_zone_classes
                    )
        return outputs

    def architecture_config(self) -> dict:
        """Serializable constructor metadata for checkpoints and reports."""
        return {
            "architecture": self.architecture_name,
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "entity_slots": self.entity_slots,
            "zone_classes": self.zone_classes,
            "canonical_zone_classes": self.canonical_zone_classes,
            "nhead": self.nhead,
            "dropout": self.dropout_rate,
            "tcn_kernel": self.tcn_kernel,
            "transformer_layers": self.transformer_layers,
            "max_seq_len": self.max_seq_len,
            "entity_stream_enabled": self.entity_stream_enabled,
            "entity_slot_width": self.entity_slot_width,
            "global_dim": self.global_dim,
            "horizon_fusion": self.horizon_fusion,
        }


# A descriptive alias makes model selection code read naturally while keeping
# one implementation and one checkpoint contract.
TCNTransformerForecastNet = HybridTemporalForecastNet


__all__ = [
    "HORIZON_POOL",
    "HybridTemporalForecastNet",
    "TCNTransformerForecastNet",
]
