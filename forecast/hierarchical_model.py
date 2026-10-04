"""Hierarchical expert forecaster with map-graph and physics-aware decoding."""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .hybrid_model import HybridTemporalForecastNet
from .temporal import HORIZONS, FEATURES_PER_SLOT, MOTION_FEATURES_PER_SLOT, MOTION_N_GLOBAL
from .entity import ENTITY_SLOTS, ROBOT_SPEED_MPS
from worldstate.canonical_zone import CANONICAL_ZONE_DIM, neighbors, CANONICAL_ZONE_LABELS


def _graph_adjacency() -> torch.Tensor:
    n = len(CANONICAL_ZONE_LABELS)
    index = {z: i for i, z in enumerate(CANONICAL_ZONE_LABELS)}
    a = torch.eye(n, dtype=torch.float32)
    for z in CANONICAL_ZONE_LABELS:
        for other in neighbors(z):
            if other in index:
                a[index[z], index[other]] = 1.0
    return a / a.sum(dim=1, keepdim=True).clamp_min(1.0)


class _GraphBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.self_proj = nn.Linear(hidden, hidden)
        self.neighbor_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("ij,bjh->bih", adjacency, x)
        y = self.self_proj(x) + self.neighbor_proj(neigh)
        return self.norm(x + self.dropout(F.gelu(y)))


class HierarchicalExpertForecastNet(HybridTemporalForecastNet):
    """Causal hierarchical model.

    It retains the v1 output contract so existing decoders continue to work.
    Additional outputs are ``entity_death_risk`` and ``entity_uncertainty``.
    The x/y entity fields are decoded as bounded displacements from the latest
    observed position, with per-type speed caps and horizon-dependent range.
    """

    architecture_name = "hierarchical_expert_v3_kinematic"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.hidden
        self.slot_width = MOTION_FEATURES_PER_SLOT if self.input_dim == 329 else 14
        self.unit_encoder = nn.Sequential(
            nn.Linear(self.slot_width, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.role_embedding = nn.Parameter(torch.zeros(2, hidden))
        self.type_embedding = nn.Parameter(torch.zeros(len(ENTITY_SLOTS) // 2, hidden))
        nn.init.normal_(self.role_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        self.group_encoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.region_embedding = nn.Parameter(torch.zeros(CANONICAL_ZONE_DIM, hidden))
        nn.init.normal_(self.region_embedding, std=0.02)
        self.graph_blocks = nn.ModuleList([
            _GraphBlock(hidden, self.dropout_rate), _GraphBlock(hidden, self.dropout_rate)
        ])
        self.hier_fusion = nn.Sequential(
            nn.Linear(hidden * 3 + MOTION_N_GLOBAL, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(self.dropout_rate),
        )
        # Distinct experts avoid forcing a single representation to solve the
        # global mode and per-unit coordinate task with the same geometry.
        self.global_expert = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.entity_expert = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.entity_delta_heads = nn.ModuleDict({
            str(h): nn.Linear(hidden, self.entity_slots * 2) for h in HORIZONS
        })
        self.entity_death_heads = nn.ModuleDict({
            str(h): nn.Linear(hidden, self.entity_slots) for h in HORIZONS
        })
        self.entity_uncertainty_heads = nn.ModuleDict({
            str(h): nn.Linear(hidden, self.entity_slots * 2) for h in HORIZONS
        })
        speed_caps = []
        for role, entity_type in ENTITY_SLOTS:
            speed_caps.append(float(ROBOT_SPEED_MPS.get(entity_type, 0.35)))
        self.register_buffer("speed_caps", torch.tensor(speed_caps, dtype=torch.float32), persistent=True)
        self.register_buffer("map_adjacency", _graph_adjacency(), persistent=True)
        # A bounded kinematic decoder is used instead of asking the network to
        # rediscover constant-velocity motion from scratch.  The residual
        # head remains learnable, while the causal velocity drift provides a
        # stable short-horizon anchor.
        # Initial residual budgets are deliberately conservative.  The
        # persistence baseline is strong for stopped/slow units, so a short
        # horizon must not be allowed to invent a large displacement.
        initial_residual = {10: -1.75, 30: -1.40, 60: -0.90, 120: -0.65}
        self.residual_scale = nn.ParameterDict({
            str(h): nn.Parameter(torch.tensor(initial_residual[h]))
            for h in HORIZONS
        })
        self.reachable_bias = nn.ParameterDict({
            str(h): nn.Parameter(torch.tensor(0.75 if h <= 30 else 0.35))
            for h in HORIZONS
        })

    def _reachable_zone_bias(self, current_zone: torch.Tensor, horizon: int,
                             device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a soft graph prior for canonical-zone logits.

        The radius grows with the forecast horizon: 10s permits the current
        cell and its immediate neighbors, while longer horizons admit more
        graph hops.  This is a bias rather than a hard mask so censored or
        unusual transitions are still representable.
        """
        n = CANONICAL_ZONE_DIM
        hops = {10: 1, 30: 2, 60: 4, 120: 8}[int(horizon)]
        adjacency = self.map_adjacency.to(device=device, dtype=dtype)
        reach = torch.eye(n, device=device, dtype=dtype)
        frontier = reach
        for _ in range(hops):
            frontier = (frontier @ adjacency).clamp_min(0.0)
            reach = torch.maximum(reach, (frontier > 0).to(dtype))
        # current_zone is [B,S] and may contain zero for unknown tracks.
        idx = current_zone.clamp(0, n - 1).long()
        allowed = reach[idx]  # [B,S,n]
        return (allowed - 1.0) * 0.35

    def _hierarchical_snapshot(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        if f < self.entity_slots * self.slot_width + MOTION_N_GLOBAL:
            return super()._embed_input(x), None
        slots = x[..., :self.entity_slots * self.slot_width].reshape(
            b * t, self.entity_slots, self.slot_width)
        unit = self.unit_encoder(slots)
        role = torch.arange(self.entity_slots, device=x.device) // (self.entity_slots // 2)
        typ = torch.arange(self.entity_slots, device=x.device) % (self.entity_slots // 2)
        unit = unit + self.role_embedding[role].view(1, self.entity_slots, -1)
        unit = unit + self.type_embedding[typ].view(1, self.entity_slots, -1)
        # First geometry feature in every slot is the canonical zone index
        # normalized to [0,1].  Rounding is only used for graph aggregation;
        # raw x/y and motion remain continuous inputs to the unit encoder.
        zone_scalar = slots[..., 10].clamp(0.0, 1.0)
        zone_idx = torch.round(zone_scalar * float(CANONICAL_ZONE_DIM - 1)).long()
        zone_idx = zone_idx.clamp(0, CANONICAL_ZONE_DIM - 1)
        nodes = self.region_embedding.unsqueeze(0).expand(b * t, -1, -1).clone()
        nodes.scatter_add_(1, zone_idx.unsqueeze(-1).expand(-1, -1, self.hidden), unit)
        counts = torch.zeros((b * t, CANONICAL_ZONE_DIM), device=x.device)
        counts.scatter_add_(1, zone_idx, torch.ones_like(zone_scalar))
        nodes = nodes / (1.0 + counts.unsqueeze(-1))
        for block in self.graph_blocks:
            nodes = block(nodes, self.map_adjacency.to(dtype=nodes.dtype, device=x.device))
        region = nodes.mean(dim=1)
        self_group = unit[:, :self.entity_slots // 2].mean(dim=1)
        opp_group = unit[:, self.entity_slots // 2:].mean(dim=1)
        global_features = x[..., self.entity_slots * self.slot_width:].reshape(
            b * t, -1)
        # Preserve the side/time/visibility features in the final fusion.
        fused = self.hier_fusion(torch.cat([self_group, opp_group, region, global_features], dim=-1))
        return fused.reshape(b, t, self.hidden), unit.reshape(b, t, self.entity_slots, self.hidden)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        snapshot, _ = self._hierarchical_snapshot(x)
        tcn = snapshot
        for block in self.tcn:
            tcn = block(tcn)
        tcn = self.tcn_norm(tcn)
        transformer = self.transformer(
            self.position(snapshot), mask=self.causal_mask(snapshot.shape[1], snapshot.device)
        )
        transformer = self.transformer_norm(transformer)
        return self.fusion(torch.cat([tcn, transformer], dim=-1))

    def _enemy_region_motion_prior(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """Causal soft prior for the coarse enemy-centroid region.

        The region task is defined by a 4x3 centroid grid.  A small prior from
        the latest visible opponent centroid and its measured displacement
        helps 10/30s predictions without replacing the learned global head.
        Unknown or stationary tracks fall back to their current centroid.
        """
        b = x.shape[0]
        if self.slot_width < MOTION_FEATURES_PER_SLOT or self.entity_slots < 16:
            return torch.zeros((b, 12), device=x.device, dtype=x.dtype)
        slots = x[:, -1, :self.entity_slots * self.slot_width].reshape(
            b, self.entity_slots, self.slot_width)[:, self.entity_slots // 2:]
        valid = (slots[..., 3] > 0.5) & (slots[..., 9] > 0.5)
        weights = valid.to(dtype=x.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        current = (slots[..., :2] * weights.unsqueeze(-1)).sum(dim=1) / denom
        motion = slots[..., 14:16]
        moving = (slots[..., 16:17] > 0.025).to(dtype=x.dtype)
        dt = (x[:, -1, -1].clamp(0.2, 1.0) * 5.0).view(b, 1, 1)
        drift = (motion * moving) * (float(horizon) / dt)
        drift = drift * weights.unsqueeze(-1)
        drift = drift.sum(dim=1) / denom
        center = (current + drift).clamp(0.0, 1.0)
        xs = (torch.arange(4, device=x.device, dtype=x.dtype) + 0.5) / 4.0
        ys = (torch.arange(3, device=x.device, dtype=x.dtype) + 0.5) / 3.0
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        dist = ((center[:, None, :] - grid[None, :, :]) ** 2).sum(dim=-1)
        temperature = {10: 0.055, 30: 0.075, 60: 0.105, 120: 0.14}[int(horizon)]
        prior = -dist / temperature
        weight = {10: 0.30, 30: 0.22, 60: 0.12, 120: 0.06}[int(horizon)]
        return prior * weight

    def forward(self, x: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected [batch,time,{self.input_dim}], got {tuple(x.shape)}")
        encoded = self._encode(x)
        outputs: Dict[int, Dict[str, torch.Tensor]] = {}
        for horizon in HORIZONS:
            pooled = self._horizon_representation(encoded, horizon)
            global_repr = self.global_expert(pooled)
            entity_repr = self.entity_expert(pooled)
            outputs[horizon] = {
                task: head(global_repr) for task, head in self.heads[str(horizon)].items()
            }
            outputs[horizon]["enemy_region"] = (
                outputs[horizon]["enemy_region"]
                + self._enemy_region_motion_prior(x, horizon)
            )
            if self.entity_slots:
                raw = self.entity_heads[str(horizon)](entity_repr).reshape(
                    x.shape[0], self.entity_slots, 5)
                # Physics-aware displacement decoder.  Speed caps are metres/s;
                # use the latest causal displacement as a constant-velocity
                # prior, then let a bounded residual correct turns/stops.
                current = x[:, -1, :self.entity_slots * self.slot_width].reshape(
                    x.shape[0], self.entity_slots, self.slot_width)
                cur_xy = current[..., :2]
                delta_raw = self.entity_delta_heads[str(horizon)](entity_repr).reshape(
                    x.shape[0], self.entity_slots, 2)
                caps = self.speed_caps.to(device=x.device, dtype=x.dtype).view(1, self.entity_slots, 1)
                axis_scale = torch.tensor([28.0, 15.0], device=x.device, dtype=x.dtype).view(1, 1, 2)
                max_delta = caps * float(horizon) / axis_scale
                if self.slot_width >= MOTION_FEATURES_PER_SLOT:
                    motion = current[..., FEATURES_PER_SLOT:MOTION_FEATURES_PER_SLOT]
                    obs_dt = (x[:, -1, -1].clamp(0.2, 1.0) * 5.0).view(-1, 1, 1)
                    # motion dx/dy are map-normalized displacements over the
                    # previous observed interval.  Fall back to zero drift if
                    # the first frame has no valid interval.
                    drift = motion[..., :2] * (float(horizon) / obs_dt)
                    # A near-zero measured speed is treated as stationary;
                    # this prevents noise in a single frame from becoming a
                    # multi-step jump.
                    drift = drift * (motion[..., 2:3] > 0.025).to(dtype=x.dtype)
                    drift = torch.nan_to_num(drift, nan=0.0, posinf=0.0, neginf=0.0)
                    norm = drift.norm(dim=-1, keepdim=True)
                    cap_norm = max_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    drift = drift * torch.minimum(torch.ones_like(norm), cap_norm / norm.clamp_min(1e-6))
                else:
                    drift = torch.zeros_like(max_delta)
                residual_fraction = torch.sigmoid(self.residual_scale[str(horizon)]).to(dtype=x.dtype)
                residual_cap = max_delta * residual_fraction
                pos = (cur_xy + drift + torch.tanh(delta_raw) * residual_cap).clamp(0.0, 1.0)
                raw = raw.clone()
                raw[..., 1:3] = torch.logit(pos.clamp(1e-4, 1.0 - 1e-4))
                outputs[horizon]["entity_state"] = raw
                outputs[horizon]["entity_death_risk"] = self.entity_death_heads[str(horizon)](entity_repr)
                outputs[horizon]["entity_uncertainty"] = self.entity_uncertainty_heads[str(horizon)](entity_repr).reshape(
                    x.shape[0], self.entity_slots, 2)
                if self.zone_classes:
                    outputs[horizon]["entity_zone_state"] = self.entity_zone_heads[str(horizon)](entity_repr).reshape(
                        x.shape[0], self.entity_slots, self.zone_classes)
                if self.canonical_zone_classes:
                    zone_logits = self.entity_canonical_zone_heads[str(horizon)](entity_repr).reshape(
                        x.shape[0], self.entity_slots, self.canonical_zone_classes)
                    zone_scalar = current[..., 10].clamp(0.0, 1.0)
                    current_zone = torch.round(zone_scalar * float(CANONICAL_ZONE_DIM - 1))
                    zone_logits = zone_logits + (
                        self._reachable_zone_bias(current_zone, horizon, x.device, zone_logits.dtype)
                        * torch.sigmoid(self.reachable_bias[str(horizon)]).to(zone_logits.dtype)
                    )
                    outputs[horizon]["entity_canonical_zone_state"] = zone_logits
        return outputs


__all__ = ["HierarchicalExpertForecastNet"]
