# -*- coding: utf-8 -*-
"""哨兵单位学生脚手架：feat → move / fire。torch 惰性 import。

未训练权重时不要在热路径启用；TieredDecider.unit_student=None 为默认。
"""
from __future__ import annotations

import json
import os
import numpy as np
from schema import layered as sl
from rl import unit_features as uf

FEAT_DIM = uf.FEAT_DIM
N_MOVE = uf.N_MOVE
HIDDEN = 64


def _build_net(device='cpu'):
    import torch
    import torch.nn as nn

    class UnitNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(FEAT_DIM, HIDDEN), nn.Tanh(),
                nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            )
            self.move_head = nn.Linear(HIDDEN, N_MOVE)
            self.fire_head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            h = self.backbone(x)
            return self.move_head(h), self.fire_head(h).squeeze(-1)

    return UnitNet().to(device)


class UnitStudent:
    """哨兵 move/fire 小 MLP。"""

    def __init__(self, device='cpu'):
        import torch
        self._torch = torch
        self.device = device
        self.net = _build_net(device)
        self.net.eval()

    def predict(self, feat):
        torch = self._torch
        x = torch.as_tensor(np.asarray(feat, dtype=np.float32).reshape(-1, FEAT_DIM),
                            device=self.device)
        with torch.no_grad():
            logits_m, logit_f = self.net(x)
        mi = int(logits_m.argmax(dim=-1).item())
        fire = bool(float(logit_f.item()) > 0.0)
        return uf.id_to_move(mi), fire

    def apply(self, decision, dp, bounds, w, opp_intent=None, team='红'):
        """覆盖决策中哨兵通道；其余单位不动。"""
        d = dict(decision) if decision else sl.build_empty_decision((w or {}).get('t', 0.0))
        mode = d.get('global_mode', '均衡')
        feat = uf.pack_unit_feat(dp, bounds, mode=mode, opp_intent=opp_intent, team=team)
        move, fire = self.predict(feat)
        units = dict(d.get('units') or {})
        cur = dict(units.get(uf.UNIT_FOCUS) or {
            'target': '无', 'move': '无', 'fire': False, 'risk': '中'})
        cur['move'] = move if move in sl.MOVES else cur.get('move', '无')
        cur['fire'] = bool(fire)
        if cur.get('target') not in sl.TARGETS:
            cur['target'] = '无'
        if cur.get('risk') not in sl.RISKS:
            cur['risk'] = '中'
        units[uf.UNIT_FOCUS] = cur
        d['units'] = units
        return d

    def save(self, out_dir):
        import torch
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'adapter.pt')
        torch.save(self.net.state_dict(), path)
        meta = {'feat_dim': FEAT_DIM, 'unit_class': uf.UNIT_FOCUS, 'kind': 'sentry_move_fire'}
        with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path_or_dir, device='cpu'):
        import torch
        pt = path_or_dir
        if os.path.isdir(path_or_dir):
            pt = os.path.join(path_or_dir, 'adapter.pt')
        obj = cls(device=device)
        obj.net.load_state_dict(torch.load(pt, map_location=device))
        obj.net.eval()
        return obj


def train_unit_student(X, y_move, y_fire, epochs=10, lr=1e-3, batch=64, device='cpu', seed=0):
    """脚手架训练；本机可不跑全量。返回 (student, stats)。"""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    X = np.asarray(X, dtype=np.float32)
    y_move = np.asarray(y_move, dtype=np.int64)
    y_fire = np.asarray(y_fire, dtype=np.float32)
    n = len(y_move)
    stu = UnitStudent(device=device)
    opt = torch.optim.Adam(stu.net.parameters(), lr=lr)
    stu.net.train()
    losses = []
    for _ in range(epochs):
        idx = np.random.permutation(n)
        ep = 0.0
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            xb = torch.as_tensor(X[bi], device=device)
            ym = torch.as_tensor(y_move[bi], device=device)
            yf = torch.as_tensor(y_fire[bi], device=device)
            lm, lf = stu.net(xb)
            loss = F.cross_entropy(lm, ym) + F.binary_cross_entropy_with_logits(lf, yf)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep += float(loss.item())
        losses.append(ep / max(1, (n + batch - 1) // batch))
    stu.net.eval()
    return stu, {'epochs': epochs, 'n': n, 'loss_last': round(losses[-1], 4) if losses else None}
