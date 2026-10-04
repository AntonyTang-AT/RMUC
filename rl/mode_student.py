# -*- coding: utf-8 -*-
"""Tier1 global_mode 学生：40→64→64→3 MLP。torch 仅在训练/推理内惰性 import。"""
from __future__ import annotations

import json
import os
import time
import numpy as np
from schema import layered as sl
from rl import mode_features as mf

OBS_DIM = mf.OBS_DIM
N_CLASS = len(mf.MODES)
HIDDEN = 64


def _build_net(device='cpu'):
    import torch
    import torch.nn as nn

    class ModeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(),
                nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                nn.Linear(HIDDEN, N_CLASS),
            )

        def forward(self, x):
            return self.net(x)

    net = ModeNet()
    return net.to(device)


class ModeStudent:
    def __init__(self, device='cpu'):
        import torch
        self._torch = torch
        self.device = device
        self.net = _build_net(device)
        self.net.eval()

    def predict_logits(self, obs):
        torch = self._torch
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM),
                            device=self.device)
        with torch.no_grad():
            return self.net(x).cpu().numpy()

    def predict_mode(self, obs):
        logits = self.predict_logits(obs)
        return mf.id_to_mode(int(np.argmax(logits, axis=-1)[0]))

    def predict_proba(self, obs):
        """返回 (mode, conf, probs_dict)。conf = softmax max。"""
        logits = self.predict_logits(obs).reshape(-1)
        # 数值稳定 softmax
        x = logits - float(np.max(logits))
        e = np.exp(x)
        p = e / max(float(e.sum()), 1e-12)
        i = int(np.argmax(p))
        probs = {mf.id_to_mode(j): round(float(p[j]), 4) for j in range(len(mf.MODES))}
        return mf.id_to_mode(i), float(p[i]), probs

    def predict_modes_batch(self, X):
        logits = self.predict_logits(X)
        ids = np.argmax(logits, axis=-1)
        return [mf.id_to_mode(i) for i in ids]

    def save(self, out_dir):
        import torch
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'adapter.pt')
        torch.save(self.net.state_dict(), path)
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


def train_mode_student(X, y, epochs=20, lr=1e-3, batch=256, device='cpu', seed=0):
    """加权 CE 训练。返回 (student, stats)。优先传入 Tier1 子集。"""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n == 0:
        raise ValueError('empty train set')

    counts = np.bincount(y, minlength=N_CLASS).astype(np.float32)
    raw_counts = {mf.id_to_mode(i): int(counts[i]) for i in range(N_CLASS)}
    counts_w = np.maximum(counts, 1.0)
    w = (counts_w.sum() / (N_CLASS * counts_w)).astype(np.float32)

    net = _build_net(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(y, device=device)
    wt = torch.as_tensor(w, device=device)
    n_batch = max(1, (n + batch - 1) // batch)
    t0 = time.time()
    net.train()
    last_loss = 0.0
    for ep in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for b in range(n_batch):
            idx = perm[b * batch:(b + 1) * batch]
            logits = net(xt[idx])
            loss = F.cross_entropy(logits, yt[idx], weight=wt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
        last_loss = ep_loss / n_batch
    net.eval()
    stu = ModeStudent(device=device)
    stu.net = net
    stats = {
        'n': n, 'epochs': int(epochs), 'lr': lr, 'batch': batch,
        'seconds': round(time.time() - t0, 3),
        'last_loss': round(last_loss, 5),
        'class_counts': raw_counts,
        'device': str(device),
    }
    return stu, stats


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, n_class=N_CLASS):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    f1s = []
    for c in range(n_class):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def legal_rate_after_refill(dps, bounds, modes, team='红'):
    from worldstate import worldstate as wsmod
    ok = 0
    n = 0
    for dp, mode in zip(dps, modes):
        w = wsmod.build_worldstate(dp, bounds)
        d = mf.refill_units_for_mode(sl.build_empty_decision(dp.get('t', 0.0)), w, mode, team=team)
        n += 1
        if sl.validate(d)[0]:
            ok += 1
    return ok / max(n, 1)


class ModeSmoother:
    """近 k 帧多数表决，抑制 mode 帧间乱跳。"""

    def __init__(self, window=3):
        self.window = max(1, int(window))
        self._buf = []

    def push(self, mode):
        m = mode if mode in mf.MODES else '均衡'
        self._buf.append(m)
        if len(self._buf) > self.window:
            self._buf = self._buf[-self.window:]
        # 多数；平票取最近
        # 未来可替换为 TemporalModeStudent（短时序），接口同 push(mode)->mode
        from collections import Counter
        c = Counter(self._buf)
        top = c.most_common()
        best_n = top[0][1]
        cands = [m for m, n in top if n == best_n]
        for m in reversed(self._buf):
            if m in cands:
                return m
        return m

    def reset(self):
        self._buf = []
