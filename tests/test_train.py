# -*- coding: utf-8 -*-
"""rl.train IPPO 单测: 验证 PPO ratio bug 已修复(行为策略 old-log-prob, clip 生效)。
torch 惰性 import(pytest 缺 torch 则 skip), 保持其余测试离线无 torch 约束。
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 守护: 仅本测试需要 torch; 缺失则跳过(不破坏整体"无 torch"测试约束)
torch = pytest.importorskip("torch")

from rl import train as tr


class _TinyEnv:
    """obs=8 维常数向量, 4 步结束, 恒定 reward。仅用于验证训练循环不崩溃 + ratio 不再恒 1。"""
    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return {'英雄': np.zeros(8, dtype=np.float32)}

    def obs_dict(self):
        return {'英雄': np.zeros(8, dtype=np.float32)}

    def step(self, actions):
        self.t += 1
        done = self.t >= 4
        return self.obs_dict(), 0.1, done, done, {'roles': ['英雄']}


class _RatioSpy(tr.IPPO):
    """拦截 _ppo_step, 记录 ratio 是否恒为 1 及其 std, 用于断言 clip 真会触发。"""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ratios = []

    def _ppo_step(self, obs_b, act_b, ret_b, adv_b, old_lp_b):
        import torch
        logits, _v = self.policy(obs_b)
        dist = torch.distributions.Categorical(logits=logits)
        lp = dist.log_prob(act_b)
        ratio = torch.exp(lp - old_lp_b)
        self.ratios.append({
            'all_one': bool(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)),
            'std': float(ratio.std().detach()),
        })
        super()._ppo_step(obs_b, act_b, ret_b, adv_b, old_lp_b)


def test_ppo_train_finite_and_ratio_not_one():
    """训练几轮后: 更新次数达标, mean_reward 有限, 且存在某步 ratio≠1(即 clip 有意义)。"""
    torch.manual_seed(0)
    ppo = _RatioSpy(obs_dim=8, lr=5e-2, epochs=3)
    stats = ppo.train(_TinyEnv(), n_updates=3)
    assert stats['updates'] >= 3
    assert np.isfinite(stats['mean_reward'])
    # 至少到后期(策略已被更新)某批 ratio 不再是全 1
    assert any(not r['all_one'] for r in ppo.ratios)
    assert max(r['std'] for r in ppo.ratios) > 1e-4
    # 最后一批的 ratio 有离 1 的偏移(策略确实偏离采样时的旧策略)
    assert not ppo.ratios[-1]['all_one']


def test_rollout_trace_carries_old_log_prob():
    """一条轨迹的每步都带行为策略 log-prob(old_lp), 且是有限标量。"""
    torch.manual_seed(0)
    ppo = tr.IPPO(obs_dim=8)
    trace = ppo._rollout_one_episode(_TinyEnv())
    assert trace
    for step in trace:
        assert len(step) == 7                      # (ag, obs, act, rew, next, done, old_lp)
        assert np.isfinite(float(step[6]))


def test_train_and_rollout_restore_grad_enabled_state():
    """训练和推理不得泄漏调用方的全局 autograd 开关。"""
    ppo = tr.IPPO(obs_dim=8)
    original = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(False)
        ppo.train(_TinyEnv(), n_updates=1)
        assert torch.is_grad_enabled() is False

        torch.set_grad_enabled(True)
        ppo.train(_TinyEnv(), n_updates=1)
        assert torch.is_grad_enabled() is True

        torch.set_grad_enabled(False)
        ppo.rollout_actions([None], {}, env=_TinyEnv())
        assert torch.is_grad_enabled() is False

        torch.set_grad_enabled(True)
        ppo.rollout_actions([None], {}, env=_TinyEnv())
        assert torch.is_grad_enabled() is True
    finally:
        torch.set_grad_enabled(original)
