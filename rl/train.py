# -*- coding: utf-8 -*-
"""M2 IPPO(参数共享): 多单位全队 MARL 训练器。

- 每个单位类 agent 共享同一 ActorCritic 网络; 每个 agent 有自己的 per-agent 观察。
- torch 仅在 `IPPO` 构造 / `train` / `rollout` / `save` / `load` 内惰性 import。
- `rollout_actions`: 对给定决策点序列, 每 agent 用当前策略(采样或 argmax)选 (move, fire) -> 供蒸馏。
"""
import numpy as np


def actor_critic(obs_dim, hidden=64):
    import torch
    import torch.nn as nn
    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
            self.pol = nn.Linear(hidden, 12)
            self.val = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.fc(x)
            return self.pol(h), self.val(h).squeeze(-1)

    return AC()


class IPPO:
    def __init__(self, obs_dim, hidden=64, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip=0.2, epochs=4, batch=512, ent_coef=0.01, vf_coef=0.5):
        import torch
        self._torch = torch
        self.obs_dim = obs_dim
        self.gamma, self.gae_lambda, self.clip = gamma, gae_lambda, clip
        self.epochs, self.batch, self.ent_coef, self.vf_coef = epochs, batch, ent_coef, vf_coef
        self.policy = actor_critic(obs_dim, hidden)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    # ---------- 单轨迹 rollout ----------
    def _rollout_one_episode(self, env, deterministic=False):
        torch = self._torch
        obs = env.reset()
        done = trunc = False
        trace = []  # (agent, obs, action, reward, next_obs, done, old_log_prob)
        role_list = list(env.obs_dict().keys())
        while not (done or trunc):
            act_map, old_lp_map = {}, {}
            for ag, o in obs.items():
                with torch.no_grad():
                    logits, _v = self.policy(torch.tensor(o, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                a = (int(torch.argmax(logits, dim=-1).item())
                     if deterministic else int(dist.sample().item()))
                # 行为策略 log-prob: 采样时的旧策略, PPO off-policy ratio 用(修复 old=curr 恒1的bug)
                with torch.no_grad():
                    old_lp = dist.log_prob(torch.tensor([a], dtype=torch.long)).item()
                act_map[ag] = a
                old_lp_map[ag] = old_lp
            next_obs, rew, done, trunc, info = env.step(act_map)
            for ag in role_list:
                trace.append((ag, obs[ag], act_map[ag], rew, next_obs[ag], done, old_lp_map[ag]))
            obs = next_obs
        return trace

    def train(self, env, n_updates=200, device='cpu', verbose=False):
        """参数共享 IPPO: 反复采样多 agent 轨迹 -> 合并过渡做 PPO 更新。返回统计 dict。"""
        import torch
        torch.set_grad_enabled(True)
        stats = {'updates': 0, 'mean_reward': 0.0}
        ep_rewards = []
        for _up in range(n_updates):
            trace = self._rollout_one_episode(env)
            if trace:
                ep_rewards.append(sum(r for _a, _o, _act, r, _no, _d, _lp in trace) / len(set(a for a, *_r in trace)))
            # 组装批量 tensors
            obs_t = torch.tensor(np.array([t[1] for t in trace]), dtype=torch.float32, device=device)
            act_t = torch.tensor([t[2] for t in trace], dtype=torch.long, device=device)
            rew_t = torch.tensor([t[3] for t in trace], dtype=torch.float32, device=device)
            dones = [t[5] for t in trace]
            next_t = torch.tensor(np.array([t[4] for t in trace]), dtype=torch.float32, device=device)
            old_lp_t = torch.tensor([t[6] for t in trace], dtype=torch.float32, device=device)
            # GAE
            values = self._value(obs_t)
            next_values = self._value(next_t)
            adv, ret = self._gae(values, next_values, rew_t, torch.tensor(dones, dtype=torch.float32, device=device))
            for _e in range(self.epochs):
                idx = torch.randperm(len(trace), device=device)
                for s in range(0, len(idx), self.batch):
                    b = idx[s:s + self.batch]
                    self._ppo_step(obs_t[b], act_t[b], ret[b], adv[b], old_lp_t[b])
            stats['updates'] += 1
            stats['mean_reward'] = float(np.mean(ep_rewards[-10:])) if ep_rewards else 0.0
        torch.set_grad_enabled(False)
        return stats

    def _value(self, x):
        import torch
        with torch.no_grad():
            return self.policy(x)[1]

    def _gae(self, values, next_values, rewards, dones):
        torch = self._torch
        last_gae = torch.zeros_like(rewards[0])
        adv = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            adv[t] = last_gae
        ret = adv + values
        return adv, ret

    def _ppo_step(self, obs_b, act_b, ret_b, adv_b, old_lp_b):
        torch = self._torch
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        logits, values = self.policy(obs_b)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(act_b)
        # old_lp_b 是**行为策略**(rollout 采样时)的 log-prob; 与当前策略 log_prob 不同 -> ratio≠1
        # (修复前用 log_prob.detach() 使 ratio 恒 1, clip 失效退化为 A2C/REINFORCE+GAE)
        ratio = torch.exp(log_prob - old_lp_b)
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_b
        pol_loss = -torch.min(surr1, surr2).mean()
        val_loss = (values - ret_b).pow(2).mean()
        ent = dist.entropy().mean()
        loss = pol_loss + self.vf_coef * val_loss - self.ent_coef * ent
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    # ---------- 蒸馏入口 ----------
    def rollout_actions(self, dps, bounds, deterministic=True, env=None):
        """对每个决策点, 每类 agent 用当前策略输出 action_index ->
        返回 list[dict{cls: (move, fire)}]。

        env: 复用/注入的环境; 缺省建 M2Env(replay-bandit)。若传入 PhysicsEnv, 则用 env.step
        做**闭环前向推进**(动作改变未来), 而非 bandit 的 `t+=1`。两种 env 统一用 step 推进,
        使 rollout 与训练同构(真 MDP 策略)。"""
        from rl.env import M2Env, ACTION_MEANINGS
        if env is None:
            env = M2Env(dps, bounds)
        out = []
        self._torch.set_grad_enabled(False)
        for _t in range(len(dps)):
            obs_d = env.obs_dict()
            act_map = {}
            for ag, o in obs_d.items():
                import torch
                logits, _v = self.policy(torch.tensor(o, dtype=torch.float32).unsqueeze(0))
                a = int(torch.argmax(logits, dim=-1).item()) if deterministic else 0
                act_map[ag] = a
            out.append({ag: ACTION_MEANINGS[act_map[ag]] for ag in obs_d})
            # 真 step: M2Env->下一真实决策点(行为与旧 t+=1 等价), PhysicsEnv->模拟推进(闭环)
            env.step(act_map)
        self._torch.set_grad_enabled(True)
        return out

    def save(self, path):
        import os, torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        import torch
        self.policy.load_state_dict(torch.load(path, map_location='cpu'))
