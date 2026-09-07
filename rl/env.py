# -*- coding: utf-8 -*-
"""M2 RL 环境: 一场真实比赛的决策点序列作为多单位全队 MARL 环境。

- 红方每单位类(英雄/工程/步兵3/步兵4/空中/哨兵)是一个 agent, 共享团队胜负条件奖励 -> 参数共享 IPPO。
- 动作 = (move, fire) 12 种离散组合; target/risk 由规则确定性填充(见 cfg / distill_labels)。
- reward = RM 胜负层级复合项: base/outpost/damage/数存活差 + 终端胜负加成。
- 纯 numpy, 无 torch(单测离线可跑, 满足测试约束)。
"""
import numpy as np
from schema import layered as sl
from worldstate import worldstate as ws

# 12 种 (move, fire) = 6 moves × 2 fire
ACTION_MEANINGS = [(m, bool(f)) for m in sl.MOVES for f in (0, 1)]
AGENTS = list(sl.UNIT_CLASSES)
BASE_HP = 5000.0    # 基地最大物理血量(仅量纲参考)
OUTPOST_HP = 1500.0  # 前哨站最大物理血量
DMG_REF = 3000.0     # 团队累计发弹归一化上限
N_AGENT = len(AGENTS)  # 6


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class M2Env:
    """多单位全队 MARL 环境。dps 为 features.build_decision_points 的决策点序列。"""

    def __init__(self, decision_points, bounds, team='红'):
        self.dps = decision_points
        self.bounds = bounds
        self.team = team
        self.enemy_team = '蓝' if team == '红' else '红'
        self.t = 0
        # 预构建 worldstate(每决策点), 供 step 直接取值, 避免反复 build
        self.ws = [ws.build_worldstate(dp, bounds) for dp in decision_points]
        self.obs_dim = self._obs_dim()
        self.action_meanings = ACTION_MEANINGS
        self.n_actions = len(ACTION_MEANINGS)

    def _obs_dim(self):
        # 自身类(4) + 每敌方类(6*4) + 全局(6: base红/蓝, out红/蓝, 存活红/蓝)
        return 4 + len(AGENTS) * 4 + 6

    # ---------- 观察 ----------
    def _class_feat(self, w, cls, team):
        """team 侧 unit class cls 特征; 缺失 -> 0 填充, dim 恒定。返回长度 4。"""
        for r in w['robots'].values():
            if r['team'] == team and r['type'] == cls:
                return [r['hp_ratio'], r['norm_xy'][0], r['norm_xy'][1], r.get('dfire', 0.0)]
        return [0.0, 0.0, 0.0, 0.0]

    def _count_team(self, w, team):
        return sum(1 for r in w['robots'].values() if r['team'] == team)

    def _global_feats(self, w):
        base = {'红': 1.0, '蓝': 1.0}
        outpost = {'红': 1.0, '蓝': 1.0}
        for b in w['buildings'].values():
            t = b.get('team')
            if t in base and b['type'] == '基地':
                base[t] = b['hp_ratio']
            elif t in outpost and b['type'] == '前哨站':
                outpost[t] = b['hp_ratio']
        r_red = self._count_team(w, '红')
        r_blue = self._count_team(w, '蓝')
        return [base['红'], base['蓝'], outpost['红'], outpost['蓝'],
                r_red / N_AGENT, r_blue / N_AGENT]

    def obs(self, cls='哨兵'):
        """单 agent 观察向量(per-agent)。"""
        w = self.ws[self.t]
        vec = self._class_feat(w, cls, self.team)
        for ec in AGENTS:
            vec += self._class_feat(w, ec, self.enemy_team)
        vec += self._global_feats(w)
        return np.array(vec, dtype=np.float32)

    def obs_dict(self):
        return {ag: self.obs(ag) for ag in AGENTS}

    def reset(self):
        self.t = 0
        return self.obs_dict()

    # ---------- 奖励: RM 胜负层级复合 ----------
    def _health(self, w):
        base = {'红': 1.0, '蓝': 1.0}
        outpost = {'红': 1.0, '蓝': 1.0}
        for b in w['buildings'].values():
            t = b.get('team')
            if t in base and b['type'] == '基地':
                base[t] = b['hp_ratio']
            elif t in outpost and b['type'] == '前哨站':
                outpost[t] = b['hp_ratio']
        return base, outpost

    def _damage(self, w):
        r = sum(x.get('ammo17', 0) + x.get('ammo42', 0) for x in w['robots'].values() if x['team'] == '红')
        b = sum(x.get('ammo17', 0) + x.get('ammo42', 0) for x in w['robots'].values() if x['team'] == '蓝')
        return r, b

    def _reward(self, w):
        base, outpost = self._health(w)
        term_base = _clip(base['红'] - base['蓝'])
        term_out = _clip(outpost['红'] - outpost['蓝'])
        r_d, b_d = self._damage(w)
        term_dmg = _clip((r_d - b_d) / DMG_REF)
        n_red = self._count_team(w, '红')
        n_blue = self._count_team(w, '蓝')
        term_alive = _clip((n_red - n_blue) / N_AGENT)
        return (0.5 * term_base + 0.3 * term_out + 0.2 * term_dmg + 0.1 * term_alive)

    def step(self, actions):
        """actions: {cls: action_index(0..11)}。返回 (next_obs_dict, reward, done, trunc, info)。"""
        prev_t = self.t
        rew = 0.0
        if self.t + 1 < len(self.ws):
            self.t += 1
            rew += self._reward(self.ws[self.t])
        done = (self.t >= len(self.ws) - 1)
        trunc = done
        if done:
            rew += self.win_condition()
        return self.obs_dict(), float(rew), done, trunc, {'t': self.t, 'roles': AGENTS}

    # ---------- RM 胜负判定(时间耗尽/基地被毁时依次比较) ----------
    def win_condition(self):
        w = self.ws[-1]
        base, outpost = self._health(w)
        if abs(base['红'] - base['蓝']) > 1e-9:
            return 1.0 if base['红'] > base['蓝'] else -1.0
        if abs(outpost['红'] - outpost['蓝']) > 1e-9:
            return 1.0 if outpost['红'] > outpost['蓝'] else -1.0
        r_d, b_d = self._damage(w)
        if abs(r_d - b_d) > 1e-9:
            return 1.0 if r_d > b_d else -1.0
        n_red = self._count_team(w, '红')
        n_blue = self._count_team(w, '蓝')
        if n_red != n_blue:
            return 1.0 if n_red > n_blue else -1.0
        return 0.0


def action_to_intent(action_index, unit, worldstate, global_mode):
    """把 RL 动作索引映射为合法 4 通道意图: 保留规则 target/risk, 用 RL 的 move/fire 覆盖。"""
    from tree import global_mode as gm
    move, fire = ACTION_MEANINGS[action_index]
    it = gm.derive_unit_intent(unit, worldstate, global_mode)
    it['move'] = move
    it['fire'] = bool(fire)
    return it
