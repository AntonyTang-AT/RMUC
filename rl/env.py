# -*- coding: utf-8 -*-
"""M2 RL 环境: 一场真实比赛的决策点序列作为多单位全队 MARL 环境。

- 红方每单位类(英雄/工程/步兵3/步兵4/空中/哨兵)是一个 agent, 共享团队胜负条件奖励 -> 参数共享 IPPO。
- 动作 = (move, fire) 12 种离散组合; target/risk 由规则确定性填充(见 cfg / distill_labels)。
- reward = reward.compute_reward(对局结果接地五组组件, 权重可学) + 终端胜负加成。
- 观测 = 自己类(4) + 每敌方类(5, 含 vis_prior 部分可观测) + 全局(6)。
  部分可观测与 RMUC-OfflineRL feature.py 对齐: 敌方位置在哨兵感知半径 self.vis_radius 外=不可见,
  位置隐藏(0.5,0.5)、vis_prior=0; 避免 train-vs-deploy gap(裁判数据全知 vs 哨兵实感)。
  敌方能力档/敌情先验(与 opstrength 对应)留给 game.theory 层, 此处只做空间可见性。
- 纯 numpy, 无 torch(单测离线可跑, 满足测试约束)。
"""
import numpy as np
import config
from schema import layered as sl
from worldstate import worldstate as ws
from data.features import BUILDING_TYPES
from rl import reward as rw
from rl import physics as ph

# 12 种 (move, fire) = 6 moves × 2 fire
ACTION_MEANINGS = [(m, bool(f)) for m in sl.MOVES for f in (0, 1)]
AGENTS = list(sl.UNIT_CLASSES)
BASE_HP = 5000.0    # 基地最大物理血量(仅量纲参考)
OUTPOST_HP = 1500.0  # 前哨站最大物理血量
N_AGENT = len(AGENTS)  # 6


class M2Env:
    """多单位全队 MARL 环境。dps 为 features.build_decision_points 的决策点序列。"""

    def __init__(self, decision_points, bounds, team='红', vis_radius=8.0, reward_conf=None):
        self.dps = decision_points
        self.bounds = bounds
        self.team = team
        self.enemy_team = '蓝' if team == '红' else '红'
        self.t = 0
        self.vis_radius = vis_radius                 # 哨兵感知半径(米); 敌方在此半径外=不可见
        self.reward_conf = reward_conf or rw.DEFAULT_CONF
        # 预构建 worldstate(每决策点), 供 step 直接取值, 避免反复 build
        self.ws = [ws.build_worldstate(dp, bounds) for dp in decision_points]
        self.obs_dim = self._obs_dim()
        self.action_meanings = ACTION_MEANINGS
        self.n_actions = len(ACTION_MEANINGS)

    def _obs_dim(self):
        # 自身类(4) + 每敌方类(6*5: 含 vis_prior) + 全局(6: base红/蓝, out红/蓝, 存活红/蓝)
        return 4 + len(AGENTS) * 5 + 6

    # ---------- 观察 ----------
    def _class_feat(self, w, cls, team):
        """team 侧 unit class cls 特征; 缺失 -> 0 填充, dim 恒定。返回长度 4(自身/己方用)。"""
        for r in w['robots'].values():
            if r['team'] == team and r['type'] == cls:
                return [r['hp_ratio'], r['norm_xy'][0], r['norm_xy'][1], r.get('dfire', 0.0)]
        return [0.0, 0.0, 0.0, 0.0]

    # -------- 部分可观测: 哨兵感知半径内的敌方可见性 --------
    def _real_xy(self, rid):
        r = self.dps[self.t]['robots'].get(rid)
        if r is None:
            return None
        return (r.get('x', 0.0), r.get('y', 0.0))

    def _ego_sentry_xy(self, w):
        """红方哨兵真实坐标(米); 无哨兵 -> None。"""
        for rid, r in self.dps[self.t]['robots'].items():
            if r['type'] == '哨兵' and r['team'] == self.team:
                return (r.get('x', 0.0), r.get('y', 0.0))
        return None

    def _vis(self, ego, e):
        """哨兵感知半径内为可见; 任一缺失 -> 不可见。"""
        if ego is None or e is None:
            return False
        dx, dy = ego[0] - e[0], ego[1] - e[1]
        return (dx * dx + dy * dy) ** 0.5 <= self.vis_radius

    def _enemy_feat(self, w, cls, ego):
        """敌方 unit class cls 特征(5维, 含 vis_prior): 哨兵半径外隐藏位置(0.5,0.5)且 vis=0。"""
        for rid, r in w['robots'].items():
            if r['team'] == self.enemy_team and r['type'] == cls:
                x, y = r['norm_xy'][0], r['norm_xy'][1]
                vis = 1.0 if self._vis(ego, self._real_xy(rid)) else 0.0
                if vis == 0.0:
                    x, y = 0.5, 0.5  # 未知位置 -> 场地中心中性值
                return [r['hp_ratio'], x, y, r.get('dfire', 0.0), vis]
        return [0.0, 0.0, 0.0, 0.0, 0.0]

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
        """单 agent 观察向量(per-agent)。敌方带 vis_prior, 位置按哨兵感知半径可选隐藏。"""
        w = self.ws[self.t]
        ego = self._ego_sentry_xy(w)
        vec = self._class_feat(w, cls, self.team)
        for ec in AGENTS:
            vec += self._enemy_feat(w, ec, ego)
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
        """对局结果接地五组组件加权(攻击/防守/自身/经济); 权重默认 DEFAULT_CONF, 可被学习权重替换。"""
        return rw.compute_reward(w, self.reward_conf, self.team)

    # 局势描述: 供动作一致性 reward 使用(与规则头同一信号, 但作用在动作上)
    def _situation(self, w):
        from tree import global_mode as gm
        mode = gm.derive_global_mode(w)
        team, en = self.team, self.enemy_team
        enemy_in_half = any(r for r in w['robots'].values()
                            if r['team'] == en and r['norm_xy'][0] < 0.5)
        our_push = sum(1 for r in w['robots'].values()
                       if r['team'] == team and r['norm_xy'][0] > 0.5)
        return {'mode': mode, 'enemy_in_half': enemy_in_half, 'our_push': our_push}

    def _action_coherence(self, actions, w):
        """动作一致性分: 根据局势奖励/惩罚每个单位的 (move, fire) -> 提供动作条件梯度。
        注意: 这是"动作依赖的 reward 整形", 使 IPPO 学到局势敏感且合法的策略;
        与规则头相关但不等同(规则头是逐 step 启发式, 这是 RL 自己学到的价值判断)。
        返回团队级标量(可正可负)。"""
        s = self._situation(w)
        score = 0.0
        for cls, a in actions.items():
            move, fire = ACTION_MEANINGS[int(a) % len(ACTION_MEANINGS)]
            fe = self._class_feat(w, cls, self.team)  # [hp,x,y,dfire]
            hp = fe[0]
            if hp <= 0.0:
                score -= 0.5          # 不指挥已损单位
                continue
            aggressive = move in ('进攻', '推进')
            if fire:
                score += 0.15
            if aggressive:
                if s['enemy_in_half']:
                    score -= 0.2      # 己方半场受袭仍强攻 -> 不合理
                elif s['our_push'] >= 3:
                    score += 0.3      # 大部队推进 -> 进攻合理
            if move == '回防':
                score += (0.3 if s['enemy_in_half'] else -0.1)
        return score

    def step(self, actions):
        """actions: {cls: action_index(0..11)}。返回 (next_obs_dict, reward, done, trunc, info)。
        reward = 动作一致性(动作条件梯度) + 下一决策点 RM 进展 + 终端胜负。"""
        rew = self._action_coherence(actions, self.ws[self.t])  # 动作依赖, 关键信号
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


class PhysicsEnv(M2Env):
    """B2 前向动力学环境: 用 rl.physics 的符号物理在 (state, 我方动作) 上推进 Δt,
    替换 M2Env 的 `t+=1`(replay-bandit)。动作决定性地改变位置/冷却/掉血 -> 真 MDP。
    复用父类 obs/reward/win_condition; 只覆盖 reset/step, 并用自建的模拟轨迹充当 self.ws。"""

    def __init__(self, decision_points, bounds, team='红', vis_radius=8.0,
                 reward_conf=None, params=None, horizon=None):
        self._sup = super()
        self._sup.__init__(decision_points, bounds, team, vis_radius, reward_conf)
        self.dt = config.DF_STEP
        self.params = params or {}
        self.horizon = horizon if horizon is not None else len(decision_points) - 1
        # self.ws 由父类预建为真实轨迹; 这里重定向为"模拟轨迹", 首元素=真实 t0, 后面 step 追加。
        self.ws = [self.ws[0]]
        self.live = self._copy_units(decision_points[0])

    def _copy_units(self, dp):
        return {rid: dict(r) for rid, r in dp['robots'].items()}

    def _worldstate(self, live):
        """由模拟单位重建 worldstate; 排除阵亡单位(建筑保留), 供 obs/reward/win 复用。"""
        robots = {}
        for rid, r in live.items():
            if r['type'] in BUILDING_TYPES:
                robots[rid] = r
            elif r.get('hp', 0.0) > 0.0:
                robots[rid] = r
        dp = {'t': self.dps[self.t]['t'], 'robots': robots, 'team': self.dps[self.t]['team']}
        return ws.build_worldstate(dp, self.bounds)

    # -------- 部分可观测: PhysicsEnv 的坐标必须用"模拟态", 而非回放态 --------
    def _real_xy(self, rid):
        """敌方/任一单位的真实坐标(米): PhysicsEnv 里取模拟 live; 模拟与实际回放偏离后仍需一致。"""
        r = self.live.get(rid)
        if r is None:
            return None
        return (r.get('x', 0.0), r.get('y', 0.0))

    def _ego_sentry_xy(self, w):
        """己方哨兵真实坐标(米): 取模拟 live; 否则用回放坐标会与喂给 obs 的模拟状态不一致。"""
        for rid, r in self.live.items():
            if r['type'] == '哨兵' and r['team'] == self.team:
                return (r.get('x', 0.0), r.get('y', 0.0))
        return None

    def _team_unit(self, w, cls):
        for r in w['robots'].values():
            if r['team'] == self.team and r['type'] == cls:
                return r
        return None

    def _phys_actions(self, actions):
        """RL 动作索引 -> (move, fire) + 意图 target。move/fire 取自 RL 动作, target 取自规则意图
        (derive_unit_intent), 使物理目标选择遵守意图(攻击追建筑/防守只打缠斗)。
        观测只看到己方动作, 未来由物理确定性决定 -> 意图 target 是**导出**信号, 不引入真值泄漏。"""
        from tree import global_mode as gm
        w = self.ws[self.t]
        mode = gm.derive_global_mode(w)
        out = {}
        for cls, a in actions.items():
            move, fire = ACTION_MEANINGS[int(a) % len(ACTION_MEANINGS)]
            unit = self._team_unit(w, cls)
            target = gm.derive_unit_intent(unit, w, mode)['target'] if unit is not None else '无'
            out[cls] = {'move': move, 'fire': bool(fire), 'target': target}
        return out

    def _enemy_actions(self, w):
        """敌方反应式对抗: 读己方局势 -> enemy_best_response -> commit_actions(含 target)。
        注意: 这是**模式级**应对(非学到的策略), 与 rollout 标注同构。"""
        from tree import global_mode as gm
        from labels import rollout_label as rrl
        ego = gm.derive_global_mode(w)
        e_m = rrl.enemy_best_response(ego)
        return rrl.commit_actions(w, e_m, self.enemy_team)

    def _phys_params(self):
        """可选: 用真实游戏标定的速度/弹种覆盖默认。"""
        return self.params

    def reset(self):
        self.t = 0
        self.live = self._copy_units(self.dps[0])
        self.ws = [self.ws[0]]
        return self.obs_dict()

    def step(self, actions):
        rew = self._action_coherence(actions, self.ws[self.t])
        if self.t < self.horizon:
            w = self.ws[self.t]
            my_act = self._phys_actions(actions)
            en_act = self._enemy_actions(w)
            # 双方对称: 我敌方各一次 step_robots, 都关返回敌火(避免双算; 敌方已按自身模式行动)
            self.live = ph.step_robots(self.live, self.team, my_act, self.dt,
                                       self._phys_params(), enemy_fire=False)
            self.live = ph.step_robots(self.live, self.enemy_team, en_act, self.dt,
                                       self._phys_params(), enemy_fire=False)
            self.t += 1
            self.ws.append(self._worldstate(self.live))
            rew += self._reward(self.ws[self.t])
        done = self.t >= self.horizon
        trunc = done
        if done:
            rew += self.win_condition()
        return self.obs_dict(), float(rew), done, trunc, {'t': self.t, 'roles': AGENTS}


def action_to_intent(action_index, unit, worldstate, global_mode):
    """把 RL 动作索引映射为合法 4 通道意图: 保留规则 target/risk, 用 RL 的 move/fire 覆盖。"""
    from tree import global_mode as gm
    move, fire = ACTION_MEANINGS[action_index]
    it = gm.derive_unit_intent(unit, worldstate, global_mode)
    it['move'] = move
    it['fire'] = bool(fire)
    return it
