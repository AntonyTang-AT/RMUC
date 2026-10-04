# -*- coding: utf-8 -*-
"""M2 RL 奖励: RMUC-OfflineRL 风格的对局结果接地奖励组件 + 可学权重。

数据集没有 reward 列 -> 好/坏行为只能从**可观测对局结果**定义。我们把一场比赛的
胜负拆成若干**未加权整形组件**(攻击/防守/自身/经济), 每项从 worldstate 直接可算,
再用 reward_model 逻辑回归从"没赢/赢"学出权重, 与手调对比。

与 rl/env.py 的 _reward(三表手调 0.5/0.3/0.2/0.1) 的区别:
- 攻击侧(打爆敌方基地/前哨/击杀)与防守侧(保住己方基地/前哨/少死)分开;
- 有哨兵自身 ego 项(存活 + 保住血量 + 过热惩罚), 这是之前完全没有的;
- 有经济项(金币领先);
- 权重可由 RewardConfig 配置, 默认值镜像 RMUC-OfflineRL, 可被 reward_learn 学到的权重替换。

离线纯 numpy, 无 torch(单测可跑)。
"""
from schema import layered as sl

# 归一化参考(与 RMUC-OfflineRL reward.py 一致)
BASE_HP = 5000.0      # 基地最大物理血量
OUTPOST_HP = 1500.0   # 前哨站最大物理血量
HEAT_NORM = 260.0     # 17mm 热量上限参考(哨兵/步兵)
COIN_NORM = 1000.0    # 队伍金币归一化上限
N_AGENT = 6           # 每方可移动单位数(英雄/工程/步兵3/步兵4/空中/哨兵)


# 手设默认权重(镜像 RMUC-OfflineRL: w_base_atk=5/w_kill=2/... 按我们的五组改写)。
# 注意: defender 侧一项"己方被伤/死亡"值是负的(好), 权重为正 -> 产品是负分, 天然惩罚。
# 我们这里攻/守分开, 敌方损是正、己方损是负, 权重统一为正。
DEFAULT_CONF = {
    # 攻击侧(对敌方造成损失 >= 对我方好)
    'outpost_atk': 3.0,
    'base_atk': 5.0,
    'kill': 2.0,            # 击毁敌方单位(用存活数差近似)
    'dmg_deal': 1.0,        # 我方放弹(伤害代理)
    # 防守侧(我方损失少 >= 对我方好; 这些项本身带符号)
    'outpost_def': 3.0,
    'base_def': 5.0,
    'death': 2.0,           # 我方可移动单位被击毁(存活数差近似)
    'dmg_take': 1.0,
    # 自身(哨兵)ego
    'ego_surv': 0.02,
    'ego_hp': 1.0,
    'heat': 0.5,
    # 经济
    'econ': 0.5,
}

# 组说明(供文档/可读性)
GROUP_NAMES = {
    'attack': ['outpost_atk', 'base_atk', 'kill', 'dmg_deal'],
    'defend': ['outpost_def', 'base_def', 'death', 'dmg_take'],
    'ego': ['ego_surv', 'ego_hp', 'heat'],
    'economy': ['econ'],
}


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def feat_names():
    return list(DEFAULT_CONF.keys())


def _building(w, btype, team, default=1.0):
    """team 侧 building btype 的 hp_ratio; 缺失(规则里当前站存满)用 default。"""
    for b in w['buildings'].values():
        if b['type'] == btype and b['team'] == team:
            return b['hp_ratio']
    return default


def _team_coins(w, team):
    return w['team'].get(team, {}).get('coins', 0.0)


def _alive(w, team):
    return sum(1 for r in w['robots'].values() if r['team'] == team and r['hp_ratio'] > 0.0)


def _sentry(w, team):
    for r in w['robots'].values():
        if r['type'] == '哨兵' and r['team'] == team:
            return r
    return None


def reward_components(w, team='红'):
    """从单个 worldstate 计算未加权、带符号、归一化的组件 dict(对 team 越大越好)。

    每一项都是"此刻对 team 的优势势能", 与 RMUC-OfflineRL 的组件语义对齐但按局势计算。
    返回 {name: float}。
    """
    enemy = '蓝' if team == '红' else '红'
    own_base = _building(w, '基地', team)
    own_out = _building(w, '前哨站', team)
    en_base = _building(w, '基地', enemy)
    en_out = _building(w, '前哨站', enemy)

    # 攻击侧(敌方血量越低越好 -> 与满血差)
    outpost_atk = _clip(1.0 - en_out)
    base_atk = _clip(1.0 - en_base)

    alive_own = _alive(w, team)
    alive_en = _alive(w, enemy)
    # kill/death 共用存活数差: 我方多存活 -> kill+ / death- (同源两项, 一正一负拆开)
    kill = _clip((alive_own - alive_en) / N_AGENT)
    death = _clip((alive_en - alive_own) / N_AGENT)

    # 伤害代理(放弹差, 我方多 -> 大)
    own_ammo = sum(r.get('ammo17', 0) + r.get('ammo42', 0) for r in w['robots'].values()
                   if r['team'] == team)
    en_ammo = sum(r.get('ammo17', 0) + r.get('ammo42', 0) for r in w['robots'].values()
                  if r['team'] == enemy)
    dmg_deal = _clip((own_ammo - en_ammo) / 3000.0)
    dmg_take = _clip((en_ammo - own_ammo) / 3000.0)

    # 防守侧(己方血量越高越好)
    outpost_def = _clip(own_out)
    base_def = _clip(own_base)

    # 哨兵 ego(存活 + 保血 + 过热惩罚)
    sentry = _sentry(w, team)
    if sentry is None or sentry['hp_ratio'] <= 0.0:
        ego_surv = 0.0
        ego_hp = 0.0
        heat = -1.0  # 哨兵被毁即满惩罚
    else:
        ego_surv = 1.0
        ego_hp = _clip(sentry['hp_ratio'], 0.0, 1.0)
        heat = _clip(sentry.get('heat', 0.0) / HEAT_NORM) * -1.0  # 散热超载惩罚

    # 经济(金币领先)
    econ = _clip((_team_coins(w, team) - _team_coins(w, enemy)) / COIN_NORM)

    return {
        'outpost_atk': outpost_atk,
        'base_atk': base_atk,
        'kill': kill,
        'dmg_deal': dmg_deal,
        'outpost_def': outpost_def,
        'base_def': base_def,
        'death': death,
        'dmg_take': dmg_take,
        'ego_surv': ego_surv,
        'ego_hp': ego_hp,
        'heat': heat,
        'econ': econ,
    }


def compute_reward(w, conf=None, team='红'):
    """加权后的标量奖励(对 team 的有符号优势势能)。conf 缺省用 DEFAULT_CONF。
    注意: 显式传空 dict 表示"无权重项" -> 返回 0, 不能用 `conf or` 兜底(NULL/空都会被短路)。"""
    if conf is None:
        conf = DEFAULT_CONF
    comp = reward_components(w, team)
    return sum(conf.get(k, 0.0) * v for k, v in comp.items())


def reward_features(ws_list, team='红'):
    """按局累加未加权组件 -> 每局一个特征向量(供 reward_model 逻辑回归学权重)。"""
    acc = {k: 0.0 for k in DEFAULT_CONF}
    for w in ws_list:
        comp = reward_components(w, team)
        for k in acc:
            acc[k] += comp[k]
    return acc


# 与本项目 schema 动作空间无关, 纯奖励工具; 暴露 MOVES 供外部对齐检查(避免死导入)。
MOVES = list(sl.MOVES)
