# -*- coding: utf-8 -*-
"""回看(look-ahead)策略标签 + 滞后一致率 —— Q2 的可落地实现。

思路(对应 Q2 提问): 用 [t, t+Δ] 一段"前向"连续数据(单位位置推进 + 建筑掉血)推断 t 时刻的
"真实短时策略"(sl.GLOBAL_MODES 之一), 作为**标注**; 再拿它对"几秒前(t-δ)的那次预测"做滞后
校验 —— 即预测当时是否准, 并不断校准。纯回放数据处理, 不加载模型/不吃 GPU; 未来可接 LLM 决策。

★ 局限(诚实标注):
   - 这是"短时行为锚定", 不是"最终胜负锚定"(后者在 eval/outcome_backtest)。
   - 策略本身有迟滞/歧义(敌后撤一次≠反攻), Δ 与阈值需消融, margin 默认 0.12 可调。
   - 无单一"观测真值", 标签是信号叠加的启发判据; 用作蒸馏伪标签要谨慎继承误差。
"""
from schema import layered as sl

# 权重/阈值(可消融)
W_PUSH = 1.0        # 位置推进(帧占比)
W_BLDG = 1.0        # 建筑掉血(hp_ratio 损失)
DEF_MARGIN = 0.12   # off-def 超过该值才判 进攻/防守, 否则 均衡


def _in_enemy_half(nx, team):
    """归一化 x -> 是否在敌方半场。红方主场=低x(中线 0.5), 蓝方=高x。"""
    return nx > 0.5 if team == '红' else nx < 0.5


def forward_strategy(ws_list, i, delta, team='红', margin=DEF_MARGIN,
                     w_push=W_PUSH, w_bldg=W_BLDG):
    """从前向窗口 [i, i+delta] 推断 t=i 的真实短时策略('进攻'/'均衡'/'防守')。
    ws_list: build_worldstate 序列(已对齐 bounds)。纯可观测轨迹, 非模型输出。
    进攻 = 己方单位进敌半场 + 敌建筑掉血; 防守 = 敌单位进己方半场 + 己建筑掉血。"""
    lo = i
    hi = min(i + delta, len(ws_list) - 1)
    n = hi - lo + 1
    if n <= 0:
        return '均衡'
    own_push = enemy_push = 0
    bldg_first, bldg_last = {}, {}
    for j in range(lo, hi + 1):
        w = ws_list[j]
        for r in w.get('robots', {}).values():
            nx = (r.get('norm_xy') or [0.5, 0.5])[0]
            if r.get('team') == team:
                if _in_enemy_half(nx, team):
                    own_push += 1
            else:
                if not _in_enemy_half(nx, team):   # 敌人进入己方半场
                    enemy_push += 1
        for bid, b in w.get('buildings', {}).items():
            key = (bid, b.get('team'))
            hp = b.get('hp_ratio', 1.0)
            bldg_first.setdefault(key, hp)
            bldg_last[key] = hp
    off = w_push * (own_push / n)
    deff = w_push * (enemy_push / n)
    for key, hp0 in bldg_first.items():
        loss = max(0.0, hp0 - bldg_last.get(key, hp0))
        if key[1] == team:
            deff += w_bldg * loss        # 己方建筑掉血 -> 敌方进攻
        else:
            off += w_bldg * loss         # 敌方建筑掉血 -> 己方进攻
    if off - deff > margin:
        return '进攻'
    if deff - off > margin:
        return '防守'
    return '均衡'


def strategy_series(ws_list, delta, team='红', margin=DEF_MARGIN):
    """全序列[0..len-1] 的逐点回看标签。"""
    return [forward_strategy(ws_list, i, delta, team, margin) for i in range(len(ws_list))]


def lag_consistency(preds, labels, horizon):
    """滞后一致率。preds[i]=i 时刻的预测; labels[i]=回看推断的 i 时刻真实策略。
    对照 preds[i-horizon](horizon 帧前那次预测) 与 labels[i](curr 真实) -> 一致率。
    horizon=0 即"当期一致"基线; horizon>0 度量"几秒前预测是否仍准"的衰减。"""
    hits = n = 0
    for i in range(len(labels)):
        j = i - horizon
        if j < 0 or j >= len(preds):
            continue
        n += 1
        hits += (preds[j] == labels[i])
    return {'rate': round(hits / max(n, 1), 4), 'n': n, 'horizon': horizon}


def lag_curve(preds, labels, horizons=(0, 1, 2, 3, 5, 8)):
    """不同滞后帧的滞后一致率曲线。"""
    return [lag_consistency(preds, labels, h) for h in horizons]


# ---- 哨兵三态(def/cruise/hitbuilding)的回看窗口判定 ----
# 窗口优先序: def(敌进己半场, 最需警觉) > hitbuilding(哨兵越线+开火) > cruise。
def sentry_state_from_window(ws_list, i, delta, team='红'):
    """从未来窗口 [i, i+delta] 判定 t=i 的"真实哨兵状态"(def/cruise/hitbuilding)。

    与瞬时 label_sentry 的区别: 用未来 Δ 帧内**是否出现**该特征, 而非只看当前一帧。
    敌方单位(排除建筑)在窗口内任一帧进己方半场 -> def; 否则若哨兵越线且开火 -> hitbuilding;
    否则 cruise。忠于口径要求的第三态 hitbuilding(global_mode->sentry 的 2 态映射会丢掉它)。"""
    lo, hi = i, min(i + delta, len(ws_list) - 1)
    enemy_seen = push_fire_seen = False
    for j in range(lo, hi + 1):
        w = ws_list[j]
        for r in w.get('robots', {}).values():
            if r.get('team') != team:
                if not _in_enemy_half((r.get('norm_xy') or [0.5, 0.5])[0], team):
                    enemy_seen = True
            elif r.get('type') == '哨兵':
                nx = (r.get('norm_xy') or [0.5, 0.5])[0]
                if _in_enemy_half(nx, team) and r.get('dfire', 0) > 0:
                    push_fire_seen = True
    if enemy_seen:
        return 'def'
    if push_fire_seen:
        return 'hitbuilding'
    return 'cruise'


def sentry_state_series(ws_list, delta, team='红'):
    """全序列逐点回看哨兵三态。"""
    return [sentry_state_from_window(ws_list, i, delta, team) for i in range(len(ws_list))]


def relabel_decision_series(ws_list, decisions, delta, team='红', margin=DEF_MARGIN):
    """回看重标注决策系列: 每决策点 global_mode <- 前向窗口真实策略; 再按该 mode 用
    derive_unit_intent 重派生全部单位(整体自洽)。

    用途: 让训练数据的状态(global_mode + 各单位动作)反映"真实短时策略", 而非瞬时逐帧规则。
    ws_list: build_worldstate 序列; decisions: 与 ws_list 等长的分层决策。
    返回 (relabeled_decisions, labels); labels[i]=全局模式(供分布/校准)。"""
    from tree import global_mode as gm
    labels = [forward_strategy(ws_list, i, delta, team, margin) for i in range(len(ws_list))]
    out = []
    for i, (mode, d) in enumerate(zip(labels, decisions)):
        if d is None:
            continue
        w = ws_list[i]
        nd = dict(d)                 # 顶层拷贝
        nd['global_mode'] = mode
        units = {}
        for cls in sl.UNIT_CLASSES:
            unit = next((r for r in w.get('robots', {}).values()
                         if r.get('team') == team and r.get('type') == cls), None)
            if unit is None:
                units[cls] = dict(d['units'].get(cls, {'target': '无', 'move': '无',
                                                       'fire': False, 'risk': '低'}))
            else:
                units[cls] = gm.derive_unit_intent(unit, w, mode, team=team)
        nd['units'] = units
        out.append(nd)
    return out, labels
