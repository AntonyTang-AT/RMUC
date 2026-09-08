# -*- coding: utf-8 -*-
"""M2 决策-胜势对齐度(纯 numpy, 无 torch): 模型/标签的 global_mode 是否落在恰当的胜势区间。

为什么做: retrain_eval(Track 1) 的承重判据 §4.2 —— "模型把进攻放在高胜势区间、防守放在低胜势区间"。
用逐状态有利度 `fav_from_worldstate`(win 相关组件和, 无窗、无 GPU)而非 Plan A 单点胜率机:
后者用 T 窗特征训练、喂单帧会塌到先验 sigmoid(b) 失去区分度(见 rollout-labeling 设计 §5)。

语义: 对每个决策点, 用其 global_mode(模型 decode 或标签)与红方有利度 fav 配对;
  aggreg: 按 mode 分组 fav 均值;
  gap = mean_fav(进攻点) − mean_fav(防守点)。期望 gap > 0(进攻落在更高胜势)。

★ 诚实前提: 关联性/条件有利度, 非因果。gap>0 只说明"该模式出现在对的胜势区间",
  不证明"是这些决策赢的"; 其可靠性地基见 outcome_backtest.cv_auc_win_model(状态确实携带胜负)。
纯 numpy + `eval.outcome_backtest.fav_from_worldstate`, 单测离线可跑。
"""
import numpy as np

from eval.outcome_backtest import fav_from_worldstate

# 判别用的两个模式; gap = 进攻 − 防守
POS = '进攻'
NEG = '防守'


def global_mode_align_gap(modes, worldstates, team='红'):
    """modes: list[str] 每决策点的 global_mode(模型预测或标签); worldstates: 对齐的 list[worldstate]。

    返回 {'mean_fav_by_mode': {mode: 均值}, 'counts': {mode: 数量}, 'gap': float|None}。
    gap = mean_fav(进攻) − mean_fav(防守); 任一缺失(无进攻或无防守点)则 None。
    """
    by = {}
    for m, w in zip(modes, worldstates):
        fav = fav_from_worldstate(w, team)
        by.setdefault(m, []).append(fav)
    means = {m: float(np.mean(v)) for m, v in by.items() if v}
    counts = {m: len(v) for m, v in by.items() if v}
    gap = None
    if POS in means and NEG in means:
        gap = round(means[POS] - means[NEG], 4)
    return {'mean_fav_by_mode': {m: round(v, 4) for m, v in means.items()},
            'counts': counts, 'gap': gap}


def gap_lte(meta_or_none, rhs=None):
    """便捷比较: 供 'rollout gap >= lookahead gap' 复判。空/None 视为不可比。"""
    g = meta_or_none.get('gap') if isinstance(meta_or_none, dict) else meta_or_none
    if g is None or rhs is None:
        return None
    return g >= rhs
