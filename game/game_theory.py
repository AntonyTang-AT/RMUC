from schema.layered import GLOBAL_MODES

# 默认对称 3 意图 payoff (我方行, 对方列)。进攻 vs 防守 高打低/低打高 有收益差; 同类对拼近零和。
# 注: spec §9 指出真实 payoff 需依赖 RL/对手模型估计, 此为手设先验默认值。
PAYOFF_3 = [
    [-0.5, 0.3, 0.8],    # 我方进攻
    [0.0, 0.0, 0.2],     # 我方均衡
    [-0.3, -0.5, 0.5],   # 我方防守
]

def regret_matching(matrix, iterations=500):
    n = len(matrix)
    regret = [0.0] * n
    strat_sum = [0.0] * n
    strategy = [1.0 / n] * n
    for _ in range(iterations):
        payoff = [sum(matrix[i][j] * strategy[j] for j in range(n)) for i in range(n)]
        avg = sum(payoff) / n
        regret = [r + (payoff[i] - avg) for i, r in enumerate(regret)]
        pos = [max(r, 0.0) for r in regret]
        total = sum(pos)
        if total > 0:
            strategy = [p / total for p in pos]
        else:
            strategy = [1.0 / n] * n
        for i in range(n):
            strat_sum[i] += strategy[i]
    total = sum(strat_sum)
    return [s / total for s in strat_sum]

def best_response(matrix, opp_strategy):
    exp = [sum(matrix[i][j] * opp_strategy[j] for j in range(len(matrix))) for i in range(len(matrix))]
    return max(range(len(exp)), key=lambda i: exp[i])

def choose_mode(matrix, opp_estimate, iterations=500):
    # spec §6.1: 对抗推演对手最优反应。有对手估计时用 best_response 做纯反应,
    # 无对手模型(opp_estimate=None)时回退到 regret-matching 自学习分布的 argmax。
    if opp_estimate is not None:
        return GLOBAL_MODES[best_response(matrix, opp_estimate)]
    strategy = regret_matching(matrix, iterations)
    idx = max(range(len(strategy)), key=lambda i: strategy[i])
    return GLOBAL_MODES[idx]

def choose_global_mode(worldstate, opp_estimate=None):
    """M3 慢环全局模式: 有对手模型 -> 博弈 best_response(强对抗推演); 否则**回退局势感知规则**。

    PAYOFF_3 在均匀对手下恒落 row0(进攻), 导致 mode_counts 恒一类、行为先验频率退化。
    无对手模型时用 derive_global_mode(读世界状态, 会变)作上下文驱动, 修复病态; 对手模型就绪时仍走博弈。
    """
    if opp_estimate is not None:
        return choose_mode(PAYOFF_3, opp_estimate)
    from tree import global_mode as gm
    return gm.derive_global_mode(worldstate)
