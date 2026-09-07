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
    strategy = regret_matching(matrix, iterations)
    # 学到的混合策略给出"我方倾向哪一意图"分布, argmax 即最优先意图
    idx = max(range(len(strategy)), key=lambda i: strategy[i])
    return GLOBAL_MODES[idx]
