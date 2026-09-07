# eval/metrics.py
from model.prompt import VALID_STATES

def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    m = [[0] * n for _ in range(n)]
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            m[idx[a]][idx[b]] += 1
    return {labels[i]: m[i] for i in range(n)}

def legal_rate(results):
    if not results:
        return 0.0
    ok = sum(1 for r in results if r.get('state') in VALID_STATES)
    return ok / len(results)

def f1_macro(y_true, y_pred):
    # 在真实出现的标签上做宏平均, 无 scikit-learn 依赖
    labels = [l for l in VALID_STATES if l in set(y_true) or l in set(y_pred)]
    if not labels:
        return 0.0
    scores = []
    for l in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == l and b == l)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != l and b == l)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == l and b != l)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        scores.append(f)
    return sum(scores) / len(scores)
