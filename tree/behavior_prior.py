# tree/behavior_prior.py
from collections import defaultdict

def state_sig(worldstate, mode):
    s = worldstate.get('robots', {}).get(7, {})
    zone = s.get('zone', 'Z--')
    return f"{mode}|{zone}"

def mine(trajectories, k=3):
    counts = defaultdict(int)
    for traj in trajectories:
        for sig, decision in traj:
            counts[(sig, decision)] += 1
    by_sig = defaultdict(list)
    for (sig, decision), c in counts.items():
        by_sig[sig].append((decision, c))
    prior = []
    for sig, items in by_sig.items():
        total = sum(c for _, c in items)
        items.sort(key=lambda x: -x[1])
        for decision, c in items[:k]:
            prior.append({'sig': sig, 'decision': decision, 'count': c, 'freq': c / total})
    return prior

def build_prior_context(prior, cur_sig):
    lines = []
    for p in prior:
        if p['sig'] == cur_sig:
            lines.append(f"行为树先验: 在 {p['sig']} 局势下, 历史决策倾向 {p['decision']} (频率 {p['freq']:.0%}, 样本 {p['count']})")
    return lines
