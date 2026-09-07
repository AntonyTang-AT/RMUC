# labels/sentry_labels.py
from data.features import BUILDING_TYPES

SENTRY_STATES = ('def', 'hitbuilding', 'cruise')

def _sentry(dp, team, sentry_rid):
    return dp['robots'].get(sentry_rid)

def _midfield_x(bounds):
    return (bounds['xmin'] + bounds['xmax']) / 2.0

def _enemy_in_half(dp, bounds, team, sentry):
    # 敌方是否存在机器人进入"己方半场"(以中线为界)。team 红则己方 x 偏小侧。
    mid = _midfield_x(bounds)
    if team == '红':
        return any(r['team'] != '红' and r['x'] < mid for r in dp['robots'].values())
    else:
        return any(r['team'] != '蓝' and r['x'] > mid for r in dp['robots'].values())

def _sentry_pushing(dp, bounds, team, sentry):
    # 哨兵越过中线 (进入敌方半场)
    mid = _midfield_x(bounds)
    return (sentry['x'] > mid) if team == '红' else (sentry['x'] < mid)

def label_sentry(dp, bounds, team='红', sentry_rid=7):
    s = _sentry(dp, team, sentry_rid)
    if s is None:
        return 'cruise'
    if _sentry_pushing(dp, bounds, team, s) and (s.get('dfire', 0) or s.get('ammo17', 0) > 0):
        return 'hitbuilding'
    if _enemy_in_half(dp, bounds, team, s):
        return 'def'
    return 'cruise'

def label_sentry_series(dps, bounds, team='红', sentry_rid=7):
    out = []
    for dp in dps:
        out.append((dp['t'], label_sentry(dp, bounds, team=team, sentry_rid=sentry_rid)))
    return out

def smooth_sequence(labels, min_dur=5.0):
    if not labels:
        return []
    # collapse consecutive duplicate labels into runs [start_time, state, end_time]
    runs = []
    start, cur, end = labels[0][0], labels[0][1], labels[0][0]
    for t, lbl in labels[1:]:
        if lbl == cur:
            end = t
        else:
            runs.append([start, cur, end])
            cur, start, end = lbl, t, t
    runs.append([start, cur, end])
    # absorb runs shorter than min_dur into the previous run (keep prev label, extend its span); head never absorbed
    out = []
    for s, lbl, e in runs:
        if out and (e - s) < min_dur:
            ps, plbl, pe = out[-1]
            out[-1] = [ps, plbl, e]
        else:
            out.append([s, lbl, e])
    return [(s, lbl) for s, lbl, e in out]
