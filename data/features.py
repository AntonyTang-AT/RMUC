# data/features.py
import math
from config import DF_STEP, TREND

ROBOT_TYPES = ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')
BUILDING_TYPES = ('前哨站', '基地')

def _zero_trend():
    return {'dx': 0.0, 'dy': 0.0, 'dhp': 0.0, 'dfire': 0.0}

def _row_to_state(row):
    return {
        'type': row['机器人类型'], 'team': row['阵营'], 'school': row['学校名'],
        'hp': row['当前血量'] or 0.0, 'maxhp': row['最大血量'] or 1.0,
        'x': row['x'] or 0.0, 'y': row['y'] or 0.0, 'z': row['z'] or 0.0,
        'yaw': row['枪口朝向'] or 0.0, 'power': row['底盘功率'] or 0.0,
        'small_heat': row['小热量'] or 0.0, 'big_heat': row['大热量'] or 0.0,
        'ammo17': row['累计17mm发弹'] or 0.0, 'ammo42': row['累计42mm发弹'] or 0.0,
        'vuln': row['是否易伤'] or 0,
        'team_coins_total': row['队伍总金币'] or 0.0,
        'team_coins_left': row['队伍剩余金币'] or 0.0,
    }

def build_decision_points(rows, game_id, step=DF_STEP, trend=TREND):
    # 按 robot 分组, 每 robot 按时刻排序的 list
    by_robot = {}
    for row in rows:
        by_robot.setdefault(row['robot_id'], []).append(row)
    for k, v in by_robot.items():
        v.sort(key=lambda r: r['时刻秒'])
    # 时间轴: 该局从 0 到 时长, 按 step 采决策点
    t_end = max(r['时刻秒'] for r in rows)
    t_end = float(math.ceil(t_end))
    times = [float(i) for i in range(0, int(t_end) + 1) if abs(i % step) < 1e-9]
    dps = []
    for t in times:
        robots = {}
        for rid, series in by_robot.items():
            # 找 <=t 的最近行, 以及 <=t-trend 的最近行(算趋势)
            cur = _closest(series, t, forward=False)
            if cur is None:
                continue
            base = _closest(series, t - trend, forward=False)
            s = _row_to_state(cur)
            s.update(_zero_trend())   # ensure dx/dy/dhp/dfire always present
            if base is not None:
                b = _row_to_state(base)
                s['dx'] = s['x'] - b['x']
                s['dy'] = s['y'] - b['y']
                s['dhp'] = s['hp'] - b['hp']
                s['dfire'] = (s['ammo17'] + s['ammo42']) - (b['ammo17'] + b['ammo42'])
            robots[rid] = s
        team = {}
        for rid, s in robots.items():
            tq = s['team']
            team.setdefault(tq, {'coins_total': s['team_coins_total'], 'coins_left': s['team_coins_left']})
        dps.append({'t': t, 'robots': robots, 'team': team})
    return dps

def _closest(series, t, forward=True):
    # series 已按 时刻秒 升序; 返回时刻最接近 t 且 <=t 的一行(时间近似)
    cand = None
    for r in series:
        if r['时刻秒'] <= t + 1e-9:
            cand = r
        else:
            break
    return cand if cand is not None else (series[0] if series else None)

def field_bounds(decision_points):
    xs, ys = [], []
    for dp in decision_points:
        for rid, r in dp['robots'].items():
            if r['type'] in BUILDING_TYPES:
                continue
            xs.append(r['x']); ys.append(r['y'])
    return {'xmin': min(xs), 'xmax': max(xs), 'ymin': min(ys), 'ymax': max(ys)}
