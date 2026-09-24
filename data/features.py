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

def build_decision_points(rows, game_id, step=DF_STEP, trend=TREND, start_time=None):
    # 按 robot 分组, 每 robot 按时刻排序的 list
    by_robot = {}
    for row in rows:
        by_robot.setdefault(row['robot_id'], []).append(row)
    for k, v in by_robot.items():
        v.sort(key=lambda r: r['时刻秒'])
    if not rows:
        return []
    # 时间轴从首个观测（或显式起点）开始，后续决策点按 step 对齐。
    first_observed = min(r['时刻秒'] for r in rows)
    t_end = max(r['时刻秒'] for r in rows)
    origin = float(first_observed if start_time is None else start_time)
    if origin > t_end:
        return []
    # Keep decision points inside the observed game interval.  A ceiling here
    # would fabricate a final point after the last replay observation.
    last_index = int(math.floor((t_end - origin) / step + 1e-9))
    times = [origin + i * step for i in range(last_index + 1)]
    dps = []
    for t in times:
        robots = {}
        for rid, series in by_robot.items():
            # 找 <=t 的最近行, 以及 <=t-trend 的最近行(算趋势)
            cur = _closest(series, t, forward=False)
            if cur is None:
                continue
            # Trend deltas must be based on observations inside [t-trend, t].
            # Use the earliest prior observation in that window; if the entity
            # has no prior sample in the window, leave its trend at zero.
            lower_bound = t - trend
            base = next((r for r in series
                         if lower_bound <= r['时刻秒'] < cur['时刻秒']), None)
            s = _row_to_state(cur)
            s['observed_at'] = float(cur['时刻秒'])
            s['age_seconds'] = float(t - cur['时刻秒'])
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

def _closest(series, t, forward=False):
    # series 已按 时刻秒 升序; 返回时刻最接近 t 且 <=t 的一行(时间近似)
    cand = None
    for r in series:
        if r['时刻秒'] <= t:
            cand = r
        else:
            break
    return cand

def field_bounds(decision_points):
    xs, ys = [], []
    for dp in decision_points:
        for rid, r in dp['robots'].items():
            if r['type'] in BUILDING_TYPES:
                continue
            xs.append(r['x']); ys.append(r['y'])
    return {'xmin': min(xs), 'xmax': max(xs), 'ymin': min(ys), 'ymax': max(ys)}
