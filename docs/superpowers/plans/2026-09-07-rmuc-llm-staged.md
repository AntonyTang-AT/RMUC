# RMUC 2026 大模型局势分析 — M0→M3 分阶段实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从官方全场数据构建 WorldState(Zone语义化)，用大模型慢环做战略规划 + 行为树快环做实时决策，产出哨兵 def/hitbuilding/cruise 及全队分层结构化宏观决策。

**Architecture:** 快慢两环 + 三引擎(LLM/博弈论/RL)分层。离线侧(基于 613 局 sqlite)构建 `数据→特征采样→WorldState(Zone)→哨兵规则标签` 管线，跑通 LLM 提示词基线；在线侧用行为树/规则快环保证毫秒级响应，RL(PettingZoo)与博弈论增强策略与标签。计划按里程碑 M0→M3 分期。

**Tech Stack:** Python3、sqlite3(内置)、numpy、matplotlib、scikit-learn、pandas/polars；LLM 用 torch+transformers+peft(微调 Qwen2.5-3B/1.5B)；RL 用 PettingZoo+stable-baselines3(PPO)。M0/M1 仅依赖：sqlite3、numpy、matplotlib、scikit-learn。

**Spec:** `docs/superpowers/specs/2026-09-07-rmuc-llm-situation-analysis-design.md`

## 范围说明（scope check）

spec 覆盖**多个独立子系统**。本轮把 **M0、M1 完整落地**为可执行的 bite-sized TDD 任务；**M2(RL)、M3(博弈论)** 是独立子系统，规划为**子计划大纲**（给出文件结构 + 对 M0/M1 的接口契约 + 任务清单），建议在 M0/M1 交付后各自展开为独立 plan 文档再执行。当前计划从 M0 开始即产生可运行、可测试的软件。

---

## Global Constraints

- 环境：Windows 11, Python 3，bash（Git Bash）。中文列名在 bash 直接查询会乱码，**所有 SQL 用 Python(sqlite3) 执行**，不直接用 `sqlite3` CLI 跑含中文列名的语句。
- 数据库只读访问：`数据与资料/rmuc_2026_region_dataset.sqlite`。三表 `matches/timeseries/events`，用 `game_id` 关联。
- 字段名（从表读取原样）：时刻秒、robot_id、机器人类型、阵营、学校名、当前血量、最大血量、x、y、z、枪口朝向、底盘功率、小热量、累计17mm发弹、累计42mm发弹、队伍总金币、队伍剩余金币、是否易伤。
- 决策粒度：`step=3.0`s，趋势窗 `trend=15`s（conf 常量）。
- 输出哨兵状态枚举：`def` / `hitbuilding` / `cruise`。
- 测试框架 pytest。依赖最小化：M0/M1 只用 sqlite3/numpy/matplotlib/scikit-learn。
- 数据关键坑：**建筑(前哨站/基地)坐标全为 0**、**受击事件无攻击方字段**、单局坐标干净但全局有基准偏差 → Zone 用"单局相对坐标归一化"，不能用建筑距离。
- 每任务末 commit。项目当前已 `git init` 需确认（若未，首任务初始化）。

---

# Phase 0 — M0：单局数据管线 + Zone + 哨兵规则标签 + 可视化

目标：对单个 `game_id` 跑通 `①→②→③`，画出哨兵状态随时间条带图，确认 Zone/标签合理。以下所有任务基于同一局：`GAME_REF = 1779323658229`（东部第1场第1局）。

### Task 0.1: 仓库脚手架 + DB 只读访问层

**Files:**
- Create: `requirements.txt`
- Create: `data/__init__.py`（空）
- Create: `data/db.py`
- Test: `tests/test_db.py`
- Modify: `config.py`（创建，含常量）

**Interfaces:**
- Produces:
  - `config.DB_PATH: str`（绝对路径）
  - `config.DF_STEP: float = 3.0`、`config.TREND: float = 15.0`、`config.GAME_REF: int = 1779323658229`
  - `data.db.connect() -> sqlite3.Connection`（`row_factory=sqlite3.Row`）
  - `data.db.timeseries_for(con, game_id:int) -> list[sqlite3.Row]`
  - `data.db.events_for(con, game_id:int) -> list[sqlite3.Row]`
  - `data.db.match_meta(con, game_id:int) -> sqlite3.Row | None`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_db.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db

def test_connect_and_meta():
    con = db.connect()
    meta = db.match_meta(con, config.GAME_REF)
    assert meta is not None
    assert meta['game_id'] == config.GAME_REF
    assert meta['红方学校'] and meta['蓝方学校']
    con.close()

def test_timeseries_nonempty():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    assert len(rows) > 1000
    con.close()

def test_events_for_nonempty():
    con = db.connect()
    rows = db.events_for(con, config.GAME_REF)
    assert len(rows) > 0
    con.close()
```

- [ ] **Step 2: 运行确认失败**

```bash
cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_db.py -q
```
Expected: FAIL（`ModuleNotFoundError: No module named '../data'` 或 import 错误）。

- [ ] **Step 3: 写最小实现**

`config.py`
```python
import os
BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, '数据与资料', 'rmuc_2026_region_dataset.sqlite')
DF_STEP = 3.0
TREND = 15.0
GAME_REF = 1779323658229
```

`data/db.py`
```python
import sqlite3
from config import DB_PATH

def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def timeseries_for(con, game_id):
    return con.execute("SELECT * FROM timeseries WHERE game_id=?", (game_id,)).fetchall()

def events_for(con, game_id):
    return con.execute("SELECT * FROM events WHERE game_id=?", (game_id,)).fetchall()

def match_meta(con, game_id):
    return con.execute("SELECT * FROM matches WHERE game_id=?", (game_id,)).fetchone()
```

- [ ] **Step 4: 运行确认通过**

```bash
cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_db.py -q
```
Expected: PASS（3 passed）。

- [ ] **Step 5: 初始化 git（若未）并提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git init -q 2>/dev/null; printf '数据与资料/*.sqlite\n_work/\n__pycache__/\n*.pyc\n' > .gitignore; git add config.py data/ tests/ requirements.txt .gitignore; git commit -m "feat(m0): db read-only access layer + scaffold"
```

---

### Task 0.2: 特征采样 —— build_decision_points

**Files:**
- Create: `data/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `data.db.timeseries_for(con, game_id) -> list[Row]`（Task 0.1）
- Produces:
  - `data.features.ROBOT_TYPES: tuple[str,...]`（地面+空中）
  - `data.features.BUILDING_TYPES: tuple[str,...] = ('前哨站','基地')`
  - `data.features.build_decision_points(rows, game_id:int, step:float=config.DF_STEP, trend:float=config.TREND) -> list[dict]`
    每个 dict：`{'t':float, 'robots':dict[int,dict], 'team':{'红':{...},'蓝':{...}}}`
    robot dict 键：`type,team,school,hp,maxhp,x,y,z,yaw,power,small_heat,big_heat,ammo17,ammo42,vuln,dx,dy,dhp,dfire`
    team dict（该决策点）键：`coins_total, coins_left`（红/蓝各自）
  - `data.features.field_bounds(decision_points) -> dict`：`{'xmin','xmax','ymin','ymax'}`（用 robots 坐标，忽略建筑）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_features.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features

def test_build_decision_points_structure():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF, step=3.0, trend=15.0)
    con.close()
    assert len(dps) > 50
    dp = dps[0]
    assert dp['t'] == 0.0
    assert 'robots' in dp and 'team' in dp
    # 红哨兵 robot_id=7 应在
    assert 7 in dp['robots']
    r = dp['robots'][7]
    for k in ('type','team','hp','maxhp','x','y','dx','dy','dhp','dfire'):
        assert k in r

def test_decision_point_t_spacing():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF, step=3.0, trend=15.0)
    con.close()
    ts = [d['t'] for d in dps]
    assert all(abs(b - a - 3.0) < 1e-6 for a, b in zip(ts, ts[1:]))

def test_field_bounds():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    b = features.field_bounds(dps)
    con.close()
    assert b['xmin'] < b['xmax'] and b['ymin'] < b['ymax']
```

- [ ] **Step 2: 运行确认失败**

```bash
cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_features.py -q
```
Expected: FAIL（`AttributeError: module 'data' has no attribute 'features'` 或 `build_decision_points` 不存在）。

- [ ] **Step 3: 写最小实现**

```python
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
    if cand is None:
        return None
    return cand
```

> 简化说明：`_closest` 取 `<=t` 的最近一行（机器人从开局起有数据则稳定）。趋势用 `t-trend` 与 `t` 差分。后续可改用插值。

```python
# data/features.py 追加
def field_bounds(decision_points):
    xs, ys = [], []
    for dp in decision_points:
        for rid, r in dp['robots'].items():
            if r['type'] in BUILDING_TYPES:
                continue
            xs.append(r['x']); ys.append(r['y'])
    return {'xmin': min(xs), 'xmax': max(xs), 'ymin': min(ys), 'ymax': max(ys)}
```

- [ ] **Step 4: 运行确认通过**

```bash
cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_features.py -q
```
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add data/ tests/ && git commit -m "feat(m0): build_decision_points + field_bounds"
```

---

### Task 0.3: Zone 划分与归属（provisional 语义网格）

**Files:**
- Create: `worldstate/__init__.py`（空）
- Create: `worldstate/zone.py`
- Test: `tests/test_zone.py`

**Interfaces:**
- Consumes: `data.features.field_bounds(dps) -> dict`
- Produces:
  - `worldstate.zone.ZONE_COLS: int = 4`、`worldstate.zone.ZONE_ROWS: int = 3`
  - `worldstate.zone.zone_index(x, y, bounds) -> tuple[int,int]`（列/行，0-based，钳制到 [0,ZONE_COLS-1]/[0,ZONE_ROWS-1]）
  - `worldstate.zone.zone_label(ci, ri, team) -> str`：返回语义标签（如 `Z{ci}{ri}`, 附加阵营侧语义，M0 用 `Z{ri}{ci}` 即够）
  - `worldstate.zone.assign_zone(x, y, bounds, team) -> str`
  - `worldstate.zone.ZONE_LABELS: list[str]`（全部可取值，供校验）

> M0 用归一化网格作**占位分区**；spec §3.2 的"场地先验骨架 + 行为事件校准"（不规则多边形）留待后续校准任务，不阻塞 M0。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_zone.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from worldstate import zone

def test_zone_index_bounds():
    b = {'xmin': 0.0, 'xmax': 10.0, 'ymin': 0.0, 'ymax': 10.0}
    assert zone.zone_index(0, 0, b) == (0, 0)
    assert zone.zone_index(10, 10, b) == (zone.ZONE_COLS - 1, zone.ZONE_ROWS - 1)
    # 越界钳制
    assert zone.zone_index(-5, 20, b) == (0, zone.ZONE_ROWS - 1)

def test_assign_zone_returns_valid_label():
    b = {'xmin': 0.0, 'xmax': 10.0, 'ymin': 0.0, 'ymax': 10.0}
    lbl = zone.assign_zone(5.0, 5.0, b, '红')
    assert lbl in zone.ZONE_LABELS

def test_zone_labels_unique():
    assert len(set(zone.ZONE_LABELS)) == len(zone.ZONE_LABELS)
```

- [ ] **Step 2: 运行确认失败**

Expected: FAIL（`module 'worldstate' has no attribute 'zone'`）。

- [ ] **Step 3: 写最小实现**

```python
# worldstate/zone.py
from data.features import BUILDING_TYPES

ZONE_COLS = 4
ZONE_ROWS = 3

def zone_index(x, y, bounds):
    span_x = bounds['xmax'] - bounds['xmin']
    span_y = bounds['ymax'] - bounds['ymin']
    span_x = span_x if span_x > 1e-9 else 1e-9
    span_y = span_y if span_y > 1e-9 else 1e-9
    ci = int((x - bounds['xmin']) / span_x * ZONE_COLS)
    ri = int((y - bounds['ymin']) / span_y * ZONE_ROWS)
    return (max(0, min(ci, ZONE_COLS - 1)), max(0, min(ri, ZONE_ROWS - 1)))

def zone_label(ci, ri, team):
    return f"Z{ri}{ci}"

def assign_zone(x, y, bounds, team):
    ci, ri = zone_index(x, y, bounds)
    return zone_label(ci, ri, team)

ZONE_LABELS = sorted({zone_label(ci, ri, '红') for ci in range(ZONE_COLS) for ri in range(ZONE_ROWS)})
```

- [ ] **Step 4: 运行确认通过**

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add worldstate/ tests/ && git commit -m "feat(m0): provisional tactical grid zones"
```

---

### Task 0.4: 哨兵三分类规则标签（含平滑）

**Files:**
- Create: `labels/__init__.py`（空）
- Create: `labels/sentry_labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `data.features`（decision point / field_bounds）、`worldstate.zone`
- Produces:
  - `labels.sentry_labels.SENTRY_STATES = ('def','hitbuilding','cruise')`
  - `labels.sentry_labels.label_sentry(dp, bounds, team='红', sentry_rid=7) -> str`
  - `labels.sentry_labels.label_sentry_series(dps, bounds, team='红', sentry_rid=7) -> list[tuple[float,str]]`
  - `labels.sentry_labels.smooth_sequence(labels:list[tuple[float,str]], min_dur:float=5.0) -> list[tuple[float,str]]`

> 标签规则（M0 启发式，spec §5.1）：
> - hitbuilding：哨兵 Zone 在敌方前哨站/基地区（用 sentry x 位于己方前哨站侧或越中线近似）且 (哨兵开火 dfire>0 或 敌方建筑受击时窗) 
>   —— M0 简化：哨兵**越过中线压制**（x 高于中场阈值）且近期开火 → hitbuilding。
> - def：敌方单位逼近己方基地/前哨站（敌单位 x 进入己方半场）且哨兵留守 → def。
> - cruise：else。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_labels.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features
from worldstate import zone
from labels import sentry_labels as sl

def _dps():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    con.close()
    return dps

def test_label_sentry_returns_valid():
    dps = _dps()
    b = features.field_bounds(dps)
    lbl = sl.label_sentry(dps[0], b, team='红', sentry_rid=7)
    assert lbl in sl.SENTRY_STATES

def test_label_series_grows_and_valid():
    dps = _dps()
    b = features.field_bounds(dps)
    series = sl.label_sentry_series(dps[:20], b, team='红', sentry_rid=7)
    assert len(series) == min(20, len(dps))
    assert all(s[1] in sl.SENTRY_STATES for s in series)

def test_smooth_sequence_removes_flicker():
    raw = [(float(i), ('cruise' if i % 2 == 0 else 'def')) for i in range(10)]
    out = sl.smooth_sequence(raw, min_dur=5.0)
    # 单帧抖动(时长<5)应被合并
    assert len(out) < len(raw)
    # 首尾时间覆盖
    assert out[0][0] == raw[0][0] and out[-1][0] == raw[-1][0]
```

- [ ] **Step 2: 运行确认失败**

Expected: FAIL（`import` 错误 / `SENTRY_STATES` 不存在）。

- [ ] **Step 3: 写最小实现**

```python
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
    groups = []
    cur_lbl, start, end = labels[0][1], labels[0][0], labels[0][0]
    for t, lbl in labels[1:]:
        if lbl == cur_lbl:
            end = t
        else:
            if (end - start) >= min_dur:
                groups.append((start, cur_lbl, end))
            cur_lbl, start, end = lbl, t, t
    groups.append((start, cur_lbl, end))
    # 合并过短段->合并到前一段的标签(简单向后填充)
    merged = []
    for start, lbl, end in groups:
        if merged and (end - start) < min_dur:
            # 并入前一段
            merged[-1] = (merged[-1][0], merged[-1][1], end)
        else:
            merged.append((start, lbl, end))
    return [(s, lbl) for s, lbl, e in merged]
```

- [ ] **Step 4: 运行确认通过**

Expected: PASS（若 `test_smooth_sequence` 断言过严可调 min_dur 或断言为 `<=`）。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add labels/ tests/ && git commit -m "feat(m0): sentry rule labels + smoothing"
```

---

### Task 0.5: WorldState 构建（含 Zone 语义 + 归一化坐标）

**Files:**
- Create: `worldstate/worldstate.py`
- Test: `tests/test_worldstate.py`

**Interfaces:**
- Consumes: `data.features`、`worldstate.zone`
- Produces:
  - `worldstate.worldstate.accept_targets: set[str]` = 合法目标对象集合（如 `{'敌方哨兵','敌方英雄','敌方前哨站','敌方基地','能量机关','飞镖','无'}`, M0 非空占位即可）
  - `worldstate.worldstate.build_worldstate(dp, bounds) -> dict`：每决策点 JSON 化。返回：
    ```python
    {
      't': float,
      'team': {'红': {'coins':..}, '蓝': {...}},
      'robots': {rid: {
          'type', 'team', 'zone', 'norm_xy':[nx,ny],
          'hp_ratio', 'power', 'heat', 'vuln', 'ammo17', 'ammo42', 'dfire'
      }},
      'buildings': {rid: {'type','team','hp_ratio'}},
    }
    ```
  - `worldstate.worldstate.normalize_xy(x, y, bounds) -> tuple[float,float]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_worldstate.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features
from worldstate import worldstate as ws

def test_normalize_xy_clamps():
    b = {'xmin': 0.0, 'xmax': 10.0, 'ymin': 0.0, 'ymax': 10.0}
    nx, ny = ws.normalize_xy(5.0, 5.0, b)
    assert 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
    nx2, _ = ws.normalize_xy(-100.0, 5.0, b)
    assert nx2 == 0.0

def test_build_worldstate_shape():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    con.close()
    b = features.field_bounds(dps)
    w = ws.build_worldstate(dps[0], b)
    assert 't' in w and 'robots' in w and 'team' in w
    s7 = w['robots'][7]
    assert s7['zone'] in ws.ZONE_LABELS
    assert len(s7['norm_xy']) == 2
    assert 0.0 <= s7['hp_ratio'] <= 1.0
    assert s7['team'] == '红'
```

（注：`ws.ZONE_LABELS` 从 worldstate.zone 再导出）

- [ ] **Step 2: 运行确认失败**

Expected: FAIL（`build_worldstate` 不存在）。

- [ ] **Step 3: 写最小实现**

```python
# worldstate/worldstate.py
from data.features import BUILDING_TYPES
from worldstate import zone as zn

ZONE_LABELS = zn.ZONE_LABELS
accept_targets = {'敌方哨兵', '敌方英雄', '敌方步兵', '敌方前哨站', '敌方基地', '能量机关', '飞镖', '无'}

def normalize_xy(x, y, bounds):
    span_x = bounds['xmax'] - bounds['xmin']
    span_y = bounds['ymax'] - bounds['ymin']
    span_x = span_x if span_x > 1e-9 else 1e-9
    span_y = span_y if span_y > 1e-9 else 1e-9
    nx = max(0.0, min(1.0, (x - bounds['xmin']) / span_x))
    ny = max(0.0, min(1.0, (y - bounds['ymin']) / span_y))
    return (nx, ny)

def build_worldstate(dp, bounds):
    robots = {}
    buildings = {}
    for rid, r in dp['robots'].items():
        nx, ny = normalize_xy(r['x'], r['y'], bounds)
        entry = {
            'type': r['type'], 'team': r['team'],
            'zone': zn.assign_zone(r['x'], r['y'], bounds, r['team']),
            'norm_xy': [round(nx, 3), round(ny, 3)],
            'hp_ratio': round(r['hp'] / r['maxhp'], 3) if r['maxhp'] else 0.0,
            'power': round(r['power'], 1), 'heat': round(r['small_heat'] + r['big_heat'], 1),
            'vuln': r['vuln'], 'ammo17': r['ammo17'], 'ammo42': r['ammo42'],
            'dfire': round(r.get('dfire', 0.0), 1),
        }
        if r['type'] in BUILDING_TYPES:
            buildings[rid] = {'type': r['type'], 'team': r['team'], 'hp_ratio': entry['hp_ratio']}
        else:
            robots[rid] = entry
    team = {k: {'coins': v['coins_left']} for k, v in dp['team'].items()}
    return {'t': dp['t'], 'team': team, 'robots': robots, 'buildings': buildings}
```

- [ ] **Step 4: 运行确认通过**

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add worldstate/ tests/ && git commit -m "feat(m0): worldstate builder with zone + norm coords"
```

---

### Task 0.6: M0 端到端脚本 + 哨兵状态条带可视化

**Files:**
- Create: `eval/__init__.py`（空）
- Create: `eval/viz.py`
- Create: `run_m0.py`
- Test: `tests/test_viz.py`
- Modify: `requirements.txt`（加 matplotlib）

**Interfaces:**
- Consumes: `data`/`features`/`zone`/`sentry_labels`/`worldstate`
- Produces:
  - `eval.viz.plot_state_strip(t_axis:list[float], labels:list[str], path:str)`：画"状态随时间"条带图（def/hitbuilding/cruise 三段着色）。
  - `eval.viz.plot_heatmap(xs:list[float], ys:list[float], path:str)`：单位位置热力图。
  - `run_m0.py` CLI：`python run_m0.py [--game GAME_REF] [--out out_m0]` 生成 `out_m0/sentry_strip.png`、`out_m0/heatmap.png`、`out_m0/report.txt`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_viz.py
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import viz

def test_plot_state_strip_writes_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'strip.png')
        viz.plot_state_strip([0.0, 3.0, 6.0], ['def', 'hitbuilding', 'cruise'], p)
        assert os.path.exists(p) and os.path.getsize(p) > 0

def test_plot_heatmap_writes_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'heat.png')
        viz.plot_heatmap([0.0, 1.0, 2.0], [0.0, 0.5, 1.0], p)
        assert os.path.exists(p) and os.path.getsize(p) > 0
```

- [ ] **Step 2: 运行确认失败**

Expected: FAIL（import 错误）。

- [ ] **Step 3: 写最小实现**

`eval/viz.py`
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

STATE_COLORS = {'def': '#1e88e5', 'hitbuilding': '#e53935', 'cruise': '#43a047'}

def plot_state_strip(t_axis, labels, path):
    fig, ax = plt.subplots(figsize=(14, 2.5))
    for i, (t, lbl) in enumerate(zip(t_axis, labels)):
        ax.barh(0, 1, left=t, height=1.0, color=STATE_COLORS[lbl], edgecolor='none')
    ax.set_yticks([])
    ax.set_xlabel('time (s)')
    ax.set_xlim(0, max(t_axis) + 1 if t_axis else 1)
    ax.set_title('sentry state over time')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in STATE_COLORS.items()], loc='upper right')
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_heatmap(xs, ys, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, ys, s=2, alpha=0.3)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.axis('equal'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path); plt.close()
```

`run_m0.py`
```python
# -*- coding: utf-8 -*-
import os, argparse, json
import config
from data import db, features
from worldstate import zone, worldstate as ws
from labels import sentry_labels as sl
from eval import viz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', type=int, default=config.GAME_REF)
    ap.add_argument('--out', type=str, default='out_m0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = db.connect()
    rows = db.timeseries_for(con, args.game)
    meta = db.match_meta(con, args.game)
    con.close()
    dps = features.build_decision_points(rows, args.game)
    b = features.field_bounds(dps)
    series = sl.label_sentry_series(dps, b, team='红', sentry_rid=7)
    strip = sl.smooth_sequence(series, min_dur=5.0)
    viz.plot_state_strip([t for t, _ in strip], [l for _, l in strip],
                         os.path.join(args.out, 'sentry_strip.png'))
    xs = [r['x'] for dp in dps for r in dp['robots'].values() if r['type'] not in features.BUILDING_TYPES]
    ys = [r['y'] for dp in dps for r in dp['robots'].values() if r['type'] not in features.BUILDING_TYPES]
    viz.plot_heatmap(xs, ys, os.path.join(args.out, 'heatmap.png'))
    # 统计
    counts = {k: 0 for k in sl.SENTRY_STATES}
    for _, l in strip: counts[l] += 1
    w0 = ws.build_worldstate(dps[0], b)
    report = {
        'game': args.game, 'meta': dict(meta) if meta else None,
        'decision_points': len(dps), 'zone_bounds': b,
        'sentry_label_counts': counts,
        'sample_worldstate': w0,
    }
    with open(os.path.join(args.out, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"[M0] wrote {args.out}/sentry_strip.png, heatmap.png, report.txt")
    print('sentry label counts:', counts)

if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行确认通过 + 手动跑一次**

```bash
cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_viz.py -q && python run_m0.py
```
Expected: PASS + `[M0] wrote out_m0/...` + 打印 label counts。人工查看 `out_m0/sentry_strip.png` 确认三状态分布合理。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add eval/ run_m0.py tests/ requirements.txt && git commit -m "feat(m0): single-game pipeline + sentry state strip viz"
```

---

# Phase 1 — M1：慢环提示词基线 + 快环行为树 + 评测

目标：LLM 慢环对 `WorldState→哨兵3状态` 做推理（提示词基线，已达达线）；接快环行为树+可行性校验；评测准确率/F1/合法率/时延。

### Task 1.1：提示词构建 + 决策解析

**Files:**
- Create: `model/__init__.py`（空）
- Create: `model/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Consumes: `worldstate.worldstate.build_worldstate -> dict`
- Produces:
  - `model.prompt.build_prompt(worldstate:dict, few_shot:list[tuple[dict,str]]|None=None) -> str`
    （中文 prompt，含 WorldState JSON + few-shot 示例 + 要求输出 JSON。）
  - `model.prompt.parse_decision(text:str) -> dict`：
    ```python
    {'state': 'def'|'hitbuilding'|'cruise'}   # 默认 'cruise'，非法回退
    ```
  - `model.prompt.VALID_STATES = ('def','hitbuilding','cruise')`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_prompt.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import prompt

def test_build_prompt_contains_worldstate_and_states():
    ws = {'t': 0.0, 'team': {'红': {'coins': 100}}, 'robots': {7: {'type':'哨兵','team':'红','zone':'Z00','norm_xy':[0.1,0.2],'hp_ratio':1.0,'vuln':0,'dfire':0}}, 'buildings': {}}
    p = prompt.build_prompt(ws)
    assert '哨兵' in p and 'def' in p and 'hitbuilding' in p and 'cruise' in p
    assert 'Z00' in p

def test_parse_decision_recovers_state():
    o = prompt.parse_decision('{"state": "hitbuilding"}')
    assert o['state'] == 'hitbuilding'

def test_parse_decision_falls_back():
    o = prompt.parse_decision('???')   # 非法 JSON
    assert o['state'] == 'cruise'
```

- [ ] **Step 2: 运行确认失败**

Expected: FAIL（import 错误）。

- [ ] **Step 3: 写最小实现**

```python
# model/prompt.py
import json
VALID_STATES = ('def', 'hitbuilding', 'cruise')

SYSTEM = (
    "你是 RoboMaster 比赛局势分析专家。给定全场 WorldState(JSON)，请判断己方哨兵的宏观决策——只输出一个状态："
    "def(守基地) / hitbuilding(攻击敌方建筑) / cruise(巡逻)。"
    "严格输出 JSON：{\"state\": \"<def|hitbuilding|cruise>\"}。仅输出 JSON，不要多余文字。"
)

def build_prompt(worldstate, few_shot=None):
    parts = [SYSTEM]
    if few_shot:
        parts.append("示例：")
        for ws, lbl in few_shot:
            parts.append(f"输入: {json.dumps(ws, ensure_ascii=False)}\n输出: {{\"state\": \"{lbl}\"}}")
    parts.append(f"当前: {json.dumps(worldstate, ensure_ascii=False)}")
    return "\n\n".join(parts)

def parse_decision(text):
    try:
        data = json.loads(text)
        s = str(data.get('state', '')).strip().lower()
        if s in VALID_STATES:
            return {'state': s}
    except Exception:
        pass
    return {'state': 'cruise'}
```

可选：从 text 提取第一个 `{...}` JSON 子串再解析（鲁棒性）。M1 先保简单。

- [ ] **Step 4: 运行确认通过**

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add model/ tests/ && git commit -m "feat(m1): prompt builder + decision parser"
```

---

### Task 1.2：快环行为树 + 规则可行性校验

**Files:**
- Create: `fastloop/__init__.py`（空）
- Create: `fastloop/behavior.py`
- Test: `tests/test_behavior.py`

**Interfaces:**
- Consumes: `worldstate.worldstate`（worldstate dict）、`model.prompt`（state 枚举）
- Produces:
  - `fastloop.behavior.Rules`（dataclass）：`min_hp_to_attack: float = 0.2`、`allow_overheating_attack: bool=False`
  - `fastloop.behavior.feasible(proposed, snapshot:dict, rules:Rules) -> bool`：
    - 若 proposed in ('hitbuilding',) 且 哨兵 hp_ratio<rules.min_hp_to_attack → False（改 def）
    - 若 哨兵 vuln==1 且 proposed=='hitbuilding' → False
    - 其余 True
  - `fastloop.behavior.decide_fast(snapshot:dict, rules:Rules) -> str`：
    读 snapshot+黑板策略，输出确定性动作目标（Zone+意图）；M1 用哨兵状态 + 规则收敛。返回 `{'state': s, 'zone': zone, 'target': target}`。
  - `fastloop.behavior.SAFE_FALLBACK = 'def'`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_behavior.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastloop import behavior as bh

def _snap(state='hitbuilding', hp=0.15, vuln=0):
    return {'robots': {7: {'type':'哨兵','team':'红','zone':'Z00','hp_ratio':hp,'vuln':vuln}}, 'bot_state': state}

def test_feasible_low_hp_blocks_attack():
    r = bh.Rules(min_hp_to_attack=0.2)
    assert bh.feasible('hitbuilding', _snap(hp=0.15), r) is False
    assert bh.feasible('def', _snap(hp=0.15), r) is True

def test_feasible_vulnerable_blocks_attack():
    r = bh.Rules()
    assert bh.feasible('hitbuilding', _snap(hp=1.0, vuln=1), r) is False

def test_decide_fast_returns_action():
    r = bh.Rules()
    a = bh.decide_fast(_snap('hitbuilding', hp=1.0), r)
    assert a['state'] in bh.SENTRY_STATES or a['state'] in ('def', 'hitbuilding', 'cruise')
    assert a['target'] is not None
```

- [ ] **Step 2: 运行确认失败**

Expected: FAIL。

- [ ] **Step 3: 写最小实现**

```python
# fastloop/behavior.py
from dataclasses import dataclass

SENTRY_STATES = ('def', 'hitbuilding', 'cruise')
SAFE_FALLBACK = 'def'

@dataclass
class Rules:
    min_hp_to_attack: float = 0.2
    allow_overheating_attack: bool = False

def _sentry(snapshot):
    return snapshot.get('robots', {}).get(7)

def feasible(proposed, snapshot, rules):
    s = _sentry(snapshot)
    if s is None:
        return proposed in SENTRY_STATES
    hp = s.get('hp_ratio', 1.0)
    vuln = s.get('vuln', 0)
    if proposed == 'hitbuilding':
        if hp < rules.min_hp_to_attack:
            return False
        if vuln:
            return False
    return proposed in SENTRY_STATES

def decide_fast(snapshot, rules):
    proposed = snapshot.get('bot_state', 'cruise')
    if not feasible(proposed, snapshot, rules):
        proposed = SAFE_FALLBACK
    s = _sentry(snapshot) or {}
    return {
        'state': proposed,
        'zone': s.get('zone', 'Z--'),
        'target': '敌方前哨站' if proposed == 'hitbuilding' else ('无' if proposed == 'cruise' else '敌方单位'),
    }
```

- [ ] **Step 4: 运行确认通过**

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add fastloop/ tests/ && git commit -m "feat(m1): fast-loop behavior tree + feasibility rules"
```

---

### Task 1.3：评测指标

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `eval.metrics.accuracy(y_true:list[str], y_pred:list[str]) -> float`
  - `eval.metrics.f1_macro(y_true:list[str], y_pred:list[str]) -> float`
  - `eval.metrics.legal_rate(results:list[dict]) -> float`（`state` 在合法集合的比例）
  - `eval.metrics.confusion(y_true, y_pred, labels) -> dict[list[int]]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_metrics.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import metrics

def test_accuracy_perfect():
    assert metrics.accuracy(['def','cruise'], ['def','cruise']) == 1.0
    assert metrics.accuracy(['def','cruise'], ['def','hitbuilding']) == 0.5

def test_f1_macro_range():
    f = metrics.f1_macro(['def','hitbuilding','cruise'], ['def','hitbuilding','def'])
    assert 0.0 <= f <= 1.0

def test_legal_rate():
    res = [{'state':'def'}, {'state':'hitbuilding'}, {'state':'bogus'}]
    assert abs(metrics.legal_rate(res) - 2/3) < 1e-9

def test_confusion_shape():
    c = metrics.confusion(['def','cruise','def'], ['def','def','cruise'], ['def','cruise'])
    assert len(c) == 2 and all(len(v) == 2 for v in c.values())
```

- [ ] **Step 2: 运行确认失败**

Expected: FAIL。

- [ ] **Step 3: 写最小实现**

```python
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
```

- [ ] **Step 4: 运行确认通过**

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add eval/ tests/ && git commit -m "feat(m1): eval metrics (acc/f1/legal/confusion)"
```

---

### Task 1.4：慢环 LLM 预测封装（transformers 推理）+ M1 端到端离线评测

**Files:**
- Create: `model/predict.py`
- Create: `run_m1.py`
- Test: `tests/test_predict.py`（仅测非模型逻辑：如 load 守卫、JSON 提取；模型调用不进单测）

**Interfaces:**
- Consumes: `worldstate`、`model.prompt`、`eval.metrics`、`fastloop.behavior`
- Produces:
  - `model.predict.load_model(path_or_name:str, quantize:bool=True) -> object`（transformers AutoModelForCausalLM + AutoTokenizer，8bit/int4 加载）
  - `model.predict.llm_decide(model, tokenizer, worldstate:dict) -> dict`：走 `build_prompt` → `generate` → `parse_decision`，返回 `{'state':..., 'raw':...}`。
  - `model.predict.relaxed_extract(text) -> str`：从模型输出中提取首个 `{...}` JSON 子串，供 parse_decision。
  - `run_m1.py`：对若干局构建 WorldState，规则标签作真值(y_true)，LLM 慢环输出作预测(y_pred)，快环 `decide_fast` 收敛，评测 acc/f1/legal/latency，写 `out_m1/report.json`。

> **测试边界**：模型加载/生成是重操作，不放进单测。单测只测 `relaxed_extract`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_predict.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import predict

def test_relaxed_extract_finds_json():
    assert predict.relaxed_extract('input: {"state": "def"} trailing') == '{"state": "def"}'

def test_relaxed_extract_none_if_missing():
    assert predict.relaxed_extract('no json here') is None
```

- [ ] **Step 2: 运行确认失败**

Expected: FAIL。

- [ ] **Step 3: 写最小实现**

```python
# model/predict.py
import json, re
from model import prompt

def relaxed_extract(text):
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    return m.group(0) if m else None

def load_model(path_or_name, quantize=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    tok = AutoTokenizer.from_pretrained(path_or_name)
    if quantize:
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(path_or_name, quantization_config=cfg, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(path_or_name, device_map='auto')
    model.eval()
    return model, tok

def llm_decide(model, tok, worldstate):
    p = prompt.build_prompt(worldstate)
    inputs = tok(p, return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sub = relaxed_extract(text)
    parsed = prompt.parse_decision(sub) if sub else prompt.parse_decision('')
    return {'state': parsed['state'], 'raw': text}
```

`run_m1.py`
```python
# -*- coding: utf-8 -*-
import os, argparse, json, time
import config
from data import db, features
from worldstate import zone, worldstate as ws
from labels import sentry_labels as sl
from fastloop import behavior as bh
from eval import metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='*', type=int, default=None)
    ap.add_argument('--out', default='out_m1')
    ap.add_argument('--no-llm', action='store_true', help='仅评估规则基线(快环),不加载LLM')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = db.connect()
    gids = args.games or [config.GAME_REF]
    y_true, y_pred, latencies, samples = [], [], [], []
    rules = bh.Rules()
    for gid in gids:
        rows = db.timeseries_for(con, gid)
        dps = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps)
        series = sl.label_sentry_series(dps, b, team='红', sentry_rid=7)
        for dp, (t, state) in zip(dps, series):
            w = ws.build_worldstate(dp, b)
            if args.no_llm:
                pred = bh.decide_fast({'robots': w['robots'], 'bot_state': state}, rules)
                p = pred['state']
            else:
                model, tok = _load()
                t0 = time.time()
                res = _llm_decide(model, tok, w)
                latencies.append(time.time() - t0)
                pred = bh.decide_fast({'robots': w['robots'], 'bot_state': res['state']}, rules)
                p = pred['state']
            y_true.append(state); y_pred.append(p)
            samples.append({'t': t, 'true': state, 'pred': p})
    con.close()
    legal = metrics.legal_rate([{'state': s} for s in y_pred])
    report = {
        'games': gids, 'n': len(samples),
        'accuracy': metrics.accuracy(y_true, y_pred),
        'f1_macro': metrics.f1_macro(y_true, y_pred),
        'legal_rate': legal,
        'mean_latency_s': (sum(latencies)/len(latencies) if latencies else None),
        'confusion': metrics.confusion(y_true, y_pred, list(sl.SENTRY_STATES)),
    }
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

def _load():
    from model import predict
    return predict.load_model('Qwen/Qwen2.5-1.5B-Instruct', quantize=True)

def _llm_decide(model, tok, w):
    from model import predict
    return predict.llm_decide(model, tok, w)

if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行确认通过 + 手动跑无 LLM 基线**

```bash
cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_predict.py -q && python run_m1.py --no-llm
```
Expected: PASS + `out_m1/report.json` 打印 acc/f1/legal。
（本机有 GPU 时可去掉 `--no-llm` 加载 Qwen2.5-1.5B 跑真实慢环。）

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add model/ run_m1.py tests/ && git commit -m "feat(m1): slow-loop LLM wrapper + offline eval"
```

---

# Phase 2 — M2: RL(PettingZoo) · 蒸馏标签 · 跨单位迁移 · LoRA 微调（子计划大纲）

> **范围说明**：本阶段为独立子系统，规划为**子计划大纲**。在 M1 交付后展开为独立 plan 文档再执行。下面是文件结构、对 M0/M1 的接口契约、任务清单。

**文件结构（待展开）**
- `rl/env.py` — PettingZoo 2D 极简环境（单位点/圆、移动/伤害/建筑血量/Zone/热量约束）
- `rl/train.py` — PPO self-play（stable-baselines3）
- `rl/distill_labels.py` — 从训练出的策略导出 `WorldState→动作` 蒸馏标签
- `labels/transfer.py` — 跨单位标签迁移（状态匹配，借 RMUC-OfflineRL）
- `finetune/lora_train.py` — 用蒸馏+迁移+规则三源标签 LoRA 微调 Qwen2.5-3B
- `blackboard.py` — 异步黑板（慢环写/快环读），对应 spec §6.0/6.3

**接口契约（对 M0/M1）**
- 消费：`worldstate.worldstate.build_worldstate(dp, bounds) -> dict`；`labels.sentry_labels`（label 枚举）；`fastloop.behavior`（Rules/decide_fast）。
- 产出：`rl.distill_labels` 输出与 `worldstate` 同构的 `{'state':..., 'target':...}` 标签，汇入 `labels` 多源融合（spec §6.5）；`blackboard` 提供 `write(policy)/read()`，快环从中取策略（spec §6.1→6.2）。
- `finetune.lora_train` 复用 `data.features.build_decision_points` 生成样本，输出微调后 `model/` 权重，供 `model.predict` 加载。

**任务清单（每项展开成 bite-sized TDD）**
1. `rl/env.py`：环境 step/obs/action 定义 + 单测（回合结束、obs 维度、合法动作）。
2. `rl/env.py + rl/train.py`：PPO self-play 训练脚本 + 冒烟测试（几个 step 不 crash）。
3. `rl/distill_labels.py`：策略 rollout → 标签导出 + 单测（标签枚举合法）。
4. `labels/transfer.py`：跨单位状态匹配迁移 + 单测（匹配命中/规则）。
5. `blackboard.py`：异步黑板写读 + 单测（写入后读回、最新策略）。
6. `finetune/lora_train.py`：三源标签多源融合 + 数据集切分(game_id 防泄漏) + 训练 + 冒烟（batch 跑通）。

---

# Phase 3 — M3: 博弈论意图选择 · 分层 schema · 行为树先验（子计划大纲）

> **范围说明**：M0/M1/M2 交付后展开为独立 plan。下面是文件结构、接口契约、任务清单。

**文件结构（待展开）**
- `game/game_theory.py` — payoff 矩阵 + Fictitious Play / Regret Matching，得全局模式/意图选择
- `schema/layered.py` — 分层输出 schema（全局模式 + 每单位四通道），校验器（spec §6.2）
- `tree/behavior_prior.py` — 从状态转移序列挖掘高频路径，组织成行为树先验（spec §2 行为树、§4.2）
- `tree/global_mode.py` — 全局模式（攻/守/均衡）决策（借北理三模式）

**接口契约**
- 消费：`worldstate`（含 zone/norm_xy）、`fastloop.behavior`、`rl/distill_labels`。
- 产出：`game_theory.choose_mode(...)`（全局模式/意图），`schema.validate(decision)`（合规），`tree.behavior_prior(...)`（行为树先验上下文，供慢环 LLM）。
- 对接 `DecisionRequest`：`schema.validate` 保证 "严格输出结构化数据"（考核点），`game_theory` 决策并行于 LLM，行为树先验作上下文。

**任务清单（每项展开成 TDD）**
1. `schema/layered.py`：分层 schema 定义 + 校验器 + 单测（合法/非法）。
2. `game/game_theory.py`：payoff + regret matching 迭代 + 单测（收敛、合法策略）。
3. `tree/behavior_prior.py`：状态转移频繁子序列挖掘 + 单测（高频路径命中）。
4. `tree/global_mode.py`：全局模式推导 + 单测（三模式可判、约束单位意图）。
5. `run_m3.py`：整链 `慢环(LLM+博弈论) + 快环(行为树+规则) + 行为树先验` 端到端 + `DecisionRequest` 合规率评测。

---

## 自检（Self-Review）

- **Spec coverage**：§1.1 基础(哨兵3状态)→M1；§1.2 进阶(分层)→M3；§2 三引擎分层→M1/M2/M3；§4 WorldState(Zone)→M0；§5 标签(规则/迁移/蒸馏)→M0/M2；§6 快慢环+校验+记忆+黑板→M1/M2/M3；§7 评测(含性能)→M1(+M2/M3)；§8 里程碑→本计划四阶段。覆盖无缺口。
- **Placeholder 扫描**：M0/M1 无 TBD；M2/M3 按 spec scope-check 以"子计划大纲"形式给出（明确标注待展开），非空占位。
- **Type consistency**：`build_decision_points`/`field_bounds`/`assign_zone`/`label_sentry`/`build_worldstate`/`build_prompt`/`parse_decision`/`Rules`/`decide_fast`/`feasible`/`accuracy`/`f1_macro`/`relaxed_extract`/`llm_decide` 前后一致；`ZONE_LABELS` 从 `worldstate.zone` 再导出到 `worldstate.worldstate` 一致；`SENTRY_STATES`/`VALID_STATES` 统一。
- **歧义**：平滑 `smooth_sequence` 合并策略明确（并入前段、向后填充）。

---

## 执行移交

计划落盘：`docs/superpowers/plans/2026-09-07-rmuc-llm-staged.md`。

开始执行时：**从 Phase 0 / Task 0.1 起**，逐任务实现 + 测试 + 提交；M0 完成即产出可运行的 `run_m0.py` 与哨兵状态条带图。M2/M3 在对应里程碑到达时再展开为独立计划执行。
