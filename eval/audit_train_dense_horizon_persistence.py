"""Train-only dense-horizon persistence/label-switch audit (no model training).

Streams only the complete-match train game ids in the saved death-aware train
archive. Labels are reconstructed from exact one-second ground-truth states
using the same label function as the dataset builder, then parity-checked at
the established 10/30/60/120s horizons.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NPZ = ROOT / 'out_m2/temporal_forecast_death_aware_candidate/train.npz'
DB = ROOT / '数据与资料/rmuc_2026_region_dataset.sqlite'
OUT = ROOT / 'out_m2/death_aware_global_mode_experiments/dense_horizon_persistence/report.json'
MD = ROOT / 'docs/analysis/2026-09-27-train-dense-horizon-persistence.md'
HORIZONS = tuple(range(3, 121, 3))
KNOWN = (10, 30, 60, 120)
REPS = 2000
SEED = 20260927
CLASSES = (0, 1, 2)
CLASS_NAMES = ('进攻', '均衡', '防守')


def _score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if not len(y):
        return {'accuracy': float('nan'), 'macro_f1': float('nan'), 'balanced_accuracy': float('nan')}
    f1, rec = [], []
    for c in CLASSES:
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        support = tp + fn
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
        rec.append(float(tp / support) if support else 0.0)
    return {'accuracy': float(np.mean(y == pred)), 'macro_f1': float(np.mean(f1)),
            'balanced_accuracy': float(np.mean(rec))}


def _ci(vals, seed):
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {'n_games': 0, 'mean': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    boot = vals[rng.integers(0, len(vals), size=(REPS, len(vals)))].mean(axis=1)
    return {'n_games': int(len(vals)), 'mean': float(vals.mean()),
            'ci95': [float(x) for x in np.quantile(boot, [.025, .975])]}


def main():
    from forecast.temporal import MODES, _mode_from_groups

    data = dict(np.load(NPZ, allow_pickle=False))
    gids = data['game_ids'].astype(np.int64)
    sides = data['sides'].astype(str)
    issue = np.rint(data['issued_at']).astype(np.int32)
    base = data['base_mode'].astype(np.int8)
    mode_y = data['mode_y'].astype(np.int8)
    train_games = np.unique(gids)

    # Only train games are queried. Result grouping fits in memory (~152k seconds).
    con = sqlite3.connect(DB)
    con.execute('PRAGMA query_only=ON')
    placeholders = ','.join('?' for _ in train_games)
    query = (f'SELECT * FROM timeseries WHERE game_id IN ({placeholders}) '
             'ORDER BY 5,6')
    cur = con.execute(query, [int(g) for g in train_games])
    states: dict[int, dict[int, dict[str, int]]] = defaultdict(dict)
    current_key = None
    groups = {'红': [], '蓝': []}
    rows_read = 0

    def flush(key, entities):
        if key is None:
            return
        gid, sec = key
        # Use exact full-state team snapshot, including static structures.
        for side in ('红', '蓝'):
            own = entities[side]
            opp = entities['蓝' if side == '红' else '红']
            states[gid].setdefault(sec, {})[side] = MODES.index(_mode_from_groups(own, opp, side))

    for row in cur:
        rows_read += 1
        gid = int(row[4])
        sec = int(round(float(row[5])))
        key = (gid, sec)
        if current_key is not None and key != current_key:
            flush(current_key, groups)
            groups = {'红': [], '蓝': []}
        current_key = key
        side = row[8]
        if side not in groups:
            continue
        groups[side].append({'entity_type': row[7],
                             'position': [row[13], row[14]],
                             'hp': row[11], 'maxhp': row[12]})
    flush(current_key, groups)
    con.close()

    # Confirm label reconstruction is exactly equivalent to established sparse targets.
    parity = {}
    for h in KNOWN:
        hi = list(data['horizons']).index(h)
        valid = mode_y[:, hi] >= 0
        mismatch = missing = 0
        for i in np.flatnonzero(valid):
            target = states.get(int(gids[i]), {}).get(int(issue[i] + h), {}).get(sides[i])
            if target is None:
                missing += 1
            elif target != int(mode_y[i, hi]):
                mismatch += 1
        parity[str(h)] = {'saved_valid_rows': int(valid.sum()), 'missing_raw_target': missing,
                          'label_mismatches': mismatch}
    if any(p['missing_raw_target'] or p['label_mismatches'] for p in parity.values()):
        raise RuntimeError(f'raw target parity failed; refusing dense report: {parity}')

    # Per game × side cells ensure matches and sides have equal influence.
    score_cells = {}
    row_coverage = {}
    switch_cells = {}
    rows_total = len(gids)
    for h in HORIZONS:
        valid_rows = []
        by_cell = defaultdict(list)
        for i, (gid, side, t, b) in enumerate(zip(gids, sides, issue, base)):
            target = states.get(int(gid), {}).get(int(t + h), {}).get(side)
            if target is not None and 0 <= int(b) < len(MODES):
                valid_rows.append(i)
                by_cell[(int(gid), side)].append((int(b), int(target)))
        # Keep only games with both sides, mirroring the prior audit contract.
        games = sorted({g for g, s in by_cell if (g, '红') in by_cell and (g, '蓝') in by_cell})
        metric = {m: {} for m in ('accuracy', 'macro_f1', 'balanced_accuracy', 'switch_rate')}
        game_rows_coverage = {}
        for g in games:
            side_results = {m: [] for m in metric}
            for s in ('红', '蓝'):
                pairs = by_cell[(g, s)]
                y = np.asarray([p[1] for p in pairs], dtype=np.int8)
                pred = np.asarray([p[0] for p in pairs], dtype=np.int8)
                sm = _score(y, pred)
                for m in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                    side_results[m].append(sm[m])
                side_results['switch_rate'].append(float(np.mean(y != pred)))
            for m in metric:
                metric[m][g] = float(np.mean(side_results[m]))
            game_rows_coverage[g] = float(sum(len(by_cell[(g, s)]) for s in ('红', '蓝')) /
                                            max(1, sum(np.sum((gids == g) & (sides == s)) for s in ('红', '蓝'))))
        score_cells[h] = metric
        row_coverage[h] = {'scorable_rows': len(valid_rows), 'total_issue_rows': rows_total,
                           'fraction': len(valid_rows) / rows_total if rows_total else 0.0}
        switch_cells[h] = game_rows_coverage

    horizons = []
    for h in HORIZONS:
        gset = sorted(score_cells[h]['accuracy'])
        row = {'horizon_s': h, 'scorable_rows': row_coverage[h]['scorable_rows'],
               'coverage_of_train_issue_rows': row_coverage[h]['fraction'],
               'games_with_both_sides': len(gset), 'metrics_game_equal_red_blue_paired': {}}
        for j, metric in enumerate(('accuracy', 'macro_f1', 'balanced_accuracy', 'switch_rate')):
            row['metrics_game_equal_red_blue_paired'][metric] = _ci(
                [score_cells[h][metric][g] for g in gset], SEED + h * 17 + j)
        row['switch_rate_complement_of_accuracy'] = True
        horizons.append(row)

    adjacent = []
    for left, right in zip(HORIZONS, HORIZONS[1:]):
        keys = sorted(set(score_cells[left]['accuracy']) & set(score_cells[right]['accuracy']))
        metric_ci = {}
        for j, m in enumerate(('accuracy', 'macro_f1', 'balanced_accuracy', 'switch_rate')):
            metric_ci[m + '_delta_right_minus_left'] = _ci(
                [score_cells[right][m][g] - score_cells[left][m][g] for g in keys],
                SEED + left * 101 + right * 3 + j)
        adjacent.append({'from_s': left, 'to_s': right, 'paired_games': len(keys), 'metrics': metric_ci})

    worst_drop = min(adjacent, key=lambda x: x['metrics']['accuracy_delta_right_minus_left']['mean'])
    fastest_switch = max(adjacent, key=lambda x: x['metrics']['switch_rate_delta_right_minus_left']['mean'])
    broad_intervals = []
    for left, right in ((3, 15), (15, 30), (30, 60), (60, 90), (90, 120)):
        keys = sorted(set(score_cells[left]['accuracy']) & set(score_cells[right]['accuracy']))
        changes = {}
        for j, m in enumerate(('accuracy', 'macro_f1', 'balanced_accuracy', 'switch_rate')):
            changes[m + '_delta'] = _ci(
                [score_cells[right][m][g] - score_cells[left][m][g] for g in keys],
                SEED + left * 1009 + right * 37 + j)
        broad_intervals.append({'from_s': left, 'to_s': right, 'paired_games': len(keys),
                                'elapsed_s': right - left, 'metrics': changes})
    output = {
        'schema': 'train-only-dense-horizon-persistence-v1',
        'source_train_archive': str(NPZ), 'source_raw_db': str(DB),
        'opened_splits': ['train'], 'validation_access': False, 'test_access': False,
        'model_training': False, 'raw_games_queried': int(len(train_games)),
        'raw_rows_read_train_games_only': rows_read, 'raw_game_seconds_aggregated': sum(map(len, states.values())),
        'label_contract': str(data['global_mode_label_version'][0]),
        'exact_raw_second_alignment': True, 'horizon_step_seconds': 3,
        'parity_check_at_existing_horizons': parity,
        'aggregation': 'per game and side metric; red/blue averaged within game; whole-game bootstrap CI, games equally weighted; require both sides',
        'bootstrap_replicates': REPS, 'class_order': list(CLASS_NAMES),
        'macro_f1_contract': 'fixed attack/balanced/defense classes; absent class F1=0; average red/blue within game then games equally',
        'horizons': horizons, 'adjacent_horizon_paired_deltas': adjacent,
        'broad_interval_paired_deltas': broad_intervals,
        'largest_adjacent_accuracy_drop': {'from_s': worst_drop['from_s'], 'to_s': worst_drop['to_s'],
                                            **worst_drop['metrics']['accuracy_delta_right_minus_left']},
        'largest_adjacent_switch_increase': {'from_s': fastest_switch['from_s'], 'to_s': fastest_switch['to_s'],
                                              **fastest_switch['metrics']['switch_rate_delta_right_minus_left']},
        'interpretation_limit': 'Persistence/label-switch diagnostics only; they do not estimate trained-model accuracy or establish that a learned model should be discarded at a particular horizon.'
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# 训练集密集时距持久性与标签切换诊断', '',
          '本报告仅查询 train.npz 中 363 场完整比赛对应的原始一秒记录，不训练模型、不访问 validation/test。原始状态按整秒分组，使用同一标签函数重建未来全局模式；10/30/60/120 秒标签与既有 train 标签逐行校验，缺失与不一致均为 0 后才继续。', '',
          '| 时距 | 覆盖 | 状态切换率（95%比赛CI） | 持久性Acc（95%比赛CI） | Macro-F1 | Balanced Acc | 双侧比赛数 |',
          '|---:|---:|---:|---:|---:|---:|---:|']
    for r in horizons:
        if r['horizon_s'] not in (3, 9, 15, 30, 45, 60, 75, 90, 105, 120):
            continue
        ms = r['metrics_game_equal_red_blue_paired']
        def fmt(key):
            x = ms[key]
            return f"{x['mean']:.1%} [{x['ci95'][0]:.1%}, {x['ci95'][1]:.1%}]"
        md.append(f"| {r['horizon_s']}s | {r['coverage_of_train_issue_rows']:.1%} | {fmt('switch_rate')} | {fmt('accuracy')} | {fmt('macro_f1')} | {fmt('balanced_accuracy')} | {r['games_with_both_sides']} |")
    md += ['', '## 分段变化与结论', '',
           '| 区间 | 持久性Acc变化（95%比赛CI） | 切换率变化（95%比赛CI） |', '|---:|---:|---:|']
    for block in broad_intervals:
        def delta_fmt(key):
            x = block['metrics'][key + '_delta']
            return f"{x['mean']*100:+.2f}pp [{x['ci95'][0]*100:+.2f}, {x['ci95'][1]*100:+.2f}]"
        md.append(f"| {block['from_s']}→{block['to_s']}s | {delta_fmt('accuracy')} | {delta_fmt('switch_rate')} |")
    md += ['',
           f"- 最大的相邻 3 秒持久性准确率下降出现在 {worst_drop['from_s']}→{worst_drop['to_s']} 秒，变化 {worst_drop['metrics']['accuracy_delta_right_minus_left']['mean']*100:+.2f} 个百分点（比赛 bootstrap 95% CI {worst_drop['metrics']['accuracy_delta_right_minus_left']['ci95'][0]*100:+.2f} 至 {worst_drop['metrics']['accuracy_delta_right_minus_left']['ci95'][1]*100:+.2f}）。",
           f"- 最大的相邻 3 秒切换率上升出现在 {fastest_switch['from_s']}→{fastest_switch['to_s']} 秒，变化 {fastest_switch['metrics']['switch_rate_delta_right_minus_left']['mean']*100:+.2f} 个百分点（95% CI {fastest_switch['metrics']['switch_rate_delta_right_minus_left']['ci95'][0]*100:+.2f} 至 {fastest_switch['metrics']['switch_rate_delta_right_minus_left']['ci95'][1]*100:+.2f}）。",
           '- 持久性准确率与切换率在相同样本上互为补数。逐 3 秒扫描能定位四个稀疏刻度之间变化较快的区间，并显示对局提前结束造成的覆盖下降；但它只诊断标签本身随时间的可预测性，不代表已训练预测器在该时距的准确率，也不能单独作为模型硬舍弃点。',
           '- 相比现有 10/30/60/120 秒审计，本次新增了中间时距的精确标签切换、持久性准确率、Macro-F1、类别平衡准确率和覆盖趋势，以及相邻 3 秒变化的比赛级配对区间。若要得出“模型在某时距开始失效”，仍需在相同密集标签上训练/评估整局分组 OOF；本次按约束不做该训练。',
           '', '## 复现', '',
           '- 脚本：`eval/audit_train_dense_horizon_persistence.py`',
           '- 机器结果：`out_m2/death_aware_global_mode_experiments/dense_horizon_persistence/report.json`',
           '- 输入：仅 `out_m2/temporal_forecast_death_aware_candidate/train.npz` 及其 train 比赛 ID 对应的 SQLite 原始状态。']
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(OUT), 'markdown': str(MD), 'train_games': len(train_games),
                      'raw_rows': rows_read, 'horizons': len(horizons), 'parity': parity}, ensure_ascii=False))


if __name__ == '__main__':
    main()
