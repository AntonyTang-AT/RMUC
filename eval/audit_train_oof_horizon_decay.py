"""Train-OOF horizon decay and adjacent-breakpoint audit; no refit/data scan.

Reads the saved death-aware, whole-game GroupKFold OOF archive only.
Usage: python eval/audit_train_oof_horizon_decay.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/motion_candidate_oof.npz'
HORIZONS = (10, 30, 60, 120)
CLASS_IDS = (0, 1, 2)
CLASS_NAMES = ('进攻', '均衡', '防守')
REPS = 2000
SEED = 20260927


def _side_metric(y, pred):
    if not len(y):
        return None
    f1s, recalls = [], []
    for c in CLASS_IDS:
        tp = int(np.sum((y == c) & (pred == c)))
        fp = int(np.sum((y != c) & (pred == c)))
        fn = int(np.sum((y == c) & (pred != c)))
        support = tp + fn
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {'accuracy': float(np.mean(pred == y)),
            'macro_f1': float(np.mean(f1s)),
            'balanced_accuracy': float(np.mean(recalls)),
            'class_recall': recalls,
            'support': [int(np.sum(y == c)) for c in CLASS_IDS]}


def _game_cells(d, horizon_i, pred, valid):
    y, gids, sides = d['mode_y'][:, horizon_i], d['game_ids'], d['sides']
    result = {}
    for gid in np.unique(gids[valid]):
        side_metrics = []
        for side in ('红', '蓝'):
            mask = valid & (gids == gid) & (sides == side)
            if np.any(mask):
                side_metrics.append(_side_metric(y[mask], pred[mask]))
        if len(side_metrics) == 2:
            result[int(gid)] = {k: np.mean([s[k] for s in side_metrics], axis=0).tolist()
                                if k == 'class_recall' else
                                (np.mean([s[k] for s in side_metrics], axis=0).tolist()
                                 if k == 'support' else
                                 float(np.mean([s[k] for s in side_metrics])))
                                for k in ('accuracy', 'macro_f1', 'balanced_accuracy',
                                           'class_recall', 'support')}
    return result


def _mean_ci(values, seed):
    vals = np.asarray(values, dtype=float)
    if not len(vals):
        return {'n_games': 0, 'mean': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    sampled = vals[rng.integers(0, len(vals), size=(REPS, len(vals)))].mean(axis=1)
    lo, hi = np.quantile(sampled, [.025, .975])
    return {'n_games': int(len(vals)), 'mean': float(vals.mean()),
            'ci95': [float(lo), float(hi)]}


def main():
    d = dict(np.load(ARCHIVE, allow_pickle=False))
    if len(np.unique(d['fold_ids'])) != 5:
        raise ValueError('expected five saved whole-game OOF folds')
    for gid in np.unique(d['game_ids']):
        if len(np.unique(d['fold_ids'][d['game_ids'] == gid])) != 1:
            raise ValueError(f'game {gid} appears in multiple held-out folds')
    if d['mode_y'].shape[1] != len(HORIZONS):
        raise ValueError('saved horizon order/shape mismatch')

    rows, by_horizon, games_by_horizon = [], {}, {}
    prediction = d['motion_candidate'].argmax(axis=2)
    for hi, h in enumerate(HORIZONS):
        y = d['mode_y'][:, hi]
        valid = y >= 0
        base = d['base_mode'][valid]
        truth = y[valid]
        pred = prediction[valid, hi]
        game_ids, sides = d['game_ids'][valid], d['sides'][valid]
        # Use the same equal-game, equal-side aggregation as the OOF reports.
        # Keep only games with both observation sides at each horizon.
        candidate_games = _game_cells(d, hi, prediction[:, hi], valid)
        persistence_games = _game_cells(d, hi, d['base_mode'], valid)
        if set(candidate_games) != set(persistence_games):
            raise AssertionError('candidate/persistence game keys mismatch')
        game_ids_common = sorted(candidate_games)
        c = {m: _mean_ci([candidate_games[g][m] for g in game_ids_common], SEED+h+idx)
             for idx, m in enumerate(('accuracy', 'macro_f1', 'balanced_accuracy'))}
        perclass = [_mean_ci([candidate_games[g]['class_recall'][ci]
                              for g in game_ids_common], SEED+h+100+ci)
                    for ci in CLASS_IDS]
        p_acc = float(np.mean([persistence_games[g]['accuracy'] for g in game_ids_common]))
        p_f1 = float(np.mean([persistence_games[g]['macro_f1'] for g in game_ids_common]))
        p_ba = float(np.mean([persistence_games[g]['balanced_accuracy'] for g in game_ids_common]))
        pooled = _side_metric(truth, pred)
        change = {}
        # Derive label transitions relative to issue-time class with game/side balance.
        transition_cells = {}
        for gid in game_ids_common:
            side_rates = []
            for side in ('红', '蓝'):
                m = valid & (d['game_ids'] == gid) & (d['sides'] == side)
                side_rates.append(float(np.mean(d['mode_y'][m, hi] != d['base_mode'][m])))
            transition_cells[gid] = float(np.mean(side_rates))
        change = _mean_ci([transition_cells[g] for g in game_ids_common], SEED+h+333)
        # Per-row coverage, and game-side class-presence audit (fixed 3-class macro-F1 uses zero for absent class).
        present_all3 = 0
        valid_game_sides = 0
        for gid in np.unique(game_ids):
            for side in ('红', '蓝'):
                m = (game_ids == gid) & (sides == side)
                if not np.any(m):
                    continue
                valid_game_sides += 1
                present_all3 += int(set(np.unique(truth[m])) == set(CLASS_IDS))
        mode_support = [int(np.sum(valid & (y == cls))) for cls in CLASS_IDS]
        candidate_recall = [p['mean'] for p in perclass]
        row = {'horizon_s': h, 'scorable_rows': int(valid.sum()),
               'coverage_of_train_oof_rows': float(valid.mean()),
               'games_with_red_blue_scored': len(game_ids_common),
               'future_class_support_pooled': dict(zip(CLASS_NAMES, mode_support)),
               'issue_to_future_mode_change_rate_game_equal': change,
               'persistence_accuracy_game_equal': p_acc,
               'persistence_macro_f1_game_equal': p_f1,
               'persistence_balanced_accuracy_game_equal': p_ba,
               'motion_candidate_accuracy_game_equal': c['accuracy'],
               'motion_candidate_macro_f1_game_equal': c['macro_f1'],
               'motion_candidate_balanced_accuracy_game_equal': c['balanced_accuracy'],
               'pooled_row_diagnostic': {'accuracy': pooled['accuracy'],
                   'macro_f1': pooled['macro_f1'], 'balanced_accuracy': pooled['balanced_accuracy'],
                   'class_recall': dict(zip(CLASS_NAMES, pooled['class_recall']))},
               'motion_candidate_per_class_recall_game_equal': dict(zip(CLASS_NAMES, candidate_recall)),
               'game_side_rows_with_all_three_truth_classes': {
                   'n': present_all3, 'denominator': valid_game_sides,
                   'fraction': present_all3 / valid_game_sides if valid_game_sides else None},
               'motion_candidate_minus_persistence_game_paired': {
                   metric: _mean_ci([candidate_games[g][metric]-persistence_games[g][metric]
                                     for g in game_ids_common], SEED+h+700+idx)
                   for idx, metric in enumerate(('accuracy', 'macro_f1', 'balanced_accuracy'))},
               'macro_f1_contract': 'per match-side, fixed labels [attack, balanced, defense]; absent class F1=0; average red/blue within game, then games equally'}
        rows.append(row)
        by_horizon[h] = candidate_games
        games_by_horizon[h] = persistence_games

    adjacent = []
    for left, right in zip(HORIZONS, HORIZONS[1:]):
        keys = sorted(set(by_horizon[left]) & set(by_horizon[right]))
        cell = {'from_s': left, 'to_s': right, 'paired_games': len(keys), 'metrics': {}}
        for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            deltas = [by_horizon[right][g][metric] - by_horizon[left][g][metric]
                      for g in keys]
            cell['metrics'][metric+'_delta_right_minus_left'] = _mean_ci(
                deltas, SEED+left+right+sum(map(ord, metric)))
        adjacent.append(cell)
    output = {'schema': 'train-only-whole-game-oof-horizon-decay-v1',
              'source': str(ARCHIVE), 'opened_splits': ['train OOF predictions + labels'],
              'validation_access': False, 'test_access': False,
              'label_contract': 'relative_margin_v2_death_aware_hp_le_zero',
              'grouping': 'saved 5-fold GroupKFold by complete game_id; red/blue sides share held-out fold',
              'aggregation': 'game equal; red/blue sides averaged within each match; whole games bootstrap',
              'candidate': 'motion plus zone/observable-score OOF candidate, scored only to characterize horizon decay; not promoted',
              'class_order': list(CLASS_NAMES), 'horizons': rows,
              'adjacent_horizon_paired_deltas': adjacent,
              'metrics_note': {
                  'accuracy': 'correct argmax class fraction; persistence uses current mode as forecast',
                  'macro_f1': 'mean of fixed three class F1 values; a class absent in a particular match-side has F1=0, so this game-balanced measure can be substantially lower than pooled-row macro F1',
                  'balanced_accuracy': 'mean of the three class recalls under the same fixed-class match-side aggregation',
                  'mode_change_rate': 'future OOF target differs from issue-time base_mode; complements persistence accuracy on the same rows',
                  'coverage': 'rows with exact valid saved OOF target / all train OOF issue rows; long-horizon games ending early are censored'} }
    out_path = ROOT/'out_m2/death_aware_global_mode_experiments/horizon_decay_oof/report.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# 训练集 OOF 时距衰减与舍弃点审计', '',
          '只读既有 death-aware train 五折整局 OOF 概率/标签，不训练、不访问 validation/test、不扫描原始比赛。',
          '候选是 motion + zone/observable-score OOF，用于描述现有最强实验预测头的时距表现；未晋升。', '',
          '| Horizon | 可评分覆盖 | 状态切换率 | Acc 候选/持久性 | Macro-F1 候选/持久性 | 候选 Balanced Acc | 攻/均衡/守 Recall |',
          '|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        recall = r['motion_candidate_per_class_recall_game_equal']
        md.append(f"| {r['horizon_s']}s | {r['coverage_of_train_oof_rows']:.1%} | "
                  f"{r['issue_to_future_mode_change_rate_game_equal']['mean']:.1%} | "
                  f"{r['motion_candidate_accuracy_game_equal']['mean']:.1%}/{r['persistence_accuracy_game_equal']:.1%} | "
                  f"{r['motion_candidate_macro_f1_game_equal']['mean']:.1%}/{r['persistence_macro_f1_game_equal']:.1%} | "
                  f"{r['motion_candidate_balanced_accuracy_game_equal']['mean']:.1%} | "
                  f"{recall['进攻']:.1%}/{recall['均衡']:.1%}/{recall['防守']:.1%} |")
    md += ['', '## 相邻时距配对变化（后者减前者；95% 比赛 bootstrap CI）', '',
           '| 区间 | Accuracy Δ | Macro-F1 Δ | Balanced Acc. Δ |', '|---|---:|---:|---:|']
    for a in adjacent:
        def fmt(metric):
            s = a['metrics'][metric+'_delta_right_minus_left']
            return f"{s['mean']*100:+.2f}pp [{s['ci95'][0]*100:+.2f}, {s['ci95'][1]*100:+.2f}]"
        md.append(f"| {a['from_s']}→{a['to_s']}s | {fmt('accuracy')} | {fmt('macro_f1')} | {fmt('balanced_accuracy')} |")
    md += ['', '## 结论与当前建议', '',
           'OOF 只在四个既有 horizon 取精确标签，所以不能证明 10–30 秒之间或 60–120 秒之间的某个未测秒数是断点。现有证据显示准确率、Macro-F1、类别召回和有效目标覆盖随 horizon 平滑变差，没有一个突变点可作为硬截断。尤其 120s：模式切换已接近一半，候选 Macro-F1/均衡召回较弱；不应删除此宏观预测字段，但应降级为概率趋势/风险摘要，避免把单一 argmax 当确定结论。',
           '', '10/30/60/120s 可保留为兼容的标准刻度；其中 10/30s 用于较具体局面，60s 用于中期计划，120s 仅用于粗粒度趋势。若要知道更精确的失效时间，需要在 train 内做整场分组的密集时距标签与 OOF（例如每 3/5/10 秒），不能由四个稀疏点插值宣称断点。',
           '', '准确率是 argmax 对三分类标签的命中比例。Macro-F1 先对攻/均衡/守各自算 F1 再取算术平均，避免常见类压住少数类；这里主表先对每个比赛的红蓝方分别计算固定三类 Macro-F1，比赛等权平均，若某个比赛方没有某类样本，该类 F1 记 0。机器报告也列 pooled-row Macro-F1：它先合并全部行的混淆矩阵再算 F1；两者是不同的估计对象，不要直接互换。Balanced accuracy 是固定三类 Recall 的算术平均。持久性基线直接把当前态势类别当未来类别；它不是一个容易打败的弱基线：10s 当前模式保持率近69%，短时预测仅略超它。候选对持久性 Accuracy 的优势约为0.9/2.3/2.4/4.5个百分点，但比赛均衡 Macro-F1 在四档都低于持久性；单看 Accuracy 会掩盖类别质量的损失。',
           '', '## 复现', '',
           '- 脚本：`eval/audit_train_oof_horizon_decay.py`',
           '- OOF 输入：`out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/motion_candidate_oof.npz`',
           '- 机器报告：`out_m2/death_aware_global_mode_experiments/horizon_decay_oof/report.json`',
           '- 既有趋势/错分诊断：`docs/analysis/2026-09-27-death-aware-oof-error-decomposition.md`、`docs/analysis/2026-09-27-visible-zone-mass-observable-score-oof.md`']
    (ROOT/'docs/analysis/2026-09-27-train-oof-horizon-decay.md').write_text(
        '\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'json': str(out_path), 'markdown': str(ROOT/'docs/analysis/2026-09-27-train-oof-horizon-decay.md'),
                      'horizons': len(rows), 'adjacent_comparisons': len(adjacent),
                      'games': len(np.unique(d['game_ids']))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
