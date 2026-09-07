# -*- coding: utf-8 -*-
"""M2 管线编排: db -> decision points -> 规则头 + RL 头 -> 三源融合 -> 跨单元增广 ->
数据集 -> QLoRA 蒸馏 -> 留出评估 -> out_m2/m2_report.json。

用法:
  python run_m2.py --stage fusion              # 规则头 + RL 头 -> 融合 (dump fused_decisions.json)
  python run_m2.py --stage dataset             # 融合 -> instruction/completion 数据集
  python run_m2.py --stage train               # QLoRA 蒸馏 (需数据集)
  python run_m2.py --stage eval                # 留出评估 (需已微调模型)
  python run_m2.py --stage all                 # 全流程(规则头兜底, 保证可跑通)
  python run_m2.py --all-fallback              # 若 RL 不收敛: 仅规则头 + 融合(强先验)全流程
"""
import os, json, argparse
import config
from data import db, features
from schema import layered as sl
import run_m3

ALL_STAGES = ['fusion', 'dataset', 'train', 'eval', 'all']


def list_game_ids(con, offset=0, count=None):
    gids = [r[0] for r in con.execute('SELECT game_id FROM matches ORDER BY game_id')]
    if count is not None:
        return gids[offset:offset + count]
    return gids[offset:]


def load_decision_points(gid, limit=None):
    con = db.connect()
    rows = db.timeseries_for(con, gid)
    dps = features.build_decision_points(rows, gid)
    b = features.field_bounds(dps)
    con.close()
    if limit:
        dps = dps[:limit]
    return dps, b


# ---------------- 规则头 ----------------
def rule_head(dps, b):
    decisions, prior, _sigs = run_m3.compute_decision_series(dps, b)
    return decisions


# ---------------- RL 头 ----------------
def rl_head(dps, b, train_game_ids, rl_updates_per_game=60, rl_seed=0):
    """在若干局上训练 IPPO(跨局暴露), 然后 rollout 到给定决策点 -> rl 头决策。"""
    from rl.env import M2Env
    from rl.train import IPPO
    from rl import distill_labels as dl
    import numpy as np
    np.random.seed(rl_seed)
    # 先在 train_game_ids 上训练以跨局泛化
    first = None
    for gid in train_game_ids:
        try:
            tdps, tb = load_decision_points(gid)
            env = M2Env(tdps, tb)
            if first is None:
                ppo = IPPO(env.obs_dim, hidden=64)
                first = ppo
            else:
                ppo = first
            ppo.train(env, n_updates=rl_updates_per_game, device='cpu')
        except Exception as e:
            print('  rl train on game', gid, 'skipped:', e)
    if first is None:
        raise RuntimeError('no RL trainable game')
    acts = first.rollout_actions(dps, b)
    return dl.rollout_to_decisions(acts, dps, b)


def stage_fusion(dps, b, out_dir, train_game_ids, skip_rl, skip_llm, rl_updates=60):
    rule = rule_head(dps, b)
    rl = None
    if not skip_rl:
        try:
            rl = rl_head(dps, b, train_game_ids, rl_updates)
        except Exception as e:
            print('RL head unavailable -> fallback rule-only:', e)
            rl = None
    llm = None  # DeepSeek 慢环头(GPU-free, 可后续接; 默认跳过以免阻塞过夜)
    if not skip_llm:
        try:
            pass  # 预留 run_llm_m3 接入
        except Exception:
            pass
    from labels import fusion as fu
    fused = fu.fuse(rule, rl, llm) if rl else fu.fuse(rule, rule, llm)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'fused_decisions.json'), 'w', encoding='utf-8') as f:
        json.dump(fused['decisions'], f, ensure_ascii=False, indent=None)
    print('fusion done:', json.dumps({k: (v if k != 'decisions' else f"{len(v)} pts")
                                      for k, v in fused.items()}, ensure_ascii=False))
    return fused


def stage_dataset(dps, b, out_dir, fused_decisions, team='红'):
    from finetune import make_dataset as md
    recs = md.make_records(fused_decisions, dps, bounds=b, team=team)
    path = md.to_jsonl(recs, os.path.join(out_dir, 'm2_dataset.jsonl'))
    print('dataset written:', len(recs), '->', path)
    return recs


def stage_train(records, out_dir, base=None):
    from finetune.lora_train import train_lora
    adapter, stats = train_lora(records, out_dir=os.path.join(out_dir, 'finetuned'),
                                base=base, epochs=1)
    print('train done:', stats)
    return adapter, stats


def stage_eval(adapter, eval_game_ids, out_dir, limit=None):
    from finetune.eval import eval_model
    reports = []
    for gid in eval_game_ids:
        dps, b = load_decision_points(gid, limit)
        if not dps:
            continue
        rep = eval_model(adapter, dps, b)
        rep['game_id'] = gid
        reports.append(rep)
        print('eval', gid, json.dumps(rep, ensure_ascii=False))
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=ALL_STAGES)
    ap.add_argument('--gid', type=int, default=config.GAME_REF)
    ap.add_argument('--skip-rl', action='store_true')
    ap.add_argument('--skip-llm', action='store_true')
    ap.add_argument('--all-fallback', action='store_true', help='仅规则头(RL/LLM 全关)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default='out_m2')
    ap.add_argument('--train-games', type=int, default=3)
    ap.add_argument('--dataset-games', type=int, default=24)
    ap.add_argument('--eval-games', type=int, default=4)
    ap.add_argument('--rl-updates', type=int, default=80)
    ap.add_argument('--base', default=None)
    args = ap.parse_args()

    if args.all_fallback:
        args.skip_rl = True
        args.skip_llm = True
    os.makedirs(args.out, exist_ok=True)

    con = db.connect()
    all_gids = list_game_ids(con)
    con.close()
    total = len(all_gids)
    train_ids = all_gids[0:args.train_games]
    dataset_ids = all_gids[args.train_games:args.train_games + args.dataset_games]
    eval_ids = all_gids[args.train_games + args.dataset_games:
                        args.train_games + args.dataset_games + args.eval_games]
    if not dataset_ids and args.stage in ('dataset', 'all'):
        dataset_ids = [args.gid]

    # 单局工厂: 规则头 + RL 头(按配置) -> 融合 -> 决策; 并按需转记录(指令/补全)
    def process_game(gid, make_records):
        dps, b = load_decision_points(gid, args.limit)
        fused = stage_fusion(dps, b, args.out, train_ids, args.skip_rl, args.skip_llm, args.rl_updates)['decisions']
        if make_records:
            recs = stage_dataset(dps, b, args.out, fused)
            return recs
        return fused

    all_records = []
    if args.stage in ('fusion', 'dataset', 'all'):
        for gid in dataset_ids:
            try:
                recs = process_game(gid, make_records=(args.stage in ('dataset', 'all')))
                all_records += recs
            except Exception as e:
                print(f'game {gid} process failed:', e)
        print('total records:', len(all_records))
        # 汇总一份跨局数据集(便于 train 一次性读取)
        if all_records and args.stage in ('dataset', 'all'):
            from finetune.make_dataset import to_jsonl
            to_jsonl(all_records, os.path.join(args.out, 'm2_dataset.jsonl'))

    adapter = None
    stats = None
    if args.stage in ('train', 'all'):
        if not all_records:
            # 读已生成的数据集 jsonl
            ds_path = os.path.join(args.out, 'm2_dataset.jsonl')
            if os.path.exists(ds_path):
                import sys
                from finetune.make_dataset import to_hf_dataset
                recs = [json.loads(l) for l in open(ds_path, encoding='utf-8') if l.strip()]
                all_records = recs
        if all_records:
            adapter, stats = stage_train(all_records, args.out, base=args.base)
        else:
            print('no records -> cannot train')

    report = {'dataset_games': dataset_ids[:10], 'eval_games': eval_ids[:4],
              'train_stats': stats}
    # eval 阶段单独跑时自动探测已训练适配器 (out_m2/finetuned/adapter)
    candidate = os.path.join(args.out, 'finetuned', 'adapter')
    if adapter is None and args.stage == 'eval' and os.path.isdir(candidate):
        adapter = candidate
    if args.stage in ('eval', 'all') and adapter:
        reports = stage_eval(adapter, eval_ids, args.out, args.limit)
        report['eval'] = reports
        report['note'] = ('fine-tuned LLM reads raw worldstate -> layered decision; '
                          'compare against prompt baseline 0.25 / rule head')
    with open(os.path.join(args.out, 'm2_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('REPORT @', os.path.join(args.out, 'm2_report.json'))


if __name__ == '__main__':
    main()
