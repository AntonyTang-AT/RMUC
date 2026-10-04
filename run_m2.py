# -*- coding: utf-8 -*-
"""M2 管线编排: db -> decision points -> 规则头 + RL 头 -> 融合 -> lookahead 重标 ->
数据集 -> QLoRA 蒸馏 -> 留出评估 -> out_m2/m2_report.json。

交付主线(--profile mainline, 默认): 规则保底 + 通道修复融合(可选 RL) + lookahead 标签
QLoRA；Tier2 默认选择当前正式候选，无法加载时自动回退规则。默认跳过的是旧的
融合 LLM 第三源，而非 Tier2。Track2/Track1 为研究附录, 见 AGENT_PROJECT_OVERVIEW。

用法:
  python run_m2.py --profile mainline --stage all   # 交付主线(默认 profile)
  python run_m2.py --profile research --stage fusion  # 旧实验复现(可开 LLM / 关 lookahead)
  python run_m2.py --stage fusion
  python run_m2.py --all-fallback                   # 仅规则头全流程
"""
import os, json, argparse
import config
from data import db, features
from schema import layered as sl
import run_m3

ALL_STAGES = ['fusion', 'dataset', 'train', 'eval', 'all']
PROFILES = ('mainline', 'research')
MAINLINE_LOOKAHEAD_DELTA = 5


def apply_profile(profile, skip_llm, lookahead_delta, skip_rl=False):
    """将 profile 解析为实际 flags。纯函数, 供 CLI 与单测共用。

    mainline: 强制 skip_llm; lookahead_delta==0 时注入 MAINLINE_LOOKAHEAD_DELTA。
    research: 不改动传入 flags(复现旧实验)。
    返回 dict: profile, skip_llm, lookahead_delta, skip_rl。
    """
    if profile not in PROFILES:
        raise ValueError(f'unknown profile: {profile}')
    out_skip_llm = bool(skip_llm)
    out_delta = int(lookahead_delta)
    if profile == 'mainline':
        out_skip_llm = True
        if out_delta <= 0:
            out_delta = MAINLINE_LOOKAHEAD_DELTA
    return {
        'profile': profile,
        'skip_llm': out_skip_llm,
        'lookahead_delta': out_delta,
        'skip_rl': bool(skip_rl),
    }


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


def load_observable_snapshots(gid, team='红', limit=None):
    """Load causal radar/sensor snapshots aligned to decision-point seconds.

    This helper is intentionally separate from ``load_decision_points`` so
    existing rule/RL callers keep their two-value API.  The returned rows are
    visibility-masked by the RMUC protocol adapter before a temporal model or
    Tier2 LLM sees them.
    """
    from data.radar_visibility_replay import load_radar_visibility_snapshots
    con = db.connect()
    try:
        snapshots = load_radar_visibility_snapshots(con, gid).get(team, [])
    finally:
        con.close()
    if limit:
        snapshots = snapshots[:int(limit)]
    return snapshots


def align_observable_snapshots(dps, snapshots):
    """Join full one-second snapshots to the coarser decision-point clock."""
    by_time = {round(float(row.get('issued_at', 0.0)), 5): row
               for row in (snapshots or []) if isinstance(row, dict)}
    return [by_time.get(round(float(dp.get('t', 0.0)), 5))
            for dp in (dps or [])]


def precompute_temporal_forecasts(dps, snapshots, forecaster):
    """Run a forecaster over every causal snapshot and retain decision times.

    Decision points are three seconds apart while radar replay snapshots are
    one second apart.  Calling ``predict`` only at decision points throws away
    two thirds of the available causal motion history.  This helper feeds all
    snapshots in chronological order, then returns one forecast per decision
    point for TieredDecider.  A failed optional forecast leaves that slot as
    ``None`` so the normal runtime fallback remains available.
    """
    if forecaster is None:
        return None
    decision_index = {
        round(float(dp.get('t', 0.0)), 5): i
        for i, dp in enumerate(dps or [])
    }
    outputs = [None] * len(dps or [])
    rows = sorted(
        [row for row in (snapshots or []) if isinstance(row, dict)],
        key=lambda row: float(row.get('issued_at', 0.0)),
    )
    for row in rows:
        try:
            forecast = forecaster.predict(row)
        except Exception:
            forecast = None
        index = decision_index.get(round(float(row.get('issued_at', 0.0)), 5))
        if index is not None:
            outputs[index] = forecast
    return outputs


def resolve_tier2_adapter(adapter, *, all_fallback=False):
    """Return the formal Tier2 candidate unless an explicit full fallback is requested."""
    if all_fallback:
        return 'none'
    return 'auto' if adapter in (None, '') else str(adapter)


def needs_observable_snapshots(tier2_adapter, temporal_checkpoint):
    """Tier2 and temporal forecasting both require issue-time visible snapshots."""
    tier2_enabled = tier2_adapter not in (None, '', 'none', 'null', '-')
    temporal_enabled = temporal_checkpoint not in (None, '', 'none', 'null', '-')
    return bool(tier2_enabled or temporal_enabled)


# ---------------- 规则头 ----------------
def rule_head(dps, b):
    decisions, prior, _sigs = run_m3.compute_decision_series(dps, b)
    return decisions


# ---------------- RL 头 ----------------
_RL_CACHE = {}  # 策略只训练一次, 多局 rollout 复用(避免每局重训浪费)


def rl_head(dps, b, train_game_ids, rl_updates_per_game=60, rl_seed=0,
            physics=False, reward_conf=None, coherence_scale=1.0, ent_coef=None,
            reward_delta=False, mode_source='from_units', fav_align_scale=0.0):
    """在若干局上训练 IPPO(跨局暴露), 然后 rollout 到给定决策点 -> rl 头决策。

    physics: 用 PhysicsEnv(B2 前向物理)训练+rollout -> 真 MDP 策略; 否则 M2Env(replay-bandit)。
    reward_conf: 覆盖默认奖励权重; None 用 DEFAULT_CONF。
    coherence_scale: Track2 常用 0.25。
    ent_coef: None=IPPO默认0.01; Track2 可传 0.02。
    reward_delta: True 用势能差分 R(t)-R(t-1); False 保旧绝对势能。
    mode_source: 'from_units'(默认) | 'game_theory'(对照)。
    fav_align_scale: Track2 常用 0.08; 0=关闭。
    缓存键含 coherence/ent/delta/fav。"""
    from rl.env import M2Env, PhysicsEnv
    from rl.train import IPPO
    from rl import physics as ph
    from rl import distill_labels as dl
    import numpy as np
    rw_tag = (reward_conf is None) and 'default' or 'learned'
    coh = float(coherence_scale)
    ent_tag = 'default' if ent_coef is None else float(ent_coef)
    delta = bool(reward_delta)
    fav_s = float(fav_align_scale)
    key = (tuple(train_game_ids), rl_updates_per_game, rl_seed, bool(physics),
           rw_tag, coh, ent_tag, delta, fav_s)
    ppo = _RL_CACHE.get(key)
    if ppo is None:
        loaded = []
        for gid in train_game_ids:
            try:
                tdps, tb = load_decision_points(gid)
                loaded.append((gid, tdps, tb))
            except Exception as e:
                print('  rl load game', gid, 'skipped:', e)
        if not loaded:
            raise RuntimeError('no RL trainable game')
        pfx = ph.calibrate([tdps for _, tdps, _ in loaded]) if physics else None
        env_kw = dict(reward_conf=reward_conf, coherence_scale=coh, reward_delta=delta,
                      fav_align_scale=fav_s)
        envs = [PhysicsEnv(tdps, tb, params=pfx, **env_kw) if physics
                else M2Env(tdps, tb, **env_kw)
                for _, tdps, tb in loaded]
        np.random.seed(rl_seed)
        try:
            import torch
            torch.manual_seed(rl_seed)
        except Exception:
            pass
        ippo_kw = {'hidden': 64}
        if ent_coef is not None:
            ippo_kw['ent_coef'] = float(ent_coef)
        first = IPPO(envs[0].obs_dim, **ippo_kw)
        for env in envs:
            first.train(env, n_updates=rl_updates_per_game, device='cpu')
        _RL_CACHE[key] = first
        ppo = first
        print(f'  RL训练完成(physics={physics}, reward={rw_tag}, coh={coh}, ent={ent_tag}, '
              f'delta={delta}, fav={fav_s}, 共{len(loaded)}局, {rl_updates_per_game}轮/局; 策略缓存)',
              flush=True)
    env_kw = dict(reward_conf=reward_conf, coherence_scale=coh, reward_delta=delta,
                  fav_align_scale=fav_s)
    if physics:
        pfx = ph.calibrate([load_decision_points(x)[0] for x in train_game_ids])
        rollout_env = PhysicsEnv(dps, b, params=pfx, **env_kw)
    else:
        rollout_env = M2Env(dps, b, **env_kw)
    acts = ppo.rollout_actions(dps, b, env=rollout_env)
    return dl.rollout_to_decisions(acts, dps, b, mode_source=mode_source)



_MODE_STUDENT_CACHE = {}


def _load_mode_student(path, device='cpu'):
    if not path:
        return None
    key = (os.path.abspath(path), str(device))
    if key not in _MODE_STUDENT_CACHE:
        from rl.mode_student import ModeStudent
        _MODE_STUDENT_CACHE[key] = ModeStudent.load(path, device=device)
        print(f'  mode_student loaded: {path}', flush=True)
    return _MODE_STUDENT_CACHE[key]


def stage_fusion(dps, b, out_dir, train_game_ids, skip_rl, skip_llm, rl_updates=60,
                 physics=False, reward_conf=None, mode_student=None, mode_student_device='cpu',
                 tier2_adapter='auto', tier2_device='auto', unit_student=None,
                 temporal_checkpoint='none', temporal_device='cpu',
                 observable_snapshots=None):
    rule = rule_head(dps, b)
    rl = None
    if not skip_rl:
        try:
            rl = rl_head(dps, b, train_game_ids, rl_updates, physics=physics, reward_conf=reward_conf)
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
    from runtime.decide import TieredDecider
    from runtime.tier2 import resolve_tier2_backend
    from runtime.single_executor import SingleExecutorArbiter
    from runtime.decision_request import from_decision_request
    fused = fu.fuse(rule, rl, llm) if rl else fu.fuse(rule, rule, llm)
    student = _load_mode_student(mode_student, device=mode_student_device) if mode_student else None
    # ``auto`` loads the formal V5 candidate only when CUDA is available;
    # otherwise resolve_tier2_backend returns NullTier2 and the rule path remains active.
    t2 = resolve_tier2_backend(adapter=tier2_adapter, device=tier2_device)
    temporal_forecaster = None
    raw_observable_snapshots = list(observable_snapshots or [])
    # Tier2 receives one strictly issue-time snapshot per decision point.
    # Temporal forecasting may consume every one-second row below, but must
    # return to this same decision-clock alignment before inference.
    aligned_snapshots = (align_observable_snapshots(dps, raw_observable_snapshots)
                         if raw_observable_snapshots else None)
    precomputed_temporal_forecasts = None
    if temporal_checkpoint not in (None, '', 'none', 'null', '-'):
        from runtime.temporal_forecast import resolve_temporal_forecaster
        temporal_forecaster = resolve_temporal_forecaster(
            temporal_checkpoint, device=temporal_device, team='红', context=12,
            adaptive_router='auto', auto_mature_feedback=True)
        raw_snapshots = raw_observable_snapshots
        precomputed_temporal_forecasts = precompute_temporal_forecasts(
            dps, raw_snapshots, temporal_forecaster)
        aligned_snapshots = align_observable_snapshots(dps, raw_snapshots)
    # The default mainline terminates at one explicit DecisionRequest
    # executor. This adapter records a shadow receipt; it does not claim
    # hardware execution until a real controller registers in its place.
    executor_arbiter = SingleExecutorArbiter()
    executor_arbiter.register(
        'mainline_shadow',
        lambda request: {'ack_status': 'shadow_recorded', 'executed': False},
    )
    decider = TieredDecider(
        mode_student=student, tier2_backend=t2, unit_student=unit_student, team='红',
        temporal_forecaster=temporal_forecaster, as_decision_request=True,
        executor_arbiter=executor_arbiter)
    enriched_requests = decider.run_series(
        dps, b, base_decisions=fused['decisions'],
        conflict_rate=fused.get('conflict_rate') or 0.0,
        observable_snapshots=aligned_snapshots,
        temporal_forecasts=precomputed_temporal_forecasts)
    # Keep the explicit external boundary and the legacy layered internal
    # view separate. Dataset construction and replay scoring consume the
    # latter; the former is retained for dispatch/audit evidence.
    enriched = []
    for request in enriched_requests:
        layered = from_decision_request(request)
        for key in ('decision_meta', 'opponent_intent', 'semantic_situation',
                    'global_analysis', 'temporal_forecast'):
            if key in request:
                layered[key] = request[key]
        enriched.append(layered)
    fused['decisions'] = enriched
    fused['decision_requests'] = enriched_requests
    fused['n_need_llm'] = decider.stats['n_need_llm']
    fused['n_mode_student'] = decider.stats['n_mode_student']
    fused['n_rule_fallback'] = decider.stats['n_rule_fallback']
    fused['n_tier2_ok'] = decider.stats['n_tier2_ok']
    fused['n_tier2_fallback'] = decider.stats['n_tier2_fallback']
    fused['executor_name'] = executor_arbiter.executor_name
    fused['executor_dispatches'] = len(executor_arbiter.history)
    fused['executor_mode'] = 'shadow_recorded_not_hardware_executed'
    fused['opp_intent_enabled'] = True
    fused['mode_student'] = bool(student)
    fused['tier2_backend'] = type(t2).__name__
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'fused_decisions.json'), 'w', encoding='utf-8') as f:
        json.dump(fused['decisions'], f, ensure_ascii=False, indent=None)
    with open(os.path.join(out_dir, 'decision_requests.json'), 'w', encoding='utf-8') as f:
        json.dump(fused['decision_requests'], f, ensure_ascii=False, indent=None)
    print('fusion done:', json.dumps({k: (v if k != 'decisions' else f"{len(v)} pts")
                                      for k, v in fused.items()}, ensure_ascii=False))
    return fused


def stage_dataset(dps, b, out_dir, fused_decisions, team='红', temporal_forecasts=None):
    from finetune import make_dataset as md
    recs = md.make_records(
        fused_decisions,
        dps,
        bounds=b,
        team=team,
        temporal_forecasts=temporal_forecasts,
    )
    path = md.to_jsonl(recs, os.path.join(out_dir, 'm2_dataset.jsonl'))
    print('dataset written:', len(recs), '->', path)
    return recs


def stage_train(records, out_dir, base=None, max_len=1024,
                batch_size=1, grad_accum=8):
    from finetune.lora_train import train_lora
    adapter, stats = train_lora(records, out_dir=os.path.join(out_dir, 'finetuned'),
                                base=base, epochs=1, max_len=int(max_len),
                                batch_size=int(batch_size), grad_accum=int(grad_accum))
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
    ap.add_argument('--profile', default='mainline', choices=PROFILES,
                    help='mainline=交付主线(默认跳过 LLM + 注入 lookahead); research=旧实验复现')
    ap.add_argument('--stage', default='all', choices=ALL_STAGES)
    ap.add_argument('--gid', type=int, default=config.GAME_REF)
    ap.add_argument('--skip-rl', action='store_true')
    ap.add_argument('--skip-llm', action='store_true',
                    help='跳过旧融合 LLM 第三源；Tier2 由 --tier2-adapter 单独控制')
    ap.add_argument('--all-fallback', action='store_true', help='仅规则头(RL/LLM 全关)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default='out_m2')
    ap.add_argument('--train-games', type=int, default=3)
    ap.add_argument('--dataset-games', type=int, default=24)
    ap.add_argument('--eval-games', type=int, default=4)
    ap.add_argument('--rl-updates', type=int, default=80)
    ap.add_argument('--base', default=None)
    ap.add_argument('--train-max-len', type=int, default=1024,
                    help='LoRA 训练最大 token 长度；压缩后的时序预测上下文默认 1024')
    ap.add_argument('--train-batch-size', type=int, default=1,
                    help='LoRA 微批量；显存允许时可设为 2')
    ap.add_argument('--train-grad-accum', type=int, default=8,
                    help='LoRA 梯度累积步数；保持有效批量时与 train-batch-size 反向调整')
    ap.add_argument('--dataset', default=None,
                    help='train 用的数据集 jsonl(默认 out/m2_dataset.jsonl; 若要重平衡集传入对应文件)')
    ap.add_argument('--physics', action='store_true', help='RL 头用 PhysicsEnv(B2 前向物理)训练+rollout -> 真 MDP 策略')
    ap.add_argument('--reward-weights', default=None,
                    help='待学的 reward 权重 JSON(_work/rl_weights.json); 加载后接入训练 reward_conf')
    ap.add_argument('--lookahead-delta', type=int, default=0,
                    help='回看窗口帧数(>0 启用; mainline 且为 0 时自动用 '
                         f'{MAINLINE_LOOKAHEAD_DELTA}; research 保持 0=关闭)')
    ap.add_argument('--lookahead-margin', type=float, default=None,
                    help='回看攻防判定 margin(默认 0.12); 调大成均衡少、调小成极端多 (偏差调优)')
    ap.add_argument('--mode-student', default=None,
                    help='Tier1 mode 学生目录(含 adapter.pt); need_llm=False 时覆盖 global_mode')
    ap.add_argument('--mode-student-device', default='cpu',
                    help='mode 学生推理设备(默认 cpu)')
    ap.add_argument('--tier2-adapter', default='auto',
                    help='Tier2 LoRA 目录; auto=当前正式 V5 候选，无 CUDA 时自动规则回退')
    ap.add_argument('--tier2-device', default='auto',
                    help='Tier2 设备; auto=无 CUDA 则 NullTier2')
    ap.add_argument('--temporal-checkpoint', default='none',
                    help='可选时序预测检查点；启用后用可见传感器/雷达快照给 Tier2 参考')
    ap.add_argument('--temporal-device', default='cpu',
                    help='时序预测设备，默认 cpu 以控制内存')
    args = ap.parse_args()

    if args.all_fallback:
        args.skip_rl = True
        args.skip_llm = True
    args.tier2_adapter = resolve_tier2_adapter(
        args.tier2_adapter, all_fallback=bool(args.all_fallback))
    flags = apply_profile(args.profile, args.skip_llm, args.lookahead_delta, args.skip_rl)
    args.skip_llm = flags['skip_llm']
    args.lookahead_delta = flags['lookahead_delta']
    args.skip_rl = flags['skip_rl']
    print(
        f"PROFILE={flags['profile']} skip_llm={flags['skip_llm']} "
        f"lookahead_delta={flags['lookahead_delta']} skip_rl={flags['skip_rl']}",
        flush=True,
    )
    # 可学的 reward 权重(若指定): 从 reward_learn 输出 JSON 映射成 per-step reward_conf
    reward_conf = None
    if args.reward_weights:
        from rl import reward_learn as rlrn
        reward_conf = rlrn.weights_json_to_conf(args.reward_weights)
        print(f'  load learned reward weights from {args.reward_weights} -> {len(reward_conf)} 项')
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
        fusion_result = stage_fusion(
            dps, b, args.out, train_ids, args.skip_rl, args.skip_llm,
            args.rl_updates, physics=args.physics, reward_conf=reward_conf,
            mode_student=args.mode_student,
            mode_student_device=args.mode_student_device,
            tier2_adapter=args.tier2_adapter,
            tier2_device=args.tier2_device,
            temporal_checkpoint=args.temporal_checkpoint,
            temporal_device=args.temporal_device,
            observable_snapshots=load_observable_snapshots(
                gid, team='红')
            if needs_observable_snapshots(args.tier2_adapter, args.temporal_checkpoint) else None)
        # Preserve the structured temporal forecast sidecar independently of
        # the decision relabeling below.  This makes the forecast context
        # available to the LLM training prompt, instead of injecting it only
        # at runtime during Tier2 inference.
        fused = fusion_result['decisions']
        temporal_forecasts = [
            row.get('temporal_forecast') if isinstance(row, dict) else None
            for row in fused
        ]
        if not any(temporal_forecasts):
            temporal_forecasts = None
        if args.lookahead_delta > 0:
            from worldstate import worldstate as ws
            from labels import strategy_label as slb
            from collections import Counter
            ws_list = [ws.build_worldstate(dp, b) for dp in dps]
            marg = args.lookahead_margin if args.lookahead_margin is not None else slb.DEF_MARGIN
            fused, labels = slb.relabel_decision_series(ws_list, fused, args.lookahead_delta,
                                                        team='红', margin=marg)
            print(f'  look-ahead mode dist(gid={gid}, Δ={args.lookahead_delta}, margin={marg}): '
                  f'{dict(Counter(labels))}', flush=True)
        if make_records:
            recs = stage_dataset(
                dps,
                b,
                args.out,
                fused,
                temporal_forecasts=temporal_forecasts,
            )
            return recs
        return fused

    all_records = []
    if args.stage in ('fusion', 'dataset', 'all'):
        N = len(dataset_ids)
        for i, gid in enumerate(dataset_ids):
            try:
                recs = process_game(gid, make_records=(args.stage in ('dataset', 'all')))
                all_records += recs
                print(f'[dataset] game {i + 1}/{N} gid={gid} records={len(recs)}', flush=True)
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
            # 读已生成的数据集 jsonl(--dataset 优先, 否则默认 out/m2_dataset.jsonl)
            ds_path = args.dataset or os.path.join(args.out, 'm2_dataset.jsonl')
            if os.path.exists(ds_path):
                import sys
                from finetune.make_dataset import to_hf_dataset
                recs = [json.loads(l) for l in open(ds_path, encoding='utf-8') if l.strip()]
                all_records = recs
        if all_records:
            adapter, stats = stage_train(all_records, args.out, base=args.base,
                                         max_len=args.train_max_len,
                                         batch_size=args.train_batch_size,
                                         grad_accum=args.train_grad_accum)
        else:
            print('no records -> cannot train')

    report = {'profile': flags['profile'],
              'skip_llm': flags['skip_llm'],
              'lookahead_delta': flags['lookahead_delta'],
              'dataset_games': dataset_ids[:10], 'eval_games': eval_ids[:4],
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
