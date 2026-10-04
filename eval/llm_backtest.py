# -*- coding: utf-8 -*-
"""把 LLM(微调) 决策接入 outcome-based backtest: 测 LLM 本身与真实胜负的有利度对齐。

与 eval/outcome_backtest.py 的区别: 那个只测了**规则**(label_sentry)的对齐 — 因为规则是模型被
训练模仿的对象。本模块加载微调模型, 在决策点上跑真实的 LLM 分层决策, 记录每点的
(sentry_move, fav) 与 (global_mode, fav), 再用"决策进攻度 -> 有利度"的对齐度度量 LLM 是否
在**低有利度(受威胁)倾向守、高有利度(占优)倾向攻** — 锚定真实胜负(胜势有利度), 非规则一致。

★ 诚实前提: 仍是**关联性/条件有利度**(env 是 replay, 动作不改未来; 因果需前向动力学-路线 B)。
★ LLM 是 3B+adapter, 用与 eval_honest 完全相同的 4bit 加载 + 分层模板(训练分布), 保证可比。

纯逻辑部分(move->aggression, fav_by_bucket)离线可测; 加载模型/解码需 GPU(评估结束后再跑)。

用法:
  python -m eval.llm_backtest --adapter out_m2/finetuned_rebalanced/adapter \
      --games 1778668127515 1778668980465 --limit 30 --out _work/llm_backtest.json
"""
import os, sys, json, time, argparse, importlib.util, hashlib, inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

# 哨兵 move -> 进攻度(越高越攻)。与 brief 三状态的语义梯度对齐: def(守)<cruise<侵略(攻)。
MOVE_AGGRESSION = {'推进': 1.0, '游击': 0.5, '绕行': 0.0, '无': 0.0, '坚守': -0.5, '回防': -1.0}
# 全局模式 -> 进攻度
MODE_AGGRESSION = {'进攻': 1.0, '均衡': 0.0, '防守': -1.0}


def aggression_of_move(move, default=0.0):
    return MOVE_AGGRESSION.get(move, default)


def aggression_of_mode(mode, default=0.0):
    return MODE_AGGRESSION.get(mode, default)


def alignment(aggr_fav_pairs):
    """aggr_fav_pairs: [(aggression, fav)] -> {mean_fav_by_action, def_vs_aggr_gap}。
    返回 def(进攻度<0) vs 攻(进攻度>0) 的平均有利度差: 负=守落在更低有利度, 合理。
    纯函数, 可单测。"""
    def_fav = [f for a, f in aggr_fav_pairs if a < 0]
    aggr_fav = [f for a, f in aggr_fav_pairs if a > 0]
    def_mean = float(np.mean(def_fav)) if def_fav else None
    aggr_mean = float(np.mean(aggr_fav)) if aggr_fav else None
    gap = (def_mean - aggr_mean) if (def_mean is not None and aggr_mean is not None) else None
    return {'def_mean_fav': def_mean, 'aggr_mean_fav': aggr_mean, 'def_minus_aggr_gap': gap,
            'n_def': len(def_fav), 'n_aggr': len(aggr_fav)}


def make_row_record(*, game_id, worldstate, prompt, raw_output, decision, error,
                    latency_s, prompt_tokens, completion_tokens, adapter,
                    run_metadata=None, visibility_policy='unspecified'):
    """Create one immutable, issue-time-only record for later response scoring.

    Future outcomes must be joined by (game_id, issued_at_s) in a separate
    evaluator; never add them to this record before generation. The caller must
    label input visibility because an issue-time replay snapshot may still be
    omniscient.
    """
    return {
        'schema_version': 1,
        'game_id': int(game_id),
        'issued_at_s': float(worldstate['t']),
        'perspective': 'both_sides',
        'visibility_policy': visibility_policy,
        'input_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        'prompt_text': prompt,
        'issue_state': worldstate,
        'adapter': str(adapter),
        'run_metadata': run_metadata or {},
        'raw_output': raw_output,
        'decision': decision,
        'parse_valid': decision is not None,
        'parse_error': error or None,
        'latency_s': float(latency_s),
        'prompt_tokens': int(prompt_tokens),
        'completion_tokens': int(completion_tokens),
    }


def _sha256_tree(path):
    """Hash an adapter file/tree once; do not hash the multi-GB base model."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    if not os.path.isdir(path):
        return None
    tree = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace('\\', '/')
            tree.update(rel.encode('utf-8'))
            tree.update(b'\0')
            with open(full, 'rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    tree.update(chunk)
    return tree.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--games', nargs='+', type=int, required=True)
    ap.add_argument('--max-new', type=int, default=200)
    ap.add_argument('--max-len', type=int, default=896)
    ap.add_argument('--limit', type=int, default=0, help='每局隔点抽样到 N 决策点(0=全部)')
    ap.add_argument('--out', default='_work/llm_backtest.json')
    ap.add_argument('--rows-out', default='',
                    help='可选：逐点JSONL路径，保留原始输出与当前输入快照；不含未来结果')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch
    from data import db, features

    # 复用 eval_honest(_template/_extract_decision/BASE), 用 importlib 避免依赖 _work 是包
    _eh_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_work', 'eval_honest.py')
    _spec = importlib.util.spec_from_file_location('eval_honest', _eh_path)
    eh = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(eh)
    from worldstate import worldstate as ws
    from eval import outcome_backtest as ob

    BASE = eh.BASE
    print('load base', BASE)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    _idx = torch.cuda.current_device()
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={'': _idx})
    base_for_adapter = getattr(__import__('finetune.lora_train', fromlist=['LORA_BASE']), 'LORA_BASE')
    meta = os.path.join(os.path.dirname(args.adapter), 'meta.json')
    if os.path.exists(meta):
        base_for_adapter = json.load(open(meta, encoding='utf-8')).get('base', base_for_adapter)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    model.eval()
    print('loaded adapter', args.adapter, 'base_from_meta', base_for_adapter)
    run_metadata = {
        'base_model_id': str(BASE),
        'adapter_path': os.path.abspath(args.adapter),
        'adapter_tree_sha256': _sha256_tree(args.adapter),
        'base_model_from_adapter_meta': str(base_for_adapter),
        'prompt_template_sha256': hashlib.sha256(
            inspect.getsource(eh._template).encode('utf-8')
        ).hexdigest(),
        'system_prompt_sha256': hashlib.sha256(eh.SYSTEM.encode('utf-8')).hexdigest(),
        'generation': {'max_new_tokens': int(args.max_new),
                       'max_prompt_tokens': int(args.max_len),
                       'do_sample': False, 'use_cache': True},
    }

    con = db.connect()
    # (move,fav) 与 (mode,fav) 对
    mv_pairs, mo_pairs = [], []
    fails = 0
    rows_path = args.rows_out or (os.path.splitext(args.out)[0] + '.rows.jsonl')
    row_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(rows_path)), exist_ok=True)
    rows_file = open(rows_path, 'w', encoding='utf-8')
    for gid in args.games:
        rows = db.timeseries_for(con, gid)
        dps_all = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps_all)
        if args.limit and len(dps_all) > args.limit:
            n_all = len(dps_all)
            step = max(1, n_all // args.limit)
            inds = sorted(set(list(range(0, n_all, step))))[:args.limit]
            dps = [dps_all[i] for i in inds]
        else:
            dps = dps_all
        print(f'[game {gid}] {len(dps)} pts', flush=True)
        t0 = time.time()
        for i, dp in enumerate(dps):
            w = ws.build_worldstate(dp, b)
            prompt = eh._template(w)
            ids = tok(prompt, return_tensors='pt', truncation=True, max_length=args.max_len)
            input_ids, attn = ids.input_ids.to(model.device), ids.attention_mask.to(model.device)
            infer_t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(input_ids, attention_mask=attn, max_new_tokens=args.max_new,
                                     do_sample=False, pad_token_id=tok.eos_token_id,
                                     eos_token_id=tok.eos_token_id, use_cache=True)
            latency_s = time.perf_counter() - infer_t0
            text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            d, err = eh._extract_decision(text)
            row_record = make_row_record(
                game_id=gid, worldstate=w, prompt=prompt, raw_output=text,
                decision=d, error=err, latency_s=latency_s,
                prompt_tokens=int(input_ids.shape[1]),
                completion_tokens=int(out.shape[1] - input_ids.shape[1]),
                adapter=args.adapter, run_metadata=run_metadata,
                visibility_policy='legacy_complete_timeseries_current_state_not_radar_masked')
            rows_file.write(json.dumps(row_record, ensure_ascii=False, separators=(',', ':')) + '\n')
            row_count += 1
            if d is None:
                fails += 1
                continue
            fav = ob.fav_from_worldstate(w)
            mv_pairs.append((aggression_of_move(d['units'].get('哨兵', {}).get('move', '无')), fav))
            mo_pairs.append((aggression_of_mode(d.get('global_mode', '均衡')), fav))
        print(f'  {len(dps)} pts in {int(time.time() - t0)}s', flush=True)
    rows_file.close()
    con.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        'adapter': args.adapter,
        'input_visibility_policy': 'legacy_complete_timeseries_current_state_not_radar_masked',
        'n_points': len(mv_pairs),
        'n_failed_decode': fails,
        'row_records': row_count,
        'row_records_path': rows_path,
        'sentry_move_alignment': alignment(mv_pairs),
        'global_mode_alignment': alignment(mo_pairs),
        'caveat': ('输入使用完整比赛回放当前帧，并非雷达可见部署快照；不得作为部署质量结论。'
                   '另：关联性/条件有利度, 非因果(env 是 replay)。'
                   'def_minus_aggr_gap<0: 守(进攻度<0)落在更低有利度=合理.'),
        'rule_ref': {'align_gap_def_minus_cruise': -0.371},   # 来自 outcome_backtest(label_sentry)
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'type': 'SAVE', 'out': args.out, 'rows_out': rows_path,
                      'row_records': row_count, 'n': len(mv_pairs),
                      'sentry_gap': result['sentry_move_alignment']['def_minus_aggr_gap'],
                      'mode_gap': result['global_mode_alignment']['def_minus_aggr_gap']},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
