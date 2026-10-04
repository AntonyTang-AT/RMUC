# -*- coding: utf-8 -*-
"""M2 留出评估: 加载 base+LoRA 适配器, 对决策点 decode 分层决策, 度量 legal/agreement/compliance/conflict。
torch/transformers/peft 惰性 import; 模块级零 torch。单点生成见 finetune.infer_one。"""
from schema import layered as sl
from tree import global_mode as gm
from finetune import infer_one as io

# 哨兵三态 -> move(用于与规则头 agreement 比对)
_SENTRY_TO_MOVE = {'def': '回防', 'hitbuilding': '推进', 'cruise': '游击'}


def _extract_decisions(text):
    return io.extract_decision(text)


def eval_model(adapter_path, dps, bounds, base=None, device='cuda', max_new=200, batch=None,
               return_predictions=False):
    """返回 {'n','legal_rate','agreement_with_rule','compliance_rate','conflict_rate'}。
    若 return_predictions=True, 追加 'predictions'(合法决策 dict 列表)与 'states'(对齐 worldstate 列表),
    供上层的 global_mode_align_gap 做逐点胜势对齐(见 Track 1 retrain_eval)。"""
    from worldstate import worldstate as ws

    tok, model = io.load_lora_bundle(adapter_path, base=base, device=device)

    decs = []
    states = []
    sent_moves = []
    rule_sent = []
    for dp in dps:
        w = ws.build_worldstate(dp, bounds)
        d = io.generate_with_bundle(tok, model, dp, bounds, max_new=max_new)
        if d is not None:
            decs.append(d)
            states.append(w)
            sent_moves.append(d['units'].get('哨兵', {}).get('move', '无'))
        rule_sent.append(_SENTRY_TO_MOVE.get(gm.derive_sentry_state(w), '游击'))

    n = len(dps)
    legal = sum(1 for d in decs if sl.validate(d)[0]) / max(n, 1)
    agree = sum(1 for a, b in zip(sent_moves, rule_sent) if a == b) / max(len(decs), 1)
    conf = sl.conflict_rate(decs) if len(decs) > 1 else 0.0
    res = {'n': len(decs), 'legal_rate': legal,
           'agreement_with_rule': agree, 'conflict_rate': conf}
    if return_predictions:
        res['predictions'] = decs
        res['states'] = states
    return res
