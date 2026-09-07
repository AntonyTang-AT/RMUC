# -*- coding: utf-8 -*-
"""用 DeepSeek(OpenAI 兼容) API 真实跑一次 M1 LLM 慢环: WorldState -> 哨兵 3 状态。

复用 M1 的构建/解析/快环/评测接口, 仅把 model/predict.llm_decide(本地 transformers)
替换为 API 调用, 不再需要 GPU。key/base/model 全部从 .env 读取(.env 已 gitignore)。

用法:
  python run_llm_api.py                     # 对 GAME_REF 头 --limit 个决策点跑(默认 20)
  python run_llm_api.py --limit 0           # 跑满该局全部决策点
  python run_llm_api.py --ping              # 仅做一次连通性自检
"""
import os, json, time, argparse, urllib.request, urllib.error

import config
from data import db, features
from worldstate import worldstate as ws
from labels import sentry_labels as sl
from fastloop import behavior as bh
from eval import metrics
from model import prompt
from tree import global_mode as gm

BASE_URL = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')


def _situation(w):
    """把世界状态压缩成哨兵决策的判别性摘要(与 label_sentry 的三个触发器对齐)。

    标注器: hitbuilding = 哨兵越中线且 dfire>0; def = 敌方进入己方半场; 其余 cruise。
    只在推理时给 LLM 一个能读懂的高层语义, 原始 JSON 仍一并提供。
    """
    mode = gm.derive_global_mode(w)
    reds = [r for r in w['robots'].values() if r['team'] == '红']
    blues = [r for r in w['robots'].values() if r['team'] == '蓝']
    enemy_in_half = any(r['norm_xy'][0] < 0.5 for r in blues)  # 蓝方进入红半场
    sentry = next((r for r in reds if r['type'] == '哨兵'), None)
    sentry_past_mid = bool(sentry and sentry['norm_xy'][0] > 0.5)
    return {
        'mode': mode,
        'enemy_in_half': enemy_in_half,
        'sentry_past_mid': sentry_past_mid,
        'sentry_hp': sentry['hp_ratio'] if sentry else None,
        'sentry_dfire': sentry['dfire'] if sentry else 0.0,
        'red_n': len(reds), 'blue_n': len(blues),
        'red_push': int(sum(1 for r in reds if r['norm_xy'][0] > 0.5)),
        'blue_push': int(sum(1 for r in blues if r['norm_xy'][0] < 0.5)),
    }


def _render_situation(s):
    return (
        "局势摘要:\n"
        f"  全局模式: {s['mode']}\n"
        f"  敌方进入我方半场: {'是' if s['enemy_in_half'] else '否'}\n"
        f"  哨兵越过中线: {'是' if s['sentry_past_mid'] else '否'}\n"
        f"  哨兵血量: {s['sentry_hp']}\n"
        f"  哨兵开火(dfire): {s['sentry_dfire']}\n"
        f"  我方/敌方单位数: {s['red_n']}/{s['blue_n']}  推进 我方/敌方: {s['red_push']}/{s['blue_push']}"
    )


def _build_user_msg(w):
    """构造带局势摘要 + few-shot 的用户消息: 让 LLM 在有判别力语义上做推理。"""
    s = _situation(w)
    return "\n".join([
        "判定规则(从重要到次要): 敌方进入我方半场 -> def; 哨兵越线并进攻建筑 -> hitbuilding; 其余 -> cruise。",
        "多数情况是 cruise, 仅当发生上述情况才切换。",
        "示例1: 敌方未进入我方半场 -> cruise",
        "示例2: 敌方进入我方半场 -> def",
        "示例3: 哨兵越线并攻击建筑 -> hitbuilding",
        _render_situation(s),
        "完整世界状态(JSON, 数值供参考): %s" % json.dumps(w, ensure_ascii=False),
    ])


def _load_env():
    """解析 .env, 返回 {key, base_url, model}。key 缺失则报错(其余用默认)。"""
    cfg = {'DEEPSEEK_API_KEY': os.environ.get('DEEPSEEK_API_KEY'),
           'DEEPSEEK_BASE_URL': BASE_URL,
           'DEEPSEEK_MODEL': MODEL}
    p = os.path.join(config.BASE, '.env')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k in cfg:
                    cfg[k] = v
    if not cfg['DEEPSEEK_API_KEY']:
        raise RuntimeError(f'缺少 DEEPSEEK_API_KEY: 请写入 {p} 或设环境变量')
    return cfg


def _chat(system, user, api_key, max_tokens=64, temp=0.0, base_url=None, model=None, debug=False):
    """单次 DeepSeek chat 调用, 返回助手文本。"""
    base_url = base_url or BASE_URL
    model = model or MODEL
    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': user}],
        'max_tokens': max_tokens,
        'temperature': temp,
        'stream': False,
    }
    url = f'{base_url}/chat/completions'
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json',
                 'Authorization': f'Bearer {api_key}'},
        method='POST')
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode('utf-8')
    data = json.loads(raw)
    if debug:
        return data, url
    return data['choices'][0]['message']['content']


def llm_state(w, api_key, base_url=None, model=None):
    """LLM 慢环: 输入世界状态 -> 哨兵 3 状态。单次 API 调用。"""
    text = _chat(prompt.SYSTEM, _build_user_msg(w), api_key, base_url=base_url, model=model)
    parsed = prompt.parse_decision(text)
    return parsed['state'], text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='*', type=int, default=None)
    ap.add_argument('--limit', type=int, default=20, help='跑多少个决策点(0=全部)')
    ap.add_argument('--ping', action='store_true', help='仅连通性自检')
    ap.add_argument('--debug', action='store_true', help='打印 DeepSeek 原始响应')
    args = ap.parse_args()

    cfg = _load_env()
    api_key = cfg['DEEPSEEK_API_KEY']
    base_url, model = cfg['DEEPSEEK_BASE_URL'], cfg['DEEPSEEK_MODEL']

    if args.ping:
        data, _url = _chat('你是测试助手,只回 ok。', 'ping', api_key,
                           max_tokens=8, base_url=base_url, model=model, debug=True)
        content = data['choices'][0]['message']['content'] if 'choices' in data else None
        print(json.dumps({
            'model': model, 'base_url': base_url,
            'http_choices': data.get('choices'), 'content': content,
        }, ensure_ascii=False))
        return

    con = db.connect()
    gids = args.games or [config.GAME_REF]
    rules = bh.Rules()
    samples = []
    for gid in gids:
        rows = db.timeseries_for(con, gid)
        dps = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps)
        series = sl.label_sentry_series(dps, b, team='红', sentry_rid=7)
        if args.limit > 0:
            dps = dps[:args.limit]
            series = series[:args.limit]
        for dp, (t, true_state) in zip(dps, series):
            w = ws.build_worldstate(dp, b)
            t0 = time.time()
            raw_state, raw_text = llm_state(w, api_key, base_url, model)
            lat_ms = (time.time() - t0) * 1000
            final = bh.decide_fast({'robots': w['robots'], 'bot_state': raw_state}, rules)
            samples.append({'t': t, 'true': true_state,
                            'llm': raw_state, 'final': final['state'],
                            'lat_ms': round(lat_ms, 1), 'raw': raw_text})
    con.close()

    n = len(samples)
    if n == 0:
        print("无决策点可跑。")
        return
    y_true = [s['true'] for s in samples]
    y_llm = [s['llm'] for s in samples]
    y_final = [s['final'] for s in samples]

    def rep(name, y_pred):
        return {
            'accuracy': metrics.accuracy(y_true, y_pred),
            'f1_macro': metrics.f1_macro(y_true, y_pred),
            'legal_rate': metrics.legal_rate([{'state': s} for s in y_pred]),
            'confusion': metrics.confusion(y_true, y_pred, list(sl.SENTRY_STATES)),
        }

    print(json.dumps({
        'games': gids, 'n': n, 'limit_used': args.limit,
        'model': model,
        'mean_latency_ms': round(sum(s['lat_ms'] for s in samples) / n, 1),
        'label_distribution': {s: y_true.count(s) for s in sl.SENTRY_STATES},
        'llm_raw': rep('llm', y_llm),
        'llm+fastloop_final': rep('final', y_final),
        'agreement_llm_vs_final': sum(1 for a, b in zip(y_llm, y_final) if a == b) / n,
        'sample': samples,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
