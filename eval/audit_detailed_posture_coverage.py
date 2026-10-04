"""Exploratory coverage/cross-tab audit for the rule-derived detailed posture.

Uses the same fixed first-eight-game sample as the recommendation audit,
red/blue radar-view snapshots every 15 seconds. No labels are called truth:
the cross-tab with the broad mode is a consistency view, not a confusion
matrix against ground truth.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from data.db import connect, timeseries_for
from data.radar_visibility_replay import build_radar_visibility_snapshots
from forecast.temporal import MAP_X, MAP_Y
from worldstate.semantic import derive_global_analysis

GAME_IDS=(1778631047459,1778631715625,1778632575621,1778633483984,
          1778634274546,1778637960624,1778638792994,1778639477645)
SIDES=('红','蓝')

def world_from_snapshot(snapshot):
    world={'t':float(snapshot['issued_at']),'robots':{},'buildings':{},'team':{}}
    for entity in (snapshot.get('entities') or {}).values():
        typ=entity.get('entity_type'); side=entity.get('team')
        pos=entity.get('position')
        if not typ or side not in SIDES: continue
        hp=entity.get('hp'); mx=entity.get('maxhp')
        ratio=(float(hp)/float(mx) if isinstance(hp,(int,float)) and isinstance(mx,(int,float)) and mx>0 else None)
        if typ in ('基地','前哨站'):
            world['buildings'][entity['entity_id']]={'type':typ,'team':side,'hp_ratio':ratio}
        elif pos and len(pos)>=2:
            world['robots'][entity['entity_id']]={'type':typ,'team':side,
                'norm_xy':[float(pos[0])/MAP_X,float(pos[1])/MAP_Y],
                'hp_ratio':ratio,'life_state':entity.get('life_state')}
    return world

def main():
    counts=Counter(); cross=defaultdict(Counter); game_rows=Counter(); samples=0
    codes=Counter(); prior=defaultdict(dict); games=[]
    with connect() as con:
      for gid in GAME_IDS:
        rows=timeseries_for(con,gid)
        if not rows: continue
        games.append(gid)
        snapshots=build_radar_visibility_snapshots(rows,game_id=gid)
        first=min(float(r['时刻秒']) for r in rows)
        for side in SIDES:
          for snap in snapshots[side]:
            t=float(snap['issued_at'])
            k=round((t-first)/15)
            if abs((t-first)-15*k)>1e-6: continue
            w=world_from_snapshot(snap)
            result=derive_global_analysis(w,team=side,prev_w=prior[(gid,side)].get('world'))
            prior[(gid,side)]={'t':t,'world':w}
            posture=result['detailed_posture']; code=posture['code']
            counts[code]+=1; codes[code]+=1; cross[code][result['current_mode']]+=1
            game_rows[(gid,side,code)]+=1; samples+=1
    fallback=counts['formation_and_patrol']
    report={'schema':'detailed-posture-coverage-audit-v1','sample_games':games,
      'sample_cadence_seconds':15,'view':'red/blue radar snapshots','samples':samples,
      'labels_are_rule_outputs_not_ground_truth':True,
      'coverage':{'non_fallback_n':samples-fallback,'non_fallback_rate':(samples-fallback)/samples if samples else None,
                  'fallback_n':fallback,'fallback_rate':fallback/samples if samples else None},
      'posture_counts':dict(counts),
      'posture_by_coarse_mode':{code:dict(v) for code,v in cross.items()},
      'limitations':['This measures rule coverage, not strategy correctness or action benefit.',
                     'Cross-tab with attack/defense/balanced is not a ground-truth confusion matrix.',
                     'Fixed exploratory sample; visible-state replay approximation.']}
    out=ROOT/'out_m2'/'detailed_posture_coverage_audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# Detailed posture coverage audit','','- Validation unit: fixed first-eight-game exploratory sample, red and blue radar view.','- Sampling: every 15 seconds.','- Interpretation: output coverage only; no ground-truth strategy label exists.','','## Rule coverage','',f"- Non-fallback: {samples-fallback}/{samples} ({report['coverage']['non_fallback_rate']:.1%})" if samples else '- No samples',f"- Fallback: {fallback}/{samples} ({report['coverage']['fallback_rate']:.1%})" if samples else '', '', '## Counts and coarse-mode cross-tab','','| Detailed posture code | Count | Defensive | Balanced | Offensive |','|---|---:|---:|---:|---:|']
    for code,n in counts.most_common():
      v=cross[code];md.append(f"| {code} | {n} | {v.get('防守',0)} | {v.get('均衡',0)} | {v.get('进攻',0)} |")
    md += ['', 'The cross-tab is descriptive consistency information, not a confusion matrix against truth. It does not establish which posture is tactically correct.']
    (ROOT/'docs'/'analysis'/'2026-09-27-detailed-posture-coverage-audit.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'samples':samples,'coverage':report['coverage'],'posture_counts':dict(counts)},ensure_ascii=False))

if __name__=='__main__': main()
