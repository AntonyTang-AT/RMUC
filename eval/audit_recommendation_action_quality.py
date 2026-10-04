"""Auditable action-label outcome proxies with paired stay-put baselines.

Fixed exploratory sample: the same first eight chronological games used by
audit_unit_recommendations.py, both sides. Spatial metrics compare each
recommendation on its exact issue/entity sample with a same-sample policy that
keeps the unit in its current zone. HP/target deltas are descriptive outcomes;
they are not caused by, or counterfactual to, a recommendation.
"""
from __future__ import annotations
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from data.db import connect,timeseries_for
from data.radar_visibility_replay import build_radar_visibility_snapshots
from eval.build_battle_replay_visual import _compact_current
from worldstate.functional_zone import assign_functional_zone,zone_centroid
from worldstate.unit_state import candidate_moves

GAME_IDS=(1778631047459,1778631715625,1778632575621,1778633483984,
          1778634274546,1778637960624,1778638792994,1778639477645)
SIDES=('红','蓝');MOBILE={'英雄','工程','步兵3','步兵4','空中','哨兵'}
HORIZONS=(3,10);REPS=4000;SEED=20260927

def _zone(row):
    if row is None or row['x'] is None or row['y'] is None:return None
    x,y=float(row['x']),float(row['y'])
    return None if x==0 and y==0 else assign_functional_zone(x,y)
def _dist(row,goal):
    c=zone_centroid(goal);z=_zone(row)
    if row is None or c is None or z is None:return None
    return ((float(row['x'])-c[0])**2+(float(row['y'])-c[1])**2)**.5
def _hp_ratio(row):
    if row is None:return None
    hp=row['当前血量'];mx=row['最大血量']
    return float(hp)/float(mx) if isinstance(hp,(int,float)) and isinstance(mx,(int,float)) and mx>0 else None
def _band(v):return 'unknown' if v is None else ('low' if v<.3 else 'mid' if v<.7 else 'high')
def _hops(unit,side,start,goal):
    if not start or not goal:return -1
    if start==goal:return 0
    q=[(start,0)];seen={start}
    while q:
        z,d=q.pop(0)
        for n in candidate_moves(unit,side,z):
            if n==goal:return d+1
            if n not in seen:seen.add(n);q.append((n,d+1))
    return -1
def _bootstrap(game_deltas,seed=SEED):
    vals=list(game_deltas.values())
    if not vals:return {'complete_games':0,'mean_delta':None,'ci95':[None,None]}
    rng=random.Random(seed);boots=[]
    for _ in range(REPS):boots.append(sum(rng.choice(vals) for _ in vals)/len(vals))
    boots.sort();return {'complete_games':len(vals),'mean_delta':sum(vals)/len(vals),
      'ci95':[boots[int(.025*REPS)],boots[min(REPS-1,int(.975*REPS))]]}
def _paired_action(action_rows,metric):
    """Same-sample action minus stay baseline, aggregated by game and side.

    The prior implementation equally averaged unit-type rates inside each
    game. That changed the estimand (a rare unit type counted as much as a
    common one) and made paired deltas disagree with the displayed pooled
    rates. Keep each issued recommendation as one observation, then cluster
    only at the game level.
    """
    fields={
      'goal_hits':('goal_n','goal_hits','stay_goal_hits'),
      'progress_hits':('route_n','progress_hits',None),
      'distance_reduction':('distance_n','distance_reduction',None),
    }
    n_key,action_key,stay_key=fields[metric]
    by_gs=defaultdict(lambda:{'n':0,'action':0.,'stay':0.})
    for r in action_rows:
        if r.get(n_key,0):
            n=r[n_key]
            cell=by_gs[(r['game_id'],r['side'])]
            cell['n']+=n;cell['action']+=r[action_key]
            cell['stay']+=r[stay_key] if stay_key else 0.
    by_game_side=defaultdict(dict)
    for (g,s),cell in by_gs.items():
        by_game_side[g][s]=(cell['action']-cell['stay'])/cell['n']
    side_results={}
    for side in SIDES:
        vals={g:v[side] for g,v in by_game_side.items() if side in v}
        side_results[side]=_bootstrap(vals)
    complete={g:(v['红']+v['蓝'])/2 for g,v in by_game_side.items()
              if '红' in v and '蓝' in v}
    balanced=_bootstrap(complete)
    return {'red_blue_equal_game_paired_bootstrap':balanced,'by_side':side_results,
            'game_side_cells':len(by_gs)}
def _summarize(rows):
    out={}
    for h in HORIZONS:
        selected=[r for r in rows if r['horizon']==h]
        n=sum(r['goal_n'] for r in selected)
        if not n:
            out[str(h)]={'scorable_recommendations':0,
                         'spatial_goal_metrics':{'status':'unknown',
                           'reason':'action_has_no_movement_goal_or_exact_future_zone'}}
            continue
        out[str(h)]={
          'scorable_recommendations':n,
          'target_zone_hit_rate':sum(r['goal_hits'] for r in selected)/n,
          'stay_put_baseline_target_zone_hit_rate':sum(r['stay_goal_hits'] for r in selected)/n,
          'target_zone_hit_delta_pp':100*(sum(r['goal_hits']-r['stay_goal_hits'] for r in selected)/n),
          'legal_route_progress_rate':sum(r['progress_hits'] for r in selected)/max(1,sum(r['route_n'] for r in selected)),
          'mean_legal_hops_reduced':sum(r['hop_reduction'] for r in selected)/max(1,sum(r['route_n'] for r in selected)),
          'mean_goal_distance_reduction_m':sum(r['distance_reduction'] for r in selected)/max(1,sum(r['distance_n'] for r in selected)),
          'positive_distance_progress_rate':sum(r['distance_progress_hits'] for r in selected)/max(1,sum(r['distance_n'] for r in selected)),
          'own_hp_delta_mean':sum(r['hp_delta_sum'] for r in selected)/max(1,sum(r['hp_n'] for r in selected)),
          'own_hp_scored':sum(r['hp_n'] for r in selected),
          'identified_target_hp_delta_mean':sum(r['target_hp_delta_sum'] for r in selected)/max(1,sum(r['target_hp_n'] for r in selected)),
          'identified_target_hp_scored':sum(r['target_hp_n'] for r in selected),
          'current_zone_retention_rate':sum(r['zone_retention_hits'] for r in selected)/max(1,sum(r['zone_retention_n'] for r in selected)),
          'current_zone_retention_scored':sum(r['zone_retention_n'] for r in selected),
          'own_survival_rate':sum(r['own_alive_hits'] for r in selected)/max(1,sum(r['own_alive_n'] for r in selected)),
          'own_survival_scored':sum(r['own_alive_n'] for r in selected),
          'identified_target_alive_rate':sum(r['target_alive_hits'] for r in selected)/max(1,sum(r['target_alive_n'] for r in selected)),
          'identified_target_alive_scored':sum(r['target_alive_n'] for r in selected),
        }
    return out

def main():
    # Aggregated only by action/game/side/type/horizon; avoids treating frames
    # as independent units in inference.
    actions=defaultdict(lambda:{'game_id':None,'side':None,'unit_type':None,'horizon':None,
      'goal_n':0,'goal_hits':0,'stay_goal_hits':0,'route_n':0,'progress_hits':0,
      'hop_reduction':0.,'distance_n':0,'distance_reduction':0.,'distance_progress_hits':0,
      'hp_n':0,'hp_delta_sum':0.,'target_hp_n':0,'target_hp_delta_sum':0.,
      'zone_retention_n':0,'zone_retention_hits':0,
      'own_alive_n':0,'own_alive_hits':0,
      'target_alive_n':0,'target_alive_hits':0})
    label_counts={s:Counter() for s in SIDES};eligible={s:Counter() for s in SIDES}
    feasibility_counts=defaultdict(Counter)
    copy_examples={};games_used=[]
    # No-action units provide a matched observational comparator for health
    # outcomes; they are not treated as commands to hold position.
    hp_cells=defaultdict(lambda:{'no_action':[],'action':[]})
    with connect() as con:
      for gid in GAME_IDS:
        rows=sorted(timeseries_for(con,gid),key=lambda r:(float(r['时刻秒']),r['阵营'],str(r['robot_id'])))
        if not rows:continue
        games_used.append(gid);truth=defaultdict(dict)
        for r in rows:truth[float(r['时刻秒'])][f"{r['阵营']}:{r['robot_id']}"]=r
        snapshots=build_radar_visibility_snapshots(rows,game_id=gid);first=min(truth)
        for side in SIDES:
          for snap in snapshots[side]:
            t=float(snap['issued_at'])
            if abs((t-first)/3-round((t-first)/3))>1e-6:continue
            later={h:truth.get(t+h) for h in HORIZONS}
            if not later[3]:continue
            entities=list(snap['entities'].values())
            for entity in entities:
              if entity.get('team')!=side or entity.get('entity_type') not in MOBILE:continue
              current=_compact_current(entity,entities,side)
              if current.get('life_state')=='destroyed':continue
              typ=current.get('type');zone=current.get('zone')
              if not typ or not zone:continue
              advice=(current.get('special_state') or {}).get('decision_guidance') or {}
              label=advice.get('action_label','legacy_unlabeled')
              label_counts[side][label]+=1
              self_truth=truth.get(t,{}).get(entity['entity_id'])
              hp0=_hp_ratio(self_truth)
              hpband=_band(hp0)
              # Record no-action comparison outcomes by exact game/side/type/HP band.
              for h in HORIZONS:
                actual=(later[h] or {}).get(entity['entity_id'])
                hp1=_hp_ratio(actual);delta=(hp1-hp0) if hp1 is not None and hp0 is not None else None
                cell=(gid,side,typ,hpband,h)
                if label=='no_action_evidence':
                  if delta is not None:hp_cells[cell]['no_action'].append(delta)
                elif advice.get('applicable') and advice.get('action') and delta is not None:
                  hp_cells[cell]['action'].append(delta)
              if not advice.get('applicable') or not advice.get('action'):
                continue
              eligible[side][label]+=1
              scorecard=advice.get('decision_scorecard') or {}
              feasibility_counts[label][(scorecard.get('feasibility') or {}).get('status','unknown')]+=1
              copy_examples.setdefault(label,{'action':advice.get('action'),'short_text':advice.get('short_text'),
                                               'current_state':advice.get('current_state'),
                                               'target_unit_id':advice.get('target_unit_id'),
                                               'target_unit_type':advice.get('target_unit_type'),
                                               'target_distance_m':advice.get('target_distance_m'),
                                               'target_zone':advice.get('target_zone'),
                                               'movement_goal_zone':advice.get('movement_goal_zone'),
                                               'first_step_zone':advice.get('first_step_zone'),
                                               'first_step_distance_m':advice.get('first_step_distance_m'),
                                               'path_zones':advice.get('path_zones'),
                                               'trigger':advice.get('trigger_condition'),
                                               'stop':advice.get('cancel_condition'),
                                               'evaluation':advice.get('evaluation'),
                                               'decision_scorecard':scorecard})
              # Only a movement goal is scored as a destination. Target-zone
              # (e.g. an enemy's current zone) must not be treated as a route
              # assignment. No goal means the spatial metric is unavailable.
              goal=advice.get('movement_goal_zone')
              if goal is None and label in ('retreat_low_hp','return_resource_to_supply',
                                            'continue_resource_route'):
                goal=advice.get('target_zone')
              target_info=advice.get('target_unit_id')
              if not target_info:
                target_info=((current.get('special_state') or {}).get('attacking_target') or
                             (current.get('special_state') or {}).get('possible_target') or {}).get('entity_id')
              for h in HORIZONS:
                bucket=actions[(label,gid,side,typ,h)]
                bucket.update({'game_id':gid,'side':side,'unit_type':typ,'horizon':h})
                actual=(later[h] or {}).get(entity['entity_id'])
                actual_zone=_zone(actual)
                if actual_zone is not None and goal is not None:
                  bucket['goal_n']+=1;bucket['goal_hits']+=int(actual_zone==goal)
                  bucket['stay_goal_hits']+=int(zone==goal)
                  h0=_hops(typ,side,zone,goal);ha=_hops(typ,side,actual_zone,goal)
                  if h0>=0 and ha>=0:
                    bucket['route_n']+=1;reduced=h0-ha
                    bucket['hop_reduction']+=reduced;bucket['progress_hits']+=int(reduced>0)
                  d0=_dist(self_truth,goal);d1=_dist(actual,goal)
                  if d0 is not None and d1 is not None:
                    bucket['distance_n']+=1;red=d0-d1;bucket['distance_reduction']+=red
                    bucket['distance_progress_hits']+=int(red>1e-6)
                if label=='cover_and_confirm_target' and actual_zone is not None:
                  bucket['zone_retention_n']+=1
                  bucket['zone_retention_hits']+=int(actual_zone==zone)
                hp1=_hp_ratio(actual)
                if hp1 is not None:
                  bucket['own_alive_n']+=1;bucket['own_alive_hits']+=int(hp1>0)
                if hp0 is not None and hp1 is not None:
                  bucket['hp_n']+=1;bucket['hp_delta_sum']+=hp1-hp0
                if target_info:
                  target0=truth.get(t,{}).get(str(target_info));target1=(later[h] or {}).get(str(target_info))
                  a0=_hp_ratio(target0);a1=_hp_ratio(target1)
                  if a0 is not None and a1 is not None:
                    bucket['target_hp_n']+=1;bucket['target_hp_delta_sum']+=a1-a0
                    bucket['target_alive_n']+=1
                    bucket['target_alive_hits']+=int(a1>0)
    records_by_label=defaultdict(list)
    for (label,gid,side,typ,h),row in actions.items():records_by_label[label].append(row)
    by_action={}
    for label,rows in records_by_label.items():
      by_action[label]={'sample_count':sum(r['goal_n'] for r in rows),
        'by_horizon':_summarize(rows),
        'paired_vs_stay_put':{str(h):{metric:_paired_action([r for r in rows if r['horizon']==h],metric)
                                      for metric in ('goal_hits','progress_hits','distance_reduction')}
                              for h in HORIZONS},
        'by_side':{s:_summarize([r for r in rows if r['side']==s]) for s in SIDES},
        'by_unit_type':{typ:{'by_horizon':_summarize([r for r in rows if r['unit_type']==typ]),
          'paired_vs_stay_put':{str(h):{metric:_paired_action([r for r in rows if r['unit_type']==typ and r['horizon']==h],metric)
                                        for metric in ('goal_hits','progress_hits','distance_reduction')}
                                for h in HORIZONS}}
          for typ in sorted({r['unit_type'] for r in rows})}}
    action_matched_metrics={}
    for label,rows in records_by_label.items():
      horizons={}
      for h in HORIZONS:
        selected=[r for r in rows if r['horizon']==h]
        n_zone=sum(r['zone_retention_n'] for r in selected)
        n_target=sum(r['target_alive_n'] for r in selected)
        n_hp=sum(r['hp_n'] for r in selected)
        n_alive=sum(r['own_alive_n'] for r in selected)
        n_target_hp=sum(r['target_hp_n'] for r in selected)
        common={
          'zone_retention': ({'status':'scored','n':n_zone,
             'rate':sum(r['zone_retention_hits'] for r in selected)/n_zone} if n_zone else
             {'status':'unknown','reason':'no_exact_future_zone_observation'}),
          'own_survival': ({'status':'scored','n':n_alive,
             'rate':sum(r['own_alive_hits'] for r in selected)/max(1,sum(r['own_alive_n'] for r in selected))}
             if n_alive else
             {'status':'unknown','reason':'exact_future_hp_missing'}),
          'own_hp_change': ({'status':'scored','n':n_hp,
             'mean_delta':sum(r['hp_delta_sum'] for r in selected)/n_hp,
             'interpretation':'observational; not attributed to this advice'} if n_hp else
             {'status':'unknown','reason':'exact_hp_missing'}),
        }
        if label=='continue_attack_visible_target':
          common['identified_target_alive']=({'status':'scored','n':n_target,
             'rate':sum(r['target_alive_hits'] for r in selected)/n_target,
             'interpretation':'observational; target outcomes are confounded by all combatants'} if n_target else
             {'status':'unknown','reason':'target_id_or_exact_hp_missing'})
          common['identified_target_hp_change']=({'status':'scored','n':n_target_hp,
             'mean_delta':sum(r['target_hp_delta_sum'] for r in selected)/n_target_hp,
             'interpretation':'observational and confounded; not attributed to this advice'} if n_target_hp else
             {'status':'unknown','reason':'target_id_or_exact_hp_missing'})
        horizons[str(h)]=common
      action_matched_metrics[label]={
        'by_horizon':horizons,
        'resource_completion': {'status':'unknown','reason':'competition_timeseries_has_no_per_robot_carry_or_exchange_field'}
          if label in ('return_resource_to_supply','continue_resource_route') else None,
        'defense_effect': {'status':'unknown','reason':'no_explicit_defense_order_or_facility_damage_attribution_field'}
          if label=='guard_defense_zone' else None,
        'enemy_confirmation': {'status':'unknown','reason':'radar future visibility/target confirmation is not in ground-truth rows'}
          if label=='cover_and_confirm_target' else None,
      }
    decision_scorecard_by_action={}
    for label, statuses in feasibility_counts.items():
      issued=sum(eligible[s][label] for s in SIDES)
      route_scored=statuses.get('scored',0)+statuses.get('failed',0)
      route_legal=statuses.get('scored',0)
      consequence_coverage={}
      for h, metrics in action_matched_metrics[label]['by_horizon'].items():
        consequence_coverage[h]={}
        for name, value in metrics.items():
          if isinstance(value,dict) and 'n' in value:
            consequence_coverage[h][name]={
              'scored':value['n'], 'issued':issued,
              'coverage':value['n']/issued if issued else None,
              'status':value.get('status','scored'),
            }
          elif isinstance(value,dict):
            consequence_coverage[h][name]={**value,'scored':0,'issued':issued,
                                           'coverage':0.0 if issued else None}
      decision_scorecard_by_action[label]={
        'issued_recommendations':issued,
        'feasibility':{
          'route_applicable':route_scored,
          'route_coverage':route_scored/issued if issued else None,
          'legal_route_rate':route_legal/route_scored if route_scored else None,
          'no_zone_change_required':statuses.get('not_required',0),
          'unknown':statuses.get('unknown',0),
          'interpretation':'functional-zone reachability only; not success probability or ETA',
        },
        'execution':{
          'status':'unknown','scored':0,'unknown':issued,
          'coverage':0.0 if issued else None,
          'reason':'no controller command/acknowledgement logs; observed movement is not proof of execution',
        },
        'consequences_by_horizon':consequence_coverage,
        'causal_benefit':{'status':'not_identifiable','scored':0,'coverage':0.0 if issued else None,
          'reason':'replay does not intervene on controls or identify the counterfactual outcome'},
      }
    # Matched action/no-action HP comparison: exact game, side, unit type,
    # initial HP band, and horizon. This is observational, not counterfactual.
    hp_deltas=defaultdict(lambda:defaultdict(list))
    hp_support=Counter()
    for (gid,side,typ,band,h),vals in hp_cells.items():
      if vals['action'] and vals['no_action']:
        d=sum(vals['action'])/len(vals['action'])-sum(vals['no_action'])/len(vals['no_action'])
        hp_deltas[h][(gid,side)].append((typ,band,d));hp_support[h]+=1
    hp_comparison={}
    for h in HORIZONS:
      gs=defaultdict(lambda:defaultdict(list))
      for (gid,side),strata in hp_deltas[h].items():
        for typ,band,d in strata:gs[gid][side].append(d)
      side_ci={}
      for side in SIDES:
        vals={g:sum(sv[side])/len(sv[side]) for g,sv in gs.items() if sv[side]}
        side_ci[side]=_bootstrap(vals)
      balanced={g:(sum(x['红'])/len(x['红'])+sum(x['蓝'])/len(x['蓝']))/2
                for g,x in gs.items() if x['红'] and x['蓝']}
      hp_comparison[str(h)]={'matched_game_side_type_hpband_strata':int(hp_support[h]),
        'action_minus_no_action_observed_hp_delta':{'red_blue_equal_game_paired_bootstrap':_bootstrap(balanced),'by_side':side_ci},
        'interpretation':'matched observational HP-change comparison; no-action does not mean controller hold and this is not a causal effect'}
    report={'schema':'recommendation-action-quality-audit-v1','games':games_used,'sides':list(SIDES),
      'issue_cadence_seconds':3,'horizons_seconds':list(HORIZONS),
      'design':{'spatial_baseline':'same unit-time, same start/goal: stay-put predicts current zone and zero route progress; compare actual t+3/t+10 zone and legal shortest-path progress.',
       'stratification':'game × side × unit type for route scores; compare recommendation with its own same-sample stay-put baseline. Game is the bootstrap cluster.',
       'hp_control':'for health changes only, compare actionable recommendation observations to no-action-evidence observations matched by game/side/unit type/initial HP band; descriptive association only.',
       'target_control':'identified target HP change is descriptive because no comparable target-bearing no-action counterfactual exists.',
       'noncausal_warning':'No controller command log exists. Observed movement/HP/target changes do not establish recommendation execution or benefit.'},
      'action_label_counts_by_side':{s:dict(label_counts[s]) for s in SIDES},
      'explicit_action_counts_by_side':{s:dict(eligible[s]) for s in SIDES},
      'action_copy_examples':copy_examples,'by_action_label':by_action,
      'action_matched_metrics':action_matched_metrics,
      'decision_scorecard_by_action':decision_scorecard_by_action,
      'metric_availability':{
        'attack_target_hp_and_survival':'scored where exact target ID and HP rows exist; confounded by all combatants',
        'possible_attack_target_confirmation':{'status':'unknown','reason':'radar future visibility/target confirmation is absent'},
        'defense_facility_protection':{'status':'unknown','reason':'no explicit defense order and no per-facility damage attribution in this telemetry'},
        'resource_arrival_or_exchange':{'status':'unknown','reason':'competition timeseries does not contain per-robot carried-resource/exchange fields'},
        'retreat_survival_and_hp_change':'scored descriptively from exact future HP; no causal attribution',
        'controller_execution':{'status':'unknown','reason':'no controller command/acknowledgement log'},
      },
      'matched_hp_comparison':hp_comparison,
      'limitations':['Fixed first-eight-game exploratory sample; not a fresh blind test.',
       'Goal-hit and route progress are observed-behavior compatibility proxies, not execution or causal effectiveness.',
       'HP and target changes are affected by all units and game events.',
       'Target HP is scored only where the currently visible target ID is present in exact truth rows.']}
    out=ROOT/'out_m2'/'recommendation_action_quality_audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'games':games_used,'explicit_actions':report['explicit_action_counts_by_side'],
      'by_action_label':{a:{h:{k:v for k,v in d.items() if k in ('scorable_recommendations','target_zone_hit_rate','target_zone_hit_delta_pp','legal_route_progress_rate','mean_legal_hops_reduced','mean_goal_distance_reduction_m','own_hp_delta_mean','identified_target_hp_delta_mean')} for h,d in q['by_horizon'].items()} for a,q in by_action.items()},
      'matched_hp_comparison':hp_comparison},ensure_ascii=False))

if __name__=='__main__':main()
