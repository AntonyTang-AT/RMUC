"""Frozen train-only OOF experiment adding causal motion features.

Only death-aware train JSONL/NPZ and the already-computed train OOF candidate
artifacts are opened. Motion features use current observable positions and the
immediately previous issue-time snapshot for the same tracked entity, with
actual elapsed time (nominally 3s) used for normalization. No targets are read
as features; no validation/test data is accepted.
"""
from __future__ import annotations

import argparse, gzip, json, time, sys
from pathlib import Path
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from forecast.temporal import HORIZONS, MODES, MOBILE_TYPES, load_compact
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS, locate_zone, zone_index
from eval.experiment_death_aware_mode_lightgbm import features as direct_features
from eval.experiment_death_aware_zone_mass_oof import metrics, game_scores, bootstrap_delta

TYPE_ORDER=tuple(sorted(MOBILE_TYPES)); ZONES=tuple(CANONICAL_ZONE_LABELS)
MOBILE_MAX=6; MAP_W=28.0; MAP_H=15.0

def finite(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(float(v))

def named_bootstrap_delta(candidate_scores, reference_scores, *, candidate_name,
                          reference_name, seed, reps):
    """Keep paired-CI field names truthful for each comparator."""
    result = bootstrap_delta(candidate_scores, reference_scores, seed, reps)
    payload = result.pop('delta_candidate_minus_direct_lgbm')
    result[f'delta_{candidate_name}_minus_{reference_name}'] = payload
    return result

def as_entities(v):
    if isinstance(v,dict): return list(v.values())
    if isinstance(v,list): return v
    return []

def current_mobile(items, issue):
    out=[]
    for e in items:
        if e.get('entity_type') not in MOBILE_TYPES: continue
        if e.get('presence_state') not in (None,'present'): continue
        if e.get('observability_state') not in (None,'current'): continue
        p=e.get('position')
        if not (isinstance(p,(tuple,list)) and len(p)>=2 and finite(p[0]) and finite(p[1])): continue
        obs_at=e.get('observed_at',issue)
        if not finite(obs_at) or float(obs_at)>float(issue)+1e-6: continue
        eid=e.get('entity_id')
        if eid is None: continue
        out.append(e)
    return out

def motion_feature_names():
    """Names parallel the exact block order appended to each row."""
    names=[]
    for role in ('self','opponent'):
        names += [f'{role}_motion_{x}' for x in ('matched_coverage','mean_dx_mapspan_per_3s',
          'mean_dy_mapspan_per_3s','mean_speed_mapspan_per_3s','p90_speed_mapspan_per_3s',
          'moving_fraction_gt_0.01_mapspan','zone_change_fraction')]
        for typ in TYPE_ORDER:
            names += [f'{role}_{typ}_{x}' for x in ('matched_coverage','mean_dx_mapspan_per_3s',
              'mean_dy_mapspan_per_3s','mean_speed_mapspan_per_3s','moving_fraction','zone_change_fraction',
              'heading_sin','heading_cos','heading_coverage')]
        names += [f'{role}_transition_to_{z}' for z in ZONES]
    role_block_dim=7+len(TYPE_ORDER)*9+len(ZONES)
    assert len(names)==2*role_block_dim
    return names

def extract_motion(npz_path,jsonl_path):
    d=load_compact(npz_path); gids=np.asarray(d['game_ids'],dtype=np.int64)
    sides=np.asarray(d['sides']).astype(str); times=np.asarray(d['issued_at'],dtype=np.float64)
    if len(gids)!=len(times): raise ValueError('NPZ alignment arrays differ')
    # Keep feature-name blocks in precisely the same per-role order as each
    # row's team summary, archetype summaries, and destination-zone masses.
    names=motion_feature_names()
    X=np.zeros((len(gids),len(names)),np.float32); previous={}; rows=0
    delta_hist=[]; matched_total=0; stale_excluded=0; obs_future_violation=0
    with gzip.open(jsonl_path,'rt',encoding='utf-8') as f:
      for line in f:
        if not line.strip(): continue
        r=json.loads(line)
        if r.get('split')!='train': raise ValueError(f"non-train row at {rows}")
        if rows>=len(gids): raise ValueError('JSONL longer than NPZ')
        t=float(r['issued_at']); key_expected=(int(gids[rows]),str(sides[rows]),round(float(times[rows]),5))
        key_actual=(int(r['game_id']),str(r['side']),round(t,5))
        if key_actual!=key_expected: raise ValueError(f'NPZ/JSONL misalignment row {rows}: {key_actual}!={key_expected}')
        if finite(r.get('max_input_observed_at')) and float(r['max_input_observed_at'])>t+1e-6:
            obs_future_violation+=1
        obs=r.get('observable') or {}; row_features=[]
        for role,container in (('self','self_entities'),('opponent','opponent_entities')):
            entities=current_mobile(as_entities(obs.get(container)),t)
            group=(int(r['game_id']),str(r['side']),role); old=previous.setdefault(group,{})
            velocities=[]; type_vel={typ:[] for typ in TYPE_ORDER}; zone_dest=np.zeros(len(ZONES),np.float32)
            next_state={}
            for e in entities:
                eid=str(e['entity_id']); p=e['position']; pos=(float(p[0])/MAP_W,float(p[1])/MAP_H)
                z=zone_index(locate_zone(float(p[0]),float(p[1])))
                obs_t=float(e.get('observed_at',t)); typ=e['entity_type']
                prior=old.get(eid)
                if prior is not None:
                    dt=obs_t-prior['t']
                    # Only short observed intervals are meaningful; account for
                    # exact elapsed seconds and discard larger visibility gaps.
                    if 0.0 < dt <= 6.01:
                        scale=3.0/dt; dx=(pos[0]-prior['pos'][0])*scale; dy=(pos[1]-prior['pos'][1])*scale
                        speed=float(np.hypot(dx,dy)); changed=(z>=0 and prior['zone']>=0 and z!=prior['zone'])
                        item=(dx,dy,speed,float(changed),typ,z)
                        velocities.append(item); type_vel[typ].append(item); matched_total+=1; delta_hist.append(dt)
                        if changed: zone_dest[z]+=1.0
                    else: stale_excluded+=1
                next_state[eid]={'t':obs_t,'pos':pos,'zone':z,'type':typ}
            # Replace with current issue snapshot, so absent tracks never bridge
            # long gaps invisibly.
            previous[group]=next_state
            n=max(1,len(entities)); speeds=[v[2] for v in velocities]
            row_features += [len(velocities)/MOBILE_MAX,
               float(np.mean([v[0] for v in velocities])) if velocities else 0.,
               float(np.mean([v[1] for v in velocities])) if velocities else 0.,
               float(np.mean(speeds)) if speeds else 0.,
               float(np.quantile(speeds,.9)) if speeds else 0.,
               sum(s>.01 for s in speeds)/len(speeds) if speeds else 0.,
               float(np.mean([v[3] for v in velocities])) if velocities else 0.]
            for typ in TYPE_ORDER:
                tv=type_vel[typ]; ts=[v[2] for v in tv]
                curr_typed=[e for e in entities if e.get('entity_type')==typ]
                yaw=[float(e['yaw']) for e in curr_typed if finite(e.get('yaw'))]
                yaw_rad=np.deg2rad(yaw)
                row_features += [len(tv)/max(1,sum(e['entity_type']==typ for e in entities)),
                  float(np.mean([v[0] for v in tv])) if tv else 0.,
                  float(np.mean([v[1] for v in tv])) if tv else 0.,
                  float(np.mean(ts)) if ts else 0.,
                  sum(s>.01 for s in ts)/len(ts) if ts else 0.,
                  float(np.mean([v[3] for v in tv])) if tv else 0.,
                  float(np.mean(np.sin(yaw_rad))) if len(yaw) else 0.,
                  float(np.mean(np.cos(yaw_rad))) if len(yaw) else 0.,
                  len(yaw)/max(1,len(curr_typed))]
            row_features += (zone_dest/MOBILE_MAX).tolist()
        X[rows]=np.asarray(row_features,np.float32); rows+=1
        if rows%20000==0: print(json.dumps({'motion_rows':rows}),flush=True)
    if rows!=len(gids): raise ValueError(f'row count mismatch {rows} vs {len(gids)}')
    # Ensure the feature count matches deterministic names contract.
    if X.shape[1]!=len(names): raise AssertionError((X.shape,len(names)))
    return d,X,{'split':'train_only','rows_aligned':rows,'npz_jsonl_order':'exact',
      'max_input_observed_at_violations':obs_future_violation,'entity_feature_source':'observable.self_entities / observable.opponent_entities only',
      'targets_used_as_features':False,'target_keys_read':False,'validation_read':False,'test_read':False,
      'motion_definition':'same entity_id consecutive issue snapshots; (position_t-position_prev) normalized by map span and multiplied by 3s/actual elapsed seconds; elapsed must be <=6.01s; unobserved gaps do not bridge',
      'nominal_interval_seconds':3.0,'interval_observed_min_s':float(min(delta_hist)) if delta_hist else None,
      'interval_observed_median_s':float(np.median(delta_hist)) if delta_hist else None,
      'interval_observed_max_s':float(max(delta_hist)) if delta_hist else None,
      'matched_entity_deltas':matched_total,'excluded_long_or_invalid_deltas':stale_excluded,
      'heading_contract':'current observable yaw degrees encoded as sin/cos separately from position-difference velocity; enemy radar snapshots without yaw yield zero heading values and zero coverage',
      'feature_column_layout':'self team summary, self per-archetype summaries, self destination-zone transitions, then the same three blocks for opponent; name blocks follow this order',
      'feature_name_order_matches_row_builder':True,
      'added_feature_dim':int(X.shape[1]),'added_feature_names':names}

def run(train_npz,train_jsonl,base_oof,best_oof,outdir,folds=5,threads=4,seed=1927,reps=2000):
    start=time.perf_counter(); d,motion,audit=extract_motion(train_npz,train_jsonl)
    prior=np.load(base_oof,allow_pickle=False); best=np.load(best_oof,allow_pickle=False)
    for arr in (prior,best):
      for k in ('game_ids','sides','issued_at','mode_y','base_mode'):
        if not np.array_equal(np.asarray(d[k]),np.asarray(arr[k])): raise ValueError(f'{k} mismatch with OOF reference')
    # zone+observable-score feature matrix is recomputed from train JSONL by the
    # established extractor. The feature rows are aligned to this same train NPZ.
    from eval.experiment_death_aware_zone_mass_oof import extract_train_aggregates
    _, zscore, z_audit=extract_train_aggregates(train_npz,train_jsonl)
    X=np.concatenate([direct_features(d),zscore,motion],axis=1)
    gids=np.asarray(d['game_ids'],np.int64); sides=np.asarray(d['sides']).astype(str)
    y=np.asarray(d['mode_y'],np.int64); base_mode=np.asarray(d['base_mode'],np.int64)
    direct_key='direct_lightgbm' if 'direct_lightgbm' in prior.files else 'lightgbm'
    basep=np.asarray(prior[direct_key],np.float64); bestp=np.asarray(best['aggregate_candidate'],np.float64)
    cand=np.full_like(basep,np.nan); foldids=np.full(len(gids),-1,np.int8); folds_report=[]
    cv=GroupKFold(n_splits=folds)
    for fold,(tr,held) in enumerate(cv.split(X,groups=gids)):
      foldids[held]=fold; hor={}
      for j,h in enumerate(HORIZONS):
        trm=(y[tr,j]>=0)&(base_mode[tr]>=0)&(base_mode[tr]<3)
        hem=(y[held,j]>=0)&(base_mode[held]>=0)&(base_mode[held]<3)
        trr,hrr=tr[trm],held[hem]
        model=LGBMClassifier(objective='multiclass',num_class=3,n_estimators=260,learning_rate=.045,
          num_leaves=15,max_depth=-1,min_child_samples=90,reg_lambda=4.,colsample_bytree=.8,
          subsample=.85,subsample_freq=1,class_weight=None,verbosity=-1,n_jobs=threads,random_state=seed+fold)
        model.fit(X[trr],y[trr,j]); ph=np.zeros((len(hrr),3),np.float32)
        for col,cls in enumerate(model.classes_): ph[:,int(cls)]=model.predict_proba(X[hrr])[:,col]
        cand[hrr,j]=ph; hor[f'{h}s']={'train_rows':len(trr),'heldout_rows':len(hrr)}
      folds_report.append({'fold':fold,'train_games':len(np.unique(gids[tr])),'heldout_games':len(np.unique(gids[held])),'horizons':hor})
      print(json.dumps({'fold':fold,'horizons':hor}),flush=True)
    results={}
    for j,h in enumerate(HORIZONS):
      valid=(y[:,j]>=0)&np.isfinite(cand[:,j,0]); gs={}
      for name,p in (('motion_candidate',cand[:,j]),('zone_score_oof',bestp[:,j]),('direct_lgbm_oof',basep[:,j])):
        gs[name]=game_scores(y[:,j],p,gids,sides,valid)
      results[f'{h}s']={'valid_rows':int(valid.sum()),'pooled':{
        name:metrics(y[valid,j],p[valid]) for name,p in (('motion_candidate',cand[:,j]),('zone_score_oof',bestp[:,j]),('direct_lgbm_oof',basep[:,j]))},
        'game_equal_red_blue_equal':{name:{k:float(np.mean([v[k] for v in vals.values()])) for k in next(iter(vals.values()))}
          for name,vals in gs.items()},
        'paired_game_bootstrap_vs_best_zone_score':named_bootstrap_delta(
          gs['motion_candidate'],gs['zone_score_oof'],candidate_name='motion_candidate',
          reference_name='zone_score_oof',seed=seed+j,reps=reps),
        'paired_game_bootstrap_vs_direct_lightgbm':named_bootstrap_delta(
          gs['motion_candidate'],gs['direct_lgbm_oof'],candidate_name='motion_candidate',
          reference_name='direct_lightgbm_oof',seed=seed+20+j,reps=reps)}
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    report={'schema':'death-aware-motion-zone-score-train-group-oof-v1','access':{'opened_data_splits':['train'],'validation':False,'test':False},
      'label_contract':str(d['global_mode_label_version'][0]),'grouping':'5-fold GroupKFold by full game_id; red/blue views stay in each game fold',
      'estimator':'same frozen LightGBM config as zone+observable-score OOF; no tuning/search',
      'features':'existing direct temporal inputs + established current-observable zone/score candidate features + causal current-vs-immediately-previous observable motion features',
      'feature_audit':audit,'zone_score_feature_audit':z_audit,'train_rows':len(gids),'train_games':len(np.unique(gids)),
      'folds':folds_report,'horizons':results,'runtime_seconds':time.perf_counter()-start}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'motion_candidate_oof.npz',game_ids=gids,sides=sides,issued_at=d['issued_at'],mode_y=y,
      base_mode=base_mode,fold_ids=foldids,motion_candidate=cand.astype(np.float32),best_zone_score_oof=bestp.astype(np.float32),
      direct_lightgbm_oof=basep.astype(np.float32))
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train-npz',required=True); p.add_argument('--train-jsonl',required=True)
    p.add_argument('--direct-oof',required=True); p.add_argument('--best-oof',required=True); p.add_argument('--output',required=True)
    p.add_argument('--folds',type=int,default=5); p.add_argument('--threads',type=int,default=4); p.add_argument('--seed',type=int,default=1927); p.add_argument('--bootstrap-reps',type=int,default=2000)
    a=p.parse_args(); r=run(a.train_npz,a.train_jsonl,a.direct_oof,a.best_oof,a.output,a.folds,a.threads,a.seed,a.bootstrap_reps)
    print(json.dumps({h:v['game_equal_red_blue_equal'] for h,v in r['horizons'].items()},ensure_ascii=False))
if __name__=='__main__': main()
