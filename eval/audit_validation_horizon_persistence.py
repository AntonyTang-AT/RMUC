"""Exact-second persistence audit on validation matches only.

Only validation.npz is opened, to whitelist game/side/issue rows and read the
current causal features. Raw rows are streamed one validation game at a time.
Future labels use exact ``issued_at + horizon`` seconds; no interpolation,
nearest timestamp, forward fill, model training, or test access is performed.
"""
from __future__ import annotations
import argparse, hashlib, json, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config import DB_PATH
from forecast.temporal import MODES, _mode_from_groups
from forecast.entity import ENTITY_SLOTS, ENTITY_ZONE_LABELS
from worldstate.functional_zone import assign_functional_zone
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS, locate_zone

HORIZONS=(5,10,15,20,30,45,60,90,120)
TYPES=('英雄','工程','步兵3','步兵4','空中','哨兵','前哨站','基地')
MOBILE=set(TYPES[:6]); MODE_ID={v:i for i,v in enumerate(MODES)}
FZ_ID={v:i for i,v in enumerate(ENTITY_ZONE_LABELS)}
CZ_ID={v:i for i,v in enumerate(CANONICAL_ZONE_LABELS)}
TASKS={'global_mode':len(MODES),'functional_zone':len(ENTITY_ZONE_LABELS),
       'canonical_zone':len(CANONICAL_ZONE_LABELS)}

def entity(r):
    return {'entity_id':f"{r['阵营']}:{r['robot_id']}",'team':r['阵营'],
            'entity_type':r['机器人类型'],'position':[r['x'],r['y']],
            'hp':r['当前血量'],'maxhp':r['最大血量']}

def alive(e): return not (isinstance(e.get('hp'),(int,float)) and e['hp']<=0)

def zone_f(e):
    if e['entity_type']=='基地': name='红基地' if e['team']=='红' else '蓝基地'
    elif e['entity_type']=='前哨站': name='红前哨区' if e['team']=='红' else '蓝前哨区'
    else:
        p=e.get('position')
        if p is None or p[0] is None or p[1] is None: return None
        name=assign_functional_zone(float(p[0]),float(p[1]))
    return FZ_ID.get(name)

def boot_ci(deltas,seed=20260927,reps=2000):
    vals=np.asarray(list(deltas.values()),dtype=float)
    if not len(vals): return {'n_games':0,'mean_delta':None,'ci95':[None,None]}
    rng=np.random.default_rng(seed)
    m=vals[rng.integers(0,len(vals),size=(reps,len(vals)))].mean(axis=1)
    return {'n_games':int(len(vals)),'mean_delta':float(vals.mean()),
            'ci95':[float(np.quantile(m,.025)),float(np.quantile(m,.975))]}

def matrix_metrics(c):
    n=int(c.sum())
    if not n:return None
    acc=float(np.trace(c)/n); f1=[]
    for i in range(c.shape[0]):
        tp=float(c[i,i]); fp=float(c[:,i].sum()-tp); fn=float(c[i,:].sum()-tp)
        den=2*tp+fp+fn; f1.append(2*tp/den if den else 0.0)
    return {'accuracy':acc,'macro_f1':float(np.mean(f1)),
            'transition_rate':float((n-np.trace(c))/n),'n':n}

def audit(db_path=DB_PATH, validation_npz=ROOT/'out_m2/temporal_forecast_canonical_side/validation.npz',
          output=ROOT/'out_m2/temporal_forecast_canonical_side/horizon_persistence_audit.json'):
    path=Path(validation_npz)
    with np.load(path,allow_pickle=False) as v:
        gids=v['game_ids'].astype(np.int64); sides=v['sides'].astype(str)
        issued=v['issued_at'].astype(np.float64); base=v['base_mode'].astype(np.int64)
        # Current state is the last causal feature frame. Drop the 12-frame
        # context immediately so the archive's largest tensor can be released.
        xx=v['x']; cur=np.array(xx[:,-1,:],dtype=np.float32,copy=True); del xx
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): sha.update(chunk)
    games=sorted(set(map(int,gids))); game_index={g:i for i,g in enumerate(games)}
    side_index={'红':0,'蓝':1}
    issue_rows=defaultdict(list)
    for i,(g,s,t) in enumerate(zip(gids,sides,issued)):
        issue_rows[int(g)].append((i,str(s),int(round(float(t)))))
    # Per-game side confusion matrices/counts are compact and enable proper
    # red-blue equal weighting plus game-cluster bootstrap.
    conf={(h,k):np.zeros((len(games),2,n,n),dtype=np.uint32)
          for h in HORIZONS for k,n in TASKS.items()}
    pos=np.zeros((len(HORIZONS),len(games),2,4),dtype=np.float64) # L1,m,within1,count
    exact=np.zeros(len(HORIZONS),dtype=np.int64); current_found=0

    con=sqlite3.connect(f'file:{Path(db_path).resolve().as_posix()}?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    fields='game_id, "时刻秒", 阵营, robot_id, 机器人类型, x, y, 当前血量, 最大血量'
    try:
        for gi,gid in enumerate(games):
            states=defaultdict(list)
            # Stream one validation game's compact truth state, not a Python
            # object graph for the complete split.
            for r in con.execute(f'SELECT {fields} FROM timeseries WHERE game_id=? ORDER BY "时刻秒"',(gid,)):
                states[int(float(r['时刻秒']))].append(entity(r))
            for row_i,side,t in issue_rows[gid]:
                if t not in states: continue
                current_found+=1; si=side_index[side]; fi=cur[row_i]
                xy=[]; fz=[]; cz=[]
                for j,(role,typ) in enumerate(ENTITY_SLOTS):
                    off=j*14
                    if fi[off+9]<=.5 or fi[off+3]<=.5:
                        xy.append(None); fz.append(None); cz.append(None); continue
                    px,py=float(fi[off]*28),float(fi[off+1]*15); xy.append((px,py))
                    team=side if role=='self' else ('蓝' if side=='红' else '红')
                    if typ=='基地': name='红基地' if team=='红' else '蓝基地'
                    elif typ=='前哨站': name='红前哨区' if team=='红' else '蓝前哨区'
                    else: name=assign_functional_zone(px,py)
                    fz.append(FZ_ID.get(name)); cz.append(CZ_ID.get(locate_zone(px,py)))
                current_mode=int(base[row_i])
                for hi,h in enumerate(HORIZONS):
                    target=states.get(t+h)
                    if not target: continue
                    exact[hi]+=1
                    own=[e for e in target if e['team']==side and
                         (e['entity_type'] not in MOBILE or alive(e))]
                    opp=[e for e in target if e['team']!=side and
                         (e['entity_type'] not in MOBILE or alive(e))]
                    true_mode=MODE_ID[_mode_from_groups(own,opp,side)]
                    conf[(h,'global_mode')][gi,si,current_mode,true_mode]+=1
                    for j,(role,typ) in enumerate(ENTITY_SLOTS):
                        team=side if role=='self' else ('蓝' if side=='红' else '红')
                        match=[e for e in target if e['team']==team and e['entity_type']==typ and alive(e)]
                        if not match: continue
                        fzt=zone_f(match[0])
                        if fzt is not None and fz[j] is not None:
                            conf[(h,'functional_zone')][gi,si,int(fz[j]),int(fzt)]+=1
                        p=match[0].get('position')
                        if p and p[0] is not None and p[1] is not None:
                            czt=CZ_ID.get(locate_zone(float(p[0]),float(p[1])))
                            if czt is not None and cz[j] is not None:
                                conf[(h,'canonical_zone')][gi,si,int(cz[j]),int(czt)]+=1
                        pts=[e['position'] for e in match if e.get('position') is not None
                             and e['position'][0] is not None and e['position'][1] is not None]
                        if pts and xy[j] is not None:
                            tx=float(np.mean([p[0] for p in pts])); ty=float(np.mean([p[1] for p in pts]))
                            dx,dy=xy[j][0]-tx,xy[j][1]-ty; slot=pos[hi,gi,si]
                            slot[0]+=(abs(dx)/28+abs(dy)/15)/2
                            slot[1]+=float(np.hypot(dx,dy)); slot[2]+=float(np.hypot(dx,dy)<=1.0); slot[3]+=1
    finally: con.close()
    del cur

    out={}; game_acc={}
    for hi,h in enumerate(HORIZONS):
        hm={}
        for task,nc in TASKS.items():
            a=conf[(h,task)]; gs={}; side_conf=[]
            for gi,gid in enumerate(games):
                sm={}
                for si,s in enumerate(('红','蓝')):
                    m=matrix_metrics(a[gi,si])
                    if m: sm[s]=m
                if {'红','蓝'}<=sm.keys(): gs[gid]=sm
            metrics=[]
            for key in ('accuracy','macro_f1','transition_rate'):
                metrics=[(gs[g]['红'][key]+gs[g]['蓝'][key])/2 for g in gs]
                hm.setdefault(task,{})[key+'_game_equal_red_blue_equal']=float(np.mean(metrics)) if metrics else None
            paired={g:(gs[g]['红']['accuracy']+gs[g]['蓝']['accuracy'])/2 for g in gs}
            game_acc[(task,h)]=paired
            pooled_by_side={}
            for si,s in enumerate(('红','蓝')):
                c=a[:,si].sum(axis=0); pooled_by_side[s]=matrix_metrics(c)
            scored=int(a.sum()); denom=len(gids) if task=='global_mode' else len(gids)*len(ENTITY_SLOTS)
            hm[task].update({'scored_count':scored,'possible_count':int(denom),
                             'coverage':float(scored/denom) if denom else None,
                             'by_side_pooled':pooled_by_side,
                             'paired_game_bootstrap_accuracy_ci95':boot_ci(paired),
                             'complete_red_blue_games':len(gs)})
        # Position continuous metrics; first collapse slots inside game-side,
        # equal-weight sides per complete game, then equal-weight games.
        arr=pos[hi]; gm=[]
        for gi,gid in enumerate(games):
            side_metrics=[]
            for si in (0,1):
                q=arr[gi,si]
                if q[3]>0: side_metrics.append(q[:3]/q[3])
            if len(side_metrics)==2: gm.append(np.mean(side_metrics,axis=0))
        pm={}
        mean=np.mean(gm,axis=0) if gm else [None,None,None]
        for nm,val in zip(('normalized_l1_mae','mean_error_m','within_1m'),mean):
            pm[nm+'_game_equal_red_blue_equal']=float(val) if val is not None else None
        pooled=arr.sum(axis=(0,1)); denom=len(gids)*len(ENTITY_SLOTS)
        pm.update({'scored_count':int(pooled[3]),'possible_count':int(denom),
                   'coverage':float(pooled[3]/denom) if denom else None,
                   'pooled_normalized_l1_mae':float(pooled[0]/pooled[3]) if pooled[3] else None,
                   'pooled_mean_error_m':float(pooled[1]/pooled[3]) if pooled[3] else None,
                   'pooled_within_1m':float(pooled[2]/pooled[3]) if pooled[3] else None,
                   'position_metric_note':'continuous position has no class Macro-F1; within_1m means Euclidean error <=1 metre'})
        hm['position']=pm; out[str(h)]=hm
    deltas={}
    for task in TASKS:
        d={}
        for a,b in zip(HORIZONS[:-1],HORIZONS[1:]):
            aa,bb=game_acc[(task,a)],game_acc[(task,b)]; common=set(aa)&set(bb)
            d[f'{a}_to_{b}s']=boot_ci({g:bb[g]-aa[g] for g in common})
        deltas[task]=d
    report={'schema':'validation-exact-second-persistence-audit-v3',
      'data_contract':{'validation_archive':str(path.relative_to(ROOT)).replace('\\','/'),
        'validation_archive_sha256':sha.hexdigest(),'validation_game_count':len(games),
        'validation_issue_count':len(gids),'rows_with_raw_current_snapshot':current_found,
        'exact_future_snapshots_by_horizon':{str(h):int(exact[k]) for k,h in enumerate(HORIZONS)},
        'split_scope':'validation only; script opens no train or test split archive',
        'raw_target_rule':'exact integer timestamp issued_at + horizon; no interpolation, nearest lookup, or forward fill',
        'raw_resolution_seconds':1,'issue_cadence_seconds':3,
        'global_mode_label':'relative_margin_v2 via forecast.temporal._mode_from_groups; current persistence class from validation base_mode',
        'entity_labels':'existing per-side/per-type slot contract; future position is mean live XY, zone is first live typed row; base/outpost special zones retained',
        'position_accuracy':'Euclidean error <= 1 metre; also report normalized L1 MAE and mean metre error',
        'aggregation':'per-game-side scores; equal red/blue average within complete games; equal weight per game; bootstrap resamples games'},
      'horizons':out,'adjacent_horizon_paired_accuracy_deltas':deltas,
      'interpretation':{'scope':'Persistence-label dynamics only, not learned model accuracy.',
        'macro_f1':'fixed complete vocabulary and zero-divisions set to zero.',
        'coverage':'Exact future samples beyond game end are unavailable and not imputed.',
        'position':'Functional/canonical zone transition rate is the discrete measure of spatial movement; position metrics are continuous.'}}
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--db',default=DB_PATH)
    p.add_argument('--validation-npz',default=str(ROOT/'out_m2/temporal_forecast_canonical_side/validation.npz'))
    p.add_argument('--output',default=str(ROOT/'out_m2/temporal_forecast_canonical_side/horizon_persistence_audit.json'))
    a=p.parse_args(); r=audit(a.db,a.validation_npz,a.output)
    compact={h:{k:{m:v for m,v in d.items() if m.endswith('_equal_red_blue_equal') or m in ('coverage','complete_red_blue_games')}
                for k,d in vals.items()} for h,vals in r['horizons'].items()}
    print(json.dumps({'games':r['data_contract']['validation_game_count'],'issues':r['data_contract']['validation_issue_count'],
      'raw_current':r['data_contract']['rows_with_raw_current_snapshot'],'horizons':compact,'output':a.output},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
