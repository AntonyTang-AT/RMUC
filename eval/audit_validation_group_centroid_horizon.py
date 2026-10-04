"""15-second sampled exact-horizon centroid/zone drift audit (validation only)."""
from __future__ import annotations
import argparse,json,hashlib,sqlite3,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from config import DB_PATH
from forecast.entity import ENTITY_TYPES
from worldstate.functional_zone import assign_functional_zone
HORIZONS=(5,10,15,20,30,45,60,90,120);MOBILE=set(ENTITY_TYPES[:6])
def _ent(r):return {'team':r['阵营'],'type':r['机器人类型'],'x':r['x'],'y':r['y'],'hp':r['当前血量']}
def _alive(e):return not(isinstance(e.get('hp'),(int,float)) and e['hp']<=0)
def _centroid(points):return tuple(np.mean(np.asarray(points,dtype=float),axis=0)) if points else None
def _stats(game_side):
    # game_side[gid][side] is list of (distance, zone_match, scored)
    gs={}
    for gid,sides in game_side.items():
        if {'红','蓝'}<=sides.keys():
            vals=[]
            for side in ('红','蓝'):
                rows=sides[side]
                if rows: vals.append((float(np.mean([r[0] for r in rows])),float(np.mean([r[1] for r in rows])),len(rows)))
            if len(vals)==2:gs[gid]=(np.mean([v[0] for v in vals]),np.mean([v[1] for v in vals]),sum(v[2] for v in vals))
    pooled=[r[0] for ss in game_side.values() for rr in ss.values() for r in rr]
    zone=[r[1] for ss in game_side.values() for rr in ss.values() for r in rr]
    n=len(pooled);return {'centroid_displacement_mean_m_game_equal_red_blue_equal':float(np.mean([v[0] for v in gs.values()])) if gs else None,
       'centroid_displacement_median_m_pooled':float(np.median(pooled)) if n else None,
       'centroid_displacement_p90_m_pooled':float(np.quantile(pooled,.9)) if n else None,
       'persistence_current_zone_hit_rate_game_equal_red_blue_equal':float(np.mean([v[1] for v in gs.values()])) if gs else None,
       'zone_transition_rate_game_equal_red_blue_equal':1-float(np.mean([v[1] for v in gs.values()])) if gs else None,
       'scored_samples':n,'complete_red_blue_games':len(gs),'coverage':None,
       'pooled_zone_hit_rate':float(np.mean(zone)) if zone else None}
def audit(db=DB_PATH,validation=ROOT/'out_m2/temporal_forecast_canonical_side/validation.npz',
          output=ROOT/'out_m2/temporal_forecast_canonical_side/group_centroid_horizon_validation.json',sample_every_s=15):
    p=Path(validation)
    with np.load(p,allow_pickle=False) as z:
        gids=z['game_ids'].astype(np.int64);sides=z['sides'].astype(str);times=z['issued_at'].astype(np.float64)
        # 3-second input cadence; deterministic per-game clock sampling (1,16,31,...).
        keep=np.isclose((times-1.0)%sample_every_s,0.0,atol=1e-5)
        selected=[(i,int(gids[i]),str(sides[i]),int(round(times[i]))) for i in np.flatnonzero(keep)]
        current=np.array(z['x'][keep,-1,:],dtype=np.float32,copy=True)
    by_game=defaultdict(list)
    for j,item in enumerate(selected):by_game[item[1]].append((j,*item))
    # Store one row per sampled issue, horizon, and own/enemy grouping.
    records={(h,r):defaultdict(lambda:{'红':[],'蓝':[]}) for h in HORIZONS for r in ('self','opponent')}
    con=sqlite3.connect(f'file:{Path(db).resolve().as_posix()}?mode=ro',uri=True);con.row_factory=sqlite3.Row
    fields='game_id, "时刻秒", 阵营, 机器人类型, x, y, 当前血量'
    try:
      for gid,rows in by_game.items():
        raw=defaultdict(list)
        for r in con.execute(f'SELECT {fields} FROM timeseries WHERE game_id=? ORDER BY "时刻秒"',(gid,)):
          raw[int(float(r['时刻秒']))].append(_ent(r))
        for local_idx,orig_i,gg,side,t in rows:
          feat=current[local_idx];now={}
          for role,lo,hi in (('self',0,8),('opponent',8,16)):
            pts=[]
            for slot in range(lo,hi):
              typ=ENTITY_TYPES[slot-lo if role=='self' else slot-8]
              off=slot*14
              if typ not in MOBILE or feat[off+9]<=.5 or feat[off+3]<=.5:continue
              pts.append((float(feat[off]*28),float(feat[off+1]*15)))
            now[role]=_centroid(pts)
          for h in HORIZONS:
            target=raw.get(t+h)
            if not target:continue
            for role in ('self','opponent'):
              team=side if role=='self' else ('蓝' if side=='红' else '红')
              pts=[(float(e['x']),float(e['y'])) for e in target if e['team']==team and e['type'] in MOBILE and _alive(e)
                   and e['x'] is not None and e['y'] is not None]
              future=_centroid(pts);present=now[role]
              if future is None or present is None:continue
              dist=float(np.hypot(future[0]-present[0],future[1]-present[1]))
              z0=assign_functional_zone(*present);z1=assign_functional_zone(*future)
              records[(h,role)][gg][side].append((dist,float(z0==z1)))
    finally:con.close()
    output_h={}
    for h in HORIZONS:
      by_role={}
      for role in ('self','opponent'):
        gs=records[(h,role)];possible=len(selected)
        # Reuse aggregate method and calculate coverage against sampled rows.
        met=_stats(gs);met['coverage']=met['scored_samples']/possible if possible else None
        met['possible_sampled_issue_rows']=possible;by_role[role]=met
      output_h[str(h)]=by_role
    sha=hashlib.sha256()
    with p.open('rb') as f:
      for chunk in iter(lambda:f.read(1<<20),b''):sha.update(chunk)
    report={'schema':'validation-group-centroid-exact-horizon-v1','data_contract':{
      'archive':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha.hexdigest(),
      'scope':'validation matches only; no test split archive opened','games':len(by_game),
      'sample_every_seconds':sample_every_s,'sampling':'deterministic issue times 1, 1+15, 1+30, ... s per game and side',
      'validation_issue_rows_total':len(gids),'sampled_issue_rows':len(selected),
      'target':'exact integer issued_at+horizon raw timestamp; no interpolation or nearest-time fill',
      'current_centroid':'mean of causally observable per-type mobile-unit slots in final validation feature frame; enemy centroid may be sensor/radar partial',
      'future_centroid':'mean of all living raw mobile units for that side at exact future second',
      'interpretation':'descriptive movement/zone drift of groups, not model forecast accuracy; paired/red-blue aggregates weight sides equally within each complete game'},
      'horizons':output_h}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report
def main():
 import argparse
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--db',default=DB_PATH);a.add_argument('--validation',default=str(ROOT/'out_m2/temporal_forecast_canonical_side/validation.npz'));a.add_argument('--output',default=str(ROOT/'out_m2/temporal_forecast_canonical_side/group_centroid_horizon_validation.json'));a.add_argument('--sample-every-s',type=int,default=15);x=a.parse_args()
 r=audit(x.db,x.validation,x.output,x.sample_every_s)
 print(json.dumps({'sampled_rows':r['data_contract']['sampled_issue_rows'],'horizons':{h:{role:{k:v for k,v in stats.items() if k.endswith('_game_equal_red_blue_equal') or k in ('coverage','scored_samples')} for role,stats in roles.items()} for h,roles in r['horizons'].items()},'output':x.output},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
