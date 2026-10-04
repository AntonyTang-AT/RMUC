"""Persistence audit for entity zones/positions at archived validation horizons.

Reads only the canonical validation archive. Current state comes from the
causal final feature frame; exact future targets come from existing validation
labels (10/30/60/120s), produced at issued_at+horizon without interpolation.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__('sys').path: __import__('sys').path.insert(0,str(ROOT))
from forecast.entity import ENTITY_SLOTS,ENTITY_ZONE_LABELS
from worldstate.functional_zone import assign_functional_zone
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS,locate_zone
HORIZONS=(10,30,60,120)
FZ={v:i for i,v in enumerate(ENTITY_ZONE_LABELS)}; CZ={v:i for i,v in enumerate(CANONICAL_ZONE_LABELS)}

def met(c):
 n=int(c.sum())
 if not n:return None
 fs=[]
 for i in range(len(c)):
  tp=float(c[i,i]); den=2*tp+float(c[:,i].sum()-tp)+float(c[i,:].sum()-tp)
  fs.append(2*tp/den if den else 0.)
 return {'accuracy':float(np.trace(c)/n),'macro_f1':float(np.mean(fs)),
         'transition_rate':float((n-np.trace(c))/n),'n':n}
def audit(npz=ROOT/'out_m2/temporal_forecast_canonical_side/validation.npz',
          out=ROOT/'out_m2/temporal_forecast_canonical_side/entity_horizon_validation.json'):
 p=Path(npz)
 with np.load(p,allow_pickle=False) as z:
  gids=z['game_ids'].astype(np.int64); sides=z['sides'].astype(str); x=np.array(z['x'][:,-1,:],np.float32,copy=True)
  yz=z['entity_zone_y']; yc=z['entity_canonical_zone_y']; ye=z['entity_y']
 games=sorted(set(map(int,gids))); gi={g:i for i,g in enumerate(games)}
 out_h={}
 for hi,h in enumerate(HORIZONS):
  cf=np.zeros((len(games),2,len(FZ),len(FZ)),np.uint32)
  cc=np.zeros((len(games),2,len(CZ),len(CZ)),np.uint32)
  ps=np.zeros((len(games),2,4),np.float64) # norm L1, metres, within1m, count
  for row,(gid,side) in enumerate(zip(gids,sides)):
   g=gi[int(gid)]; si=0 if side=='红' else 1; f=x[row]
   for j,(role,typ) in enumerate(ENTITY_SLOTS):
    off=j*14
    if f[off+9]<=.5 or f[off+3]<=.5: continue
    px,py=float(f[off]*28),float(f[off+1]*15)
    team=side if role=='self' else ('蓝' if side=='红' else '红')
    if typ=='基地': name='红基地' if team=='红' else '蓝基地'
    elif typ=='前哨站': name='红前哨区' if team=='红' else '蓝前哨区'
    else:name=assign_functional_zone(px,py)
    pz=FZ.get(name); pcz=CZ.get(locate_zone(px,py))
    tz=int(yz[row,hi,j]); tcz=int(yc[row,hi,j])
    if pz is not None and tz>=0: cf[g,si,pz,tz]+=1
    if pcz is not None and tcz>=0: cc[g,si,pcz,tcz]+=1
    if ye[row,hi,j,0]>.5:
     tx,ty=float(ye[row,hi,j,1]*28),float(ye[row,hi,j,2]*15)
     d=float(np.hypot(px-tx,py-ty)); ps[g,si,0]+=(abs(px-tx)/28+abs(py-ty)/15)/2
     ps[g,si,1]+=d;ps[g,si,2]+=float(d<=1);ps[g,si,3]+=1
  m={}
  for name,c in [('functional_zone',cf),('canonical_zone',cc)]:
   per={}
   for g in range(len(games)):
    red=met(c[g,0]);blue=met(c[g,1])
    if red and blue: per[games[g]]={k:(red[k]+blue[k])/2 for k in ('accuracy','macro_f1','transition_rate')}
   sides_out={}
   for si,s in enumerate(('红','蓝')):sides_out[s]=met(c[:,si].sum(0))
   pooled=c.sum(axis=(0,1)); n=int(pooled.sum());den=len(gids)*len(ENTITY_SLOTS)
   m[name]={'accuracy_game_equal_red_blue_equal':float(np.mean([v['accuracy'] for v in per.values()])) if per else None,
    'macro_f1_game_equal_red_blue_equal':float(np.mean([v['macro_f1'] for v in per.values()])) if per else None,
    'transition_rate_game_equal_red_blue_equal':float(np.mean([v['transition_rate'] for v in per.values()])) if per else None,
    'scored_count':n,'possible_count':den,'coverage':n/den if den else None,
    'by_side_pooled':sides_out,'complete_red_blue_games':len(per)}
  pp=[]
  for g in range(len(games)):
   sm=[]
   for si in (0,1):
    q=ps[g,si]
    if q[3]>0:sm.append(q[:3]/q[3])
   if len(sm)==2:pp.append(np.mean(sm,axis=0))
  sumpos=ps.sum(axis=(0,1));den=len(gids)*len(ENTITY_SLOTS)
  m['position']={'normalized_l1_mae_game_equal_red_blue_equal':float(np.mean([q[0] for q in pp])) if pp else None,
   'mean_error_m_game_equal_red_blue_equal':float(np.mean([q[1] for q in pp])) if pp else None,
   'within_1m_game_equal_red_blue_equal':float(np.mean([q[2] for q in pp])) if pp else None,
   'scored_count':int(sumpos[3]),'possible_count':den,'coverage':float(sumpos[3]/den) if den else None,
   'pooled_normalized_l1_mae':float(sumpos[0]/sumpos[3]) if sumpos[3] else None,
   'pooled_mean_error_m':float(sumpos[1]/sumpos[3]) if sumpos[3] else None,
   'pooled_within_1m':float(sumpos[2]/sumpos[3]) if sumpos[3] else None,
   'position_metric_note':'No class Macro-F1 for continuous XY; within_1m is Euclidean error <=1m.'}
  out_h[str(h)]=m
 sha=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):sha.update(chunk)
 report={'schema':'validation-entity-persistence-audit-v1','data_contract':{
  'archive':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha.hexdigest(),
  'scope':'validation only; no test archive accessed','games':len(games),'issue_rows':len(gids),
  'horizons_seconds':list(HORIZONS),'future_labels':'existing validation.npz targets from exact issued_at+horizon raw timestamp; no interpolation',
  'current_state':'last causal feature frame; only present slots with visible/current XY are scored',
  'position_accuracy':'Euclidean XY error <=1 metre; normalized L1 and metre error additionally reported',
  'aggregation':'per game-side; equal red/blue within complete game, then equal weight games',
  'coverage_denominator':'validation issue rows × 16 entity slots; current/future XY or zone must be available'},
  'horizons':out_h}
 o=Path(out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report
def main():
 import argparse
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--validation',default=str(ROOT/'out_m2/temporal_forecast_canonical_side/validation.npz'))
 a.add_argument('--output',default=str(ROOT/'out_m2/temporal_forecast_canonical_side/entity_horizon_validation.json'));z=a.parse_args()
 r=audit(z.validation,z.output)
 print(json.dumps({'horizons':{h:{k:{m:v for m,v in d.items() if m.endswith('_equal_red_blue_equal') or m in ('coverage','scored_count','complete_red_blue_games')} for k,d in q.items()} for h,q in r['horizons'].items()},'output':z.output},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
