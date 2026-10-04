"""Incremental five-fold game-group transition-prior experiment (train/val only)."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from forecast.temporal import HORIZONS,MODES,load_compact
from eval.experiment_death_aware_mode_lightgbm import scores,persistence_probs,per_game_scores,paired_bootstrap

def matrix(y,base,alpha):
 c=np.full((3,3),float(alpha),dtype=float)
 for b,t in zip(base,y):
  if 0<=b<3 and 0<=t<3:c[int(b),int(t)]+=1
 return c/c.sum(1,keepdims=True)

def mix_pick(y,p,prior):
 best=None
 for lam in np.linspace(0,1,101):
  q=lam*p+(1-lam)*prior;m=scores(y,q)
  key=(.5*m['accuracy']+.5*m['macro_f1'],m['macro_f1'],m['accuracy'],-lam)
  if best is None or key>best[0]:best=(key,float(lam),m)
 return best[1]

def report_h(train,val,j,h,folds,seed):
 y=train['mode_y'][:,j].astype(int);valid=y>=0;rows=np.flatnonzero(valid);yy=y[valid]
 base=train['base_mode'].astype(int);groups=train['game_ids'];oof=np.zeros((len(y),3),float)
 # Select transition smoothing only from group-held-out training OOF predictions.
 alpha_scores={};alpha_oof={}
 cv=GroupKFold(n_splits=folds)
 for alpha in (.5,1.,2.,5.,10.):
  p=np.zeros((len(y),3),float)
  for fold,(trix,teix) in enumerate(cv.split(rows,yy,groups=groups[valid])):
   tr,te=rows[trix],rows[teix];mat=matrix(y[tr],base[tr],alpha)
   p[te]=mat[np.clip(base[te],0,2)]
  alpha_oof[alpha]=p[valid];alpha_scores[str(alpha)]=scores(yy,p[valid])
 # Choose alpha and model-vs-persistence weight solely from train OOF.
 alpha=max((float(a) for a in alpha_scores),key=lambda a:(.5*alpha_scores[str(a)]['accuracy']+.5*alpha_scores[str(a)]['macro_f1'],alpha_scores[str(a)]['macro_f1']))
 trans=alpha_oof[alpha];persist=persistence_probs(base)[valid]
 lam=mix_pick(yy,trans,persist);trainmix=lam*trans+(1-lam)*persist
 # Validation only once after all choices frozen from OOF.
 vy=val['mode_y'][:,j].astype(int);vv=vy>=0;mat=matrix(y[valid],base[valid],alpha)
 vtrans=mat[np.clip(val['base_mode'].astype(int)[vv],0,2)];vpersist=persistence_probs(val['base_mode'])[vv]
 vmix=lam*vtrans+(1-lam)*vpersist
 gids=val['game_ids'][vv];sides=val['sides'][vv];allmask=np.ones(int(vv.sum()),bool)
 baseg=per_game_scores(vy[vv],vpersist,gids,sides,allmask)
 candg=per_game_scores(vy[vv],vmix,gids,sides,allmask)
 return {'train_oof_alpha_scores':alpha_scores,'selected_dirichlet_alpha_train_oof':alpha,
  'selected_transition_weight_train_oof':lam,'train_oof_transition_raw':scores(yy,trans),
  'train_oof_persistence':scores(yy,persist),'train_oof_blend':scores(yy,trainmix),
  'train_transition_counts':np.array([[int(np.sum(valid&(base==b)&(y==t))) for t in range(3)] for b in range(3)]).tolist(),
  'validation_transition_counts':np.array([[int(np.sum(vv&(val['base_mode']==b)&(vy==t))) for t in range(3)] for b in range(3)]).tolist(),
  'validation_target_distribution':np.bincount(vy[vv],minlength=3).tolist(),
  'validation':{'transition':scores(vy[vv],vtrans),'persistence':scores(vy[vv],vpersist),'blend':scores(vy[vv],vmix)},
  'validation_paired_game_bootstrap_delta_vs_0.98_persistence':{
   'transition':paired_bootstrap(per_game_scores(vy[vv],vtrans,gids,sides,allmask),baseg,seed),
   'blend':paired_bootstrap(candg,baseg,seed+1)}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',required=True);p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=97);a=p.parse_args()
 tr=load_compact(a.train);va=load_compact(a.validation)
 if str(tr['global_mode_label_version'][0])!=str(va['global_mode_label_version'][0]):raise ValueError('label-contract mismatch')
 if set(tr['game_ids'].tolist())&set(va['game_ids'].tolist()):raise ValueError('train/validation games overlap')
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 report={'schema':'death-aware-transition-prior-incremental-v1','test_access':False,
  'train_npz':str(a.train),'validation_npz':str(a.validation),'label_contract':str(tr['global_mode_label_version'][0]),
  'fold_contract':'GroupKFold by game_id; no row from held-out game contributes to its transition prior',
  'probability_baseline':'persistence 0.98 current mode, 0.01 each alternative; transition prior uses symmetric Dirichlet alpha selected on OOF train',
  'horizons':{}}
 for j,h in enumerate(HORIZONS):
  report['horizons'][f'{h}s']=report_h(tr,va,j,h,a.folds,a.seed+j*13)
  out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps({'horizon':h,'validation':report['horizons'][f'{h}s']['validation'],'selected_alpha':report['horizons'][f'{h}s']['selected_dirichlet_alpha_train_oof'],'lambda':report['horizons'][f'{h}s']['selected_transition_weight_train_oof']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
