"""Causal score-stat feature candidate with per-split incremental caches.

Only death-aware train/validation NPZ and JSONL inputs are accepted. JSONL is
streamed with orjson; the summary uses the runtime mode-score/mobility helpers
and never reads target fields. Each split cache and parity metadata are written
before the next split is parsed. Model selection is whole-game OOF only.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import orjson
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from forecast.temporal import HORIZONS,MODES,load_compact
from eval.experiment_death_aware_mode_lightgbm import (scores,temp_scale,fit_temperature,
    persistence_probs,per_game_scores,paired_bootstrap)
from eval.experiment_global_mode_feature_ablation import _summary,FEATURE_NAMES

LABEL_VERSION='relative_margin_v2_death_aware_hp_le_zero'

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def build_cache(split,npz_path,jsonl_path,cache_dir,context=12):
 started=time.perf_counter();cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
 d=np.load(npz_path,allow_pickle=False);n=len(d['game_ids'])
 version=str(d['global_mode_label_version'][0])
 if version!=LABEL_VERSION:raise ValueError(f'{split}: expected death-aware label version, got {version}')
 names=list(FEATURE_NAMES);dim=len(names);window=np.zeros((n,context,dim),np.float32)
 key_rows=0;groups=0;bad_mode=0;previous_key=None;previous_time=None;ring=[]
 with gzip.open(jsonl_path,'rb') as stream:
  for i,line in enumerate(stream):
   if not line.strip():continue
   row=orjson.loads(line)
   actual=(int(row['game_id']),str(row['side']),round(float(row['issued_at']),5))
   expected=(int(d['game_ids'][key_rows]),str(d['sides'][key_rows]),round(float(d['issued_at'][key_rows]),5))
   if actual!=expected:raise ValueError(f'{split}: row-key mismatch at {key_rows}: {actual} != {expected}')
   key=(actual[0],actual[1])
   if key!=previous_key:
    ring=[];previous_time=float(row['issued_at']);previous_key=key;groups+=1
   vec,raw=_summary(row,previous_time)
   mode_id=MODES.index(raw['current_global_mode'])
   if mode_id!=int(d['base_mode'][key_rows]):bad_mode+=1
   ring.append(vec)
   if len(ring)>context:ring.pop(0)
   window[key_rows,-len(ring):]=np.asarray(ring,np.float32)
   previous_time=float(row['issued_at']);key_rows+=1
 if key_rows!=n:raise ValueError(f'{split}: parsed rows {key_rows} != compact rows {n}')
 if bad_mode:raise ValueError(f'{split}: {bad_mode} current base_mode mismatches')
 cache_path=cache_dir/f'{split}_score_stats.npz'
 np.savez_compressed(cache_path,summary_windows=window,game_ids=d['game_ids'],sides=d['sides'],issued_at=d['issued_at'],base_mode=d['base_mode'])
 metadata={'schema':'death-aware-score-stat-cache-v1','split':split,'test_access':False,
  'label_contract':version,'feature_schema':str(d['feature_schema_version'][0]),
  'source_npz':str(npz_path),'source_jsonl':str(jsonl_path),'cache_path':str(cache_path),
  'source_npz_sha256':sha256(npz_path),'source_jsonl_sha256':sha256(jsonl_path),
  'rows':n,'game_side_groups':groups,'context':context,'feature_dim_per_timestep':dim,
  'feature_names':names,'orjson_version':orjson.__version__,'row_key_parity':'exact (game_id, side, issued_at to 1e-5)',
  'base_mode_parity':'exact against runtime _mode_from_groups/_mode_score; alive mobility excludes hp<=0/destroyed; missing base/HP flags preserved',
  'target_fields_read_by_builder':False,'elapsed_seconds':round(time.perf_counter()-started,3)}
 meta_path=cache_dir/f'{split}_metadata.json';meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
 print(orjson.dumps({'cache_written':str(cache_path),'metadata_written':str(meta_path),'rows':n,'groups':groups,'row_key_parity':True,'base_mode_parity':True,'elapsed_seconds':metadata['elapsed_seconds']}).decode(),flush=True)
 return metadata

def model(seed,threads):
 return LGBMClassifier(objective='multiclass',num_class=3,n_estimators=120,learning_rate=.05,
  num_leaves=9,min_child_samples=120,reg_lambda=5.,colsample_bytree=.9,
  subsample=.85,subsample_freq=1,verbosity=-1,n_jobs=threads,random_state=seed)

def features(cache):
 s=cache['summary_windows'];cur=s[:,-1];m3=s[:,-3:].mean(1);m6=s[:,-6:].mean(1);d1=cur-s[:,-2];d3=cur-s[:,-4]
 base=np.eye(3,dtype=np.float32)[np.clip(cache['base_mode'].astype(int),0,2)]
 return np.concatenate([cur,m3,m6,d1,d3,base],axis=1)

def report_h(train,val,X,V,j,h,folds,seed,threads):
 y=train['mode_y'][:,j].astype(int);valid=y>=0;rows=np.flatnonzero(valid);yy=y[valid];gids=train['game_ids'];sides=train['sides'];base=train['base_mode'].astype(int)
 oof=np.full((len(y),3),np.nan);cv=GroupKFold(n_splits=folds)
 for fold,(itr,ite) in enumerate(cv.split(X[valid],yy,groups=gids[valid])):
  tr,te=rows[itr],rows[ite];m=model(seed+fold,threads);m.fit(X[tr],y[tr]);oof[te]=m.predict_proba(X[te])
 pp=oof[valid];temp=fit_temperature(yy,pp);cal=temp_scale(pp,temp);persist=persistence_probs(base)[valid]
 best=None
 for lam in np.linspace(0,1,21):
  q=lam*cal+(1-lam)*persist;sc=scores(yy,q);key=(.5*sc['accuracy']+.5*sc['macro_f1'],sc['macro_f1'],sc['accuracy'],-lam)
  if best is None or key>best[0]:best=(key,float(lam),q)
 lam=best[1];oofmix=best[2]
 vg=np.asarray(val['game_ids']);vside=np.asarray(val['sides']);vy=val['mode_y'][:,j].astype(int);vv=vy>=0
 final=model(seed,threads);final.fit(X[valid],y[valid]);raw=final.predict_proba(V[vv]);vcal=temp_scale(raw,temp)
 vpersist=persistence_probs(val['base_mode'])[vv];vmix=lam*vcal+(1-lam)*vpersist
 oof_base=per_game_scores(yy,persist,gids[valid],sides[valid],np.ones(valid.sum(),bool))
 oof_cand=per_game_scores(yy,oofmix,gids[valid],sides[valid],np.ones(valid.sum(),bool))
 vbase=per_game_scores(vy[vv],vpersist,vg[vv],vside[vv],np.ones(vv.sum(),bool))
 vcand=per_game_scores(vy[vv],vmix,vg[vv],vside[vv],np.ones(vv.sum(),bool))
 return {'n_train_oof':int(valid.sum()),'n_validation':int(vv.sum()),
  'train_oof':{'candidate':scores(yy,oofmix),'persistence':scores(yy,persist),
    'paired_game_ci_vs_persistence':paired_bootstrap(oof_cand,oof_base,seed+20)},
  'validation':{'candidate':scores(vy[vv],vmix),'persistence':scores(vy[vv],vpersist),
    'paired_game_ci_vs_persistence':paired_bootstrap(vcand,vbase,seed+40)},
  'temperature_from_train_oof_only':temp,'candidate_weight_from_train_oof_only':lam}

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--train-jsonl',required=True);p.add_argument('--validation-jsonl',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--folds',type=int,default=5);p.add_argument('--threads',type=int,default=6);p.add_argument('--seed',type=int,default=103);p.add_argument('--context',type=int,default=12);a=p.parse_args()
 out=Path(a.output_dir);cache_dir=out/'cache';out.mkdir(parents=True,exist_ok=True)
 train_meta=build_cache('train',a.train,a.train_jsonl,cache_dir,a.context)
 val_meta=build_cache('validation',a.validation,a.validation_jsonl,cache_dir,a.context)
 trc=np.load(train_meta['cache_path'],allow_pickle=False);vac=np.load(val_meta['cache_path'],allow_pickle=False)
 tr=load_compact(a.train);va=load_compact(a.validation)
 if set(tr['game_ids'].tolist())&set(va['game_ids'].tolist()):raise ValueError('train/validation games overlap')
 X=features(trc);V=features(vac)
 report={'schema':'death-aware-score-stats-lightgbm-groupcv-v1','test_access':False,
  'cache_metadata':{'train':str(cache_dir/'train_metadata.json'),'validation':str(cache_dir/'validation_metadata.json')},
  'model':'LightGBM multiclass; 120 trees, 9 leaves, min_child_samples 120, l2=5; score-stat only (no raw entity vectors)',
  'grouping':'GroupKFold by entire training game; validation set remains untouched until final candidate scoring',
  'probability_baseline':'0.98 current class + 0.01 each other class','horizons':{}}
 report_path=out/'report.json'
 for j,h in enumerate(HORIZONS):
  report['horizons'][f'{h}s']=report_h(tr,va,X,V,j,h,a.folds,a.seed+31*j,a.threads)
  report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
  print(orjson.dumps({'horizon':h,**report['horizons'][f'{h}s']}).decode(),flush=True)
if __name__=='__main__':main()
