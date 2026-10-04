"""NPZ-only causal base_mode history feature experiment.

Only death-aware train/validation NPZ are opened. No JSONL parsing, test data,
or production/default artifacts are used. Adds past mode lags, current run
length, and change counts to the same compact causal LightGBM features.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from forecast.temporal import HORIZONS, MODES, load_compact
try:
    from .experiment_death_aware_mode_lightgbm import features, scores, per_game_scores, equal_game_summary, paired_bootstrap, persistence_probs
except ImportError:  # direct `python eval/script.py` execution
    from experiment_death_aware_mode_lightgbm import features, scores, per_game_scores, equal_game_summary, paired_bootstrap, persistence_probs

def history(data):
    """History on each (game,side), ordered by issued_at, causally through t."""
    n=len(data['base_mode']); base=np.asarray(data['base_mode'],dtype=np.int64)
    gids=np.asarray(data['game_ids']); sides=np.asarray(data['sides']); ts=np.asarray(data['issued_at'],dtype=np.float64)
    # lags 3/6/9/15 seconds; rows normally arrive at 3-second cadence.
    # Exact timestamp matching avoids treating a missing interval as adjacent.
    out=np.zeros((n,15),dtype=np.float32)
    out[:,:4]=-1.0
    for gid in np.unique(gids):
        for side in np.unique(sides[gids==gid]):
            ix=np.flatnonzero((gids==gid)&(sides==side))
            ix=ix[np.argsort(ts[ix],kind='stable')]
            at={round(float(ts[k]),3):int(k) for k in ix}
            for pos,k in enumerate(ix):
                t=round(float(ts[k]),3)
                for j,delta in enumerate((3,6,9,15)):
                    old=at.get(round(t-delta,3),None)
                    if old is not None: out[k,j]=base[old]
                # Consecutive same-mode run length (in 3s snapshots, capped 15)
                run=1
                for q in range(pos-1,max(-1,pos-15),-1):
                    prev=ix[q]
                    nxt=ix[q+1]
                    if base[prev]!=base[k] or abs(ts[nxt]-ts[prev]-3)>1e-3: break
                    run+=1
                out[k,4]=min(run,15)
                # number of mode changes in the preceding 3/6/9/15 seconds;
                # missing exact steps remain absent, not inferred.
                for j,span in enumerate((3,6,9,15),start=5):
                    earlier=[at.get(round(t-d,3)) for d in range(0,span+1,3)]
                    pairs=[(int(a),int(b)) for a,b in zip(earlier,earlier[1:]) if a is not None and b is not None]
                    out[k,j]=sum(base[a]!=base[b] for a,b in pairs)
                # fraction of preceding observed snapshots matching current mode
                for j,span in enumerate((6,15),start=9):
                    prev=[at.get(round(t-d,3)) for d in range(3,span+1,3)]
                    prev=[int(v) for v in prev if v is not None]
                    out[k,j]=sum(base[a]==base[k] for a in prev)/max(1,len(prev))
                # elapsed time since last observed transition, capped 45 sec
                since=0
                for q in range(pos-1,-1,-1):
                    prev=ix[q]
                    if abs(ts[k]-ts[prev]-(pos-q)*3)>1e-3 or base[prev]!=base[k]: break
                    since+=3
                out[k,11]=min(since,45)
                # one-hot past mode at 6s lag (unknown all zero)
                lag=out[k,1]
                if lag>=0: out[k,12+int(lag)]=1
    return out

def make_model(seed,threads):
    return LGBMClassifier(objective='multiclass',num_class=3,n_estimators=140,
        learning_rate=.055,num_leaves=9,min_child_samples=120,reg_lambda=5.,
        colsample_bytree=.82,subsample=.85,subsample_freq=1,verbosity=-1,
        n_jobs=threads,random_state=seed)

def run(train_path,val_path,out_path,folds=5,seed=43,threads=4):
    tr=load_compact(train_path); va=load_compact(val_path)
    if str(tr['global_mode_label_version'][0])!=str(va['global_mode_label_version'][0]): raise ValueError('label contracts differ')
    x0=features(tr); xv0=features(va)
    lag=history(tr); vlag=history(va)
    x1=np.concatenate([x0,lag],1); xv1=np.concatenate([xv0,vlag],1)
    gids=np.asarray(tr['game_ids']);vgids=np.asarray(va['game_ids']); sides=np.asarray(tr['sides']);vsides=np.asarray(va['sides'])
    cv=GroupKFold(n_splits=folds)
    report={'schema':'death-aware-mode-lag-feature-experiment-v1','test_access':False,
      'train_npz':str(train_path),'validation_npz':str(val_path),
      'label_contract':str(tr['global_mode_label_version'][0]),'groups':'5-fold GroupKFold by whole game_id',
      'history_features':['base_mode lag 3/6/9/15s exact timestamps','current-mode run length capped 15 samples',
        'mode change count in exact observed 3/6/9/15s windows','same-mode ratio over 6/15s','seconds since last transition capped 45','6s lag mode one-hot'],
      'feature_coverage':{'train_rows':len(tr['base_mode']),'train_nonmissing_exact_3s_lag':float(np.mean(lag[:,0]>=0)),
       'train_nonmissing_exact_6s_lag':float(np.mean(lag[:,1]>=0)),'validation_rows':len(va['base_mode']),
       'validation_nonmissing_exact_3s_lag':float(np.mean(vlag[:,0]>=0)),'validation_nonmissing_exact_6s_lag':float(np.mean(vlag[:,1]>=0))},
      'horizons':{}}
    out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True)
    for j,h in enumerate(HORIZONS):
        y=np.asarray(tr['mode_y'][:,j],int); m=y>=0; rows=np.flatnonzero(m); yy=y[m]
        oof0=np.full((len(y),3),np.nan);oof1=np.full((len(y),3),np.nan)
        for fold,(a,b) in enumerate(cv.split(x0[m],yy,gids[m])):
            ai=rows[a];bi=rows[b]
            model=make_model(seed+fold,threads);model.fit(x0[ai],y[ai]);oof0[bi]=model.predict_proba(x0[bi])
            model=make_model(seed+fold,threads);model.fit(x1[ai],y[ai]);oof1[bi]=model.predict_proba(x1[bi])
        vv=np.asarray(va['mode_y'][:,j],int); vm=vv>=0; vr=np.flatnonzero(vm)
        valid_metrics={}; val_games={}; train_metrics={}
        for name,xx,oo in (('current_raw',x0,oof0),('with_mode_history',x1,oof1)):
            p=oo[m]
            train_metrics[name]=scores(yy,p)
            model=make_model(seed,threads);model.fit(x0[m] if name=='current_raw' else x1[m],yy)
            pv=model.predict_proba(xv0[vm] if name=='current_raw' else xv1[vm])
            valid_metrics[name]=scores(vv[vm],pv)
            val_games[name]=per_game_scores(vv[vm],pv,vgids[vm],vsides[vm],np.ones(vm.sum(),bool))
        baseline=per_game_scores(vv[vm],persistence_probs(va['base_mode'])[vm],vgids[vm],vsides[vm],np.ones(vm.sum(),bool))
        game_equal={name:equal_game_summary(gs) for name,gs in val_games.items()}
        game_equal['persistence']=equal_game_summary(baseline)
        pair={}
        for ix,name in enumerate(('current_raw','with_mode_history')):
            pair[name+'_vs_current_raw']=paired_bootstrap(val_games[name],val_games['current_raw'],seed+900+j*7+ix) if name!='current_raw' else None
            pair[name+'_vs_persistence']=paired_bootstrap(val_games[name],baseline,seed+950+j*7+ix)
        # The shared helper has a persistence-specific legacy field name;
        # this experiment compares both raw and history models to named bases.
        for metrics in pair.values():
            if metrics is None:
                continue
            for metric in metrics.values():
                metric['paired_mean_delta_model_minus_comparator'] = metric.pop(
                    'paired_mean_delta_candidate_minus_persistence')
        report['horizons'][f'{h}s']={'valid_train_rows':int(m.sum()),'valid_validation_rows':int(vm.sum()),
          'train_oof_5fold':train_metrics,'validation':valid_metrics,
          'validation_game_equal_red_blue_equal':game_equal,'validation_paired_game_ci':pair}
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'horizon':h,'train_oof':train_metrics,'validation':valid_metrics,
          'paired_history_vs_raw':pair['with_mode_history_vs_current_raw']},ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',required=True);p.add_argument('--folds',type=int,default=5);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    run(a.train,a.validation,a.output,a.folds,threads=a.threads)
if __name__=='__main__': main()
