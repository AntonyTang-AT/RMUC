"""Causal summary-feature and transition-prior global-mode experiments.

Only the death-aware train and validation NPZ/JSONL contracts are accepted.
All candidate and calibration choices use five-fold game-group OOF train
predictions; validation is a single confirmation set. No test path exists.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from forecast.temporal import HORIZONS, MODES
from eval.experiment_death_aware_mode_lightgbm import (features as base_features,
    scores, temp_scale, fit_temperature, persistence_probs, per_game_scores, paired_bootstrap)
from eval.experiment_global_mode_feature_ablation import _read_groups, _summary, FEATURE_NAMES

def summary_windows(npz_path,jsonl_path):
    d=np.load(npz_path,allow_pickle=False); n=len(d['game_ids']); stats=np.zeros((n,len(FEATURE_NAMES)),np.float32)
    groups=[];cursor=0
    for (gid,side),rows in _read_groups(Path(jsonl_path)):
        start=cursor; previous=float(rows[0]['issued_at'])
        local=[]
        for i,row in enumerate(rows):
            expected=(int(d['game_ids'][cursor]),str(d['sides'][cursor]),round(float(d['issued_at'][cursor]),5))
            actual=(int(gid),str(side),round(float(row['issued_at']),5))
            if actual!=expected: raise ValueError(f'row alignment error {cursor}: {actual} != {expected}')
            vec,raw=_summary(row,previous if i else float(row['issued_at']))
            if MODES.index(raw['current_global_mode'])!=int(d['base_mode'][cursor]):
                raise ValueError(f'death-aware base_mode parity failure at {cursor}')
            local.append(vec);stats[cursor]=vec;previous=float(row['issued_at']);cursor+=1
        groups.append((start,cursor,int(gid),str(side)))
    if cursor!=n: raise ValueError(f'JSONL rows {cursor} != NPZ rows {n}')
    win=np.zeros((n,d['x'].shape[1],stats.shape[1]),np.float32)
    for start,end,_gid,_side in groups:
        for i in range(start,end):
            a=max(start,i-win.shape[1]+1); z=stats[a:i+1];win[i,-len(z):]=z
    return d,win,{'rows':n,'groups':len(groups),'feature_dim_added':win.shape[-1],
                  'context':win.shape[1],'base_mode_parity':'exact','alignment':'game_id,side,issued_at verified'}

def feature_views(d,stats):
    x=d['x'];cur=x[:,-1];m3=x[:,-3:].mean(1);m6=x[:,-6:].mean(1);d1=cur-x[:,-2];d3=cur-x[:,-4]
    sx=stats;sc=sx[:,-1];sm3=sx[:,-3:].mean(1);sm6=sx[:,-6:].mean(1);sd1=sc-sx[:,-2];sd3=sc-sx[:,-4]
    base=base_features(d)
    extra=np.concatenate([sc,sm3,sm6,sd1,sd3],axis=1)
    return base,np.concatenate([base,extra],axis=1)

def trans_prior(y,base,alpha=2.0):
    c=np.ones((3,3),dtype=np.float64)*float(alpha)
    for a,b in zip(base,y):
        if 0<=a<3 and 0<=b<3:c[int(a),int(b)]+=1
    c/=c.sum(1,keepdims=True)
    return c

def transition_probs(base,matrix):
    b=np.clip(np.asarray(base,dtype=int),0,2);return matrix[b]

def class_bias_fit(y,p):
    """Small train-OOF-only coordinate search for macro-F1 logit offsets."""
    logits=np.log(np.clip(p,1e-8,1));bias=np.zeros(3);best=scores(y,p)['macro_f1']
    for _ in range(3):
        for c in range(3):
            chosen=(best,float(np.mean(np.asarray(y)==np.argmax(logits+bias,1))),0.)
            # Allow enough log-odds movement to recover a weakly predicted
            # class, while still limiting the selected offsets to +/-1.5.
            for delta in np.linspace(-1.5,1.5,61):
                b=bias.copy();b[c]+=delta
                pp=logits+b;pp-=pp.max(1,keepdims=True);pp=np.exp(pp);pp/=pp.sum(1,keepdims=True)
                m=scores(y,pp);key=(m['macro_f1'],m['accuracy'],-abs(delta))
                if key>chosen:chosen=key;chosen_b=b
            if 'chosen_b' in locals() and chosen[0]>=best:
                bias=chosen_b;best=chosen[0]
            if 'chosen_b' in locals():del chosen_b
    return bias.tolist()

def apply_bias(p,bias):
    z=np.log(np.clip(p,1e-8,1))+np.asarray(bias);z-=z.max(1,keepdims=True);z=np.exp(z);return z/z.sum(1,keepdims=True)

def class_diagnostics(y,p):
    y=np.asarray(y,int);pred=np.asarray(p).argmax(1);cm=np.zeros((3,3),dtype=int);rows=[]
    for a,b in zip(y,pred):cm[a,b]+=1
    for c in range(3):
        tp=cm[c,c];fp=cm[:,c].sum()-tp;fn=cm[c,:].sum()-tp
        rows.append({'class':MODES[c],'precision':float(tp/max(1,tp+fp)),
                     'recall':float(tp/max(1,tp+fn)),
                     'f1':float(2*tp/max(1,2*tp+fp+fn))})
    return {'confusion_true_rows_pred_cols':cm.tolist(),'per_class':rows}

def choose_mix(y,p,trans):
    best=None
    for lam in np.linspace(0,1,21):
        q=lam*p+(1-lam)*trans;m=scores(y,q);key=(.5*m['accuracy']+.5*m['macro_f1'],m['macro_f1'],m['accuracy'],-lam)
        if best is None or key>best[0]:best=(key,float(lam),m)
    return best[1]

def model(seed,threads):
    return LGBMClassifier(objective='multiclass',num_class=3,n_estimators=160,learning_rate=.05,
        num_leaves=15,min_child_samples=100,reg_lambda=4.,colsample_bytree=.8,
        subsample=.85,subsample_freq=1,verbosity=-1,n_jobs=threads,random_state=seed)

def run(a):
    tr,ts,tmeta=summary_windows(a.train,a.train_jsonl);va,vs,vmeta=summary_windows(a.validation,a.validation_jsonl)
    X0,X1=feature_views(tr,ts);V0,V1=feature_views(va,vs)
    gids=np.asarray(tr['game_ids']);vgids=np.asarray(va['game_ids']);sides=np.asarray(tr['sides']);vsides=np.asarray(va['sides'])
    cv=GroupKFold(n_splits=a.folds)
    report={'schema':'death-aware-mode-summary-transition-groupcv-v1','test_access':False,
        'train':tmeta,'validation':vmeta,'feature_names_added':list(FEATURE_NAMES),
        'feature_contract':'20 raw-observable summary scalars per causal snapshot; same _mode_score/_hp_ratio/_mobile functions and missing flags as runtime; 12-step history aligned to compact row windows',
        'split_contract':'five-fold group split by whole game_id; validation only confirms train-OOF-selected candidates',
        'horizons':{}}
    for j,h in enumerate(HORIZONS):
        y=np.asarray(tr['mode_y'][:,j],int);valid=y>=0;rows=np.flatnonzero(valid);yy=y[valid]
        oof={'base':np.full((len(y),3),np.nan),'summary':np.full((len(y),3),np.nan),'transition':np.full((len(y),3),np.nan)}
        for fold,(itr,ite) in enumerate(cv.split(X0[valid],yy,groups=gids[valid])):
            trr,ter=rows[itr],rows[ite]
            # Estimate transition prior from only this fold's train games.
            mat=trans_prior(y[trr],tr['base_mode'][trr],alpha=2.)
            oof['transition'][ter]=transition_probs(tr['base_mode'][ter],mat)
            m0=model(a.seed+fold,a.threads);m1=model(a.seed+fold,a.threads)
            m0.fit(X0[trr],y[trr]);m1.fit(X1[trr],y[trr])
            oof['base'][ter]=m0.predict_proba(X0[ter]);oof['summary'][ter]=m1.predict_proba(X1[ter])
        trans=oof['transition'][valid];basep=oof['base'][valid];summp=oof['summary'][valid]
        temp0=fit_temperature(yy,basep);temp1=fit_temperature(yy,summp)
        basecal=temp_scale(basep,temp0);sumcal=temp_scale(summp,temp1)
        lam0=choose_mix(yy,basecal,trans);lam1=choose_mix(yy,sumcal,trans)
        base_mix=lam0*basecal+(1-lam0)*trans;sum_mix=lam1*sumcal+(1-lam1)*trans
        bias0=class_bias_fit(yy,base_mix);bias1=class_bias_fit(yy,sum_mix)
        base_macro=apply_bias(base_mix,bias0);sum_macro=apply_bias(sum_mix,bias1)
        pbase=persistence_probs(tr['base_mode'])[valid]
        vmat=trans_prior(y[valid],tr['base_mode'][valid],alpha=2.)
        vy=np.asarray(va['mode_y'][:,j],int);vv=vy>=0
        validation={};oofmetrics={}
        variants={
            'base_tree_mix':(base_mix[...,:],lambda p: p),
            'summary_tree_mix':(sum_mix,lambda p:p),
            'base_macro_bias':(base_macro,lambda p:apply_bias(p,bias0)),
            'summary_macro_bias':(sum_macro,lambda p:apply_bias(p,bias1)),
        }
        for name,p in (('base_tree',basep),('summary_tree',summp),('base_mix',base_mix),('summary_mix',sum_mix),('base_macro_bias',base_macro),('summary_macro_bias',sum_macro),('transition',trans),('persistence',pbase)):
            oofmetrics[name]=scores(yy,p)
        # Full-train models and prior for the single validation confirmation.
        tmatrix=trans_prior(y[valid],tr['base_mode'][valid],alpha=2.)
        pvt=transition_probs(va['base_mode'][vv],tmatrix)
        ppersist=persistence_probs(va['base_mode'])[vv]
        val_predictions={'transition':pvt,'persistence':ppersist}
        for variant,xx,temp,lam,bias in [('base',X0,temp0,lam0,bias0),('summary',X1,temp1,lam1,bias1)]:
            mm=model(a.seed,a.threads);mm.fit(xx[valid],y[valid]);raw=mm.predict_proba((V0 if variant=='base' else V1)[vv]);cal=temp_scale(raw,temp);mix=lam*cal+(1-lam)*pvt
            val_predictions[f'{variant}_mix']=mix;val_predictions[f'{variant}_macro_bias']=apply_bias(mix,bias)
        vg={'persistence':per_game_scores(vy[vv],val_predictions['persistence'],vgids[vv],vsides[vv],np.ones(vv.sum(),bool))}
        boot={};valmetrics={}
        for name,p in val_predictions.items():
            valmetrics[name]=scores(vy[vv],p)
            if name!='persistence':
                gs=per_game_scores(vy[vv],p,vgids[vv],vsides[vv],np.ones(vv.sum(),bool))
                boot[name]=paired_bootstrap(gs,vg['persistence'],a.seed+100*j+len(boot))
        report['horizons'][f'{h}s']={'target_distribution_train':np.bincount(yy,minlength=3).tolist(),
          'target_distribution_validation':np.bincount(vy[vv],minlength=3).tolist(),
          'transition_counts_current_rows_future_cols_train':[[int(np.sum(valid&(tr['base_mode']==c)&(y==z))) for z in range(3)] for c in range(3)],
          'transition_counts_current_rows_future_cols_validation':[[int(np.sum(vv&(va['base_mode']==c)&(vy==z))) for z in range(3)] for c in range(3)],
          'train_oof':oofmetrics,'temperature_train_oof_only':{'base':temp0,'summary':temp1},
          'transition_blend_weight_train_oof_only':{'base':lam0,'summary':lam1},
          'macro_f1_class_bias_train_oof_only':{'base':bias0,'summary':bias1},
          'validation':valmetrics,
          'validation_class_diagnostics':{name:class_diagnostics(vy[vv],p) for name,p in val_predictions.items()},
          'validation_paired_game_bootstrap_delta_vs_0.98_persistence':boot}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--train-jsonl',required=True);p.add_argument('--validation-jsonl',required=True);p.add_argument('--output',required=True);p.add_argument('--folds',type=int,default=5);p.add_argument('--threads',type=int,default=8);p.add_argument('--seed',type=int,default=73);a=p.parse_args();r=run(a)
    print(json.dumps({h:{'oof':v['train_oof'],'val':v['validation'],'lambda':v['transition_blend_weight_train_oof_only']} for h,v in r['horizons'].items()},ensure_ascii=False))
if __name__=='__main__':main()
