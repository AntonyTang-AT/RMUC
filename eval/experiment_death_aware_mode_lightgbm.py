"""Train-split group-crossfit non-neural global-mode candidates.

Reads only the death-aware train and validation NPZ files. Five-fold CV groups
whole games; validation is evaluated once after candidate construction. No test
artifact is opened, and output paths are always explicit/new.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from forecast.temporal import HORIZONS, MODES, load_compact


def features(data):
    x = data["x"].astype(np.float32)
    # Current snapshot plus causal trailing means/slopes; retain all 12-frame
    # input for the baseline GRU, but provide the tree a compact interpretable
    # set of current state and recent change features.
    cur = x[:, -1]
    mean3 = x[:, -3:].mean(axis=1)
    mean6 = x[:, -6:].mean(axis=1)
    d1 = cur - x[:, -2]
    d3 = cur - x[:, -4]
    # Explicit current mode one-hot guarantees a strong persistence feature.
    base = np.eye(len(MODES), dtype=np.float32)[np.clip(data["base_mode"], 0, 2)]
    return np.concatenate([cur, mean3, mean6, d1, d3, base], axis=1)


def scores(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = p.argmax(1)
    f1 = []
    for c in range(len(MODES)):
        tp = np.sum((y == c) & (pred == c)); fp = np.sum((y != c) & (pred == c)); fn = np.sum((y == c) & (pred != c))
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    onehot = np.eye(len(MODES))[y]
    conf = p.max(1); correct = (pred == y)
    ece = 0.
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf >= lo) & (conf < lo + .1 if lo < .9 else conf <= 1.)
        if m.any(): ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {"n": int(len(y)), "accuracy": float(correct.mean()),
            "macro_f1": float(np.mean(f1)),
            "brier": float(np.mean(np.sum((p-onehot)**2, axis=1))),
            "log_loss": float(log_loss(y, p, labels=np.arange(3))), "ece": float(ece)}


def temp_scale(p, temp):
    lp = np.log(np.clip(p, 1e-8, 1.)) / temp
    lp -= lp.max(1, keepdims=True)
    q = np.exp(lp); return q / q.sum(1, keepdims=True)


def persistence_probs(base_mode):
    """Fair probabilistic version of persistence: 0.98 on the current class."""
    base=np.asarray(base_mode,dtype=np.int64)
    p=np.full((len(base),3),0.01,dtype=np.float64)
    legal=(base>=0)&(base<3)
    p[np.arange(len(base))[legal],base[legal]]=0.98
    return p


def fit_temperature(y, p):
    result = minimize_scalar(lambda z: log_loss(y, temp_scale(p, np.exp(z)), labels=np.arange(3)),
                             bounds=(-2.3, 2.3), method="bounded")
    return float(np.exp(result.x))


def per_game_scores(y, p, gids, sides, mask):
    out = {}
    for gid in np.unique(gids):
        vals=[]
        for side in ("红", "蓝"):
            m=mask & (gids==gid) & (sides==side)
            if m.any(): vals.append(scores(y[m], p[m]))
        if len(vals)==2:
            out[int(gid)]={k:float(np.mean([v[k] for v in vals])) for k in ("accuracy","macro_f1","brier","log_loss","ece")}
    return out


def equal_game_summary(game_scores):
    """Aggregate metrics using the same game-equal units as paired bootstrap."""
    if not game_scores:
        return {}
    metrics = ("accuracy", "macro_f1", "brier", "log_loss", "ece")
    return {key: float(np.mean([row[key] for row in game_scores.values()])) for key in metrics}


def paired_bootstrap(candidate, baseline, seed=0, reps=2000):
    rng=np.random.default_rng(seed); result={}
    for metric in ("accuracy","macro_f1","brier","log_loss","ece"):
        ids=sorted(set(candidate)&set(baseline))
        d=np.array([candidate[g][metric]-baseline[g][metric] for g in ids])
        idx=rng.integers(0,len(d),size=(reps,len(d)))
        boot=d[idx].mean(1)
        result[metric]={"paired_mean_delta_candidate_minus_persistence":float(d.mean()),
                        "game_bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
                        "n_games_red_blue_paired":len(ids)}
    return result


def choose_mix(y, probs, prior):
    # Candidate p=lambda*tree +(1-lambda)*persistence. Choose lambda on OOF
    # accuracy first, then macro-F1; deterministic tie breaks toward shrinkage.
    best=None
    for lam in np.linspace(0,1,21):
        p=lam*probs+(1-lam)*prior
        m=scores(y,p)
        key=(m["accuracy"],m["macro_f1"],-lam)
        if best is None or key>best[0]: best=(key,float(lam),m)
    return best[1],best[2]


def run(train_path, val_path, output, folds=5, seed=23, threads=8):
    train=load_compact(train_path); val=load_compact(val_path)
    if str(train["global_mode_label_version"][0]) != str(val["global_mode_label_version"][0]):
        raise ValueError("train/validation label contracts differ")
    X=features(train); Xv=features(val)
    gids=np.asarray(train["game_ids"]); vgids=np.asarray(val["game_ids"])
    sides=np.asarray(train["sides"]); vsides=np.asarray(val["sides"])
    cv=GroupKFold(n_splits=folds)
    report={"schema":"death-aware-global-mode-lightgbm-groupcv-v1","test_access":False,
            "train_npz":str(train_path),"validation_npz":str(val_path),
            "label_contract":str(train["global_mode_label_version"][0]),
            "features":"current, trailing 3/6 means, causal 1/3 snapshot deltas, current-mode one-hot",
            "grouping":"5-fold group split by whole game_id; each game red/blue remain together",
            "horizons":{}}
    for j,h in enumerate(HORIZONS):
        y=np.asarray(train["mode_y"][:,j],dtype=np.int64); valid=y>=0
        oof=np.full((len(y),3),np.nan,dtype=np.float64)
        for fold,(tr,te) in enumerate(cv.split(X[valid],y[valid],groups=gids[valid])):
            valid_rows=np.flatnonzero(valid); tr=valid_rows[tr]; te=valid_rows[te]
            model=LGBMClassifier(objective="multiclass",num_class=3,n_estimators=260,
                learning_rate=.045,num_leaves=15,max_depth=-1,min_child_samples=90,
                reg_lambda=4.,colsample_bytree=.8,subsample=.85,subsample_freq=1,
                class_weight=None,verbosity=-1,n_jobs=threads,random_state=seed+fold)
            model.fit(X[tr],y[tr]); oof[te]=model.predict_proba(X[te])
        vo=valid & np.isfinite(oof[:,0]); yy=y[vo]; pp=oof[vo]
        tt=fit_temperature(yy,pp); pcal=temp_scale(pp,tt)
        prior=persistence_probs(train["base_mode"])[vo]
        lam,cv_mix=choose_mix(yy,pcal,prior)
        # For the frozen holdout, refit once on all training games.
        model=LGBMClassifier(objective="multiclass",num_class=3,n_estimators=260,
            learning_rate=.045,num_leaves=15,min_child_samples=90,reg_lambda=4.,
            colsample_bytree=.8,subsample=.85,subsample_freq=1,verbosity=-1,
            n_jobs=threads,random_state=seed)
        model.fit(X[valid],y[valid])
        vy=np.asarray(val["mode_y"][:,j],dtype=np.int64); vv=vy>=0
        raw=model.predict_proba(Xv[vv]); calibrated=temp_scale(raw,tt)
        vprior=persistence_probs(val["base_mode"])[vv]
        blend=lam*calibrated+(1-lam)*vprior
        fold_cv={}
        for name,p in (("tree_raw",pp),("tree_temp_calibrated",pcal),("persistence",prior),("oof_blend",lam*pcal+(1-lam)*prior)):
            fold_cv[name]=scores(yy,p)
        train_group_baseline=per_game_scores(yy,prior,gids[vo],sides[vo],np.ones(vo.sum(),bool))
        train_group_boot={}
        for i,(name,p) in enumerate((("tree_raw",pp),("tree_temperature",pcal),
                                     ("oof_blend",lam*pcal+(1-lam)*prior))):
            cand=per_game_scores(yy,p,gids[vo],sides[vo],np.ones(vo.sum(),bool))
            train_group_boot[name]=paired_bootstrap(cand,train_group_baseline,seed+1000*j+i)
        # Pair CI on held-out games, red/blue averaged within each game.
        vm=np.zeros(len(vy),dtype=bool);vm[np.flatnonzero(vv)]=True
        vg_base=persistence_probs(val["base_mode"])
        vvpreds={"tree_raw":np.zeros((len(vy),3)),"tree_temperature":np.zeros((len(vy),3)),
                 "tree_blend":np.zeros((len(vy),3)),"persistence":vg_base}
        vvpreds["tree_raw"][vv]=raw;vvpreds["tree_temperature"][vv]=calibrated;vvpreds["tree_blend"][vv]=blend
        base_game=per_game_scores(vy[vv],vprior,vgids[vv],vsides[vv],np.ones(vv.sum(),bool))
        val_metrics={"tree_raw":scores(vy[vv],raw),"tree_temperature":scores(vy[vv],calibrated),
                     "tree_blend":scores(vy[vv],blend),"persistence":scores(vy[vv],vprior)}
        game_scores={name:per_game_scores(vy[vv],p,vgids[vv],vsides[vv],np.ones(vv.sum(),bool))
                     for name,p in (("tree_raw",raw),("tree_temperature",calibrated),("tree_blend",blend))}
        validation_game_equal={name:equal_game_summary(gs) for name,gs in game_scores.items()}
        validation_game_equal["persistence"]=equal_game_summary(base_game)
        report["horizons"][f"{h}s"]={"valid_train_rows":int(vo.sum()),"valid_validation_rows":int(vv.sum()),
            "oof_train_groupcv":fold_cv,"temperature_from_oof_train_only":tt,
            "persistence_blend_weight_from_oof_train_only":lam,
            "oof_train_group_bootstrap_delta_vs_smoothed_persistence":train_group_boot,
            "persistence_probabilities":"0.98 current mode, 0.01 other modes",
            "validation":val_metrics,
            "validation_game_equal_red_blue_equal":validation_game_equal,
            "validation_paired_game_bootstrap_delta_vs_persistence":{
                name:paired_bootstrap(gs,base_game,seed+101*j+i) for i,(name,gs) in enumerate(game_scores.items())}}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True);p.add_argument("--validation",required=True)
    p.add_argument("--output",required=True);p.add_argument("--folds",type=int,default=5)
    p.add_argument("--threads",type=int,default=8)
    a=p.parse_args();r=run(a.train,a.validation,a.output,a.folds,threads=a.threads)
    print(json.dumps({h:{"oof":v["oof_train_groupcv"],"validation":v["validation"],"lambda":v["persistence_blend_weight_from_oof_train_only"]} for h,v in r["horizons"].items()},ensure_ascii=False))

if __name__=="__main__": main()
