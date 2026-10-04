"""Fixed train-only whole-game OOF test of moderate class weighting.

Uses only the death-aware training split and cached causal train features.
Weights are prescribed as inverse square-root class prevalence, normalized to
mean one within each training fold and horizon. No weight search or val/test
access. The comparator is the saved same-fold motion+zone/score model.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from forecast.temporal import HORIZONS, MODES, load_compact
from eval.experiment_death_aware_mode_lightgbm import features as direct_features

SPLIT=ROOT/'out_m2/temporal_forecast_death_aware_candidate/train.npz'
ZONE=ROOT/'out_m2/death_aware_global_mode_experiments/nested_transition_expert_30s_5outer_pilot/causal_train_features.npz'
MOTION=ROOT/'out_m2/death_aware_global_mode_experiments/visible_events_train_audit/motion_features.npz'
BASE=ROOT/'out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/motion_candidate_oof.npz'
OUT=ROOT/'out_m2/death_aware_global_mode_experiments/moderate_class_weight_oof'
DOC=ROOT/'docs/analysis/2026-09-27-death-aware-moderate-class-weight-oof.md'
SEED=1927

def scores(y,p):
    pred=p.argmax(1); conf=np.zeros((3,3),dtype=np.int64); np.add.at(conf,(y,pred),1)
    recalls=[]; f1=[]
    for c in range(3):
        tp=conf[c,c]; fp=conf[:,c].sum()-tp; fn=conf[c].sum()-tp
        recalls.append(tp/conf[c].sum() if conf[c].sum() else 0.)
        f1.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.)
    pp=np.clip(p,1e-12,1.)
    one=np.eye(3)[y]
    return {'accuracy':float(np.mean(pred==y)),'macro_f1':float(np.mean(f1)),
      'balanced_accuracy':float(np.mean(recalls)),
      'brier':float(np.mean(np.sum((p-one)**2,axis=1))),
      'log_loss':float(-np.mean(np.log(pp[np.arange(len(y)),y])))}

def cell_scores(y,p,gids,sides,valid):
    out={}
    for gid in np.unique(gids[valid]):
        sidevals=[]
        for side in ('红','蓝'):
            m=valid&(gids==gid)&(sides==side)
            if m.any(): sidevals.append(scores(y[m],p[m]))
        if len(sidevals)==2:
            out[int(gid)]={k:float(np.mean([s[k] for s in sidevals])) for k in sidevals[0]}
    return out

def paired(a,b,seed,reps=2000):
    ids=sorted(set(a)&set(b)); rng=np.random.default_rng(seed); ix=rng.integers(0,len(ids),(reps,len(ids)))
    out={'paired_games':len(ids),'replicates':reps,'delta_candidate_minus_motion':{}}
    for k in a[ids[0]]:
        d=np.asarray([a[g][k]-b[g][k] for g in ids]); boot=d[ix].mean(1)
        out['delta_candidate_minus_motion'][k]={'mean':float(d.mean()),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
    return out

def main():
    start=time.perf_counter(); d=load_compact(SPLIT)
    with np.load(ZONE,allow_pickle=False) as z, np.load(MOTION,allow_pickle=False) as m, np.load(BASE,allow_pickle=False) as b:
        for k in ('game_ids','sides','issued_at','mode_y','base_mode'):
            if not np.array_equal(d[k],b[k]): raise ValueError(f'baseline alignment mismatch: {k}')
            if k in z.files and not np.array_equal(d[k],z[k]): raise ValueError(f'zone alignment mismatch: {k}')
        x=np.concatenate([direct_features(d),z['extra_features'],m['x']],axis=1).astype(np.float32)
        baseline=np.asarray(b['motion_candidate'],dtype=np.float32)
        expected_folds=np.asarray(b['fold_ids']);
    gids=np.asarray(d['game_ids'],np.int64); sides=np.asarray(d['sides']).astype(str)
    y=np.asarray(d['mode_y'],np.int64); cur=np.asarray(d['base_mode'],np.int64)
    pred=np.full_like(baseline,np.nan); folds=np.full(len(gids),-1,np.int8); fold_reports=[]
    for fold,(tr,held) in enumerate(GroupKFold(n_splits=5).split(x,groups=gids)):
        train_fold=tr.copy(); held_fold=held.copy()
        folds[held]=fold
        if not np.all(expected_folds[held]==fold): raise ValueError('OOF fold mismatch')
        hr={}
        for j,h in enumerate(HORIZONS):
            tr=train_fold[(y[train_fold,j]>=0)&(cur[train_fold]>=0)&(cur[train_fold]<3)]
            heldj=held_fold[(y[held_fold,j]>=0)&(cur[held_fold]>=0)&(cur[held_fold]<3)]
            counts=np.bincount(y[tr,j],minlength=3).astype(float)
            if np.any(counts==0): raise ValueError(f'class missing in train fold {fold} horizon {h}: {counts}')
            w=np.sqrt(len(tr)/(3*counts)); w/=w.mean()
            sw=w[y[tr,j]]
            model=LGBMClassifier(objective='multiclass',num_class=3,n_estimators=260,learning_rate=.045,
                num_leaves=15,max_depth=-1,min_child_samples=90,reg_lambda=4.,colsample_bytree=.8,
                subsample=.85,subsample_freq=1,class_weight=None,verbosity=-1,n_jobs=4,random_state=SEED+fold)
            model.fit(x[tr],y[tr,j],sample_weight=sw)
            ph=model.predict_proba(x[heldj])
            for ci,cls in enumerate(model.classes_): pred[heldj,j,int(cls)]=ph[:,ci]
            hr[f'{h}s']={'train_rows':len(tr),'held_rows':len(heldj),'train_class_counts':counts.astype(int).tolist(),
                         'sqrt_inverse_prevalence_weights':w.tolist()}
        fold_reports.append({'fold':fold,'train_games':len(np.unique(gids[train_fold])),'heldout_games':len(np.unique(gids[held_fold])),'horizons':hr})
        print(json.dumps({'fold':fold,'elapsed_seconds':round(time.perf_counter()-start,1)}),flush=True)
    if not np.array_equal(folds,expected_folds): raise ValueError('fold ids differ')
    results={}
    for j,h in enumerate(HORIZONS):
        valid=(y[:,j]>=0)&np.isfinite(pred[:,j,0]);
        if not valid.any(): raise ValueError(f'no valid rows at {h}')
        candidate=cell_scores(y[:,j],pred[:,j],gids,sides,valid)
        reference=cell_scores(y[:,j],baseline[:,j],gids,sides,valid)
        paired_delta=paired(candidate,reference,SEED+h)
        pooled_candidate=scores(y[valid,j],pred[valid,j]); pooled_base=scores(y[valid,j],baseline[valid,j])
        mean_cell=lambda cells:{k:float(np.mean([v[k] for v in cells.values()])) for k in next(iter(cells.values()))}
        results[f'{h}s']={'valid_rows':int(valid.sum()),'pooled':{'candidate':pooled_candidate,'motion_baseline':pooled_base},
          'game_equal_red_blue_equal':{'candidate':mean_cell(candidate),'motion_baseline':mean_cell(reference)},'paired_game_bootstrap':paired_delta}
    OUT.mkdir(parents=True,exist_ok=True)
    report={'schema':'death-aware-moderate-class-weight-train-group-oof-v1','access':{'opened_splits':['train'],'validation':False,'test':False},
      'label_contract':str(d['global_mode_label_version'][0]),'train_rows':len(gids),'train_games':len(np.unique(gids)),
      'grouping':'same saved 5-fold GroupKFold by complete game_id; red/blue views kept together',
      'candidate':'same features and LightGBM hyperparameters; per-fold/per-horizon sample weights sqrt(N/(3*n_class)), normalized to mean one; fixed a priori, not tuned',
      'features':'direct temporal inputs + cached causal zone/observable-score aggregates + cached causal current-observable motion features',
      'comparator':'saved same-fold motion+zone/score OOF; exact game/split/fold alignment verified',
      'folds':fold_reports,'horizons':results,'runtime_seconds':time.perf_counter()-start}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'candidate_oof.npz',game_ids=gids,sides=sides,issued_at=d['issued_at'],mode_y=y,base_mode=cur,fold_ids=folds,candidate=pred,motion_baseline=baseline)
    lines=['# 中等类别加权候选：death-aware train 整局 OOF','',
      '只用训练集 363 场；5 折按完整比赛分组，红蓝视角同折。候选与 motion+zone/score 基线使用相同输入、相同 LightGBM 配置和相同折；每折每时距只按训练标签频数赋予固定的逆平方根频率权重，权重均值归一为 1。没有访问 validation/test，也未搜索权重。数值按比赛等权、每场红蓝等权；置信区间对比赛成对重采样。','',
      '| 时距 | 指标 | 加权候选 | motion基线 | 差值及95%比赛CI |','|---:|---|---:|---:|---:|']
    for h in HORIZONS:
        r=results[f'{h}s']
        for metric in ('accuracy','macro_f1','balanced_accuracy','brier','log_loss'):
            c=r['game_equal_red_blue_equal']['candidate'][metric]; b=r['game_equal_red_blue_equal']['motion_baseline'][metric]; q=r['paired_game_bootstrap']['delta_candidate_minus_motion'][metric];
            lines.append(f'| {h}s | {metric} | {c:.4f} | {b:.4f} | {q["mean"]:+.4f} [{q["ci95"][0]:+.4f}, {q["ci95"][1]:+.4f}] |')
    lines += ['', '解释：候选属于一次预先固定强度的类别平衡尝试，结果仅是开发训练集的交叉拟合证据；若准确率/损失受损或CI跨零，不建议替换生产默认。即便 train OOF 有正收益，也需未来单独锁定的评估确认。','',
      '复现：`python eval/experiment_death_aware_moderate_class_weight_oof.py`。']
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.json'),'doc':str(DOC),'runtime_seconds':report['runtime_seconds']},ensure_ascii=False))

if __name__=='__main__': main()
