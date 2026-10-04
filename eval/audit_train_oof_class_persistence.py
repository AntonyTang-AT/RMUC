"""Per-current-class future mode persistence from existing train-only OOF labels.

Read-only diagnostic: no fitting, no validation/test, no source-data rescan.
Each transition row is computed inside each match-side, averaged red/blue per
match, then matches receive equal weight. Paired bootstrap resamples matches.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/motion_candidate_oof.npz'
OUT=ROOT/'out_m2/death_aware_global_mode_experiments/class_persistence_oof'
DOC=ROOT/'docs/analysis/2026-09-27-train-oof-class-persistence.md'
CLASSES=('attack','balanced','defense'); HORIZONS=(10,30,60,120); SEED=1927; REPS=2000

def main():
    with np.load(SRC,allow_pickle=False) as z:
        y=np.asarray(z['mode_y'],np.int64); cur=np.asarray(z['base_mode'],np.int64)
        gid=np.asarray(z['game_ids'],np.int64); side=np.asarray(z['sides']).astype(str)
    if y.shape!=(len(gid),4): raise ValueError(f'unexpected label shape {y.shape}')
    results={}
    rng=np.random.default_rng(SEED)
    for j,h in enumerate(HORIZONS):
        valid=(y[:,j]>=0)&(cur>=0)&(cur<3)
        games=np.unique(gid[valid]); matrix_by_game={}; supports={}
        for g in games:
            side_mats=[]
            side_sup=[]
            for s in ('红','蓝'):
                gm=(gid==g)&(side==s)&valid
                if not gm.any(): continue
                cm=np.zeros((3,3),np.float64); ns=np.zeros(3,np.int64)
                for c in range(3):
                    m=gm&(cur==c); ns[c]=int(m.sum())
                    if ns[c]:
                        for d in range(3): cm[c,d]=np.mean(y[m,j]==d)
                side_mats.append(cm); side_sup.append(ns)
            if len(side_mats)==2:
                # A match-side with no observations in a current class must
                # not contribute a zero transition row. Average only sides
                # where that current class was actually observed.
                game_matrix=np.full((3,3),np.nan,np.float64)
                for c in range(3):
                    active=[side_mats[k][c] for k in range(2) if side_sup[k][c]>0]
                    if active: game_matrix[c]=np.mean(active,axis=0)
                matrix_by_game[int(g)]=game_matrix
                supports[int(g)]=np.sum(side_sup,axis=0)
        ids=sorted(matrix_by_game)
        mats=np.stack([matrix_by_game[g] for g in ids])
        row={}
        for c,name in enumerate(CLASSES):
            used=np.isfinite(mats[:,c,0]); class_mats=mats[used,c,:]
            class_ids=np.asarray(ids)[used]; mean=class_mats.mean(axis=0)
            boot_idx=rng.integers(0,len(class_ids),size=(REPS,len(class_ids)))
            boot=class_mats[boot_idx].mean(axis=1)
            row[name]={}
            for d,dname in enumerate(CLASSES):
                vals=boot[:,d]
                row[name][dname]={'rate':float(mean[d]),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
            pooled=int(np.sum([supports[g][c] for g in ids]))
            row[name]['current_sample_count_pooled']=pooled
            row[name]['games_with_current_class']=int(used.sum())
            row[name]['retention']=row[name][name]
        results[f'{h}s']={'scorable_rows':int(valid.sum()),'matches':len(ids),'transition_rates_by_current_class':row}
    report={'schema':'train-only-oof-class-persistence-v1','source':str(SRC.relative_to(ROOT)),
      'opened_splits':['train OOF labels only'],'validation_access':False,'test_access':False,
      'training_or_tuning':False,'grouping':'existing 5-fold OOF labels; transitions averaged within each match-side, red/blue equal within match, games equal',
      'bootstrap':'2,000 paired match resamples; 95% percentile intervals; each class/horizon resampled with fixed seed',
      'interpretation':'P(future class | current class), not model performance. Rates are match-equal. Pooled sample counts are descriptive.',
      'class_order':list(CLASSES),'horizons':results}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Train OOF 分类别持续率与转移矩阵','',
      '仅读取已生成的 death-aware train OOF 标签和当前状态类别，不拟合模型、不访问 validation/test。每场先分别计算红蓝的条件转移比例，再红蓝平均；最后比赛等权。95%区间按比赛重采样 2,000 次。样本数是所有可评分行的描述性计数。','',
      '表格单元为 P(未来类别 | 当前类别)，例如当前攻势行的攻势列就是“攻势保持率”。','']
    for h in HORIZONS:
        lines += [f'## {h}s', '', '| 当前状态 | 未来攻势 | 未来均衡 | 未来防守 | 当前类样本数 |','|---|---:|---:|---:|---:|']
        t=results[f'{h}s']['transition_rates_by_current_class']
        for c in CLASSES:
            cells=[f'{100*t[c][d]["rate"]:.1f}% [{100*t[c][d]["ci95"][0]:.1f}, {100*t[c][d]["ci95"][1]:.1f}]' for d in CLASSES]
            lines.append(f'| {c} | {cells[0]} | {cells[1]} | {cells[2]} | {t[c]["current_sample_count_pooled"]:,} |')
        lines.append('')
    lines += ['## 解释限制','',
      '该统计衡量标签本身的类条件持续性，不是模型预测召回，也不表示进入某状态后一定会延续。已有模型能在总体准确率上高于持久性基线，但模型 Macro-F1 与 Balanced Accuracy 并不随之同步提升；请与 horizon-decay OOF 报告结合阅读。','',
      '复现：`python eval/audit_train_oof_class_persistence.py`。']
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.json'),'document':str(DOC),'horizons':{h:v['transition_rates_by_current_class'] for h,v in results.items()}},ensure_ascii=False))

if __name__=='__main__': main()
