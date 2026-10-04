from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(r'F:\Desktop\学校\rm考核项目')
OOF=ROOT/'out_m2/death_aware_global_mode_experiments/gru_lgbm_50_50_oof/oof_probabilities.npz'
TRAIN=ROOT/'out_m2/temporal_forecast_death_aware_candidate/train.npz'
OUT=ROOT/'out_m2/death_aware_global_mode_experiments/gru_lgbm_50_50_oof/error_decomposition.json'
DOC=ROOT/'docs/analysis/2026-09-27-death-aware-oof-error-decomposition.md'
HORIZONS=[10,30,60,120]; CLASSES=['进攻','均衡','防守']; SIDES=['红','蓝']

def safe_div(a,b): return float(a/b) if b else None

def reconstruct_current_score(x,sides):
    # Compact schema: 8 entity slots/team, 14 values/slot; indices 0=x,2=HP ratio,3=has-position.
    z=x[:,-1,:-8].reshape(len(x),16,14)
    score=np.zeros(len(x),dtype=float); visible=np.zeros(len(x),dtype=int)
    for i,side in enumerate(sides):
        own=z[i,:8]; opp=z[i,8:]
        visible[i]=int(np.rint(opp[:,3].sum()))
        if side=='蓝':
            own_push=np.sum((own[:6,3]>.5)&(own[:6,0]<.5))
            opp_push=np.sum((opp[:6,3]>.5)&(opp[:6,0]>=.5))
        else:
            own_push=np.sum((own[:6,3]>.5)&(own[:6,0]>.5))
            opp_push=np.sum((opp[:6,3]>.5)&(opp[:6,0]<=.5))
        own_base=float(own[7,2]) if own[7,3]>.5 else 1.0
        opp_base=float(opp[7,2]) if opp[7,3]>.5 else 1.0
        score[i]=(own_push-opp_push)+2.0*(opp_base-own_base)
    return score,visible

def game_ci(y,p,gids,sides,mask,reps=1200,seed=731):
    ids,inv=np.unique(gids[mask],return_inverse=True); yy=y[mask]; pp=p[mask].argmax(1); ss=sides[mask]
    scores=[]
    for side in SIDES:
        sm=ss==side; cnt=np.bincount(inv[sm],minlength=len(ids)); cor=np.bincount(inv[sm],weights=(pp[sm]==yy[sm]).astype(float),minlength=len(ids))
        scores.append(np.divide(cor,cnt,out=np.zeros(len(ids),float),where=cnt>0))
    nside=np.stack([np.bincount(inv[ss==side],minlength=len(ids))>0 for side in SIDES]).sum(0)
    arr=((scores[0]+scores[1])/np.maximum(nside,1))[nside>0]
    if not len(arr): return {'game_n':0,'bootstrap_replicates':int(reps),'mean':None,'ci95':[None,None]}
    rng=np.random.default_rng(seed); b=arr[rng.integers(0,len(arr),size=(reps,len(arr)))].mean(1)
    return {'game_n':int(len(arr)),'bootstrap_replicates':int(reps),'mean':float(arr.mean()),'ci95':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
def mode_metrics(y,p,gids,sides,mask,seed):
    yy=y[mask]; pr=p[mask].argmax(1)
    conf=np.zeros((3,3),dtype=np.int64); np.add.at(conf,(yy,pr),1)
    recalls=[]
    for c in range(3): recalls.append(safe_div(int(conf[c,c]),int(conf[c].sum())))
    return {'n':int(mask.sum()),'accuracy':safe_div(int(np.trace(conf)),int(conf.sum())),
      'per_class_recall':{CLASSES[c]:recalls[c] for c in range(3)},'confusion_true_rows_pred_cols':conf.tolist(),
      'game_equal_red_blue_equal_accuracy_ci':game_ci(y,p,gids,sides,mask,seed=seed)}

def main():
    a=np.load(OOF,allow_pickle=False); d=np.load(TRAIN,allow_pickle=False)
    if not np.array_equal(a['game_ids'],d['game_ids']) or not np.array_equal(a['mode_y'],d['mode_y']) or not np.array_equal(a['issued_at'],d['issued_at']):
        raise ValueError('OOF and train rows are not identically aligned')
    n=len(a['game_ids']); gids=a['game_ids']; sides=a['sides']; y=a['mode_y']
    score,visible=reconstruct_current_score(d['x'],sides)
    rem=np.maximum(0,420.-a['issued_at'])
    bins={
      'current_mode':[(str(c),a['base_mode']==i) for i,c in enumerate(CLASSES)],
      'current_score_distance_to_±1_proxy':[
       ('≤0.25',np.minimum(abs(score-1),abs(score+1))<=.25),
       ('0.25–0.75',(np.minimum(abs(score-1),abs(score+1))>.25)&(np.minimum(abs(score-1),abs(score+1))<=.75)),
       ('0.75–1.5',(np.minimum(abs(score-1),abs(score+1))>.75)&(np.minimum(abs(score-1),abs(score+1))<=1.5)),
       ('>1.5',np.minimum(abs(score-1),abs(score+1))>1.5)],
      'remaining_time_seconds':[
       ('0–60',(rem>=0)&(rem<60)),('60–120',(rem>=60)&(rem<120)),('120–240',(rem>=120)&(rem<240)),
       ('240–360',(rem>=240)&(rem<360)),('360–420',rem>=360)],
      'visible_opponent_type_slots_proxy':[(f'{i}',visible==i) for i in range(0,7)]+[('8',visible>=8)]
    }
    report={'schema':'death-aware-oof-error-decomposition-v1','oof_path':str(OOF),'train_path':str(TRAIN),
      'test_access':False,'validation_access':False,'training_performed':False,'rows':n,'games':int(len(np.unique(gids))),
      'label_contract':str(d['global_mode_label_version'][0]),'methods':['blend','lightgbm'],
      'important_proxies':{
       'current_score':'Recomputed from causal compact x current snapshot using at most six selected mobile type slots per team and base HP ratio: (own push slots - opponent push slots)+2*(opponent base HP ratio-own base HP ratio). It approximates temporal._mode_score but loses multiplicity because compact data retains representative slots.',
       'enemy_visibility':'Count of the 8 opponent entity-type slots with a position in current compact snapshot. This is visible type-slot coverage, NOT robot count; do not interpret as exact enemies seen.',
       'remaining_time':'420 seconds minus issued_at, clipped at zero; source-game duration may vary and the exported training rows may censor horizons.'},'horizons':{}}
    for j,h in enumerate(HORIZONS):
      vm=a['valid'][:,j]&(y[:,j]>=0)
      section={'valid_rows':int(vm.sum()),'overall':{},'strata':{}}
      for k,name in enumerate(('blend','lightgbm')):
        section['overall'][name]=mode_metrics(y[:,j],a[name][:,j],gids,sides,vm,900+j*10+k)
      for axis,groups in bins.items():
        section['strata'][axis]={}
        for ii,(label,bm) in enumerate(groups):
          m=vm&bm
          section['strata'][axis][label]={name:mode_metrics(y[:,j],a[name][:,j],gids,sides,m,1300+j*100+ii*3+mi)
              for mi,name in enumerate(('blend','lightgbm'))}
          section['strata'][axis][label]['red_blue_pooled_accuracy']={side:safe_div(int(np.sum((y[m&(sides==side),j]==a['blend'][m&(sides==side),j].argmax(1)))),int(np.sum(m&(sides==side)))) for side in SIDES}
      report['horizons'][f'{h}s']=section
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Death-aware OOF 错误分解','',f'- 样本：{n:,} 行，{len(np.unique(gids))} 局比赛；标签合同 `{report["label_contract"]}`。','- 仅读取 OOF 概率与 death-aware train.npz；没有训练，也没有打开 validation/test。','- 指标：逐局内先按红蓝分别算准确率，再对有样本的队伍等权、最后以比赛为单位做 1,200 次 bootstrap 95% CI。','- 当前分数和敌方可见度是压缩特征代理，不能当成真实完整机器人数量或原始标签分数。详见 JSON 中的 `important_proxies`。','', '## 整体与分桶准确率','', '| 时距 | 方法 | 全体行准确率 | 比赛等权准确率 [95% CI] |']
    for h,sec in report['horizons'].items():
      for method in ('blend','lightgbm'):
       o=sec['overall'][method]; gm=o['game_equal_red_blue_equal_accuracy_ci']; ci=gm['ci95']; lines.append(f"| {h} | {method} | {o['accuracy']:.3f} | {gm['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    # Add stratified compact tables for blend; coverage and CI make small groups visible.
    for axis,groups in bins.items():
      lines += ['',f'## {axis}（blend）','', '| 时距 | 分桶 | n | 行准确率 | 比赛等权准确率 [95% CI] | 进攻/均衡/防守召回 |','|---|---:|---:|---:|---:|---|']
      for h,sec in report['horizons'].items():
       for label,_ in groups:
        o=sec['strata'][axis][label]['blend']; gm=o['game_equal_red_blue_equal_accuracy_ci']; ci=gm['ci95']; rc=o['per_class_recall']
        gmval=gm['mean'] if gm['mean'] is not None else float('nan'); lo=ci[0] if ci[0] is not None else float('nan'); hi=ci[1] if ci[1] is not None else float('nan')
        lines.append(f"| {h} | {label} | {o['n']} | {o['accuracy'] if o['accuracy'] is not None else float('nan'):.3f} | {gmval:.3f} [{lo:.3f}, {hi:.3f}] | {rc['进攻'] if rc['进攻'] is not None else float('nan'):.2f}/{rc['均衡'] if rc['均衡'] is not None else float('nan'):.2f}/{rc['防守'] if rc['防守'] is not None else float('nan'):.2f} |")
    lines += ['', '## 解释限制','', '分桶结果是诊断相关性，不证明因果。低可见槽可能表示遮挡或状态缺失，也会改变可评估样本构成；剩余时间是回放时间剩余代理。若压缩 schema 对单类兵种只留一个代表，当前局面 push-score 会漏掉其余同类单位。请优先把“游戏等权准确率及区间、分桶覆盖率、类别召回”作为诊断读数。','']
    DOC.parent.mkdir(parents=True,exist_ok=True); DOC.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'json':str(OUT),'markdown':str(DOC),'horizons':{h:{'blend':v['overall']['blend']['accuracy'],'lgbm':v['overall']['lightgbm']['accuracy']} for h,v in report['horizons'].items()}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()


