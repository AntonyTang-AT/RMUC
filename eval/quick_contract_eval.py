"""Fast batch evaluator for the frozen forecast contract.

It reports core model-vs-simple-baseline metrics without building the large
per-row JSON ledger.  The full ledger can still be produced separately.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from forecast.train import load_model
from forecast.temporal import HORIZONS, MODES, TRENDS, REGION_LABELS, load_compact
from forecast.contract import contract_metadata


def macro_f1(y, p, n):
    vals=[]
    for c in range(n):
        tp=np.sum((y==c)&(p==c)); fp=np.sum((y!=c)&(p==c)); fn=np.sum((y==c)&(p!=c))
        d=2*tp+fp+fn; vals.append(float(2*tp/d) if d else 0.0)
    return float(np.mean(vals))

def score_cat(y, p, n):
    if len(y)==0: return {"n":0,"accuracy":None,"macro_f1":None,"balanced_accuracy":None}
    rec=[]
    for c in range(n):
        den=np.sum(y==c); rec.append(float(np.sum((y==c)&(p==c))/den) if den else 0.0)
    return {"n":int(len(y)),"accuracy":float(np.mean(y==p)),"macro_f1":macro_f1(y,p,n),"balanced_accuracy":float(np.mean(rec))}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--test-npz',required=True); ap.add_argument('--out',required=True); ap.add_argument('--batch-size',type=int,default=1024); ap.add_argument('--device',default='auto'); a=ap.parse_args(argv)
    dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    model, ckpt=load_model(a.model, device=dev); model.eval(); data=load_compact(a.test_npz); x=data['x']; n=len(x)
    pred={h:{} for h in HORIZONS}; ent={h:{'state':[],'zone':[],'canonical':[]} for h in HORIZONS}
    with torch.no_grad():
        for s in range(0,n,a.batch_size):
            out=model(torch.from_numpy(x[s:s+a.batch_size]).to(dev))
            for h in HORIZONS:
                for task in ('global_mode','opponent_trend','enemy_region'):
                    pred[h].setdefault(task,[]).append(out[h][task].argmax(-1).detach().cpu().numpy())
                ent[h]['state'].append(out[h]['entity_state'].detach().cpu().numpy())
                ent[h]['zone'].append(out[h]['entity_zone_state'].argmax(-1).detach().cpu().numpy())
                ent[h]['canonical'].append(out[h]['entity_canonical_zone_state'].argmax(-1).detach().cpu().numpy())
    for h in HORIZONS:
        for k in pred[h]: pred[h][k]=np.concatenate(pred[h][k])
        for k in ent[h]: ent[h][k]=np.concatenate(ent[h][k])
    result={'contract':contract_metadata(),'device':str(dev),'model':ckpt.get('model_type'), 'loss_profile':ckpt.get('loss_profile'),'horizons':{}}
    widths=20 if x.shape[-1]==329 else 14
    last=x[:,-1,:]
    for hi,h in enumerate(HORIZONS):
        row={'global':{},'entity':{}}
        for task,key,classes,base_key in [('global_mode','mode_y',len(MODES),'base_mode'),('opponent_trend','trend_y',len(TRENDS),None),('enemy_region','region_y',len(REGION_LABELS),'base_region')]:
            y=data[key][:,hi]; valid=y>=0; yy=y[valid]; pp=pred[h][task][valid]; base=None
            if base_key: base=data[base_key][valid]
            elif task=='opponent_trend': base=np.full(len(yy),1,dtype=np.int64)
            row['global'][task]={'model':score_cat(yy,pp,classes),'persistence':score_cat(yy,base,classes)}
        st=ent[h]['state']; yt=data['entity_y'][:,hi]; valid=yt[:,:,0]>=0; present=valid&(yt[:,:,0]>.5)
        pred_presence=(1/(1+np.exp(-st[:,:,0])))>=.5; base_presence=last[:,0::widths][:,:yt.shape[1]]*0+0 # overwritten below
        base_presence=np.stack([last[:,i*widths+3]>.5 for i in range(yt.shape[1])],axis=1)
        yp=yt[:,:,0][valid]>.5; mp=pred_presence[valid]; bp=base_presence[valid]
        pos_pred=1/(1+np.exp(-st[:,:,1:3])); hp_pred=1/(1+np.exp(-st[:,:,3]))
        y_pos=yt[:,:,1:3][present]; p_pos=pos_pred[present]; b_pos=np.stack([last[:,i*widths:i*widths+2] for i in range(yt.shape[1])],axis=1)[present]
        y_hp=yt[:,:,3][present]; p_hp=hp_pred[present]; b_hp=np.stack([last[:,i*widths+2] for i in range(yt.shape[1])],axis=1)[present]
        zone_y=data['entity_zone_y'][:,hi]; zv=zone_y>=0; zp=ent[h]['zone'][zv]
        # The current functional zone is recovered from the stored target-side
        # feature geometry by the same runtime utility in the full evaluator;
        # here we report model zone accuracy and coverage only.
        erow={'presence':{'model_accuracy':float(np.mean(mp==yp)) if len(yp) else None,'persistence_accuracy':float(np.mean(bp==yp)) if len(yp) else None,'n':int(len(yp))},
              'position':{'model_mae_normalized':float(np.mean(np.abs(p_pos-y_pos))) if len(y_pos) else None,'persistence_mae_normalized':float(np.mean(np.abs(b_pos-y_pos))) if len(y_pos) else None,'n':int(len(y_pos))},
              'hp':{'model_mae':float(np.mean(np.abs(p_hp-y_hp))) if len(y_hp) else None,'persistence_mae':float(np.mean(np.abs(b_hp-y_hp))) if len(y_hp) else None,'n':int(len(y_hp))},
              'functional_zone':{'model_accuracy':float(np.mean(zp==zone_y[zv])) if np.any(zv) else None,'coverage':float(np.mean(zv))}}
        row['entity']=erow; result['horizons'][str(h)]=row
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
