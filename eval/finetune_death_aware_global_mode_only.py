"""Warm-start global-mode-only fine-tuning, train/validation NPZ only."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from forecast.temporal import HORIZONS, MODES, load_compact
from forecast.train import load_model

def metrics(y,p):
    pred=p.argmax(1); f=[]
    for c in range(3):
        tp=np.sum((y==c)&(pred==c));fp=np.sum((y!=c)&(pred==c));fn=np.sum((y==c)&(pred!=c))
        f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(pred==y)),float(np.mean(f))

def evaluate(model,data,device,batch=512):
    model.eval(); probs={h:[] for h in HORIZONS}
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(torch.from_numpy(data['x'])),batch_size=batch):
            out=model(x.to(device))
            for h in HORIZONS: probs[h].append(torch.softmax(out[h]['global_mode'],-1).cpu().numpy())
    return {h:np.concatenate(probs[h]) for h in HORIZONS}

def run(a):
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    tr=load_compact(a.train);va=load_compact(a.validation)
    if str(tr['global_mode_label_version'][0])!=str(va['global_mode_label_version'][0]): raise ValueError('contract mismatch')
    model,ckpt=load_model(a.initial_checkpoint,device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    ds=TensorDataset(torch.from_numpy(tr['x']),torch.from_numpy(tr['mode_y'].astype(np.int64)))
    loader=DataLoader(ds,batch_size=a.batch,shuffle=True,pin_memory=device.type=='cuda')
    vy=va['mode_y'].astype(int);vvalid=vy>=0;vside=np.asarray(va['sides']);vbase=va['base_mode'].astype(int)
    best_score=-1;best_state=None;history=[];patience=3;stale=0
    for epoch in range(1,a.epochs+1):
        model.train();loss_sum=0.;count=0
        for x,y in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            out=model(x); losses=[]
            for j,h in enumerate(HORIZONS):
                ok=y[:,j]>=0
                if ok.any(): losses.append(torch.nn.functional.cross_entropy(out[h]['global_mode'][ok],y[ok,j]))
            loss=sum(losses)/len(losses)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            loss_sum+=float(loss.detach())*len(x);count+=len(x)
        p=evaluate(model,va,device,a.batch)
        horizon={}; side_score=[]
        for j,h in enumerate(HORIZONS):
            good=vvalid[:,j]; sidescores=[]
            for side in ('红','蓝'):
                m=good&(vside==side);acc,f1=metrics(vy[m,j],p[h][m]);sidescores.append((acc,f1));side_score.append(.6*acc+.4*f1)
            horizon[f'{h}s']={'accuracy':float(np.mean([z[0] for z in sidescores])), 'macro_f1':float(np.mean([z[1] for z in sidescores]))}
        score=float(np.mean(side_score));history.append({'epoch':epoch,'train_loss':loss_sum/max(count,1),'val_equal_side_score':score,'horizons':horizon})
        print(json.dumps(history[-1]),flush=True)
        if score>best_score+1e-5:
            best_score=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            Path(a.output).parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':best_state,'model_type':ckpt.get('model_type'),'architecture':ckpt.get('architecture'),
                'input_dim':ckpt['input_dim'],'hidden':ckpt['hidden'],'layers':ckpt['layers'],
                'entity_slots':ckpt.get('entity_slots',0),'zone_classes':ckpt.get('zone_classes',0),
                'canonical_zone_classes':ckpt.get('canonical_zone_classes',0),'loss_profile':'global_mode_only_finetune',
                'warm_start':str(a.initial_checkpoint),'seed':a.seed,'selected_epoch':epoch},a.output)
        else:
            stale+=1
            if stale>=patience: break
    Path(a.history).parent.mkdir(parents=True,exist_ok=True)
    Path(a.history).write_text(json.dumps({'history':history,'selected_epoch':int(torch.load(a.output,map_location='cpu',weights_only=False)['selected_epoch']),
        'device':str(device),'objective':'mean unweighted cross entropy over available global_mode horizon labels; validation early-stop on equal-side mean 0.6 accuracy + 0.4 macro-F1',
        'validation_is_selection_data':True,'test_access':False},indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--initial-checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--history',required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch',type=int,default=256);p.add_argument('--lr',type=float,default=0.0002);p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=83);run(p.parse_args())
if __name__=='__main__':main()
