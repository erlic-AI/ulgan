from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from ulgan import ULGAN, TASKS


def label_for(bits, task):
    if task == 0: return int(sum(bits) % 2)
    if task == 1: return int(sum(bits) > len(bits) / 2)
    return int(bits[-1])


def make_batch(batch_size, min_len, max_len, device):
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)
    tasks = torch.randint(0, 3, (batch_size,), device=device)
    tokens = torch.full((batch_size, int(lengths.max().item()) + 1), 2, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, device=device)
    for i, n in enumerate(lengths.tolist()):
        bits = torch.randint(0, 2, (n,), device=device)
        tokens[i, :n] = bits
        targets[i] = label_for(bits.tolist(), int(tasks[i]))
    return tokens, lengths, tasks, targets


def train_epoch(model, optimizer, steps, batch_size, max_len, device, ponder_weight):
    model.train(); losses=[]; accs=[]; used=[]
    for _ in range(steps):
        tokens, lengths, tasks, targets = make_batch(batch_size, 1, max_len, device)
        logits, halt_logits = model(tokens, lengths, tasks)
        # Deep supervision makes every possible hard-stop output useful at inference.
        per_step = F.binary_cross_entropy_with_logits(logits, targets[:, None].expand_as(logits), reduction='none')
        step_weights = torch.linspace(1.0, 0.55, model.max_think_steps, device=device)
        cls = (per_step * step_weights[None, :]).mean()
        # Encourage early stopping only after every step has learned the target.
        p = torch.sigmoid(halt_logits)
        survival = torch.cumprod(torch.cat([torch.ones_like(p[:, :1]), 1-p[:, :-1]], dim=1), dim=1)
        halt_mass = survival * p
        halt_mass = torch.cat([halt_mass[:, :-1], survival[:, -1:]], dim=1)
        expected_steps = (halt_mass * torch.arange(1, model.max_think_steps+1, device=device)[None, :]).sum(1)
        loss = cls + ponder_weight * expected_steps.mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        accs.append(((logits[:, -1] >= 0) == targets.bool()).float().mean().item())
        losses.append(loss.item()); used.append(expected_steps.mean().item())
    return {'loss':float(np.mean(losses)), 'fixed8_training_accuracy':float(np.mean(accs)), 'expected_steps':float(np.mean(used))}


@torch.no_grad()
def make_eval(n, lo, hi, task, device):
    lengths=torch.randint(lo,hi+1,(n,),device=device)
    tokens=torch.full((n,hi+1),2,dtype=torch.long,device=device); targets=torch.zeros(n,device=device)
    for i, ln in enumerate(lengths.tolist()):
        bits=torch.randint(0,2,(ln,),device=device); tokens[i,:ln]=bits
        targets[i]=label_for(bits.tolist(),task)
    return tokens,lengths,torch.full((n,),task,dtype=torch.long,device=device),targets


@torch.no_grad()
def evaluate(model,n,lo,hi,device,threshold=.95):
    model.eval(); out={};
    for name,task in TASKS.items():
        tokens,lengths,tasks,targets=make_eval(n,lo,hi,task,device)
        h=model.encode(tokens,lengths,tasks); te=model.task_embedding(tasks)
        stopped=torch.zeros(n,dtype=torch.bool,device=device); used=torch.zeros(n,device=device)
        pred=torch.zeros(n,device=device); cumulative=torch.zeros(n,device=device)
        for step in range(model.max_think_steps):
            h,logits,halt_logit=model.think_step(h,te,tasks)
            active=~stopped; used += active.float(); pred=torch.where(active,logits,pred)
            p=torch.sigmoid(halt_logit); cumulative=1-(1-cumulative)*(1-p)
            stopped |= cumulative >= threshold
            if bool(stopped.all()): break
        out[name]={'n':n,'accuracy':float(((pred>=0)==targets.bool()).float().mean()),'mean_think_steps':float(used.mean())}
    out['macro_accuracy']=float(np.mean([out[k]['accuracy'] for k in TASKS]))
    out['macro_mean_think_steps']=float(np.mean([out[k]['mean_think_steps'] for k in TASKS]))
    return out


@torch.no_grad()
def evaluate_fixed(model,n,lo,hi,device):
    model.eval(); out={}
    for name,task in TASKS.items():
        tokens,lengths,tasks,targets=make_eval(n,lo,hi,task,device)
        h=model.encode(tokens,lengths,tasks); te=model.task_embedding(tasks)
        for _ in range(model.max_think_steps): h,logits,_=model.think_step(h,te,tasks)
        out[name]={'n':n,'accuracy':float(((logits>=0)==targets.bool()).float().mean()),'think_steps':model.max_think_steps}
    out['macro_accuracy']=float(np.mean([out[k]['accuracy'] for k in TASKS]))
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=24); ap.add_argument('--steps-per-epoch',type=int,default=100)
    ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--ponder-weight',type=float,default=.001); ap.add_argument('--eval-samples',type=int,default=1000)
    ap.add_argument('--train-max-length',type=int,default=64); ap.add_argument('--out',default='artifacts'); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ULGAN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1.2e-3,weight_decay=1e-5)
    history=[]; started=time.time()
    for epoch in range(1,args.epochs+1):
        m=train_epoch(model,opt,args.steps_per_epoch,args.batch_size,args.train_max_length,device,args.ponder_weight); history.append(m)
        if epoch==1 or epoch%4==0 or epoch==args.epochs:
            print(f"epoch={epoch:02d}/{args.epochs} loss={m['loss']:.4f} fixed8_acc={m['fixed8_training_accuracy']:.4f} expected_steps={m['expected_steps']:.2f}",flush=True)
    os.makedirs(args.out,exist_ok=True); model.cpu()
    config={'hidden_size':64,'embedding_size':16,'task_embedding_size':8,'max_think_steps':8}
    torch.save({'state_dict':model.state_dict(),'config':config,'tasks':TASKS,'seed':args.seed,'version':'0.4','ponder_weight':args.ponder_weight},os.path.join(args.out,'ulgan_v0.4.pt'))
    model.to(device)
    report={'version':'0.4','seed':args.seed,'device':device,'parameters':sum(p.numel() for p in model.parameters()),'training_examples_presented':args.epochs*args.steps_per_epoch*args.batch_size,'train_max_length':args.train_max_length,'ponder_weight':args.ponder_weight,'training_seconds':time.time()-started,'history':history,
      'adaptive_seen_1_64':evaluate(model,args.eval_samples,1,64,device),'fixed8_seen_1_64':evaluate_fixed(model,args.eval_samples,1,64,device),
      'adaptive_unseen_65_128':evaluate(model,args.eval_samples,65,128,device),'fixed8_unseen_65_128':evaluate_fixed(model,args.eval_samples,65,128,device)}
    with open(os.path.join(args.out,'metrics.json'),'w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
