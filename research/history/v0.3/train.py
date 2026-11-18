import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from model import ULGAN, TASKS


def label_for(bits, task):
    if task == 0: return int(sum(bits) % 2)
    if task == 1: return int(sum(bits) > len(bits) / 2)
    return int(bits[-1])


def make_batch(batch_size, min_len, max_len, device):
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)
    tasks = torch.randint(0, 3, (batch_size,), device=device)
    T = int(lengths.max().item()) + 1
    tokens = torch.full((batch_size, T), 2, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, device=device)
    for i, n in enumerate(lengths.tolist()):
        bits = torch.randint(0, 2, (n,), device=device)
        tokens[i, :n] = bits
        targets[i] = label_for(bits.tolist(), int(tasks[i]))
    return tokens, lengths, tasks, targets


def train_epoch(model, optimizer, steps, batch_size, device, ponder_weight):
    model.train(); losses=[]; accs=[]; ponder=[]
    for _ in range(steps):
        tokens, lengths, tasks, targets = make_batch(batch_size, 1, 16, device)
        logits, expected_steps, _, _ = model(tokens, lengths, tasks)
        per_item = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        task_weights = torch.where(tasks == 0, 2.5, 1.0)
        cls = (per_item * task_weights).mean()
        # ACT ponder cost: encourage the controller to stop using fewer latent steps.
        loss = cls + ponder_weight * expected_steps.mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(loss.item()); accs.append(((logits >= 0) == targets.bool()).float().mean().item()); ponder.append(expected_steps.mean().item())
    return {'loss':float(np.mean(losses)), 'accuracy':float(np.mean(accs)), 'expected_think_steps':float(np.mean(ponder))}


@torch.no_grad()
def make_eval_batch(n, min_len, max_len, task_id, device):
    lengths = torch.randint(min_len, max_len + 1, (n,), device=device)
    tokens = torch.full((n, max_len + 1), 2, dtype=torch.long, device=device)
    targets = torch.zeros(n, device=device)
    for i, ln in enumerate(lengths.tolist()):
        bits = torch.randint(0, 2, (ln,), device=device)
        tokens[i, :ln] = bits
        targets[i] = label_for(bits.tolist(), task_id)
    tasks = torch.full((n,), task_id, dtype=torch.long, device=device)
    return tokens, lengths, tasks, targets


@torch.no_grad()
def evaluate(model, n, min_len, max_len, device, threshold=.95):
    model.eval(); results={}
    for name, task_id in TASKS.items():
        tokens, lengths, tasks, targets = make_eval_batch(n,min_len,max_len,task_id,device)
        h=model.encode(tokens,lengths,tasks); te=model.task_embedding(tasks)
        cumulative=torch.zeros(n,device=device); stopped=torch.zeros(n,dtype=torch.bool,device=device)
        pred=torch.zeros(n,device=device); used=torch.zeros(n,device=device)
        for step in range(model.max_think_steps):
            h,logits,halt_logits=model.think_step(h,te,tasks)
            p=torch.sigmoid(halt_logits)
            cumulative=1-(1-cumulative)*(1-p)
            active=~stopped
            used += active.float()
            pred=torch.where(active,logits,pred)
            stopped |= cumulative >= threshold
            if bool(stopped.all()): break
        accuracy=((pred>=0)==targets.bool()).float().mean().item()
        results[name]={'n':n,'accuracy':accuracy,'mean_think_steps':used.mean().item(),'mean_cumulative_halt_probability':cumulative.mean().item()}
    results['macro_accuracy']=float(np.mean([results[k]['accuracy'] for k in TASKS]))
    results['macro_mean_think_steps']=float(np.mean([results[k]['mean_think_steps'] for k in TASKS]))
    return results


@torch.no_grad()
def fixed_step_evaluate(model, n, min_len, max_len, device, fixed_steps=8):
    model.eval(); result={}
    for name, task_id in TASKS.items():
        tokens,lengths,tasks,targets=make_eval_batch(n,min_len,max_len,task_id,device)
        h=model.encode(tokens,lengths,tasks); te=model.task_embedding(tasks)
        for _ in range(fixed_steps): h,logits,_=model.think_step(h,te,tasks)
        result[name]={'n':n,'accuracy':((logits>=0)==targets.bool()).float().mean().item(),'fixed_think_steps':fixed_steps}
    result['macro_accuracy']=float(np.mean([result[k]['accuracy'] for k in TASKS]))
    result['macro_mean_think_steps']=float(fixed_steps)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--steps-per-epoch',type=int,default=80); ap.add_argument('--batch-size',type=int,default=96); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--ponder-weight',type=float,default=.015); ap.add_argument('--eval-samples',type=int,default=300); ap.add_argument('--out',default='artifacts'); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ULGAN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1.5e-3,weight_decay=1e-5)
    history=[]; started=time.time()
    for epoch in range(1,args.epochs+1):
        m=train_epoch(model,opt,args.steps_per_epoch,args.batch_size,device,args.ponder_weight); history.append(m)
        print(f"epoch={epoch:02d} loss={m['loss']:.4f} accuracy={m['accuracy']:.4f} expected_steps={m['expected_think_steps']:.3f}",flush=True)
    os.makedirs(args.out,exist_ok=True); model.cpu()
    ckpt={'state_dict':model.state_dict(),'config':{'hidden_size':48,'embedding_size':16,'task_embedding_size':8,'max_think_steps':8},'tasks':TASKS,'seed':args.seed,'version':'0.3','ponder_weight':args.ponder_weight}
    torch.save(ckpt,os.path.join(args.out,'ulgan_v0.3.pt'))
    # Fixed and adaptive evaluations use identical test ranges and sample counts.
    model.to(device)
    seen_adapt=evaluate(model,args.eval_samples,1,16,device); unseen_adapt=evaluate(model,args.eval_samples,17,32,device)
    seen_fixed=fixed_step_evaluate(model,args.eval_samples,1,16,device); unseen_fixed=fixed_step_evaluate(model,args.eval_samples,17,32,device)
    report={'version':'0.3','seed':args.seed,'device':device,'parameters':sum(p.numel() for p in model.parameters()),'training_examples_presented':args.epochs*args.steps_per_epoch*args.batch_size,'ponder_weight':args.ponder_weight,'training_seconds':time.time()-started,'history':history,'adaptive_test_seen_lengths':seen_adapt,'adaptive_test_unseen_lengths':unseen_adapt,'fixed_8_step_test_seen_lengths':seen_fixed,'fixed_8_step_test_unseen_lengths':unseen_fixed}
    with open(os.path.join(args.out,'metrics.json'),'w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
