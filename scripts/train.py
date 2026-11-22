from pathlib import Path
import argparse, json, random, time, sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ulgan import ULGAN, TASKS


def label(bits, task):
    if task == 0: return int(sum(bits) % 2)
    if task == 1: return int(sum(bits) > len(bits) / 2)
    return int(bits[-1])


def batch(bs, lo, hi, device):
    lengths = torch.randint(lo, hi + 1, (bs,), device=device)
    tasks = torch.randint(0, 3, (bs,), device=device)
    bits = torch.randint(0, 2, (bs, hi), device=device)
    mask = torch.arange(hi, device=device)[None, :] < lengths[:, None]
    bits = bits * mask
    tokens = torch.full((bs, hi + 1), 2, dtype=torch.long, device=device)
    tokens[:, :hi] = torch.where(mask, bits, torch.full_like(bits, 2))
    counts = bits.sum(1)
    parity = counts.remainder(2)
    majority = (counts > (lengths.float() / 2)).long()
    last = bits.gather(1, (lengths - 1)[:, None]).squeeze(1)
    targets = torch.where(tasks == 0, parity, torch.where(tasks == 1, majority, last)).float()
    prefix_targets = bits.cumsum(1).remainder(2).float()
    return tokens, lengths, tasks, targets, prefix_targets


def train_epoch(model, opt, steps, bs, max_len, device, ponder, prefix_weight):
    model.train(); losses=[]; accs=[]
    for _ in range(steps):
        tok, lens, tasks, y, py = batch(bs, 1, max_len, device)
        logits, halt_logits, prefix_logits = model(tok, lens, tasks)
        # ACT-style differentiable halting: classification quality is measured
        # under the same halting distribution used at inference.
        p = torch.sigmoid(halt_logits)
        survival = torch.cumprod(torch.cat((torch.ones_like(p[:, :1]), 1-p[:, :-1]), 1), 1)
        mass = survival * p
        # Any probability mass still alive after the final step must use its final prediction.
        mass = torch.cat((mass[:, :-1], survival[:, -1:]), 1)
        per_step = F.binary_cross_entropy_with_logits(
            logits, y[:, None].expand_as(logits), reduction='none')
        cls = (mass * per_step).sum(1).mean()
        expected = (mass * torch.arange(1, model.max_think_steps+1, device=device)).sum(1).mean()
        prefix_loss = torch.zeros((), device=device)
        if prefix_logits is not None:
            mask = torch.arange(py.shape[1], device=device)[None, :] < lens[:, None]
            parity_mask = (tasks == TASKS['parity'])[:, None] & mask
            raw = F.binary_cross_entropy_with_logits(prefix_logits[:, :py.shape[1]], py, reduction='none')
            prefix_loss = (raw * parity_mask).sum() / parity_mask.sum().clamp_min(1)
        loss = cls + ponder * expected + prefix_weight * prefix_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(loss.detach())); accs.append(float(((logits[:, -1] >= 0) == y.bool()).float().mean()))
    return {'loss': float(np.mean(losses)), 'fixed_steps_accuracy': float(np.mean(accs))}


@torch.no_grad()
def evaluate(model, n, lo, hi, device, threshold=.95):
    model.eval(); result={}
    for name, tid in TASKS.items():
        correct=0; total_steps=0
        gen = torch.Generator(device='cpu').manual_seed(910000 + tid * 100003 + lo * 1009 + hi)
        remaining = n
        while remaining > 0:
            bs = min(256, remaining)
            remaining -= bs
            lengths=torch.randint(lo,hi+1,(bs,),generator=gen,device='cpu').to(device)
            bits=torch.randint(0,2,(bs,hi),generator=gen,device='cpu').to(device)
            mask=torch.arange(hi,device=device)[None,:] < lengths[:,None]
            bits=bits*mask
            tokens=torch.full((bs,hi+1),2,dtype=torch.long,device=device)
            tokens[:,:hi]=torch.where(mask,bits,torch.full_like(bits,2))
            counts=bits.sum(1)
            if tid==0: target=counts.remainder(2)
            elif tid==1: target=(counts > lengths.float()/2).long()
            else: target=bits.gather(1,(lengths-1)[:,None]).squeeze(1)
            tasks=torch.full((bs,),tid,dtype=torch.long,device=device)
            h,encoded=model.encode(tokens,lengths,tasks); te=model.task_embedding(tasks)
            stopped=torch.zeros(bs,dtype=torch.bool,device=device); used=torch.zeros(bs,device=device)
            pred=torch.zeros(bs,device=device); survival=torch.ones(bs,device=device)
            for _step in range(model.max_think_steps):
                h,lg,hz=model.think_step(h,te,tasks,encoded,lengths); active=~stopped
                used += active.float(); pred=torch.where(active,lg,pred)
                survival *= (1-torch.sigmoid(hz)); stopped |= (1-survival)>=threshold
                if bool(stopped.all()): break
            correct += int((((pred>=0)==target.bool()).sum()).item())
            total_steps += float(used.sum().item())
        result[name]={'accuracy':correct/n, 'mean_think_steps':total_steps/n}
    result['macro_accuracy']=float(np.mean([result[k]['accuracy'] for k in TASKS]))
    result['macro_mean_think_steps']=float(np.mean([result[k]['mean_think_steps'] for k in TASKS]))
    return result


@torch.no_grad()
def evaluate_fixed(model, n, lo, hi, device):
    model.eval(); result={}
    for name, tid in TASKS.items():
        correct=0; gen = torch.Generator(device='cpu').manual_seed(910000 + tid * 100003 + lo * 1009 + hi); remaining=n
        while remaining:
            bs=min(256,remaining); remaining-=bs
            lengths=torch.randint(lo,hi+1,(bs,),generator=gen,device='cpu').to(device)
            bits=torch.randint(0,2,(bs,hi),generator=gen,device='cpu').to(device)
            mask=torch.arange(hi,device=device)[None,:] < lengths[:,None]
            bits=bits*mask
            tokens=torch.full((bs,hi+1),2,dtype=torch.long,device=device)
            tokens[:,:hi]=torch.where(mask,bits,torch.full_like(bits,2))
            counts=bits.sum(1)
            if tid==0: target=counts.remainder(2)
            elif tid==1: target=(counts > lengths.float()/2).long()
            else: target=bits.gather(1,(lengths-1)[:,None]).squeeze(1)
            tasks=torch.full((bs,),tid,dtype=torch.long,device=device)
            h,encoded=model.encode(tokens,lengths,tasks); te=model.task_embedding(tasks)
            for _ in range(model.max_think_steps): h,lg,_=model.think_step(h,te,tasks,encoded,lengths)
            correct += int((((lg>=0)==target.bool()).sum()).item())
        result[name]={'accuracy':correct/n,'think_steps':model.max_think_steps}
    result['macro_accuracy']=float(np.mean([result[k]['accuracy'] for k in TASKS]))
    return result



@torch.no_grad()
def evaluate_paired(model, n, lo, hi, device, threshold=.95):
    """Evaluate adaptive and fixed-step inference on exactly the same examples."""
    model.eval(); adaptive={}; fixed={}
    for name, tid in TASKS.items():
        correct_ad=0; steps_ad=0.0; correct_fx=0; per_step_correct=torch.zeros(model.max_think_steps,device=device); halt_prob_sum=torch.zeros(model.max_think_steps,device=device); gen = torch.Generator(device='cpu').manual_seed(910000 + tid * 100003 + lo * 1009 + hi); remaining=n
        while remaining:
            bs=min(256,remaining); remaining-=bs
            lengths=torch.randint(lo,hi+1,(bs,),generator=gen,device='cpu').to(device)
            bits=torch.randint(0,2,(bs,hi),generator=gen,device='cpu').to(device)
            mask=torch.arange(hi,device=device)[None,:] < lengths[:,None]
            bits=bits*mask
            tokens=torch.full((bs,hi+1),2,dtype=torch.long,device=device)
            tokens[:,:hi]=torch.where(mask,bits,torch.full_like(bits,2))
            counts=bits.sum(1)
            if tid==0: target=counts.remainder(2)
            elif tid==1: target=(counts > lengths.float()/2).long()
            else: target=bits.gather(1,(lengths-1)[:,None]).squeeze(1)
            tasks=torch.full((bs,),tid,dtype=torch.long,device=device)
            h,encoded=model.encode(tokens,lengths,tasks); te=model.task_embedding(tasks)
            stopped=torch.zeros(bs,dtype=torch.bool,device=device); used=torch.zeros(bs,device=device)
            pred_ad=torch.zeros(bs,device=device); survival=torch.ones(bs,device=device)
            h_fixed=h.clone(); pred_fx=None
            for step in range(model.max_think_steps):
                h,lg,hz=model.think_step(h,te,tasks,encoded,lengths)
                active=~stopped; used += active.float(); pred_ad=torch.where(active,lg,pred_ad)
                survival *= (1-torch.sigmoid(hz)); stopped |= (1-survival)>=threshold
                h_fixed,lg_fixed,_=model.think_step(h_fixed,te,tasks,encoded,lengths)
                pred_fx=lg_fixed
                per_step_correct[step] += ((lg_fixed >= 0) == target.bool()).float().sum()
                halt_prob_sum[step] += torch.sigmoid(hz).sum()
            correct_ad += int((((pred_ad>=0)==target.bool()).sum()).item())
            correct_fx += int((((pred_fx>=0)==target.bool()).sum()).item())
            steps_ad += float(used.sum().item())
        adaptive[name]={'accuracy':correct_ad/n,'mean_think_steps':steps_ad/n}
        fixed[name]={'accuracy':correct_fx/n,'think_steps':model.max_think_steps,
                     'per_step_accuracy':(per_step_correct / n).cpu().tolist()}
        adaptive[name]['mean_halt_probability_by_step']=(halt_prob_sum / n).cpu().tolist()
    adaptive['macro_accuracy']=float(np.mean([adaptive[k]['accuracy'] for k in TASKS]))
    adaptive['macro_mean_think_steps']=float(np.mean([adaptive[k]['mean_think_steps'] for k in TASKS]))
    fixed['macro_accuracy']=float(np.mean([fixed[k]['accuracy'] for k in TASKS]))
    return adaptive, fixed

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,nargs='+',default=[11,23,42]); ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--steps-per-epoch',type=int,default=100); ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--train-max-length',type=int,default=128); ap.add_argument('--eval-samples',type=int,default=2000)
    ap.add_argument('--ponder-weight',type=float,default=.0005); ap.add_argument('--prefix-weight',type=float,default=.25); ap.add_argument('--disable-prefix-supervision',action='store_true'); ap.add_argument('--causal-encoder',action='store_true')
    ap.add_argument('--out',default='artifacts/ulgan_1_attention_act'); args=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'; reports=[]
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
        model=ULGAN(prefix_supervision=not args.disable_prefix_supervision, bidirectional=not args.causal_encoder).to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5); start=time.time(); history=[]
        for ep in range(1,args.epochs+1):
            m=train_epoch(model,opt,args.steps_per_epoch,args.batch_size,args.train_max_length,device,args.ponder_weight,args.prefix_weight); history.append(m)
            if ep==1 or ep%5==0 or ep==args.epochs: print(f'seed={seed} epoch={ep}/{args.epochs} loss={m["loss"]:.4f} acc={m["fixed_steps_accuracy"]:.4f}',flush=True)
        seen_key=f'1_{args.train_max_length}'
        unseen_lo=args.train_max_length+1; unseen_hi=args.train_max_length*2
        unseen_key=f'{unseen_lo}_{unseen_hi}'
        adaptive_seen, fixed_seen = evaluate_paired(model,args.eval_samples,1,args.train_max_length,device)
        adaptive_unseen, fixed_unseen = evaluate_paired(model,args.eval_samples,unseen_lo,unseen_hi,device)
        report={'seed':seed,'device':device,'parameters':sum(p.numel() for p in model.parameters()),'train_examples':args.epochs*args.steps_per_epoch*args.batch_size,'seconds':time.time()-start,'history':history,
                f'adaptive_seen_{seen_key}':adaptive_seen, f'fixed12_seen_{seen_key}':fixed_seen,
                f'adaptive_unseen_{unseen_key}':adaptive_unseen, f'fixed12_unseen_{unseen_key}':fixed_unseen}
        reports.append(report)
        out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':model.cpu().state_dict(),'config':{'hidden_size':96,'embedding_size':24,'task_embedding_size':12,'max_think_steps':12,'prefix_supervision':not args.disable_prefix_supervision,'bidirectional':not args.causal_encoder,'attention_context':True,'halting_objective':'act_weighted_classification'},'seed':seed,'name':'ulgan 1'},out/f'ulgan_1_seed_{seed}.pt')
    # Aggregate across seeds; sample standard deviation is reported (0 for a single seed).
    seen_key=f'1_{args.train_max_length}'
    unseen_key=f'{args.train_max_length+1}_{args.train_max_length*2}'
    metric_paths=[f'adaptive_seen_{seen_key}',f'fixed12_seen_{seen_key}',f'adaptive_unseen_{unseen_key}',f'fixed12_unseen_{unseen_key}']
    aggregate={}
    for path in metric_paths:
        aggregate[path]={}
        keys=['macro_accuracy'] + list(TASKS)
        for key in keys:
            vals=[r[path][key]['accuracy'] if key!='macro_accuracy' else r[path][key] for r in reports]
            aggregate[path][key]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}
        if path.startswith('adaptive'):
            vals=[r[path]['macro_mean_think_steps'] for r in reports]
            aggregate[path]['macro_mean_think_steps']={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}
    summary={'name':'ulgan 1','seeds':args.seeds,'settings':vars(args),'aggregate':aggregate,'runs':reports}
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
