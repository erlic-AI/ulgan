import argparse, json, os, random
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
    task_ids = torch.randint(0, 3, (batch_size,), device=device)
    T = int(lengths.max().item()) + 1
    tokens = torch.full((batch_size, T), 2, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, device=device)
    for i, n in enumerate(lengths.tolist()):
        bits = torch.randint(0, 2, (n,), device=device)
        tokens[i, :n] = bits
        tokens[i, n] = 2
        targets[i] = label_for(bits.tolist(), int(task_ids[i].item()))
    return tokens, lengths, task_ids, targets


def run_epoch(model, optimizer, steps, batch_size, device):
    model.train(); losses=[]; accs=[]; halt_accs=[]
    for _ in range(steps):
        tokens, lengths, tasks, targets = make_batch(batch_size, 1, 16, device)
        h = torch.zeros(batch_size, model.hidden_size, device=device)
        final_logits = torch.zeros(batch_size, device=device)
        halt_losses=[]; halt_correct=[]
        for t in range(tokens.shape[1]):
            active = (lengths >= t)
            new_h, logits, halt_logits = model.step(tokens[:, t], tasks, h)
            h = torch.where(active[:, None], new_h, h)
            is_eos = (lengths == t)
            final_logits = torch.where(is_eos, logits, final_logits)
            halt_losses.append(F.binary_cross_entropy_with_logits(halt_logits, is_eos.float(), reduction='none') * active.float())
            halt_correct.append((((torch.sigmoid(halt_logits) >= .5) == is_eos).float()) * active.float())
        denom = (lengths.float() + 1).sum().clamp_min(1)
        halt_loss = torch.stack(halt_losses).sum() / denom
        per_item_loss = F.binary_cross_entropy_with_logits(final_logits, targets, reduction='none')
        task_weights = torch.where(tasks == 0, 2.5, 1.0)
        cls_loss = (per_item_loss * task_weights).mean()
        loss = cls_loss + 0.2 * halt_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(loss.item()); accs.append(((final_logits >= 0).float() == targets).float().mean().item())
        halt_accs.append((torch.stack(halt_correct).sum() / denom).item())
    return {'loss':float(np.mean(losses)), 'accuracy':float(np.mean(accs)), 'halt_token_accuracy':float(np.mean(halt_accs))}


@torch.no_grad()
def evaluate(model, n, min_len, max_len, device='cpu'):
    model.eval(); per_task={}; all_acc=[]; all_halt=[]; all_steps=[]
    for task_name, task_id in TASKS.items():
        lengths = torch.randint(min_len, max_len + 1, (n,), device=device)
        tokens = torch.full((n, max_len + 1), 2, dtype=torch.long, device=device)
        targets = torch.zeros(n, device=device)
        for i, ln in enumerate(lengths.tolist()):
            bits = torch.randint(0, 2, (ln,), device=device)
            tokens[i, :ln] = bits
            targets[i] = label_for(bits.tolist(), task_id)
        tasks = torch.full((n,), task_id, dtype=torch.long, device=device)
        h = torch.zeros(n, model.hidden_size, device=device)
        stopped = torch.zeros(n, dtype=torch.bool, device=device)
        pred_logits = torch.zeros(n, device=device)
        used = torch.zeros(n, device=device)
        correct_eos = torch.zeros(n, dtype=torch.bool, device=device)
        for t in range(max_len + 1):
            active = (~stopped) & (lengths >= t)
            if not active.any(): break
            new_h, logits, halt_logits = model.step(tokens[:, t], tasks, h)
            h = torch.where(active[:, None], new_h, h)
            used += active.float()
            halt = active & (torch.sigmoid(halt_logits) >= 0.5)
            pred_logits = torch.where(halt, logits, pred_logits)
            correct_eos |= halt & (lengths == t)
            stopped |= halt
        # A missing halt is scored using the last valid state only for output accuracy;
        # it remains visible in the separate halt metric.
        accuracy = ((pred_logits >= 0).float() == targets).float().mean().item()
        halt_rate = correct_eos.float().mean().item()
        mean_steps = used.mean().item()
        per_task[task_name] = {'n': n, 'accuracy': accuracy, 'correct_eos_halt_rate': halt_rate,
                               'non_eos_or_missing_halt_rate': 1.0 - halt_rate, 'mean_steps': mean_steps}
        all_acc.append(accuracy); all_halt.append(halt_rate); all_steps.append(mean_steps)
    return {'length_range':[min_len,max_len], 'tasks':per_task,
            'macro_accuracy':float(np.mean(all_acc)), 'macro_correct_eos_halt_rate':float(np.mean(all_halt)),
            'macro_non_eos_or_missing_halt_rate':float(1-np.mean(all_halt)), 'macro_mean_steps':float(np.mean(all_steps))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--steps-per-epoch',type=int,default=120); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',default='artifacts'); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ULGAN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1.5e-3,weight_decay=1e-5)
    history=[]
    for epoch in range(1,args.epochs+1):
        m=run_epoch(model,opt,args.steps_per_epoch,args.batch_size,device); history.append(m)
        print(f"epoch={epoch:02d} loss={m['loss']:.4f} accuracy={m['accuracy']:.4f} halt_acc={m['halt_token_accuracy']:.4f}",flush=True)
    os.makedirs(args.out,exist_ok=True)
    model.cpu(); ckpt={'state_dict':model.state_dict(),'config':{'hidden_size':48,'embedding_size':16,'task_embedding_size':8,'num_tasks':3},'tasks':TASKS,'seed':args.seed,'version':'0.2'}
    torch.save(ckpt,os.path.join(args.out,'ulgan_v0.2.pt'))
    report={'version':'0.2','seed':args.seed,'device':device,'parameters':sum(p.numel() for p in model.parameters()),'training_examples_presented':args.epochs*args.steps_per_epoch*args.batch_size,'history':history,
            'test_seen_lengths':evaluate(model,1000,1,16), 'test_unseen_lengths':evaluate(model,1000,17,32)}
    with open(os.path.join(args.out,'metrics.json'),'w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
