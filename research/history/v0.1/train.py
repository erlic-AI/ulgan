import argparse, json, os, random
import numpy as np
import torch
import torch.nn.functional as F
from model import ULGAN

def make_batch(batch_size, min_len, max_len, device):
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)
    T = int(lengths.max().item()) + 1
    tokens = torch.full((batch_size, T), 2, dtype=torch.long, device=device) # EOS padding after actual EOS
    targets = torch.zeros(batch_size, device=device)
    for i, n in enumerate(lengths.tolist()):
        bits = torch.randint(0, 2, (n,), device=device)
        tokens[i, :n] = bits
        tokens[i, n] = 2
        targets[i] = bits.sum().remainder(2).float()
    return tokens, lengths, targets

def run_epoch(model, optimizer, steps, batch_size, device, train=True):
    model.train(train)
    losses=[]; accs=[]; halt_accs=[]
    for _ in range(steps):
        tokens, lengths, targets = make_batch(batch_size, 1, 12, device)
        h = torch.zeros(batch_size, model.core.hidden_size, device=device)
        final_logits = torch.zeros(batch_size, device=device)
        halt_losses=[]; halt_correct=[]
        for t in range(tokens.shape[1]):
            active = (lengths >= t).float()  # includes EOS at t == length
            token = tokens[:, t]
            new_h, logits, halt_logits = model.step(token, h)
            mask = active.unsqueeze(1)
            h = new_h * mask + h * (1-mask)
            is_eos = (lengths == t).float()
            final_logits = torch.where(is_eos.bool(), logits, final_logits)
            halt_losses.append(F.binary_cross_entropy_with_logits(halt_logits, is_eos, reduction='none') * active)
            halt_correct.append((((torch.sigmoid(halt_logits) >= .5) == is_eos.bool()).float() * active))
        denom = (lengths.float() + 1).sum().clamp_min(1)
        halt_loss = torch.stack(halt_losses).sum() / denom
        cls_loss = F.binary_cross_entropy_with_logits(final_logits, targets)
        loss = cls_loss + 0.25 * halt_loss
        if train:
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        pred = (torch.sigmoid(final_logits) >= .5).float()
        losses.append(loss.item()); accs.append((pred == targets).float().mean().item())
        hc = torch.stack(halt_correct).sum() / denom
        halt_accs.append(hc.item())
    return {'loss':float(np.mean(losses)), 'accuracy':float(np.mean(accs)), 'halt_token_accuracy':float(np.mean(halt_accs))}

@torch.no_grad()
def evaluate(model, n=2000, min_len=1, max_len=12, device='cpu'):
    model.eval()
    lengths = torch.tensor([random.randint(min_len, max_len) for _ in range(n)], device=device)
    max_t = int(lengths.max().item()) + 1
    tokens = torch.full((n, max_t), 2, dtype=torch.long, device=device)
    targets = torch.zeros(n, device=device)
    for i, ln in enumerate(lengths.tolist()):
        bits = torch.randint(0, 2, (ln,), device=device)
        tokens[i, :ln] = bits
        tokens[i, ln] = 2
        targets[i] = bits.sum().remainder(2).float()
    h = torch.zeros(n, model.core.hidden_size, device=device)
    stopped = torch.zeros(n, dtype=torch.bool, device=device)
    pred_logits = torch.zeros(n, device=device)
    used = torch.zeros(n, device=device)
    stop_at_eos = torch.zeros(n, dtype=torch.bool, device=device)
    premature = torch.zeros(n, dtype=torch.bool, device=device)
    for t in range(max_t):
        active = (~stopped) & (lengths >= t)
        if not active.any(): break
        new_h, logits, halt_logits = model.step(tokens[:, t], h)
        h = torch.where(active.unsqueeze(1), new_h, h)
        used += active.float()
        halt = active & (torch.sigmoid(halt_logits) >= 0.5)
        pred_logits = torch.where(halt, logits, pred_logits)
        stop_at_eos |= halt & (lengths == t)
        premature |= halt & (lengths > t)
        stopped |= halt
    missing = ~stopped
    # For sequences that did not halt, use their final available output for diagnostic accuracy.
    accuracy = ((torch.sigmoid(pred_logits) >= .5).float() == targets).float().mean().item()
    return {'n':n,'length_range':[min_len,max_len],'accuracy':accuracy,'halt_rate':stopped.float().mean().item(),
            'correct_halt_rate':stop_at_eos.float().mean().item(),'premature_halt_rate':premature.float().mean().item(),
            'missing_halt_rate':missing.float().mean().item(),'mean_steps':used.mean().item()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=6); ap.add_argument('--steps-per-epoch',type=int,default=80); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',default='artifacts'); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ULGAN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    history=[]
    for epoch in range(1,args.epochs+1):
        metrics=run_epoch(model,opt,args.steps_per_epoch,args.batch_size,device,True)
        history.append(metrics)
        print(f"epoch={epoch:02d} loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f} halt_acc={metrics['halt_token_accuracy']:.4f}",flush=True)
    os.makedirs(args.out,exist_ok=True)
    torch.save({'state_dict':model.cpu().state_dict(),'config':{'hidden_size':32,'embedding_size':16},'seed':args.seed},os.path.join(args.out,'ulgan_v0.1.pt'))
    report={'device':device,'parameters':sum(p.numel() for p in model.parameters()),'history':history,
            'test_seen_lengths':evaluate(model,2000,1,12,device='cpu'),
            'test_unseen_lengths':evaluate(model,2000,13,20,device='cpu')}
    with open(os.path.join(args.out,'metrics.json'),'w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
