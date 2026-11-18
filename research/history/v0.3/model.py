"""ULGAN v0.3: task-conditioned recurrent model with Adaptive Computation Time (ACT)."""
from __future__ import annotations
import torch
from torch import nn

TASKS = {"parity": 0, "majority": 1, "last_bit": 2}

class ULGAN(nn.Module):
    def __init__(self, hidden_size: int = 48, embedding_size: int = 16,
                 task_embedding_size: int = 8, max_think_steps: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_think_steps = max_think_steps
        self.embedding = nn.Embedding(3, embedding_size)  # 0, 1, EOS=2
        self.task_embedding = nn.Embedding(len(TASKS), task_embedding_size)
        self.encoder = nn.GRUCell(embedding_size + task_embedding_size, hidden_size)
        self.think_token = nn.Parameter(torch.zeros(1, embedding_size))
        self.thinker = nn.GRUCell(embedding_size + task_embedding_size, hidden_size)
        self.output_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(len(TASKS))])
        self.halt = nn.Linear(hidden_size, 1)

    def encode(self, tokens: torch.Tensor, lengths: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        h = torch.zeros(batch, self.hidden_size, device=tokens.device)
        te = self.task_embedding(tasks)
        for t in range(tokens.shape[1]):
            active = lengths >= t
            x = torch.cat([self.embedding(tokens[:, t]), te], dim=-1)
            nh = self.encoder(x, h)
            h = torch.where(active[:, None], nh, h)
        return h

    def think_step(self, h: torch.Tensor, task_emb: torch.Tensor, tasks: torch.Tensor):
        x = torch.cat([self.think_token.expand(h.shape[0], -1), task_emb], dim=-1)
        h = self.thinker(x, h)
        all_logits = torch.cat([head(h) for head in self.output_heads], dim=1)
        logits = all_logits.gather(1, tasks.view(-1, 1)).squeeze(1)
        return h, logits, self.halt(h).squeeze(-1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, tasks: torch.Tensor):
        h = self.encode(tokens, lengths, tasks)
        te = self.task_embedding(tasks)
        batch = tokens.shape[0]
        survival = torch.ones(batch, device=tokens.device)
        weighted_logits = torch.zeros(batch, device=tokens.device)
        expected_steps = torch.zeros(batch, device=tokens.device)
        halt_probs = []
        step_logits = []
        for step in range(self.max_think_steps):
            h, logits, halt_logits = self.think_step(h, te, tasks)
            p = torch.sigmoid(halt_logits)
            # ACT: probability mass that halts at this step.
            if step == self.max_think_steps - 1:
                weight = survival
            else:
                weight = survival * p
            survival = survival - weight
            weighted_logits = weighted_logits + weight * logits
            expected_steps = expected_steps + weight * float(step + 1)
            halt_probs.append(p)
            step_logits.append(logits)
        return weighted_logits, expected_steps, torch.stack(halt_probs, dim=1), torch.stack(step_logits, dim=1)

    @torch.no_grad()
    def predict_one(self, bits, task_id: int, threshold: float = 0.95):
        self.eval()
        device = next(self.parameters()).device
        seq = list(map(int, bits)) + [2]
        tokens = torch.tensor([seq], dtype=torch.long, device=device)
        lengths = torch.tensor([len(seq)-1], dtype=torch.long, device=device)
        tasks = torch.tensor([int(task_id)], dtype=torch.long, device=device)
        h = self.encode(tokens, lengths, tasks)
        te = self.task_embedding(tasks)
        cumulative = 0.0
        for step in range(self.max_think_steps):
            h, logits, halt_logits = self.think_step(h, te, tasks)
            p = float(torch.sigmoid(halt_logits).item())
            cumulative = 1.0 - (1.0-cumulative)*(1.0-p)
            if cumulative >= threshold or step == self.max_think_steps-1:
                return int(logits.item() >= 0), step+1, cumulative
        raise RuntimeError('unreachable')
