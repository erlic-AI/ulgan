"""ULGAN v0.4: task-conditioned recurrent model with supervised adaptive halting."""
from __future__ import annotations
import torch
from torch import nn

TASKS = {"parity": 0, "majority": 1, "last_bit": 2}

class ULGAN(nn.Module):
    def __init__(self, hidden_size: int = 64, embedding_size: int = 16,
                 task_embedding_size: int = 8, max_think_steps: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_think_steps = max_think_steps
        self.embedding = nn.Embedding(3, embedding_size)
        self.task_embedding = nn.Embedding(len(TASKS), task_embedding_size)
        self.encoder = nn.GRU(embedding_size + task_embedding_size, hidden_size, batch_first=True)
        self.think_token = nn.Parameter(torch.zeros(1, embedding_size))
        self.thinker = nn.GRUCell(embedding_size + task_embedding_size, hidden_size)
        self.output_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in TASKS])
        self.halt = nn.Linear(hidden_size, 1)

    def encode(self, tokens, lengths, tasks):
        te = self.task_embedding(tasks)
        token_emb = self.embedding(tokens)
        task_seq = te[:, None, :].expand(-1, tokens.shape[1], -1)
        x = torch.cat((token_emb, task_seq), dim=-1)
        outputs, _ = self.encoder(x)
        # Gather the state after the final real bit, not after EOS/padding.
        idx = (lengths - 1).clamp_min(0)
        return outputs[torch.arange(tokens.shape[0], device=tokens.device), idx]

    def think_step(self, h, task_emb, tasks):
        x = torch.cat((self.think_token.expand(h.shape[0], -1), task_emb), dim=-1)
        h = self.thinker(x, h)
        all_logits = torch.cat([head(h) for head in self.output_heads], dim=1)
        logits = all_logits.gather(1, tasks[:, None]).squeeze(1)
        return h, logits, self.halt(h).squeeze(-1)

    def forward(self, tokens, lengths, tasks):
        h = self.encode(tokens, lengths, tasks)
        te = self.task_embedding(tasks)
        logits, halt_logits = [], []
        for _ in range(self.max_think_steps):
            h, y, z = self.think_step(h, te, tasks)
            logits.append(y); halt_logits.append(z)
        return torch.stack(logits, 1), torch.stack(halt_logits, 1)

    @torch.no_grad()
    def predict_one(self, bits, task_id: int, threshold: float = 0.95):
        self.eval(); device = next(self.parameters()).device
        seq = list(map(int, bits)) + [2]
        tokens = torch.tensor([seq], dtype=torch.long, device=device)
        lengths = torch.tensor([len(bits)], dtype=torch.long, device=device)
        tasks = torch.tensor([int(task_id)], dtype=torch.long, device=device)
        h = self.encode(tokens, lengths, tasks); te = self.task_embedding(tasks)
        cumulative = 0.0
        for step in range(self.max_think_steps):
            h, logits, halt_logit = self.think_step(h, te, tasks)
            p = float(torch.sigmoid(halt_logit).item())
            cumulative = 1.0 - (1.0 - cumulative) * (1.0 - p)
            if cumulative >= threshold or step == self.max_think_steps - 1:
                return int(logits.item() >= 0), step + 1, cumulative
        raise RuntimeError("unreachable")
