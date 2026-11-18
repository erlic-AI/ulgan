"""ULGAN v0.1: compact recurrent parity model with a learned halting head."""
import torch
from torch import nn

class ULGAN(nn.Module):
    def __init__(self, hidden_size: int = 32, embedding_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_size)  # 0, 1, EOS=2
        self.core = nn.GRUCell(embedding_size, hidden_size)
        self.output_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(), nn.Linear(hidden_size // 2, 1))
        self.halt_head = nn.Linear(hidden_size, 1)

    def step(self, token, hidden):
        hidden = self.core(self.embedding(token), hidden)
        logits = self.output_head(hidden).squeeze(-1)
        halt_logits = self.halt_head(hidden).squeeze(-1)
        return hidden, logits, halt_logits

    @torch.no_grad()
    def predict_one(self, sequence, threshold=0.5, max_steps=64, device=None):
        device = device or next(self.parameters()).device
        h = torch.zeros(1, self.core.hidden_size, device=device)
        used = 0
        for token in sequence[:max_steps]:
            x = torch.tensor([int(token)], dtype=torch.long, device=device)
            h, logits, halt_logits = self.step(x, h)
            used += 1
            if torch.sigmoid(halt_logits).item() >= threshold:
                return int(torch.sigmoid(logits).item() >= 0.5), used, True
        return int(torch.sigmoid(logits).item() >= 0.5), used, False
