"""ULGAN v0.2: task-conditioned recurrent model for three binary sequence tasks."""
import torch
from torch import nn

TASKS = {"parity": 0, "majority": 1, "last_bit": 2}

class ULGAN(nn.Module):
    def __init__(self, hidden_size: int = 48, embedding_size: int = 16, task_embedding_size: int = 8, num_tasks: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(3, embedding_size)  # 0, 1, EOS=2
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_size)
        self.core = nn.GRUCell(embedding_size + task_embedding_size, hidden_size)
        self.output_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)) for _ in range(num_tasks)])
        self.halt_head = nn.Linear(hidden_size, 1)

    def step(self, token, task, hidden):
        x = torch.cat([self.embedding(token), self.task_embedding(task)], dim=-1)
        hidden = self.core(x, hidden)
        all_logits = torch.cat([head(hidden) for head in self.output_heads], dim=1)
        logits = all_logits.gather(1, task.view(-1, 1)).squeeze(1)
        return hidden, logits, self.halt_head(hidden).squeeze(-1)

    @torch.no_grad()
    def predict_one(self, bits, task_id, threshold=0.5, max_steps=256):
        device = next(self.parameters()).device
        h = torch.zeros(1, self.hidden_size, device=device)
        task = torch.tensor([int(task_id)], dtype=torch.long, device=device)
        used = 0
        for token in list(bits) + [2]:
            x = torch.tensor([int(token)], dtype=torch.long, device=device)
            h, logits, halt_logits = self.step(x, task, h)
            used += 1
            if torch.sigmoid(halt_logits).item() >= threshold:
                return int(torch.sigmoid(logits).item() >= 0.5), used, token == 2
            if used >= max_steps:
                break
        return int(torch.sigmoid(logits).item() >= 0.5), used, False
