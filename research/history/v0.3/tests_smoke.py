"""Minimal smoke test: checkpoint loads and all task interfaces return valid values."""
import torch
from model import ULGAN, TASKS

ckpt = torch.load('artifacts/ulgan_v0.3.pt', map_location='cpu', weights_only=False)
model = ULGAN(**ckpt['config'])
model.load_state_dict(ckpt['state_dict'])
model.eval()
for name, task_id in TASKS.items():
    pred, steps, halt_score = model.predict_one([1, 0, 1, 1], task_id)
    assert pred in (0, 1)
    assert 1 <= steps <= model.max_think_steps
    assert 0.0 <= halt_score <= 1.0
    print(name, pred, steps, round(halt_score, 4))
print('smoke_test=passed')
