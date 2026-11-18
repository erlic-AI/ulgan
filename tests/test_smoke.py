from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ulgan import ULGAN, TASKS

def test_checkpoint_loads_and_predicts():
    path = ROOT / "artifacts/checkpoints/ulgan_v0.4.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ULGAN(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    assert sum(p.numel() for p in model.parameters()) == 34908
    for task_id in TASKS.values():
        prediction, steps, halt = model.predict_one([1, 0, 1, 1], task_id)
        assert prediction in (0, 1)
        assert 1 <= steps <= model.max_think_steps
        assert 0.0 <= halt <= 1.0

def test_forward_shapes():
    model = ULGAN()
    tokens = torch.tensor([[1, 0, 1, 2], [0, 1, 2, 2]])
    lengths = torch.tensor([3, 2])
    tasks = torch.tensor([0, 1])
    logits, halt_logits = model(tokens, lengths, tasks)
    assert logits.shape == (2, 8)
    assert halt_logits.shape == (2, 8)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(halt_logits).all()
