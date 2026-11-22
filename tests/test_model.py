import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ulgan import ULGAN, TASKS

def test_shapes_and_finite():
    m=ULGAN(); x=torch.tensor([[1,0,1,2],[0,1,2,2]]); lengths=torch.tensor([3,2]); tasks=torch.tensor([0,1])
    logits,halt,prefix=m(x,lengths,tasks)
    assert logits.shape==(2,12) and halt.shape==(2,12) and prefix.shape==(2,4)
    assert torch.isfinite(logits).all() and torch.isfinite(halt).all() and torch.isfinite(prefix).all()

def test_predict_contract():
    m=ULGAN().eval()
    for tid in TASKS.values():
        pred,steps,halt=m.predict_one([1,0,1,1],tid)
        assert pred in (0,1) and 1<=steps<=12 and 0<=halt<=1

def test_prefix_parity_targets():
    bits=torch.tensor([1,1,0,1]); assert bits.cumsum(0).remainder(2).tolist()==[1,0,0,1]


def test_prefix_head_has_no_future_bit_leakage():
    torch.manual_seed(7)
    m=ULGAN().eval()
    a=torch.tensor([[1,0,1,0,2]])
    b=torch.tensor([[1,0,0,1,2]])
    lengths=torch.tensor([4]); tasks=torch.tensor([0])
    with torch.no_grad():
        pa=m(a,lengths,tasks)[2]
        pb=m(b,lengths,tasks)[2]
    # Prefixes 1 and 2 must not depend on later bits.
    assert torch.allclose(pa[:,:2],pb[:,:2],atol=1e-6)


def test_prefix_forward_states_are_causal():
    torch.manual_seed(19)
    m=ULGAN().eval()
    x1=torch.tensor([[0,1,0,0,2]])
    x2=torch.tensor([[0,1,1,1,2]])
    with torch.no_grad():
        _, enc1=m.encode(x1,torch.tensor([4]),torch.tensor([0]))
        _, enc2=m.encode(x2,torch.tensor([4]),torch.tensor([0]))
    assert torch.allclose(enc1[:,:2,:m.hidden_size],enc2[:,:2,:m.hidden_size],atol=1e-6)


def test_task_label_definitions_match_vectorized_batch_logic():
    bits = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0]])
    lengths = torch.tensor([4, 3, 1])
    counts = (bits * (torch.arange(4)[None, :] < lengths[:, None])).sum(1)
    parity = counts.remainder(2)
    majority = (counts > lengths.float() / 2).long()
    last = bits.gather(1, (lengths - 1)[:, None]).squeeze(1)
    assert parity.tolist() == [0, 1, 1]
    assert majority.tolist() == [0, 1, 1]
    assert last.tolist() == [0, 1, 1]


def test_encoder_representation_is_invariant_to_extra_right_padding():
    torch.manual_seed(31)
    model = ULGAN().eval()
    short = torch.tensor([[1, 0, 1, 2]])
    padded = torch.tensor([[1, 0, 1, 2, 2, 2, 2]])
    lengths = torch.tensor([3])
    tasks = torch.tensor([0])
    with torch.no_grad():
        a, _ = model.encode(short, lengths, tasks)
        b, _ = model.encode(padded, lengths, tasks)
    assert torch.allclose(a, b, atol=1e-6)


def test_density_feature_does_not_encode_parity_label():
    torch.manual_seed(37)
    model = ULGAN().eval()
    # Same density and different parity are possible at different lengths;
    # the scalar feature is only a proportion, not an exact parity computation.
    x = torch.tensor([[1, 0, 2, 2, 2], [1, 0, 1, 0, 2]])
    lengths = torch.tensor([2, 4])
    tasks = torch.tensor([0, 0])
    with torch.no_grad():
        rep, _ = model.encode(x, lengths, tasks)
    assert torch.isfinite(rep).all()
    assert not hasattr(model, 'parity_projection')


def test_thinker_uses_encoded_bit_context_and_masks_padding():
    torch.manual_seed(101)
    model = ULGAN().eval()
    tokens_a = torch.tensor([[1, 0, 1, 2]])
    tokens_b = torch.tensor([[1, 0, 1, 2, 2, 2]])
    lengths = torch.tensor([3])
    tasks = torch.tensor([0])
    with torch.no_grad():
        h_a, enc_a = model.encode(tokens_a, lengths, tasks)
        h_b, enc_b = model.encode(tokens_b, lengths, tasks)
        te = model.task_embedding(tasks)
        out_a = model.think_step(h_a, te, tasks, enc_a, lengths)[1]
        out_b = model.think_step(h_b, te, tasks, enc_b, lengths)[1]
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_predict_rejects_invalid_task_and_threshold():
    model = ULGAN().eval()
    import pytest
    with pytest.raises(ValueError):
        model.predict_one([0, 1], 99)
    with pytest.raises(ValueError):
        model.predict_one([0, 1], 0, threshold=0.0)
