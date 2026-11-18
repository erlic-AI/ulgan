"""Run a trained ULGAN v0.1 checkpoint on a bit sequence."""
import argparse
import torch
from model import ULGAN


def main():
    parser = argparse.ArgumentParser(description='Predict parity with ULGAN v0.1')
    parser.add_argument('bits', nargs='+', type=int, help='A sequence of 0/1 values, e.g. 1 0 1 1')
    parser.add_argument('--checkpoint', default='artifacts/ulgan_v0.1.pt')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    if any(x not in (0, 1) for x in args.bits):
        parser.error('All input values must be 0 or 1.')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = ULGAN(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    pred, steps, halted = model.predict_one(args.bits + [2], threshold=args.threshold, device='cpu')
    print({'prediction': pred, 'processed_tokens': steps, 'halted': halted,
           'input_length': len(args.bits), 'expected_parity_for_check': sum(args.bits) % 2})

if __name__ == '__main__':
    main()
