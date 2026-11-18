from pathlib import Path
import argparse
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ulgan import ULGAN, TASKS

def main():
    parser = argparse.ArgumentParser(description="Run the ulgan v0.4 reference checkpoint.")
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("bits", nargs="+", type=int, help="Binary sequence, e.g. 1 0 1 1")
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts/checkpoints/ulgan_v0.4.pt"))
    args = parser.parse_args()
    if any(bit not in (0, 1) for bit in args.bits):
        parser.error("Every sequence element must be 0 or 1.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ULGAN(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    prediction, steps, halt = model.predict_one(args.bits, TASKS[args.task])
    print({"task": args.task, "bits": args.bits, "prediction": prediction,
           "think_steps": steps, "cumulative_halt_probability": round(halt, 4)})

if __name__ == "__main__":
    main()
