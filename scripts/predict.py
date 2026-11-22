from pathlib import Path
import argparse, sys, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ulgan import ULGAN, TASKS

def main():
    ap=argparse.ArgumentParser(description='Run an ulgan 1 checkpoint.')
    ap.add_argument('task',choices=TASKS)
    ap.add_argument('bits',nargs='+',type=int)
    ap.add_argument('--checkpoint',required=True)
    args=ap.parse_args()
    if any(x not in (0,1) for x in args.bits): ap.error('Bits must be 0 or 1.')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=ULGAN(**ckpt['config']); model.load_state_dict(ckpt['state_dict'])
    pred,steps,halt=model.predict_one(args.bits,TASKS[args.task])
    print({'task':args.task,'bits':args.bits,'prediction':pred,'think_steps':steps,'cumulative_halt_probability':round(halt,5)})
if __name__=='__main__': main()
