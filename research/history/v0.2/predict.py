import argparse, torch
from model import ULGAN, TASKS

def main():
    p=argparse.ArgumentParser(description='ULGAN v0.2 multi-task binary sequence inference')
    p.add_argument('task',choices=TASKS.keys()); p.add_argument('bits',nargs='+',type=int)
    p.add_argument('--checkpoint',default='artifacts/ulgan_v0.2.pt'); a=p.parse_args()
    if any(x not in (0,1) for x in a.bits): p.error('Bits must be 0 or 1')
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=False); model=ULGAN(**c['config']); model.load_state_dict(c['state_dict']); model.eval()
    pred,steps,halted=model.predict_one(a.bits,TASKS[a.task])
    print({'task':a.task,'prediction':pred,'processed_tokens':steps,'halted_at_eos':halted})
if __name__=='__main__': main()
