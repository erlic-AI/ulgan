import argparse, torch
from model import ULGAN, TASKS

p=argparse.ArgumentParser(description='Run ULGAN v0.3 with adaptive latent computation.')
p.add_argument('task',choices=TASKS); p.add_argument('bits',nargs='+',type=int); p.add_argument('--checkpoint',default='artifacts/ulgan_v0.3.pt'); p.add_argument('--threshold',type=float,default=.95)
a=p.parse_args()
if any(x not in (0,1) for x in a.bits): p.error('bits must contain only 0 or 1')
ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
m=ULGAN(**ckpt['config']); m.load_state_dict(ckpt['state_dict']); m.eval()
pred,steps,halt=m.predict_one(a.bits,TASKS[a.task],a.threshold)
print({'task':a.task,'input':a.bits,'prediction':pred,'adaptive_think_steps':steps,'cumulative_halt_probability':round(halt,4)})
