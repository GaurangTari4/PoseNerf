"""Bounded, read-only M14 frozen-view diagnostics."""
from __future__ import annotations
import argparse, csv, json, math, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from posenerf.milestone3 import load_run_config, _ssim
from posenerf.config import build_paths
from posenerf.data import load_chess_dataset
from posenerf.model import HierarchicalNeRF, get_ray_directions, render_image
from posenerf.utils import get_device, safe_torch_load, seed_everything

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--indices',type=int,nargs='+',required=True);p.add_argument('--coarse',type=int,default=48);p.add_argument('--fine',type=int,default=48);a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 cfg=load_run_config(a.run); dev=get_device();seed_everything(cfg.seed); data=load_chess_dataset(cfg,build_paths(a.run.parent.parent),dev,storage_device=torch.device('cpu'),include_test=False)
 payload=safe_torch_load(a.run/'checkpoints'/a.checkpoint,dev); model=HierarchicalNeRF(cfg).to(dev);model.load_state_dict(payload['model_state_dict']);model.eval(); dirs=get_ray_directions(cfg,dev)
 rows=[]; a.output.mkdir(parents=True)
 for i in a.indices:
  if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev);torch.cuda.synchronize(dev)
  t=time.perf_counter(); pred,pdepth=render_image(model,data.val_poses[i].to(dev),dirs,cfg,num_coarse_samples=a.coarse,num_fine_samples=a.fine)
  if dev.type=='cuda': torch.cuda.synchronize(dev)
  rgb=data.val_images[i].to(dev); gt=data.val_depths[i].to(dev);mask=data.val_masks[i].to(dev)>0.5;err=(pdepth-gt).abs();
  valid=err[mask]; bins=[]
  for lo,hi in ((0.1,1.5667),(1.5667,3.0333),(3.0333,4.5)):
   m=mask&(gt>=lo)&(gt<hi);bins.append(float(err[m].mean()) if m.any() else float('nan'))
  gray=rgb.mean(-1);gx=F.pad((gray[:,2:]-gray[:,:-2]).abs(),(1,1));gy=F.pad((gray[2:,:]-gray[:-2,:]).abs(),(0,0,1,1));edge=(gx+gy)>=torch.quantile((gx+gy)[mask],.75);bound=mask&edge;inter=mask&~edge
  rows.append({'frame_index':i,'psnr':-10*math.log10(max(float(F.mse_loss(pred,rgb)),1e-12)),'ssim':_ssim(pred,rgb),'mae':float(valid.mean()),'rmse':float(valid.square().mean().sqrt()),'absrel':float((valid/gt[mask].clamp_min(1e-6)).mean()),'valid_pixels':int(mask.sum()),'near_mae':bins[0],'middle_mae':bins[1],'far_mae':bins[2],'boundary_mae':float(err[bound].mean()),'interior_mae':float(err[inter].mean()),'latency_ms':(time.perf_counter()-t)*1000,'cuda_peak_mb':float(torch.cuda.max_memory_allocated(dev)/2**20) if dev.type=='cuda' else 0.0})
  np.save(a.output/f'residual_{i:02d}.npy',err.detach().cpu().numpy())
 with (a.output/'report.json').open('w') as f: json.dump({'run':a.run.name,'checkpoint':a.checkpoint,'checkpoint_step':payload.get('step'),'coarse':a.coarse,'fine':a.fine,'rows':rows},f,indent=2)
 with (a.output/'rows.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps(rows))
if __name__=='__main__':main()
