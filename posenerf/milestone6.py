"""Leakage-free RGB-only validation localization benchmark."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .config import build_paths
from .data import load_chess_dataset
from .milestone3 import _checkpoint_for, _safe_run, load_run_config
from .model import HierarchicalNeRF, get_ray_directions, render_image
from .pose import retrieve_initial_pose, optimize_pose_from_rgb, rotation_error_deg, translation_error_m, split_pose_rays
from .utils import get_device, save_json, seed_everything

def _id(frame, scene): return str(frame.color_path.relative_to(scene)).replace('\\','/')
def _stats(values):
    values=np.asarray(values,float); return {'median':float(np.median(values)),'mean':float(np.mean(values)),'p90':float(np.percentile(values,90))}

@torch.inference_mode()
def _render(model, pose, dirs, config): return render_image(model,pose,dirs,config,num_coarse_samples=config.pose.render_coarse_samples,num_fine_samples=config.pose.render_fine_samples)[0]

def evaluate_run(workspace: Path, name: str, output: Path, steps: int, seed: int=612, limit: int | None=None, sampling_mode: str='uniform', conservative: bool=False) -> dict:
    run=_safe_run(workspace,name)
    if output.exists(): raise FileExistsError(f'refusing to overwrite {output}')
    config=load_run_config(run); config.pose.iters=steps; config.pose.batch_rays=512; config.pose.lr=3e-3
    device=get_device(); seed_everything(seed)
    data=load_chess_dataset(config,build_paths(workspace),device,storage_device=torch.device('cpu'),include_test=False)
    protocol=json.loads((run/'frozen_protocol.json').read_text())
    ids=[_id(x,build_paths(workspace).scene_dir) for x in data.val_frames]
    if ids != [x.replace('\\','/') for x in protocol['validation_frames']]: raise ValueError('validation protocol drift')
    train_ids={_id(x,build_paths(workspace).scene_dir) for x in data.train_frames}
    if train_ids.intersection(ids): raise ValueError('query/retrieval database overlap')
    if limit is not None: ids=ids[:limit]
    checkpoint,payload=_checkpoint_for(run,'hierarchical_nerf_best.pth',config,device)
    model=HierarchicalNeRF(config).to(device); model.load_state_dict(payload['model_state_dict'],strict=True); model.eval()
    weights={k:v.detach().clone() for k,v in model.state_dict().items()}; dirs=get_ray_directions(config,device)
    rows=[]
    for index, query_id in enumerate(ids):
        # INFERENCE BOUNDARY: only RGB plus training RGB/poses cross it.
        query_rgb=data.val_images[index].to(device); init, retrieved=retrieve_initial_pose(query_rgb,data.train_images.to(device),data.train_poses.to(device))
        generator=torch.Generator(device=device).manual_seed(seed+index)
        optimize_pixels, holdout_pixels = split_pose_rays(query_rgb, config.pose.batch_rays, config.pose.batch_rays, generator, sampling_mode)
        init_render=_render(model,init.to(device),dirs,config); init_loss=float(F.mse_loss(init_render,query_rgb))
        started=time.perf_counter(); proposed,history=optimize_pose_from_rgb(model,query_rgb,init.to(device),dirs,config,device,f'RGB-only localize {index}',generator,sampling_mode,conservative,optimize_pixels)
        if device.type=='cuda': torch.cuda.synchronize(device)
        runtime=time.perf_counter()-started
        proposed_render=_render(model,proposed,dirs,config); proposed_loss=float(F.mse_loss(proposed_render,query_rgb)); displacement=translation_error_m(proposed,init.to(device)); hy,hx=holdout_pixels; holdout_init=float(F.mse_loss(init_render[hy,hx],query_rgb[hy,hx])); holdout_final=float(F.mse_loss(proposed_render[hy,hx],query_rgb[hy,hx])); accepted=(not conservative) or (math.isfinite(holdout_final) and holdout_final < holdout_init and displacement <= .10)
        refined=proposed if accepted else init.to(device)
        gt=data.val_poses[index].to(device) # EVALUATION BOUNDARY: never passed to retrieval/optimization.
        rows.append({'query_frame_id':query_id,'retrieval_frame_id':_id(data.train_frames[retrieved],build_paths(workspace).scene_dir),'initial_pose':init.tolist(),'refined_pose':refined.tolist(),'initial_translation_error_m':translation_error_m(init.to(device),gt),'initial_rotation_error_deg':rotation_error_deg(init.to(device),gt),'refined_translation_error_m':translation_error_m(refined,gt),'refined_rotation_error_deg':rotation_error_deg(refined,gt),'translation_improvement_m':translation_error_m(init.to(device),gt)-translation_error_m(refined,gt),'rotation_improvement_deg':rotation_error_deg(init.to(device),gt)-rotation_error_deg(refined,gt),'initial_loss':init_loss,'final_loss':proposed_loss,'holdout_initial_loss':holdout_init,'holdout_final_loss':holdout_final,'holdout_improvement':holdout_init-holdout_final,'pose_displacement_m':displacement,'accepted':accepted,'rejection_reason':None if accepted else 'holdout_or_displacement','runtime_seconds':runtime,'iterations':steps,'status':'ok' if torch.isfinite(refined).all() else 'invalid'})
    if any(not torch.equal(weights[k],v) for k,v in model.state_dict().items()): raise RuntimeError('pose optimization modified NeRF weights')
    def summary(prefix):
        t=[r[f'{prefix}_translation_error_m'] for r in rows]; q=[r[f'{prefix}_rotation_error_deg'] for r in rows]
        return {'translation_m':_stats(t),'rotation_deg':_stats(q),'success_5cm_5deg':sum(a<=.05 and b<=5 for a,b in zip(t,q))/len(rows),'success_10cm_10deg':sum(a<=.1 and b<=10 for a,b in zip(t,q))/len(rows)}
    report={'run':name,'checkpoint':str(checkpoint),'checkpoint_step':int(payload.get('step',-1)),'protocol_sha256':protocol['protocol_sha256'],'inference_contract':'query RGB, trained checkpoint, training RGB descriptors/poses, and intrinsics only; query pose/depth excluded','steps':steps,'seed':seed,'sampling_mode':sampling_mode,'conservative':conservative,'accepted_count':sum(r['accepted'] for r in rows),'initial':summary('initial'),'refined':summary('refined'),'frames':rows,'cuda_peak_mb':float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=='cuda' else 0.0}
    output.mkdir(parents=True); save_json(output/'localization_report.json',report); return report

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,default=Path('workspace'));p.add_argument('--run-name',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--limit',type=int);p.add_argument('--sampling-mode',choices=('uniform','informative','mixed'),default='uniform');p.add_argument('--conservative',action='store_true');a=p.parse_args(); print(json.dumps(evaluate_run(a.workspace.resolve(),a.run_name,a.output.resolve(),a.steps,limit=a.limit,sampling_mode=a.sampling_mode,conservative=a.conservative),indent=2))
if __name__=='__main__': main()
