"""Experimental geometry-guided iNeRF atop the preserved Milestone 9 PnP pipeline."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .config import build_paths
from .data import load_chess_dataset
from .milestone3 import _checkpoint_for, _safe_run, load_run_config
from .milestone9 import build_reference_database, camera_matrix, match_candidates, pnp_ransac, _frame_id
from .model import HierarchicalNeRF, get_ray_directions, get_rays, render_image, render_rays
from .pose import compose_pose_delta, retrieve_initial_pose, rotation_error_deg, translation_error_m
from .utils import get_device, save_json, seed_everything

def reprojection_residual(c2w, world, pixels, K):
    """Differentiable OpenCV +Z projection; invalid/behind points are masked."""
    R,t=c2w[:3,:3],c2w[:3,3]; camera=(world-t)@R
    z=camera[:,2]; pred=torch.stack((K[0,0]*camera[:,0]/z+K[0,2],K[1,1]*camera[:,1]/z+K[1,2]),-1)
    return pred-pixels, z>1e-5

def refine(model, rgb, initial, dirs, world, pixels, K, config, steps=80, photo_weight=.2, geometry_weight=1., prior_weight=.02, seed=612):
    """Frozen-NeRF SE(3) refinement using RGB, Huber reprojection and PnP prior."""
    if len(world)<6: return initial,[],"degenerate_correspondences"
    dev=initial.device; rv=torch.nn.Parameter(torch.zeros(3,device=dev)); tv=torch.nn.Parameter(torch.zeros(3,device=dev)); opt=torch.optim.Adam([rv,tv],lr=8e-4)
    states=[p.requires_grad for p in model.parameters()]
    for p in model.parameters():p.requires_grad_(False)
    gen=torch.Generator(device=dev).manual_seed(seed); history=[]
    try:
      for _ in range(steps):
        pose=compose_pose_delta(initial,rv,tv); residual,valid=reprojection_residual(pose,world,pixels,K)
        if valid.sum()<6: return initial,history,"behind_camera"
        geo=F.huber_loss(residual[valid],torch.zeros_like(residual[valid]),delta=1.)
        y=torch.randint(rgb.shape[0],(256,),device=dev,generator=gen);x=torch.randint(rgb.shape[1],(256,),device=dev,generator=gen)
        ro,rd=get_rays(dirs[y,x],pose); pred,_,_=render_rays(model,ro,rd,config,perturb=False,num_coarse_samples=16,num_fine_samples=16)
        photo=F.mse_loss(pred,rgb[y,x]); prior=rv.square().sum()+tv.square().sum(); loss=geometry_weight*geo+photo_weight*(photo/.02)+prior_weight*prior
        if not torch.isfinite(loss): return initial,history,"nonfinite"
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([rv,tv],1.);opt.step();history.append({'loss':float(loss.detach()),'geometry':float(geo.detach()),'photo':float(photo.detach())})
    finally:
      for p,s in zip(model.parameters(),states):p.requires_grad_(s)
    return compose_pose_delta(initial,rv,tv).detach(),history,"ok"

@torch.no_grad()
def rgb_loss(model,pose,rgb,dirs,config):
  # Fixed sparse diagnostic residual: avoids a full image render per candidate.
  y=torch.arange(0,rgb.shape[0],8,device=rgb.device).repeat_interleave(rgb.shape[1]//8)[:240];x=torch.arange(0,rgb.shape[1],8,device=rgb.device).repeat(rgb.shape[0]//8)[:240]
  ro,rd=get_rays(dirs[y,x],pose);pred,_,_=render_rays(model,ro,rd,config,perturb=False,num_coarse_samples=16,num_fine_samples=16);return float(F.mse_loss(pred,rgb[y,x]))

def evaluate(workspace, run_name, output, cache, steps=80, seed=612):
  run=_safe_run(workspace,run_name)
  if output.exists():raise FileExistsError(output)
  output.mkdir(parents=True);config=load_run_config(run);dev=get_device();seed_everything(seed); data=load_chess_dataset(config,build_paths(workspace),dev,storage_device=torch.device('cpu'),include_test=False); scene=build_paths(workspace).scene_dir
  protocol=json.loads((run/'frozen_protocol.json').read_text()); ids=[_frame_id(f,scene) for f in data.val_frames]
  if ids!=[x.replace('\\','/') for x in protocol['validation_frames']]:raise ValueError('protocol drift')
  ck,payload=_checkpoint_for(run,'hierarchical_nerf_best.pth',config,dev);model=HierarchicalNeRF(config).to(dev);model.load_state_dict(payload['model_state_dict']);model.eval(); frozen={k:v.clone() for k,v in model.state_dict().items()};dirs=get_ray_directions(config,dev);db=build_reference_database(data,config,scene,cache);K=torch.tensor(camera_matrix(config),dtype=torch.float32,device=dev);rows=[]
  for i,qid in enumerate(ids):
    start=time.perf_counter();rgb=data.val_images[i].to(dev);retr,nearest=retrieve_initial_pose(rgb,data.train_images.to(dev),data.train_poses.to(dev)); errors=(data.train_images-rgb.cpu()).square().mean((1,2,3)).numpy();cand=np.argsort(errors)[:5].tolist();qp,wp,_,_,_,_=match_candidates(rgb.cpu(),db,cand);pn=pnp_ransac(qp,wp,camera_matrix(config),seed+i)
    if not pn['ok']:
      chosen=retr;rows.append({'query_frame_id':qid,'source':'retrieval_fallback','reason':pn['reason'],'pnp_ok':False});continue
    pnp=torch.from_numpy(pn['pose']).to(dev); idx=pn['inliers']; rng=np.random.default_rng(seed+i);rng.shuffle(idx);cut=max(6,int(.8*len(idx)));train,hold=idx[:cut],idx[cut:];world=torch.tensor(wp,dtype=torch.float32,device=dev);pixels=torch.tensor(qp,dtype=torch.float32,device=dev)
    base_hold=float(F.huber_loss(reprojection_residual(pnp,world[hold],pixels[hold],K)[0],torch.zeros_like(pixels[hold]))) if len(hold) else float('inf');before=rgb_loss(model,pnp,rgb,dirs,config); candidate,hist,status=refine(model,rgb,pnp,dirs,world[train],pixels[train],K,config,steps,seed=seed+i); after=rgb_loss(model,candidate,rgb,dirs,config);res,valid=reprojection_residual(candidate,world[hold],pixels[hold],K);hold_loss=float(F.huber_loss(res[valid],torch.zeros_like(res[valid]))) if valid.any() else float('inf'); displacement=translation_error_m(candidate,pnp); accept=status=='ok' and math.isfinite(hold_loss) and hold_loss<=base_hold*1.05 and displacement<=.05
    chosen=candidate if accept else pnp;gt=data.val_poses[i].to(dev);rows.append({'query_frame_id':qid,'source':'geometry_guided_accepted' if accept else 'pnp_fallback','reason':status if not accept else None,'pnp_ok':True,'inliers':int(len(idx)),'pnp_reprojection_rmse_px':pn['reprojection_rmse_px'],'pnp_translation_error_m':translation_error_m(pnp,gt),'pnp_rotation_error_deg':rotation_error_deg(pnp,gt),'final_translation_error_m':translation_error_m(chosen,gt),'final_rotation_error_deg':rotation_error_deg(chosen,gt),'rgb_before':before,'rgb_after':after,'holdout_geometry_before':base_hold,'holdout_geometry_after':hold_loss,'displacement_m':displacement,'history':hist,'runtime_seconds':time.perf_counter()-start})
  if any(not torch.equal(frozen[k],v) for k,v in model.state_dict().items()):raise RuntimeError('NeRF weights changed')
  good=[r for r in rows if r.get('pnp_ok')]; metric=lambda key:{'median':float(np.median([r[key] for r in good])),'mean':float(np.mean([r[key] for r in good])),'p90':float(np.percentile([r[key] for r in good],90))}
  report={'method':'B5/B6 PnP + frozen-NeRF RGB + Huber 2D-3D reprojection + PnP prior; held-out geometry acceptance','checkpoint':str(ck),'protocol_sha256':protocol['protocol_sha256'],'steps':steps,'accepted':sum(r['source']=='geometry_guided_accepted' for r in rows),'pnp_fallback':sum(r['source']=='pnp_fallback' for r in rows),'retrieval_fallback':sum(r['source']=='retrieval_fallback' for r in rows),'pnp':{'translation_m':metric('pnp_translation_error_m'),'rotation_deg':metric('pnp_rotation_error_deg')},'final':{'translation_m':metric('final_translation_error_m'),'rotation_deg':metric('final_rotation_error_deg')},'frames':rows}
  save_json(output/'hybrid_report.json',report);return report
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('workspace'));p.add_argument('--run-name',default='m3_e3_capacity_8x192_12k');p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=80);a=p.parse_args();print(json.dumps(evaluate(a.workspace.resolve(),a.run_name,a.output.resolve(),a.cache.resolve(),a.steps),indent=2))
if __name__=='__main__':main()
