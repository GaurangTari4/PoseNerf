"""Matched measured-depth and frozen-NeRF-depth localization benchmark."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from .config import build_paths
from .data import load_chess_dataset
from .milestone3 import _checkpoint_for,_safe_run,load_run_config
from .milestone9 import ReferenceDatabase,backproject_pixels,build_reference_database,c2w_points,camera_matrix,match_candidates,pnp_ransac,render_feature_geometry,_frame_id
from .model import HierarchicalNeRF
from .pose import retrieve_initial_pose,rotation_error_deg,translation_error_m
from .utils import get_device,save_json,seed_everything

def _load_model(workspace,name,device):
 run=_safe_run(workspace,name);cfg=load_run_config(run);checkpoint,payload=_checkpoint_for(run,'hierarchical_nerf_best.pth',cfg,device);model=HierarchicalNeRF(cfg).to(device);model.load_state_dict(payload['model_state_dict']);model.eval();return cfg,model,checkpoint,hashlib.sha256(checkpoint.read_bytes()).hexdigest()

def nerf_database(measured, data, render_cfg, model, checkpoint, checkpoint_hash, cache, device):
 """D1/D2: exact D0 feature descriptors/pixels, only points are NeRF-derived."""
 if cache.exists(): raise FileExistsError(cache)
 start=time.perf_counter(); K=camera_matrix(render_cfg); desc=[]; pix=[]; points=[]; per=[]; depth_errors=[]; total=valid=0
 for i,(d,p) in enumerate(zip(measured.descriptors,measured.pixels)):
  z,mask=render_feature_geometry(model,data.train_poses[i],p,render_cfg,device)
  total+=len(p);valid+=int(mask.sum());desc.append(d[mask]);pix.append(p[mask]);points.append(c2w_points(backproject_pixels(p[mask],z[mask],K),data.train_poses[i].numpy()).astype(np.float32));per.append({'frame_id':measured.frame_ids[i],'features':len(p),'valid':int(mask.sum()),'invalid':int((~mask).sum())})
  # Evaluation-only: D0 is read only after all NeRF geometry has been rendered.
  if mask.any():
   observed_world=measured.world_points[i][mask]; pose=np.linalg.inv(data.train_poses[i].numpy()); observed_z=(observed_world@pose[:3,:3].T+pose[:3,3])[:,2]; depth_errors.extend((z[mask]-observed_z).tolist())
 manifest={'source':'frozen_nerf_rendered_depth','checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'frames':measured.frame_ids,'feature_detector':measured.manifest['algorithm'],'intrinsics':K.tolist(),'opacity_min':.5,'features_total':total,'features_valid':valid,'features_invalid':total-valid,'valid_fraction':valid/max(total,1),'build_seconds':time.perf_counter()-start,'depth_vs_measured_evaluation_only':{'mae_m':float(np.mean(np.abs(depth_errors))),'rmse_m':float(np.sqrt(np.mean(np.square(depth_errors)))),'bias_m':float(np.mean(depth_errors)),'count':len(depth_errors)},'per_frame':per}
 cache.mkdir(parents=True);np.savez_compressed(cache/'geometry.npz',descriptors=np.concatenate(desc),pixels=np.concatenate(pix),world_points=np.concatenate(points),offsets=np.cumsum([0]+[len(x) for x in desc]));save_json(cache/'manifest.json',manifest)
 return ReferenceDatabase(measured.frame_ids,desc,pix,points,manifest)

def _metrics(rows,key):
 prefix=f'{key}_' if key else ''
 t=np.asarray([r[f'{prefix}translation_error_m'] for r in rows]);q=np.asarray([r[f'{prefix}rotation_error_deg'] for r in rows]);return {'translation_m':{'median':float(np.median(t)),'mean':float(np.mean(t)),'p90':float(np.percentile(t,90))},'rotation_deg':{'median':float(np.median(q)),'mean':float(np.mean(q)),'p90':float(np.percentile(q,90))},'success_5cm_5deg':float(np.mean((t<=.05)&(q<=5))),'success_10cm_10deg':float(np.mean((t<=.1)&(q<=10)))}

def benchmark(data, scene, databases, config, seed):
 rows=[];K=camera_matrix(config)
 for i,frame in enumerate(data.val_frames):
  rgb=data.val_images[i];retr,nearest=retrieve_initial_pose(rgb,data.train_images,data.train_poses);errors=(data.train_images-rgb).square().mean((1,2,3)).numpy();candidate=np.argsort(errors)[:5].tolist();row={'query_frame_id':_frame_id(frame,scene),'retrieval_frame_id':databases['D0'].frame_ids[nearest],'retrieval_translation_error_m':translation_error_m(retr,data.val_poses[i]),'retrieval_rotation_error_deg':rotation_error_deg(retr,data.val_poses[i]),'methods':{}}
  for label,db in databases.items():
   started=time.perf_counter();qp,wp,_,_,detail,features=match_candidates(rgb,db,candidate);p=pnp_ransac(qp,wp,K,seed+i);estimate=retr if not p['ok'] else torch.from_numpy(p['pose']);row['methods'][label]={'pnp_ok':p['ok'],'fallback':not p['ok'],'reason':p.get('reason'),'features':features,'raw_matches':sum(x['raw_matches'] for x in detail),'correspondences':len(qp),'inliers':p.get('inlier_count',0),'inlier_fraction':p.get('inlier_fraction',0.),'rmse_px':p.get('reprojection_rmse_px'),'translation_error_m':translation_error_m(estimate,data.val_poses[i]),'rotation_error_deg':rotation_error_deg(estimate,data.val_poses[i]),'runtime_seconds':time.perf_counter()-started}
  rows.append(row)
 report={'frames':rows,'retrieval':_metrics(rows,'retrieval')}
 for label in databases:
  flat=[dict(x['methods'][label]) for x in rows]
  report[label]={'metrics':_metrics(flat,''),'pnp_success':sum(x['pnp_ok'] for x in flat),'fallback':sum(x['fallback'] for x in flat),'mean_correspondences':float(np.mean([x['correspondences'] for x in flat])),'mean_inlier_fraction':float(np.mean([x['inlier_fraction'] for x in flat])),'mean_rmse_px':float(np.nanmean([x['rmse_px'] if x['rmse_px'] is not None else np.nan for x in flat]))}
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('workspace'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();out=a.output.resolve();
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);dev=get_device();seed_everything(612);cfg,m3,c3,h3=_load_model(w,'m3_e3_capacity_8x192_12k',dev);cfg6,m6,c6,h6=_load_model(w,'m5b_e6_capacity_8x384_12k',dev);data=load_chess_dataset(cfg,build_paths(w),dev,storage_device=torch.device('cpu'),include_test=False);scene=build_paths(w).scene_dir;d0=build_reference_database(data,cfg,scene,w/'outputs'/'m9_geometry_reference_e3_20261105');d1=nerf_database(d0,data,cfg,m3,c3,h3,out/'d1_e3',dev);d2=nerf_database(d0,data,cfg6,m6,c6,h6,out/'d2_e6',dev);report=benchmark(data,scene,{'D0':d0,'D1_E3':d1,'D2_E6':d2},cfg,612);report.update({'protocol_sha256':json.loads((_safe_run(w,'m3_e3_capacity_8x192_12k')/'frozen_protocol.json').read_text())['protocol_sha256'],'databases':{'D0':d0.manifest,'D1_E3':d1.manifest,'D2_E6':d2.manifest},'cuda_peak_mb':float(torch.cuda.max_memory_allocated(dev)/2**20) if dev.type=='cuda' else 0.});save_json(out/'milestone10_report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
