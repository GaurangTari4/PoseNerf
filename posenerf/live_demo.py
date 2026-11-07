"""One-view executable rendering and RGB-only localization demonstration."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from .config import build_paths
from .data import load_chess_dataset
from .milestone3 import _checkpoint_for,_safe_run,load_run_config
from .milestone9 import build_reference_database,camera_matrix,match_candidates,pnp_ransac,_frame_id,_save_matches
from .model import HierarchicalNeRF,get_ray_directions,render_image
from .pose import retrieve_initial_pose,rotation_error_deg,translation_error_m
from .utils import get_device,save_json,seed_everything
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('workspace'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();o=a.output.resolve()
 if o.exists():raise FileExistsError(o)
 o.mkdir(parents=True);dev=get_device();seed_everything(612);run=_safe_run(w,'m3_e3_capacity_8x192_12k');cfg=load_run_config(run);data=load_chess_dataset(cfg,build_paths(w),dev,storage_device=torch.device('cpu'),include_test=False);ck,payload=_checkpoint_for(run,'hierarchical_nerf_best.pth',cfg,dev);model=HierarchicalNeRF(cfg).to(dev);model.load_state_dict(payload['model_state_dict']);model.eval();weights={k:v.clone() for k,v in model.state_dict().items()};dirs=get_ray_directions(cfg,dev);q=data.val_images[0].to(dev);gt=data.val_poses[0].to(dev)
 start=time.perf_counter();pred,depth=render_image(model,gt,dirs,cfg);torch.cuda.synchronize(dev) if dev.type=='cuda' else None;latency=time.perf_counter()-start;mse=float(F.mse_loss(pred,q));psnr=-10*np.log10(max(mse,1e-12));fig,ax=plt.subplots(1,4,figsize=(14,3));items=[(q,'Known-pose reference RGB'),(pred,'E3 predicted RGB'),((q-pred).abs(),'Absolute RGB error'),(depth,'Predicted depth')]
 for a1,(im,title) in zip(ax,items):a1.imshow(im.detach().cpu().numpy(),cmap='magma' if title=='Predicted depth' else None);a1.set_title(title);a1.axis('off')
 fig.tight_layout();fig.savefig(o/'known_pose_e3_render.png',dpi=160);plt.close(fig)
 scene=build_paths(w).scene_dir;cache=w/'outputs'/'m9_geometry_reference_e3_20261105';db=build_reference_database(data,cfg,scene,cache);retr,nearest=retrieve_initial_pose(q,data.train_images.to(dev),data.train_poses.to(dev));scores=(data.train_images-q.cpu()).square().mean((1,2,3)).numpy();cand=np.argsort(scores)[:5].tolist();qp,wp,rp,src,detail,features=match_candidates(q.cpu(),db,cand);result=pnp_ransac(qp,wp,camera_matrix(cfg),612);est=retr if not result['ok'] else torch.from_numpy(result['pose']).to(dev);first=cand[0];keep=src==first;inliers=result.get('inliers');local=None if inliers is None else np.flatnonzero(np.isin(np.flatnonzero(keep),inliers));_save_matches(o/'e3_nerf_depth_pnp_matches.png',q,data.train_images[first],qp[keep],rp[keep],local)
 fig,ax=plt.subplots(1,2,figsize=(7,3));ax[0].imshow(q.cpu());ax[0].set_title('Validation query RGB');ax[1].imshow(data.train_images[nearest]);ax[1].set_title('RGB retrieval reference');[x.axis('off') for x in ax];fig.tight_layout();fig.savefig(o/'query_reference.png',dpi=160);plt.close(fig)
 report={'query_frame_id':_frame_id(data.val_frames[0],scene),'retrieval_frame_id':_frame_id(data.train_frames[nearest],scene),'checkpoint':str(ck),'poses_evaluation_only':{'retrieval_c2w':retr.cpu().tolist(),'pnp_c2w':est.cpu().tolist(),'gt_c2w':gt.cpu().tolist()},'known_pose_render':{'psnr_db':psnr,'mse':mse,'latency_seconds':latency},'rgb_only_pnp':{'ok':result['ok'],'features':features,'correspondences':len(qp),'inliers':result.get('inlier_count',0),'rmse_px':result.get('reprojection_rmse_px'),'translation_error_m_evaluation_only':translation_error_m(est,gt),'rotation_error_deg_evaluation_only':rotation_error_deg(est,gt),'fallback':not result['ok']}}
 if any(not torch.equal(weights[k],v) for k,v in model.state_dict().items()):raise RuntimeError('checkpoint mutation')
 save_json(o/'live_demo_report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
