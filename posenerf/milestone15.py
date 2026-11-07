"""M15-A: frozen four-model RGB-only localization foundation (no test inference)."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from .config import build_paths
from .data import load_chess_dataset
from .milestone3 import load_run_config
from .milestone9 import build_reference_database
from .milestone10 import nerf_database
from .model import HierarchicalNeRF
from .utils import get_device, safe_torch_load, save_json, seed_everything

MODELS={
 "E7_final":("m12_e7_extended_8x384_48k","hierarchical_nerf_trained.pth",8,384),
 "E8_rgb_best":("m13_e8_manual_10x512_48k","hierarchical_nerf_trained.pth",10,512),
 "E8_D1":("m14_e8_d1_lrfix_500","hierarchical_nerf_continuation_trained.pth",10,512),
 "E8_D2":("m14_e8_d2_lrfix_500","hierarchical_nerf_continuation_trained.pth",10,512),
}
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def registry(workspace:Path):
 out={}
 for label,(run,file,layers,hidden) in MODELS.items():
  path=workspace/'outputs'/run/'checkpoints'/file
  if not path.is_file():raise FileNotFoundError(f'{label} checkpoint missing: {path}')
  p=safe_torch_load(path,torch.device('cpu')); cfg=p.get('config',{}).get('model',{})
  if cfg.get('layers')!=layers or cfg.get('hidden')!=hidden or 'model_state_dict' not in p:raise ValueError(f'{label} checkpoint architecture mismatch')
  if label.startswith('E8_D') and p.get('export_kind')!='terminal_continuation_state':raise ValueError(f'{label} is not a terminal continuation export')
  out[label]={'run':run,'checkpoint':str(path.resolve()),'sha256':sha(path),'layers':layers,'hidden':hidden,'parameters':sum(x.numel() for x in p['model_state_dict'].values()),'export_kind':p.get('export_kind')}
 return out
def reference_manifest(workspace:Path):
 cfg=load_run_config(workspace/'outputs'/MODELS['E8_rgb_best'][0]); paths=build_paths(workspace);data=load_chess_dataset(cfg,paths,torch.device('cpu'),storage_device=torch.device('cpu'),include_test=False)
 ids=sorted(str(f.color_path.relative_to(paths.scene_dir)).replace('\\','/') for f in data.train_frames)
 if any(not x.startswith(('seq-01/','seq-02/','seq-04/','seq-06/')) for x in ids):raise ValueError('reference list is not restricted to authorized training sequences')
 return {'dataset':'Microsoft 7-Scenes Chess','authorized_sequences':['01','02','04','06'],'official_test_excluded':['03','05'],'frame_ids':ids,'frame_list_sha256':hashlib.sha256(canonical(ids).encode()).hexdigest(),'intrinsics':{'fx':cfg.scene.fx,'fy':cfg.scene.fy,'cx':cfg.scene.cx,'cy':cfg.scene.cy},'pose_convention':'OpenCV +Z camera coordinates; c2w training-reference poses'}
def protocol(reg,refs):
 p={'version':'M15-A-v1','models':reg,'reference_frame_sha256':refs['frame_list_sha256'],'reference_geometry':{'coarse':48,'fine':48,'opacity_min':.5,'format':'nerf-depth-sift-worldpoints-v1'},'retrieval':{'top_k':5,'metric':'RGB MSE'},'matching':{'feature':'SIFT','nfeatures':500,'contrast_threshold':.01,'bf':'L2','mutual_ratio':.75},'pnp':{'iterations':300,'reprojection_px':2.5,'confidence':.999,'flag':'EPNP','inlier_fraction_min':.15},'fallback':'RGB retrieval pose','success_thresholds':['5cm/5deg','10cm/10deg'],'query_contract':'RGB only; GT pose/depth unavailable to inference'};p['sha256']=hashlib.sha256(canonical(p).encode()).hexdigest();return p
def smoke_database(workspace:Path,label:str,out:Path,limit:int):
 reg=registry(workspace)[label];run=workspace/'outputs'/reg['run'];cfg=load_run_config(run);dev=get_device();data=load_chess_dataset(cfg,build_paths(workspace),dev,storage_device=torch.device('cpu'),include_test=False);base=build_reference_database(data,cfg,build_paths(workspace).scene_dir,out/'feature_cache');model=HierarchicalNeRF(cfg).to(dev);payload=safe_torch_load(Path(reg['checkpoint']),dev);model.load_state_dict(payload['model_state_dict']);model.eval()
 if limit<len(base.frame_ids):
  base=type(base)(base.frame_ids[:limit],base.descriptors[:limit],base.pixels[:limit],base.world_points[:limit],base.manifest)
 db=nerf_database(base,data,cfg,model,Path(reg['checkpoint']),reg['sha256'],out/'geometry',dev);m=db.manifest|{'model_id':label,'reference_frame_sha256':hashlib.sha256(canonical(base.frame_ids).encode()).hexdigest(),'smoke_only':limit<len(data.train_frames)};save_json(out/'geometry'/'m15_manifest.json',m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,default=Path('workspace'));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke-model',choices=tuple(MODELS));ap.add_argument('--smoke-frames',type=int,default=2);a=ap.parse_args();w=a.workspace.resolve();o=a.output.resolve();o.mkdir(parents=True,exist_ok=True);r=registry(w);refs=reference_manifest(w);p=protocol(r,refs);save_json(o/'checkpoint_registry.json',r);save_json(o/'reference_frame_manifest.json',refs);save_json(o/'protocol.json',p);(o/'protocol_sha256.txt').write_text(p['sha256']+'\n');
 if a.smoke_model:save_json(o/f'{a.smoke_model}_smoke.json',smoke_database(w,a.smoke_model,o/a.smoke_model,a.smoke_frames))
 print(json.dumps({'protocol_sha256':p['sha256'],'models':list(r)},indent=2))
if __name__=='__main__':main()
