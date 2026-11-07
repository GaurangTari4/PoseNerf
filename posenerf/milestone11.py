"""Frozen-protocol official Chess test evaluation for M10 databases."""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from .config import build_paths
from .data import collect_frames,load_chess_dataset,load_pose,load_rgb,read_split_file
from .milestone3 import _safe_run,load_run_config
from .milestone9 import ReferenceDatabase,camera_matrix,match_candidates,pnp_ransac,_frame_id
from .milestone10 import _metrics
from .pose import retrieve_initial_pose,rotation_error_deg,translation_error_m
from .utils import get_device,save_json,seed_everything

def load_db(cache,ids):
 path=cache/'geometry.npz'
 if not path.exists(): path=cache/'reference_geometry.npz'
 p=np.load(path,allow_pickle=False);o=p['offsets'];m=json.loads((cache/'manifest.json').read_text());return ReferenceDatabase(ids,[p['descriptors'][o[i]:o[i+1]] for i in range(len(ids))],[p['pixels'][o[i]:o[i+1]] for i in range(len(ids))],[p['world_points'][o[i]:o[i+1]] for i in range(len(ids))],m)
def _append(path, item):
 with path.open('a',encoding='utf8') as f: f.write(json.dumps(item,separators=(',',':'))+'\n');f.flush();__import__('os').fsync(f.fileno())
def _journal(path, protocol_hash):
 done={}
 if not path.exists(): return done
 for n,line in enumerate(path.read_text(encoding='utf8').splitlines(),1):
  try:x=json.loads(line)
  except json.JSONDecodeError: raise ValueError(f'malformed journal line {n}')
  if x.get('protocol_hash')!=protocol_hash or not x.get('query_frame_id') or x['query_frame_id'] in done:raise ValueError('journal identity mismatch or duplicate')
  done[x['query_frame_id']]=x
 return done
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,default=Path('workspace'));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');a=ap.parse_args();w=a.workspace.resolve();out=a.output.resolve();
 if out.exists() and not a.resume:raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=True);seed=612;run=_safe_run(w,'m3_e3_capacity_8x192_12k');cfg=load_run_config(run);m10=w/'outputs'/'m10_nerf_geometry_20261105';m10r=json.loads((m10/'milestone10_report.json').read_text());protocol={'seed':seed,'source_m10_report_sha256':hashlib.sha256((m10/'milestone10_report.json').read_bytes()).hexdigest(),'training_frame_ids':m10r['databases']['D1_E3']['frames'],'intrinsics':m10r['databases']['D1_E3']['intrinsics'],'matching':'SIFT BF L2 mutual ratio .75; RGB retrieval top five; fixed solvePnPRansac EPNP 300, 2.5px, .999','databases':{k:m10r['databases'][k] for k in ['D0','D1_E3','D2_E6']},'checkpoint_sha256':{k:m10r['databases'][k].get('checkpoint_sha256') for k in ['D1_E3','D2_E6']},'official_test_sequences':['sequence03','sequence05'],'loader':'streaming RGB-only v1'};ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
 if (out/'frozen_protocol.json').exists() and json.loads((out/'frozen_protocol.json').read_text())!=protocol:raise ValueError('frozen protocol mismatch')
 save_json(out/'frozen_protocol.json',protocol)
 dev=get_device();seed_everything(seed);data=load_chess_dataset(cfg,build_paths(w),dev,storage_device=torch.device('cpu'),include_test=False);scene=build_paths(w).scene_dir;ids=protocol['training_frame_ids'];d0=load_db(w/'outputs'/'m9_geometry_reference_e3_20261105',ids);d1=load_db(m10/'d1_e3',ids);d2=load_db(m10/'d2_e6',ids);dbs={'D0':d0,'D1_E3':d1,'D2_E6':d2};test=collect_frames(scene,read_split_file(build_paths(w).scene_dir/'TestSplit.txt'));test=test if a.limit is None else test[:a.limit];K=camera_matrix(cfg);journal=out/'results.jsonl';done=_journal(journal,ph);rows=list(done.values())
 for frame in test:
  qid=_frame_id(frame,scene)
  if qid in done:continue
  rgb=torch.from_numpy(load_rgb(frame.color_path,cfg.scene.downsample)).float();retr,nearest=retrieve_initial_pose(rgb,data.train_images,data.train_poses);e=(data.train_images-rgb.unsqueeze(0)).square().mean((1,2,3)).numpy();cand=np.argsort(e)[:5].tolist();methods={}
  for name,db in dbs.items():
   t=time.perf_counter();qp,wp,_,_,detail,features=match_candidates(rgb,db,cand);p=pnp_ransac(qp,wp,K,seed+len(rows));est=retr if not p['ok'] else torch.from_numpy(p['pose'])
   methods[name]={'pnp_ok':p['ok'],'fallback':not p['ok'],'reason':p.get('reason'),'features':features,'correspondences':len(qp),'inliers':p.get('inlier_count',0),'inlier_fraction':p.get('inlier_fraction',0.),'rmse_px':p.get('reprojection_rmse_px'),'estimated_pose':est.tolist(),'runtime_seconds':time.perf_counter()-t}
  gt=torch.from_numpy(load_pose(frame.pose_path)) # evaluation-only after all predictions
  for method in methods.values():
   estimate=torch.tensor(method.pop('estimated_pose'));method['translation_error_m']=translation_error_m(estimate,gt);method['rotation_error_deg']=rotation_error_deg(estimate,gt)
  row={'protocol_hash':ph,'query_frame_id':qid,'sequence':frame.sequence,'status':'complete','retrieval_translation_error_m':translation_error_m(retr,gt),'retrieval_rotation_error_deg':rotation_error_deg(retr,gt),'methods':methods};_append(journal,row);rows.append(row)
 peak = float(torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == 'cuda' else 0.0
 report = {'protocol': protocol, 'protocol_hash': ph, 'queries_available': len(test), 'queries_evaluated': len(rows), 'complete': len(rows) == len(test), 'retrieval': _metrics(rows, 'retrieval'), 'frames': rows, 'cuda_peak_mb': peak}
 for name in dbs:
  flat=[x['methods'][name] for x in rows];report[name]={'metrics':_metrics(flat,''),'pnp_success':sum(x['pnp_ok'] for x in flat),'fallback':sum(x['fallback'] for x in flat),'by_sequence':{s:_metrics([x['methods'][name] for x in rows if x['sequence']==s],'') for s in sorted(set(x['sequence'] for x in rows))}}
 save_json(out/'m11_report.json',report);print(json.dumps({'queries':len(rows),'D0':report['D0'],'D1_E3':report['D1_E3'],'D2_E6':report['D2_E6']},indent=2))
if __name__=='__main__':main()
