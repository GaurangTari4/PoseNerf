import tempfile
import unittest
from pathlib import Path
import torch

from posenerf.config import ExperimentConfig, build_paths, ensure_directories
from posenerf.data import LoadedDataset
from posenerf.model import HierarchicalNeRF, get_ray_directions
from posenerf.training import train_model
from posenerf.utils import capture_rng_state, safe_torch_load, seed_everything


class E8ContinuationResumeTests(unittest.TestCase):
    def assert_state_equal(self,a,b):
        if torch.is_tensor(a): self.assertTrue(torch.equal(a,b)); return
        if isinstance(a,dict): self.assertEqual(a.keys(),b.keys()); [self.assert_state_equal(a[k],b[k]) for k in a]; return
        if isinstance(a,(list,tuple)): self.assertEqual(len(a),len(b)); [self.assert_state_equal(x,y) for x,y in zip(a,b)]; return
        self.assertEqual(a,b)
    def config(self):
        c=ExperimentConfig(seed=91);c.scene.orig_w=c.scene.orig_h=4;c.scene.orig_fx=c.scene.orig_fy=2;c.scene.orig_cx=c.scene.orig_cy=2;c.scene.downsample=1
        c.model.hidden,c.model.layers,c.model.pos_freqs,c.model.dir_freqs=8,4,2,1;c.train.batch_rays,c.train.accumulation_steps=8,2;c.train.num_coarse_samples=c.train.num_fine_samples=4;c.train.num_steps=4;c.train.checkpoint_every=1;c.train.val_render_interval=100;c.train.val_frame_count=1;c.train.use_amp=False;c.train.resume=True
        return c
    def data(self):
        x=torch.rand(2,4,4,3);d=torch.ones(2,4,4);m=torch.ones(2,4,4);p=torch.eye(4).repeat(2,1,1)
        return LoadedDataset([],[],[],x,d,m,p,x[:1],d[:1],m[:1],p[:1],x[:1],d[:1],m[:1],p[:1])
    def source(self,c,path):
        torch.manual_seed(1234)
        model=HierarchicalNeRF(c);opt=torch.optim.Adam(model.parameters(),lr=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=4)
        state=sched.state_dict();state['last_epoch']=48000;state['_last_lr']=[4e-5];sched.load_state_dict(state)
        seed_everything(c.seed); g=torch.Generator(device='cpu').manual_seed(c.seed)
        torch.save({'step':0,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sched.state_dict(),'scaler_state_dict':{},'sampler_generator_state':g.get_state(),'rng_state':capture_rng_state(),'config':c.to_dict(),'config_signature':c.signature(),'optimizer_step_count':0},path)
    def execute_continuation(self,root,interrupt):
        c=self.config();seed_everything(c.seed);d=self.data();p=build_paths(root);ensure_directories(p);src=root/'source.pth';self.source(c,src);seed_everything(c.seed);m=HierarchicalNeRF(c);b=HierarchicalNeRF(c);dirs=get_ray_directions(c,torch.device('cpu'));cont={'type':'cosine','successful_updates':3,'start_lr':4e-5,'final_lr':1e-5,'source_sha256':'x','resume_existing':False}
        if interrupt:(p.outputs_dir/'graceful_stop.request').write_text('test')
        train_model(m,b,d,dirs,c,p,torch.device('cpu'),external_resume=src,continuation=cont)
        if interrupt:
            (p.outputs_dir/'graceful_stop.request').unlink();cont['resume_existing']=True
            train_model(m,b,d,dirs,c,p,torch.device('cpu'),external_resume=p.checkpoints_dir/'hierarchical_nerf_resume_latest.pth',continuation=cont)
        return safe_torch_load(p.checkpoints_dir/'hierarchical_nerf_continuation_trained.pth',torch.device('cpu'))
    def test_split_resume_matches_uninterrupted_continuation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as td:
            a=self.execute_continuation(Path(td)/'a',False);b=self.execute_continuation(Path(td)/'b',True)
            for k in a['model_state_dict']:self.assertTrue(torch.equal(a['model_state_dict'][k],b['model_state_dict'][k]))
            self.assert_state_equal(a['optimizer_state_dict'],b['optimizer_state_dict']);self.assert_state_equal(a['scheduler_state_dict'],b['scheduler_state_dict'])
            for k in ('continuation_success_updates','attempted_iterations','skipped_amp_updates'):self.assertEqual(a[k],b[k])
            ca,cb=dict(a['continuation']),dict(b['continuation']);ca.pop('resume_existing',None);cb.pop('resume_existing',None);self.assertEqual(ca,cb)
            self.assertEqual(a['continuation_success_updates'],3);self.assertEqual(a['skipped_amp_updates'],0)
            self.assertAlmostEqual(a['optimizer_state_dict']['param_groups'][0]['lr'],1e-5,places=10)
            self.assertEqual(a['export_kind'],'terminal_continuation_state')
            terminal=safe_torch_load(Path(td)/'a'/'outputs'/'checkpoints'/'hierarchical_nerf_resume_latest.pth',torch.device('cpu'))
            for key in a['model_state_dict']: self.assertTrue(torch.equal(a['model_state_dict'][key],terminal['model_state_dict'][key]))
