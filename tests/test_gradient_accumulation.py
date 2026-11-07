import unittest
import torch


class GradientAccumulationTests(unittest.TestCase):
    def test_four_microbatches_match_full_batch_gradient_and_update(self):
        torch.manual_seed(9); x=torch.randn(1024,3); y=torch.randn(1024,2)
        a=torch.nn.Linear(3,2); b=torch.nn.Linear(3,2); b.load_state_dict(a.state_dict())
        oa=torch.optim.SGD(a.parameters(),lr=.01); ob=torch.optim.SGD(b.parameters(),lr=.01)
        oa.zero_grad(); torch.nn.functional.mse_loss(a(x),y).backward(); oa.step()
        ob.zero_grad()
        for chunk_x,chunk_y in zip(x.chunk(4),y.chunk(4)):
            (torch.nn.functional.mse_loss(b(chunk_x),chunk_y)/4).backward()
        ob.step()
        for left,right in zip(a.parameters(),b.parameters()): self.assertTrue(torch.allclose(left,right,atol=1e-7,rtol=0))

    def test_accumulation_is_part_of_compatibility_signature(self):
        from posenerf.config import ExperimentConfig
        a=ExperimentConfig(); b=ExperimentConfig(); b.train.accumulation_steps=4
        self.assertNotEqual(a.signature(),b.signature())
