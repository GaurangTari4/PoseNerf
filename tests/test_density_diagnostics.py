import unittest
import torch
from posenerf.density import termination_statistics

class DensityDiagnosticsTests(unittest.TestCase):
    def test_normalized_expected_depth_and_variance(self):
        z=torch.tensor([[1.,2.,3.]]); w=torch.tensor([[.2,.6,.2]])
        s=termination_statistics(z,w); self.assertAlmostEqual(float(s['expected_depth']),2.0); self.assertAlmostEqual(float(s['variance']),.4); self.assertEqual(int(s['peak_count']),1)
    def test_low_opacity_is_explicit_and_finite(self):
        s=termination_statistics(torch.tensor([[1.,2.]]),torch.zeros(1,2)); self.assertTrue(bool(s['low_opacity'])); self.assertTrue(torch.isfinite(s['entropy']).all()); self.assertAlmostEqual(float(s['background_weight']),1.0)
