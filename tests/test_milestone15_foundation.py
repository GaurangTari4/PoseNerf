import unittest
from posenerf.milestone15 import MODELS, canonical, protocol

class M15FoundationTests(unittest.TestCase):
 def test_four_model_registry_complete(self):self.assertEqual(set(MODELS),{'E7_final','E8_rgb_best','E8_D1','E8_D2'})
 def test_protocol_deterministic(self):
  r={k:{'sha256':k} for k in MODELS};f={'frame_list_sha256':'frames'};self.assertEqual(protocol(r,f)['sha256'],protocol(r,f)['sha256'])
 def test_canonical_ordering(self):self.assertEqual(canonical({'b':1,'a':2}),'{"a":2,"b":1}')
