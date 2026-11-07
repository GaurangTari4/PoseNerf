import unittest
from pathlib import Path

class E8FreshStartSafetyTests(unittest.TestCase):
    def test_source_hash_is_computed_before_e8_comparison(self):
        text=Path('posenerf/stage_c.py').read_text(encoding='utf8')
        self.assertLess(text.index('source_hash = hashlib.sha256(args.resume_from.read_bytes()).hexdigest()'),text.index('args.source_sha256.lower() != source_hash.lower()'))
    def test_monitor_handles_uninitialized_metrics(self):
        text=Path('scripts/monitor_e8_finetune.ps1').read_text(encoding='utf8')
        self.assertIn('Metrics not initialized',text)
