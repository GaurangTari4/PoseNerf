import unittest


class ThermalPacingTests(unittest.TestCase):
    def test_pacing_is_recorded_in_checkpoint_compatibility_signature(self):
        from posenerf.config import ExperimentConfig
        baseline = ExperimentConfig(); paced = ExperimentConfig()
        paced.train.optimizer_pause_seconds = 0.5
        self.assertNotEqual(baseline.signature(), paced.signature())

    def test_pacing_occurs_after_optimizer_and_scheduler_updates(self):
        source = open("posenerf/training.py", encoding="utf-8").read()
        self.assertLess(source.index("scheduler.step()"), source.index("time.sleep(train_cfg.optimizer_pause_seconds)"))

    def test_validation_uses_inference_mode_without_changing_protocol(self):
        source = open("posenerf/training.py", encoding="utf-8").read()
        self.assertIn("@torch.inference_mode()\ndef evaluate_model", source)
