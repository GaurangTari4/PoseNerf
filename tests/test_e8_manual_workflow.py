import json
import tempfile
import unittest
from pathlib import Path

from posenerf.telemetry import RunTelemetry


class E8ManualWorkflowTests(unittest.TestCase):
    def test_manual_scripts_and_every_1000_snapshot_schedule_exist(self):
        start = Path("scripts/start_e8_manual.ps1").read_text(encoding="utf-8")
        for name in ("start_e8_manual.ps1", "monitor_e8_manual.ps1", "stop_e8_manual.ps1", "resume_e8_manual.ps1"):
            self.assertTrue((Path("scripts") / name).is_file())
        for step in range(1000, 48001, 1000): self.assertIn(str(step), start)
        self.assertIn("--steps 48000", start)

    def test_audit_log_appends_without_rewriting_history(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw:
            root = Path(raw); telemetry = RunTelemetry(root / "live", audit_dir=root / "e8_logs")
            telemetry.publish("train_step", 1, loss=1.0, lr=0.1, cuda_allocated_mb=1.0, cuda_reserved_mb=2.0)
            telemetry.publish("train_step", 2, loss=0.5, lr=0.1, cuda_allocated_mb=1.0, cuda_reserved_mb=2.0)
            events = [json.loads(line) for line in (root / "e8_logs" / "training_events.jsonl").read_text().splitlines()]
            self.assertEqual([item["step"] for item in events], [1, 2])
            self.assertEqual(len((root / "e8_logs" / "training_metrics.csv").read_text().splitlines()), 3)
