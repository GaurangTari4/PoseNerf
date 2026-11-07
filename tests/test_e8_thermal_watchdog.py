import importlib.util
import unittest
from pathlib import Path


def module():
    spec = importlib.util.spec_from_file_location("e8_watchdog", Path("scripts/e8_thermal_watchdog.py"))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


class E8ThermalWatchdogTests(unittest.TestCase):
    def test_warning_does_not_request_stop(self):
        decision, count = module().thermal_decision(85.0, 0, warning_c=85, critical_c=88, emergency_c=90)
        self.assertEqual((decision, count), ("warning_temperature", 0))

    def test_critical_requires_three_readings(self):
        watchdog = module(); count = 0
        for expected in ("critical_warning", "critical_warning", "critical_temperature"):
            decision, count = watchdog.thermal_decision(88.0, count, warning_c=85, critical_c=88, emergency_c=90)
            self.assertEqual(decision, expected)

    def test_emergency_is_immediate(self):
        decision, count = module().thermal_decision(90.0, 0, warning_c=85, critical_c=88, emergency_c=90)
        self.assertEqual(decision, "emergency_temperature")
        self.assertEqual(count, 0)
