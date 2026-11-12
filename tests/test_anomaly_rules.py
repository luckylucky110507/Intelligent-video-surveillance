import unittest

from anomaly import detect_rule_anomaly


class RuleAnomalyTests(unittest.TestCase):
    def test_threshold_values_do_not_trigger(self):
        self.assertEqual((False, []), detect_rule_anomaly(3, 2.5))

    def test_crowd_and_high_motion_are_reported(self):
        self.assertEqual(
            (True, ["crowd", "high_motion"]),
            detect_rule_anomaly(4, 2.6),
        )


if __name__ == "__main__":
    unittest.main()
