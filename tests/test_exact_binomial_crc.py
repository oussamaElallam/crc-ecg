import unittest

import numpy as np

from crc import MultiLabelCRC


class ExactBinomialCRCTests(unittest.TestCase):
    def test_zero_miss_upper_bound_matches_closed_form(self):
        n = 72
        delta = 0.05
        observed = MultiLabelCRC.exact_upper_bound(0, n, delta)
        expected = 1.0 - delta ** (1.0 / n)
        self.assertAlmostEqual(observed, expected, places=12)

    def test_upper_bound_increases_with_misses(self):
        bounds = [MultiLabelCRC.exact_upper_bound(m, 100, 0.05) for m in range(6)]
        self.assertTrue(all(a < b for a, b in zip(bounds, bounds[1:])))

    def test_simultaneous_delta_allocation(self):
        probs = np.array([[0.9], [0.8], [0.7], [0.6], [0.5]])
        labels = np.ones_like(probs, dtype=np.int8)
        model = MultiLabelCRC(confidence=0.95, simultaneous=True, family_size=9)
        model.calibrate(probs, labels, {0: 0.8}, controlled_classes=[0])
        self.assertAlmostEqual(model.calibration_info[0]["delta"], 0.05 / 9)

    def test_guard_is_never_less_inclusive(self):
        probs = np.linspace(0.1, 0.99, 200).reshape(-1, 1)
        labels = np.ones_like(probs, dtype=np.int8)
        model = MultiLabelCRC(confidence=0.95, simultaneous=False)
        model.calibrate(probs, labels, {0: 0.2}, controlled_classes=[0])
        original = model.lambdas[0]
        guarded = model.guarded_copy({0: min(1.0, original + 0.05)}, probs, labels)
        self.assertGreaterEqual(guarded.lambdas[0], original)
        self.assertLessEqual(
            guarded.calibration_info[0]["exact_ucb"],
            model.calibration_info[0]["exact_ucb"] + 1e-12,
        )

    def test_metrics_name_macro_class_coverage(self):
        pred = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8)
        truth = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
        metrics = MultiLabelCRC.compute_metrics(pred, truth, ["A", "B"])
        self.assertIn("macro_class_coverage", metrics)
        self.assertEqual(metrics["macro_class_coverage"], metrics["marginal_coverage"])


if __name__ == "__main__":
    unittest.main()
