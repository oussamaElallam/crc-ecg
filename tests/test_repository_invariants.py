import json
import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RepositoryInvariantTests(unittest.TestCase):
    def test_example_headline(self):
        data = json.loads((ROOT / "example_results.json").read_text())
        headline = data["headline"]
        self.assertEqual(headline["exact_binomial_rcps_empirical_targets_met"], 9)
        self.assertEqual(headline["exact_binomial_rcps_nonvacuous_guarantees"], 9)
        self.assertEqual(headline["uniform_standard_cp_targets_met"], 5)

    def test_cooccurrence_counts_are_nonnegative(self):
        data = json.loads(
            (ROOT / "results" / "verification_results_20260712_1446.json").read_text()
        )
        for key in ["ptbxl_counts", "chapman_counts"]:
            matrix = np.asarray(data["cooccurrence"][key], dtype=np.int64)
            self.assertTrue(np.all(matrix >= 0))

    def test_ptb_patient_disjoint_flag(self):
        data = json.loads(
            (ROOT / "results" / "verification_results_20260712_1446.json").read_text()
        )
        self.assertTrue(data["ptbxl"]["patient_disjoint"])
        self.assertEqual(data["ptbxl"]["n_total_retained"], 21388)


if __name__ == "__main__":
    unittest.main()
