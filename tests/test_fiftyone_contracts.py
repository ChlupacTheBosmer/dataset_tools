from __future__ import annotations

import inspect
import unittest

import fiftyone.brain as fob  # type: ignore
from fiftyone.core.collections import SampleCollection  # type: ignore


class FiftyOneContractsTests(unittest.TestCase):
    def test_required_brain_functions_exist(self):
        required = [
            "compute_visualization",
            "compute_similarity",
            "compute_uniqueness",
            "compute_mistakenness",
            "compute_hardness",
            "compute_representativeness",
            "compute_near_duplicates",
            "compute_exact_duplicates",
            "compute_leaky_splits",
        ]
        for name in required:
            with self.subTest(name=name):
                self.assertTrue(hasattr(fob, name), f"Missing fiftyone.brain.{name}")

    def test_signature_contracts(self):
        mistakenness = inspect.signature(fob.compute_mistakenness).parameters
        self.assertIn("pred_field", mistakenness)
        self.assertIn("label_field", mistakenness)

        hardness = inspect.signature(fob.compute_hardness).parameters
        self.assertIn("label_field", hardness)
        self.assertEqual(hardness["label_field"].default, inspect._empty)

        leaky = inspect.signature(fob.compute_leaky_splits).parameters
        self.assertIn("splits", leaky)
        self.assertEqual(leaky["splits"].default, inspect._empty)

        near = inspect.signature(fob.compute_near_duplicates).parameters
        self.assertNotIn("brain_key", near)

        viz = inspect.signature(fob.compute_visualization).parameters
        self.assertIn("brain_key", viz)

        sim = inspect.signature(fob.compute_similarity).parameters
        self.assertIn("brain_key", sim)

    def test_unsupported_sample_collection_helpers_not_assumed(self):
        for name in ("find_duplicates", "find_unique", "match_duplicates"):
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(SampleCollection, name),
                    f"SampleCollection unexpectedly has '{name}'",
                )


if __name__ == "__main__":
    unittest.main()
