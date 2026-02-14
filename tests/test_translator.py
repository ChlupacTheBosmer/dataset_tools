from __future__ import annotations

import unittest

from dataset_tools.label_studio.translator import fo_detection_to_ls_result


class _FakeDetection:
    def __init__(self, label, bbox, confidence=None):
        self.label = label
        self.bounding_box = bbox
        self.confidence = confidence


class TranslatorTests(unittest.TestCase):
    def test_detection_to_ls_rectangle(self):
        det = _FakeDetection("rodent", [0.1, 0.2, 0.3, 0.4], confidence=0.75)
        out = fo_detection_to_ls_result(det)

        self.assertEqual(out["type"], "rectanglelabels")
        self.assertEqual(out["value"]["rectanglelabels"], ["rodent"])
        self.assertAlmostEqual(out["value"]["x"], 10.0)
        self.assertAlmostEqual(out["value"]["y"], 20.0)
        self.assertAlmostEqual(out["value"]["width"], 30.0)
        self.assertAlmostEqual(out["value"]["height"], 40.0)
        self.assertEqual(out["score"], 0.75)


if __name__ == "__main__":
    unittest.main()
