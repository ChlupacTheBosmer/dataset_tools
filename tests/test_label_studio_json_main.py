from __future__ import annotations

import os
import tempfile
import unittest

from dataset_tools import label_studio_json as ls_json


class LabelStudioJsonMainTests(unittest.TestCase):
    def _build_fixture(self, tmpdir):
        img_dir = os.path.join(tmpdir, "images", "nested")
        lbl_dir = os.path.join(tmpdir, "labels", "nested")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        img_path = os.path.join(img_dir, "a.jpg")
        with open(img_path, "wb") as f:
            f.write(b"img")

        label_path = os.path.join(lbl_dir, "a.txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.4\n")

    def test_build_tasks_reads_yolo_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_fixture(tmpdir)
            tasks = ls_json.build_tasks(root_dir=ls_json.Path(tmpdir), ls_root="/data/local-files/?d=images")

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertIn("data", task)
        self.assertIn("predictions", task)
        self.assertIn("rectanglelabels", task["predictions"][0]["result"][0]["value"])


if __name__ == "__main__":
    unittest.main()
