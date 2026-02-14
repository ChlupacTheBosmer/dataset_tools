from __future__ import annotations

import unittest

from dataset_tools.sync_from_fo_to_disk import infer_label_path


class SyncUtilsTests(unittest.TestCase):
    def test_infer_label_path_images(self):
        path = "/mnt/data/images/a/b/img_01.jpg"
        out = infer_label_path(path, (("/images/", "/labels/"),))
        self.assertEqual(out, "/mnt/data/labels/a/b/img_01.txt")

    def test_infer_label_path_frames(self):
        path = "/mnt/data/frames/a/b/img_01.png"
        out = infer_label_path(path, (("/frames/", "/labels/"),))
        self.assertEqual(out, "/mnt/data/labels/a/b/img_01.txt")

    def test_infer_label_path_missing_rule(self):
        path = "/mnt/data/custom/a/b/img_01.png"
        out = infer_label_path(path, (("/frames/", "/labels/"),))
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
