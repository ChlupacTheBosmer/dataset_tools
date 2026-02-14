from __future__ import annotations

import os
import tempfile
import types
import unittest
from unittest.mock import patch

from dataset_tools.config import load_config
from dataset_tools.sync_from_fo_to_disk import backup_file, sync_corrections_to_disk


class _FakeDetection:
    def __init__(self, label, bounding_box):
        self.label = label
        self.bounding_box = bounding_box


class _FakeSample(dict):
    def __init__(self, filepath, detections, tags=None):
        super().__init__()
        self.filepath = filepath
        self.tags = list(tags or [])
        self["ls_corrections"] = types.SimpleNamespace(detections=list(detections))


class _FakeView:
    def __init__(self, samples):
        self._samples = list(samples)

    def match_tags(self, tag):
        return _FakeView([s for s in self._samples if tag in s.tags])

    def __iter__(self):
        return iter(self._samples)


class _FakeDataset:
    def __init__(self, samples):
        self._samples = list(samples)

    def match(self, _expr):
        return _FakeView(self._samples)


class _FakeViewField:
    def __init__(self, _name):
        self._name = _name

    def exists(self):
        return True


class SyncFromFoToDiskExtendedTests(unittest.TestCase):
    def test_backup_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, "labels.txt")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("x")
            backup = backup_file(fp, suffix_format="%Y%m%d")
            self.assertTrue(backup is not None and os.path.exists(backup))
            self.assertIsNone(backup_file(os.path.join(tmpdir, "missing.txt")))

    def test_sync_corrections_to_disk_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "images", "a.jpg")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(b"img")

            sample = _FakeSample(
                filepath=image_path,
                detections=[_FakeDetection("rodent", [0.1, 0.1, 0.2, 0.2])],
                tags=["fix"],
            )
            dataset = _FakeDataset([sample])

            cfg = load_config(
                local_config_path="/tmp/does-not-exist.json",
                overrides={
                    "dataset": {"name": "ds", "label_to_class_id": {"rodent": 0}},
                    "disk_sync": {"path_replacements": [["/images/", "/labels/"]]},
                },
            )
            fake_fo = types.SimpleNamespace(
                list_datasets=lambda: ["ds"],
                load_dataset=lambda name: dataset,
                ViewField=_FakeViewField,
            )

            with patch("dataset_tools.sync_from_fo_to_disk.load_config", return_value=cfg), patch.dict(
                "sys.modules", {"fiftyone": fake_fo}
            ):
                count = sync_corrections_to_disk(dataset_name="ds", dry_run=True, tag_filter="fix")

            self.assertEqual(count, 1)
            expected_label = image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            self.assertFalse(os.path.exists(expected_label))

    def test_sync_corrections_to_disk_writes_labels_and_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "images", "b.jpg")
            label_path = os.path.join(tmpdir, "labels", "b.txt")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(b"img")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("old")

            sample = _FakeSample(
                filepath=image_path,
                detections=[
                    _FakeDetection("rodent", [0.2, 0.3, 0.4, 0.5]),
                    _FakeDetection("unknown", [1.2, -0.2, 2.0, -1.0]),
                ],
            )
            dataset = _FakeDataset([sample])

            cfg = load_config(
                local_config_path="/tmp/does-not-exist.json",
                overrides={
                    "dataset": {"name": "ds", "default_class_id": 9, "label_to_class_id": {"rodent": 0}},
                    "disk_sync": {
                        "path_replacements": [["/images/", "/labels/"]],
                        "backup_suffix_format": "%Y%m%d",
                    },
                },
            )
            fake_fo = types.SimpleNamespace(
                list_datasets=lambda: ["ds"],
                load_dataset=lambda name: dataset,
                ViewField=_FakeViewField,
            )

            with patch("dataset_tools.sync_from_fo_to_disk.load_config", return_value=cfg), patch.dict(
                "sys.modules", {"fiftyone": fake_fo}
            ):
                count = sync_corrections_to_disk(dataset_name="ds", dry_run=False)

            self.assertEqual(count, 1)
            with open(label_path, "r", encoding="utf-8") as f:
                content = f.read().strip().splitlines()
            self.assertEqual(len(content), 2)
            self.assertTrue(content[0].startswith("0 "))
            self.assertTrue(content[1].startswith("9 "))
            backups = [p for p in os.listdir(os.path.dirname(label_path)) if p.startswith("b.txt.") and p.endswith(".bak")]
            self.assertTrue(backups)

    def test_sync_corrections_to_disk_missing_dataset_raises(self):
        cfg = load_config(local_config_path="/tmp/does-not-exist.json", overrides={"dataset": {"name": "missing"}})
        fake_fo = types.SimpleNamespace(
            list_datasets=lambda: [],
            load_dataset=lambda name: None,
            ViewField=_FakeViewField,
        )
        with patch("dataset_tools.sync_from_fo_to_disk.load_config", return_value=cfg), patch.dict(
            "sys.modules", {"fiftyone": fake_fo}
        ):
            with self.assertRaises(RuntimeError):
                sync_corrections_to_disk(dataset_name="missing")


if __name__ == "__main__":
    unittest.main()
