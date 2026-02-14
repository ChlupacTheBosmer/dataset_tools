from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dataset_tools import loader as legacy_loader
from dataset_tools.loaders import base as base_loader
from dataset_tools.loaders import coco as coco_loader
from dataset_tools.loaders import path_resolvers
from dataset_tools.loaders import yolo as yolo_loader


class _FakeDataset:
    def __init__(self, name="ds"):
        self.name = name
        self.persistent = False
        self.samples = []

    def add_samples(self, samples):
        self.samples.extend(samples)

    def __len__(self):
        return len(self.samples)


class _FakeSample(dict):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath


class _FakeDetection:
    def __init__(self, **kwargs):
        self.label = kwargs.get("label")
        self.bounding_box = kwargs.get("bounding_box")
        self.confidence = kwargs.get("confidence")


class _FakeDetections:
    def __init__(self, detections):
        self.detections = list(detections)


class _NoopLoader(base_loader.BaseDatasetLoader):
    def load(self, dataset_name: str, overwrite: bool = False, persistent: bool = True):
        return base_loader.LoaderResult(dataset_name=dataset_name, sample_count=0)


class LoadersAndWrappersTests(unittest.TestCase):
    def test_base_loader_create_or_replace(self):
        loader = _NoopLoader()
        created = _FakeDataset("new")

        with patch("dataset_tools.loaders.base.fo.list_datasets", side_effect=[["ds"], []]), patch(
            "dataset_tools.loaders.base.fo.delete_dataset"
        ) as delete_ds, patch("dataset_tools.loaders.base.fo.Dataset", return_value=created):
            out = loader._create_or_replace_dataset("ds", overwrite=True, persistent=True)

        delete_ds.assert_called_once_with("ds")
        self.assertIs(out, created)
        self.assertTrue(created.persistent)

    def test_base_loader_load_existing(self):
        loader = _NoopLoader()
        existing = _FakeDataset("existing")
        with patch("dataset_tools.loaders.base.fo.list_datasets", return_value=["existing"]), patch(
            "dataset_tools.loaders.base.fo.load_dataset",
            return_value=existing,
        ):
            out = loader._create_or_replace_dataset("existing", overwrite=False, persistent=True)
        self.assertIs(out, existing)

    def test_path_resolvers_and_filter(self):
        root = Path("/tmp/root")
        resolver = path_resolvers.ImagesLabelsSubdirResolver(root_dir=root)
        label_path = resolver.label_path_for(root / "images" / "a" / "b.jpg")
        self.assertEqual(label_path, root / "labels" / "a" / "b.txt")

        mirror = path_resolvers.MirroredRootsPathResolver(
            images_root=Path("/images"),
            labels_root=Path("/labels"),
        )
        self.assertEqual(mirror.label_path_for(Path("/images/x/y.png")), Path("/labels/x/y.txt"))

        self.assertTrue(path_resolvers.default_image_filter(Path("a.JPG")))
        self.assertFalse(path_resolvers.default_image_filter(Path("a.txt")))

    def test_yolo_parse_file_with_optional_confidence(self):
        fake_fo = SimpleNamespace(Detection=_FakeDetection, Detections=_FakeDetections, Sample=_FakeSample)
        loader = yolo_loader.YoloDatasetLoader(
            resolver=SimpleNamespace(images_root=Path("."), label_path_for=lambda p: Path(".")),
            parser_config=yolo_loader.YoloParserConfig(class_id_to_label={0: "rodent"}, include_confidence=True),
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(yolo_loader, "fo", fake_fo):
            labels = Path(tmpdir) / "a.txt"
            labels.write_text("0 0.5 0.5 0.4 0.2 0.9\ninvalid row\n", encoding="utf-8")
            out = loader._parse_yolo_file(labels)

        self.assertIsNotNone(out)
        self.assertEqual(len(out.detections), 1)
        det = out.detections[0]
        self.assertEqual(det.label, "rodent")
        self.assertAlmostEqual(det.confidence, 0.9)

    def test_yolo_load_dataset(self):
        fake_fo = SimpleNamespace(Detection=_FakeDetection, Detections=_FakeDetections, Sample=_FakeSample)
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(yolo_loader, "fo", fake_fo):
            root = Path(tmpdir)
            images = root / "images"
            labels = root / "labels"
            (images / "sub").mkdir(parents=True)
            (labels / "sub").mkdir(parents=True)
            (images / "sub" / "a.jpg").write_bytes(b"img")
            (images / "sub" / "b.png").write_bytes(b"img")
            (labels / "sub" / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            resolver = yolo_loader.ImagesLabelsSubdirResolver(root_dir=root)
            loader = yolo_loader.YoloDatasetLoader(
                resolver=resolver,
                parser_config=yolo_loader.YoloParserConfig(class_id_to_label={1: "mouse"}),
                sample_metadata_fields={"split": lambda p: "train" if p.name.startswith("a") else "val"},
            )
            dataset = _FakeDataset(name="loaded")
            with patch.object(loader, "_create_or_replace_dataset", return_value=dataset):
                result = loader.load(dataset_name="loaded", overwrite=False, persistent=True)

        self.assertEqual(result.sample_count, 2)
        self.assertEqual(len(dataset.samples), 2)
        self.assertIn("ground_truth", dataset.samples[0])
        self.assertIn("split", dataset.samples[0])

    def test_coco_loader_uses_from_dir(self):
        config = coco_loader.CocoLoaderConfig(dataset_dir=Path("/tmp/coco"), data_path="data", labels_path="labels.json")
        loader = coco_loader.CocoDatasetLoader(config)
        imported = _FakeDataset(name="coco_ds")
        imported.samples = [1, 2, 3]

        with patch.object(loader, "_create_or_replace_dataset", return_value=_FakeDataset("ignore")), patch(
            "dataset_tools.loaders.coco.fo.Dataset.from_dir",
            return_value=imported,
        ) as from_dir:
            result = loader.load(dataset_name="coco_ds", overwrite=True, persistent=False)

        from_dir.assert_called_once()
        self.assertEqual(result.dataset_name, "coco_ds")
        self.assertEqual(result.sample_count, 3)

    def test_legacy_loader_wrappers(self):
        fake_result = base_loader.LoaderResult(dataset_name="ds", sample_count=12)
        fake_loader_instance = MagicMock()
        fake_loader_instance.load.return_value = fake_result

        with patch("dataset_tools.loader.YoloDatasetLoader", return_value=fake_loader_instance):
            out1 = legacy_loader.import_yolo_dataset_from_root(
                root_dir="/tmp/root",
                dataset_name="ds",
                overwrite=False,
            )
            out2 = legacy_loader.import_yolo_dataset_from_roots(
                images_root="/tmp/images",
                labels_root="/tmp/labels",
                dataset_name="ds2",
                overwrite=True,
            )

        self.assertEqual(out1.sample_count, 12)
        self.assertEqual(out2.sample_count, 12)
        self.assertEqual(fake_loader_instance.load.call_count, 2)

    def test_get_or_create_dataset(self):
        existing = _FakeDataset("existing")
        with patch("dataset_tools.loader.fo.list_datasets", return_value=["existing"]), patch(
            "dataset_tools.loader.fo.load_dataset",
            return_value=existing,
        ):
            out = legacy_loader.get_or_create_dataset("existing")
        self.assertIs(out, existing)

        created = _FakeDataset("new")
        with patch("dataset_tools.loader.fo.list_datasets", return_value=[]), patch(
            "dataset_tools.loader.fo.Dataset",
            return_value=created,
        ):
            out2 = legacy_loader.get_or_create_dataset("new")
        self.assertIs(out2, created)
        self.assertTrue(created.persistent)


if __name__ == "__main__":
    unittest.main()
