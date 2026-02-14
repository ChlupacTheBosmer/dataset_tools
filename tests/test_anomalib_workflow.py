from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset_tools.anomaly.anomalib import (
    AnomalibArtifact,
    PreparedAnomalibDataset,
    _parse_image_size,
    prepare_anomalib_folder_dataset,
    score_with_anomalib_artifact,
    train_and_export_anomalib,
)


class _FakeView:
    def __init__(self, samples, tags_map):
        self._samples = list(samples)
        self._tags_map = tags_map
        self._saved = {}

    def __len__(self):
        return len(self._samples)

    def values(self, field):
        if field == "id":
            return [s.id for s in self._samples]
        if field == "filepath":
            return [s.filepath for s in self._samples]
        raise KeyError(field)

    def set_values(self, field, payload, key_field="id"):
        self._saved[field] = dict(payload)

    def iter_samples(self, progress=False):
        return iter(self._samples)

    def match_tags(self, tag):
        selected = [s for s in self._samples if tag in self._tags_map.get(s.id, [])]
        return _FakeView(selected, self._tags_map)


class _FakeDataset(_FakeView):
    pass


class _FakeInferencer:
    def predict(self, image):
        if str(image).endswith("a.jpg"):
            return types.SimpleNamespace(
                pred_score=0.9,
                pred_label=1,
                anomaly_map=np.ones((4, 4), dtype=float),
                pred_mask=np.ones((4, 4), dtype=np.uint8),
            )
        return types.SimpleNamespace(
            pred_score=0.1,
            pred_label=0,
            anomaly_map=np.zeros((4, 4), dtype=float),
            pred_mask=np.zeros((4, 4), dtype=np.uint8),
        )


class _FakeTrainer:
    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("ckpt", encoding="utf-8")


class _FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []
        self.trainer = _FakeTrainer()

    def fit(self, model, datamodule):
        self.fit_calls.append((model, datamodule))

    def export(self, model, export_type, export_root, datamodule, input_size):
        if str(export_type) == "torch":
            return Path(export_root) / "weights" / "torch" / "model.pt"
        return Path(export_root) / "weights" / "openvino" / "model.xml"


class _FakeSample:
    def __init__(self, sid, filepath, mask_path=None):
        self.id = sid
        self.filepath = filepath
        self.defect_mask = types.SimpleNamespace(mask_path=mask_path)


class _FakePredictEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def predict(self, model, data_path, ckpt_path, return_predictions):
        self.calls.append((model, data_path, ckpt_path, return_predictions))
        items = []
        for image in sorted(Path(data_path).glob("*")):
            is_anomaly = image.stem.startswith("s1")
            items.append(
                types.SimpleNamespace(
                    image_path=str(image),
                    pred_score=np.array([0.9 if is_anomaly else 0.1]),
                    pred_label=np.array([1 if is_anomaly else 0]),
                    anomaly_map=np.ones((4, 4), dtype=float),
                    pred_mask=np.ones((4, 4), dtype=np.uint8),
                )
            )
        return [items]


class AnomalibWorkflowTests(unittest.TestCase):
    def test_parse_image_size(self):
        self.assertEqual(_parse_image_size("256"), (256, 256))
        self.assertEqual(_parse_image_size("320,240"), (320, 240))
        self.assertIsNone(_parse_image_size(None))
        self.assertIsNone(_parse_image_size(" "))

    def test_prepare_anomalib_folder_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_a = root / "a.jpg"
            img_b = root / "b.jpg"
            img_c = root / "c.jpg"
            mask_c = root / "c_mask.png"
            img_a.write_bytes(b"a")
            img_b.write_bytes(b"b")
            img_c.write_bytes(b"c")
            mask_c.write_bytes(b"m")

            samples = [
                _FakeSample("s1", str(img_a)),
                _FakeSample("s2", str(img_b)),
                _FakeSample("s3", str(img_c), mask_path=str(mask_c)),
            ]
            tags = {
                "s1": ["normal"],
                "s2": ["normal"],
                "s3": ["abnormal"],
            }
            dataset = _FakeDataset(samples, tags)

            with patch("dataset_tools.anomaly.anomalib._load_dataset", return_value=dataset):
                prepared = prepare_anomalib_folder_dataset(
                    dataset_name="d",
                    output_root=root / "anom_data",
                    normal_tag="normal",
                    abnormal_tag="abnormal",
                    mask_field="defect_mask",
                    symlink=False,
                    overwrite_data=False,
                )

            self.assertEqual(prepared.normal_count, 2)
            self.assertEqual(prepared.abnormal_count, 1)
            self.assertEqual(prepared.mask_count, 1)
            self.assertEqual(prepared.missing_masks, 0)
            self.assertTrue((Path(prepared.normal_dir) / "s1.jpg").exists())
            self.assertTrue((Path(prepared.abnormal_dir or "") / "s3.jpg").exists())
            self.assertTrue((Path(prepared.mask_dir or "") / "s3.jpg").exists())

    def test_score_with_anomalib_artifact(self):
        samples = [
            _FakeSample("s1", "/tmp/a.jpg"),
            _FakeSample("s2", "/tmp/b.jpg"),
        ]
        dataset = _FakeDataset(samples, {"s1": ["eval"], "s2": ["eval"]})

        artifact = AnomalibArtifact(
            dataset_name="d",
            model_ref="anomalib:padim",
            export_type="openvino",
            model_path="/tmp/model.xml",
            artifact_dir="/tmp",
        )

        with patch("dataset_tools.anomaly.anomalib._load_dataset", return_value=dataset), patch(
            "dataset_tools.anomaly.anomalib._resolve_anomalib_artifact",
            return_value=artifact,
        ), patch(
            "dataset_tools.anomaly.anomalib._build_inferencer",
            return_value=_FakeInferencer(),
        ):
            payload = score_with_anomalib_artifact(
                dataset_name="d",
                artifact="/tmp/anom/artifact.json",
                threshold=0.5,
                score_field="score",
                flag_field="flag",
                label_field="label",
                map_field="map",
                mask_field="mask",
                tag_filter=None,
            )

        self.assertEqual(payload["backend"], "anomalib")
        self.assertEqual(payload["scored_samples"], 2)
        self.assertEqual(payload["anomaly_count"], 1)
        self.assertIn("score", dataset._saved)
        self.assertIn("flag", dataset._saved)
        self.assertIn("label", dataset._saved)
        self.assertIn("map", dataset._saved)
        self.assertIn("mask", dataset._saved)
        self.assertEqual(payload["inference_mode"], "inferencer")

    def test_score_with_anomalib_artifact_uses_engine_predict_for_torch(self):
        samples = [
            _FakeSample("s1", "/tmp/a.jpg"),
            _FakeSample("s2", "/tmp/b.jpg"),
        ]
        dataset = _FakeDataset(samples, {"s1": [], "s2": []})

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.ckpt"
            checkpoint_path.write_text("ckpt", encoding="utf-8")
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_text("pt", encoding="utf-8")
            artifact = AnomalibArtifact(
                dataset_name="d",
                model_ref="anomalib:padim",
                export_type="torch",
                model_path=str(model_path),
                artifact_dir=str(Path(tmpdir)),
                checkpoint_path=str(checkpoint_path),
            )
            fake_engine = _FakePredictEngine()

            with patch("dataset_tools.anomaly.anomalib._load_dataset", return_value=dataset), patch(
                "dataset_tools.anomaly.anomalib._resolve_anomalib_artifact",
                return_value=artifact,
            ), patch(
                "dataset_tools.anomaly.anomalib._import_anomalib_components",
                return_value=(
                    lambda **kwargs: fake_engine,
                    object,
                    object,
                    object,
                    object,
                ),
            ), patch(
                "dataset_tools.anomaly.anomalib.load_model",
                return_value=types.SimpleNamespace(model=object()),
            ), patch.dict(os.environ, {"TRUST_REMOTE_CODE": "1"}, clear=False):
                payload = score_with_anomalib_artifact(
                    dataset_name="d",
                    artifact=str(model_path),
                    threshold=0.5,
                    score_field="score",
                    flag_field="flag",
                    label_field="label",
                    map_field="map",
                    mask_field="mask",
                )

        self.assertEqual(payload["backend"], "anomalib")
        self.assertEqual(payload["scored_samples"], 2)
        self.assertEqual(payload["inference_mode"], "engine_predict")
        self.assertEqual(payload["anomaly_count"], 1)
        self.assertIn("score", dataset._saved)
        self.assertIn("flag", dataset._saved)
        self.assertIn("label", dataset._saved)

    def test_train_and_export_anomalib(self):
        prepared = PreparedAnomalibDataset(
            dataset_name="d",
            root_dir="/tmp/anom_data",
            normal_dir="/tmp/anom_data/normal",
            abnormal_dir="/tmp/anom_data/abnormal",
            mask_dir=None,
            normal_tag="normal",
            abnormal_tag="abnormal",
            mask_field=None,
            normal_count=2,
            abnormal_count=1,
            mask_count=0,
            missing_masks=0,
        )
        loaded_model = types.SimpleNamespace(model=object())

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifact"
            artifact_json = artifact_dir / "artifact.json"
            (artifact_dir / "weights" / "openvino").mkdir(parents=True, exist_ok=True)
            (artifact_dir / "weights" / "openvino" / "model.xml").write_text("xml", encoding="utf-8")
            (artifact_dir / "weights" / "torch").mkdir(parents=True, exist_ok=True)
            (artifact_dir / "weights" / "torch" / "model.pt").write_text("pt", encoding="utf-8")

            with patch(
                "dataset_tools.anomaly.anomalib.prepare_anomalib_folder_dataset",
                return_value=prepared,
            ), patch(
                "dataset_tools.anomaly.anomalib._create_datamodule",
                return_value=object(),
            ), patch(
                "dataset_tools.anomaly.anomalib.load_model",
                return_value=loaded_model,
            ), patch(
                "dataset_tools.anomaly.anomalib._import_anomalib_components",
                return_value=(
                    _FakeEngine,
                    lambda value: value,
                    object,
                    object,
                    object,
                ),
            ):
                artifact = train_and_export_anomalib(
                    dataset_name="d",
                    model_ref="anomalib:padim",
                    artifact_dir=artifact_dir,
                    artifact_json=artifact_json,
                    artifact_format="torch",
                    normal_tag="normal",
                    abnormal_tag="abnormal",
                    overwrite_data=True,
                )

            self.assertEqual(artifact.dataset_name, "d")
            self.assertEqual(artifact.export_type, "torch")
            self.assertEqual(artifact.model_ref, "anomalib:padim")
            self.assertIsNotNone(artifact.checkpoint_path)
            self.assertTrue(Path(artifact.checkpoint_path).exists())
            self.assertTrue(artifact_json.exists())

    def test_torch_artifact_requires_trust_remote_code(self):
        samples = [_FakeSample("s1", "/tmp/a.jpg")]
        dataset = _FakeDataset(samples, {"s1": []})

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_text("pt", encoding="utf-8")
            artifact = AnomalibArtifact(
                dataset_name="d",
                model_ref="anomalib:padim",
                export_type="torch",
                model_path=str(model_path),
                artifact_dir=str(Path(tmpdir)),
            )
            with patch("dataset_tools.anomaly.anomalib._load_dataset", return_value=dataset), patch(
                "dataset_tools.anomaly.anomalib._resolve_anomalib_artifact",
                return_value=artifact,
            ), patch.dict(os.environ, {"TRUST_REMOTE_CODE": "0"}, clear=False):
                with self.assertRaises(RuntimeError):
                    score_with_anomalib_artifact(
                        dataset_name="d",
                        artifact=str(model_path),
                    )


if __name__ == "__main__":
    unittest.main()
