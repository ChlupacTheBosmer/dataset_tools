from __future__ import annotations

import unittest
from unittest.mock import patch

from dataset_tools.metrics.field_metric import FieldMetricComputation
from dataset_tools.metrics.hardness import HardnessComputation
from dataset_tools.metrics.mistakenness import MistakennessComputation
from dataset_tools.metrics.representativeness import RepresentativenessComputation
from dataset_tools.metrics.uniqueness import UniquenessComputation


class _FakeDataset:
    class _FakeField:
        def __init__(self, document_type):
            self.document_type = document_type

    def __init__(
        self,
        fields: set[str] | None = None,
        field_types: dict[str, object] | None = None,
        size: int = 25,
    ):
        self._fields = set(fields or ())
        self._field_types = dict(field_types or {})
        self._size = size

    def has_sample_field(self, name: str) -> bool:
        return name in self._fields

    def get_field(self, name: str):
        return self._FakeField(self._field_types.get(name))

    def __len__(self):
        return self._size


class _FakeFieldMetric(FieldMetricComputation):
    def __init__(self, dataset_name: str, required_fields):
        super().__init__(dataset_name=dataset_name, required_fields=required_fields)
        self.compute_called = False

    def compute(self, dataset):
        self.compute_called = True
        return {"ok": True}


class MetricsOperationsTests(unittest.TestCase):
    def test_field_metric_validates_required_fields(self):
        metric = _FakeFieldMetric(dataset_name="d", required_fields=("a", "b"))
        metric.load_dataset = lambda: _FakeDataset(fields={"a"})  # type: ignore[assignment]

        with self.assertRaises(RuntimeError):
            metric.run()

    def test_field_metric_injects_dataset_into_result(self):
        metric = _FakeFieldMetric(dataset_name="d", required_fields=("a",))
        metric.load_dataset = lambda: _FakeDataset(fields={"a"})  # type: ignore[assignment]
        payload = metric.run()
        self.assertTrue(metric.compute_called)
        self.assertEqual(payload["dataset"], "d")

    def test_uniqueness_requires_embeddings_field_when_provided(self):
        metric = UniquenessComputation(dataset_name="d", embeddings_field="emb", output_field="uniq")
        metric.load_dataset = lambda: _FakeDataset(fields=set())  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            metric.run()

    def test_uniqueness_calls_fiftyone_brain(self):
        dataset = _FakeDataset(fields={"emb"})
        metric = UniquenessComputation(dataset_name="d", embeddings_field="emb", output_field="uniq")
        metric.load_dataset = lambda: dataset  # type: ignore[assignment]

        with patch("dataset_tools.metrics.uniqueness.fob.compute_uniqueness") as call:
            payload = metric.run()

        self.assertEqual(payload["field"], "uniq")
        _, kwargs = call.call_args
        self.assertEqual(kwargs["embeddings"], "emb")
        self.assertEqual(kwargs["uniqueness_field"], "uniq")

    def test_mistakenness_validates_required_fields(self):
        metric = MistakennessComputation(dataset_name="d", pred_field="pred", gt_field="gt")
        metric.load_dataset = lambda: _FakeDataset(fields={"pred"})  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            metric.run()

    def test_mistakenness_calls_fiftyone_brain(self):
        dataset = _FakeDataset(fields={"pred", "gt"})
        metric = MistakennessComputation(
            dataset_name="d",
            pred_field="pred",
            gt_field="gt",
            mistakenness_field="mist",
            missing_field="miss",
            spurious_field="spur",
        )
        metric.load_dataset = lambda: dataset  # type: ignore[assignment]

        with patch("dataset_tools.metrics.mistakenness.fob.compute_mistakenness") as call:
            payload = metric.run()

        self.assertEqual(payload["field"], "mist")
        _, kwargs = call.call_args
        self.assertEqual(kwargs["pred_field"], "pred")
        self.assertEqual(kwargs["label_field"], "gt")
        self.assertEqual(kwargs["mistakenness_field"], "mist")
        self.assertEqual(kwargs["missing_field"], "miss")
        self.assertEqual(kwargs["spurious_field"], "spur")

    def test_hardness_validates_label_field(self):
        metric = HardnessComputation(dataset_name="d", label_field="gt")
        metric.load_dataset = lambda: _FakeDataset(fields=set())  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            metric.run()

    def test_hardness_calls_fiftyone_brain(self):
        import fiftyone as fo  # type: ignore

        dataset = _FakeDataset(fields={"gt"}, field_types={"gt": fo.Classification})
        metric = HardnessComputation(dataset_name="d", label_field="gt", output_field="hard")
        metric.load_dataset = lambda: dataset  # type: ignore[assignment]

        with patch("dataset_tools.metrics.hardness.fob.compute_hardness") as call:
            payload = metric.run()

        self.assertEqual(payload["field"], "hard")
        _, kwargs = call.call_args
        self.assertEqual(kwargs["label_field"], "gt")
        self.assertEqual(kwargs["hardness_field"], "hard")

    def test_hardness_requires_classification_label_type(self):
        import fiftyone as fo  # type: ignore

        dataset = _FakeDataset(fields={"gt"}, field_types={"gt": fo.Detections})
        metric = HardnessComputation(dataset_name="d", label_field="gt", output_field="hard")
        metric.load_dataset = lambda: dataset  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            metric.run()

    def test_representativeness_validates_optional_fields(self):
        metric = RepresentativenessComputation(
            dataset_name="d",
            output_field="repr",
            method="cluster-center",
            embeddings_field="emb",
            roi_field="patches",
        )
        metric.load_dataset = lambda: _FakeDataset(fields={"emb"})  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            metric.run()

    def test_representativeness_calls_fiftyone_brain(self):
        dataset = _FakeDataset(fields={"emb", "patches"}, size=30)
        metric = RepresentativenessComputation(
            dataset_name="d",
            output_field="repr",
            method="cluster-center",
            embeddings_field="emb",
            roi_field="patches",
        )
        metric.load_dataset = lambda: dataset  # type: ignore[assignment]

        with patch("dataset_tools.metrics.representativeness.fob.compute_representativeness") as call:
            payload = metric.run()

        self.assertEqual(payload["field"], "repr")
        _, kwargs = call.call_args
        self.assertEqual(kwargs["representativeness_field"], "repr")
        self.assertEqual(kwargs["method"], "cluster-center")
        self.assertEqual(kwargs["embeddings"], "emb")
        self.assertEqual(kwargs["roi_field"], "patches")

    def test_representativeness_requires_min_samples_for_cluster_methods(self):
        dataset = _FakeDataset(fields={"emb"}, size=5)
        metric = RepresentativenessComputation(
            dataset_name="d",
            output_field="repr",
            method="cluster-center",
            embeddings_field="emb",
        )
        metric.load_dataset = lambda: dataset  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            metric.run()


if __name__ == "__main__":
    unittest.main()
