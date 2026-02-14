from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import fiftyone as fo  # type: ignore

from dataset_tools.config import load_config
from dataset_tools.tag_workflow.context import TagWorkflowContext
from dataset_tools.tag_workflow.operations.analysis import (
    ComputeHardnessOperation,
    ComputeLeakySplitsOperation,
    ComputeNearDuplicatesOperation,
    ComputeRepresentativenessOperation,
    ComputeSimilarityIndexOperation,
    ComputeUniquenessOperation,
)
from dataset_tools.tag_workflow.operations.core import default_operations_registry


class _FakeField:
    def __init__(self, document_type):
        self.document_type = document_type


class _FakeCollection:
    def __init__(
        self,
        *,
        size: int,
        fields: set[str] | None = None,
        field_types: dict[str, object] | None = None,
    ):
        self._size = size
        self._fields = set(fields or ())
        self._field_types = dict(field_types or {})

    def __len__(self):
        return self._size

    def has_sample_field(self, name: str):
        return name in self._fields

    def get_field(self, name: str):
        return _FakeField(self._field_types.get(name))


class _FakeSimilarityIndex:
    key = "sim_key"
    index_size = 10
    total_index_size = 12


class _FakeNearIndex:
    def __init__(self):
        self.neighbors_map = {
            "s1": [("s2", 0.1)],
            "s3": [("s4", 0.2), ("s5", 0.25)],
        }
        self.called_threshold = None

    def find_duplicates(self, threshold):
        self.called_threshold = threshold


class _FakeLeakyIndex:
    def __init__(self):
        self.leak_ids = ["a", "b", "c"]
        self.split_views = {"train": object(), "val": object(), "test": object()}
        self.called_threshold = None

    def find_leaks(self, threshold):
        self.called_threshold = threshold


def _make_context(dataset):
    cfg = load_config(local_config_path="/tmp/does-not-exist.json")
    return TagWorkflowContext(
        dataset=dataset,
        dataset_name="d",
        app_config=cfg,
    )


class TagWorkflowAnalysisTests(unittest.TestCase):
    def test_default_registry_contains_analysis_operations(self):
        registry = default_operations_registry()
        for name in (
            "compute_uniqueness",
            "compute_hardness",
            "compute_representativeness",
            "compute_similarity_index",
            "compute_exact_duplicates",
            "compute_near_duplicates",
            "compute_leaky_splits",
            "compute_anomaly_scores",
        ):
            self.assertIn(name, registry)

    def test_uniqueness_defaults_to_view_scope(self):
        dataset = _FakeCollection(size=10, fields={"emb"})
        view = _FakeCollection(size=3, fields={"emb"})
        context = _make_context(dataset)

        with patch("dataset_tools.tag_workflow.operations.analysis.fob.compute_uniqueness") as call:
            payload = ComputeUniquenessOperation().execute(
                context=context,
                view=view,
                params={"embeddings_field": "emb", "output_field": "uniq"},
                tag="fix",
            )

        self.assertEqual(payload["scope"], "view")
        self.assertEqual(payload["sample_count"], 3)
        self.assertIs(call.call_args[0][0], view)

    def test_uniqueness_can_target_dataset_scope(self):
        dataset = _FakeCollection(size=10, fields={"emb"})
        view = _FakeCollection(size=3, fields={"emb"})
        context = _make_context(dataset)

        with patch("dataset_tools.tag_workflow.operations.analysis.fob.compute_uniqueness") as call:
            payload = ComputeUniquenessOperation().execute(
                context=context,
                view=view,
                params={"scope": "dataset", "embeddings_field": "emb"},
                tag="fix",
            )

        self.assertEqual(payload["scope"], "dataset")
        self.assertEqual(payload["sample_count"], 10)
        self.assertIs(call.call_args[0][0], dataset)

    def test_similarity_index_requires_explicit_dataset_scope(self):
        context = _make_context(_FakeCollection(size=10))
        with self.assertRaises(ValueError):
            ComputeSimilarityIndexOperation().execute(
                context=context,
                view=_FakeCollection(size=3),
                params={"scope": "view"},
                tag="fix",
            )

    def test_similarity_index_uses_dataset_scope(self):
        dataset = _FakeCollection(size=10)
        context = _make_context(dataset)
        with patch(
            "dataset_tools.tag_workflow.operations.analysis.fob.compute_similarity",
            return_value=_FakeSimilarityIndex(),
        ) as call:
            payload = ComputeSimilarityIndexOperation().execute(
                context=context,
                view=_FakeCollection(size=3),
                params={"scope": "dataset", "embeddings_field": "emb", "brain_key": "sim_key"},
                tag="fix",
            )

        self.assertEqual(payload["scope"], "dataset")
        self.assertEqual(payload["brain_key"], "sim_key")
        self.assertTrue(payload["persisted"])
        self.assertIs(call.call_args[0][0], dataset)

    def test_near_duplicates_summary(self):
        dataset = _FakeCollection(size=10)
        context = _make_context(dataset)
        index = _FakeNearIndex()

        with patch(
            "dataset_tools.tag_workflow.operations.analysis.fob.compute_near_duplicates",
            return_value=index,
        ):
            payload = ComputeNearDuplicatesOperation().execute(
                context=context,
                view=_FakeCollection(size=3),
                params={"scope": "dataset", "threshold": 0.2},
                tag="fix",
            )

        self.assertEqual(index.called_threshold, 0.2)
        self.assertEqual(payload["duplicate_source_count"], 2)
        self.assertEqual(payload["duplicate_pair_count"], 3)
        self.assertEqual(payload["affected_sample_count"], 5)

    def test_leaky_splits_accepts_csv_splits(self):
        dataset = _FakeCollection(size=10)
        context = _make_context(dataset)
        index = _FakeLeakyIndex()

        with patch(
            "dataset_tools.tag_workflow.operations.analysis.fob.compute_leaky_splits",
            return_value=index,
        ):
            payload = ComputeLeakySplitsOperation().execute(
                context=context,
                view=_FakeCollection(size=3),
                params={"scope": "dataset", "splits": "train,val,test", "threshold": 0.2},
                tag="fix",
            )

        self.assertEqual(index.called_threshold, 0.2)
        self.assertEqual(payload["splits"], ["train", "val", "test"])
        self.assertEqual(payload["leak_count"], 3)
        self.assertEqual(payload["split_count"], 3)
        self.assertEqual(payload["sample_count"], 10)

    def test_hardness_requires_classification_type(self):
        dataset = _FakeCollection(
            size=10,
            fields={"lbl"},
            field_types={"lbl": fo.Detections},
        )
        context = _make_context(dataset)
        with self.assertRaises(RuntimeError):
            ComputeHardnessOperation().execute(
                context=context,
                view=dataset,
                params={"label_field": "lbl"},
                tag="fix",
            )

    def test_representativeness_min_sample_guard(self):
        dataset = _FakeCollection(size=5, fields={"emb"})
        context = _make_context(dataset)
        with self.assertRaises(RuntimeError):
            ComputeRepresentativenessOperation().execute(
                context=context,
                view=dataset,
                params={"embeddings_field": "emb", "method": "cluster-center"},
                tag="fix",
            )

    def test_anomaly_scores_embedding_distance(self):
        from dataset_tools.tag_workflow.operations.analysis import ComputeAnomalyScoresOperation

        dataset = _FakeCollection(size=10, fields={"emb"})
        context = _make_context(dataset)
        with patch(
            "dataset_tools.tag_workflow.operations.analysis.fit_embedding_distance_reference",
            return_value=types.SimpleNamespace(to_dict=lambda: {"backend": "embedding_distance"}),
        ), patch(
            "dataset_tools.tag_workflow.operations.analysis.score_with_embedding_distance",
            return_value={"backend": "embedding_distance", "dataset": "d"},
        ):
            payload = ComputeAnomalyScoresOperation().execute(
                context=context,
                view=_FakeCollection(size=3, fields={"emb"}),
                params={"backend": "embedding_distance", "embeddings_field": "emb"},
                tag="fix",
            )

        self.assertEqual(payload["backend"], "embedding_distance")

    def test_anomaly_scores_anomalib(self):
        from dataset_tools.tag_workflow.operations.analysis import ComputeAnomalyScoresOperation

        dataset = _FakeCollection(size=10, fields={"emb"})
        context = _make_context(dataset)
        with patch(
            "dataset_tools.tag_workflow.operations.analysis.score_with_anomalib",
            return_value={"backend": "anomalib", "dataset": "d"},
        ):
            payload = ComputeAnomalyScoresOperation().execute(
                context=context,
                view=_FakeCollection(size=3, fields={"emb"}),
                params={"backend": "anomalib", "artifact": "/tmp/anom/artifact.json"},
                tag="fix",
            )

        self.assertEqual(payload["backend"], "anomalib")

    def test_anomaly_scores_anomalib_requires_artifact(self):
        from dataset_tools.tag_workflow.operations.analysis import ComputeAnomalyScoresOperation

        dataset = _FakeCollection(size=10, fields={"emb"})
        context = _make_context(dataset)

        with self.assertRaises(ValueError):
            ComputeAnomalyScoresOperation().execute(
                context=context,
                view=_FakeCollection(size=3, fields={"emb"}),
                params={"backend": "anomalib"},
                tag="fix",
            )


if __name__ == "__main__":
    unittest.main()
