from __future__ import annotations

import unittest
from unittest.mock import patch

from dataset_tools.brain.duplicates import ExactDuplicatesOperation, NearDuplicatesOperation
from dataset_tools.brain.leaky_splits import LeakySplitsOperation
from dataset_tools.brain.similarity import SimilarityOperation
from dataset_tools.brain.visualization import VisualizationOperation


class _FakeResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeNearIndex:
    def __init__(self):
        self.neighbors_map = {
            "s1": [("s2", 0.1)],
            "s3": [("s4", 0.2), ("s5", 0.25)],
        }
        self.threshold_called_with = None

    def find_duplicates(self, threshold):
        self.threshold_called_with = threshold


class _FakeLeakyIndex:
    def __init__(self):
        self.leak_ids = ["a", "b"]
        self.split_views = {"train": object(), "val": object()}
        self.threshold_called_with = None

    def find_leaks(self, threshold):
        self.threshold_called_with = threshold


class _FakeDataset:
    def __len__(self):
        return 10


class BrainOperationsTests(unittest.TestCase):
    def test_visualization_operation_returns_key_and_index_stats(self):
        fake_result = _FakeResult(key="viz_key", index_size=9, total_index_size=10)
        with patch("dataset_tools.brain.visualization.fob.compute_visualization", return_value=fake_result) as call:
            job = VisualizationOperation(
                dataset_name="d",
                method="pca",
                num_dims=2,
                embeddings="emb",
                brain_key="viz_key",
            )
            payload = job.execute(dataset=object())

        self.assertEqual(payload["brain_key"], "viz_key")
        self.assertTrue(payload["persisted"])
        self.assertEqual(payload["index_size"], 9)
        self.assertEqual(payload["total_index_size"], 10)
        _, kwargs = call.call_args
        self.assertEqual(kwargs["method"], "pca")
        self.assertEqual(kwargs["num_dims"], 2)
        self.assertEqual(kwargs["embeddings"], "emb")
        self.assertEqual(kwargs["brain_key"], "viz_key")

    def test_similarity_operation_returns_key_and_index_stats(self):
        fake_result = _FakeResult(key="sim_key", index_size=12, total_index_size=12)
        with patch("dataset_tools.brain.similarity.fob.compute_similarity", return_value=fake_result) as call:
            job = SimilarityOperation(
                dataset_name="d",
                embeddings="emb",
                backend="sklearn",
                brain_key="sim_key",
            )
            payload = job.execute(dataset=object())

        self.assertEqual(payload["brain_key"], "sim_key")
        self.assertTrue(payload["persisted"])
        self.assertEqual(payload["index_size"], 12)
        self.assertEqual(payload["total_index_size"], 12)
        _, kwargs = call.call_args
        self.assertEqual(kwargs["embeddings"], "emb")
        self.assertEqual(kwargs["backend"], "sklearn")
        self.assertEqual(kwargs["brain_key"], "sim_key")

    def test_exact_duplicates_summary(self):
        duplicates = {
            "s1": ["s2", "s3"],
            "s4": ["s5"],
        }
        with patch("dataset_tools.brain.duplicates.fob.compute_exact_duplicates", return_value=duplicates):
            payload = ExactDuplicatesOperation(dataset_name="d").execute(dataset=object())

        self.assertEqual(payload["duplicate_source_count"], 2)
        self.assertEqual(payload["duplicate_sample_count"], 3)
        self.assertEqual(payload["affected_sample_count"], 5)

    def test_near_duplicates_uses_threshold_without_brain_key(self):
        index = _FakeNearIndex()
        with patch("dataset_tools.brain.duplicates.fob.compute_near_duplicates", return_value=index) as call:
            payload = NearDuplicatesOperation(
                dataset_name="d",
                threshold=0.25,
                embeddings="emb",
                roi_field="detections",
            ).execute(dataset=object())

        self.assertEqual(index.threshold_called_with, 0.25)
        self.assertEqual(payload["duplicate_source_count"], 2)
        self.assertEqual(payload["duplicate_pair_count"], 3)
        self.assertEqual(payload["affected_sample_count"], 5)
        _, kwargs = call.call_args
        self.assertNotIn("brain_key", kwargs)
        self.assertEqual(kwargs["threshold"], 0.25)
        self.assertEqual(kwargs["embeddings"], "emb")
        self.assertEqual(kwargs["roi_field"], "detections")

    def test_near_duplicates_threshold_validation(self):
        with self.assertRaises(ValueError):
            NearDuplicatesOperation(dataset_name="d", threshold=0).execute(dataset=object())

    def test_leaky_splits_summary(self):
        index = _FakeLeakyIndex()
        dataset = _FakeDataset()
        with patch("dataset_tools.brain.leaky_splits.fob.compute_leaky_splits", return_value=index) as call:
            payload = LeakySplitsOperation(
                dataset_name="d",
                splits=["train", "val"],
                threshold=0.2,
                embeddings="emb",
            ).execute(dataset=dataset)

        self.assertEqual(index.threshold_called_with, 0.2)
        self.assertEqual(payload["leak_count"], 2)
        self.assertEqual(payload["sample_count"], 10)
        self.assertEqual(payload["split_count"], 2)
        _, kwargs = call.call_args
        self.assertNotIn("brain_key", kwargs)
        self.assertEqual(kwargs["splits"], ["train", "val"])
        self.assertEqual(kwargs["threshold"], 0.2)
        self.assertEqual(kwargs["embeddings"], "emb")

    def test_leaky_splits_validates_inputs(self):
        with self.assertRaises(ValueError):
            LeakySplitsOperation(dataset_name="d", splits=[], threshold=0.2).execute(dataset=object())

        with self.assertRaises(ValueError):
            LeakySplitsOperation(dataset_name="d", splits=["train"], threshold=0).execute(dataset=object())


if __name__ == "__main__":
    unittest.main()
