from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_tools.anomaly.base import AnomalyReference
from dataset_tools.anomaly.pipeline import (
    fit_embedding_distance_reference,
    load_reference,
    run_embedding_distance,
    save_reference,
    score_with_anomalib,
    score_with_embedding_distance,
)


class _FakeView:
    def __init__(self, ids, embeddings, filepaths, tags_map=None):
        self._ids = list(ids)
        self._embeddings = list(embeddings)
        self._filepaths = list(filepaths)
        self._saved = {}
        self._tags_map = dict(tags_map or {})

    def values(self, field):
        if field == "id":
            return list(self._ids)
        if field == "filepath":
            return list(self._filepaths)
        return list(self._embeddings)

    def set_values(self, field, payload, key_field="id"):
        self._saved[field] = dict(payload)

    def match_tags(self, tag):
        filtered = [idx for idx, sample_id in enumerate(self._ids) if tag in self._tags_map.get(sample_id, [])]
        return _FakeView(
            ids=[self._ids[i] for i in filtered],
            embeddings=[self._embeddings[i] for i in filtered],
            filepaths=[self._filepaths[i] for i in filtered],
            tags_map=self._tags_map,
        )


class _FakeDataset(_FakeView):
    pass


class AnomalyPipelineTests(unittest.TestCase):
    def _fake_dataset(self):
        ids = ["s1", "s2", "s3", "s4"]
        embeddings = [[0.0, 0.0], [0.2, 0.1], [0.1, 0.2], [3.5, 3.8]]
        paths = [f"/tmp/{sid}.jpg" for sid in ids]
        tags = {"s1": ["normal"], "s2": ["normal"], "s3": ["normal"], "s4": ["eval"]}
        return _FakeDataset(ids, embeddings, paths, tags)

    def test_reference_roundtrip(self):
        ref = AnomalyReference(
            backend="embedding_distance",
            embeddings_field="emb",
            threshold=1.2,
            centroid=[0.1, 0.1],
            metadata={"x": 1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "ref.json"
            save_reference(p, ref)
            loaded = load_reference(p)
        self.assertEqual(loaded.backend, "embedding_distance")
        self.assertEqual(loaded.threshold, 1.2)
        self.assertEqual(loaded.metadata["x"], 1)

    def test_fit_and_score_embedding_distance(self):
        dataset = self._fake_dataset()
        with patch("dataset_tools.anomaly.pipeline.fo.list_datasets", return_value=["d"]), patch(
            "dataset_tools.anomaly.pipeline.fo.load_dataset",
            return_value=dataset,
        ):
            ref = fit_embedding_distance_reference(
                dataset_name="d",
                embeddings_field="emb",
                normal_tag="normal",
                threshold_quantile=0.9,
            )
            self.assertEqual(ref.backend, "embedding_distance")
            payload = score_with_embedding_distance(
                dataset_name="d",
                reference=ref,
                score_field="a_score",
                flag_field="a_flag",
                tag_filter=None,
            )

        self.assertEqual(payload["dataset"], "d")
        self.assertGreaterEqual(payload["scored_samples"], 1)
        self.assertIn("a_score", dataset._saved)
        self.assertIn("a_flag", dataset._saved)

    def test_run_embedding_distance(self):
        dataset = self._fake_dataset()
        with patch("dataset_tools.anomaly.pipeline.fo.list_datasets", return_value=["d"]), patch(
            "dataset_tools.anomaly.pipeline.fo.load_dataset",
            return_value=dataset,
        ):
            payload = run_embedding_distance(
                dataset_name="d",
                embeddings_field="emb",
                normal_tag="normal",
                score_tag="eval",
            )

        self.assertEqual(payload["backend"], "embedding_distance")
        self.assertEqual(payload["tag_filter"], "eval")
        self.assertIn("reference", payload)

    def test_anomalib_scoring_delegates_to_artifact_scorer(self):
        with patch(
            "dataset_tools.anomaly.pipeline.score_with_anomalib_artifact",
            return_value={"backend": "anomalib", "dataset": "d", "scored_samples": 4},
        ) as scorer:
            payload = score_with_anomalib(
                dataset_name="d",
                artifact_path="/tmp/anomalib_artifact.json",
                artifact_format="openvino",
                threshold=0.7,
                score_field="anom_score",
                flag_field="anom_flag",
                label_field="anom_label",
                map_field="anom_map",
                mask_field="anom_mask",
                tag_filter="eval",
                device="CPU",
            )

        self.assertEqual(payload["backend"], "anomalib")
        self.assertEqual(payload["scored_samples"], 4)
        scorer.assert_called_once_with(
            dataset_name="d",
            artifact="/tmp/anomalib_artifact.json",
            artifact_format="openvino",
            threshold=0.7,
            score_field="anom_score",
            flag_field="anom_flag",
            label_field="anom_label",
            map_field="anom_map",
            mask_field="anom_mask",
            tag_filter="eval",
            device="CPU",
            trust_remote_code=False,
        )


if __name__ == "__main__":
    unittest.main()
