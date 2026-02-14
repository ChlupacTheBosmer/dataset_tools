from __future__ import annotations

import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from dataset_tools.metrics import embeddings as emb_mod


class _FakeInputs(dict):
    def to(self, _device):
        return self


class _FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        return _FakeInputs({"pixel_values": torch.tensor([[1.0]])})


class _FakeModelLastHidden:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **_kwargs):
        return types.SimpleNamespace(last_hidden_state=torch.tensor([[[1.0, 2.0, 3.0]]]))


class _FakeModelPooler:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **_kwargs):
        return types.SimpleNamespace(pooler_output=torch.tensor([[4.0, 5.0]]))


class _FakeView:
    def __init__(self, embeddings, ids):
        self._embeddings = list(embeddings)
        self._ids = list(ids)
        self.selected_ids = None
        self.set_values_calls = []

    def values(self, field):
        if field == "id":
            return list(self._ids)
        return list(self._embeddings)

    def select(self, ids):
        self.selected_ids = list(ids)
        return self

    def set_values(self, field, values, key_field="id"):
        self.set_values_calls.append((field, dict(values), key_field))


class _FakeDataset(_FakeView):
    def __init__(self, embeddings, ids):
        super().__init__(embeddings=embeddings, ids=ids)
        self.compute_embeddings_calls = []
        self.compute_patch_embeddings_calls = []
        self.patches_view = _FakeView(embeddings=embeddings, ids=ids)

    def compute_embeddings(self, model, embeddings_field, progress=True):
        self.compute_embeddings_calls.append((model, embeddings_field, progress))

    def compute_patch_embeddings(self, model, patches_field, embeddings_field, force_square, progress):
        self.compute_patch_embeddings_calls.append(
            (model, patches_field, embeddings_field, force_square, progress)
        )

    def to_patches(self, _patches_field):
        return self.patches_view


class _FakeKMeans:
    calls = []

    def __init__(self, n_clusters, n_init, random_state):
        _FakeKMeans.calls.append((n_clusters, n_init, random_state))

    def fit_predict(self, matrix):
        return np.array([idx % 2 for idx in range(len(matrix))])


class MetricsEmbeddingsExtendedTests(unittest.TestCase):
    def test_huggingface_model_embed_from_path_and_embed_all(self):
        with patch("dataset_tools.models.providers.huggingface.torch.cuda.is_available", return_value=False), patch(
            "dataset_tools.models.providers.huggingface.AutoImageProcessor.from_pretrained",
            return_value=_FakeProcessor(),
        ), patch(
            "dataset_tools.models.providers.huggingface.AutoModel.from_pretrained",
            return_value=_FakeModelLastHidden(),
        ):
            model = emb_mod.HuggingFaceEmbeddingModel("dummy/model")
            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                Image.new("RGB", (16, 16), color=(10, 20, 30)).save(tmp.name)
                vec = model.embed(tmp.name)
                self.assertEqual(vec.shape[0], 3)
                stacked = model.embed_all([tmp.name, np.zeros((8, 8, 3), dtype=np.uint8)])
                self.assertEqual(stacked.shape[0], 2)

    def test_huggingface_model_embed_pooler_fallback(self):
        with patch("dataset_tools.models.providers.huggingface.torch.cuda.is_available", return_value=False), patch(
            "dataset_tools.models.providers.huggingface.AutoImageProcessor.from_pretrained",
            return_value=_FakeProcessor(),
        ), patch(
            "dataset_tools.models.providers.huggingface.AutoModel.from_pretrained",
            return_value=_FakeModelPooler(),
        ):
            model = emb_mod.HuggingFaceEmbeddingModel("dummy/model")
            vec = model.embed(np.zeros((8, 8, 3), dtype=np.uint8))
            self.assertEqual(vec.tolist(), [4.0, 5.0])

    def test_embeddings_computation_full_flow_dataset_level(self):
        embeddings = [np.array([1.0, 2.0]), None, np.array([3.0, 4.0])]
        ids = ["a", "b", "c"]
        dataset = _FakeDataset(embeddings=embeddings, ids=ids)

        with patch("dataset_tools.metrics.embeddings.load_model") as load_model, patch(
            "dataset_tools.metrics.embeddings.fob.compute_visualization"
        ) as compute_vis, patch("dataset_tools.metrics.embeddings.KMeans", _FakeKMeans):
            load_model.return_value = types.SimpleNamespace(
                model=object(),
                ref=types.SimpleNamespace(provider="hf", model_id="m"),
            )
            job = emb_mod.EmbeddingsComputation(
                dataset_name="d",
                model_name="m",
                embeddings_field="emb",
                use_umap=True,
                use_cluster=True,
                n_clusters=10,
            )
            out = job.compute(dataset)

        self.assertEqual(out["field"], "emb")
        self.assertEqual(len(dataset.compute_embeddings_calls), 1)
        compute_vis.assert_called_once()
        self.assertEqual(dataset.selected_ids, ["a", "c"])
        self.assertEqual(len(dataset.set_values_calls), 1)
        field, label_map, key_field = dataset.set_values_calls[0]
        self.assertEqual(field, "emb_cluster")
        self.assertEqual(set(label_map.keys()), {"a", "c"})
        self.assertEqual(key_field, "id")
        self.assertEqual(_FakeKMeans.calls[-1][0], 2)  # min(n_clusters, valid_samples)

    def test_embeddings_computation_patch_flow_without_valid_embeddings(self):
        embeddings = [None, None]
        ids = ["x", "y"]
        dataset = _FakeDataset(embeddings=embeddings, ids=ids)

        with patch("dataset_tools.metrics.embeddings.load_model") as load_model, patch(
            "dataset_tools.metrics.embeddings.fob.compute_visualization"
        ) as compute_vis, patch("dataset_tools.metrics.embeddings.KMeans", _FakeKMeans):
            load_model.return_value = types.SimpleNamespace(
                model=object(),
                ref=types.SimpleNamespace(provider="hf", model_id="m"),
            )
            job = emb_mod.EmbeddingsComputation(
                dataset_name="d",
                model_name="m",
                embeddings_field="emb",
                patches_field="patches",
                use_umap=True,
                use_cluster=True,
                n_clusters=3,
            )
            job.compute(dataset)

        self.assertEqual(len(dataset.compute_patch_embeddings_calls), 1)
        compute_vis.assert_not_called()
        self.assertEqual(dataset.patches_view.set_values_calls, [])


if __name__ == "__main__":
    unittest.main()
