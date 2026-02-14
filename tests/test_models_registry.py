from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from dataset_tools.models import registry
from dataset_tools.models.providers.fiftyone_zoo import FiftyOneZooProvider
from dataset_tools.models.spec import LoadedModel, ModelRef, normalize_provider, parse_model_ref


class _FakeProvider:
    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded
        self.calls = []

    def load(self, model_ref, *, task=None, **kwargs):
        self.calls.append((model_ref, task, kwargs))
        return self.loaded

    def list_models(self, contains=None, limit=None):
        values = ["alpha", "beta", "gamma"]
        if contains:
            values = [v for v in values if contains in v]
        if limit is not None:
            values = values[:limit]
        return values


class ModelsRegistryTests(unittest.TestCase):
    def test_parse_and_normalize(self):
        self.assertEqual(normalize_provider("huggingface"), "hf")
        self.assertEqual(normalize_provider("foz"), "foz")
        self.assertEqual(parse_model_ref("hf:facebook/dinov2-base").provider, "hf")
        self.assertEqual(parse_model_ref("facebook/dinov2-base").provider, "hf")
        self.assertEqual(parse_model_ref("zoo:mobilenet-v2-imagenet-torch").provider, "foz")
        with self.assertRaises(ValueError):
            normalize_provider("unknown")
        with self.assertRaises(ValueError):
            parse_model_ref("")

    def test_registry_helpers(self):
        providers = registry.list_providers()
        self.assertIn("hf", providers)
        self.assertIn("foz", providers)
        self.assertIn("anomalib", providers)
        self.assertEqual(registry.resolve_model_ref("hf:x").model_id, "x")

    def test_load_model_capability_check(self):
        loaded = LoadedModel(
            ref=ModelRef(provider="hf", model_id="m"),
            model=object(),
            capabilities=("inference",),
        )
        fake = _FakeProvider(loaded)
        with patch("dataset_tools.models.registry.get_provider", return_value=fake):
            out = registry.load_model("hf:m", capability=None)
            self.assertIs(out, loaded)
            with self.assertRaises(RuntimeError):
                registry.load_model("hf:m", capability="embeddings")

    def test_provider_model_list(self):
        loaded = LoadedModel(
            ref=ModelRef(provider="hf", model_id="m"),
            model=object(),
            capabilities=("embeddings",),
        )
        fake = _FakeProvider(loaded)
        with patch("dataset_tools.models.registry.get_provider", return_value=fake):
            models = registry.provider_model_list("hf", contains="a", limit=2)
        self.assertEqual(models, ["alpha", "beta"])

    def test_fiftyone_zoo_provider_list_and_load(self):
        provider = FiftyOneZooProvider()
        with patch("dataset_tools.models.providers.fiftyone_zoo.foz.list_zoo_models", return_value=["clip", "mobile"]), patch(
            "dataset_tools.models.providers.fiftyone_zoo.foz.load_zoo_model",
            return_value=types.SimpleNamespace(has_embeddings=True),
        ):
            names = provider.list_models(contains="c", limit=5)
            self.assertEqual(names, ["clip"])
            out = provider.load(ModelRef(provider="foz", model_id="clip"))
        self.assertIn("embeddings", out.capabilities)


if __name__ == "__main__":
    unittest.main()
