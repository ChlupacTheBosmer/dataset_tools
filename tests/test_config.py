from __future__ import annotations

import json
import os
import tempfile
import unittest

from dataset_tools.config import load_config


class ConfigLoadTests(unittest.TestCase):
    def test_local_env_and_override_precedence(self):
        local_payload = {
            "label_studio": {
                "url": "https://local.example",
                "batch_size": 20,
            },
            "dataset": {
                "name": "local_dataset",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "local_config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(local_payload, f)

            old_url = os.environ.get("LABEL_STUDIO_URL")
            old_dataset = os.environ.get("FIFTYONE_DATASET_NAME")
            os.environ["LABEL_STUDIO_URL"] = "https://env.example"
            os.environ["FIFTYONE_DATASET_NAME"] = "env_dataset"

            try:
                cfg = load_config(
                    local_config_path=path,
                    overrides={"label_studio": {"batch_size": 5}},
                )
            finally:
                if old_url is None:
                    os.environ.pop("LABEL_STUDIO_URL", None)
                else:
                    os.environ["LABEL_STUDIO_URL"] = old_url

                if old_dataset is None:
                    os.environ.pop("FIFTYONE_DATASET_NAME", None)
                else:
                    os.environ["FIFTYONE_DATASET_NAME"] = old_dataset

            self.assertEqual(cfg.label_studio.url, "https://env.example")
            self.assertEqual(cfg.dataset.name, "env_dataset")
            self.assertEqual(cfg.label_studio.batch_size, 5)

    def test_default_upload_strategy_is_sdk_batched(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing_local_config.json")
            cfg = load_config(local_config_path=missing_path)
            self.assertEqual(cfg.label_studio.upload_strategy, "sdk_batched")


if __name__ == "__main__":
    unittest.main()
