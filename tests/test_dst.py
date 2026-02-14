from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from dataset_tools import dst as dst_cli


class DSTTests(unittest.TestCase):
    def test_parser_wires_top_level_commands(self):
        parser = dst_cli.build_parser()

        cases = [
            (["config", "show"], "cmd_config_show"),
            (["ls", "project", "list"], "cmd_ls_project_list"),
            (["data", "load", "yolo", "--root", "/tmp/data"], "cmd_data_load_yolo"),
            (["data", "load", "coco", "--dataset-dir", "/tmp/coco"], "cmd_data_load_coco"),
            (["metrics", "uniqueness", "--dataset", "d"], "cmd_metrics_uniqueness"),
            (["metrics", "hardness", "--dataset", "d"], "cmd_metrics_hardness"),
            (["metrics", "representativeness", "--dataset", "d"], "cmd_metrics_representativeness"),
            (["brain", "visualization", "--dataset", "d"], "cmd_brain_visualization"),
            (["brain", "similarity", "--dataset", "d"], "cmd_brain_similarity"),
            (["brain", "duplicates", "exact", "--dataset", "d"], "cmd_brain_duplicates_exact"),
            (["brain", "duplicates", "near", "--dataset", "d"], "cmd_brain_duplicates_near"),
            (["brain", "leaky-splits", "--dataset", "d", "--splits", "train,val"], "cmd_brain_leaky_splits"),
            (["models", "list"], "cmd_models_list"),
            (["models", "resolve", "--model-ref", "hf:facebook/dinov2-base"], "cmd_models_resolve"),
            (["models", "validate", "--model-ref", "foz:mobilenet-v2-imagenet-torch"], "cmd_models_validate"),
            (["anomaly", "fit", "--dataset", "d"], "cmd_anomaly_fit"),
            (["anomaly", "train", "--dataset", "d"], "cmd_anomaly_train"),
            (["anomaly", "score", "--dataset", "d"], "cmd_anomaly_score"),
            (["anomaly", "run", "--dataset", "d"], "cmd_anomaly_run"),
            (
                ["workflow", "roundtrip", "--dataset", "d", "--quiet-logs", "--output-json", "/tmp/out.json"],
                "cmd_workflow_roundtrip",
            ),
            (
                [
                    "workflow",
                    "tags",
                    "run",
                    "--workflow",
                    "/tmp/wf.json",
                    "--quiet-logs",
                    "--output-json",
                    "/tmp/out.json",
                ],
                "cmd_workflow_tags_run",
            ),
            (
                [
                    "workflow",
                    "tags",
                    "inline",
                    "--dataset",
                    "d",
                    "--rule",
                    '{"operation":"delete_samples"}',
                    "--quiet-logs",
                    "--output-json",
                    "/tmp/out.json",
                ],
                "cmd_workflow_tags_inline",
            ),
            (["sync", "disk"], "cmd_sync_disk"),
            (["app", "open", "--dataset", "d"], "cmd_app_open"),
        ]

        for argv, expected_func_name in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertEqual(args.func.__name__, expected_func_name)

    def test_config_show_masks_api_key(self):
        payload = {
            "label_studio": {
                "api_key": "supersecretkey",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "local_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            parser = dst_cli.build_parser()
            args = parser.parse_args(["config", "show", "--config", cfg_path])
            out = args.func(args)

            self.assertEqual(out["label_studio"]["api_key"], "**********tkey")

    def test_config_show_can_reveal_secrets(self):
        payload = {
            "label_studio": {
                "api_key": "supersecretkey",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "local_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            parser = dst_cli.build_parser()
            args = parser.parse_args(["config", "show", "--config", cfg_path, "--show-secrets"])
            out = args.func(args)

            self.assertEqual(out["label_studio"]["api_key"], "supersecretkey")

    def test_parse_path_replacements(self):
        out = dst_cli._parse_path_replacements(["/images/=/labels/", "/frames/=/labels/"])
        self.assertEqual(out, (("/images/", "/labels/"), ("/frames/", "/labels/")))

    def test_parse_path_replacements_rejects_invalid(self):
        with self.assertRaises(ValueError):
            dst_cli._parse_path_replacements(["/images/->/labels/"])

    def test_parse_csv_list(self):
        self.assertEqual(
            dst_cli._parse_csv_list("train, val ,test", "--splits"),
            ["train", "val", "test"],
        )

    def test_parse_csv_list_rejects_empty(self):
        with self.assertRaises(ValueError):
            dst_cli._parse_csv_list(" ,  ", "--splits")

    def test_write_json_output(self):
        payload = {"a": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "nested", "result.json")
            dst_cli._write_json_output(out_path, payload)
            with open(out_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        self.assertEqual(loaded, payload)

    def test_execute_with_optional_log_capture_suppresses_stdout(self):
        def noisy():
            print("noisy stdout")
            return {"ok": True}

        with patch("sys.stdout", new_callable=StringIO) as fake_stdout:
            out = dst_cli._execute_with_optional_log_capture(noisy, quiet_logs=True)

        self.assertEqual(out, {"ok": True})
        self.assertEqual(fake_stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
