from __future__ import annotations

import unittest
from unittest.mock import patch

from dataset_tools.config import load_config
from dataset_tools.workflows.roundtrip import CurationRoundtripWorkflow, RoundtripWorkflowConfig


class _FakeEngine:
    def __init__(self, app_config):
        self.app_config = app_config
        self.ran_config = None

    def run(self, config):
        self.ran_config = config
        return [{"ok": True, "rule_count": len(config.rules)}]


class WorkflowsRoundtripTests(unittest.TestCase):
    def test_roundtrip_builds_all_rules(self):
        cfg = load_config(local_config_path="/tmp/does-not-exist.json")
        fake_engine = _FakeEngine(cfg)
        with patch("dataset_tools.workflows.roundtrip.TagWorkflowEngine", return_value=fake_engine):
            wf = CurationRoundtripWorkflow(app_config=cfg)
            result = wf.run(
                RoundtripWorkflowConfig(
                    dataset_name="ds",
                    send_tag="fix",
                    project_title="P1",
                    corrections_field="ls_corrections",
                    upload_strategy="sdk_batched",
                    additional_send_params={"batch": 10},
                    additional_pull_params={"pull_strategy": "sdk_meta"},
                    additional_sync_params={"dry_run": True},
                )
            )

        self.assertEqual(result[0]["rule_count"], 3)
        rules = fake_engine.ran_config.rules
        self.assertEqual(rules[0].operation, "send_to_label_studio")
        self.assertEqual(rules[1].operation, "pull_from_label_studio")
        self.assertEqual(rules[2].operation, "sync_corrections_to_disk")
        self.assertEqual(rules[0].params["upload_strategy"], "sdk_batched")
        self.assertEqual(rules[1].params["upload_strategy"], "sdk_batched")
        self.assertEqual(rules[0].params["project_title"], "P1")
        self.assertTrue(rules[2].params["dry_run"])

    def test_roundtrip_respects_skip_flags(self):
        cfg = load_config(local_config_path="/tmp/does-not-exist.json")
        fake_engine = _FakeEngine(cfg)
        with patch("dataset_tools.workflows.roundtrip.TagWorkflowEngine", return_value=fake_engine):
            wf = CurationRoundtripWorkflow(app_config=cfg)
            wf.run(
                RoundtripWorkflowConfig(
                    dataset_name="ds",
                    send_to_label_studio=False,
                    pull_from_label_studio=True,
                    sync_to_disk=False,
                )
            )

        rules = fake_engine.ran_config.rules
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].operation, "pull_from_label_studio")
        self.assertIsNone(rules[0].tag)


if __name__ == "__main__":
    unittest.main()
