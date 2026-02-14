from __future__ import annotations

import os
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dataset_tools.config import load_config
from dataset_tools.tag_workflow.context import TagWorkflowContext
from dataset_tools.tag_workflow.operations import core


class _FakeSample:
    def __init__(self, sample_id: str, filepath: str):
        self.id = sample_id
        self.filepath = filepath


class _FakeView:
    def __init__(self, samples):
        self._samples = list(samples)

    def __len__(self):
        return len(self._samples)

    def values(self, field):
        if field != "filepath":
            raise KeyError(field)
        return [s.filepath for s in self._samples]

    def __iter__(self):
        return iter(self._samples)


class _FakeDataset:
    def __init__(self):
        self.deleted = []
        self.match_tags_calls = []
        self.by_tag = {}

    def delete_samples(self, view):
        self.deleted.append(view)

    def match_tags(self, tag):
        self.match_tags_calls.append(tag)
        return self.by_tag.get(tag, _FakeView([]))


class _FakeProject:
    def __init__(self, title="Proj"):
        self.title = title
        self.id = 55


def _context(dataset=None):
    ds = dataset or _FakeDataset()
    cfg = load_config(local_config_path="/tmp/does-not-exist.json")
    return TagWorkflowContext(dataset=ds, dataset_name="ds", app_config=cfg)


class TagWorkflowCoreOperationTests(unittest.TestCase):
    def test_app_config_overrides_applied(self):
        cfg = load_config(local_config_path="/tmp/does-not-exist.json")
        out = core._app_config_with_ls_overrides(
            cfg,
            {
                "url": "https://new-ls.example",
                "batch_size": 99,
                "label_to_class_id": {"rodent": 4},
                "default_class_id": 5,
            },
        )
        self.assertEqual(out.label_studio.url, "https://new-ls.example")
        self.assertEqual(out.label_studio.batch_size, 99)
        self.assertEqual(out.dataset.label_to_class_id, {"rodent": 4})
        self.assertEqual(out.dataset.default_class_id, 5)

    def test_delete_samples_operation(self):
        dataset = _FakeDataset()
        view = _FakeView([_FakeSample("a", "/tmp/a.jpg"), _FakeSample("b", "/tmp/b.jpg")])
        payload = core.DeleteSamplesOperation().execute(_context(dataset), view, {}, "delete")
        self.assertEqual(payload["deleted_samples"], 2)
        self.assertEqual(len(dataset.deleted), 1)

    def test_delete_files_and_samples_operation(self):
        dataset = _FakeDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "a.jpg")
            p2 = os.path.join(tmpdir, "b.jpg")
            with open(p1, "wb") as f:
                f.write(b"a")
            # p2 intentionally missing
            view = _FakeView([_FakeSample("a", p1), _FakeSample("b", p2)])
            payload = core.DeleteFilesAndSamplesOperation().execute(_context(dataset), view, {}, "delete")

        self.assertEqual(payload["deleted_files"], 1)
        self.assertEqual(payload["deleted_samples"], 2)
        self.assertEqual(len(dataset.deleted), 1)

    def test_move_samples_to_dataset_requires_target(self):
        with self.assertRaises(ValueError):
            core.MoveSamplesToDatasetOperation().execute(_context(), _FakeView([]), {}, "dup")

    def test_move_samples_to_existing_dataset(self):
        dataset = _FakeDataset()
        view = _FakeView([_FakeSample("a", "/tmp/a.jpg")])
        target = SimpleNamespace(add_samples=lambda v: setattr(target, "added", v))

        with patch("dataset_tools.tag_workflow.operations.core.fo.list_datasets", return_value=["dst_b"]), patch(
            "dataset_tools.tag_workflow.operations.core.fo.load_dataset",
            return_value=target,
        ):
            payload = core.MoveSamplesToDatasetOperation().execute(
                _context(dataset),
                view,
                {"target_dataset": "dst_b", "remove_from_source": False},
                "dup",
            )

        self.assertEqual(payload["moved_samples"], 1)
        self.assertEqual(payload["target_dataset"], "dst_b")
        self.assertIs(target.added, view)
        self.assertEqual(len(dataset.deleted), 0)

    def test_move_samples_to_new_dataset_and_remove(self):
        dataset = _FakeDataset()
        view = _FakeView([_FakeSample("a", "/tmp/a.jpg")])
        created = SimpleNamespace(name="new_ds", persistent=False)
        created.add_samples = lambda v: setattr(created, "added", v)

        with patch("dataset_tools.tag_workflow.operations.core.fo.list_datasets", return_value=[]), patch(
            "dataset_tools.tag_workflow.operations.core.fo.Dataset",
            return_value=created,
        ):
            payload = core.MoveSamplesToDatasetOperation().execute(
                _context(dataset),
                view,
                {"target_dataset": "new_ds"},
                "dup",
            )

        self.assertTrue(created.persistent)
        self.assertIs(created.added, view)
        self.assertEqual(payload["moved_samples"], 1)
        self.assertEqual(len(dataset.deleted), 1)

    def test_send_to_label_studio_noop_on_empty_view(self):
        out = core.SendToLabelStudioOperation().execute(_context(), _FakeView([]), {}, "fix")
        self.assertEqual(out["sent_samples"], 0)

    def test_send_to_label_studio_sdk_strategy_with_cache(self):
        context = _context()
        view = _FakeView([_FakeSample("a", "/tmp/a.jpg"), _FakeSample("b", "/tmp/b.jpg")])
        context.caches["ls_client"] = "cached-client"

        with patch("dataset_tools.tag_workflow.operations.core.ensure_project", return_value=_FakeProject("P")) as ensure_project, patch(
            "dataset_tools.tag_workflow.operations.core.preflight_validate_upload",
            return_value={"ok": True},
        ) as preflight, patch(
            "dataset_tools.tag_workflow.operations.core.delete_project_tasks",
        ) as clear_tasks, patch(
            "dataset_tools.tag_workflow.operations.core.push_view_to_label_studio_sdk",
            return_value=2,
        ) as push_sdk:
            out = core.SendToLabelStudioOperation().execute(
                context,
                view,
                {
                    "project_title": "P",
                    "upload_strategy": "sdk_batched",
                    "clear_project_tasks": True,
                    "strict_preflight": True,
                },
                "fix",
            )

        self.assertEqual(out["strategy"], "sdk_batched")
        self.assertEqual(out["sent_samples"], 2)
        ensure_project.assert_called_once()
        preflight.assert_called_once()
        clear_tasks.assert_called_once()
        push_sdk.assert_called_once()
        self.assertEqual(context.caches["ls_project"].title, "P")

    def test_send_to_label_studio_annotate_strategy(self):
        context = _context()
        view = _FakeView([_FakeSample("a", "/tmp/a.jpg")])
        result = SimpleNamespace(uploaded_tasks={1: "a"})

        with patch("dataset_tools.tag_workflow.operations.core.ensure_label_studio_client", return_value="ls") as ensure_client, patch(
            "dataset_tools.tag_workflow.operations.core.ensure_project",
            return_value=_FakeProject("P"),
        ), patch(
            "dataset_tools.tag_workflow.operations.core.preflight_validate_upload",
            return_value={"ok": True},
        ), patch(
            "dataset_tools.tag_workflow.operations.core.push_view_to_label_studio",
            return_value=result,
        ) as push_annotate:
            out = core.SendToLabelStudioOperation().execute(
                context,
                view,
                {"project_title": "P", "upload_strategy": "annotate_batched", "annotation_key": "runX"},
                "fix",
            )

        ensure_client.assert_called_once()
        push_annotate.assert_called_once()
        self.assertEqual(out["strategy"], "annotate_batched")
        self.assertEqual(out["sent_samples"], 1)

    def test_pull_from_label_studio_annotate_run_strategy(self):
        context = _context()
        with patch("dataset_tools.tag_workflow.operations.core.ensure_label_studio_client", return_value="ls"), patch(
            "dataset_tools.tag_workflow.operations.core.pull_labeled_tasks_from_annotation_run",
            return_value=3,
        ) as pull:
            out = core.PullFromLabelStudioOperation().execute(
                context,
                _FakeView([]),
                {"project_title": "P", "upload_strategy": "annotate_batched", "annotation_key": "run1"},
                None,
            )

        pull.assert_called_once()
        self.assertEqual(out["strategy"], "annotate_run")
        self.assertEqual(out["updated_samples"], 3)

    def test_pull_from_label_studio_sdk_meta_project_not_found(self):
        with patch("dataset_tools.tag_workflow.operations.core.ensure_label_studio_client", return_value="ls"), patch(
            "dataset_tools.tag_workflow.operations.core.find_project",
            return_value=None,
        ):
            with self.assertRaises(RuntimeError):
                core.PullFromLabelStudioOperation().execute(
                    _context(),
                    _FakeView([]),
                    {"project_title": "missing", "pull_strategy": "sdk_meta"},
                    None,
                )

    def test_pull_from_label_studio_sdk_meta_can_create_missing(self):
        context = _context()
        with patch("dataset_tools.tag_workflow.operations.core.ensure_label_studio_client", return_value="ls"), patch(
            "dataset_tools.tag_workflow.operations.core.find_project",
            return_value=None,
        ), patch(
            "dataset_tools.tag_workflow.operations.core.ensure_project",
            return_value=_FakeProject("P"),
        ) as ensure_project, patch(
            "dataset_tools.tag_workflow.operations.core.pull_labeled_tasks_to_fiftyone",
            return_value=4,
        ) as pull_sdk:
            out = core.PullFromLabelStudioOperation().execute(
                context,
                _FakeView([]),
                {"project_title": "P", "pull_strategy": "sdk_meta", "create_if_missing": True},
                None,
            )

        ensure_project.assert_called_once()
        pull_sdk.assert_called_once()
        self.assertEqual(out["updated_samples"], 4)
        self.assertEqual(out["strategy"], "sdk_meta")

    def test_sync_corrections_to_disk_operation(self):
        context = _context()
        with patch("dataset_tools.tag_workflow.operations.core.sync_corrections_to_disk", return_value=7) as sync:
            out = core.SyncCorrectionsToDiskOperation().execute(
                context,
                _FakeView([]),
                {"dry_run": True, "default_class_id": 9, "backup_suffix_format": "%Y%m%d"},
                "fix",
            )
        sync.assert_called_once()
        self.assertEqual(out["synced_files"], 7)
        self.assertTrue(out["dry_run"])

    def test_default_registry_contains_core_operations(self):
        registry = core.default_operations_registry()
        for name in (
            "delete_samples",
            "delete_files_and_samples",
            "move_samples_to_dataset",
            "send_to_label_studio",
            "pull_from_label_studio",
            "sync_corrections_to_disk",
        ):
            self.assertIn(name, registry)


if __name__ == "__main__":
    unittest.main()
