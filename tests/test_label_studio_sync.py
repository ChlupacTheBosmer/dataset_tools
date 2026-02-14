from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from dataset_tools.config import load_config
from dataset_tools.label_studio.sync import preflight_validate_upload, push_view_to_label_studio


class _FakeView:
    def __init__(self, filepaths):
        self._filepaths = list(filepaths)
        self.annotate_calls = []

    def values(self, field):
        if field != "filepath":
            raise KeyError(field)
        return list(self._filepaths)

    def list_annotation_runs(self):
        return []

    def annotate(self, *args, **kwargs):
        self.annotate_calls.append((args, kwargs))
        return {"ok": True}


class _FakeProject:
    def __init__(self, title, import_paths, export_paths, storage_type="localfiles"):
        self.title = title
        self._import_paths = list(import_paths)
        self._export_paths = list(export_paths)
        self._storage_type = storage_type

    def get_import_storages(self):
        return [{"type": self._storage_type, "path": p} for p in self._import_paths]

    def get_export_storages(self):
        return [{"type": self._storage_type, "path": p} for p in self._export_paths]


class LabelStudioSyncTests(unittest.TestCase):
    def test_push_annotate_uses_project_name(self):
        cfg = load_config(
            local_config_path="/tmp/does-not-exist.json",
            overrides={
                "label_studio": {
                    "url": "https://ls.example",
                    "api_key": "abc",
                }
            },
        )
        view = _FakeView([])

        with patch("dataset_tools.label_studio.sync.install_batched_upload_patch", return_value=None):
            push_view_to_label_studio(
                view=view,
                config=cfg,
                project_name="MyProject",
                annotation_key="MyRun",
                overwrite_annotation_run=False,
            )

        self.assertEqual(len(view.annotate_calls), 1)
        _, kwargs = view.annotate_calls[0]
        self.assertEqual(kwargs.get("project_name"), "MyProject")

    def test_preflight_sdk_accepts_mappable_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = os.path.join(tmpdir, "a.jpg")
            with open(frame, "wb") as f:
                f.write(b"x")

            cfg = load_config(
                local_config_path="/tmp/does-not-exist.json",
                overrides={
                    "mount": {
                        "host_root": tmpdir,
                        "ls_document_root": "/data/images",
                        "local_files_prefix": "/data/local-files/?d=",
                    },
                    "label_studio": {
                        "source_path": "/data/images/source",
                        "target_path": "/data/images/target",
                    },
                },
            )

            project = _FakeProject(
                title="P",
                import_paths=["/data/images/source"],
                export_paths=["/data/images/target"],
            )
            view = _FakeView([frame])

            out = preflight_validate_upload(
                view=view,
                project=project,
                config=cfg,
                strategy="sdk_batched",
                strict=True,
            )
            self.assertEqual(out["total_samples"], 1)
            self.assertEqual(out["mappable_samples"], 1)
            self.assertEqual(out["skipped_samples"], 0)

    def test_preflight_sdk_rejects_unmappable_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outside = "/tmp/outside_preflight.jpg"
            with open(outside, "wb") as f:
                f.write(b"x")

            cfg = load_config(
                local_config_path="/tmp/does-not-exist.json",
                overrides={
                    "mount": {
                        "host_root": tmpdir,
                        "ls_document_root": "/data/images",
                        "local_files_prefix": "/data/local-files/?d=",
                    },
                    "label_studio": {
                        "source_path": "/data/images/source",
                        "target_path": "/data/images/target",
                    },
                },
            )

            project = _FakeProject(
                title="P",
                import_paths=["/data/images/source"],
                export_paths=["/data/images/target"],
            )
            view = _FakeView([outside])

            with self.assertRaises(RuntimeError):
                preflight_validate_upload(
                    view=view,
                    project=project,
                    config=cfg,
                    strategy="sdk_batched",
                    strict=True,
                )

            os.remove(outside)

    def test_preflight_rejects_missing_project_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = os.path.join(tmpdir, "a.jpg")
            with open(frame, "wb") as f:
                f.write(b"x")

            cfg = load_config(
                local_config_path="/tmp/does-not-exist.json",
                overrides={
                    "mount": {"host_root": tmpdir},
                    "label_studio": {
                        "source_path": "/data/images/source",
                        "target_path": "/data/images/target",
                    },
                },
            )
            project = _FakeProject(
                title="P",
                import_paths=["/data/images/other_source"],
                export_paths=["/data/images/target"],
            )
            view = _FakeView([frame])

            with self.assertRaises(RuntimeError):
                preflight_validate_upload(
                    view=view,
                    project=project,
                    config=cfg,
                    strategy="sdk_batched",
                    strict=True,
                )


if __name__ == "__main__":
    unittest.main()
