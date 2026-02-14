from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dataset_tools.config import load_config
from dataset_tools.label_studio import storage, uploader


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return dict(self._payload)


class _FakeProject:
    def __init__(self, title="P", import_storages=None, export_storages=None):
        self.title = title
        self.id = 123
        self._import_storages = list(import_storages or [])
        self._export_storages = list(export_storages or [])
        self.connected_local = []
        self.predictions = []
        self._task_ids = [901, 902, 903, 904]

    def get_import_storages(self):
        return list(self._import_storages)

    def get_export_storages(self):
        return list(self._export_storages)

    def connect_local_import_storage(self, path):
        self.connected_local.append(path)

    def get_tasks(self, only_ids=False):
        if only_ids:
            return list(self._task_ids)
        return [{"id": i} for i in self._task_ids]

    def create_predictions(self, preds):
        self.predictions.append(preds)


class _FakeLS:
    def __init__(self, projects=None, create_project_result=None):
        self.projects = list(projects or [])
        self.create_project_result = create_project_result
        self.calls = []
        self.deleted = []
        self.responses = []

    def list_projects(self):
        return list(self.projects)

    def get_projects(self):
        return list(self.projects)

    def create_project(self, title, label_config):
        self.calls.append(("create_project", title, label_config))
        return self.create_project_result

    def make_request(self, method, path, json=None, data=None, files=None, params=None):
        self.calls.append((method, path, json, data, files, params))
        if self.responses:
            return self.responses.pop(0)
        return _Response(201, {"id": 777})


class _FakeClient:
    def __init__(self, list_projects_result):
        self._projects = list_projects_result
        self.headers = {}
        self.requests = []

    def list_projects(self):
        return list(self._projects)

    def get_project(self, project_id):
        return SimpleNamespace(id=project_id, title=f"restored-{project_id}")

    def make_request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if path.endswith("/import") and kwargs.get("files") is not None:
            return _Response(200, {"file_upload_ids": [1, 2, 3]})
        return _Response(200, {"ok": True})


class LabelStudioStorageTests(unittest.TestCase):
    def _config(self, tmpdir):
        return load_config(
            local_config_path="/tmp/does-not-exist.json",
            overrides={
                "mount": {"host_root": tmpdir, "ls_document_root": "/data/images"},
                "label_studio": {
                    "source_path": "/data/images/source",
                    "source_title": "Source",
                    "target_path": "/data/images/target",
                    "project_title": "P",
                },
                "dataset": {"label_to_class_id": {"rodent": 0}},
            },
        )

    def test_helpers(self):
        self.assertTrue(storage._is_local_storage({"type": "local"}))
        self.assertTrue(storage._is_local_storage({"type": "localfiles"}))
        self.assertFalse(storage._is_local_storage({"type": "s3"}))

        xml = storage.build_rectangle_label_config(["rodent", "rat"])
        self.assertIn('value="rodent"', xml)
        self.assertIn('value="rat"', xml)
        self.assertIn("RectangleLabels", xml)

        xml_default = storage.build_rectangle_label_config([])
        self.assertIn('value="Object"', xml_default)

    def test_list_projects_fallbacks(self):
        ls = _FakeLS(projects=[SimpleNamespace(id=1, title="A")])
        self.assertEqual(len(storage._list_projects(ls)), 1)

        class OnlyGet:
            def get_projects(self):
                return [SimpleNamespace(id=2, title="B")]

        self.assertEqual(len(storage._list_projects(OnlyGet())), 1)

        with self.assertRaises(RuntimeError):
            storage._list_projects(object())

    def test_find_project(self):
        ls = _FakeLS(projects=[SimpleNamespace(id=1, title="A"), SimpleNamespace(id=2, title="B")])
        found = storage.find_project(ls, "B")
        self.assertEqual(found.id, 2)
        self.assertIsNone(storage.find_project(ls, "missing"))

    def test_ensure_project_reuses_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)
            existing = _FakeProject(title="P")
            ls = _FakeLS(projects=[existing])
            with patch("dataset_tools.label_studio.storage.ensure_local_storage") as ensure_local, patch(
                "dataset_tools.label_studio.storage.ensure_target_storage"
            ) as ensure_target:
                project = storage.ensure_project(ls, cfg)
            self.assertIs(project, existing)
            ensure_local.assert_called_once()
            ensure_target.assert_called_once()

    def test_ensure_project_creates_new(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)
            created = _FakeProject(title="P")
            ls = _FakeLS(projects=[], create_project_result=created)
            with patch("dataset_tools.label_studio.storage.ensure_local_storage"), patch(
                "dataset_tools.label_studio.storage.ensure_target_storage"
            ):
                project = storage.ensure_project(ls, cfg, title="P")
            self.assertIs(project, created)
            self.assertTrue(any(call[0] == "create_project" for call in ls.calls))

    def test_ensure_project_raises_without_create_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)

            class NoCreate:
                def list_projects(self):
                    return []

            with self.assertRaises(RuntimeError):
                storage.ensure_project(NoCreate(), cfg)

    def test_ensure_local_storage_create_and_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)
            project = _FakeProject(import_storages=[])
            ls = _FakeLS()
            ls.responses = [_Response(201, {"id": 11}), _Response(200, {})]
            storage.ensure_local_storage(ls, project, cfg)
            paths = [call[1] for call in ls.calls if call[0] == "POST"]
            self.assertIn("/api/storages/localfiles", paths)
            self.assertIn("/api/storages/localfiles/11/sync", paths)

    def test_ensure_local_storage_noop_if_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)
            project = _FakeProject(import_storages=[{"type": "localfiles", "path": cfg.label_studio.source_path}])
            ls = _FakeLS()
            storage.ensure_local_storage(ls, project, cfg)
            self.assertEqual(ls.calls, [])

    def test_ensure_local_storage_failure_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)
            project = _FakeProject(import_storages=[])
            ls = _FakeLS()
            ls.responses = [_Response(500, {}, text="boom")]
            with self.assertRaises(RuntimeError):
                storage.ensure_local_storage(ls, project, cfg)

    def test_ensure_target_storage_create_and_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)
            project = _FakeProject(export_storages=[])
            ls = _FakeLS()
            ls.responses = [_Response(201, {"id": 22}), _Response(200, {})]
            storage.ensure_target_storage(ls, project, cfg)
            host_target = cfg.label_studio.target_path.replace(cfg.mount.ls_document_root, cfg.mount.host_root)
            self.assertTrue(os.path.isdir(host_target))
            paths = [call[1] for call in ls.calls if call[0] == "POST"]
            self.assertIn("/api/storages/export/localfiles", paths)
            self.assertIn("/api/storages/export/localfiles/22/sync", paths)

    def test_ensure_target_storage_noop_if_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)
            project = _FakeProject(export_storages=[{"type": "localfiles", "path": cfg.label_studio.target_path}])
            ls = _FakeLS()
            storage.ensure_target_storage(ls, project, cfg)
            self.assertEqual(ls.calls, [])

    def test_ensure_target_storage_failure_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(tmpdir)
            project = _FakeProject(export_storages=[])
            ls = _FakeLS()
            ls.responses = [_Response(500, {}, text="bad target")]
            with self.assertRaises(RuntimeError):
                storage.ensure_target_storage(ls, project, cfg)


class LabelStudioUploaderTests(unittest.TestCase):
    def setUp(self):
        uploader._PATCHED_BATCH_SIZE = None

    def _install_patch(self, batch_size=2):
        fake_fols = types.ModuleType("fiftyone.utils.labelstudio")

        class API:
            def __init__(self, client):
                self._client = client

            def _init_project(self, config, samples):
                return {"original": True, "project_name": config.project_name}

        fake_fols.LabelStudioAnnotationAPI = API

        fake_fiftyone = types.ModuleType("fiftyone")
        fake_utils = types.ModuleType("fiftyone.utils")
        fake_utils.labelstudio = fake_fols
        fake_fiftyone.utils = fake_utils

        patcher = patch.dict(
            sys.modules,
            {
                "fiftyone": fake_fiftyone,
                "fiftyone.utils": fake_utils,
                "fiftyone.utils.labelstudio": fake_fols,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        uploader.install_batched_upload_patch(batch_size=batch_size)
        return fake_fols.LabelStudioAnnotationAPI

    def test_install_patch_reuses_existing_project(self):
        API = self._install_patch(batch_size=2)

        listed = [SimpleNamespace(title="MyProject", id=42)]
        client = _FakeClient(listed)
        api = API(client)
        cfg = SimpleNamespace(project_name="MyProject")
        samples = SimpleNamespace(_root_dataset=SimpleNamespace(name="ds"))

        out = api._init_project(cfg, samples)
        self.assertEqual(out.id, 42)

    def test_install_patch_falls_back_to_original_init(self):
        API = self._install_patch(batch_size=2)
        client = _FakeClient([])
        api = API(client)
        cfg = SimpleNamespace(project_name="Missing")
        samples = SimpleNamespace(_root_dataset=SimpleNamespace(name="ds"))

        out = api._init_project(cfg, samples)
        self.assertEqual(out["project_name"], "Missing")

    def test_batched_upload_local_storage_path(self):
        API = self._install_patch(batch_size=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "a.jpg")
            f2 = os.path.join(tmpdir, "b.jpg")
            for fp in (f1, f2):
                with open(fp, "wb") as f:
                    f.write(b"x")

            os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = tmpdir
            os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
            self.addCleanup(os.environ.pop, "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", None)
            self.addCleanup(os.environ.pop, "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED", None)

            project = _FakeProject()
            client = _FakeClient([])
            api = API(client)
            tasks = [
                {"source_id": "s1", "media_type": "image", "image": f1},
                {"source_id": "s2", "media_type": "image", "image": f2},
            ]
            preds = {"ground_truth": {"s1": [{"x": 1}], "s2": [{"x": 2}]}}

            with patch(
                "dataset_tools.label_studio.uploader.os.path.commonprefix",
                return_value=f"{tmpdir}/shared",
            ):
                mapping = api._upload_tasks(project, tasks, predictions=preds)

        self.assertEqual(set(mapping.values()), {"s1", "s2"})
        self.assertTrue(project.connected_local)
        self.assertEqual(len(project.predictions), 1)

    def test_batched_upload_file_upload_path(self):
        API = self._install_patch(batch_size=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "a.jpg")
            f2 = os.path.join(tmpdir, "b.jpg")
            for fp in (f1, f2):
                with open(fp, "wb") as f:
                    f.write(b"img")

            os.environ.pop("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", None)
            os.environ.pop("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED", None)

            project = _FakeProject()
            client = _FakeClient([])
            api = API(client)
            tasks = [
                {"source_id": "s1", "media_type": "image", "image": f1},
                {"source_id": "s2", "media_type": "image", "image": f2},
            ]

            mapping = api._upload_tasks(project, tasks, predictions=None)

        self.assertEqual(set(mapping.values()), {"s1", "s2"})
        self.assertNotIn("Content-Type", client.headers)
        paths = [item[1] for item in client.requests]
        self.assertIn(f"/api/projects/{project.id}/import", paths)
        self.assertIn(f"/api/projects/{project.id}/reimport", paths)


if __name__ == "__main__":
    unittest.main()
