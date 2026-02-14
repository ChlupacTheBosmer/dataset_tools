from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from argparse import Namespace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dataset_tools import dst as dst_cli


class _FakeProject:
    def __init__(self, pid, title, tasks=None):
        self.id = pid
        self.title = title
        self._tasks = list(tasks or [])
        self.cleared = False

    def get_tasks(self):
        return list(self._tasks)

    def delete_all_tasks(self):
        self.cleared = True


class _FakeJob:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def run(self):
        return {"ok": True, "kwargs": self.kwargs}


class DSTCommandTests(unittest.TestCase):
    def test_json_parsers_and_helpers(self):
        self.assertEqual(dst_cli._parse_json_dict(None, "--x"), {})
        self.assertEqual(dst_cli._parse_json_dict('{"a":1}', "--x"), {"a": 1})
        self.assertEqual(dst_cli._parse_json_list("[1,2]", "--x"), [1, 2])
        self.assertEqual(dst_cli._parse_class_map('{"0":"rodent"}'), {0: "rodent"})
        self.assertEqual(dst_cli._parse_class_map(None), {})

        with self.assertRaises(ValueError):
            dst_cli._parse_json_dict("[]", "--x")
        with self.assertRaises(ValueError):
            dst_cli._parse_json_list("{}", "--x")
        with self.assertRaises(ValueError):
            dst_cli._parse_json_value("{oops", "--x")

    def test_misc_helpers(self):
        self.assertEqual(dst_cli._mask_api_key(""), "")
        self.assertEqual(dst_cli._mask_api_key("1234"), "****")

        class LSGetOnly:
            def get_projects(self):
                return [1, 2]

        self.assertEqual(dst_cli._list_projects(LSGetOnly()), [1, 2])
        with self.assertRaises(RuntimeError):
            dst_cli._list_projects(object())

        p = _FakeProject(1, "A", tasks=[1, 2])
        self.assertEqual(dst_cli._project_payload(p, include_task_count=True)["task_count"], 2)

        class BrokenProject(_FakeProject):
            def get_tasks(self):
                raise RuntimeError("boom")

        b = BrokenProject(2, "B")
        self.assertIsNone(dst_cli._project_payload(b, include_task_count=True)["task_count"])

    def test_print_result_variants(self):
        with patch("sys.stdout", new_callable=StringIO) as out:
            dst_cli._print_result({"a": 1})
            dst_cli._print_result([1, 2])
            dst_cli._print_result("x")
            dst_cli._print_result(None)
            text = out.getvalue()
        self.assertIn('"a": 1', text)
        self.assertIn("[", text)
        self.assertIn("x", text)

    def test_resolve_project(self):
        projects = [_FakeProject(1, "A"), _FakeProject(2, "B")]
        ls = SimpleNamespace(
            get_project=lambda pid: _FakeProject(pid, f"P{pid}"),
            list_projects=lambda: projects,
        )
        self.assertEqual(dst_cli._resolve_project(ls, 9, None).id, 9)
        self.assertEqual(dst_cli._resolve_project(ls, None, "A").id, 1)

        with self.assertRaises(ValueError):
            dst_cli._resolve_project(ls, None, None)
        with self.assertRaises(RuntimeError):
            dst_cli._resolve_project(ls, None, "missing")

        ls_dup = SimpleNamespace(list_projects=lambda: [_FakeProject(1, "A"), _FakeProject(2, "A")])
        with self.assertRaises(RuntimeError):
            dst_cli._resolve_project(ls_dup, None, "A")

    def test_cmd_ls_commands(self):
        cfg = types.SimpleNamespace(label_studio=types.SimpleNamespace(url="https://ls"), dataset=types.SimpleNamespace(name="d"))
        projects = [_FakeProject(1, "Alpha", tasks=[1]), _FakeProject(2, "Beta", tasks=[1, 2])]
        ls = types.SimpleNamespace(
            list_projects=lambda: projects,
            delete_project=MagicMock(),
            get_project=lambda pid: projects[0],
        )

        with patch("dataset_tools.dst._load_app_config", return_value=cfg), patch(
            "dataset_tools.label_studio.client.ensure_label_studio_client",
            return_value=ls,
        ):
            out = dst_cli.cmd_ls_test(Namespace(list_projects=True))
            self.assertEqual(out["project_count"], 2)
            out_list = dst_cli.cmd_ls_project_list(
                Namespace(contains="alp", case_sensitive=False, limit=1, with_task_count=True)
            )
            self.assertEqual(out_list["count"], 1)

            out_clear_dry = dst_cli.cmd_ls_project_clear_tasks(
                Namespace(id=1, title=None, dry_run=True)
            )
            self.assertTrue(out_clear_dry["dry_run"])
            self.assertFalse(projects[0].cleared)

            out_clear = dst_cli.cmd_ls_project_clear_tasks(Namespace(id=1, title=None, dry_run=False))
            self.assertEqual(out_clear["cleared"], 1)
            self.assertTrue(projects[0].cleared)

            out_cleanup = dst_cli.cmd_ls_project_cleanup(
                Namespace(keyword=["alpha"], dry_run=False, case_sensitive=False)
            )
            self.assertEqual(out_cleanup["matched_count"], 1)
            ls.delete_project.assert_called_once_with(1)

    def test_cmd_data_load_yolo_and_coco(self):
        cfg = types.SimpleNamespace(dataset=types.SimpleNamespace(name="default_ds"))
        fake_result = types.SimpleNamespace(dataset_name="loaded", sample_count=5)
        yolo_loader_instance = MagicMock()
        yolo_loader_instance.load.return_value = fake_result
        coco_loader_instance = MagicMock()
        coco_loader_instance.load.return_value = fake_result

        with patch("dataset_tools.dst._load_app_config", return_value=cfg), patch(
            "dataset_tools.loaders.YoloDatasetLoader",
            return_value=yolo_loader_instance,
        ), patch("dataset_tools.loaders.YoloParserConfig", side_effect=lambda **k: ("parser", k)), patch(
            "dataset_tools.loaders.ImagesLabelsSubdirResolver",
            side_effect=lambda **k: ("subdir", k),
        ), patch(
            "dataset_tools.loaders.MirroredRootsPathResolver",
            side_effect=lambda **k: ("mirror", k),
        ):
            out_root = dst_cli.cmd_data_load_yolo(
                Namespace(
                    dataset=None,
                    root="/tmp/root",
                    images_subdir="images",
                    labels_subdir="labels",
                    images_root=None,
                    labels_root=None,
                    class_map='{"0":"rodent"}',
                    no_confidence=False,
                    overwrite=True,
                    persistent=True,
                )
            )
            self.assertEqual(out_root["samples"], 5)

            out_mirror = dst_cli.cmd_data_load_yolo(
                Namespace(
                    dataset="x",
                    root=None,
                    images_subdir="images",
                    labels_subdir="labels",
                    images_root="/tmp/images",
                    labels_root="/tmp/labels",
                    class_map=None,
                    no_confidence=True,
                    overwrite=False,
                    persistent=False,
                )
            )
            self.assertFalse(out_mirror["persistent"])

            with self.assertRaises(ValueError):
                dst_cli.cmd_data_load_yolo(
                    Namespace(
                        dataset="x",
                        root=None,
                        images_subdir="images",
                        labels_subdir="labels",
                        images_root=None,
                        labels_root=None,
                        class_map=None,
                        no_confidence=False,
                        overwrite=False,
                        persistent=True,
                    )
                )

        with patch("dataset_tools.dst._load_app_config", return_value=cfg), patch(
            "dataset_tools.loaders.CocoLoaderConfig",
            side_effect=lambda **k: ("cfg", k),
        ), patch(
            "dataset_tools.loaders.CocoDatasetLoader",
            return_value=coco_loader_instance,
        ):
            out = dst_cli.cmd_data_load_coco(
                Namespace(
                    dataset=None,
                    dataset_dir="/tmp/coco",
                    data_path="data",
                    labels_path="labels.json",
                    overwrite=False,
                    persistent=True,
                )
            )
            self.assertEqual(out["dataset"], "loaded")

    def test_cmd_data_export_ls_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "tasks.json")
            with patch("dataset_tools.label_studio_json.build_tasks", return_value=[{"a": 1}, {"b": 2}]):
                out = dst_cli.cmd_data_export_ls_json(
                    Namespace(root=tmpdir, ls_root="/data/local-files/?d=x", output=out_path)
                )
            self.assertEqual(out["tasks"], 2)
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(len(payload), 2)

    def test_cmd_metric_and_brain_commands(self):
        with patch("dataset_tools.metrics.EmbeddingsComputation", _FakeJob), patch(
            "dataset_tools.metrics.UniquenessComputation",
            _FakeJob,
        ), patch("dataset_tools.metrics.MistakennessComputation", _FakeJob), patch(
            "dataset_tools.metrics.HardnessComputation",
            _FakeJob,
        ), patch(
            "dataset_tools.metrics.RepresentativenessComputation",
            _FakeJob,
        ), patch("dataset_tools.brain.VisualizationOperation", _FakeJob), patch(
            "dataset_tools.brain.SimilarityOperation",
            _FakeJob,
        ), patch("dataset_tools.brain.ExactDuplicatesOperation", _FakeJob), patch(
            "dataset_tools.brain.NearDuplicatesOperation",
            _FakeJob,
        ), patch(
            "dataset_tools.brain.LeakySplitsOperation",
            _FakeJob,
        ):
            self.assertTrue(dst_cli.cmd_metrics_embeddings(Namespace(dataset="d", model="m", model_ref=None, embeddings_field="emb", patches_field=None, no_umap=False, no_cluster=False, n_clusters=2))["ok"])
            self.assertTrue(dst_cli.cmd_metrics_uniqueness(Namespace(dataset="d", embeddings_field="emb", output_field="u"))["ok"])
            self.assertTrue(dst_cli.cmd_metrics_mistakenness(Namespace(dataset="d", pred_field="p", gt_field="g", mistakenness_field="m", missing_field="mi", spurious_field="sp"))["ok"])
            self.assertTrue(dst_cli.cmd_metrics_hardness(Namespace(dataset="d", label_field="l", output_field="h"))["ok"])
            self.assertTrue(dst_cli.cmd_metrics_representativeness(Namespace(dataset="d", output_field="r", method="cluster-center", embeddings_field="emb", roi_field=None))["ok"])
            self.assertTrue(dst_cli.cmd_brain_visualization(Namespace(dataset="d", method="umap", num_dims=2, embeddings_field=None, patches_field=None, brain_key=None))["ok"])
            self.assertTrue(dst_cli.cmd_brain_similarity(Namespace(dataset="d", embeddings_field=None, patches_field=None, roi_field=None, backend=None, brain_key=None))["ok"])
            self.assertTrue(dst_cli.cmd_brain_duplicates_exact(Namespace(dataset="d"))["ok"])
            self.assertTrue(dst_cli.cmd_brain_duplicates_near(Namespace(dataset="d", threshold=0.2, embeddings_field=None, roi_field=None))["ok"])
            self.assertTrue(dst_cli.cmd_brain_leaky_splits(Namespace(dataset="d", splits="train,val", threshold=0.2, embeddings_field=None, roi_field=None))["ok"])

    def test_cmd_models_commands(self):
        loaded = types.SimpleNamespace(
            ref=types.SimpleNamespace(provider="hf", model_id="m"),
            model=object(),
            capabilities=("embeddings",),
            metadata={"task": "embeddings"},
        )
        with patch("dataset_tools.models.list_providers", return_value=["hf", "foz"]), patch(
            "dataset_tools.models.provider_model_list",
            return_value=["a", "b"],
        ), patch("dataset_tools.models.load_model", return_value=loaded):
            providers = dst_cli.cmd_models_list(
                Namespace(provider=None, contains=None, limit=10)
            )
            self.assertEqual(providers["providers"], ["hf", "foz"])

            listed = dst_cli.cmd_models_list(
                Namespace(provider="foz", contains="a", limit=2)
            )
            self.assertEqual(listed["count"], 2)

            resolved = dst_cli.cmd_models_resolve(
                Namespace(model_ref="hf:m", default_provider="hf", task="embeddings", capability="embeddings")
            )
            self.assertEqual(resolved["provider"], "hf")

            valid = dst_cli.cmd_models_validate(
                Namespace(model_ref="hf:m", default_provider="hf", task="embeddings", capability="embeddings")
            )
            self.assertTrue(valid["valid"])

    def test_cmd_anomaly_commands(self):
        ref = types.SimpleNamespace(
            to_dict=lambda: {
                "backend": "embedding_distance",
                "embeddings_field": "emb",
                "threshold": 1.0,
                "centroid": [0.1, 0.2],
                "metadata": {},
            }
        )
        artifact = types.SimpleNamespace(
            artifact_dir="/tmp/anom",
            to_dict=lambda: {
                "dataset_name": "d",
                "model_ref": "anomalib:padim",
                "export_type": "openvino",
                "model_path": "/tmp/anom/model.xml",
                "artifact_dir": "/tmp/anom",
            },
        )

        with patch("dataset_tools.anomaly.fit_embedding_distance_reference", return_value=ref), patch(
            "dataset_tools.anomaly.save_reference"
        ) as save_ref, patch(
            "dataset_tools.anomaly.load_reference",
            return_value=ref,
        ), patch(
            "dataset_tools.anomaly.score_with_embedding_distance",
            return_value={"backend": "embedding_distance", "dataset": "d"},
        ), patch(
            "dataset_tools.anomaly.score_with_anomalib",
            return_value={"backend": "anomalib", "dataset": "d"},
        ), patch(
            "dataset_tools.anomaly.run_embedding_distance",
            return_value={"backend": "embedding_distance", "dataset": "d"},
        ), patch(
            "dataset_tools.anomaly.train_and_export_anomalib",
            return_value=artifact,
        ):
            fit_out = dst_cli.cmd_anomaly_fit(
                Namespace(
                    dataset="d",
                    backend="embedding_distance",
                    embeddings_field="emb",
                    normal_tag="normal",
                    threshold=None,
                    threshold_quantile=0.95,
                    reference_json="/tmp/ref.json",
                )
            )
            self.assertEqual(fit_out["dataset"], "d")
            save_ref.assert_called_once()

            train_out = dst_cli.cmd_anomaly_train(
                Namespace(
                    dataset="d",
                    model_ref="anomalib:padim",
                    normal_tag="normal",
                    abnormal_tag="anomaly",
                    mask_field=None,
                    artifact_dir="/tmp/anom",
                    data_dir="/tmp/anom_data",
                    artifact_format="openvino",
                    artifact_json="/tmp/anom/artifact.json",
                    image_size="256,256",
                    train_batch_size=8,
                    eval_batch_size=8,
                    num_workers=0,
                    normal_split_ratio=0.2,
                    test_split_mode="from_dir",
                    test_split_ratio=0.2,
                    val_split_mode="from_test",
                    val_split_ratio=0.5,
                    seed=13,
                    max_epochs=2,
                    accelerator="cpu",
                    devices="1",
                    copy_media=False,
                    overwrite_data=True,
                )
            )
            self.assertEqual(train_out["action"], "train_export")

            score_out = dst_cli.cmd_anomaly_score(
                Namespace(
                    dataset="d",
                    backend="embedding_distance",
                    embeddings_field="emb",
                    normal_tag=None,
                    threshold=None,
                    threshold_quantile=0.95,
                    reference_json="/tmp/ref.json",
                    model_ref="anomalib:padim",
                    artifact=None,
                    artifact_format=None,
                    anomaly_threshold=0.5,
                    device=None,
                    trust_remote_code=False,
                    tag=None,
                    score_field="anomaly_score",
                    flag_field="is_anomaly",
                    label_field=None,
                    map_field=None,
                    mask_field=None,
                )
            )
            self.assertEqual(score_out["dataset"], "d")

            score_anom_out = dst_cli.cmd_anomaly_score(
                Namespace(
                    dataset="d",
                    backend="anomalib",
                    embeddings_field="emb",
                    normal_tag=None,
                    threshold=None,
                    threshold_quantile=0.95,
                    reference_json=None,
                    model_ref="anomalib:padim",
                    artifact="/tmp/anom/artifact.json",
                    artifact_format="openvino",
                    anomaly_threshold=0.5,
                    device="CPU",
                    trust_remote_code=False,
                    tag=None,
                    score_field="anomaly_score",
                    flag_field="is_anomaly",
                    label_field="anomaly_label",
                    map_field="anomaly_map",
                    mask_field="anomaly_mask",
                )
            )
            self.assertEqual(score_anom_out["backend"], "anomalib")

            run_out = dst_cli.cmd_anomaly_run(
                Namespace(
                    dataset="d",
                    backend="embedding_distance",
                    embeddings_field="emb",
                    normal_tag=None,
                    threshold=None,
                    threshold_quantile=0.95,
                    reference_json=None,
                    model_ref="anomalib:padim",
                    artifact=None,
                    artifact_format=None,
                    anomaly_threshold=0.5,
                    device=None,
                    trust_remote_code=False,
                    tag=None,
                    score_field="anomaly_score",
                    flag_field="is_anomaly",
                    label_field=None,
                    map_field=None,
                    mask_field=None,
                )
            )
            self.assertEqual(run_out["backend"], "embedding_distance")

            run_anom_out = dst_cli.cmd_anomaly_run(
                Namespace(
                    dataset="d",
                    backend="anomalib",
                    embeddings_field="emb",
                    normal_tag=None,
                    threshold=None,
                    threshold_quantile=0.95,
                    reference_json=None,
                    model_ref="anomalib:padim",
                    artifact="/tmp/anom/artifact.json",
                    artifact_format="openvino",
                    anomaly_threshold=0.5,
                    device="CPU",
                    trust_remote_code=False,
                    tag=None,
                    score_field="anomaly_score",
                    flag_field="is_anomaly",
                    label_field="anomaly_label",
                    map_field=None,
                    mask_field=None,
                )
            )
            self.assertEqual(run_anom_out["backend"], "anomalib")

    def test_cmd_workflow_roundtrip_and_tags(self):
        cfg = types.SimpleNamespace(label_studio=types.SimpleNamespace(project_title="P"))
        fake_workflow = MagicMock()
        fake_workflow.run.return_value = [{"ok": True}]
        fake_engine = MagicMock()
        fake_engine.run.return_value = [{"engine": True}]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "out.json")
            rules_json = os.path.join(tmpdir, "rules.json")
            with open(rules_json, "w", encoding="utf-8") as f:
                json.dump([{"operation": "delete_samples", "tag": "delete"}], f)

            wf_args = Namespace(
                dataset="ds",
                tag="fix",
                project="P2",
                label_field="ground_truth",
                corrections_field="ls_corrections",
                skip_send=False,
                skip_pull=False,
                skip_sync_disk=False,
                dry_run_sync=True,
                clear_project_tasks=True,
                upload_strategy="sdk_batched",
                pull_strategy="sdk_meta",
                annotation_key="run1",
                launch_editor=False,
                create_if_missing=True,
                strict_preflight=True,
                overwrite_annotation_run=True,
                send_params='{"x":1}',
                pull_params='{"y":2}',
                sync_params='{"z":3}',
                quiet_logs=True,
                output_json=out_json,
                config=None,
                overrides=None,
            )
            with patch("dataset_tools.dst._load_app_config", return_value=cfg), patch(
                "dataset_tools.workflows.CurationRoundtripWorkflow",
                return_value=fake_workflow,
            ), patch(
                "dataset_tools.dst._execute_with_optional_log_capture",
                side_effect=lambda fn, quiet_logs: fn(),
            ):
                out = dst_cli.cmd_workflow_roundtrip(wf_args)
            self.assertEqual(out, [{"ok": True}])
            self.assertTrue(os.path.exists(out_json))

            run_args = Namespace(
                workflow=rules_json,
                dataset="ds",
                fail_fast=None,
                quiet_logs=True,
                output_json=out_json,
                config=None,
                overrides=None,
            )
            with patch("dataset_tools.dst._load_app_config", return_value=cfg), patch(
                "dataset_tools.tag_workflow.TagWorkflowEngine",
                return_value=fake_engine,
            ), patch(
                "dataset_tools.dst._execute_with_optional_log_capture",
                side_effect=lambda fn, quiet_logs: fn(),
            ):
                run_out = dst_cli.cmd_workflow_tags_run(run_args)
            self.assertEqual(run_out, [{"engine": True}])

            with open(rules_json, "w", encoding="utf-8") as f:
                json.dump("bad", f)
            with patch("dataset_tools.dst._load_app_config", return_value=cfg):
                with self.assertRaises(ValueError):
                    dst_cli.cmd_workflow_tags_run(run_args)

            inline_args = Namespace(
                dataset="ds",
                rule=['{"operation":"delete_samples","tag":"delete"}'],
                no_fail_fast=False,
                quiet_logs=True,
                output_json=out_json,
                config=None,
                overrides=None,
            )
            with patch("dataset_tools.dst._load_app_config", return_value=cfg), patch(
                "dataset_tools.tag_workflow.TagWorkflowEngine",
                return_value=fake_engine,
            ), patch(
                "dataset_tools.dst._execute_with_optional_log_capture",
                side_effect=lambda fn, quiet_logs: fn(),
            ):
                inline_out = dst_cli.cmd_workflow_tags_inline(inline_args)
            self.assertEqual(inline_out, [{"engine": True}])

    def test_cmd_sync_disk(self):
        cfg = types.SimpleNamespace(dataset=types.SimpleNamespace(name="cfg_ds"))
        with patch("dataset_tools.dst._load_app_config", return_value=cfg), patch(
            "dataset_tools.sync_from_fo_to_disk.sync_corrections_to_disk",
            return_value=9,
        ) as sync:
            out = dst_cli.cmd_sync_disk(
                Namespace(
                    dataset=None,
                    dry_run=True,
                    tag="fix",
                    corrections_field="ls_corrections",
                    label_to_class_id='{"rodent":0}',
                    default_class_id=0,
                    path_replacement=["/images/=/labels/"],
                    backup_suffix_format="%Y%m%d",
                )
            )
        sync.assert_called_once()
        self.assertEqual(out["synced_files"], 9)
        self.assertEqual(out["dataset"], "cfg_ds")

    def test_cmd_app_open_no_block_and_blocking(self):
        fake_session = MagicMock()
        fake_fo = types.SimpleNamespace(
            list_datasets=lambda: ["d"],
            load_dataset=lambda name: {"name": name},
            launch_app=lambda ds, port, address: fake_session,
        )
        with patch.dict("sys.modules", {"fiftyone": fake_fo}):
            out = dst_cli.cmd_app_open(Namespace(dataset="d", port=5151, address="0.0.0.0", no_block=True))
            self.assertTrue(out["session_open"])

            with patch("dataset_tools.dst.time.sleep", side_effect=KeyboardInterrupt), patch(
                "dataset_tools.dst.print"
            ):
                out2 = dst_cli.cmd_app_open(
                    Namespace(dataset="d", port=5151, address="0.0.0.0", no_block=False)
                )
            self.assertIsNone(out2)
            fake_session.close.assert_called_once()

        with patch.dict("sys.modules", {"fiftyone": types.SimpleNamespace(list_datasets=lambda: [])}):
            with self.assertRaises(RuntimeError):
                dst_cli.cmd_app_open(Namespace(dataset="missing", port=1, address="x", no_block=True))

    def test_main_success_and_error(self):
        with patch("dataset_tools.dst.build_parser") as build_parser:
            parser = MagicMock()
            args = Namespace(func=lambda a: {"ok": True})
            parser.parse_args.return_value = args
            build_parser.return_value = parser
            rc = dst_cli.main(["config", "show"])
        self.assertEqual(rc, 0)

        with patch("dataset_tools.dst.build_parser") as build_parser:
            parser = MagicMock()
            def _raise(_args):
                raise ValueError("bad input")
            parser.parse_args.return_value = Namespace(func=_raise)
            parser.error.side_effect = SystemExit(2)
            build_parser.return_value = parser
            with self.assertRaises(SystemExit):
                dst_cli.main(["x"])


if __name__ == "__main__":
    unittest.main()
