from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image

import fiftyone as fo  # type: ignore
import fiftyone.zoo as foz  # type: ignore

from dataset_tools import dst as dst_cli
from dataset_tools.metrics import EmbeddingsComputation
from dataset_tools.sync_from_fo_to_disk import infer_label_path

RUN_REAL_LIFE_TESTS = os.getenv("RUN_REAL_LIFE_TESTS") == "1"
RUN_REAL_LIFE_NETWORK_TESTS = os.getenv("RUN_REAL_LIFE_NETWORK_TESTS", "1") == "1"


def _sample_bbox(index: int) -> tuple[float, float, float, float]:
    x = 0.2 + (index % 3) * 0.02
    y = 0.25 + (index % 2) * 0.02
    w = 0.3
    h = 0.3
    return x, y, w, h


@unittest.skipUnless(
    RUN_REAL_LIFE_TESTS,
    "Set RUN_REAL_LIFE_TESTS=1 to run real-life integration tests",
)
class RealLifeDatasetToolsTests(unittest.TestCase):
    def setUp(self):
        self._tmp_dirs: list[str] = []
        self._datasets: set[str] = set()

    def tearDown(self):
        for dataset_name in sorted(self._datasets):
            if dataset_name in fo.list_datasets():
                fo.delete_dataset(dataset_name)

        for path in self._tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)

    def _new_name(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _register_dataset(self, dataset_name: str):
        self._datasets.add(dataset_name)

    def _mktempdir(self, prefix: str) -> str:
        path = tempfile.mkdtemp(prefix=prefix)
        self._tmp_dirs.append(path)
        return path

    def _run_dst_func(self, argv: list[str]):
        parser = dst_cli.build_parser()
        args = parser.parse_args(argv)
        return args.func(args)

    def _run_dst_main_json(self, argv: list[str]) -> tuple[object, str]:
        buf = StringIO()
        with redirect_stdout(buf):
            code = dst_cli.main(argv)
        self.assertEqual(code, 0)

        raw = buf.getvalue().strip()
        self.assertTrue(raw, f"No stdout payload for argv={argv}")
        return json.loads(raw), raw

    def _create_yolo_fixture(self, root: str, num_images: int = 30):
        images_root = Path(root) / "images"
        labels_root = Path(root) / "labels"
        images_root.mkdir(parents=True, exist_ok=True)
        labels_root.mkdir(parents=True, exist_ok=True)

        for idx in range(num_images):
            split = ["train", "val", "test"][idx % 3]
            image_subdir = images_root / split
            label_subdir = labels_root / split
            image_subdir.mkdir(parents=True, exist_ok=True)
            label_subdir.mkdir(parents=True, exist_ok=True)

            if idx in (0, 1):
                color = (250, 20, 20)  # exact duplicate image signal
            else:
                color = ((idx * 37) % 255, (idx * 19) % 255, (idx * 13) % 255)

            image_path = image_subdir / f"img_{idx:03d}.jpg"
            Image.new("RGB", (96, 96), color=color).save(image_path)

            x, y, w, h = _sample_bbox(idx)
            label_path = label_subdir / f"img_{idx:03d}.txt"
            with label_path.open("w", encoding="utf-8") as f:
                if idx % 2 == 0:
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                else:
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} 0.91\n")

    def _create_coco_fixture(self, root: str, num_images: int = 6):
        data_root = Path(root) / "data"
        data_root.mkdir(parents=True, exist_ok=True)

        images = []
        annotations = []
        categories = [{"id": 1, "name": "rodent", "supercategory": "animal"}]

        ann_id = 1
        for idx in range(num_images):
            file_name = f"coco_{idx:03d}.jpg"
            image_path = data_root / file_name
            Image.new(
                "RGB",
                (128, 128),
                color=((idx * 41) % 255, (idx * 23) % 255, (idx * 7) % 255),
            ).save(image_path)

            images.append(
                {
                    "id": idx + 1,
                    "file_name": file_name,
                    "width": 128,
                    "height": 128,
                }
            )

            x, y, w, h = _sample_bbox(idx)
            # COCO bbox is absolute pixels [x, y, w, h]
            abs_bbox = [x * 128, y * 128, w * 128, h * 128]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": 1,
                    "bbox": abs_bbox,
                    "area": abs_bbox[2] * abs_bbox[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        labels_path = Path(root) / "labels.json"
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with labels_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _enrich_dataset_for_analysis(self, dataset_name: str):
        dataset = fo.load_dataset(dataset_name)
        samples = list(dataset.sort_by("filepath"))
        self.assertGreaterEqual(len(samples), 24)

        for idx, sample in enumerate(samples):
            gt = sample.ground_truth
            gt_dets = list(getattr(gt, "detections", []))
            self.assertGreater(len(gt_dets), 0)

            predictions = []
            for det in gt_dets:
                x, y, w, h = det.bounding_box
                predictions.append(
                    fo.Detection(
                        label=det.label,
                        bounding_box=[x + 0.01, y, w, h],
                        confidence=0.9,
                    )
                )
            sample["predictions"] = fo.Detections(detections=predictions)

            sample["label_cls"] = fo.Classification(
                label="rodent",
                logits=[0.1 + (idx % 2) * 0.2, 0.9 - (idx % 2) * 0.2],
            )

            if idx in (0, 1):
                sample["emb_real"] = [0.0, 1.0, 2.0, 3.0]
            else:
                sample["emb_real"] = [float(idx), float(idx + 1), float(idx + 2), float(idx + 3)]

            tags = set(sample.tags or [])
            tags.add(["train", "val", "test"][idx % 3])
            if idx % 2 == 0:
                tags.add("fix")
            sample.tags = sorted(tags)
            sample.save()

    def test_end_to_end_yolo_metrics_brain_workflow_and_sync(self):
        root = self._mktempdir("dst_real_yolo_")
        self._create_yolo_fixture(root, num_images=30)

        dataset_name = self._new_name("dst_real_yolo")
        self._register_dataset(dataset_name)

        load_payload = self._run_dst_func(
            [
                "data",
                "load",
                "yolo",
                "--dataset",
                dataset_name,
                "--root",
                root,
                "--class-map",
                '{"0":"rodent"}',
                "--overwrite",
            ]
        )
        self.assertEqual(load_payload["dataset"], dataset_name)
        self.assertEqual(load_payload["samples"], 30)

        self._enrich_dataset_for_analysis(dataset_name)
        dataset = fo.load_dataset(dataset_name)

        export_path = str(Path(root) / "import_tasks.json")
        export_payload = self._run_dst_func(
            [
                "data",
                "export",
                "ls-json",
                "--root",
                root,
                "--ls-root",
                "/data/local-files/?d=test/images",
                "--output",
                export_path,
            ]
        )
        self.assertEqual(export_payload["tasks"], 30)
        self.assertTrue(Path(export_path).exists())
        with Path(export_path).open("r", encoding="utf-8") as f:
            exported_tasks = json.load(f)
        self.assertEqual(len(exported_tasks), 30)

        metrics_ops = [
            [
                "metrics",
                "uniqueness",
                "--dataset",
                dataset_name,
                "--embeddings-field",
                "emb_real",
                "--output-field",
                "uniq_real",
            ],
            [
                "metrics",
                "mistakenness",
                "--dataset",
                dataset_name,
                "--pred-field",
                "predictions",
                "--gt-field",
                "ground_truth",
                "--mistakenness-field",
                "mist_real",
                "--missing-field",
                "miss_real",
                "--spurious-field",
                "spur_real",
            ],
            [
                "metrics",
                "hardness",
                "--dataset",
                dataset_name,
                "--label-field",
                "label_cls",
                "--output-field",
                "hard_real",
            ],
            [
                "metrics",
                "representativeness",
                "--dataset",
                dataset_name,
                "--output-field",
                "repr_real",
                "--method",
                "cluster-center",
                "--embeddings-field",
                "emb_real",
            ],
        ]
        for argv in metrics_ops:
            payload = self._run_dst_func(argv)
            self.assertEqual(payload["dataset"], dataset_name)

        brain_ops = [
            [
                "brain",
                "visualization",
                "--dataset",
                dataset_name,
                "--method",
                "pca",
                "--num-dims",
                "2",
                "--embeddings-field",
                "emb_real",
                "--brain-key",
                "viz_real",
            ],
            [
                "brain",
                "similarity",
                "--dataset",
                dataset_name,
                "--embeddings-field",
                "emb_real",
                "--brain-key",
                "sim_real",
            ],
            [
                "brain",
                "duplicates",
                "exact",
                "--dataset",
                dataset_name,
            ],
            [
                "brain",
                "duplicates",
                "near",
                "--dataset",
                dataset_name,
                "--embeddings-field",
                "emb_real",
                "--threshold",
                "0.2",
            ],
            [
                "brain",
                "leaky-splits",
                "--dataset",
                dataset_name,
                "--splits",
                "train,val,test",
                "--embeddings-field",
                "emb_real",
                "--threshold",
                "0.2",
            ],
        ]
        brain_payloads = [self._run_dst_func(argv) for argv in brain_ops]
        self.assertEqual(brain_payloads[0]["brain_key"], "viz_real")
        self.assertEqual(brain_payloads[1]["brain_key"], "sim_real")
        self.assertGreaterEqual(brain_payloads[2]["duplicate_source_count"], 1)
        self.assertGreaterEqual(brain_payloads[3]["duplicate_source_count"], 1)
        self.assertIn("leak_count", brain_payloads[4])

        # Verify fields and brain runs are present on dataset
        dataset.reload()
        self.assertTrue(dataset.has_sample_field("uniq_real"))
        self.assertTrue(dataset.has_sample_field("mist_real"))
        self.assertTrue(dataset.has_sample_field("hard_real"))
        self.assertTrue(dataset.has_sample_field("repr_real"))
        self.assertTrue(dataset.has_sample_field("miss_real"))
        self.assertTrue(dataset.has_sample_field("spur_real"))
        self.assertIn("viz_real", dataset.list_brain_runs())
        self.assertIn("sim_real", dataset.list_brain_runs())

        workflow_path = Path(root) / "workflow.json"
        workflow_output = Path(root) / "workflow_result.json"
        workflow_payload = {
            "dataset_name": dataset_name,
            "fail_fast": True,
            "rules": [
                {
                    "tag": "fix",
                    "operation": "compute_uniqueness",
                    "params": {
                        "scope": "view",
                        "embeddings_field": "emb_real",
                        "output_field": "wf_uniq_fix",
                    },
                },
                {
                    "tag": None,
                    "operation": "compute_similarity_index",
                    "params": {
                        "scope": "dataset",
                        "embeddings_field": "emb_real",
                        "brain_key": "wf_sim_real",
                    },
                },
                {
                    "tag": None,
                    "operation": "compute_near_duplicates",
                    "params": {
                        "scope": "dataset",
                        "embeddings_field": "emb_real",
                        "threshold": 0.2,
                    },
                },
            ],
        }
        with workflow_path.open("w", encoding="utf-8") as f:
            json.dump(workflow_payload, f)

        workflow_result, raw_stdout = self._run_dst_main_json(
            [
                "workflow",
                "tags",
                "run",
                "--workflow",
                str(workflow_path),
                "--quiet-logs",
                "--output-json",
                str(workflow_output),
            ]
        )
        self.assertIsInstance(workflow_result, list)
        self.assertEqual(len(workflow_result), 3)
        self.assertNotIn("Computing uniqueness", raw_stdout)
        with workflow_output.open("r", encoding="utf-8") as f:
            workflow_file_result = json.load(f)
        self.assertEqual(workflow_file_result, workflow_result)

        dataset.reload()
        fix_count = len(dataset.match_tags("fix"))
        wf_uniqueness_non_null = sum(v is not None for v in dataset.values("wf_uniq_fix"))
        self.assertEqual(wf_uniqueness_non_null, fix_count)
        self.assertIn("wf_sim_real", dataset.list_brain_runs())

        # Prepare corrections and test disk sync writes + backup behavior
        fix_view = dataset.match_tags("fix").limit(2)
        for sample in fix_view:
            sample["ls_corrections"] = fo.Detections(
                detections=[
                    fo.Detection(
                        label="mouse",
                        bounding_box=[0.2, 0.2, 0.3, 0.3],
                    )
                ]
            )
            sample.save()

        sync_payload = self._run_dst_func(
            [
                "sync",
                "disk",
                "--dataset",
                dataset_name,
                "--tag",
                "fix",
                "--corrections-field",
                "ls_corrections",
                "--label-to-class-id",
                '{"mouse":5}',
                "--default-class-id",
                "9",
                "--path-replacement",
                "/images/=/labels/",
            ]
        )
        self.assertEqual(sync_payload["dataset"], dataset_name)
        self.assertEqual(sync_payload["synced_files"], 2)

        # Check that corrected labels were rewritten and backups were created
        backup_count = 0
        for sample in fix_view:
            label_path = infer_label_path(sample.filepath, [("/images/", "/labels/")])
            self.assertIsNotNone(label_path)
            assert label_path is not None
            with open(label_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            self.assertTrue(first_line.startswith("5 "))

            label_dir = Path(label_path).parent
            stem = Path(label_path).name
            backups = [p for p in label_dir.glob(f"{stem}.*.bak")]
            backup_count += len(backups)

        self.assertGreaterEqual(backup_count, 2)

    def test_coco_loader_real_life_fixture(self):
        root = self._mktempdir("dst_real_coco_")
        self._create_coco_fixture(root, num_images=6)

        dataset_name = self._new_name("dst_real_coco")
        self._register_dataset(dataset_name)

        payload = self._run_dst_func(
            [
                "data",
                "load",
                "coco",
                "--dataset",
                dataset_name,
                "--dataset-dir",
                root,
                "--data-path",
                "data",
                "--labels-path",
                "labels.json",
                "--overwrite",
            ]
        )
        self.assertEqual(payload["dataset"], dataset_name)
        self.assertEqual(payload["samples"], 6)

        dataset = fo.load_dataset(dataset_name)
        self.assertEqual(len(dataset), 6)
        label_field = None
        for candidate in ("ground_truth", "detections"):
            if dataset.has_sample_field(candidate):
                label_field = candidate
                break

        self.assertIsNotNone(label_field, "Expected COCO import to create a detections label field")
        assert label_field is not None
        self.assertGreater(
            sum(len(dets.detections) for dets in dataset.values(label_field) if dets is not None),
            0,
        )


@unittest.skipUnless(
    RUN_REAL_LIFE_TESTS and RUN_REAL_LIFE_NETWORK_TESTS,
    "Set RUN_REAL_LIFE_TESTS=1 and RUN_REAL_LIFE_NETWORK_TESTS=1 to run network/model-download tests",
)
class RealLifeNetworkModelTests(unittest.TestCase):
    def setUp(self):
        self._tmp_dirs: list[str] = []
        self._datasets: set[str] = set()

    def tearDown(self):
        for dataset_name in sorted(self._datasets):
            if dataset_name in fo.list_datasets():
                fo.delete_dataset(dataset_name)
        for path in self._tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)

    def _new_name(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _mktempdir(self, prefix: str) -> str:
        path = tempfile.mkdtemp(prefix=prefix)
        self._tmp_dirs.append(path)
        return path

    def test_model_zoo_and_hf_embeddings_download_and_inference(self):
        dataset_name = self._new_name("dst_real_network")
        self._datasets.add(dataset_name)

        root = self._mktempdir("dst_real_network_")
        samples = []
        for idx in range(8):
            img_path = Path(root) / f"net_{idx:02d}.jpg"
            Image.new(
                "RGB",
                (224, 224),
                color=((idx * 17) % 255, (idx * 29) % 255, (idx * 43) % 255),
            ).save(img_path)
            samples.append(fo.Sample(filepath=str(img_path)))

        dataset = fo.Dataset(dataset_name)
        dataset.add_samples(samples)
        dataset.persistent = True

        # Model Zoo: downloads model weights (if not cached) and applies inference
        model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
        dataset.apply_model(model, label_field="zoo_predictions")
        self.assertTrue(dataset.has_sample_field("zoo_predictions"))
        zoo_labels = dataset.values("zoo_predictions.label")
        self.assertEqual(len(zoo_labels), len(dataset))
        self.assertGreater(sum(label is not None for label in zoo_labels), 0)

        # HuggingFace embeddings model: downloads weights (if not cached)
        out = EmbeddingsComputation(
            dataset_name=dataset_name,
            model_name="hf-internal-testing/tiny-random-vit",
            embeddings_field="hf_real_embeddings",
            use_umap=False,
            use_cluster=False,
        ).run()
        self.assertEqual(out["dataset"], dataset_name)
        self.assertEqual(out["field"], "hf_real_embeddings")
        emb_values = dataset.values("hf_real_embeddings")
        self.assertEqual(len(emb_values), len(dataset))
        self.assertGreater(sum(v is not None for v in emb_values), 0)


if __name__ == "__main__":
    unittest.main()
