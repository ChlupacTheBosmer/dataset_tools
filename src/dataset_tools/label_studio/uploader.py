"""Implementation module for Label Studio integration.
"""
from __future__ import annotations

import json
import os
from typing import Any

_PATCHED_BATCH_SIZE: int | None = None


def install_batched_upload_patch(batch_size: int = 10):
    """Perform install batched upload patch.

Args:
    batch_size: Batch size controlling transfer or inference throughput.

Returns:
    Result object consumed by the caller or downstream workflow.
    """

    global _PATCHED_BATCH_SIZE

    if _PATCHED_BATCH_SIZE == batch_size:
        return

    import fiftyone.utils.labelstudio as fols  # type: ignore

    def batched_upload_tasks(self, project, tasks, predictions=None):
        all_uploaded_tasks: dict[int, str] = {}
        total = len(tasks)
        print(f"Uploading {total} tasks in batches of {batch_size}...")

        for idx in range(0, total, batch_size):
            batch_tasks = tasks[idx : idx + batch_size]
            current_predictions = None
            if predictions:
                batch_source_ids = {task["source_id"] for task in batch_tasks}
                current_predictions = {}
                for field, preds in predictions.items():
                    current_predictions[field] = {
                        sid: labels for sid, labels in preds.items() if sid in batch_source_ids
                    }

            ls_root = os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", None)
            ls_local_enabled = os.getenv("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED", None)
            local_storage_enabled = (
                ls_root is not None
                and ls_local_enabled is not None
                and ls_local_enabled.lower() == "true"
            )

            common_prefix = os.path.commonprefix(
                [task[task["media_type"]] for task in batch_tasks]
            ).rstrip("/")
            common_prefix = os.path.dirname(common_prefix)

            if local_storage_enabled and ls_root in common_prefix:
                project.connect_local_import_storage(common_prefix)

                def _get_file_path(file_path):
                    return "/data/local-files?d=" + os.path.relpath(file_path, ls_root)

                files_data = [
                    {task["media_type"]: _get_file_path(task[task["media_type"]])}
                    for task in batch_tasks
                ]

                self._client.make_request(
                    "POST",
                    f"/api/projects/{project.id}/import",
                    json=files_data,
                )
            else:
                files = []
                opened_files = []
                for task in batch_tasks:
                    path = task[task["media_type"]]
                    fh = open(path, "rb")
                    opened_files.append(fh)
                    files.append((task["source_id"], (os.path.basename(path), fh)))

                try:
                    upload_resp = self._client.make_request(
                        "POST",
                        f"/api/projects/{project.id}/import",
                        params={"commit_to_project": True},
                        files=files,
                    )
                finally:
                    for fh in opened_files:
                        fh.close()

                payload = json.dumps(
                    {
                        "file_upload_ids": upload_resp.json()["file_upload_ids"],
                        "files_as_tasks_list": False,
                    }
                )

                self._client.headers.update({"Content-Type": "application/json"})
                self._client.make_request(
                    "POST", f"/api/projects/{project.id}/reimport", data=payload
                )
                if "Content-Type" in self._client.headers:
                    del self._client.headers["Content-Type"]

            uploaded_ids = project.get_tasks(only_ids=True)[-len(batch_tasks) :]

            batch_uploaded_tasks = {
                task_id: task["source_id"] for task_id, task in zip(uploaded_ids, batch_tasks)
            }

            if current_predictions:
                source_to_task = {source_id: task_id for task_id, source_id in batch_uploaded_tasks.items()}
                for _, label_predictions in current_predictions.items():
                    ls_predictions = [
                        {
                            "task": source_to_task[sample_id],
                            "result": pred,
                        }
                        for sample_id, pred in label_predictions.items()
                        if sample_id in source_to_task
                    ]
                    if ls_predictions:
                        project.create_predictions(ls_predictions)

            all_uploaded_tasks.update(batch_uploaded_tasks)
            print(
                f"  Batch {idx // batch_size + 1} complete. "
                f"({len(batch_uploaded_tasks)} tasks)"
            )

        return all_uploaded_tasks

    original_init_project = fols.LabelStudioAnnotationAPI._init_project

    def init_project_reuse_existing(self, config, samples):
        """
        Reuses an existing project with the same title instead of creating
        suffixed duplicates like `<name>_<timestamp>`.
        """
        project_name = config.project_name
        if project_name is None:
            dataset_name = samples._root_dataset.name.replace(" ", "_")
            project_name = f"FiftyOne_{dataset_name}"

        for one in self._client.list_projects():
            title = getattr(one, "title", None)
            if title is None and hasattr(one, "params"):
                title = one.params.get("title")

            if title != project_name:
                continue

            project_id = getattr(one, "id", None)
            if project_id is None and hasattr(one, "params"):
                project_id = one.params.get("id")

            if project_id is not None:
                return self._client.get_project(project_id)

            return one

        return original_init_project(self, config, samples)

    fols.LabelStudioAnnotationAPI._upload_tasks = batched_upload_tasks
    fols.LabelStudioAnnotationAPI._init_project = init_project_reuse_existing
    _PATCHED_BATCH_SIZE = batch_size
