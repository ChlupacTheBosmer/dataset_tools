"""Push/pull synchronization helpers between FiftyOne and Label Studio.

This module contains the low-level transfer and reconciliation logic used by
workflow operations:

- push samples from a FiftyOne view into Label Studio tasks
- validate mount/storage preconditions before upload
- pull submitted annotations back into FiftyOne fields
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from dataset_tools.config import AppConfig
from dataset_tools.label_studio.translator import (
    fo_detection_to_ls_result,
    ls_rectangle_result_to_fo_detection,
)
from dataset_tools.label_studio.uploader import install_batched_upload_patch


def _set_label_studio_env(url: str, api_key: str):
    """Export LS credentials into env vars used by FiftyOne annotate backend."""
    os.environ["FIFTYONE_LABELSTUDIO_URL"] = url
    os.environ["FIFTYONE_LABELSTUDIO_API_KEY"] = api_key
    os.environ["LABEL_STUDIO_URL"] = url
    os.environ["LABEL_STUDIO_API_KEY"] = api_key


def push_view_to_label_studio(
    view,
    config: AppConfig,
    project_name: str | None = None,
    annotation_key: str | None = None,
    label_field: str | None = None,
    launch_editor: bool = False,
    overwrite_annotation_run: bool = True,
):
    """Push a view to Label Studio using FiftyOne's annotate backend.

    This path preserves annotation-run metadata (``uploaded_tasks``), which can
    later be used for annotation-run based pulling.

    Args:
        view: FiftyOne view whose samples should be exported.
        config: Resolved application configuration.
        project_name: Optional Label Studio project title.
        annotation_key: Optional annotation run key; defaults to project/timestamp.
        label_field: Optional source detections field. Defaults from config.
        launch_editor: If true, launch editor immediately after task creation.
        overwrite_annotation_run: If true, delete existing annotation run with
            the same key before creating a new one.

    Returns:
        FiftyOne annotation results object returned by ``view.annotate(...)``.
    """
    ls_cfg = config.label_studio
    label_field = label_field or config.dataset.label_field

    install_batched_upload_patch(batch_size=ls_cfg.batch_size)
    _set_label_studio_env(url=ls_cfg.url, api_key=ls_cfg.api_key)

    key = annotation_key or project_name or f"ls_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    if overwrite_annotation_run and hasattr(view, "list_annotation_runs"):
        runs = set(view.list_annotation_runs())
        if key in runs and hasattr(view, "delete_annotation_run"):
            view.delete_annotation_run(key)

    return view.annotate(
        key,
        label_field=label_field,
        launch_editor=launch_editor,
        backend="labelstudio",
        url=ls_cfg.url,
        api_key=ls_cfg.api_key,
        project_name=project_name,
    )


def _to_local_files_url(filepath: str, config: AppConfig) -> str | None:
    """Convert an absolute filepath into Label Studio local-files URL.

    Returns ``None`` when the filepath is outside ``mount.host_root``.
    """
    mount = config.mount
    host_root = os.path.abspath(mount.host_root)
    absolute_path = os.path.abspath(filepath)

    try:
        common = os.path.commonpath([host_root, absolute_path])
    except ValueError:
        return None

    if common != host_root:
        return None

    rel_path = os.path.relpath(absolute_path, host_root).replace(os.sep, "/")
    return f"{mount.local_files_prefix}{rel_path}"


def preflight_validate_upload(
    view,
    project,
    config: AppConfig,
    strategy: str,
    strict: bool = True,
    ls_client=None,
) -> dict[str, Any]:
    """Validate upload prerequisites before sending tasks to Label Studio.

    Checks:
    - required import/export local storages are attached to project
    - sample filepaths exist on disk
    - filepaths can be mapped to LS local-files URLs for SDK upload path

    Args:
        view: FiftyOne view planned for upload.
        project: Target Label Studio project.
        config: Resolved application configuration.
        strategy: Upload strategy (``annotate_batched`` or ``sdk_batched``).
        strict: When true with ``sdk_batched``, reject partial mappability.
        ls_client: Optional client used to re-fetch project after storage changes.

    Returns:
        Diagnostics payload summarizing preflight checks.

    Raises:
        RuntimeError: If any required storage/filesystem/mapping check fails.
    """
    filepaths = list(view.values("filepath"))
    total = len(filepaths)

    source_path = config.label_studio.source_path
    target_path = config.label_studio.target_path

    import_paths = [
        storage.get("path")
        for storage in project.get_import_storages()
        if storage.get("type") in ("local", "localfiles")
    ]
    export_paths = [
        storage.get("path")
        for storage in project.get_export_storages()
        if storage.get("type") in ("local", "localfiles")
    ]

    has_source_storage = source_path in import_paths
    has_target_storage = target_path in export_paths

    if (not has_source_storage or not has_target_storage) and ls_client is not None:
        project_id = getattr(project, "id", None)
        if project_id is not None and hasattr(ls_client, "get_project"):
            project = ls_client.get_project(project_id)
            import_paths = [
                storage.get("path")
                for storage in project.get_import_storages()
                if storage.get("type") in ("local", "localfiles")
            ]
            export_paths = [
                storage.get("path")
                for storage in project.get_export_storages()
                if storage.get("type") in ("local", "localfiles")
            ]
            has_source_storage = source_path in import_paths
            has_target_storage = target_path in export_paths

    if not has_source_storage or not has_target_storage:
        raise RuntimeError(
            "Label Studio project storage preflight failed. "
            f"source_path configured={source_path!r}, attached={import_paths}; "
            f"target_path configured={target_path!r}, attached={export_paths}"
        )

    missing_files = [fp for fp in filepaths if not os.path.exists(fp)]
    if missing_files:
        preview = ", ".join(missing_files[:3])
        raise RuntimeError(
            "Upload preflight failed: dataset contains missing filepaths. "
            f"examples: {preview}"
        )

    mappable = [fp for fp in filepaths if _to_local_files_url(fp, config) is not None]
    skipped = total - len(mappable)

    if strategy == "sdk_batched":
        if total > 0 and not mappable:
            raise RuntimeError(
                "Upload preflight failed: no sample filepath can be mapped to "
                "Label Studio local-files URL. Check mount.host_root and dataset filepaths."
            )
        if strict and skipped > 0:
            raise RuntimeError(
                "Upload preflight failed: some sample filepaths are outside mount.host_root "
                f"({config.mount.host_root!r}). total={total}, mappable={len(mappable)}, skipped={skipped}"
            )

    return {
        "strategy": strategy,
        "total_samples": total,
        "mappable_samples": len(mappable),
        "skipped_samples": skipped,
        "missing_files": len(missing_files),
        "has_source_storage": has_source_storage,
        "has_target_storage": has_target_storage,
        "project_title": getattr(project, "title", None),
    }


def push_view_to_label_studio_sdk(
    view,
    project,
    config: AppConfig,
    label_field: str | None = None,
) -> int:
    """Push a view to Label Studio via direct SDK batched task import.

    This path writes ``meta.fiftyone_id`` into each task for robust SDK-based
    pull mapping, and chunks uploads by configured batch size.

    Args:
        view: FiftyOne view to export.
        project: Target Label Studio project.
        config: Resolved application configuration.
        label_field: Optional source detections field for predictions payload.

    Returns:
        Number of tasks imported into the project.
    """
    label_field = label_field or config.dataset.label_field
    tasks: list[dict[str, Any]] = []

    for sample in view:
        ls_path = _to_local_files_url(sample.filepath, config)
        if not ls_path:
            continue

        task: dict[str, Any] = {
            "data": {"image": ls_path},
            "meta": {"fiftyone_id": sample.id},
        }

        fo_labels = getattr(sample, label_field, None)
        detections = getattr(fo_labels, "detections", None)
        if detections:
            predictions = {
                "model_version": "fiftyone_export",
                "result": [fo_detection_to_ls_result(det) for det in detections],
            }
            task["predictions"] = [predictions]

        tasks.append(task)

    batch_size = max(1, int(config.label_studio.batch_size))
    imported = 0
    for idx in range(0, len(tasks), batch_size):
        batch = tasks[idx : idx + batch_size]
        project.import_tasks(batch)
        imported += len(batch)

    return imported


def delete_project_tasks(project):
    """Delete all tasks from a Label Studio project."""
    project.delete_all_tasks()


def pull_labeled_tasks_to_fiftyone(
    dataset,
    project,
    corrections_field: str = "ls_corrections",
) -> int:
    """Pull submitted LS tasks into FiftyOne using task metadata mapping.

    Mapping priority:
    - ``task.meta.fiftyone_id``
    - ``task.data.meta.fiftyone_id`` (legacy shape)

    Only rectangle labels are converted and stored in ``corrections_field``.

    Args:
        dataset: Target FiftyOne dataset.
        project: Label Studio project containing labeled tasks.
        corrections_field: Destination field for pulled detections.

    Returns:
        Number of samples updated in FiftyOne.
    """
    import fiftyone as fo  # type: ignore

    tasks = project.get_labeled_tasks()
    updated = 0

    for task in tasks:
        sample_id = (
            task.get("meta", {}).get("fiftyone_id")
            or task.get("data", {}).get("meta", {}).get("fiftyone_id")
        )
        if not sample_id:
            continue

        try:
            sample = dataset[sample_id]
        except KeyError:
            continue

        annotations = task.get("annotations") or []
        if not annotations:
            continue

        result = annotations[-1].get("result", [])
        detections = []
        for item in result:
            if item.get("type") != "rectanglelabels":
                continue
            detections.append(ls_rectangle_result_to_fo_detection(item))

        sample[corrections_field] = fo.Detections(detections=detections)
        sample.save()
        updated += 1

    return updated


def pull_labeled_tasks_from_annotation_run(
    dataset,
    ls_client,
    annotation_key: str,
    corrections_field: str = "ls_corrections",
) -> int:
    """Pull LS annotations by replaying a FiftyOne annotation-run task map.

    This path is intended for annotate-based upload runs where task IDs are
    tracked in ``uploaded_tasks`` within annotation results.

    Args:
        dataset: Target FiftyOne dataset.
        ls_client: Connected Label Studio client.
        annotation_key: Annotation run key used during send.
        corrections_field: Destination field for pulled detections.

    Returns:
        Number of samples updated in FiftyOne.

    Raises:
        RuntimeError: If annotation run metadata cannot be loaded.
    """
    import fiftyone as fo  # type: ignore

    try:
        results = dataset.load_annotation_results(annotation_key)
    except Exception as e:
        raise RuntimeError(f"Annotation run '{annotation_key}' not found in dataset '{dataset.name}'") from e

    project = ls_client.get_project(results.project_id)
    task_map = dict(getattr(results, "uploaded_tasks", {}) or {})
    task_ids = list(task_map.keys())
    if not task_ids:
        return 0

    tasks = project.get_tasks(selected_ids=task_ids)
    updated = 0

    for task in tasks:
        task_id = task.get("id")
        sample_id = task_map.get(task_id)
        if sample_id is None and isinstance(task_id, str) and task_id.isdigit():
            sample_id = task_map.get(int(task_id))
        if sample_id is None and isinstance(task_id, int):
            sample_id = task_map.get(str(task_id))
        if not sample_id:
            continue

        annotations = task.get("annotations") or []
        if not annotations:
            continue

        latest_annotation = sorted(
            annotations,
            key=lambda x: x.get("updated_at", ""),
        )[-1]
        result = latest_annotation.get("result", [])

        detections = []
        for item in result:
            if item.get("type") != "rectanglelabels":
                continue
            detections.append(ls_rectangle_result_to_fo_detection(item))

        try:
            sample = dataset[sample_id]
        except KeyError:
            continue

        sample[corrections_field] = fo.Detections(detections=detections)
        sample.save()
        updated += 1

    return updated
