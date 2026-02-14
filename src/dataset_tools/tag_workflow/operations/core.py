"""Core tag-workflow operations for mutation, LS sync, and disk sync.

These operations are intentionally dataset-agnostic and receive behavior through
rule parameters, which keeps project-specific logic out of reusable tooling.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import fiftyone as fo  # type: ignore

from dataset_tools.config import AppConfig
from dataset_tools.label_studio import (
    delete_project_tasks,
    ensure_label_studio_client,
    ensure_project,
    find_project,
    preflight_validate_upload,
    pull_labeled_tasks_from_annotation_run,
    pull_labeled_tasks_to_fiftyone,
    push_view_to_label_studio,
    push_view_to_label_studio_sdk,
)
from dataset_tools.sync_from_fo_to_disk import sync_corrections_to_disk
from dataset_tools.tag_workflow.context import TagWorkflowContext
from dataset_tools.tag_workflow.operations.analysis import (
    ComputeAnomalyScoresOperation,
    ComputeExactDuplicatesOperation,
    ComputeHardnessOperation,
    ComputeLeakySplitsOperation,
    ComputeNearDuplicatesOperation,
    ComputeRepresentativenessOperation,
    ComputeSimilarityIndexOperation,
    ComputeUniquenessOperation,
)
from dataset_tools.tag_workflow.operations.base import TagOperation


def _app_config_with_ls_overrides(config: AppConfig, params: dict[str, Any]) -> AppConfig:
    """Return config copy with optional Label Studio/dataset overrides from rule params.

    Supported keys in ``params`` are mapped onto ``AppConfig.label_studio`` and
    selected dataset sync fields so a single workflow can target different LS
    projects/storages without mutating global config.
    """
    ls_cfg = config.label_studio
    ds_cfg = config.dataset

    ls_updates = {}
    for key in (
        "url",
        "api_key",
        "project_title",
        "source_path",
        "source_title",
        "target_path",
        "batch_size",
        "clear_existing_tasks",
        "upload_strategy",
    ):
        if key in params:
            ls_updates[key] = params[key]

    ds_updates = {}
    if "label_to_class_id" in params:
        ds_updates["label_to_class_id"] = dict(params["label_to_class_id"])
    if "default_class_id" in params:
        ds_updates["default_class_id"] = int(params["default_class_id"])

    updated_ls = replace(ls_cfg, **ls_updates) if ls_updates else ls_cfg
    updated_ds = replace(ds_cfg, **ds_updates) if ds_updates else ds_cfg
    return replace(config, label_studio=updated_ls, dataset=updated_ds)


class DeleteSamplesOperation(TagOperation):
    """Delete selected samples from the active dataset."""
    name = "delete_samples"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Delete all samples in ``view`` and report count."""
        count = len(view)
        context.dataset.delete_samples(view)
        return {"operation": self.name, "tag": tag, "deleted_samples": count}


class DeleteFilesAndSamplesOperation(TagOperation):
    """Delete media files on disk and then delete corresponding samples."""
    name = "delete_files_and_samples"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Remove existing files referenced by ``view`` and drop samples."""
        deleted_files = 0
        for filepath in view.values("filepath"):
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_files += 1

        deleted_samples = len(view)
        context.dataset.delete_samples(view)
        return {
            "operation": self.name,
            "tag": tag,
            "deleted_files": deleted_files,
            "deleted_samples": deleted_samples,
        }


class MoveSamplesToDatasetOperation(TagOperation):
    """Copy/move selected samples into a target dataset."""
    name = "move_samples_to_dataset"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Add ``view`` samples to target dataset and optionally remove source samples.

        Required params:
            - ``target_dataset``: target dataset name

        Optional params:
            - ``remove_from_source`` (default ``True``)
        """
        target_dataset_name = params.get("target_dataset")
        if not target_dataset_name:
            raise ValueError("'target_dataset' is required for move_samples_to_dataset")

        if target_dataset_name in fo.list_datasets():
            target_dataset = fo.load_dataset(target_dataset_name)
        else:
            target_dataset = fo.Dataset(target_dataset_name)
            target_dataset.persistent = True

        count = len(view)
        target_dataset.add_samples(view)

        if params.get("remove_from_source", True):
            context.dataset.delete_samples(view)

        return {
            "operation": self.name,
            "tag": tag,
            "moved_samples": count,
            "target_dataset": target_dataset_name,
        }


class SendToLabelStudioOperation(TagOperation):
    """Send selected samples to Label Studio using configured upload strategy."""
    name = "send_to_label_studio"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Push samples to LS and return transfer diagnostics.

        Strategy options:
            - ``sdk_batched``: direct SDK task import (preferred default)
            - ``annotate_batched``: FiftyOne annotate backend with annotation runs
        """
        if len(view) == 0:
            return {"operation": self.name, "tag": tag, "sent_samples": 0}

        app_cfg = _app_config_with_ls_overrides(context.app_config, params)
        ls = context.caches.get("ls_client")
        if ls is None:
            ls = ensure_label_studio_client(app_cfg)
            context.caches["ls_client"] = ls

        project_title = params.get("project_title", app_cfg.label_studio.project_title)
        project = ensure_project(ls, app_cfg, title=project_title)
        context.caches["ls_project"] = project

        label_field = params.get("label_field", app_cfg.dataset.label_field)
        upload_strategy = params.get("upload_strategy", app_cfg.label_studio.upload_strategy)
        strict_preflight = bool(params.get("strict_preflight", True))
        preflight = preflight_validate_upload(
            view=view,
            project=project,
            config=app_cfg,
            strategy=upload_strategy,
            strict=strict_preflight,
            ls_client=ls,
        )

        clear_tasks = params.get("clear_project_tasks", app_cfg.label_studio.clear_existing_tasks)
        if clear_tasks:
            delete_project_tasks(project)

        if upload_strategy == "sdk_batched":
            sent = push_view_to_label_studio_sdk(view=view, project=project, config=app_cfg, label_field=label_field)
            return {
                "operation": self.name,
                "tag": tag,
                "project": project_title,
                "sent_samples": sent,
                "strategy": upload_strategy,
                "preflight": preflight,
            }

        annotation_key = params.get("annotation_key", project_title)
        result = push_view_to_label_studio(
            view=view,
            config=app_cfg,
            project_name=project_title,
            annotation_key=annotation_key,
            label_field=label_field,
            launch_editor=bool(params.get("launch_editor", False)),
            overwrite_annotation_run=bool(params.get("overwrite_annotation_run", True)),
        )

        uploaded_count = len(view)
        if hasattr(result, "uploaded_tasks"):
            uploaded_count = len(result.uploaded_tasks)

        return {
            "operation": self.name,
            "tag": tag,
            "project": project_title,
            "sent_samples": uploaded_count,
            "strategy": "annotate_batched",
            "preflight": preflight,
        }


class PullFromLabelStudioOperation(TagOperation):
    """Pull submitted LS annotations back into FiftyOne correction fields."""
    name = "pull_from_label_studio"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Pull LS annotations using strategy-compatible mapping path.

        Pull strategy defaults by upload strategy:
            - annotate upload -> ``annotate_run`` pull
            - sdk upload -> ``sdk_meta`` pull
        """
        app_cfg = _app_config_with_ls_overrides(context.app_config, params)
        corrections_field = params.get("corrections_field", app_cfg.dataset.corrections_field)
        project_title = params.get("project_title", app_cfg.label_studio.project_title)

        upload_strategy = params.get("upload_strategy", app_cfg.label_studio.upload_strategy)
        pull_strategy = params.get("pull_strategy")
        if not pull_strategy:
            pull_strategy = "annotate_run" if upload_strategy == "annotate_batched" else "sdk_meta"

        ls = context.caches.get("ls_client")
        if ls is None:
            ls = ensure_label_studio_client(app_cfg)
            context.caches["ls_client"] = ls

        if pull_strategy == "annotate_run":
            annotation_key = params.get("annotation_key", project_title)
            updated = pull_labeled_tasks_from_annotation_run(
                dataset=context.dataset,
                ls_client=ls,
                annotation_key=annotation_key,
                corrections_field=corrections_field,
            )
            return {
                "operation": self.name,
                "tag": tag,
                "project": project_title,
                "annotation_key": annotation_key,
                "updated_samples": updated,
                "corrections_field": corrections_field,
                "strategy": pull_strategy,
            }

        project = find_project(ls, project_title)
        if not project:
            if params.get("create_if_missing", False):
                project = ensure_project(ls, app_cfg, title=project_title)
            else:
                raise RuntimeError(f"Label Studio project '{project_title}' not found")

        updated = pull_labeled_tasks_to_fiftyone(
            dataset=context.dataset,
            project=project,
            corrections_field=corrections_field,
        )
        return {
            "operation": self.name,
            "tag": tag,
            "project": project_title,
            "updated_samples": updated,
            "corrections_field": corrections_field,
            "strategy": pull_strategy,
        }


class SyncCorrectionsToDiskOperation(TagOperation):
    """Write correction fields from FiftyOne samples back to label files on disk."""
    name = "sync_corrections_to_disk"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Run disk sync with workflow/config overrides and report synced file count."""
        app_cfg = _app_config_with_ls_overrides(context.app_config, params)

        synced = sync_corrections_to_disk(
            dataset_name=context.dataset_name,
            dry_run=bool(params.get("dry_run", False)),
            tag_filter=tag,
            corrections_field=params.get("corrections_field", app_cfg.dataset.corrections_field),
            label_to_class_id=params.get("label_to_class_id", app_cfg.dataset.label_to_class_id),
            default_class_id=int(params.get("default_class_id", app_cfg.dataset.default_class_id)),
            path_replacements=params.get("path_replacements", app_cfg.disk_sync.path_replacements),
            backup_suffix_format=params.get(
                "backup_suffix_format", app_cfg.disk_sync.backup_suffix_format
            ),
        )

        return {
            "operation": self.name,
            "tag": tag,
            "synced_files": synced,
            "dry_run": bool(params.get("dry_run", False)),
        }


def default_operations_registry() -> dict[str, TagOperation]:
    """Return default registry of core and analysis operations by operation name."""
    operations = [
        DeleteSamplesOperation(),
        DeleteFilesAndSamplesOperation(),
        MoveSamplesToDatasetOperation(),
        SendToLabelStudioOperation(),
        PullFromLabelStudioOperation(),
        SyncCorrectionsToDiskOperation(),
        ComputeUniquenessOperation(),
        ComputeHardnessOperation(),
        ComputeRepresentativenessOperation(),
        ComputeSimilarityIndexOperation(),
        ComputeExactDuplicatesOperation(),
        ComputeNearDuplicatesOperation(),
        ComputeLeakySplitsOperation(),
        ComputeAnomalyScoresOperation(),
    ]
    return {op.name: op for op in operations}
