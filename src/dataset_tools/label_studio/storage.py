"""Implementation module for Label Studio integration.
"""
from __future__ import annotations

import os
from typing import Iterable

from dataset_tools.config import AppConfig


def _is_local_storage(storage: dict) -> bool:
    """Internal helper for is local storage.

Args:
    storage: Value controlling storage for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    return storage.get("type") in ("local", "localfiles")


def build_rectangle_label_config(labels: Iterable[str]) -> str:
    """Build rectangle label config for downstream steps.

Args:
    labels: Value controlling labels for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    labels = list(labels)
    if not labels:
        labels = ["Object"]

    label_nodes = []
    colors = ["green", "red", "blue", "orange", "magenta", "cyan"]
    for idx, label in enumerate(labels):
        color = colors[idx % len(colors)]
        label_nodes.append(f'<Label value="{label}" background="{color}"/>')

    labels_xml = "\n            ".join(label_nodes)
    return f"""
        <View>
          <Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
            {labels_xml}
          </RectangleLabels>
        </View>
    """.strip()


def _list_projects(ls):
    """List available projects.

Args:
    ls: Connected Label Studio client instance.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if hasattr(ls, "list_projects"):
        return ls.list_projects()
    if hasattr(ls, "get_projects"):
        return ls.get_projects()
    raise RuntimeError("Label Studio client does not support listing projects")


def find_project(ls, title: str):
    """Perform find project.

Args:
    ls: Connected Label Studio client instance.
    title: Value controlling title for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    for project in _list_projects(ls):
        if getattr(project, "title", None) == title:
            return project
    return None


def ensure_project(ls, config: AppConfig, title: str | None = None, label_config: str | None = None):
    """Ensure project exists and return it.

Args:
    ls: Connected Label Studio client instance.
    config: Configuration object controlling runtime behavior.
    title: Value controlling title for this routine.
    label_config: Value controlling label config for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    project_title = title or config.label_studio.project_title

    project = find_project(ls, project_title)
    if project:
        ensure_local_storage(ls, project, config)
        ensure_target_storage(ls, project, config)
        return project

    if not label_config:
        label_config = build_rectangle_label_config(config.dataset.label_to_class_id.keys())

    if not hasattr(ls, "create_project"):
        raise RuntimeError("Label Studio client does not support project creation")

    project = ls.create_project(title=project_title, label_config=label_config)
    ensure_local_storage(ls, project, config)
    ensure_target_storage(ls, project, config)
    return project


def ensure_local_storage(ls, project, config: AppConfig):
    """Ensure local storage exists and return it.

Args:
    ls: Connected Label Studio client instance.
    project: Label Studio project object.
    config: Configuration object controlling runtime behavior.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    source_path = config.label_studio.source_path
    source_title = config.label_studio.source_title

    storages = project.get_import_storages()
    for storage in storages:
        if _is_local_storage(storage) and storage.get("path") == source_path:
            return

    payload = {
        "project": project.id,
        "path": source_path,
        "title": source_title,
        "use_blob_urls": True,
        "regex_filter": ".*(jpg|jpeg|png|bmp)",
        "description": "Configured by dataset_tools",
    }

    response = ls.make_request("POST", "/api/storages/localfiles", json=payload)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Failed to configure local source storage: {response.status_code} {response.text}")

    storage_id = response.json().get("id")
    if storage_id:
        ls.make_request("POST", f"/api/storages/localfiles/{storage_id}/sync")


def ensure_target_storage(ls, project, config: AppConfig):
    """Ensure target storage exists and return it.

Args:
    ls: Connected Label Studio client instance.
    project: Label Studio project object.
    config: Configuration object controlling runtime behavior.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    target_path = config.label_studio.target_path

    storages = project.get_export_storages()
    for storage in storages:
        if _is_local_storage(storage) and storage.get("path") == target_path:
            return

    host_path = target_path.replace(config.mount.ls_document_root, config.mount.host_root)
    os.makedirs(host_path, exist_ok=True)

    payload = {
        "project": project.id,
        "path": target_path,
        "title": "Annotations Export",
        "use_blob_urls": True,
        "can_delete_objects": True,
    }

    response = ls.make_request("POST", "/api/storages/export/localfiles", json=payload)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Failed to configure local target storage: {response.status_code} {response.text}")

    storage_id = response.json().get("id")
    if storage_id:
        ls.make_request("POST", f"/api/storages/export/localfiles/{storage_id}/sync")
