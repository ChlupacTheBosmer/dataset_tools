"""Implementation module for dataset tools runtime.
"""
from __future__ import annotations

import datetime
import os
import shutil
from typing import Iterable

from dataset_tools.config import load_config


def backup_file(filepath: str, suffix_format: str = "%Y%m%d_%H%M%S") -> str | None:
    """Perform backup file.

Args:
    filepath: Filesystem path to a file.
    suffix_format: Value controlling suffix format for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if not os.path.exists(filepath):
        return None

    timestamp = datetime.datetime.now().strftime(suffix_format)
    backup_path = f"{filepath}.{timestamp}.bak"
    shutil.copy2(filepath, backup_path)
    return backup_path


def infer_label_path(
    image_path: str,
    path_replacements: Iterable[tuple[str, str]],
) -> str | None:
    """Perform infer label path.

Args:
    image_path: Value controlling image path for this routine.
    path_replacements: Ordered path replacement rules used for path translation.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    for src, dst in path_replacements:
        if src in image_path:
            return os.path.splitext(image_path.replace(src, dst))[0] + ".txt"
    return None


def sync_corrections_to_disk(
    dataset_name: str | None = None,
    dry_run: bool = False,
    tag_filter: str | None = None,
    corrections_field: str | None = None,
    label_to_class_id: dict[str, int] | None = None,
    default_class_id: int | None = None,
    path_replacements: Iterable[tuple[str, str]] | None = None,
    backup_suffix_format: str | None = None,
) -> int:
    """Perform sync corrections to disk.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    dry_run: If true, report actions without writing changes.
    tag_filter: Sample tag filter used to restrict processing scope.
    corrections_field: Field containing corrected annotations.
    label_to_class_id: Mapping from label string to YOLO class id integer.
    default_class_id: Fallback class id used when mapping is missing a label.
    path_replacements: Ordered path replacement rules used for path translation.
    backup_suffix_format: strftime format used when creating backup filenames.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    import fiftyone as fo  # type: ignore

    config = load_config()

    dataset_name = dataset_name or config.dataset.name
    corrections_field = corrections_field or config.dataset.corrections_field
    label_to_class_id = label_to_class_id or config.dataset.label_to_class_id
    default_class_id = config.dataset.default_class_id if default_class_id is None else default_class_id
    path_replacements = path_replacements or config.disk_sync.path_replacements
    backup_suffix_format = backup_suffix_format or config.disk_sync.backup_suffix_format

    if dataset_name not in fo.list_datasets():
        raise RuntimeError(f"Dataset '{dataset_name}' not found")

    dataset = fo.load_dataset(dataset_name)
    view = dataset.match(fo.ViewField(corrections_field).exists())
    if tag_filter:
        view = view.match_tags(tag_filter)

    updated_count = 0
    for sample in view:
        label_path = infer_label_path(sample.filepath, path_replacements)
        if not label_path:
            continue

        detections_obj = sample[corrections_field]
        detections = getattr(detections_obj, "detections", [])

        if dry_run:
            updated_count += 1
            continue

        label_dir = os.path.dirname(label_path)
        os.makedirs(label_dir, exist_ok=True)

        if os.path.exists(label_path):
            backup_file(label_path, suffix_format=backup_suffix_format)

        lines = []
        for det in detections:
            tl_x, tl_y, width, height = det.bounding_box
            center_x = max(0.0, min(1.0, tl_x + width / 2))
            center_y = max(0.0, min(1.0, tl_y + height / 2))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            class_id = label_to_class_id.get(det.label, default_class_id)
            lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        updated_count += 1

    return updated_count
