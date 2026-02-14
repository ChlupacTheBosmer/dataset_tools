"""Implementation module for dataset loading.
"""
from __future__ import annotations

from pathlib import Path

import fiftyone as fo  # type: ignore

from dataset_tools.loaders import (
    ImagesLabelsSubdirResolver,
    MirroredRootsPathResolver,
    YoloDatasetLoader,
    YoloParserConfig,
)


def import_yolo_dataset_from_root(
    root_dir: str,
    dataset_name: str,
    image_subdir: str = "images",
    labels_subdir: str = "labels",
    class_id_to_label: dict[int, str] | None = None,
    overwrite: bool = True,
):
    """Perform import yolo dataset from root.

Args:
    root_dir: Root directory used by resolver or loader logic.
    dataset_name: Name of the FiftyOne dataset to operate on.
    image_subdir: Value controlling image subdir for this routine.
    labels_subdir: Value controlling labels subdir for this routine.
    class_id_to_label: Value controlling class id to label for this routine.
    overwrite: Whether existing resources should be replaced.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    resolver = ImagesLabelsSubdirResolver(
        root_dir=Path(root_dir),
        images_subdir=image_subdir,
        labels_subdir=labels_subdir,
    )
    loader = YoloDatasetLoader(
        resolver=resolver,
        parser_config=YoloParserConfig(class_id_to_label=class_id_to_label or {}),
    )
    return loader.load(dataset_name=dataset_name, overwrite=overwrite, persistent=True)


def import_yolo_dataset_from_roots(
    images_root: str,
    labels_root: str,
    dataset_name: str,
    class_id_to_label: dict[int, str] | None = None,
    overwrite: bool = True,
):
    """Perform import yolo dataset from roots.

Args:
    images_root: Value controlling images root for this routine.
    labels_root: Value controlling labels root for this routine.
    dataset_name: Name of the FiftyOne dataset to operate on.
    class_id_to_label: Value controlling class id to label for this routine.
    overwrite: Whether existing resources should be replaced.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    resolver = MirroredRootsPathResolver(images_root=Path(images_root), labels_root=Path(labels_root))
    loader = YoloDatasetLoader(
        resolver=resolver,
        parser_config=YoloParserConfig(class_id_to_label=class_id_to_label or {}),
    )
    return loader.load(dataset_name=dataset_name, overwrite=overwrite, persistent=True)


def get_or_create_dataset(name: str):
    """Perform get or create dataset.

Args:
    name: Name identifier for the resource being created or retrieved.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if name in fo.list_datasets():
        return fo.load_dataset(name)

    dataset = fo.Dataset(name=name)
    dataset.persistent = True
    return dataset
