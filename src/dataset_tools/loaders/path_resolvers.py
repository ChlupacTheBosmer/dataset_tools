"""Implementation module for dataset loading.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MirroredRootsPathResolver:
    """MirroredRootsPathResolver used by dataset loading.
    """
    images_root: Path
    labels_root: Path
    label_ext: str = ".txt"

    def label_path_for(self, image_path: Path) -> Path:
        """Perform label path for.

Args:
    image_path: Value controlling image path for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        rel_path = image_path.relative_to(self.images_root)
        return self.labels_root / rel_path.with_suffix(self.label_ext)


@dataclass(frozen=True)
class ImagesLabelsSubdirResolver:
    """ImagesLabelsSubdirResolver used by dataset loading.
    """
    root_dir: Path
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    label_ext: str = ".txt"

    @property
    def images_root(self) -> Path:
        """Perform images root.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return self.root_dir / self.images_subdir

    @property
    def labels_root(self) -> Path:
        """Perform labels root.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return self.root_dir / self.labels_subdir

    def label_path_for(self, image_path: Path) -> Path:
        """Perform label path for.

Args:
    image_path: Value controlling image path for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        rel_path = image_path.relative_to(self.images_root)
        return self.labels_root / rel_path.with_suffix(self.label_ext)


def default_image_filter(path: Path) -> bool:
    """Perform default image filter.

Args:
    path: Filesystem path used for reading/writing artifacts.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
