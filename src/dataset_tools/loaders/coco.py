"""Implementation module for dataset loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fiftyone as fo  # type: ignore

from dataset_tools.loaders.base import BaseDatasetLoader, LoaderResult


@dataclass(frozen=True)
class CocoLoaderConfig:
    """Configuration dataclass for dataset loading.
    """
    dataset_dir: Path
    data_path: str = "data"
    labels_path: str = "labels.json"


class CocoDatasetLoader(BaseDatasetLoader):
    """Dataset loader that imports source media/annotations into FiftyOne.
    """
    def __init__(self, config: CocoLoaderConfig):
        """Initialize `CocoDatasetLoader` with runtime parameters.

Args:
    config: Configuration object controlling runtime behavior.

Returns:
    None.
        """
        self.config = config

    def load(self, dataset_name: str, overwrite: bool = False, persistent: bool = True) -> LoaderResult:
        """Perform load.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    overwrite: Whether existing resources should be replaced.
    persistent: Whether created FiftyOne datasets should be persistent.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        dataset = self._create_or_replace_dataset(dataset_name, overwrite=overwrite, persistent=persistent)

        imported = fo.Dataset.from_dir(
            dataset_dir=str(self.config.dataset_dir),
            dataset_type=fo.types.COCODetectionDataset,
            data_path=self.config.data_path,
            labels_path=self.config.labels_path,
            name=dataset_name,
            persistent=persistent,
            overwrite=overwrite,
        )

        return LoaderResult(dataset_name=imported.name, sample_count=len(imported))
