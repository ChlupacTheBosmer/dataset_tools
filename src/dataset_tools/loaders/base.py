"""Implementation module for dataset loading.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import fiftyone as fo  # type: ignore


@dataclass(frozen=True)
class LoaderResult:
    """LoaderResult used by dataset loading.
    """
    dataset_name: str
    sample_count: int


class BaseDatasetLoader(ABC):
    """Dataset loader that imports source media/annotations into FiftyOne.
    """

    @abstractmethod
    def load(self, dataset_name: str, overwrite: bool = False, persistent: bool = True) -> LoaderResult:
        """Perform load.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    overwrite: Whether existing resources should be replaced.
    persistent: Whether created FiftyOne datasets should be persistent.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        pass

    @staticmethod
    def _create_or_replace_dataset(name: str, overwrite: bool, persistent: bool):
        """Internal helper for create or replace dataset.

Args:
    name: Name identifier for the resource being created or retrieved.
    overwrite: Whether existing resources should be replaced.
    persistent: Whether created FiftyOne datasets should be persistent.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if overwrite and name in fo.list_datasets():
            fo.delete_dataset(name)

        if name in fo.list_datasets():
            return fo.load_dataset(name)

        dataset = fo.Dataset(name=name)
        dataset.persistent = persistent
        return dataset
