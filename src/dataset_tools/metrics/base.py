"""Implementation module for metric computation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import fiftyone as fo  # type: ignore


class BaseMetricComputation(ABC):
    """BaseMetricComputation used by metric computation.
    """
    def __init__(self, dataset_name: str):
        """Initialize `BaseMetricComputation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.

Returns:
    None.
        """
        self.dataset_name = dataset_name

    def load_dataset(self):
        """Load dataset required by this module.

Returns:
    Loaded object/data required by downstream workflow steps.
        """
        if self.dataset_name not in fo.list_datasets():
            raise RuntimeError(f"Dataset '{self.dataset_name}' not found")
        return fo.load_dataset(self.dataset_name)

    def run(self):
        """Run the operation and return execution results.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        dataset = self.load_dataset()
        return self.compute(dataset)

    @abstractmethod
    def compute(self, dataset):
        """Perform compute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        pass
