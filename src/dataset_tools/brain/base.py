"""Implementation module for FiftyOne Brain analysis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import fiftyone as fo  # type: ignore


class BrainOperation(ABC):
    """Operation class used in FiftyOne Brain analysis.
    """

    def __init__(self, dataset_name: str, brain_key: str | None = None):
        """Initialize `BrainOperation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    brain_key: Value controlling brain key for this routine.

Returns:
    None.
        """
        self.dataset_name = dataset_name
        self.brain_key = brain_key

    def load_dataset(self):
        """Load dataset required by this module.

Returns:
    Loaded object/data required by downstream workflow steps.
        """
        if self.dataset_name not in fo.list_datasets():
            raise RuntimeError(f"Dataset '{self.dataset_name}' not found")
        return fo.load_dataset(self.dataset_name)

    def run(self) -> dict[str, Any]:
        """Run the operation and return execution results.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        dataset = self.load_dataset()
        payload = self.execute(dataset)
        if "dataset" not in payload:
            payload["dataset"] = self.dataset_name
        if self.brain_key and "brain_key" not in payload:
            payload["brain_key"] = self.brain_key
        return payload

    def ensure_brain_run_exists(self, dataset, brain_key: str):
        """Ensure brain run exists exists and return it.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.
    brain_key: Value controlling brain key for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if not dataset.has_brain_run(brain_key):
            raise RuntimeError(
                f"Brain run '{brain_key}' not found in dataset '{self.dataset_name}'"
            )

    @staticmethod
    def _normalize_ids(values) -> list[str]:
        """Internal helper for normalize ids.

Args:
    values: Value controlling values for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return [str(value) for value in values]

    @abstractmethod
    def execute(self, dataset) -> dict[str, Any]:
        """Perform execute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        raise NotImplementedError
