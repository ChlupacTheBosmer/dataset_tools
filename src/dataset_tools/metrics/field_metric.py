"""Implementation module for metric computation.
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

from dataset_tools.metrics.base import BaseMetricComputation


class FieldMetricComputation(BaseMetricComputation):
    """FieldMetricComputation used by metric computation.
    """

    def __init__(self, dataset_name: str, required_fields: Iterable[str] | None = None):
        """Initialize `FieldMetricComputation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    required_fields: Value controlling required fields for this routine.

Returns:
    None.
        """
        super().__init__(dataset_name)
        self.required_fields = tuple(required_fields or ())

    def validate_required_fields(self, dataset):
        """Perform validate required fields.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        missing = [field for field in self.required_fields if not dataset.has_sample_field(field)]
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                f"Dataset '{self.dataset_name}' is missing required sample field(s): {missing_str}"
            )

    def run(self):
        """Run the operation and return execution results.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        dataset = self.load_dataset()
        self.validate_required_fields(dataset)
        result = self.compute(dataset)
        if isinstance(result, dict) and "dataset" not in result:
            result["dataset"] = self.dataset_name
        return result

    @abstractmethod
    def compute(self, dataset) -> Any:
        """Perform compute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        raise NotImplementedError
