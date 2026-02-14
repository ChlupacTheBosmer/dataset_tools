"""Implementation module for metric computation.
"""
from __future__ import annotations

import fiftyone.brain as fob  # type: ignore

from dataset_tools.metrics.field_metric import FieldMetricComputation


class UniquenessComputation(FieldMetricComputation):
    """UniquenessComputation used by metric computation.
    """
    def __init__(self, dataset_name: str, embeddings_field: str | None = None, output_field: str = "uniqueness"):
        """Initialize `UniquenessComputation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    embeddings_field: Field containing embeddings vectors.
    output_field: Field name where this operation stores its output.

Returns:
    None.
        """
        required_fields = (embeddings_field,) if embeddings_field else ()
        super().__init__(dataset_name=dataset_name, required_fields=required_fields)
        self.embeddings_field = embeddings_field
        self.output_field = output_field

    def compute(self, dataset):
        """Perform compute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        fob.compute_uniqueness(
            dataset,
            embeddings=self.embeddings_field,
            uniqueness_field=self.output_field,
        )
        return {"dataset": self.dataset_name, "field": self.output_field}
