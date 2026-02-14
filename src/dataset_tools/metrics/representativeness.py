"""Implementation module for metric computation.
"""
from __future__ import annotations

import fiftyone.brain as fob  # type: ignore

from dataset_tools.metrics.field_metric import FieldMetricComputation


class RepresentativenessComputation(FieldMetricComputation):
    """RepresentativenessComputation used by metric computation.
    """
    MIN_CLUSTER_SAMPLES = 20

    def __init__(
        self,
        dataset_name: str,
        output_field: str = "representativeness",
        method: str = "cluster-center",
        embeddings_field: str | None = None,
        roi_field: str | None = None,
    ):
        """Initialize `RepresentativenessComputation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    output_field: Field name where this operation stores its output.
    method: Value controlling method for this routine.
    embeddings_field: Field containing embeddings vectors.
    roi_field: Value controlling roi field for this routine.

Returns:
    None.
        """
        required_fields = tuple(
            field
            for field in (embeddings_field, roi_field)
            if field is not None
        )
        super().__init__(dataset_name=dataset_name, required_fields=required_fields)
        self.output_field = output_field
        self.method = method
        self.embeddings_field = embeddings_field
        self.roi_field = roi_field

    def compute(self, dataset):
        """Perform compute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if self.method in ("cluster-center", "cluster-center-downweight") and len(dataset) < self.MIN_CLUSTER_SAMPLES:
            raise RuntimeError(
                "Representativeness clustering requires at least "
                f"{self.MIN_CLUSTER_SAMPLES} samples; dataset '{self.dataset_name}' has {len(dataset)}"
            )

        fob.compute_representativeness(
            dataset,
            representativeness_field=self.output_field,
            method=self.method,
            embeddings=self.embeddings_field,
            roi_field=self.roi_field,
        )
        return {
            "dataset": self.dataset_name,
            "field": self.output_field,
            "method": self.method,
        }
