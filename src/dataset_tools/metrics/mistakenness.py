"""Implementation module for metric computation.
"""
from __future__ import annotations

import fiftyone.brain as fob  # type: ignore

from dataset_tools.metrics.field_metric import FieldMetricComputation


class MistakennessComputation(FieldMetricComputation):
    """MistakennessComputation used by metric computation.
    """
    def __init__(
        self,
        dataset_name: str,
        pred_field: str = "predictions",
        gt_field: str = "ground_truth",
        mistakenness_field: str = "mistakenness",
        missing_field: str = "possible_missing",
        spurious_field: str = "possible_spurious",
    ):
        """Initialize `MistakennessComputation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    pred_field: Prediction field name used as input.
    gt_field: Ground-truth field name used as input.
    mistakenness_field: Value controlling mistakenness field for this routine.
    missing_field: Value controlling missing field for this routine.
    spurious_field: Value controlling spurious field for this routine.

Returns:
    None.
        """
        super().__init__(dataset_name=dataset_name, required_fields=(pred_field, gt_field))
        self.pred_field = pred_field
        self.gt_field = gt_field
        self.mistakenness_field = mistakenness_field
        self.missing_field = missing_field
        self.spurious_field = spurious_field

    def compute(self, dataset):
        """Perform compute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        fob.compute_mistakenness(
            dataset,
            pred_field=self.pred_field,
            label_field=self.gt_field,
            mistakenness_field=self.mistakenness_field,
            missing_field=self.missing_field,
            spurious_field=self.spurious_field,
        )
        return {"dataset": self.dataset_name, "field": self.mistakenness_field}
