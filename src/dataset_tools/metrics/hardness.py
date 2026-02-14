"""Implementation module for metric computation.
"""
from __future__ import annotations

import fiftyone as fo  # type: ignore
import fiftyone.brain as fob  # type: ignore

from dataset_tools.metrics.field_metric import FieldMetricComputation


class HardnessComputation(FieldMetricComputation):
    """HardnessComputation used by metric computation.
    """
    def __init__(
        self,
        dataset_name: str,
        label_field: str = "ground_truth",
        output_field: str = "hardness",
    ):
        """Initialize `HardnessComputation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    label_field: Field name containing labels for this operation.
    output_field: Field name where this operation stores its output.

Returns:
    None.
        """
        super().__init__(dataset_name=dataset_name, required_fields=(label_field,))
        self.label_field = label_field
        self.output_field = output_field

    def compute(self, dataset):
        """Perform compute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        field = dataset.get_field(self.label_field)
        doc_type = getattr(field, "document_type", None)
        if doc_type not in (fo.Classification, fo.Classifications):
            raise RuntimeError(
                "Hardness requires a classification-style field "
                f"({fo.Classification.__name__} or {fo.Classifications.__name__}); "
                f"'{self.label_field}' is {getattr(doc_type, '__name__', str(doc_type))}"
            )

        fob.compute_hardness(
            dataset,
            label_field=self.label_field,
            hardness_field=self.output_field,
        )
        return {
            "dataset": self.dataset_name,
            "field": self.output_field,
            "label_field": self.label_field,
        }
