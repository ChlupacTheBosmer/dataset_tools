"""Implementation module for FiftyOne Brain analysis.
"""
from __future__ import annotations

from typing import Any

import fiftyone.brain as fob  # type: ignore

from dataset_tools.brain.base import BrainOperation


class ExactDuplicatesOperation(BrainOperation):
    """Operation class used in FiftyOne Brain analysis.
    """
    def execute(self, dataset) -> dict[str, Any]:
        """Perform execute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        result = fob.compute_exact_duplicates(dataset)
        if not isinstance(result, dict):
            return {
                "operation": "brain.duplicates.exact",
                "result_type": type(result).__name__,
            }

        duplicate_source_count = len(result)
        duplicate_sample_count = sum(len(dups) for dups in result.values())
        affected_ids = set(result.keys())
        for dups in result.values():
            affected_ids.update(dups)

        return {
            "operation": "brain.duplicates.exact",
            "duplicate_source_count": duplicate_source_count,
            "duplicate_sample_count": duplicate_sample_count,
            "affected_sample_count": len(affected_ids),
            "result_type": type(result).__name__,
        }


class NearDuplicatesOperation(BrainOperation):
    """Operation class used in FiftyOne Brain analysis.
    """
    def __init__(
        self,
        dataset_name: str,
        threshold: float = 0.2,
        embeddings: str | None = None,
        roi_field: str | None = None,
    ):
        """Initialize `NearDuplicatesOperation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    threshold: Decision/filter threshold used by this operation.
    embeddings: Value controlling embeddings for this routine.
    roi_field: Value controlling roi field for this routine.

Returns:
    None.
        """
        super().__init__(dataset_name=dataset_name, brain_key=None)
        self.threshold = threshold
        self.embeddings = embeddings
        self.roi_field = roi_field

    def execute(self, dataset) -> dict[str, Any]:
        """Perform execute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if self.threshold <= 0:
            raise ValueError("threshold must be > 0")

        kwargs: dict[str, Any] = {
            "threshold": self.threshold,
        }
        if self.embeddings:
            kwargs["embeddings"] = self.embeddings
        if self.roi_field:
            kwargs["roi_field"] = self.roi_field

        result = fob.compute_near_duplicates(dataset, **kwargs)
        if hasattr(result, "find_duplicates"):
            result.find_duplicates(self.threshold)
        neighbors_map = getattr(result, "neighbors_map", None) or {}

        duplicate_source_count = len(neighbors_map)
        duplicate_pair_count = sum(len(neighbors) for neighbors in neighbors_map.values())
        affected_ids = set(neighbors_map.keys())
        for neighbors in neighbors_map.values():
            for neighbor in neighbors:
                if isinstance(neighbor, (tuple, list)) and neighbor:
                    affected_ids.add(str(neighbor[0]))
                else:
                    affected_ids.add(str(neighbor))

        return {
            "operation": "brain.duplicates.near",
            "threshold": self.threshold,
            "duplicate_source_count": duplicate_source_count,
            "duplicate_pair_count": duplicate_pair_count,
            "affected_sample_count": len(affected_ids),
            "result_type": type(result).__name__,
        }
