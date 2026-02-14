"""Implementation module for FiftyOne Brain analysis.
"""
from __future__ import annotations

from typing import Any

import fiftyone.brain as fob  # type: ignore

from dataset_tools.brain.base import BrainOperation


class LeakySplitsOperation(BrainOperation):
    """Operation class used in FiftyOne Brain analysis.
    """
    def __init__(
        self,
        dataset_name: str,
        splits: list[str],
        threshold: float = 0.2,
        embeddings: str | None = None,
        roi_field: str | None = None,
    ):
        """Initialize `LeakySplitsOperation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    splits: Value controlling splits for this routine.
    threshold: Decision/filter threshold used by this operation.
    embeddings: Value controlling embeddings for this routine.
    roi_field: Value controlling roi field for this routine.

Returns:
    None.
        """
        super().__init__(dataset_name=dataset_name, brain_key=None)
        self.splits = splits
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
        if not self.splits:
            raise ValueError("At least one split must be provided")
        if self.threshold <= 0:
            raise ValueError("threshold must be > 0")

        kwargs: dict[str, Any] = {
            "splits": self.splits,
            "threshold": self.threshold,
        }
        if self.embeddings:
            kwargs["embeddings"] = self.embeddings
        if self.roi_field:
            kwargs["roi_field"] = self.roi_field

        result = fob.compute_leaky_splits(dataset, **kwargs)
        if hasattr(result, "find_leaks"):
            result.find_leaks(self.threshold)

        leak_ids = getattr(result, "leak_ids", []) or []
        split_views = getattr(result, "split_views", {}) or {}

        return {
            "operation": "brain.leaky_splits",
            "splits": list(self.splits),
            "threshold": self.threshold,
            "leak_count": len(leak_ids),
            "sample_count": len(dataset),
            "split_count": len(split_views),
            "result_type": type(result).__name__,
        }
