"""Implementation module for FiftyOne Brain analysis.
"""
from __future__ import annotations

from typing import Any

import fiftyone.brain as fob  # type: ignore

from dataset_tools.brain.base import BrainOperation


class VisualizationOperation(BrainOperation):
    """Operation class used in FiftyOne Brain analysis.
    """
    def __init__(
        self,
        dataset_name: str,
        method: str = "umap",
        num_dims: int = 2,
        embeddings: str | None = None,
        patches_field: str | None = None,
        brain_key: str | None = None,
    ):
        """Initialize `VisualizationOperation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    method: Value controlling method for this routine.
    num_dims: Value controlling num dims for this routine.
    embeddings: Value controlling embeddings for this routine.
    patches_field: Value controlling patches field for this routine.
    brain_key: Value controlling brain key for this routine.

Returns:
    None.
        """
        super().__init__(dataset_name=dataset_name, brain_key=brain_key)
        self.method = method
        self.num_dims = num_dims
        self.embeddings = embeddings
        self.patches_field = patches_field

    def execute(self, dataset) -> dict[str, Any]:
        """Perform execute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if self.num_dims <= 0:
            raise ValueError("num_dims must be > 0")

        kwargs: dict[str, Any] = {
            "method": self.method,
            "num_dims": self.num_dims,
        }
        if self.embeddings:
            kwargs["embeddings"] = self.embeddings
        if self.patches_field:
            kwargs["patches_field"] = self.patches_field
        if self.brain_key:
            kwargs["brain_key"] = self.brain_key

        result = fob.compute_visualization(dataset, **kwargs)
        key = self.brain_key or getattr(result, "key", None) or getattr(result, "brain_key", None)
        return {
            "operation": "brain.visualization",
            "method": self.method,
            "num_dims": self.num_dims,
            "brain_key": key,
            "persisted": bool(key),
            "index_size": getattr(result, "index_size", None),
            "total_index_size": getattr(result, "total_index_size", None),
        }
