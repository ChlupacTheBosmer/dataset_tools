"""Implementation module for FiftyOne Brain analysis.
"""
from __future__ import annotations

from typing import Any

import fiftyone.brain as fob  # type: ignore

from dataset_tools.brain.base import BrainOperation


class SimilarityOperation(BrainOperation):
    """Operation class used in FiftyOne Brain analysis.
    """
    def __init__(
        self,
        dataset_name: str,
        embeddings: str | None = None,
        patches_field: str | None = None,
        roi_field: str | None = None,
        backend: str | None = None,
        brain_key: str | None = None,
    ):
        """Initialize `SimilarityOperation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    embeddings: Value controlling embeddings for this routine.
    patches_field: Value controlling patches field for this routine.
    roi_field: Value controlling roi field for this routine.
    backend: Value controlling backend for this routine.
    brain_key: Value controlling brain key for this routine.

Returns:
    None.
        """
        super().__init__(dataset_name=dataset_name, brain_key=brain_key)
        self.embeddings = embeddings
        self.patches_field = patches_field
        self.roi_field = roi_field
        self.backend = backend

    def execute(self, dataset) -> dict[str, Any]:
        """Perform execute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        kwargs: dict[str, Any] = {}
        if self.embeddings:
            kwargs["embeddings"] = self.embeddings
        if self.patches_field:
            kwargs["patches_field"] = self.patches_field
        if self.roi_field:
            kwargs["roi_field"] = self.roi_field
        if self.backend:
            kwargs["backend"] = self.backend
        if self.brain_key:
            kwargs["brain_key"] = self.brain_key

        result = fob.compute_similarity(dataset, **kwargs)
        key = self.brain_key or getattr(result, "key", None) or getattr(result, "brain_key", None)
        return {
            "operation": "brain.similarity",
            "backend": self.backend,
            "brain_key": key,
            "persisted": bool(key),
            "index_size": getattr(result, "index_size", None),
            "total_index_size": getattr(result, "total_index_size", None),
        }
