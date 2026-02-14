"""Implementation module for anomaly analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnomalyReference:
    """AnomalyReference used by anomaly analysis.
    """
    backend: str
    embeddings_field: str
    threshold: float
    centroid: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Perform to dict.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return {
            "backend": self.backend,
            "embeddings_field": self.embeddings_field,
            "threshold": self.threshold,
            "centroid": self.centroid,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "AnomalyReference":
        """Perform from dict.

Args:
    payload: JSON-like payload consumed by this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return AnomalyReference(
            backend=str(payload["backend"]),
            embeddings_field=str(payload["embeddings_field"]),
            threshold=float(payload["threshold"]),
            centroid=list(payload["centroid"]) if payload.get("centroid") is not None else None,
            metadata=dict(payload.get("metadata", {})),
        )
