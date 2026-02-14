"""Implementation module for model provider registry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dataset_tools.models.spec import LoadedModel, ModelRef


class ModelProvider(ABC):
    """Model provider adapter that resolves and loads backend-specific models.
    """
    name: str

    @abstractmethod
    def load(self, model_ref: ModelRef, *, task: str | None = None, **kwargs: Any) -> LoadedModel:
        """Perform load.

Args:
    model_ref: Provider-qualified model reference (for example `hf:...`, `foz:...`, `anomalib:...`).
    task: Value controlling task for this routine.
    **kwargs: Additional keyword arguments forwarded to downstream APIs.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        pass

    def list_models(self, contains: str | None = None, limit: int | None = None) -> list[str]:
        """List available models.

Args:
    contains: Value controlling contains for this routine.
    limit: Value controlling limit for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return []
