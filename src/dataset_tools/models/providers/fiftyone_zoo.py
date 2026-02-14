"""Implementation module for model provider registry.
"""
from __future__ import annotations

from typing import Any

import fiftyone.zoo as foz  # type: ignore

from dataset_tools.models.base import ModelProvider
from dataset_tools.models.spec import LoadedModel, ModelRef


class FiftyOneZooProvider(ModelProvider):
    """Model provider adapter that resolves and loads backend-specific models.
    """
    name = "foz"

    def load(self, model_ref: ModelRef, *, task: str | None = None, **kwargs: Any) -> LoadedModel:
        """Perform load.

Args:
    model_ref: Provider-qualified model reference (for example `hf:...`, `foz:...`, `anomalib:...`).
    task: Value controlling task for this routine.
    **kwargs: Additional keyword arguments forwarded to downstream APIs.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        model = foz.load_zoo_model(model_ref.model_id, **kwargs)

        capabilities = {"inference"}
        if bool(getattr(model, "has_embeddings", False)) or hasattr(model, "embed"):
            capabilities.add("embeddings")

        return LoadedModel(
            ref=model_ref,
            model=model,
            capabilities=tuple(sorted(capabilities)),
            metadata={
                "task": task,
                "model_type": type(model).__name__,
            },
        )

    def list_models(self, contains: str | None = None, limit: int | None = None) -> list[str]:
        """List available models.

Args:
    contains: Value controlling contains for this routine.
    limit: Value controlling limit for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        models = list(foz.list_zoo_models())
        if contains:
            needle = contains.lower()
            models = [name for name in models if needle in name.lower()]

        if limit is not None:
            models = models[: max(0, int(limit))]
        return models
